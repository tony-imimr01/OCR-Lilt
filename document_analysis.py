#!/usr/bin/env python3
"""
Document Analysis API - LiLT + Email Extraction + Fingerprint Verification
Combined API with:
- LiLT-based document analysis and form field extraction
- OCR fingerprint + LILT embedding hybrid verification
- Email and phone number extraction
- Individual checkbox detection (separate fields for each option)
- Form template matching
Run:
python3 combined_document_analysis_api.py \
--model_path lilt_text_extraction_model05 \
--qa_model deepset/roberta-base-squad2 \
--fingerprints_out fingerprints.pkl \
--port 8000 --debug
"""
import os
import re
import time
import json
import pickle
import tempfile
import argparse
import logging
import hashlib
from typing import List, Dict, Optional, Tuple, Any, Union
from collections import defaultdict, Counter
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from datetime import datetime

# ---- CUDA / Torch env ----
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
try:
    import torch
    TORCH_AVAILABLE = True
    def _validate_cuda_state():
        try:
            if torch.cuda.is_available():
                _ = torch.zeros(10, device="cuda")
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                return True
        except Exception as e:
            logging.warning(f"CUDA validation failed: {e}")
            return False
    CUDA_AVAILABLE = _validate_cuda_state()
except Exception as e:
    logging.warning(f"PyTorch import failed: {e}")
    torch = None
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    pytesseract = None
    TESSERACT_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except Exception:
    PDF2IMAGE_AVAILABLE = False

try:
    from transformers import (
        pipeline,
        AutoModel,
        AutoConfig,
        AutoTokenizer,
        AutoModelForSequenceClassification,
    )
    TRANSFORMERS_AVAILABLE = True
except Exception as e:
    logging.warning(f"Transformers import failed: {e}")
    TRANSFORMERS_AVAILABLE = False

EASYOCR_AVAILABLE = False
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    easyocr = None

# LiLT models
LILT_AVAILABLE = False
try:
    from models.LiLTRobertaLike import (
        LiLTRobertaLikeForRelationExtraction,
        LiLTRobertaLikeConfig,
        LiLTRobertaLikeForTokenClassification
    )
    LILT_AVAILABLE = True
    logging.info("LiLT models imported from models.LiLTRobertaLike")
except ImportError:
    try:
        from LiLTRobertaLike import (
            LiLTRobertaLikeForRelationExtraction,
            LiLTRobertaLikeConfig,
            LiLTRobertaLikeForTokenClassification
        )
        LILT_AVAILABLE = True
        logging.info("LiLT models imported from LiLTRobertaLike")
    except ImportError:
        logging.warning("LiLT model not found. Using fallback.")
        LILT_AVAILABLE = False

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# ------- Logging -------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("combined_document_analysis_api")

# ------- Constants / Config -------
class EntityTypes:
    HEADER = "header"
    QUESTION = "question"
    ANSWER = "answer"
    OTHER = "other"
    CONTENT = "content"
    METADATA = "metadata"
    STRUCTURE = "structure"

class LiLTConfig:
    def __init__(self, model_path: Optional[str] = None, qa_model_path: Optional[str] = None):
        self.model_path = model_path
        self.qa_model_path = qa_model_path or "deepset/roberta-base-squad2"
        self.coordinate_range = 1000
        self.min_confidence = 0.15
        self.num_rel_labels = 2
        self.max_entities_per_doc = 1500
        self.min_entity_length = 1
        self.max_seq_length = 512
        self.max_word_length = 300
        self.enable_qa_linking = True
        self.max_qa_distance = 500
        self.qa_threshold = 0.01

# ------- Pydantic models -------
class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0

class ExtractedEntity(BaseModel):
    field: str
    value: str
    bbox: BoundingBox
    confidence: float = 1.0
    page_number: int = 1
    semantic_type: Optional[str] = None
    semantic_confidence: Optional[float] = None

class DataField(BaseModel):
    field_name: str
    field_type: str
    value: str
    confidence: float
    bbox: BoundingBox

class KeyFieldResult(BaseModel):
    field_name: str
    value: str
    structured_value: Optional[DataField] = None
    confidence: float
    bbox: BoundingBox
    context_entities: List[ExtractedEntity] = []
    meta: Optional[Dict[str, Any]] = None
    extraction_method: Optional[str] = None

class ExtractionResult(BaseModel):
    document_name: str
    page_count: int
    total_entities: int
    entities: List[ExtractedEntity]
    key_field_result: Optional[Union[KeyFieldResult, List[KeyFieldResult]]] = None
    full_text_snippet: str = ""
    processing_time: float = 0.0
    language_used: str = "eng"
    model_used: bool = False

class AnalysisResponse(BaseModel):
    status: str
    message: str
    result: Optional[ExtractionResult] = None
    error: Optional[str] = None

class VerificationResult(BaseModel):
    filename: str
    predicted_class: str
    confidence: float
    form_name: str
    classication_type: str
    in_training_data: bool
    training_similarity: float
    training_info: str
    extracted_text_preview: str
    processing_time: float
    method: str

class OCRResult(BaseModel):
    filename: str
    width: int
    height: int
    word_count: int
    words: List[str]
    bboxes: List[List[int]]
    text_joined: str

class FormNameResult(BaseModel):
    filename: str
    form_name: Optional[str]
    similarity: float

# ------- OCR Utilities -------
def _clean_text(s: str) -> str:
    """Clean OCR text"""
    s2 = re.sub(r"[^\w\s\-.,!?;:()'\"/]", "", s)
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2

def _sha16(s: str) -> str:
    """Generate short hash"""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def generate_fingerprint(words: List[str], boxes: List[List[int]], image_w: int, image_h: int) -> Dict[str, str]:
    """Generate fingerprint for document verification"""
    important_tokens = [w.lower() for w in words if len(w) > 3]
    anchor_hash = _sha16(" ".join(sorted(set(important_tokens))))
    grid = np.zeros((4, 4), dtype=int)
    for w, b in zip(words, boxes):
        cx = (b[0] + b[2]) / 2.0 / max(1.0, image_w)
        cy = (b[1] + b[3]) / 2.0 / max(1.0, image_h)
        gx = min(3, int(cx * 4))
        gy = min(3, int(cy * 4))
        grid[gy, gx] += 1
    layout_hash = _sha16(",".join(map(str, grid.flatten().tolist())))
    top_tokens = [t for t, _ in Counter(important_tokens).most_common(20)]
    token_hash = _sha16(",".join(top_tokens))
    return {
        "anchor_hash": anchor_hash,
        "layout_hash": layout_hash,
        "token_hash": token_hash,
    }

def fingerprint_similarity(fp1: Dict[str, str], fp2: Dict[str, str]) -> float:
    """Calculate fingerprint similarity score"""
    score = 0.0
    if fp1["anchor_hash"] == fp2["anchor_hash"]:
        score += 0.6
    if fp1["layout_hash"] == fp2["layout_hash"]:
        score += 0.3
    if fp1["token_hash"] == fp2["token_hash"]:
        score += 0.1
    return score

def ocr_extract_words_bboxes(image_path: str, conf_thresh: int = 30):
    try:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        ocr = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT,
                                        config="--psm 6 --oem 3")
        words, bboxes = [], []
        for i in range(len(ocr["text"])):
            text = str(ocr["text"][i]).strip()
            conf = float(ocr["conf"][i] or 0)
            if not text or conf < conf_thresh:
                continue
            # 1. Get raw values
            l, t, w, h = ocr["left"][i], ocr["top"][i], ocr["width"][i], ocr["height"][i]
            # 2. Calculate coordinates using direct clamping
            x1 = max(0, min(int(l), width - 1))
            y1 = max(0, min(int(t), height - 1))
            x2 = max(x1 + 1, min(int(l + w), width))
            y2 = max(y1 + 1, min(int(t + h), height))
            cleaned = _clean_text(text)
            if cleaned:
                words.append(cleaned)
                bboxes.append([x1, y1, x2, y2])
        return words, bboxes, width, height
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return [], [], 1654, 2339

def find_json_files(train_dir: str) -> List[str]:
    """Find JSON files in training directory"""
    jsons = []
    for root, _, files in os.walk(train_dir):
        for f in files:
            if f.lower().endswith(".json"):
                jsons.append(os.path.join(root, f))
    return jsons

def build_fingerprints(train_json_dir: str, out_path: str):
    """Build fingerprints from training JSON files"""
    jsons = find_json_files(train_json_dir)
    logger.info(f"Found {len(jsons)} JSON files under {train_json_dir}")
    fp_db = {}
    # Use tqdm for progress bar
    from tqdm import tqdm
    for j in tqdm(jsons, desc="Building fingerprints"):
        try:
            with open(j, "r", encoding="utf-8") as f:
                data = json.load(f)
            docs = data.get("documents", [])
            if not docs:
                continue
            doc = docs[0]
            img_info = doc.get("img", {})
            fname = next(
                (img_info[k] for k in ("fname", "filename", "file_name", "image_name") if k in img_info),
                os.path.splitext(os.path.basename(j))[0],
            )
            words = doc.get("words", [])
            bboxes = doc.get("bboxes", [])
            width = img_info.get("width") or 1000
            height = img_info.get("height") or 1000
            if not words or not bboxes:
                candidate = os.path.join(os.path.dirname(j), fname)
                # If the candidate does not exist, try adding common image extensions
                if not os.path.exists(candidate):
                    for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']:
                        candidate_with_ext = candidate + ext
                        if os.path.exists(candidate_with_ext):
                            candidate = candidate_with_ext
                            break
                if os.path.exists(candidate) and candidate.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    words, bboxes, width, height = ocr_extract_words_bboxes(candidate)
                else:
                    words, bboxes, width, height = (
                        ["document", "text"],
                        [[0, 0, 100, 20], [0, 30, 180, 60]],
                        1000,
                        1000,
                    )
            fp = generate_fingerprint(words, bboxes, int(width), int(height))
            fp_db[fname] = fp
        except Exception as e:
            logger.warning(f"Skipping {j}: {e}")
    with open(out_path, "wb") as f:
        pickle.dump(fp_db, f)
    logger.info(f"Saved fingerprints ({len(fp_db)}) to {out_path}")

def _get_installed_tesseract_langs() -> List[str]:
    """Get available Tesseract languages"""
    if not TESSERACT_AVAILABLE:
        return ["eng", "chi_sim", "chi_tra"]
    try:
        langs = pytesseract.get_languages(config="")
        if isinstance(langs, (list, tuple)) and langs:
            return list(sorted(set(langs)))
    except Exception:
        pass
    return [
        "eng",
        "spa",
        "fra",
        "deu",
        "ita",
        "por",
        "chi_sim",
        "chi_tra",
        "jpn",
        "kor",
        "rus",
        "ara",
    ]
AVAILABLE_LANGUAGES = _get_installed_tesseract_langs()

def _clean_word(word: str, language_code: str = "eng") -> str:
    """Clean word based on language"""
    if not word:
        return ""
    # For Chinese text
    if language_code.startswith(("chi", "jpn", "kor")):
        cleaned = re.sub(r'[^\u4e00-\u9fff\w\s\-\.,:/$#@%()@+\u00C0-\u017F✓✔☑√]', '', word)
    else:
        cleaned = re.sub(r'[^\w\s\-\.,:/$#@%()@+\u00C0-\u017F]', '', word)
    return cleaned.strip()

def _validate_and_normalize_langs(lang_input: Optional[str]) -> str:
    """Validate and normalize language input"""
    if not lang_input:
        return "eng"
    parts = re.split(r"[,+\s]+", lang_input.strip())
    parts = [p for p in parts if p]
    if not parts:
        return "eng"
    inst = set(AVAILABLE_LANGUAGES)
    valid = [p for p in parts if p in inst]
    if not valid:
        for p in parts:
            for ins in inst:
                if ins.startswith(p):
                    valid.append(ins)
                    break
    if not valid:
        logger.warning(f"Requested languages {parts} unavailable; falling back to 'eng'")
        return "eng"
    return "+".join(sorted(set(valid)))

def validate_bbox(bbox: Dict[str, Any]) -> bool:
    """Validate bounding box"""
    if not isinstance(bbox, dict):
        return False
    required_fields = ["x", "y", "width", "height"]
    if not all(field in bbox for field in required_fields):
        return False
    try:
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        return x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= 20000 and y + h <= 20000
    except (ValueError, TypeError):
        return False

def extract_phone_from_text(text: str) -> Optional[str]:
    """Extract phone number from text"""
    if not text:
        return None
    cleaned_text = re.sub(r"[^\w\s\+\(\)\-\.\,\/\:\=\#\@\$\%]", " ", text)
    phone_patterns = [
        r"\+?[\d\s\-()]{7,}[\d]",
        r"\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}",
        r"\d{4}[\s\-]?\d{3}[\s\-]?\d{4}",
        r"[\d\+\-\(\)\s]{7,}[\d]",
        r"\d{8,12}",
    ]
    best_match = None
    best_score = 0
    for pattern in phone_patterns:
        matches = re.finditer(pattern, cleaned_text, re.IGNORECASE)
        for match in matches:
            matched_text = match.group()
            if not matched_text:
                continue
            cleaned_match = re.sub(r"[^\d\+\-\(\)\s]", "", matched_text)
            digits = re.sub(r"\D", "", cleaned_match)
            digit_count = len(digits)
            if digit_count < 7 or digit_count > 15:
                continue
            score = digit_count / 10.0
            if "+" in matched_text:
                score += 0.2
            if any(sep in matched_text for sep in ["-", ".", "(", ")", " "]):
                score += 0.1
            if score > best_score:
                best_score = score
                best_match = matched_text
    if best_match and best_score > 0.3:
        final_clean = re.sub(r"[^\d\+\-\(\)\s]", "", best_match).strip()
        return final_clean if final_clean else None
    return None

def clean_extracted_entities(entities: List[Dict]) -> List[Dict]:
    """Clean up extracted entities before returning"""
    cleaned = []
    for entity in entities:
        cleaned_entity = entity.copy()
        if "field" in cleaned_entity:
            field = cleaned_entity["field"]
            cleaned_field = re.sub(r'[^\w_]', '', field)
            if not cleaned_field:
                cleaned_field = "entity"
            cleaned_entity["field"] = cleaned_field
        if "value" in cleaned_entity:
            value = cleaned_entity["value"]
            words = value.split()
            unique_words = []
            for word in words:
                if word not in unique_words:
                    unique_words.append(word)
            cleaned_entity["value"] = " ".join(unique_words)
            cleaned_entity["value"] = re.sub(r'\s+', ' ', cleaned_entity["value"]).strip()
        cleaned.append(cleaned_entity)
    return cleaned

def is_checkbox_option(text: str) -> bool:
    """Check if text looks like a checkbox option"""
    if not text:
        return False
    text_lower = text.lower().strip()
    skip_patterns = [
        r"tick.*box.*only",
        r"please.*tick",
        r"appropriate.*box",
        r"only.*one.*allowed",
        r"hereby.*apply.*following",
        r"receipt.*date",
        r"receipt.*number",
        r"application.*fee",
        r"section.*[a-z]",
        r"general.*information",
        r"name.*organization",
        r"contact.*tel",
        r"email.*address",
        r"government.*hong.*kong",
        r"electricity.*ordinance",
        r"registration.*regulations",
        r"certificate.*of",
        r"generating.*facility",
    ]
    for pattern in skip_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return False
    checkbox_patterns = [
        r"^loss.*certificate.*$",
        r"^replacement.*copy.*certificate.*registration.*generating.*facility.*due.*to$",
        r"^damage.*\(.*damaged.*certificate.*must.*returned.*deletion.*\)$",
    ]
    has_checkmark = any(symbol in text for symbol in ["✓", "✔", "☑", "√"])
    for pattern in checkbox_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    has_parentheses = "(" in text and ")" in text
    content_words = ["loss", "damage", "replacement", "deletion", "returned", "certificate"]
    content_count = sum(1 for word in content_words if word in text_lower)
    text_len = len(text)
    if text_len < 20 or text_len > 300:
        return False
    return (content_count >= 2) or (has_parentheses and content_count >= 1) or has_checkmark

def find_checkbox_region(entities: List[Dict]) -> Tuple[int, int, int, int]:
    """Find the bounding box of the checkbox section"""
    checkbox_section_keywords = [
        "hereby apply for the following",
        "tick one box only",
        "please tick one box",
        "please tick in the appropriate box",
        "only one is allowed",
        "due to:",
        "reason for:",
    ]
    region_entities = []
    for e in entities:
        text = e.get("value", "").lower()
        for keyword in checkbox_section_keywords:
            if keyword in text:
                region_entities.append(e)
                logger.info(f"Found checkbox section header: '{e.get('value', '')}'")
                break
    if not region_entities:
        for e in entities:
            text = e.get("value", "").lower()
            if "replacement copy" in text or "damage (" in text:
                region_entities.append(e)
                logger.info(f"Using checkbox option as anchor: '{e.get('value', '')}'")
    if not region_entities:
        logger.warning("No checkbox region found, using entire page")
        return 0, 0, 10000, 10000
    
    min_x = min(e.get("bbox", {}).get("x", 0) for e in region_entities)
    max_x = max(e.get("bbox", {}).get("x", 0) + e.get("bbox", {}).get("width", 0)
                for e in region_entities)
    min_y = min(e.get("bbox", {}).get("y", 0) for e in region_entities)
    max_y = max(e.get("bbox", {}).get("y", 0) + e.get("bbox", {}).get("height", 0)
                for e in region_entities)
    x_padding = 100
    y_padding = 200
    return (max(0, min_x - x_padding), max(0, min_y - y_padding),
            max_x + x_padding, max_y + y_padding)

def _is_in_region(entity: Dict, region: Tuple[int, int, int, int]) -> bool:
    """Check if entity is within the specified region"""
    if not region:
        return True
    bbox = entity.get("bbox", {})
    x = bbox.get("x", 0)
    y = bbox.get("y", 0)
    width = bbox.get("width", 0)
    height = bbox.get("height", 0)
    min_x, min_y, max_x, max_y = region
    return not (x + width < min_x or x > max_x or
                y + height < min_y or y > max_y)

def merge_entities(entities: List[Dict]) -> List[Dict]:
    """Merge overlapping or adjacent entities with improved validation"""
    if not entities:
        return []
    
    logger.info("=" * 80)
    logger.info("ALL EXTRACTED ENTITIES (RAW):")
    logger.info("=" * 80)
    for idx, e in enumerate(entities):
        text = e.get("value", "")
        logger.info(f"Entity {idx}: '{text}'")
    logger.info("=" * 80)
    logger.info("PHASE 1: Pre-merge phone extraction")
    logger.info("=" * 80)
    
    for e in entities:
        if "value" in e and e["value"]:
            e["value"] = e["value"].strip()
            words = e["value"].split()
            unique_words = []
            for word in words:
                if word not in unique_words:
                    unique_words.append(word)
            e["value"] = " ".join(unique_words)
    
    phone_entities = []
    for idx, e in enumerate(entities):
        text = e.get("value", "")
        phone = extract_phone_from_text(text)
        if phone:
            logger.info(f"  ✓ FOUND PHONE in entity #{idx}: {phone}")
            phone_entities.append(
                {
                    "field": f"phone_{len(phone_entities)+1}",
                    "value": phone,
                    "bbox": e.get("bbox", {}).copy(),
                    "confidence": max(0.85, e.get("confidence", 0.7)),
                    "page_number": e.get("page_number", 1),
                    "semantic_type": "phone_number",
                    "extracted_from": text[:100],
                    "extraction_phase": "pre_merge",
                }
            )
    
    by_page = defaultdict(list)
    for e in entities:
        by_page[e["page_number"]].append(e)
    
    merged = []
    for page, page_entities in by_page.items():
        page_entities.sort(key=lambda e: (e["bbox"]["y"], e["bbox"]["x"]))
        current = None
        for e in page_entities:
            if current is None:
                current = e.copy()
                continue
            
            curr_bbox = current["bbox"]
            e_bbox = e["bbox"]
            y_diff = abs(e_bbox["y"] - curr_bbox["y"])
            x_diff = e_bbox["x"] - (curr_bbox["x"] + curr_bbox["width"])
            
            curr_has_phone = extract_phone_from_text(current["value"])
            e_has_phone = extract_phone_from_text(e["value"])
            curr_is_label = any(
                pattern in current["value"].lower()
                for pattern in ["tel", "phone", "contact", "铬", "?l", ":"]
            )
            e_is_label = any(
                pattern in e["value"].lower()
                for pattern in ["tel", "phone", "contact", "铬", "?l", ":"]
            )
            curr_is_checkbox = is_checkbox_option(current["value"])
            e_is_checkbox = is_checkbox_option(e["value"])
            
            should_not_merge = (
                curr_has_phone
                or e_has_phone
                or (curr_is_label and not e_is_label)
                or curr_is_checkbox
                or e_is_checkbox
                or y_diff >= 15
                or x_diff >= 50
            )
            
            if should_not_merge:
                merged.append(current)
                current = e.copy()
            else:
                current_words = current["value"].split()
                new_words = e["value"].split()
                for word in new_words:
                    if word not in current_words:
                        current_words.append(word)
                current["value"] = " ".join(current_words)
                
                new_x = min(curr_bbox["x"], e_bbox["x"])
                new_y = min(curr_bbox["y"], e_bbox["y"])
                new_width = max(
                    curr_bbox["x"] + curr_bbox["width"],
                    e_bbox["x"] + e_bbox["width"],
                ) - new_x
                new_height = max(
                    curr_bbox["y"] + curr_bbox["height"],
                    e_bbox["y"] + e_bbox["height"],
                ) - new_y
                
                # Validate and clamp the new height
                MAX_HEIGHT = 120  # Reasonable max height for merged text
                if new_height > MAX_HEIGHT:
                    logger.warning(f"Clamping merged height from {new_height} to {MAX_HEIGHT}")
                    new_height = MAX_HEIGHT
                
                current["bbox"] = {
                    "x": new_x,
                    "y": new_y,
                    "width": new_width,
                    "height": new_height,
                }
        
        if current:
            merged.append(current)
    
    merged.extend(phone_entities)
    entity_counter = 1
    for m in merged:
        if not m.get("field") or m.get("field", "").startswith("pre_merge_"):
            field_name = f"entity_{entity_counter}"
            m["field"] = field_name
            entity_counter += 1
        else:
            current_field = m.get("field", "")
            cleaned_field = re.sub(r'[^\w_]', '', current_field)
            if not cleaned_field or len(cleaned_field) == 1:
                cleaned_field = f"entity_{entity_counter}"
                entity_counter += 1
            m["field"] = cleaned_field
        if "value" in m:
            m["value"] = m["value"].strip()
            m["value"] = re.sub(r'\s+', ' ', m["value"])
    
    logger.info(f"Merge complete: {len(merged)} entities ({len(phone_entities)} phones)")
    merged = clean_extracted_entities(merged)
    return merged

def is_valid_bbox(bbox: Dict[str, Any], img_width: int, img_height: int, idx: int = 0) -> Tuple[bool, str]:
    """Validate bounding box with detailed error messages"""
    if not isinstance(bbox, dict):
        return False, "not a dictionary"
    
    x = bbox.get("x", 0)
    y = bbox.get("y", 0)
    w = bbox.get("width", 0)
    h = bbox.get("height", 0)
    
    # Check for negative values
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        return False, f"negative/zero dimensions (x={x}, y={y}, w={w}, h={h})"
    
    # Check if bbox is completely outside the image (with generous padding)
    if x > img_width + 500 or y > img_height + 500 or x + w < -500 or y + h < -500:
        return False, f"completely outside image bounds (x={x}, y={y}, w={w}, h={h}, img={img_width}x{img_height})"
    
    # Check if bbox is too large (more than 95% of image dimensions)
    if w > img_width * 0.95 or h > img_height * 0.95:
        return False, f"too large (w={w}, h={h}, img={img_width}x{img_height})"
    
    # Check if bbox is partially outside with more than 50% outside
    visible_width = min(x + w, img_width) - max(x, 0)
    visible_height = min(y + h, img_height) - max(y, 0)
    visible_area = visible_width * visible_height
    total_area = w * h
    
    if visible_area < total_area * 0.3:  # Less than 30% visible
        return False, f"mostly outside image ({visible_area/total_area:.1%} visible)"
    
    return True, "valid"

def clamp_bbox_to_image(bbox: Dict[str, Any], img_width: int, img_height: int) -> Dict[str, Any]:
    """Clamp bounding box coordinates to be within image boundaries"""
    x = max(0, min(bbox.get("x", 0), img_width - 1))
    y = max(0, min(bbox.get("y", 0), img_height - 1))
    w = max(1, min(bbox.get("width", 1), img_width - x))
    h = max(1, min(bbox.get("height", 1), img_height - y))
    
    # Ensure reasonable minimum dimensions
    w = max(w, 1)
    h = max(h, 1)
    
    return {
        "x": x,
        "y": y,
        "width": w,
        "height": h
    }

def _extract_text_multipage(
    file_path: str, languages: str = "eng", conf_threshold: float = 0.15
) -> Tuple[List[Dict], str, int, List[Image.Image]]:
    """Extract text from multi-page document with robust bbox validation"""
    entities: List[Dict] = []
    full_text_parts: List[str] = []
    ext = os.path.splitext(file_path)[1].lower()
    images: List[Image.Image] = []
    clamped_count = 0
    validation_stats = {"total": 0, "valid": 0, "filtered": 0, "reasons": defaultdict(int)}

    try:
        if ext == ".pdf":
            if not PDF2IMAGE_AVAILABLE:
                raise RuntimeError("pdf2image not available")
            poppler_path = os.environ.get("POPPLER_PATH")
            kwargs = {"poppler_path": poppler_path} if poppler_path else {}
            images = convert_from_path(file_path, dpi=300, **kwargs)
        else:
            img = Image.open(file_path).convert("RGB")
            img = ImageOps.exif_transpose(img)
            # Apply gentle enhancement - avoid over-processing
            gray = img.convert("L")
            # Use milder enhancement factors
            enhanced = ImageEnhance.Contrast(gray).enhance(1.8)  # Reduced from 2.5
            enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.5)  # Reduced from 2.0
            # Skip median filter for now - it can blur text positions
            images = [enhanced]
    except Exception as e:
        logger.exception("Failed to convert file to images: %s", e)
        return [], "", 0, []

    global_idx = 0
    logger.info("STARTING OCR EXTRACTION")

    if EASYOCR_AVAILABLE:
        try:
            gpu_available = CUDA_AVAILABLE and torch.cuda.is_available()
            langs = ["ch_sim", "en"] if "chi_sim" in languages else ["en"]
            reader = easyocr.Reader(langs, gpu=gpu_available, verbose=False, detector=True)

            for page_idx, img in enumerate(images):
                page_num = page_idx + 1
                img_width, img_height = img.size
                logger.info(f"Processing page {page_num} ({img_width}x{img_height}) with EasyOCR...")

                try:
                    # Get both word-level and paragraph-level results for redundancy
                    word_results = reader.readtext(
                        np.array(img), detail=1, paragraph=False, min_size=5, contrast_ths=0.1
                    )
                    
                    # Add paragraph results for context (optional)
                    para_results = reader.readtext(
                        np.array(img), detail=1, paragraph=True, min_size=15, contrast_ths=0.1
                    )
                    
                    all_results = word_results + para_results

                    for res_idx, res in enumerate(all_results):
                        try:
                            if len(res) == 3:  # (bbox, text, confidence)
                                bbox, text, conf = res
                            elif len(res) == 2:  # (bbox, text)
                                bbox, text = res
                                conf = 0.8
                            else:
                                continue
                        except Exception as e:
                            logger.warning(f"Result parsing failed: {e}")
                            continue

                        if conf < conf_threshold:
                            continue
                        if not text.strip():
                            continue

                        # ✅ ROBUST BBOX CALCULATION - FIXED VERSION
                        try:
                            # Convert polygon coordinates to proper rectangle
                            polygon = [(float(p[0]), float(p[1])) for p in bbox]
                            
                            # Get min/max coordinates from polygon
                            xs = [p[0] for p in polygon]
                            ys = [p[1] for p in polygon]
                            
                            left = max(0, min(xs))
                            top = max(0, min(ys))
                            right = min(img_width - 1, max(xs))
                            bottom = min(img_height - 1, max(ys))
                            
                            # Ensure valid rectangle
                            if right <= left:
                                right = left + 1
                            if bottom <= top:
                                bottom = top + 1
                            
                            min_x = int(round(left))
                            min_y = int(round(top))
                            bbox_width = int(round(right - left))
                            bbox_height = int(round(bottom - top))
                            
                            # Final validation and clamping
                            if bbox_width <= 0 or bbox_height <= 0:
                                logger.debug(f"Invalid bbox dimensions: {bbox_width}x{bbox_height}, skipping")
                                continue
                            
                            # Clamp to reasonable heights
                            MAX_HEIGHT = 100  # Reasonable max height for text
                            if bbox_height > MAX_HEIGHT:
                                bbox_height = MAX_HEIGHT
                                bottom = top + bbox_height
                                clamped_count += 1
                            
                            # Validate coordinates are within image bounds
                            min_x = max(0, min(min_x, img_width - 1))
                            min_y = max(0, min(min_y, img_height - 1))
                            bbox_width = max(1, min(bbox_width, img_width - min_x))
                            bbox_height = max(1, min(bbox_height, img_height - min_y))

                        except Exception as e:
                            logger.warning(f"BBox calculation failed for text '{text}': {e}")
                            continue

                        cleaned = _clean_word(text, language_code=languages.split("+")[0])
                        if not cleaned:
                            continue
                            
                        if "✓" in cleaned:
                            logger.warning(f"Found checkmark: '{text}' -> '{cleaned}'")
                            cleaned = cleaned.replace("✓", "").strip()

                        entities.append({
                            "field": f"word_{global_idx+1}",
                            "value": cleaned,
                            "bbox": {
                                "x": min_x,
                                "y": min_y,
                                "width": bbox_width,
                                "height": bbox_height
                            },
                            "confidence": float(conf),
                            "page_number": page_num,
                            "raw_text": text,
                            "ocr_method": "easyocr",
                        })
                        full_text_parts.append(cleaned)
                        global_idx += 1
                        
                        # Debug logging for large bboxes
                        if bbox_height > 80:
                            logger.debug(f"Large bbox detected: {bbox_height}px - '{text}' at ({min_x},{min_y})")
                            
                except Exception as e:
                    logger.warning(f"EasyOCR failed for page {page_num}: {e}")
                    continue
        except Exception as e:
            logger.warning(f"EasyOCR initialization failed: {e}")

    # Fallback to Tesseract with PROPER bbox validation
    if (not entities or len(entities) < 5) and TESSERACT_AVAILABLE:
        logger.info("Falling back to Tesseract with proper bbox validation")
        
        for page_idx, img in enumerate(images):
            page_num = page_idx + 1
            try:
                # Use Tesseract with better config
                ocr_data = pytesseract.image_to_data(
                    img, 
                    output_type=pytesseract.Output.DICT,
                    config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,:;!?()[]{}-_$#@%&*+=/\\"\'\\s'
                )
                
                for i in range(len(ocr_data['text'])):
                    text = ocr_data['text'][i].strip()
                    conf = float(ocr_data['conf'][i]) if ocr_data['conf'][i] != '-1' else 0.0
                    
                    if not text or conf < conf_threshold * 100:  # Tesseract uses 0-100 scale
                        continue
                    
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]
                    
                    # Validate and clamp coordinates
                    img_width, img_height = img.size
                    x = max(0, min(x, img_width - 1))
                    y = max(0, min(y, img_height - 1))
                    w = max(1, min(w, img_width - x))
                    h = max(1, min(h, img_height - y))
                    
                    # Clamp height
                    MAX_HEIGHT = 100
                    if h > MAX_HEIGHT:
                        h = MAX_HEIGHT
                        clamped_count += 1
                    
                    cleaned = _clean_word(text, language_code=languages.split("+")[0])
                    if not cleaned:
                        continue
                    
                    entities.append({
                        "field": f"tess_{global_idx+1}",
                        "value": cleaned,
                        "bbox": {"x": x, "y": y, "width": w, "height": h},
                        "confidence": conf / 100.0,  # Normalize to 0-1
                        "page_number": page_num,
                        "ocr_method": "tesseract"
                    })
                    full_text_parts.append(cleaned)
                    global_idx += 1
                    
            except Exception as e:
                logger.warning(f"Tesseract failed for page {page_num}: {e}")

    logger.info(f"Total entities BEFORE merge: {len(entities)}")
    
    # ✅ ROBUST SPATIAL VALIDATION BEFORE MERGING
    if entities:
        # Get image dimensions for validation
        img_width, img_height = images[0].size if images else (1654, 2339)
        
        # First pass: validate and clamp bboxes
        validated_entities = []
        for e in entities:
            bbox = e.get("bbox", {})
            validation_stats["total"] += 1
            
            # Validate bbox
            is_valid, reason = is_valid_bbox(bbox, img_width, img_height)
            
            if is_valid:
                validation_stats["valid"] += 1
                validated_entities.append(e)
            else:
                validation_stats["filtered"] += 1
                validation_stats["reasons"][reason] += 1
                logger.debug(f"Filtered entity '{e.get('value', '')}' - {reason}")
                
                # Try to recover by clamping
                clamped_bbox = clamp_bbox_to_image(bbox, img_width, img_height)
                # Check if clamped bbox is valid now
                is_valid_after_clamp, _ = is_valid_bbox(clamped_bbox, img_width, img_height)
                if is_valid_after_clamp:
                    e_clamped = e.copy()
                    e_clamped["bbox"] = clamped_bbox
                    validated_entities.append(e_clamped)
                    logger.debug(f"Recovered entity by clamping: '{e.get('value', '')}'")
        
        # Log validation statistics
        if validation_stats["filtered"] > 0:
            logger.info(f"Entity validation results: {validation_stats['valid']}/{validation_stats['total']} valid, {validation_stats['filtered']} filtered")
            for reason, count in validation_stats["reasons"].items():
                logger.info(f"  - {count} entities filtered due to: {reason}")
        
        # Only proceed with validated entities
        if validated_entities:
            entities = validated_entities
            entities = merge_entities(entities)
        else:
            logger.warning("No valid entities after validation, keeping original entities")
            entities = merge_entities(entities)
    
    full_text = " ".join(full_text_parts)
    logger.info(f"Final count AFTER merge: {len(entities)}, {clamped_count} bboxes clamped")
    
    return entities, full_text, len(images), images

# ------- LiLT Relation Extractor -------
class LiLTRelationExtractor:
    def __init__(self, model_path: str, config: LiLTConfig, device: Optional[int] = None):
        self.model_path = model_path
        self.config = config
        self.device = device if device is not None else (
            0 if (TORCH_AVAILABLE and torch.cuda.is_available()) else -1
        )
        self.available = False
        if not (LILT_AVAILABLE and TORCH_AVAILABLE):
            logger.warning("LiLT model or torch not available")
            return
        try:
            logger.info(f"Loading LiLT from: {model_path}")
            try:
                encoder = AutoModel.from_pretrained(model_path)
                logger.info("Loaded encoder from model path")
            except Exception as e1:
                logger.warning(f"Failed to load encoder from model path: {e1}")
                try:
                    encoder = AutoModel.from_pretrained("SCUT-DLVCLab/lilt-roberta-en-base")
                    logger.info("Loaded default LiLT encoder")
                except Exception as e2:
                    logger.warning(f"Failed to load default encoder: {e2}")
                    encoder_config = AutoConfig.from_pretrained("roberta-base")
                    encoder = AutoModel.from_config(encoder_config)
                    logger.info("Created encoder from config")
            self.model = LiLTRobertaLikeForRelationExtraction(
                encoder=encoder, num_rel_labels=config.num_rel_labels
            )
            logger.info(f"Created LiLT model with num_rel_labels={config.num_rel_labels}")
            model_file = self._find_model_file(model_path)
            if model_file:
                logger.info(f"Attempting to load LiLT weights from: {model_file}")
                try:
                    state_obj = torch.load(model_file, map_location="cpu", weights_only=False)
                    if isinstance(state_obj, dict) and all(
                        isinstance(k, str) for k in state_obj.keys()
                    ):
                        if "state_dict" in state_obj:
                            state_dict = state_obj["state_dict"]
                        elif "model" in state_obj and isinstance(state_obj["model"], dict):
                            state_dict = state_obj["model"]
                        else:
                            state_dict = state_obj
                        model_dict = self.model.state_dict()
                        filtered = {
                            k: v
                            for k, v in state_dict.items()
                            if k in model_dict and hasattr(v, "shape")
                            and v.shape == model_dict[k].shape
                        }
                        self.model.load_state_dict(filtered, strict=False)
                        logger.info(f"Loaded {len(filtered)} LiLT weights")
                    else:
                        logger.warning(
                            "Checkpoint does not look like a plain state_dict; skipping head loading."
                        )
                except (pickle.UnpicklingError, RuntimeError, EOFError) as e:
                    logger.warning(
                        f"LiLT checkpoint at '{model_file}' is not a valid torch state_dict; "
                        f"using randomly initialized head. Error: {e}"
                    )
                except Exception as e:
                    logger.warning(f"Unexpected error while loading LiLT weights: {e}")
            else:
                logger.info("No LiLT head weights file found; using initialized model")
            device_name = f"cuda:{self.device}" if self.device >= 0 else "cpu"
            self.model = self.model.to(device_name)
            self.model.eval()
            self.available = True
            logger.info(f"LiLT model ready on {device_name}")
        except Exception as e:
            logger.error(f"Failed to initialize LiLT model: {e}")
            self.available = False
    def _find_model_file(self, model_path: str) -> Optional[str]:
        candidate_files = [
            "re_head.pt",
            "re_head.bin",
            "pytorch_model.bin",
            "model.pt",
            "model.bin",
        ]
        for fname in candidate_files:
            path = os.path.join(model_path, fname)
            if os.path.isfile(path) and os.path.getsize(path) > 1024:
                logger.info(
                    f"Found candidate LiLT checkpoint: {path} (size={os.path.getsize(path)})"
                )
                return path
        return None
    def is_available(self) -> bool:
        return self.available

# ------- Enhanced QA Model -------
class EnhancedQAModel:
    def __init__(self, model_name: str, device: Optional[int] = None):
        self.model_name = model_name
        self.device = device if device is not None else (
            0 if (TORCH_AVAILABLE and torch.cuda.is_available()) else -1
        )
        self.qa_pipeline = None
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available")
            return
        try:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            logger.info(f"Loading QA model: {model_name}")
            self.qa_pipeline = pipeline(
                "question-answering",
                model=model_name,
                tokenizer=model_name,
                device=self.device if self.device >= 0 else -1,
                batch_size=8,
            )
            logger.info("QA model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load QA model: {e}")
            self.qa_pipeline = None
    def is_available(self) -> bool:
        return self.qa_pipeline is not None

# ------- Verifier: Fingerprint + LILT hybrid -------
class Verifier:
    def __init__(self, fingerprints_path: Optional[str], lilt_classifier_model_path: Optional[str] = None, fp_threshold: float = 0.85):
        self.fp_threshold = fp_threshold
        self.fingerprints = {}
        if fingerprints_path and os.path.exists(fingerprints_path):
            try:
                with open(fingerprints_path, "rb") as f:
                    self.fingerprints = pickle.load(f)
                if not isinstance(self.fingerprints, dict):
                    self.fingerprints = dict(self.fingerprints)
                logger.info(f"Loaded {len(self.fingerprints)} fingerprints")
            except Exception as e:
                logger.warning(f"Failed to load fingerprints: {e}")
        else:
            logger.info("No fingerprints loaded")
        # LILT classifier
        self.classifier = None
        self.tokenizer = None
        self.device = torch.device("cuda" if (TORCH_AVAILABLE and torch is not None and torch.cuda.is_available()) else "cpu")
        if lilt_classifier_model_path and os.path.exists(lilt_classifier_model_path) and LILT_AVAILABLE:
            try:
                config = LiLTRobertaLikeConfig.from_pretrained(lilt_classifier_model_path)
                config.trust_remote_code = True
                self.tokenizer = AutoTokenizer.from_pretrained(
                    lilt_classifier_model_path,
                    use_fast=True,
                    config=config,
                )
                self.classifier = LiLTRobertaLikeForTokenClassification.from_pretrained(
                    lilt_classifier_model_path,
                    config=config,
                    trust_remote_code=True,
                ).to(self.device)
                logger.info("Loaded custom LILT classifier")
            except Exception as e:
                logger.warning(f"Failed to load LILT classifier: {e}")
                self.classifier = None
                self.tokenizer = None
    def _get_first_page_image(self, path: str) -> Optional[str]:
        """Convert first page of PDF to a temporary PNG and return its path, or return original path for images."""
        ext = os.path.splitext(path)[1].lower()
        if ext != ".pdf":
            return path
        if not PDF2IMAGE_AVAILABLE:
            logger.warning("pdf2image not available; cannot process PDF for verification")
            return None
        try:
            poppler_path = os.environ.get("POPPLER_PATH")
            kwargs = {"poppler_path": poppler_path} if poppler_path else {}
            pages = convert_from_path(path, dpi=200, **kwargs)
            if not pages:
                logger.warning(f"No pages converted from PDF: {path}")
                return None
            tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            pages[0].save(tmp_img.name, format="PNG")
            tmp_img.close()
            return tmp_img.name
        except Exception as e:
            logger.warning(f"Failed to convert PDF to image for verification: {e}")
            return None
    def verify(self, image_path: str) -> Dict:
        t0 = time.time()
        # Handle PDFs by converting first page to image
        processed_path = self._get_first_page_image(image_path)
        if processed_path is None:
            # Fallback: no OCR possible, return dummy result
            return {
                "filename": os.path.basename(image_path),
                "predicted_class": "Unknown",
                "confidence": 0.0,
                "form_name": "Unknown",
                "in_training_data": False,
                "training_similarity": 0.0,
                "training_info": "No OCR available for given file",
                "extracted_text_preview": "",
                "processing_time": time.time() - t0,
                "method": "hybrid",
            }
        try:
            words, boxes, w, h = ocr_extract_words_bboxes(processed_path)
        finally:
            # Delete temporary image if created
            if processed_path != image_path and os.path.exists(processed_path):
                try:
                    os.remove(processed_path)
                except Exception:
                    pass
        preview = " ".join(words)[:400]
        query_fp = generate_fingerprint(words, boxes, w, h)
        best_fp_score = 0.0
        best_fp_name = None
        for fname, fp in self.fingerprints.items():
            try:
                score = fingerprint_similarity(query_fp, fp)
            except Exception:
                continue
            if score > best_fp_score:
                best_fp_score = score
                best_fp_name = fname
        if best_fp_score >= self.fp_threshold:
            label, conf = self._predict_label(preview)
            return {
                "filename": os.path.basename(image_path),
                "predicted_class": label,
                "confidence": conf,
                "form_name": best_fp_name,
                "in_training_data": True,
                "training_similarity": float(best_fp_score),
                "training_info": f"Fingerprint match: {best_fp_name}",
                "extracted_text_preview": preview,
                "processing_time": time.time() - t0,
                "method": "fingerprint",
            }
        label, conf = self._predict_label(preview)
        return {
            "filename": os.path.basename(image_path),
            "predicted_class": label,
            "confidence": conf,
            "form_name": best_fp_name if best_fp_name else "Unknown",
            "in_training_data": False,
            "training_similarity": float(best_fp_score),
            "training_info": "Fingerprint weak, LILT embedding used",
            "extracted_text_preview": preview,
            "processing_time": time.time() - t0,
            "method": "hybrid",
        }
    def _predict_label(self, text: str):
        if self.classifier is None or self.tokenizer is None:
            return "Unknown", 0.0
        try:
            enc = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
                return_special_tokens_mask=True
            ).to(self.device)
            input_ids = enc['input_ids']
            batch_size, seq_len = input_ids.shape
            dummy_bboxes = torch.zeros((batch_size, seq_len, 4), dtype=torch.long).to(self.device)
            special_tokens_mask = enc.get('special_tokens_mask', torch.zeros_like(input_ids))
            dummy_bboxes[special_tokens_mask == 0] = torch.tensor([100, 100, 200, 200], dtype=torch.long).to(self.device)
            with torch.no_grad():
                outputs = self.classifier(
                    input_ids=input_ids,
                    attention_mask=enc['attention_mask'],
                    bbox=dummy_bboxes
                )
                logits = outputs.logits.detach().cpu().numpy()
            if logits.ndim == 3:
                logits = logits[:, 0, :] if logits.shape[1] > 0 else logits.mean(axis=1)
            elif logits.ndim == 2:
                pass
            else:
                raise ValueError(f"Unexpected logits shape: {logits.shape}")
            probs = np.squeeze(
                np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
            )
            if probs.ndim == 0:
                idx = 0
                score = float(probs)
            else:
                idx = int(np.argmax(probs))
                score = float(probs[idx])
            label = (
                self.classifier.config.id2label.get(idx, f"class_{idx}")
                if hasattr(self.classifier.config, "id2label")
                else f"class_{idx}"
            )
            return label, score
        except Exception as e:
            logger.warning(f"Classifier prediction failed: {e}")
            return "Unknown", 0.0
    def extract_form_name_only(self, image_path: str) -> Dict:
        """Extract only form name using fingerprints"""
        words, boxes, w, h = ocr_extract_words_bboxes(image_path)
        query_fp = generate_fingerprint(words, boxes, w, h)
        best_fp = None
        best_score = 0.0
        for fname, fp in self.fingerprints.items():
            try:
                score = fingerprint_similarity(query_fp, fp)
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_fp = fname
        return {
            "filename": os.path.basename(image_path),
            "form_name": best_fp,
            "similarity": float(best_score),
        }

# ------- Form Field Detector (phone + email + general key-value) -------
class FormFieldDetector:
    def __init__(
        self,
        lilt_extractor: Optional[LiLTRelationExtractor] = None,
        qa_model: Optional[EnhancedQAModel] = None,
    ):
        self.lilt_extractor = lilt_extractor
        self.qa_model = qa_model
    def _is_email_label(self, text: str) -> bool:
        if not text:
            return False
        t = text.lower()
        return any(k in t for k in ["email", "e-mail", "mail address", "電子郵件", "邮箱", "電郵"])
    def _is_phone_label(self, text: str) -> bool:
        if not text:
            return False
        t = text.lower()
        return any(k in t for k in ["phone", "tel", "telephone", "contact", "電話", "手机", "聯絡", "铬", "?l"])
    def _extract_email(self, text: str) -> Optional[str]:
        if not text:
            return None
        patterns = [
            r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
            r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+",
            r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9]+",
            r"\S+@\S+",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if '@' in match and len(match) > 3:
                    email = match.strip()
                    #if not any(email.endswith(ext) for ext in ['.com', '.net', '.org', '.edu', '.gov', '.io', '.co', '.hk']):
                    #    return email
                    return email
        return None
    def has_excessive_spaces(self, text: str) -> bool:
        return " " not in text
    def _normalize_text_for_matching(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[:;.]+', ':', text)
        text = re.sub(r'[`\'"~]', '', text)
        text = text.replace('(', ' (')
        text = text.replace(')', ') ')
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    def _find_label_entity(self, entities: List[Dict], label_text: str) -> Optional[Dict]:
        if not label_text:
            return None
        if len(label_text.strip()) == 1 and label_text.strip() in "✓✔☑√":
            logger.warning(f"Skipping single symbol label: '{label_text}'")
            return None
        search_text = self._normalize_text_for_matching(label_text)
        search_text = re.sub(r'[✓✔☑√]', '', search_text).strip()
        strategies = [
            lambda e, s: self._normalize_text_for_matching(e.get("value", "")) == s,
            lambda e, s: s in self._normalize_text_for_matching(e.get("value", "")),
            lambda e, s: self._normalize_text_for_matching(e.get("value", "")) in s,
            lambda e, s: self._partial_match_score(
                self._normalize_text_for_matching(e.get("value", "")),
                s
            ) >= 0.8,
            lambda e, s: self._clean_for_comparison(e.get("value", "")) == self._clean_for_comparison(label_text),
        ]
        for strategy in strategies:
            for e in entities:
                if strategy(e, search_text):
                    logger.info(f"Found label using strategy {strategies.index(strategy)+1}: '{e.get('value', '')}'")
                    return e
        return None
    def _clean_for_comparison(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'[^\w\s:().\-]', '', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    def _partial_match_score(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        if not words1 or not words2:
            return 0.0
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0.0
    import re
    def _find_nearest_value_entity(self, entities: List[Dict], label_entity: Dict, label_text: str) -> Optional[Dict]:
        if not label_entity:
            return None
        label_bbox = label_entity.get("bbox", {})
        label_page = label_entity.get("page_number", 1)
        label_x = label_bbox.get("x", 0)
        label_y = label_bbox.get("y", 0)
        label_width = label_bbox.get("width", 0)
        label_height = label_bbox.get("height", 0)
        page_entities = [e for e in entities if e.get("page_number", 1) == label_page]
        page_entities.sort(key=lambda e: (e.get("bbox", {}).get("y", 0), e.get("bbox", {}).get("x", 0)))
        label_index = -1
        for i, e in enumerate(page_entities):
            if e.get("value", "") == label_entity.get("value", ""):
                label_index = i
                break
        candidates = []
        if label_index >= 0 and label_index + 1 < len(page_entities):
            next_entity = page_entities[label_index + 1]
            next_bbox = next_entity.get("bbox", {})
            next_x = next_bbox.get("x", 0)
            next_y = next_bbox.get("y", 0)
            if (next_x > label_x and
                abs(next_y - label_y) < label_height * 2 and
                abs((next_y + next_bbox.get("height", 0)) - (label_y + label_height)) < label_height * 2):
                candidates.append({
                    "entity": next_entity,
                    "distance": abs(next_x - (label_x + label_width)),
                    "type": "immediate_right",
                    "score": 1.0
                })
        for e in entities:
            if e is label_entity:
                continue
            if e.get("page_number", 1) != label_page:
                continue
            e_bbox = e.get("bbox", {})
            e_x = e_bbox.get("x", 0)
            e_y = e_bbox.get("y", 0)
            e_width = e_bbox.get("width", 0)
            e_height = e_bbox.get("height", 0)
            is_to_right = e_x > label_x
            if is_to_right:
                horizontal_dist = e_x - (label_x + label_width)
                label_center_y = label_y + label_height / 2
                e_center_y = e_y + e_height / 2
                vertical_diff = abs(e_center_y - label_center_y)
                max_vertical_diff = max(label_height, e_height) * 1.5
                vertical_score = max(0, 1 - (vertical_diff / max_vertical_diff))
                score = vertical_score * 0.7 + (1 / (horizontal_dist + 1)) * 0.3
                if horizontal_dist < 1000 and vertical_diff < label_height * 3:
                    candidates.append({
                        "entity": e,
                        "distance": horizontal_dist,
                        "type": "spatial_right",
                        "score": score,
                        "vertical_diff": vertical_diff,
                        "horizontal_dist": horizontal_dist
                    })
        for e in entities:
            if e is label_entity:
                continue
            if e.get("page_number", 1) != label_page:
                continue
            e_bbox = e.get("bbox", {})
            e_x = e_bbox.get("x", 0)
            e_y = e_bbox.get("y", 0)
            e_width = e_bbox.get("width", 0)
            e_height = e_bbox.get("height", 0)
            is_below = e_y > label_y + label_height
            if is_below:
                horizontal_alignment = abs(e_x - label_x) < label_width * 2
                if horizontal_alignment:
                    vertical_dist = e_y - (label_y + label_height)
                    score = 1 / (vertical_dist + 1)
                    candidates.append({
                        "entity": e,
                        "distance": vertical_dist,
                        "type": "below",
                        "score": score,
                        "vertical_dist": vertical_dist
                    })
        candidates.sort(key=lambda x: x["score"], reverse=True)
        logger.info(f"Top 5 candidates for label '{label_text}':")
        for i, cand in enumerate(candidates[:5]):
            logger.info(f"  {i+1}. '{cand['entity'].get('value', '')}' (type: {cand['type']}, score: {cand['score']:.3f})")
        if candidates and candidates[0]["score"] > 0.1:
            best = candidates[0]["entity"]
            logger.info(f"Selected value: '{best.get('value', '')}'")
            return best
        return None
    
    def detect_individual_checkbox_fields(self, entities: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Detect individual checkbox options as separate fields with robust validation"""
        logger.info("Starting INDIVIDUAL checkbox detection")
        
        region = find_checkbox_region(entities)
        logger.info(f"Checkbox region: {region}")
        
        region_entities = [e for e in entities if _is_in_region(e, region)]
        logger.info(f"Entities in checkbox region: {len(region_entities)}")
        
        section_headers = []
        section_header_patterns = [
            r"tick.*box.*only",
            r"please.*tick",
            r"appropriate.*box",
            r"only.*one.*allowed",
            r"hereby.*apply.*following",
        ]
        
        for e in region_entities:
            text = e.get("value", "").lower()
            for pattern in section_header_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    section_headers.append(e)
                    logger.info(f"Excluding section header: '{e.get('value', '')}'")
                    break
        
        checkbox_patterns = [
            r"loss.*of.*certificate",
            r"replacement.*copy.*certificate.*registration.*generating.*facility.*due.*to",
            r"damage.*\(.*damaged.*certificate.*must.*returned.*deletion.*\)",
            r"deletion.*of.*generating.*facility",
        ]
        
        checkbox_candidates = []
        for e in region_entities:
            if e in section_headers:
                continue
                
            value = e.get("value", "").strip()
            if not value:
                continue
            
            value_lower = value.lower()
            is_checkbox = False
            for pattern in checkbox_patterns:
                if re.search(pattern, value_lower, re.IGNORECASE):
                    is_checkbox = True
                    break
            
            if not is_checkbox:
                has_checkmark = any(symbol in value for symbol in ["✓", "✔", "☑", "√"])
                has_parentheses = "(" in value and ")" in value
                content_words = ["loss", "damage", "replacement", "deletion", "returned", "certificate"]
                content_count = sum(1 for word in content_words if word in value_lower)
                text_len = len(value)
                if (3 < content_count < 10 and text_len >= 20 and text_len <= 300):
                    is_checkbox = True
            
            if is_checkbox:
                checkbox_candidates.append(e)
                logger.info(f"Found checkbox candidate: '{value}'")
        
        # Robust deduplication
        unique_candidates = []
        seen_texts = set()
        seen_positions = set()
        
        for cand in checkbox_candidates:
            text = cand.get("value", "").strip()
            bbox = cand.get("bbox", {})
            pos_key = (bbox.get("x", 0), bbox.get("y", 0))
            
            # Skip if exact position already seen
            if pos_key in seen_positions:
                continue
                
            # Skip if exact text already seen
            if text in seen_texts:
                continue
                
            # Check for similar text (within 85% similarity)
            is_similar = False
            for seen_text in seen_texts:
                if self._partial_match_score(text, seen_text) >= 0.85:
                    is_similar = True
                    break
            
            if not is_similar:
                unique_candidates.append(cand)
                seen_texts.add(text)
                seen_positions.add(pos_key)
                logger.info(f"Added unique checkbox: '{cand.get('value', '').strip()}'")
        
        unique_candidates.sort(key=lambda e: e.get("bbox", {}).get("y", 0))
        
        individual_results = []
        filtered_entities = []
        
        for idx, cand in enumerate(unique_candidates):
            value = cand.get("value", "").strip()
            bbox = cand.get("bbox", {"x": 0, "y": 0, "width": 1, "height": 1})
            conf = float(cand.get("confidence", 0.9))
            page_num = cand.get("page_number", 1)
            
            # Clean the value
            words = value.split()
            unique_words = []
            for word in words:
                if word not in unique_words:
                    unique_words.append(word)
            cleaned_value = " ".join(unique_words)
            cleaned_value = re.sub(r'\s+', ' ', cleaned_value).strip()
            cleaned_value = re.sub(r'^[:\-.,\s]+|[:\-.,\s]+$', '', cleaned_value)
            
            # Check if it's checked
            is_checked = any(symbol in cleaned_value for symbol in ["✓", "✔", "☑", "√", "[x]", "[X]"])
            if is_checked:
                cleaned_value = re.sub(r'[✓✔☑√\[x\]X]', '', cleaned_value).strip()
            
            field_name = f"checkbox_option_{idx+1}"
            
            result = {
                "field_name": field_name,
                "value": cleaned_value,
                "structured_value": {
                    "field_name": field_name,
                    "field_type": "individual_checkbox",
                    "value": cleaned_value,
                    "confidence": conf,
                    "bbox": bbox,
                },
                "confidence": conf,
                "bbox": bbox,
                "context_entities": [],
                "extraction_method": "individual_checkbox_detection",
                "meta": {
                    "original_text": value,
                    "is_checked": is_checked,
                    "checkbox_index": idx + 1,
                    "total_checkboxes": len(unique_candidates),
                    "region_found": region != (0, 0, 10000, 10000),
                },
                "page_number": page_num
            }
            
            filtered_entity = {
                "field": field_name,
                "value": cleaned_value,
                "bbox": bbox,
                "confidence": conf,
                "page_number": page_num,
                "semantic_type": EntityTypes.ANSWER,
                "semantic_confidence": conf,
            }
            
            individual_results.append(result)
            filtered_entities.append(filtered_entity)
            
            logger.info(f"Individual checkbox {idx+1}: '{cleaned_value}' (checked: {is_checked})")
        
        if individual_results:
            logger.info(f"Found {len(individual_results)} individual checkboxes")
            return individual_results, filtered_entities
        
        logger.warning("No individual checkboxes found")
        return [], []
    
    def detect_checkbox_field(self, entities: List[Dict]) -> Tuple[Optional[Dict], List[Dict]]:
        """Detect checkbox fields with robust validation"""
        logger.info("Starting specialized checkbox detection")
        
        region = find_checkbox_region(entities)
        logger.info(f"Checkbox region: {region}")
        
        region_entities = [e for e in entities if _is_in_region(e, region)]
        logger.info(f"Entities in checkbox region: {len(region_entities)}")
        
        logger.info("=" * 80)
        logger.info("ENTITIES IN CHECKBOX REGION:")
        for idx, e in enumerate(region_entities):
            logger.info(f"  {idx}: '{e.get('value', '')}'")
        logger.info("=" * 80)
        
        section_headers = []
        section_header_patterns = [
            r"tick.*box.*only",
            r"please.*tick",
            r"appropriate.*box",
            r"only.*one.*allowed",
            r"hereby.*apply.*following",
        ]
        
        for e in region_entities:
            text = e.get("value", "").lower()
            for pattern in section_header_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    section_headers.append(e)
                    logger.info(f"Excluding section header: '{e.get('value', '')}'")
                    break
        
        checkbox_patterns = [
            r"replacement.*copy.*certificate.*registration.*generating.*facility.*due.*to",
            r"damage.*\(.*damaged.*certificate.*must.*returned.*deletion.*\)",
        ]
        
        checkbox_candidates = []
        for e in region_entities:
            if e in section_headers:
                continue
                
            value = e.get("value", "").strip()
            if not value:
                continue
            
            value_lower = value.lower()
            for pattern in checkbox_patterns:
                if re.search(pattern, value_lower, re.IGNORECASE):
                    checkbox_candidates.append(e)
                    logger.info(f"Found checkbox option via pattern '{pattern}': '{value}'")
                    break
        
        if not checkbox_candidates:
            logger.info("No candidates via pattern matching, trying general approach")
            for e in region_entities:
                if e in section_headers:
                    continue
                    
                value = e.get("value", "").strip()
                if not value:
                    continue
                
                if is_checkbox_option(value):
                    checkbox_candidates.append(e)
        
        # Robust deduplication
        unique_candidates = []
        seen_texts = set()
        seen_positions = set()
        
        for cand in checkbox_candidates:
            text = cand.get("value", "").strip()
            bbox = cand.get("bbox", {})
            pos_key = (bbox.get("x", 0), bbox.get("y", 0))
            
            # Skip if exact position already seen
            if pos_key in seen_positions:
                continue
                
            # Skip if exact text already seen
            if text in seen_texts:
                continue
                
            # Check for similar text (within 85% similarity)
            is_similar = False
            for seen_text in seen_texts:
                if self._partial_match_score(text, seen_text) >= 0.85:
                    is_similar = True
                    break
            
            if not is_similar:
                unique_candidates.append(cand)
                seen_texts.add(text)
                seen_positions.add(pos_key)
                logger.info(f"Added unique checkbox candidate: '{text}'")
        
        unique_candidates.sort(key=lambda e: e.get("bbox", {}).get("y", 0))
        
        final_candidates = []
        for cand in unique_candidates:
            value = cand.get("value", "").strip()
            
            if any(header_word in value.lower() for header_word in [
                "tick", "please", "appropriate", "allowed", "hereby", "apply", "following"
            ]):
                continue
                
            if any(option_word in value.lower() for option_word in [
                "replacement", "damage", "loss", "certificate", "deletion"
            ]):
                final_candidates.append(cand)
        
        if final_candidates:
            # Create INDIVIDUAL results for each checkbox
            individual_results = []
            filtered_entities = []
            
            for idx, cand in enumerate(final_candidates):
                value = cand.get("value", "").strip()
                bbox = cand.get("bbox", {"x": 0, "y": 0, "width": 1, "height": 1})
                conf = float(cand.get("confidence", 0.9))
                page_num = cand.get("page_number", 1)
                
                # Clean the value
                words = value.split()
                unique_words = []
                for word in words:
                    if word not in unique_words:
                        unique_words.append(word)
                cleaned_value = " ".join(unique_words)
                cleaned_value = re.sub(r'\s+', ' ', cleaned_value).strip()
                cleaned_value = re.sub(r'^[:\-.,\s]+|[:\-.,\s]+$', '', cleaned_value)
                
                # Check if it's checked
                is_checked = any(symbol in cleaned_value for symbol in ["✓", "✔", "☑", "√", "[x]", "[X]"])
                if is_checked:
                    cleaned_value = re.sub(r'[✓✔☑√\[x\]X]', '', cleaned_value).strip()
                
                field_name = f"checkbox_option_{idx+1}"
                
                result = {
                    "field_name": field_name,
                    "value": cleaned_value,
                    "structured_value": {
                        "field_name": field_name,
                        "field_type": "individual_checkbox",
                        "value": cleaned_value,
                        "confidence": conf,
                        "bbox": bbox,
                    },
                    "confidence": conf,
                    "bbox": bbox,
                    "context_entities": [],
                    "extraction_method": "checkbox_region_detection_individual",
                    "meta": {
                        "original_text": value,
                        "is_checked": is_checked,
                        "checkbox_index": idx + 1,
                        "total_options": len(final_candidates),
                        "region_found": region != (0, 0, 10000, 10000),
                    },
                    "page_number": page_num
                }
                
                filtered_entity = {
                    "field": field_name,
                    "value": cleaned_value,
                    "bbox": bbox,
                    "confidence": conf,
                    "page_number": page_num,
                    "semantic_type": EntityTypes.ANSWER,
                    "semantic_confidence": conf,
                }
                
                individual_results.append(result)
                filtered_entities.append(filtered_entity)
            
            logger.info(f"Found {len(final_candidates)} individual checkbox options")
            return individual_results, filtered_entities
        
        logger.warning("No checkbox options found after filtering")
        empty_result = {
            "field_name": "checkbox_options",
            "value": "No checkbox options detected",
            "structured_value": {
                "field_name": "checkbox_options",
                "field_type": "checkbox_group",
                "value": "No checkbox options detected",
                "confidence": 0.0,
                "bbox": {"x": 0, "y": 0, "width": 1, "height": 1},
            },
            "confidence": 0.0,
            "bbox": {"x": 0, "y": 0, "width": 1, "height": 1},
            "context_entities": [],
            "extraction_method": "none",
            "meta": {"reason": "No checkbox options found after filtering"},
        }
        return [empty_result], [empty_result]
       
    def detect_field(self, entities: List[Dict], key_field: str) -> Tuple[Union[Dict, List[Dict], None], List[Dict]]:
        logger.info(f"Detecting field: '{key_field}'")
        logger.info(f"Total entities: {len(entities)}")
        
        key_field_clean = key_field.strip().lower()
        
        # Individual checkbox detection mode
        individual_checkbox_patterns = [
            "individual_check", "individual_checkbox", "individual_checkboxes",
            "separate_check", "separate_checkbox", "separate_checkboxes",
            "checkbox_ind", "checkbox_individual", "checkbox_separate",
            "checkbox_option", "checkbox_options"
        ]
        
        if any(pattern in key_field_clean for pattern in individual_checkbox_patterns):
            logger.info("Individual checkbox detection mode activated")
            return self.detect_individual_checkbox_fields(entities)
        
        # Combined checkbox detection mode (legacy)
        if key_field_clean in ["✓", "✔", "☑", "√", "check", "checkbox", "tick"]:
            logger.info("Using combined checkbox detection (legacy mode)")
            return self.detect_checkbox_field(entities)
        
        key_field = key_field.strip()
        if not key_field:
            logger.warning("Empty key_field provided")
            empty_result = {
                "field_name": "unknown",
                "value": "Not found",
                "structured_value": {
                    "field_name": "unknown",
                    "field_type": "text",
                    "value": "Not found",
                    "confidence": 0.0,
                    "bbox": {"x": 0, "y": 0, "width": 1, "height": 1},
                },
                "confidence": 0.0,
                "bbox": {"x": 0, "y": 0, "width": 1, "height": 1},
                "context_entities": [],
                "extraction_method": "none",
                "meta": {"reason": "Empty key field"},
            }
            return empty_result, []
        
        logger.info("=" * 80)
        logger.info("ALL ENTITIES (for debugging):")
        for idx, e in enumerate(entities):
            logger.info(f"  Entity {idx}: '{e.get('value', '')}'")
        logger.info("=" * 80)
        
        phone_entities = [e for e in entities if e.get("extraction_phase") == "pre_merge"]
        if phone_entities and self._is_phone_label(key_field):
            best_phone = phone_entities[0]
            bbox = best_phone["bbox"]
            conf = float(best_phone["confidence"])
            result = {
                "field_name": key_field,
                "value": best_phone["value"],
                "structured_value": {
                    "field_name": key_field,
                    "field_type": "phone",
                    "value": best_phone["value"],
                    "confidence": conf,
                    "bbox": bbox,
                },
                "confidence": conf,
                "bbox": bbox,
                "context_entities": [],
                "extraction_method": "pre_merge_phone_extraction",
                "meta": {"extracted_from": best_phone.get("extracted_from", "")},
            }
            filtered_entities = [
                {
                    "field": key_field,
                    "value": best_phone["value"],
                    "bbox": bbox,
                    "confidence": conf,
                    "page_number": best_phone.get("page_number", 1),
                    "semantic_type": EntityTypes.ANSWER,
                    "semantic_confidence": conf,
                }
            ]
            return result, filtered_entities
        
        if self._is_email_label(key_field):
            logger.info("Using email-specific extraction logic for key_field")
            candidates = []
            for e in entities:
                val = e.get("value", "")
                if not val:
                    continue
                if self._is_email_label(val):
                    label_page = e.get("page_number", 1)
                    lb = e.get("bbox", {})
                    lx = lb.get("x", 0)
                    ly = lb.get("y", 0)
                    lheight = lb.get("height", 0)
                    for e2 in entities:
                        if e2 is e:
                            continue
                        if e2.get("page_number", 1) != label_page:
                            continue
                        b2 = e2.get("bbox", {})
                        x2 = b2.get("x", 0)
                        y2 = b2.get("y", 0)
                        h2 = b2.get("height", 0)
                        if abs(y2 - ly) <= max(10, int(0.5 * max(lheight, h2))) and x2 > lx:
                            email_val = self._extract_email(e2.get("value", ""))
                            if email_val:
                                candidates.append((email_val, e2))
            if not candidates:
                for e in entities:
                    v = e.get("value", {})
                    if self.has_excessive_spaces(v):
                        email_val = self._extract_email(e.get("value", ""))                        
                        if email_val:
                            candidates.append((email_val, e))
            if candidates:
                email_val, ent = candidates[0]
                logger.info(f"Detected email value: {email_val}")
                bbox = ent.get("bbox", {"x": 0, "y": 0, "width": 1, "height": 1})
                conf = float(ent.get("confidence", 0.9))
                result = {
                    "field_name": key_field,
                    "value": email_val,
                    "structured_value": {
                        "field_name": key_field,
                        "field_type": "email",
                        "value": email_val,
                        "confidence": conf,
                        "bbox": bbox,
                    },
                    "confidence": conf,
                    "bbox": bbox,
                    "context_entities": [],
                    "extraction_method": "regex_email_from_entities",
                    "meta": {},
                }
                filtered_entities = [
                    {
                        "field": key_field,
                        "value": email_val,
                        "bbox": bbox,
                        "confidence": conf,
                        "page_number": ent.get("page_number", 1),
                        "semantic_type": EntityTypes.ANSWER,
                        "semantic_confidence": conf,
                    }
                ]
                return result, filtered_entities
        
        logger.info(f"Attempting general key-value detection for: '{key_field}'")
        label_entity = self._find_label_entity(entities, key_field)
        if label_entity:
            logger.info(f"Found label entity: '{label_entity.get('value', '')}'")
            logger.info(f"Label entity page: {label_entity.get('page_number', 1)}")
            value_entity = self._find_nearest_value_entity(entities, label_entity, key_field)
            if value_entity:
                value_text = value_entity.get("value", "").strip()
                logger.info(f"Found value entity: '{value_text}'")
                bbox = value_entity.get("bbox", {"x": 0, "y": 0, "width": 1, "height": 1})
                conf = float(value_entity.get("confidence", 0.9))
                result = {
                    "field_name": key_field,
                    "value": value_text,
                    "structured_value": {
                        "field_name": key_field,
                        "field_type": "text",
                        "value": value_text,
                        "confidence": conf,
                        "bbox": bbox,
                    },
                    "confidence": conf,
                    "bbox": bbox,
                    "context_entities": [],
                    "extraction_method": "general_key_value",
                    "meta": {
                        "label_entity": label_entity.get("value", ""),
                        "label_confidence": label_entity.get("confidence", 0),
                    },
                }
                filtered_entities = [
                    {
                        "field": key_field,
                        "value": value_text,
                        "bbox": bbox,
                        "confidence": conf,
                        "page_number": value_entity.get("page_number", 1),
                        "semantic_type": EntityTypes.ANSWER,
                        "semantic_confidence": conf,
                    }
                ]
                return result, filtered_entities
            else:
                logger.warning(f"Found label but no value entity for: '{key_field}'")
        else:
            logger.warning(f"No label entity found for: '{key_field}'")
        
        for e in entities:
            if e.get("value", "").strip() and len(e.get("value", "").strip()) > 3:
                value_text = e.get("value", "").strip()
                if not any(marker in value_text.lower() for marker in [":", ":", "：", "is", "are", "the"]):
                    logger.info(f"Fallback: using entity as value: '{value_text}'")
                    bbox = e.get("bbox", {"x": 0, "y": 0, "width": 1, "height": 1})
                    conf = float(e.get("confidence", 0.9))
                    result = {
                        "field_name": key_field,
                        "value": value_text,
                        "structured_value": {
                            "field_name": key_field,
                            "field_type": "text",
                            "value": value_text,
                            "confidence": conf,
                            "bbox": bbox,
                        },
                        "confidence": conf,
                        "bbox": bbox,
                        "context_entities": [],
                        "extraction_method": "fallback_value",
                        "meta": {"note": "Label not found, using fallback value"},
                    }
                    filtered_entities = [
                        {
                            "field": key_field,
                            "value": value_text,
                            "bbox": bbox,
                            "confidence": conf,
                            "page_number": e.get("page_number", 1),
                            "semantic_type": EntityTypes.ANSWER,
                            "semantic_confidence": conf,
                        }
                    ]
                    return result, filtered_entities
        
        logger.warning(f"No value found for: {key_field}")
        empty_result = {
            "field_name": key_field,
            "value": "Not found",
            "structured_value": {
                "field_name": key_field,
                "field_type": "text",
                "value": "Not found",
                "confidence": 0.0,
                "bbox": {"x": 0, "y": 0, "width": 1, "height": 1},
            },
            "confidence": 0.0,
            "bbox": {"x": 0, "y": 0, "width": 1, "height": 1},
            "context_entities": [],
            "extraction_method": "none",
            "meta": {"reason": "No matching value found"},
        }
        return empty_result, []

# ------- Document Analyzer -------
class DocumentAnalyzer:
    def __init__(self, config: LiLTConfig):
        self.config = config
        self.lilt_extractor: Optional[LiLTRelationExtractor] = None
        if config.model_path and LILT_AVAILABLE and TORCH_AVAILABLE:
            try:
                device = 0 if (CUDA_AVAILABLE and torch.cuda.is_available()) else -1
                self.lilt_extractor = LiLTRelationExtractor(
                    model_path=config.model_path, config=config, device=device
                )
            except Exception as e:
                logger.error(f"LiLT init failed: {e}")
                self.lilt_extractor = None
        self.qa_model: Optional[EnhancedQAModel] = None
        if config.qa_model_path and TRANSFORMERS_AVAILABLE:
            try:
                device = 0 if (CUDA_AVAILABLE and torch.cuda.is_available()) else -1
                self.qa_model = EnhancedQAModel(
                    model_name=config.qa_model_path, device=device
                )
            except Exception as e:
                logger.error(f"QA init failed: {e}")
                self.qa_model = None
        self.field_detector = FormFieldDetector(
            lilt_extractor=self.lilt_extractor, qa_model=self.qa_model
        )
    
    def analyze_file(
        self, file_path: str, key_field: Optional[str], language_input: Optional[str]
    ) -> Dict[str, Any]:
        start_time = time.time()
        langs_norm = _validate_and_normalize_langs(language_input)
        logger.info(f"Analyzing file with languages: {langs_norm}")
        entities, full_text, page_count, _ = _extract_text_multipage(
            file_path, languages=langs_norm, conf_threshold=0.05
        )
        kf_result = None
        filtered_entities: List[Dict[str, Any]] = []
        if key_field:
            kf_result, filtered_entities = self.field_detector.detect_field(
                entities, key_field
            )
        else:
            filtered_entities = entities
        filtered_entities = clean_extracted_entities(filtered_entities)
        processing_time = time.time() - start_time
        logger.info(f"Analysis completed in {processing_time:.2f} seconds")
        return {
            "document_name": os.path.basename(file_path),
            "page_count": page_count,
            "total_entities": len(filtered_entities),
            "entities": filtered_entities,
            "key_field_result": kf_result,
            "full_text": full_text,
            "processing_time": processing_time,
            "language_used": langs_norm,
            "model_used": self.lilt_extractor is not None
            and self.lilt_extractor.is_available()
            if self.lilt_extractor
            else False,
            "qa_model_used": self.qa_model is not None
            and self.qa_model.is_available()
            if self.qa_model
            else False,
            "lilt_model_used": self.lilt_extractor is not None
            and self.lilt_extractor.is_available()
            if self.lilt_extractor
            else False,
        }

# ------- FastAPI app -------
app = FastAPI(title="Document Analysis API - Combined LiLT + Verification")
# ------- Combined endpoints -------
class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int
class ExtractedEntity(BaseModel):
    field: str
    value: str
    bbox: BoundingBox
    confidence: float
    page_number: int = 1
    semantic_type: Optional[str] = None
    semantic_confidence: Optional[float] = None
class DataField(BaseModel):
    field_name: str
    field_type: str
    value: str
    confidence: float
    bbox: BoundingBox
class KeyFieldResult(BaseModel):
    field_name: str
    value: str
    structured_value: Optional[DataField] = None
    confidence: float
    bbox: BoundingBox
    context_entities: List[ExtractedEntity] = []
    meta: Optional[Dict] = None
    extraction_method: Optional[str] = None
class ExtractionResult(BaseModel):
    document_name: str
    page_count: int
    total_entities: int
    entities: List[ExtractedEntity]
    key_field_result: Optional[Union[KeyFieldResult, List[KeyFieldResult]]] = None
    full_text_snippet: str
    processing_time: float
    language_used: str
    model_used: bool
class VerificationResult(BaseModel):
    filename: str
    predicted_class: str
    confidence: float
    form_name: str
    classification_type: str
    in_training_data: bool
    training_similarity: float
    training_info: str
    extracted_text_preview: str
    processing_time: float
    method: str
class MultiKeyFieldResult(BaseModel):
    key_field: str
    extraction: Optional[ExtractionResult] = None
    error: Optional[str] = None
class GetDataResponse(BaseModel):
    status: str
    message: str
    results: List[MultiKeyFieldResult]
    verification: Optional[VerificationResult] = None
    error: Optional[str] = None
# ------- Helper Functions -------
def _validate_and_normalize_langs(language: Optional[str]) -> str:
    """Validate and normalize language input."""
    if not language:
        return "eng"
    langs = language.lower().strip()
    lang_map = {
        "en": "eng",
        "english": "eng",
        "zh": "chi_sim",
        "chinese": "chi_sim",
        "ja": "jpn",
        "japanese": "jpn",
        "ko": "kor",
        "korean": "kor",
        "es": "spa",
        "spanish": "spa",
        "fr": "fra",
        "french": "fra",
        "de": "deu",
        "german": "deu",
    }
    return lang_map.get(langs, langs)
def _scale_bbox_to_target(raw_bbox: Dict, document_width: int, document_height: int) -> Dict:
    """Scale bbox to target 1654x2339."""
    x = float(raw_bbox.get("x", 0))
    y = float(raw_bbox.get("y", 0))
    width = float(raw_bbox.get("width", 0))
    height = float(raw_bbox.get("height", 0))
    scale_x = 1654 / max(1, document_width)
    scale_y = 2339 / max(1, document_height)
    return {
        "x": int(round(x * scale_x)),
        "y": int(round(y * scale_y)),
        "width": int(round(width * scale_x)),
        "height": int(round(height * scale_y)),
        "confidence": raw_bbox.get("confidence", 1.0)
    }
def _build_result_json_payload(
    key_fields: List[str],
    results: List[MultiKeyFieldResult],
    verification_result: Optional[VerificationResult],
    document_width: int,
    document_height: int
) -> dict:
    """Build result.json: ONE FIELD PER KEY_FIELD + PAGE NUMBER."""
    out = {
        "form_name": verification_result.form_name if verification_result else None,
        "classification_type": verification_result.classification_type if verification_result else None,
        "document_dimensions": {"width": document_width, "height": document_height},
        "target_dimensions": {"width": 1654, "height": 2339},
        "total_key_fields": len(key_fields),
        "fields": [],
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "stats": {
            "found": 0,
            "missing": 0,
            "pages": 1
        }
    }
    
    # Build lookup map
    result_map = {r.key_field: r for r in results}
    logger.info(f"Building result.json for {len(key_fields)} key fields")
    
    # Initialize sequential index counter
    next_index = 1
    
    for idx, kf in enumerate(key_fields):
        item = result_map.get(kf)
        page_number = 1
        found = False
        
        if item and item.extraction and item.extraction.key_field_result:
            kfr = item.extraction.key_field_result
            
            # Handle individual checkbox results (multiple results)
            if isinstance(kfr, list):
                for sub_idx, checkbox_result in enumerate(kfr):
                    page_number = getattr(checkbox_result, 'page_number', 1) or 1
                    raw_bbox = {
                        "x": checkbox_result.bbox.x,
                        "y": checkbox_result.bbox.y,
                        "width": checkbox_result.bbox.width,
                        "height": checkbox_result.bbox.height
                    }
                    scaled_bbox = _scale_bbox_to_target(raw_bbox, document_width, document_height)
                    
                    out["fields"].append({
                        "index": next_index,
                        "key_field": checkbox_result.field_name,
                        "field_name": checkbox_result.field_name,
                        "value": checkbox_result.value or "",
                        "confidence": round(float(checkbox_result.confidence or 0.0), 4),
                        "page_number": page_number,
                        "bbox": scaled_bbox,
                        "found": True,
                        "source": "individual_checkbox"
                    })
                    next_index += 1
                    found = True
            
            # Handle single result
            elif kfr:
                page_number = getattr(kfr, 'page_number', 1) or 1
                raw_bbox = {
                    "x": kfr.bbox.x, "y": kfr.bbox.y,
                    "width": kfr.bbox.width, "height": kfr.bbox.height
                }
                scaled_bbox = _scale_bbox_to_target(raw_bbox, document_width, document_height)
                
                out["fields"].append({
                    "index": next_index,
                    "key_field": kf,
                    "field_name": kfr.field_name or kf,
                    "value": kfr.value or "",
                    "confidence": round(float(kfr.confidence or 0.0), 4),
                    "page_number": page_number,
                    "bbox": scaled_bbox,
                    "found": True,
                    "source": "key_field_result"
                })
                next_index += 1
                found = True
        
        if not found:
            # Missing fields get placeholder
            placeholder_bbox = _scale_bbox_to_target(
                {"x": 0, "y": 0, "width": 1, "height": 1},
                document_width, document_height
            )
            out["fields"].append({
                "index": next_index,
                "key_field": kf,
                "field_name": kf,
                "value": "",
                "confidence": 0.0,
                "page_number": page_number,
                "bbox": placeholder_bbox,
                "found": False,
                "error": item.error if item and item.error else "Not detected",
                "source": "missing"
            })
            next_index += 1
    
    # Update stats (now we count based on "found" field)
    found_count = sum(1 for f in out["fields"] if f["found"])
    out["stats"]["found"] = found_count
    out["stats"]["missing"] = len(key_fields) - found_count
    
    # No need to sort by index anymore since we've maintained sequential order
    # But keep this for backward compatibility
    out["fields"].sort(key=lambda f: f["index"])
    
    logger.info(f"result.json: {found_count}/{len(key_fields)} fields found")
    return out

import csv

FORM_NAME_CSV_PATH = "form_name_mapping.csv"  # adjust path as needed

def map_form_display(raw_form_name: str, csv_path: str = FORM_NAME_CSV_PATH) -> Tuple[str, str]:
    """
    Lookup raw_form_name in CSV col1, return col2 (display_form_name) and col3 (display_classification_type).
    
    CSV format (3 columns per row):
    raw_form_name,display_form_name,display_classification_type
    
    Input: "FC0005_Test_page_1.png"
    Output: ("FORM GF2", "EMSD")
    """
    if not raw_form_name or raw_form_name == "Unknown":
        return "Unknown", "Unknown"
    
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3:
                    continue
                
                key = row[0].strip()
                form_display = row[1].strip()
                class_display = row[2].strip()
                
                # Exact match (case-sensitive for filenames)
                if key == raw_form_name.strip():
                    return form_display or raw_form_name, class_display or "Unknown"
                    
    except FileNotFoundError:
        logger.warning(f"CSV not found: {csv_path}")
    except Exception as e:
        logger.warning(f"CSV read error: {e}")
    
    # No match: return original
    return raw_form_name, "Unknown"

@app.post("/api/get_data")
async def get_data_api(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
):
    """
    Combined endpoint - RETURNS ONLY result.json content directly
    """
    tmp_path = None
    document_width = 0
    document_height = 0
    try:
        if not hasattr(app.state, "analyzer") or app.state.analyzer is None:
            raise HTTPException(503, "Analyzer not initialized")
        if not hasattr(app.state, "verifier") or app.state.verifier is None:
            raise HTTPException(503, "Verifier not initialized")
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]:
            raise HTTPException(400, "Unsupported file type")
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tf:
            content = await file.read()
            if not content:
                raise HTTPException(400, "Empty file")
            tf.write(content)
            tmp_path = tf.name
        # Get document dimensions
        try:
            ext = os.path.splitext(file.filename)[1].lower()
            if ext == ".pdf":
                if not PDF2IMAGE_AVAILABLE:
                    raise RuntimeError("pdf2image not available")
                poppler_path = os.environ.get("POPPLER_PATH")
                kwargs = {"poppler_path": poppler_path} if poppler_path else {}
                images = convert_from_path(tmp_path, dpi=300, **kwargs)
                for page_idx, img in enumerate(images):
                    page_num = page_idx + 1
                    img_width, img_height = img.size  # 🚨 CRITICAL: PAGE DIMENSIONS
                    document_width, document_height = img.size
            else:
                # Image handling with Pillow
                img = Image.open(tmp_path).convert("RGB")
                img = ImageOps.exif_transpose(img)
                document_width, document_height = img.size
            logger.info(f"Document: {document_width}x{document_height}")
        except Exception as e:
            logger.warning(f"Dimension extraction failed ({e}), using default")
            document_width, document_height = 1654, 2339
        langs_norm = _validate_and_normalize_langs(language)
        analyzer = app.state.analyzer
        verifier = app.state.verifier
        # Load key_fields from key_field.txt
        key_fields: List[str] = []
        try:
            with open("key_field.txt", "r", encoding="utf-8") as f:
                key_fields = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            logger.warning("key_field.txt not found")
            key_fields = []
        results: List[MultiKeyFieldResult] = []
        # Analyze each key_field
        for kf in key_fields:
            try:
                raw_result = analyzer.analyze_file(tmp_path, kf, language_input=langs_norm)
                # Convert entities (for API response)
                entities_model = []
                for e in raw_result.get("entities", []):
                    try:
                        bbox_data = e.get("bbox", {"x": 0, "y": 0, "width": 1, "height": 1})
                        entities_model.append(ExtractedEntity(
                            field=e.get("field", ""),
                            value=e.get("value", ""),
                            bbox=BoundingBox(**bbox_data),
                            confidence=float(e.get("confidence", 0.0)),
                            page_number=int(e.get("page_number", 1))
                        ))
                    except Exception:
                        continue
                # Convert key_field_result
                kfr_model = None
                if raw_result.get("key_field_result"):
                    kf_res = raw_result["key_field_result"]
                    try:
                        if isinstance(kf_res, list):
                            # Multiple individual checkbox results
                            kfr_list = []
                            for res in kf_res:
                                try:
                                    bbox_data = res.get("bbox", {"x": 0, "y": 0, "width": 1, "height": 1})
                                    page_num = int(res.get("page_number", 1))
                                    # Create DataField for structured_value
                                    structured_value = None
                                    if res.get("structured_value"):
                                        sv = res["structured_value"]
                                        structured_value = DataField(
                                            field_name=sv.get("field_name", res.get("field_name", "")),
                                            field_type=sv.get("field_type", "individual_checkbox"),
                                            value=sv.get("value", ""),
                                            confidence=float(sv.get("confidence", 0.0)),
                                            bbox=BoundingBox(**sv.get("bbox", bbox_data))
                                        )
                                    
                                    kfr_list.append(KeyFieldResult(
                                        field_name=res.get("field_name", ""),
                                        value=res.get("value", ""),
                                        structured_value=structured_value,
                                        confidence=float(res.get("confidence", 0.0)),
                                        bbox=BoundingBox(**bbox_data),
                                        page_number=page_num,
                                        meta=res.get("meta"),
                                        extraction_method=res.get("extraction_method")
                                    ))
                                except Exception as e:
                                    logger.warning(f"Individual checkbox conversion failed: {e}")
                                    continue
                            kfr_model = kfr_list
                        else:
                            # Single result
                            bbox_data = kf_res.get("bbox", {"x": 0, "y": 0, "width": 1, "height": 1})
                            page_num = int(kf_res.get("page_number", 1))
                            # Create DataField for structured_value
                            structured_value = None
                            if kf_res.get("structured_value"):
                                sv = kf_res["structured_value"]
                                structured_value = DataField(
                                    field_name=sv.get("field_name", kf_res.get("field_name", "")),
                                    field_type=sv.get("field_type", "text"),
                                    value=sv.get("value", ""),
                                    confidence=float(sv.get("confidence", 0.0)),
                                    bbox=BoundingBox(**sv.get("bbox", bbox_data))
                                )
                            
                            kfr_model = KeyFieldResult(
                                field_name=kf_res.get("field_name", ""),
                                value=kf_res.get("value", ""),
                                structured_value=structured_value,
                                confidence=float(kf_res.get("confidence", 0.0)),
                                bbox=BoundingBox(**bbox_data),
                                page_number=page_num,
                                meta=kf_res.get("meta"),
                                extraction_method=kf_res.get("extraction_method")
                            )
                    except Exception as e:
                        logger.warning(f"Key field result conversion failed: {e}")
                        kfr_model = None
                extraction = ExtractionResult(
                    document_name=file.filename,
                    page_count=raw_result.get("page_count", 1),
                    total_entities=len(entities_model),
                    entities=entities_model,
                    key_field_result=kfr_model,
                    full_text_snippet=raw_result.get("full_text", "")[:1000],
                    processing_time=float(raw_result.get("processing_time", 0.0)),
                    language_used=langs_norm,
                    model_used=raw_result.get("model_used", False)
                )
                results.append(MultiKeyFieldResult(key_field=kf, extraction=extraction))
            except Exception as e:
                results.append(MultiKeyFieldResult(key_field=kf, error=str(e)))
        # Run verifier
        verification_result = None
        try:
            v_res = verifier.verify(tmp_path)  
            raw_form_name = v_res.get("form_name", "Unknown")
            mapped_form_name, mapped_classication_type = map_form_display(raw_form_name)
            verification_result = VerificationResult(
                filename=v_res.get("filename", file.filename),
                predicted_class=v_res.get("predicted_class", "Unknown"),
                confidence=float(v_res.get("confidence", 0.0)),
                form_name=mapped_form_name,  # ✅ mapped via CSV
                classification_type=mapped_classication_type,
                in_training_data=bool(v_res.get("in_training_data", False)),
                training_similarity=float(v_res.get("training_similarity", 0.0)),
                training_info=v_res.get("training_info", ""),
                extracted_text_preview=v_res.get("extracted_text_preview", ""),
                processing_time=float(v_res.get("processing_time", 0.0)),
                method=v_res.get("method", "hybrid")
            )
        except Exception as e:
            logger.exception("Verification failed")
        # BUILD result.json and RETURN IT DIRECTLY
        result_json_content = _build_result_json_payload(
            key_fields=key_fields,
            results=results,
            verification_result=verification_result,
            document_width=document_width,
            document_height=document_height
        )
        # Save to disk
        try:
            with open("result.json", "w", encoding="utf-8") as f:
                json.dump(result_json_content, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved result.json: {result_json_content['stats']['found']}/{len(key_fields)}")
        except Exception as e:
            logger.warning(f"result.json save failed: {e}")
        # RETURN ONLY result.json_content
        return result_json_content
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception("get_data_api error")
        return {
            "status": "error",
            "message": "internal error",
            "form_name": None,
            "fields": [],
            "stats": {"found": 0, "missing": 0, "pages": 1}
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
# ------- Health check -------
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "analyzer_available": hasattr(app.state, "analyzer") and app.state.analyzer is not None,
        "verifier_available": hasattr(app.state, "verifier") and app.state.verifier is not None,
        "lilt_available": LILT_AVAILABLE,
        "torch_available": TORCH_AVAILABLE,
        "cuda_available": CUDA_AVAILABLE,
        "transformers_available": TRANSFORMERS_AVAILABLE,
        "tesseract_available": TESSERACT_AVAILABLE,
        "easyocr_available": EASYOCR_AVAILABLE,
        "pdf2image_available": PDF2IMAGE_AVAILABLE,
    }
    return JSONResponse(content=status)
# ------- Main -------
def main():
    parser = argparse.ArgumentParser(
        description="Document Analysis API - Combined LiLT + Verification"
    )
    # Document analyzer arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to LiLT model for relation extraction",
    )
    parser.add_argument(
        "--qa_model", type=str, default="deepset/roberta-base-squad2", help="QA model"
    )
    # Verifier arguments
    parser.add_argument(
        "--train_json_dir",
        default=None,
        help="Training JSON directory (for building fingerprints)",
    )
    parser.add_argument(
        "--generate_fingerprints",
        action="store_true",
        help="Generate fingerprints.pkl from training JSONs",
    )
    parser.add_argument(
        "--fingerprints_out", default="fingerprints.pkl", help="Output path for fingerprints"
    )
    parser.add_argument(
        "--lilt_classifier_model_path",
        default=None,
        help="LILT classifier model path for verification"
    )
    parser.add_argument(
        "--fp_threshold",
        type=float,
        default=0.85,
        help="Fingerprint similarity threshold (0..1)"
    )
    # Server arguments
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    # CLI verification arguments
    parser.add_argument("--verify", action="store_true", help="Run verification on --input")
    parser.add_argument("--input", default=None, help="Input image path for verification")
    args = parser.parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    # Generate fingerprints if requested
    if args.generate_fingerprints:
        if not args.train_json_dir:
            logger.error("--train_json_dir required for fingerprint generation")
            return
        build_fingerprints(args.train_json_dir, args.fingerprints_out)
        return
    # CLI verification if requested
    if args.verify:
        if not args.input:
            logger.error("--input required for verification")
            return
        verifier = Verifier(
            fingerprints_path=args.fingerprints_out
            if os.path.exists(args.fingerprints_out)
            else None,
            lilt_classifier_model_path=args.lilt_classifier_model_path,
            fp_threshold=args.fp_threshold,
        )
        result = verifier.verify(args.input)
        print(json.dumps(result, indent=2))
        return
    # Initialize the API
    logger.info("Starting Combined Document Analysis API...")
    # Initialize document analyzer
    config = LiLTConfig(model_path=args.model_path, qa_model_path=args.qa_model)
    app.state.analyzer = DocumentAnalyzer(config)
    # Initialize verifier
    app.state.verifier = Verifier(
        fingerprints_path=args.fingerprints_out,
        lilt_classifier_model_path=args.lilt_classifier_model_path,
        fp_threshold=args.fp_threshold,
    )
    logger.info(f"LiLT model path: {args.model_path}")
    logger.info(f"QA model: {args.qa_model}")
    logger.info(f"Fingerprints: {args.fingerprints_out}")
    logger.info(f"LILT classifier model: {args.lilt_classifier_model_path}")
    logger.info(
        f"Analyzer available: {app.state.analyzer is not None}"
    )
    logger.info(
        f"Verifier available: {app.state.verifier is not None}"
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
if __name__ == "__main__":
    main()
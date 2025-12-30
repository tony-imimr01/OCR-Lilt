#!/usr/bin/env python3
"""
Document Analysis API - LiLT + Email Extraction + Fingerprint Verification

Combined API with:
- LiLT-based document analysis and form field extraction
- OCR fingerprint + LILT embedding hybrid verification
- Email and phone number extraction
- Checkbox detection
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
    key_field_result: Optional[KeyFieldResult] = None
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


def ocr_extract_words_bboxes(image_path: str, conf_thresh: int = 30) -> Tuple[List[str], List[List[int]], int, int]:
    try:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        # Try to run Tesseract with a timeout of 30 seconds
        ocr = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config="--psm 6 --oem 3", timeout=30)

        words, bboxes = [], []
        n = len(ocr.get("text", []))

        for i in range(n):
            text = str(ocr["text"][i]).strip()
            if not text:
                continue

            try:
                conf = float(ocr["conf"][i])
            except Exception:
                conf = 0.0

            if conf < conf_thresh:
                continue

            try:
                left = int(ocr["left"][i])
                top = int(ocr["top"][i])
                w = int(ocr["width"][i])
                h = int(ocr["height"][i])
            except Exception:
                continue

            x1 = max(0, left)
            y1 = max(0, top)
            x2 = min(width, left + w)
            y2 = min(height, top + h)

            cleaned = _clean_text(text)
            if cleaned:
                words.append(cleaned)
                bboxes.append([x1, y1, x2, y2])

        if not words:
            return (
                ["document", "text"],
                [[0, 0, min(100, width), min(30, height)],
                 [0, 40, min(100, width), min(70, height)]],
                width,
                height,
            )

        return words, bboxes, width, height

    except Exception as e:
        logger.warning(f"OCR failed for {image_path}: {e}")
        return ["document", "text"], [[0, 0, 300, 150], [400, 100, 600, 200]], 1000, 1000


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
        r"掼毁.*琨有损毁澄明善必须交遣本署鞋艄.*damage",
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
    """Merge overlapping or adjacent entities"""
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


def _extract_text_multipage(
    file_path: str, languages: str = "eng", conf_threshold: float = 0.15
) -> Tuple[List[Dict], str, int, List[Image.Image]]:
    """Extract text from multi-page document"""
    entities: List[Dict] = []
    full_text_parts: List[str] = []
    ext = os.path.splitext(file_path)[1].lower()
    images: List[Image.Image] = []

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
            gray = img.convert("L")
            enhanced = ImageEnhance.Contrast(gray).enhance(2.5)
            enhanced = ImageEnhance.Sharpness(enhanced).enhance(2.0)
            enhanced = enhanced.filter(ImageFilter.MedianFilter(size=3))
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
            reader = easyocr.Reader(langs, gpu=gpu_available, verbose=False)

            for page_idx, img in enumerate(images):
                page_num = page_idx + 1
                logger.info(f"Processing page {page_num} with EasyOCR...")

                try:
                    ocr_results_words = reader.readtext(
                        np.array(img), detail=1, paragraph=False, min_size=5
                    )
                    ocr_results_para = reader.readtext(
                        np.array(img), detail=1, paragraph=True, min_size=10
                    )
                    all_results = ocr_results_words + ocr_results_para

                    for res_idx, res in enumerate(all_results):
                        try:
                            bbox, text, conf = res
                        except ValueError:
                            bbox, text = res
                            conf = 0.8

                        if conf < conf_threshold:
                            continue
                        if not text.strip():
                            continue

                        x_coords = [p[0] for p in bbox]
                        y_coords = [p[1] for p in bbox]
                        min_x, max_x = min(x_coords), max(x_coords)
                        min_y, max_y = min(y_coords), max(y_coords)

                        cleaned = _clean_word(text, language_code=languages.split("+")[0])
                        if not cleaned:
                            continue
                        
                        if "✓" in cleaned:
                            logger.warning(f"Found checkmark in OCR text: '{text}' -> '{cleaned}'")
                            cleaned = cleaned.replace("✓", "").strip()

                        entities.append(
                            {
                                "field": f"word_{global_idx+1}",
                                "value": cleaned,
                                "bbox": {
                                    "x": max(0, int(min_x)),
                                    "y": max(0, int(min_y)),
                                    "width": max(1, int(max_x - min_x)),
                                    "height": max(1, int(max_y - min_y)),
                                },
                                "confidence": float(conf),
                                "page_number": page_num,
                                "raw_text": text,
                                "ocr_method": "easyocr",
                            }
                        )
                        full_text_parts.append(cleaned)
                        global_idx += 1

                except Exception as e:
                    logger.warning(f"EasyOCR failed for page {page_num}: {e}")
                    continue
        except Exception as e:
            logger.warning(f"EasyOCR initialization failed: {e}")

    if TESSERACT_AVAILABLE and (not entities or len(entities) < 5):
        logger.info("Falling back to Tesseract (stub)")

    logger.info(f"Total entities BEFORE merge: {len(entities)}")
    if entities:
        entities = merge_entities(entities)

    full_text = " ".join(full_text_parts)
    logger.info(f"Final count AFTER merge: {len(entities)}")
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
                    if not any(email.endswith(ext) for ext in ['.com', '.net', '.org', '.edu', '.gov', '.io', '.co', '.hk']):
                        return email
                    return email
        
        return None

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

    def detect_checkbox_field(self, entities: List[Dict]) -> Tuple[Optional[Dict], List[Dict]]:
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
        
        unique_candidates = []
        seen_texts = set()
        
        for cand in checkbox_candidates:
            text = cand.get("value", "").strip()
            
            if text in seen_texts:
                continue
                
            is_duplicate = False
            seen_texts_copy = list(seen_texts)
            
            for seen in seen_texts_copy:
                if text in seen or seen in text:
                    is_duplicate = True
                    if len(text) > len(seen):
                        seen_texts.remove(seen)
                        for uc in unique_candidates:
                            if uc.get("value", "").strip() == seen:
                                unique_candidates.remove(uc)
                                break
                        is_duplicate = False
                    else:
                        break
            
            if not is_duplicate and text not in seen_texts:
                seen_texts.add(text)
                unique_candidates.append(cand)
        
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
            checkbox_values = [c.get("value", "").strip() for c in final_candidates]
            
            cleaned_values = []
            for val in checkbox_values:
                words = val.split()
                unique_words = []
                for word in words:
                    if word not in unique_words:
                        unique_words.append(word)
                val = " ".join(unique_words)
                val = re.sub(r'\s+', ' ', val).strip()
                val = re.sub(r'^[:\-.,\s]+|[:\-.,\s]+$', '', val)
                cleaned_values.append(val)
            
            combined_value = " | ".join(cleaned_values)
            
            first_entity = final_candidates[0]
            bbox = first_entity.get("bbox", {"x": 0, "y": 0, "width": 1, "height": 1})
            conf = float(first_entity.get("confidence", 0.9))
            
            filtered_entities = []
            for idx, cand in enumerate(final_candidates):
                filtered_entities.append({
                    "field": f"checkbox_option_{idx+1}",
                    "value": cand.get("value", "").strip(),
                    "bbox": cand.get("bbox", {}),
                    "confidence": float(cand.get("confidence", 0.9)),
                    "page_number": cand.get("page_number", 1),
                    "semantic_type": EntityTypes.ANSWER,
                    "semantic_confidence": float(cand.get("confidence", 0.9)),
                })
            
            result = {
                "field_name": "checkbox_options",
                "value": combined_value,
                "structured_value": {
                    "field_name": "checkbox_options",
                    "field_type": "checkbox_group",
                    "value": combined_value,
                    "confidence": conf,
                    "bbox": bbox,
                },
                "confidence": conf,
                "bbox": bbox,
                "context_entities": [],
                "extraction_method": "checkbox_region_detection",
                "meta": {
                    "total_options": len(cleaned_values),
                    "individual_options": cleaned_values,
                    "region_found": region != (0, 0, 10000, 10000),
                },
            }
            
            logger.info(f"Found {len(cleaned_values)} checkbox options: {cleaned_values}")
            return result, filtered_entities
        
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
        return empty_result, []

    def detect_field(self, entities: List[Dict], key_field: str) -> Tuple[Optional[Dict], List[Dict]]:
        logger.info(f"Detecting field: '{key_field}'")
        logger.info(f"Total entities: {len(entities)}")
        
        if key_field.strip() in ["✓", "✔", "☑", "√", "check", "checkbox", "tick"]:
            logger.info("Using specialized checkbox detection")
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
#!/usr/bin/env python3
"""
Fixed API endpoint that produces only ONE field result per key_field in result.json
"""

from typing import List, Optional, Dict
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tempfile
import os
import json
import logging

logger = logging.getLogger(__name__)

# ------- Models -------
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
    key_field_result: Optional[KeyFieldResult] = None
    full_text_snippet: str
    processing_time: float
    language_used: str
    model_used: bool

class VerificationResult(BaseModel):
    filename: str
    predicted_class: str
    confidence: float
    form_name: str
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

def validate_bbox(bbox_data: Dict) -> bool:
    """Validate bbox has required fields and valid values."""
    if not isinstance(bbox_data, dict):
        return False
    required = ["x", "y", "width", "height"]
    if not all(k in bbox_data for k in required):
        return False
    try:
        w = float(bbox_data["width"])
        h = float(bbox_data["height"])
        return w > 0 and h > 0
    except (ValueError, TypeError):
        return False

def _scale_bbox_to_target(
    bbox: Dict, 
    orig_width: int, 
    orig_height: int,
    target_width: int = 1654, 
    target_height: int = 2339
) -> Dict:
    """Scale bbox coordinates to target page size 1654x2339."""
    x = float(bbox.get("x", 0))
    y = float(bbox.get("y", 0))
    width = float(bbox.get("width", 0))
    height = float(bbox.get("height", 0))
    
    scale_x = target_width / max(1, orig_width)
    scale_y = target_height / max(1, orig_height)
   
    return {
        "x": int(round(x * scale_x)),
        "y": int(round(y * scale_y)),
        "width": int(round(width * scale_x)),
        "height": int(round(height * scale_y)),
        "confidence": bbox.get("confidence", 1.0)
    }

def _build_result_json_payload(
    results: List[MultiKeyFieldResult],
    verification_result: Optional[VerificationResult],
    document_width: int,
    document_height: int
) -> dict:
    """
    Build JSON payload for result.json with scaled bboxes.
    
    CRITICAL FIX: Only include ONE field per key_field (the key_field_result)
    NOT all entities, which was causing duplicates.
    """
    out: dict = {
        "form_name": None,
        "document_dimensions": {
            "width": document_width, 
            "height": document_height
        },
        "target_dimensions": {
            "width": 1654, 
            "height": 2339
        },
        "fields": [],
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    # Form name from verification (if any)
    if verification_result is not None:
        out["form_name"] = verification_result.form_name

    # FIXED: Only add key_field_result (master field), NOT all entities
    for item in results:
        if not item.extraction:
            continue
        
        extraction = item.extraction

        # Only process key_field_result (the main field we're looking for)
        if extraction.key_field_result is not None:
            kf = extraction.key_field_result
            
            # Scale bbox to target dimensions
            scaled_bbox = _scale_bbox_to_target(
                {
                    "x": kf.bbox.x, 
                    "y": kf.bbox.y, 
                    "width": kf.bbox.width, 
                    "height": kf.bbox.height,
                    "confidence": kf.confidence
                },
                document_width, 
                document_height
            )
            
            # Add single field entry for this key_field
            field_entry = {
                "key_field": item.key_field,
                "field_name": kf.field_name,
                "value": kf.value,
                "confidence": kf.confidence,
                "bbox": scaled_bbox,
            }
            
            # Optionally include extraction method and metadata
            if kf.extraction_method:
                field_entry["extraction_method"] = kf.extraction_method
            if kf.meta:
                field_entry["meta"] = kf.meta
            
            # If there's structured_value, include it
            if kf.structured_value:
                sv = kf.structured_value
                scaled_sv_bbox = _scale_bbox_to_target(
                    {
                        "x": sv.bbox.x,
                        "y": sv.bbox.y,
                        "width": sv.bbox.width,
                        "height": sv.bbox.height
                    },
                    document_width,
                    document_height
                )
                field_entry["structured_value"] = {
                    "field_name": sv.field_name,
                    "field_type": sv.field_type,
                    "value": sv.value,
                    "confidence": sv.confidence,
                    "bbox": scaled_sv_bbox
                }
            
            out["fields"].append(field_entry)
        else:
            # If no key_field_result found, log a warning
            logger.warning(
                f"No key_field_result found for key_field '{item.key_field}'"
            )

    return out

def _validate_and_normalize_langs(language: Optional[str]) -> str:
    """Validate and normalize language input."""
    if not language:
        return "eng"
    
    # Normalize to lowercase and remove spaces
    langs = language.lower().strip()
    
    # Map common language codes
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

# ------- API Endpoint -------

app = FastAPI()

# ------- Combined endpoints -------
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

def bboxes_overlap(bbox1: Dict, bbox2: Dict, threshold: float = 0.5) -> bool:
    x1, y1, w1, h1 = bbox1["x"], bbox1["y"], bbox1["width"], bbox1["height"]
    x2, y2, w2, h2 = bbox2["x"], bbox2["y"], bbox2["width"], bbox2["height"]

    # Intersection
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    if xi2 <= xi1 or yi2 <= yi1:
        return False

    inter_area = (xi2 - xi1) * (yi2 - yi1)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    iou = inter_area / union_area if union_area > 0 else 0
    return iou > threshold

def _build_result_json_payload(
    key_fields: List[str],
    results: List[MultiKeyFieldResult],
    verification_result: Optional[VerificationResult],
    document_width: int,
    document_height: int
) -> dict:
    out = {
        "form_name": verification_result.form_name if verification_result else None,
        "document_dimensions": {"width": document_width, "height": document_height},
        "target_dimensions": {"width": 1654, "height": 2339},
        "fields": [],
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    # Build lookup map
    result_map = {r.key_field: r for r in results}

    logger.info(f"Building result.json for {len(key_fields)} key fields")

    for idx, kf in enumerate(key_fields):
        item = result_map.get(kf)

        if item and item.extraction and item.extraction.key_field_result:
            kfr = item.extraction.key_field_result
            raw_bbox = {"x": kfr.bbox.x, "y": kfr.bbox.y,
                        "width": kfr.bbox.width, "height": kfr.bbox.height}
            scaled_bbox = _scale_bbox_to_target(raw_bbox, document_width, document_height)

            out["fields"].append({
                "key_field": kf,
                "field_name": kfr.field_name or kf,
                "value": kfr.value or "",
                "confidence": round(float(kfr.confidence or 0.0), 4),
                "bbox": scaled_bbox,
                "found": True,
                "index": idx + 1
            })
        else:
            # Always include missing ones
            placeholder = _scale_bbox_to_target(
                {"x": 0, "y": 0, "width": 1, "height": 1},
                document_width, document_height
            )
            out["fields"].append({
                "key_field": kf,
                "field_name": kf,
                "value": "",
                "confidence": 0.0,
                "bbox": placeholder,
                "found": False,
                "error": item.error if item and item.error else "Not detected",
                "index": idx + 1
            })

    logger.info(f"Generated {len(out['fields'])} fields in result.json")
    return out

@app.post("/api/get_data", response_model=GetDataResponse)
async def get_data_api(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
):
    """
    Combined endpoint:
    - Read key_field values line-by-line from key_field.txt
    - For each key_field, run DocumentAnalyzer.analyze_file
    - Also run Verifier.verify once on the same file
    - Save aggregated data into result.json with bboxes scaled to 1654x2339
    """
    tmp_path = None
    document_width = 0
    document_height = 0
    
    try:
        # Check analyzer and verifier availability
        if not hasattr(app.state, "analyzer") or app.state.analyzer is None:
            raise HTTPException(503, "Analyzer not initialized")
        if not hasattr(app.state, "verifier") or app.state.verifier is None:
            raise HTTPException(503, "Verifier not initialized")

        # Validate file type
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]:
            raise HTTPException(400, "Unsupported file type")

        # Save uploaded file to temp path
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tf:
            content = await file.read()
            if not content:
                raise HTTPException(400, "Empty file")
            tf.write(content)
            tmp_path = tf.name

        # Get document dimensions from first page (for bbox scaling)
        try:
            from PIL import Image
            img = Image.open(tmp_path)
            document_width, document_height = img.size
            logger.info(f"Document dimensions: {document_width}x{document_height}")
        except Exception:
            document_width, document_height = 1654, 2339  # fallback

        # Normalized language
        langs_norm = _validate_and_normalize_langs(language)

        analyzer: DocumentAnalyzer = app.state.analyzer
        verifier: Verifier = app.state.verifier

        # Load key_field list from key_field.txt (one per line, strip blanks)
        key_fields: List[str] = []
        try:
            with open("key_field.txt", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        key_fields.append(line)
        except FileNotFoundError:
            logger.warning("key_field.txt not found; using empty key_field list")
            key_fields = []

        results: List[MultiKeyFieldResult] = []

        # Run analysis for each key_field
        for kf in key_fields:
            try:
                result = analyzer.analyze_file(tmp_path, kf, language_input=langs_norm)

                # Convert entities
                entities_model: List[ExtractedEntity] = []
                for e in result.get("entities", []):
                    try:
                        bbox_data = e.get("bbox", {"x": 0, "y": 0, "width": 0, "height": 0})
                        if not validate_bbox(bbox_data):
                            bbox_data = {"x": 0, "y": 0, "width": 1, "height": 1}
                        entities_model.append(
                            ExtractedEntity(
                                field=e.get("field", ""),
                                value=e.get("value", ""),
                                bbox=BoundingBox(**bbox_data),
                                confidence=float(e.get("confidence", 0.0)),
                                page_number=int(e.get("page_number", 1)),
                                semantic_type=e.get("semantic_type"),
                                semantic_confidence=e.get("semantic_confidence"),
                            )
                        )
                    except Exception as exc:
                        logger.warning(
                            f"Entity conversion failed for key_field '{kf}': {exc}"
                        )
                        continue

                # Convert key_field_result, if any
                kfr_model = None
                if result.get("key_field_result"):
                    kf_res = result["key_field_result"]
                    try:
                        bbox_data = kf_res.get(
                            "bbox", {"x": 0, "y": 0, "width": 0, "height": 0}
                        )
                        if not validate_bbox(bbox_data):
                            bbox_data = {"x": 0, "y": 0, "width": 1, "height": 1}
                        structured_value = None
                        if kf_res.get("structured_value"):
                            sv = kf_res["structured_value"]
                            structured_value = DataField(
                                field_name=sv["field_name"],
                                field_type=sv["field_type"],
                                value=sv["value"],
                                confidence=float(sv["confidence"]),
                                bbox=BoundingBox(
                                    **sv.get(
                                        "bbox",
                                        {"x": 0, "y": 0, "width": 1, "height": 1},
                                    )
                                ),
                            )
                        kfr_model = KeyFieldResult(
                            field_name=kf_res["field_name"],
                            value=kf_res["value"],
                            structured_value=structured_value,
                            confidence=float(kf_res["confidence"]),
                            bbox=BoundingBox(**bbox_data),
                            context_entities=[],
                            meta=kf_res.get("meta"),
                            extraction_method=kf_res.get("extraction_method"),
                        )
                    except Exception as exc:
                        logger.warning(
                            f"Key field result conversion failed for key_field '{kf}': {exc}"
                        )
                        kfr_model = None

                extraction = ExtractionResult(
                    document_name=result.get("document_name", ""),
                    page_count=result.get("page_count", 0),
                    total_entities=result.get("total_entities", 0),
                    entities=entities_model,
                    key_field_result=kfr_model,
                    full_text_snippet=(
                        result.get("full_text", "")[:1000]
                        + ("..." if len(result.get("full_text", "")) > 1000 else "")
                    ),
                    processing_time=float(result.get("processing_time", 0.0)),
                    language_used=result.get("language_used", "eng"),
                    model_used=result.get("model_used", False),
                )

                results.append(
                    MultiKeyFieldResult(
                        key_field=kf,
                        extraction=extraction,
                        error=None,
                    )
                )
            except Exception as e:
                logger.exception("Analysis failed for key_field '%s': %s", kf, e)
                results.append(
                    MultiKeyFieldResult(
                        key_field=kf,
                        extraction=None,
                        error=str(e),
                    )
                )

        # Run verifier on the same file (once)
        verification_result: Optional[VerificationResult] = None
        try:
            v_res = verifier.verify(tmp_path)
            verification_result = VerificationResult(
                filename=v_res.get("filename", os.path.basename(file.filename)),
                predicted_class=v_res.get("predicted_class", "Unknown"),
                confidence=float(v_res.get("confidence", 0.0)),
                form_name=v_res.get("form_name", "Unknown"),
                in_training_data=bool(v_res.get("in_training_data", False)),
                training_similarity=float(v_res.get("training_similarity", 0.0)),
                training_info=v_res.get("training_info", ""),
                extracted_text_preview=v_res.get("extracted_text_preview", ""),
                processing_time=float(v_res.get("processing_time", 0.0)),
                method=v_res.get("method", "hybrid"),
            )
        except Exception as e:
            logger.exception("Verification failed: %s", e)

        # ---- Save to result.json with scaled bboxes ----
        try:
            payload = _build_result_json_payload(
                key_fields=key_fields,                  # ← NEW: First argument
                results=results,
                verification_result=verification_result,
                document_width=document_width,
                document_height=document_height
            )
            with open("result.json", "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved result.json with {len(payload['fields'])} fields (matches key_field.txt)")
        except Exception as e:
            logger.warning(f"Failed to write result.json: {e}")

        msg = f"Processed {len(results)} key_fields"
        if verification_result is not None:
            msg += f" with verification (form: {verification_result.form_name})"

        return GetDataResponse(
            status="success",
            message=msg,
            results=results,
            verification=verification_result,
            error=None,
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception("get_data_api error: %s", e)
        return GetDataResponse(
            status="error",
            message="internal error",
            results=[],
            verification=None,
            error=str(e),
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
       
@app.post("/api/extract-text", response_model=AnalysisResponse)
async def extract_text_api(
    file: UploadFile = File(...),
    key_field: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
):
    """Extract text and form fields using LiLT analysis"""
    tmp = None
    try:
        if not hasattr(app.state, "analyzer"):
            raise HTTPException(503, "Analyzer not initialized")

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]:
            raise HTTPException(400, "Unsupported file type")

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tf:
            content = await file.read()
            if not content:
                raise HTTPException(400, "Empty file")
            tf.write(content)
            tmp = tf.name

        langs_norm = _validate_and_normalize_langs(language)
        analyzer: DocumentAnalyzer = app.state.analyzer
        result = analyzer.analyze_file(tmp, key_field, language_input=langs_norm)

        entities_model: List[ExtractedEntity] = []
        for e in result.get("entities", []):
            try:
                bbox_data = e.get("bbox", {"x": 0, "y": 0, "width": 0, "height": 0})
                if not validate_bbox(bbox_data):
                    bbox_data = {"x": 0, "y": 0, "width": 1, "height": 1}
                entities_model.append(
                    ExtractedEntity(
                        field=e.get("field", ""),
                        value=e.get("value", ""),
                        bbox=BoundingBox(**bbox_data),
                        confidence=float(e.get("confidence", 0.0)),
                        page_number=int(e.get("page_number", 1)),
                        semantic_type=e.get("semantic_type"),
                        semantic_confidence=e.get("semantic_confidence"),
                    )
                )
            except Exception as exc:
                logger.warning(f"Entity conversion failed: {exc}")
                continue

        kfr_model = None
        if result.get("key_field_result"):
            kf = result["key_field_result"]
            try:
                bbox_data = kf.get("bbox", {"x": 0, "y": 0, "width": 0, "height": 0})
                if not validate_bbox(bbox_data):
                    bbox_data = {"x": 0, "y": 0, "width": 1, "height": 1}
                structured_value = None
                if kf.get("structured_value"):
                    sv = kf["structured_value"]
                    structured_value = DataField(
                        field_name=sv["field_name"],
                        field_type=sv["field_type"],
                        value=sv["value"],
                        confidence=float(sv["confidence"]),
                        bbox=BoundingBox(
                            **sv.get("bbox", {"x": 0, "y": 0, "width": 1, "height": 1})
                        ),
                    )
                kfr_model = KeyFieldResult(
                    field_name=kf["field_name"],
                    value=kf["value"],
                    structured_value=structured_value,
                    confidence=float(kf["confidence"]),
                    bbox=BoundingBox(**bbox_data),
                    context_entities=[],
                    meta=kf.get("meta"),
                    extraction_method=kf.get("extraction_method"),
                )
            except Exception as exc:
                logger.warning(f"Key field result conversion failed: {exc}")
                kfr_model = None

        extraction = ExtractionResult(
            document_name=result.get("document_name", ""),
            page_count=result.get("page_count", 0),
            total_entities=result.get("total_entities", 0),
            entities=entities_model,
            key_field_result=kfr_model,
            full_text_snippet=(
                result.get("full_text", "")[:1000]
                + ("..." if len(result.get("full_text", "")) > 1000 else "")
            ),
            processing_time=float(result.get("processing_time", 0.0)),
            language_used=result.get("language_used", "eng"),
            model_used=result.get("model_used", False),
        )

        message = f"Analyzed with language '{result.get('language_used', 'eng')}'"
        if result.get("lilt_model_used"):
            message += " using LiLT model"
        elif result.get("qa_model_used"):
            message += " using QA model"

        return AnalysisResponse(
            status="success", message=message, result=extraction, error=None
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception("API error: %s", e)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "internal error",
                "error": str(e),
            },
        )
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except Exception:
                pass


@app.post("/verify")
async def verify_endpoint(file: UploadFile = File(...)):
    """Verify document using fingerprint + LILT hybrid"""
    if not hasattr(app.state, "verifier") or app.state.verifier is None:
        return JSONResponse(
            content={"error": "Verifier not initialized"}, status_code=500
        )

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(file.filename)[1]
    ) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        verifier: Verifier = app.state.verifier
        result = verifier.verify(tmp_path)
        return JSONResponse(content=result)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


@app.post("/api/extract-text-raw")
async def api_extract_text_raw(file: UploadFile = File(...)):
    """Extract raw OCR text and bounding boxes"""
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(file.filename)[1]
    ) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        words, boxes, w, h = ocr_extract_words_bboxes(tmp_path)

        out = {
            "filename": file.filename,
            "width": w,
            "height": h,
            "word_count": len(words),
            "words": words,
            "bboxes": boxes,
            "text_joined": " ".join(words),
        }

        return JSONResponse(content=out)
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass


@app.post("/api/extract_form_name")
async def extract_form_name(file: UploadFile = File(...)):
    """Extract only form name using fingerprints"""
    if not hasattr(app.state, "verifier") or app.state.verifier is None:
        return JSONResponse({"error": "Verifier not initialized"}, status_code=500)

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(file.filename)[1]
    ) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        verifier: Verifier = app.state.verifier
        result = verifier.extract_form_name_only(tmp_path)
        return JSONResponse(content=result)
    finally:
        try:
            os.remove(tmp_path)
        except:
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
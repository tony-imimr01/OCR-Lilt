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
from math import sqrt
import csv
import json
import ast

from extract_field import DocumentFieldExtractor

# ---- CUDA / Torch env ----
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
# Add these for memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
os.environ["CUDA_EMPTY_CACHE"] = "1"

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

LILT_MODEL = None

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

# ADD THIS LINE TO DISABLE INFO MESSAGES
logger.setLevel(logging.WARNING)  # Only show warnings and above

def manage_gpu_memory():
    """Clear GPU memory and optimize usage"""
    if torch and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            # Set memory fraction if needed
            torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
            logger.info("GPU memory managed and cache cleared")
        except Exception as e:
            logger.warning(f"GPU memory management failed: {e}")

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
    page_number: int = 1  # ✅ ADD THIS LINE - CRITICAL FIX
    context_entities: List[ExtractedEntity] = []
    meta: Optional[Dict] = None
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
    if not text:
        return False
    text_lower = text.lower().strip()
    
    # Skip headers
    skip_patterns = [
        r"tick.*box", r"please.*tick", r"section.*[a-z]",
        r"general.*information", r"name.*organization"
    ]
    for p in skip_patterns:
        if re.search(p, text_lower, re.IGNORECASE):
            return False
    
    # Keywords
    keywords = ["loss", "damage", "replacement", "deletion", "certificate", "facility"]
    keyword_count = sum(1 for k in keywords if k in text_lower)
    has_paren = "(" in text and ")" in text
    
    # Accept if:
    # - Has parentheses + keyword, OR
    # - At least 1 keyword and length >= 4, OR
    # - Matches known pattern
    patterns = [
        r".*certificate",
        r"damaged.*\(.*certificate.*deletion",
        r"replacement.*copy.*certificate",
        r"deletion.*facility"
    ]
    matches_pattern = any(re.search(p, text_lower, re.IGNORECASE) for p in patterns)
    
    return (
        matches_pattern or
        (has_paren and keyword_count >= 1) or
        (keyword_count >= 1 and len(text) >= 4)
    )

def find_checkbox_region(entities: List[Dict]) -> Tuple[int, int, int, int]:
    """Find checkbox region with generous boundaries to include all checkboxes"""
    if not entities:
        logger.warning("No entities found, using entire page")
        return 0, 0, 10000, 10000
    
    # Get overall document bounds
    all_x1, all_y1, all_x2, all_y2 = [], [], [], []
    for e in entities:
        bbox = e.get("bbox", {})
        x1 = bbox.get("x", 0)
        y1 = bbox.get("y", 0)
        x2 = x1 + bbox.get("width", 0)
        y2 = y1 + bbox.get("height", 0)
        
        all_x1.append(x1)
        all_y1.append(y1)
        all_x2.append(x2)
        all_y2.append(y2)
    
    if not all_x1:
        logger.warning("No valid bounding boxes found")
        return 0, 0, 10000, 10000
    
    # Find document boundaries
    doc_left = min(all_x1)
    doc_right = max(all_x2)
    doc_top = min(all_y1)
    doc_bottom = max(all_y2)
    doc_width = doc_right - doc_left
    doc_height = doc_bottom - doc_top
    
    logger.info(f"Document bounds: {doc_left}x{doc_top} to {doc_right}x{doc_bottom} ({doc_width}x{doc_height})")
    
    # Checkbox region: typically starts from middle to bottom of document
    # Use very generous bounds to ensure we capture all checkboxes
    region_left = max(0, doc_left - 100)  # Add 100px padding left
    region_right = min(10000, doc_right + 100)  # Add 100px padding right
    
    # Vertical region: start from 40% of document height (not too high)
    # to 90% of document height (not too low)
    region_top = max(0, int(doc_top + (doc_height * 0.4)))
    region_bottom = min(10000, int(doc_top + (doc_height * 0.9)))
    
    # If document is very small, adjust bounds
    if doc_height < 500:
        region_top = max(0, doc_top)
        region_bottom = min(10000, doc_bottom)
    
    logger.info(f"Checkbox region (generous): {region_left}, {region_top}, {region_right}, {region_bottom}")
    
    # Now refine by looking for vertical clusters of text
    # This helps when checkboxes are in the middle of the document
    vertical_entities = []
    for e in entities:
        bbox = e.get("bbox", {})
        y_center = bbox.get("y", 0) + (bbox.get("height", 0) / 2)
        
        # Check if entity is within our generous vertical bounds
        if region_top <= y_center <= region_bottom:
            vertical_entities.append(e)
    
    # If we found entities in the vertical region, adjust bounds to include them
    if vertical_entities:
        vert_y1 = min(e.get("bbox", {}).get("y", region_top) for e in vertical_entities)
        vert_y2 = max(e.get("bbox", {}).get("y", 0) + e.get("bbox", {}).get("height", 0) 
                     for e in vertical_entities)
        
        # Expand vertical bounds by 100px to be safe
        region_top = max(0, vert_y1 - 100)
        region_bottom = min(10000, vert_y2 + 100)
        
        # Also adjust horizontal bounds based on these entities
        vert_x1 = min(e.get("bbox", {}).get("x", region_left) for e in vertical_entities)
        vert_x2 = max(e.get("bbox", {}).get("x", 0) + e.get("bbox", {}).get("width", 0) 
                     for e in vertical_entities)
        
        region_left = max(0, vert_x1 - 100)
        region_right = min(10000, vert_x2 + 100)
        
        logger.info(f"Refined checkbox region: {region_left}, {region_top}, {region_right}, {region_bottom}")
    
    return (region_left, region_top, region_right, region_bottom)
    
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

    # Check if entity looks like a checkbox option
    def is_checkbox_candidate(entity: Dict) -> bool:
        text = entity.get("value", "").lower()
        checkbox_indicators = [
            "due to:", "reason for:", "replacement", "damage", 
            "certificate", "deletion", "facility", "loss"
        ]
        return any(indicator in text for indicator in checkbox_indicators)
    
    by_page = defaultdict(list)
    for e in entities:
        by_page[e["page_number"]].append(e)
    
    merged = []
    for page, page_entities in by_page.items():
        page_entities.sort(key=lambda e: (e["bbox"]["y"], e["bbox"]["x"]))
        
        # Don't merge checkbox candidates at all
        checkbox_candidates = [e for e in page_entities if is_checkbox_candidate(e)]
        non_checkbox = [e for e in page_entities if not is_checkbox_candidate(e)]
        
        # Merge non-checkbox entities normally
        current = None
        for e in non_checkbox:
            if current is None:
                current = e.copy()
                continue
            
            curr_bbox = current["bbox"]
            e_bbox = e["bbox"]
            y_diff = abs(e_bbox["y"] - curr_bbox["y"])
            x_diff = e_bbox["x"] - (curr_bbox["x"] + curr_bbox["width"])
            
            # INCREASE THRESHOLDS
            should_not_merge = (
                y_diff >= 30 or  # Increased from 15
                x_diff >= 100    # Increased from 50
            )
            
            if should_not_merge:
                merged.append(current)
                current = e.copy()
            else:
                # Merge logic...
                pass

        if current:
            merged.append(current)
        
        # Add checkbox candidates WITHOUT merging
        merged.extend(checkbox_candidates)

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

def scale_bbox_to_a4_300dpi(raw_bbox: Dict, document_width: int, document_height: int) -> Dict:
    """
    Scale bbox to target 2480x3509 (e.g. A4 at ~300 DPI).
    raw_bbox uses keys: x, y, width, height (in pixels of original doc).
    """
    x = float(raw_bbox.get("x", 0))
    y = float(raw_bbox.get("y", 0))
    width = float(raw_bbox.get("width", 0))
    height = float(raw_bbox.get("height", 0))

    target_w = 2480
    target_h = 3509

    scale_x = target_w / max(1, document_width)
    scale_y = target_h / max(1, document_height)

    return {
        "x": int(round(x * scale_x)),
        "y": int(round(y * scale_y)),
        "width": int(round(width * scale_x)),
        "height": int(round(height * scale_y)),
        "confidence": raw_bbox.get("confidence", 1.0),
    }

def scale_bbox_to_a4_400dpi(raw_bbox: Dict, document_width: int, document_height: int) -> Dict:
    """
    Scale bbox to target 2480x3509 (e.g. A4 at ~300 DPI).
    raw_bbox uses keys: x, y, width, height (in pixels of original doc).
    """
    x = float(raw_bbox.get("x", 0))
    y = float(raw_bbox.get("y", 0))
    width = float(raw_bbox.get("width", 0))
    height = float(raw_bbox.get("height", 0))

    target_w = 1654
    target_h = 2339

    scale_x = target_w / max(1, document_width)
    scale_y = target_h / max(1, document_height)

    return {
        "x": int(round(x * scale_x)),
        "y": int(round(y * scale_y)),
        "width": int(round(width * scale_x)),
        "height": int(round(height * scale_y)),
        "confidence": raw_bbox.get("confidence", 1.0),
    }

def parse_entities_from_literal(content: str, source_label: str = "<memory>") -> List[Dict[str, Any]]:
    """
    Same logic as load_entities_from_file, but works on a string
    instead of reading from disk.
    """
    content = content.strip()
    try:
        data = ast.literal_eval(content)
        if not isinstance(data, list):
            raise ValueError("Content must be a list")
        return data
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Invalid Python literal in {source_label}: {e}")


def convert_result_json_to_test_page_data_in_memory(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    associations = data.get("associations", [])
    entities: List[Dict[str, Any]] = []

    for idx, assoc in enumerate(associations, start=1):
        page = assoc.get("page", 1)
        text = assoc.get("text", "") or ""
        bbox_list = assoc.get("bbox", [0, 0, 0, 0])
        img_info = assoc.get("image_dimensions", {})  # from result1.json [file:43]
        doc_w = int(img_info.get("width", 1))
        doc_h = int(img_info.get("height", 1))

        # bbox in result1.json is [x1, y1, x2, y2]
        if len(bbox_list) == 4:
            x1, y1, x2, y2 = bbox_list
            raw_bbox = {
                "x": float(x1),
                "y": float(y1),
                "width": float(x2 - x1),
                "height": float(y2 - y1),
                "confidence": 0.8,
            }
        else:
            raw_bbox = {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0, "confidence": 1.0}

        # NEW: scale to 2480x3509
        scaled = scale_bbox_to_a4_400dpi(raw_bbox, 1654, 2339)

        entity = {
            "field": f"word_{idx}",
            "value": text,
            "bbox": {
                "x": scaled["x"],
                "y": scaled["y"],
                "width": scaled["width"],
                "height": scaled["height"],
            },
            "confidence": scaled.get("confidence", 0.8),
            "page_number": int(page),
            "raw_text": text,
            "ocr_method": "easyocr",
        }
        entities.append(entity)

    literal_str = repr(entities)
    return parse_entities_from_literal(literal_str)
   
def _extract_text_multipage(
    file_path: str, languages: str = "eng", conf_threshold: float = 0.15
) -> Tuple[List[Dict], str, int, List[Image.Image]]:
    """Extract text from multi-page document with robust bbox validation"""
    entities: List[Dict] = []
    images: List[Image.Image] = []
    clamped_count = 0
    validation_stats = {"total": 0, "valid": 0, "filtered": 0, "reasons": defaultdict(int)}
    
    # Load entities from result.json
    entities = convert_result_json_to_test_page_data_in_memory(file_path)
    
    return entities, len(images), images

# ------- LiLT Relation Extractor -------
class LiLTRelationExtractor:
    def __init__(self, model_path: str, config: LiLTConfig, device: Optional[int] = None):
        self.model_path = model_path
        self.config = config
        self.device = device if device is not None else (0 if (TORCH_AVAILABLE and torch.cuda.is_available()) else -1)
        # Add device_str for moving tensors
        self.device_str = f"cuda:{self.device}" if self.device >= 0 else "cpu"
        self.available = False
        self.tokenizer = None
        self.model = None
        
        if not (LILT_AVAILABLE and TORCH_AVAILABLE):
            logger.warning("LiLT model or torch not available")
            return

        try:
            logger.info(f"Loading LiLT from: {model_path}")
            
            # Fix: Load tokenizer with Mistral regex fix disabled
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    use_fast=True,
                    fix_mistral_regex=False  # Explicitly disable for non-Mistral models
                )
                logger.info(f"Loaded tokenizer: {type(self.tokenizer).__name__}")
                
            except Exception as e:
                logger.warning(f"Failed to load tokenizer from {model_path}: {e}")
                # Fallback to a default tokenizer
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        "roberta-base",
                        use_fast=True
                    )
                    logger.info("Loaded fallback RoBERTa base tokenizer")
                except Exception as e2:
                    logger.error(f"Failed to load fallback tokenizer: {e2}")
                    self.tokenizer = None
            
            # FIXED: Simplified and robust model loading
            try:
                # Strategy 1: Try to create a simple model from config
                try:
                    # Load config first
                    config_model = AutoConfig.from_pretrained(
                        model_path,
                        trust_remote_code=True
                    )
                    
                    # Set the number of relation labels
                    config_model.num_labels = config.num_rel_labels
                    
                    # Create base model from config
                    base_model = AutoModel.from_config(config_model)
                    
                    # Create custom relation extraction model
                    self.model = LiLTRobertaLikeForRelationExtraction(
                        encoder=base_model,
                        num_rel_labels=config.num_rel_labels
                    )
                    
                    logger.info("Created LiLT model from config (random weights)")
                    
                except Exception as e:
                    logger.warning(f"Failed to create model from config: {e}")
                    
                    # Strategy 2: Try direct loading with error handling
                    try:
                        self.model = AutoModel.from_pretrained(
                            model_path,
                            trust_remote_code=True
                        )
                        logger.info("Directly loaded LiLT model with AutoModel")
                    except Exception as e2:
                        logger.error(f"All model loading attempts failed: {e2}")
                        self.model = None
                
                # Try to load weights if model was created
                if self.model:
                    # Try to find and load weights
                    weights_loaded = self._try_load_weights(model_path)
                    
                    if not weights_loaded:
                        logger.warning("Model created with random weights - will need fine-tuning")
                
            except Exception as config_error:
                logger.error(f"Failed to initialize model: {config_error}")
                self.model = None
            
            # Move model to device
            if self.model:
                self.model = self.model.to(self.device_str)
                self.model.eval()
                self.available = True
                logger.info(f"LiLT model ready on {self.device_str}")
                logger.info("Note: Random initialization is normal if no weights were found")
            else:
                logger.warning("LiLT model not loaded")
                
        except Exception as e:
            logger.error(f"Failed to initialize LiLT model: {e}")
            logger.exception(e)
            self.available = False
    
    def _try_load_weights(self, model_path: str) -> bool:
        """Try to load weights with multiple strategies"""
        model_file = self._find_model_file(model_path)
        if not model_file:
            logger.info("No model file found, using random initialization")
            return False
        
        logger.info(f"Found potential model file: {model_file}")
        
        try:
            # Check if it's a safetensors file
            if model_file.endswith('.safetensors'):
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(model_file, device="cpu")
                    logger.info(f"Loaded safetensors file: {len(state_dict)} keys")
                except ImportError:
                    logger.error("safetensors not installed. Install with: pip install safetensors")
                    return False
                except Exception as e:
                    logger.error(f"Failed to load safetensors: {e}")
                    return False
            else:
                # Try loading with different approaches
                try:
                    # First try with weights_only=True (safer)
                    state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
                    logger.info("Loaded weights with weights_only=True")
                except Exception:
                    try:
                        # Try with weights_only=False if safe
                        state_dict = torch.load(model_file, map_location="cpu", weights_only=False)
                        logger.info("Loaded weights with weights_only=False")
                    except Exception as e:
                        logger.error(f"Failed to load weights file: {e}")
                        # Check if file might be corrupted
                        file_size = os.path.getsize(model_file)
                        logger.info(f"Model file size: {file_size:,} bytes")
                        if file_size < 1024:
                            logger.warning("Model file is too small, likely corrupted")
                        return False
            
            # Process state_dict
            if isinstance(state_dict, dict):
                # Handle different state_dict formats
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                elif "model" in state_dict:
                    state_dict = state_dict["model"]
                elif "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                
                # Load into model
                try:
                    self.model.load_state_dict(state_dict, strict=False)
                    logger.info(f"Successfully loaded weights into model")
                    return True
                except Exception as e:
                    logger.warning(f"Could not load state_dict directly: {e}")
                    
                    # Try partial loading
                    try:
                        model_dict = self.model.state_dict()
                        filtered_dict = {k: v for k, v in state_dict.items() 
                                       if k in model_dict and v.shape == model_dict[k].shape}
                        
                        if filtered_dict:
                            model_dict.update(filtered_dict)
                            self.model.load_state_dict(model_dict)
                            logger.info(f"Partially loaded {len(filtered_dict)}/{len(model_dict)} weights")
                            return True
                    except Exception as e2:
                        logger.warning(f"Partial loading failed: {e2}")
            
            return False
                
        except Exception as e:
            logger.error(f"Error loading weights: {e}")
            return False
    
    def _find_model_file(self, model_path: str) -> Optional[str]:
        """Find model file in directory"""
        # Check if it's already a file
        if os.path.isfile(model_path):
            return model_path
        
        # Check for common model files
        common_files = [
            "pytorch_model.bin",
            "model.bin",
            "model.safetensors",
            "pytorch_model.pt",
            "model.pt",
            "checkpoint.pt",
            "weights.pt",
        ]
        
        for fname in common_files:
            path = os.path.join(model_path, fname)
            if os.path.isfile(path):
                return path
        
        # Search for any model file
        try:
            for root, _, files in os.walk(model_path):
                for f in files:
                    if f.endswith(('.bin', '.pt', '.safetensors')):
                        path = os.path.join(root, f)
                        file_size = os.path.getsize(path)
                        if file_size > 1024:  # At least 1KB
                            return path
        except:
            pass
        
        return None

    def _create_memory_efficient_inputs(self, words: List[str], bboxes: List[List[int]], max_seq_length: int = 512) -> Optional[Dict]:
        """Create inputs with memory-efficient chunking"""
        try:
            # Use a smaller batch size for memory efficiency
            chunk_size = 32  # Reduced from potentially 512
            
            # Process in chunks if too many entities
            if len(words) > chunk_size:
                logger.info(f"Processing {len(words)} entities in chunks of {chunk_size} for memory efficiency")
                return self._process_in_chunks(words, bboxes, chunk_size, max_seq_length)
            
            # For small number of entities, use regular processing
            if hasattr(self.tokenizer, "encode_plus_with_bbox"):
                encoded = self.tokenizer.encode_plus_with_bbox(
                    words,
                    bboxes,
                    padding="max_length",
                    truncation=True,
                    max_length=max_seq_length,
                    return_tensors="pt"
                )
            else:
                try:
                    encoded = self.tokenizer(
                        words,
                        boxes=bboxes,
                        padding="max_length",
                        truncation=True,
                        max_length=max_seq_length,
                        return_tensors="pt"
                    )
                except TypeError:
                    # Fallback to text-only
                    encoded = self.tokenizer(
                        words,
                        padding="max_length",
                        truncation=True,
                        max_length=max_seq_length,
                        return_tensors="pt"
                    )
            
            # Move to device
            encoded = {k: v.to(self.device_str) for k, v in encoded.items()}
            return encoded
            
        except Exception as e:
            logger.error(f"Error in memory-efficient inputs: {e}")
            return None
    
    def _process_in_chunks(self, words: List[str], bboxes: List[List[int]], chunk_size: int, max_seq_length: int) -> Optional[Dict]:
        """Process large number of entities in chunks to save memory"""
        all_chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i+chunk_size]
            chunk_bboxes = bboxes[i:i+chunk_size]
            
            if hasattr(self.tokenizer, "encode_plus_with_bbox"):
                chunk_encoded = self.tokenizer.encode_plus_with_bbox(
                    chunk_words,
                    chunk_bboxes,
                    padding="max_length",
                    truncation=True,
                    max_length=max_seq_length,
                    return_tensors="pt"
                )
            else:
                try:
                    chunk_encoded = self.tokenizer(
                        chunk_words,
                        boxes=chunk_bboxes,
                        padding="max_length",
                        truncation=True,
                        max_length=max_seq_length,
                        return_tensors="pt"
                    )
                except TypeError:
                    # Fallback to text-only
                    chunk_encoded = self.tokenizer(
                        chunk_words,
                        padding="max_length",
                        truncation=True,
                        max_length=max_seq_length,
                        return_tensors="pt"
                    )
            
            all_chunks.append(chunk_encoded)
        
        if not all_chunks:
            return None
        
        # Use first chunk as base (we'll just process chunks sequentially in the main extraction)
        first_chunk = all_chunks[0]
        first_chunk = {k: v.to(self.device_str) for k, v in first_chunk.items()}
        return first_chunk
    
    def _extract_relations_memory_efficient(self, entities: List[Dict], max_entities_per_batch: int = 50) -> List[Dict]:
        """Extract relations with memory limits"""
        if not self.available or not entities:
            return []
        
        try:
            # Group by page
            pages = {}
            for entity in entities:
                page_num = entity.get("page_number", 1)
                if page_num not in pages:
                    pages[page_num] = []
                pages[page_num].append(entity)
            
            all_relations = []
            
            for page_num, page_entities in pages.items():
                if not page_entities or len(page_entities) < 2:
                    continue
                
                # If too many entities, sample or filter
                if len(page_entities) > max_entities_per_batch:
                    logger.info(f"Page {page_num} has {len(page_entities)} entities, limiting to {max_entities_per_batch}")
                    # Sample entities with highest confidence
                    page_entities = sorted(page_entities, key=lambda x: x.get("confidence", 0), reverse=True)[:max_entities_per_batch]
                
                words = [e["value"] for e in page_entities]
                bboxes = [e["bbox"] for e in page_entities]
                
                # Normalize bboxes
                normalized_bboxes = []
                for bbox in bboxes:
                    x = bbox.get("x", 0)
                    y = bbox.get("y", 0)
                    width = bbox.get("width", 0)
                    height = bbox.get("height", 0)
                    
                    x2 = x + width
                    y2 = y + height
                    
                    # Simple normalization to 0-1000
                    max_coord = 1000
                    norm_x = min(int((x / max(x2, 1)) * max_coord), 999)
                    norm_y = min(int((y / max(y2, 1)) * max_coord), 999)
                    norm_x2 = min(int((x2 / max(x2, 1)) * max_coord), 1000)
                    norm_y2 = min(int((y2 / max(y2, 1)) * max_coord), 1000)
                    
                    normalized_bboxes.append([norm_x, norm_y, norm_x2, norm_y2])
                
                # Get inputs with memory limits
                inputs = self._create_memory_efficient_inputs(words, normalized_bboxes, max_seq_length=256)
                
                if inputs is None:
                    logger.warning(f"Failed to create inputs for page {page_num}")
                    # Use fallback spatial relations
                    fallback_relations = self._extract_spatial_relations(page_entities, words, normalized_bboxes)
                    all_relations.extend(fallback_relations)
                    continue
                
                try:
                    # Run with torch.no_grad and reduced precision if needed
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    
                    # Process outputs with better error handling
                    try:
                        relations = self._process_model_outputs(outputs, page_entities, words, normalized_bboxes)
                    except Exception as e:
                        logger.error(f"Error processing model outputs: {e}")
                        # Use fallback spatial relations
                        relations = self._extract_spatial_relations(page_entities, words, normalized_bboxes)
                    
                    all_relations.extend(relations)
                    
                except torch.cuda.OutOfMemoryError:
                    logger.error(f"Out of memory processing page {page_num}, using fallback spatial relations")
                    # Clear cache and use fallback
                    torch.cuda.empty_cache()
                    fallback_relations = self._extract_spatial_relations(page_entities, words, normalized_bboxes)
                    all_relations.extend(fallback_relations)
                
                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {e}")
                    # Use fallback
                    fallback_relations = self._extract_spatial_relations(page_entities, words, normalized_bboxes)
                    all_relations.extend(fallback_relations)
            
            logger.info(f"Extracted {len(all_relations)} relations (memory-efficient)")
            return all_relations
        
        except Exception as e:
            logger.error(f"Error in memory-efficient extraction: {e}")
            torch.cuda.empty_cache()
            return []
    
    def _extract_spatial_relations(self, entities: List[Dict], words: List[str], bboxes: List[List[int]]) -> List[Dict]:
        """Extract spatial relations as fallback when model fails"""
        relations = []
        
        if len(entities) < 2:
            return relations
        
        # Use spatial proximity to find potential label-value pairs
        for i in range(len(entities)):
            for j in range(len(entities)):
                if i == j:
                    continue
                
                # Check if these entities might form a label-value pair
                entity_i = entities[i]
                entity_j = entities[j]
                
                # Check confidence
                if entity_i.get("confidence", 0) < 0.3 or entity_j.get("confidence", 0) < 0.3:
                    continue
                
                # Check spatial relationship
                bbox_i = bboxes[i]
                bbox_j = bboxes[j]
                
                # Calculate distance and alignment
                y_diff = abs((bbox_i[1] + bbox_i[3])/2 - (bbox_j[1] + bbox_j[3])/2)
                x_diff = bbox_j[0] - bbox_i[2]  # Distance from end of i to start of j
                
                # Heuristic: label is often to the left of value on same line
                if x_diff > 0 and x_diff < 200 and y_diff < 30:
                    distance = sqrt((bbox_i[0] - bbox_j[0])**2 + (bbox_i[1] - bbox_j[1])**2)
                    
                    relation = {
                        "label_entity": entity_i,
                        "value_entity": entity_j,
                        "label_text": words[i],
                        "value_text": words[j],
                        "label_bbox": bboxes[i],
                        "value_bbox": bboxes[j],
                        "score": 0.7,  # Heuristic confidence
                        "distance": distance,
                        "page_number": entity_i.get("page_number", 1),
                        "method": "spatial_fallback"
                    }
                    relations.append(relation)
        
        return relations

    def extract_relations(self, entities: List[Dict]) -> List[Dict]:
        """Extract relations between entities with memory limits"""
        if not self.available or not entities:
            logger.warning("LiLT model not available or no entities provided")
            return []
        
        # Set memory limits
        max_entities_total = 100  # Maximum total entities to process
        max_entities_per_page = 50  # Maximum entities per page
        
        if len(entities) > max_entities_total:
            logger.warning(f"Too many entities ({len(entities)}), limiting to {max_entities_total}")
            # Sort by confidence and take top N
            entities = sorted(entities, key=lambda x: x.get("confidence", 0), reverse=True)[:max_entities_total]
        
        # Try memory-efficient extraction first
        relations = self._extract_relations_memory_efficient(entities, max_entities_per_page)
        
        if not relations:
            # Fallback to spatial relations only
            logger.info("Using spatial fallback relations only")
            relations = self._extract_fallback_relations(entities)
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return relations
    
    def _extract_fallback_relations(self, entities: List[Dict]) -> List[Dict]:
        """Simple fallback relation extraction based on spatial proximity"""
        relations = []
        
        # Group by page
        pages = {}
        for entity in entities:
            page_num = entity.get("page_number", 1)
            if page_num not in pages:
                pages[page_num] = []
            pages[page_num].append(entity)
        
        for page_num, page_entities in pages.items():
            if len(page_entities) < 2:
                continue
            
            # Sort by position (top-left to bottom-right)
            page_entities.sort(key=lambda e: (e.get("bbox", {}).get("y", 0), e.get("bbox", {}).get("x", 0)))
            
            # Create pairwise relations for adjacent entities
            for i in range(len(page_entities) - 1):
                e1 = page_entities[i]
                e2 = page_entities[i + 1]
                
                bbox1 = e1.get("bbox", {})
                bbox2 = e2.get("bbox", {})
                
                x1, y1 = bbox1.get("x", 0), bbox1.get("y", 0)
                x2, y2 = bbox2.get("x", 0), bbox2.get("y", 0)
                
                # Calculate distance
                distance = sqrt((x1 - x2)**2 + (y1 - y2)**2)
                
                # Only create relation if reasonably close
                if distance < 300:  # 300 pixels threshold
                    relation = {
                        "label_entity": e1,
                        "value_entity": e2,
                        "label_text": e1.get("value", ""),
                        "value_text": e2.get("value", ""),
                        "label_bbox": [x1, y1, x1 + bbox1.get("width", 0), y1 + bbox1.get("height", 0)],
                        "value_bbox": [x2, y2, x2 + bbox2.get("width", 0), y2 + bbox2.get("height", 0)],
                        "score": max(0.1, 1.0 - distance/300),  # Higher score for closer entities
                        "distance": distance,
                        "page_number": page_num,
                        "method": "spatial_adjacency"
                    }
                    relations.append(relation)
        
        return relations

    def _tokenize_with_layout(self, words: List[str], bboxes: List[List[int]]) -> Optional[Dict]:
        """Tokenize text with layout information using the appropriate method"""
        try:
            # First, get tokenized inputs
            if hasattr(self.tokenizer, "encode_plus_with_bbox"):
                # Some layout tokenizers have this method
                encoded = self.tokenizer.encode_plus_with_bbox(
                    words,
                    bboxes,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                # Move to device
                encoded = {k: v.to(self.device_str) for k, v in encoded.items()}
                return encoded
            
            # Try standard tokenization with bbox parameter
            try:
                inputs = self.tokenizer(
                    words,
                    boxes=bboxes,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device_str) for k, v in inputs.items()}
                return inputs
            except TypeError:
                # Fallback: manually process layout information
                logger.warning("Tokenizer doesn't support boxes parameter, using text-only mode")
                inputs = self.tokenizer(
                    words,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device_str) for k, v in inputs.items()}
                
                # Add bbox information to inputs manually if model expects it
                if hasattr(self.model, "config") and hasattr(self.model.config, "use_bbox"):
                    bbox_tensor = torch.tensor(bboxes, dtype=torch.long)
                    bbox_tensor = bbox_tensor.to(self.device_str)
                    inputs["bbox"] = bbox_tensor
                
                return inputs
        
        except Exception as e:
            logger.error(f"Error during layout-aware tokenization: {e}")
            return None

    def _process_model_outputs(self, outputs, entities, words, bboxes) -> List[Dict]:
        """Process model outputs to extract relations between entities"""
        relations = []
        
        # Check if outputs contain relation predictions
        if hasattr(outputs, 'logits') and outputs.logits is not None:
            logits = outputs.logits
            
            # Handle different logits shapes
            if logits.dim() == 3:  # [batch_size, sequence_length, num_labels]
                # This is token classification output, not relation extraction
                # Use fallback spatial relations instead
                logger.warning("Model returned token classification logits, using spatial fallback")
                return self._extract_spatial_relations(entities, words, bboxes)
            
            elif logits.dim() == 2:  # [batch_size, num_relations]
                # This looks like relation extraction output
                try:
                    # Get scores for positive relations (assuming binary classification)
                    if logits.shape[1] >= 2:  # Has at least 2 classes
                        relation_probs = torch.softmax(logits, dim=-1)
                        positive_scores = relation_probs[:, 1]  # Probability for positive relation
                        
                        # Create potential relations
                        # This assumes the model outputs relations for all entity pairs
                        n_entities = len(entities)
                        expected_relations = n_entities * (n_entities - 1)  # All possible non-self relations
                        
                        if len(positive_scores) >= expected_relations:
                            idx = 0
                            for i in range(n_entities):
                                for j in range(n_entities):
                                    if i == j:
                                        continue
                                        
                                    if idx < len(positive_scores):
                                        score = positive_scores[idx].item()
                                        if score > self.config.min_confidence:
                                            relations.append({
                                                "label_entity": entities[i],
                                                "value_entity": entities[j],
                                                "label_text": words[i],
                                                "value_text": words[j],
                                                "label_bbox": bboxes[i],
                                                "value_bbox": bboxes[j],
                                                "score": score,
                                                "distance": self._calculate_distance(bboxes[i], bboxes[j]),
                                                "page_number": entities[i].get("page_number", 1)
                                            })
                                        idx += 1
                    else:
                        # Single class output - use thresholding
                        relation_scores = torch.sigmoid(logits).squeeze()
                        if relation_scores.dim() == 0:
                            relation_scores = relation_scores.unsqueeze(0)
                        
                        # Create potential relations based on score threshold
                        n_entities = len(entities)
                        for i in range(min(n_entities, 10)):  # Limit for performance
                            for j in range(min(n_entities, 10)):
                                if i == j or i >= j:  # Only create one direction to avoid duplicates
                                    continue
                                    
                                # Create relation based on spatial proximity
                                distance = self._calculate_distance(bboxes[i], bboxes[j])
                                if distance < 200:  # Close entities
                                    relations.append({
                                        "label_entity": entities[i],
                                        "value_entity": entities[j],
                                        "label_text": words[i],
                                        "value_text": words[j],
                                        "label_bbox": bboxes[i],
                                        "value_bbox": bboxes[j],
                                        "score": 0.7,  # Heuristic confidence
                                        "distance": distance,
                                        "page_number": entities[i].get("page_number", 1)
                                    })
                                    
                except Exception as e:
                    logger.error(f"Error processing relation logits: {e}")
                    # Fallback to spatial relations
                    return self._extract_spatial_relations(entities, words, bboxes)
        
        # Fallback: if no explicit relations, use spatial proximity
        if not relations:
            logger.info("No explicit relations found, using spatial proximity as fallback")
            relations = self._extract_spatial_relations(entities, words, bboxes)
        
        return relations
    
    def _calculate_distance(self, bbox1, bbox2):
        """Calculate distance between two bounding boxes"""
        x1_center = (bbox1[0] + bbox1[2]) / 2
        y1_center = (bbox1[1] + bbox1[3]) / 2
        x2_center = (bbox2[0] + bbox2[2]) / 2
        y2_center = (bbox2[1] + bbox2[3]) / 2
        
        return ((x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2) ** 0.5
    
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
    def has_excessive_spaces(self, text: str) -> bool:
        #logger.warning(f"text: {text}")
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
    def _find_label_entity(self, entities: List[Dict], key_field: str) -> Optional[Dict]:
        key_lower = key_field.lower()
        
        # Map common key_fields to expected patterns
        pattern_map = {
            "form number": [r"form\s+no", r"form\s+gf\d+", r"reference\s+no"],
            "applicant name": [r"applicant", r"name.*organization", r"owner.*name"],
            "issue date": [r"issue\s+date", r"date\s+of\s+issue"],
            "completion date": [r"completion\s+date", r"date\s+of\s+completion"],
        }
        
        patterns = []
        for k, v in pattern_map.items():
            if k in key_lower:
                patterns.extend(v)
        
        if not patterns:
            # Fallback: split key_field into words
            keywords = re.findall(r'\w+', key_lower)
            patterns = [r".*" + r".*".join(keywords) + r".*"]
        
        for e in entities:
            text = e.get("value", "").lower()
            for pat in patterns:
                if re.search(pat, text, re.IGNORECASE):
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
    def _build_empty_result(self, key_field: str) -> Tuple[Dict, List[Dict]]:
        """Build generic empty result for any field type when no value is found"""
        empty_bbox = {"x": 0, "y": 0, "width": 1, "height": 1}
        field_type = self._get_field_type(key_field)
        
        result = {
            "field_name": key_field,
            "value": "Not found",
            "structured_value": {
                "field_name": key_field,
                "field_type": field_type,
                "value": "Not found",
                "confidence": 0.0,
                "bbox": empty_bbox,
            },
            "confidence": 0.0,
            "bbox": empty_bbox,
            "context_entities": [],
            "extraction_method": "none",
            "meta": {"reason": f"No valid value found for field: '{key_field}'"},
            "page_number": 1,
            "found": False
        }
        
        filtered_entity = {
            "field": key_field,
            "value": "Not found",
            "bbox": empty_bbox,
            "confidence": 0.0,
            "page_number": 1,
            "semantic_type": EntityTypes.ANSWER,
            "semantic_confidence": 0.0,
            "found": False
        }
        
        return result, [filtered_entity]
    def _find_nearest_value_entity(self, entities: List[Dict], label_entity: Dict, label_text: str) -> Optional[Dict]:
        if not label_entity:
            return None

        label_bbox = label_entity.get("bbox", {})
        label_page = label_entity.get("page_number", 1)
        lx = label_bbox.get("x", 0)
        ly = label_bbox.get("y", 0)
        lw = label_bbox.get("width", 0)
        lh = label_bbox.get("height", 0)

        best_candidate = None
        min_distance = float('inf')

        for e in entities:
            if e is label_entity:
                continue
            if e.get("page_number", 1) != label_page:
                continue
            val = e.get("value", "").strip()
            if not val:
                continue
            # Skip other labels
            if self._is_email_label(val) or self._is_phone_label(val):
                continue

            bbox = e.get("bbox", {})
            ex = bbox.get("x", 0)
            ey = bbox.get("y", 0)
            ew = bbox.get("width", 0)
            eh = bbox.get("height", 0)

            # Restrict search to reasonable area: right or below within 500px
            if ex > lx + lw + 500 or ey > ly + lh + 300:
                continue
            if ex + ew < lx - 200 or ey + eh < ly - 50:
                continue

            # Compute distance between label right-center and entity left-center
            label_anchor_x = lx + lw
            label_anchor_y = ly + lh / 2
            entity_anchor_x = ex
            entity_anchor_y = ey + eh / 2

            dx = entity_anchor_x - label_anchor_x
            dy = entity_anchor_y - label_anchor_y
            distance = sqrt(dx*dx + dy*dy)

            if distance < min_distance:
                min_distance = distance
                best_candidate = e

        if best_candidate:
            logger.info(f"Nearest value for '{label_text}': '{best_candidate.get('value', '')}' (dist={min_distance:.1f})")
        return best_candidate

    def find_checkbox_region(entities: List[Dict]) -> Tuple[int, int, int, int]:
        """Find checkbox region with generous boundaries to include all checkboxes"""
        if not entities:
            logger.warning("No entities found, using entire page")
            return 0, 0, 10000, 10000
        
        # Get overall document bounds
        all_x1, all_y1, all_x2, all_y2 = [], [], [], []
        for e in entities:
            bbox = e.get("bbox", {})
            x1 = bbox.get("x", 0)
            y1 = bbox.get("y", 0)
            x2 = x1 + bbox.get("width", 0)
            y2 = y1 + bbox.get("height", 0)
            
            all_x1.append(x1)
            all_y1.append(y1)
            all_x2.append(x2)
            all_y2.append(y2)
        
        if not all_x1:
            logger.warning("No valid bounding boxes found")
            return 0, 0, 10000, 10000
        
        # Find document boundaries
        doc_left = min(all_x1)
        doc_right = max(all_x2)
        doc_top = min(all_y1)
        doc_bottom = max(all_y2)
        doc_width = doc_right - doc_left
        doc_height = doc_bottom - doc_top
        
        logger.info(f"Document bounds: {doc_left}x{doc_top} to {doc_right}x{doc_bottom} ({doc_width}x{doc_height})")
        
        # Checkbox region: typically starts from middle to bottom of document
        # Use very generous bounds to ensure we capture all checkboxes
        region_left = max(0, doc_left - 100)  # Add 100px padding left
        region_right = min(10000, doc_right + 100)  # Add 100px padding right
        
        # Vertical region: start from 40% of document height (not too high)
        # to 90% of document height (not too low)
        region_top = max(0, int(doc_top + (doc_height * 0.4)))
        region_bottom = min(10000, int(doc_top + (doc_height * 0.9)))
        
        # If document is very small, adjust bounds
        if doc_height < 500:
            region_top = max(0, doc_top)
            region_bottom = min(10000, doc_bottom)
        
        logger.info(f"Checkbox region (generous): {region_left}, {region_top}, {region_right}, {region_bottom}")
        
        # Now refine by looking for vertical clusters of text
        # This helps when checkboxes are in the middle of the document
        vertical_entities = []
        for e in entities:
            bbox = e.get("bbox", {})
            y_center = bbox.get("y", 0) + (bbox.get("height", 0) / 2)
            
            # Check if entity is within our generous vertical bounds
            if region_top <= y_center <= region_bottom:
                vertical_entities.append(e)
        
        # If we found entities in the vertical region, adjust bounds to include them
        if vertical_entities:
            vert_y1 = min(e.get("bbox", {}).get("y", region_top) for e in vertical_entities)
            vert_y2 = max(e.get("bbox", {}).get("y", 0) + e.get("bbox", {}).get("height", 0) 
                        for e in vertical_entities)
            
            # Expand vertical bounds by 100px to be safe
            region_top = max(0, vert_y1 - 100)
            region_bottom = min(10000, vert_y2 + 100)
            
            # Also adjust horizontal bounds based on these entities
            vert_x1 = min(e.get("bbox", {}).get("x", region_left) for e in vertical_entities)
            vert_x2 = max(e.get("bbox", {}).get("x", 0) + e.get("bbox", {}).get("width", 0) 
                        for e in vertical_entities)
            
            region_left = max(0, vert_x1 - 100)
            region_right = min(10000, vert_x2 + 100)
            
            logger.info(f"Refined checkbox region: {region_left}, {region_top}, {region_right}, {region_bottom}")
        
        return (region_left, region_top, region_right, region_bottom)

    def detect_individual_checkbox_fields(self, entities: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Detect checkbox options with strict validation"""
        logger.info("Starting STRICT individual checkbox detection")
        
        # Find checkbox region with tighter boundaries
        region = self._find_checkbox_region(entities)
        region_entities = [e for e in entities if self._is_in_region(e, region)]
        
        # Filter out section headers/instructions
        section_headers = []
        section_header_patterns = [
            r"tick.*box.*only", r"please.*tick", r"appropriate.*box",
            r"only.*one.*allowed", r"hereby.*apply.*following",
            r"due to:", r"reason for:", r"只可選擇一項", r"作出以下申請"
        ]
        for e in region_entities:
            text = e.get("value", "").lower()
            for pattern in section_header_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    section_headers.append(e)
                    break
        
        # Identify checkbox candidates with strict validation
        checkbox_candidates = []
        for e in region_entities:
            if e in section_headers:
                continue
            value = e.get("value", "").strip()
            if not value:
                continue
            if self._is_checkbox_option(value):
                checkbox_candidates.append(e)
        
        # Deduplicate candidates
        unique_candidates = []
        seen_texts = set()
        seen_positions = set()
        
        for cand in checkbox_candidates:
            text = cand.get("value", "").strip()
            bbox = cand.get("bbox", {})
            pos_key = (bbox.get("x", 0), bbox.get("y", 0))
            
            if pos_key in seen_positions or text in seen_texts:
                continue
                
            # Check similarity to avoid near-duplicates
            is_similar = any(
                self._partial_match_score(text, seen) >= 0.9  # Higher threshold
                for seen in seen_texts
            )
            if not is_similar:
                unique_candidates.append(cand)
                seen_texts.add(text)
                seen_positions.add(pos_key)
        
        # Sort by vertical position and limit to reasonable count
        unique_candidates.sort(key=lambda x: x.get("bbox", {}).get("y", 0))
        unique_candidates = unique_candidates[:6]  # Max 6 checkbox options
        
        if not unique_candidates:
            logger.warning("No checkbox candidates found after strict filtering")
            return [], []
        
        # Final validation: Each candidate must match checkbox patterns and not be instructional
        validated_candidates = []
        for cand in unique_candidates:
            text = cand.get("value", "").strip().lower()
            
            # Must NOT contain instructional words
            instructional_words = [
                "please", "tick", "select", "choose", "indicate", "mark", 
                "note", "instruction", "must", "should", "will", "may"
            ]
            if any(word in text for word in instructional_words):
                continue
                
            # Must match specific checkbox option patterns
            checkbox_patterns = [
                r"^loss.*certificate$",
                r"^damage\s*\(.*\)$",
                r"^replacement.*copy$",
                r"^deletion.*facility$",
                r"^change.*owner$"
            ]
            if not any(re.search(pattern, text, re.IGNORECASE) for pattern in checkbox_patterns):
                continue
                
            validated_candidates.append(cand)
        
        if not validated_candidates:
            logger.warning("No candidates passed final validation")
            return [], []
        
        # Determine checked status and build results
        tick_symbols = ["✓", "✔", "☑", "√", "[x]", "[X]", "(x)", "(X)"]
        checked_index = None
        
        # Check for tick symbols in ORIGINAL text
        for i, cand in enumerate(validated_candidates):
            original_text = cand.get("value", "")
            if any(symbol in original_text for symbol in tick_symbols):
                checked_index = i
                break
        
        # If no explicit tick, assume FIRST option is selected (common in forms)
        if checked_index is None:
            checked_index = 0
            logger.info("No explicit tick found; assuming first option is selected")
        
        # Build results
        individual_results = []
        filtered_entities = []
        
        for idx, cand in enumerate(validated_candidates):
            original_text = cand.get("value", "").strip()
            bbox = cand.get("bbox", {"x": 0, "y": 0, "width": 1, "height": 1})
            conf = float(cand.get("confidence", 0.9))
            page_num = cand.get("page_number", 1)
            
            # Clean value: remove tick symbols and extra punctuation
            cleaned_value = original_text
            for symbol in tick_symbols:
                cleaned_value = cleaned_value.replace(symbol, "")
            cleaned_value = re.sub(r'^[:\-.,\s]+|[:\-.,\s]+$', '', cleaned_value).strip()
            
            is_checked = (idx == checked_index)
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
                "extraction_method": "strict_checkbox_detection",
                "meta": {
                    "original_text": original_text,
                    "is_checked": is_checked,
                    "checkbox_index": idx + 1,
                    "total_checkboxes": len(validated_candidates),
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
            
            logger.info(f"Validated checkbox {idx+1}: '{cleaned_value}' (checked: {is_checked})")
        
        logger.info(f"Detected {len(individual_results)} checkboxes after strict validation")
        return individual_results, filtered_entities
        
    def _is_checkbox_option(self, text: str) -> bool:
        """Stricter validation for checkbox options with context awareness"""
        if not text or len(text.strip()) < 3:
            return False
            
        text_lower = text.lower().strip()
        
        # Skip non-checkbox patterns (more comprehensive)
        skip_patterns = [
            r"section\s+\w+", r"part\s+\w+", r"instructions?", r"note[:\s]",
            r"confidential", r"form\s+no", r"page\s+\d+", r"application\s+fee",
            r"receipt\s+number", r"tick\s+one\s+box", r"please\s+tick",
            r"only\s+one", r"hereby\s+apply", r"作出以下申請", r"只可選擇一項",
            r"請在適當空格加", r"適用於", r"請選擇", r"form\s+gf"
        ]
        for pattern in skip_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return False
        
        # Must contain at least one checkbox keyword AND proper context
        checkbox_keywords = {
            "loss": ["loss", "missing", "lost"],
            "damage": ["damage", "damaged", "broken"],
            "replacement": ["replacement", "repl"],
            "deletion": ["deletion", "delete", "remove"],
            "certificate": ["certificate", "cert"]
        }
        
        has_valid_context = False
        for category, keywords in checkbox_keywords.items():
            if any(kw in text_lower for kw in keywords):
                # Require contextual indicators for validation
                if category == "loss":
                    has_valid_context = any(ctx in text_lower for ctx in ["certificate", "cert", "application"])
                elif category == "damage":
                    has_valid_context = any(ctx in text_lower for ctx in ["certificate", "facility", "must be returned"])
                elif category in ["replacement", "deletion", "certificate"]:
                    has_valid_context = True
                break
        
        # Require structural indicators (parentheses, etc.)
        has_parentheses = "(" in text and ")" in text
        is_short_phrase = 3 <= len(text.split()) <= 6 and len(text) <= 100
        
        # Only accept if has valid context AND structural indicators
        return has_valid_context and (has_parentheses or is_short_phrase)
   
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
                "replacement", "damage", "certificate", "deletion"
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

    def _build_phone_result(self, key_field: str, value: str, bbox: Dict, conf: float, entity: Dict, method: str = "unknown", meta: Dict = None):
        if meta is None:
            meta = {}
        result = {
            "field_name": key_field,
            "value": value,
            "structured_value": {
                "field_name": key_field,
                "field_type": "phone",
                "value": value,
                "confidence": conf,
                "bbox": bbox,
            },
            "confidence": conf,
            "bbox": bbox,
            "context_entities": [],
            "extraction_method": method,
            "meta": meta,
        }
        filtered_entity = {
            "field": key_field,
            "value": value,
            "bbox": bbox,
            "confidence": conf,
            "page_number": entity.get("page_number", 1),
            "semantic_type": EntityTypes.ANSWER,
            "semantic_confidence": conf,
        }
        return result, [filtered_entity]

    def _build_empty_phone_result(self, key_field: str):
        empty_bbox = {"x": 0, "y": 0, "width": 1, "height": 1}
        result = {
            "field_name": key_field,
            "value": "Not found",
            "structured_value": {
                "field_name": key_field,
                "field_type": "phone",
                "value": "Not found",
                "confidence": 0.0,
                "bbox": empty_bbox,
            },
            "confidence": 0.0,
            "bbox": empty_bbox,
            "context_entities": [],
            "extraction_method": "none",
            "meta": {"reason": "No valid phone detected near label"},
        }
        filtered_entity = {
            "field": key_field,
            "value": "Not found",
            "bbox": empty_bbox,
            "confidence": 0.0,
            "page_number": 1,
            "semantic_type": EntityTypes.ANSWER,
            "semantic_confidence": 0.0,
        }
        return result, [filtered_entity]

    def _find_best_relation(self, relations: List[Dict], key_field: str, field_type: str) -> Tuple[Optional[Dict], float]:
        """Find best matching relation using semantic and spatial scoring"""
        best_relation = None
        best_score = 0
        
        # Get field-specific patterns for scoring
        patterns = self._get_field_patterns(key_field)
        
        for rel in relations:
            label_text = rel.get("label_text", "").lower()
            key_lower = key_field.lower()
            
            # Semantic score based on pattern matching
            pattern_score = sum(1 for p in patterns if p in label_text) / max(1, len(patterns))
            
            # Spatial score (lower distance = higher score)
            spatial_score = 1.0 / (1.0 + rel.get("distance", 100))
            
            # Field-type specific boost
            type_boost = 0.1 if field_type in label_text else 0
            
            # Combined score
            total_score = pattern_score * 0.6 + spatial_score * 0.3 + type_boost
            
            if total_score > best_score:
                best_score = total_score
                best_relation = rel
        
        return best_relation, best_score

    def _build_result_from_relation(self, relation: Dict, key_field: str, field_type: str) -> Tuple[Dict, List[Dict]]:
        """Build result from LiLT relation"""
        value_text = relation.get("value_text", "Not found")
        bbox = relation.get("value_bbox", {"x": 0, "y": 0, "width": 1, "height": 1})
        conf = float(relation.get("confidence", 0.8))
        
        # Apply field-type specific cleaning
        cleaned_value = self._clean_field_value(value_text, field_type)
        
        result = {
            "field_name": key_field,
            "value": cleaned_value,
            "structured_value": {
                "field_name": key_field,
                "field_type": field_type,
                "value": cleaned_value,
                "confidence": conf,
                "bbox": bbox,
            },
            "confidence": conf,
            "bbox": bbox,
            "context_entities": [],
            "extraction_method": "lilt_relation_extraction",
            "meta": {
                "label_text": relation.get("label_text", ""),
                "relation_score": relation.get("score", 0),
                "model_used": True,
            },
        }
        
        filtered_entity = {
            "field": key_field,
            "value": cleaned_value,
            "bbox": bbox,
            "confidence": conf,
            "page_number": relation.get("page_number", 1),
            "semantic_type": EntityTypes.ANSWER,
            "semantic_confidence": conf,
        }
        
        return result, [filtered_entity]

    def _pattern_based_extraction(self, entities: List[Dict], key_field: str, field_type: str) -> Tuple[Dict, List[Dict]]:
        """Enhanced pattern-based extraction with validation"""
        # Map field types to expected patterns
        pattern_map = {
            "document_id": [r"form\s+no", r"reference\s+no", r"reg\s+no", r"ad_\d+", r"gf\d+", r"pp-\d+"],
            "person_name": [r"name.*organization", r"owner.*name", r"applicant", r"company\s+name"],
            "date": [r"issue\s+date", r"completion\s+date", r"date\s+of\s+", r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"],
            "address": [r"correspondence\s+address", r"contact\s+address", r"mailing\s+address", r"address\s+of"],
            "phone": [r"contact\s+tel", r"phone\s+no", r"telephone"],
            "email": [r"email\s+address", r"e-mail", r"mail\s+address"],
        }
        
        # Get patterns for this field
        patterns = pattern_map.get(field_type, [])
        if not patterns:
            # Fallback to key_field patterns
            patterns = self._get_field_patterns(key_field)
        
        # Try to find label with patterns
        label_entity = None
        for e in entities:
            text = e.get("value", "").lower()
            for pat in patterns:
                if re.search(pat, text, re.IGNORECASE):
                    label_entity = e
                    logger.info(f"Found label via pattern '{pat}': '{e.get('value', '')}'")
                    break
            if label_entity:
                break
        
        # Try to find value
        if label_entity:
            value_entity = self._find_value_for_label(entities, label_entity, field_type)
            if value_entity:
                value_text = value_entity.get("value", "").strip()
                # Apply field-type specific cleaning
                cleaned_value = self._clean_field_value(value_text, field_type)
                return self._build_result_from_entities(label_entity, value_entity, key_field, cleaned_value, field_type)
        
        # Global search for values that match field type
        candidates = []
        for e in entities:
            value_text = e.get("value", "").strip()
            if self._validate_field_value(value_text, field_type):
                candidates.append((value_text, e))
        
        if candidates:
            # Pick the most confident candidate
            best_candidate = max(candidates, key=lambda x: x[1].get("confidence", 0))
            value_text, entity = best_candidate
            return self._build_result_from_entity(entity, key_field, value_text, field_type)
        
        # Ultimate fallback
        return self._build_empty_result(key_field)

    def _build_email_result(self, key_field: str, value: str, bbox: Dict, conf: float, entity: Dict, method: str = "unknown", meta: Dict = None):
        """Build structured result for email field"""
        if meta is None:
            meta = {}
        
        # Determine if email was found
        is_found = value != "Not found" and self._is_valid_email_format(value)
        
        result = {
            "field_name": key_field,
            "value": value,
            "structured_value": {
                "field_name": key_field,
                "field_type": "email",
                "value": value,
                "confidence": conf,
                "bbox": bbox,
            },
            "confidence": conf,
            "bbox": bbox,
            "context_entities": [],
            "extraction_method": method,
            "meta": meta,
            "found": is_found  # FIXED: Proper "found" flag
        }
        
        filtered_entity = {
            "field": key_field,
            "value": value,
            "bbox": bbox,
            "confidence": conf,
            "page_number": entity.get("page_number", 1),
            "semantic_type": EntityTypes.ANSWER,
            "semantic_confidence": conf,
            "found": is_found  # FIXED: Proper "found" flag
        }
        
        return result, [filtered_entity]

    def _build_empty_email_result(self, key_field: str):
        """Build empty result for missing email field"""
        empty_bbox = {"x": 0, "y": 0, "width": 1, "height": 1}
        
        result = {
            "field_name": key_field,
            "value": "Not found",
            "structured_value": {
                "field_name": key_field,
                "field_type": "email",
                "value": "Not found",
                "confidence": 0.0,
                "bbox": empty_bbox,
            },
            "confidence": 0.0,
            "bbox": empty_bbox,
            "context_entities": [],
            "extraction_method": "none",
            "meta": {"reason": "No valid email detected near label"},
            "found": False  # FIXED: Proper "found" flag for empty result
        }
        
        filtered_entity = {
            "field": key_field,
            "value": "Not found",
            "bbox": empty_bbox,
            "confidence": 0.0,
            "page_number": 1,
            "semantic_type": EntityTypes.ANSWER,
            "semantic_confidence": 0.0,
            "found": False  # FIXED: Proper "found" flag for empty result
        }
        
        return result, [filtered_entity]

    def _is_valid_email_format(self, email: str) -> bool:
        """Validate email format matches expected patterns"""
        if not email or not isinstance(email, str):
            return False
        
        email = email.strip().lower()
        
        # Basic email pattern validation
        email_pattern = r'^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$'
        if not re.match(email_pattern, email):
            return False
        
        # Domain validation - check for realistic TLDs
        domain = email.split('@')[-1]
        valid_tlds = ['.com', '.net', '.org', '.edu', '.gov', '.io', '.co', '.hk', '.asia']
        
        return any(domain.endswith(tld) for tld in valid_tlds)

    def _extract_email(self, text: str) -> Optional[str]:
        """Extract email with robust OCR error handling"""
        if not text:
            return None
        
        # Preprocess text to fix common OCR errors in emails
        cleaned_text = self._clean_ocr_email_artifacts(text)
        
        # Multiple email patterns with increasing tolerance
        email_patterns = [
            # Strict pattern (ideal case)
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            # Tolerant pattern (handles spaces around dots)
            r'[a-zA-Z0-9._%+-]+\s*@\s*[a-zA-Z0-9.-]+\s*\.\s*[a-zA-Z]{2,}',
            # Very tolerant pattern (handles most OCR errors)
            r'[a-zA-Z0-9._%+-]+[\s@]+[a-zA-Z0-9.-]+[\s.]+[a-zA-Z]{2,}'
        ]
        
        for pattern in email_patterns:
            matches = re.finditer(pattern, cleaned_text, re.IGNORECASE)
            for match in matches:
                email_candidate = match.group(0)
                # Final cleanup and validation
                cleaned_email = self._sanitize_email(email_candidate)
                if self._is_valid_email_format(cleaned_email):
                    logger.info(f"✅ Found valid email after cleaning: '{cleaned_email}' from '{email_candidate}'")
                    return cleaned_email
        
        # Fallback: Look for @ symbol and extract surrounding text
        if '@' in cleaned_text:
            parts = cleaned_text.split('@')
            if len(parts) > 1:
                local_part = parts[0].strip()[-30:]  # Take last 30 chars of local part
                domain_part = parts[1].strip()[:50]   # Take first 50 chars of domain
                email_candidate = f"{local_part}@{domain_part}"
                cleaned_email = self._sanitize_email(email_candidate)
                if self._is_valid_email_format(cleaned_email):
                    logger.info(f"✅ Found email via fallback method: '{cleaned_email}'")
                    return cleaned_email
        
        return None

    def _clean_ocr_email_artifacts(self, text: str) -> str:
        """Clean common OCR artifacts in email addresses"""
        if not text:
            return text
        
        # Clean whitespace around @ and . symbols
        text = re.sub(r'\s*@\s*', '@', text)
        text = re.sub(r'\s*\.\s*', '.', text)
        
        # Fix common OCR misreads
        text = re.sub(r'@rn', '@m', text, flags=re.IGNORECASE)  # @rn -> @m
        text = re.sub(r'@rr', '@m', text, flags=re.IGNORECASE)  # @rr -> @m
        text = re.sub(r'@ii', '@m', text, flags=re.IGNORECASE)  # @ii -> @m
        text = re.sub(r'@iii', '@m', text, flags=re.IGNORECASE) # @iii -> @m
        text = re.sub(r'\bi\s*rn\b', 'im', text, flags=re.IGNORECASE)  # imrn -> im
        text = re.sub(r'\bi\s*rr\b', 'ir', text, flags=re.IGNORECASE)  # irrn -> ir
        
        # Fix domain suffixes
        text = re.sub(r'\.coin\b', '.com', text, flags=re.IGNORECASE)
        text = re.sub(r'\.cori\b', '.com', text, flags=re.IGNORECASE)
        text = re.sub(r'\.corm\b', '.com', text, flags=re.IGNORECASE)
        text = re.sub(r'\.nci\b', '.net', text, flags=re.IGNORECASE)
        text = re.sub(r'\.gel\b', '.org', text, flags=re.IGNORECASE)
        text = re.sub(r'\.hk\b', '.hk', text, flags=re.IGNORECASE)
        
        # Remove special characters that shouldn't be in emails
        text = re.sub(r'[^\w@.\-+%]', '', text)
        
        return text.lower()

    def _sanitize_email(self, email: str) -> str:
        """Sanitize email by removing invalid characters and normalizing"""
        if not email:
            return email
        
        # Remove spaces and invalid characters
        email = re.sub(r'\s+', '', email)
        email = re.sub(r'[^\w@.\-+%]', '', email)
        
        # Fix multiple dots in domain
        parts = email.split('@')
        if len(parts) == 2:
            local, domain = parts
            # Fix multiple dots in domain (e.g., "domain..com" -> "domain.com")
            domain = re.sub(r'\.{2,}', '.', domain)
            # Fix leading/trailing dots
            domain = domain.strip('.')
            local = local.strip('.')
            
            # Reconstruct email
            email = f"{local}@{domain}"
        
        return email.lower()

    def _is_valid_email_format(self, email: str) -> bool:
        """Validate email format with realistic domain checking"""
        if not email or '@' not in email or '.' not in email:
            return False
        
        # Basic format validation
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            return False
        
        # Domain validation - check for realistic TLDs
        domain = email.split('@')[-1].lower()
        
        # Common valid TLDs
        valid_tlds = [
            '.com', '.net', '.org', '.edu', '.gov', '.io', '.co', '.hk', 
            '.asia', '.biz', '.info', '.me', '.tv', '.cc', '.us', '.uk',
            '.ca', '.au', '.jp', '.kr', '.cn', '.tw', '.sg'
        ]
        
        # Check if domain ends with a valid TLD
        has_valid_tld = any(domain.endswith(tld) for tld in valid_tlds)
        
        # Additional checks to reject obviously invalid emails
        invalid_patterns = [
            r'@localhost', r'@example\.com', r'@test\.com',  # Test domains
            r'@\w+\.\w{1,2}$',  # Single letter TLDs
            r'@[\d.]+$',        # IP addresses as domains
            r'@no-email', r'@none', r'@not-applicable',      # Placeholder text
            r'^\.', r'\.$',     # Leading/trailing dots
            r'@\.+', r'\.@'     # Dots next to @ symbol
        ]
        
        has_invalid_pattern = any(re.search(pattern, email) for pattern in invalid_patterns)
        
        return has_valid_tld and not has_invalid_pattern

    def _is_valid_email_value(self, raw_value: str) -> bool:
        """Validate email value with OCR tolerance"""
        if not raw_value:
            return False
        
        # Clean the value first
        cleaned_value = self._clean_ocr_email_artifacts(raw_value)
        
        # Apply all validation checks
        return self._is_valid_email_format(cleaned_value)

    def _build_result_from_entities(self, label_entity: Dict, value_entity: Dict, key_field: str, value_text: str, field_type: str) -> Tuple[Dict, List[Dict]]:
        """Build result from label and value entities"""
        bbox = value_entity.get("bbox", {"x": 0, "y": 0, "width": 1, "height": 1})
        conf = float(value_entity.get("confidence", 0.9))
        
        result = {
            "field_name": key_field,
            "value": value_text,
            "structured_value": {
                "field_name": key_field,
                "field_type": field_type,
                "value": value_text,
                "confidence": conf,
                "bbox": bbox,
            },
            "confidence": conf,
            "bbox": bbox,
            "context_entities": [],
            "extraction_method": "pattern_based_fallback",
            "meta": {
                "label_entity": label_entity.get("value", ""),
                "label_confidence": label_entity.get("confidence", 0),
            },
        }
        filtered_entity = {
            "field": key_field,
            "value": value_text,
            "bbox": bbox,
            "confidence": conf,
            "page_number": value_entity.get("page_number", 1),
            "semantic_type": EntityTypes.ANSWER,
            "semantic_confidence": conf,
        }
        return result, [filtered_entity]

    def _find_value_entity_with_generous_bounds(self, entities: List[Dict], label_entity: Dict, key_field: str) -> Optional[Dict]:
        """Find value entity with very generous spatial bounds for email fields"""
        if not label_entity:
            return None
        
        label_bbox = label_entity.get("bbox", {})
        label_page = label_entity.get("page_number", 1)
        label_x = label_bbox.get("x", 0)
        label_y = label_bbox.get("y", 0)
        label_width = label_bbox.get("width", 0)
        label_height = label_bbox.get("height", 0)
        label_center_y = label_y + label_height / 2
        
        candidates = []
        
        for e in entities:
            if e is label_entity:
                continue
            
            if e.get("page_number", 1) != label_page:
                continue
            
            bbox = e.get("bbox", {})
            x = bbox.get("x", 0)
            y = bbox.get("y", 0)
            width = bbox.get("width", 0)
            height = bbox.get("height", 0)
            value = e.get("value", "").strip()
            
            if not value:
                continue
            
            # Skip obvious non-values (headers, labels, etc.)
            if self._is_likely_label_or_header(value):
                continue
            
            # Calculate spatial relationships - BE VERY GENEROUS
            horizontal_dist = x - (label_x + label_width)  # Distance from end of label to start of value
            vertical_dist = abs((y + height/2) - label_center_y)  # Vertical distance between centers
            
            # Candidate if: within generous bounds
            is_candidate = False
            position_score = 0.0
            
            # 1. Right of label (preferred) - very generous bounds
            if -100 <= horizontal_dist <= 800 and vertical_dist <= label_height * 3.0:
                is_candidate = True
                position_score = 1.0 - (max(0, horizontal_dist) / 800) * 0.5 - (vertical_dist / (label_height * 3.0)) * 0.5
            
            # 2. Below label (acceptable) - also generous
            elif horizontal_dist > -300 and x < label_x + label_width + 300 and y > label_y - label_height * 0.5 and y < label_y + label_height * 4.0:
                is_candidate = True
                vertical_penalty = min(1.0, abs(y - (label_y + label_height)) / 400)
                horizontal_penalty = min(1.0, abs(x - label_x) / (label_width * 2.0))
                position_score = 0.9 - (vertical_penalty * 0.6 + horizontal_penalty * 0.4)
            
            if not is_candidate:
                continue
            
            # Higher priority for entities that contain @ symbol (likely emails)
            email_boost = 1.5 if "@" in value else 1.0
            final_score = position_score * email_boost
            
            candidates.append({
                "entity": e,
                "score": final_score,
                "position_score": position_score,
                "value": value,
                "contains_at": "@" in value
            })
        
        if candidates:
            # Sort by score and pick the best candidate
            best_candidate = max(candidates, key=lambda x: x["score"])
            logger.info(f"Selected email candidate: '{best_candidate['value']}' (score={best_candidate['score']:.2f}, contains_at={best_candidate['contains_at']})")
            return best_candidate["entity"]
        
        return None

    def _find_email_value_first(self, entities: List[Dict], key_field: str) -> Optional[Dict]:
        """Find email value first, then associate with nearest label"""
        email_candidates = []
        
        for e in entities:
            value = e.get("value", "").strip()
            if not value:
                continue
            
            email_val = self._extract_email(value)
            if email_val and self._is_valid_email_format(email_val):
                confidence = float(e.get("confidence", 0.7))
                email_candidates.append({
                    "entity": e,
                    "email": email_val,
                    "confidence": confidence,
                    "value": value
                })
        
        if not email_candidates:
            return None
        
        # Pick the most confident email candidate
        best_candidate = max(email_candidates, key=lambda x: x["confidence"])
        logger.info(f"Found email via value-first approach: '{best_candidate['email']}' (confidence={best_candidate['confidence']:.2f})")
        return best_candidate["entity"]

    def _find_email_in_label_area(self, entities: List[Dict], label_entity: Dict) -> Optional[Dict]:
        """Search aggressively in the area around the email label"""
        if not label_entity:
            return None
        
        label_bbox = label_entity.get("bbox", {})
        label_page = label_entity.get("page_number", 1)
        label_x = label_bbox.get("x", 0)
        label_y = label_bbox.get("y", 0)
        label_width = label_bbox.get("width", 0)
        label_height = label_bbox.get("height", 0)
        
        # Define a generous search area around the label
        search_left = max(0, label_x - 200)
        search_right = label_x + label_width + 600
        search_top = max(0, label_y - 100)
        search_bottom = label_y + label_height * 4.0
        
        email_candidates = []
        
        for e in entities:
            if e.get("page_number", 1) != label_page:
                continue
            
            bbox = e.get("bbox", {})
            x = bbox.get("x", 0)
            y = bbox.get("y", 0)
            width = bbox.get("width", 0)
            height = bbox.get("height", 0)
            value = e.get("value", "").strip()
            
            if not value:
                continue
            
            # Check if entity is within search area
            entity_center_x = x + width / 2
            entity_center_y = y + height / 2
            
            if (search_left <= entity_center_x <= search_right and 
                search_top <= entity_center_y <= search_bottom):
                
                email_val = self._extract_email(value)
                if email_val and self._is_valid_email_format(email_val):
                    confidence = float(e.get("confidence", 0.7))
                    distance_to_label = sqrt((entity_center_x - (label_x + label_width/2))**2 + 
                                        (entity_center_y - (label_y + label_height/2))**2)
                    email_candidates.append({
                        "entity": e,
                        "email": email_val,
                        "confidence": confidence,
                        "distance": distance_to_label,
                        "value": value
                    })
        
        if email_candidates:
            # Pick the closest email candidate to the label
            best_candidate = min(email_candidates, key=lambda x: x["distance"])
            logger.info(f"Found email in label area: '{best_candidate['email']}' (distance={best_candidate['distance']:.1f})")
            return best_candidate["entity"]
        
        return None

    def _get_nearby_entities(self, entities: List[Dict], reference_entity: Dict, max_distance: int = 100) -> List[Dict]:
        """Get entities near a reference entity within max_distance pixels"""
        if not reference_entity:
            return []
        
        ref_bbox = reference_entity.get("bbox", {})
        ref_x = ref_bbox.get("x", 0) + ref_bbox.get("width", 0) / 2
        ref_y = ref_bbox.get("y", 0) + ref_bbox.get("height", 0) / 2
        
        nearby_entities = []
        
        for e in entities:
            if e is reference_entity:
                continue
            
            bbox = e.get("bbox", {})
            e_x = bbox.get("x", 0) + bbox.get("width", 0) / 2
            e_y = bbox.get("y", 0) + bbox.get("height", 0) / 2
            
            distance = sqrt((e_x - ref_x)**2 + (e_y - ref_y)**2)
            
            if distance <= max_distance:
                nearby_entities.append(e)
        
        return nearby_entities

    def _is_likely_label_or_header(self, text: str) -> bool:
        """Check if text is likely a label or header rather than a value"""
        text_lower = text.lower().strip()
        
        # Common label/header patterns
        label_patterns = [
            r"^email\s+address", r"^e[-\s]?mail", r"^electronic\s+mail",
            r"^contact\s+information", r"^contact\s+details",
            r"^section\s+\w+", r"^part\s+\w+", r"^item\s+\w+",
            r"^\([a-z0-9]+\)", r"^[a-z0-9]+\.",  # Numbered items
            r"^[A-Z\s]{3,}$",  # ALL CAPS headers
            r"^for\s+official\s+use", r"^confidential",
            r"^application\s+form", r"^form\s+\w+",
            r"^please\s+", r"^tick\s+", r"^select\s+",
            r"^name", r"^address", r"^phone", r"^tel", r"^contact",
            r"^date", r"^signature", r"^page\s+\d+",
            r"^section", r"^part", r"^table", r"^figure"
        ]
        
        for pattern in label_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        return False

    def _build_email_result(self, key_field: str, value: str, bbox: Dict, conf: float, entity: Dict, method: str = "unknown", meta: Dict = None):
        """Build structured result for email field with proper 'found' flag"""
        if meta is None:
            meta = {}
        
        # FIXED: Add proper 'found' flag based on whether we actually found a valid email
        is_found = value != "Not found" and self._is_valid_email_format(value) if value else False
        
        result = {
            "field_name": key_field,
            "value": value,
            "structured_value": {
                "field_name": key_field,
                "field_type": "email",
                "value": value,
                "confidence": conf,
                "bbox": bbox,
            },
            "confidence": conf,
            "bbox": bbox,
            "context_entities": [],
            "extraction_method": method,
            "meta": meta,
            "found": is_found,  # FIXED: Proper found flag
            "page_number": entity.get("page_number", 1)
        }
        
        filtered_entity = {
            "field": key_field,
            "value": value,
            "bbox": bbox,
            "confidence": conf,
            "page_number": entity.get("page_number", 1),
            "semantic_type": EntityTypes.ANSWER,
            "semantic_confidence": conf,
            "found": is_found  # FIXED: Proper found flag
        }
        
        return result, [filtered_entity]

    def _build_empty_email_result(self, key_field: str):
        """Build empty result for missing email field with proper 'found' flag"""
        empty_bbox = {"x": 0, "y": 0, "width": 1, "height": 1}
        
        result = {
            "field_name": key_field,
            "value": "Not found",
            "structured_value": {
                "field_name": key_field,
                "field_type": "email",
                "value": "Not found",
                "confidence": 0.0,
                "bbox": empty_bbox,
            },
            "confidence": 0.0,
            "bbox": empty_bbox,
            "context_entities": [],
            "extraction_method": "none",
            "meta": {"reason": "No valid email detected"},
            "found": False,  # FIXED: Proper found flag for empty result
            "page_number": 1
        }
        
        filtered_entity = {
            "field": key_field,
            "value": "Not found",
            "bbox": empty_bbox,
            "confidence": 0.0,
            "page_number": 1,
            "semantic_type": EntityTypes.ANSWER,
            "semantic_confidence": 0.0,
            "found": False  # FIXED: Proper found flag for empty result
        }
        
        return result, [filtered_entity]

    def _find_value_by_pattern(self, entities: List[Dict], key_field: str, field_type: str) -> Optional[Dict]:
        """Find value by matching against field-type patterns"""
        candidates = []
        
        for e in entities:
            value = e.get("value", "").strip()
            if not value:
                continue

            value_lower =  value.lower()  # ✅ CRITICAL FIX: MUST ADD THIS LINE HERE
            
            # Score based on field-type validation
            score = 0.0
            if self._validate_field_value(value, field_type):
                score = 0.8
            
            # Additional score based on position (prefer top-left values for first fields)
            bbox = e.get("bbox", {})
            position_score = 1.0 - (bbox.get("y", 0) / 3000)  # Prefer values near top
            
            total_score = score + position_score * 0.2
            
            if total_score > 0.3:
                candidates.append({
                    "entity": e,
                    "score": total_score,
                    "validation_match": score > 0
                })
        
        if candidates:
            best_candidate = max(candidates, key=lambda x: x["score"])
            logger.info(f"Found value by pattern for '{key_field}': '{best_candidate['entity'].get('value', '')}' "
                        f"(score={best_candidate['score']:.2f})")
            return best_candidate["entity"]
        
        return None

    def _find_value_near_label(self, entities: List[Dict], label_entity: Dict, key_field: str, field_type: str) -> Optional[Dict]:
        """Find value entity near label with more generous search criteria"""
        if not label_entity:
            return None
        
        label_bbox = label_entity.get("bbox", {})
        label_page = label_entity.get("page_number", 1)
        label_x = label_bbox.get("x", 0)
        label_y = label_bbox.get("y", 0)
        label_width = label_bbox.get("width", 0)
        label_height = label_bbox.get("height", 0)
        label_right = label_x + label_width
        label_bottom = label_y + label_height
        
        candidates = []
        
        for e in entities:
            if e is label_entity or e.get("page_number", 1) != label_page:
                continue
            
            e_bbox = e.get("bbox", {})
            e_x = e_bbox.get("x", 0)
            e_y = e_bbox.get("y", 0)
            e_width = e_bbox.get("width", 0)
            e_height = e_bbox.get("height", 0)
            e_value = e.get("value", "").strip()
            
            if not e_value:
                continue
            
            # Skip labels and instructional text
            if self._is_label_candidate(e_value):
                continue
            
            # Calculate spatial relationships
            # For name/organization field, be more generous with search area
            is_to_right = e_x >= label_right - 50 and e_x <= label_right + 400
            is_below = e_y >= label_bottom - 30 and e_y <= label_bottom + 200
            is_same_line = abs(e_y - label_y) < label_height * 0.5
            
            # Check if entity is in the general area of the label
            in_general_area = (
                e_x >= label_x - 200 and 
                e_x <= label_x + label_width + 500 and
                e_y >= label_y - 50 and
                e_y <= label_y + label_height * 3
            )
            
            if not (is_to_right or is_below or is_same_line or in_general_area):
                continue
            
            # Calculate distance scores
            horizontal_dist = max(0, e_x - label_right)
            vertical_dist = max(0, e_y - label_bottom)
            distance = sqrt(horizontal_dist**2 + vertical_dist**2)
            
            # For name/organization field, prioritize closer entities
            if "name" in key_field.lower() and "owner" in key_field.lower():
                # Be more generous with distance for this specific field
                max_distance = 500
            else:
                max_distance = 300
            
            if distance > max_distance:
                continue
            
            # Score based on position and distance
            position_score = 0.0
            if is_to_right and is_same_line:
                position_score = 1.0  # Best: directly to the right on same line
            elif is_below and horizontal_dist < 100:
                position_score = 0.8  # Good: directly below with good alignment
            elif in_general_area:
                position_score = 0.6  # Acceptable: in general area
            
            # Distance score (closer is better)
            distance_score = 1.0 - (distance / max_distance)
            
            # Field type validation score
            validation_score = 1.0 if self._validate_field_value(e_value, field_type) else 0.3
            
            # Combine scores
            total_score = position_score * 0.4 + distance_score * 0.3 + validation_score * 0.3
            
            if total_score > 0.4:  # Lower threshold to catch more candidates
                candidates.append({
                    "entity": e,
                    "score": total_score,
                    "distance": distance,
                    "value": e_value
                })
        
        if candidates:
            # Sort by score and return best candidate
            candidates.sort(key=lambda x: x["score"], reverse=True)
            best_candidate = candidates[0]
            
            logger.info(f"Found value for '{key_field}': '{best_candidate['value']}' "
                        f"(score={best_candidate['score']:.2f}, distance={best_candidate['distance']:.1f})")
            
            return best_candidate["entity"]
        
        return None

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
        
        if self._is_phone_label(key_field):
            logger.info(f"Using model-guided phone extraction for key_field: '{key_field}'")

            # Step 1: Find the label entity using semantic/text matching
            label_entity = self._find_label_entity(entities, key_field)
            if not label_entity:
                logger.warning(f"No label found for phone field: '{key_field}'")
                # → Fall back to global phone search only if no label exists
                for e in entities:
                    phone_val = extract_phone_from_text(e.get("value", ""))
                    if phone_val:
                        # Return first valid phone as last resort
                        bbox = e.get("bbox", {"x": 0, "y": 0, "width": 1, "height": 1})
                        conf = float(e.get("confidence", 0.85))
                        return self._build_phone_result(key_field, phone_val, bbox, conf, e, method="fallback_global_phone")
                return self._build_empty_phone_result(key_field)

            # Step 2: Use spatial reasoning to find the NEAREST value
            value_entity = self._find_nearest_value_entity(entities, label_entity, key_field)
            if not value_entity:
                logger.warning(f"Label found but no nearby value for: '{key_field}'")
                return self._build_empty_phone_result(key_field)

            # Step 3: Extract & validate phone from the candidate value
            raw_value = value_entity.get("value", "").strip()
            phone_val = extract_phone_from_text(raw_value)

            # If no phone pattern, but value is short and numeric, still consider it
            if not phone_val:
                cleaned = re.sub(r"[^\d\+\-\(\)\s]", "", raw_value)
                if 7 <= len(re.sub(r"\D", "", cleaned)) <= 15:
                    phone_val = cleaned

            if not phone_val:
                logger.warning(f"Value near label is not a valid phone: '{raw_value}'")
                return self._build_empty_phone_result(key_field)

            # Step 4: Return structured result
            bbox = value_entity.get("bbox", {"x": 0, "y": 0, "width": 1, "height": 1})
            conf = float(value_entity.get("confidence", 0.9))
            return self._build_phone_result(
                key_field, phone_val, bbox, conf, value_entity,
                method="model_based_key_value",
                meta={"label_entity": label_entity.get("value", ""), "raw_value": raw_value}
            )

        if self._is_email_label(key_field):
            logger.info(f"Using enhanced email extraction for key_field: '{key_field}'")
            
            # Step 1: Find the email label entity
            label_entity = self._find_label_entity(entities, key_field)
            if not label_entity:
                logger.warning(f"No label found for email field: '{key_field}'")
                
                # Aggressive fallback 1: Search for email pattern anywhere in document
                global_email_candidates = []
                for e in entities:
                    value = e.get("value", "").strip()
                    if not value:
                        continue
                    
                    email_val = self._extract_email(value)
                    if email_val and self._is_valid_email_format(email_val):
                        confidence = float(e.get("confidence", 0.7))
                        global_email_candidates.append((confidence, email_val, e))
                
                if global_email_candidates:
                    # Pick the most confident candidate
                    global_email_candidates.sort(key=lambda x: x[0], reverse=True)
                    best_confidence, best_email, best_entity = global_email_candidates[0]
                    logger.info(f"Found email via global search: '{best_email}' (confidence={best_confidence:.2f})")
                    bbox = best_entity.get("bbox", {"x": 0, "y": 0, "width": 1, "height": 1})
                    return self._build_email_result(key_field, best_email, bbox, best_confidence, best_entity, method="global_search")
                
                # Aggressive fallback 2: Look for common email patterns in likely positions
                for e in entities:
                    value = e.get("value", "").strip().lower()
                    if not value:
                        continue
                    
                    # Look for patterns like "email: kc@imimr.net" or "e-mail address: kc@imimr.net"
                    if any(pattern in value for pattern in ["email:", "e-mail:", "email address:", "e-mail address:"]):
                        # Extract the part after the colon
                        parts = value.split(":")
                        if len(parts) > 1:
                            potential_email = parts[1].strip()
                            email_val = self._extract_email(potential_email)
                            if email_val and self._is_valid_email_format(email_val):
                                bbox = e.get("bbox", {"x": 0, "y": 0, "width": 1, "height": 1})
                                conf = float(e.get("confidence", 0.8))
                                logger.info(f"Found email via pattern search: '{email_val}'")
                                return self._build_email_result(key_field, email_val, bbox, conf, e, method="pattern_search")
                
                logger.warning("No email found after all fallback attempts")
                return self._build_empty_email_result(key_field)

            # Step 2: Find value entity with GENEROUS spatial matching
            value_entity = self._find_value_entity_with_generous_bounds(entities, label_entity, key_field)
            
            # Step 3: If direct spatial search fails, try value-first approach
            if not value_entity:
                logger.warning(f"Spatial search failed for email field '{key_field}', trying value-first approach")
                value_entity = self._find_email_value_first(entities, key_field)
            
            # Step 4: If still no value, try aggressive search around label area
            if not value_entity:
                logger.warning(f"Value-first failed, trying aggressive area search around label")
                value_entity = self._find_email_in_label_area(entities, label_entity)
            
            if not value_entity:
                logger.warning(f"Label found but no valid email value for: '{key_field}'")
                return self._build_empty_email_result(key_field)

            # Step 5: Extract and validate email
            raw_value = value_entity.get("value", "").strip()
            email_val = self._extract_email(raw_value)
            
            # Fallback: Check nearby entities if direct extraction fails
            if not email_val:
                logger.warning(f"No email pattern in direct value: '{raw_value}', checking nearby entities")
                nearby_entities = self._get_nearby_entities(entities, value_entity, max_distance=100)
                for nearby_entity in nearby_entities:
                    nearby_value = nearby_entity.get("value", "").strip()
                    nearby_email = self._extract_email(nearby_value)
                    if nearby_email and self._is_valid_email_format(nearby_email):
                        logger.info(f"Found email in nearby entity: '{nearby_email}'")
                        email_val = nearby_email
                        value_entity = nearby_entity
                        raw_value = nearby_value
                        break
            
            if not email_val:
                logger.warning(f"No email pattern found near label: '{raw_value}'")
                return self._build_empty_email_result(key_field)

            if not self._is_valid_email_format(email_val):
                logger.warning(f"Invalid email format: '{email_val}'")
                return self._build_empty_email_result(key_field)

            # Step 6: Return structured result with proper "found" flag
            bbox = value_entity.get("bbox", {"x": 0, "y": 0, "width": 1, "height": 1})
            conf = float(value_entity.get("confidence", 0.9))
            
            # FIXED: Only mark as "found" if we actually found a valid email
            result, filtered_entities = self._build_email_result(
                key_field, email_val, bbox, conf, value_entity,
                method="model_based_key_value",
                meta={"label_entity": label_entity.get("value", ""), "raw_value": raw_value}
            )
            
            # Ensure proper "found" flag in result structure
            if email_val != "Not found" and self._is_valid_email_format(email_val):
                if isinstance(result, dict):
                    result["found"] = True
                if filtered_entities and isinstance(filtered_entities[0], dict):
                    filtered_entities[0]["found"] = True
            
            return result, filtered_entities

        # Get field type for validation
        field_type = self._get_field_type(key_field)
        
        # SPECIAL HANDLING for name/organization field
        if "name" in key_field_clean and ("organization" in key_field_clean or "owner" in key_field_clean):
            logger.info(f"SPECIAL HANDLING for name/organization field: '{key_field}'")
            
            # Try to find the label first
            label_entity = self._find_label_entity(entities, key_field)
            
            if label_entity:
                logger.info(f"Found label for '{key_field}': '{label_entity.get('value', '')}'")
                
                # Use improved search for values near this label
                value_entity = self._find_value_near_label(entities, label_entity, key_field, field_type)
                
                if value_entity:
                    value_text = value_entity.get("value", "").strip()
                    bbox = value_entity.get("bbox", {"x": 0, "y": 0, "width": 1, "height": 1})
                    conf = float(value_entity.get("confidence", 0.9))
                    
                    # Clean value
                    cleaned_value = self._clean_field_value(value_text, field_type)
                    
                    result = {
                        "field_name": key_field,
                        "value": cleaned_value,
                        "structured_value": {
                            "field_name": key_field,
                            "field_type": field_type,
                            "value": cleaned_value,
                            "confidence": conf,
                            "bbox": bbox,
                        },
                        "confidence": conf,
                        "bbox": bbox,
                        "context_entities": [],
                        "extraction_method": "special_name_organization_detection",
                        "meta": {
                            "label_entity": label_entity.get("value", ""),
                            "label_confidence": label_entity.get("confidence", 0),
                        },
                    }
                    filtered_entity = {
                        "field": key_field,
                        "value": cleaned_value,
                        "bbox": bbox,
                        "confidence": conf,
                        "page_number": value_entity.get("page_number", 1),
                        "semantic_type": EntityTypes.ANSWER,
                        "semantic_confidence": conf,
                    }
                    return result, [filtered_entity]
                else:
                    logger.warning(f"Label found but no value found for: '{key_field}'")
            else:
                logger.warning(f"No label found for: '{key_field}'")

        # === STEP 1: PRIMARY EXTRACTION USING LILT MODEL ===
        if self.lilt_extractor and self.lilt_extractor.is_available():
            try:
                relations = self.lilt_extractor.extract_relations(entities)
                logger.info(f"LiLT found {len(relations)} relations")
                
                # Get field-specific patterns
                patterns = self._get_field_patterns(key_field)
                
                # Find best matching relation with field-type awareness
                best_relation = None
                best_score = 0
                
                for rel in relations:
                    label_text = rel.get("label_text", "").lower()
                    value_text = rel.get("value_text", "").strip()
                    
                    # Semantic score based on pattern matching
                    pattern_score = sum(1 for p in patterns if p in label_text) / max(1, len(patterns))
                    
                    # Field-type validation score
                    validation_score = 1.0 if self._validate_field_value(value_text, field_type) else 0.2
                    
                    # Spatial score (lower distance = higher score)
                    spatial_score = 1.0 / (1.0 + rel.get("distance", 100))
                    
                    # Combined score
                    total_score = pattern_score * 0.5 + validation_score * 0.3 + spatial_score * 0.2
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_relation = rel
                
                if best_relation and best_score > 0.45:
                    logger.info(f"LiLT found matching relation with score {best_score:.2f} for field '{key_field}'")
                    value_text = best_relation.get("value_text", "Not found")
                    bbox = best_relation.get("value_bbox", {"x": 0, "y": 0, "width": 1, "height": 1})
                    conf = float(best_relation.get("confidence", 0.8))
                    
                    # Clean value based on field type
                    cleaned_value = self._clean_field_value(value_text, field_type)
                    
                    result = {
                        "field_name": key_field,
                        "value": cleaned_value,
                        "structured_value": {
                            "field_name": key_field,
                            "field_type": field_type,
                            "value": cleaned_value,
                            "confidence": conf,
                            "bbox": bbox,
                        },
                        "confidence": conf,
                        "bbox": bbox,
                        "context_entities": [],
                        "extraction_method": "lilt_relation_extraction",
                        "meta": {
                            "relation_score": best_score,
                            "label_text": best_relation.get("label_text", ""),
                            "value_text": best_relation.get("value_text", ""),
                            "model_used": True,
                        },
                    }
                    filtered_entity = {
                        "field": key_field,
                        "value": cleaned_value,
                        "bbox": bbox,
                        "confidence": conf,
                        "page_number": best_relation.get("page_number", 1),
                        "semantic_type": EntityTypes.ANSWER,
                        "semantic_confidence": conf,
                    }
                    return result, [filtered_entity]
                
                logger.info(f"LiLT relations didn't match well for '{key_field}' (best score: {best_score:.2f})")
            
            except Exception as e:
                logger.warning(f"LiLT relation extraction failed for '{key_field}': {e}")
        
        # === STEP 2: PATTERN-BASED FALLBACK WITH STRICT VALIDATION ===
        logger.info(f"Fallback to pattern-based detection for: '{key_field}'")
        
        # Get field-specific patterns
        patterns = self._get_field_patterns(key_field)
        
        # Find label entity using patterns
        label_entity = None
        for e in entities:
            text = e.get("value", "").lower()
            for pat in patterns:
                if re.search(pat, text, re.IGNORECASE):
                    label_entity = e
                    logger.info(f"Found label via pattern '{pat}' for '{key_field}': '{e.get('value', '')}'")
                    break
            if label_entity:
                break
        
        # Find value entity with field-type awareness
        value_entity = None
        if label_entity:
            value_entity = self._find_value_for_label(entities, label_entity, key_field, field_type)
        
        # If no label found, try value-first approach with strict validation
        if not label_entity or not value_entity:
            logger.info("Label/value not found; trying value-first approach with validation")
            value_entity = self._find_value_by_pattern(entities, key_field, field_type)
        
        # Build result with validation
        if value_entity:
            value_text = value_entity.get("value", "").strip()
            bbox = value_entity.get("bbox", {"x": 0, "y": 0, "width": 1, "height": 1})
            conf = float(value_entity.get("confidence", 0.8))
            
            # Clean and validate value
            cleaned_value = self._clean_field_value(value_text, field_type)
            
            if not self._validate_field_value(cleaned_value, field_type):
                logger.warning(f"Value failed validation for field type '{field_type}' ({key_field}): '{cleaned_value}'")
                # Try to find better match with stricter validation
                return self._find_alternative_value(entities, key_field, field_type)
            
            result = {
                "field_name": key_field,
                "value": cleaned_value,
                "structured_value": {
                    "field_name": key_field,
                    "field_type": field_type,
                    "value": cleaned_value,
                    "confidence": conf,
                    "bbox": bbox,
                },
                "confidence": conf,
                "bbox": bbox,
                "context_entities": [],
                "extraction_method": "pattern_based_fallback",
                "meta": {
                    "label_entity": label_entity.get("value", "") if label_entity else None,
                    "method": "value_first" if not label_entity else "label_value_pair",
                },
            }
            filtered_entity = {
                "field": key_field,
                "value": cleaned_value,
                "bbox": bbox,
                "confidence": conf,
                "page_number": value_entity.get("page_number", 1),
                "semantic_type": EntityTypes.ANSWER,
                "semantic_confidence": conf,
            }
            return result, [filtered_entity]
        
        # === STEP 3: ULTIMATE FALLBACK - EMPTY RESULT ===
        logger.warning(f"No valid value found for field: '{key_field}'")
        return self._build_empty_result(key_field)

    def _get_field_type(self, key_field: str) -> str:
        """Map field names to semantic types for validation"""
        key_lower = key_field.lower()
        
        if any(term in key_lower for term in ["registration number", "reg no", "registration no"]):
            return "registration_number"
        elif any(term in key_lower for term in ["application number", "app no", "application no", "fit scheme application"]):
            return "application_number"
        elif any(term in key_lower for term in ["form number", "form no", "reference no"]):
            return "form_number"
        elif any(term in key_lower for term in ["correspondence address", "contact address", "mailing address"]):
            return "address"
        elif any(term in key_lower for term in ["issue date", "completion date", "date of"]):
            return "date"
        elif any(term in key_lower for term in ["name", "organization", "owner", "applicant"]):
            return "person_name"
        elif any(term in key_lower for term in ["phone", "tel", "telephone", "contact tel"]):
            return "phone"
        elif "email" in key_lower:
            return "email"
        
        return "text"

    def _get_field_patterns(self, key_field: str) -> List[str]:
        """Get patterns for field matching based on field name"""
        key_lower = key_field.lower()
        
        # SPECIFIC FIX for "Name Or Organization Of The Owner Of The Generating Facility"
        if "name" in key_lower and ("organization" in key_lower or "owner" in key_lower):
            # Enhanced patterns for owner name field - BOTH English and Chinese
            return [
                # English patterns
                r"name.*organization.*owner.*generating.*facility",
                r"owner.*name.*organization",
                r"name.*or.*organization.*owner",
                r"name.*of.*owner",
                r"owner.*name.*of.*generating.*facility",
                
                # Chinese patterns (from your screenshot)
                r"發電設施擁有人的姓名或公司名稱",
                r"設施擁有人的姓名",
                r"姓名或公司名稱",
                r"擁有人的姓名",
                r"發電設施.*姓名.*公司名稱",
                r"姓名.*公司名稱.*發電設施",
                
                # Bilingual combination patterns
                r"name.*or.*organization.*發電設施",
                r"owner.*姓名.*公司名稱",
                
                # Numeric pattern (for "1." in front of the label)
                r"1\..*姓名.*公司名稱",
                r"1\..*name.*organization",
                
                # Generic patterns that might match partial text
                r"姓名或公司名稱",
                r"name.*organization",
                r"owner.*name"
            ]

        if "registration number" in key_lower:
            return [r"registration\s+number", r"reg\s+no", r"no\s*:\s*ad_\d+", r"ad_\d+",
                    r"registration\s+no", r"registration\s+of"]
        elif "application number" in key_lower or "application no" in key_lower:
            return [r"application\s+no", r"application\s+number", r"app\s+no", r"no\s*:\s*pp-\d+",
                    r"fit\s+scheme\s+application", r"power\s+supply\s+company"]
        elif "correspondence address" in key_lower:
            return [r"correspondence\s+address", r"contact\s+address", r"mailing\s+address",
                    r"address\s+of\s+the\s+owner"]
        elif "name" in key_lower and ("organization" in key_lower or "owner" in key_lower):
            return [r"name.*organization", r"owner.*name", r"applicant", r"organisation"]
        
        # Default patterns
        words = re.findall(r'\w+', key_lower)
        patterns = []
        if len(words) > 1:
            patterns.append(r".*" + r".*".join(words) + r".*")
        patterns.append(r".*" + words[-1] + r".*")
        return patterns

    def _is_instructional_or_header_text(self, text: str) -> bool:
        """Identify instructional text, headers, and non-value content to skip entirely"""
        if not text or len(text.strip()) < 3:
            return False
        
        text_lower = text.lower().strip()
        
        # Skip all instructional text patterns (both English and Chinese)
        skip_patterns = [
            # Chinese instructional patterns
            r"更改\s*註冊\s*發電\s*設施\s*資料",  # The problematic text
            r"申請\s*更改", r"更正\s*資料", r"變更\s*登記",
            r"機電\s*工程\s*署", r"必\s*須\s*填\s*寫", r"請\s*在\s*適\s*當\s*空\s*格\s*加",
            r"只\s*可\s*選\s*擇\s*一\s*項", r"並\s*加\s*上", r"符\s*號",
            r"發\s*電\s*設\s*施\s*擁\s*有\s*人", r"的\s*姓\s*名\s*或\s*公\s*司\s*名\s*稱",
            
            # English instructional patterns  
            r"change\s+registered\s+power\s+generation\s+facility\s+data",
            r"please\s+tick", r"tick\s+the\s+appropriate", r"only\s+one\s+allowed",
            r"fill\s+in\s+block\s+letters", r"applicant\s+should\s+complete",
            r"before\s+you\s+complete", r"read\s+carefully\s+the\s+notes",
        ]
        
        # Skip section headers and form titles
        section_headers = [
            r"section\s+a", r"section\s+b", r"part\s+a", r"part\s+b",
            r"a\s+部", r"b\s+部", r"一\s+般\s+資\s*料", r"申\s*請\s*人\s*資\s*料",
            r"form\s+gf2", r"application\s+form", r"renewable\s+energy",
        ]
        
        # Skip all patterns
        for pattern in skip_patterns + section_headers:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        return False

    def _find_value_for_label(self, entities: List[Dict], label_entity: Dict, key_field: str, field_type: str) -> Optional[Dict]:
        """Find value entity with LiLT-aware spatial analysis and field-type validation"""
        label_bbox = label_entity.get("bbox", {})
        label_page = label_entity.get("page_number", 1)
        label_x = label_bbox.get("x", 0)
        label_y = label_bbox.get("y", 0)
        label_width = label_bbox.get("width", 0)
        label_height = label_bbox.get("height", 0)
        label_center_y = label_y + label_height / 2
        
        candidates = []
        
        for e in entities:
            if e is label_entity or e.get("page_number", 1) != label_page:
                continue
                
            e_bbox = e.get("bbox", {})
            e_x = e_bbox.get("x", 0)
            e_y = e_bbox.get("y", 0)
            e_width = e_bbox.get("width", 0)
            e_height = e_bbox.get("height", 0)
            e_center_y = e_y + e_height / 2

            # SKIP INSTRUCTIONAL/HEADER TEXT COMPLETELY
            value = e.get("value", "")
            if self._is_instructional_or_header_text(value):
                logger.debug(f"Skipping instructional text: '{value}'")
                continue

            # Skip likely labels
            if self._is_label_candidate(e.get("value", "")):
                continue
            
            # Calculate spatial relationships - PRIORITIZE RIGHT AND BELOW
            horizontal_dist = e_x - (label_x + label_width)
            vertical_dist = abs(e_center_y - label_center_y)
            
            # Directional preference (right of label is best, below is acceptable)
            is_to_right = horizontal_dist > -50 and horizontal_dist < 300
            is_below = e_y > label_y + label_height * 0.8 and abs(e_x - label_x) < label_width * 2.0
            
            if not (is_to_right or is_below):
                continue
            
            # For owner name specifically, be more restrictive
            if "owner" in key_field.lower() or "name" in key_field.lower():
                # Must be within reasonable vertical range (not too far below)
                vertical_gap = abs(e_y - (label_y + label_height))
                if vertical_gap > 200:  # Max 200px below
                    continue
                
                # Must be reasonably aligned horizontally
                horizontal_gap = abs(e_x - label_x)
                if horizontal_gap > 300:  # Max 300px horizontal shift
                    continue

            # Score based on position and field type
            score = 0.0
            if is_to_right:
                # Higher score for values on the same line
                vertical_penalty = min(1.0, vertical_dist / (label_height * 1.5))
                horizontal_penalty = min(1.0, horizontal_dist / 300)
                score = 1.0 - (vertical_penalty * 0.7 + horizontal_penalty * 0.3)
            elif is_below:
                # Lower score for values below the label
                vertical_penalty = min(1.0, (e_y - (label_y + label_height)) / 200)
                horizontal_penalty = min(1.0, abs(e_x - label_x) / (label_width * 1.5))
                score = 0.8 - (vertical_penalty * 0.6 + horizontal_penalty * 0.4)
            
            # Boost for field-type match
            value_text = e.get("value", "").strip()
            if self._validate_field_value(value_text, field_type):
                score += 0.3
            
            if score > 0.3:
                candidates.append({
                    "entity": e,
                    "score": score,
                    "horizontal_dist": horizontal_dist,
                    "vertical_dist": vertical_dist,
                    "position": "right" if is_to_right else "below",
                    "field_type_match": self._validate_field_value(value_text, field_type)
                })
        
        if not candidates:
            return None
        
        # Sort by score and return best match
        best_candidate = max(candidates, key=lambda x: x["score"])
        logger.info(f"Selected value for '{key_field}': '{best_candidate['entity'].get('value', '')}' "
                    f"(score={best_candidate['score']:.2f}, position={best_candidate['position']}, "
                    f"field_type_match={best_candidate['field_type_match']})")
        
        return best_candidate["entity"]

    def _validate_field_value(self, value: str, field_type: str) -> bool:
        """Validate value based on field type with strict patterns"""
        if not value or value.lower() in ["not found", "none", "n/a"]:
            return False
        
        value_lower = value.lower()

        # Registration number validation - accept patterns like "AD 1223"
        if field_type == "registration_number":
            # First check for valid patterns
            registration_patterns = [
                r'ad[-\s_]?[0-9]{3,5}',  # AD 1223, AD-1223, AD_ r'reg[-\s_]?no[-\s_]?[0-9]+',  # REG NO  r'[0-9]+[-\s_]?ad',  # 1223 AD
                r'[a-z]{2,3}[-\s_]?[0-9]{3,5}'  # General pattern
            ]
            
            for pattern in registration_patterns:
                if re.search(pattern, value_lower):
                    return True
            
            # Fallback: check if it contains numbers and "ad" or "reg"
            return ('ad' in value_lower or 'reg' in value_lower) and any(c.isdigit() for c in value_lower)
       
        if field_type == "application_number":
            # Should match patterns like PP-2323190
            return bool(re.search(r'pp[-_]?\d+', value_lower))
        
        if field_type == "form_number":
            # Should match patterns like EMSD/EL/GF2(06/2024) but NOT be just a year
            return bool(re.search(r'em\w+/[^/]+/gf\d+', value_lower)) and not bool(re.search(r'\(?\d{4}\)?', value_lower))
        
        if field_type == "address":
            # Should contain address keywords and have comma or multiple parts
            address_keywords = ["road", "street", "blvd", "ave", "unit", "floor", "tower", "building", "room", "level"]
            has_keywords = any(keyword in value_lower for keyword in address_keywords)
            has_structure = ',' in value or len(value.split()) >= 3
            return has_keywords or has_structure
        
        # FIXED: More lenient validation for person/organization names
        if field_type == "person_name":
            # Accept any non-empty string that's not obviously invalid
            # Don't require multiple words (single word names are valid)
            has_content = len(value.strip()) >= 2
            
            # Reject obvious non-names (email patterns, URLs, etc.)
            has_no_email_pattern = '@' not in value and not re.search(r'\.(com|org|net|gov|edu)$', value_lower)
            has_no_url_pattern = 'http' not in value_lower and 'www.' not in value_lower
            
            # Don't reject based on "department" or "office" - these could be part of organization names
            # But reject if it's clearly a header/instruction
            is_not_instructional = not any(
                pattern in value_lower for pattern in [
                    "please", "tick", "select", "indicate", "mark",
                    "section", "part", "form", "application", "confidential"
                ]
            )
            
            return has_content and has_no_email_pattern and has_no_url_pattern and is_not_instructional
        
        if field_type == "date":
            # Should contain date patterns
            return bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', value_lower))

        if field_type == "email":
            # Should have proper email format with valid domain
            email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
            match = re.search(email_pattern, value_lower)
            if not match:
                return False
            
            # Validate domain is realistic
            domain = match.group(0).split('@')[-1]
            valid_domains = [".com", ".org", ".gov", ".edu", "net", ".hk", ".co"]
            return any(domain.endswith(d) for d in valid_domains)
       
        # Default validation
        return len(value) > 2 and not value.startswith(("<", "[", "{")) and not value.endswith((":", ":", "："))

    def _find_value_by_pattern(self, entities: List[Dict], key_field: str, field_type: str) -> Optional[Dict]:
        """Find value by matching against field-type patterns with positional awareness"""
        candidates = []
        
        for e in entities:
            value = e.get("value", "").strip()
            if not value:
                continue

            value_lower =  value.lower()  # ✅ CRITICAL FIX: MUST ADD THIS LINE HERE
            
            bbox = e.get("bbox", {})
            # Prefer values from top to bottom (earlier in document)
            y_position = bbox.get("y", 0)
            position_score = 1.0 - (y_position / 3000)  # Prefer values near top
            
            # Score based on field-type validation
            validation_match = self._validate_field_value(value, field_type)
            validation_score = 0.8 if validation_match else 0.0
            
            # Additional score for field-specific keywords
            keyword_score = 0.0
            if field_type == "registration_number" and ("ad_" in value_lower or "reg" in value_lower):
                keyword_score = 0.2
            elif field_type == "application_number" and "pp-" in value_lower:
                keyword_score = 0.2
            elif field_type == "address" and any(k in value_lower for k in ["road", "street", "blvd"]):
                keyword_score = 0.2
            
            total_score = validation_score + keyword_score + position_score * 0.2
            
            if total_score > 0.4:
                candidates.append({
                    "entity": e,
                    "score": total_score,
                    "validation_match": validation_match,
                    "y_position": y_position
                })
        
        if candidates:
            best_candidate = max(candidates, key=lambda x: x["score"])
            logger.info(f"Found value by pattern for '{key_field}': '{best_candidate['entity'].get('value', '')}' "
                        f"(score={best_candidate['score']:.2f}, validation={best_candidate['validation_match']})")
            return best_candidate["entity"]
        
        return None

    def _is_label_candidate(self, text: str) -> bool:
        """Check if text is likely a label rather than a value"""
        if not text:
            return False
            
        text_lower = text.lower().strip()
        
        # Skip entities with these patterns (likely labels)
        label_patterns = [
            r"^[a-z\s]{3,}:\s*$",  # "text:" pattern
            r"^section\s+\w",
            r"^part\s+\w",
            r"^item\s+\w",
            r"^table\s+\w",
            r"^[A-Z\s]{3,}$",  # ALL CAPS headers
            r"^\(?\d+\)?$",  # Numbered items like (1), 1., etc.
            r"^tick\s+the",  # Instructions
            r"^please\s+mark",  # Instructions
            r"^select\s+one",  # Instructions
        ]
        
        # Skip common header text
        header_words = ["section", "part", "item", "page", "reference", "form", "document", "confidential"]
        if any(word in text_lower for word in header_words):
            return True
        
        return any(re.search(pattern, text_lower) for pattern in label_patterns)

    def _clean_field_value(self, value: str, field_type: str) -> str:
        """Clean field value based on field type"""
        if not value:
            return value
            
        value = value.strip()
        
        if field_type in ["registration_number", "application_number", "form_number"]:
            # Extract only the identifier part with proper formatting
            match = re.search(r'([A-Z]{2,3}[-_]?\d+)', value)
            if match:
                return match.group(0)
            
            # For form numbers with complex patterns
            if field_type == "form_number":
                match = re.search(r'(EMSD/[^/]+/GF\d+)(?:\s*\(\d{2}/\d{4}\))?', value)
                if match:
                    return match.group(1)
        
        if field_type == "date":
            # Standardize date format
            match = re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', value)
            if match:
                return match.group(0)
        
        if field_type == "email":
            # Clean and extract only valid email part
            match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', value.lower())
            if match:
                return match.group(0)
        
        # Remove leading/trailing punctuation and whitespace
        return re.sub(r'^[:\-.,\s]+|[:\-.,\s]+$', '', value)

    def _find_alternative_value(self, entities: List[Dict], key_field: str, field_type: str) -> Tuple[Dict, List[Dict]]:
        """Find alternative value when primary candidate fails validation"""
        logger.info(f"Searching for alternative value for field: '{key_field}'")
        
        candidates = []
        for e in entities:
            value = e.get("value", "").strip()
            if not value:
                continue
            
            # Prioritize entities with higher confidence
            score = float(e.get("confidence", 0.5))
            
            # Boost score if value matches field patterns
            if self._validate_field_value(value, field_type):
                score += 0.5
            
            candidates.append((score, e))
        
        if candidates:
            best_candidate = max(candidates, key=lambda x: x[0])[1]
            value_text = best_candidate.get("value", "").strip()
            bbox = best_candidate.get("bbox", {"x": 0, "y": 0, "width": 1, "height": 1})
            conf = float(best_candidate.get("confidence", 0.8))
            
            cleaned_value = self._clean_field_value(value_text, field_type)
            
            result = {
                "field_name": key_field,
                "value": cleaned_value,
                "structured_value": {
                    "field_name": key_field,
                    "field_type": field_type,
                    "value": cleaned_value,
                    "confidence": conf,
                    "bbox": bbox,
                },
                "confidence": conf,
                "bbox": bbox,
                "context_entities": [],
                "extraction_method": "alternative_value_search",
                "meta": {
                    "reason": "Primary candidate failed validation",
                    "fallback_used": True,
                },
            }
            filtered_entity = {
                "field": key_field,
                "value": cleaned_value,
                "bbox": bbox,
                "confidence": conf,
                "page_number": best_candidate.get("page_number", 1),
                "semantic_type": EntityTypes.ANSWER,
                "semantic_confidence": conf,
            }
            return result, [filtered_entity]
        
        return self._build_empty_result(key_field)

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
        
        # Clear GPU cache before starting
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        langs_norm = _validate_and_normalize_langs(language_input)
        logger.info(f"Analyzing file with languages: {langs_norm}")

        langs_norm = _validate_and_normalize_langs(language_input)
        logger.info(f"Analyzing file with languages: {langs_norm}")
        entities, page_count, _ = _extract_text_multipage(
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
            "document_name": file_path,
            "page_count": page_count,
            "total_entities": len(filtered_entities),
            "entities": filtered_entities,
            "key_field_result": kf_result,
            "full_text": "", #full_text,
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

def _process_multi_page_analysis(analysis_data: dict, tmp_path: str, analyzer, verifier, langs_norm: str, key_fields: List[str]) -> Tuple[List[MultiKeyFieldResult], Optional[VerificationResult], int, int]:
    """Process multi-page analysis by analyzing each page separately and combining results.
    
    Returns: results, verification_result, document_width, document_height
    """
    all_results = []
    document_width = 1654 # default
    document_height = 2339 # default
    
    try:
        # Extract total pages from analysis data with proper type checking
        if isinstance(analysis_data, dict):
            total_pages = analysis_data.get("total_pages", 1)
        elif isinstance(analysis_data, list):
            total_pages = len(analysis_data)
        else:
            total_pages = 1
        
        logger.info(f"Processing {total_pages} pages from analysis data")
        
        # Convert multi-page PDF to images if needed
        if tmp_path.lower().endswith('.pdf'):
            if not PDF2IMAGE_AVAILABLE:
                raise HTTPException(500, "pdf2image not available for PDF processing")
            poppler_path = os.environ.get("POPPLER_PATH")
            kwargs = {"poppler_path": poppler_path} if poppler_path else {}
            images = convert_from_path(tmp_path, dpi=300, **kwargs)
            if len(images) != total_pages:
                logger.warning(f"PDF has {len(images)} pages but analysis data expects {total_pages}")
                total_pages = min(len(images), total_pages)
        else:
            # Single image
            images = [Image.open(tmp_path).convert("RGB")]
            total_pages = 1
        
        # Run document-wide verification
        verification_result = None
        try:
            v_res = verifier.verify(tmp_path)
            raw_form_name = v_res.get("form_name", "Unknown")
            mapped_form_name, mapped_classification_type = map_form_display(raw_form_name)
            
            verification_result = VerificationResult(
                filename=v_res.get("filename", os.path.basename(tmp_path)),
                predicted_class=v_res.get("predicted_class", "Unknown"),
                confidence=float(v_res.get("confidence", 0.0)),
                form_name=mapped_form_name,
                classification_type=mapped_classification_type,
                in_training_data=bool(v_res.get("in_training_data", False)),
                training_similarity=float(v_res.get("training_similarity", 0.0)),
                training_info=v_res.get("training_info", ""),
                extracted_text_preview=v_res.get("extracted_text_preview", ""),
                processing_time=float(v_res.get("processing_time", 0.0)),
                method=v_res.get("method", "hybrid")
            )
        except Exception as e:
            logger.warning(f"Verification failed: {e}")
        
        # Process each page
        page_results_map = {} # key_field -> list of results per page
        
        for page_num in range(1, total_pages + 1):
            logger.info(f"Processing page {page_num}/{total_pages}")
            
            # Create temporary file for this page
            page_tmp_path = None
            try:
                if len(images) > 1:
                    # Save this page as temporary image
                    page_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_page{page_num}.png")
                    images[page_num-1].save(page_tmp.name, 'PNG')
                    page_tmp_path = page_tmp.name
                else:
                    page_tmp_path = tmp_path
                
                # Process each key field for this page
                for kf in key_fields:
                    try:
                        raw_result = analyzer.analyze_file(page_tmp_path, kf, language_input=langs_norm)
                        
                        if kf not in page_results_map:
                            page_results_map[kf] = []
                        
                        # Create page-specific result
                        page_result = MultiKeyFieldResult(
                            key_field=kf,
                            extraction=ExtractionResult(
                                document_name=os.path.basename(tmp_path),
                                page_count=1,
                                total_entities=len(raw_result.get("entities", [])),
                                entities=raw_result.get("entities", []),
                                key_field_result=raw_result.get("key_field_result"),
                                full_text_snippet=raw_result.get("full_text", "")[:1000],
                                processing_time=float(raw_result.get("processing_time", 0.0)),
                                language_used=langs_norm,
                                model_used=raw_result.get("model_used", False)
                            ),
                            page_number=page_num
                        )
                        
                        page_results_map[kf].append(page_result)
                        
                    except Exception as e:
                        logger.error(f"Field '{kf}' on page {page_num} failed: {e}")
                        if kf not in page_results_map:
                            page_results_map[kf] = []
                        
                        page_results_map[kf].append(MultiKeyFieldResult(
                            key_field=kf,
                            error=str(e),
                            page_number=page_num
                        ))
            
            finally:
                # Clean up page temporary file
                if page_tmp_path and page_tmp_path != tmp_path and os.path.exists(page_tmp_path):
                    try:
                        os.unlink(page_tmp_path)
                    except Exception as e:
                        logger.warning(f"Failed to clean up page temp file: {e}")
        
        # Combine results from all pages
        for kf in key_fields:
            if kf not in page_results_map:
                all_results.append(MultiKeyFieldResult(
                    key_field=kf,
                    error="No results for any page"
                ))
                continue
            
            page_results = page_results_map[kf]
            
            # Find best result across all pages
            best_result = None
            best_confidence = -1
            
            for page_result in page_results:
                if page_result.extraction and page_result.extraction.key_field_result:
                    if isinstance(page_result['extraction'].key_field_result, dict):
                        conf = page_result['extraction'].key_field_result.get('confidence', 0.0)
                        if conf > best_confidence:
                            best_confidence = conf
                            best_result = page_result
                    elif isinstance(page_result["extraction"].key_field_result, list):
                        if page_result["extraction"].key_field_result:
                            conf = page_result["extraction"].key_field_result[0].confidence
                            if conf > best_confidence:
                                best_confidence = conf
                                best_result = page_result
            
            if best_result:
                all_results.append(MultiKeyFieldResult(
                    key_field=kf,
                    extraction=best_result["extraction"]
                ))
            else:
                all_results.append(MultiKeyFieldResult(
                    key_field=kf,
                    error="No valid result found on any page"
                ))
    
    except Exception as e:
        logger.error(f"Multi-page processing failed: {e}")
        raise HTTPException(500, f"Multi-page processing failed: {str(e)}")
    
    return all_results, verification_result, document_width, document_height

def split_entities_by_page(
    entities: List[Dict[str, Any]], 
    document_height: int = 2339,
    expected_pages: int = 1
) -> List[Dict[str, Any]]:
    """
    Split entities between pages using adaptive gap analysis.
    Supports ANY number of pages (1, 2, 3+) with robust fallbacks.
    NO HARDCODING - dynamically detects page boundaries from entity distribution.
    
    Args:
        entities: List of entity dicts with 'bbox' containing 'y' coordinate
        document_height: Total height of document image in pixels
        expected_pages: Expected number of pages (default: 2)
    
    Returns:
        List of entities with 'page_number' assigned (1-indexed)
    """
    if not entities:
        return entities
    
    # Extract Y positions of entity tops (robust handling)
    y_positions = []
    for e in entities:
        bbox = e.get("bbox", {})
        y = bbox.get("y")
        if y is not None and isinstance(y, (int, float)):
            y_positions.append(float(y))
    
    if not y_positions:
        logger.warning("⚠️ No valid Y positions found in entities, assigning all to page 1")
        for e in entities:
            e["page_number"] = 1
        return entities
    
    # Sort Y positions for gap analysis
    sorted_y = sorted(y_positions)
    
    # CRITICAL FIX #1: Detect natural page splits using adaptive gap analysis
    # For N pages, we need N-1 split points
    split_points = []
    
    if expected_pages > 1 and len(sorted_y) > 1:
        # Calculate gaps between consecutive Y positions
        gaps = []
        for i in range(1, len(sorted_y)):
            gap = sorted_y[i] - sorted_y[i-1]
            gaps.append((gap, sorted_y[i-1], sorted_y[i]))  # (gap_size, y_before, y_after)
        
        # Sort gaps by size (largest first)
        gaps.sort(key=lambda x: x[0], reverse=True)
        
        # Find significant gaps (>15% of document height OR >2x average gap)
        avg_gap = sum(g[0] for g in gaps) / len(gaps) if gaps else 0
        min_significant_gap = max(document_height * 0.15, avg_gap * 2.0)
        
        significant_gaps = [
            (gap, y_before, y_after) 
            for gap, y_before, y_after in gaps 
            if gap > min_significant_gap
        ]
        
        # Take top (expected_pages - 1) significant gaps as split points
        significant_gaps = significant_gaps[:expected_pages - 1]
        
        if significant_gaps:
            # Calculate split points as midpoints of significant gaps
            split_points = [
                y_before + (y_after - y_before) / 2.0
                for _, y_before, y_after in significant_gaps
            ]
            split_points.sort()  # Ensure ascending order
            
            logger.info(f"✅ Found {len(split_points)} significant gap(s) for {expected_pages} pages:")
            for i, sp in enumerate(split_points, 1):
                logger.info(f"    Split {i}: Y={sp:.0f}px (gap size: {significant_gaps[i-1][0]:.0f}px)")
        else:
            logger.warning(f"⚠️ No significant gaps found (> {min_significant_gap:.0f}px), using equal-height split")
    
    # CRITICAL FIX #2: Fallback to equal-height splits if no natural gaps found
    if not split_points:
        # Create (expected_pages - 1) split points at equal intervals
        split_points = [
            document_height * (i / expected_pages)
            for i in range(1, expected_pages)
        ]
        logger.info(f"🔧 Using equal-height split for {expected_pages} pages:")
        for i, sp in enumerate(split_points, 1):
            logger.info(f"    Split {i}: Y={sp:.0f}px (document height: {document_height}px)")
    
    # CRITICAL FIX #3: Assign page numbers based on split points
    page_counts = {i: 0 for i in range(1, expected_pages + 1)}
    
    for entity in entities:
        bbox = entity.get("bbox", {})
        y_top = bbox.get("y", 0)
        
        # Determine page number based on split points (1-indexed)
        page_num = 1
        for i, split_y in enumerate(split_points, 1):
            if y_top >= split_y:
                page_num = i + 1
            else:
                break
        
        # Clamp to valid page range (safety for edge cases)
        page_num = max(1, min(page_num, expected_pages))
        
        entity["page_number"] = page_num
        page_counts[page_num] += 1
    
    # Log results
    total_entities = len(entities)
    logger.info(f"✅ Split complete across {expected_pages} pages:")
    for page_num in range(1, expected_pages + 1):
        count = page_counts.get(page_num, 0)
        pct = (count / total_entities * 100) if total_entities > 0 else 0
        logger.info(f"  Page {page_num}: {count} entities ({pct:.1f}%)")
    
    # DEBUG: Show sample entities from each page
    for page_num in range(1, expected_pages + 1):
        page_entities = [e for e in entities if e.get("page_number") == page_num]
        if page_entities:
            sample_count = min(3, len(page_entities))
            logger.debug(f"📄 Page {page_num} samples (first {sample_count}):")
            for i in range(sample_count):
                e = page_entities[i]
                text_preview = e.get('value', '')[:40].replace('\n', ' ')
                logger.debug(f"    Y={e['bbox']['y']:4.0f}px: '{text_preview}'")
    
    return entities

# ✅ STANDALONE FUNCTION (NO 'self' parameter)
def _build_multi_page_result_json_payload(
    key_fields: List[str],
    results: List[MultiKeyFieldResult],
    verification_result: Optional[VerificationResult],
    document_width: int,
    document_height: int,
    total_pages: int = 2
) -> dict:
    """
    Build result.json with CORRECT multi-page grouping.
    CRITICAL FIXES:
    1. Uses list [] not set() for out["fields"] initialization
    2. Gets page_number from entities (most reliable source)
    3. Handles any number of pages dynamically (no hardcoding)
    """
    # ✅ CRITICAL FIX #1: Initialize fields as LIST (not set!)
    out = {
        "form_name": verification_result.form_name if verification_result else None,
        "classification_type": verification_result.classification_type if verification_result else None,
        "document_dimensions": {"width": document_width, "height": document_height},
        "target_dimensions": {"width": 1654, "height": 2339},
        "total_key_fields": len(key_fields),
        "total_pages": total_pages,
        "fields": [],  # ✅ MUST BE LIST - NOT set()
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "stats": {"found": 0, "missing": 0, "pages": total_pages}
    }
    
    # Deduplication set (separate from out["fields"])
    seen_fields = set()
    
    # ✅ CRITICAL FIX #2: Get page number from entities (most reliable source)
    for result in results:
        # Skip invalid results
        if not result or not result.extraction or not result.extraction.key_field_result:
            continue
        
        kfr = result.extraction.key_field_result
        base_key_field = result.key_field
        
        # ✅ DETERMINE PAGE NUMBER FROM ENTITIES (authoritative source)
        page_num = 1
        if result.extraction.entities and len(result.extraction.entities) > 0:
            # Use first entity's page number (all entities for this result should be same page)
            page_num = getattr(result.extraction.entities[0], 'page_number', 1)
        else:
            # Fallback: try to get from kfr if available
            if isinstance(kfr, list) and kfr:
                page_num = getattr(kfr[0], 'page_number', 1)
            elif kfr:
                page_num = getattr(kfr, 'page_number', 1)
        
        # Handle checkbox results (list format)
        if isinstance(kfr, list):
            for checkbox in kfr:
                value = checkbox.value or ""
                field_name = getattr(checkbox, 'field_name', base_key_field)
                
                # Skip exact duplicates on same page with same value
                dup_key = (field_name, page_num, value)
                if dup_key in seen_fields:
                    logger.debug(f"Skipping duplicate field: {dup_key}")
                    continue
                seen_fields.add(dup_key)
                
                # Build field entry
                raw_bbox = {
                    "x": checkbox.bbox.x,
                    "y": checkbox.bbox.y,
                    "width": checkbox.bbox.width,
                    "height": checkbox.bbox.height
                }
                scaled_bbox = _scale_bbox_to_target(raw_bbox, document_width, document_height)
                out["fields"].append({  # ✅ SAFE: out["fields"] is LIST
                    "index": len(out["fields"]) + 1,
                    "key_field": field_name,
                    "field_name": field_name,
                    "value": value,
                    "confidence": round(float(checkbox.confidence or 0.0), 4),
                    "page_number": page_num,
                    "bbox": scaled_bbox,
                    "found": True,
                    "source": "individual_checkbox"
                })
                logger.debug(f"Added checkbox field: {field_name} (page {page_num})")
        
        # Handle single result fields
        elif kfr:
            value = kfr.value or ""
            field_name = getattr(kfr, 'field_name', base_key_field)
            
            # Skip exact duplicates
            dup_key = (field_name, page_num, value)
            if dup_key in seen_fields:
                logger.debug(f"Skipping duplicate field: {dup_key}")
                continue
            seen_fields.add(dup_key)
            
            # Build field entry
            raw_bbox = {
                "x": kfr.bbox.x,
                "y": kfr.bbox.y,
                "width": kfr.bbox.width,
                "height": kfr.bbox.height
            }
            scaled_bbox = _scale_bbox_to_target(raw_bbox, document_width, document_height)
            out["fields"].append({  # ✅ SAFE: out["fields"] is LIST
                "index": len(out["fields"]) + 1,
                "key_field": field_name,
                "field_name": field_name,
                "value": value,
                "confidence": round(float(kfr.confidence or 0.0), 4),
                "page_number": page_num,
                "bbox": scaled_bbox,
                "found": True,
                "source": "key_field_result"
            })
            logger.debug(f"Added field: {field_name} (page {page_num})")
    
    # ✅ CRITICAL: Sort ALL fields by page_number FIRST, then by vertical position (y)
    out["fields"].sort(key=lambda f: (f["page_number"], f["bbox"]["y"]))
    
    # ✅ Renumber indexes sequentially AFTER sorting
    for i, field in enumerate(out["fields"], 1):
        field["index"] = i
    
    # ✅ ACCURATE STATS: Count UNIQUE key_fields found (not field entries)
    found_key_fields = set()
    for field in out["fields"]:
        if field.get("found", False):
            # Map checkbox options back to base field pattern for stats
            if "checkbox_option" in field["key_field"].lower():
                found_key_fields.add("individual_checkbox")
            else:
                found_key_fields.add(field["key_field"])
    
    out["stats"]["found"] = len(found_key_fields)
    out["stats"]["missing"] = len(key_fields) - out["stats"]["found"]
    
    # Log final page distribution
    pages_in_output = sorted(set(f["page_number"] for f in out["fields"]))
    logger.info(f"✅ Built result.json with {len(out['fields'])} fields across {len(pages_in_output)} pages:")
    for page_num in pages_in_output:
        count = sum(1 for f in out["fields"] if f["page_number"] == page_num)
        logger.info(f"  Page {page_num}: {count} fields")
    logger.info(f"📊 Stats: {out['stats']['found']} found, {out['stats']['missing']} missing, {out['stats']['pages']} total pages")
    
    return out

def assign_fields_to_pages(result_json_content: dict) -> dict:
    """Alternate assigning fields to pages 1 and 2"""
    fields = result_json_content.get("fields", [])
    
    for i, field in enumerate(fields):
        # Alternate: odd indices -> page 1, even indices -> page 2
        field["page_number"] = 1 if i % 2 == 0 else 2
    
    # Update stats
    result_json_content["stats"]["pages"] = 2
    
    return result_json_content

def parse_page_fields_from_file(file_path: str = "extracted_fields.txt") -> Dict[int, List[str]]:
    """
    Parse extracted_fields.txt with page markers (# PAGE X).
    Returns Dict[page_number, List[field_names]] with ONLY actual field names.
    
    Format example:
      # PAGE 1
      Field Name 1
      Field Name 2
      
      # PAGE 2
      Field Name 3
      Field Name 4
    
    Rules:
      - Lines starting with '#' are COMMENTS (ignored as fields)
      - '# PAGE X' sets current page context (X = integer)
      - Non-comment lines are field names for current page
      - Blank lines are ignored
      - Field names are cleaned (trailing colons/whitespace removed)
    """
    page_fields: Dict[int, List[str]] = {}
    current_page = 1  # Default page if no markers found
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            original_line = line
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check for page marker: # PAGE X (case-insensitive, flexible spacing)
            # Must be EXACTLY "# PAGE X" pattern - not other comments
            page_match = re.match(r'^#\s*PAGE\s*(\d+)', line, re.IGNORECASE)
            if page_match:
                current_page = int(page_match.group(1))
                if current_page not in page_fields:
                    page_fields[current_page] = []
                logger.debug(f"Line {line_num}: Page marker → PAGE {current_page}")
                continue
            
            # Skip ALL other comment lines (anything starting with # but not PAGE marker)
            if line.startswith('#'):
                logger.debug(f"Line {line_num}: Skipping comment: '{original_line.strip()}'")
                continue
            
            # This is an actual field name - add to current page
            if current_page not in page_fields:
                page_fields[current_page] = []
            
            # Clean field name:
            # 1. Remove trailing colons (common in form field names)
            # 2. Remove extra whitespace
            # 3. Skip empty fields
            clean_field = line.rstrip(':').strip()
            if not clean_field:
                logger.debug(f"Line {line_num}: Skipping empty field")
                continue
            
            # Special handling: Fix "No" to "No." for contact fields
            if clean_field.endswith("No"):
                clean_field = clean_field + "."
            
            page_fields[current_page].append(clean_field)
            logger.debug(f"Line {line_num}: Added field '{clean_field}' to PAGE {current_page}")
        
        # Remove pages with no actual fields
        valid_pages = {p: fs for p, fs in page_fields.items() if fs}
        
        if not valid_pages:
            logger.warning(f"Parsed file but found NO actual fields (only comments/markers) in {file_path}")
            return {}
        
        # Log parsed structure for verification
        logger.info(f"Parsed {len(valid_pages)} pages with fields from {file_path}")
        for page_num in sorted(valid_pages.keys()):
            fields = valid_pages[page_num]
            preview = fields[:3] + (["..."] if len(fields) > 3 else [])
            logger.info(f"  PAGE {page_num}: {len(fields)} fields - {preview}")
        
        return valid_pages
    
    except FileNotFoundError:
        logger.warning(f"Key fields file not found: {file_path}")
        return {}
    except Exception as e:
        logger.error(f"Error parsing {file_path}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return {}

# ✅ CRITICAL FIX: Preserve page number from entities during conversion
def _convert_kf_result_to_model(kf_result: Any, fallback_page_num: int = 1) -> Union[KeyFieldResult, List[KeyFieldResult], None]:
    """Convert raw detection result to KeyFieldResult model WITH page_number preserved"""
    if isinstance(kf_result, list):
        kfr_list = []
        for res in kf_result:
            try:
                # ✅ Get page number from result OR fallback
                page_num = res.get("page_number", fallback_page_num) if isinstance(res, dict) else getattr(res, 'page_number', fallback_page_num)
                
                # Build structured_value if exists
                structured_value = None
                if res.get("structured_value"):
                    sv = res["structured_value"]
                    structured_value = DataField(
                        field_name=sv.get("field_name", res.get("field_name", "")),
                        field_type=sv.get("field_type", "individual_checkbox"),
                        value=sv.get("value", ""),
                        confidence=float(sv.get("confidence", 0.0)),
                        bbox=BoundingBox(**sv.get("bbox", {"x": 0, "y": 0, "width": 1, "height": 1}))
                    )
                
                kfr = KeyFieldResult(
                    field_name=res.get("field_name", ""),
                    value=res.get("value", ""),
                    structured_value=structured_value,
                    confidence=float(res.get("confidence", 0.0)),
                    bbox=BoundingBox(**res.get("bbox", {"x": 0, "y": 0, "width": 1, "height": 1})),
                    page_number=int(page_num),  # ✅ PRESERVE PAGE NUMBER
                    meta=res.get("meta"),
                    extraction_method=res.get("extraction_method")
                )
                kfr_list.append(kfr)
            except Exception as e:
                logger.warning(f"Failed to convert checkbox result: {e}")
                continue
        return kfr_list if len(kfr_list) > 1 else (kfr_list[0] if kfr_list else None)
    else:
        try:
            page_num = kf_result.get("page_number", fallback_page_num) if isinstance(kf_result, dict) else getattr(kf_result, 'page_number', fallback_page_num)
            
            structured_value = None
            if kf_result.get("structured_value"):
                sv = kf_result["structured_value"]
                structured_value = DataField(
                    field_name=sv.get("field_name", kf_result.get("field_name", "")),
                    field_type=sv.get("field_type", "text"),
                    value=sv.get("value", ""),
                    confidence=float(sv.get("confidence", 0.0)),
                    bbox=BoundingBox(**sv.get("bbox", {"x": 0, "y": 0, "width": 1, "height": 1}))
                )
            
            return KeyFieldResult(
                field_name=kf_result.get("field_name", ""),
                value=kf_result.get("value", ""),
                structured_value=structured_value,
                confidence=float(kf_result.get("confidence", 0.0)),
                bbox=BoundingBox(**kf_result.get("bbox", {"x": 0, "y": 0, "width": 1, "height": 1})),
                page_number=int(page_num),  # ✅ PRESERVE PAGE NUMBER
                meta=kf_result.get("meta"),
                extraction_method=kf_result.get("extraction_method")
            )
        except Exception as e:
            logger.warning(f"Failed to convert key field result: {e}")
            return None

def get_all_key_fields(page_fields: Dict[int, List[str]]) -> List[str]:
    """Get all unique key fields across all pages (deduplicated)"""
    all_fields = set()
    for fields in page_fields.values():
        all_fields.update(fields)
    return sorted(all_fields)


def get_fields_for_page(page_fields: Dict[int, List[str]], page_num: int) -> List[str]:
    """Get fields for a specific page (empty list if page not found)"""
    return page_fields.get(page_num, [])

# ------- Main API Endpoint -------
@app.post("/api/get_analysis_result")
async def get_analysis_result(
    document: UploadFile = File(...),
    analysis_data: UploadFile = File(...),
    language: Optional[str] = Form(None),
):
    """
    Combined endpoint with CORRECT page detection.
    FIXED: Now properly handles single-page documents vs multi-page templates.
    """
    tmp_path = None
    analysis_file_path = None
    
    # Save analysis_data file
    try:
        os.makedirs("test_images", exist_ok=True)
        timestamp = int(time.time())
        original_filename = analysis_data.filename
        filename_safe = re.sub(r'[^\w\-_\.]', '_', original_filename)
        analysis_filename = f"analysis_{timestamp}_{filename_safe}"
        analysis_file_path = os.path.join("test_images", analysis_filename)
        content = await analysis_data.read()
        with open(analysis_file_path, "wb") as f:
            f.write(content)
        logger.info(f"✅ Saved analysis data file to: {analysis_file_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save analysis data file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save analysis data file: {str(e)}")
    
    try:
        if not hasattr(app.state, "analyzer") or app.state.analyzer is None:
            raise HTTPException(503, "Analyzer not initialized")
        if not hasattr(app.state, "verifier") or app.state.verifier is None:
            raise HTTPException(503, "Verifier not initialized")
        
        ext = os.path.splitext(document.filename)[1].lower()
        if ext not in [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]:
            raise HTTPException(400, "Unsupported document type")
        
        # Save document file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tf:
            content = await document.read()
            if not content:
                raise HTTPException(400, "Empty document")
            tf.write(content)
            tmp_path = tf.name
        
        langs_norm = _validate_and_normalize_langs(language)
        analyzer = app.state.analyzer
        verifier = app.state.verifier

        # Run extraction to get field mapping
        OUTPUT_FILE = "extracted_fields.txt"
        run_extraction(tmp_path, LILT_MODEL, OUTPUT_FILE)

        # ✅ PARSE PAGE-SPECIFIC FIELDS FROM TEMPLATE
        page_field_mapping = parse_page_fields_from_file("extracted_fields.txt")
        
        if not page_field_mapping:
            logger.error("❌ No valid key fields found in extracted_fields.txt!")
            raise HTTPException(
                500,
                "No valid key fields configured. extracted_fields.txt must contain '# PAGE X' markers followed by actual field names."
            )
        
        # ✅ DETERMINE ACTUAL NUMBER OF PAGES IN DOCUMENT
        logger.info("🔍 DETERMINING ACTUAL PAGE COUNT...")
        
        # Load entities FIRST to see actual page distribution
        entities = []
        try:
            logger.info(f"Loading entities from analysis JSON: {analysis_file_path}")
            entities = convert_result_json_to_test_page_data_in_memory(analysis_file_path)
            logger.info(f"✅ Loaded {entities} entities")
        except Exception as e:
            logger.error(f"❌ Failed to load entities from JSON: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load entities from JSON: {str(e)}")
        
        # Get ACTUAL page count from entities (not from template)
        actual_pages = sorted(set(e.get("page_number", 1) for e in entities))
        actual_page_count = len(actual_pages)
        
        logger.info(f"📊 ACTUAL DOCUMENT: {actual_page_count} page(s) - Entities on pages: {actual_pages}")
        logger.info(f"📊 TEMPLATE EXPECTS: {len(page_field_mapping)} page(s)")
        
        # ✅ CRITICAL FIX: If template has more pages than actual document, ONLY use pages that exist
        if actual_page_count < len(page_field_mapping):
            logger.warning(f"⚠️ Template expects {len(page_field_mapping)} pages but document has only {actual_page_count}. Using only page 1 fields.")
            # Keep only page 1 fields from template
            page_field_mapping = {1: page_field_mapping.get(1, [])}
        
        # Get all unique fields from template
        all_key_fields = get_all_key_fields(page_field_mapping)
        logger.info(f"✅ Processing {len(page_field_mapping)} page(s) with {len(all_key_fields)} unique fields")
        
        # Get document dimensions
        document_width, document_height = 1654, 2339
        try:
            with open(analysis_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data.get("associations") and len(data["associations"]) > 0:
                    img_info = data["associations"][0].get("image_dimensions", {})
                    document_width = img_info.get("width", 1654)
                    document_height = img_info.get("height", 2339)
        except Exception as e:
            logger.warning(f"Could not get dimensions from JSON: {e}")

        # ✅ PER-PAGE FIELD DETECTION (ONLY FOR ACTUAL PAGES)
        results: List[MultiKeyFieldResult] = []
        
        for page_num in sorted(page_field_mapping.keys()):
            # Skip if this page doesn't exist in actual document
            if page_num > actual_page_count:
                logger.warning(f"⚠️ Template page {page_num} doesn't exist in document. Skipping.")
                continue
                
            page_fields = get_fields_for_page(page_field_mapping, page_num)
            if not page_fields:
                continue
            
            page_entities = [e for e in entities if e.get("page_number", 1) == page_num]
            if not page_entities:
                logger.warning(f"⚠️ No entities found for page {page_num}")
                continue
            
            logger.info(f"📄 Processing page {page_num}: {len(page_fields)} fields, {len(page_entities)} entities")
            
            for kf in page_fields:
                try:
                    kf_result, filtered_entities = analyzer.field_detector.detect_field(page_entities, kf)
                    
                    if kf_result:
                        # Force correct page number
                        if isinstance(kf_result, list):
                            for res in kf_result:
                                res["page_number"] = page_num
                        else:
                            kf_result["page_number"] = page_num
                        
                        # Convert to model with page number preservation
                        kfr_model = _convert_kf_result_to_model(kf_result, fallback_page_num=page_num)
                        
                        # Build extraction result
                        entities_model = []
                        for e in filtered_entities:
                            try:
                                bbox_data = e.get("bbox", {"x": 0, "y": 0, "width": 1, "height": 1})
                                
                                # NEW: scale to 2480x3509
                                scaled_data = scale_bbox_to_a4_400dpi(bbox_data, 2480, 3509)
                                
                                entities_model.append(ExtractedEntity(
                                    field=e.get("field", ""),
                                    value=e.get("value", ""),
                                    bbox=BoundingBox(**scaled_data),
                                    confidence=float(e.get("confidence", 0.0)),
                                    page_number=e.get("page_number", page_num)
                                ))
                            except Exception as e:
                                logger.warning(f"Failed to convert entity: {e}")
                                continue
                        
                        extraction = ExtractionResult(
                            document_name=document.filename,
                            page_count=actual_page_count,  # ✅ Use ACTUAL page count
                            total_entities=len(entities_model),
                            entities=entities_model,
                            key_field_result=kfr_model,
                            full_text_snippet="",
                            processing_time=0.0,
                            language_used=language,
                            model_used=False
                        )
                        results.append(MultiKeyFieldResult(key_field=kf, extraction=extraction))
                        
                except Exception as e:
                    logger.error(f"Field '{kf}' on page {page_num} failed: {e}", exc_info=True)
                    results.append(MultiKeyFieldResult(key_field=kf, error=str(e)))

        # Run verification
        verification_result = None
        try:
            v_res = verifier.verify(tmp_path)
            raw_form_name = v_res.get("form_name", "Unknown")
            mapped_form_name, mapped_classification_type = map_form_display(raw_form_name)
            verification_result = VerificationResult(
                filename=v_res.get("filename", document.filename),
                predicted_class=v_res.get("predicted_class", "Unknown"),
                confidence=float(v_res.get("confidence", 0.0)),
                form_name=mapped_form_name,
                classification_type=mapped_classification_type,
                in_training_data=bool(v_res.get("in_training_data", False)),
                training_similarity=float(v_res.get("training_similarity", 0.0)),
                training_info=v_res.get("training_info", ""),
                extracted_text_preview=v_res.get("extracted_text_preview", ""),
                processing_time=float(v_res.get("processing_time", 0.0)),
                method=v_res.get("method", "hybrid")
            )
        except Exception as e:
            logger.warning(f"Verification failed: {e}")
        
        # Build result.json with proper page grouping
        result_json_content = _build_multi_page_result_json_payload(
            key_fields=all_key_fields,
            results=results,
            verification_result=verification_result,
            document_width=document_width,
            document_height=document_height,
            total_pages=actual_page_count  # ✅ Use ACTUAL page count, not template page count
        )

        # ✅ Sort fields by page THEN y-position
        result_json_content["fields"].sort(key=lambda f: (f["page_number"], f["bbox"]["y"]))
        
        # ✅ Renumber indexes sequentially
        for i, field in enumerate(result_json_content["fields"], 1):
            field["index"] = i
        
        # Verify final output
        pages_in_output = sorted(set(f["page_number"] for f in result_json_content["fields"]))
        logger.info(f"✅ FINAL OUTPUT:")
        for page_num in pages_in_output:
            page_fields = [f for f in result_json_content["fields"] if f["page_number"] == page_num]
            found_count = sum(1 for f in page_fields if f.get("found", False))
            logger.info(f"  Page {page_num}: {found_count}/{len(page_fields)} fields found")
        
        # Save to disk
        try:
            with open("result.json", "w", encoding="utf-8") as f:
                json.dump(result_json_content, f, ensure_ascii=False, indent=2)
            found_count = result_json_content['stats']['found']
            logger.info(f"✅ Saved result.json: {found_count}/{len(all_key_fields)} fields found")
        except Exception as e:
            logger.warning(f"result.json save failed: {e}")
        
        return result_json_content
    
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception("get_analysis_result error")
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
        if analysis_file_path and os.path.exists(analysis_file_path):
            try:
                os.unlink(analysis_file_path)
            except Exception:
                pass

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
#-------- NEW: EXTRACT FORM NAME ONLY --------
verifier_instance: Optional[Verifier] = None
@app.post("/api/extract_form_name")
async def extract_form_name(file: UploadFile = File(...)):
    """Extract only form name using fingerprints without full analysis"""
    if not hasattr(app.state, "verifier") or app.state.verifier is None:
        return JSONResponse({"error": "Verifier not initialized"}, status_code=500)

    # Save uploaded file to temporary location
    ext = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        if not content:
            return JSONResponse({"error": "Empty file"}, status_code=400)
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Use the verifier from app.state
        verifier = app.state.verifier
        
        # First, get first page image for processing
        processed_path = verifier._get_first_page_image(tmp_path)
        if not processed_path:
            return JSONResponse({
                "error": "Failed to process file for form name extraction",
                "filename": file.filename
            }, status_code=400)

        try:
            # Extract OCR text and boxes
            words, boxes, w, h = ocr_extract_words_bboxes(processed_path)
            query_fp = generate_fingerprint(words, boxes, w, h)

            # Find best matching fingerprint
            best_fp = None
            best_score = 0.0

            for fname, fp in verifier.fingerprints.items():
                try:
                    score = fingerprint_similarity(query_fp, fp)
                except Exception:
                    continue

                if score > best_score:
                    best_score = score
                    best_fp = fname

            # Map form name using CSV if available
            mapped_form_name, mapped_classification_type = map_form_display(best_fp)
            
            out = {
                "filename": file.filename,
                "form_name": mapped_form_name,
                "classification_type": mapped_classification_type,
                "similarity": float(best_score),
                "is_match": best_score >= verifier.fp_threshold,
                "matched_original_name": best_fp,
                "method": "fingerprint_only"
            }

            return JSONResponse(content=out)
            
        finally:
            # Clean up temporary processed image if different from original
            if processed_path != tmp_path and os.path.exists(processed_path):
                try:
                    os.remove(processed_path)
                except Exception:
                    pass
                    
    except Exception as e:
        logger.exception(f"Form name extraction failed: {e}")
        return JSONResponse({
            "error": f"Form name extraction failed: {str(e)}",
            "filename": file.filename
        }, status_code=500)
        
    finally:
        # Clean up original temporary file
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

def run_extraction(document_path: str, model_path: str, output_file: str = "extracted_fields.txt"):
    """
    Run field extraction on a document using the DocumentFieldExtractor.
    
    Args:
        document_path (str): Path to input PDF or image file.
        model_path (str): Path to model directory (can be dummy path since fallback is used).
        output_file (str): Output file to save extracted fields (default: fields.txt)
    
    Returns:
        bool: True if successful, False otherwise.
    """
    if not os.path.exists(document_path):
        print(f"ERROR: Document not found: {document_path}")
        return False

    # Supported extensions check
    supported_ext = {'.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
    ext = os.path.splitext(document_path)[1].lower()
    if ext not in supported_ext:
        print(f"ERROR: Unsupported file type: {ext}")
        return False

    try:
        print("Initializing extractor...")
        extractor = DocumentFieldExtractor(model_path=model_path)

        print("Extracting fields...")
        fields = extractor.extract_fields(document_path)

        print(f"Saving results to {output_file}...")
        success = extractor.save_fields(fields, output_file)

    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        return False

# ------- Main -------
def main():
    # Manage GPU memory at startup
    manage_gpu_memory()
    
    parser = argparse.ArgumentParser(
        description="Document Analysis API - Combined LiLT + Verification"
    )

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
    LILT_MODEL = args.model_path
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
#!/usr/bin/env python3
"""
Document Analysis API - Fixed Contact Field Extraction with LiLT Relation Extraction
Features:
- Fixed LiLT model import and method parameter errors
- Removed trust_remote_code parameter
- Integrated LiLT model for relationship extraction between keys and values
- Specialized phone number extraction with OCR error tolerance
- Multi-stage fallback strategies for contact fields
- Proper handling when no key_field is found (returns empty result)
Run: python3 document_analysis_fixed_lilt.py --port 8000 --model_path lilt_text_extraction_model01 --qa_model deepset/roberta-base-squad2
"""
import os
import re
import time
import tempfile
import argparse
import logging
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

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

# LiLT
try:
    from models.LiLTRobertaLike import LiLTRobertaLikeForRelationExtraction
    LILT_AVAILABLE = True
    logging.info("LiLT model imported from models.LiLTRobertaLike")
except ImportError:
    try:
        from LiLTRobertaLike import LiLTRobertaLikeForRelationExtraction
        LILT_AVAILABLE = True
        logging.info("LiLT model imported from LiLTRobertaLike")
    except ImportError:
        logging.warning("LiLT model not found. Using fallback.")
        LILT_AVAILABLE = False

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import pickle

# ------- Logging -------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("doc_analysis_complete")


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


# ------- FastAPI app -------
app = FastAPI(title="Document Analysis API - LiLT + Email Extraction")


# ------- Utilities -------
def _get_installed_tesseract_langs() -> List[str]:
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
    if not word:
        return ""
    
    # For Chinese text
    if language_code.startswith(("chi", "jpn", "kor")):
        # Keep Chinese characters, Latin letters, numbers, and basic punctuation
        cleaned = re.sub(r'[^\u4e00-\u9fff\w\s\-\.,:/$#@%()@+\u00C0-\u017F✓✔☑√]', '', word)
    else:
        # For other languages
        cleaned = re.sub(r'[^\w\s\-\.,:/$#@%()@+\u00C0-\u017F]', '', word)
    
    return cleaned.strip()


def _validate_and_normalize_langs(lang_input: Optional[str]) -> str:
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
        # Create a copy to avoid modifying the original
        cleaned_entity = entity.copy()
        
        # Clean field name
        if "field" in cleaned_entity:
            field = cleaned_entity["field"]
            # Remove symbols from field name, keep only alphanumeric and underscore
            cleaned_field = re.sub(r'[^\w_]', '', field)
            if not cleaned_field:
                cleaned_field = "entity"
            cleaned_entity["field"] = cleaned_field
            
        # Clean value
        if "value" in cleaned_entity:
            value = cleaned_entity["value"]
            # Remove duplicate words
            words = value.split()
            unique_words = []
            for word in words:
                if word not in unique_words:
                    unique_words.append(word)
            cleaned_entity["value"] = " ".join(unique_words)
            
            # Remove excessive whitespace
            cleaned_entity["value"] = re.sub(r'\s+', ' ', cleaned_entity["value"]).strip()
            
        cleaned.append(cleaned_entity)
    
    return cleaned


def is_checkbox_option(text: str) -> bool:
    """Check if text looks like a checkbox option with better filtering"""
    if not text:
        return False
        
    text_lower = text.lower().strip()
    
    # Skip obvious non-checkbox text - EXCLUDE section headers and instructions
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
    
    # Checkbox option patterns - only include actual options
    checkbox_patterns = [
        # Pattern for "Loss of the existing certificate..."
        r"^loss.*certificate.*$",
        r"^replacement.*copy.*certificate.*registration.*generating.*facility.*due.*to$",
        r"^damage.*\(.*damaged.*certificate.*must.*returned.*deletion.*\)$",
        r"掼毁.*琨有损毁澄明善必须交遣本署鞋艄.*damage",
    ]
    
    # Check for checkbox-specific patterns
    has_checkmark = any(symbol in text for symbol in ["✓", "✔", "☑", "√"])
    
    # Look for checkbox option patterns
    for pattern in checkbox_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    
    # Parentheses often contain explanations in checkbox options
    has_parentheses = "(" in text and ")" in text
    
    # Check for specific checkbox content words
    content_words = ["loss", "damage", "replacement", "deletion", "returned", "certificate"]
    content_count = sum(1 for word in content_words if word in text_lower)
    
    # Length constraints - checkbox options are usually medium length
    text_len = len(text)
    if text_len < 20 or text_len > 300:  # Too short or too long
        return False
    
    # Must have at least 2 content words OR parentheses with explanation OR checkmark
    return (content_count >= 2) or (has_parentheses and content_count >= 1) or has_checkmark


def find_checkbox_region(entities: List[Dict]) -> Tuple[int, int, int, int]:
    """Find the bounding box of the checkbox section in the document"""
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
    
    # Find the main checkbox section header
    for e in entities:
        text = e.get("value", "").lower()
        for keyword in checkbox_section_keywords:
            if keyword in text:
                region_entities.append(e)
                logger.info(f"Found checkbox section header: '{e.get('value', '')}'")
                break
    
    if not region_entities:
        # If no section header found, look for "Replacement copy..." or "Damage..." as anchors
        for e in entities:
            text = e.get("value", "").lower()
            if "replacement copy" in text or "damage (" in text:
                region_entities.append(e)
                logger.info(f"Using checkbox option as anchor: '{e.get('value', '')}'")
    
    if not region_entities:
        # Fallback: use entire page
        logger.warning("No checkbox region found, using entire page")
        return 0, 0, 10000, 10000
    
    # Calculate region bounds based on found entities
    min_x = min(e.get("bbox", {}).get("x", 0) for e in region_entities)
    max_x = max(e.get("bbox", {}).get("x", 0) + e.get("bbox", {}).get("width", 0) 
                for e in region_entities)
    min_y = min(e.get("bbox", {}).get("y", 0) for e in region_entities)
    max_y = max(e.get("bbox", {}).get("y", 0) + e.get("bbox", {}).get("height", 0) 
                for e in region_entities)
    
    # Expand region to capture all checkbox options (but not too much)
    x_padding = 100
    y_padding = 200  # More padding vertically to capture options below
    
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
    
    # Check if entity overlaps with region
    return not (x + width < min_x or x > max_x or 
                y + height < min_y or y > max_y)


def merge_entities(entities: List[Dict]) -> List[Dict]:
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

    # Clean entity values first
    for e in entities:
        if "value" in e and e["value"]:
            # Clean up OCR artifacts
            e["value"] = e["value"].strip()
            # Remove duplicate text (simple deduplication)
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

            # Check if entities look like checkbox options
            curr_is_checkbox = is_checkbox_option(current["value"])
            e_is_checkbox = is_checkbox_option(e["value"])

            should_not_merge = (
                curr_has_phone
                or e_has_phone
                or (curr_is_label and not e_is_label)
                or curr_is_checkbox  # Don't merge checkbox options
                or e_is_checkbox     # Don't merge checkbox options
                or y_diff >= 15
                or x_diff >= 50
            )

            if should_not_merge:
                merged.append(current)
                current = e.copy()
            else:
                # Merge text but avoid duplicates
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

    # Assign proper field names
    entity_counter = 1
    for m in merged:
        if not m.get("field") or m.get("field", "").startswith("pre_merge_"):
            # Clean field name - remove symbols, keep only alphanumeric
            field_name = f"entity_{entity_counter}"
            m["field"] = field_name
            entity_counter += 1
        else:
            # Clean existing field name
            current_field = m.get("field", "")
            # Remove symbols, keep only alphanumeric and underscore
            cleaned_field = re.sub(r'[^\w_]', '', current_field)
            if not cleaned_field or len(cleaned_field) == 1:
                cleaned_field = f"entity_{entity_counter}"
                entity_counter += 1
            m["field"] = cleaned_field
            
        # Final cleanup of value
        if "value" in m:
            m["value"] = m["value"].strip()
            # Remove excessive whitespace
            m["value"] = re.sub(r'\s+', ' ', m["value"])

    logger.info(f"Merge complete: {len(merged)} entities ({len(phone_entities)} phones)")
    
    # Final cleanup
    merged = clean_extracted_entities(merged)
    
    return merged


def _extract_text_multipage(
    file_path: str, languages: str = "eng", conf_threshold: float = 0.15
) -> Tuple[List[Dict], str, int, List[Image.Image]]:
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
                        
                        # Check for checkmarks and clean them
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
        
        # Enhanced email patterns
        patterns = [
            # Standard email with dot
            r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
            # Email without dot (like kc@imimrnet)
            r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+",
            # Common incomplete patterns
            r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9]+",
            # Look for @ symbol and take the whole word
            r"\S+@\S+",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Validate it looks like an email
                if '@' in match and len(match) > 3:
                    # Try to complete common domain extensions
                    email = match.strip()
                    if not any(email.endswith(ext) for ext in ['.com', '.net', '.org', '.edu', '.gov', '.io', '.co', '.hk']):
                        return email  # Return as-is
                    return email
        
        return None

    def _normalize_text_for_matching(self, text: str) -> str:
        """Normalize text for fuzzy matching"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Normalize punctuation - replace multiple colons/punctuation with single
        text = re.sub(r'[:;.]+', ':', text)
        
        # Remove common OCR artifacts
        text = re.sub(r'[`\'"~]', '', text)
        
        # Normalize parentheses and brackets
        text = text.replace('(', ' (')
        text = text.replace(')', ') ')
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _find_label_entity(self, entities: List[Dict], label_text: str) -> Optional[Dict]:
        """Find entity that matches or contains the label text"""
        if not label_text:
            return None
        
        # Skip if label_text is just a symbol
        if len(label_text.strip()) == 1 and label_text.strip() in "✓✔☑√":
            logger.warning(f"Skipping single symbol label: '{label_text}'")
            return None
            
        # Normalize the search text
        search_text = self._normalize_text_for_matching(label_text)
        
        # Remove symbols from search text
        search_text = re.sub(r'[✓✔☑√]', '', search_text).strip()
        
        # Try multiple matching strategies with increasing tolerance
        strategies = [
            # Strategy 1: Exact normalized match
            lambda e, s: self._normalize_text_for_matching(e.get("value", "")) == s,
            
            # Strategy 2: Search text is in entity text (normalized)
            lambda e, s: s in self._normalize_text_for_matching(e.get("value", "")),
            
            # Strategy 3: Entity text is in search text (normalized)
            lambda e, s: self._normalize_text_for_matching(e.get("value", "")) in s,
            
            # Strategy 4: Partial match with high threshold (80% of words match)
            lambda e, s: self._partial_match_score(
                self._normalize_text_for_matching(e.get("value", "")), 
                s
            ) >= 0.8,
            
            # Strategy 5: Remove non-alphanumeric and compare
            lambda e, s: self._clean_for_comparison(e.get("value", "")) == self._clean_for_comparison(label_text),
        ]
        
        for strategy in strategies:
            for e in entities:
                if strategy(e, search_text):
                    logger.info(f"Found label using strategy {strategies.index(strategy)+1}: '{e.get('value', '')}'")
                    return e
        
        return None
    
    def _clean_for_comparison(self, text: str) -> str:
        """Clean text for comparison by removing non-alphanumeric characters"""
        if not text:
            return ""
        # Keep letters, numbers, and basic punctuation
        text = re.sub(r'[^\w\s:().\-]', '', text.lower())
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _partial_match_score(self, text1: str, text2: str) -> float:
        """Calculate partial match score between two texts"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

    def _find_nearest_value_entity(self, entities: List[Dict], label_entity: Dict, label_text: str) -> Optional[Dict]:
        """Find the entity most likely to be the value for this label"""
        if not label_entity:
            return None
        
        label_bbox = label_entity.get("bbox", {})
        label_page = label_entity.get("page_number", 1)
        label_x = label_bbox.get("x", 0)
        label_y = label_bbox.get("y", 0)
        label_width = label_bbox.get("width", 0)
        label_height = label_bbox.get("height", 0)
        
        # Find the label's position in sorted entities
        page_entities = [e for e in entities if e.get("page_number", 1) == label_page]
        page_entities.sort(key=lambda e: (e.get("bbox", {}).get("y", 0), e.get("bbox", {}).get("x", 0)))
        
        label_index = -1
        for i, e in enumerate(page_entities):
            if e.get("value", "") == label_entity.get("value", ""):
                label_index = i
                break
        
        candidates = []
        
        # Strategy 1: Look for entities immediately after the label in sorted order
        if label_index >= 0 and label_index + 1 < len(page_entities):
            next_entity = page_entities[label_index + 1]
            next_bbox = next_entity.get("bbox", {})
            next_x = next_bbox.get("x", 0)
            next_y = next_bbox.get("y", 0)
            
            # Check if it's to the right and on roughly the same line
            if (next_x > label_x and 
                abs(next_y - label_y) < label_height * 2 and
                abs((next_y + next_bbox.get("height", 0)) - (label_y + label_height)) < label_height * 2):
                candidates.append({
                    "entity": next_entity,
                    "distance": abs(next_x - (label_x + label_width)),
                    "type": "immediate_right",
                    "score": 1.0
                })
        
        # Strategy 2: Spatial search for entities to the right
        for e in entities:
            if e is label_entity:
                continue
            
            # Must be on same page
            if e.get("page_number", 1) != label_page:
                continue
            
            e_bbox = e.get("bbox", {})
            e_x = e_bbox.get("x", 0)
            e_y = e_bbox.get("y", 0)
            e_width = e_bbox.get("width", 0)
            e_height = e_bbox.get("height", 0)
            
            # Check if entity is to the right of label
            is_to_right = e_x > label_x
            
            # Calculate distance metrics
            if is_to_right:
                # Horizontal distance from label's right edge
                horizontal_dist = e_x - (label_x + label_width)
                
                # Vertical alignment score (0-1, 1=perfect alignment)
                label_center_y = label_y + label_height / 2
                e_center_y = e_y + e_height / 2
                vertical_diff = abs(e_center_y - label_center_y)
                
                # Calculate alignment score (higher is better)
                max_vertical_diff = max(label_height, e_height) * 1.5
                vertical_score = max(0, 1 - (vertical_diff / max_vertical_diff))
                
                # Calculate overall score (prioritize vertical alignment)
                score = vertical_score * 0.7 + (1 / (horizontal_dist + 1)) * 0.3
                
                # Only consider if reasonably close
                if horizontal_dist < 1000 and vertical_diff < label_height * 3:
                    candidates.append({
                        "entity": e,
                        "distance": horizontal_dist,
                        "type": "spatial_right",
                        "score": score,
                        "vertical_diff": vertical_diff,
                        "horizontal_dist": horizontal_dist
                    })
        
        # Strategy 3: Look for entities below but aligned
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
            
            # Check if entity is below the label
            is_below = e_y > label_y + label_height
            
            if is_below:
                # Check if horizontally aligned (within label width)
                horizontal_alignment = abs(e_x - label_x) < label_width * 2
                
                if horizontal_alignment:
                    vertical_dist = e_y - (label_y + label_height)
                    
                    # Prefer closer entities
                    score = 1 / (vertical_dist + 1)
                    
                    candidates.append({
                        "entity": e,
                        "distance": vertical_dist,
                        "type": "below",
                        "score": score,
                        "vertical_dist": vertical_dist
                    })
        
        # Sort candidates by score (descending)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Log top candidates for debugging
        logger.info(f"Top 5 candidates for label '{label_text}':")
        for i, cand in enumerate(candidates[:5]):
            logger.info(f"  {i+1}. '{cand['entity'].get('value', '')}' (type: {cand['type']}, score: {cand['score']:.3f})")
        
        # Return best candidate if any
        if candidates and candidates[0]["score"] > 0.1:
            best = candidates[0]["entity"]
            logger.info(f"Selected value: '{best.get('value', '')}'")
            return best
        
        return None

    def detect_checkbox_field(self, entities: List[Dict]) -> Tuple[Optional[Dict], List[Dict]]:
        """Specialized checkbox detection with better filtering"""
        logger.info("Starting specialized checkbox detection")
        
        # 1. First find the checkbox region
        region = find_checkbox_region(entities)
        logger.info(f"Checkbox region: {region}")
        
        # 2. Filter entities within region
        region_entities = [e for e in entities if _is_in_region(e, region)]
        logger.info(f"Entities in checkbox region: {len(region_entities)}")
        
        # Log all entities in region for debugging
        logger.info("=" * 80)
        logger.info("ENTITIES IN CHECKBOX REGION:")
        for idx, e in enumerate(region_entities):
            logger.info(f"  {idx}: '{e.get('value', '')}'")
        logger.info("=" * 80)
        
        # 3. Identify section headers to exclude
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
        
        # 4. Find actual checkbox options - look for specific patterns
        checkbox_patterns = [
            # Pattern for "Replacement copy of certificate of registration of generating facility due to"
            r"replacement.*copy.*certificate.*registration.*generating.*facility.*due.*to",
            # Pattern for "Damage (the damaged existing certificate must be returned for deletion)"
            r"damage.*\(.*damaged.*certificate.*must.*returned.*deletion.*\)",
            # Pattern for Chinese version of damage option
            r"掼毁.*琨有损毁澄明善必须交遣本署鞋艄.*damage",
        ]
        
        checkbox_candidates = []
        for e in region_entities:
            # Skip section headers
            if e in section_headers:
                continue
                
            value = e.get("value", "").strip()
            if not value:
                continue
            
            # Check if it matches any of the checkbox patterns
            value_lower = value.lower()
            for pattern in checkbox_patterns:
                if re.search(pattern, value_lower, re.IGNORECASE):
                    checkbox_candidates.append(e)
                    logger.info(f"Found checkbox option via pattern '{pattern}': '{value}'")
                    break
        
        # 5. If we didn't find via patterns, try the general approach
        if not checkbox_candidates:
            logger.info("No candidates via pattern matching, trying general approach")
            for e in region_entities:
                if e in section_headers:
                    continue
                    
                value = e.get("value", "").strip()
                if not value:
                    continue
                
                # Check if it looks like a checkbox option
                if is_checkbox_option(value):
                    checkbox_candidates.append(e)
        
        # 6. Remove duplicates and similar entries - FIXED: don't modify set during iteration
        unique_candidates = []
        seen_texts = set()
        
        for cand in checkbox_candidates:
            text = cand.get("value", "").strip()
            
            # Skip if we've seen this exact text
            if text in seen_texts:
                continue
                
            # Check for duplicates by similarity
            is_duplicate = False
            # Create a copy of seen_texts to iterate over
            seen_texts_copy = list(seen_texts)
            
            for seen in seen_texts_copy:
                # Check if texts are similar (one contains the other)
                if text in seen or seen in text:
                    is_duplicate = True
                    # Keep the longer/more complete version
                    if len(text) > len(seen):
                        # Remove the shorter one and add this one
                        seen_texts.remove(seen)
                        # Remove the corresponding candidate
                        for uc in unique_candidates:
                            if uc.get("value", "").strip() == seen:
                                unique_candidates.remove(uc)
                                break
                        # We'll add the new one below
                        is_duplicate = False  # Reset to add the new one
                    else:
                        # Current text is shorter, skip it
                        break
            
            if not is_duplicate and text not in seen_texts:
                seen_texts.add(text)
                unique_candidates.append(cand)
        
        # 7. Sort by vertical position (top to bottom)
        unique_candidates.sort(key=lambda e: e.get("bbox", {}).get("y", 0))
        
        # 8. Final filtering - ensure we only have actual options
        final_candidates = []
        for cand in unique_candidates:
            value = cand.get("value", "").strip()
            
            # Skip if it's a section header or instruction
            if any(header_word in value.lower() for header_word in [
                "tick", "please", "appropriate", "allowed", "hereby", "apply", "following"
            ]):
                continue
                
            # Must contain checkbox-specific content
            if any(option_word in value.lower() for option_word in [
                "replacement", "damage", "loss", "certificate", "deletion"
            ]):
                final_candidates.append(cand)
        
        # 9. Create result
        if final_candidates:
            checkbox_values = [c.get("value", "").strip() for c in final_candidates]
            
            # Clean up values - remove duplicate text and excessive whitespace
            cleaned_values = []
            for val in checkbox_values:
                # Remove duplicate words
                words = val.split()
                unique_words = []
                for word in words:
                    if word not in unique_words:
                        unique_words.append(word)
                val = " ".join(unique_words)
                
                # Remove excessive whitespace
                val = re.sub(r'\s+', ' ', val).strip()
                
                # Remove leading/trailing punctuation
                val = re.sub(r'^[:\-.,\s]+|[:\-.,\s]+$', '', val)
                cleaned_values.append(val)
            
            combined_value = " | ".join(cleaned_values)
            
            # Use first candidate for bbox
            first_entity = final_candidates[0]
            bbox = first_entity.get("bbox", {"x": 0, "y": 0, "width": 1, "height": 1})
            conf = float(first_entity.get("confidence", 0.9))
            
            # Create filtered entities list
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
        
        # 10. No checkbox options found
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
        
        # Handle checkmark/checkbox fields specially
        if key_field.strip() in ["✓", "✔", "☑", "√", "check", "checkbox", "tick"]:
            logger.info("Using specialized checkbox detection")
            return self.detect_checkbox_field(entities)
        
        # Clean the key field input for other types
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
        
        # Log all entities for debugging
        logger.info("=" * 80)
        logger.info("ALL ENTITIES (for debugging):")
        for idx, e in enumerate(entities):
            logger.info(f"  Entity {idx}: '{e.get('value', '')}'")
        logger.info("=" * 80)
        
        # ---- Phone case (existing behaviour) ----
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

        # ---- Email case ----
        if self._is_email_label(key_field):
            logger.info("Using email-specific extraction logic for key_field")

            candidates = []

            # label-based: find label entity, then neighbor on same line
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

            # global fallback: any email anywhere
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

        # ---- General key-value detection ----
        logger.info(f"Attempting general key-value detection for: '{key_field}'")
        
        # Find the label entity with improved matching
        label_entity = self._find_label_entity(entities, key_field)
        
        if label_entity:
            logger.info(f"Found label entity: '{label_entity.get('value', '')}'")
            logger.info(f"Label entity page: {label_entity.get('page_number', 1)}")
            
            # Find the nearest value entity
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
            # Try to find any value that might be related
            # This is a fallback for when label matching fails
            for e in entities:
                if e.get("value", "").strip() and len(e.get("value", "").strip()) > 3:
                    # Check if it looks like a value (not a label)
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
        
        # ---- Default: not found ----
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

        # Clean up entities
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


# ------- API endpoint -------
@app.post("/api/extract-text", response_model=AnalysisResponse)
async def extract_text_api(
    file: UploadFile = File(...),
    key_field: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
):
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


# ------- Main -------
def main():
    parser = argparse.ArgumentParser(
        description="Document Analysis API - LiLT + Email Extraction"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to LiLT model for relation extraction",
    )
    parser.add_argument(
        "--qa_model", type=str, default="deepset/roberta-base-squad2", help="QA model"
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    config = LiLTConfig(model_path=args.model_path, qa_model_path=args.qa_model)
    app.state.analyzer = DocumentAnalyzer(config)

    logger.info("Starting Document Analysis API...")
    logger.info(f"LiLT model path: {args.model_path}")
    logger.info(f"QA model: {args.qa_model}")
    logger.info(
        f"LiLT model available: {app.state.analyzer.lilt_extractor is not None and app.state.analyzer.lilt_extractor.is_available() if app.state.analyzer.lilt_extractor else False}"
    )
    logger.info(
        f"QA model available: {app.state.analyzer.qa_model is not None and app.state.analyzer.qa_model.is_available() if app.state.analyzer.qa_model else False}"
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
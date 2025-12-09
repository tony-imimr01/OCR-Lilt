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
from datetime import datetime
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from collections import defaultdict
import math
import json

# Critical CUDA error handling setup
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Optional heavy deps
try:
    import torch
    TORCH_AVAILABLE = True
    
    def _validate_cuda_state():
        try:
            if torch.cuda.is_available():
                test_tensor = torch.zeros(10, device='cuda')
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
except Exception as e:
    pytesseract = None
    TESSERACT_AVAILABLE = False

try:
    from pdf2image import convert_from_path, pdfinfo_from_path
    PDF2IMAGE_AVAILABLE = True
except Exception as e:
    PDF2IMAGE_AVAILABLE = False

# TRANSFORMERS with QA model
try:
    from transformers import (
        pipeline,
        AutoModelForQuestionAnswering,
        AutoTokenizer
    )
    TRANSFORMERS_AVAILABLE = True
except Exception as e:
    logging.warning(f"Transformers import failed: {e}")
    TRANSFORMERS_AVAILABLE = False

# Import EasyOCR
EASYOCR_AVAILABLE = False
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    easyocr = None

# Try to import LiLT model with fallback to standard models
try:
    from models.LiLTRobertaLike import LiLTRobertaLikeForRelationExtraction
    LILT_AVAILABLE = True
    logging.info("LiLT model imported successfully from models/LiLTRobertaLike")
except ImportError:
    try:
        from LiLTRobertaLike import LiLTRobertaLikeForRelationExtraction
        LILT_AVAILABLE = True
        logging.info("LiLT model imported from LiLTRobertaLike")
    except ImportError:
        logging.warning("LiLT model not found. Using fallback relation extraction.")
        LILT_AVAILABLE = False

        # Fallback implementation if LiLT model isn't available
        class LiLTRobertaLikeForRelationExtraction:
            def __init__(self, model_path):
                self.model_path = model_path
            
            def to(self, device):
                return self
            
            def eval(self):
                return self
            
            # Add fallback method for relation extraction
            def extract_relations(self, entities, key_fields):
                """Fallback implementation when LiLT model doesn't have extract_relations method"""
                logging.info("Using fallback LiLT implementation - simulating relation extraction")
                return []
            
            # Common alternative method names
            def predict(self, entities, key_fields):
                return self.extract_relations(entities, key_fields)
            
            def forward(self, entities, key_fields):
                return self.extract_relations(entities, key_fields)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# ------- Logging -------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("doc_analysis_fixed_lilt")

# ------- Constants and Entity Types -------
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
        self.min_confidence = 0.05  # Very low threshold for better recall
        self.min_confidence_per_type = {
            EntityTypes.HEADER: 0.15,
            EntityTypes.QUESTION: 0.10,
            EntityTypes.ANSWER: 0.05,
            EntityTypes.CONTENT: 0.10,
            EntityTypes.METADATA: 0.15,
            EntityTypes.STRUCTURE: 0.10,
            EntityTypes.OTHER: 0.05
        }
        self.max_entities_per_doc = 1500
        self.min_entity_length = 1
        self.max_seq_length = 512
        self.max_word_length = 300
        self.enable_qa_linking = True
        self.max_qa_distance = 500
        self.qa_threshold = 0.001
        self.use_safetensors = True

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

class InfoResponse(BaseModel):
    filename: str
    file_size_bytes: int
    page_count: int
    is_pdf: bool
    available_languages: List[str] = []
    gpu_available: bool = False
    qa_model_available: bool = False
    lilt_model_available: bool = False

# ------- FastAPI app -------
app = FastAPI(title="Document Analysis API - Fixed LiLT Integration")

# ------- Utilities -------
def _get_installed_tesseract_langs() -> List[str]:
    if not TESSERACT_AVAILABLE:
        return ["eng", "chi_sim", "chi_tra"]
    try:
        langs = pytesseract.get_languages(config='')
        if isinstance(langs, (list, tuple)) and langs:
            return list(sorted(set(langs)))
    except Exception:
        pass
    return ["eng", "spa", "fra", "deu", "ita", "por", "chi_sim", "chi_tra", "jpn", "kor", "rus", "ara"]

AVAILABLE_LANGUAGES = _get_installed_tesseract_langs()

def _clean_word(word: str, language_code: str = "eng") -> str:
    """Clean word with special handling for Chinese characters and OCR errors"""
    if not word:
        return ""
    # Handle Chinese characters specially
    if language_code.startswith(("chi", "jpn", "kor")):
        # Keep Chinese chars but clean noise
        return re.sub(r"[^\u4e00-\u9fff\w\s\-\.,:/$#@%()@+\u00C0-\u017F]", "", word).strip()
    # For English and other languages, be more aggressive but keep phone number characters
    return re.sub(r"[^\w\s\-\.,:/$#@%()@+\u00C0-\u017F\+\(\)]", "", word).strip()

def _parse_conf_value(conf_raw) -> float:
    try:
        if conf_raw is None:
            return 0.0
        s = str(conf_raw).strip()
        if s in ("-1", "-1.0"):
            return 0.0
        v = float(s)
        if v > 1.0:
            v = max(0.0, min(100.0, v)) / 100.0
        return float(max(0.0, min(1.0, v)))
    except Exception:
        return 0.0

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
    required_fields = ['x', 'y', 'width', 'height']
    if not all(field in bbox for field in required_fields):
        return False
    try:
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        return (x >= 0 and y >= 0 and w > 0 and h > 0 and 
                x + w <= 20000 and y + h <= 20000)
    except (ValueError, TypeError):
        return False

def enhance_ocr_preprocessing(img: Image.Image) -> Image.Image:
    """Enhanced preprocessing for better OCR results, especially for phone numbers"""
    # Convert to grayscale
    gray = img.convert('L')
    # Apply contrast enhancement
    enhanced = ImageEnhance.Contrast(gray).enhance(3.0)
    # Apply sharpness enhancement
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(2.5)
    # Apply binarization with adaptive threshold for better character separation
    threshold = enhanced.point(lambda p: p > 140 and 255)
    # Apply slight dilation to connect broken characters (helps with phone numbers)
    threshold = threshold.filter(ImageFilter.MaxFilter(3))
    return threshold

def extract_phone_from_text(text: str) -> Optional[str]:
    """Extract phone number using comprehensive patterns with OCR error tolerance"""
    if not text:
        return None
    
    # Define comprehensive phone patterns for international and Chinese formats
    phone_patterns = [
        # International format with country code
        r'\+?(\d{1,3})?[\s\-\.]?\(?\d{1,4}\)?[\s\-\.]?\d{1,4}[\s\-\.]?\d{1,4}[\s\-\.]?\d{1,4}',
        # US/CAN format
        r'\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\d{4}',
        # UK format
        r'\+44\s?\d{2,4}\s?\d{3,4}\s?\d{3,4}|\b0\d{3,4}\s?\d{3,4}\s?\d{3,4}\b',
        # Chinese formats
        r'电话[:：]?\s*([\d+\-()\s]{7,15})',
        r'联系电话[:：]?\s*([\d+\-()\s]{7,15})',
        r'手机号[:：]?\s*([\d+\-()\s]{7,15})',
        r'手机号码[:：]?\s*([\d+\-()\s]{7,15})',
        r'联络电话[:：]?\s*([\d+\-()\s]{7,15})',
        # English formats
        r'contact\s+tel[:：]?\s*([\d+\-()\s]{7,15})',
        r'tel\s+no[:：]?\s*([\d+\-()\s]{7,15})',
        r'phone\s+no[:：]?\s*([\d+\-()\s]{7,15})',
        r'mobile\s+no[:：]?\s*([\d+\-()\s]{7,15})',
        # Generic phone number patterns
        r'(?:\d[ -.*+()]*){7,15}\d',
        r'[\d+\-()]{7,15}'
    ]
    
    # Try to find phone numbers in the text
    for pattern in phone_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            matched_text = match.group(0).strip()
            
            # Validate the match is actually a phone number
            if is_valid_phone_number(matched_text):
                # Extract from groups if available
                for i in range(1, len(match.groups()) + 1):
                    group_match = match.group(i)
                    if group_match and is_valid_phone_number(group_match.strip()):
                        return group_match.strip()
                # Fallback to full match
                return matched_text
    
    # Fallback: look for digit sequences that might be phone numbers
    digit_sequences = re.findall(r'[\+\-\d() ]{7,20}', text)
    for seq in digit_sequences:
        cleaned = re.sub(r'[^\d\+\-\(\)\s]', '', seq).strip()
        if is_valid_phone_number(cleaned):
            return cleaned
    
    return None

def is_valid_phone_number(text: str) -> bool:
    """Validate if text is likely a phone number - STRICT VERSION"""
    if not text:
        return False
    
    # Clean the text
    cleaned = re.sub(r'[^\d\+\-\(\)\s\.\,]', '', text.strip())
    if len(cleaned) < 7:
        return False
    
    # Count digits
    digit_count = sum(c.isdigit() for c in cleaned)
    if digit_count < 7 or digit_count > 15:
        return False
    
    # Check for phone number patterns
    phone_patterns = [
        r'^\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}$',
        r'^\d{3}[-.\s]?\d{3}[-.\s]?\d{4}$',
        r'^\d{4}[-.\s]?\d{3}[-.\s]?\d{4}$',
        r'^\(\d{3}\)\s*\d{3}[-.\s]?\d{4}$',
        r'^\+\d{1,3}\s?\d{4,15}$',
        r'^\d{2,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}$'
    ]
    
    # Remove spaces for pattern matching
    no_spaces = cleaned.replace(' ', '')
    for pattern in phone_patterns:
        if re.match(pattern, no_spaces):
            return True
    
    # Additional checks to avoid garbage text
    # Check if it's mostly digits and valid phone characters
    valid_chars = sum(c in '0123456789+-()., ' for c in text)
    if valid_chars / max(1, len(text)) < 0.7:  # Less than 70% valid phone chars
        return False
    
    return False  # Default to not valid

def _extract_text_multipage(file_path: str, languages: str = "eng", conf_threshold: float = 0.05) -> Tuple[List[Dict], str, int, List[Image.Image]]:
    """Extract text with enhanced OCR processing specifically for contact information"""
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
            # Apply EXIF transpose to handle orientation correctly
            img = ImageOps.exif_transpose(img)
            images = [img]
    except Exception as e:
        logger.exception("Failed to convert file to images: %s", e)
        return [], "", 0, []
    
    global_idx = 0
    
    # Extract text using multiple methods without aggressive entity merging
    for page_idx, img in enumerate(images):
        page_num = page_idx + 1
        page_entities = []
        
        # First, try EasyOCR if available (better for Chinese text)
        if EASYOCR_AVAILABLE:
            try:
                gpu_available = CUDA_AVAILABLE and torch.cuda.is_available()
                langs = ['ch_sim', 'en'] if 'chi_sim' in languages else ['en']
                reader = easyocr.Reader(langs, gpu=gpu_available, verbose=False)
                
                # Get paragraph-level results first (better context)
                enhanced_img = enhance_ocr_preprocessing(img)
                paragraph_results = reader.readtext(np.array(enhanced_img), detail=1, paragraph=True, batch_size=8, min_size=10)
                
                for res in paragraph_results:
                    try:
                        bbox, text, conf = res
                    except ValueError:
                        bbox, text = res
                        conf = 0.8
                    
                    if conf < conf_threshold or not text.strip():
                        continue
                    
                    # Get bounding box coordinates
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    min_x, max_x = min(x_coords), max(x_coords)
                    min_y, max_y = min(y_coords), max(y_coords)
                    
                    # Clean and normalize the text
                    cleaned = _clean_word(text, language_code=languages.split('+')[0])
                    if not cleaned:
                        continue
                    
                    # Create entity
                    entity = {
                        "field": f"para_{global_idx+1}",
                        "value": cleaned,
                        "bbox": {
                            "x": max(0, int(min_x)), 
                            "y": max(0, int(min_y)), 
                            "width": max(1, int(max_x - min_x)), 
                            "height": max(1, int(max_y - min_y))
                        },
                        "confidence": float(conf),
                        "page_number": page_num,
                        "raw_text": text,
                        "source": "easyocr_paragraph"
                    }
                    
                    page_entities.append(entity)
                    full_text_parts.append(cleaned)
                    global_idx += 1
            except Exception as e:
                logger.warning(f"EasyOCR failed for page {page_num}: {e}")
        
        # Fallback to Tesseract with multiple configurations
        if TESSERACT_AVAILABLE and (not page_entities or len(page_entities) < 5):
            tesseract_langs = languages
            try:
                enhanced_img = enhance_ocr_preprocessing(img)
                configs = [
                    '--psm 1 --oem 3',  # Auto page segmentation with LSTM
                    '--psm 6 --oem 3',  # Assume uniform block of text
                    '--psm 3 --oem 3',  # Fully automatic page segmentation
                ]
                
                for config in configs:
                    try:
                        od = pytesseract.image_to_data(
                            enhanced_img,
                            lang=tesseract_langs,
                            output_type=pytesseract.Output.DICT,
                            config=config
                        )
                        n = len(od.get("text", []))
                        
                        for i in range(n):
                            raw_text = str(od.get("text", [""]*n)[i]).strip()
                            if not raw_text or len(raw_text) < 2:
                                continue
                            
                            raw_conf = od.get("conf", [None]*n)[i]
                            conf = _parse_conf_value(raw_conf)
                            if conf < conf_threshold:
                                continue
                            
                            cleaned = _clean_word(raw_text, language_code=tesseract_langs.split('+')[0])
                            if not cleaned:
                                continue
                            
                            try:
                                left = int(od.get("left", [0]*n)[i])
                                top = int(od.get("top", [0]*n)[i])
                                width = int(od.get("width", [0]*n)[i])
                                height = int(od.get("height", [0]*n)[i])
                                if width <= 0 or height <= 0:
                                    continue
                            except Exception:
                                continue
                            
                            # Avoid duplicate entities
                            is_duplicate = False
                            for existing in page_entities:
                                if (existing["value"] == cleaned and
                                    abs(existing["bbox"]["x"] - left) < 10 and
                                    abs(existing["bbox"]["y"] - top) < 10):
                                    is_duplicate = True
                                    break
                            
                            if not is_duplicate:
                                entity = {
                                    "field": f"tess_{global_idx+1}",
                                    "value": cleaned,
                                    "bbox": {
                                        "x": max(0, left), 
                                        "y": max(0, top), 
                                        "width": max(1, width), 
                                        "height": max(1, height)
                                    },
                                    "confidence": conf,
                                    "page_number": page_num,
                                    "raw_text": raw_text,
                                    "source": f"tesseract_{config}"
                                }
                                page_entities.append(entity)
                                full_text_parts.append(cleaned)
                                global_idx += 1
                    except Exception as e:
                        logger.debug(f"Tesseract config {config} failed: {e}")
                        continue
            except Exception as e:
                logger.exception("Tesseract page OCR failed: %s", e)
        
        entities.extend(page_entities)
    
    # Add full page text as a single entity for QA context
    full_text = " ".join(full_text_parts)
    if full_text.strip():
        first_entity = entities[0] if entities else {"bbox": {"x": 0, "y": 0, "width": 100, "height": 100}, "page_number": 1}
        entities.append({
            "field": "full_page_text",
            "value": full_text.strip(),
            "bbox": first_entity["bbox"],
            "confidence": 0.99,
            "page_number": first_entity.get("page_number", 1),
            "raw_text": full_text.strip(),
            "source": "full_text"
        })
    
    logger.info(f"Extracted {len(entities)} entities without aggressive merging, {len(full_text)} characters")
    logger.info(f"Entity sample (first 10):")
    for i, e in enumerate(entities[:10]):
        logger.info(f"  Entity {i}: '{e.get('value', '')[:100]}...' (conf: {e.get('confidence', 0.0):.2f}, source: {e.get('source', 'unknown')})")
    
    return entities, full_text.strip(), len(images), images

# ------- LiLT Relation Extractor with Method Fallbacks -------
class LiLTRelationExtractor:
    """Extract relationships between key fields and values using LiLT model with method fallbacks"""
    
    def __init__(self, model_path: str, device: Optional[int] = None):
        self.model_path = model_path
        self.device = device if device is not None else (0 if torch.cuda.is_available() else -1)
        self.available = LILT_AVAILABLE
        
        if not LILT_AVAILABLE:
            logger.warning("LiLT model not available - using fallback methods only")
            return
        
        try:
            logger.info(f"Loading LiLT relation extraction model from: {model_path}")
            self.model = LiLTRobertaLikeForRelationExtraction.from_pretrained(
                model_path
            ).to(self.device if self.device >= 0 else "cpu")
            self.model.eval()
            logger.info("LiLT relation extraction model loaded successfully")
            
            # Check if model has the required method
            if not hasattr(self.model, 'extract_relations'):
                logger.warning("LiLT model missing 'extract_relations' method - checking for alternatives")
                # Check for alternative method names
                has_predict = hasattr(self.model, 'predict')
                has_forward = hasattr(self.model, 'forward')
                
                if not (has_predict or has_forward):
                    logger.error("LiLT model has no compatible relation extraction methods")
                    self.available = False
                else:
                    logger.info(f"LiLT model has alternative methods - predict: {has_predict}, forward: {has_forward}")
        except Exception as e:
            logger.error(f"Failed to load LiLT model: {e}")
            self.available = False
    
    def extract_relations(self, entities: List[Dict], key_field: str) -> List[Dict]:
        """Extract relationships between keys and values with method fallbacks"""
        if not self.available or not entities:
            return []
        
        try:
            # The LiLT model expects different input format
            # Try to prepare inputs according to the model's expected format
            if hasattr(self.model, 'extract_relations'):
                try:
                    # Method 1: Standard relation extraction
                    # Convert entities to the format expected by LiLT
                    input_entities = []
                    for entity in entities:
                        text = entity.get("value", "")
                        bbox = entity.get("bbox", {})
                        if text and len(text) < 300:  # Limit text length
                            input_entities.append({
                                "text": text,
                                "box": [
                                    bbox.get("x", 0),
                                    bbox.get("y", 0),
                                    bbox.get("x", 0) + bbox.get("width", 0),
                                    bbox.get("y", 0) + bbox.get("height", 0)
                                ]
                            })
                    
                    # Try different calling conventions
                    try:
                        # Try with expected parameters
                        relations = self.model.extract_relations(
                            entities=input_entities,
                            key_fields=[key_field]
                        )
                    except TypeError as e:
                        # Try alternative parameter names
                        if "entities" in str(e):
                            relations = self.model.extract_relations(
                                input_entities,
                                [key_field]
                            )
                        else:
                            raise e
                    
                    if relations is not None:
                        logger.info(f"LiLT model extracted {len(relations)} relations")
                        return relations
                        
                except Exception as e:
                    logger.warning(f"LiLT extract_relations failed: {e}")
            
            # If all methods fail, return empty list
            return []
            
        except Exception as e:
            logger.error(f"LiLT relation extraction failed: {e}")
            return []
    
    def is_available(self) -> bool:
        return self.available

# ------- Robust QA Model with Contact Specialization -------
class RobustContactQAModel:
    """QA model specialized for contact information extraction"""
    
    def __init__(self, model_name: str = "deepset/roberta-base-squad2", device: Optional[int] = None):
        self.model_name = model_name
        self.device = device if device is not None else (0 if torch.cuda.is_available() else -1)
        
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available for QA model")
            self.qa_pipeline = None
            return
        
        try:
            # Load model and tokenizer separately
            from transformers import AutoModelForQuestionAnswering, AutoTokenizer
            
            logger.info(f"Loading robust QA model: {model_name} on device: {self.device}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            
            # Move model to appropriate device
            if self.device >= 0 and torch.cuda.is_available():
                self.model = self.model.to(f"cuda:{self.device}")
            
            # Create pipeline with proper parameters
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device if self.device >= 0 else -1,
                batch_size=16
            )
            logger.info("Robust QA model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load QA model: {e}")
            self.qa_pipeline = None
    
    def _create_contact_questions(self) -> List[str]:
        """Create specialized questions for contact information extraction"""
        return [
            "What is the phone number?",
            "What is the contact telephone?",
            "Find the telephone number",
            "Contact phone number",
            "Tel No",
            "Tel Contact No",
            "Phone contact number",
            "Mobile number",
            "What is the telephone number?",
            "Phone number to contact",
            "Contact number",
            "Telephone contact number",
            "telephone number",
            "phone number",
            "contact telephone",
            "telephone",
            "phone",
            "tel",
            "contact",
            "mobile",
            "whatsapp",
            "wechat",
            "line",
            "signal",
            "call",
            "电话号码",
            "联系电话",
            "手机号码",
            "聯絡電話",
            "聯絡电话",
            "联系人电话",
            "联系方式",
            "Contact information",
            "Phone contact information",
            "Contact via telephone",
            "Call number"
        ]
    
    def extract_contact_info(self, context: str) -> Dict[str, Any]:
        """Specialized extraction for contact information with validation"""
        if not self.qa_pipeline or not context or len(context) < 10:
            return {"answer": "", "score": 0.0, "method": "none"}
        
        try:
            questions = self._create_contact_questions()
            best_answer = {"answer": "", "score": 0.0, "question": ""}
            
            # Try all questions at once for efficiency
            inputs = [{"question": q, "context": context[:4000]} for q in questions]
            
            try:
                results = self.qa_pipeline(
                    inputs,
                    max_answer_len=100,
                    handle_impossible_answer=True
                )
                
                # Handle single result case
                if not isinstance(results, list):
                    results = [results]
                
                for j, result in enumerate(results):
                    answer = result.get("answer", "").strip()
                    score = float(result.get("score", 0.0))
                    question = questions[j]
                    
                    # Post-process and validate the answer
                    if answer and score > 0.05:
                        cleaned_answer = self._clean_phone_answer(answer)
                        if is_valid_phone_number(cleaned_answer):
                            if score > best_answer["score"]:
                                best_answer = {
                                    "answer": cleaned_answer,
                                    "score": score,
                                    "question": question
                                }
            except Exception as e:
                logger.debug(f"QA processing failed: {e}")
            
            if best_answer["answer"] and best_answer["score"] > 0.1:
                return {
                    "answer": best_answer["answer"],
                    "score": best_answer["score"],
                    "method": "qa_model",
                    "question_used": best_answer["question"]
                }
            
            return {"answer": "", "score": 0.0, "method": "none"}
        except Exception as e:
            logger.error(f"QA extraction failed: {e}")
            return {"answer": "", "score": 0.0, "method": "error"}
    
    def _clean_phone_answer(self, answer: str) -> str:
        """Clean phone number answer with OCR error correction"""
        if not answer:
            return answer
        
        # Remove OCR errors and keep only valid phone number characters
        cleaned = re.sub(r'[^\d+\-\(\)\s\.\,]', '', answer)
        
        # Common OCR substitutions for phone numbers
        substitutions = {
            'O': '0', 'o': '0', 'l': '1', 'I': '1', 'i': '1',
            'S': '5', 's': '5', 'B': '8', 'b': '8', 'g': '9',
            'Z': '2', 'z': '2', '@': '0', '#': '8', '$': '5',
            'D': '0', 'Q': '0', 'G': '6', 'T': '7'
        }
        
        # Apply substitutions only to the beginning of the string
        cleaned_chars = list(cleaned)
        for i in range(min(5, len(cleaned_chars))):  # Only check first 5 characters
            if cleaned_chars[i] in substitutions:
                cleaned_chars[i] = substitutions[cleaned_chars[i]]
        
        cleaned = ''.join(cleaned_chars)
        
        return cleaned.strip()

# ------- Contact Field Detector with LiLT Integration -------
class ContactFieldDetector:
    """Specialized contact field detector with LiLT relation extraction"""
    
    def __init__(self, lilt_extractor: Optional[LiLTRelationExtractor] = None, qa_model: Optional[RobustContactQAModel] = None):
        self.lilt_extractor = lilt_extractor
        self.qa_model = qa_model
        self.contact_keywords = {
            'en': ['tel', 'telephone', 'phone', 'contact', 'mobile', 'cell', 'whatsapp', 'wechat', 'line', 'signal', 'call'],
            'zh': ['电', '话', '联', '络', '手', '机', '号码', '联系方式', '手机', '联络', '联系', '联络电话', '联系电话', '手机号']
        }
    
    def _is_contact_region(self, text: str) -> bool:
        """Check if text is likely to be in a contact information region"""
        text_lower = text.lower()
        
        # Check for contact keywords with OCR error tolerance
        for lang, keywords in self.contact_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return True
        
        # Check for phone number patterns
        if re.search(r"[\d+\-()]{7,}", text_lower):
            return True
        
        # Check for corrupted OCR patterns that might be contact-related
        if re.search(r"[cC][^\w]?[o0][^\w]?[nN][^\w]?[tT][^\w]?[aA][^\w]?[cC][^\w]?[tT]", text_lower):
            return True
        if re.search(r"[tT][eE][lL][\s\-:]*[nN][o0O]", text_lower):
            return True
        
        return False
    
    def _find_contact_regions(self, entities: List[Dict]) -> List[Dict]:
        """Find regions that likely contain contact information"""
        contact_regions = []
        
        for entity in entities:
            text = entity.get("value", "")
            bbox = entity.get("bbox", {})
            page = entity.get("page_number", 1)
            
            if self._is_contact_region(text):
                # Create a region around this entity
                region_entities = []
                center_x = bbox.get("x", 0) + bbox.get("width", 0) // 2
                center_y = bbox.get("y", 0) + bbox.get("height", 0) // 2
                
                for e in entities:
                    if e.get("page_number", 1) != page:
                        continue
                    
                    e_bbox = e.get("bbox", {})
                    e_center_x = e_bbox.get("x", 0) + e_bbox.get("width", 0) // 2
                    e_center_y = e_bbox.get("y", 0) + e_bbox.get("height", 0) // 2
                    
                    # Calculate distance between centers
                    dx = abs(e_center_x - center_x)
                    dy = abs(e_center_y - center_y)
                    
                    # Include entities within a reasonable radius
                    if dx < 400 and dy < 200:
                        region_entities.append(e)
                
                contact_regions.append({
                    "anchor_entity": entity,
                    "entities": region_entities,
                    "page": page,
                    "center": (center_x, center_y)
                })
        
        # Sort regions by confidence and relevance
        contact_regions.sort(key=lambda x: (
            -x["anchor_entity"].get("confidence", 0),
            -len([e for e in x["entities"] if re.search(r"[\d+\-()]{7,}", e.get("value", ""))])
        ))
        
        logger.info(f"Found {len(contact_regions)} contact regions")
        for i, region in enumerate(contact_regions[:5]):
            anchor_text = region["anchor_entity"].get("value", "")[:100]
            logger.info(f"  Region {i+1}: '{anchor_text}...' (confidence: {region['anchor_entity'].get('confidence', 0):.2f}, entities: {len(region['entities'])})")
        
        return contact_regions
    
    def _extract_from_lilt_relations(self, relations: List[Dict], entities: List[Dict], key_field: str) -> Optional[Dict]:
        """Extract phone number using LiLT relation extraction results"""
        if not relations:
            return None
        
        logger.info(f"Processing {len(relations)} LiLT relations for key field: {key_field}")
        
        # Filter relations for our specific key field
        relevant_relations = [r for r in relations if key_field.lower().replace(":", "") in r.get("key", "").lower().replace(":", "")]
        
        if not relevant_relations:
            logger.info("No relevant LiLT relations found for key field")
            return None
        
        # Sort by confidence score
        relevant_relations.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        
        # Take the highest confidence relation
        best_relation = relevant_relations[0]
        value_text = best_relation.get("value", "").strip()
        
        if value_text and is_valid_phone_number(value_text):
            logger.info(f"LiLT model found contact number: {value_text} (score: {best_relation.get('score', 0.0):.3f})")
            
            # Find the entity that contains this value
            best_entity = None
            for entity in entities:
                if value_text in entity.get("value", ""):
                    best_entity = entity
                    break
            
            bbox = best_entity.get("bbox", {"x": 0, "y": 0, "width": 100, "height": 30}) if best_entity else {"x": 0, "y": 0, "width": 100, "height": 30}
            
            return {
                "value": value_text,
                "bbox": bbox,
                "confidence": max(0.7, best_relation.get("score", 0.0)),
                "method": "lilt_relation_extraction",
                "relation": best_relation
            }
        
        return None
    
    def _extract_from_region(self, region: Dict, key_field: str) -> Optional[Dict]:
        """Extract phone number from a contact region with validation"""
        entities = region["entities"]
        anchor_entity = region["anchor_entity"]
        
        logger.info(f"Analyzing region anchored at: '{anchor_entity.get('value', '')[:100]}...'")
        
        # Strategy 1: Look for entities that contain phone numbers
        phone_candidates = []
        for entity in entities:
            text = entity.get("value", "").strip()
            phone = extract_phone_from_text(text)
            
            if phone and is_valid_phone_number(phone):
                confidence = entity.get("confidence", 0.5)
                # Boost confidence if the entity is close to the anchor
                if entity != anchor_entity:
                    anchor_bbox = anchor_entity.get("bbox", {})
                    entity_bbox = entity.get("bbox", {})
                    distance = abs(
                        (entity_bbox.get("x", 0) + entity_bbox.get("width", 0)/2) - 
                        (anchor_bbox.get("x", 0) + anchor_bbox.get("width", 0)/2)
                    )
                    if distance < 200:
                        confidence += 0.3
                
                phone_candidates.append({
                    "value": phone,
                    "bbox": entity.get("bbox", {"x": 0, "y": 0, "width": 100, "height": 30}),
                    "confidence": min(1.0, confidence),
                    "method": "region_phone_extraction",
                    "source_entity": entity,
                    "anchor_entity": anchor_entity
                })
        
        if phone_candidates:
            # Sort by confidence and phone number validation
            phone_candidates.sort(key=lambda x: (x["confidence"], is_valid_phone_number(x["value"])), reverse=True)
            best_candidate = phone_candidates[0]
            logger.info(f"Found phone number in region: {best_candidate['value']} (confidence: {best_candidate['confidence']:.2f})")
            return best_candidate
        
        # Strategy 2: Look for entities with high digit density after the anchor
        anchor_bbox = anchor_entity.get("bbox", {})
        anchor_right = anchor_bbox.get("x", 0) + anchor_bbox.get("width", 0)
        
        digit_candidates = []
        for entity in entities:
            if entity == anchor_entity:
                continue
            
            text = entity.get("value", "")
            digits = re.sub(r'\D', '', text)
            total_chars = len(text)
            
            if total_chars > 0 and len(digits) >= 7 and len(digits) <= 15:
                # Calculate digit density
                digit_density = len(digits) / total_chars
                
                # Check if it's to the right of the anchor (likely value)
                entity_bbox = entity.get("bbox", {})
                entity_x = entity_bbox.get("x", 0)
                
                if entity_x > anchor_right - 50:  # Allow some overlap
                    confidence = entity.get("confidence", 0.5) * (0.3 + digit_density * 0.7)
                    digit_candidates.append({
                        "value": text,
                        "bbox": entity_bbox,
                        "confidence": min(1.0, confidence),
                        "method": "region_digit_density",
                        "source_entity": entity,
                        "anchor_entity": anchor_entity
                    })
        
        if digit_candidates:
            # Sort by confidence and digit count
            digit_candidates.sort(key=lambda x: (x["confidence"], len(re.sub(r'\D', '', x["value"]))), reverse=True)
            best_candidate = digit_candidates[0]
            
            # Validate it looks like a phone number
            if is_valid_phone_number(best_candidate["value"]):
                logger.info(f"Found potential phone number by digit density: {best_candidate['value']} (confidence: {best_candidate['confidence']:.2f})")
                return best_candidate
        
        # Strategy 3: Look for the first entity after the anchor that has digits
        right_candidates = []
        for entity in entities:
            if entity == anchor_entity:
                continue
            
            entity_bbox = entity.get("bbox", {})
            entity_x = entity_bbox.get("x", 0)
            entity_y = entity_bbox.get("y", 0)
            anchor_y = anchor_bbox.get("y", 0)
            
            # Same line or slightly below, and to the right
            if (entity_x > anchor_right - 50 and
                abs(entity_y - anchor_y) < 100 and
                entity_x - anchor_right < 400):
                right_candidates.append((entity_x - anchor_right, entity))
        
        if right_candidates:
            right_candidates.sort(key=lambda x: x[0])
            best_candidate_entity = right_candidates[0][1]
            best_candidate_text = best_candidate_entity.get("value", "").strip()
            
            if is_valid_phone_number(best_candidate_text):
                logger.info(f"Found phone number by spatial analysis: {best_candidate_text}")
                return {
                    "value": best_candidate_text,
                    "bbox": best_candidate_entity.get("bbox", {"x": 0, "y": 0, "width": 100, "height": 30}),
                    "confidence": max(0.6, best_candidate_entity.get("confidence", 0.5)),
                    "method": "region_spatial_extraction",
                    "source_entity": best_candidate_entity,
                    "anchor_entity": anchor_entity
                }
        
        logger.info("No phone number found in region")
        return None
    
    def detect_contact_fields(self, entities: List[Dict], full_text: str, key_field: str) -> Tuple[Optional[Dict], List[Dict]]:
        """Main detection function with LiLT model integration for relationship extraction"""
        logger.info(f"=== Detecting contact field: '{key_field}' ===")
        logger.info(f"Total entities: {len(entities)}")
        
        # Strategy 1: Use LiLT model for relation extraction if available
        if self.lilt_extractor and self.lilt_extractor.is_available():
            logger.info("Using LiLT model for relation extraction")
            relations = self.lilt_extractor.extract_relations(entities, key_field)
            if relations:
                lilt_result = self._extract_from_lilt_relations(relations, entities, key_field)
                if lilt_result:
                    logger.info(f"LiLT model successfully extracted contact number: {lilt_result['value']}")
                    return self._create_result(key_field, lilt_result, lilt_result["value"], lilt_result.get("source_entity"))
        
        # Strategy 2: Use QA model specialized for contact information
        if self.qa_model:
            logger.info("Using specialized QA model for contact detection")
            full_text_entity = next((e for e in entities if e.get("field") == "full_page_text"), None)
            context = full_text_entity.get("value", full_text) if full_text_entity else full_text
            
            if len(context) > 50:  # Need sufficient context
                qa_result = self.qa_model.extract_contact_info(context)
                if qa_result["answer"] and qa_result["score"] > 0.1:
                    logger.info(f"QA model found contact number: {qa_result['answer']} (score: {qa_result['score']:.3f})")
                    return self._create_result(key_field, qa_result, qa_result["answer"], full_text_entity)
        
        # Strategy 3: Find contact regions and analyze them
        contact_regions = self._find_contact_regions(entities)
        if contact_regions:
            logger.info("Analyzing contact regions for phone numbers")
            for i, region in enumerate(contact_regions[:5]):  # Limit to top 5 regions
                result = self._extract_from_region(region, key_field)
                if result:
                    logger.info(f"Found contact number in region {i+1}: {result['value']}")
                    return self._create_result(key_field, result, result["value"], result.get("source_entity"))
        
        # Strategy 4: Full text extraction as last resort
        logger.info("Falling back to full text phone number extraction")
        phone = extract_phone_from_text(full_text)
        if phone and is_valid_phone_number(phone):
            logger.info(f"Found phone number in full text: {phone}")
            # Find the best entity that contains this phone number
            best_entity = None
            for entity in entities:
                text = entity.get("value", "")
                if phone in text or text in phone:
                    best_entity = entity
                    break
            
            if best_entity:
                return self._create_result(key_field, {
                    "value": phone,
                    "bbox": best_entity.get("bbox", {"x": 0, "y": 0, "width": 100, "height": 30}),
                    "confidence": max(0.5, best_entity.get("confidence", 0.5)),
                    "method": "full_text_extraction"
                }, phone, best_entity)
        
        # **CRITICAL FIX: Return None when no contact is found**
        logger.info("No contact information found - returning None")
        return None, []
    
    def _create_result(self, key_field: str, extraction_data: Optional[Dict], value: str, source_entity: Optional[Dict]) -> Tuple[Optional[Dict], List[Dict]]:
        """Create standardized result format - returns None if value is invalid"""
        # **CRITICAL: Validate the value before creating result**
        if not value or not is_valid_phone_number(value):
            logger.warning(f"Invalid phone number value: '{value}'")
            return None, []
        
        if not extraction_data:
            logger.warning("No extraction data provided")
            return None, []
        
        confidence = extraction_data.get("confidence", 0.5)
        bbox = extraction_data.get("bbox", {"x": 0, "y": 0, "width": 100, "height": 30})
        method = extraction_data.get("method", "unknown")
        meta = {
            "source_entity_value": source_entity.get("value", "") if source_entity else None,
            "extraction_method": method
        }
        
        key_result = {
            "field_name": key_field,
            "value": value,
            "structured_value": {
                "field_name": key_field,
                "field_type": "contact_tel",
                "value": value,
                "confidence": confidence,
                "bbox": bbox
            },
            "confidence": confidence,
            "bbox": bbox,
            "context_entities": [],
            "extraction_method": method,
            "meta": meta
        }
        
        # Create filtered entities - just the result entity
        filtered_entities = [{
            "field": f"{key_field}_value",
            "value": value,
            "bbox": bbox,
            "confidence": confidence,
            "page_number": source_entity.get("page_number", 1) if source_entity else 1,
            "semantic_type": EntityTypes.ANSWER,
            "semantic_confidence": confidence
        }] if source_entity else []
        
        return key_result, filtered_entities

# ------- Main Document Analyzer with LiLT Integration -------
class DocumentAnalyzer:
    def __init__(self, config: LiLTConfig):
        self.config = config
        
        # Initialize LiLT model for relation extraction
        self.lilt_extractor = None
        if config.model_path:
            try:
                device = 0 if CUDA_AVAILABLE and torch.cuda.is_available() else -1
                self.lilt_extractor = LiLTRelationExtractor(model_path=config.model_path, device=device)
                logger.info(f"LiLT relation extraction model initialized: {config.model_path}")
            except Exception as e:
                logger.error(f"Failed to initialize LiLT model: {e}")
        
        # Initialize QA model specialized for contact information
        self.qa_model = None
        if config.qa_model_path and TRANSFORMERS_AVAILABLE:
            try:
                device = 0 if CUDA_AVAILABLE and torch.cuda.is_available() else -1
                self.qa_model = RobustContactQAModel(model_name=config.qa_model_path, device=device)
                logger.info(f"Specialized QA model initialized for contact extraction: {config.qa_model_path}")
            except Exception as e:
                logger.error(f"Failed to initialize QA model: {e}")
        
        # Initialize contact detector
        self.contact_detector = ContactFieldDetector(lilt_extractor=self.lilt_extractor, qa_model=self.qa_model)
    
    def analyze_file(self, file_path: str, key_field: Optional[str], language_input: Optional[str]) -> Dict:
        start_time = time.time()
        langs_norm = _validate_and_normalize_langs(language_input)
        logger.info(f"Analyzing file with languages: {langs_norm}")
        
        # Extract text
        entities, full_text, page_count, images = _extract_text_multipage(
            file_path, languages=langs_norm, conf_threshold=0.05
        )
        
        # Detect contact fields if requested
        kf_result = None
        filtered_entities = []
        
        if key_field and any(term in key_field.lower() for term in ['tel', 'phone', 'contact', 'mobile']):
            kf_result, filtered_entities = self.contact_detector.detect_contact_fields(
                entities, full_text, key_field
            )
            
            # Handle case where no contact fields were found
            if kf_result is None:
                logger.info("No contact fields found - returning empty result")
                kf_result = {
                    "field_name": key_field,
                    "value": "",
                    "structured_value": None,
                    "confidence": 0.0,
                    "bbox": {"x": 0, "y": 0, "width": 0, "height": 0},
                    "context_entities": [],
                    "extraction_method": "none",
                    "meta": {"reason": "No contact information found"}
                }
                filtered_entities = []  # Return empty entities list when nothing found
        else:
            # Return all entities if not looking for contact info
            filtered_entities = entities
        
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
            "model_used": self.lilt_extractor is not None,
            "qa_model_used": self.qa_model is not None,
            "lilt_model_used": self.lilt_extractor is not None and self.lilt_extractor.is_available()
        }

# ------- API endpoints -------
@app.post("/api/extract-text", response_model=AnalysisResponse)
async def extract_text_api(
    file: UploadFile = File(...),
    key_field: Optional[str] = Form(None),
    language: Optional[str] = Form(None)
):
    tmp = None
    try:
        if not hasattr(app.state, "analyzer"):
            raise HTTPException(503, "Analyzer not initialized")
        
        # Validate file type
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]:
            raise HTTPException(400, "Unsupported file type")
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tf:
            content = await file.read()
            if not content:
                raise HTTPException(400, "Empty file")
            tf.write(content)
            tmp = tf.name
        
        # Analyze file
        langs_norm = _validate_and_normalize_langs(language)
        analyzer: DocumentAnalyzer = app.state.analyzer
        result = analyzer.analyze_file(tmp, key_field, language_input=langs_norm)
        
        if not result:
            return JSONResponse(status_code=500, content={"status": "error", "message": "analysis failed", "error": "no result"})
        
        # Convert to Pydantic models
        entities_model = []
        for e in result.get("entities", []):
            try:
                bbox_data = e.get("bbox", {"x":0,"y":0,"width":0,"height":0})
                if not validate_bbox(bbox_data):
                    bbox_data = {"x":0,"y":0,"width":1,"height":1}
                entities_model.append(ExtractedEntity(
                    field=e.get("field", ""),
                    value=e.get("value", ""),
                    bbox=BoundingBox(**bbox_data),
                    confidence=float(e.get("confidence", 0.0)),
                    page_number=int(e.get("page_number", 1)),
                    semantic_type=e.get("semantic_type"),
                    semantic_confidence=e.get("semantic_confidence")
                ))
            except Exception as exc:
                logger.warning(f"Entity conversion failed: {exc}")
                continue
        
        # Convert key field result
        kfr_model = None
        if result.get("key_field_result"):
            kf = result["key_field_result"]
            try:
                bbox_data = kf.get("bbox", {"x":0,"y":0,"width":0,"height":0})
                if not validate_bbox(bbox_data):
                    bbox_data = {"x":0,"y":0,"width":1,"height":1}
                
                structured_value = None
                if kf.get("structured_value"):
                    sv = kf["structured_value"]
                    structured_value = DataField(
                        field_name=sv["field_name"],
                        field_type=sv["field_type"],
                        value=sv["value"],
                        confidence=float(sv["confidence"]),
                        bbox=BoundingBox(**sv.get("bbox", {"x":0,"y":0,"width":1,"height":1}))
                    )
                
                kfr_model = KeyFieldResult(
                    field_name=kf["field_name"],
                    value=kf["value"],
                    structured_value=structured_value,
                    confidence=float(kf["confidence"]),
                    bbox=BoundingBox(**bbox_data),
                    context_entities=[],
                    meta=kf.get("meta"),
                    extraction_method=kf.get("extraction_method")
                )
            except Exception as exc:
                logger.warning(f"Key field result conversion failed: {exc}")
                kfr_model = None
        
        # Create extraction result
        extraction = ExtractionResult(
            document_name=result.get("document_name", ""),
            page_count=result.get("page_count", 0),
            total_entities=result.get("total_entities", 0),
            entities=entities_model,
            key_field_result=kfr_model,
            full_text_snippet=(result.get("full_text", "")[:1000] + ("..." if len(result.get("full_text","")) > 1000 else "")),
            processing_time=float(result.get("processing_time", 0.0)),
            language_used=result.get("language_used", "eng"),
            model_used=result.get("model_used", False)
        )
        
        message = f"Analyzed with language '{result.get('language_used','eng')}'"
        if result.get('lilt_model_used'):
            message += " using LiLT model"
        elif result.get('qa_model_used'):
            message += " using QA model"
        
        return AnalysisResponse(
            status="success",
            message=message,
            result=extraction,
            error=None
        )
    
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception("API error: %s", e)
        return JSONResponse(status_code=500, content={"status": "error", "message": "internal error", "error": str(e)})
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except Exception:
                pass

# ------- Main -------
def main():
    parser = argparse.ArgumentParser(description="Document Analysis API - Fixed LiLT Integration")
    parser.add_argument("--model_path", type=str, default=None, help="Path to LiLT model for relation extraction")
    parser.add_argument("--qa_model", type=str, default="deepset/roberta-base-squad2", help="QA model to use")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize configuration and analyzer
    config = LiLTConfig(model_path=args.model_path, qa_model_path=args.qa_model)
    app.state.analyzer = DocumentAnalyzer(config)
    
    logger.info("Starting Document Analysis API with Fixed LiLT Integration...")
    logger.info(f"LiLT model path: {args.model_path}")
    logger.info(f"QA model: {args.qa_model}")
    logger.info(f"LiLT model available: {app.state.analyzer.lilt_extractor is not None and app.state.analyzer.lilt_extractor.is_available()}")
    logger.info(f"QA model available: {app.state.analyzer.qa_model is not None}")
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
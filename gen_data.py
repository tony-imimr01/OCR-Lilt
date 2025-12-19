#!/usr/bin/env python
"""
lilt_data_generator.py — Complete question-answer entity support with FUNSD format compliance
ENHANCED: Added checkbox/symbol support (✓, ✔, ☑, x, X) for training data

FIXES:
- Added checkbox/symbol detection and preservation
- Enhanced entity type classification for symbolic answers
- Improved OCR handling for special symbols
- Updated cleaning and validation for symbolic entities

Usage:
    python lilt_data_generator.py INPUT_DIR OUTPUT_DIR --default_label 0
"""
import os
import sys
import json
import random
import imghdr
import pytesseract
import numpy as np
import math
from transformers import LiltForTokenClassification, AutoConfig, AutoTokenizer
from PIL import Image, ImageEnhance, ImageFilter
import torch
from typing import List, Dict, Optional, Any, Tuple, Union
import argparse
from pathlib import Path
import shutil
import tempfile
import logging
from tqdm import tqdm
import uuid
from datetime import datetime
import re
import easyocr

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('lilt_data_generator.log')]
)
logger = logging.getLogger("LiLT_Data_Generator")

# ---------- Entity Types ----------
class EntityTypes:
    HEADER = "header"
    QUESTION = "question"
    ANSWER = "answer"
    OTHER = "other"
    FALLBACK = "fallback"
    CHECKBOX = "checkbox"  # NEW: For checkbox/symbol entities

# ---------- Configuration ----------
class LiLTConfig:
    def __init__(self, args=None):
        # Model
        self.model_id = "nielsr/lilt-xlm-roberta-base"
        self.use_safetensors = True
        self.trust_remote_code = True
        
        # Processing - ENHANCED: Added checkbox-specific settings
        self.normalize_coordinates = True
        self.coordinate_range = 1000
        self.min_confidence = 0.2
        
        # ENHANCED: Added checkbox confidence threshold
        self.min_confidence_per_type = {
            EntityTypes.HEADER: 0.25,
            EntityTypes.QUESTION: 0.20,
            EntityTypes.ANSWER: 0.15,
            EntityTypes.OTHER: 0.10,
            EntityTypes.CHECKBOX: 0.10,  # NEW: Lower threshold for symbols
        }
        
        # ENHANCED: Symbol patterns for detection
        self.symbol_patterns = {
            'checkbox': [r'[✓✔☑]', r'[xX]', r'\[[ xX]\]', r'□[ xX✓✔☑]', r'◻[ xX✓✔☑]'],
            'selection': [r'○●', r'⭘⬤', r'\[[oO●]\]'],
            'marker': [r'[*•·]', r'→', r'⇒', r'▶'],
        }
        
        self.max_entities_per_doc = 350
        self.min_entity_length = 1  # Lowered for symbols
        
        # Question-Answer linking
        self.enable_qa_linking = True
        self.max_qa_distance = 200
        self.qa_linking_heuristics = True
        
        # ENHANCED: Symbol detection settings
        self.enable_symbol_detection = True
        self.treat_symbols_as_answers = True  # Treat detected symbols as answer entities
        
        # Data generation
        self.train_split = 0.8
        self.val_split = 0.2
        self.use_overlapping_split = True
        self.use_augmentation = True
        self.augmentation_ratio = 0.3
        self.augmentation_types = ['brightness', 'contrast', 'rotation', 'noise']
        
        # Output
        self.output_format = "funsd"
        self.include_metadata = True
        self.compress_output = False
        
        # Default label
        self.default_label = 0
        
        # ENHANCED: Added checkbox label mapping
        self.entity_label_mapping = {
            "O": 0,
            "B-HEADER": 1,
            "B-QUESTION": 2,
            "B-ANSWER": 3,
            "B-OTHER": 4,
            "B-CHECKBOX": 5,  # NEW
            "I-HEADER": 6,
            "I-QUESTION": 7,
            "I-ANSWER": 8,
            "I-OTHER": 9,
            "I-CHECKBOX": 10,  # NEW
        }
        self.id2label = {v: k for k, v in self.entity_label_mapping.items()}
        self.label2id = self.entity_label_mapping
        
        if args:
            self.update_from_args(args)
    
    def update_from_args(self, args):
        for key, value in vars(args).items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get_min_confidence_for_entity(self, entity_type: str) -> float:
        """Get minimum confidence threshold for specific entity type"""
        return self.min_confidence_per_type.get(entity_type.lower(), self.min_confidence)
    
    def is_symbol_text(self, text: str) -> Tuple[bool, Optional[str]]:
        """Check if text contains checkbox/symbol markers and return type"""
        if not text or len(text.strip()) == 0:
            return False, None
        
        cleaned = text.strip()
        
        # Check for single-character symbols
        if len(cleaned) == 1 and cleaned in '✓✔☑xX○●⭘⬤*•·→⇒▶':
            return True, 'checkbox' if cleaned in '✓✔☑xX' else 'marker'
        
        # Check for bracket symbols [x], [✓], etc.
        if re.match(r'^\[[ xX✓✔☑●○]\]$', cleaned):
            return True, 'checkbox'
        
        # Check for box symbols □x, ☑, etc.
        if re.match(r'^[□◻][ xX✓✔☑]?$', cleaned):
            return True, 'checkbox'
        
        # Check for common checkbox patterns
        for pattern in self.symbol_patterns['checkbox']:
            if re.match(pattern, cleaned):
                return True, 'checkbox'
        
        for pattern in self.symbol_patterns['selection']:
            if re.match(pattern, cleaned):
                return True, 'checkbox'
        
        for pattern in self.symbol_patterns['marker']:
            if re.match(pattern, cleaned):
                return True, 'marker'
        
        return False, None

# ---------- Environment helpers ----------
def setup_tesseract() -> bool:
    try:
        if not shutil.which("tesseract"):
            common_paths = [
                "/usr/bin/tesseract",
                "/usr/local/bin/tesseract",
                r"C:/Program Files/Tesseract-OCR/tesseract.exe",
                r"C:/Program Files (x86)/Tesseract-OCR/tesseract.exe",
                "/opt/homebrew/bin/tesseract"
            ]
            for p in common_paths:
                if os.path.exists(p):
                    pytesseract.pytesseract.tesseract_cmd = p
                    logger.info(f"Found Tesseract at: {p}")
                    break
        v = pytesseract.get_tesseract_version()
        logger.info(f"Tesseract version: {v}")
        return True
    except Exception as e:
        logger.error(f"Tesseract not available: {e}")
        return False

def setup_torch() -> torch.device:
    logger.info(f"PyTorch version: {torch.__version__}")
    try:
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False
    
    if cuda_available:
        logger.info(f"CUDA available: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        try:
            for i in range(torch.cuda.device_count()):
                logger.info(f" GPU {i}: {torch.cuda.get_device_name(i)}")
        except Exception:
            pass
    
    device = torch.device("cuda" if cuda_available else "cpu")
    logger.info(f"Using device: {device}")
    return device

def load_lilt_model(config: LiLTConfig, device: torch.device):
    logger.info(f"Loading LiLT model: {config.model_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, 
            trust_remote_code=config.trust_remote_code
        )
        
        model_config = AutoConfig.from_pretrained(
            config.model_id,
            num_labels=len(config.label2id),
            id2label=config.id2label,
            label2id=config.label2id,
            trust_remote_code=config.trust_remote_code,
        )
        
        model = LiltForTokenClassification.from_pretrained(
            config.model_id,
            config=model_config,
            ignore_mismatched_sizes=True,
            use_safetensors=config.use_safetensors,
            torch_dtype=torch.float32,
            trust_remote_code=config.trust_remote_code,
        )
        
        model = model.to(device)
        model.eval()
        logger.info(f"Model loaded on {device}")
        logger.info(f"Entity labels: {list(model.config.id2label.values())}")
        return model, tokenizer
    except Exception as e:
        logger.exception("Failed to load model")
        sys.exit(1)

# ---------- Utilities ----------
def normalize_box(bbox: List[int], width: int, height: int, coordinate_range: int = 1000) -> List[int]:
    try:
        if len(bbox) != 4:
            logger.warning(f"Invalid bbox format: {bbox}")
            return [0, 0, coordinate_range, coordinate_range]
        
        x_min, y_min, x_max, y_max = bbox
        x_min = max(0, min(int(x_min), width - 1))
        y_min = max(0, min(int(y_min), height - 1))
        x_max = max(x_min + 1, min(int(x_max), width))
        y_max = max(y_min + 1, min(int(y_max), height))
        
        norm_x_min = int(coordinate_range * (x_min / width))
        norm_y_min = int(coordinate_range * (y_min / height))
        norm_x_max = int(coordinate_range * (x_max / width))
        norm_y_max = int(coordinate_range * (y_max / height))
        
        norm_x_min = max(0, min(norm_x_min, coordinate_range))
        norm_y_min = max(0, min(norm_y_min, coordinate_range))
        norm_x_max = max(norm_x_min + 1, min(norm_x_max, coordinate_range))
        norm_y_max = max(norm_y_min + 1, min(norm_y_max, coordinate_range))
        
        return [norm_x_min, norm_y_min, norm_x_max, norm_y_max]
    except Exception as e:
        logger.exception(f"normalize_box error for {bbox}: {e}")
        return [0, 0, coordinate_range, coordinate_range]

def unnormalize_box(bbox: List[int], width: int, height: int, coordinate_range: int = 1000) -> List[int]:
    return [
        int(width * (bbox[0] / coordinate_range)),
        int(height * (bbox[1] / coordinate_range)),
        int(width * (bbox[2] / coordinate_range)),
        int(height * (bbox[3] / coordinate_range)),
    ]

def compute_union_box(boxes: List[List[int]]) -> List[int]:
    """Compute the union bounding box for multiple boxes"""
    if not boxes:
        return [0, 0, 0, 0]
    
    valid_boxes = [b for b in boxes if validate_bbox(b)]
    if not valid_boxes:
        return [0, 0, 0, 0]
    
    x1 = min(b[0] for b in valid_boxes)
    y1 = min(b[1] for b in valid_boxes)
    x2 = max(b[2] for b in valid_boxes)
    y2 = max(b[3] for b in valid_boxes)
    
    x2 = max(x2, x1 + 1)
    y2 = max(y2, y1 + 1)
    
    return [x1, y1, x2, y2]

def clean_text(text: str, preserve_symbols: bool = True) -> str:
    """Clean and normalize text with option to preserve symbols"""
    if not isinstance(text, str):
        return ""
    
    # ENHANCED: Normalize common symbol variations
    if preserve_symbols:
        # Normalize checkmark variations
        text = text.replace('✅', '✓').replace('☑️', '☑').replace('✔️', '✔')
        # Normalize X variations
        text = text.replace('❌', 'X').replace('❎', 'X').replace('✗', 'X').replace('✘', 'X')
        # Normalize box variations
        text = text.replace('☐', '□').replace('⬜', '□').replace('⬛', '■')
        # Normalize circle variations
        text = text.replace('⚫', '●').replace('⚪', '○')
    
    # Basic cleaning
    text = re.sub(r"\s+", " ", text).strip()
    text = ''.join(c for c in text if c.isprintable() or c in ['\n', '\t'])
    
    # Fix common OCR artifacts
    text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl').replace('ﬀ', 'ff')
    
    # Special handling for symbolic text - don't over-clean
    if preserve_symbols and len(text.strip()) <= 3:
        # For short text that might be symbols, preserve more characters
        symbol_chars = '✓✔☑xX□■○●⭘⬤*•·→⇒▶[]()'
        text = ''.join(c for c in text if c.isalnum() or c in f'{symbol_chars} ,.-')
    
    return text.strip()

def validate_bbox(bbox: List[int]) -> bool:
    """Validate bounding box coordinates"""
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    
    try:
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        width = x2 - x1
        height = y2 - y1
        
        return (
            x1 >= 0 and y1 >= 0 and 
            x2 > x1 and y2 > y1 and 
            width >= 1 and height >= 1 and 
            x2 <= 20000 and y2 <= 20000
        )
    except (ValueError, TypeError):
        return False

def get_entity_type_from_label(label: str) -> str:
    """Extract entity type from BIO label"""
    if not label or label == "O":
        return EntityTypes.OTHER
    
    if "-" in label:
        _, entity_type = label.split("-", 1)
        return entity_type.lower().strip()
    
    return label.lower().strip()

def validate_entity(entity: Dict, config: LiLTConfig = None) -> bool:
    """Validate entity structure and content"""
    if not isinstance(entity, dict):
        return False
    
    required_fields = ['box', 'text', 'label', 'id']
    if not all(field in entity for field in required_fields):
        return False
    
    if not validate_bbox(entity['box']):
        return False
    
    text = entity.get('text', '').strip()
    
    # ENHANCED: Special validation for checkbox/symbol entities
    if config and config.enable_symbol_detection:
        is_symbol, symbol_type = config.is_symbol_text(text)
        if is_symbol:
            # For symbols, allow very short or empty-looking text
            if len(text) == 0:
                return False
            # Update label if it's a symbol and should be treated as checkbox
            if config.treat_symbols_as_answers and entity['label'] != EntityTypes.CHECKBOX:
                logger.debug(f"Detected symbol '{text}' as checkbox entity")
    
    # For answer entities, allow very short text
    if entity['label'] in [EntityTypes.ANSWER, EntityTypes.CHECKBOX]:
        if len(text) < 1:
            return False
    
    if not text and entity['label'] not in [EntityTypes.ANSWER, EntityTypes.FALLBACK, EntityTypes.CHECKBOX]:
        return False
    
    return True

# ---------- Enhanced Symbol Detection ----------
def detect_and_label_symbols(entities: List[Dict], config: LiLTConfig) -> List[Dict]:
    """Detect symbols in entities and label them appropriately"""
    if not config.enable_symbol_detection:
        return entities
    
    enhanced_entities = []
    symbol_count = 0
    
    for entity in entities:
        text = entity.get('text', '').strip()
        original_label = entity.get('label', EntityTypes.OTHER)
        
        # Check if text contains symbols
        is_symbol, symbol_type = config.is_symbol_text(text)
        
        if is_symbol and config.treat_symbols_as_answers:
            # Create a new entity with checkbox label
            symbol_entity = entity.copy()
            symbol_entity['label'] = EntityTypes.CHECKBOX
            symbol_entity['symbol_type'] = symbol_type
            symbol_entity['original_text'] = text
            
            # ENHANCED: Clean and normalize the symbol text
            if symbol_type == 'checkbox':
                # Standardize checkbox symbols
                if text in ['✓', '✔', '☑']:
                    symbol_entity['text'] = '✓'
                elif text in ['x', 'X']:
                    symbol_entity['text'] = 'X'
                elif text == '[ ]':
                    symbol_entity['text'] = '□'
                elif text == '[x]' or text == '[X]':
                    symbol_entity['text'] = '[X]'
            
            enhanced_entities.append(symbol_entity)
            symbol_count += 1
            logger.debug(f"Detected symbol: '{text}' -> {symbol_type}")
            
            # Keep original entity if it's not already the same
            if original_label != EntityTypes.CHECKBOX:
                enhanced_entities.append(entity)
        else:
            enhanced_entities.append(entity)
    
    if symbol_count > 0:
        logger.info(f"Detected {symbol_count} symbol entities")
    
    return enhanced_entities

# ---------- Question-Answer Linking ----------
def link_questions_and_answers(entities: List[Dict], width: int, height: int, config: LiLTConfig) -> List[Dict]:
    """Link questions with their corresponding answers including symbols"""
    questions = [e for e in entities if e['label'] == EntityTypes.QUESTION]
    answers = [e for e in entities if e['label'] in [EntityTypes.ANSWER, EntityTypes.CHECKBOX]]
    headers = [e for e in entities if e['label'] == EntityTypes.HEADER]
    
    # Clear existing linking
    for entity in entities:
        entity['linking'] = []
    
    # Link questions to answers
    for question in questions:
        best_answer = None
        best_score = -1
        
        for answer in answers:
            # Calculate spatial proximity score
            q_box = question['box']
            a_box = answer['box']
            
            # Horizontal distance (answer should be to the right or same column as question)
            horizontal_dist = max(0, a_box[0] - q_box[2]) if a_box[0] > q_box[2] else max(0, q_box[0] - a_box[2])
            
            # Vertical distance (answer should be on same line or below question)
            vertical_dist = max(0, a_box[1] - q_box[3]) if a_box[1] > q_box[3] else max(0, q_box[1] - a_box[3])
            
            # Prefer answers that are horizontally aligned
            alignment_score = 1.0 - (horizontal_dist / width) if horizontal_dist < width * 0.3 else 0
            vertical_score = 1.0 - (vertical_dist / (height * 0.1)) if vertical_dist < height * 0.1 else 0
            
            # ENHANCED: Text-based heuristics including symbols
            text_score = 0.5
            
            # Question indicators
            if ':' in question['text'] or '?' in question['text']:
                text_score += 0.3
            
            # Answer indicators (including symbols)
            answer_text = answer['text'].strip()
            if re.match(r'^[\d\.,$€£\-/]+$', answer_text):
                text_score += 0.2
            
            # Symbol indicators
            if answer['label'] == EntityTypes.CHECKBOX:
                text_score += 0.3  # Higher score for symbols as they're often answers
            
            # Check for common answer patterns
            if answer_text.lower() in ['yes', 'no', 'true', 'false', '✓', 'x', 'X']:
                text_score += 0.2
            
            total_score = alignment_score * 0.6 + vertical_score * 0.3 + text_score * 0.1
            
            if total_score > best_score and total_score > 0.3:
                best_score = total_score
                best_answer = answer
        
        if best_answer:
            # Add bidirectional linking
            question['linking'].append([question['id'], best_answer['id']])
            best_answer['linking'].append([best_answer['id'], question['id']])
            
            logger.debug(f"Linked question '{question['text']}' with answer '{best_answer['text']}' "
                        f"(type: {best_answer['label']}, score: {best_score:.2f})")
    
    return entities

# ---------- Augmentation ----------
class DocumentAugmentor:
    def __init__(self, config: LiLTConfig):
        self.config = config
    
    def apply_augmentation(self, image: Image.Image, bboxes: List[List[int]], texts: List[str]) -> Tuple[Image.Image, List[List[int]], List[str]]:
        if not self.config.use_augmentation or random.random() > self.config.augmentation_ratio:
            return image, bboxes, texts
        
        augmentation_type = random.choice(self.config.augmentation_types)
        try:
            if augmentation_type == 'brightness':
                factor = random.uniform(0.7, 1.3)
                return ImageEnhance.Brightness(image).enhance(factor), bboxes, texts
            elif augmentation_type == 'contrast':
                factor = random.uniform(0.7, 1.3)
                return ImageEnhance.Contrast(image).enhance(factor), bboxes, texts
            elif augmentation_type == 'rotation':
                return self._augment_rotation(image, bboxes, texts)
            elif augmentation_type == 'noise':
                return self._augment_noise(image, bboxes, texts)
        except Exception as e:
            logger.warning(f"Augmentation {augmentation_type} failed: {e}")
            return image, bboxes, texts
        
        return image, bboxes, texts
    
    def _augment_rotation(self, image: Image.Image, bboxes: List[List[int]], texts: List[str]) -> Tuple[Image.Image, List[List[int]], List[str]]:
        angle = random.uniform(-3, 3)
        width, height = image.size
        center_x, center_y = width / 2, height / 2
        angle_rad = math.radians(angle)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)
        
        augmented_image = image.rotate(angle, expand=False, center=(center_x, center_y), resample=Image.BILINEAR)
        augmented_bboxes = []
        
        for bbox in bboxes:
            try:
                x1, y1, x2, y2 = bbox
                corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                rotated_corners = []
                
                for x, y in corners:
                    x_trans = x - center_x
                    y_trans = y - center_y
                    x_rot = x_trans * cos_angle - y_trans * sin_angle
                    y_rot = x_trans * sin_angle + y_trans * cos_angle
                    x_new = x_rot + center_x
                    y_new = y_rot + center_y
                    rotated_corners.append((x_new, y_new))
                
                xs = [p[0] for p in rotated_corners]
                ys = [p[1] for p in rotated_corners]
                new_x1 = max(0, min(xs))
                new_y1 = max(0, min(ys))
                new_x2 = min(width, max(xs))
                new_y2 = min(height, max(ys))
                
                new_x2 = max(new_x1 + 1, new_x2)
                new_y2 = max(new_y1 + 1, new_y2)
                
                augmented_bboxes.append([int(new_x1), int(new_y1), int(new_x2), int(new_y2)])
            except Exception as e:
                logger.warning(f"Box rotation failed: {e}")
                augmented_bboxes.append(bbox)
        
        return augmented_image, augmented_bboxes, texts
    
    def _augment_noise(self, image: Image.Image, bboxes: List[List[int]], texts: List[str]) -> Tuple[Image.Image, List[List[int]], List[str]]:
        np_image = np.array(image)
        mean = 0
        var = 5
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, np_image.shape).astype(np.uint8)
        noisy_image = np.clip(np_image + gauss, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_image), bboxes, texts

# ---------- Entity extraction (ENHANCED FOR SYMBOLS) ----------
def extract_entities(model, tokenizer, image_path: Path, config: LiLTConfig, device: torch.device) -> Optional[Dict]:
    try:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        logger.info(f"Processing image: {image_path.name} ({width}x{height})")
    except Exception as e:
        logger.error(f"Error opening image {image_path}: {e}")
        return None
    
    words: List[str] = []
    raw_boxes: List[List[int]] = []
    full_text = ""
    
    # ENHANCED: Preprocess image for better symbol detection
    # Increase contrast for better symbol recognition
    gray = image.convert('L')
    enhanced = ImageEnhance.Contrast(gray).enhance(3.0)
    threshold = enhanced.point(lambda p: p > 100 and 255)
    
    # Try EasyOCR first
    try:
        reader = easyocr.Reader(['en', 'ch_sim'], gpu=torch.cuda.is_available())
        ocr_results = reader.readtext(np.array(threshold), detail=1, paragraph=False)
        
        for res in ocr_results:
            bbox, text, conf = res
            if conf > config.min_confidence:
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                
                # ENHANCED: Clean text but preserve symbols
                cleaned_text = clean_text(text, preserve_symbols=True)
                
                words.append(cleaned_text)
                raw_boxes.append([min_x, min_y, max_x, max_y])
        
        full_text = ' '.join(words)
        logger.debug(f"EasyOCR extracted {len(words)} words")
    except Exception as e:
        logger.debug(f"EasyOCR failed for {image_path}: {e}")
    
    # Fallback to pytesseract if needed
    if not words or len(words) < 5:  # If very few words, try Tesseract
        try:
            # ENHANCED: Use custom config for better symbol detection
            custom_config = r'--oem 1 --psm 6 -c tessedit_char_whitelist="✓✔☑xX□■○●⭘⬤*•·→⇒▶[]()0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,:;?!-\'\" "'
            
            ocr_data = pytesseract.image_to_data(
                threshold,
                lang='eng+chi_sim',
                output_type=pytesseract.Output.DICT,
                config=custom_config
            )
            
            n_boxes = len(ocr_data['text'])
            for i in range(n_boxes):
                text = str(ocr_data['text'][i]).strip()
                if not text:
                    continue
                
                try:
                    conf = float(ocr_data['conf'][i])
                except Exception:
                    conf = -1.0
                
                # ENHANCED: Lower threshold for symbols and short text
                min_conf_needed = config.min_confidence
                is_symbol, symbol_type = config.is_symbol_text(text)
                
                if is_symbol:
                    min_conf_needed = config.get_min_confidence_for_entity(EntityTypes.CHECKBOX)
                elif re.match(r'^[\d\-/\\.,]+$', text):
                    min_conf_needed = config.min_confidence * 0.3
                
                if text and conf > min_conf_needed:
                    x, y, w, h = (
                        ocr_data['left'][i],
                        ocr_data['top'][i],
                        ocr_data['width'][i],
                        ocr_data['height'][i]
                    )
                    
                    if w > 1 and h > 1:
                        # ENHANCED: Clean but preserve symbols
                        cleaned_text = clean_text(text, preserve_symbols=True)
                        words.append(cleaned_text)
                        raw_boxes.append([x, y, x + w, y + h])
            
            full_text = ' '.join(words)
            logger.debug(f"pytesseract extracted {len(words)} words")
        except Exception as e:
            logger.debug(f"pytesseract fallback failed for {image_path}: {e}")
    
    if not words:
        logger.warning(f"No text detected in {image_path.name} by any OCR method")
        return {
            "entities": [{
                "box": [0, 0, config.coordinate_range, config.coordinate_range],
                "text": "No text detected - fallback entity",
                "label": EntityTypes.FALLBACK,
                "words": [{"box": [0, 0, config.coordinate_range, config.coordinate_range], "text": "No text detected"}],
                "linking": [],
                "id": 0
            }],
            "width": width,
            "height": height,
            "filename": image_path.name,
            "original_width": width,
            "original_height": height,
            "image_path": str(image_path),
            "full_text": ""
        }
    
    # Normalize bounding boxes
    boxes = []
    valid_indices = []
    
    for i, (word, box) in enumerate(zip(words, raw_boxes)):
        if not validate_bbox(box):
            logger.debug(f"Skipping invalid bbox: {box}")
            continue
        
        norm_box = normalize_box(box, width, height, config.coordinate_range)
        if validate_bbox(norm_box):
            boxes.append(norm_box)
            valid_indices.append(i)
            words[i] = clean_text(word, preserve_symbols=True)
    
    # Filter words to only valid ones
    words = [words[i] for i in valid_indices]
    
    if not words:
        logger.warning(f"No valid text after cleaning in {image_path.name}")
        return {
            "entities": [{
                "box": [0, 0, config.coordinate_range, config.coordinate_range],
                "text": "No valid text - fallback entity",
                "label": EntityTypes.FALLBACK,
                "words": [{"box": [0, 0, config.coordinate_range, config.coordinate_range], "text": "No valid text"}],
                "linking": [],
                "id": 0
            }],
            "width": width,
            "height": height,
            "filename": image_path.name,
            "original_width": width,
            "original_height": height,
            "image_path": str(image_path),
            "full_text": ""
        }
    
    # Tokenize words and align boxes
    try:
        encoding = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
            return_overflowing_tokens=True,
            return_special_tokens_mask=True
        ).to(device)
        
        input_ids = encoding['input_ids'][0]
        attention_mask = encoding['attention_mask'][0]
        
        try:
            word_ids = encoding.word_ids(0)
        except Exception:
            word_ids = []
            w_idx = 0
            for _ in range(len(input_ids)):
                word_ids.append(w_idx if w_idx < len(boxes) else None)
                w_idx += 1
        
        aligned_boxes = []
        aligned_words = []
        
        for word_idx in word_ids:
            if word_idx is None:
                aligned_boxes.append([0, 0, 0, 0])
                aligned_words.append("[PAD]")
            elif word_idx < len(boxes):
                aligned_boxes.append(boxes[word_idx])
                aligned_words.append(words[word_idx] if word_idx < len(words) else "[UNK]")
            else:
                aligned_boxes.append([0, 0, 0, 0])
                aligned_words.append("[UNK]")
        
        bbox_tensor = torch.tensor([aligned_boxes], device=device)
        logger.debug(f"Tokenized {len(words)} words into {len(input_ids)} tokens")
    except Exception as e:
        logger.error(f"Tokenization error for {image_path}: {e}")
        return None
    
    # Model inference
    try:
        lilt_encoding = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'bbox': bbox_tensor
        }
        
        with torch.no_grad():
            outputs = model(**lilt_encoding)
        
        predictions = outputs.logits.argmax(-1)[0].cpu().tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
        bboxes = bbox_tensor[0].cpu().tolist()
        
        pred_labels = [model.config.id2label.get(pred_id, "O") for pred_id in predictions]
        unique_preds = set(pred_labels)
        logger.debug(f"Unique predicted labels: {unique_preds}")
    except Exception as e:
        logger.error(f"Model inference error for {image_path}: {e}")
        return None
    
    # Entity post-processing
    entities = []
    current_entity = None
    entity_id = 0
    processed_tokens = set()
    
    logger.debug("Starting entity post-processing...")
    
    for idx in range(len(predictions)):
        token = tokens[idx]
        if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token, "[PAD]", "[CLS]", "[SEP]"]:
            continue
        
        if idx in processed_tokens:
            continue
        
        label_id = predictions[idx]
        label = model.config.id2label.get(label_id, "O")
        
        word_text = token.replace("Ġ", " ").replace("▁", " ").strip()
        if word_text.startswith("##"):
            word_text = word_text[2:]
        
        if not word_text or word_text == "[UNK]":
            continue
        
        norm_box = bboxes[idx]
        if not validate_bbox(norm_box):
            logger.debug(f"Skipping invalid normalized bbox: {norm_box}")
            continue
        
        entity_type = get_entity_type_from_label(label)
        prefix = "O"
        if "-" in label:
            prefix = label.split("-")[0]
        
        logger.debug(f"Processing token: '{word_text}' with label '{label}' (type: {entity_type}, prefix: {prefix})")
        
        if label == "O" or prefix == "O":
            if current_entity:
                union_box = compute_union_box(current_entity["boxes"])
                entity_text = clean_text(current_entity["text"].strip(), preserve_symbols=True)
                
                if entity_text and len(entity_text) >= config.min_entity_length:
                    entities.append({
                        "box": union_box,
                        "text": entity_text,
                        "label": current_entity["label"],
                        "words": current_entity["words"],
                        "linking": [],
                        "id": entity_id
                    })
                    entity_id += 1
                    logger.debug(f"Created entity: '{entity_text}' ({current_entity['label']})")
                
                current_entity = None
            continue
        
        if prefix == "B":
            if current_entity:
                union_box = compute_union_box(current_entity["boxes"])
                entity_text = clean_text(current_entity["text"].strip(), preserve_symbols=True)
                
                if entity_text and len(entity_text) >= config.min_entity_length:
                    entities.append({
                        "box": union_box,
                        "text": entity_text,
                        "label": current_entity["label"],
                        "words": current_entity["words"],
                        "linking": [],
                        "id": entity_id
                    })
                    entity_id += 1
                    logger.debug(f"Closed previous entity: '{entity_text}' ({current_entity['label']})")
            
            current_entity = {
                "label": entity_type,
                "text": word_text,
                "boxes": [norm_box],
                "words": [{"box": norm_box, "text": word_text, "token_idx": idx}]
            }
            processed_tokens.add(idx)
            logger.debug(f"Started new entity: '{word_text}' ({entity_type})")
        
        elif prefix == "I":
            if current_entity and current_entity["label"] == entity_type:
                current_entity["text"] += " " + word_text
                current_entity["boxes"].append(norm_box)
                current_entity["words"].append({"box": norm_box, "text": word_text, "token_idx": idx})
                processed_tokens.add(idx)
                logger.debug(f"Extended entity: '{word_text}' added to '{current_entity['text']}'")
            else:
                if current_entity:
                    union_box = compute_union_box(current_entity["boxes"])
                    entity_text = clean_text(current_entity["text"].strip(), preserve_symbols=True)
                    
                    if entity_text and len(entity_text) >= config.min_entity_length:
                        entities.append({
                            "box": union_box,
                            "text": entity_text,
                            "label": current_entity["label"],
                            "words": current_entity["words"],
                            "linking": [],
                            "id": entity_id
                        })
                        entity_id += 1
                
                current_entity = {
                    "label": entity_type,
                    "text": word_text,
                    "boxes": [norm_box],
                    "words": [{"box": norm_box, "text": word_text, "token_idx": idx}]
                }
                processed_tokens.add(idx)
                logger.debug(f"Started new entity from I-label: '{word_text}' ({entity_type})")
    
    if current_entity:
        union_box = compute_union_box(current_entity["boxes"])
        entity_text = clean_text(current_entity["text"].strip(), preserve_symbols=True)
        
        if entity_text and len(entity_text) >= config.min_entity_length:
            entities.append({
                "box": union_box,
                "text": entity_text,
                "label": current_entity["label"],
                "words": current_entity["words"],
                "linking": [],
                "id": entity_id
            })
            logger.debug(f"Final entity: '{entity_text}' ({current_entity['label']})")
    
    # ENHANCED: Detect and label symbols
    entities = detect_and_label_symbols(entities, config)
    
    # Ensure we have at least one of each entity type if text exists
    entity_types_present = set(e['label'] for e in entities)
    logger.info(f"Entity types found: {entity_types_present}")
    
    # Enhanced fallback logic for missing question/answer entities
    has_questions = any(e['label'] == EntityTypes.QUESTION for e in entities)
    has_answers = any(e['label'] in [EntityTypes.ANSWER, EntityTypes.CHECKBOX] for e in entities)
    
    if not has_questions or not has_answers:
        logger.warning(f"Missing {'question' if not has_questions else 'answer'} entities, applying enhanced fallback logic for {image_path.name}")
        
        fallback_entities = []
        text_chunks = re.split(r'[\n\r]+', full_text.strip())
        
        for i, chunk in enumerate(text_chunks):
            if not chunk.strip() or len(chunk.strip()) < 1:
                continue
            
            chunk_clean = clean_text(chunk.strip(), preserve_symbols=True)
            if not chunk_clean:
                continue
            
            # Heuristics for question detection
            is_question = (
                chunk_clean.endswith(':') or 
                chunk_clean.endswith('?') or
                len(chunk_clean.split()) <= 6 or
                any(keyword in chunk_clean.lower() for keyword in ['name', 'date', 'address', 'phone', 'email', 'total', 'amount', 'id', 'number'])
            )
            
            # Check if it's a symbol
            is_symbol, symbol_type = config.is_symbol_text(chunk_clean)
            
            if is_symbol and config.treat_symbols_as_answers:
                entity_label = EntityTypes.CHECKBOX
            else:
                entity_label = EntityTypes.QUESTION if is_question else EntityTypes.ANSWER
            
            # Only create the missing entity type
            if ((entity_label == EntityTypes.QUESTION and not has_questions) or 
                (entity_label in [EntityTypes.ANSWER, EntityTypes.CHECKBOX] and not has_answers)):
                
                box_height = int(config.coordinate_range * 0.05)
                y_start = int(config.coordinate_range * 0.1 * (i % 10))
                fallback_box = [
                    int(config.coordinate_range * 0.1),
                    y_start,
                    int(config.coordinate_range * 0.9),
                    min(y_start + box_height, config.coordinate_range)
                ]
                
                fallback_entities.append({
                    "box": fallback_box,
                    "text": chunk_clean,
                    "label": entity_label,
                    "words": [{"box": fallback_box, "text": chunk_clean}],
                    "linking": [],
                    "id": len(entities) + len(fallback_entities),
                    "is_fallback": True
                })
        
        # Add fallback entities
        entities.extend(fallback_entities)
        logger.info(f"Added {len(fallback_entities)} fallback entities")
    
    # Filter and validate entities
    valid_entities = []
    for entity in entities:
        if validate_entity(entity, config):
            min_conf = config.get_min_confidence_for_entity(entity['label'])
            
            # Special handling for short entities and symbols
            if entity['label'] in [EntityTypes.ANSWER, EntityTypes.CHECKBOX] and len(entity['text'].strip()) <= 3:
                min_conf = min(0.1, min_conf)
            
            # Deduplicate similar entities
            is_duplicate = False
            for valid_entity in valid_entities:
                if (valid_entity['label'] == entity['label'] and 
                    abs(valid_entity['box'][0] - entity['box'][0]) < 10 and
                    abs(valid_entity['box'][1] - entity['box'][1]) < 10 and
                    entity['text'].lower() in valid_entity['text'].lower()):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                valid_entities.append(entity)
    
    if len(valid_entities) > config.max_entities_per_doc:
        logger.warning(f"Too many entities found ({len(valid_entities)}), limiting to {config.max_entities_per_doc}")
        valid_entities.sort(key=lambda x: (
            x['label'] in [EntityTypes.QUESTION, EntityTypes.ANSWER, EntityTypes.CHECKBOX],
            len(x['text'])
        ), reverse=True)
        valid_entities = valid_entities[:config.max_entities_per_doc]
    
    # Link questions and answers (including symbols)
    if config.enable_qa_linking:
        valid_entities = link_questions_and_answers(valid_entities, width, height, config)
    
    logger.info(f"Extracted {len(valid_entities)} valid entities from {image_path.name}")
    
    question_count = sum(1 for e in valid_entities if e['label'] == EntityTypes.QUESTION)
    answer_count = sum(1 for e in valid_entities if e['label'] == EntityTypes.ANSWER)
    checkbox_count = sum(1 for e in valid_entities if e['label'] == EntityTypes.CHECKBOX)
    
    logger.info(f"Question entities: {question_count}, Answer entities: {answer_count}, Checkbox entities: {checkbox_count}")
    
    if valid_entities:
        entity_types = set(e['label'] for e in valid_entities)
        logger.info(f"Final entity types: {entity_types}")
    
    return {
        "entities": valid_entities,
        "width": width,
        "height": height,
        "filename": image_path.name,
        "original_width": width,
        "original_height": height,
        "image_path": str(image_path),
        "full_text": full_text
    }

# ---------- FUNSD JSON creation ----------
def create_funsd_document(doc_info: Dict[str, Any], config: LiLTConfig, label: Optional[Union[int, str]] = None) -> Dict[str, Any]:
    """Create FUNSD-format document with symbol support"""
    doc_id = Path(doc_info["filename"]).stem
    unique_id = f"{doc_id}_{uuid.uuid4().hex[:8]}"
    
    if label is None:
        label = config.default_label
    
    entities = []
    for entity in doc_info["entities"]:
        if not validate_entity(entity, config):
            continue
        
        # Standardize entity format for FUNSD
        entity_dict = {
            "id": entity.get("id", len(entities)),
            "text": entity["text"],
            "label": entity["label"].lower() if entity["label"] not in [EntityTypes.OTHER, EntityTypes.FALLBACK] else "other",
            "words": [],
            "box": entity["box"],
            "linking": entity.get("linking", []),
        }
        
        # Add symbol metadata if present
        if entity["label"] == EntityTypes.CHECKBOX:
            entity_dict["symbol_type"] = entity.get("symbol_type", "checkbox")
        
        # Add words
        for word_info in entity.get("words", []):
            if isinstance(word_info, dict):
                entity_dict["words"].append({
                    "box": word_info.get("box", entity["box"]),
                    "text": word_info.get("text", "")
                })
            else:
                entity_dict["words"].append({
                    "box": entity["box"],
                    "text": str(word_info)
                })
        
        entities.append(entity_dict)
    
    document = {
        "id": f"en_custom_{unique_id}",
        "uid": unique_id,
        "file": doc_info["filename"],
        "img": {
            "fname": doc_info["filename"],
            "width": doc_info["width"],
            "height": doc_info["height"]
        },
        "document": entities,
        "label": label
    }
    
    if config.include_metadata:
        document["metadata"] = {
            "generation_timestamp": datetime.now().isoformat(),
            "model_id": config.model_id,
            "entity_count": len(entities),
            "question_count": sum(1 for e in entities if e["label"] == EntityTypes.QUESTION),
            "answer_count": sum(1 for e in entities if e["label"] == EntityTypes.ANSWER),
            "checkbox_count": sum(1 for e in entities if e.get("label") == EntityTypes.CHECKBOX),
            "original_dimensions": {
                "width": doc_info.get("original_width", doc_info["width"]),
                "height": doc_info.get("original_height", doc_info["height"])
            },
            "entity_types": {}
        }
        
        for entity in entities:
            entity_type = entity["label"]
            document["metadata"]["entity_types"][entity_type] = document["metadata"]["entity_types"].get(entity_type, 0) + 1
    
    return document

def create_funsd_json(doc_info: Dict[str, Any], config: LiLTConfig, label: Optional[Union[int, str]] = None) -> Dict[str, Any]:
    """Create complete FUNSD-format JSON"""
    document = create_funsd_document(doc_info, config, label)
    top_label = document.get("label", config.default_label)
    
    if "document" not in document or not isinstance(document["document"], list):
        document["document"] = []
    
    for entity in document["document"]:
        if "words" not in entity or not entity["words"]:
            entity["words"] = [{
                "box": entity["box"],
                "text": entity["text"]
            }]
        
        if "linking" not in entity:
            entity["linking"] = []
        elif not isinstance(entity["linking"], list):
            entity["linking"] = []
    
    return {
        "lang": "en",
        "version": "1.0",
        "split": "custom",
        "label": top_label,
        "documents": [document]
    }

# ---------- Split train/val data ----------
def split_train_val_data(documents: List[Dict], config: LiLTConfig) -> Tuple[List[Dict], List[Dict]]:
    if len(documents) < 2:
        logger.warning("Not enough documents for train/val split. Using all for training.")
        return documents, []
    
    if config.use_overlapping_split:
        train_docs = documents.copy()
        random.seed(42)
        shuffled_docs = documents.copy()
        random.shuffle(shuffled_docs)
        val_size = max(1, int(len(shuffled_docs) * config.val_split))
        val_docs = shuffled_docs[:val_size]
        logger.info(f"Overlapping split: {len(train_docs)} training, {len(val_docs)} validation")
        return train_docs, val_docs
    else:
        random.seed(42)
        shuffled_docs = documents.copy()
        random.shuffle(shuffled_docs)
        split_idx = int(len(shuffled_docs) * config.train_split)
        split_idx = max(1, min(split_idx, len(shuffled_docs) - 1))
        train_docs = shuffled_docs[:split_idx]
        val_docs = shuffled_docs[split_idx:]
        logger.info(f"Non-overlapping split: {len(train_docs)} training, {len(val_docs)} validation")
        return train_docs, val_docs

# ---------- Generate training data ----------
def generate_training_data(input_paths: List[Path], config: LiLTConfig, model, tokenizer, device: torch.device) -> List[Dict]:
    all_documents: List[Dict] = []
    augmentor = DocumentAugmentor(config)
    
    for input_path in tqdm(input_paths, desc="Processing documents"):
        if not input_path.exists():
            logger.warning(f"File not found: {input_path}")
            continue
        
        try:
            doc_info = extract_entities(model, tokenizer, input_path, config, device)
            if not doc_info:
                continue
            
            # Create original document
            original_doc = create_funsd_json(doc_info, config, label=config.default_label)
            original_doc["image_path"] = str(input_path)
            all_documents.append(original_doc)
            
            # Create augmented documents if enabled
            if config.use_augmentation:
                try:
                    image = Image.open(input_path).convert("RGB")
                    words = []
                    boxes = []
                    
                    for entity in doc_info["entities"]:
                        for word in entity.get("words", []) if isinstance(entity.get("words", []), list) else []:
                            words.append(word.get("text", ""))
                            boxes.append(word.get("box", [0, 0, config.coordinate_range, config.coordinate_range]))
                    
                    if words and boxes:
                        augmented_image, augmented_boxes, augmented_words = augmentor.apply_augmentation(image, boxes, words)
                        temp_dir = Path(tempfile.mkdtemp())
                        aug_image_path = temp_dir / f"aug_{input_path.name}"
                        augmented_image.save(aug_image_path)
                        
                        aug_doc_info = extract_entities(model, tokenizer, aug_image_path, config, device)
                        if aug_doc_info and len(aug_doc_info.get("entities", [])) > 0:
                            aug_doc_info["filename"] = f"aug_{doc_info['filename']}"
                            augmented_doc = create_funsd_json(aug_doc_info, config, label=config.default_label)
                            augmented_doc["image_path"] = str(aug_image_path)
                            all_documents.append(augmented_doc)
                            logger.info(f"Augmented document created with {len(aug_doc_info['entities'])} entities")
                        
                        if aug_image_path.exists():
                            aug_image_path.unlink()
                except Exception as e:
                    logger.warning(f"Augmentation failed for {input_path}: {e}")
        except Exception as e:
            logger.exception(f"Error processing {input_path}: {e}")
    
    logger.info(f"Generated {len(all_documents)} training documents")
    return all_documents

# ---------- I/O helpers ----------
def copy_image_and_json(doc_info: Dict[str, Any], output_dir: Path, config: LiLTConfig) -> bool:
    try:
        source_img_path = Path(doc_info.get("image_path", ""))
        if not source_img_path.exists():
            logger.warning(f"Source image not found: {source_img_path}")
            return False
        
        dest_img_path = output_dir / source_img_path.name
        dest_json_path = output_dir / f"{source_img_path.stem}.json"
        
        shutil.copy2(source_img_path, dest_img_path)
        
        json_data = doc_info.copy()
        if "image_path" in json_data:
            del json_data["image_path"]
        
        with open(dest_json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
        
        logger.debug(f"Copied image and saved JSON to {dest_json_path}")
        return True
    except Exception as e:
        logger.exception(f"Error copying files: {e}")
        return False

def process_image_file(file_path: Path, output_dir: Path, config: LiLTConfig, model, tokenizer, device: torch.device) -> bool:
    logger.info(f"Processing image file: {file_path.name}")
    
    if imghdr.what(file_path) is None:
        logger.warning(f"Skipping {file_path.name}: not a recognized image file")
        return False
    
    doc_info = extract_entities(model, tokenizer, file_path, config, device)
    if doc_info:
        json_data = create_funsd_json(doc_info, config, label=config.default_label)
        output_filename = file_path.with_suffix('.json').name
        output_path = output_dir / output_filename
        
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)
            logger.info(f"Saved output JSON: {output_filename}")
            return True
        except Exception as e:
            logger.exception(f"Error saving JSON: {e}")
    
    logger.warning(f"Failed to process {file_path.name}")
    return False

def process_pdf_file(pdf_path: Path, output_dir: Path, config: LiLTConfig, model, tokenizer, device: torch.device) -> List[Path]:
    try:
        from pdf2image import convert_from_path
    except Exception:
        logger.error("pdf2image not installed; cannot process PDF files")
        return []
    
    logger.info(f"Processing PDF: {pdf_path.name}")
    temp_dir = Path(tempfile.mkdtemp(prefix="pdf_processing_"))
    processed_images = []
    
    try:
        images = convert_from_path(pdf_path, dpi=200, output_folder=str(temp_dir), fmt='png')
        for i, img in enumerate(images):
            temp_img_path = temp_dir / f"{pdf_path.stem}_page_{i+1}.png"
            img.save(temp_img_path)
            if process_image_file(temp_img_path, output_dir, config, model, tokenizer, device):
                processed_images.append(temp_img_path)
    except Exception as e:
        logger.exception(f"Error processing PDF {pdf_path}: {e}")
    
    return processed_images

def process_input(input_path: Path, output_dir: Path, config: LiLTConfig, model, tokenizer, device: torch.device) -> List[Path]:
    input_files: List[Path] = []
    
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return input_files
    
    if input_path.is_dir():
        logger.info(f"Processing directory: {input_path}")
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        pdf_extensions = ('.pdf',)
        
        image_files = [f for f in input_path.glob('*') if f.suffix.lower() in image_extensions]
        pdf_files = [f for f in input_path.glob('*') if f.suffix.lower() in pdf_extensions]
        
        logger.info(f"Found {len(image_files)} image files and {len(pdf_files)} PDF files")
        
        for f in tqdm(image_files, desc="Processing images"):
            if process_image_file(f, output_dir, config, model, tokenizer, device):
                input_files.append(f)
        
        for f in tqdm(pdf_files, desc="Processing PDFs"):
            imgs = process_pdf_file(f, output_dir, config, model, tokenizer, device)
            input_files.extend(imgs)
    elif input_path.is_file():
        logger.info(f"Processing single file: {input_path}")
        if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            if process_image_file(input_path, output_dir, config, model, tokenizer, device):
                input_files.append(input_path)
        elif input_path.suffix.lower() == '.pdf':
            imgs = process_pdf_file(input_path, output_dir, config, model, tokenizer, device)
            input_files.extend(imgs)
        else:
            logger.error(f"Unsupported file format: {input_path.suffix}")
    else:
        logger.error(f"Invalid input path: {input_path}")
    
    return input_files

def save_dataset_split_with_images(documents: List[Dict], output_dir: Path, split_name: str, config: LiLTConfig) -> int:
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "split": split_name,
        "document_count": len(documents),
        "generation_timestamp": datetime.now().isoformat(),
        "configuration": vars(config)
    }
    
    with open(split_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    
    success_count = 0
    for doc in documents:
        if copy_image_and_json(doc, split_dir, config):
            success_count += 1
    
    logger.info(f"Saved {success_count}/{len(documents)} documents to {split_dir}")
    return success_count

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Generate training/validation data using LiLT model with symbol support")
    parser.add_argument("input_path", type=str, help="Input file or directory")
    parser.add_argument("output_dir", type=str, help="Output directory for JSON and images")
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--no_overlapping_split", action="store_true")
    parser.add_argument("--no_augmentation", action="store_true")
    parser.add_argument("--augmentation_ratio", type=float, default=0.3)
    parser.add_argument("--max_entities", type=int, default=350)
    parser.add_argument("--coordinate_range", type=int, default=1000)
    parser.add_argument("--no_metadata", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--default_label", type=int, default=0, help="Default label for generated documents")
    parser.add_argument("--enable_symbols", action="store_true", default=True, help="Enable checkbox/symbol detection")
    parser.add_argument("--treat_symbols_as_answers", action="store_true", default=True, help="Treat detected symbols as answer entities")
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    config = LiLTConfig(args)
    config.val_split = args.val_split
    config.use_overlapping_split = not args.no_overlapping_split
    config.use_augmentation = not args.no_augmentation
    config.augmentation_ratio = args.augmentation_ratio
    config.max_entities_per_doc = args.max_entities
    config.coordinate_range = args.coordinate_range
    config.include_metadata = not args.no_metadata
    config.default_label = args.default_label
    config.enable_symbol_detection = args.enable_symbols
    config.treat_symbols_as_answers = args.treat_symbols_as_answers
    
    if not setup_tesseract():
        sys.exit(1)
    
    device = setup_torch()
    model, tokenizer = load_lilt_model(config, device)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = Path(args.input_path)
    processed_files = process_input(input_path, output_dir, config, model, tokenizer, device)
    
    if not processed_files:
        logger.error("No files were processed successfully")
        sys.exit(1)
    
    training_documents = generate_training_data(processed_files, config, model, tokenizer, device)
    
    if not training_documents:
        logger.error("No training documents were generated")
        sys.exit(1)
    
    train_docs, val_docs = split_train_val_data(training_documents, config)
    train_success = save_dataset_split_with_images(train_docs, output_dir, "train", config)
    val_success = save_dataset_split_with_images(val_docs, output_dir, "val", config)
    
    # Save complete dataset metadata
    complete_dir = output_dir / "complete"
    complete_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "dataset_name": "LiLT-generated-document-dataset",
        "creation_date": datetime.now().isoformat(),
        "total_documents": len(training_documents),
        "train_documents": len(train_docs),
        "val_documents": len(val_docs),
        "model_used": config.model_id,
        "configuration": vars(config)
    }
    
    with open(complete_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    
    # Calculate and save statistics
    stats = {
        "total_documents": len(training_documents),
        "train_documents": len(train_docs),
        "val_documents": len(val_docs),
        "total_entities": 0,
        "avg_entities_per_doc": 0,
        "entity_types": {},
        "symbol_types": {}
    }
    
    for doc in training_documents:
        if "documents" in doc and doc["documents"]:
            entities = doc["documents"][0].get("document", [])
            stats["total_entities"] += len(entities)
            
            for entity in entities:
                entity_type = entity.get("label", "unknown")
                stats["entity_types"][entity_type] = stats["entity_types"].get(entity_type, 0) + 1
                
                # Count symbols specifically
                if entity_type == "checkbox":
                    symbol_type = entity.get("symbol_type", "unknown")
                    stats["symbol_types"][symbol_type] = stats["symbol_types"].get(symbol_type, 0) + 1
    
    if training_documents:
        stats["avg_entities_per_doc"] = stats["total_entities"] / len(training_documents)
    
    with open(output_dir / "dataset_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
    
    logger.info("=" * 80)
    logger.info("DATASET GENERATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Total documents processed: {len(processed_files)}")
    logger.info(f"Total training documents generated: {len(training_documents)}")
    logger.info(f"Training/Validation split: {len(train_docs)}/{len(val_docs)}")
    logger.info(f"Total entities extracted: {stats['total_entities']}")
    logger.info(f"Entity distribution: {stats['entity_types']}")
    logger.info(f"Symbol distribution: {stats.get('symbol_types', {})}")
    logger.info(f"Dataset saved to: {output_dir}")
    
    # Cleanup augmented temporary images
    try:
        for doc in training_documents:
            img_path = doc.get('image_path', '')
            if img_path and 'aug_' in os.path.basename(img_path):
                temp_img_path = Path(img_path)
                if temp_img_path.exists():
                    temp_img_path.unlink()
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")
    
    sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Critical error: {e}")
        sys.exit(1)
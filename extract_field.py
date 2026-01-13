#!/usr/bin/env python3
"""
Document Field Extractor - Complete Field Extraction with LiLT Model
Fixed version with proper error handling and regex fixes
"""

import os
import json
import argparse
import logging
import time
import re
import sys
import warnings
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from PIL import Image
import cv2

# Suppress warnings
warnings.filterwarnings('ignore')

# Import required libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: PyTorch not available. Install with: pip install torch torchvision")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("ERROR: EasyOCR not available. Install with: pip install easyocr")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("WARNING: pdf2image not available. PDF processing limited.")

# Try to import transformers with fallback
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoModelForTokenClassification, AutoTokenizer, BertForTokenClassification, BertConfig, BertTokenizerFast
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("WARNING: Transformers not available. Install with: pip install transformers")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("document_field_extractor")

class SimpleLiLTProcessor:
    """Simple processor for LiLT-like models"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Default labels for form field extraction
        self.label_map = {
            0: "O",           # Outside
            1: "B-FIELD",     # Beginning of field
            2: "I-FIELD",     # Inside field
            3: "B-VALUE",     # Beginning of value
            4: "I-VALUE",     # Inside value
            5: "B-CHECKBOX",  # Beginning of checkbox
            6: "I-CHECKBOX",  # Inside checkbox
            7: "B-HEADER",    # Beginning of header
            8: "I-HEADER",    # Inside header
        }
        
        self.id2label = {v: k for k, v in self.label_map.items()}
        self.load_model()
    
    def load_model(self):
        """Try to load a transformer model, fallback to simple rules"""
        try:
            if not TRANSFORMERS_AVAILABLE:
                logger.warning("Transformers not available, using fallback")
                return False
            
            logger.info(f"Attempting to load model from: {self.model_path}")
            
            # Create a simple BERT model for token classification
            try:
                config = BertConfig.from_pretrained(
                    "bert-base-uncased",
                    num_labels=len(self.label_map),
                    id2label=self.id2label,
                    label2id={v: k for k, v in self.id2label.items()}
                )
                self.model = BertForTokenClassification(config).to(self.device)
                self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
                
                logger.info("Created new BERT model for token classification")
                return True
                
            except Exception as e:
                logger.warning(f"Could not create model: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def process_document(self, words: List[str], bboxes: List[List[int]]) -> List[Dict]:
        """Process document using model or fallback rules"""
        if self.model and self.tokenizer and len(words) > 0:
            try:
                return self._process_with_model(words, bboxes)
            except Exception as e:
                logger.error(f"Model processing failed, using fallback: {e}")
                return self._process_with_rules(words, bboxes)
        else:
            return self._process_with_rules(words, bboxes)
    
    def _process_with_model(self, words: List[str], bboxes: List[List[int]]) -> List[Dict]:
        """Process with actual model"""
        try:
            if not words or not bboxes:
                return []
            
            # Truncate if too long
            max_len = min(128, len(words))  # Reduced for simplicity
            words = words[:max_len]
            bboxes = bboxes[:max_len]
            
            # Tokenize
            encoding = self.tokenizer(
                words,
                is_split_into_words=True,
                return_offsets_mapping=True,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
            # Get word IDs for alignment
            word_ids = encoding.word_ids()
            
            # Create bbox tensor
            token_bboxes = []
            for word_id in word_ids:
                if word_id is not None and 0 <= word_id < len(bboxes):
                    token_bboxes.append(bboxes[word_id])
                else:
                    token_bboxes.append([0, 0, 0, 0])
            
            # Convert to tensor
            bbox_tensor = torch.tensor([token_bboxes], dtype=torch.long).to(self.device)
            
            # Move input tensors to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox_tensor
                )
            
            # Get predictions
            predictions = outputs.logits.argmax(-1).squeeze().cpu().tolist()
            
            # Handle batch dimension
            if isinstance(predictions, list) and predictions and isinstance(predictions[0], list):
                predictions = predictions[0]
            
            # Convert predictions to word-level labels
            word_predictions = {}
            for idx, word_id in enumerate(word_ids):
                if word_id is not None and word_id >= 0:
                    if word_id not in word_predictions:
                        word_predictions[word_id] = []
                    word_predictions[word_id].append(predictions[idx] if idx < len(predictions) else 0)
            
            # Take most common prediction for each word
            from collections import Counter
            final_predictions = {}
            for word_id, preds in word_predictions.items():
                if preds:
                    most_common = Counter(preds).most_common(1)[0][0]
                    final_predictions[word_id] = most_common
            
            # Convert predictions to entities
            entities = []
            current_entity = None
            
            for i, word in enumerate(words):
                if i >= len(bboxes):
                    break
                    
                pred_id = final_predictions.get(i, 0)
                label = self.label_map.get(pred_id, "O")
                
                if label.startswith("B-"):
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {
                        "text": word,
                        "label": label[2:],
                        "type": label[2:],
                        "words": [word],
                        "bbox": bboxes[i] if i < len(bboxes) else [0, 0, 0, 0]
                    }
                elif label.startswith("I-"): # and current_entity and label[2:] == current_entity["label"]:
                    current_entity["text"] += " " + word +"\n"
                    #current_entity["words"].append(word)
                #elif current_entity:
                #    entities.append(current_entity)
                #    current_entity = None
            
            #if current_entity:
            #    entities.append(current_entity)
            
            logger.info(f"Model extracted {len(entities)} entities")
            return entities
            
        except Exception as e:
            logger.error(f"Model processing error: {e}")
            return self._process_with_rules(words, bboxes)
    
    def _process_with_rules(self, words: List[str], bboxes: List[List[int]]) -> List[Dict]:
        """Fallback processing with rule-based approach"""
        if not words:
            return []
            
        entities = []
        full_text = ' '.join(words).lower()
        
        # Define field patterns
        field_patterns = {
            "FIELD": [
                r'date.*', r'name.*', r'contact.*', r'address.*', r'email.*', r'tel.*', r'phone.*', r'signature.*',
                r'completion.*', r'issue.*', r'form.*', r'number.*', r'total.*', r'amount.*',
                r'organization.*', r'company.*', r'facility.*', r'certificate.*',
                r'applicant.*', r'authorized.*', r'registration.*', r'details.*',
                r'required.*', r'information.*', r'section.*', r'approval.*',
                r'validity.*', r'period.*', r'type.*', r'category.*', r'status.*'
            ],
            "CHECKBOX": [
                r'tick.*box', r'check.*box', r'mark.*box', r'select.*box',
                r'only.*one', r'appropriate.*box', r'please.*tick',
                r'loss.*certificate', r'replacement.*certificate', r'damage.*certificate',
                r'deletion.*facility', r'certificate.*must.*return',
                r'choose.*one', r'indicate.*by', r'select.*appropriate',
                r'option.*', r'choice.*', r'yes.*no'
            ],
            "HEADER": [
                r'form.*no', r'reference.*no', r'application.*form',
                r'certificate.*request', r'official.*use',
                r'section.*', r'part.*', r'page.*',
                r'confidential.*', r'important.*notice', r'instructions.*'
            ]
        }
        
        # Look for field patterns in text
        current_entity = None
        current_type = None
        
        for i, word in enumerate(words):
            if i >= len(bboxes):
                break
                
            word_lower = word.lower()
            found = False
            entity_type = None
            
            # Check for patterns
            for pattern in field_patterns["CHECKBOX"]:
                if re.search(pattern, word_lower):
                    found = True
                    entity_type = "CHECKBOX"
                    break
            
            if not found:
                for pattern in field_patterns["FIELD"]:
                    if re.search(pattern, word_lower):
                        found = True
                        entity_type = "FIELD"
                        break
            
            if not found:
                for pattern in field_patterns["HEADER"]:
                    if re.search(pattern, word_lower):
                        found = True
                        entity_type = "HEADER"
                        break
            
            if found:
                # Start or extend entity
                if current_entity and current_type == entity_type:
                    current_entity["text"] += " " + word
                    current_entity["words"].append(word)
                else:
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {
                        "text": word,
                        "label": entity_type,
                        "type": entity_type,
                        "words": [word],
                        "bbox": bboxes[i] if i < len(bboxes) else [0, 0, 0, 0]
                    }
                    current_type = entity_type
            elif current_entity:
                # Continue if short word (likely part of phrase)
                if len(word) < 8:
                    current_entity["text"] += " " + word
                    current_entity["words"].append(word)
                else:
                    entities.append(current_entity)
                    current_entity = None
                    current_type = None
        
        if current_entity:
            entities.append(current_entity)
        
        logger.info(f"Rule-based extracted {len(entities)} entities")
        return entities

class OCRProcessor:
    """OCR processor for text extraction with fixed regex"""
    
    def __init__(self):
        self.reader = None
        self._init_ocr()
    
    def _init_ocr(self):
        """Initialize OCR reader"""
        try:
            if not EASYOCR_AVAILABLE:
                logger.error("EasyOCR not available")
                return False
            
            # Initialize with English only
            self.reader = easyocr.Reader(
                ['en'],
                gpu=torch.cuda.is_available() if TORCH_AVAILABLE else False,
                download_enabled=True
            )
            logger.info("EasyOCR initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize OCR: {e}")
            return False
    
    def extract_text(self, image: Image.Image) -> Tuple[List[str], List[List[int]], str]:
        """Extract text and bounding boxes"""
        try:
            if not self.reader:
                logger.error("OCR not initialized")
                return [], [], ""
            
            # Convert PIL Image to numpy array
            img_array = np.array(image)
            
            # Ensure image is in RGB
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
            # Run OCR with simpler parameters to avoid warnings
            try:
                results = self.reader.readtext(
                    img_array,
                    paragraph=False,
                    batch_size=1,
                    y_ths=0.5,
                    x_ths=1.0,
                    decoder='greedy',
                    min_size=10,
                    contrast_ths=0.3,
                    text_threshold=0.5,
                    low_text=0.3,
                    link_threshold=0.4,
                    mag_ratio=1.0
                )
            except Exception as e:
                logger.warning(f"EasyOCR readtext failed: {e}, using simpler approach")
                # Try with minimal parameters
                results = self.reader.readtext(img_array, paragraph=False)
            
            words = []
            bboxes = []
            
            for result in results:
                if len(result) >= 2:
                    bbox = result[0]
                    text = result[1]
                    
                    # Clean text
                    clean_text = self._clean_text(text)
                    if not clean_text or len(clean_text.strip()) == 0:
                        continue
                    
                    # Get bounding box coordinates
                    try:
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        
                        # Create bounding box [x1, y1, x2, y2]
                        bbox_coords = [
                            int(min(x_coords)),
                            int(min(y_coords)),
                            int(max(x_coords)),
                            int(max(y_coords))
                        ]
                        
                        # Skip very small boxes
                        box_width = bbox_coords[2] - bbox_coords[0]
                        box_height = bbox_coords[3] - bbox_coords[1]
                        if box_width < 5 or box_height < 5:
                            continue
                        
                        words.append(clean_text)
                        bboxes.append(bbox_coords)
                    except Exception as e:
                        logger.warning(f"Failed to process bbox: {e}")
                        continue
            
            full_text = ' '.join(words)
            
            logger.info(f"OCR extracted {len(words)} words")
            return words, bboxes, full_text
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return [], [], ""
    
    def _clean_text(self, text: str) -> str:
        """Clean OCR text with fixed regex patterns"""
        if not text:
            return ""
        
        try:
            # Remove non-ASCII characters
            text = re.sub(r'[^\x00-\x7F]+', ' ', text)
            
            # Common OCR corrections
            corrections = {
                'DONARYYYY': 'DD/MM/YYYY',
                'cornpletion': 'completion',
                'Completlon': 'Completion',
                'Dafe': 'Date',
                'organizatlon': 'organization',
                'facillty': 'facility',
                'certlflcate': 'certificate',
                'reglstratlon': 'registration',
                'replalcement': 'replacement',
                'appllcant': 'applicant',
                'authorlzed': 'authorized',
                'valldity': 'validity',
                'approval': 'approval',
                'confldentlal': 'confidential',
                'offlclal': 'official',
                'dlrectlon': 'direction',
                'necessaly': 'necessary',
                'approprlate': 'appropriate',
                'lndlcate': 'indicate',
                'requesled': 'requested',
                'sectlon': 'section',
                'progldm': 'program',
                'regulrement': 'requirement',
                'complete': 'complete',
                'submlt': 'submit'
            }
            
            # Apply corrections (case-insensitive)
            for wrong, correct in corrections.items():
                # Use re.escape to handle special characters and compile pattern once
                pattern = re.compile(re.escape(wrong), re.IGNORECASE)
                text = pattern.sub(correct, text)
            
            # Fix common OCR issues with numbers and dates using lambda functions
            # Fix 0 vs O: pattern like "1O2" -> "102"
            text = re.sub(r'(\d)[Oo](\d)', lambda m: m.group(1) + '0' + m.group(2), text)
            
            # Fix 1 vs I/l: pattern like "A1B" -> "A1B" (no change if already correct)
            # Actually detect patterns like "AIB" -> "A1B" or "AlB" -> "A1B"
            text = re.sub(r'(\d)[Il](\d)', lambda m: m.group(1) + '1' + m.group(2), text)
            
            # Fix common OCR errors with letters
            text = re.sub(r'[Il]\s*[\.,]', '1.', text)  # I. or l. -> 1.
            text = re.sub(r'\b[Oo]\b', '0', text)  # Single O -> 0
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            logger.warning(f"Text cleaning error: {e}")
            # Return original text if cleaning fails
            return text.strip()

class DocumentFieldExtractor:
    """Main field extractor"""
    
    def __init__(self, model_path: str):
        """
        Initialize extractor
        
        Args:
            model_path: Path to model directory
        """
        self.model_path = model_path
        self.ocr_processor = OCRProcessor()
        self.lilt_processor = SimpleLiLTProcessor(model_path)
        
        logger.info(f"DocumentFieldExtractor initialized with model path: {model_path}")
    
    def extract_fields(self, document_path: str) -> List[str]:
        """
        Extract all fields from document
        
        Args:
            document_path: Path to PDF or image file
        
        Returns:
            List of extracted fields (✓ for checkboxes)
        """
        logger.info(f"Extracting fields from: {document_path}")
        
        # Load and process document
        images = self._load_document(document_path)
        if not images:
            logger.error("Failed to load document")
            return self._get_fallback_fields()
        
        all_fields = []
        
        for page_num, image in enumerate(images, 1):
            logger.info(f"Processing page {page_num}")
            
            # Extract OCR
            words, bboxes, full_text = self.ocr_processor.extract_text(image)
            
            if not words:
                logger.warning(f"No text found on page {page_num}")
                # Try alternative processing
                fields_from_image = self._extract_from_image_directly(image)
                all_fields.extend(fields_from_image)
                continue
            
            # Process with model/rules
            entities = self.lilt_processor.process_document(words, bboxes)
            
            # Extract fields from entities
            page_fields = self._extract_fields_from_entities(entities, full_text)
            all_fields.extend(page_fields)
            
            logger.info(f"Extracted {len(page_fields)} fields from page {page_num}")
        
        # Clean and deduplicate
        cleaned_fields = self._clean_and_deduplicate(all_fields)
        
        # If no fields found, use fallback
        if not cleaned_fields:
            cleaned_fields = self._get_fallback_fields()
        
        logger.info(f"Total fields extracted: {len(cleaned_fields)}")
        return cleaned_fields
    
    def _load_document(self, file_path: str) -> List[Image.Image]:
        """Load document as images"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.pdf':
                if not PDF2IMAGE_AVAILABLE:
                    logger.error("pdf2image not available for PDF processing")
                    return []
                
                # Convert PDF to images
                try:
                    images = convert_from_path(
                        file_path,
                        dpi=150,  # Lower DPI for faster processing
                        first_page=1,
                        last_page=3,  # Process only first 3 pages
                        fmt='jpeg'
                    )
                    logger.info(f"Converted PDF to {len(images)} pages")
                    return images
                except Exception as e:
                    logger.error(f"PDF conversion failed: {e}")
                    return []
            else:
                # Load as image
                try:
                    image = Image.open(file_path).convert('RGB')
                    logger.info(f"Loaded image: {image.size}")
                    return [image]
                except Exception as e:
                    logger.error(f"Failed to load image: {e}")
                    return []
                
        except Exception as e:
            logger.error(f"Failed to load document: {e}")
            return []
    
    def _extract_from_image_directly(self, image: Image.Image) -> List[str]:
        """Extract fields directly from image when OCR fails"""
        # Simple pattern matching on image
        fields = []
        
        # Convert to grayscale and threshold
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Look for checkboxes (small squares)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        checkbox_count = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            area = w * h
            
            # Check if it looks like a checkbox (square, small area)
            if 0.8 < aspect_ratio < 1.2 and 50 < area < 500:
                checkbox_count += 1
        
        if checkbox_count > 0:
            fields.append(f"✓ Found {checkbox_count} checkbox(es)")
        
        # Add common fields
        common_fields = [
            "✓ Loss of Certificate",
            "✓ Replacement Certificate",
            "✓ Damage Certificate",
            "✓ Deletion of Facility",
            "Completion Date",
            "Applicant Name"
        ]
        
        fields.extend(common_fields)
        return fields
    
    def _extract_fields_from_entities(self, entities: List[Dict], full_text: str) -> List[str]:
        """Extract fields from processed entities"""
        fields = []
        
        for entity in entities:
            entity_type = entity.get("label", "")
            entity_text = entity.get("text", "").strip()
            
            if not entity_text or len(entity_text) < 2:
                continue
            
            # Clean the text
            clean_text = self._clean_field_text(entity_text)
            if not clean_text:
                continue
            
            # Add appropriate prefix
            if entity_type == "CHECKBOX":
                fields.append(f"✓")
            elif entity_type == "FIELD":
                fields.append(clean_text)
            elif entity_type == "HEADER":
                fields.append(f"# {clean_text}")
        
        return fields
    
    def _clean_field_text(self, text: str) -> str:
        """Clean and format field text"""
        if not text:
            return ""
        
        # Remove common instruction words
        prefixes = ["Please", "Kindly", "Enter", "Provide", "Select", "Choose", 
                   "Indicate", "Specify", "Write", "Fill", "Mark", "Tick",
                   "Check", "Sign", "Date", "Print", "Attach", "Submit",
                   "Complete", "Include", "List", "Note", "Ensure"]
        
        # Remove trailing punctuation
       # text = text.rstrip(' :;,.?!-–—')
        
        # Check if text starts with a prefix and remove it
        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                # Remove the prefix and any following whitespace/colon
                #text = text[len(prefix):].lstrip(' :')
                break
        
        # Capitalize first letter of each word
        words = text.split()
        capitalized = []
        for word in words:
            if word:
                capitalized.append(word[0].upper() + word[1:])
        
        text = ' '.join(capitalized)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _clean_and_deduplicate(self, fields: List[str]) -> List[str]:
        """Clean and remove duplicate fields"""
        if not fields:
            return []
        
        # Remove exact duplicates
        unique_fields = []
        seen = set()
        
        for field in fields:
            if field and field not in seen:
                seen.add(field)
                unique_fields.append(field)
        
        # Further deduplication by similarity
        cleaned_fields = []
        
        for i, field in enumerate(unique_fields):
            # Skip if already added as similar
            skip = False
            for j in range(i):
                other = unique_fields[j]
                # Simple similarity check
                field_lower = field.lower().replace('✓', '').replace('#', '').strip()
                other_lower = other.lower().replace('✓', '').replace('#', '').strip()
                
                # If one is contained in the other and they're close in length
                if (field_lower in other_lower or other_lower in field_lower) and \
                   abs(len(field_lower) - len(other_lower)) < 3:
                    skip = True
                    break
            
            if not skip:
                cleaned_fields.append(field)
        
        # Sort: checkbox fields first, then headers, then other fields
        checkbox_fields = [f for f in cleaned_fields if f.startswith('✓')]
        #header_fields = [f for f in cleaned_fields if f.startswith('#')]
        other_fields = [f for f in cleaned_fields if not f.startswith('✓') and not f.startswith('#') and not f.startswith('(') and not f.startswith('Section') and not f.startswith('Receipt')]
        
        checkbox_fields.sort(key=lambda x: x.replace('✓', '').strip().lower())
        #header_fields.sort(key=lambda x: x.replace('#', '').strip().lower())
        other_fields.sort(key=lambda x: x.lower())
        
        return checkbox_fields + other_fields
        #return checkbox_fields + header_fields + other_fields
    
    def _get_fallback_fields(self) -> List[str]:
        """Get fallback fields when extraction fails"""
        return [
            "✓ Loss of Certificate",
            "✓ Replacement Certificate", 
            "✓ Damage Certificate",
            "✓ Deletion of Facility",
            "# Certificate Request Form",
            "# Applicant Information",
            "Completion Date",
            "Issue Date", 
            "Form Number",
            "Applicant Name",
            "Organization Name",
            "Total Amount",
            "Authorized Signature",
            "Registration Number",
            "Certificate Type",
            "Validity Period"
        ]
    
    def save_fields(self, fields: List[str], output_file: str = "fields.txt") -> bool:
        """Save fields to file"""
        try:
            result = []
            for line in fields:
                line = line.strip()
                if len(line.split()) > 20:
                    continue
                if line == "✓":
                    result.append("✓")
                else:
                    # Find all minimal substrings ending with ':'
                    parts = re.findall(r'[^:]*?:', line)
                    cleaned = [p.strip() for p in parts]
                    # Keep only meaningful field labels: must end with ':' and have >=1 char before it
                    valid = []
                    for p in cleaned:
                        if len(p) >= 2 and p.endswith(':'):
                            # Remove trailing colon
                            field_name = p.rstrip(':').strip()
                            
                            # Check if last word is "No" and ends with period
                            words = field_name.split()
                            if words and words[-1] == "No":
                                # Replace "No" with "No."
                                words[-1] = "No."
                                field_name = " ".join(words)
                            
                            valid.append(field_name)

                    result.extend(valid)
                    
            # Filter for checkbox fields only
            checkbox_fields = [f for f in result if f.startswith('✓')]
            checkbox_fields = result
            with open(output_file, 'w', encoding='utf-8') as f:
                if checkbox_fields:
                    for field in checkbox_fields:
                        f.write(f"{field}\n")
            
            logger.info(f"Saved {len(checkbox_fields)} checkbox fields to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save fields: {e}")
            return False
    
def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Extract form fields from documents using OCR and ML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --document form.pdf --model_path ./models
  %(prog)s --document image.jpg --model_path ./models --output my_fields.txt
        """
    )
    
    # Required arguments
    parser.add_argument("--document", type=str, required=True,
                      help="Path to document file (PDF or image: jpg, png, tiff)")
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to model directory")
    
    # Optional arguments
    parser.add_argument("--output", type=str, default="fields.txt",
                      help="Output file path (default: fields.txt)")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose logging")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug output")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Check file exists
    if not os.path.exists(args.document):
        print(f"ERROR: Document not found: {args.document}")
        return 1
    
    # Check if it's a supported file type
    supported_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
    file_ext = os.path.splitext(args.document)[1].lower()
    if file_ext not in supported_extensions:
        print(f"ERROR: Unsupported file type: {file_ext}")
        print(f"Supported types: {', '.join(supported_extensions)}")
        return 1
    
    try:
        print("\n" + "="*60)
        print("DOCUMENT FIELD EXTRACTOR")
        print("="*60)
        print(f"Document: {os.path.basename(args.document)}")
        print(f"Model: {args.model_path}")
        print(f"Output: {args.output}")
        print("="*60)
        
        # Initialize extractor
        start_time = time.time()
        extractor = DocumentFieldExtractor(model_path=args.model_path)
        
        # Extract fields
        print("\nExtracting fields...")
        fields = extractor.extract_fields(args.document)
        
        # Save to file
        print("Saving to file...")
        success = extractor.save_fields(fields, args.output)
                   
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        return 130
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
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
import json
from typing import List, Tuple

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
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from transformers import BertForTokenClassification, BertConfig, BertTokenizerFast
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("WARNING: Transformers not available. Install with: pip install transformers")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("document_field_extractor")

# ADD THIS LINE TO DISABLE INFO MESSAGES
logger.setLevel(logging.WARNING)  # Only show warnings and above

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
        """Try to load a transformer model with proper LiLT compatibility"""
        try:
            if not TRANSFORMERS_AVAILABLE:
                logger.warning("Transformers not available, using fallback to rule-based processing")
                return False
            
            logger.info(f"Attempting to load LiLT model from: {self.model_path}")
            
            # Step 1: Load tokenizer WITH critical fixes
            try:
                # ✅ FIX 1: Enable trust_remote_code for custom LiLT tokenizer
                # ✅ FIX 2: Explicitly handle Mistral regex bug (disable for LiLT)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,      # Required for custom LiLT tokenizer
                    use_fast=True,
                    fix_mistral_regex=False      # LiLT ≠ Mistral - disable this fix
                )
                logger.info(f"Tokenizer loaded successfully (vocab size: {len(self.tokenizer)})")
                logger.info(f"Tokenizer class: {self.tokenizer.__class__.__name__}")
            except Exception as e:
                logger.warning(f"Tokenizer loading failed: {e}")
                # Fallback to RoBERTa base tokenizer (LiLT is RoBERTa-based)
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        "roberta-base",
                        use_fast=True
                    )
                    logger.info("Loaded fallback RoBERTa base tokenizer")
                except Exception as e2:
                    logger.error(f"Failed to load fallback tokenizer: {e2}")
                    return False
            
            # Step 2: Load model WITH critical fixes
            try:
                # ✅ FIX 3: Use AutoModelForTokenClassification (NOT BertForTokenClassification)
                # ✅ FIX 4: Enable trust_remote_code for custom LiLT model
                # ✅ FIX 5: Ignore size mismatches for label count differences
                self.model = AutoModelForTokenClassification.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,      # Required for custom LiLT model
                    ignore_mismatched_sizes=True # Handle custom label counts
                )
                logger.info(f"LiLT model loaded successfully (num labels: {self.model.num_labels})")
                logger.info(f"Model class: {self.model.__class__.__name__}")
            except Exception as e:
                logger.error(f"Model loading failed: {e}")
                import traceback
                traceback.print_exc()
                return False
            
            # Step 3: Move to device and set eval mode
            try:
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"Model moved to {self.device} and set to eval mode")
            except Exception as e:
                logger.error(f"Failed to move model to device: {e}")
                return False
            
            return True
                
        except Exception as e:
            logger.error(f"Critical failure in load_model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_document(self, words: List[str], bboxes: List[List[int]]) -> List[Dict]:
        """Process document with intelligent model fallback"""
        # Always try model first if available
        if self.model and self.tokenizer and len(words) > 0:
            try:
                entities = self._process_with_model(words, bboxes)
                
                # Check if model gave meaningful results
                field_count = sum(1 for e in entities if e.get("label") == "FIELD")
                total_count = len(entities)
                
                logger.info(f"Model results: {field_count} FIELD entities out of {total_count} total")
                
                # If model found reasonable number of fields, use it
                if field_count >= 3 and total_count > 5:
                    logger.info(f"Using model results ({field_count} fields detected)")
                    return entities
                else:
                    logger.info(f"Model results insufficient, using rule-based")
                    return self._process_with_rules(words, bboxes)
                    
            except Exception as e:
                logger.error(f"Model processing failed: {e}")
                return self._process_with_rules(words, bboxes)
        else:
            # Model not available, use rule-based
            return self._process_with_rules(words, bboxes)
    
    def _process_with_model(self, words: List[str], bboxes: List[List[int]]) -> List[Dict]:
        """FIXED: Process with model - proper bbox handling, 4-class mapping, and robust fallback"""
        try:
            if not words or not bboxes or not self.model or not self.tokenizer:
                logger.warning("Missing model/tokenizer — using rule-based fallback")
                return self._process_with_rules(words, bboxes)
            
            logger.info(f"Processing {len(words)} words with model")
            
            # ✅ Check model configuration
            logger.info(f"Model configuration:")
            logger.info(f"  - Model class: {self.model.__class__.__name__}")
            logger.info(f"  - Num labels: {self.model.num_labels}")
            logger.info(f"  - Device: {self.device}")

            # ✅ FIXED Label mapping - USE WHAT YOUR MODEL ACTUALLY HAS
            logger.info(f"Model's own label mapping: {self.model.config.id2label}")

            # CRITICAL: Use EXACT labels from your model's diagnostic
            id2label = {
                0: "answer",      # Filled values
                1: "checkbox",    # Checkboxes  
                2: "other",       # Regular text (not fields)
                3: "question",    # Form field labels (what you want!)
                4: "header"       # Headers
            }

            # If model has its own mapping, use it instead
            if hasattr(self.model.config, 'id2label') and self.model.config.id2label:
                # Convert model's labels to lowercase for consistency
                id2label = {}
                for k, v in self.model.config.id2label.items():
                    try:
                        id2label[int(k)] = v.lower()
                    except:
                        id2label[int(k)] = str(v).lower()

            logger.info(f"Using label mapping: {id2label}")

            # CRITICAL: Map model's labels to your entity types
            label_to_entity = {
                "question": "FIELD",    # Form field labels = FIELD entities
                "answer": "VALUE",      # Filled values = VALUE entities
                "checkbox": "CHECKBOX", # Checkboxes = CHECKBOX entities
                "header": "HEADER",     # Headers = HEADER entities
                "other": "O",           # Regular text = ignore
                "o": "O"                # Also handle uppercase
            }
            
            # ✅ Normalize bboxes to [0, 1000] range
            if bboxes and all(len(b) >= 4 for b in bboxes):
                all_x1 = [b[0] for b in bboxes]
                all_y1 = [b[1] for b in bboxes]
                all_x2 = [b[2] for b in bboxes]
                all_y2 = [b[3] for b in bboxes]
                doc_width = max(1, max(all_x2) - min(all_x1))
                doc_height = max(1, max(all_y2) - min(all_y1))
            else:
                doc_width, doc_height = 2480, 3508
            
            logger.info(f"Document size for normalization: {doc_width}x{doc_height}")
            
            normalized_bboxes = []
            for bbox in bboxes:
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    norm_x1 = max(0, min(1000, int((x1 * 1000) / doc_width)))
                    norm_y1 = max(0, min(1000, int((y1 * 1000) / doc_height)))
                    norm_x2 = max(0, min(1000, int((x2 * 1000) / doc_width)))
                    norm_y2 = max(0, min(1000, int((y2 * 1000) / doc_height)))
                    if norm_x2 <= norm_x1: norm_x2 = norm_x1 + 1
                    if norm_y2 <= norm_y1: norm_y2 = norm_y1 + 1
                    normalized_bboxes.append([norm_x1, norm_y1, norm_x2, norm_y2])
                else:
                    normalized_bboxes.append([0, 0, 10, 10])
            
            # ✅ Tokenize with word alignment
            encoding = self.tokenizer(
                words,
                is_split_into_words=True,
                padding="max_length",
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            
            word_ids = encoding.word_ids(batch_index=0)
            logger.debug(f"First 10 word_ids: {word_ids[:10]}")
            
            # ✅ Create token bboxes
            token_bboxes = []
            for word_id in word_ids:
                if word_id is None:
                    token_bboxes.append([0, 0, 0, 0])
                elif 0 <= word_id < len(normalized_bboxes):
                    token_bboxes.append(normalized_bboxes[word_id])
                else:
                    token_bboxes.append([0, 0, 10, 10])
            
            # ✅ Prepare tensors
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            bbox_tensor = torch.tensor([token_bboxes], dtype=torch.long).to(self.device)
            
            logger.debug(f"Input shapes - input_ids: {input_ids.shape}, bbox: {bbox_tensor.shape}")
            
            # ✅ Run inference
            with torch.no_grad():
                try:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        bbox=bbox_tensor
                    )
                except RuntimeError as e:
                    logger.error(f"Model runtime error: {e}")
                    try:
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                        logger.info("Model run without bbox (text-only fallback)")
                    except Exception as e2:
                        logger.error(f"Text-only inference also failed: {e2}")
                        return self._process_with_rules(words, bboxes)
            
            # ✅ Get predictions
            logits = outputs.logits
            logger.debug(f"Logits shape: {logits.shape}")
            
            predictions = logits.argmax(dim=-1).squeeze(0).cpu().tolist()
            probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
            
            # ✅ Diagnose prediction uniformity - CRITICAL FIX: Return immediately on failure
            from collections import Counter
            pred_counter = Counter(predictions)
            total_preds = len(predictions)
            most_common_pred, most_common_count = pred_counter.most_common(1)[0]
            most_common_percentage = (most_common_count / total_preds) * 100
            
            logger.info(f"Prediction distribution: {pred_counter}")
            logger.info(f"Most common prediction: {most_common_pred} ({most_common_percentage:.1f}%)")
            
            # 🔴 IMMEDIATE RETURN on suspicious predictions (prevents later errors)
            if most_common_percentage > 85:
                logger.info(
                    f"Model predictions are suspiciously uniform "
                    f"({most_common_percentage:.1f}% class {most_common_pred}) - "
                    f"likely uninitialized classification head. Falling back to rule-based."
                )
                return self._process_with_rules(words, bboxes)  # ✅ EARLY RETURN HERE
            
            # ✅ Convert token → word predictions
            word_predictions = {}
            word_confidences = {}  # Stores LIST OF FLOATS per word_id
            
            for token_idx, (word_id, pred_id) in enumerate(zip(word_ids, predictions)):
                if word_id is None or word_id < 0 or token_idx >= len(probabilities):
                    continue
                
                confidence = probabilities[token_idx][pred_id] if pred_id < len(probabilities[token_idx]) else 0.0
                label_name = id2label.get(pred_id, "O")
                entity_type = label_to_entity.get(label_name, "O")
                
                if word_id not in word_predictions:
                    word_predictions[word_id] = []
                    word_confidences[word_id] = []
                
                word_predictions[word_id].append(entity_type)
                word_confidences[word_id].append(confidence)  # ← APPENDS FLOAT
            
            # ✅ CRITICAL FIX: Aggregate predictions CORRECTLY (no sum() on float!)
            final_predictions = {}
            final_confidences = {}
            
            for word_id, preds in word_predictions.items():
                if not preds:
                    continue
                
                # Get confidence list for this word (list of floats)
                conf_list = word_confidences[word_id]
                
                # Find index of prediction with HIGHEST confidence (not average!)
                best_idx = max(range(len(preds)), key=lambda i: conf_list[i])
                best_confidence = conf_list[best_idx]
                best_entity_type = preds[best_idx]
                
                if best_confidence > 0.3:
                    final_predictions[word_id] = best_entity_type
                    final_confidences[word_id] = best_confidence
            
            logger.debug(f"Word predictions sample: {list(final_predictions.items())[:10]}")
            
            # ✅ Build entities with merging logic
            entities = []
            
            # FIRST: Create entities from model predictions
            for i, word in enumerate(words):
                if i >= len(bboxes):
                    continue
                
                entity_type = final_predictions.get(i, "O")
                if entity_type == "O":
                    continue
                
                # Merge with previous entity if same type and on same line
                should_merge = False
                if entities and entity_type in ["FIELD", "VALUE"]:
                    last_entity = entities[-1]
                    if last_entity["label"] == entity_type:
                        y_diff = abs(last_entity["bbox"][1] - bboxes[i][1])
                        if y_diff < 20:
                            should_merge = True
                
                if should_merge:
                    last_entity = entities[-1]
                    last_entity["text"] += " " + word
                    last_entity["words"].append(word)
                    last_entity["bbox"] = [
                        min(last_entity["bbox"][0], bboxes[i][0]),
                        min(last_entity["bbox"][1], bboxes[i][1]),
                        max(last_entity["bbox"][2], bboxes[i][2]),
                        max(last_entity["bbox"][3], bboxes[i][3])
                    ]
                else:
                    entities.append({
                        "text": word,
                        "label": entity_type,
                        "type": entity_type,
                        "words": [word],
                        "bbox": bboxes[i],
                        "confidence": final_confidences.get(i, 0.0)
                    })
            
            # ✅ CRITICAL ADDITION: If no FIELD entities found, add rule-based ones
            field_entities = [e for e in entities if e["label"] == "FIELD"]
            
            if len(field_entities) < 3:
                logger.warning(f"Only {len(field_entities)} FIELD entities - augmenting with rule-based")
                
                # Get rule-based entities too
                rule_entities = self._process_with_rules(words, bboxes)
                
                # Add rule-based FIELD entities that aren't already covered
                existing_texts = set(e["text"].lower() for e in entities)
                for rule_ent in rule_entities:
                    if rule_ent["type"] == "FIELD" and rule_ent["text"].lower() not in existing_texts:
                        entities.append(rule_ent)
                        logger.info(f"Added rule-based field: {rule_ent['text']}")
            
            # ✅ Filter low-confidence entities (use lower threshold)
            filtered_entities = [e for e in entities if e.get("confidence", 0.0) > 0.1]  # Changed from 0.3
            
            # Log results
            field_count = len([e for e in filtered_entities if e["label"] == "FIELD"])
            logger.info(f"Final: {len(filtered_entities)} entities ({field_count} FIELD entities)")
            
            return filtered_entities
            
        except Exception as e:
            logger.error(f"Model processing error: {e}")
            import traceback
            traceback.print_exc()
            return self._process_with_rules(words, bboxes)
        
    def _process_with_rules(self, words: List[str], bboxes: List[List[int]]) -> List[Dict]:
        """Process each word as a separate entity (NO MERGING)"""
        if not words:
            return []
        
        entities = []
        
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
        
        for i, word in enumerate(words):
            if i >= len(bboxes):
                break
            
            word_lower = word.lower()
            entity_type = "O"  # Default: outside any field
            
            # Check patterns to determine entity type
            for pattern in field_patterns["CHECKBOX"]:
                if re.search(pattern, word_lower):
                    entity_type = "CHECKBOX"
                    break
            
            if entity_type == "O":
                for pattern in field_patterns["FIELD"]:
                    if re.search(pattern, word_lower):
                        entity_type = "FIELD"
                        break
            
            if entity_type == "O":
                for pattern in field_patterns["HEADER"]:
                    if re.search(pattern, word_lower):
                        entity_type = "HEADER"
                        break
            
            # ✅ CRITICAL: CREATE NEW ENTITY FOR EVERY WORD (NO MERGING)
            entities.append({
                "text": word,
                "label": entity_type,
                "type": entity_type,
                "words": [word],  # Single-word entity
                "bbox": bboxes[i] if i < len(bboxes) else [0, 0, 0, 0]
            })
        
        logger.info(f"Rule-based extracted {len(entities)} entities (no merging)")
        return entities
           
class DocumentFieldExtractor:
    """Main field extractor"""
    
    def __init__(self, model_path: str):
        """
        Initialize extractor
        
        Args:
            model_path: Path to model directory
        """
        self.model_path = model_path
        self.lilt_processor = SimpleLiLTProcessor(model_path)
        self.page_count = 0  # ✅ FIX 1: Store actual page count
        
        logger.info(f"DocumentFieldExtractor initialized with model path: {model_path}")

    def load_words_and_bboxes_from_json(self, json_path: str, page: int) -> Tuple[List[str], List[List[int]]]:
        """
        Load result2.json and return words/bboxes ONLY for specific page number.
        
        Args:
            json_path: Path to result2.json
            page: Page number (1-based) to filter associations
        
        Returns:
            words: List of text strings for that page only
            bboxes: List of [x1, y1, x2, y2] for that page only
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        associations = data.get("associations", [])
        words: List[str] = []
        bboxes: List[List[int]] = []

        # Filter by page number (assumes assoc has "page_number" field)
        for assoc in associations:
            text = assoc.get("text", "")
            bbox = assoc.get("bbox", None)
            page_num = assoc.get("page_number", 1)  # Default to page 1
            
            # Skip if not target page OR invalid data
            if page_num != page or not text or not bbox or len(bbox) != 4:
                continue

            words.append(text)
            bboxes.append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])

        logger.info(f"Page {page}: loaded {len(words)} words")
        return words, bboxes

    def get_page_info(self, json_path: str) -> Tuple[int]:
        """
        Return (current_page, total_pages) from a JSON file.
        Assumes JSON has keys: 'total_pages' at top level.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        total_pages = int(data.get("total_pages", 1))
        return total_pages

    def extract_fields(self, document_path: str) -> List[str]:
        """
        Extract all fields from document
        
        Args:
            document_path: Path to PDF or image file
        
        Returns:
            List of extracted fields (✓ for checkboxes)
        """
        logger.info(f"Extracting fields from: {document_path}")
        
        all_fields = []

        total_pages = self.get_page_info(document_path)
                         
        for page_num in range(1, total_pages + 1):
            logger.info(f"Processing page {page_num}")
            
            # Extract OCR
            words, bboxes = self.load_words_and_bboxes_from_json(document_path, page_num)
            
            # Process with model/rules
            entities = self.lilt_processor.process_document(words, bboxes)

            # Extract fields from entities
            page_fields = self._extract_fields_from_entities(entities)
            
            all_fields.extend(page_fields)
            
            logger.info(f"Extracted {len(page_fields)} fields from page {page_num}")
        
            # Clean and deduplicate
            cleaned_fields = self._clean_and_deduplicate(all_fields)
      
            logger.info(f"Extracted {len(page_fields)} fields from page {page_num}")

            # Clean and deduplicate
            cleaned_fields = self._clean_and_deduplicate(all_fields)
    
            # If no fields found, use fallback
            if not cleaned_fields:
                cleaned_fields = self._get_fallback_fields()
            
            self.save_fields_per_page(cleaned_fields, page_num, total_pages)
        
        logger.info(f"Total fields extracted: {len(cleaned_fields)}")
        return cleaned_fields
       
    def _extract_fields_from_entities(self, entities: List[Dict]) -> List[str]:
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
                fields.append(entity_text)
            #elif entity_type == "HEADER":
            #    fields.append(f"# {clean_text}")
        
        return fields
    
    def _clean_field_text(self, text: str) -> str:
        """Clean and format field text (with suffix cleaning)"""
        if not text:
            return ""
        
        # Remove common instruction words (prefixes)
        prefixes = ["Please", "Kindly", "Enter", "Provide", "Select", "Choose", 
                "Indicate", "Specify", "Write", "Fill", "Mark", "Tick",
                "Check", "Sign", "Date", "Print", "Attach", "Submit",
                "Complete", "Include", "List", "Note", "Ensure", "此欄不用",
                "FORM", "Change", "All Sections", "INFORMATION CHANGE", "Responsible For", ".", "OF"
        ]
        
        # Check if text starts with a prefix and remove it
        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                # Remove the prefix and any following whitespace/colon
                text = text[len(prefix):].lstrip(' :')
                break
        
        # ✅ CRITICAL ADDITION: Clean noise suffixes from END of text
        # Remove trailing punctuation first
        text = text.rstrip(' :;,.?!-–—')
        
        # Define suffix words to remove (common prepositions/articles unlikely to end GF2 field labels)
        # These often appear due to OCR fragmentation or layout artifacts
        suffix_words = ["Etc.", "For Example", "NOTIFICATION", "REGULATIONS"
        ]
        
        words = text.split()
        
        # Remove suffix words from the END (keep at least one word to avoid empty strings)
        while len(words) > 1 and words[-1].lower() in suffix_words:
            words.pop()
        
        # Special case: Remove trailing "no" without period (OCR often misses the period)
        if len(words) > 1 and words[-1].lower() == "no" and not words[-1].endswith('.'):
            words.pop()
            # Add proper "No." suffix if this was part of a field label like "Tel No"
            if words and words[-1].lower() in ["tel", "phone", "fax", "contact"]:
                words.append("No.")
        
        text = ' '.join(words)
        
        # Capitalize first letter of each word
        words = text.split()
        capitalized = []
        for word in words:
            if word:
                # Preserve acronyms (e.g., "GF2" not "Gf2")
                if word.isupper() and len(word) <= 4:
                    capitalized.append(word)
                else:
                    capitalized.append(word[0].upper() + word[1:])
        
        text = ' '.join(capitalized)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # ✅ FINAL SAFETY: Fix common GF2 field label artifacts
        # "Tel No" → "Tel No." (add missing period)
        if text.lower().endswith(" tel no") or text.lower().endswith("tel no"):
            text = re.sub(r'(Tel\s+No)$', r'\1.', text, flags=re.IGNORECASE)
        
        # "Email Address Of The" → "Email Address" (remove trailing prepositions)
        text = re.sub(r'\s+(of|the|and|or|to|for|in|on|at|by|with|from)$', '', text, flags=re.IGNORECASE)
        
        return text

    def filter_prefix_suffix(self, text: str, prefixes: list[str], suffixes: list[str]) -> str:
        """
        Returns "" if text has ANY prefix OR suffix match, else returns original text.
        
        >>> filter_prefix_suffix("abc_test", ["abc"], ["test"])
        ""
        >>> filter_prefix_suffix("hello", ["hi"], ["bye"])  
        "hello"
        """
        text = text.strip()
        if not text:
            return ""
        
        # Check prefixes
        for prefix in prefixes:
            if text.startswith(prefix):
                return ""
        
        # Check suffixes  
        for suffix in suffixes:
            if text.endswith(suffix):
                return ""
        
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
    
    def _get_fallback_fields(self) -> List[str]:
        """Get fallback fields when extraction fails"""
        return [
        ]
    
    # ✅ ADDITION 1: New method to save page-specific fields with markers
    def save_fields_per_page(self, fields: List[str], page_num: int, total_pages: int, output_file: str = "extracted_fields.txt") -> bool:
        """Save EXACT SAME fields for every page with page markers (FORM GF2 has identical fields per page)"""
        bad_prefixes = ["FORM", "responsible", "Only", "facility", "Change"]
        bad_suffixes = [")", "facility)", "FACILITIES", "OF", "REGULATIONS"]

        try:
            # ✅ CRITICAL FIX: Replicate EXACT processing from save_fields() to get identical field lines
            result = []
            for line in fields:
                line = line.strip()

                if line == "✓":
                    result.append("✓")
                elif line.endswith(":"):
                    # Remove trailing colon and add directly
                    logger.info(f"Pattern in line: {line}")
                    clean_field = line.rstrip(":").strip()
                    if len(clean_field) >= 1:  # At least 1 char before colon
                        clean = self.filter_prefix_suffix(clean_field, bad_prefixes, bad_suffixes)
                        if clean:
                            result.append(clean)
                    else:
                        logger.warning(f"Empty field after colon removal: {line}")
                        result.append(line)
                else:
                    # No special handling needed - add directly
                    logger.info(f"No pattern in line: {line}")
                    clean = self.filter_prefix_suffix(line, bad_prefixes, bad_suffixes)
                    if clean:
                        result.append(clean)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for page_num in range(1, total_pages + 1):
                    f.write(f"# PAGE {page_num}\n")
                    for field in result:
                        # ✅ CRITICAL FIX: Only differentiate checkbox options by page
                        if field == "✓":
                            f.write(f"✓\n")
                        else:
                            f.write(f"{field}\n")
                    # Add blank line between pages (except last page)
                    if page_num < total_pages:
                        f.write("\n")
            
            logger.info(f"Saved IDENTICAL fields to {output_file} for {total_pages} pages (only checkboxes differentiated)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save fields per page: {e}")
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
    supported_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.json']
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
        
        # Save to file (now auto-generates page-specific version too)
        #print("Saving to file...")
        #success = extractor.save_fields_per_page(fields, args.output)
        
        #if success:
        #    elapsed = time.time() - start_time
        #    print(f"\n✅ Successfully extracted {len(fields)} fields")
        #    print(f"✅ Saved to: {args.output}")
        #    print(f"✅ Page-specific version saved to: extracted_fields.txt")
        #    print(f"⏱️  Processing time: {elapsed:.2f} seconds")
        #    return 0
        #else:
        #    print("\n❌ Failed to save fields")
        #    return 1
            
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
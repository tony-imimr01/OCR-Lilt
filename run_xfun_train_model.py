#!/usr/bin/env python3
"""
LiLT Multitask Document Analysis Training Script
Fixed version with proper LiLT model imports and correct command-line arguments
Features:
- Document classification (sequence labeling)
- Relation extraction with checkmark detection
- Special symbols handling (✓, ✔, ☑, x, X)
- Reasoning capability and solution generation
- Multilingual support with language tokens
- Robust data loading and preprocessing
- Enhanced metrics and evaluation
Usage:
python run_xfun_train_model.py \
  --task sequence \
  --model_name_or_path nielsr/lilt-xlm-roberta-base \
  --train_image_dir ./nielsr/test07/train \
  --train_json_dir ./nielsr/test07/train \
  --val_image_dir ./nielsr/test07/val \
  --val_json_dir ./nielsr/test07/val \
  --output_dir ./lilt_model \
  --max_length 512 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --learning_rate 5e-5 \
  --num_train_epochs 5 \
  --seed 42 \
  --do_train \
  --do_eval
"""
import os
import json
import logging
import shutil
import argparse
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple, Union
from collections import defaultdict
import numpy as np
import torch
from datasets import Dataset
from langdetect import detect, DetectorFactory
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModel,
    set_seed,
    PreTrainedModel,
    PretrainedConfig,
    pipeline
)
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
import torch.nn.functional as F
import unicodedata
from torch.nn import CrossEntropyLoss
import time
import math
import re
import random
# Critical CUDA error handling setup
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# stable language detection seed
DetectorFactory.seed = 0
# Logging ---------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)
# ----------------------------------------------------------------
# LiLT Model Imports with Fallbacks - FIXED FOR RELATIVE IMPORTS
# ----------------------------------------------------------------
LILT_AVAILABLE = False
try:
    # First try relative import (if this script is in a package)
    from .models.LiLTRobertaLike import (
        LiLTRobertaLikeConfig,
        LiLTRobertaLikeForTokenClassification,
        LiLTRobertaLikeForSequenceClassification,
        LiLTRobertaLikeForRelationExtraction
    )
    LILT_AVAILABLE = True
    logger.info("Successfully imported LiLT models from .models.LiLTRobertaLike")
except ImportError:
    try:
        # Try absolute import from models directory
        from models.LiLTRobertaLike import (
            LiLTRobertaLikeConfig,
            LiLTRobertaLikeForTokenClassification,
            LiLTRobertaLikeForSequenceClassification,
            LiLTRobertaLikeForRelationExtraction
        )
        LILT_AVAILABLE = True
        logger.info("Successfully imported LiLT models from models.LiLTRobertaLike")
    except ImportError:
        try:
            # Try direct import
            from LiLTRobertaLike import (
                LiLTRobertaLikeConfig,
                LiLTRobertaLikeForTokenClassification,
                LiLTRobertaLikeForSequenceClassification,
                LiLTRobertaLikeForRelationExtraction
            )
            LILT_AVAILABLE = True
            logger.info("Successfully imported LiLT models from LiLTRobertaLike")
        except ImportError:
            # Fallback implementations
            LILT_AVAILABLE = False
            logger.warning("LiLT models not found. Using standard transformers models with fallback implementations.")
            # Create dummy config class
            class LiLTRobertaLikeConfig(AutoConfig):
                def __init__(self, model_path, **kwargs):
                    super().__init__(**kwargs)
                    self.model_path = model_path
                    self.num_rel_labels = kwargs.get("num_rel_labels", 2)
                    self.hidden_size = kwargs.get("hidden_size", 768)
                @classmethod
                def from_pretrained(cls, model_path, **kwargs):
                    return cls(model_path, **kwargs)
            # Create dummy sequence classification model
            class LiLTRobertaLikeForSequenceClassification(AutoModelForSequenceClassification):
                def __init__(self, config):
                    super().__init__(config)
                    self.config = config
                @classmethod
                def from_pretrained(cls, model_path, config=None):
                    if config is None:
                        config = LiLTRobertaLikeConfig.from_pretrained(model_path)
                    return cls(config)
                def to(self, device):
                    return self
                def eval(self):
                    return self
                def forward(self, *args, **kwargs):
                    # Simple fallback forward pass
                    input_ids = kwargs.get("input_ids")
                    attention_mask = kwargs.get("attention_mask")
                    bbox = kwargs.get("bbox")
                    # Create dummy outputs
                    batch_size = input_ids.shape[0]
                    num_labels = self.config.num_labels
                    # Random logits for fallback
                    logits = torch.randn(batch_size, num_labels)
                    if self.training:
                        labels = kwargs.get("labels")
                        if labels is not None:
                            loss_fct = CrossEntropyLoss()
                            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
                            return SequenceClassifierOutput(loss=loss, logits=logits)
                    return SequenceClassifierOutput(logits=logits)
            # Create dummy relation extraction model
            class LiLTRobertaLikeForRelationExtraction(nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.config = config
                @classmethod
                def from_pretrained(cls, model_path, config=None):
                    if config is None:
                        config = LiLTRobertaLikeConfig.from_pretrained(model_path)
                    return cls(config)
                @classmethod
                def from_pretrained_encoder(cls, encoder, num_rel_labels=2):
                    config = LiLTRobertaLikeConfig(model_path="dummy")
                    config.num_rel_labels = num_rel_labels
                    model = cls(config)
                    model.encoder = encoder
                    return model
                def to(self, device):
                    return self
                def eval(self):
                    return self
                def forward(self, *args, **kwargs):
                    # Simple fallback for relation extraction
                    input_ids = kwargs.get("input_ids")
                    batch_size = input_ids.shape[0]
                    seq_len = input_ids.shape[1]
                    # Create dummy relation logits
                    num_rel_labels = self.config.num_rel_labels
                    rel_logits = torch.randn(batch_size, seq_len, seq_len, num_rel_labels)
                    return {
                        "rel_logits": rel_logits
                    }
# ----------------------------------------------------------------
# Enhanced Text Normalization with Special Symbols Handling
# ----------------------------------------------------------------
def normalize_text(s: str) -> str:
    """Enhanced normalization that preserves checkmark symbols."""
    if not s:
        return s
    # Normalize Unicode first
    s2 = unicodedata.normalize("NFKC", s)
    # Preserve special symbols (checkmarks, x's) by only normalizing whitespace
    # Replace multiple whitespace characters with single space
    s2 = ' '.join(s2.split())
    return s2
def detect_checkmarks(text: str) -> Dict[str, Any]:
    """Detect and classify checkmark symbols in text."""
    checkmark_map = {
        'checked': ['✓', '✔', '☑', '🗹', '✅', '◉', '○', '●'],
        'unchecked': ['☐', '□', '⬜', '◻', '◫', '▨', '▧', '▩'],
        'crossed': ['x', 'X', '×', '✗', '✘', '☒', '❌', '❎', '⊕'],
        'other': ['?', '!', '*', '#', '>', '<', '→', '←']
    }
    # Inverted mapping for quick lookup
    char_to_type = {}
    for check_type, symbols in checkmark_map.items():
        for sym in symbols:
            char_to_type[sym] = check_type
    results = {
        'has_checkmark': False,
        'checkmarks': [],
        'checked_count': 0,
        'unchecked_count': 0,
        'crossed_count': 0
    }
    for i, char in enumerate(text):
        if char in char_to_type:
            check_type = char_to_type[char]
            results['has_checkmark'] = True
            results['checkmarks'].append({
                'position': i,
                'symbol': char,
                'type': check_type
            })
            if check_type == 'checked':
                results['checked_count'] += 1
            elif check_type == 'unchecked':
                results['unchecked_count'] += 1
            elif check_type == 'crossed':
                results['crossed_count'] += 1
    return results
# ----------------------------------------------------------------
# Reasoning Pattern Classification
# ----------------------------------------------------------------
def classify_reasoning_pattern(full_text: str, words: List[str]) -> int:
    """
    Classify the type of reasoning needed for this example.
    0: No reasoning needed
    1: Checkbox verification
    2: Form completion check
    3: Data validation
    4: Compliance check
    5: Error identification
    6: Signature verification
    7: Date validation
    """
    if not full_text:
        return 0
    text_lower = full_text.lower()
    words_lower = [w.lower() for w in words] if words else []
    # Check for signature patterns
    signature_keywords = ['signature', 'sign', 'signed', 'sign here', 'authorized', 'signatory', 'digital signature']
    if any(kw in text_lower for kw in signature_keywords):
        return 6
    # Check for date patterns
    date_keywords = ['date', 'd.o.b', 'birth', 'issued', 'expiry', 'valid until', 'effective date', 'expiration']
    if any(kw in text_lower for kw in date_keywords):
        return 7
    # Checkbox verification
    checkbox_keywords = ['check', 'tick', 'mark', 'checkbox', 'option', 'select', 'choose', 'radio button', 'bullet']
    if any(kw in text_lower for kw in checkbox_keywords):
        return 1
    # Form completion
    form_keywords = ['form', 'complete', 'fill', 'applicant', 'submit', 'required', 'mandatory', 'field', 'section']
    if any(kw in text_lower for kw in form_keywords):
        return 2
    # Data validation
    validation_keywords = ['valid', 'correct', 'accurate', 'verify', 'confirm', 'match', 'consistent', 'error', 'inconsistent']
    if any(kw in text_lower for kw in validation_keywords):
        return 3
    # Compliance check
    compliance_keywords = ['comply', 'regulation', 'standard', 'requirement', 'policy', 'legal', 'mandatory', 'obligation']
    if any(kw in text_lower for kw in compliance_keywords):
        return 4
    # Error identification
    error_keywords = ['error', 'mistake', 'wrong', 'incorrect', 'invalid', 'missing', 'incomplete', 'inconsistent']
    if any(kw in text_lower for kw in error_keywords):
        return 5
    return 0
# ----------------------------------------------------------------
# Enhanced Dataset Loader with Checkmark Detection
# ----------------------------------------------------------------
def load_local_dataset(image_dir: str, json_dir: str, task: str = "sequence") -> List[Dict]:
    """
    Enhanced loader that detects checkmarks and adds metadata.
    """
    examples = []
    if not os.path.exists(json_dir):
        logger.warning("JSON directory does not exist: %s", json_dir)
        return examples
    json_files = [f for f in os.listdir(json_dir) if f.lower().endswith(".json")]
    for json_file in json_files:
        path = os.path.join(json_dir, json_file)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            logger.warning("Failed to read %s: %s", path, e)
            continue
        docs = data.get("documents") or []
        if not docs:
            # legacy layout
            words = data.get("words", [])
            bboxes = data.get("bboxes", [])
            label = data.get("label")
            fname = os.path.splitext(json_file)[0]
            # Try multiple possible image extensions
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif']:
                candidate_path = os.path.join(image_dir, fname + ext)
                if os.path.exists(candidate_path):
                    image_path = candidate_path
                    break
            if image_path is None:
                image_path = os.path.join(image_dir, fname)  # fallback
            entities = data.get("entities")
            relations = data.get("relations")
        else:
            doc = docs[0]
            img_info = doc.get("img", {}) or {}
            fname = img_info.get("fname") or img_info.get("filename") or os.path.splitext(json_file)[0]
            # Try multiple possible image extensions
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif']:
                candidate_path = os.path.join(image_dir, fname + ext)
                if os.path.exists(candidate_path):
                    image_path = candidate_path
                    break
            if image_path is None:
                image_path = os.path.join(image_dir, fname)  # fallback
            words = []
            bboxes = []
            entities = doc.get("entities") or data.get("entities")
            relations = doc.get("relations") or data.get("relations")
            # support different internal structures
            if "document" in doc and isinstance(doc["document"], list):
                for blk in doc["document"]:
                    for w in blk.get("words", []):
                        txt = w.get("text") or w.get("word") or w.get("value")
                        box = w.get("box") or w.get("bbox") or w.get("box_xy")
                        if txt is None or box is None:
                            continue
                        words.append(str(txt))
                        try:
                            bboxes.append([int(float(box[0])), int(float(box[1])), 
                                         int(float(box[2])), int(float(box[3]))])
                        except Exception:
                            bboxes.append([0, 0, 0, 0])
            else:
                # try flat lists
                words = doc.get("words", []) or []
                bboxes = doc.get("bboxes", []) or []
            label = data.get("label") or doc.get("label")
        # basic validation
        if not words or not bboxes or len(words) != len(bboxes):
            logger.warning("Skipping %s: words/bboxes missing or length mismatch", json_file)
            continue
        # extract label - handle different formats
        label = None
        if data.get("label") is not None:
            label = data.get("label")
        elif docs and docs[0].get("label") is not None:
            label = docs[0].get("label")
        elif data.get("category") is not None:
            label = data.get("category")
        elif data.get("type") is not None:
            label = data.get("type")
        # For sequence classification, if no label, assign default based on content
        if label is None and task == "sequence":
            full_text = " ".join(words).lower()
            if any(word in full_text for word in ["invoice", "receipt", "bill", "tax"]):
                label = "invoice"
            elif any(word in full_text for word in ["form", "application", "survey", "questionnaire"]):
                label = "form"
            elif any(word in full_text for word in ["contract", "agreement", "license", "terms"]):
                label = "contract"
            elif any(word in full_text for word in ["report", "statement", "summary", "analysis"]):
                label = "report"
            else:
                label = "other"
        # Convert label to appropriate format
        try:
            if isinstance(label, (int, float)):
                label = int(label)
            elif isinstance(label, str):
                # Keep as string for now, will convert to ID later
                pass
        except Exception:
            label = 0  # Default label
        # Detect checkmarks in the text
        full_text = " ".join(words)
        checkmark_info = detect_checkmarks(full_text)
        # Detect reasoning patterns
        reasoning_label = classify_reasoning_pattern(full_text, words)
        ex = {
            "id": os.path.splitext(json_file)[0],
            "words": words,
            "bboxes": bboxes,
            "image_path": image_path,
            "labels": label,
            "checkmark_info": checkmark_info,
            "reasoning_label": reasoning_label
        }
        if entities:
            ex["entities"] = entities
        if relations:
            ex["relations"] = relations
        examples.append(ex)
    logger.info("Loaded %d examples from %s", len(examples), json_dir)
    return examples
# ----------------------------------------------------------------
# Language Detection
# ----------------------------------------------------------------
def detect_language_safe(text: str) -> str:
    try:
        if not text or len(text.strip()) < 3:
            return "und"
        return detect(text)
    except Exception:
        return "und"
# ----------------------------------------------------------------
# Enhanced Model Definitions
# ----------------------------------------------------------------
class EnhancedLiLTRelationExtraction(nn.Module):
    """
    Enhanced RE model with reasoning capability for checkmark relationships.
    """
    def __init__(self, encoder, num_rel_labels: int = 3, num_checkmark_labels: int = 4, 
                 num_reasoning_labels: int = 8, hidden_size: int = 768):
        super().__init__()
        self.encoder = encoder
        self.num_rel_labels = num_rel_labels
        self.num_checkmark_labels = num_checkmark_labels
        self.num_reasoning_labels = num_reasoning_labels
        # Relation extraction head (pairwise classification)
        self.rel_head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_rel_labels)
        )
        # Checkmark classification head (token-level)
        self.checkmark_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_checkmark_labels)
        )
        # Reasoning classification head (document-level)
        self.reasoning_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_reasoning_labels)
        )
        # Solution generation head (multi-label)
        self.solution_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 8)  # 8 solution types
        )
    def forward(self, input_ids=None, attention_mask=None, bbox=None, 
                labels=None, rel_labels=None, checkmark_labels=None, 
                reasoning_labels=None, **kwargs):
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            **{k: v for k, v in kwargs.items() if k not in ['num_items_in_batch']}
        )
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden]
        batch_size, seq_len, hidden_size = sequence_output.shape
        # --------------------------------------------------
        # 1. Checkmark classification
        # --------------------------------------------------
        checkmark_logits = self.checkmark_head(sequence_output)
        # --------------------------------------------------
        # 2. Reasoning classification (using [CLS] token)
        # --------------------------------------------------
        cls_output = sequence_output[:, 0, :]  # Use [CLS] token
        reasoning_logits = self.reasoning_head(cls_output)
        # --------------------------------------------------
        # 3. Relation extraction
        # --------------------------------------------------
        # Create all possible pairs
        expanded_left = sequence_output.unsqueeze(2).expand(-1, -1, seq_len, -1)  # [b, s, s, h]
        expanded_right = sequence_output.unsqueeze(1).expand(-1, seq_len, -1, -1)  # [b, s, s, h]
        # Combine features: [head, tail, head-tail difference]
        head_tail = torch.cat([expanded_left, expanded_right], dim=-1)
        diff = expanded_left - expanded_right
        pair_features = torch.cat([head_tail, diff], dim=-1)  # [b, s, s, 3*h]
        # Apply relation head
        rel_logits = self.rel_head(pair_features)  # [b, s, s, num_rel_labels]
        # --------------------------------------------------
        # 4. Solution generation (based on CLS and reasoning)
        # --------------------------------------------------
        # Combine CLS with reasoning features
        reasoning_features = self.reasoning_head[:2](cls_output)  # Get features before final layer
        solution_features = torch.cat([cls_output, reasoning_features], dim=-1)
        solution_logits = self.solution_head(solution_features)
        # --------------------------------------------------
        # 5. Compute losses if labels provided
        # --------------------------------------------------
        total_loss = None
        if rel_labels is not None and checkmark_labels is not None and reasoning_labels is not None:
            # Relation loss
            rel_loss_fct = CrossEntropyLoss(ignore_index=-100)
            rel_loss = rel_loss_fct(
                rel_logits.view(-1, self.num_rel_labels),
                rel_labels.view(-1)
            )
            # Checkmark loss
            checkmark_loss_fct = CrossEntropyLoss(ignore_index=-100)
            checkmark_loss = checkmark_loss_fct(
                checkmark_logits.view(-1, self.num_checkmark_labels),
                checkmark_labels.view(-1)
            )
            # Reasoning loss
            reasoning_loss_fct = CrossEntropyLoss()
            reasoning_loss = reasoning_loss_fct(
                reasoning_logits.view(-1, self.num_reasoning_labels),
                reasoning_labels.view(-1)
            )
            # Total loss (weighted)
            total_loss = rel_loss + 0.5 * checkmark_loss + 0.3 * reasoning_loss
        return {
            'loss': total_loss,
            'rel_logits': rel_logits,
            'checkmark_logits': checkmark_logits,
            'reasoning_logits': reasoning_logits,
            'solution_logits': solution_logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None
        }
# ----------------------------------------------------------------
# Enhanced Preprocessing
# ----------------------------------------------------------------
def preprocess_enhanced_relation_extraction(example: Dict, tokenizer: AutoTokenizer, 
                                           max_length: int = 512,
                                           lang_token_map: Optional[Dict[str,str]] = None,
                                           insert_lang_token: bool = True,
                                           relation2id: Optional[Dict[str,int]] = None,
                                           checkmark2id: Optional[Dict[str,int]] = None,
                                           task: str = "enhanced"):
    """
    Enhanced preprocessing that includes:
    - Relation labels
    - Checkmark labels
    - Reasoning labels
    """
    # Basic preprocessing (words, bboxes, input_ids, etc.)
    words = example["words"]
    boxes = example["bboxes"]
    cleaned_words = [normalize_text(str(w)) for w in words]
    # Detect language
    full_text = " ".join(cleaned_words)
    lang = detect_language_safe(full_text)
    # Get special tokens
    cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else pad_id
    # Language token
    lang_token_id = None
    if insert_lang_token and lang_token_map and lang in lang_token_map:
        lang_token = lang_token_map[lang]
        if lang_token in tokenizer.get_vocab():
            lang_token_id = tokenizer.convert_tokens_to_ids(lang_token)
    # Tokenization with word alignment tracking
    input_ids = []
    bboxes_out = []
    attention_mask = []
    word_boundaries = []  # Track which tokens belong to which word
    token_to_word = []    # Map token index to word index
    if cls_id is not None:
        input_ids.append(cls_id)
        bboxes_out.append([0, 0, 0, 0])
        attention_mask.append(1)
        word_boundaries.append(-1)  # Special token
        token_to_word.append(-1)
    if lang_token_id is not None:
        input_ids.append(lang_token_id)
        bboxes_out.append([0, 0, 0, 0])
        attention_mask.append(1)
        word_boundaries.append(-1)
        token_to_word.append(-1)
    # Tokenize each word separately to maintain alignment
    for word_idx, (word, box) in enumerate(zip(cleaned_words, boxes)):
        try:
            tokenized = tokenizer(word, add_special_tokens=False, return_offsets_mapping=False)
            sub_ids = tokenized["input_ids"]
            if not sub_ids:
                sub_ids = [unk_id]
        except Exception:
            sub_ids = [unk_id]
        # Add start boundary marker
        if sub_ids:
            word_boundaries.append(1)  # Start of new word
        else:
            word_boundaries.append(0)
        # Add tokens for this word
        for sub_idx, sub_id in enumerate(sub_ids):
            input_ids.append(sub_id)
            try:
                nb = [max(0, min(1000, int(float(v)))) for v in box]
            except Exception:
                nb = [0, 0, 0, 0]
            bboxes_out.append(nb)
            attention_mask.append(1)
            token_to_word.append(word_idx)
            if sub_idx > 0:
                word_boundaries.append(0)  # Continuation token
    if sep_id is not None:
        input_ids.append(sep_id)
        bboxes_out.append([0, 0, 0, 0])
        attention_mask.append(1)
        word_boundaries.append(-1)
        token_to_word.append(-1)
    # Truncate if necessary
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        bboxes_out = bboxes_out[:max_length]
        attention_mask = attention_mask[:max_length]
        word_boundaries = word_boundaries[:max_length]
        token_to_word = token_to_word[:max_length]
    seq_len = len(input_ids)
    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "bbox": bboxes_out,
        "lang": lang,
        "word_boundaries": word_boundaries,
        "token_to_word": token_to_word
    }
    # Add label for sequence classification
    if task == "sequence" and "labels" in example:
        result["labels"] = example["labels"]
    # Add checkmark labels for enhanced tasks
    if task in ["enhanced", "relation"]:
        # --------------------------------------------------
        # Create checkmark labels
        # --------------------------------------------------
        checkmark2id = checkmark2id or {"none": 0, "checked": 1, "unchecked": 2, "crossed": 3}
        checkmark_labels = [0] * seq_len  # Default: no checkmark
        # Detect checkmarks in original words
        for word_idx, word in enumerate(cleaned_words):
            checkmark_info = detect_checkmarks(word)
            if checkmark_info['has_checkmark']:
                # Find all tokens that belong to this word
                for token_idx, word_map in enumerate(token_to_word):
                    if word_map == word_idx:
                        # Assign checkmark type to the first token of the word
                        if checkmark_info['checkmarks']:
                            checkmark_type = checkmark_info['checkmarks'][0]['type']
                            checkmark_labels[token_idx] = checkmark2id.get(checkmark_type, 0)
                        break
        result["checkmark_labels"] = checkmark_labels
        # --------------------------------------------------
        # Create relation labels
        # --------------------------------------------------
        relation2id = relation2id or {"NO_REL": 0, "HAS_VALUE": 1, "IS_PART_OF": 2}
        rel_labels = np.zeros((seq_len, seq_len), dtype=np.int64)
        entities = example.get("entities", [])
        relations = example.get("relations", [])
        if entities and relations:
            # Map entity spans to token spans
            entity_token_spans = []
            for entity in entities:
                # If entities are already token-level
                if "start" in entity and "end" in entity:
                    start = max(0, min(int(entity["start"]), seq_len - 1))
                    end = max(start + 1, min(int(entity["end"]), seq_len))
                    entity_token_spans.append((start, end))
                else:
                    # Fallback: map to first token of entity
                    entity_token_spans.append((0, 1))
            # Map relations
            for rel in relations:
                try:
                    head_idx = int(rel["head"])
                    tail_idx = int(rel["tail"])
                    rel_type = rel.get("type", "HAS_VALUE")
                    rel_id = relation2id.get(rel_type, 1)
                    if 0 <= head_idx < len(entity_token_spans) and 0 <= tail_idx < len(entity_token_spans):
                        hstart, hend = entity_token_spans[head_idx]
                        tstart, tend = entity_token_spans[tail_idx]
                        # Mark relation between all tokens in the spans
                        for i in range(hstart, min(hend, seq_len)):
                            for j in range(tstart, min(tend, seq_len)):
                                rel_labels[i, j] = rel_id
                except Exception:
                    continue
        result["rel_labels"] = rel_labels
        # --------------------------------------------------
        # Get reasoning label
        # --------------------------------------------------
        reasoning_label = example.get("reasoning_label", 0)
        result["reasoning_labels"] = reasoning_label
    return result
# ----------------------------------------------------------------
# Simple dataset loader (LiLT JSON-ish format) - for classification task
# ----------------------------------------------------------------
def load_local_dataset_classification(image_dir: str, json_dir: str) -> List[Dict]:
    """
    Expects JSONs where each file contains 'documents' -> first doc -> 'img' and either
    'document' blocks with 'words' entries or top-level 'words'/'bboxes'.
    Returns list of examples with fields: id, words (list[str]), bboxes (list[[4]]), image_path, labels
    """
    examples = []
    if not os.path.exists(json_dir):
        logger.warning("JSON directory does not exist: %s", json_dir)
        return examples
    json_files = [f for f in os.listdir(json_dir) if f.lower().endswith(".json")]
    for json_file in json_files:
        path = os.path.join(json_dir, json_file)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            logger.warning("Failed to read %s: %s", path, e)
            continue
        docs = data.get("documents") or []
        if not docs:
            # legacy layout
            words = data.get("words", [])
            bboxes = data.get("bboxes", [])
            label = data.get("label")
            fname = os.path.splitext(json_file)[0]
            image_path = os.path.join(image_dir, fname)
        else:
            doc = docs[0]
            img_info = doc.get("img", {}) or {}
            fname = img_info.get("fname") or img_info.get("filename") or os.path.splitext(json_file)[0]
            image_path = os.path.join(image_dir, fname)
            words = []
            bboxes = []
            # support different internal structures
            if "document" in doc and isinstance(doc["document"], list):
                for blk in doc["document"]:
                    for w in blk.get("words", []):
                        txt = w.get("text") or w.get("word") or w.get("value")
                        box = w.get("box") or w.get("bbox") or w.get("box_xy")
                        if txt is None or box is None:
                            continue
                        words.append(str(txt))
                        # box expected [x1,y1,x2,y2] in pixels
                        try:
                            bboxes.append([int(float(box[0])), int(float(box[1])), 
                                         int(float(box[2])), int(float(box[3]))])
                        except Exception:
                            bboxes.append([0, 0, 0, 0])
            else:
                # try flat lists
                words = doc.get("words", []) or []
                bboxes = doc.get("bboxes", []) or []
            # top-level label fallback
            label = data.get("label") or doc.get("label")
        # basic validation
        if not words or not bboxes or len(words) != len(bboxes):
            logger.warning("Skipping %s: words/bboxes missing or length mismatch", json_file)
            continue
        # extract label
        label = data.get("label") if 'label' in data else label
        if label is None:
            # try various fallbacks
            if docs:
                label = docs[0].get("label")
        try:
            label = int(label) if label is not None else None
        except Exception:
            # keep as-is (some tasks may have strings -> map later)
            pass
        examples.append({
            "id": os.path.splitext(json_file)[0],
            "words": words,
            "bboxes": bboxes,
            "image_path": image_path,
            "labels": label
        })
    logger.info("Loaded %d examples from %s", len(examples), json_dir)
    return examples
# ----------------------------------------------------------------
# Preprocessing for LiLT (tokenize and align bboxes to subtokens) - for classification task
# ----------------------------------------------------------------
def preprocess_example_classification(example: Dict, tokenizer: AutoTokenizer, max_length: int = 512,
                       lang_token_map: Optional[Dict[str,str]] = None, insert_lang_token: bool = True):
    """
    example: {words: [w1,w2,...], bboxes: [[x1,y1,x2,y2], ...], labels: int}
    Returns dict with keys 'input_ids', 'attention_mask', 'bbox' (list of lists), 'labels'
    """
    words = example["words"]
    boxes = example["bboxes"]
    label = example.get("labels")
    # Normalize words and detect language for this document
    cleaned_words = []
    for w in words:
        w2 = normalize_text(str(w))
        cleaned_words.append(w2)
    full_text = " ".join(cleaned_words)
    lang = detect_language_safe(full_text)
    # Optional language token id insertion (if tokenizer knows those tokens)
    lang_token_id = None
    if insert_lang_token and lang_token_map and lang in lang_token_map:
        lang_token = lang_token_map[lang]
        if lang_token in tokenizer.get_vocab():
            lang_token_id = tokenizer.convert_tokens_to_ids(lang_token)
    # Build token lists
    cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else pad_id
    input_ids = []
    bboxes_out = []
    attention_mask = []
    # Optionally add CLS token
    if cls_id is not None:
        input_ids.append(cls_id)
        bboxes_out.append([0,0,0,0])
        attention_mask.append(1)
    # Optionally add language token directly after CLS
    if lang_token_id is not None:
        input_ids.append(lang_token_id)
        bboxes_out.append([0,0,0,0])
        attention_mask.append(1)
    # Tokenize each word into subtokens and replicate bbox
    for word, box in zip(cleaned_words, boxes):
        # If the tokenizer is fast, prefer tokenizer(word, add_special_tokens=False)["input_ids"]
        try:
            tokenized = tokenizer(word, add_special_tokens=False, return_offsets_mapping=False)
            sub_ids = tokenized["input_ids"]
            if not sub_ids:
                # fallback to unk token
                sub_ids = [unk_id]
        except Exception:
            # rare tokenization failure -> fallback
            sub_ids = [unk_id]
        input_ids.extend(sub_ids)
        # Replicate bbox for each subtoken
        for _ in sub_ids:
            # Expect here box in pixel coordinates. Convert to 0..1000 normalized box expected by LiLT models.
            # If user dataset boxes are already normalized to 0..1000, skip normalization.
            bx = box
            # if coordinates appear huge, clamp to 0..1000 as safe fallback
            try:
                nb = [max(0, min(1000, int(v))) for v in bx]
            except Exception:
                nb = [0, 0, 0, 0]
            bboxes_out.append(nb)
            attention_mask.append(1)
    # add SEP
    if sep_id is not None:
        input_ids.append(sep_id)
        bboxes_out.append([0,0,0,0])
        attention_mask.append(1)
    # Truncate
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        bboxes_out = bboxes_out[:max_length]
        attention_mask = attention_mask[:max_length]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "bbox": bboxes_out,
        "labels": label,
        "lang": lang
    }
# ----------------------------------------------------------------
# Enhanced Data Collators
# ----------------------------------------------------------------
@dataclass
class DataCollatorForEnhancedRE:
    tokenizer: AutoTokenizer
    max_length: Optional[int] = 512
    padding_side: str = "right"
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Pad all sequences to same length
        max_len = max(len(f["input_ids"]) for f in features)
        if self.max_length:
            max_len = min(max_len, self.max_length)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        pad_box = [0, 0, 0, 0]
        batch_input_ids = []
        batch_attention = []
        batch_bbox = []
        batch_rel_labels = []
        batch_checkmark_labels = []
        batch_reasoning_labels = []
        for f in features:
            # Basic sequences
            ids = f["input_ids"][:max_len]
            att = f["attention_mask"][:max_len]
            bxs = f["bbox"][:max_len]
            # Pad sequences
            pad_len = max_len - len(ids)
            if pad_len > 0:
                if self.padding_side == "right":
                    ids = ids + [pad_id] * pad_len
                    att = att + [0] * pad_len
                    bxs = bxs + [pad_box] * pad_len
                else:
                    ids = [pad_id] * pad_len + ids
                    att = [0] * pad_len + att
                    bxs = [pad_box] * pad_len + bxs
            batch_input_ids.append(ids)
            batch_attention.append(att)
            batch_bbox.append(bxs)
            # Relation labels
            if "rel_labels" in f:
                rel = f["rel_labels"][:max_len, :max_len]
                rel = np.pad(rel, ((0, max_len - rel.shape[0]), (0, max_len - rel.shape[1])), 
                            constant_values=0)
                batch_rel_labels.append(rel)
            # Checkmark labels
            if "checkmark_labels" in f:
                check = f["checkmark_labels"][:max_len]
                check = check + [0] * (max_len - len(check))  # Pad with 0 (no checkmark)
                batch_checkmark_labels.append(check)
            # Reasoning labels
            if "reasoning_labels" in f:
                batch_reasoning_labels.append(f["reasoning_labels"])
        batch = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
            "bbox": torch.tensor(batch_bbox, dtype=torch.long),
        }
        if batch_rel_labels:
            batch["rel_labels"] = torch.tensor(np.stack(batch_rel_labels), dtype=torch.long)
        if batch_checkmark_labels:
            batch["checkmark_labels"] = torch.tensor(batch_checkmark_labels, dtype=torch.long)
        if batch_reasoning_labels:
            batch["reasoning_labels"] = torch.tensor(batch_reasoning_labels, dtype=torch.long)
        return batch
@dataclass
class DataCollatorForLilt:
    tokenizer: AutoTokenizer
    max_length: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None
    padding_side: str = "right"
    def __call__(self, features: List[Dict[str,Any]]) -> Dict[str, torch.Tensor]:
        # features: list of dicts returned by preprocess_example()
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        bbox = [f["bbox"] for f in features]
        labels = [f.get("labels") for f in features]
        # Verify no missing labels in the batch — Trainer expects labels for loss computation.
        if any(l is None for l in labels):
            # Provide a clear, actionable error instead of silently dropping labels.
            missing_idx = [i for i, l in enumerate(labels) if l is None]
            raise ValueError(
                "Found examples with missing labels in batch. "
                f"Missing indices in batch: {missing_idx}. "
                "All training/eval examples must have labels. Fix your dataset JSONs."
            )
        # compute max len
        max_len = max(len(x) for x in input_ids)
        if self.max_length:
            max_len = min(max_len, self.max_length)
        # pad
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        pad_box = [0,0,0,0]
        batch_input_ids = []
        batch_attention = []
        batch_bbox = []
        for ids, att, bxs in zip(input_ids, attention_mask, bbox):
            cur_len = len(ids)
            if cur_len > max_len:
                ids = ids[:max_len]
                att = att[:max_len]
                bxs = bxs[:max_len]
            pad_len = max_len - len(ids)
            if pad_len > 0:
                if self.padding_side == "right":
                    ids = ids + [pad_id]*pad_len
                    att = att + [0]*pad_len
                    bxs = bxs + [pad_box]*pad_len
                else:
                    ids = [pad_id]*pad_len + ids
                    att = [0]*pad_len + att
                    bxs = [pad_box]*pad_len + bxs
            batch_input_ids.append(ids)
            batch_attention.append(att)
            batch_bbox.append(bxs)
        # convert to tensors
        batch = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
            "bbox": torch.tensor(batch_bbox, dtype=torch.long),
            # Always include labels for Trainer. We validated earlier that none are None.
            "labels": torch.tensor(labels, dtype=torch.long)
        }
        return batch
# ----------------------------------------------------------------
# Enhanced Metrics
# ----------------------------------------------------------------
def compute_enhanced_metrics(eval_pred):
    """Compute metrics for enhanced model with multiple heads."""
    predictions, labels = eval_pred
    metrics = {}
    # Relation metrics
    if 'rel_logits' in predictions:
        rel_preds = np.argmax(predictions['rel_logits'], axis=-1)
        if 'rel_labels' in labels:
            rel_labels = labels['rel_labels']
            # Flatten ignoring padding
            mask = rel_labels != -100
            if mask.any():
                flat_rel_preds = rel_preds.flatten()[mask.flatten()]
                flat_rel_labels = rel_labels.flatten()[mask.flatten()]
                metrics.update({
                    "rel_accuracy": accuracy_score(flat_rel_labels, flat_rel_preds),
                    "rel_f1_micro": f1_score(flat_rel_labels, flat_rel_preds, average="micro", zero_division=0),
                    "rel_f1_macro": f1_score(flat_rel_labels, flat_rel_preds, average="macro", zero_division=0),
                })
    # Checkmark metrics
    if 'checkmark_logits' in predictions:
        check_preds = np.argmax(predictions['checkmark_logits'], axis=-1)
        if 'checkmark_labels' in labels:
            check_labels = labels['checkmark_labels']
            mask = check_labels != -100
            if mask.any():
                flat_check_preds = check_preds.flatten()[mask.flatten()]
                flat_check_labels = check_labels.flatten()[mask.flatten()]
                metrics.update({
                    "checkmark_accuracy": accuracy_score(flat_check_labels, flat_check_preds),
                })
                # Detailed checkmark metrics
                checkmark_report = classification_report(
                    flat_check_labels, flat_check_preds, 
                    target_names=['none', 'checked', 'unchecked', 'crossed'],
                    output_dict=True,
                    zero_division=0
                )
                for class_name, class_metrics in checkmark_report.items():
                    if isinstance(class_metrics, dict):
                        for metric_name, value in class_metrics.items():
                            metrics[f"checkmark_{class_name}_{metric_name}"] = value
    # Reasoning metrics
    if 'reasoning_logits' in predictions:
        reason_preds = np.argmax(predictions['reasoning_logits'], axis=-1)
        if 'reasoning_labels' in labels:
            reason_labels = labels['reasoning_labels']
            metrics.update({
                "reasoning_accuracy": accuracy_score(reason_labels, reason_preds),
                "reasoning_f1_micro": f1_score(reason_labels, reason_preds, average="micro", zero_division=0),
            })
    return metrics
def compute_metrics(eval_pred):
    """Standard metrics for sequence classification."""
    logits, labels = eval_pred
    # Filter out ignored labels (-100)
    mask = labels != -100
    if mask.any():
        logits = logits[mask]
        labels = labels[mask]
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
            "recall_micro": recall_score(labels, preds, average="micro", zero_division=0),
        }
    else:
        return {"accuracy": 0.0, "f1_micro": 0.0, "f1_macro": 0.0}
# ----------------------------------------------------------------
# Post-processing for Reasoning and Solutions
# ----------------------------------------------------------------
class ReasonSolutionGenerator:
    """Generates human-readable reasons and solutions from model outputs."""
    def __init__(self, tokenizer, relation2id=None, checkmark2id=None):
        self.tokenizer = tokenizer
        self.relation2id = relation2id or {"NO_REL": 0, "HAS_VALUE": 1, "IS_PART_OF": 2}
        self.checkmark2id = checkmark2id or {"none": 0, "checked": 1, "unchecked": 2, "crossed": 3}
        # Inverse mappings
        self.id2relation = {v: k for k, v in self.relation2id.items()}
        self.id2checkmark = {v: k for k, v in self.checkmark2id.items()}
        # Reasoning type descriptions
        self.reasoning_descriptions = {
            0: "No specific reasoning needed",
            1: "Checkbox verification required",
            2: "Form completion check needed",
            3: "Data validation required",
            4: "Compliance check needed",
            5: "Error identification required",
            6: "Signature verification needed",
            7: "Date validation required"
        }
        # Solution templates by reasoning type
        self.solution_templates = {
            1: [
                "Verify all checkboxes are properly marked",
                "Ensure required options are selected",
                "Check for missing selections in multiple-choice sections"
            ],
            2: [
                "Complete all mandatory form fields",
                "Fill in missing information sections",
                "Review form for incomplete sections"
            ],
            3: [
                "Validate entered data against source documents",
                "Check numerical values for accuracy",
                "Verify text fields for spelling errors"
            ],
            4: [
                "Ensure document meets regulatory requirements",
                "Check compliance with company policies",
                "Verify required documentation is attached"
            ],
            5: [
                "Identify and correct data entry errors",
                "Review document for inconsistencies",
                "Check calculations and totals"
            ],
            6: [
                "Verify signatures are present and valid",
                "Check signature dates are within range",
                "Ensure authorized personnel have signed"
            ],
            7: [
                "Verify dates are in correct format",
                "Check date ranges for consistency",
                "Ensure dates are not expired or in future"
            ]
        }
    def generate(self, model_output: Dict, input_ids: List[int], 
                 tokens: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive analysis with reasons and solutions.
        """
        if tokens is None:
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        results = {
            "checkmarks": [],
            "relations": [],
            "reasons": [],
            "solutions": [],
            "issues_found": 0,
            "recommendations": []
        }
        # 1. Extract checkmarks
        if 'checkmark_logits' in model_output:
            checkmark_preds = torch.argmax(model_output['checkmark_logits'], dim=-1)
            for i, (token, pred) in enumerate(zip(tokens, checkmark_preds[0])):
                pred = pred.item()
                if pred > 0:  # Not "none"
                    check_type = self.id2checkmark.get(pred, "unknown")
                    results["checkmarks"].append({
                        "token": token,
                        "position": i,
                        "type": check_type,
                        "attention_needed": check_type in ["unchecked", "crossed"]
                    })
                    if check_type in ["unchecked", "crossed"]:
                        results["issues_found"] += 1
        # 2. Extract relations
        if 'rel_logits' in model_output:
            rel_preds = torch.argmax(model_output['rel_logits'], dim=-1)
            seq_len = rel_preds.shape[1]
            for i in range(seq_len):
                for j in range(seq_len):
                    rel_type = rel_preds[0, i, j].item()
                    if rel_type > 0:  # Not "NO_REL"
                        rel_name = self.id2relation.get(rel_type, "UNKNOWN")
                        if i < len(tokens) and j < len(tokens):
                            results["relations"].append({
                                "head": tokens[i],
                                "tail": tokens[j],
                                "relation": rel_name,
                                "head_pos": i,
                                "tail_pos": j
                            })
        # 3. Extract reasoning
        if 'reasoning_logits' in model_output:
            reason_pred = torch.argmax(model_output['reasoning_logits'], dim=-1)
            reason_id = reason_pred[0].item()
            reason_desc = self.reasoning_descriptions.get(reason_id, "Unknown reasoning")
            results["reasons"].append({
                "type_id": reason_id,
                "description": reason_desc,
                "confidence": torch.softmax(model_output['reasoning_logits'][0], dim=-1)[reason_id].item()
            })
        # 4. Generate solutions based on reasoning
        if 'reasoning_logits' in model_output:
            reason_pred = torch.argmax(model_output['reasoning_logits'], dim=-1)
            reason_id = reason_pred[0].item()
            if reason_id > 0:  # Not "no reasoning needed"
                templates = self.solution_templates.get(reason_id, [])
                results["solutions"].extend(templates)
                # Add specific solutions based on checkmarks
                if results["checkmarks"]:
                    unchecked = [c for c in results["checkmarks"] if c["type"] == "unchecked"]
                    crossed = [c for c in results["checkmarks"] if c["type"] == "crossed"]
                    if unchecked:
                        results["solutions"].append(f"Complete {len(unchecked)} unchecked items")
                    if crossed:
                        results["solutions"].append(f"Correct {len(crossed)} crossed/incorrect entries")
        # 5. Generate recommendations
        if results["issues_found"] > 0:
            results["recommendations"].append(f"Found {results['issues_found']} issues requiring attention")
        if results["relations"]:
            results["recommendations"].append(f"Detected {len(results['relations'])} relationships to verify")
        # Add overall assessment
        if results["issues_found"] == 0 and len(results["relations"]) > 0:
            results["recommendations"].append("Document appears to be correctly filled with proper relationships")
        elif results["issues_found"] > 0:
            results["recommendations"].append("Document requires review and correction")
        return results
    def format_report(self, analysis: Dict[str, Any]) -> str:
        """Format analysis as a readable report."""
        lines = []
        lines.append("=" * 60)
        lines.append("DOCUMENT ANALYSIS REPORT")
        lines.append("=" * 60)
        # Summary
        lines.append("\nSUMMARY:")
        lines.append(f"  Issues Found: {analysis['issues_found']}")
        lines.append(f"  Checkmarks Detected: {len(analysis['checkmarks'])}")
        lines.append(f"  Relations Detected: {len(analysis['relations'])}")
        # Checkmarks
        if analysis["checkmarks"]:
            lines.append("\nCHECKMARKS:")
            for check in analysis["checkmarks"]:
                status = "✓ OK" if check["type"] == "checked" else "⚠ NEEDS ATTENTION"
                lines.append(f"  [{check['position']}] '{check['token']}': {check['type']} {status}")
        # Relations
        if analysis["relations"]:
            lines.append("\nRELATIONS:")
            for rel in analysis["relations"]:
                lines.append(f"  '{rel['head']}' → '{rel['tail']}': {rel['relation']}")
        # Reasons
        if analysis["reasons"]:
            lines.append("\nREASONING:")
            for reason in analysis["reasons"]:
                lines.append(f"  {reason['description']} (confidence: {reason['confidence']:.2%})")
        # Solutions
        if analysis["solutions"]:
            lines.append("\nRECOMMENDED SOLUTIONS:")
            for i, solution in enumerate(analysis["solutions"][:5], 1):  # Limit to top 5
                lines.append(f"  {i}. {solution}")
        # Recommendations
        if analysis["recommendations"]:
            lines.append("\nRECOMMENDATIONS:")
            for rec in analysis["recommendations"]:
                lines.append(f"  • {rec}")
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
# ----------------------------------------------------------------
# Custom Trainer for handling missing labels
# ----------------------------------------------------------------
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Override compute_loss to handle missing labels gracefully.
        """
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        # Remove unsupported kwargs
        unsupported = ["num_items_in_batch", "output_attentions", "output_hidden_states"]
        for k in list(kwargs.keys()):
            if k in unsupported:
                kwargs.pop(k)
        outputs = model(**inputs, **kwargs)
        # If we have loss in outputs, use it
        if hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss
        else:
            # Otherwise compute loss from logits and labels
            logits = outputs.get("logits")
            if logits is not None and labels is not None:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), 
                               labels.view(-1))
            else:
                # If no labels, return zero loss
                loss = torch.tensor(0.0, device=model.device)
        return (loss, outputs) if return_outputs else loss
# ----------------------------------------------------------------
# Main Training Script
# ----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train enhanced LiLT model with checkmark detection and reasoning"
    )
    parser.add_argument("--task", type=str, choices=["sequence", "relation", "enhanced"], 
                       default="enhanced", help="Task type")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                       help="Base model (e.g. nielsr/lilt-xlm-roberta-base)")
    parser.add_argument("--train_image_dir", type=str, default="train/images")
    parser.add_argument("--train_json_dir", type=str, default="train/annotations")
    parser.add_argument("--val_image_dir", type=str, default="val/images")
    parser.add_argument("--val_json_dir", type=str, default="val/annotations")
    parser.add_argument("--output_dir", type=str, default="./lilt_enhanced_model")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--num_labels", type=int, default=None, help="Number of labels for classification")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform")
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--use_lilt_model", action="store_true", help="Use actual LiLT model instead of wrapper")
    args = parser.parse_args()
    set_seed(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.overwrite_output_dir and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Loading tokenizer from %s", args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    # Add enhanced special tokens
    enhanced_special_tokens = {
        "additional_special_tokens": [
            # Checkmark tokens
            "<CHECKED>", "<UNCHECKED>", "<CROSSED>",
            # Reasoning tokens
            "<REASON_VERIFY>", "<REASON_COMPLETE>", "<REASON_VALIDATE>",
            "<REASON_COMPLY>", "<REASON_ERROR>", "<REASON_SIGNATURE>", "<REASON_DATE>",
            # Solution tokens
            "<SOLUTION_FIX>", "<SOLUTION_VERIFY>", "<SOLUTION_COMPLETE>",
            "<SOLUTION_VALIDATE>", "<SOLUTION_SIGN>", "<SOLUTION_DATE>",
            # Relation tokens
            "<REL_HAS_VALUE>", "<REL_IS_PART_OF>", "<REL_DERIVES_FROM>",
            # Language tokens (extended)
            "<lang:en>", "<lang:es>", "<lang:fr>", "<lang:de>", 
            "<lang:zh>", "<lang:ja>", "<lang:ko>", "<lang:vi>",
            "<lang:th>", "<lang:ar>", "<lang:hi>", "<lang:pt>"
        ]
    }
    # Check which tokens already exist
    existing_vocab = set(tokenizer.get_vocab().keys())
    tokens_to_add = []
    for token in enhanced_special_tokens["additional_special_tokens"]:
        if token not in existing_vocab:
            tokens_to_add.append(token)
    if tokens_to_add:
        tokenizer.add_special_tokens(enhanced_special_tokens)
        logger.info("Added %d enhanced special tokens to tokenizer", len(tokens_to_add))
    # Language token mapping
    lang_token_map = {
        "en": "<lang:en>", "es": "<lang:es>", "fr": "<lang:fr>", "de": "<lang:de>",
        "zh-cn": "<lang:zh>", "zh-tw": "<lang:zh>", "ja": "<lang:ja>", "ko": "<lang:ko>",
        "vi": "<lang:vi>", "th": "<lang:th>", "ar": "<lang:ar>", "hi": "<lang:hi>",
        "pt": "<lang:pt>"
    }
    logger.info("Loading datasets...")
    train_examples = load_local_dataset(args.train_image_dir, args.train_json_dir, task=args.task)
    val_examples = load_local_dataset(args.val_image_dir, args.val_json_dir, task=args.task)
    if not train_examples:
        logger.error("No training examples found!")
        return
    logger.info("Loaded %d training and %d validation examples", 
                len(train_examples), len(val_examples))
    # Task-specific setup
    if args.task == "sequence":
        # Sequence classification task
        logger.info("Setting up for sequence classification task")
        # Build label map from training examples
        labels = []
        for ex in train_examples:
            label = ex.get("labels")
            if label is not None:
                labels.append(str(label))
        # If no labels found in training, check validation
        if not labels and val_examples:
            for ex in val_examples:
                label = ex.get("labels")
                if label is not None:
                    labels.append(str(label))
        if labels:
            unique_labels = sorted(set(labels))
            label2id = {label: idx for idx, label in enumerate(unique_labels)}
            id2label = {idx: label for label, idx in label2id.items()}
            # Convert string labels to IDs
            for ex in train_examples + val_examples:
                label = ex.get("labels")
                if label is not None:
                    ex["labels"] = label2id.get(str(label), 0)
            logger.info("Label mapping: %s", label2id)
            num_labels = len(label2id)
            # Save label mapping
            with open(os.path.join(args.output_dir, "label_mapping.json"), "w") as f:
                json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)
        else:
            logger.warning("No labels found in dataset, using default 2-class classification")
            num_labels = args.num_labels or 2
            label2id = {"class_0": 0, "class_1": 1}
            id2label = {0: "class_0", 1: "class_1"}
        # Preprocess
        def preprocess_seq(ex):
            return preprocess_example_classification(
                ex, tokenizer, max_length=args.max_length,
                lang_token_map=lang_token_map, insert_lang_token=True
            )
        train_processed = [preprocess_seq(ex) for ex in train_examples]
        val_processed = [preprocess_seq(ex) for ex in val_examples] if val_examples else []
        train_ds = Dataset.from_list(train_processed)
        val_ds = Dataset.from_list(val_processed) if val_processed else None
        # Load base model
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            config=config
        )
        model.resize_token_embeddings(len(tokenizer))
        data_collator = DataCollatorForLilt(tokenizer=tokenizer, max_length=args.max_length)
        compute_fn = compute_metrics
        # Use custom trainer for sequence classification
        trainer_class = CustomTrainer
    elif args.task == "enhanced":
        # Enhanced relation extraction with reasoning
        logger.info("Setting up for enhanced relation extraction task")
        # Build relation mapping
        relation_types = set()
        for ex in train_examples + val_examples:
            rels = ex.get("relations", [])
            for r in rels:
                rel_type = r.get("type", "HAS_VALUE")
                if rel_type:
                    relation_types.add(rel_type)
        if not relation_types:
            relation_types = {"HAS_VALUE", "IS_PART_OF"}
            logger.warning("No relations found in dataset, using default relation types")
        relation2id = {"NO_REL": 0}
        for i, rel_type in enumerate(sorted(relation_types), 1):
            relation2id[rel_type] = i
        # Checkmark mapping
        checkmark2id = {"none": 0, "checked": 1, "unchecked": 2, "crossed": 3}
        logger.info("Relation mapping: %s", relation2id)
        logger.info("Checkmark mapping: %s", checkmark2id)
        # Save mappings
        with open(os.path.join(args.output_dir, "relation_mapping.json"), "w") as f:
            json.dump(relation2id, f, indent=2)
        with open(os.path.join(args.output_dir, "checkmark_mapping.json"), "w") as f:
            json.dump(checkmark2id, f, indent=2)
        # Preprocess
        def preprocess_enhanced(ex):
            return preprocess_enhanced_relation_extraction(
                ex, tokenizer, max_length=args.max_length,
                lang_token_map=lang_token_map, insert_lang_token=True,
                relation2id=relation2id, checkmark2id=checkmark2id,
                task="enhanced"
            )
        train_processed = [preprocess_enhanced(ex) for ex in train_examples]
        val_processed = [preprocess_enhanced(ex) for ex in val_examples] if val_examples else []
        train_ds = Dataset.from_list(train_processed)
        val_ds = Dataset.from_list(val_processed) if val_processed else None
        # Load base encoder
        encoder = AutoModel.from_pretrained(args.model_name_or_path)
        encoder.resize_token_embeddings(len(tokenizer))
        # Create enhanced model
        model = EnhancedLiLTRelationExtraction(
            encoder=encoder,
            num_rel_labels=len(relation2id),
            num_checkmark_labels=len(checkmark2id),
            num_reasoning_labels=8,  # from classify_reasoning_pattern
            hidden_size=encoder.config.hidden_size
        )
        data_collator = DataCollatorForEnhancedRE(tokenizer=tokenizer, max_length=args.max_length)
        compute_fn = compute_enhanced_metrics
        trainer_class = Trainer
    else:  # relation (original)
        logger.info("Setting up for original relation extraction task")
        # Build relation mapping
        relation_types = set()
        for ex in train_examples + val_examples:
            rels = ex.get("relations", [])
            for r in rels:
                rel_type = r.get("type", "HAS_VALUE")
                if rel_type:
                    relation_types.add(rel_type)
        if not relation_types:
            relation_types = {"HAS_VALUE", "IS_PART_OF"}
            logger.warning("No relations found in dataset, using default relation types")
        relation2id = {"NO_REL": 0}
        for i, rel_type in enumerate(sorted(relation_types), 1):
            relation2id[rel_type] = i
        logger.info("Relation mapping: %s", relation2id)
        # Preprocess
        def preprocess_rel(ex):
            return preprocess_enhanced_relation_extraction(
                ex, tokenizer, max_length=args.max_length,
                lang_token_map=lang_token_map, insert_lang_token=True,
                relation2id=relation2id,
                task="relation"
            )
        train_processed = [preprocess_rel(ex) for ex in train_examples]
        val_processed = [preprocess_rel(ex) for ex in val_examples] if val_examples else []
        train_ds = Dataset.from_list(train_processed)
        val_ds = Dataset.from_list(val_processed) if val_processed else None
        # Load original RE model or fallback
        if LILT_AVAILABLE:
            encoder = AutoModel.from_pretrained(args.model_name_or_path)
            encoder.resize_token_embeddings(len(tokenizer))
            model = LiLTRobertaLikeForRelationExtraction.from_pretrained_encoder(
                encoder, num_rel_labels=len(relation2id)
            )
        else:
            logger.warning("Original RE model not available, using enhanced model as fallback")
            encoder = AutoModel.from_pretrained(args.model_name_or_path)
            encoder.resize_token_embeddings(len(tokenizer))
            model = EnhancedLiLTRelationExtraction(
                encoder=encoder,
                num_rel_labels=len(relation2id),
                hidden_size=encoder.config.hidden_size
            )
        data_collator = DataCollatorForEnhancedRE(tokenizer=tokenizer, max_length=args.max_length)
        compute_fn = compute_enhanced_metrics
        trainer_class = Trainer
    # Training arguments - FIXED: Added num_train_epochs parameter
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,  # FIXED: Added num_train_epochs parameter
        warmup_steps=args.warmup_steps,
        eval_strategy="steps" if args.do_eval and val_ds else "no",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=3,
        load_best_model_at_end=True if args.do_eval and val_ds else False,
        metric_for_best_model="f1_micro" if args.task == "sequence" else "rel_f1_micro",
        greater_is_better=True,
        fp16=args.fp16 and torch.cuda.is_available(),
        seed=args.seed,
        remove_unused_columns=False,
        report_to="none",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        label_names=["labels", "rel_labels", "checkmark_labels", "reasoning_labels"] if args.task != "sequence" else ["labels"]
    )
    # Create trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds if args.do_eval else None,
        data_collator=data_collator,
        compute_metrics=compute_fn,
    )
    # Training
    if args.do_train:
        logger.info("Starting training...")
        try:
            train_result = trainer.train()
            # Save metrics
            trainer.save_metrics("train", train_result.metrics)
            logger.info("Training completed with metrics: %s", train_result.metrics)
            # Save model
            logger.info("Saving model and tokenizer...")
            if args.task == "sequence":
                # For sequence classification
                trainer.save_model(args.output_dir)
            else:
                # For enhanced/relation models, save custom components
                # Save encoder separately
                if hasattr(model, 'encoder'):
                    model.encoder.save_pretrained(args.output_dir)
                # Save full model state dict
                torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
                # Save config
                config_dict = {
                    "model_type": "enhanced_lilt" if args.task == "enhanced" else "lilt_re",
                    "num_rel_labels": model.num_rel_labels if hasattr(model, 'num_rel_labels') else 0,
                    "num_checkmark_labels": model.num_checkmark_labels if hasattr(model, 'num_checkmark_labels') else 0,
                    "num_reasoning_labels": model.num_reasoning_labels if hasattr(model, 'num_reasoning_labels') else 0,
                    "hidden_size": model.encoder.config.hidden_size if hasattr(model, 'encoder') else 768
                }
                with open(os.path.join(args.output_dir, "config.json"), "w") as f:
                    json.dump(config_dict, f, indent=2)
            tokenizer.save_pretrained(args.output_dir)
            # Create model config file
            config_dict = {
                "task": args.task,
                "model_type": "enhanced_lilt" if args.task == "enhanced" else "lilt",
                "max_length": args.max_length,
                "has_checkmark_detection": args.task == "enhanced",
                "has_reasoning": args.task == "enhanced",
                "has_relation_extraction": args.task in ["relation", "enhanced"]
            }
            with open(os.path.join(args.output_dir, "model_config.json"), "w") as f:
                json.dump(config_dict, f, indent=2)
            logger.info("Model saved to %s", args.output_dir)
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            import traceback
            traceback.print_exc()
            return
    # Evaluation
    if args.do_eval and val_ds:
        logger.info("Running evaluation...")
        try:
            metrics = trainer.evaluate()
            logger.info("Evaluation metrics: %s", metrics)
            # Save evaluation metrics
            trainer.save_metrics("eval", metrics)
            # Generate sample predictions
            if args.task in ["relation", "enhanced"] and len(val_examples) > 0:
                logger.info("Generating sample predictions...")
                generator = ReasonSolutionGenerator(tokenizer, relation2id=relation2id, checkmark2id=checkmark2id)
                # Test on first few validation examples
                for i, ex in enumerate(val_examples[:3]):
                    processed = preprocess_enhanced_relation_extraction(
                        ex, tokenizer, max_length=args.max_length,
                        lang_token_map=lang_token_map, insert_lang_token=True,
                        task=args.task
                    )
                    # Get model prediction
                    with torch.no_grad():
                        inputs = {k: torch.tensor([v]).to(model.device) 
                                 for k, v in processed.items() if k in ['input_ids', 'attention_mask', 'bbox']}
                        outputs = model(**inputs)
                    # Generate analysis
                    tokens = tokenizer.convert_ids_to_tokens(processed['input_ids'])
                    analysis = generator.generate(outputs, processed['input_ids'], tokens)
                    # Save sample analysis
                    report = generator.format_report(analysis)
                    sample_file = os.path.join(args.output_dir, f"sample_analysis_{i}.txt")
                    with open(sample_file, "w", encoding="utf-8") as f:
                        f.write(f"Example ID: {ex['id']}\n")
                        f.write(f"Original Text: {' '.join(ex['words'][:20])}...\n")
                        f.write(report)
                    logger.info("Saved sample analysis to %s", sample_file)
        except Exception as e:
            logger.error(f"Evaluation failed with error: {e}")
            import traceback
            traceback.print_exc()
    logger.info("Done!")
if __name__ == "__main__":
    main()
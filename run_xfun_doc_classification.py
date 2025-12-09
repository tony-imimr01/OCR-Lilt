#!/usr/bin/env python3
"""
run_xfun_doc_classification09.py

Fixed LiLT multilingual document classification training script.

Key fixes:
 - Added missing arguments: --num_labels and --fp16
 - DataCollatorForLilt enforces that every example in a training batch has a label
   (raises a clear ValueError when missing).
 - LiltWrapperForClassification removes unsupported kwargs (e.g. 'num_items_in_batch')
   before forwarding to the base model, and explicitly computes CrossEntropy loss so
   the HuggingFace Trainer receives a loss tensor.

Usage:
 python run_xfun_doc_classification09.py \
   --model_name_or_path nielsr/lilt-xlm-roberta-base \
   --train_image_dir ./nielsr/test03/train \
   --train_json_dir ./nielsr/test03/train \
   --val_image_dir ./nielsr/test03/val \
   --val_json_dir ./nielsr/test03/val \
   --output_dir ./lilt_multilingual_out \
   --max_steps 2000 \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --do_train
"""
import os
import json
import logging
import shutil
import argparse
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np
import torch
from datasets import Dataset
from langdetect import detect, DetectorFactory
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
import unicodedata

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
# LiLT Model Imports
# ----------------------------------------------------------------
try:
    # First try relative import (if this script is in a package)
    from .models.LiLTRobertaLike import (
        LiLTRobertaLikeConfig,
        LiLTRobertaLikeForTokenClassification,
        LiLTRobertaLikeForSequenceClassification
    )
    LILT_AVAILABLE = True
    logger.info("Successfully imported LiLT models from .models.LiLTRobertaLike")
except ImportError:
    try:
        # Try absolute import from models directory
        from models.LiLTRobertaLike import (
            LiLTRobertaLikeConfig,
            LiLTRobertaLikeForTokenClassification,
            LiLTRobertaLikeForSequenceClassification
        )
        LILT_AVAILABLE = True
        logger.info("Successfully imported LiLT models from models.LiLTRobertaLike")
    except ImportError:
        try:
            # Try direct import
            from LiLTRobertaLike import (
                LiLTRobertaLikeConfig,
                LiLTRobertaLikeForTokenClassification,
                LiLTRobertaLikeForSequenceClassification
            )
            LILT_AVAILABLE = True
            logger.info("Successfully imported LiLT models from LiLTRobertaLike")
        except ImportError:
            # Fallback implementations
            LILT_AVAILABLE = False
            logger.warning("LiLT models not found. Using standard transformers models.")
            
            # Create dummy classes for type hints
            class LiLTRobertaLikeConfig:
                def __init__(self, model_path, **kwargs):
                    self.model_path = model_path
                
                @classmethod
                def from_pretrained(cls, model_path, **kwargs):
                    return cls(model_path, **kwargs)
            
            class LiLTRobertaLikeForTokenClassification:
                def __init__(self, config):
                    self.config = config
                
                @classmethod
                def from_pretrained(cls, model_path, config=None):
                    return cls(config or LiLTRobertaLikeConfig(model_path))
                
                def to(self, device):
                    return self
                
                def eval(self):
                    return self
                
                def forward(self, *args, **kwargs):
                    return None
            
            class LiLTRobertaLikeForSequenceClassification:
                def __init__(self, config):
                    self.config = config
                
                @classmethod
                def from_pretrained(cls, model_path, config=None):
                    return cls(config or LiLTRobertaLikeConfig(model_path))
                
                def to(self, device):
                    return self
                
                def eval(self):
                    return self
                
                def forward(self, *args, **kwargs):
                    return None


# ----------------------------------------------------------------
# Simple dataset loader (LiLT JSON-ish format)
# ----------------------------------------------------------------
def load_local_dataset(image_dir: str, json_dir: str) -> List[Dict]:
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
                            bboxes.append([int(float(box[0])), int(float(box[1])), int(float(box[2])), int(float(box[3]))])
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
# Utilities
# ----------------------------------------------------------------
def normalize_text(s: str) -> str:
    # Unicode normalize and strip/control char removal
    s2 = unicodedata.normalize("NFKC", s)
    # collapse whitespace
    s2 = " ".join(s2.split())
    return s2


def detect_language_safe(text: str) -> str:
    try:
        # langdetect can throw on short or noisy text; catch and default to 'und'
        if not text or len(text.strip()) < 3:
            return "und"
        return detect(text)
    except Exception:
        return "und"


# ----------------------------------------------------------------
# Preprocessing for LiLT (tokenize and align bboxes to subtokens)
# ----------------------------------------------------------------
def preprocess_example(example: Dict, tokenizer: AutoTokenizer, max_length: int = 512,
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
# Data collator for padding input_ids + bbox (LiLT) — FIXED
# ----------------------------------------------------------------
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
# Metrics
# ----------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_micro": f1_score(labels, preds, average="micro"),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "precision_micro": precision_score(labels, preds, average="micro"),
        "recall_micro": recall_score(labels, preds, average="micro"),
    }


# ----------------------------------------------------------------
# Wrapper model to ensure Trainer sees loss and accepts 'bbox' kwarg
# ----------------------------------------------------------------
class LiltWrapperForClassification(torch.nn.Module):
    """
    Wraps an AutoModelForSequenceClassification instance, accepts bbox kwarg (ignored by base)
    and returns a SequenceClassifierOutput with computed CrossEntropy loss (if labels provided).
    Also strips unsupported kwargs passed by Trainer (e.g. num_items_in_batch).
    """
    def __init__(self, base_model: AutoModelForSequenceClassification):
        super().__init__()
        self.base = base_model
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, bbox=None, labels=None, **kwargs):
        # Remove kwargs that some HF components (Trainer/Accelerate) may pass but base model
        # does not expect (this was causing TypeError: unexpected keyword argument 'num_items_in_batch')
        unsupported = ["num_items_in_batch", "output_attentions", "output_hidden_states"]
        for k in list(kwargs.keys()):
            if k in unsupported:
                kwargs.pop(k)

        # Call base model without bbox (most base models will ignore unexpected kwargs).
        # If your base model accepts bbox, you can adapt this call accordingly.
        out = self.base(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        logits = out.logits

        loss = None
        if labels is not None:
            # move labels to same device as logits and ensure correct dtype
            if isinstance(labels, np.ndarray):
                labels_t = torch.tensor(labels, dtype=torch.long, device=logits.device)
            elif not isinstance(labels, torch.Tensor):
                labels_t = torch.tensor(labels, dtype=torch.long, device=logits.device)
            else:
                labels_t = labels.to(logits.device)

            loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels_t.view(-1))

        # Return SequenceClassifierOutput so Trainer can parse loss and logits.
        return SequenceClassifierOutput(loss=loss, logits=logits)


# ----------------------------------------------------------------
# LiLT-specific Wrapper for Classification (if using actual LiLT model)
# ----------------------------------------------------------------
class LiltCustomWrapperForClassification(torch.nn.Module):
    """
    Wrapper for actual LiLT models that accept bbox input.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, bbox=None, labels=None, **kwargs):
        # Remove unsupported kwargs
        unsupported = ["num_items_in_batch", "output_attentions", "output_hidden_states"]
        for k in list(kwargs.keys()):
            if k in unsupported:
                kwargs.pop(k)

        # Call LiLT model with bbox
        out = self.base(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            bbox=bbox,
            **kwargs
        )
        
        # LiLT models return different outputs - handle both cases
        if hasattr(out, 'logits'):
            logits = out.logits
        else:
            # Assume the model returns a tuple with logits as first element
            logits = out[0]

        loss = None
        if labels is not None:
            # move labels to same device as logits and ensure correct dtype
            if isinstance(labels, np.ndarray):
                labels_t = torch.tensor(labels, dtype=torch.long, device=logits.device)
            elif not isinstance(labels, torch.Tensor):
                labels_t = torch.tensor(labels, dtype=torch.long, device=logits.device)
            else:
                labels_t = labels.to(logits.device)

            loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels_t.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)


# ----------------------------------------------------------------
# Main training script
# ----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train multilingual LiLT document classifier")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Base model (e.g. nielsr/lilt-xlm-roberta-base or your LiLT checkpoint)")
    parser.add_argument("--train_image_dir", type=str, default="train/images")
    parser.add_argument("--train_json_dir", type=str, default="train/annotations")
    parser.add_argument("--val_image_dir", type=str, default="val/images")
    parser.add_argument("--val_json_dir", type=str, default="val/annotations")
    parser.add_argument("--output_dir", type=str, default="./lilt_multilingual_model")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--max_length", type=int, default=512)
    # Added missing arguments
    parser.add_argument("--num_labels", type=int, default=2, help="Number of labels for classification")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    # Optional: use actual LiLT model if available
    parser.add_argument("--use_lilt_model", action="store_true", help="Use actual LiLT model instead of wrapper")
    
    args = parser.parse_args()

    set_seed(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.overwrite_output_dir and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading tokenizer from %s", args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    # add language tokens (optional small set) to tokenizer if not present
    lang_token_map = {}
    add_lang_tokens = []
    for l in ["en","es","fr","de","zh-cn","zh-tw","ja","ko","vi","th","ar","hi"]:
        tok = f"<lang:{l}>"
        if tok not in tokenizer.get_vocab():
            add_lang_tokens.append(tok)
        lang_token_map[l] = tok
    if add_lang_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": add_lang_tokens})
        logger.info("Added %d language special tokens to tokenizer", len(add_lang_tokens))

    logger.info("Loading model config and model from %s", args.model_name_or_path)
    
    # Check if we should use actual LiLT model
    if args.use_lilt_model and LILT_AVAILABLE:
        logger.info("Using actual LiLT model for sequence classification")
        # Use LiLTRobertaLikeConfig
        config = LiLTRobertaLikeConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_labels)
        # Load LiLT sequence classification model
        base_model = LiLTRobertaLikeForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
        # Wrap with LiLT-specific wrapper
        model = LiltCustomWrapperForClassification(base_model)
    else:
        # Use standard transformers approach
        logger.info("Using standard transformers model with wrapper")
        # Use num_labels from command line argument
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_labels)
        # load base sequence classifier
        base_model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
        # Wrap base_model so Trainer will receive loss (and we accept bbox)
        model = LiltWrapperForClassification(base_model)
    
    # resize token embeddings if tokenizer changed
    try:
        if hasattr(model.base, 'resize_token_embeddings'):
            model.base.resize_token_embeddings(len(tokenizer))
        elif hasattr(model, 'resize_token_embeddings'):
            model.resize_token_embeddings(len(tokenizer))
    except Exception:
        # some models may not expose resize_token_embeddings in the same way
        logger.debug("Could not call resize_token_embeddings; proceeding without resizing")

    # Load datasets
    logger.info("Loading training dataset")
    train_examples = load_local_dataset(args.train_image_dir, args.train_json_dir)
    logger.info("Loading validation dataset")
    val_examples = load_local_dataset(args.val_image_dir, args.val_json_dir)

    # Quick dataset label checks: ensure training examples have labels
    missing_train = [ex["id"] for ex in train_examples if ex.get("labels") is None]
    if missing_train:
        logger.error("Found %d training examples with missing labels. Example ids (first 20): %s",
                     len(missing_train), missing_train[:20])
        raise ValueError("All training examples must have labels. Fix your dataset JSONs.")

    # If validation examples have no labels, we will still proceed but evaluation will be skipped.
    missing_val = [ex["id"] for ex in val_examples if ex.get("labels") is None]
    if missing_val:
        logger.warning("Found %d validation examples with missing labels; evaluation will be skipped. Example ids (first 10): %s",
                       len(missing_val), missing_val[:10])

    # If labels are None or strings, build label mapping
    def ensure_label_map(examples_list):
        labels = [ex.get("labels") for ex in examples_list if ex.get("labels") is not None]
        if not labels:
            return None
        # if labels are strings, build map
        if isinstance(labels[0], str):
            uniq = sorted(list(set(labels)))
            label2id = {l:i for i,l in enumerate(uniq)}
            for ex in examples_list:
                if ex.get("labels") is not None:
                    ex["labels"] = label2id.get(ex["labels"], 0)
            return label2id
        return None

    label_map = ensure_label_map(train_examples) or ensure_label_map(val_examples)
    if label_map:
        logger.info("Built label map: %s", label_map)

    # Preprocess examples into token-level inputs
    logger.info("Tokenizing and aligning bboxes (this may take time)...")
    train_processed = [preprocess_example(ex, tokenizer, max_length=args.max_length,
                                          lang_token_map=lang_token_map, insert_lang_token=True)
                       for ex in train_examples]
    val_processed = [preprocess_example(ex, tokenizer, max_length=args.max_length,
                                        lang_token_map=lang_token_map, insert_lang_token=True)
                     for ex in val_examples]

    # Wrap into datasets
    train_ds = Dataset.from_list(train_processed) if len(train_processed) > 0 else None
    val_ds = Dataset.from_list(val_processed) if len(val_processed) > 0 else None

    # Data collator
    data_collator = DataCollatorForLilt(tokenizer=tokenizer, max_length=args.max_length)

    # Training arguments - use fp16 from command line argument
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=200,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        logging_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        # Use fp16 from command line argument
        fp16=args.fp16 and torch.cuda.is_available(),
        seed=args.seed,
        remove_unused_columns=False,  # important to pass bbox through to collator
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds if (val_ds is not None and len(val_processed)>0 and all(p.get("labels") is not None for p in val_processed)) else None,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if args.do_train:
        logger.info("Starting training...")
        trainer.train()
        logger.info("Saving model and tokenizer...")
        # Save base model weights (the underlying AutoModel weights)
        if hasattr(model, "base") and hasattr(model.base, "save_pretrained"):
            model.base.save_pretrained(args.output_dir)
        else:
            trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    logger.info("Running final evaluation...")
    if trainer.eval_dataset is not None:
        metrics = trainer.evaluate()
        logger.info("Evaluation metrics: %s", metrics)
    else:
        logger.warning("No eval dataset provided or eval dataset lacks labels; skipping evaluation.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
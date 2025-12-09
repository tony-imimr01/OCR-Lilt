#!/usr/bin/env python3
"""
run_xfun_doc_classification08_with_re.py

Extended LiLT multilingual training script with optional Relation Extraction head.

Usage examples:
  # sequence classification (original behavior)
  python run_xfun_doc_classification08_with_re.py \
    --task sequence \
    --model_name_or_path nielsr/lilt-xlm-roberta-base \
    --train_image_dir ./nielsr/test03/train \
    --train_json_dir ./nielsr/test03/train \
    --val_image_dir ./nielsr/test03/val \
    --val_json_dir ./nielsr/test03/val \
    --output_dir ./lilt_multilingual_out \
    --do_train

  # relation extraction training (expects 'entities' and 'relations' in JSONs)
  python run_xfun_doc_classification08_with_re.py \
    --task relation \
    --model_name_or_path nielsr/lilt-xlm-roberta-base \
    --train_image_dir ./data/train/images \
    --train_json_dir ./data/train/json \
    --val_image_dir ./data/val/images \
    --val_json_dir ./data/val/json \
    --output_dir ./lilt_re_out \
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
    AutoModel,
    set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
import unicodedata

# Import from local models module
from models.LiLTRobertaLike import LiLTRobertaLikeForRelationExtraction

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
# Simple dataset loader (LiLT JSON-ish format)
# ----------------------------------------------------------------
def load_local_dataset(image_dir: str, json_dir: str) -> List[Dict]:
    """
    Expects JSONs where each file contains 'documents' -> first doc -> 'img' and either
    'document' blocks with 'words' entries or top-level 'words'/'bboxes'.
    For relation task, JSONs should also include:
      - 'entities': list of {"start": token_start, "end": token_end, "type": "<TYPE>"}  OR entities inside doc["document"]
      - 'relations': list of {"head": entity_idx, "tail": entity_idx, "type": "HAS_VALUE"}
    Returns list of examples with fields: id, words (list[str]), bboxes (list[[4]]), image_path, labels (or None),
    and optional entities/relations.
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
            entities = data.get("entities")
            relations = data.get("relations")
        else:
            doc = docs[0]
            img_info = doc.get("img", {}) or {}
            fname = img_info.get("fname") or img_info.get("filename") or os.path.splitext(json_file)[0]
            image_path = os.path.join(image_dir, fname)
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
                            bboxes.append([int(float(box[0])), int(float(box[1])), int(float(box[2])), int(float(box[3]))])
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

        # extract label
        label = data.get("label") if 'label' in data else label
        if label is None and docs:
            label = docs[0].get("label")
        try:
            label = int(label) if label is not None else None
        except Exception:
            pass

        ex = {
            "id": os.path.splitext(json_file)[0],
            "words": words,
            "bboxes": bboxes,
            "image_path": image_path,
            "labels": label
        }
        if entities:
            ex["entities"] = entities
        if relations:
            ex["relations"] = relations

        examples.append(ex)

    logger.info("Loaded %d examples from %s", len(examples), json_dir)
    return examples

# ----------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------
def normalize_text(s: str) -> str:
    s2 = unicodedata.normalize("NFKC", s)
    s2 = " ".join(s2.split())
    return s2

def detect_language_safe(text: str) -> str:
    try:
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
    Map words + bboxes into token-level input_ids, attention_mask, bbox arrays.
    Returns dict with keys 'input_ids', 'attention_mask', 'bbox', 'labels'(sequence label) and 'lang'.
    Note: This function does NOT build relation labels.
    """
    words = example["words"]
    boxes = example["bboxes"]
    label = example.get("labels")

    cleaned_words = [normalize_text(str(w)) for w in words]
    full_text = " ".join(cleaned_words)
    lang = detect_language_safe(full_text)

    lang_token_id = None
    if insert_lang_token and lang_token_map and lang in lang_token_map:
        lang_token = lang_token_map[lang]
        if lang_token in tokenizer.get_vocab():
            lang_token_id = tokenizer.convert_tokens_to_ids(lang_token)

    cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else pad_id

    input_ids = []
    bboxes_out = []
    attention_mask = []

    if cls_id is not None:
        input_ids.append(cls_id)
        bboxes_out.append([0,0,0,0])
        attention_mask.append(1)

    if lang_token_id is not None:
        input_ids.append(lang_token_id)
        bboxes_out.append([0,0,0,0])
        attention_mask.append(1)

    for word, box in zip(cleaned_words, boxes):
        try:
            tokenized = tokenizer(word, add_special_tokens=False, return_offsets_mapping=False)
            sub_ids = tokenized["input_ids"]
            if not sub_ids:
                sub_ids = [unk_id]
        except Exception:
            sub_ids = [unk_id]

        input_ids.extend(sub_ids)
        for _ in sub_ids:
            try:
                nb = [max(0, min(1000, int(float(v)))) for v in box]
            except Exception:
                nb = [0,0,0,0]
            bboxes_out.append(nb)
            attention_mask.append(1)

    if sep_id is not None:
        input_ids.append(sep_id)
        bboxes_out.append([0,0,0,0])
        attention_mask.append(1)

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
# Relation preprocessing helper: map entity-level relations -> token-level grid
# ----------------------------------------------------------------
def preprocess_for_relation_extraction(example: Dict, tokenizer: AutoTokenizer, max_length: int = 512,
                                       lang_token_map: Optional[Dict[str,str]] = None, insert_lang_token: bool = True,
                                       relation2id: Optional[Dict[str,int]] = None):
    """
    example is expected to contain:
      - words, bboxes (same as preprocess_example)
      - entities: list of {"start": token_idx_start, "end": token_idx_end, "type": ...}
        where start inclusive, end exclusive, indices are token-level positions AFTER tokenization.
        (If your entities are word-level spans, you must align to token indices externally.)
      - relations: list of {"head": entity_idx, "tail": entity_idx, "type": "HAS_VALUE"}

    Returns a dict with the same keys as preprocess_example plus:
      - rel_labels: np.ndarray (seq_len, seq_len) with integer relation ids (0 = no relation)
    """
    base = preprocess_example(example, tokenizer, max_length=max_length, lang_token_map=lang_token_map, insert_lang_token=insert_lang_token)
    seq_len = len(base["input_ids"])
    # default relation mapping
    relation2id = relation2id or {"NO_REL": 0, "HAS_VALUE": 1}

    rel_labels = np.zeros((seq_len, seq_len), dtype=np.int64)

    # If example provides relations and entities, map them.
    entities = example.get("entities", [])
    relations = example.get("relations", [])

    # If entities are given as token index spans they should fit inside seq_len.
    # If they are word-level spans or char-level spans, user must convert them externally.
    if entities and relations:
        for rel in relations:
            try:
                head_idx = int(rel["head"])
                tail_idx = int(rel["tail"])
                rel_type = rel.get("type", "HAS_VALUE")
                rel_id = relation2id.get(rel_type, relation2id.get("HAS_VALUE", 1))
                if 0 <= head_idx < len(entities) and 0 <= tail_idx < len(entities):
                    head_ent = entities[head_idx]
                    tail_ent = entities[tail_idx]
                    hstart = int(head_ent["start"])
                    hend = int(head_ent["end"])
                    tstart = int(tail_ent["start"])
                    tend = int(tail_ent["end"])
                    # clamp
                    hstart = max(0, min(hstart, seq_len-1))
                    hend = max(hstart+1, min(hend, seq_len))
                    tstart = max(0, min(tstart, seq_len-1))
                    tend = max(tstart+1, min(tend, seq_len))
                    for i in range(hstart, hend):
                        for j in range(tstart, tend):
                            rel_labels[i, j] = rel_id
            except Exception:
                continue

    base["rel_labels"] = rel_labels
    return base

# ----------------------------------------------------------------
# Data collators
# ----------------------------------------------------------------
@dataclass
class DataCollatorForLilt:
    tokenizer: AutoTokenizer
    max_length: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None
    padding_side: str = "right"

    def __call__(self, features: List[Dict[str,Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        bbox = [f["bbox"] for f in features]
        labels = [f.get("labels") for f in features]

        # For sequence classification we require labels
        if any(l is None for l in labels):
            missing_idx = [i for i, l in enumerate(labels) if l is None]
            raise ValueError(
                "Found examples with missing labels in batch. "
                f"Missing indices in batch: {missing_idx}. "
                "All training/eval examples must have labels. Fix your dataset JSONs."
            )

        max_len = max(len(x) for x in input_ids)
        if self.max_length:
            max_len = min(max_len, self.max_length)

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

        batch = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
            "bbox": torch.tensor(batch_bbox, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }
        return batch

@dataclass
class DataCollatorForLiltRE:
    tokenizer: AutoTokenizer
    max_length: Optional[int] = 512
    padding_side: str = "right"

    def __call__(self, features: List[Dict[str,Any]]) -> Dict[str, torch.Tensor]:
        # Reuse basic collating (without sequence label validation)
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        bbox = [f["bbox"] for f in features]
        # pad to same max len
        max_len = max(len(x) for x in input_ids)
        if self.max_length:
            max_len = min(max_len, self.max_length)

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

        batch = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
            "bbox": torch.tensor(batch_bbox, dtype=torch.long),
        }

        # Build rel_labels if present in features
        if "rel_labels" in features[0]:
            rels = [f["rel_labels"] for f in features]
            # Each rel is (seq_len, seq_len) numpy array; pad to max_len x max_len
            rel_batch = []
            for r in rels:
                cur = np.array(r)
                cur_h, cur_w = cur.shape
                pad_h = max_len - cur_h
                pad_w = max_len - cur_w
                if pad_h > 0 or pad_w > 0:
                    cur = np.pad(cur, ((0,pad_h),(0,pad_w)), constant_values=0)
                else:
                    cur = cur[:max_len,:max_len]
                rel_batch.append(cur)
            batch["rel_labels"] = torch.tensor(np.stack(rel_batch), dtype=torch.long)
        return batch

# ----------------------------------------------------------------
# Metrics (sequence classification)
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
# Main training script
# ----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train multilingual LiLT document classifier / relation extraction")
    parser.add_argument("--task", type=str, choices=["sequence","relation"], default="sequence",
                        help="Task to train: sequence (document classification) or relation (relation extraction)")
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
    args = parser.parse_args()

    set_seed(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.overwrite_output_dir and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading tokenizer from %s", args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    # add language tokens (optional)
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

    logger.info("Loading model config and base encoder from %s", args.model_name_or_path)
    base_encoder = AutoModel.from_pretrained(args.model_name_or_path)

    # If sequence task: wrap into classifier as before
    # If relation task: build RE model using base encoder and relation head
    logger.info("Loading datasets")
    train_examples = load_local_dataset(args.train_image_dir, args.train_json_dir)
    val_examples = load_local_dataset(args.val_image_dir, args.val_json_dir)

    # Quick checks
    if args.task == "sequence":
        missing_train = [ex["id"] for ex in train_examples if ex.get("labels") is None]
        if missing_train:
            logger.error("Found %d training examples with missing labels. Example ids (first 20): %s",
                         len(missing_train), missing_train[:20])
            raise ValueError("All training examples must have labels for sequence classification.")
        # build label map if needed
        def ensure_label_map(examples_list):
            labels = [ex.get("labels") for ex in examples_list if ex.get("labels") is not None]
            if not labels:
                return None
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

        # Tokenize
        train_processed = [preprocess_example(ex, tokenizer, max_length=args.max_length,
                                              lang_token_map=lang_token_map, insert_lang_token=True)
                           for ex in train_examples]
        val_processed = [preprocess_example(ex, tokenizer, max_length=args.max_length,
                                            lang_token_map=lang_token_map, insert_lang_token=True)
                         for ex in val_examples]

        train_ds = Dataset.from_list(train_processed) if len(train_processed) > 0 else None
        val_ds = Dataset.from_list(val_processed) if len(val_processed) > 0 else None

        # Build a classifier from AutoModelForSequenceClassification
        # we keep the standard classifier as a baseline
        base_config = AutoConfig.from_pretrained(args.model_name_or_path)
        base_model_cls = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=base_config)
        # resize embeddings if tokenizer was extended
        try:
            base_model_cls.resize_token_embeddings(len(tokenizer))
        except Exception:
            logger.debug("Could not resize token embedding matrix; proceeding.")

        # Wrap as simple model that accepts bbox in forward (we reuse LiltWrapperForClassification idea)
        class SeqModelWrapper(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base

            def forward(self, input_ids=None, attention_mask=None, bbox=None, labels=None, **kwargs):
                # drop bbox (base model may not accept it)
                out = self.base(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **{k:v for k,v in kwargs.items() if k!='num_items_in_batch'})
                return out

        model = SeqModelWrapper(base_model_cls)
        data_collator = DataCollatorForLilt(tokenizer=tokenizer, max_length=args.max_length)
        compute_fn = compute_metrics
        eval_dataset = val_ds if (val_ds is not None and len(val_processed)>0 and all(p.get("labels") is not None for p in val_processed)) else None

    else:
        # relation task
        # Build relation2id mapping across train_examples + val_examples
        relation_types = set()
        for ex in train_examples + val_examples:
            rels = ex.get("relations") or []
            for r in rels:
                relation_types.add(r.get("type", "HAS_VALUE"))
        if not relation_types:
            # default small mapping
            relation2id = {"NO_REL": 0, "HAS_VALUE": 1}
        else:
            relation2id = {"NO_REL": 0}
            for i, r in enumerate(sorted(relation_types), start=1):
                relation2id[r] = i
        logger.info("Relation mapping: %s", relation2id)

        # Preprocess for relation extraction
        train_processed = [preprocess_for_relation_extraction(ex, tokenizer, max_length=args.max_length,
                                                              lang_token_map=lang_token_map, insert_lang_token=True,
                                                              relation2id=relation2id)
                           for ex in train_examples]
        val_processed = [preprocess_for_relation_extraction(ex, tokenizer, max_length=args.max_length,
                                                            lang_token_map=lang_token_map, insert_lang_token=True,
                                                            relation2id=relation2id)
                         for ex in val_examples]

        train_ds = Dataset.from_list(train_processed) if len(train_processed) > 0 else None
        val_ds = Dataset.from_list(val_processed) if len(val_processed) > 0 else None

        # Build relation model from base_encoder using the new module
        encoder = base_encoder
        # resize encoder token embeddings if tokenizer changed
        try:
            if hasattr(encoder, "resize_token_embeddings"):
                encoder.resize_token_embeddings(len(tokenizer))
        except Exception:
            logger.debug("Could not resize encoder token embeddings; proceeding.")

        # Use the imported model class
        model = LiLTRobertaLikeForRelationExtraction.from_pretrained_encoder(
            encoder, 
            num_rel_labels=len(relation2id)
        )
        
        data_collator = DataCollatorForLiltRE(tokenizer=tokenizer, max_length=args.max_length)
        # metrics for relation are task-specific: disabling compute_metrics by default
        compute_fn = None
        eval_dataset = val_ds if (val_ds is not None and len(val_processed) > 0 and "rel_labels" in val_processed[0]) else None

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=200,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=200,
        save_steps=200,
        logging_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset is not None else False,
        metric_for_best_model="f1_micro" if compute_fn else None,
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        seed=args.seed,
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_fn,
    )

    if args.do_train:
        logger.info("Starting training...")
        trainer.train()
        logger.info("Saving model and tokenizer...")
        # For relation model save encoder weights if possible
        if args.task == "sequence":
            # base_model stored in wrapper
            try:
                trainer.save_model(args.output_dir)
            except Exception:
                pass
        else:
            # save encoder
            try:
                if hasattr(model, "encoder") and hasattr(model.encoder, "save_pretrained"):
                    model.encoder.save_pretrained(args.output_dir)
                else:
                    trainer.save_model(args.output_dir)
            except Exception:
                trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    logger.info("Running final evaluation...")
    if trainer.eval_dataset is not None:
        metrics = trainer.evaluate()
        logger.info("Evaluation metrics: %s", metrics)
    else:
        logger.warning("No eval dataset provided or eval dataset lacks required labels; skipping evaluation.")

    logger.info("Done.")

if __name__ == "__main__":
    main()
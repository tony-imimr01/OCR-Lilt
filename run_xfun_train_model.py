#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for LiLT model on document analysis tasks - Simplified Version
"""

import argparse
import json
import os
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    set_seed,
)
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def normalize_bbox(bbox, width, height):
    """Normalize bbox to 0-1000 range expected by LiLT"""
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]

def detect_dataset_structure(base_dir):
    """
    Detect the dataset structure and return the correct paths
    Handles common structures like:
    - XFUND: train/images/, train/annotations/
    - Simple: train/ (contains both images and JSON)
    - Custom: user-defined structure
    """
    logger.info(f"Detecting dataset structure in: {base_dir}")
    
    # Check for XFUND structure
    xfund_structure = {
        'images': os.path.join(base_dir, 'images'),
        'annotations': os.path.join(base_dir, 'annotations'),
        'type': 'xfund'
    }
    
    if os.path.exists(xfund_structure['images']) and os.path.exists(xfund_structure['annotations']):
        logger.info("Detected XFUND structure")
        return xfund_structure
    
    # Check for simple structure (all files in one directory)
    json_files = [f for f in os.listdir(base_dir) if f.lower().endswith('.json')]
    image_files = [f for f in os.listdir(base_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if json_files and image_files:
        simple_structure = {
            'images': base_dir,
            'annotations': base_dir,
            'type': 'simple'
        }
        logger.info("Detected simple structure (all files in one directory)")
        return simple_structure
    
    # Check for subdirectories
    subdirs = [d for d in os.listdir(base_dir) 
              if os.path.isdir(os.path.join(base_dir, d)) and d not in ['__pycache__', '.git']]
    
    for subdir in subdirs:
        subdir_path = os.path.join(base_dir, subdir)
        json_files = [f for f in os.listdir(subdir_path) if f.lower().endswith('.json')]
        if json_files:
            # This is likely the annotations directory
            images_dir = None
            # Look for images directory at same level
            for sibling in subdirs:
                sibling_path = os.path.join(base_dir, sibling)
                if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(sibling_path)):
                    images_dir = sibling_path
                    break
            
            if images_dir:
                custom_structure = {
                    'images': images_dir,
                    'annotations': subdir_path,
                    'type': 'custom'
                }
                logger.info(f"Detected custom structure: images={images_dir}, annotations={subdir_path}")
                return custom_structure
    
    # Default fallback - try to use base_dir for both
    logger.warning("Could not detect clear structure, using base directory for both images and annotations")
    return {
        'images': base_dir,
        'annotations': base_dir,
        'type': 'fallback'
    }

def inspect_json_structure(json_file):
    """Inspect JSON file structure and return the detected format"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"\nInspecting JSON structure of: {json_file}")
        logger.info(f"Type: {type(data)}")
        
        if isinstance(data, dict):
            logger.info(f"Keys: {list(data.keys())}")
            
            # Check for custom XFUND structure
            if "documents" in data and isinstance(data["documents"], list) and len(data["documents"]) > 0:
                doc = data["documents"][0]
                logger.info(f"Document keys: {list(doc.keys())}")
                
                # Check for nested document structure
                if "document" in doc and isinstance(doc["document"], list) and len(doc["document"]) > 0:
                    logger.info(f"Document item keys: {list(doc['document'][0].keys())}")
                    return "custom_xfund"
                
                # Check for form structure
                if "form" in doc and isinstance(doc["form"], list) and len(doc["form"]) > 0:
                    logger.info(f"Form item keys: {list(doc['form'][0].keys())}")
                    return "xfund"
            
            # Check for FUNSD structure
            if "form" in data and isinstance(data["form"], list):
                logger.info(f"FUNSD form item keys: {list(data['form'][0].keys())}")
                return "funsd"
            
            # Check for simple key-value structure
            if "fields" in data or "annotations" in data:
                return "simple_kv"
        
        elif isinstance(data, list) and len(data) > 0:
            logger.info(f"List item type: {type(data[0])}")
            if isinstance(data[0], dict):
                logger.info(f"List item keys: {list(data[0].keys())}")
            return "list"
        
        return "unknown"
    
    except Exception as e:
        logger.error(f"Error inspecting {json_file}: {e}")
        return "error"

def load_xfund_examples(image_dir, annotation_dir, lang="en", debug_samples=3):
    """
    Load examples from XFUND format with structure detection
    """
    examples = []
    
    # First, inspect the structure of the first few JSON files
    json_files = [f for f in os.listdir(annotation_dir) if f.lower().endswith('.json')]
    if not json_files:
        logger.warning(f"No JSON files found in annotation directory: {annotation_dir}")
        return examples
    
    logger.info(f"\n{'='*80}")
    logger.info("STRUCTURE INSPECTION - First few JSON files:")
    logger.info(f"{'='*80}")
    
    inspected_types = []
    for i, json_file in enumerate(json_files[:debug_samples]):
        json_path = os.path.join(annotation_dir, json_file)
        struct_type = inspect_json_structure(json_path)
        inspected_types.append(struct_type)
        logger.info(f"File {i+1}/{min(debug_samples, len(json_files))}: {json_file} -> {struct_type}")
    
    logger.info(f"{'='*80}")
    logger.info(f"Most common structure type: {max(set(inspected_types), key=inspected_types.count)}")
    logger.info(f"{'='*80}\n")
    
    # Determine the structure type from inspection
    structure_type = max(set(inspected_types), key=inspected_types.count) if inspected_types else "custom_xfund"
    
    # Load based on detected structure
    for json_file in tqdm(json_files, desc=f"Loading {lang} examples"):
        json_path = os.path.join(annotation_dir, json_file)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if structure_type == "custom_xfund":
                # Handle custom XFUND structure with nested "document" field
                if "documents" not in data or not isinstance(data["documents"], list):
                    logger.warning(f"No 'documents' field found in {json_file}")
                    continue
                
                for document in data["documents"]:
                    # Get document content - this is where the form data lives
                    doc_content = document.get("document", document.get("form", []))
                    if not isinstance(doc_content, list):
                        logger.warning(f"Document content is not a list in {json_file}")
                        continue
                    
                    # Get image info for bbox normalization
                    img_info = document.get("img", {})
                    width = img_info.get("width", 1000)
                    height = img_info.get("height", 1000)
                    
                    # Get document ID
                    doc_id = document.get("id", document.get("uid", os.path.splitext(json_file)[0]))
                    
                    words = []
                    bboxes = []
                    labels = []
                    
                    for item in doc_content:
                        if not isinstance(item, dict):
                            continue
                        
                        # Get text - handle multiple possible field names
                        text = None
                        for key in ["text", "words", "content", "value", "word"]:
                            if key in item:
                                text_val = item[key]
                                if isinstance(text_val, list):
                                    text = " ".join(str(t) for t in text_val if t is not None)
                                else:
                                    text = str(text_val).strip()
                                break
                        
                        if not text or text == "text":
                            continue
                        
                        # Get bounding box
                        bbox = [0, 0, 100, 50]  # Default bbox
                        for key in ["box", "bbox", "bounding_box", "coordinates", "geometry"]:
                            if key in item:
                                box_val = item[key]
                                if isinstance(box_val, list) and len(box_val) >= 4:
                                    bbox = box_val[:4]
                                elif isinstance(box_val, dict):
                                    try:
                                        # Handle different bbox formats
                                        x = box_val.get("x", box_val.get("left", 0))
                                        y = box_val.get("y", box_val.get("top", 0))
                                        w = box_val.get("width", box_val.get("w", 100))
                                        h = box_val.get("height", box_val.get("h", 50))
                                        bbox = [x, y, x + w, y + h]
                                    except Exception as e:
                                        logger.debug(f"Error parsing bbox dict: {e}")
                                break
                        
                        # Get label - handle multiple possible field names
                        label = "other"
                        for key in ["label", "class", "category", "type", "entity", "label_name"]:
                            if key in item:
                                label_val = item[key]
                                if isinstance(label_val, list) and label_val:
                                    label = str(label_val[0]).lower()
                                else:
                                    label = str(label_val).lower()
                                break
                        
                        # Normalize bbox
                        try:
                            normalized_bbox = normalize_bbox(bbox, width, height)
                        except Exception as e:
                            logger.debug(f"Error normalizing bbox: {e}")
                            normalized_bbox = [0, 0, 100, 50]
                        
                        words.append(text)
                        bboxes.append(normalized_bbox)
                        labels.append(label)
                    
                    if words:
                        examples.append({
                            "id": doc_id,
                            "words": words,
                            "bboxes": bboxes,
                            "labels": labels,
                            "lang": lang,
                            "image_path": document.get("file")
                        })
                        logger.debug(f"Added example with {len(words)} words from {json_file}")
            
            elif structure_type == "xfund":
                # XFUND format
                if "documents" not in data or not isinstance(data["documents"], list):
                    continue
                
                for document in data["documents"]:
                    if "form" not in document or not isinstance(document["form"], list):
                        continue
                    
                    doc_id = document.get("id", document.get("document_id", os.path.splitext(json_file)[0]))
                    img_info = document.get("img", {})
                    width = img_info.get("width", 1000)
                    height = img_info.get("height", 1000)
                    
                    words = []
                    bboxes = []
                    labels = []
                    
                    for item in document["form"]:
                        if "text" not in item or not item["text"].strip():
                            continue
                        
                        text = item["text"].strip()
                        bbox = item.get("box", [0, 0, 100, 50])
                        label = item.get("label", "other").lower()
                        
                        normalized_bbox = normalize_bbox(bbox, width, height)
                        
                        words.append(text)
                        bboxes.append(normalized_bbox)
                        labels.append(label)
                    
                    if words:
                        examples.append({
                            "id": doc_id,
                            "words": words,
                            "bboxes": bboxes,
                            "labels": labels,
                            "lang": lang,
                            "image_path": None
                        })
            
            elif structure_type == "funsd":
                # FUNSD format
                if "form" not in data or not isinstance(data["form"], list):
                    continue
                
                doc_id = os.path.splitext(json_file)[0]
                width, height = 1000, 1000  # Default dimensions
                
                words = []
                bboxes = []
                labels = []
                
                for item in data["form"]:
                    text = item.get("text", "").strip()
                    if not text:
                        continue
                    
                    # Get bounding box
                    box = item.get("box", [0, 0, 100, 50])
                    if isinstance(box, dict):
                        # Convert dict format to list format
                        box = [
                            box.get("x", 0),
                            box.get("y", 0),
                            box.get("x", 0) + box.get("width", 100),
                            box.get("y", 0) + box.get("height", 50)
                        ]
                    
                    label = item.get("label", "other").lower()
                    
                    normalized_bbox = normalize_bbox(box, width, height)
                    
                    words.append(text)
                    bboxes.append(normalized_bbox)
                    labels.append(label)
                
                if words:
                    examples.append({
                        "id": doc_id,
                        "words": words,
                        "bboxes": bboxes,
                        "labels": labels,
                        "lang": lang,
                        "image_path": None
                    })
            
            else:
                # Generic format - try to extract text and boxes
                if not isinstance(data, dict):
                    continue
                
                # Try common field names
                fields = None
                for key in ["form", "fields", "annotations", "data", "content", "document"]:
                    if key in data and isinstance(data[key], list):
                        fields = data[key]
                        break
                
                if not fields:
                    continue
                
                doc_id = os.path.splitext(json_file)[0]
                width, height = 1000, 1000
                
                words = []
                bboxes = []
                labels = []
                
                for item in fields:
                    if not isinstance(item, dict):
                        continue
                    
                    # Get text
                    text = None
                    for key in ["text", "content", "value", "words", "word"]:
                        if key in item:
                            text = str(item[key]).strip()
                            break
                    
                    if not text:
                        continue
                    
                    # Get bbox
                    bbox = [0, 0, 100, 50]
                    for key in ["box", "bbox", "bounding_box", "coordinates", "geometry"]:
                        if key in item:
                            val = item[key]
                            if isinstance(val, list) and len(val) >= 4:
                                bbox = val[:4]
                                break
                            elif isinstance(val, dict):
                                try:
                                    bbox = [
                                        val.get("x", 0),
                                        val.get("y", 0),
                                        val.get("x", 0) + val.get("width", val.get("w", 100)),
                                        val.get("y", 0) + val.get("height", val.get("h", 50))
                                    ]
                                    break
                                except:
                                    continue
                    
                    # Get label
                    label = "other"
                    for key in ["label", "class", "category", "type", "entity"]:
                        if key in item:
                            label = str(item[key]).lower()
                            break
                    
                    normalized_bbox = normalize_bbox(bbox, width, height)
                    
                    words.append(text)
                    bboxes.append(normalized_bbox)
                    labels.append(label)
                
                if words:
                    examples.append({
                        "id": doc_id,
                        "words": words,
                        "bboxes": bboxes,
                        "labels": labels,
                        "lang": lang,
                        "image_path": None
                    })
        
        except Exception as e:
            logger.warning(f"Error processing {json_file}: {e}")
            continue
    
    logger.info(f"Loaded {len(examples)} {lang} examples from {len(json_files)} JSON files")
    return examples

class XFUNDataset(Dataset):
    """Dataset class for XFUND format data"""
    
    def __init__(self, examples, tokenizer, label_map, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize words with bbox alignment
        tokenized_inputs = self.tokenizer(
            example["words"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            is_split_into_words=True,
        )
        
        # Create label tensors
        labels = []
        word_ids = tokenized_inputs.word_ids(batch_index=0) or []
        
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)  # Special tokens
            elif word_idx < len(example["labels"]):
                label = example["labels"][word_idx]
                labels.append(self.label_map.get(label, self.label_map.get("other", 0)))
            else:
                labels.append(-100)
        
        # Create bbox tensors
        bbox_tensors = []
        for word_idx in word_ids:
            if word_idx is None:
                bbox_tensors.append([0, 0, 0, 0])  # Special tokens
            elif word_idx < len(example["bboxes"]):
                bbox_tensors.append(example["bboxes"][word_idx])
            else:
                bbox_tensors.append([0, 0, 0, 0])
        
        return {
            "input_ids": tokenized_inputs["input_ids"].squeeze(0),
            "attention_mask": tokenized_inputs["attention_mask"].squeeze(0),
            "bbox": torch.tensor(bbox_tensors, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def train_model():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train LiLT model for document analysis")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="nielsr/lilt-xlm-roberta-base")
    parser.add_argument("--output_dir", type=str, default="./lilt_model01")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    
    # Data arguments
    parser.add_argument("--train_dir", type=str, default="./nielsr/test07/train")
    parser.add_argument("--val_dir", type=str, default="./nielsr/test07/val")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--debug_samples", type=int, default=3, help="Number of files to inspect for structure detection")
    
    # Training arguments
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    
    # Training flags
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Detect dataset structure
    logger.info("Detecting training dataset structure...")
    train_structure = detect_dataset_structure(args.train_dir)
    train_image_dir = train_structure['images']
    train_annotation_dir = train_structure['annotations']
    
    logger.info("Detecting validation dataset structure...")
    val_structure = detect_dataset_structure(args.val_dir)
    val_image_dir = val_structure['images']
    val_annotation_dir = val_structure['annotations']
    
    logger.info(f"Training structure resolved:")
    logger.info(f"  Images: {train_image_dir}")
    logger.info(f"  Annotations: {train_annotation_dir}")
    logger.info(f"Validation structure resolved:")
    logger.info(f"  Images: {val_image_dir}")
    logger.info(f"  Annotations: {val_annotation_dir}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    
    # Load datasets
    logger.info("Loading training data...")
    train_examples = load_xfund_examples(
        train_image_dir, 
        train_annotation_dir, 
        args.lang,
        debug_samples=args.debug_samples
    )
    
    logger.info("Loading validation data...")
    val_examples = load_xfund_examples(
        val_image_dir, 
        val_annotation_dir, 
        args.lang,
        debug_samples=args.debug_samples
    )
    
    if not train_examples:
        logger.error("No training examples found!")
        logger.error("Please check your dataset structure. Common XFUND structure:")
        logger.error("train/")
        logger.error("├── images/")
        logger.error("│   ├── 0001.png")
        logger.error("│   └── 0002.jpg")
        logger.error("└── annotations/")
        logger.error("    ├── 0001.json")
        logger.error("    └── 0002.json")
        
        # List directory contents for debugging
        for dir_path in [args.train_dir, train_image_dir, train_annotation_dir]:
            if os.path.exists(dir_path):
                logger.info(f"\nContents of {dir_path}:")
                for item in os.listdir(dir_path):
                    logger.info(f"  {item}")
            else:
                logger.error(f"Directory does not exist: {dir_path}")
        
        return
    
    # Create label map from all labels in training data
    all_labels = [label for ex in train_examples for label in ex["labels"]]
    unique_labels = sorted(set(all_labels))
    
    # Add common labels if not present
    common_labels = ["header", "question", "answer", "other"]
    for label in common_labels:
        if label not in unique_labels:
            unique_labels.append(label)
    
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label_map.items()}
    
    logger.info(f"Found {len(unique_labels)} unique labels: {unique_labels}")
    
    # Save label map
    with open(os.path.join(args.output_dir, "label_map.json"), "w") as f:
        json.dump({
            "label_map": label_map,
            "id2label": id2label,
            "num_labels": len(unique_labels)
        }, f, indent=2)
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = XFUNDataset(train_examples, tokenizer, label_map, args.max_length)
    val_dataset = XFUNDataset(val_examples, tokenizer, label_map, args.max_length) if val_examples else None
    
    # Load model configuration
    logger.info("Loading model configuration...")
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label_map,
    )
    
    # Create model for token classification
    logger.info("Creating token classification model...")
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True  # This is crucial for handling architecture mismatches
    )
    
    # SIMPLE MAXIMUM COMPATIBILITY TrainingArguments
    logger.info("Creating simple TrainingArguments for maximum compatibility")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=100,
        save_steps=500,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        report_to=["none"],
    )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Define compute_metrics function
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = []
        true_labels = []
        
        for i in range(len(predictions)):
            pred_list = predictions[i]
            label_list = labels[i]
            
            for j in range(len(pred_list)):
                if label_list[j] != -100:  # Not a special token
                    # Ensure indices are within bounds
                    pred_idx = int(pred_list[j]) if pred_list[j] < len(id2label) else 0
                    label_idx = int(label_list[j]) if label_list[j] < len(id2label) else 0
                    
                    if pred_idx in id2label and label_idx in id2label:
                        true_predictions.append(id2label[pred_idx])
                        true_labels.append(id2label[label_idx])
        
        # Calculate accuracy
        if not true_predictions:
            accuracy = 0.0
        else:
            accuracy = sum(p == l for p, l in zip(true_predictions, true_labels)) / len(true_predictions)
        
        return {
            "accuracy": accuracy,
        }
    
    # Create trainer with minimal evaluation
    logger.info("Creating trainer...")
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": data_collator,
        "tokenizer": tokenizer,
    }
    
    # Only add evaluation if explicitly requested and dataset exists
    if args.do_eval and val_dataset:
        try:
            trainer_kwargs["eval_dataset"] = val_dataset
            trainer_kwargs["compute_metrics"] = compute_metrics
            logger.info("Added evaluation to trainer")
        except Exception as e:
            logger.warning(f"Evaluation setup failed: {e}")
    
    trainer = Trainer(**trainer_kwargs)
    logger.info("Successfully created trainer")
    
    # Train model
    if args.do_train:
        logger.info("Starting training...")
        train_result = trainer.train()
        logger.info(f"Training completed. Metrics: {train_result}")
        
        # Save model
        logger.info("Saving model...")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Model saved to {args.output_dir}")
    
    # Evaluate model (only if explicitly requested)
    if args.do_eval and val_dataset and hasattr(trainer, 'evaluate'):
        logger.info("Evaluating model...")
        eval_result = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_result}")
        
        # Save evaluation results
        eval_output = os.path.join(args.output_dir, "eval_results.json")
        with open(eval_output, "w") as f:
            json.dump(eval_result, f, indent=2)
        logger.info(f"Evaluation results saved to {eval_output}")
    
    logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    train_model()
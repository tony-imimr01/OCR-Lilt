#!/usr/bin/env python3
"""
doc_pipeline_lilt_fingerprint_api.py

OCR Fingerprint + LILT embedding hybrid verifier with FastAPI endpoint
- Uses fingerprints for fast match
- Uses LILT embeddings for fine similarity if fingerprint is weak
- Tokenizer regex fixed for Mistral/LILT models

Now extended with:
    ✓ /api/extract-text        → raw OCR extraction
    ✓ /api/extract_form_name   → return only matched form name
"""

import os
import sys
import json
import time
import argparse
import logging
import pickle
from typing import List, Tuple, Dict, Optional
from collections import Counter
import tempfile

import numpy as np
from PIL import Image
import pytesseract
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

# Import LiLT model classes - FIXED: remove the dot for absolute import
try:
    from models.LiLTRobertaLike import (
        LiLTRobertaLikeConfig,
        LiLTRobertaLikeForTokenClassification
    )
    LILT_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger("doc_pipeline_lilt")
    logger.warning(f"LiLT models not available: {e}")
    LILT_AVAILABLE = False

# FastAPI imports
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("doc_pipeline_lilt")

# ------------------------------------------------------------
# OCR helper
# ------------------------------------------------------------
def ocr_extract_words_bboxes(image_path: str, conf_thresh: int = 30) -> Tuple[List[str], List[List[int]], int, int]:
    try:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        ocr = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config="--psm 6 --oem 3")

        words, bboxes = [], []
        n = len(ocr.get("text", []))

        for i in range(n):
            text = str(ocr["text"][i]).strip()
            if not text:
                continue

            try:
                conf = float(ocr["conf"][i])
            except Exception:
                conf = 0.0

            if conf < conf_thresh:
                continue

            try:
                left = int(ocr["left"][i])
                top = int(ocr["top"][i])
                w = int(ocr["width"][i])
                h = int(ocr["height"][i])
            except Exception:
                continue

            x1 = max(0, left)
            y1 = max(0, top)
            x2 = min(width, left + w)
            y2 = min(height, top + h)

            cleaned = _clean_text(text)
            if cleaned:
                words.append(cleaned)
                bboxes.append([x1, y1, x2, y2])

        if not words:
            return (
                ["document", "text"],
                [[0, 0, min(100, width), min(30, height)],
                 [0, 40, min(100, width), min(70, height)]],
                width,
                height,
            )

        return words, bboxes, width, height

    except Exception as e:
        logger.warning(f"OCR failed for {image_path}: {e}")
        return ["document", "text"], [[0, 0, 300, 150], [400, 100, 600, 200]], 1000, 1000


def _clean_text(s: str) -> str:
    import re
    s2 = re.sub(r"[^\w\s\-.,!?;:()'\"/]", "", s)
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2

# ------------------------------------------------------------
# Fingerprint generation
# ------------------------------------------------------------
def _sha16(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def generate_fingerprint(words: List[str], boxes: List[List[int]], image_w: int, image_h: int) -> Dict[str, str]:
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
    score = 0.0
    if fp1["anchor_hash"] == fp2["anchor_hash"]:
        score += 0.6
    if fp1["layout_hash"] == fp2["layout_hash"]:
        score += 0.3
    if fp1["token_hash"] == fp2["token_hash"]:
        score += 0.1
    return score

# ------------------------------------------------------------
# Pickle helpers
# ------------------------------------------------------------
def save_pickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# ------------------------------------------------------------
# Build fingerprints from training JSONs
# ------------------------------------------------------------
def find_json_files(train_dir: str) -> List[str]:
    jsons = []
    for root, _, files in os.walk(train_dir):
        for f in files:
            if f.lower().endswith(".json"):
                jsons.append(os.path.join(root, f))
    return jsons


def build_fingerprints(train_json_dir: str, out_path: str):
    jsons = find_json_files(train_json_dir)
    logger.info(f"Found {len(jsons)} JSON files under {train_json_dir}")

    fp_db = {}

    for j in tqdm(jsons, desc="fingerprints"):
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
                if os.path.exists(candidate):
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

    save_pickle(out_path, fp_db)
    logger.info(f"Saved fingerprints ({len(fp_db)}) to {out_path}")

# ------------------------------------------------------------
# Verifier: Fingerprint + LILT hybrid
# ------------------------------------------------------------
class Verifier:
    def __init__(self, fingerprints_path: Optional[str], model_path: Optional[str] = None, fp_threshold: float = 0.85):
        self.fp_threshold = fp_threshold

        self.fingerprints = {}
        if fingerprints_path and os.path.exists(fingerprints_path):
            try:
                self.fingerprints = load_pickle(fingerprints_path)
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_path and os.path.exists(model_path) and LILT_AVAILABLE:
            try:
                # Load config and tokenizer
                config = LiLTRobertaLikeConfig.from_pretrained(model_path)
                config.trust_remote_code = True  # Important for custom models
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path, 
                    use_fast=True,
                    config=config
                )
                
                # Load the custom LILT model
                self.classifier = LiLTRobertaLikeForTokenClassification.from_pretrained(
                    model_path,
                    config=config,
                    trust_remote_code=True
                ).to(self.device)

                logger.info("Loaded custom LILT classifier")

            except Exception as e:
                logger.warning(f"Failed to load LILT classifier: {e}")
                self.classifier = None
                self.tokenizer = None

    def verify(self, image_path: str) -> Dict:
        t0 = time.time()

        words, boxes, w, h = ocr_extract_words_bboxes(image_path)
        preview = " ".join(words)[:400]

        query_fp = generate_fingerprint(words, boxes, w, h)

        # fingerprint match
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

        # strong fingerprint match → trust it
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

        # weak → fallback to LILT classifier
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
            # Tokenize with special handling for LILT models
            enc = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                max_length=512,
                return_special_tokens_mask=True
            ).to(self.device)

            # Create dummy bbox tensor with normalized coordinates
            input_ids = enc['input_ids']
            batch_size, seq_len = input_ids.shape
            
            # Create dummy bounding boxes for LILT model (normalized to 0-1000)
            dummy_bboxes = torch.zeros((batch_size, seq_len, 4), dtype=torch.long).to(self.device)
            
            # Special tokens get [0, 0, 0, 0], other tokens get [100, 100, 200, 200]
            special_tokens_mask = enc.get('special_tokens_mask', torch.zeros_like(input_ids))
            dummy_bboxes[special_tokens_mask == 0] = torch.tensor([100, 100, 200, 200], dtype=torch.long).to(self.device)

            with torch.no_grad():
                outputs = self.classifier(
                    input_ids=input_ids,
                    attention_mask=enc['attention_mask'],
                    bbox=dummy_bboxes
                )
                logits = outputs.logits.detach().cpu().numpy()

            # Handle different output shapes
            if logits.ndim == 3:  # Token classification: (batch, seq_len, num_labels)
                # Take the first token (CLS) or average over tokens
                logits = logits[:, 0, :] if logits.shape[1] > 0 else logits.mean(axis=1)
            elif logits.ndim == 2:  # Sequence classification: (batch, num_labels)
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

# ------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------
app = FastAPI(title="OCR-Fingerprint LILT Verifier API")

verifier_instance: Optional[Verifier] = None


@app.on_event("startup")
def load_verifier():
    global verifier_instance

    parser = argparse.ArgumentParser()
    parser.add_argument("--fingerprints_out", default="fingerprints.pkl")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--fp_threshold", type=float, default=0.85)

    args, _ = parser.parse_known_args()

    verifier_instance = Verifier(
        fingerprints_path=args.fingerprints_out,
        model_path=args.model_path,
        fp_threshold=args.fp_threshold,
    )

    logger.info("Verifier instance loaded for API")


# ------------------------------------------------------------
# 1️⃣ VERIFY ENDPOINT (existing)
# ------------------------------------------------------------
@app.post("/verify")
async def verify_endpoint(file: UploadFile = File(...)):
    if verifier_instance is None:
        return JSONResponse(
            content={"error": "Verifier not initialized"}, status_code=500
        )

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(file.filename)[1]
    ) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = verifier_instance.verify(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return JSONResponse(content=result)

# ------------------------------------------------------------
# 2️⃣ NEW: EXTRACT TEXT ONLY
# ------------------------------------------------------------
@app.post("/api/extract-text")
async def api_extract_text(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(file.filename)[1]
    ) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        words, boxes, w, h = ocr_extract_words_bboxes(tmp_path)

        out = {
            "filename": file.filename,
            "width": w,
            "height": h,
            "word_count": len(words),
            "words": words,
            "bboxes": boxes,
            "text_joined": " ".join(words),
        }

    finally:
        try:
            os.remove(tmp_path)
        except:
            pass

    return JSONResponse(content=out)

# ------------------------------------------------------------
# 3️⃣ NEW: EXTRACT FORM NAME ONLY
# ------------------------------------------------------------
@app.post("/api/extract_form_name")
async def extract_form_name(file: UploadFile = File(...)):
    if verifier_instance is None:
        return JSONResponse({"error": "Verifier not initialized"}, status_code=500)

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(file.filename)[1]
    ) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        words, boxes, w, h = ocr_extract_words_bboxes(tmp_path)
        query_fp = generate_fingerprint(words, boxes, w, h)

        best_fp = None
        best_score = 0.0

        for fname, fp in verifier_instance.fingerprints.items():
            try:
                score = fingerprint_similarity(query_fp, fp)
            except:
                continue

            if score > best_score:
                best_score = score
                best_fp = fname

        out = {
            "filename": file.filename,
            "form_name": best_fp,
            "similarity": float(best_score),
        }

    finally:
        try:
            os.remove(tmp_path)
        except:
            pass

    return JSONResponse(content=out)

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="OCR-fingerprint + LILT hybrid verifier with FastAPI"
    )
    p.add_argument(
        "--train_json_dir",
        default=None,
        help="Training JSON directory (for building fingerprints)",
    )
    p.add_argument(
        "--generate_fingerprints",
        action="store_true",
        help="Generate fingerprints.pkl from training JSONs",
    )
    p.add_argument(
        "--fingerprints_out", default="fingerprints.pkl", help="Output path for fingerprints"
    )
    p.add_argument("--model_path", default=None, help="LILT classifier model path")
    p.add_argument("--verify", action="store_true", help="Run verification on --input")
    p.add_argument("--input", default=None, help="Input image path for verification")
    p.add_argument(
        "--fp_threshold", type=float, default=0.85, help="Fingerprint similarity threshold (0..1)"
    )
    p.add_argument("--serve_api", action="store_true", help="Run FastAPI server")
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    return p.parse_args()


def main():
    args = parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.generate_fingerprints:
        if not args.train_json_dir:
            logger.error("--train_json_dir required for fingerprint generation")
            sys.exit(1)

        build_fingerprints(args.train_json_dir, args.fingerprints_out)
        sys.exit(0)

    if args.verify:
        if not args.input:
            logger.error("--input required for verification")
            sys.exit(1)

        verifier = Verifier(
            fingerprints_path=args.fingerprints_out
            if os.path.exists(args.fingerprints_out)
            else None,
            model_path=args.model_path,
            fp_threshold=args.fp_threshold,
        )

        result = verifier.verify(args.input)
        print(json.dumps(result, indent=2))
        sys.exit(0)

    if args.serve_api:
        # Get port from environment variable or use default
        # port = int(os.environ.get("PORT", 8000))
        port = int(os.environ.get("PORT", 8005))
        logger.info(f"Starting FastAPI server on port {port}")
        # Fixed: use the correct module name
        uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
        sys.exit(0)

    logger.info(
        "No operation specified. Use --generate_fingerprints, --verify or --serve_api"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
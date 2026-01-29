"""
LiLT-style models for Relation Extraction and Token Classification.

Provides:
- LiLTPairwiseHead: pairwise token interaction head producing (bs, seq, seq, C) logits
- LiLTRobertaLikeForRelationExtraction: wrapper around a HuggingFace encoder + pairwise head
- LiLTRobertaLikeForTokenClassification: wrapper around a HuggingFace encoder + token classification head

Usage:
    from models.LiLTRobertaLike import LiLTRobertaLikeForRelationExtraction, LiLTRobertaLikeForTokenClassification

    encoder = AutoModel.from_pretrained("nielsr/lilt-xlm-roberta-base")
    re_model = LiLTRobertaLikeForRelationExtraction(encoder, num_rel_labels=5)
    tc_model = LiLTRobertaLikeForTokenClassification(encoder, num_labels=10)

    # save
    re_model.save_pretrained("out_dir_re")
    tc_model.save_pretrained("out_dir_tc")

    # load
    re_model = LiLTRobertaLikeForRelationExtraction.from_pretrained("out_dir_re", device="cpu")
    tc_model = LiLTRobertaLikeForTokenClassification.from_pretrained("out_dir_tc", device="cpu")
"""
from typing import Optional, Dict, Any, List
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoModelForTokenClassification
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput


__all__ = [
    "LiLTPairwiseHead",
    "LiLTRobertaLikeForRelationExtraction",
    "LiLTRobertaLikeForTokenClassification"
]


class LiLTPairwiseHead(nn.Module):
    """
    Pairwise token interaction head for Relation Extraction.
    Produces logits of shape (bs, seq_len, seq_len, num_rel_labels).
    A simple concatenation + linear classifier is used here.
    """
    def __init__(self, hidden_size: int, num_rel_labels: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_rel_labels = num_rel_labels
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_rel_labels)

        # init
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (bs, L, H)
        returns: logits (bs, L, L, C)
        """
        bs, L, H = hidden_states.size()
        # Expand to (bs, L, L, H)
        hs_i = hidden_states.unsqueeze(2).expand(bs, L, L, H)
        hs_j = hidden_states.unsqueeze(1).expand(bs, L, L, H)
        pair = torch.cat([hs_i, hs_j], dim=-1)  # (bs, L, L, 2H)
        pair = self.dropout(pair)
        logits = self.classifier(pair)  # (bs, L, L, C)
        return logits


class LiLTRobertaLikeForRelationExtraction(nn.Module):
    """
    Wrapper model combining a HuggingFace encoder and a pairwise RE head.

    save_pretrained(save_directory) will:
      - call encoder.save_pretrained(save_directory)
      - write re_head state_dict to save_directory/re_head.pt
      - write re_config.json with {"num_rel_labels": ..., "head_type": "..."}.

    from_pretrained(load_directory) will:
      - load encoder via AutoModel.from_pretrained(load_directory)
      - load re_head state from re_head.pt
      - construct and return the wrapper model.
    """
    RE_HEAD_FILENAME = "re_head.pt"
    RE_CONFIG_FILENAME = "re_config.json"

    def __init__(self, encoder: nn.Module, num_rel_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        # infer hidden size from encoder config
        if hasattr(encoder, "config") and hasattr(encoder.config, "hidden_size"):
            hidden_size = int(encoder.config.hidden_size)
        else:
            hidden_size = 768
        self.rel_head = LiLTPairwiseHead(hidden_size, num_rel_labels, dropout=dropout)
        self.num_rel_labels = int(num_rel_labels)
        self.config = getattr(encoder, "config", None)
        # loss
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                bbox: Optional[torch.Tensor] = None,
                rel_labels: Optional[torch.Tensor] = None,
                **kwargs) -> SequenceClassifierOutput:
        """
        Forward the encoder and pairwise head.

        rel_labels (optional): tensor of shape (bs, seq_len, seq_len) with label ids (0 = NO_REL).
        """
        # allow encoder to ignore unexpected kwargs
        enc_kwargs: Dict[str, Any] = {}
        if input_ids is not None:
            enc_kwargs["input_ids"] = input_ids
        if attention_mask is not None:
            enc_kwargs["attention_mask"] = attention_mask
        # many encoders accept bbox, but if not, encoder will ignore / throw: user must supply compatible encoder
        if bbox is not None:
            enc_kwargs["bbox"] = bbox
        enc_out = self.encoder(**enc_kwargs)
        # get last hidden state robustly
        if hasattr(enc_out, "last_hidden_state"):
            seq_out = enc_out.last_hidden_state
        elif isinstance(enc_out, (list, tuple)) and len(enc_out) > 0:
            seq_out = enc_out[0]
        else:
            raise RuntimeError("Encoder output does not contain last_hidden_state")

        logits = self.rel_head(seq_out)  # (bs, L, L, C)

        loss = None
        if rel_labels is not None:
            # flatten
            bs, s1, s2, C = logits.size()
            logits_flat = logits.view(-1, C)  # (bs*s1*s2, C)
            labels_flat = rel_labels.view(-1).to(logits.device)
            loss = self.loss_fct(logits_flat, labels_flat)

        return SequenceClassifierOutput(loss=loss, logits=logits)

    # ---------------- saving / loading helpers ----------------
    def save_pretrained(self, save_directory: str):
        """
        Save encoder + rel_head state.
        encoder.save_pretrained() writes encoder config/pytorch_model.bin etc.
        """
        os.makedirs(save_directory, exist_ok=True)
        # 1) save encoder (weights + config)
        if hasattr(self.encoder, "save_pretrained"):
            self.encoder.save_pretrained(save_directory)
        else:
            # fallback: save encoder state_dict
            torch.save(self.encoder.state_dict(), os.path.join(save_directory, "encoder_state_dict.pt"))

        # 2) save re head weights
        re_path = os.path.join(save_directory, self.RE_HEAD_FILENAME)
        torch.save(self.rel_head.state_dict(), re_path)

        # 3) save re config
        cfg = {
            "num_rel_labels": int(self.num_rel_labels),
            "head_class": "LiLTPairwiseHead",
        }
        cfg_path = os.path.join(save_directory, self.RE_CONFIG_FILENAME)
        with open(cfg_path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)

    @classmethod
    def from_pretrained(cls, load_directory: str, device: Optional[str] = None):
        """
        Load the encoder (via AutoModel.from_pretrained) and the re_head weights from disk.
        device may be "cpu" or "cuda" or None (leave on default).
        """
        if device is None:
            map_location = None
        else:
            map_location = torch.device(device)

        # 1) load encoder using HF AutoModel (reads config from load_directory)
        encoder = AutoModel.from_pretrained(load_directory)

        # 2) load re_config
        cfg_path = os.path.join(load_directory, cls.RE_CONFIG_FILENAME)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Missing {cls.RE_CONFIG_FILENAME} in {load_directory}")
        with open(cfg_path, "r", encoding="utf-8") as fh:
            rcfg = json.load(fh)
        num_rel_labels = int(rcfg.get("num_rel_labels", 2))

        # 3) construct wrapper and load state
        model = cls(encoder, num_rel_labels=num_rel_labels)

        re_path = os.path.join(load_directory, cls.RE_HEAD_FILENAME)
        if not os.path.exists(re_path):
            raise FileNotFoundError(f"Missing {cls.RE_HEAD_FILENAME} in {load_directory}")

        state = torch.load(re_path, map_location=map_location)
        model.rel_head.load_state_dict(state)

        if device is not None:
            model.to(device)

        model.eval()
        return model

    # convenience
    def to(self, device):
        self.encoder.to(device)
        self.rel_head.to(device)
        return self

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # combine encoder/state? we'll just return model state as default
        return super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)


class LiLTRobertaLikeForTokenClassification(nn.Module):
    """
    Wrapper model combining a HuggingFace encoder and a token classification head.

    save_pretrained(save_directory) will:
      - call encoder.save_pretrained(save_directory)
      - write tc_head state_dict to save_directory/tc_head.pt
      - write tc_config.json with {"num_labels": ..., "head_type": "..."}.

    from_pretrained(load_directory) will:
      - load encoder via AutoModel.from_pretrained(load_directory)
      - load tc_head state from tc_head.pt
      - construct and return the wrapper model.
    """
    TC_HEAD_FILENAME = "tc_head.pt"
    TC_CONFIG_FILENAME = "tc_config.json"

    def __init__(self, encoder: nn.Module, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        # infer hidden size from encoder config
        if hasattr(encoder, "config") and hasattr(encoder.config, "hidden_size"):
            hidden_size = int(encoder.config.hidden_size)
        else:
            hidden_size = 768
        self.num_labels = int(num_labels)
        self.config = getattr(encoder, "config", None)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # init classifier
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        bbox: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> TokenClassifierOutput:
        """
        Forward the encoder and token classification head.

        labels (optional): tensor of shape (bs, seq_len) with label ids.
        """
        # allow encoder to ignore unexpected kwargs
        enc_kwargs: Dict[str, Any] = {}
        if input_ids is not None:
            enc_kwargs["input_ids"] = input_ids
        if attention_mask is not None:
            enc_kwargs["attention_mask"] = attention_mask
        # handle bbox if available
        if bbox is not None:
            enc_kwargs["bbox"] = bbox

        enc_out = self.encoder(**enc_kwargs)
        # get last hidden state robustly
        if hasattr(enc_out, "last_hidden_state"):
            seq_out = enc_out.last_hidden_state
        elif isinstance(enc_out, (list, tuple)) and len(enc_out) > 0:
            seq_out = enc_out[0]
        else:
            raise RuntimeError("Encoder output does not contain last_hidden_state")

        seq_out = self.dropout(seq_out)
        logits = self.classifier(seq_out)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=enc_out.hidden_states if hasattr(enc_out, "hidden_states") else None,
            attentions=enc_out.attentions if hasattr(enc_out, "attentions") else None,
        )

    # ---------------- saving / loading helpers ----------------
    def save_pretrained(self, save_directory: str):
        """
        Save encoder + tc_head state.
        encoder.save_pretrained() writes encoder config/pytorch_model.bin etc.
        """
        os.makedirs(save_directory, exist_ok=True)
        # 1) save encoder (weights + config)
        if hasattr(self.encoder, "save_pretrained"):
            self.encoder.save_pretrained(save_directory)
        else:
            # fallback: save encoder state_dict
            torch.save(self.encoder.state_dict(), os.path.join(save_directory, "encoder_state_dict.pt"))

        # 2) save tc head weights
        tc_path = os.path.join(save_directory, self.TC_HEAD_FILENAME)
        torch.save(self.classifier.state_dict(), tc_path)

        # 3) save tc config
        cfg = {
            "num_labels": int(self.num_labels),
            "head_class": "TokenClassificationHead",
        }
        cfg_path = os.path.join(save_directory, self.TC_CONFIG_FILENAME)
        with open(cfg_path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)

    @classmethod
    def from_pretrained(cls, load_directory: str, device: Optional[str] = None):
        """
        Load the encoder (via AutoModel.from_pretrained) and the tc_head weights from disk.
        device may be "cpu" or "cuda" or None (leave on default).
        """
        if device is None:
            map_location = None
        else:
            map_location = torch.device(device)

        # 1) load encoder using HF AutoModel (reads config from load_directory)
        encoder = AutoModel.from_pretrained(load_directory)

        # 2) load tc_config
        cfg_path = os.path.join(load_directory, cls.TC_CONFIG_FILENAME)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Missing {cls.TC_CONFIG_FILENAME} in {load_directory}")
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        num_labels = int(cfg.get("num_labels", 2))

        # 3) construct wrapper and load state
        model = cls(encoder, num_labels=num_labels)

        tc_path = os.path.join(load_directory, cls.TC_HEAD_FILENAME)
        if not os.path.exists(tc_path):
            raise FileNotFoundError(f"Missing {cls.TC_HEAD_FILENAME} in {load_directory}")

        state = torch.load(tc_path, map_location=map_location)
        model.classifier.load_state_dict(state)

        if device is not None:
            model.to(device)

        model.eval()
        return model

    # convenience methods
    def to(self, device):
        self.encoder.to(device)
        self.classifier.to(device)
        return self

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # combine encoder/state? we'll just return model state as default
        return super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
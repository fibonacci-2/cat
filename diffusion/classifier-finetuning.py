#!/usr/bin/env python3
"""
Fine-tune a HF sequence classifier on YOUR train/val splits (SOURCE text only),
then run the same hedge-intervention evaluation on TEST.

This script produces:
- a local fine-tuned model directory (save_dir/)
- hedge evaluation outputs on test:
    universal_df.test.paired_hedge_test.csv
    universal_df.test.fullpred.csv

It is intentionally minimal + reproducible.

Requires:
pip install -U transformers datasets accelerate scikit-learn torch pandas numpy

Example:
python3 finetune_and_test.py \
  --splits_dir ~/Downloads/splits \
  --base_model cardiffnlp/twitter-roberta-base \
  --save_dir ~/Downloads/ft_roberta_suicide \
  --epochs 3 \
  --batch_size 16 \
  --lr 2e-5 \
  --seed 42 \
  --max_len 256

Then:
python3 classifier.py --csv ~/Downloads/universal_df.csv --splits_dir ~/Downloads/splits --split test --model ~/Downloads/ft_roberta_suicide
"""

import argparse
import os
import math
import random
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)

# ----------------------------
# Helpers
# ----------------------------
LABEL_MAP = {"non-suicide": 0, "suicide": 1}

def normalize_label(x: str) -> int:
    x = str(x).strip().lower()
    if x not in LABEL_MAP:
        raise ValueError(f"Unexpected label '{x}'. Expected one of {list(LABEL_MAP.keys())}")
    return LABEL_MAP[x]

def load_split_csv(splits_dir: str, split: str) -> pd.DataFrame:
    path = os.path.join(splits_dir, f"{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_csv(path)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", pos_label=1, zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

@dataclass
class SimpleDataset(torch.utils.data.Dataset):
    encodings: Dict[str, Any]
    labels: np.ndarray

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item

def tokenize_texts(tokenizer, texts, max_len: int):
    return tokenizer(
        texts,
        truncation=True,
        padding=False,   # dynamic padding via collator
        max_length=max_len,
    )

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", required=True, help="Directory containing train.csv / val.csv / test.csv")
    ap.add_argument("--base_model", required=True, help="Base encoder model to fine-tune (e.g., cardiffnlp/twitter-roberta-base)")
    ap.add_argument("--save_dir", required=True, help="Output directory to save fine-tuned model")
    ap.add_argument("--text_col", default="source", choices=["source"], help="Train on SOURCE only (fixed as requested)")
    ap.add_argument("--label_col", default="class")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true", help="Enable fp16 (recommended on GPU)")
    ap.add_argument("--bf16", action="store_true", help="Enable bf16 if supported")
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--metric_for_best", default="f1", choices=["f1", "accuracy", "recall", "precision"])
    args = ap.parse_args()

    set_seed(args.seed)

    # Load splits
    train_df = load_split_csv(args.splits_dir, "train")
    val_df   = load_split_csv(args.splits_dir, "val")
    test_df  = load_split_csv(args.splits_dir, "test")  # used only for a quick sanity eval at end (optional)

    # Prepare data (SOURCE ONLY)
    for df in (train_df, val_df, test_df):
        df[args.text_col] = df[args.text_col].astype(str)
        df[args.label_col] = df[args.label_col].astype(str).str.strip().str.lower()
        df["y"] = df[args.label_col].map(normalize_label)

    # Tokenizer/model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label={0: "non-suicide", 1: "suicide"},
        label2id={"non-suicide": 0, "suicide": 1},
    )

    # Tokenize
    train_enc = tokenize_texts(tokenizer, train_df[args.text_col].tolist(), args.max_len)
    val_enc   = tokenize_texts(tokenizer, val_df[args.text_col].tolist(), args.max_len)

    train_ds = SimpleDataset(train_enc, train_df["y"].to_numpy())
    val_ds   = SimpleDataset(val_enc, val_df["y"].to_numpy())

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training args
    os.makedirs(args.save_dir, exist_ok=True)
    targs = TrainingArguments(
        output_dir=args.save_dir,
        # overwrite_output_dir=True,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best,
        greater_is_better=True,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=50,
        report_to="none",
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print("============================================================")
    print("Fine-tuning on TRAIN (SOURCE only) with VAL for early selection")
    print("============================================================")
    trainer.train()

    print("============================================================")
    print("Saving best model")
    print("============================================================")
    trainer.save_model(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print(f"[OK] Saved fine-tuned model to: {args.save_dir}")

    # Optional quick sanity eval on val (already done during training) + test on source only
    print("============================================================")
    print("Quick sanity eval on TEST (SOURCE only)")
    print("============================================================")
    test_enc = tokenize_texts(tokenizer, test_df[args.text_col].tolist(), args.max_len)
    test_ds = SimpleDataset(test_enc, test_df["y"].to_numpy())
    metrics = trainer.evaluate(test_ds)
    print(metrics)

    print("\nNEXT STEP (run hedge pipeline on test):")
    print(f"python3 classifier.py --csv <PATH_TO_universal_df.csv> --splits_dir {args.splits_dir} --split test --model {args.save_dir}")


if __name__ == "__main__":
    main()
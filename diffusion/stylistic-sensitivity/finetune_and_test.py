#!/usr/bin/env python3
"""
Fine-tune a binary suicide classifier on TRAIN (source only),
select best checkpoint on VAL (source only),
and report final TEST metrics (source only).

Expected split files in --splits_dir:
  train.csv
  val.csv
  test.csv

Each split must contain:
  id, source, recover, class, gender

Label column:
  class ∈ {"suicide","non-suicide"} (case-insensitive)

Example usage:

python3 finetune_and_test.py \
  --splits_dir /SEAS/home/g21775526/Downloads/splits \
  --base_model roberta-base \
  --save_dir /SEAS/home/g21775526/Downloads/ft_roberta_base_suicide \
  --epochs 3 \
  --batch_size 16 \
  --lr 2e-5 \
  --max_len 256 \
  --seed 42 \
  --fp16
"""

import argparse
import os
import random
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing split file: {path}")
    df = pd.read_csv(path)
    required = ["id", "source", "class"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {path}")
    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    y = df["class"].astype(str).str.strip().str.lower()
    if not set(y.unique()).issubset({"suicide", "non-suicide"}):
        bad = sorted(set(y.unique()) - {"suicide", "non-suicide"})
        raise ValueError(f"Unexpected labels: {bad}")
    df = df.copy()
    df["label"] = (y == "suicide").astype(int)
    return df


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    # ----------------------------
    # Load splits
    # ----------------------------
    train_df = load_split(os.path.join(args.splits_dir, "train.csv"))
    val_df = load_split(os.path.join(args.splits_dir, "val.csv"))
    test_df = load_split(os.path.join(args.splits_dir, "test.csv"))

    train_df = encode_labels(train_df)
    val_df = encode_labels(val_df)
    test_df = encode_labels(test_df)

    # Use SOURCE only for training and evaluation
    train_df["text"] = train_df["source"].astype(str)
    val_df["text"] = val_df["source"].astype(str)
    test_df["text"] = test_df["source"].astype(str)

    # ----------------------------
    # HF Datasets
    # ----------------------------
    train_ds = Dataset.from_pandas(train_df[["text", "label"]])
    val_ds = Dataset.from_pandas(val_df[["text", "label"]])
    test_ds = Dataset.from_pandas(test_df[["text", "label"]])

    # ----------------------------
    # Tokenizer + model
    # ----------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_len,
        )

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label={0: "non-suicide", 1: "suicide"},
        label2id={"non-suicide": 0, "suicide": 1},
    )

    # ----------------------------
    # Training args
    # ----------------------------
    # Determine mixed precision settings
    use_fp16 = args.fp16 and not args.bf16 and torch.cuda.is_available()
    use_bf16 = args.bf16 and torch.cuda.is_available()
    
    # If fp16 causes issues, fall back to no mixed precision
    if use_fp16:
        print("[WARN] fp16 can cause gradient unscaling issues with some models. Consider --bf16 instead.")
    
    training_args = TrainingArguments(
        output_dir=args.save_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=args.seed,
        logging_steps=50,
        fp16=False,  # Disabled due to gradient unscaling issues
        bf16=use_bf16,
        report_to="none",
        optim="adamw_torch",  # Use standard optimizer to avoid fp16 issues
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    # ----------------------------
    # Train
    # ----------------------------
    print("============================================================")
    print("Training...")
    print("============================================================")
    trainer.train()

    print("============================================================")
    print("Saving best model")
    print("============================================================")
    trainer.save_model(args.save_dir)

    # ----------------------------
    # Final TEST evaluation
    # ----------------------------
    print("============================================================")
    print("Final evaluation on TEST (SOURCE only)")
    print("============================================================")
    test_metrics = trainer.evaluate(test_ds)
    print(test_metrics)

    # Save metrics
    metrics_path = os.path.join(args.save_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        import json
        json.dump(test_metrics, f, indent=2)

    print(f"[OK] Saved fine-tuned model to: {args.save_dir}")
    print(f"[OK] Saved test metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Gender × Hedging pipeline (directly matches your 2 goals)

GOAL 1 (Baseline):
  Does hedging correlate with performance differences across genders on SOURCE?
  -> Evaluate model on SOURCE text, compute metrics by gender, compute gender gap.

GOAL 2 (Controlled):
  Does controlling hedging (hedge-up males in RECOVER) reduce that gender gap?
  -> Evaluate model on RECOVER text, compute metrics by gender, compute gender gap.

Main estimand (Difference-in-Differences):
  gap_source  = metric_M_source  - metric_F_source
  gap_recover = metric_M_recover - metric_F_recover
  DiD = gap_recover - gap_source
Interpretation:
  If |DiD| is large and moves toward 0 gap, controlling hedging helped.

Input requirements in split CSV (train/val/test):
  columns: id, class, gender, source, recover
  class in {"suicide","non-suicide"} (case-insensitive)

Recommended usage (on TEST split):
python3 hedge_gender_gap.py \
  --splits_dir ~/Downloads/splits \
  --split test \
  --model /SEAS/home/g21775526/Downloads/ft_roberta_suicide \
  --out_dir ~/Downloads/hedge_gap_results \
  --batch_size 32 \
  --n_boot 20000 \
  --seed 42

Also run on an off-the-shelf model for comparison:
python3 hedge_gender_gap.py \
  --splits_dir ~/Downloads/splits \
  --split test \
  --model Akashpaul123/bert-suicide-detection \
  --out_dir ~/Downloads/hedge_gap_results \
  --batch_size 32
"""

import argparse
import math
import os
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ----------------------------
# Label + metrics
# ----------------------------
def ytrue_from_class(series: pd.Series) -> np.ndarray:
    y = series.astype(str).str.strip().str.lower()
    if not set(y.unique()).issubset({"suicide", "non-suicide"}):
        bad = sorted(set(y.unique()) - {"suicide", "non-suicide"})
        raise ValueError(f"Unexpected labels in 'class': {bad} (expected suicide/non-suicide)")
    return (y == "suicide").astype(int).to_numpy()

def confusion_counts(y_true: np.ndarray, y_hat: np.ndarray) -> Dict[str, int]:
    tp = int(((y_true == 1) & (y_hat == 1)).sum())
    tn = int(((y_true == 0) & (y_hat == 0)).sum())
    fp = int(((y_true == 0) & (y_hat == 1)).sum())
    fn = int(((y_true == 1) & (y_hat == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

def safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else float("nan")

def compute_metrics(y_true: np.ndarray, y_hat: np.ndarray, p_pos: np.ndarray) -> Dict[str, float]:
    c = confusion_counts(y_true, y_hat)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    n = len(y_true)

    acc = safe_div(tp + tn, n)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)            # TPR
    f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else float("nan")
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)
    mean_p = float(np.mean(p_pos)) if len(p_pos) else float("nan")

    return {
        "n": int(n),
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "mean_p_suicide": mean_p,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }


# ----------------------------
# Model loading (robust label mapping)
# ----------------------------
def load_clf(model_name: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    mdl.to(device)
    mdl.eval()

    label2id_raw = mdl.config.label2id or {}
    label2id = {str(k).strip().lower(): int(v) for k, v in label2id_raw.items()}

    # Prefer exact label names
    if "suicide" in label2id and "non-suicide" in label2id:
        suicide_id = label2id["suicide"]
        non_suicide_id = label2id["non-suicide"]
    elif "label_1" in label2id and "label_0" in label2id:
        suicide_id = label2id["label_1"]
        non_suicide_id = label2id["label_0"]
    else:
        suicide_id = None
        non_suicide_id = None
        for name, idx in label2id.items():
            n = name.replace("_", "-")
            if n == "suicide" or n.startswith("suicide-"):
                suicide_id = idx
            if n in {"non-suicide", "nonsuicide", "non suicide", "control", "negative"} or n.startswith("non-suicide-"):
                non_suicide_id = idx
        if suicide_id is None or non_suicide_id is None:
            if mdl.config.num_labels == 2:
                suicide_id, non_suicide_id = 1, 0
            else:
                raise ValueError(f"Could not infer label ids. label2id={label2id_raw}, num_labels={mdl.config.num_labels}")

    if suicide_id == non_suicide_id:
        raise ValueError(f"BUG: suicide_id == non_suicide_id == {suicide_id}. label2id={label2id_raw}")

    print(f"[INFO] label2id: {mdl.config.label2id}")
    print(f"[INFO] using suicide_id={suicide_id}, non_suicide_id={non_suicide_id}")
    return tok, mdl, suicide_id


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

@torch.no_grad()
def predict_probs(texts: List[str], tok, mdl, device: str, batch_size: int, suicide_id: int) -> Tuple[np.ndarray, np.ndarray]:
    ps = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = mdl(**enc).logits.detach().cpu().numpy()
        probs = softmax_np(logits)
        ps.append(probs[:, suicide_id])
    p = np.concatenate(ps, axis=0) if ps else np.array([])
    yhat = (p >= 0.5).astype(int)
    return p, yhat


# ----------------------------
# Bootstrap for gender gaps + DiD
# ----------------------------
def bootstrap_gap_and_did(
    df: pd.DataFrame,
    metric: str,
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    """
    Bootstraps the gender gap and DiD by resampling rows within each (gender, condition).
    Assumes df has columns:
      gender ∈ {M,F}, condition ∈ {source,recover}, metric columns precomputed per-row? (we compute on sample)
    We recompute the metric from y_true/yhat each bootstrap.
    """
    rng = np.random.default_rng(seed)

    def metric_from_subset(sub: pd.DataFrame) -> float:
        y_true = sub["y_true"].to_numpy()
        y_hat = sub["y_hat"].to_numpy()
        p = sub["p_suicide"].to_numpy()
        m = compute_metrics(y_true, y_hat, p)
        return float(m[metric])

    # split into 4 cells
    cells = {}
    for gender in ["M", "F"]:
        for cond in ["source", "recover"]:
            key = (gender, cond)
            cells[key] = df[(df["gender"] == gender) & (df["condition"] == cond)].copy()

    # need all cells non-empty for DiD
    for k, sub in cells.items():
        if len(sub) == 0:
            return {"gap_source": np.nan, "gap_recover": np.nan, "did": np.nan,
                    "gap_source_ci_low": np.nan, "gap_source_ci_high": np.nan,
                    "gap_recover_ci_low": np.nan, "gap_recover_ci_high": np.nan,
                    "did_ci_low": np.nan, "did_ci_high": np.nan}

    # point estimates
    M_src = metric_from_subset(cells[("M", "source")])
    F_src = metric_from_subset(cells[("F", "source")])
    M_rec = metric_from_subset(cells[("M", "recover")])
    F_rec = metric_from_subset(cells[("F", "recover")])

    gap_source = M_src - F_src
    gap_recover = M_rec - F_rec
    did = gap_recover - gap_source

    # bootstrap
    boot_gap_s = np.empty(n_boot, dtype=float)
    boot_gap_r = np.empty(n_boot, dtype=float)
    boot_did = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        res = {}
        for key, sub in cells.items():
            idx = rng.integers(0, len(sub), size=len(sub))
            res[key] = sub.iloc[idx]

        M_src_b = metric_from_subset(res[("M", "source")])
        F_src_b = metric_from_subset(res[("F", "source")])
        M_rec_b = metric_from_subset(res[("M", "recover")])
        F_rec_b = metric_from_subset(res[("F", "recover")])

        gs = M_src_b - F_src_b
        gr = M_rec_b - F_rec_b
        boot_gap_s[b] = gs
        boot_gap_r[b] = gr
        boot_did[b] = gr - gs

    def ci(arr: np.ndarray) -> Tuple[float, float]:
        lo = float(np.quantile(arr, 0.025))
        hi = float(np.quantile(arr, 0.975))
        return lo, hi

    gs_lo, gs_hi = ci(boot_gap_s)
    gr_lo, gr_hi = ci(boot_gap_r)
    did_lo, did_hi = ci(boot_did)

    return {
        "gap_source": float(gap_source),
        "gap_recover": float(gap_recover),
        "did": float(did),
        "gap_source_ci_low": gs_lo,
        "gap_source_ci_high": gs_hi,
        "gap_recover_ci_low": gr_lo,
        "gap_recover_ci_high": gr_hi,
        "did_ci_low": did_lo,
        "did_ci_high": did_hi,
    }


# ----------------------------
# Main evaluation
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", required=True, help="Directory with train.csv/val.csv/test.csv")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--model", required=True, help="HF model id or local fine-tuned path")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n_boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    split_path = os.path.join(args.splits_dir, f"{args.split}.csv")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Missing split file: {split_path}")

    df = pd.read_csv(split_path)

    required = ["id", "class", "gender", "source", "recover"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {split_path}")

    # normalize gender
    df["gender"] = df["gender"].astype(str).str.strip().str.upper()
    df = df[df["gender"].isin(["M", "F"])].copy()

    # true labels
    df["y_true"] = ytrue_from_class(df["class"])

    # Load model
    tok, mdl, suicide_id = load_clf(args.model, args.device)

    # Build long-form evaluation df: one row per (example, condition)
    rows = []
    for cond, text_col in [("source", "source"), ("recover", "recover")]:
        texts = df[text_col].astype(str).tolist()
        p, yhat = predict_probs(texts, tok, mdl, args.device, args.batch_size, suicide_id)

        tmp = df[["id", "gender", "y_true"]].copy()
        tmp["condition"] = cond
        tmp["p_suicide"] = p
        tmp["y_hat"] = yhat
        rows.append(tmp)

    long_df = pd.concat(rows, ignore_index=True)

    # Metrics by (condition, gender)
    out_rows = []
    for cond in ["source", "recover"]:
        for gender in ["M", "F"]:
            sub = long_df[(long_df["condition"] == cond) & (long_df["gender"] == gender)]
            m = compute_metrics(sub["y_true"].to_numpy(), sub["y_hat"].to_numpy(), sub["p_suicide"].to_numpy())
            out_rows.append({
                "model": args.model,
                "split": args.split,
                "condition": cond,
                "gender": gender,
                **m
            })

    metrics_df = pd.DataFrame(out_rows)

    # Gaps + DiD for key metrics
    # You can add more metrics here if you want.
    target_metrics = ["acc", "fnr", "fpr", "recall", "precision", "f1", "mean_p_suicide"]

    gap_rows = []
    for metric in target_metrics:
        boot = bootstrap_gap_and_did(long_df, metric=metric, n_boot=args.n_boot, seed=args.seed)
        gap_rows.append({
            "model": args.model,
            "split": args.split,
            "metric": metric,
            **boot
        })

    gaps_df = pd.DataFrame(gap_rows)

    # Save
    safe_model = args.model.replace("/", "_").replace(":", "_").replace(" ", "_")
    metrics_path = os.path.join(args.out_dir, f"{args.split}.{safe_model}.gender_by_condition_metrics.csv")
    gaps_path = os.path.join(args.out_dir, f"{args.split}.{safe_model}.gender_gaps_did.csv")
    long_path = os.path.join(args.out_dir, f"{args.split}.{safe_model}.long_predictions.csv")

    metrics_df.to_csv(metrics_path, index=False)
    gaps_df.to_csv(gaps_path, index=False)
    long_df.to_csv(long_path, index=False)

    # Print concise summary for the two main goals
    def get_val(cond, gender, key):
        return float(metrics_df[(metrics_df["condition"] == cond) & (metrics_df["gender"] == gender)][key].iloc[0])

    # Baseline gaps (SOURCE)
    gap_acc_source = get_val("source", "M", "acc") - get_val("source", "F", "acc")
    gap_fnr_source = get_val("source", "M", "fnr") - get_val("source", "F", "fnr")

    # Controlled gaps (RECOVER)
    gap_acc_recover = get_val("recover", "M", "acc") - get_val("recover", "F", "acc")
    gap_fnr_recover = get_val("recover", "M", "fnr") - get_val("recover", "F", "fnr")

    print("============================================================")
    print(f"MODEL: {args.model}")
    print(f"SPLIT: {args.split}")
    print("Goal 1 (Baseline): Does hedging relate to gender performance gap on SOURCE?")
    print(f"  Acc gap (M-F) on SOURCE: {gap_acc_source:+.4f}")
    print(f"  FNR gap (M-F) on SOURCE: {gap_fnr_source:+.4f}")
    print("Goal 2 (Controlled): Does hedging-control reduce that gap on RECOVER?")
    print(f"  Acc gap (M-F) on RECOVER: {gap_acc_recover:+.4f}")
    print(f"  FNR gap (M-F) on RECOVER: {gap_fnr_recover:+.4f}")
    print("Difference-in-Differences (RECOVER gap - SOURCE gap):")
    print(f"  DiD Acc: {gap_acc_recover - gap_acc_source:+.4f}")
    print(f"  DiD FNR: {gap_fnr_recover - gap_fnr_source:+.4f}")
    print("------------------------------------------------------------")
    print(f"[OK] wrote: {metrics_path}")
    print(f"[OK] wrote: {gaps_path}")
    print(f"[OK] wrote: {long_path}")
    print("============================================================")


if __name__ == "__main__":
    main()
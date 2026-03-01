#!/usr/bin/env python3
"""
Hedge shortcut test using proper TRAIN/VAL/TEST splits.

What this script does:
1) Optionally creates ID-level stratified splits and saves them (train/val/test).
2) Runs the paired hedge intervention test ONLY on the chosen split (default: test).
3) Computes observational error rates by hedge level on the FULL chosen split
   (not just the paired subset), to avoid misleading tables.

CSV schema:
id, source, recover, class, gender, source_hedge_level, recover_hedge_level

Usage examples:

# 1) Create splits (id-level, stratified by class) and save:
python3 classifier.py --csv universal_df.csv --make_splits --out_dir ~/Downloads/splits

# 2) Run hedge test on TEST split:
python3 classifier.py --csv universal_df.csv --splits_dir ~/Downloads/splits --split test \
  --model Akashpaul123/bert-suicide-detection --gender M

# 3) Or run on train (for diagnostic):
python3 classifier.py --csv universal_df.csv --splits_dir ~/Downloads/splits --split train \
  --model Akashpaul123/bert-suicide-detection
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split


# ----------------------------
# Utils
# ----------------------------
HEDGE_ORDER = {"small": 0, "low": 0, "medium": 1, "med": 1, "mid": 1, "high": 2, "large": 2}

def norm_hedge(x: str) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "unknown"
    return str(x).strip().lower()

def hedge_to_num(x: str) -> float:
    x = norm_hedge(x)
    return float(HEDGE_ORDER.get(x, np.nan))

def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def wilcoxon_signed_rank(x: np.ndarray) -> Tuple[float, float]:
    """
    Simple Wilcoxon signed-rank p-value approximation using normal approximation.
    (For publication use scipy.stats.wilcoxon.)
    Returns: (W, p_approx_two_sided)
    """
    d = x.copy()
    d = d[~np.isnan(d)]
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return np.nan, np.nan
    absd = np.abs(d)
    ranks = absd.argsort().argsort().astype(float) + 1.0  # 1..n
    Wpos = ranks[d > 0].sum()
    Wneg = ranks[d < 0].sum()
    W = min(Wpos, Wneg)
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0:
        return W, 1.0
    z = (W - mu) / sigma
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return W, p


# ----------------------------
# Splitting (ID-level, stratified)
# ----------------------------
def make_id_splits(
    df: pd.DataFrame,
    out_dir: str,
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> Tuple[str, str, str]:
    """
    Creates ID-level splits stratified by label (class).
    Saves: train.csv, val.csv, test.csv into out_dir.

    Assumes each id has a single label in column 'class'. If not, this will raise.
    """
    os.makedirs(out_dir, exist_ok=True)

    if "id" not in df.columns or "class" not in df.columns:
        raise ValueError("CSV must include 'id' and 'class' columns.")

    # Normalize labels
    d = df.copy()
    d["class"] = d["class"].astype(str).str.strip().str.lower()

    # Build id -> label (ensure consistent)
    id_label = d.groupby("id")["class"].nunique()
    bad_ids = id_label[id_label > 1].index.tolist()
    if bad_ids:
        raise ValueError(
            f"Found ids with multiple labels (leakage risk). Example bad id: {bad_ids[0]}. "
            "Fix dataset so each id maps to a single class."
        )

    id_to_label = d.groupby("id")["class"].first().reset_index()
    ids = id_to_label["id"].to_numpy()
    y = id_to_label["class"].to_numpy()

    # train vs temp
    train_ids, temp_ids = train_test_split(
        ids,
        test_size=(1.0 - train_frac),
        stratify=y,
        random_state=seed,
    )

    # val vs test from temp
    # temp size is (1-train_frac), so val is val_frac overall -> val fraction of temp:
    temp_df = id_to_label[id_to_label["id"].isin(temp_ids)]
    temp_y = temp_df["class"].to_numpy()
    val_frac_of_temp = val_frac / (1.0 - train_frac)

    val_ids, test_ids = train_test_split(
        temp_df["id"].to_numpy(),
        test_size=(1.0 - val_frac_of_temp),
        stratify=temp_y,
        random_state=seed,
    )

    train_df = d[d["id"].isin(train_ids)].copy()
    val_df   = d[d["id"].isin(val_ids)].copy()
    test_df  = d[d["id"].isin(test_ids)].copy()

    train_path = os.path.join(out_dir, "train.csv")
    val_path   = os.path.join(out_dir, "val.csv")
    test_path  = os.path.join(out_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[OK] Saved splits to: {out_dir}")
    print(f"  train: {len(train_df)} rows | {train_df['id'].nunique()} unique ids")
    print(f"  val:   {len(val_df)} rows   | {val_df['id'].nunique()} unique ids")
    print(f"  test:  {len(test_df)} rows  | {test_df['id'].nunique()} unique ids")

    return train_path, val_path, test_path


def load_split(csv_path: str, splits_dir: str, split: str) -> pd.DataFrame:
    if splits_dir is None:
        return pd.read_csv(csv_path)

    split = split.lower()
    if split not in ("train", "val", "test"):
        raise ValueError("--split must be one of: train, val, test")

    path = os.path.join(splits_dir, f"{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Split file not found: {path} (did you run --make_splits?)")
    return pd.read_csv(path)


# ----------------------------
# Model wrapper
# ----------------------------
@dataclass
class PredResult:
    p_suicide: np.ndarray
    yhat: np.ndarray
    label2id: dict
    suicide_id: int
    non_suicide_id: int


def load_clf(model_name: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    mdl.to(device)
    mdl.eval()

    label2id = {k.lower(): int(v) for k, v in (mdl.config.label2id or {}).items()}

    def find_label(keys: List[str]) -> Optional[int]:
        for k in keys:
            for name, idx in label2id.items():
                if k in name:
                    return idx
        return None

    suicide_id = find_label(["suicide"])
    non_suicide_id = find_label(["non-suicide", "nonsuicide", "non suicide", "control", "negative"])

    if suicide_id is None or non_suicide_id is None:
        if mdl.config.num_labels == 2:
            suicide_id = 1
            non_suicide_id = 0
        else:
            raise ValueError(
                f"Could not infer label ids from model config label2id={mdl.config.label2id}. "
                "Please use a binary model or modify mapping."
            )

    print(f"[INFO] label2id: {mdl.config.label2id}")
    print(f"[INFO] using suicide_id={suicide_id}, non_suicide_id={non_suicide_id}")

    return tok, mdl, label2id, suicide_id, non_suicide_id


@torch.no_grad()
def predict_probs(texts: List[str], tok, mdl, device: str, batch_size: int, suicide_id: int):
    ps = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = mdl(**enc).logits.detach().cpu().numpy()
        probs = softmax_np(logits)
        ps.append(probs[:, suicide_id])
    p_suicide = np.concatenate(ps, axis=0) if ps else np.array([])
    yhat = (p_suicide >= 0.5).astype(int)
    return p_suicide, yhat


# ----------------------------
# Tests
# ----------------------------
def paired_hedge_effect(df: pd.DataFrame, tok, mdl, device: str, batch_size: int, suicide_id: int) -> pd.DataFrame:
    required = ["id", "source", "recover", "class", "source_hedge_level", "recover_hedge_level"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    d = df.copy()
    d["class"] = d["class"].astype(str).str.strip().str.lower()
    d["source_hedge_level"] = d["source_hedge_level"].map(norm_hedge)
    d["recover_hedge_level"] = d["recover_hedge_level"].map(norm_hedge)

    d["source"] = d["source"].astype(str)
    d["recover"] = d["recover"].astype(str)

    # keep where hedge differs
    d = d[d["source_hedge_level"] != d["recover_hedge_level"]].copy()
    if len(d) == 0:
        raise ValueError("No rows where hedge level differs between source and recover.")

    p_src, y_src = predict_probs(d["source"].tolist(), tok, mdl, device, batch_size, suicide_id)
    p_rec, y_rec = predict_probs(d["recover"].tolist(), tok, mdl, device, batch_size, suicide_id)

    d["p_suicide_source"] = p_src
    d["p_suicide_recover"] = p_rec
    d["yhat_source"] = y_src
    d["yhat_recover"] = y_rec
    d["delta_p"] = d["p_suicide_recover"] - d["p_suicide_source"]
    d["flip"] = (d["yhat_source"] != d["yhat_recover"]).astype(int)

    d["src_hnum"] = d["source_hedge_level"].map(hedge_to_num)
    d["rec_hnum"] = d["recover_hedge_level"].map(hedge_to_num)
    d["delta_hedge"] = d["rec_hnum"] - d["src_hnum"]

    d["y_true"] = (d["class"] == "suicide").astype(int)
    return d


def error_rates_by_hedge_full(df: pd.DataFrame, use_col: str) -> pd.DataFrame:
    """
    Observational rates on FULL split (not just paired subset).
    We predict on the chosen text column and group by that column's hedge level.
    """
    assert use_col in ("source", "recover")
    hedge_col = f"{use_col}_hedge_level"
    if hedge_col not in df.columns:
        raise ValueError(f"Missing {hedge_col}")

    g = df.copy()
    g["class"] = g["class"].astype(str).str.strip().str.lower()
    g["y_true"] = (g["class"] == "suicide").astype(int)
    g[hedge_col] = g[hedge_col].map(norm_hedge)

    rows = []
    for level, sub in g.groupby(hedge_col):
        y = sub["y_true"].to_numpy()
        yhat = sub[f"yhat_{use_col}"].to_numpy()
        p = sub[f"p_suicide_{use_col}"].to_numpy()

        tp = int(((y == 1) & (yhat == 1)).sum())
        tn = int(((y == 0) & (yhat == 0)).sum())
        fp = int(((y == 0) & (yhat == 1)).sum())
        fn = int(((y == 1) & (yhat == 0)).sum())
        n = len(sub)

        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
        acc = (tp + tn) / n if n > 0 else np.nan

        rows.append({
            "hedge_level": level,
            "n": n,
            "acc": acc,
            "FPR": fpr,
            "FNR": fnr,
            "mean_p_suicide": float(np.mean(p)) if n > 0 else np.nan,
        })

    return pd.DataFrame(rows).sort_values(["hedge_level"])


def attach_predictions_full(df: pd.DataFrame, tok, mdl, device: str, batch_size: int, suicide_id: int) -> pd.DataFrame:
    """
    Predict on full split for both source and recover, so observational grouping is correct.
    """
    d = df.copy()
    d["source"] = d["source"].astype(str)
    d["recover"] = d["recover"].astype(str)

    p_src, y_src = predict_probs(d["source"].tolist(), tok, mdl, device, batch_size, suicide_id)
    p_rec, y_rec = predict_probs(d["recover"].tolist(), tok, mdl, device, batch_size, suicide_id)

    d["p_suicide_source"] = p_src
    d["yhat_source"] = y_src
    d["p_suicide_recover"] = p_rec
    d["yhat_recover"] = y_rec
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to universal_df.csv (or any full CSV).")
    ap.add_argument("--model", required=False, help="HF model name/path for sequence classification.")
    ap.add_argument("--gender", default=None, choices=["M", "F"], help="Filter gender (optional).")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # splits
    ap.add_argument("--make_splits", action="store_true", help="Create and save id-level splits.")
    ap.add_argument("--out_dir", default=None, help="Where to write splits (train.csv/val.csv/test.csv).")
    ap.add_argument("--splits_dir", default=None, help="Directory containing train.csv/val.csv/test.csv.")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"],
                    help="Which split to evaluate on (default: test).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)

    ap.add_argument("--min_pairs", type=int, default=20)

    args = ap.parse_args()

    # load full data
    full_df = pd.read_csv(args.csv)

    # optionally create splits and exit
    if args.make_splits:
        if args.out_dir is None:
            raise ValueError("--out_dir is required when using --make_splits")
        make_id_splits(full_df, out_dir=args.out_dir, seed=args.seed,
                       train_frac=args.train_frac, val_frac=args.val_frac)
        return

    if not args.model:
        raise ValueError("--model is required unless you are using --make_splits")

    # load chosen split
    df = load_split(args.csv, args.splits_dir, args.split)

    # optional gender filter
    if args.gender is not None and "gender" in df.columns:
        df = df[df["gender"].astype(str).str.upper() == args.gender].copy()

    tok, mdl, label2id, suicide_id, non_suicide_id = load_clf(args.model, args.device)

    # predict on full split for observational tables
    df_pred = attach_predictions_full(df, tok, mdl, args.device, args.batch_size, suicide_id)

    # paired analysis only uses hedge-changed subset
    paired = paired_hedge_effect(df_pred, tok, mdl, args.device, args.batch_size, suicide_id)

    if len(paired) < args.min_pairs:
        print(f"[WARN] Only {len(paired)} hedge-changed pairs in split={args.split}.", file=sys.stderr)

    # ----------------------------
    # Paired causal-style tests
    # ----------------------------
    sign = np.sign(paired["delta_hedge"].to_numpy())
    delta_p = paired["delta_p"].to_numpy()
    sign_aligned = delta_p * sign
    mask = (~np.isnan(sign_aligned)) & (sign != 0)
    sign_aligned = sign_aligned[mask]

    mean_effect = float(np.nanmean(sign_aligned)) if len(sign_aligned) else np.nan
    W, p_wil = wilcoxon_signed_rank(sign_aligned) if len(sign_aligned) else (np.nan, np.nan)

    flip_rate = float(paired["flip"].mean())
    mean_delta = float(paired["delta_p"].mean())

    print("============================================================")
    print(f"Split: {args.split} | Gender: {args.gender if args.gender else 'ALL'}")
    print("Paired Hedge Intervention Test (source vs recover)")
    print("Keep only ids where hedge level changed.")
    print("============================================================")
    print(f"Pairs (hedge changed): {len(paired)}")
    print(f"Flip rate (label changes): {flip_rate:.4f}")
    print(f"Mean ΔP(suicide) = P_recover - P_source: {mean_delta:+.4f}")
    print()
    print("Directional effect (align to hedge change direction):")
    print("  sign_aligned = ΔP(suicide) * sign(Δhedge)")
    print("  > 0 means 'more hedging' increased P(suicide) on average.")
    print(f"Mean directional effect: {mean_effect:+.4f}")
    print(f"Wilcoxon signed-rank approx: W={W:.2f}, p≈{p_wil:.4g}")
    print()

    for yval, name in [(0, "TRUE non-suicide"), (1, "TRUE suicide")]:
        sub = paired[paired["y_true"] == yval]
        if len(sub) == 0:
            continue
        sign = np.sign(sub["delta_hedge"].to_numpy())
        sign_aligned = (sub["delta_p"].to_numpy()) * sign
        mask = (~np.isnan(sign_aligned)) & (sign != 0)
        sign_aligned = sign_aligned[mask]
        mean_eff = float(np.nanmean(sign_aligned)) if len(sign_aligned) else np.nan
        W2, p2 = wilcoxon_signed_rank(sign_aligned) if len(sign_aligned) else (np.nan, np.nan)
        print(f"--- {name} ---")
        print(f"n={len(sub)} | flip_rate={float(sub['flip'].mean()):.4f} | mean ΔP={float(sub['delta_p'].mean()):+.4f}")
        print(f"mean directional effect={mean_eff:+.4f} | Wilcoxon p≈{p2:.4g}")
        print()

    # ----------------------------
    # Observational (FULL split)
    # ----------------------------
    print("============================================================")
    print("Observational: Error rates by hedge level (SOURCE text) on FULL split")
    print("============================================================")
    src_rates = error_rates_by_hedge_full(df_pred, "source")
    print(src_rates.to_string(index=False))
    print()

    print("============================================================")
    print("Observational: Error rates by hedge level (RECOVER text) on FULL split")
    print("============================================================")
    rec_rates = error_rates_by_hedge_full(df_pred, "recover")
    print(rec_rates.to_string(index=False))
    print()

    # save outputs
    base = os.path.splitext(os.path.basename(args.csv))[0]
    tag = f"{args.split}" + (f"_{args.gender}" if args.gender else "")
    out_dir = args.splits_dir if args.splits_dir else os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(out_dir, exist_ok=True)

    paired_path = os.path.join(out_dir, f"{base}.{tag}.paired_hedge_test.csv")
    fullpred_path = os.path.join(out_dir, f"{base}.{tag}.fullpred.csv")

    paired.to_csv(paired_path, index=False)
    df_pred.to_csv(fullpred_path, index=False)

    print(f"[OK] Wrote paired results to: {paired_path}")
    print(f"[OK] Wrote full-split predictions to: {fullpred_path}")


if __name__ == "__main__":
    main()
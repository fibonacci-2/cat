#!/usr/bin/env python3
"""
Test whether a suicide classifier uses HEDGING as a shortcut cue.

Input CSV schema (must include):
id, source, recover, class, gender, source_hedge_level, recover_hedge_level

Core idea:
- Use paired examples per id: (source vs recover) where meaning should be same but hedge differs.
- Measure how predictions change when only hedging changes.
- Quantify: (1) flip rate, (2) change in P(suicide), (3) causal-style effect via paired tests,
  and (4) whether hedge level predicts errors (FPR/FNR) while controlling for label.

You provide:
1) A pretrained classifier on HF OR a local model
2) Path to the CSV
"""

import argparse
import math
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ----------------------------
# Utils
# ----------------------------
HEDGE_ORDER = {"small": 0, "low": 0, "medium": 1, "mid": 1, "high": 2, "large": 2}

def norm_hedge(x: str) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "unknown"
    x = str(x).strip().lower()
    return x

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
    (Good enough for a quick test; for publication use scipy.stats.wilcoxon.)
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
    # Normal approximation
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0:
        return W, 1.0
    z = (W - mu) / sigma
    # two-sided p
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return W, p


# ----------------------------
# Model wrapper
# ----------------------------
@dataclass
class PredResult:
    p_suicide: np.ndarray       # shape [N]
    yhat: np.ndarray            # shape [N] 0/1
    label2id: dict
    suicide_id: int
    non_suicide_id: int


def load_clf(model_name: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    mdl.to(device)
    mdl.eval()

    label2id = {k.lower(): int(v) for k, v in mdl.config.label2id.items()} if mdl.config.label2id else {}
    # best-effort mapping
    # prefer explicit "suicide"/"non-suicide" labels
    def find_label(keys: List[str]) -> Optional[int]:
        for k in keys:
            for name, idx in label2id.items():
                if k in name:
                    return idx
        return None

    suicide_id = find_label(["suicide"])
    non_suicide_id = find_label(["non-suicide", "nonsuicide", "non suicide", "control", "negative"])

    # fallback: assume binary with {0,1} where 1 is positive class
    if suicide_id is None or non_suicide_id is None:
        if mdl.config.num_labels == 2:
            suicide_id = 1
            non_suicide_id = 0
        else:
            raise ValueError(
                "Could not infer suicide/non-suicide label ids from model config. "
                "Please use a binary model with clear label2id or modify mapping."
            )

    return tok, mdl, label2id, suicide_id, non_suicide_id


@torch.no_grad()
def predict_probs(texts: List[str], tok, mdl, device: str, batch_size: int,
                  suicide_id: int) -> Tuple[np.ndarray, np.ndarray]:
    ps = []
    logits_all = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = mdl(**enc)
        logits = out.logits.detach().cpu().numpy()
        probs = softmax_np(logits)
        ps.append(probs[:, suicide_id])
        logits_all.append(logits)
    p_suicide = np.concatenate(ps, axis=0)
    logits_all = np.concatenate(logits_all, axis=0)
    yhat = (p_suicide >= 0.5).astype(int)
    return p_suicide, yhat


# ----------------------------
# Hypothesis tests
# ----------------------------
def paired_hedge_effect(df: pd.DataFrame, tok, mdl, device: str, batch_size: int,
                        suicide_id: int) -> pd.DataFrame:
    """
    Within each id, compare predictions on source vs recover.
    Keep only ids where hedge changed (source_hedge_level != recover_hedge_level).
    """
    required = ["id", "source", "recover", "class", "source_hedge_level", "recover_hedge_level"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    d = df.copy()
    d["source_hedge_level"] = d["source_hedge_level"].map(norm_hedge)
    d["recover_hedge_level"] = d["recover_hedge_level"].map(norm_hedge)

    # drop rows with empty text
    d["source"] = d["source"].astype(str)
    d["recover"] = d["recover"].astype(str)

    # keep where hedge differs
    d = d[d["source_hedge_level"] != d["recover_hedge_level"]].copy()
    if len(d) == 0:
        raise ValueError("No rows where hedge level differs between source and recover.")

    # predict
    p_src, y_src = predict_probs(d["source"].tolist(), tok, mdl, device, batch_size, suicide_id)
    p_rec, y_rec = predict_probs(d["recover"].tolist(), tok, mdl, device, batch_size, suicide_id)

    d["p_suicide_source"] = p_src
    d["p_suicide_recover"] = p_rec
    d["yhat_source"] = y_src
    d["yhat_recover"] = y_rec

    d["delta_p"] = d["p_suicide_recover"] - d["p_suicide_source"]
    d["flip"] = (d["yhat_source"] != d["yhat_recover"]).astype(int)

    # direction of hedging change (numerical ordering)
    d["src_hnum"] = d["source_hedge_level"].map(hedge_to_num)
    d["rec_hnum"] = d["recover_hedge_level"].map(hedge_to_num)
    d["delta_hedge"] = d["rec_hnum"] - d["src_hnum"]  # + means more hedged in recover

    # label numeric: suicide=1 non-suicide=0
    d["y_true"] = (d["class"].astype(str).str.lower().str.strip() == "suicide").astype(int)

    return d


def error_rates_by_hedge(df: pd.DataFrame, use_col: str) -> pd.DataFrame:
    """
    Compute FPR/FNR grouped by hedge level for a given text column's predictions.
    use_col in {"source","recover"} (uses yhat_source/yhat_recover).
    """
    assert use_col in ("source", "recover")
    hedge_col = f"{use_col}_hedge_level"
    pred_col = f"yhat_{use_col}"
    p_col = f"p_suicide_{use_col}"

    g = df.copy()
    g[hedge_col] = g[hedge_col].map(norm_hedge)

    rows = []
    for level, sub in g.groupby(hedge_col):
        y = sub["y_true"].to_numpy()
        yhat = sub[pred_col].to_numpy()
        # confusion parts
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
            "mean_p_suicide": float(sub[p_col].mean()),
        })
    return pd.DataFrame(rows).sort_values(["hedge_level"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV with columns: id,source,recover,class,gender,source_hedge_level,recover_hedge_level")
    ap.add_argument("--model", required=True, help="HF model name/path for sequence classification")
    ap.add_argument("--gender", default=None, choices=["M", "F"], help="Filter gender (optional)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--min_pairs", type=int, default=20, help="Warn if too few hedge-changed pairs")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    if args.gender is not None and "gender" in df.columns:
        df = df[df["gender"].astype(str).str.upper() == args.gender].copy()

    tok, mdl, label2id, suicide_id, non_suicide_id = load_clf(args.model, args.device)

    paired = paired_hedge_effect(df, tok, mdl, args.device, args.batch_size, suicide_id)

    if len(paired) < args.min_pairs:
        print(f"[WARN] Only {len(paired)} hedge-changed pairs. Results may be unstable.", file=sys.stderr)

    # ----------------------------
    # Primary causal-style tests
    # ----------------------------
    # H1: Increasing hedging decreases P(suicide) (or increases) consistently.
    # We align delta_p to delta_hedge direction:
    #   sign_aligned = delta_p * sign(delta_hedge)
    # If hedging increases (delta_hedge>0), positive sign_aligned means P(suicide) increased with more hedging.
    sign = np.sign(paired["delta_hedge"].to_numpy())
    delta_p = paired["delta_p"].to_numpy()
    sign_aligned = delta_p * sign
    # Drop where delta_hedge is 0 or NaN mapping
    mask = (~np.isnan(sign_aligned)) & (sign != 0)
    sign_aligned = sign_aligned[mask]

    mean_effect = float(np.nanmean(sign_aligned)) if len(sign_aligned) else np.nan
    W, p_wil = wilcoxon_signed_rank(sign_aligned) if len(sign_aligned) else (np.nan, np.nan)

    flip_rate = float(paired["flip"].mean())
    mean_delta = float(paired["delta_p"].mean())

    print("============================================================")
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

    # ----------------------------
    # Stratify by true label (detect shortcut vs label-consistent behavior)
    # If hedging changes predictions similarly in BOTH classes, that's stronger shortcut evidence.
    # ----------------------------
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
    # Error rates by hedge level (observational)
    # ----------------------------
    print("============================================================")
    print("Observational: Error rates by hedge level (SOURCE text)")
    print("============================================================")
    src_rates = error_rates_by_hedge(paired, "source")
    print(src_rates.to_string(index=False))
    print()

    print("============================================================")
    print("Observational: Error rates by hedge level (RECOVER text)")
    print("============================================================")
    rec_rates = error_rates_by_hedge(paired, "recover")
    print(rec_rates.to_string(index=False))
    print()

    # ----------------------------
    # Save detailed paired results for analysis/plots
    # ----------------------------
    out_path = args.csv.replace(".csv", "") + ".paired_hedge_test.csv"
    paired.to_csv(out_path, index=False)
    print(f"[OK] Wrote paired results to: {out_path}")


if __name__ == "__main__":
    main()
import argparse
import numpy as np
import pandas as pd


HEDGE_ORDER = {"small": 0, "low": 0, "medium": 1, "med": 1, "mid": 1, "high": 2, "large": 2}


def norm_hedge(x):
    if x is None:
        return np.nan
    x = str(x).strip().lower()
    if x in ("", "nan", "none"):
        return np.nan
    return x


def hedge_to_num(x):
    x = norm_hedge(x)
    return float(HEDGE_ORDER.get(x, np.nan))


def bootstrap_diff_means(a, b, n_boot=20000, ci=0.95, seed=123):
    """Bootstrap CI for mean(a) - mean(b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return {"diff": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    rng = np.random.default_rng(seed)
    ai = rng.integers(0, len(a), size=(n_boot, len(a)))
    bi = rng.integers(0, len(b), size=(n_boot, len(b)))
    boot = a[ai].mean(axis=1) - b[bi].mean(axis=1)

    alpha = (1.0 - ci) / 2.0
    return {
        "diff": float(a.mean() - b.mean()),
        "ci_low": float(np.quantile(boot, alpha)),
        "ci_high": float(np.quantile(boot, 1.0 - alpha)),
    }


def cohens_d_ind(a, b):
    """Cohen's d for independent groups."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    va = a.var(ddof=1); vb = b.var(ddof=1)
    sp = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else np.inf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Training CSV")
    ap.add_argument("--use", choices=["source", "recover"], default="source",
                    help="Which hedge column to analyze: source_hedge_level or recover_hedge_level")
    ap.add_argument("--gender", choices=["M", "F"], default=None)
    ap.add_argument("--label_col", default="class", help="Column containing labels: 'suicide'/'non-suicide'")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    if args.gender is not None and "gender" in df.columns:
        df = df[df["gender"].astype(str).str.upper() == args.gender].copy()

    hedge_col = f"{args.use}_hedge_level"
    if hedge_col not in df.columns:
        raise ValueError(f"Missing column: {hedge_col}")

    # map labels
    y = df[args.label_col].astype(str).str.strip().str.lower()
    y_true = (y == "suicide").astype(int)

    # numeric hedge
    h = df[hedge_col].map(hedge_to_num).to_numpy()

    hs = h[y_true == 1]  # suicide
    hn = h[y_true == 0]  # non-suicide

    print("============================================================")
    print(f"Natural hedge difference by label (using {hedge_col})")
    if args.gender:
        print(f"Gender filter: {args.gender}")
    print("============================================================")
    print(f"n_suicide     = {np.sum(~np.isnan(hs))}")
    print(f"n_non_suicide = {np.sum(~np.isnan(hn))}")
    print()

    mean_s = float(np.nanmean(hs)) if np.sum(~np.isnan(hs)) else np.nan
    mean_n = float(np.nanmean(hn)) if np.sum(~np.isnan(hn)) else np.nan
    print(f"Mean hedge (suicide)     = {mean_s:.4f}")
    print(f"Mean hedge (non-suicide) = {mean_n:.4f}")
    print(f"Diff (suicide - non)     = {mean_s - mean_n:+.4f}")
    print()

    # bootstrap CI for diff in means (suicide - non)
    ci = bootstrap_diff_means(hs, hn)
    print(f"Bootstrap 95% CI for diff (suicide - non): [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]")
    print()

    # effect size
    d = cohens_d_ind(hs, hn)
    print(f"Cohen's d (independent) for hedge difference (suicide - non): {d:+.4f}")
    print()

    # helpful label-wise distribution counts
    def counts(arr):
        arr = arr[~np.isnan(arr)].astype(int)
        vals, cts = np.unique(arr, return_counts=True)
        return dict(zip(vals.tolist(), cts.tolist()))

    print("Counts by hedge bin (0=small/low, 1=med, 2=high):")
    print(f"  suicide:     {counts(hs)}")
    print(f"  non-suicide: {counts(hn)}")


if __name__ == "__main__":
    main()
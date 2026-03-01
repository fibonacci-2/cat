import numpy as np
import pandas as pd


def _cohens_d_paired(d: np.ndarray) -> float:
    """Cohen's d for paired samples = mean(diff) / std(diff)."""
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return np.nan
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else np.inf


def _hedges_g_paired(d: np.ndarray) -> float:
    """Hedges' g for paired diffs (small-sample corrected Cohen's d)."""
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 2:
        return np.nan
    cd = _cohens_d_paired(d)
    if not np.isfinite(cd):
        return cd
    J = 1.0 - (3.0 / (4.0 * n - 9.0)) if n > 2 else 1.0
    return float(J * cd)


def bootstrap_ci_mean(
    d: np.ndarray,
    n_boot: int = 20000,
    ci: float = 0.95,
    seed: int = 123,
) -> dict:
    """
    Bootstrap CI for mean paired difference.
    Returns dict with mean, ci_low, ci_high.
    """
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = d[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(boot_means, alpha))
    hi = float(np.quantile(boot_means, 1.0 - alpha))
    return {"mean": float(d.mean()), "ci_low": lo, "ci_high": hi}


def bootstrap_ci_effectsize(
    d: np.ndarray,
    n_boot: int = 20000,
    ci: float = 0.95,
    seed: int = 123,
    effect: str = "hedges_g",  # or "cohens_d"
) -> dict:
    """
    Bootstrap CI for paired effect size (Cohen's d or Hedges' g) computed on diffs.
    Returns dict with point_est, ci_low, ci_high.
    """
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 2:
        return {"point_est": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    eff_fn = _hedges_g_paired if effect.lower() in ("hedges_g", "g") else _cohens_d_paired
    point = eff_fn(d)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        boot[b] = eff_fn(d[idx[b]])

    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(boot, alpha))
    hi = float(np.quantile(boot, 1.0 - alpha))
    return {"point_est": float(point), "ci_low": lo, "ci_high": hi}


def summarize_paired_effects(paired_csv: str, label_col: str = "y_true") -> None:
    """
    Reads your saved paired results CSV (the one your script writes)
    and prints bootstrap CI + effect size for:
      - overall ΔP (recover - source)
      - by true label: non-suicide (0) and suicide (1)
    """
    df = pd.read_csv(paired_csv)

    # ensure columns exist
    for c in ["delta_p", label_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {paired_csv}")

    def report(name: str, dvals: np.ndarray):
        mci = bootstrap_ci_mean(dvals)
        gci = bootstrap_ci_effectsize(dvals, effect="hedges_g")
        dci = bootstrap_ci_effectsize(dvals, effect="cohens_d")
        print(f"=== {name} ===")
        print(f"n = {np.sum(~np.isnan(dvals))}")
        print(f"Mean ΔP(suicide) = {mci['mean']:+.4f}  [{mci['ci_low']:+.4f}, {mci['ci_high']:+.4f}]  (bootstrap {int(0.95*100)}% CI)")
        print(f"Hedges' g (paired) = {gci['point_est']:+.4f}  [{gci['ci_low']:+.4f}, {gci['ci_high']:+.4f}]")
        print(f"Cohen's d (paired) = {dci['point_est']:+.4f}  [{dci['ci_low']:+.4f}, {dci['ci_high']:+.4f}]")
        print()

    # overall
    report("Overall", df["delta_p"].to_numpy())

    # by label
    if set(df[label_col].unique()).issuperset({0, 1}):
        report("True non-suicide (y_true=0)", df.loc[df[label_col] == 0, "delta_p"].to_numpy())
        report("True suicide (y_true=1)", df.loc[df[label_col] == 1, "delta_p"].to_numpy())
    else:
        print(f"[WARN] {label_col} does not contain both 0 and 1; skipping stratified report.")


# -------------------------
# Usage
# -------------------------
summarize_paired_effects("/SEAS/home/g21775526/Downloads/universal_df.paired_hedge_test.csv")
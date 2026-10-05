"""
D25 supporting tables (exploratory; changes no model, threshold or primary result).

1. Collinearity support for the D24 explanation: Spearman correlations between the QR features
   that received offsetting E3 weights, computed on CLEAN TRAIN images only, with the locked
   E3 coefficient signs next to them.
2. What the prototype demo bands (0.30 / 0.70) mean for E1, measured on VALIDATION only
   (test split is never loaded here).

Run:  python -m src.supporting_tables
"""
from __future__ import annotations

import itertools
import warnings

import joblib
import numpy as np
import pandas as pd

from src import config
from src.train_models import load_train_val

PAIRS_OF_INTEREST = [("qr_gray_std", "qr_blur_score"), ("qr_gray_std", "qr_edge_density"),
                     ("qr_blur_score", "qr_edge_density")]
EXTRA = ["url_length"]


def correlation_table(train: pd.DataFrame, coefs: pd.Series) -> pd.DataFrame:
    feats = sorted({f for p in PAIRS_OF_INTEREST for f in p})
    rows = []
    for a, b in itertools.combinations(feats + EXTRA, 2):
        if train[a].nunique() < 2 or train[b].nunique() < 2:
            continue
        rows.append({"feature_a": a, "feature_b": b,
                     "spearman_rho_train_clean": round(float(train[a].corr(train[b], method="spearman")), 3),
                     "coef_a_E3": round(float(coefs.get(a, np.nan)), 3),
                     "coef_b_E3": round(float(coefs.get(b, np.nan)), 3),
                     "pair_of_interest": (a, b) in PAIRS_OF_INTEREST or (b, a) in PAIRS_OF_INTEREST})
    return pd.DataFrame(rows)


def band_table(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    lo, hi = config.RISK_THRESHOLDS["suspicious"], config.RISK_THRESHOLDS["high_risk"]
    band = np.where(p >= hi, "HIGH RISK", np.where(p >= lo, "SUSPICIOUS", "LOW RISK"))
    rows = []
    for name in ["LOW RISK", "SUSPICIOUS", "HIGH RISK"]:
        m = band == name
        n_mal, n_ben = int((m & (y == 1)).sum()), int((m & (y == 0)).sum())
        rows.append({"band": name, "n": int(m.sum()), "n_malicious": n_mal, "n_benign": n_ben,
                     "share_malicious": round(n_mal / m.sum(), 3) if m.sum() else float("nan"),
                     "pct_of_all_malicious": round(n_mal / (y == 1).sum(), 3),
                     "pct_of_all_benign": round(n_ben / (y == 0).sum(), 3)})
    return pd.DataFrame(rows)


def main() -> None:
    train, val = load_train_val()
    e3 = joblib.load(config.MODELS_DIR / "E3_fusion.joblib")
    e1 = joblib.load(config.MODELS_DIR / "E1_url_only.joblib")
    ct = e3["pipeline"].named_steps["prep"]
    order = [c for n, _, cols in ct.transformers_ if n != "remainder" for c in cols]
    coefs = pd.Series(e3["pipeline"].named_steps["clf"].coef_.ravel(), index=order)

    corr = correlation_table(train, coefs)
    corr.to_csv(config.TABLES_DIR / "d25_qr_feature_correlations_train.csv", index=False)
    print("Spearman on clean TRAIN images (n =", len(train), ")\n", corr.to_string(index=False))

    with warnings.catch_warnings():
        # Spurious on Apple Silicon (NumPy 2 + Accelerate); soundness is asserted, not assumed.
        warnings.filterwarnings("ignore", message=r".*encountered in matmul", category=RuntimeWarning)
        p = e1["pipeline"].predict_proba(val[e1["features"]])[:, 1]
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "non-finite E1 scores"
    bands = band_table(val["label"].to_numpy(), p)
    bands.to_csv(config.TABLES_DIR / "d25_e1_demo_bands_validation.csv", index=False)
    print("\nE1 demo bands on VALIDATION (n =", len(val), ")\n", bands.to_string(index=False))


if __name__ == "__main__":
    main()

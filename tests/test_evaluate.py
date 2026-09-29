"""Evaluation statistics tests. Run:  python -m pytest -q"""
import numpy as np
import pandas as pd

from src.evaluate_models import binary_metrics, length_matched_subset, mcnemar


def test_binary_metrics_counts():
    m = binary_metrics(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
    assert m["recall_malicious"] == 0.5 and m["fpr"] == 0.5


def test_mcnemar_holm_is_monotone_and_not_below_raw():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    preds = pd.DataFrame({"label": y})
    for name in ["M1_length_only", "M1_plus_QR", "E1_url_only", "E2_qr_only", "E3_fusion"]:
        flip = rng.random(300) < {"E3_fusion": 0.05, "E1_url_only": 0.15}.get(name, 0.3)
        preds[f"{name}_pred"] = np.where(flip, 1 - y, y)
    df = mcnemar(preds)
    assert (df["p_holm"] >= df["p_exact"] - 1e-9).all()
    assert df["p_holm"].is_monotonic_increasing


def test_length_matching_balances_classes_per_bin():
    rng = np.random.default_rng(1)
    test = pd.DataFrame({"url_length": np.r_[rng.integers(20, 80, 150), rng.integers(30, 120, 150)],
                         "label": [0] * 150 + [1] * 150})
    subset, meta = length_matched_subset(test)
    assert subset["label"].value_counts()[0] == subset["label"].value_counts()[1]
    assert meta["subset_size"] == len(subset) > 0

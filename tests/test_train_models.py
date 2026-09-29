"""Training-protocol tests. Run:  python -m pytest -q"""
import numpy as np
import pandas as pd

from src.train_models import choose_thresholds, training_filter


def test_training_filter_drops_constant_and_rare_flags():
    train = pd.DataFrame({"const": [1.0] * 10, "rare_flag": [1] + [0] * 9,
                          "ok_flag": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0], "numeric": np.arange(10.0)})
    kept, dropped = training_filter(train, list(train.columns))
    assert kept == ["ok_flag", "numeric"]
    assert set(dropped) == {"const", "rare_flag"}


def test_primary_threshold_is_highest_meeting_recall_and_fpr():
    y = np.array([1] * 10 + [0] * 10)
    p = np.concatenate([np.linspace(0.95, 0.5, 10), np.linspace(0.6, 0.05, 10)])
    thr = choose_thresholds(y, p)
    prim = thr["primary"]
    assert prim["recall_malicious"] >= 0.9 and prim["fpr"] <= 0.5
    assert prim["rule"].startswith("recall")
    # any higher threshold must break the recall target
    higher = [t for t in np.unique(p) if t > prim["threshold"]]
    assert all(((p >= t)[y == 1].mean() < 0.9) for t in higher)


def test_fallback_when_target_infeasible():
    y = np.array([1, 1, 0, 0])
    # Catching both malicious cases needs t <= 0.1, which also flags both benign ones (FPR 1.0 > 0.5)
    thr = choose_thresholds(y, np.array([0.1, 0.9, 0.8, 0.85]))
    assert thr["primary"]["rule"].startswith("fallback")

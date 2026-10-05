"""
Day 4c: pick demo / poster example images by a FIXED, pre-stated rule (no cherry-picking).

From the frozen TEST split (clean images, exact decode), score each with the locked E1 and take, in each of
four cells, the example with the MEDIAN score:
    benign   -> LOW RISK     (correct)         malicious -> HIGH RISK   (correct)
    benign   -> HIGH RISK    (false alarm)     malicious -> LOW RISK    (miss)
Inference only: no model, threshold or result changes. URLs are printed as inert text - do not open them.

Run:  python -m src.demo_examples
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src import config
from src import explainability as ex

CELLS = [("benign", "LOW RISK", "correct"), ("malicious", "HIGH RISK", "correct"),
         ("benign", "HIGH RISK", "false alarm"), ("malicious", "LOW RISK", "miss")]
TRUTH = {"benign": config.LABEL_BENIGN, "malicious": config.LABEL_MALICIOUS}


def pick_examples(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    assert set(df["label"].unique()) <= set(TRUTH.values())
    d = df.assign(score=p, band=[config.probability_to_risk_label(x) for x in p],
                  truth=np.where(df["label"] == config.LABEL_MALICIOUS, "malicious", "benign"))
    rows = []
    for truth, band, kind in CELLS:
        cell = d[(d["truth"] == truth) & (d["band"] == band)].sort_values(["score", "original_qr_id"])
        if cell.empty:
            rows.append({"cell": f"{truth} -> {band} ({kind})", "n_in_cell": 0})
            continue
        r = cell.iloc[(len(cell) - 1) // 2]                      # lower median, deterministic
        rows.append({"cell": f"{truth} -> {band} ({kind})", "n_in_cell": len(cell), "score": round(float(r["score"]), 3),
                     "image_path": r["image_path"], "payload": r["decoded_payload"], "original_qr_id": r["original_qr_id"]})
    return pd.DataFrame(rows)


def main() -> None:
    feats = pd.read_csv(config.FEATURES_CSV, keep_default_na=False, na_values=[""])
    test = feats[(feats["split"] == "test") & (feats["distortion_type"] == "none") & (feats["decode_success"] == 1)]
    assert len(test) and set(test["split"]) == {"test"}
    b = ex.load_e1()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*encountered in matmul", category=RuntimeWarning)
        p = b["pipeline"].predict_proba(test[b["features"]])[:, 1]
    assert np.isfinite(p).all()
    out = pick_examples(test.reset_index(drop=True), p)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.TABLES_DIR / "demo_examples.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 90):
        print(out.to_string(index=False))
    print("\nSaved outputs/tables/demo_examples.csv. Upload the image_path files in the app (URLs: text only, do not open).")


if __name__ == "__main__":
    main()

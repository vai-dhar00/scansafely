import numpy as np
import pandas as pd
import pytest

from src import config
from src.error_analysis_summary import select_errors, summarise


def _data(n=100, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    s = np.clip(0.3 * y + rng.uniform(0, 0.7, n), 0, 1)
    preds = pd.DataFrame({"original_qr_id": [f"Q{i:03d}" for i in range(n)], "label": y, "url_length": rng.integers(10, 90, n),
                          "M_score": s, "M_pred": (s >= 0.5).astype(int)})
    urls = pd.DataFrame({"original_qr_id": preds["original_qr_id"], "url": [f"http://x{i}.test/" for i in range(n)]})
    return preds, urls


def test_selection_is_most_confident_errors_at_locked_threshold():
    preds, urls = _data()
    ws = select_errors(preds, urls, "M", k=5)
    fp = ws[ws.error_type == "false_positive"]; fn = ws[ws.error_type == "false_negative"]
    allfp = preds[(preds.label == 0) & (preds.M_pred == 1)]; allfn = preds[(preds.label == 1) & (preds.M_pred == 0)]
    assert len(fp) == 5 and len(fn) == 5
    assert set(fp.risk_score) <= set(allfp.M_score.round(3)) and fp.risk_score.min() >= allfp.M_score.nlargest(5).min().round(3)
    assert fn.risk_score.max() <= allfn.M_score.nsmallest(5).max().round(3)
    assert (fp.true_label == "benign").all() and (fn.true_label == "suspicious/malicious").all()


def test_summarise_requires_complete_and_few_categories():
    preds, urls = _data()
    ws = select_errors(preds, urls, "M", k=4)                     # 8 rows
    with pytest.raises(SystemExit):
        summarise(ws)                                             # blank categories and notes
    many = ws.assign(your_category=[f"c{i}" for i in range(len(ws))], evidence_note="n")
    with pytest.raises(SystemExit):
        summarise(many)                                           # 8 distinct categories > 6
    ok = ws.assign(your_category=["Short URL", "short url", "Long path", "Long path"] * 2, evidence_note="n")
    out, counts = summarise(ok)
    assert set(out["category"]) == {"short url", "long path"} and counts["n_rows"].sum() == 8

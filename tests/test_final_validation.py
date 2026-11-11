"""The comparison helper behind the final integrity check must catch any changed number."""
import pandas as pd

from src.final_validation import same_as_saved


def _table():
    return pd.DataFrame({"model": ["a", "b"], "roc_auc": [0.8630, 0.6614], "threshold": [0.49, float("nan")]})


def test_identical_table_passes(tmp_path):
    path = tmp_path / "t.csv"
    _table().to_csv(path, index=False)
    ok, detail = same_as_saved(_table(), path)
    assert ok, detail


def test_changed_number_text_or_shape_fails(tmp_path):
    path = tmp_path / "t.csv"
    _table().to_csv(path, index=False)
    changed = _table()
    changed.loc[0, "roc_auc"] += 1e-4
    assert not same_as_saved(changed, path)[0]
    renamed = _table()
    renamed.loc[1, "model"] = "c"
    assert not same_as_saved(renamed, path)[0]
    assert not same_as_saved(_table().iloc[:1], path)[0]

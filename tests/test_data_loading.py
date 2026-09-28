"""Tests for cleaning and sampling. Run from the project root:  python -m pytest -q"""
import pandas as pd

from src import config
from src.data_loading import clean_urls, sample_balanced


def _frame(rows):
    return pd.DataFrame(rows, columns=["url", "label"]).assign(source="test")


def test_conflicting_labels_are_removed_completely():
    df = _frame([("https://a.example", 0), ("https://a.example", 1), ("https://b.example", 0)])
    cleaned, report = clean_urls(df)
    assert list(cleaned["url"]) == ["https://b.example"]
    assert report["dropped_conflicting_label_rows"] == 2


def test_duplicates_and_blank_rows_are_removed():
    df = _frame([("https://a.example", 0), ("https://a.example", 0), ("  ", 1), ("has space.example", 1)])
    cleaned, report = clean_urls(df)
    assert len(cleaned) == 1
    assert report["dropped_exact_duplicates"] == 1
    assert report["dropped_empty_or_whitespace"] == 2


def test_sampling_is_balanced_and_reproducible():
    df = _frame([(f"https://b{i}.example", 0) for i in range(50)] +
                [(f"https://m{i}.example", 1) for i in range(50)])
    s1 = sample_balanced(df, 20, config.RANDOM_SEED)
    s2 = sample_balanced(df, 20, config.RANDOM_SEED)
    assert s1["label"].value_counts().to_dict() == {0: 20, 1: 20}
    assert s1["url"].tolist() == s2["url"].tolist()


def test_phiusiil_label_map_is_inverted():
    # PhiUSIIL documents 1 = legitimate. Our convention is 0 = benign.
    assert config.RAW_DATASETS["phiusiil"]["label_map"] == {1: 0, 0: 1}

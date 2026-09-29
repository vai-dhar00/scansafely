"""Split logic tests. Run:  python -m pytest -q"""
import pandas as pd

from src.split_data import build_groups, is_nontrivial_template, make_candidate, score_split


def _urls(rows):
    return pd.DataFrame(rows, columns=["registered_domain", "url_template", "label"])


def test_nontrivial_template_rule():
    assert not is_nontrivial_template("/")
    assert is_nontrivial_template("/login.php")
    assert is_nontrivial_template("/?id=<V>")


def test_groups_merge_domains_and_repeated_templates_but_not_generic():
    rows = [("a.com", "/", 0), ("a.com", "/x", 0),                      # same domain -> one group
            ("b.com", "/kit/verify", 1), ("c.com", "/kit/verify", 1), ("d.com", "/kit/verify", 1),  # template x3 -> merged
            ("e.com", "/solo", 0)]
    rows += [(f"g{i}.com", "/index.html", i % 2) for i in range(40)]   # >30 uses -> generic, NOT merged
    df = _urls(rows)
    gid, report = build_groups(df)
    assert gid[0] == gid[1]
    assert gid[2] == gid[3] == gid[4]
    assert gid[5] != gid[2]
    assert gid.iloc[6:].nunique() == 40
    assert set(report["action"]) == {"merged", "reported_generic_not_merged"}


def test_candidate_is_deterministic_and_keeps_groups_whole():
    df = _urls([(f"d{i}.com", "/", i % 2) for i in range(200)])
    df["split_group_id"], _ = build_groups(df)
    groups = df.groupby("split_group_id")["label"].agg(
        n_benign=lambda s: int((s == 0).sum()), n_malicious=lambda s: int((s == 1).sum())).reset_index()
    a, b = make_candidate(groups, 42, len(df)), make_candidate(groups, 42, len(df))
    assert a == b
    trial = df.assign(split=df["split_group_id"].map(a))
    score, stats = score_split(trial)
    assert stats["test_n"] == 30 and stats["val_n"] == 30 and stats["test_mal_frac"] == 0.5
    assert score < 0.01

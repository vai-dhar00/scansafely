"""
Grouped, label-stratified train/validation/test split (D1, D12, D15), then FREEZE.

1. Split groups (indivisible units): union-find over
     - registered domain, and
     - repeated NON-TRIVIAL URL templates (>= TEMPLATE_GROUP_MIN_COUNT uses),
       except "generic" templates larger than TEMPLATE_GROUP_MAX_SIZE (reported, not merged).
2. N_SPLIT_CANDIDATES candidate splits, candidate i shuffles groups with seed RANDOM_SEED + i
   and fills test, then validation, to 15% each with ~50/50 classes; the rest is train.
3. Each candidate gets the pre-registered score; the lowest wins (ties -> lowest index).
4. Writes the split into urls.csv and features.csv and saves the frozen artefacts.

This uses ONLY ids, labels, domains and templates - never features or model outputs.

Run ONCE:  python -m src.split_data        then commit the outputs before any training.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from src import config

SPLITS = ("train", "val", "test")
TARGET = {"train": config.TRAIN_FRACTION, "val": config.VAL_FRACTION, "test": config.TEST_FRACTION}
MANIFEST_CSV = config.PROCESSED_DIR / "split_manifest.csv"
TEST_IDS_CSV = config.PROCESSED_DIR / "test_ids_frozen.csv"


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def is_nontrivial_template(template: str) -> bool:
    """Not '/', and has a non-root path or query keys."""
    path, _, query = template.partition("?")
    return template != "/" and (path not in ("", "/") or bool(query))


def build_groups(urls: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Return a group id per row and a report of template handling."""
    uf = _UnionFind(len(urls))
    for idx in urls.groupby("registered_domain").indices.values():
        for j in idx[1:]:
            uf.union(idx[0], j)

    report = []
    for template, idx in urls.groupby("url_template").indices.items():
        if not is_nontrivial_template(template) or len(idx) < config.TEMPLATE_GROUP_MIN_COUNT:
            continue
        labels = urls["label"].iloc[idx]
        generic = len(idx) > config.TEMPLATE_GROUP_MAX_SIZE
        report.append({"url_template": template, "n_urls": len(idx),
                       "n_benign": int((labels == 0).sum()), "n_malicious": int((labels == 1).sum()),
                       "action": "reported_generic_not_merged" if generic else "merged"})
        if not generic:
            for j in idx[1:]:
                uf.union(idx[0], j)

    roots = [uf.find(i) for i in range(len(urls))]
    # Stable, readable group ids ordered by first appearance
    order = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    group_id = pd.Series([f"G{order[r]:05d}" for r in roots], index=urls.index, name="split_group_id")
    return group_id, pd.DataFrame(report, columns=["url_template", "n_urls", "n_benign", "n_malicious", "action"])


def make_candidate(groups: pd.DataFrame, seed: int, n_total: int) -> dict[str, str]:
    """Greedy stratified fill of test then val using a seeded group order."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    per_class_target = {s: TARGET[s] * n_total / 2 for s in ("test", "val")}
    filled = {s: [0, 0] for s in ("test", "val")}  # [benign, malicious]
    assignment = {}
    for k in order:
        g = groups.iloc[k]
        b, m = int(g["n_benign"]), int(g["n_malicious"])
        dest = "train"
        for s in ("test", "val"):
            if filled[s][0] + b <= per_class_target[s] and filled[s][1] + m <= per_class_target[s]:
                dest = s
                filled[s][0] += b
                filled[s][1] += m
                break
        assignment[g["split_group_id"]] = dest
    return assignment


def score_split(df: pd.DataFrame) -> tuple[float, dict]:
    """Pre-registered objective (D15). Lower is better."""
    n = len(df)
    score, stats = 0.0, {}
    for s in SPLITS:
        part = df[df["split"] == s]
        frac = len(part) / n
        mal = part["label"].mean() if len(part) else np.nan
        stats[f"{s}_n"], stats[f"{s}_mal_frac"] = len(part), round(float(mal), 4) if len(part) else np.nan
        score += abs(frac - TARGET[s]) + (abs(mal - 0.5) if len(part) else 1.0)
        if len(part) == 0 or part["label"].nunique() < 2:
            score += 10
        if abs(frac - TARGET[s]) > config.SPLIT_SIZE_TOLERANCE:
            score += 10
        if s in ("val", "test") and len(part):
            if part.groupby("split_group_id").size().max() / len(part) > config.SPLIT_MAX_GROUP_SHARE:
                score += 1
    return score, stats


def run_split() -> pd.DataFrame:
    urls = pd.read_csv(config.URLS_CSV, keep_default_na=False)
    if (urls["split"] != "").any() or TEST_IDS_CSV.exists():
        raise SystemExit("A split already exists. The test set is frozen - do NOT re-split. "
                         "(Delete split files only if you are deliberately restarting, and log it.)")

    urls["split_group_id"], template_report = build_groups(urls)
    groups = (urls.groupby("split_group_id")["label"]
              .agg(n_benign=lambda s: int((s == 0).sum()), n_malicious=lambda s: int((s == 1).sum()))
              .reset_index())

    candidates, best = [], None
    for i in range(config.N_SPLIT_CANDIDATES):
        seed = config.RANDOM_SEED + i
        assign = make_candidate(groups, seed, len(urls))
        trial = urls.assign(split=urls["split_group_id"].map(assign))
        score, stats = score_split(trial)
        candidates.append({"candidate": i, "seed": seed, "score": round(score, 6), **stats})
        if best is None or score < best[0] - 1e-12:  # strict: ties keep the lower index
            best = (score, i, trial)
    score, best_i, final = best

    # Hard assertions: no group, domain or merged template in more than one split
    assert (final.groupby("split_group_id")["split"].nunique() == 1).all()
    assert (final.groupby("registered_domain")["split"].nunique() == 1).all()
    merged = template_report.loc[template_report["action"] == "merged", "url_template"]
    assert (final[final["url_template"].isin(merged)].groupby("url_template")["split"].nunique() == 1).all()

    # --- write outputs
    config.ensure_directories()
    urls_out = final.drop(columns=["split_group_id"])
    urls_out.to_csv(config.URLS_CSV, index=False)
    manifest = final[["original_qr_id", "label", "registered_domain", "split_group_id", "split"]]
    manifest.to_csv(MANIFEST_CSV, index=False)
    test_ids = manifest[manifest["split"] == "test"][["original_qr_id", "label", "split_group_id"]]
    test_ids.to_csv(TEST_IDS_CSV, index=False)

    feats = pd.read_csv(config.FEATURES_CSV, keep_default_na=False, na_values=[""])
    feats["split"] = feats["original_qr_id"].map(manifest.set_index("original_qr_id")["split"])
    feats.to_csv(config.FEATURES_CSV, index=False)

    pd.DataFrame(candidates).to_csv(config.TABLES_DIR / "split_candidate_scores.csv", index=False)
    if len(template_report):
        dist = final[final["url_template"].isin(template_report["url_template"])]
        per_split = dist.groupby(["url_template", "split"]).size().unstack(fill_value=0)
        template_report = template_report.merge(per_split, left_on="url_template", right_index=True, how="left")
    template_report.to_csv(config.TABLES_DIR / "split_template_report.csv", index=False)

    audit = []
    for s in SPLITS:
        part = final[final["split"] == s]
        audit.append({"split": s, "n": len(part), "fraction": round(len(part) / len(final), 4),
                      "n_benign": int((part["label"] == 0).sum()), "n_malicious": int((part["label"] == 1).sum()),
                      "n_groups": part["split_group_id"].nunique(),
                      "largest_group": int(part.groupby("split_group_id").size().max()),
                      "n_domains": part["registered_domain"].nunique()})
    pd.DataFrame(audit).to_csv(config.TABLES_DIR / "split_audit.csv", index=False)

    sha = hashlib.sha256(TEST_IDS_CSV.read_bytes()).hexdigest()
    (config.TABLES_DIR / "split_freeze.json").write_text(json.dumps({
        "selected_candidate": best_i, "seed": config.RANDOM_SEED + best_i, "score": round(score, 6),
        "n_groups": int(final["split_group_id"].nunique()),
        "templates_merged": int((template_report["action"] == "merged").sum()) if len(template_report) else 0,
        "templates_generic_reported": int((template_report["action"] != "merged").sum()) if len(template_report) else 0,
        "test_ids_sha256": sha,
    }, indent=2))
    return pd.DataFrame(audit)


if __name__ == "__main__":
    audit = run_split()
    freeze = json.loads((config.TABLES_DIR / "split_freeze.json").read_text())
    print(audit.to_string(index=False))
    print(f"\nSelected candidate {freeze['selected_candidate']} (seed {freeze['seed']}), score {freeze['score']}")
    print(f"Split groups: {freeze['n_groups']} | templates merged: {freeze['templates_merged']} | "
          f"generic templates reported (not merged): {freeze['templates_generic_reported']}")
    print(f"Frozen test ids SHA-256: {freeze['test_ids_sha256']}")
    print("\nNEXT: python -m src.validate_dataset, then COMMIT before any training.")

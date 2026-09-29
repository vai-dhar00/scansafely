"""
Automated dataset validation report: PASS / FAIL / INFO for every rule we claim holds.

Checks: dataset integrity, domain cap, split leakage (once splits exist), image files,
clean decode outcomes, features.csv consistency, QR-version exclusion (D10), a static
safety scan for network/browser code, file hashes and a config snapshot.

Run:  python -m src.validate_dataset        (exit code 1 if any check FAILS)
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from datetime import datetime

import pandas as pd

from src import config
from src.feature_pipeline import MODEL_FEATURES
from src.url_features import URL_FEATURES

# Modules that could reach the network or open a browser. None may be imported in project code.
_FORBIDDEN = r"^\s*(import|from)\s+(requests|httpx|aiohttp|urllib\.request|urllib3|http\.client|socket|webbrowser|selenium|playwright)\b"
_results: list[dict] = []


def _check(name: str, ok: bool | None, detail: str = "") -> None:
    status = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    _results.append({"check": name, "status": status, "detail": detail})


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_validation() -> bool:
    urls = pd.read_csv(config.URLS_CSV, keep_default_na=False)
    n = len(urls)

    # --- Dataset integrity
    _check("urls: expected row count", n == 2 * config.N_PER_CLASS, f"{n} rows")
    _check("urls: original_qr_id unique", urls["original_qr_id"].is_unique)
    _check("urls: url unique", urls["url"].is_unique)
    _check("urls: canonical_url unique", urls["canonical_url"].is_unique)
    _check("urls: labels are 0/1 only", set(urls["label"]) <= {0, 1})
    counts = urls["label"].value_counts().to_dict()
    _check("urls: classes balanced", counts.get(0) == counts.get(1), str(counts))
    max_dom = int(urls.groupby("registered_domain").size().max())
    if config.DOMAIN_CAP is not None:
        _check("urls: domain cap respected", max_dom <= config.DOMAIN_CAP, f"largest group {max_dom} <= {config.DOMAIN_CAP}")

    # --- Split leakage (only meaningful once Day 2 assigns splits)
    if (urls["split"] == "").all():
        _check("split: assigned", None, "not yet assigned (Day 2) - leakage checks skipped")
    else:
        _check("split: every row assigned", (urls["split"] != "").all())
        doms = urls.groupby("registered_domain")["split"].nunique()
        _check("split: no registered domain in >1 split", (doms <= 1).all(), f"{int((doms > 1).sum())} leaking domains")
        _check("split: fractions", None, urls["split"].value_counts(normalize=True).round(3).to_dict().__str__())
        manifest_path = config.PROCESSED_DIR / "split_manifest.csv"
        freeze_path = config.TABLES_DIR / "split_freeze.json"
        test_ids_path = config.PROCESSED_DIR / "test_ids_frozen.csv"
        if manifest_path.exists():
            man = pd.read_csv(manifest_path)
            grp = man.groupby("split_group_id")["split"].nunique()
            _check("split: no split group (domain+template) in >1 split", (grp <= 1).all(),
                   f"{int((grp > 1).sum())} leaking groups")
        if freeze_path.exists() and test_ids_path.exists():
            frozen = json.loads(freeze_path.read_text())["test_ids_sha256"]
            _check("split: frozen test ids unchanged (SHA-256)", _sha256(test_ids_path) == frozen)

    # --- Images and clean decoding
    manifest = pd.read_csv(config.QR_MANIFEST_CSV)
    missing = [p for p in manifest["image_path"] if not (config.PROJECT_ROOT / p).exists()]
    _check("images: every QR image file exists", not missing, f"{len(missing)} missing")
    _check("images: one per URL", set(manifest["original_qr_id"]) == set(urls["original_qr_id"]))
    _check("decode: all clean images exact match", (manifest["decode_outcome"] == "exact").all(),
           str(manifest["decode_outcome"].value_counts().to_dict()))

    # --- features.csv
    feats = pd.read_csv(config.FEATURES_CSV, keep_default_na=False, na_values=[""])
    clean = feats[feats["distortion_type"] == "none"]
    _check("features: one clean row per URL", len(clean) == n and clean["original_qr_id"].is_unique, f"{len(clean)} rows")
    _check("features: all model feature columns present", set(MODEL_FEATURES) <= set(feats.columns))
    _check("features: generator QR version NOT a model feature (D10)", "qr_version_generator" not in MODEL_FEATURES)
    exact = feats[feats["decode_outcome"] == "exact"]
    _check("features: no missing URL features when decoded", not exact[URL_FEATURES].isna().any().any())
    merged = feats.merge(urls[["original_qr_id", "split", "label"]], on="original_qr_id", suffixes=("", "_u"))
    _check("features: split/label consistent with urls.csv",
           (merged["split"].fillna("") == merged["split_u"]).all() and (merged["label"] == merged["label_u"]).all())

    # --- Static safety scan of project code
    code_files = sorted(config.PROJECT_ROOT.glob("src/*.py")) + sorted(config.PROJECT_ROOT.glob("app.py"))
    offenders = [f"{f.name}:{i}" for f in code_files if f.name != "validate_dataset.py"
                 for i, line in enumerate(f.read_text().splitlines(), 1) if re.match(_FORBIDDEN, line)]
    _check("safety: no network/browser imports in project code", not offenders, ", ".join(offenders) or "clean")

    # --- Provenance: hashes + config snapshot
    raw = config.RAW_DIR / config.RAW_DATASETS[config.PRIMARY_DATASET]["file"]
    hashes = {p.name: _sha256(p) for p in (raw, config.URLS_CSV, config.QR_MANIFEST_CSV, config.FEATURES_CSV) if p.exists()}
    shutil.copy(config.PROJECT_ROOT / "src" / "config.py", config.OUTPUTS_DIR / "config_snapshot.py")
    _check("provenance: hashes + config snapshot saved", None, "outputs/validation_report.json, outputs/config_snapshot.py")

    passed = all(r["status"] != "FAIL" for r in _results)
    report = {"generated": datetime.now().isoformat(timespec="seconds"),
              "overall": "PASS" if passed else "FAIL", "checks": _results, "sha256": hashes}
    (config.OUTPUTS_DIR / "validation_report.json").write_text(json.dumps(report, indent=2))
    return passed


if __name__ == "__main__":
    ok = run_validation()
    print("DATASET VALIDATION REPORT\n" + "-" * 60)
    for r in _results:
        print(f"[{r['status']}] {r['check']}" + (f"  ({r['detail']})" if r["detail"] else ""))
    print("-" * 60 + f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)

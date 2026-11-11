"""
Final integrity and reproducibility check (D33). READ-ONLY apart from one report file.

It never retrains, never re-splits and never overwrites a saved table, figure or model.
It recomputes results IN MEMORY from the locked models and the frozen test set and compares
them with what is saved on disk, then checks every number quoted in the paper against the
saved tables.

Sections
  A. Frozen artefacts   test-id hash, six model hashes, robustness run used the same hashes
  B. Split              sizes, class balance, no group or registered domain in more than one split
  C. Reproduction       re-score the test set; metrics, predictions, bootstrap CIs, McNemar and
                        length-matched metrics must equal the saved tables
  D. Robustness tables  row count, decode counts, zero mismatches, E1 scores unchanged
  E. Paper numbers      each quoted value equals the saved table at the quoted precision
  F. Housekeeping       required files, decision log, lock file, config snapshot, git state

Run:    python -m src.final_validation
Writes: outputs/final_validation_report.json   (nothing else)
Exit code 0 only if there is no FAIL and no ERROR. WARN items need a human decision.
"""
from __future__ import annotations

import hashlib
import io
import json
import platform
import re
import subprocess
import sys
import warnings
from datetime import datetime
from importlib import metadata

import numpy as np
import pandas as pd

from src import config
from src import evaluate_models as ev

REPORT = config.OUTPUTS_DIR / "final_validation_report.json"
T = config.TABLES_DIR
LAST_DECISION = 32          # highest D-number expected in the log BEFORE this check is logged
E1, E3, E2, M1 = "E1_url_only", "E3_fusion", "E2_qr_only", "M1_length_only"

_results: list[dict] = []


def _record(section: str, check: str, status: str, detail: str = "") -> None:
    _results.append({"section": section, "check": check, "status": status, "detail": str(detail)})


def _run(section: str, check: str, fn) -> None:
    """fn returns (ok, detail) or (status_string, detail). Any exception becomes ERROR, never a crash."""
    try:
        ok, detail = fn()
        status = ok if isinstance(ok, str) else ("PASS" if ok else "FAIL")
    except Exception as exc:  # noqa: BLE001 - a broken check must be reported, not hide the others
        status, detail = "ERROR", f"{type(exc).__name__}: {exc}"
    _record(section, check, status, detail)


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _roundtrip(df: pd.DataFrame) -> pd.DataFrame:
    """Write to CSV in memory and read back, so a fresh frame has the same dtypes as a saved one."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return pd.read_csv(buf)


def same_as_saved(fresh: pd.DataFrame, path, atol: float = 1e-9) -> tuple[bool, str]:
    """Compare a recomputed table with the saved CSV: same shape, columns, text and numbers (within atol)."""
    saved, fresh = pd.read_csv(path), _roundtrip(fresh)
    if list(saved.columns) != list(fresh.columns):
        return False, f"columns differ: saved {list(saved.columns)} vs recomputed {list(fresh.columns)}"
    if saved.shape != fresh.shape:
        return False, f"shape differs: saved {saved.shape} vs recomputed {fresh.shape}"
    worst = 0.0
    for col in saved.columns:
        a, b = saved[col], fresh[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            if not (a.isna() == b.isna()).all():
                return False, f"{col}: missing values in different places"
            diff = float((a - b).abs().max()) if a.notna().any() else 0.0
            worst = max(worst, diff)
            if diff > atol:
                return False, f"{col}: max abs difference {diff:.3g} (allowed {atol:g})"
        elif not (a.fillna("").astype(str) == b.fillna("").astype(str)).all():
            return False, f"{col}: text values differ"
    return True, f"identical to {path.name} ({saved.shape[0]} rows, max numeric difference {worst:.2g})"


# --------------------------------------------------------------------------- A. frozen artefacts
def check_frozen() -> None:
    freeze = json.loads((T / "split_freeze.json").read_text())
    manifest = json.loads(ev.MANIFEST_JSON.read_text())

    _run("A", "frozen test ids match split_freeze.json (SHA-256)",
         lambda: (_sha(config.PROCESSED_DIR / "test_ids_frozen.csv") == freeze["test_ids_sha256"],
                  freeze["test_ids_sha256"][:16] + "..."))
    for name in ev.MODELS:
        _run("A", f"model unchanged since training: {name}",
             lambda n=name: (_sha(config.MODELS_DIR / f"{n}.joblib") == manifest["models"][n]["model_sha256"],
                             manifest["models"][n]["model_sha256"][:16] + "..."))

    def robustness_hashes():
        meta = json.loads((T / "robustness_run_metadata.json").read_text())
        ok = meta["frozen_test_ids_sha256"] == freeze["test_ids_sha256"] and all(
            meta["model_sha256"][n] == manifest["models"][n]["model_sha256"] for n in (E1, E3))
        return ok, f"robustness run of {meta.get('run', '?')}"
    _run("A", "robustness run used the same test ids and the same E1/E3 models", robustness_hashes)


# --------------------------------------------------------------------------- B. split
def check_split() -> None:
    def sizes():
        man = pd.read_csv(config.PROCESSED_DIR / "split_manifest.csv")
        counts = man["split"].value_counts().to_dict()
        return len(man) == 2000 and sorted(counts.values()) == [300, 300, 1400] and counts.get("test") == 300, counts
    _run("B", "split sizes are 1400 / 300 / 300 of 2000", sizes)

    def balance():
        man = pd.read_csv(config.PROCESSED_DIR / "split_manifest.csv")
        overall = man["label"].value_counts().to_dict()
        test = man[man["split"] == "test"]["label"].value_counts().to_dict()
        return (sorted(overall.values()) == [1000, 1000] and sorted(test.values()) == [150, 150],
                f"overall {overall}, test {test}")
    _run("B", "labels are 1000/1000 overall and 150/150 in test", balance)

    for col, what in (("split_group_id", "split group"), ("registered_domain", "registered domain")):
        def leak(c=col):
            man = pd.read_csv(config.PROCESSED_DIR / "split_manifest.csv")
            n = int((man.groupby(c)["split"].nunique() > 1).sum())
            return n == 0, f"{n} leaking, {man[c].nunique()} distinct"
        _run("B", f"no {what} appears in more than one split", leak)

    def frozen_rows():
        test = ev.load_test()                       # asserts clean test rows == frozen ids
        return len(test) == 300 and test["original_qr_id"].is_unique, f"{len(test)} clean test rows"
    _run("B", "features.csv clean test rows are exactly the frozen ids", frozen_rows)


# --------------------------------------------------------------------------- C. reproduction
def check_reproduction() -> dict:
    out: dict = {}
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)   # macOS Accelerate matmul noise
            test = ev.load_test()
            metrics, preds, _ = ev.score_models(test)
        score_cols = [c for c in preds.columns if c.endswith("_score")]
        assert np.isfinite(preds[score_cols].to_numpy()).all(), "non-finite scores"
    except Exception as exc:  # noqa: BLE001
        _record("C", "re-score the frozen test set with the locked models", "ERROR", f"{type(exc).__name__}: {exc}")
        return out
    _record("C", "re-score the frozen test set with the locked models", "PASS", f"{len(test)} rows, 6 models")
    out.update(test=test, preds=preds)

    _run("C", "test_predictions.csv reproduces", lambda: same_as_saved(preds, T / "test_predictions.csv"))
    _run("C", "test_metrics.csv reproduces", lambda: same_as_saved(metrics, T / "test_metrics.csv"))
    _run("C", "mcnemar.csv reproduces", lambda: same_as_saved(ev.mcnemar(preds), T / "mcnemar.csv"))
    _run("C", "bootstrap_ci_cluster.csv reproduces (seeded, 2000 resamples)",
         lambda: same_as_saved(ev.bootstrap(preds, cluster=True), T / "bootstrap_ci_cluster.csv"))

    def length_matched():
        ids = pd.read_csv(T / "length_matched_ids.csv")["original_qr_id"]
        subset = test[test["original_qr_id"].isin(ids)].reset_index(drop=True)
        if len(subset) != len(ids):
            return False, f"{len(ids)} saved ids but {len(subset)} found in the test set"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            lm, _, _ = ev.score_models(subset)
        ok, detail = same_as_saved(lm, T / "length_matched_metrics.csv")
        return ok, f"n={len(subset)}; {detail}"
    _run("C", "length_matched_metrics.csv reproduces from the saved subset ids", length_matched)
    return out


# --------------------------------------------------------------------------- D. robustness tables
def _robustness() -> pd.DataFrame:
    return pd.read_csv(T / "robustness_detailed_results.csv", keep_default_na=False, na_values=[""])


def _exact_counts(res: pd.DataFrame) -> pd.Series:
    d = res[res["distortion_type"] != "none"]
    return (d["decode_outcome"] == "exact").groupby([d["distortion_type"], d["distortion_severity"].astype(str)]).sum()


def _cond_fpr(res: pd.DataFrame, model: str, dtype: str, severity: str | None = None) -> float:
    d = res[(res["distortion_type"] == dtype) & (res["decode_outcome"] == "exact") & (res["label"] == 0)]
    if severity is not None:
        d = d[d["distortion_severity"].astype(str) == severity]
    if d.empty:
        raise ValueError(f"no exactly decoded benign rows for {dtype}/{severity}")
    return float(d[f"{model}_pred"].mean())


def check_robustness() -> None:
    def shape():
        res = _robustness()
        conds = res.groupby(["distortion_type", "distortion_severity"]).size()
        return (len(res) == 5400 and len(conds) == 18 and (conds == 300).all(),
                f"{len(res)} rows, {len(conds)} conditions including clean")
    _run("D", "5,400 attempts: clean + 17 conditions x 300 images", shape)

    def outcomes():
        res = _robustness()
        counts = res["decode_outcome"].value_counts().to_dict()
        return counts.get("mismatch", 0) == 0 and set(counts) <= {"exact", "failure", "mismatch"}, counts
    _run("D", "zero decode mismatches (a wrong payload was never returned)", outcomes)

    def unclassified():
        res = _robustness()
        bad = res[(res["decode_outcome"] != "exact") & res[f"{E1}_score"].notna()]
        return bad.empty, f"{len(bad)} non-exact rows carry a score"
    _run("D", "images that did not decode exactly were never scored (D6)", unclassified)

    def e1_unchanged():
        res = _robustness()
        clean = pd.read_csv(T / "test_predictions.csv").set_index("original_qr_id")[f"{E1}_score"]
        ex = res[res["decode_outcome"] == "exact"]
        diff = float((ex[f"{E1}_score"] - ex["original_qr_id"].map(clean)).abs().max())
        return diff < 1e-9, f"max abs difference {diff:.2g} over {len(ex)} exactly decoded images"
    _run("D", "E1 score is identical to the clean score for every exactly decoded image", e1_unchanged)


# --------------------------------------------------------------------------- E. paper numbers
def _claim(name: str, getter, claimed: float, dp: int) -> None:
    def fn():
        actual = float(getter())
        return round(actual, dp) == round(claimed, dp), f"paper {claimed:.{dp}f}, saved {actual:.4f}"
    _run("E", name, fn)


def check_paper_numbers() -> None:
    def metric(table: str, model: str, col: str):
        m = pd.read_csv(T / table)
        return m[(m["model"] == model) & (m["threshold_view"] == "primary")][col].iloc[0]

    def ci(key: str, col: str):
        b = pd.read_csv(T / "bootstrap_ci_cluster.csv")
        return b[(b["model_or_difference"] == key) & (b["metric"] == "roc_auc")][col].iloc[0]

    for model, auc in ((E1, 0.8630), (M1, 0.6614), (E2, 0.6688), (E3, 0.8585)):
        _claim(f"test ROC-AUC {model}", lambda m=model: metric("test_metrics.csv", m, "roc_auc"), auc, 4)
    for model, auc in ((E2, 0.5273), (M1, 0.4971), (E1, 0.8502), (E3, 0.8421)):
        _claim(f"length-matched ROC-AUC {model}", lambda m=model: metric("length_matched_metrics.csv", m, "roc_auc"), auc, 4)

    for key, vals in ((f"{E3} - {E1}", (-0.0045, -0.0123, 0.0027)), (f"{E2} - {M1}", (0.0073, -0.0235, 0.0365))):
        for col, v in zip(("point", "ci95_low", "ci95_high"), vals):
            _claim(f"AUC difference {key}: {col}", lambda k=key, c=col: ci(k, c), v, 4)

    manifest = json.loads(ev.MANIFEST_JSON.read_text())
    _claim("E1 locked primary threshold", lambda: manifest["models"][E1]["thresholds"]["primary"], 0.490, 3)
    _claim("E1 recall at the locked threshold", lambda: metric("test_metrics.csv", E1, "recall_malicious"), 0.753, 3)
    _claim("E1 false-positive rate at the locked threshold", lambda: metric("test_metrics.csv", E1, "fpr"), 0.187, 3)

    def full_decode():
        counts = _exact_counts(_robustness())
        return int((counts == 300).sum()) == 16 and len(counts) == 17, f"{int((counts == 300).sum())} of {len(counts)} conditions at 300/300"
    _run("E", "exact decoding is 300/300 in 16 of 17 distortion conditions", full_decode)
    _claim("exact decodes under strong low resolution (of 300)",
           lambda: _exact_counts(_robustness())[("low_resolution", "strong")], 86, 0)

    for label, dtype, sev, v in (("clean", "none", None, 0.2333), ("blur mild", "gaussian_blur", "mild", 0.1933),
                                 ("blur moderate", "gaussian_blur", "moderate", 0.3000),
                                 ("blur strong", "gaussian_blur", "strong", 0.4800),
                                 ("JPEG q30", "jpeg", "q30", 0.3067),
                                 ("perspective mild", "perspective", "mild", 0.3733),
                                 ("perspective moderate", "perspective", "moderate", 0.4600)):
        _claim(f"E3 false-positive rate given exact decode: {label}",
               lambda d=dtype, s=sev: _cond_fpr(_robustness(), E3, d, s), v, 4)


# --------------------------------------------------------------------------- F. housekeeping
REQUIRED = [
    "requirements.txt", "requirements-lock.txt", "README.md", "docs/decisions_log.md", "docs/data_sources.md",
    "outputs/config_snapshot.py", "outputs/model_manifest.json", "outputs/evaluation_log.json",
    "outputs/validation_report.json", "data/processed/split_manifest.csv", "data/processed/test_ids_frozen.csv",
    "outputs/tables/split_freeze.json", "outputs/tables/split_audit.csv", "outputs/tables/test_metrics.csv",
    "outputs/tables/test_predictions.csv", "outputs/tables/bootstrap_ci_cluster.csv", "outputs/tables/mcnemar.csv",
    "outputs/tables/length_matched_metrics.csv", "outputs/tables/length_matched_ids.csv",
    "outputs/tables/robustness_detailed_results.csv", "outputs/tables/robustness_summary.csv",
    "outputs/tables/robustness_run_metadata.json", "outputs/tables/diag_padding_control.json",
    "outputs/tables/d25_qr_feature_correlations_train.csv", "outputs/tables/d25_e1_demo_bands_validation.csv",
    "outputs/tables/demo_examples.csv",
]
KEY_PACKAGES = ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "opencv-python", "Pillow", "qrcode",
                "tldextract", "shap", "matplotlib", "streamlit"]


def _installed(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def check_housekeeping(git_state: dict) -> None:
    root = config.PROJECT_ROOT

    def required():
        missing = [p for p in REQUIRED if not (root / p).exists()]
        return not missing, "missing: " + ", ".join(missing) if missing else f"all {len(REQUIRED)} present"
    _run("F", "required files exist", required)

    def decisions():
        text = (root / "docs" / "decisions_log.md").read_text()
        found = {int(n) for n in re.findall(r"^\|\s*D(\d+)\s*\|", text, flags=re.M)}
        missing = sorted(set(range(1, LAST_DECISION + 1)) - found)
        return ("PASS" if not missing else "WARN",
                f"D1-D{LAST_DECISION} all present" if not missing else "not found as table rows: " + ", ".join(f"D{n}" for n in missing))
    _run("F", f"decision log has D1 to D{LAST_DECISION}", decisions)

    def lock():
        path = root / "requirements-lock.txt"
        if not path.exists():
            return "WARN", "no lock file; create with: pip freeze > requirements-lock.txt"
        pinned = {}
        for line in path.read_text().splitlines():
            m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)", line.strip())
            if m:
                pinned[m.group(1).lower().replace("_", "-")] = m.group(2)
        diffs = []
        for pkg in KEY_PACKAGES:
            want, have = pinned.get(pkg.lower()), _installed(pkg)
            if want is None:
                diffs.append(f"{pkg} not pinned")
            elif want != have:
                diffs.append(f"{pkg} lock {want} vs installed {have}")
        return ("PASS" if not diffs else "WARN"), "; ".join(diffs) or "installed versions match the lock for all key packages"
    _run("F", "installed packages match requirements-lock.txt", lock)

    def run_versions():
        meta = json.loads((T / "robustness_run_metadata.json").read_text())["package_versions"]
        now = {"python": platform.python_version(), "numpy": _installed("numpy"), "pandas": _installed("pandas"),
               "scikit-learn": _installed("scikit-learn"), "matplotlib": _installed("matplotlib")}
        diffs = [f"{k} then {meta[k]} / now {v}" for k, v in now.items() if k in meta and meta[k] != v]
        return ("PASS" if not diffs else "WARN"), "; ".join(diffs) or "same versions as the robustness run"
    _run("F", "environment matches the one recorded at the robustness run", run_versions)

    def snapshot():
        same = (root / "outputs" / "config_snapshot.py").read_bytes() == (root / "src" / "config.py").read_bytes()
        return ("PASS" if same else "WARN"), ("snapshot equals src/config.py" if same else
                                              "snapshot differs from src/config.py: review `diff outputs/config_snapshot.py src/config.py`")
    _run("F", "config snapshot equals current config", snapshot)

    _record("F", "git working tree clean when the check started",
            "PASS" if git_state.get("clean") else "WARN",
            f"HEAD {git_state.get('head', 'unknown')}" + ("" if git_state.get("clean") else f"; {git_state.get('detail', '')}"))


def _git() -> dict:
    try:
        root = config.PROJECT_ROOT
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        return {"head": head, "clean": not dirty, "detail": f"{len(dirty.splitlines())} changed/untracked paths" if dirty else ""}
    except Exception as exc:  # noqa: BLE001
        return {"head": "unknown", "clean": False, "detail": f"git not available: {exc}"}


def _hash_outputs() -> dict:
    """SHA-256 of every file the paper's numbers and figures are built from."""
    root, files = config.PROJECT_ROOT, []
    files += sorted(T.glob("*.csv")) + sorted(T.glob("*.json")) + sorted(config.MODELS_DIR.glob("*.joblib"))
    files += [config.OUTPUTS_DIR / "model_manifest.json", config.PROCESSED_DIR / "split_manifest.csv",
              config.PROCESSED_DIR / "test_ids_frozen.csv", config.URLS_CSV, config.FEATURES_CSV, config.QR_MANIFEST_CSV]
    return {p.relative_to(root).as_posix(): _sha(p) for p in files if p.exists()}


# --------------------------------------------------------------------------- main
def main() -> int:
    git_state = _git()                      # before this script writes its own report
    for section, fn in (("A", check_frozen), ("B", check_split), ("C", check_reproduction), ("D", check_robustness),
                        ("E", check_paper_numbers), ("F", lambda: check_housekeeping(git_state))):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - e.g. a manifest file is missing; report it and continue
            _record(section, "section could not start", "ERROR", f"{type(exc).__name__}: {exc}")

    counts = {s: sum(r["status"] == s for r in _results) for s in ("PASS", "WARN", "FAIL", "ERROR")}
    overall = "FAIL" if counts["FAIL"] or counts["ERROR"] else ("PASS WITH WARNINGS" if counts["WARN"] else "PASS")
    report = {"generated": datetime.now().isoformat(timespec="seconds"), "overall": overall, "counts": counts,
              "git_head": git_state.get("head"), "git_clean_at_start": git_state.get("clean"),
              "python": platform.python_version(), "platform": platform.platform(),
              "packages": {p: _installed(p) for p in KEY_PACKAGES},
              "checks": _results, "sha256": _hash_outputs()}
    REPORT.write_text(json.dumps(report, indent=2))

    print("FINAL VALIDATION REPORT\n" + "-" * 78)
    section = None
    for r in _results:
        if r["section"] != section:
            section = r["section"]
            print(f"\n[{section}]")
        print(f"  {r['status']:5s} {r['check']}" + (f"\n          {r['detail']}" if r["detail"] else ""))
    print("\n" + "-" * 78)
    print(f"OVERALL: {overall}   " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"Saved {REPORT.relative_to(config.PROJECT_ROOT)} ({len(report['sha256'])} file hashes)")
    return 0 if overall != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())

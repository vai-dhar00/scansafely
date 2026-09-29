"""
Day 3: robustness of the LOCKED pipeline on the FROZEN test images (D19-D23).

For every test image x every core distortion condition (plus the clean reference):
  distort -> same fixed decoder chain -> exact / mismatch / failure
  -> if EXACT: score E1 (primary, D20) and E3 (secondary) at their LOCKED thresholds.
No retraining, re-thresholding or feature changes.

Three reporting layers per condition (D21), overall and by class, with Wilson 95% CIs (D23):
  1. decoder: exact / mismatch / failure rates
  2. conditional: E1/E3 metrics among exact decodes (only if >= 30 per class, else flagged)
  3. end-to-end: (exact AND correct) / attempted; end-to-end malicious recall
Plus decode rates by label x QR-version band (D22).

Key built-in check: when a code decodes EXACTLY, E1's input is the identical URL string, so
its score must equal its clean-test score. The script asserts this, so any change in E1's
conditional metrics can only come from WHICH images survived decoding (selection), never
from the classifier behaving differently.

Run:  python -m src.robustness_experiment --preview   (inspect the grid first)
      python -m src.robustness_experiment             (full run, ~5 min)
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.distortions import CORE_DISTORTIONS, apply, conditions
from src.evaluate_models import binary_metrics, verify_integrity
from src.feature_pipeline import process_image
from src.qr_decoder import load_image

PRIMARY, SECONDARY = "E1_url_only", "E3_fusion"
MIN_PER_CLASS = 30
BLUE, ORANGE = "#2a78d6", "#eb6834"


def version_band(v: int) -> str:
    return "v1-v3" if v <= 3 else "v4-v6" if v <= 6 else "v7+"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_test_meta() -> pd.DataFrame:
    urls = pd.read_csv(config.URLS_CSV, keep_default_na=False)
    man = pd.read_csv(config.QR_MANIFEST_CSV)[["original_qr_id", "image_path", "qr_version"]]
    test = urls[urls["split"] == "test"].merge(man, on="original_qr_id")
    frozen = pd.read_csv(config.PROCESSED_DIR / "test_ids_frozen.csv")
    assert set(test["original_qr_id"]) == set(frozen["original_qr_id"])
    test["version_band"] = test["qr_version"].map(version_band)
    return test.reset_index(drop=True)


# --------------------------------------------------------------------------- preview
def preview_grid(test: pd.DataFrame) -> None:
    """3 test codes (one per version band) x every condition, for a visual check before the run."""
    picks = [test[test["version_band"] == b].sort_values("qr_version").iloc[len(test[test["version_band"] == b]) // 2]
             for b in ("v1-v3", "v4-v6", "v7+") if (test["version_band"] == b).any()]
    conds = [("none", "clean", 0)] + conditions()
    fig, axes = plt.subplots(len(picks), len(conds), figsize=(1.6 * len(conds), 1.9 * len(picks)), squeeze=False)
    for r, rec in enumerate(picks):
        img = load_image(config.PROJECT_ROOT / rec["image_path"])
        for c, (t, sev, val) in enumerate(conds):
            out = img if t == "none" else apply(img, t, val)
            ax = axes[r, c]
            ax.imshow(out, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{t.replace('_', ' ')}\n{sev}", fontsize=7)
            if c == 0:
                ax.set_ylabel(f"v{rec['qr_version']}", fontsize=9)
    fig.suptitle("Distortion preview (3 frozen-test codes) - check before the full run. Do NOT scan with a phone.",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "distortion_preview.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- run
def run(save_images: bool = True) -> pd.DataFrame:
    verify_integrity()
    test = load_test_meta()
    bundles = {n: joblib.load(config.MODELS_DIR / f"{n}.joblib") for n in (PRIMARY, SECONDARY)}
    clean_scores = pd.read_csv(config.TABLES_DIR / "test_predictions.csv").set_index("original_qr_id")
    config.DISTORTED_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    conds = [("none", "clean", 0)] + conditions()
    for rec in test.itertuples(index=False):
        img = load_image(config.PROJECT_ROOT / rec.image_path)
        for t, sev, val in conds:
            out = img if t == "none" else apply(img, t, val)
            t0 = time.perf_counter()
            feats = process_image(out, original_url=rec.url)
            qr_id = f"{rec.original_qr_id}__{t}__{sev}"
            exact = feats["decode_outcome"] == "exact"
            row = {"qr_id": qr_id, "original_qr_id": rec.original_qr_id, "split": "test", "label": rec.label,
                   "url_length": len(rec.url), "qr_version_generator": rec.qr_version, "version_band": rec.version_band,
                   "image_path_clean": rec.image_path,
                   "image_path_distorted": "" if t == "none" else
                       (config.DISTORTED_DIR / f"{qr_id}.png").relative_to(config.PROJECT_ROOT).as_posix(),
                   "distortion_type": t, "distortion_severity": sev, "distortion_value": val,
                   "distortion_seed": "n/a (deterministic)",
                   "decode_outcome": feats["decode_outcome"], "decoder_used": feats["decoder_name"],
                   "decoded_payload_exact_match": exact, "classification_possible": exact,
                   "notes": "" if exact else f"decode {feats['decode_outcome']}: not classified (undecodable input, "
                                             "never treated as benign/suspicious/high risk)"}
            if exact:
                X = pd.DataFrame([feats])
                for name, b in bundles.items():
                    s = float(b["pipeline"].predict_proba(X[b["features"]])[:, 1][0])
                    pred = int(s >= b["thresholds"]["primary"]["threshold"])
                    row.update({f"{name}_score": s, f"{name}_pred": pred, f"{name}_correct": int(pred == rec.label)})
                # Determinism check: identical URL -> identical E1 score as on the clean test run
                assert abs(row[f"{PRIMARY}_score"] - clean_scores.loc[rec.original_qr_id, f"{PRIMARY}_score"]) < 1e-9
            row["inference_time_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            rows.append(row)
            if save_images and t != "none":
                import cv2
                cv2.imwrite(str(config.DISTORTED_DIR / f"{row['qr_id']}.png"), out)

    res = pd.DataFrame(rows)
    res.to_csv(config.TABLES_DIR / "robustness_detailed_results.csv", index=False)
    return res


# --------------------------------------------------------------------------- summaries
def _rate(k, n):
    lo, hi = wilson(k, n)
    return {"k": int(k), "n": int(n), "rate": round(k / n, 4) if n else np.nan, "ci_low": round(lo, 4), "ci_high": round(hi, 4)}


def summarise(res: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = []
    for (t, sev), g in res.groupby(["distortion_type", "distortion_severity"], sort=False):
        exact = g["decode_outcome"] == "exact"
        row = {"distortion_type": t, "distortion_severity": sev, "attempted": len(g)}
        for oc in ("exact", "mismatch", "failure"):
            r = _rate(int((g["decode_outcome"] == oc).sum()), len(g))
            row.update({f"{oc}_rate": r["rate"], f"{oc}_ci_low": r["ci_low"], f"{oc}_ci_high": r["ci_high"]})
        for lab, lname in ((0, "benign"), (1, "malicious")):
            gl = g[g["label"] == lab]
            r = _rate(int((gl["decode_outcome"] == "exact").sum()), len(gl))
            row.update({f"exact_{lname}": f"{r['k']}/{r['n']}", f"exact_rate_{lname}": r["rate"],
                        f"exact_{lname}_ci_low": r["ci_low"], f"exact_{lname}_ci_high": r["ci_high"]})
        for name in (PRIMARY, SECONDARY):
            ge = g[exact]
            enough = min((ge["label"] == 0).sum(), (ge["label"] == 1).sum()) >= MIN_PER_CLASS
            if len(ge) and ge["label"].nunique() == 2:
                m = binary_metrics(ge["label"].to_numpy(), ge[f"{name}_pred"].astype(int).to_numpy())
                row.update({f"{name}_cond_recall": round(m["recall_malicious"], 4), f"{name}_cond_fpr": round(m["fpr"], 4),
                            f"{name}_cond_f1": round(m["f1"], 4)})
            row[f"{name}_cond_status"] = "ok" if enough else f"exploratory (<{MIN_PER_CLASS}/class)"
            mal = g[g["label"] == 1]
            k = int(((mal["decode_outcome"] == "exact") & (mal[f"{name}_pred"] == 1)).sum())
            r = _rate(k, len(mal))
            row.update({f"{name}_e2e_mal_recall": r["rate"], f"{name}_e2e_mal_recall_ci_low": r["ci_low"],
                        f"{name}_e2e_mal_recall_ci_high": r["ci_high"]})
            r = _rate(int(((g["decode_outcome"] == "exact") & (g[f"{name}_correct"] == 1)).sum()), len(g))
            row.update({f"{name}_e2e_correct": r["rate"]})
        out.append(row)
    summary = pd.DataFrame(out)

    ver = []
    for (t, sev, band, lab), g in res.groupby(["distortion_type", "distortion_severity", "version_band", "label"], sort=False):
        r = _rate(int((g["decode_outcome"] == "exact").sum()), len(g))
        ver.append({"distortion_type": t, "distortion_severity": sev, "version_band": band,
                    "class": "malicious" if lab else "benign", "exact": f"{r['k']}/{r['n']}",
                    "exact_rate": r["rate"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                    "sparse_cell": r["n"] < 10})
    return summary, pd.DataFrame(ver)


def write_tables(summary: pd.DataFrame, by_version: pd.DataFrame) -> None:
    """The five reporting tables required by the Day 3 protocol (all derived from `summary`)."""
    key = ["distortion_type", "distortion_severity", "attempted"]
    T = config.TABLES_DIR
    summary[key + [c for c in summary.columns if c.split("_")[0] in ("exact", "mismatch", "failure")
                   and "benign" not in c and "malicious" not in c]].to_csv(T / "robustness_decode_summary.csv", index=False)
    summary[key + [c for c in summary.columns if "_cond_" in c]].to_csv(
        T / "robustness_classification_exact_decode.csv", index=False)
    summary[key + [c for c in summary.columns if "_e2e_" in c]].to_csv(T / "robustness_end_to_end_summary.csv", index=False)
    summary[key + [c for c in summary.columns if "benign" in c or "malicious" in c and "e2e" not in c]].to_csv(
        T / "robustness_by_label.csv", index=False)
    by_version.to_csv(T / "robustness_by_qr_version_band.csv", index=False)
    summary.to_csv(T / "robustness_summary.csv", index=False)  # combined table used by the figures


def write_metadata(n_attempts: int, minutes: float) -> None:
    import platform
    from datetime import datetime

    import cv2 as _cv2
    import sklearn
    manifest = json.loads((config.OUTPUTS_DIR / "model_manifest.json").read_text())
    freeze = json.loads((config.TABLES_DIR / "split_freeze.json").read_text())
    meta = {
        "run": datetime.now().isoformat(timespec="seconds"), "random_seed": config.RANDOM_SEED,
        "frozen_test_ids_sha256": freeze["test_ids_sha256"],
        "model_sha256": {n: manifest["models"][n]["model_sha256"] for n in (PRIMARY, SECONDARY)},
        "locked_thresholds": {n: manifest["models"][n]["thresholds"]["primary"] for n in (PRIMARY, SECONDARY)},
        "decoder_policy": "OpenCV QRCodeDetector, then QRCodeDetectorAruco only if the first returns nothing; "
                          "no preprocessing; outcome exact/mismatch/failure; only exact is classified (D6, D21)",
        "distortion_settings": {k: config.DISTORTIONS[k] for k in CORE_DISTORTIONS},
        "distortion_implementation": {
            "rotation": "about centre, white fill, canvas expanded to fit (no cropping)",
            "perspective": "keystone: top-left corner -> (frac*w, 0), top-right -> (w - frac*w, 0), bottom corners fixed",
            "low_resolution": "INTER_AREA down, INTER_NEAREST up; controlled digital degradation, not a camera model",
            "jpeg": "encode + decode in memory with OpenCV at the given quality",
            "occlusion": "excluded from core study (D19)"},
        "total_images_attempted": n_attempts, "runtime_minutes": minutes,
        "e1_determinism_check": "passed",
        "config_snapshot": "outputs/config_snapshot.py",
        "package_versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                             "scikit-learn": sklearn.__version__, "opencv": _cv2.__version__,
                             "matplotlib": matplotlib.__version__},
    }
    import shutil
    shutil.copy(config.PROJECT_ROOT / "src" / "config.py", config.OUTPUTS_DIR / "config_snapshot.py")
    (config.TABLES_DIR / "robustness_run_metadata.json").write_text(json.dumps(meta, indent=2))


def _x_axis(sub_rows: pd.DataFrame, t: str) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Shared x-axis: clean first (or at 0 degrees for rotation), then severities in config order."""
    if t == "rotation":
        vals = [0.0 if s == "clean" else float(s) for s in sub_rows["distortion_severity"]]
        sub_rows = sub_rows.assign(_x=vals).sort_values("_x")
        xs = sub_rows["_x"].to_numpy()
        return xs, [f"{int(x):+d}°" if x else "0°" for x in xs], sub_rows
    return np.arange(len(sub_rows)), list(sub_rows["distortion_severity"]), sub_rows


def plot_by_version(by_version: pd.DataFrame) -> None:
    """Exact decode rate by QR-version band (both classes pooled per band; per-class cells are in the CSV)."""
    shades = {"v1-v3": "#86b6ef", "v4-v6": "#2a78d6", "v7+": "#104281"}
    pooled = (by_version.assign(k=by_version["exact"].str.split("/").str[0].astype(int),
                                n=by_version["exact"].str.split("/").str[1].astype(int))
              .groupby(["distortion_type", "distortion_severity", "version_band"], sort=False)[["k", "n"]].sum()
              .reset_index())
    fig, axes = plt.subplots(1, len(CORE_DISTORTIONS), figsize=(4.0 * len(CORE_DISTORTIONS), 4.4), sharey=True)
    clean = pooled[pooled["distortion_type"] == "none"]
    for ax, t in zip(axes, CORE_DISTORTIONS):
        for band, col in shades.items():
            rows = pd.concat([clean[clean["version_band"] == band],
                              pooled[(pooled["distortion_type"] == t) & (pooled["version_band"] == band)]])
            if rows.empty:
                continue
            xs, labels, rows = _x_axis(rows, t)
            rate = rows["k"] / rows["n"]
            ax.plot(xs, rate, color=col, linewidth=2, marker="o", markersize=5,
                    label=f"{band} (n={int(rows['n'].iloc[0])})")
            ax.set_xticks(xs, labels, fontsize=9)
        ax.set_title(t.replace("_", " "), fontsize=12, loc="left")
        ax.set_ylim(0, 1.03); ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Exact decode rate", fontsize=11)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02),
               title="QR version band (denser codes = darker)")
    fig.suptitle("Exact decoding by QR version band (frozen test set)", fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(config.FIGURES_DIR / "robustness_by_version.png", dpi=200)
    plt.close(fig)


def plot_e1_vs_e3(summary: pd.DataFrame) -> None:
    """Conditional recall/FPR among EXACT decodes; points with < 30/class are hollow (exploratory)."""
    fig, axes = plt.subplots(2, len(CORE_DISTORTIONS), figsize=(4.0 * len(CORE_DISTORTIONS), 6.8), sharey="row")
    clean = summary[summary["distortion_type"] == "none"]
    for c, t in enumerate(CORE_DISTORTIONS):
        rows = pd.concat([clean, summary[summary["distortion_type"] == t]])
        xs, labels, rows = _x_axis(rows, t)
        for r, (metric, ylabel) in enumerate((("cond_recall", "Recall | exact decode"), ("cond_fpr", "FPR | exact decode"))):
            ax = axes[r, c]
            for name, col, lab in ((PRIMARY, BLUE, "E1 URL-only (primary)"), (SECONDARY, ORANGE, "E3 fusion (secondary)")):
                y = rows[f"{name}_{metric}"].astype(float).to_numpy()
                ok = (rows[f"{name}_cond_status"] == "ok").to_numpy()
                ax.plot(xs, y, color=col, linewidth=2, label=lab)
                ax.plot(xs[ok], y[ok], "o", color=col, markersize=6)
                ax.plot(xs[~ok], y[~ok], "o", color=col, markerfacecolor="white", markersize=6)
            ax.set_xticks(xs, labels, fontsize=9)
            ax.set_ylim(0, 1.03); ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
            if r == 0:
                ax.set_title(t.replace("_", " "), fontsize=12, loc="left")
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=11)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Classification among exactly decoded codes (hollow = < 30 per class, exploratory). "
                 "E1 changes = which codes decoded; E3 changes can also be image-feature shift.",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(config.FIGURES_DIR / "robustness_e1_vs_e3.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- figure
def plot_robustness(summary: pd.DataFrame) -> None:
    clean = summary[summary["distortion_type"] == "none"].iloc[0]
    fig, axes = plt.subplots(1, len(CORE_DISTORTIONS), figsize=(4.0 * len(CORE_DISTORTIONS), 4.4), sharey=True)
    for ax, t in zip(axes, CORE_DISTORTIONS):
        sub = pd.concat([clean.to_frame().T, summary[summary["distortion_type"] == t]])
        if t == "rotation":  # clean sits at 0 degrees, in the middle
            sub = sub.assign(_x=[0] + [float(s) for s in summary.loc[summary["distortion_type"] == t, "distortion_severity"]])
            sub = sub.sort_values("_x"); xs = sub["_x"].to_numpy(); labels = [f"{int(x):+d}°" if x else "0°" for x in xs]
        else:
            xs = np.arange(len(sub)); labels = ["clean"] + list(sub["distortion_severity"].iloc[1:])
        # malicious drawn first; benign on top with hollow markers so equal values stay visible
        for lname, col, mk, face, lw in (("malicious", ORANGE, "^", ORANGE, 2.5), ("benign", BLUE, "o", "white", 1.5)):
            y = sub[f"exact_rate_{lname}"].astype(float).to_numpy()
            lo, hi = sub[f"exact_{lname}_ci_low"].astype(float), sub[f"exact_{lname}_ci_high"].astype(float)
            ax.fill_between(xs, lo, hi, color=col, alpha=0.12, linewidth=0)
            ax.plot(xs, y, color=col, linewidth=lw, marker=mk, markersize=7, markerfacecolor=face,
                    markeredgewidth=1.5, label=f"exact decode · {lname}")
        e2e = sub[f"{PRIMARY}_e2e_mal_recall"].astype(float).to_numpy()
        ax.plot(xs, e2e, color=ORANGE, linewidth=2, linestyle="--", marker="s", markersize=4,
                label="end-to-end malicious recall (E1)")
        ax.set_xticks(xs, labels, fontsize=9, rotation=0)
        ax.set_title(t.replace("_", " "), fontsize=12, loc="left")
        ax.set_ylim(0, 1.03); ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Rate (95% Wilson CI)", fontsize=11)
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Robustness on the frozen test set (n = 300 per condition): decoding vs end-to-end detection",
                 fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(config.FIGURES_DIR / "robustness.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=r".*encountered in matmul", category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="only write the distortion preview grid")
    args = ap.parse_args()
    config.ensure_directories()
    test_meta = load_test_meta()
    if args.preview:
        preview_grid(test_meta)
        print("Saved outputs/figures/distortion_preview.png - inspect it, then run without --preview.")
    else:
        t0 = time.time()
        results = run()
        summary, by_version = summarise(results)
        write_tables(summary, by_version)
        write_metadata(len(results), round((time.time() - t0) / 60, 1))
        plot_robustness(summary)
        plot_by_version(by_version)
        plot_e1_vs_e3(summary)
        cols = ["distortion_type", "distortion_severity", "exact_rate", "mismatch_rate", "failure_rate",
                "exact_benign", "exact_malicious", f"{PRIMARY}_cond_recall", f"{PRIMARY}_cond_fpr",
                f"{PRIMARY}_e2e_mal_recall", f"{SECONDARY}_cond_recall", f"{SECONDARY}_cond_fpr",
                f"{PRIMARY}_cond_status"]
        pd.set_option("display.width", 250)
        print("E1 determinism check PASSED (exact decodes reproduce the clean E1 scores).\n")
        print(summary[cols].to_string(index=False))
        print("\nSaved tables: robustness_detailed_results, _decode_summary, _classification_exact_decode,")
        print("              _end_to_end_summary, _by_label, _by_qr_version_band (.csv), robustness_run_metadata.json")
        print("Saved figures: robustness.png, robustness_by_version.png, robustness_e1_vs_e3.png")

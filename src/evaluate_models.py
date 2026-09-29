"""
Phase 7b: evaluate the LOCKED models on the FROZEN test set (D14).

Before touching the test set this script VERIFIES:
  - the test ids still match the SHA-256 recorded at the freeze (Phase 6)
  - every model file still matches the SHA-256 recorded when it was locked (Phase 7a)
Nothing is retrained, re-tuned or re-thresholded here.

Outputs (outputs/tables, outputs/figures):
  test_metrics.csv                 every model x threshold view (+ constant references)
  test_predictions.csv             per-row scores/predictions (no URL text)
  bootstrap_ci_cluster.csv         primary uncertainty: cluster bootstrap over split groups
  bootstrap_ci_url_level.csv       sensitivity: URL-level bootstrap
  mcnemar.csv                      exact McNemar, Holm-adjusted
  length_matched_*.csv/json        length-matched sensitivity subset (re-score only)
  error_analysis_E3.csv            10 highest-confidence FPs + 10 FNs of E3 (URLs as TEXT)
  model_comparison.png, confusion_matrix_E3.png, length_matched_auc.png

Run:  python -m src.evaluate_models
"""
from __future__ import annotations

import hashlib
import json
import warnings
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score

from src import config
from src.train_models import MANIFEST_JSON

MODELS = ["M1_length_only", "M1_plus_QR", "E1_url_only", "E2_qr_only", "E3_fusion", "E3_reducedQR"]
PAIRS = [("E3_fusion", "E1_url_only"), ("M1_plus_QR", "M1_length_only"), ("E2_qr_only", "M1_length_only")]
EXTRA_DIFFS = [("E3_reducedQR", "E1_url_only")]  # supplementary, CI only (not in the Holm family)
BLUE, ORANGE = "#2a78d6", "#eb6834"


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- integrity
def verify_integrity() -> dict:
    freeze = json.loads((config.TABLES_DIR / "split_freeze.json").read_text())
    test_ids = config.PROCESSED_DIR / "test_ids_frozen.csv"
    if _sha(test_ids) != freeze["test_ids_sha256"]:
        raise SystemExit("ABORT: test ids do not match the frozen SHA-256. The test set changed.")
    manifest = json.loads(MANIFEST_JSON.read_text())
    for name in MODELS:
        if _sha(config.MODELS_DIR / f"{name}.joblib") != manifest["models"][name]["model_sha256"]:
            raise SystemExit(f"ABORT: {name}.joblib differs from the locked model (Phase 7a).")
    return {"test_ids_sha256": freeze["test_ids_sha256"], "models_verified": MODELS}


def load_test() -> pd.DataFrame:
    feats = pd.read_csv(config.FEATURES_CSV, keep_default_na=False, na_values=[""])
    feats = feats[(feats["distortion_type"] == "none") & (feats["split"] == "test")]
    frozen = pd.read_csv(config.PROCESSED_DIR / "test_ids_frozen.csv")
    assert set(feats["original_qr_id"]) == set(frozen["original_qr_id"]), "test rows != frozen ids"
    return feats.merge(frozen[["original_qr_id", "split_group_id"]], on="original_qr_id").reset_index(drop=True)


# --------------------------------------------------------------------------- metrics
def binary_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"accuracy": (tp + tn) / len(y), "precision": prec, "recall_malicious": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else 0.0, "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def score_models(test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    y = test["label"].to_numpy()
    preds = test[["original_qr_id", "label", "split_group_id", "url_length"]].copy()
    rows, bundles = [], {}
    for name in MODELS:
        b = joblib.load(config.MODELS_DIR / f"{name}.joblib")
        bundles[name] = b
        p = b["pipeline"].predict_proba(test[b["features"]])[:, 1]
        assert np.isfinite(p).all()
        preds[f"{name}_score"] = p
        auc, brier = roc_auc_score(y, p), brier_score_loss(y, p)
        for view, t in b["thresholds"].items():
            m = binary_metrics(y, (p >= t["threshold"]).astype(int))
            rows.append({"model": name, "threshold_view": view, "threshold": round(t["threshold"], 6),
                         **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
                         "roc_auc": round(auc, 4), "brier": round(brier, 4)})
        preds[f"{name}_pred"] = (p >= b["thresholds"]["primary"]["threshold"]).astype(int)
    for name, const in (("REF_constant_negative", 0), ("REF_constant_positive", 1)):
        m = binary_metrics(y, np.full_like(y, const))
        rows.append({"model": name, "threshold_view": "n/a", "threshold": np.nan,
                     **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
                     "roc_auc": np.nan, "brier": np.nan})
    return pd.DataFrame(rows), preds, bundles


# --------------------------------------------------------------------------- bootstrap
def _boot_stats(y, scores, preds) -> dict:
    out = {}
    both = len(np.unique(y)) == 2
    for name in MODELS:
        m = binary_metrics(y, preds[name])
        out[(name, "f1")], out[(name, "recall_malicious")], out[(name, "precision")] = m["f1"], m["recall_malicious"], m["precision"]
        out[(name, "roc_auc")] = roc_auc_score(y, scores[name]) if both else np.nan
    for a, b in PAIRS + EXTRA_DIFFS:
        for k in ("f1", "recall_malicious", "roc_auc"):
            out[(f"{a} - {b}", k)] = out[(a, k)] - out[(b, k)]
    return out


def bootstrap(preds: pd.DataFrame, cluster: bool) -> pd.DataFrame:
    rng = np.random.default_rng(config.RANDOM_SEED)
    y = preds["label"].to_numpy()
    scores = {n: preds[f"{n}_score"].to_numpy() for n in MODELS}
    hard = {n: preds[f"{n}_pred"].to_numpy() for n in MODELS}
    units = (list(preds.groupby("split_group_id").indices.values()) if cluster
             else [np.array([i]) for i in range(len(preds))])
    draws = []
    for _ in range(config.BOOTSTRAP_RESAMPLES):
        idx = np.concatenate([units[i] for i in rng.integers(0, len(units), len(units))])
        draws.append(_boot_stats(y[idx], {n: s[idx] for n, s in scores.items()}, {n: h[idx] for n, h in hard.items()}))
    point = _boot_stats(y, scores, hard)
    rows = []
    for key in point:
        vals = np.array([d[key] for d in draws], dtype=float)
        vals = vals[~np.isnan(vals)]
        rows.append({"model_or_difference": key[0], "metric": key[1], "point": round(point[key], 4),
                     "ci95_low": round(np.percentile(vals, 2.5), 4), "ci95_high": round(np.percentile(vals, 97.5), 4),
                     "valid_draws": len(vals)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- McNemar
def mcnemar(preds: pd.DataFrame) -> pd.DataFrame:
    y = preds["label"].to_numpy()
    rows = []
    for a, b in PAIRS:
        ca, cb = preds[f"{a}_pred"].to_numpy() == y, preds[f"{b}_pred"].to_numpy() == y
        n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())  # A right/B wrong, A wrong/B right
        p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        rows.append({"comparison": f"{a} vs {b}", "a_right_b_wrong": n01, "a_wrong_b_right": n10, "p_exact": p})
    df = pd.DataFrame(rows).sort_values("p_exact").reset_index(drop=True)
    m = len(df)  # Holm step-down
    adj = np.maximum.accumulate([min(1.0, (m - i) * p) for i, p in enumerate(df["p_exact"])])
    df["p_holm"] = np.round(adj, 4)
    df["p_exact"] = df["p_exact"].round(4)
    return df


# --------------------------------------------------------------------------- length matching
def length_matched_subset(test: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """D12: pooled length quantile bins; keep min(class counts) per class per bin; seed 42.
    Uses ONLY url_length and label."""
    chosen = None
    for n_bins in config.LENGTH_MATCH_BIN_OPTIONS:
        bins = pd.qcut(test["url_length"], q=n_bins, duplicates="drop")
        counts = pd.crosstab(bins, test["label"])
        if (counts.min(axis=1) >= config.LENGTH_MATCH_MIN_PER_CLASS).all():
            chosen = (n_bins, bins, counts, "all bins meet minimum")
            break
    if chosen is None:
        n_bins = config.LENGTH_MATCH_BIN_OPTIONS[-1]
        bins = pd.qcut(test["url_length"], q=n_bins, duplicates="drop")
        chosen = (n_bins, bins, pd.crosstab(bins, test["label"]), "sparse bins dropped (below minimum)")
    n_bins, bins, counts, note = chosen
    keep_idx = []
    for interval, row in counts.iterrows():
        k = int(row.min())
        if k < config.LENGTH_MATCH_MIN_PER_CLASS:
            continue
        for label in (0, 1):
            pool = test[(bins == interval) & (test["label"] == label)]
            keep_idx.extend(pool.sample(n=k, random_state=config.RANDOM_SEED).index)
    subset = test.loc[sorted(keep_idx)]
    n = len(subset)
    tier = ("strong" if n >= 200 else "useful (show CIs)" if n >= 150 else
            "exploratory" if n >= 100 else "descriptive only")
    meta = {"n_bins": int(n_bins), "note": note, "subset_size": n,
            "per_class": subset["label"].value_counts().to_dict(), "interpretation_tier": tier,
            "bin_counts": {str(k): v for k, v in counts.to_dict(orient="index").items()}}
    return subset, meta


# --------------------------------------------------------------------------- figures
def plot_model_comparison(ci: pd.DataFrame) -> None:
    order = ["M1_length_only", "E2_qr_only", "M1_plus_QR", "E1_url_only", "E3_fusion", "E3_reducedQR"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, metric, title in zip(axes, ("f1", "recall_malicious"), ("F1", "Malicious-class recall")):
        sub = ci[(ci["metric"] == metric) & ci["model_or_difference"].isin(order)].set_index("model_or_difference").loc[order]
        y = np.arange(len(order))
        ax.errorbar(sub["point"], y, xerr=[sub["point"] - sub["ci95_low"], sub["ci95_high"] - sub["point"]],
                    fmt="o", color=BLUE, ecolor=BLUE, elinewidth=2, capsize=4, markersize=8)
        for yi, v, hi in zip(y, sub["point"], sub["ci95_high"]):
            ax.text(min(hi + 0.02, 1.0), yi, f"{v:.2f}", va="center", fontsize=10)
        ax.set_yticks(y, order, fontsize=11)
        ax.set_xlim(0, 1.1)
        ax.set_title(title, fontsize=13, loc="left")
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle("Frozen test set · primary threshold · 95% cluster-bootstrap CI", fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "model_comparison.png", dpi=200)
    plt.close(fig)


def plot_confusion(test_metrics: pd.DataFrame, name: str = "E3_fusion") -> None:
    r = test_metrics[(test_metrics["model"] == name) & (test_metrics["threshold_view"] == "primary")].iloc[0]
    cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
    fig, ax = plt.subplots(figsize=(4.8, 4.3))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.3)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=18,
                    color="white" if cm[i, j] > cm.max() * 0.65 else "#1a1a19")
    ax.set_xticks([0, 1], ["benign", "malicious"], fontsize=11)
    ax.set_yticks([0, 1], ["benign", "malicious"], fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12); ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"{name} · test · threshold {r['threshold']:.3f}", fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"confusion_matrix_{name.split('_')[0]}.png", dpi=200)
    plt.close(fig)


def plot_length_matched(full: pd.DataFrame, matched: pd.DataFrame) -> None:
    order = ["M1_length_only", "E2_qr_only", "M1_plus_QR", "E1_url_only", "E3_fusion", "E3_reducedQR"]
    f = full[full["threshold_view"] == "primary"].set_index("model").loc[order, "roc_auc"]
    m = matched[matched["threshold_view"] == "primary"].set_index("model").loc[order, "roc_auc"]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.5, 4.9))
    ax.axvline(0.5, color="#999999", linestyle="--", linewidth=1, label="chance (AUC 0.5)")
    ax.hlines(y, m, f, color="#c3c2b7", linewidth=2)
    ax.plot(f, y, "o", color=BLUE, markersize=9, label="full test set")
    ax.plot(m, y, "o", color=ORANGE, markersize=9, label="length-matched subset")
    ax.set_yticks(y, order, fontsize=11); ax.invert_yaxis()
    ax.set_xlim(0.35, 1.0)
    ax.set_xlabel("ROC-AUC (test)", fontsize=12)
    ax.set_title("Does performance survive when URL length is controlled?", fontsize=13, loc="left")
    ax.legend(frameon=False, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    ax.grid(axis="x", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "length_matched_auc.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- error analysis
def error_analysis(test: pd.DataFrame, preds: pd.DataFrame, name: str = "E3_fusion", k: int = 10) -> pd.DataFrame:
    urls = pd.read_csv(config.URLS_CSV, keep_default_na=False)[["original_qr_id", "url"]]
    d = preds.merge(urls, on="original_qr_id").merge(
        test[["original_qr_id", "url_suspicious_token_count", "url_has_https", "url_subdomain_count", "url_path_depth"]],
        on="original_qr_id")
    fp = d[(d["label"] == 0) & (d[f"{name}_pred"] == 1)].nlargest(k, f"{name}_score").assign(error="false_positive")
    fn = d[(d["label"] == 1) & (d[f"{name}_pred"] == 0)].nsmallest(k, f"{name}_score").assign(error="false_negative")
    out = pd.concat([fp, fn])[["error", "original_qr_id", f"{name}_score", "url_length", "url_suspicious_token_count",
                               "url_has_https", "url_subdomain_count", "url_path_depth", "url"]]
    out = out.rename(columns={"url": "url_TEXT_ONLY_do_not_open"})
    out["your_category"] = ""
    return out


# --------------------------------------------------------------------------- main
def evaluate() -> dict:
    config.ensure_directories()
    integrity = verify_integrity()
    test = load_test()
    metrics, preds, bundles = score_models(test)
    metrics.to_csv(config.TABLES_DIR / "test_metrics.csv", index=False)
    preds.to_csv(config.TABLES_DIR / "test_predictions.csv", index=False)

    ci_cluster = bootstrap(preds, cluster=True)
    ci_cluster.to_csv(config.TABLES_DIR / "bootstrap_ci_cluster.csv", index=False)
    bootstrap(preds, cluster=False).to_csv(config.TABLES_DIR / "bootstrap_ci_url_level.csv", index=False)
    mc = mcnemar(preds)
    mc.to_csv(config.TABLES_DIR / "mcnemar.csv", index=False)

    subset, meta = length_matched_subset(test)
    subset[["original_qr_id", "label", "url_length"]].to_csv(config.TABLES_DIR / "length_matched_ids.csv", index=False)
    lm_metrics, _, _ = score_models(subset.reset_index(drop=True))
    lm_metrics.to_csv(config.TABLES_DIR / "length_matched_metrics.csv", index=False)
    (config.TABLES_DIR / "length_matched_meta.json").write_text(json.dumps(meta, indent=2))

    error_analysis(test, preds).to_csv(config.TABLES_DIR / "error_analysis_E3.csv", index=False)
    plot_model_comparison(ci_cluster)
    plot_confusion(metrics)
    plot_length_matched(metrics, lm_metrics)

    log = {"run": datetime.now().isoformat(timespec="seconds"), **integrity, "n_test": len(test),
           "bootstrap_resamples": config.BOOTSTRAP_RESAMPLES, "length_matched": meta}
    (config.OUTPUTS_DIR / "evaluation_log.json").write_text(json.dumps(log, indent=2))
    return {"metrics": metrics, "ci": ci_cluster, "mcnemar": mc, "lm": lm_metrics, "meta": meta}


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=r".*encountered in matmul", category=RuntimeWarning)
    r = evaluate()
    pd.set_option("display.width", 200)
    prim = r["metrics"][r["metrics"]["threshold_view"].isin(["primary", "n/a"])]
    print("INTEGRITY: frozen test ids and all 6 locked models verified by SHA-256.\n")
    print("TEST SET - primary thresholds")
    print(prim[["model", "threshold", "f1", "recall_malicious", "precision", "fpr", "roc_auc", "brier",
                "tp", "fp", "tn", "fn"]].to_string(index=False))
    diffs = r["ci"][r["ci"]["model_or_difference"].str.contains(" - ")]
    print("\nPAIRED DIFFERENCES (cluster bootstrap 95% CI)")
    print(diffs.to_string(index=False))
    print("\nEXACT McNEMAR (Holm-adjusted)")
    print(r["mcnemar"].to_string(index=False))
    lm = r["lm"][r["lm"]["threshold_view"] == "primary"][["model", "f1", "recall_malicious", "roc_auc"]]
    print(f"\nLENGTH-MATCHED SUBSET: n={r['meta']['subset_size']} {r['meta']['per_class']} "
          f"bins={r['meta']['n_bins']} -> {r['meta']['interpretation_tier']}")
    print(lm.to_string(index=False))
    print("\nFigures: model_comparison.png, confusion_matrix_E3.png, length_matched_auc.png")

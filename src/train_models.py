"""
Phase 7a: train, tune and LOCK every model using TRAIN and VALIDATION only (D14).
The frozen test split is never loaded here.

For each model:
  1. training-only feature filter (zero variance / rare flags)              - train
  2. C chosen by grouped 5-fold CV (D12 group ids), mean F1, ties -> smaller - train
  3. final fit                                                                - train
  4. thresholds: primary (recall >= 0.90 and FPR <= 0.50), F1-optimal, 0.50  - validation
  5. calibration: Brier score + reliability diagram                           - validation
Outputs: models/<name>.joblib, outputs/model_manifest.json, validation tables/figures.

Run:  python -m src.train_models      then COMMIT before Phase 7b (test evaluation).
"""
from __future__ import annotations

import hashlib
import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src import config
from src.qr_features import QR_FEATURES, QR_LOG1P_FEATURES
from src.url_features import URL_FEATURES, URL_LOG1P_FEATURES

LOG1P = set(URL_LOG1P_FEATURES) | set(QR_LOG1P_FEATURES)
MANIFEST_JSON = config.OUTPUTS_DIR / "model_manifest.json"


# --------------------------------------------------------------------------- data
def load_train_val() -> tuple[pd.DataFrame, pd.DataFrame]:
    feats = pd.read_csv(config.FEATURES_CSV, keep_default_na=False, na_values=[""])
    feats = feats[feats["distortion_type"] == "none"]
    groups = pd.read_csv(config.PROCESSED_DIR / "split_manifest.csv")[["original_qr_id", "split_group_id"]]
    feats = feats.merge(groups, on="original_qr_id", how="left", validate="one_to_one")
    train, val = feats[feats["split"] == "train"].copy(), feats[feats["split"] == "val"].copy()
    assert len(train) and len(val) and "test" not in set(train["split"]) | set(val["split"])
    return train, val


# --------------------------------------------------------------------------- model specs
def reduced_qr_features(train: pd.DataFrame) -> tuple[list[str], dict[str, float]]:
    """QR features with |Spearman rho| with url_length <= threshold, computed on TRAIN only."""
    rhos = {}
    for f in QR_FEATURES:
        s = train[f]
        rhos[f] = float("nan") if s.nunique() < 2 else float(train["url_length"].corr(s, method="spearman"))
    keep = [f for f, r in rhos.items() if not np.isnan(r) and abs(r) <= config.REDUCED_QR_MAX_ABS_RHO]
    return keep, {k: round(v, 4) for k, v in rhos.items()}


def model_specs(train: pd.DataFrame) -> tuple[dict[str, list[str]], dict]:
    reduced, rhos = reduced_qr_features(train)
    specs = {
        "M1_length_only": ["url_length"],
        "M1_plus_QR": ["url_length"] + QR_FEATURES,
        "E1_url_only": URL_FEATURES,
        "E2_qr_only": QR_FEATURES,
        "E3_fusion": URL_FEATURES + QR_FEATURES,
        "E3_reducedQR": URL_FEATURES + reduced,
    }
    return specs, {"reduced_qr_kept": reduced, "train_spearman_with_url_length": rhos}


def training_filter(train: pd.DataFrame, features: list[str]) -> tuple[list[str], dict[str, str]]:
    """D9: drop zero-variance features and 0/1 flags with too few positives - TRAIN only."""
    kept, dropped = [], {}
    for f in features:
        s = train[f].dropna()
        if s.nunique() < 2:
            dropped[f] = "zero variance in train"
        elif set(s.unique()) <= {0, 1} and s.sum() < config.MIN_BINARY_POSITIVES_TRAIN:
            dropped[f] = f"< {config.MIN_BINARY_POSITIVES_TRAIN} positives in train"
        else:
            kept.append(f)
    return kept, dropped


def build_pipeline(features: list[str], C: float) -> Pipeline:
    log_cols = [f for f in features if f in LOG1P]
    plain_cols = [f for f in features if f not in LOG1P]
    branches = []
    if log_cols:
        branches.append(("log", Pipeline([("impute", SimpleImputer(strategy="median")),
                                          ("log1p", FunctionTransformer(np.log1p)),
                                          ("scale", StandardScaler())]), log_cols))
    if plain_cols:
        branches.append(("plain", Pipeline([("impute", SimpleImputer(strategy="median")),
                                            ("scale", StandardScaler())]), plain_cols))
    return Pipeline([("prep", ColumnTransformer(branches)),
                     ("clf", LogisticRegression(C=C, max_iter=2000, random_state=config.RANDOM_SEED))])


# --------------------------------------------------------------------------- tuning
def select_C(train: pd.DataFrame, features: list[str]) -> tuple[float, list[dict]]:
    cv = StratifiedGroupKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    X, y, g = train[features], train["label"].to_numpy(), train["split_group_id"].to_numpy()
    rows = []
    for C in config.LR_C_GRID:
        scores = []
        for tr, te in cv.split(X, y, g):
            pipe = build_pipeline(features, C).fit(X.iloc[tr], y[tr])
            scores.append(f1_score(y[te], pipe.predict(X.iloc[te])))
        rows.append({"C": C, "fold_f1": [round(s, 4) for s in scores], "mean_f1": round(float(np.mean(scores)), 4)})
    best = max(rows, key=lambda r: (round(r["mean_f1"], config.CV_TIE_DECIMALS), -r["C"]))  # tie -> smaller C
    return best["C"], rows


def metrics_at(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    pred = (p >= t).astype(int)
    neg = y == 0
    return {"threshold": round(float(t), 6),
            "f1": round(f1_score(y, pred, zero_division=0), 4),
            "recall_malicious": round(recall_score(y, pred, zero_division=0), 4),
            "precision": round(precision_score(y, pred, zero_division=0), 4),
            "fpr": round(float(pred[neg].mean()) if neg.any() else float("nan"), 4)}


def choose_thresholds(y: np.ndarray, p: np.ndarray) -> dict:
    """All thresholds from VALIDATION scores. Prediction rule: malicious if p >= t."""
    candidates = np.unique(np.concatenate([p, [0.5]]))
    grid = [metrics_at(y, p, t) for t in candidates]
    feasible = [m for m in grid if m["recall_malicious"] >= config.THRESHOLD_MIN_RECALL
                and m["fpr"] <= config.THRESHOLD_MAX_FPR]
    f1_opt = max(grid, key=lambda m: (m["f1"], m["threshold"]))  # tie -> higher threshold
    if feasible:
        primary, rule = max(feasible, key=lambda m: m["threshold"]), "recall>=0.90 & FPR<=0.50"
    else:
        primary, rule = f1_opt, "fallback: F1-optimal (recall/FPR target infeasible)"
    return {"primary": {**primary, "rule": rule}, "f1_optimal": f1_opt, "fixed_0.50": metrics_at(y, p, 0.5)}


def calibration_stats(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    table = [{"bin": b, "n": int((bins == b).sum()),
              "mean_pred": float(p[bins == b].mean()) if (bins == b).any() else np.nan,
              "frac_malicious": float(y[bins == b].mean()) if (bins == b).any() else np.nan}
             for b in range(n_bins)]
    ece = sum(r["n"] / len(y) * abs(r["mean_pred"] - r["frac_malicious"]) for r in table if r["n"])
    return {"brier": round(brier_score_loss(y, p), 4), "ece": round(float(ece), 4), "bins": table}


# --------------------------------------------------------------------------- figure
def plot_reliability(calib: dict[str, dict]) -> None:
    """Small multiples: one reliability panel per model (validation, bins with n >= 5)."""
    names = list(calib)
    cols = 3
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows), sharex=True, sharey=True, squeeze=False)
    for ax, name in zip(axes.flat, names):
        c = calib[name]
        ax.plot([0, 1], [0, 1], color="#999999", linewidth=1, linestyle="--")
        pts = [(b["mean_pred"], b["frac_malicious"], b["n"]) for b in c["bins"] if b["n"] >= 5]
        if pts:
            xs, ys, ns = zip(*pts)
            ax.plot(xs, ys, color="#2a78d6", linewidth=2, marker="o", markersize=6)
        ax.set_title(f"{name}\nBrier {c['brier']:.3f} · ECE {c['ece']:.3f}", fontsize=11, loc="left")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    fig.supxlabel("Mean predicted malicious score (validation)", fontsize=12)
    fig.supylabel("Observed fraction malicious", fontsize=12)
    fig.suptitle("Calibration on the validation set (dashed = perfect)", fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "calibration_validation.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def train_all() -> dict:
    config.ensure_directories()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train, val = load_train_val()
    specs, reduced_info = model_specs(train)
    y_tr, y_val = train["label"].to_numpy(), val["label"].to_numpy()

    manifest = {"protocol": "D14", "final_fit": config.FINAL_FIT, "n_train": len(train), "n_val": len(val),
                "reduced_qr": reduced_info, "models": {}}
    val_rows, cv_rows, calib = [], [], {}
    for name, feats in specs.items():
        kept, dropped = training_filter(train, feats)
        C, cv = select_C(train, kept)
        pipe = build_pipeline(kept, C).fit(train[kept], y_tr)
        p_val = pipe.predict_proba(val[kept])[:, 1]
        # Guard: the matmul RuntimeWarnings seen on macOS/Accelerate are silenced below,
        # so we PROVE numerical soundness instead of trusting silence.
        coef = pipe[-1].coef_
        assert np.isfinite(coef).all() and np.isfinite(p_val).all(), f"{name}: non-finite coefficients/scores"
        thr = choose_thresholds(y_val, p_val)
        calib[name] = calibration_stats(y_val, p_val)

        path = config.MODELS_DIR / f"{name}.joblib"
        joblib.dump({"name": name, "pipeline": pipe, "features": kept, "thresholds": thr}, path)
        manifest["models"][name] = {
            "features_used": kept, "features_dropped": dropped, "C": C,
            "cv_mean_f1": next(r["mean_f1"] for r in cv if r["C"] == C),
            "thresholds": {k: v["threshold"] for k, v in thr.items()},
            "primary_rule": thr["primary"]["rule"],
            "val_auc": round(roc_auc_score(y_val, p_val), 4),
            "val_brier": calib[name]["brier"], "val_ece": calib[name]["ece"],
            "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for view, m in thr.items():
            val_rows.append({"model": name, "threshold_view": view, **{k: v for k, v in m.items() if k != "rule"},
                             "val_auc": manifest["models"][name]["val_auc"]})
        for r in cv:
            cv_rows.append({"model": name, **r})
        print(f"{name:16s} features={len(kept):2d} (dropped {len(dropped)})  C={C:<5}  "
              f"val AUC={manifest['models'][name]['val_auc']:.3f}  primary t={thr['primary']['threshold']:.3f} "
              f"[{'target met' if thr['primary']['rule'].startswith('recall') else 'FALLBACK'}]")

    pd.DataFrame(val_rows).to_csv(config.TABLES_DIR / "validation_metrics.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(config.TABLES_DIR / "cv_c_selection.csv", index=False)
    pd.DataFrame([{"model": k, "brier": v["brier"], "ece": v["ece"]} for k, v in calib.items()]).to_csv(
        config.TABLES_DIR / "calibration_validation.csv", index=False)
    plot_reliability(calib)
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    # Spurious on Apple Silicon (NumPy 2 + Accelerate); soundness is asserted per model above.
    warnings.filterwarnings("ignore", message=r".*encountered in matmul", category=RuntimeWarning)
    m = train_all()
    print(f"\nReduced-QR model keeps: {m['reduced_qr']['reduced_qr_kept']}")
    print("\nSaved: models/*.joblib, outputs/model_manifest.json, outputs/tables/validation_metrics.csv,")
    print("       outputs/tables/cv_c_selection.csv, outputs/figures/calibration_validation.png")
    print("NEXT: commit (models locked) BEFORE running the test evaluation.")

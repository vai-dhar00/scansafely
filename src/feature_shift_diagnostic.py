"""
D24 - POST-HOC EXPLORATORY diagnostic: WHY does E3 fusion become fragile under distortion?
Primary Day 2 / Day 3 results are frozen; this script changes no model, threshold or result.

1. PAIRED feature-shift table: every survivor (exact decode) is compared with ITS OWN clean
   original. For conditions with < 100% exact decoding the results describe the survivor
   subset only and are labelled so (review v5).
   Columns: feature_name, feature_group, clean_median, distorted_median, raw_delta
   (median of paired differences), transformed_delta (mean paired difference after the model's
   own log1p/scaling), model_coefficient, mean_logit_contribution_clean/_distorted,
   mean_logit_shift.  Correlated features may share or redistribute contribution.
2. E3 risk-score distributions, benign vs malicious, clean vs each condition, with the locked
   threshold drawn - shows whether benign scores cross the threshold.
3. Padding control: pad_like_rot15 adds the white margin of a 15-degree rotation WITHOUT
   rotating. Verdict rule fixed before running (share of the rotation +15 QR-group logit shift
   reproduced by padding alone): < 10% "none", 10-60% "part", > 60% "most".
4. Strong low resolution: attempted / exact counts by label WITHIN each QR-version band;
   cells with < 10 attempts flagged sparse.

Run:  python -m src.feature_shift_diagnostic
"""
from __future__ import annotations

import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.distortions import apply, rotation
from src.feature_pipeline import process_image
from src.qr_decoder import load_image
from src.qr_features import QR_FEATURES
from src.robustness_experiment import load_test_meta

MODEL = "E3_fusion"
CONDITIONS = [  # (label, type, value) - selected before running
    ("clean", "none", 0),
    ("blur strong", "gaussian_blur", 11),
    ("jpeg q30", "jpeg", 30),
    ("perspective 10%", "perspective", 0.10),
    ("rotation +5", "rotation", 5),
    ("rotation +15", "rotation", 15),
    ("pad_like_rot15 (control)", "pad", 15),
    ("low-res mild", "low_resolution", 0.50),
    ("low-res strong", "low_resolution", 0.15),
]
PAD_VERDICT = [(0.10, "none"), (0.60, "part"), (float("inf"), "most")]
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9a9a94"


def pad_like_rotation(img: np.ndarray, degrees: float) -> np.ndarray:
    """White border so the canvas matches a rotation's expanded size - but no rotation."""
    target = rotation(img, degrees).shape
    h, w = img.shape
    top, left = (target[0] - h) // 2, (target[1] - w) // 2
    out = np.full(target, 255, dtype=np.uint8)
    out[top:top + h, left:left + w] = img
    return out


def model_parts(bundle: dict, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Transformed values Z, per-feature logit contributions (coef x Z), and coefficients."""
    pipe = bundle["pipeline"]
    ct = pipe.named_steps["prep"]
    order = [c for name, _, cols in ct.transformers_ if name != "remainder" for c in cols]
    Z = pd.DataFrame(ct.transform(X[bundle["features"]]), columns=order, index=X.index)
    coef = pd.Series(pipe.named_steps["clf"].coef_.ravel(), index=order)
    return Z, Z * coef, coef


def collect(test: pd.DataFrame) -> dict[str, pd.DataFrame]:
    per_cond = {}
    for label, t, v in CONDITIONS:
        rows = []
        for rec in test.itertuples(index=False):
            img = load_image(config.PROJECT_ROOT / rec.image_path)
            out = img if t == "none" else pad_like_rotation(img, v) if t == "pad" else apply(img, t, v)
            f = process_image(out, original_url=rec.url)
            if f["decode_outcome"] == "exact":
                rows.append({"original_qr_id": rec.original_qr_id, "label": rec.label, **f})
        per_cond[label] = pd.DataFrame(rows).set_index("original_qr_id")
    return per_cond


def analyse(per_cond: dict[str, pd.DataFrame], bundle: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    clean = per_cond["clean"]
    Zc, Cc, coef = model_parts(bundle, clean)
    intercept = bundle["pipeline"].named_steps["clf"].intercept_[0]
    threshold = bundle["thresholds"]["primary"]["threshold"]
    n_total = len(clean)

    feat_rows, summ_rows, scores = [], [], {}
    for label, df in per_cond.items():
        ids = df.index.intersection(clean.index)          # PAIRED: survivors vs their own clean originals
        Zd, Cd, _ = model_parts(bundle, df.loc[ids])
        subset = "full test set" if len(ids) == n_total else f"survivor subset ({len(ids)}/{n_total}, paired)"
        p_dist = 1 / (1 + np.exp(-(Cd.sum(axis=1) + intercept)))
        p_clean = 1 / (1 + np.exp(-(Cc.loc[ids].sum(axis=1) + intercept)))
        scores[label] = pd.DataFrame({"label": df.loc[ids, "label"], "score_clean": p_clean, "score_dist": p_dist})
        benign = df.loc[ids, "label"] == 0
        url_cols = [c for c in Cd.columns if c not in QR_FEATURES]
        qr_cols = [c for c in Cd.columns if c in QR_FEATURES]
        delta = Cd - Cc.loc[ids]
        summ_rows.append({
            "condition": label, "analysis_set": subset, "n_exact": len(ids),
            "mean_logit_shift_URL_group": round(float(delta[url_cols].sum(axis=1).mean()), 4),
            "mean_logit_shift_QR_group": round(float(delta[qr_cols].sum(axis=1).mean()), 4),
            "benign_FPR_clean_same_images": round(float((p_clean[benign] >= threshold).mean()), 4),
            "benign_FPR_distorted": round(float((p_dist[benign] >= threshold).mean()), 4),
        })
        for f in Cd.columns:
            raw_c, raw_d = clean.loc[ids, f].astype(float), df.loc[ids, f].astype(float)
            feat_rows.append({
                "condition": label, "analysis_set": subset, "feature_name": f,
                "feature_group": "QR/image" if f in QR_FEATURES else "URL",
                "clean_median": round(float(raw_c.median()), 4), "distorted_median": round(float(raw_d.median()), 4),
                "raw_delta": round(float((raw_d - raw_c).median()), 4),
                "transformed_delta": round(float((Zd[f] - Zc.loc[ids, f]).mean()), 4),
                "model_coefficient": round(float(coef[f]), 4),
                "mean_logit_contribution_clean": round(float(Cc.loc[ids, f].mean()), 4),
                "mean_logit_contribution_distorted": round(float(Cd[f].mean()), 4),
                "mean_logit_shift": round(float(delta[f].mean()), 4),
            })
    summary, features = pd.DataFrame(summ_rows), pd.DataFrame(feat_rows)

    rot = summary.set_index("condition").loc["rotation +15", "mean_logit_shift_QR_group"]
    pad = summary.set_index("condition").loc["pad_like_rot15 (control)", "mean_logit_shift_QR_group"]
    share = pad / rot if abs(rot) > 1e-9 else float("nan")
    verdict = next(v for cut, v in PAD_VERDICT if (abs(share) if np.isfinite(share) else 0) < cut)
    pad_info = {"rotation_15_QR_logit_shift": round(float(rot), 4), "padding_only_QR_logit_shift": round(float(pad), 4),
                "share_explained_by_padding": round(float(share), 3), "verdict": verdict,
                "rule": "<10% none, 10-60% part, >60% most (fixed before running)"}
    return summary, features, {"pad": pad_info, "scores": scores, "threshold": threshold}


def within_band_table() -> pd.DataFrame:
    res = pd.read_csv(config.TABLES_DIR / "robustness_detailed_results.csv")
    lr = res[(res["distortion_type"] == "low_resolution") & (res["distortion_severity"] == "strong")]
    t = (lr.assign(exact=lr["decode_outcome"] == "exact")
         .groupby(["version_band", "label"])["exact"].agg(exact="sum", attempted="count").reset_index())
    t["class"] = t["label"].map({0: "benign", 1: "malicious"})
    t["exact_rate"] = (t["exact"] / t["attempted"]).round(3)
    t["sparse_lt10"] = t["attempted"] < 10
    return t[["version_band", "class", "exact", "attempted", "exact_rate", "sparse_lt10"]]


def plot_heatmap(features: pd.DataFrame) -> None:
    conds = [c for c, _, _ in CONDITIONS if c != "clean"]
    q = features[(features["feature_group"] == "QR/image") & features["condition"].isin(conds)]
    piv = q.pivot(index="feature_name", columns="condition", values="mean_logit_shift")[conds]
    piv = piv.loc[piv.abs().max(axis=1).sort_values().index]
    vmax = max(0.5, float(piv.abs().max().max()))
    fig, ax = plt.subplots(figsize=(12, 0.55 * len(piv) + 2.2))
    im = ax.imshow(piv.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > 0.6 * vmax else "#1a1a19")
    ax.set_xticks(range(len(conds)), conds, rotation=25, ha="right", fontsize=10)
    ax.set_yticks(range(len(piv)), piv.index, fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    cb.set_label("mean paired shift in E3 log-odds vs own clean image\n(+ = towards 'malicious')", fontsize=9)
    ax.set_title("Which QR/image features move E3? (exploratory; low-res strong = survivor subset)",
                 fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "diag_e3_feature_shift.png", dpi=200)
    plt.close(fig)


def plot_scores(scores: dict[str, pd.DataFrame], threshold: float) -> None:
    conds = [c for c, _, _ in CONDITIONS if c != "clean"]
    bins = np.linspace(0, 1, 26)
    fig, axes = plt.subplots(2, len(conds), figsize=(2.6 * len(conds), 5.4), sharex=True, sharey="row")
    for j, c in enumerate(conds):
        s = scores[c]
        for i, (lab, name, col) in enumerate(((0, "benign", BLUE), (1, "malicious", ORANGE))):
            ax = axes[i, j]
            sub = s[s["label"] == lab]
            ax.hist(sub["score_clean"], bins=bins, histtype="step", color=GREY, linewidth=1.5, label="clean (same images)")
            ax.hist(sub["score_dist"], bins=bins, histtype="step", color=col, linewidth=2, label="distorted")
            ax.axvline(threshold, color="#1a1a19", linestyle="--", linewidth=1)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8)
            if i == 0:
                ax.set_title(f"{c}\n(n={len(s)})", fontsize=9, loc="left")
            if j == 0:
                ax.set_ylabel(f"{name}\ncount", fontsize=10)
    for j in range(len(conds)):
        axes[1, j].set_xlabel("E3 score", fontsize=9)
    h0, l0 = axes[0, 0].get_legend_handles_labels()
    h1, _ = axes[1, 0].get_legend_handles_labels()
    fig.legend([h0[0], h0[1], h1[1]], ["clean (same images)", "distorted · benign", "distorted · malicious"],
               loc="lower center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"E3 risk-score distributions, paired clean vs distorted (dashed = locked threshold {threshold:.3f})",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(config.FIGURES_DIR / "diag_e3_score_distributions.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=r".*encountered in matmul", category=RuntimeWarning)
    bundle = joblib.load(config.MODELS_DIR / f"{MODEL}.joblib")
    per_cond = collect(load_test_meta())
    summary, features, extra = analyse(per_cond, bundle)
    within = within_band_table()

    T = config.TABLES_DIR
    summary.to_csv(T / "diag_e3_shift_summary.csv", index=False)
    features.to_csv(T / "diag_e3_feature_shifts.csv", index=False)
    within.to_csv(T / "diag_lowres_strong_class_within_band.csv", index=False)
    (T / "diag_padding_control.json").write_text(json.dumps(extra["pad"], indent=2))
    plot_heatmap(features)
    plot_scores(extra["scores"], extra["threshold"])

    pd.set_option("display.width", 220)
    print("D24 EXPLORATORY DIAGNOSTIC (primary results unchanged)\n")
    print("E3 LOGIT SHIFT BY FEATURE GROUP - paired against each image's own clean original")
    print(summary.to_string(index=False))
    pad = extra["pad"]
    print(f"\nPADDING CONTROL: padding alone reproduces {pad['share_explained_by_padding']:.0%} of the rotation +15 "
          f"QR-group shift -> explains '{pad['verdict']}' of it ({pad['rule']})")
    print("\nTOP 3 QR-FEATURE SHIFTS PER CONDITION")
    q = features[(features["feature_group"] == "QR/image") & (features["condition"] != "clean")]
    q = q.reindex(q["mean_logit_shift"].abs().sort_values(ascending=False).index)
    print(q.groupby("condition", sort=False).head(3)[["condition", "feature_name", "clean_median", "distorted_median",
                                                       "raw_delta", "model_coefficient", "mean_logit_shift"]]
          .to_string(index=False))
    print("\nSTRONG LOW RESOLUTION - EXACT DECODE BY CLASS WITHIN QR-VERSION BAND")
    print(within.to_string(index=False))
    print("\nSaved: diag_e3_shift_summary.csv, diag_e3_feature_shifts.csv, diag_lowres_strong_class_within_band.csv,")
    print("       diag_padding_control.json, figures/diag_e3_feature_shift.png, figures/diag_e3_score_distributions.png")

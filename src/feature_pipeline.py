"""
One entry point that turns a QR image into a feature row:
    image -> local decode -> URL features (from the decoded payload) + QR/image features

`process_image` is the ONLY path used by features.csv, the robustness experiment and
the Streamlit demo, so all three compute features identically.

Run:  python -m src.feature_pipeline     (builds features.csv for the clean images and
                                          the QR-feature vs URL-length correlation table)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.qr_decoder import decode_outcome, decode_qr, load_image
from src.qr_features import QR_DESCRIPTIVE, QR_FEATURES, extract_qr_features
from src.url_features import URL_FEATURES, extract_url_features

MODEL_FEATURES: list[str] = URL_FEATURES + QR_FEATURES


def process_image(source: str | Path | np.ndarray, original_url: str | None = None) -> dict:
    """Decode + extract every feature for one image. URL features are NaN when decoding fails.
    `original_url` is only known in experiments; the demo passes None."""
    gray = load_image(source)
    result = decode_qr(gray)
    row: dict = {
        "decode_success": int(result.success),
        "decoded_payload": result.payload,
        "decoder_name": result.decoder_name,
        "decode_outcome": decode_outcome(result, original_url) if original_url is not None
                          else ("decoded" if result.success else "failure"),
    }
    url_feats = extract_url_features(result.payload) if result.success else {f: np.nan for f in URL_FEATURES}
    row.update(url_feats)
    row.update(extract_qr_features(gray, result))
    return row


def build_features_csv() -> pd.DataFrame:
    """features.csv: one row per CLEAN image (distorted rows are added on Day 3)."""
    for path in (config.URLS_CSV, config.QR_MANIFEST_CSV):
        if not path.exists():
            raise SystemExit(f"Missing {path.name}. Run data_loading and qr_generation first.")
    urls = pd.read_csv(config.URLS_CSV, keep_default_na=False)
    manifest = pd.read_csv(config.QR_MANIFEST_CSV)[["original_qr_id", "image_path", "qr_version"]]
    meta = urls.merge(manifest, on="original_qr_id", how="left", validate="one_to_one")

    rows = []
    for rec in meta.itertuples(index=False):
        feats = process_image(config.PROJECT_ROOT / rec.image_path, original_url=rec.url)
        rows.append({
            "qr_id": f"{rec.original_qr_id}__clean",
            "original_qr_id": rec.original_qr_id,
            "label": rec.label,
            "split": rec.split,
            "registered_domain": rec.registered_domain,
            "image_path": rec.image_path,
            "distortion_type": "none",
            "distortion_severity": "none",
            "qr_version_generator": rec.qr_version,  # DESCRIPTIVE ONLY (D10)
            **feats,
        })
    df = pd.DataFrame(rows)
    df.to_csv(config.FEATURES_CSV, index=False)
    return df


def length_correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of each QR/image feature with URL length (clean images)."""
    rows = []
    for feat in QR_FEATURES + QR_DESCRIPTIVE + ["qr_version_generator"]:
        s = df[feat]
        rho = np.nan if s.nunique(dropna=True) < 2 else df["url_length"].corr(s, method="spearman")
        rows.append({"feature": feat,
                     "role": "model input" if feat in QR_FEATURES else "descriptive only",
                     "spearman_rho_with_url_length": round(float(rho), 4) if pd.notna(rho) else np.nan,
                     "n_unique_values": int(s.nunique(dropna=True))})
    table = pd.DataFrame(rows).sort_values("spearman_rho_with_url_length", key=lambda x: -x.abs(), na_position="last")
    table.to_csv(config.TABLES_DIR / "qr_feature_length_correlation.csv", index=False)
    return table


def plot_length_correlation(table: pd.DataFrame) -> None:
    t = table[table["role"] == "model input"].dropna(subset=["spearman_rho_with_url_length"])
    t = t.iloc[::-1]  # strongest at the top
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(t) + 1.5))
    ax.barh(t["feature"], t["spearman_rho_with_url_length"].abs(), color="#2a78d6", height=0.6)
    for y, v in enumerate(t["spearman_rho_with_url_length"]):
        ax.text(abs(v) + 0.01, y, f"{v:+.2f}", va="center", fontsize=11)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("|Spearman ρ| with URL length (clean images)", fontsize=12)
    ax.set_title("QR/image model features vs URL length", fontsize=13, loc="left")
    ax.tick_params(labelsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "qr_feature_length_correlation.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    config.ensure_directories()
    features = build_features_csv()
    corr = length_correlation_table(features)
    plot_length_correlation(corr)
    print(f"Wrote {config.FEATURES_CSV.relative_to(config.PROJECT_ROOT)}: {len(features)} rows, "
          f"{len(MODEL_FEATURES)} model features ({len(URL_FEATURES)} URL + {len(QR_FEATURES)} QR)")
    print("Decode outcomes:", features["decode_outcome"].value_counts().to_dict())
    print("\nQR/image features vs URL length (Spearman, clean images):")
    print(corr.to_string(index=False))
    print("\nSaved: outputs/tables/qr_feature_length_correlation.csv, "
          "outputs/figures/qr_feature_length_correlation.png")

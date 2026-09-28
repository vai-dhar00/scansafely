"""
Generate one clean QR image per URL in urls.csv, then verify every image decodes
back to exactly the original URL.

Every QR uses the SAME settings from config.py (error correction, box size,
border), so rendering settings cannot carry label information. The only thing
that differs between images is the payload - which is why QR structure
(version / module count) is expected to track URL length. We measure that here.

Outputs:
  data/qr_images/<original_qr_id>.png
  data/processed/qr_manifest.csv            (one row per clean image)
  outputs/tables/qr_generation_summary.json

Run:  python -m src.qr_generation
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import qrcode
from qrcode.constants import ERROR_CORRECT_H, ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_Q

from src import config
from src.qr_decoder import decode_outcome, decode_qr

_EC_LEVELS = {"L": ERROR_CORRECT_L, "M": ERROR_CORRECT_M, "Q": ERROR_CORRECT_Q, "H": ERROR_CORRECT_H}


def generate_qr_image(payload: str, out_path: Path) -> int:
    """Render `payload` as a QR PNG using the fixed project settings. Returns the QR version (1-40)."""
    qr = qrcode.QRCode(
        version=None,  # smallest version that fits the payload
        error_correction=_EC_LEVELS[config.QR_ERROR_CORRECTION],
        box_size=config.QR_BOX_SIZE,
        border=config.QR_BORDER,
    )
    qr.add_data(payload)
    qr.make(fit=True)
    qr.make_image(fill_color="black", back_color="white").save(out_path)
    return int(qr.version)


def build_qr_dataset() -> pd.DataFrame:
    if not config.URLS_CSV.exists():
        raise SystemExit("urls.csv not found. Run: python -m src.data_loading")
    config.ensure_directories()
    urls = pd.read_csv(config.URLS_CSV, keep_default_na=False)

    rows = []
    for rec in urls.itertuples(index=False):
        img_path = config.QR_IMAGES_DIR / f"{rec.original_qr_id}.png"
        version = generate_qr_image(rec.url, img_path)
        result = decode_qr(img_path)
        rows.append({
            "original_qr_id": rec.original_qr_id,
            "label": rec.label,
            "image_path": img_path.relative_to(config.PROJECT_ROOT).as_posix(),
            "url_length": len(rec.url),
            "qr_version": version,
            "qr_modules_per_side": 17 + 4 * version,  # QR standard: version v has 17+4v modules per side
            "decode_success": int(result.success),
            "decoded_matches_original": int(result.payload == rec.url),
            "decode_outcome": decode_outcome(result, rec.url),
            "decoder_name": result.decoder_name,
        })

    manifest = pd.DataFrame(rows)
    manifest.to_csv(config.QR_MANIFEST_CSV, index=False)

    # Early look at the payload-length confound (descriptive statistic on all clean images, no model involved).
    spearman = manifest["url_length"].corr(manifest["qr_version"], method="spearman")
    summary = {
        "n_images": len(manifest),
        "qr_settings": {"error_correction": config.QR_ERROR_CORRECTION,
                        "box_size": config.QR_BOX_SIZE, "border": config.QR_BORDER},
        "clean_decode_rate": round(manifest["decode_success"].mean(), 4),
        "clean_exact_match_rate": round(manifest["decoded_matches_original"].mean(), 4),
        "decode_outcome_counts": manifest["decode_outcome"].value_counts().to_dict(),
        "decoder_used_counts": manifest["decoder_name"].value_counts().to_dict(),
        "qr_version_counts": {int(k): int(v) for k, v in manifest["qr_version"].value_counts().sort_index().items()},
        "median_qr_version_by_label": {config.LABEL_NAMES[int(k)]: float(v)
                                       for k, v in manifest.groupby("label")["qr_version"].median().items()},
        "spearman_url_length_vs_qr_version": round(float(spearman), 4),
    }
    (config.TABLES_DIR / "qr_generation_summary.json").write_text(json.dumps(summary, indent=2))
    return manifest


if __name__ == "__main__":
    m = build_qr_dataset()
    s = json.loads((config.TABLES_DIR / "qr_generation_summary.json").read_text())
    print(f"Generated {s['n_images']} QR images in data/qr_images/")
    print(f"Clean decode rate:       {s['clean_decode_rate']:.2%}")
    print(f"Exact payload match:     {s['clean_exact_match_rate']:.2%}")
    print(f"Decode outcomes:         {s['decode_outcome_counts']}")
    print(f"Decoder used:            {s['decoder_used_counts']}")
    print(f"QR version counts:       {s['qr_version_counts']}")
    print(f"Median version by label: {s['median_qr_version_by_label']}")
    print(f"Spearman(URL length, QR version) = {s['spearman_url_length_vs_qr_version']}")
    failures = m[m["decoded_matches_original"] == 0]
    if len(failures):
        print(f"\n{len(failures)} clean image(s) did not decode to the exact URL. IDs (first 10):",
              failures["original_qr_id"].head(10).tolist())
    print("\nSaved: data/processed/qr_manifest.csv, outputs/tables/qr_generation_summary.json")

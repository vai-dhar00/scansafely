"""
QR/image features computed from PIXELS ONLY (D10), so the same function works for
clean generated images, distorted test images and uploaded demo images.

Scale invariance (D13): uploads can be any resolution, so every pixel statistic is
computed on a standardised 512x512 grayscale copy. Raw width/height are kept as
DESCRIPTIVE columns only and are never model inputs.

Expected finding (not hidden): under fixed rendering, most of these features follow
payload length. `qr_decoded_modules_per_side` is the decoder's reading of the code
size (17 + 4 x version) and is expected to be an almost perfect URL-length proxy.
"""
from __future__ import annotations

import cv2
import numpy as np

from src import config
from src.qr_decoder import DecodeResult

STD_SIZE = 512  # all pixel statistics use this standardised resolution

QR_FEATURES: list[str] = [
    "qr_aspect_ratio",
    "qr_gray_mean",
    "qr_gray_std",
    "qr_dark_ratio",
    "qr_contrast",
    "qr_blur_score",
    "qr_edge_density",
    "qr_dark_component_density",
    "qr_quiet_zone_ratio",
    "qr_decoded_modules_per_side",
]
# log1p before scaling (same rule as D9: heavily skewed, non-negative magnitudes)
QR_LOG1P_FEATURES: list[str] = ["qr_blur_score", "qr_dark_component_density", "qr_decoded_modules_per_side"]
QR_DESCRIPTIVE: list[str] = ["qr_width", "qr_height"]


def extract_qr_features(gray: np.ndarray, decode: DecodeResult | None = None) -> dict[str, float]:
    """All QR_FEATURES + QR_DESCRIPTIVE for one grayscale uint8 image."""
    h, w = gray.shape[:2]
    std = cv2.resize(gray, (STD_SIZE, STD_SIZE), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(std, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = binary == 0

    # Connected dark blobs per 10,000 px: more modules -> more, smaller blobs.
    n_labels, _ = cv2.connectedComponents(dark.astype(np.uint8), connectivity=8)
    components = max(n_labels - 1, 0)

    # Quiet zone: smallest blank margin around the dark content, relative to image side.
    rows, cols = np.where(dark)
    if rows.size:
        margin = min(rows.min(), cols.min(), STD_SIZE - 1 - rows.max(), STD_SIZE - 1 - cols.max())
        quiet = margin / STD_SIZE
    else:
        quiet = np.nan

    p1, p99 = np.percentile(std, [1, 99])
    edges = cv2.Canny(std, config.CANNY_LOW, config.CANNY_HIGH)

    return {
        "qr_aspect_ratio": w / h if h else np.nan,
        "qr_gray_mean": float(std.mean()) / 255,
        "qr_gray_std": float(std.std()) / 255,
        "qr_dark_ratio": float((std < config.DARK_PIXEL_THRESHOLD).mean()),
        "qr_contrast": float(p99 - p1) / 255,
        "qr_blur_score": float(cv2.Laplacian(std, cv2.CV_64F).var()),
        "qr_edge_density": float((edges > 0).mean()),
        "qr_dark_component_density": components / (STD_SIZE * STD_SIZE / 10_000),
        "qr_quiet_zone_ratio": float(quiet),
        "qr_decoded_modules_per_side": decode.modules_per_side if decode and decode.success else np.nan,
        "qr_width": float(w),
        "qr_height": float(h),
    }

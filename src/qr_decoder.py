"""
Local, offline QR decoding. Used by dataset building, the robustness experiment
and the Streamlit demo, so all three decode in exactly the same way.

Decoding strategy (fixed; documented as decision D6):
  1. cv2.QRCodeDetector        - OpenCV's standard detector
  2. cv2.QRCodeDetectorAruco   - OpenCV's alternative detector, tried only if (1) fails
No other preprocessing or retries are applied, so decode rates are comparable
across conditions.

SAFETY: the payload is returned as a plain string. Nothing here opens it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class DecodeResult:
    success: bool
    payload: str          # "" when decoding failed
    decoder_name: str     # which detector succeeded, or "none"
    # Modules per side of the rectified code the decoder read (17 + 4 x version).
    # Read from the PIXELS by the decoder, so it is available for any decodable image (D13).
    modules_per_side: float = float("nan")


_DETECTORS = (
    ("opencv", cv2.QRCodeDetector()),
    ("opencv_aruco", cv2.QRCodeDetectorAruco()),
)


def load_image(source: str | Path | np.ndarray) -> np.ndarray:
    """Return a grayscale uint8 image from a file path or an array."""
    if isinstance(source, np.ndarray):
        img = source
    else:
        img = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not read image: {source}")
    if img.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        img = cv2.cvtColor(img, code)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def decode_outcome(result: DecodeResult, original: str) -> str:
    """'exact' = payload identical to the stored URL; 'mismatch' = a different non-empty
    payload (NOT counted as a success); 'failure' = nothing decoded."""
    if not result.success:
        return "failure"
    return "exact" if result.payload == original else "mismatch"


def decode_qr(source: str | Path | np.ndarray) -> DecodeResult:
    """Try each detector in order. Never raises on an unreadable QR - returns success=False."""
    gray = load_image(source)
    for name, detector in _DETECTORS:
        try:
            payload, points, straight = detector.detectAndDecode(gray)
        except cv2.error:
            continue
        if payload:
            modules = float(straight.shape[0]) if straight is not None and straight.size else float("nan")
            return DecodeResult(True, payload, name, modules)
    return DecodeResult(False, "", "none")

"""
Day 4b: everything the demo does with an uploaded file, kept OUT of app.py so it can be unit-tested.

SAFETY (D25 / D28): uploads are processed in memory only (nothing written to disk, the filename is never
read), validated before any pixel decoding, and the decoded payload is only ever returned as inert text.
No network access of any kind.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from src import explainability as ex
from src.qr_decoder import decode_qr

MAX_UPLOAD_BYTES = 5 * 1024 * 1024      # also enforced by .streamlit/config.toml (maxUploadSize = 5 MB)
MAX_SIDE = 6000
MAX_PIXELS = 16_000_000
MIN_SIDE = 32
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}   # content-sniffed by Pillow, not trusted from the extension


class UploadRejected(Exception):
    """Carries a message that is safe to show to the user (never contains file content)."""


def validate_and_load(data: bytes) -> np.ndarray:
    """Bytes -> grayscale uint8 array, or UploadRejected. Header checks come BEFORE pixel decoding."""
    if not data:
        raise UploadRejected("The file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadRejected(f"The file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    try:
        im = Image.open(io.BytesIO(data))          # reads the header only
    except (UnidentifiedImageError, OSError, ValueError):
        raise UploadRejected("This is not a readable PNG, JPG or WEBP image.") from None
    if im.format not in ALLOWED_FORMATS:
        raise UploadRejected("Only PNG, JPG and WEBP images are accepted.")
    w, h = im.size
    if min(w, h) < MIN_SIDE:
        raise UploadRejected(f"The image is too small (minimum {MIN_SIDE} pixels per side).")
    if max(w, h) > MAX_SIDE or w * h > MAX_PIXELS:
        raise UploadRejected(f"The image is too large (maximum {MAX_SIDE} pixels per side, "
                             f"{MAX_PIXELS // 1_000_000} megapixels).")
    try:
        im.seek(0)                                   # animated files: first frame only
        im.load()
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))   # transparent -> white, not black
            im = Image.alpha_composite(bg, rgba)
        gray = np.asarray(im.convert("L"), dtype=np.uint8)
    except Exception:  # noqa: BLE001 - any decoder failure becomes a generic, content-free message
        raise UploadRejected("This image could not be read.") from None
    return gray


def analyse_upload(data: bytes, bundle: dict) -> dict:
    """Validate -> decode locally -> triage payload -> (maybe) E1 score + explanation."""
    gray = validate_and_load(data)
    result = decode_qr(gray)
    out = ex.analyse_payload(result.payload if result.success else None, result.decoder_name, bundle)
    out["image_size"] = (int(gray.shape[1]), int(gray.shape[0]))
    return out

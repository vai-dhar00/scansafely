"""
Deterministic image distortions for the robustness experiment (D19).

Levels come from config.DISTORTIONS (fixed on Day 1). Every function takes and
returns a grayscale uint8 image and involves NO randomness, so each distorted
image is exactly reproducible.

Implementation choices (reviewed before running):
  gaussian_blur  - cv2.GaussianBlur, odd kernel size
  rotation       - about the centre, white fill, canvas EXPANDED to fit, so no part of
                   the code or its quiet zone is cropped at any angle
  jpeg           - encode/decode in memory at the given quality
  low_resolution - downscale with INTER_AREA, upscale back with INTER_NEAREST
                   (controlled digital degradation, not a camera/print simulation)
  perspective    - fixed "keystone" pattern: the two top corners move inward by
                   `frac` x width each (like viewing the code from below); white fill
Occlusion is excluded from the core study (D19).
"""
from __future__ import annotations

import cv2
import numpy as np

from src import config

CORE_DISTORTIONS = ("gaussian_blur", "rotation", "jpeg", "low_resolution", "perspective")


def gaussian_blur(img: np.ndarray, k: float) -> np.ndarray:
    k = int(k) | 1  # force odd
    return cv2.GaussianBlur(img, (k, k), 0)


def rotation(img: np.ndarray, degrees: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w, new_h = int(np.ceil(h * sin + w * cos)), int(np.ceil(h * cos + w * sin))
    M[0, 2] += new_w / 2 - w / 2
    M[1, 2] += new_h / 2 - h / 2
    return cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=255)


def jpeg(img: np.ndarray, quality: float) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def low_resolution(img: np.ndarray, factor: float) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * factor)), max(1, int(h * factor))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def perspective(img: np.ndarray, frac: float) -> np.ndarray:
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[frac * w, 0], [w - frac * w, 0], [w, h], [0, h]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h),
                               flags=cv2.INTER_LINEAR, borderValue=255)


_FUNCS = {"gaussian_blur": gaussian_blur, "rotation": rotation, "jpeg": jpeg,
          "low_resolution": low_resolution, "perspective": perspective}


def conditions() -> list[tuple[str, str, float]]:
    """All (type, severity_label, value) conditions of the core study, in config order."""
    return [(t, sev, val) for t in CORE_DISTORTIONS for sev, val in config.DISTORTIONS[t].items()]


def apply(img: np.ndarray, distortion_type: str, value: float) -> np.ndarray:
    out = _FUNCS[distortion_type](img, value)
    assert out.dtype == np.uint8 and out.ndim == 2
    return out

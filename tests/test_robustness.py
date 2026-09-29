"""Distortion and robustness helper tests. Run:  python -m pytest -q"""
import numpy as np

from src.distortions import apply, conditions, rotation
from src.qr_decoder import decode_qr, load_image
from src.qr_generation import generate_qr_image
from src.robustness_experiment import version_band, wilson


def test_every_condition_is_deterministic_and_uint8(tmp_path):
    generate_qr_image("https://example.com/path", tmp_path / "q.png")
    img = load_image(tmp_path / "q.png")
    for t, _, v in conditions():
        a, b = apply(img, t, v), apply(img, t, v)
        assert a.dtype == np.uint8 and np.array_equal(a, b), t


def test_rotation_never_crops_the_code(tmp_path):
    v = generate_qr_image("https://example.com/" + "a" * 900, tmp_path / "big.png")  # high version
    img = load_image(tmp_path / "big.png")
    out = rotation(img, 15)
    edge = np.concatenate([out[0], out[-1], out[:, 0], out[:, -1]])
    assert v >= 20 and edge.min() > 200  # border stays white -> nothing cut off


def test_mild_distortions_keep_small_codes_decodable(tmp_path):
    generate_qr_image("https://example.com/x", tmp_path / "q.png")
    img = load_image(tmp_path / "q.png")
    for t, v in (("gaussian_blur", 3), ("jpeg", 90), ("rotation", 5), ("perspective", 0.05)):
        assert decode_qr(apply(img, t, v)).payload == "https://example.com/x", t


def test_wilson_and_bands():
    lo, hi = wilson(150, 150)
    assert hi == 1.0 and 0.97 < lo < 1.0
    assert all(np.isnan(x) for x in wilson(0, 0))  # empty denominator -> undefined, not a crash
    assert [version_band(v) for v in (1, 3, 4, 6, 7, 31)] == ["v1-v3"] * 2 + ["v4-v6"] * 2 + ["v7+"] * 2

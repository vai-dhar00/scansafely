"""QR generation/decoding tests. Run:  python -m pytest -q"""
import numpy as np

from src.qr_decoder import DecodeResult, decode_outcome, decode_qr
from src.qr_generation import generate_qr_image


def test_roundtrip_exact(tmp_path):
    url = "https://example.com/path/to/page?id=42&x=%20"
    path = tmp_path / "q.png"
    generate_qr_image(url, path)
    result = decode_qr(path)
    assert result.success and result.payload == url


def test_blank_image_fails_gracefully():
    result = decode_qr(np.full((300, 300), 255, dtype=np.uint8))
    assert not result.success and result.payload == "" and result.decoder_name == "none"


def test_longer_payload_needs_higher_version(tmp_path):
    short_v = generate_qr_image("https://example.com", tmp_path / "s.png")
    long_v = generate_qr_image("https://example.com/" + "a" * 200, tmp_path / "l.png")
    assert long_v > short_v


def test_decode_outcome_three_way():
    assert decode_outcome(DecodeResult(True, "a", "opencv"), "a") == "exact"
    assert decode_outcome(DecodeResult(True, "b", "opencv"), "a") == "mismatch"
    assert decode_outcome(DecodeResult(False, "", "none"), "a") == "failure"

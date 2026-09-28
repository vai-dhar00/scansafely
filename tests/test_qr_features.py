"""QR/image feature tests. Run:  python -m pytest -q"""
import cv2
import numpy as np

from src.feature_pipeline import MODEL_FEATURES, process_image
from src.qr_decoder import decode_qr, load_image
from src.qr_features import QR_FEATURES, QR_LOG1P_FEATURES, extract_qr_features
from src.qr_generation import generate_qr_image


def test_decoder_reads_modules_per_side_exactly(tmp_path):
    for n in (10, 150, 600):
        v = generate_qr_image("https://example.com/" + "a" * n, tmp_path / "q.png")
        assert decode_qr(tmp_path / "q.png").modules_per_side == 17 + 4 * v


def test_features_are_scale_invariant(tmp_path):
    generate_qr_image("https://example.com/page", tmp_path / "q.png")
    g = load_image(tmp_path / "q.png")
    big = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    a, b = extract_qr_features(g), extract_qr_features(big)
    for f in ("qr_dark_ratio", "qr_edge_density", "qr_quiet_zone_ratio", "qr_dark_component_density"):
        assert abs(a[f] - b[f]) < 0.02 * max(1, abs(a[f])), f


def test_process_image_failure_gives_nan_url_features():
    row = process_image(np.full((300, 300), 255, dtype=np.uint8), original_url="https://example.com")
    assert row["decode_outcome"] == "failure"
    assert np.isnan(row["url_length"]) and np.isnan(row["qr_decoded_modules_per_side"])
    assert all(f in row for f in MODEL_FEATURES)


def test_process_image_demo_mode_without_original(tmp_path):
    generate_qr_image("https://example.com/x", tmp_path / "q.png")
    row = process_image(tmp_path / "q.png")
    assert row["decode_outcome"] == "decoded" and row["decoded_payload"] == "https://example.com/x"


def test_log1p_list_is_subset():
    assert set(QR_LOG1P_FEATURES) <= set(QR_FEATURES)

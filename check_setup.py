"""
Phase 1 environment check for ScanSafely.

Run from the project root:   python check_setup.py

It checks that every package imports, creates the project folders, and does an
OFFLINE round-trip: generate a QR for a harmless placeholder string -> decode it
with OpenCV -> compare. No network access is used.
"""
from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

REQUIRED = [
    ("numpy", "numpy"), ("pandas", "pandas"), ("sklearn", "scikit-learn"),
    ("joblib", "joblib"), ("cv2", "opencv-python"), ("PIL", "Pillow"),
    ("qrcode", "qrcode"), ("shap", "shap"), ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"), ("streamlit", "streamlit"), ("pytest", "pytest"),
]


def check_python() -> bool:
    ok = (3, 10) <= sys.version_info[:2] <= (3, 12)
    status = "OK  " if ok else "WARN"
    print(f"[{status}] Python {sys.version.split()[0]} (target 3.10-3.12)")
    return True  # warning only


def check_packages() -> bool:
    all_ok = True
    for module, pip_name in REQUIRED:
        try:
            mod = importlib.import_module(module)
            print(f"[OK  ] {pip_name:<15} {getattr(mod, '__version__', '?')}")
        except Exception as exc:  # noqa: BLE001 - report any import failure
            print(f"[FAIL] {pip_name:<15} {type(exc).__name__}: {exc}")
            all_ok = False
    return all_ok


def check_config() -> bool:
    try:
        from src import config
        config.ensure_directories()
        assert config.probability_to_risk_label(0.10) == "SAFE"
        assert config.probability_to_risk_label(0.30) == "SUSPICIOUS"
        assert config.probability_to_risk_label(0.70) == "HIGH RISK"
        print(f"[OK  ] src/config.py loaded; seed={config.RANDOM_SEED}; folders created")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] src/config.py: {type(exc).__name__}: {exc}")
        return False


def check_qr_roundtrip() -> bool:
    try:
        import cv2
        import qrcode

        payload = "https://example.com/scansafely-setup-test"  # reserved test domain, never visited
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_qr.png"
            qrcode.make(payload).save(path)
            img = cv2.imread(str(path))
            decoded, _, _ = cv2.QRCodeDetector().detectAndDecode(img)
        if decoded == payload:
            print("[OK  ] Offline QR round-trip: generated and decoded correctly")
            return True
        print(f"[FAIL] QR round-trip mismatch: got {decoded!r}")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] QR round-trip: {type(exc).__name__}: {exc}")
        return False


if __name__ == "__main__":
    print("ScanSafely - Phase 1 setup check\n" + "-" * 40)
    results = [check_python(), check_packages(), check_config(), check_qr_roundtrip()]
    print("-" * 40)
    if all(results):
        print("ALL CHECKS PASSED - environment is ready.")
        sys.exit(0)
    print("SOME CHECKS FAILED - copy this whole output and send it back.")
    sys.exit(1)

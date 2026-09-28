"""
Central configuration for ScanSafely.

Every other module imports settings from here, so that:
  * the random seed is defined in exactly one place,
  * file paths work on Windows/macOS/Linux without hardcoding your machine's paths,
  * thresholds and experiment settings are documented and easy to report on the poster.

RULE: if you change a value here after the test set is frozen, record the change
in outputs/experiment_log.json and explain it in the README. Never change a value
because of what you saw on the test set.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Safety boundary (documentation flag - no module in this project may make
# network calls, DNS lookups, WHOIS queries, HTTP requests or open a browser).
# Decoded URLs are treated as inert text strings only.
# ---------------------------------------------------------------------------
ALLOW_NETWORK: bool = False

# ---------------------------------------------------------------------------
# Paths (all relative to the project root, resolved automatically)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
QR_IMAGES_DIR: Path = DATA_DIR / "qr_images"
DISTORTED_DIR: Path = DATA_DIR / "distorted_test_images"

MODELS_DIR: Path = PROJECT_ROOT / "models"

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
TABLES_DIR: Path = OUTPUTS_DIR / "tables"
SHAP_DIR: Path = OUTPUTS_DIR / "shap"
SCREENSHOTS_DIR: Path = OUTPUTS_DIR / "demo_screenshots"

# Key files
URLS_CSV: Path = PROCESSED_DIR / "urls.csv"                  # one row per original URL
FEATURES_CSV: Path = PROCESSED_DIR / "features.csv"          # one row per QR image
ROBUSTNESS_CSV: Path = TABLES_DIR / "robustness_results.csv"
EXPERIMENT_LOG: Path = OUTPUTS_DIR / "experiment_log.json"

ALL_DIRS: tuple[Path, ...] = (
    RAW_DIR, PROCESSED_DIR, QR_IMAGES_DIR, DISTORTED_DIR, MODELS_DIR,
    FIGURES_DIR, TABLES_DIR, SHAP_DIR, SCREENSHOTS_DIR,
)


def ensure_directories() -> None:
    """Create every project folder if it does not exist yet (safe to call repeatedly)."""
    for folder in ALL_DIRS:
        folder.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
LABEL_BENIGN: int = 0
LABEL_MALICIOUS: int = 1
LABEL_NAMES: dict[int, str] = {LABEL_BENIGN: "benign", LABEL_MALICIOUS: "suspicious/malicious"}

# ---------------------------------------------------------------------------
# Data split (grouped by original_qr_id, stratified by label)
# ---------------------------------------------------------------------------
TRAIN_FRACTION: float = 0.70
VAL_FRACTION: float = 0.15
TEST_FRACTION: float = 0.15

# ---------------------------------------------------------------------------
# Risk-label policy for the demo.
# These are PROTOTYPE POLICY THRESHOLDS, not ground-truth classes and not a
# security standard. The model is trained on binary labels only.
# ---------------------------------------------------------------------------
RISK_THRESHOLDS: dict[str, float] = {
    "suspicious": 0.30,  # p >= 0.30 -> SUSPICIOUS
    "high_risk": 0.70,   # p >= 0.70 -> HIGH RISK
}

RISK_RECOMMENDATIONS: dict[str, str] = {
    "SAFE": ("No strong risk indicators were detected by this prototype. "
             "Still verify the destination before entering sensitive information."),
    "SUSPICIOUS": ("Some risk indicators were detected. Verify the destination "
                   "through an independent trusted source before proceeding."),
    "HIGH RISK": ("Multiple risk indicators were detected. Do not navigate to this "
                  "destination unless it is independently verified."),
}

DISCLAIMER: str = "Prototype decision support only. Do not treat this result as a guarantee of safety."


def probability_to_risk_label(prob_malicious: float) -> str:
    """Map a predicted malicious-class probability to SAFE / SUSPICIOUS / HIGH RISK."""
    if not 0.0 <= prob_malicious <= 1.0:
        raise ValueError(f"Probability must be between 0 and 1, got {prob_malicious}")
    if prob_malicious >= RISK_THRESHOLDS["high_risk"]:
        return "HIGH RISK"
    if prob_malicious >= RISK_THRESHOLDS["suspicious"]:
        return "SUSPICIOUS"
    return "SAFE"


# ---------------------------------------------------------------------------
# QR generation (identical settings for every URL, so settings cannot leak the label)
# ---------------------------------------------------------------------------
QR_ERROR_CORRECTION: str = "M"   # L, M, Q or H
QR_BOX_SIZE: int = 10            # pixels per module
QR_BORDER: int = 4               # quiet zone in modules (4 is the QR standard minimum)

# ---------------------------------------------------------------------------
# Image feature settings
# ---------------------------------------------------------------------------
DARK_PIXEL_THRESHOLD: int = 128  # grayscale value below this counts as "dark"
CANNY_LOW: int = 100
CANNY_HIGH: int = 200

# ---------------------------------------------------------------------------
# URL lexical features: transparent, fixed suspicious-token list
# (matched case-insensitively as substrings of the URL string)
# ---------------------------------------------------------------------------
SUSPICIOUS_TOKENS: tuple[str, ...] = (
    "login", "verify", "verification", "secure", "account", "update", "bank",
    "wallet", "password", "signin", "confirm", "payment", "bonus", "reward", "urgent",
)

# ---------------------------------------------------------------------------
# Robustness experiment (applied to the frozen TEST split only).
# Fixed BEFORE running the experiment - do not adjust after seeing test results.
# ---------------------------------------------------------------------------
DISTORTIONS: dict[str, dict[str, float]] = {
    "gaussian_blur": {"mild": 3, "moderate": 7, "strong": 11},          # kernel size (odd, px)
    "rotation": {"-15": -15, "-10": -10, "-5": -5, "+5": 5, "+10": 10, "+15": 15},  # degrees
    "jpeg": {"q90": 90, "q60": 60, "q30": 30},                           # JPEG quality
    "low_resolution": {"mild": 0.50, "moderate": 0.30, "strong": 0.15},  # downscale factor
    "perspective": {"mild": 0.05, "moderate": 0.10},                     # corner shift, fraction of side
    "occlusion": {"small": 0.05, "medium": 0.10},                        # OPTIONAL: fraction of area covered
}

"""
Offline URL lexical features. ONE function, `extract_url_features`, is used by
dataset building, training, the robustness experiment and the Streamlit demo.

The input is always the DECODED QR payload, treated as an inert string.
No network, DNS, WHOIS or browser access of any kind.

Feature list is fixed in URL_FEATURES (decided before any modelling, D8).
"""
from __future__ import annotations

import ipaddress
import math
from collections import Counter
from urllib.parse import parse_qsl, urlsplit

import pandas as pd

from src import config
from src.data_loading import TLD_EXTRACT  # shared offline public-suffix parser (bundled list, no download)

TOKEN_FLAGS: tuple[str, ...] = ("login", "verify", "account", "secure", "update", "bank")
_SPECIAL_EXEMPT = set(".:/")

URL_FEATURES: list[str] = [
    "url_length", "url_hostname_length", "url_path_length", "url_query_length",
    "url_dot_count", "url_subdomain_count", "url_hyphen_count", "url_digit_count",
    "url_digit_ratio", "url_special_char_count", "url_at_count", "url_question_count",
    "url_equals_count", "url_ampersand_count", "url_percent_count", "url_path_depth",
    "url_query_param_count", "url_has_https", "url_has_ip", "url_has_punycode",
    "url_has_nonstandard_port", "url_has_double_slash_path", "url_entropy",
    "url_suspicious_token_count",
] + [f"url_has_token_{t}" for t in TOKEN_FLAGS]


# D9 (fixed before training): these non-negative counts/lengths get log1p before
# standardisation. Flags, digit ratio and entropy are standardised without log.
URL_LOG1P_FEATURES: list[str] = [
    "url_length", "url_hostname_length", "url_path_length", "url_query_length",
    "url_dot_count", "url_subdomain_count", "url_hyphen_count", "url_digit_count",
    "url_special_char_count", "url_at_count", "url_question_count", "url_equals_count",
    "url_ampersand_count", "url_percent_count", "url_path_depth", "url_query_param_count",
    "url_suspicious_token_count",
]


def shannon_entropy(text: str) -> float:
    """Average information per character (bits). Random-looking strings score higher."""
    if not text:
        return 0.0
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in Counter(text).values())


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def extract_url_features(url: str) -> dict[str, float]:
    """Return every feature in URL_FEATURES for one URL string. Never raises on odd input."""
    url = url or ""
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        try:
            port = parts.port
        except ValueError:  # malformed port like ':abc'
            port = None
    except ValueError:  # e.g. malformed IPv6 brackets
        parts, host, port = urlsplit(""), "", None

    path, query = parts.path or "", parts.query or ""
    lower = url.lower()
    is_ip = _is_ip(host) if host else False
    subdomain = "" if is_ip else TLD_EXTRACT(host).subdomain if host else ""
    tokens_found = [t for t in config.SUSPICIOUS_TOKENS if t in lower]
    digits = sum(ch.isdigit() for ch in url)

    feats: dict[str, float] = {
        "url_length": len(url),
        "url_hostname_length": len(host),
        "url_path_length": len(path),
        "url_query_length": len(query),
        "url_dot_count": url.count("."),
        "url_subdomain_count": len(subdomain.split(".")) if subdomain else 0,
        "url_hyphen_count": url.count("-"),
        "url_digit_count": digits,
        "url_digit_ratio": digits / len(url) if url else 0.0,
        "url_special_char_count": sum(not ch.isalnum() and ch not in _SPECIAL_EXEMPT for ch in url),
        "url_at_count": url.count("@"),
        "url_question_count": url.count("?"),
        "url_equals_count": url.count("="),
        "url_ampersand_count": url.count("&"),
        "url_percent_count": url.count("%"),
        "url_path_depth": len([seg for seg in path.split("/") if seg]),
        "url_query_param_count": len(parse_qsl(query, keep_blank_values=True)),
        "url_has_https": int(parts.scheme.lower() == "https"),
        "url_has_ip": int(is_ip),
        "url_has_punycode": int("xn--" in lower),
        "url_has_nonstandard_port": int(port is not None and port not in (80, 443)),
        "url_has_double_slash_path": int("//" in path),
        "url_entropy": shannon_entropy(url),
        "url_suspicious_token_count": len(tokens_found),
    }
    for t in TOKEN_FLAGS:
        feats[f"url_has_token_{t}"] = int(t in lower)
    return feats


def url_feature_frame(urls: pd.Series) -> pd.DataFrame:
    """Vectorised helper: one row of URL features per URL, columns in URL_FEATURES order."""
    return pd.DataFrame([extract_url_features(u) for u in urls], index=urls.index)[URL_FEATURES]


if __name__ == "__main__":
    # Sanity check on urls.csv: value ranges and per-class medians (descriptive only;
    # the feature list above was fixed before this summary was looked at).
    urls = pd.read_csv(config.URLS_CSV, keep_default_na=False)
    feats = url_feature_frame(urls["url"])
    assert list(feats.columns) == URL_FEATURES and not feats.isna().any().any()
    summary = feats.groupby(urls["label"].map(config.LABEL_NAMES)).median().T
    summary["min"], summary["max"] = feats.min(), feats.max()
    config.ensure_directories()
    summary.round(3).to_csv(config.TABLES_DIR / "url_feature_summary.csv")
    print(f"Extracted {len(URL_FEATURES)} URL features for {len(urls)} URLs, no missing values.\n")
    with pd.option_context("display.width", 160):
        print(summary.round(2).to_string())
    print("\nSaved: outputs/tables/url_feature_summary.csv")

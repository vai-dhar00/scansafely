"""
Day 4a: payload triage + E1 risk score + exact per-feature explanation (D25, D26).

POLICY (D25): E1 (URL-only) is the ONLY operational model. No QR-image-quality judgements, no E3.
Everything here is offline and treats the decoded payload as inert text: no network, DNS, browser.

Explanation method: E1 is a logistic regression on standardised (and for counts, log1p) features, so
    logit = intercept + sum_i coef_i * z_i          (z_i = transformed, standardised value)
Each term coef_i * z_i is an EXACT additive contribution relative to an "average training URL"
(z = 0). It equals SHAP LinearExplainer values (verified in tests). Contributions are also summed
into feature groups, because correlated features can share credit (e.g. the length features).
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
import warnings
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd

from src import config
from src.url_features import extract_url_features

MAX_URL_CHARS = 2000          # training data was cleaned to <= 2,000 characters (D3/D25)
MAX_DISPLAY_CHARS = 500       # payload display is truncated for the screen only
MIN_FACTOR_LOGIT = 0.05       # ignore contributions smaller than this in the plain-language list
TOP_FACTORS = 3

EXPLANATION_CAVEAT = "This is a model explanation, not proof that the destination is malicious."
CORRELATION_NOTE = ("Several URL features are correlated (for example, the length-related ones), so "
                    "credit can be shared between them. The group totals are the more stable summary.")
SCORE_NOTE = ("Risk score is a prototype model output, not the probability that a destination is malicious.")
NON_URL_MESSAGE = ("Decoded payload is not a web URL (http:// or https://), so URL risk analysis is "
                   "unavailable. It is shown below as plain text and has not been opened or run.")
TOO_LONG_MESSAGE = (f"Decoded payload is longer than {MAX_URL_CHARS} characters, outside the range this model "
                    "was trained on, so URL risk analysis is unavailable. It is shown as plain text.")
UNDECODABLE_MESSAGE = "Could not decode a QR payload from this image. Try a clearer, less distorted image."

# Neutral analysis-status states (review v6)
STATUS_COMPLETE, STATUS_FALLBACK = "ANALYSIS COMPLETE", "FALLBACK DECODER USED"
STATUS_UNDECODABLE, STATUS_NO_URL = "UNDECODABLE", "DECODED - URL ANALYSIS UNAVAILABLE"

# --------------------------------------------------------------------------- feature groups
FEATURE_GROUPS: dict[str, list[str]] = {
    "URL length and size": ["url_length", "url_hostname_length", "url_path_length", "url_query_length"],
    "Character mix": ["url_dot_count", "url_hyphen_count", "url_digit_count", "url_digit_ratio",
                      "url_special_char_count", "url_at_count", "url_percent_count", "url_entropy"],
    "URL structure": ["url_subdomain_count", "url_path_depth", "url_query_param_count", "url_question_count",
                      "url_equals_count", "url_ampersand_count", "url_has_double_slash_path"],
    "Connection and host type": ["url_has_https", "url_has_ip", "url_has_punycode", "url_has_nonstandard_port"],
    "Keywords": ["url_suspicious_token_count"] + [f"url_has_token_{t}" for t in
                                                  ("login", "verify", "account", "secure", "update", "bank")],
}
GROUP_OF = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}

# (plain label, value formatter). Wording describes what the MODEL does, never what a feature "means".
_N = lambda v: f"{v:.0f}"
_CH = lambda v: f"{v:.0f} character" + ("" if round(v) == 1 else "s")
_YN = lambda v: "yes" if v >= 0.5 else "no"
LABELS: dict[str, tuple[str, callable]] = {
    "url_length": ("Total URL length", _CH), "url_hostname_length": ("Domain name length", _CH),
    "url_path_length": ("Path length", _CH), "url_query_length": ("Query-string length", _CH),
    "url_dot_count": ("Number of dots", _N), "url_hyphen_count": ("Number of hyphens", _N),
    "url_digit_count": ("Number of digits", _N), "url_digit_ratio": ("Share of characters that are digits", lambda v: f"{100*v:.0f}%"),
    "url_special_char_count": ("Number of special characters", _N), "url_at_count": ("Number of '@' symbols", _N),
    "url_percent_count": ("Number of '%' encodings", _N), "url_entropy": ("Character randomness", lambda v: f"{v:.2f} bits/char"),
    "url_subdomain_count": ("Number of subdomain parts", _N), "url_path_depth": ("Path depth (folders)", _N),
    "url_query_param_count": ("Number of query parameters", _N), "url_question_count": ("Number of '?' symbols", _N),
    "url_equals_count": ("Number of '=' symbols", _N), "url_ampersand_count": ("Number of '&' symbols", _N),
    "url_has_double_slash_path": ("Double slash inside the path", _YN),
    "url_has_https": ("Uses https", _YN), "url_has_ip": ("Uses a raw IP address as the host", _YN),
    "url_has_punycode": ("Contains punycode (xn--)", _YN), "url_has_nonstandard_port": ("Uses a non-standard port", _YN),
    "url_suspicious_token_count": ("Number of watch-list words present", _N),
    **{f"url_has_token_{t}": (f"Contains the word '{t}'", _YN) for t in
       ("login", "verify", "account", "secure", "update", "bank")},
}
assert set(LABELS) == set(GROUP_OF), "every URL feature needs a label and a group"


# --------------------------------------------------------------------------- payload triage
def escape_for_display(payload: str, limit: int = MAX_DISPLAY_CHARS) -> str:
    """Make a payload safe to show: control, format (bidi / zero-width), unassigned and non-ASCII
    space characters are shown as \\uXXXX escapes; long text is truncated. Never a link."""
    out = []
    for ch in payload[:limit]:
        cat = unicodedata.category(ch)
        if cat[0] == "C" or cat in {"Zl", "Zp"} or (cat == "Zs" and ch != " "):
            out.append(f"\\u{ord(ch):04x}" if ord(ch) <= 0xFFFF else f"\\U{ord(ch):08x}")
        else:
            out.append(ch)
    text = "".join(out)
    return text + (f" ... [{len(payload) - limit} more characters not shown]" if len(payload) > limit else "")


def classify_payload(payload: str | None) -> str:
    """'undecodable' | 'url' | 'too_long' | 'non_url'. Conservative: only plain http(s) URLs are analysed."""
    if payload is None or payload == "":
        return "undecodable"
    if any(ch.isspace() for ch in payload) or any(unicodedata.category(c)[0] == "C" for c in payload):
        return "non_url"  # training URLs never contained whitespace (D3); do not guess
    if not payload.lower().startswith(("http://", "https://")):
        return "non_url"
    if len(payload) > MAX_URL_CHARS:
        return "too_long"
    try:
        if not urlsplit(payload).hostname:
            return "non_url"
    except ValueError:
        return "non_url"
    return "url"


# --------------------------------------------------------------------------- model
def load_e1(verify: bool = True) -> dict:
    """Load the locked E1 bundle; refuse to run if its SHA-256 differs from the training manifest."""
    path = config.MODELS_DIR / "E1_url_only.joblib"
    if verify:
        manifest = json.loads((config.OUTPUTS_DIR / "model_manifest.json").read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["models"]["E1_url_only"]["model_sha256"]:
            raise RuntimeError("E1 model file does not match its locked hash; refusing to run.")
    return joblib.load(path)


def transformed_parts(bundle: dict, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, float]:
    """Z (transformed values), coefficients, intercept - in the pipeline's own column order."""
    pipe = bundle["pipeline"]
    ct = pipe.named_steps["prep"]
    order = [c for name, _, cols in ct.transformers_ if name != "remainder" for c in cols]
    with warnings.catch_warnings():
        # Spurious on Apple Silicon (NumPy 2 + Accelerate); finiteness is asserted below instead.
        warnings.filterwarnings("ignore", message=r".*encountered in matmul", category=RuntimeWarning)
        Z = pd.DataFrame(ct.transform(X[bundle["features"]]), columns=order, index=X.index)
    clf = pipe.named_steps["clf"]
    coef = pd.Series(clf.coef_.ravel(), index=order)
    assert np.isfinite(Z.to_numpy()).all() and np.isfinite(coef.to_numpy()).all()
    return Z, coef, float(clf.intercept_[0])


@dataclass
class Explanation:
    score: float
    band: str
    recommendation: str
    logit: float
    intercept: float
    group_contributions: dict[str, float]
    raising: list[str] = field(default_factory=list)
    lowering: list[str] = field(default_factory=list)
    feature_contributions: dict[str, float] = field(default_factory=dict)


def _sentence(feat: str, raw: float, z: float, contrib: float) -> str:
    label, fmt = LABELS[feat]
    effect = "raises" if contrib > 0 else "lowers"
    if feat.startswith(("url_has_",)):
        return f"{label}: {fmt(raw)}. This {effect} the risk score."
    side = "above" if z > 0 else "below"
    return f"{label}: {fmt(raw)} ({side} the training average). This {effect} the risk score."


def build_factor_sentences(feats: dict[str, float], z: pd.Series, contrib: pd.Series,
                           found_words: list[str]) -> tuple[list[str], list[str]]:
    """Plain-language 'raising' / 'lowering' lists.

    Correlated features get offsetting weights (D24), so a single feature can point AGAINST the
    net direction of its group (e.g. one keyword flag 'lowers' while the keyword count 'raises').
    Showing those reads as a contradiction, so:
      - a feature sentence is shown only if it agrees in sign with its group's net contribution
        and the group's net effect is itself >= MIN_FACTOR_LOGIT;
      - the keyword group is summarised in ONE sentence (words found + group effect).
    Group totals are always shown separately by the app and are exact."""
    gnet: dict[str, float] = {}
    for f, c in contrib.items():
        gnet[GROUP_OF[f]] = gnet.get(GROUP_OF[f], 0.0) + float(c)

    items: list[tuple[float, str]] = []  # (signed contribution used for ranking, sentence)
    kw_net = gnet.get("Keywords", 0.0)
    if abs(kw_net) >= MIN_FACTOR_LOGIT:
        words = ("Watch-list words found in the URL text: " + ", ".join(f"'{t}'" for t in found_words)
                 if found_words else "No watch-list words found in the URL text")
        items.append((kw_net, f"{words}. Together, the keyword features "
                              f"{'raise' if kw_net > 0 else 'lower'} the risk score."))
    for f, c in contrib.items():
        g = GROUP_OF[f]
        if g == "Keywords" or abs(c) < MIN_FACTOR_LOGIT or abs(gnet[g]) < MIN_FACTOR_LOGIT:
            continue
        if np.sign(c) != np.sign(gnet[g]):
            continue
        items.append((float(c), _sentence(f, feats[f], float(z[f]), float(c))))
    raising = [t for c, t in sorted(items, key=lambda x: -x[0]) if c > 0][:TOP_FACTORS]
    lowering = [t for c, t in sorted(items, key=lambda x: x[0]) if c < 0][:TOP_FACTORS]
    return raising, lowering


def explain_url(url: str, bundle: dict) -> Explanation:
    """Score one http(s) URL with E1 and explain it. Caller must have checked classify_payload == 'url'."""
    feats = extract_url_features(url)
    X = pd.DataFrame([feats])
    Z, coef, intercept = transformed_parts(bundle, X)
    contrib = (Z.iloc[0] * coef)
    logit = intercept + float(contrib.sum())
    score = float(1.0 / (1.0 + np.exp(-logit)))
    assert np.isfinite(score) and 0.0 <= score <= 1.0

    groups = {g: 0.0 for g in FEATURE_GROUPS}
    for f, c in contrib.items():
        groups[GROUP_OF[f]] += float(c)
    # Same expression the feature extractor uses (substring match on the 15-word watch-list), so the words
    # we name are exactly the ones the model's keyword count saw.
    found_words = [t for t in config.SUSPICIOUS_TOKENS if t in url.lower()]
    assert len(found_words) == int(feats["url_suspicious_token_count"])
    raising, lowering = build_factor_sentences(feats, Z.iloc[0], contrib, found_words)

    band = config.probability_to_risk_label(score)
    return Explanation(score=score, band=band, recommendation=config.RISK_RECOMMENDATIONS[band], logit=logit,
                       intercept=intercept, group_contributions=groups, raising=raising, lowering=lowering,
                       feature_contributions={f: float(c) for f, c in contrib.items()})


def analyse_payload(payload: str | None, decoder_name: str | None, bundle: dict) -> dict:
    """Single entry point for the app: triage -> (maybe) E1 -> everything the UI may display."""
    kind = classify_payload(payload)
    base = {"kind": kind, "disclaimer": config.DISCLAIMER, "decoder_name": decoder_name}
    if kind == "undecodable":
        return {**base, "status": STATUS_UNDECODABLE, "message": UNDECODABLE_MESSAGE, "explanation": None}
    shown = escape_for_display(payload)
    if kind in ("non_url", "too_long"):
        return {**base, "status": STATUS_NO_URL, "payload_display": shown, "explanation": None,
                "message": NON_URL_MESSAGE if kind == "non_url" else TOO_LONG_MESSAGE}
    expl = explain_url(payload, bundle)
    fallback = bool(decoder_name) and "aruco" in decoder_name.lower()
    return {**base, "status": STATUS_FALLBACK if fallback else STATUS_COMPLETE, "payload_display": shown,
            "explanation": expl, "message": None, "caveat": EXPLANATION_CAVEAT,
            "correlation_note": CORRELATION_NOTE, "score_note": SCORE_NOTE}


if __name__ == "__main__":
    # Prints E1's locked coefficients (standardised space) so unintuitive signs can be inspected
    # and reported honestly. Descriptive only: nothing is retrained or changed.
    b = load_e1()
    Z, coef, icpt = transformed_parts(b, pd.DataFrame([extract_url_features("https://example.com/")]))
    tab = pd.DataFrame({"group": [GROUP_OF[f] for f in coef.index], "coef": coef.round(3)}, index=coef.index)
    print(f"intercept {icpt:.3f}\n")
    print(tab.sort_values(["group", "coef"]).to_string())
    print("\nGroup sums of |coef|:", tab.assign(a=tab.coef.abs()).groupby("group").a.sum().round(2).to_dict())

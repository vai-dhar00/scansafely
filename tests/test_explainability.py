import numpy as np
import pandas as pd
import pytest

from src import config
from src import explainability as ex
from src.train_models import build_pipeline
from src.url_features import URL_FEATURES, extract_url_features

pytestmark = pytest.mark.filterwarnings("ignore:.*encountered in matmul:RuntimeWarning")


def _synthetic_bundle(n=400, seed=0):
    rng = np.random.default_rng(seed)
    words = ["shop", "news", "login", "verify", "bank", "photo", "blog", "update", "secure", "home"]
    urls, y = [], []
    for i in range(n):
        mal = i % 2
        host = "-".join(rng.choice(words, 1 + 2 * mal)) + f"{rng.integers(0, 9999) if mal else ''}.example{i % 7}.com"
        path = "/".join(rng.choice(words, 1 + 3 * mal))
        urls.append(("http://" if mal and rng.random() < .6 else "https://") + host + "/" + path)
        y.append(mal)
    X = pd.DataFrame([extract_url_features(u) for u in urls])[URL_FEATURES]
    keep = [f for f in URL_FEATURES if X[f].nunique() > 1]
    pipe = build_pipeline(keep, 1.0).fit(X[keep], y)
    return {"name": "E1_url_only", "pipeline": pipe, "features": keep, "thresholds": {}}, X, urls


@pytest.fixture(scope="module")
def synth():
    return _synthetic_bundle()


def test_every_feature_has_label_and_group():
    assert set(ex.LABELS) == set(ex.GROUP_OF) == set(URL_FEATURES)


@pytest.mark.parametrize("payload,expected", [
    (None, "undecodable"), ("", "undecodable"),
    ("https://example.com/a?b=1", "url"), ("HTTP://Example.com", "url"),
    ("WIFI:S:home;T:WPA;P:secret;;", "non_url"), ("mailto:a@b.com", "non_url"),
    ("javascript:alert(1)", "non_url"), ("example.com/login", "non_url"),
    ("https://exa mple.com", "non_url"), ("https://example.com/‮evil", "non_url"),
    ("https://", "non_url"), ("https://" + "a" * 2000 + ".com", "too_long"),
])
def test_classify_payload(payload, expected):
    assert ex.classify_payload(payload) == expected


def test_escape_for_display_hides_controls_and_truncates():
    s = ex.escape_for_display("a‮b\x00c​d")
    assert "‮" not in s and "\x00" not in s and "​" not in s
    assert "\\u202e" in s and "\\u0000" in s
    assert "more characters not shown" in ex.escape_for_display("x" * 600)
    assert ex.escape_for_display("https://example.com/a") == "https://example.com/a"


def test_contributions_sum_to_logit_and_score(synth):
    bundle, _, _ = synth
    e = ex.explain_url("http://login-verify1234.example1.com/bank/update", bundle)
    assert abs(e.intercept + sum(e.feature_contributions.values()) - e.logit) < 1e-9
    assert abs(sum(e.group_contributions.values()) - sum(e.feature_contributions.values())) < 1e-9
    p = bundle["pipeline"].predict_proba(pd.DataFrame([extract_url_features("http://login-verify1234.example1.com/bank/update")])[bundle["features"]])[0, 1]
    assert abs(p - e.score) < 1e-9 and e.band == config.probability_to_risk_label(e.score)


def test_matches_shap_linear_explainer(synth):
    shap = pytest.importorskip("shap")
    bundle, X, _ = synth
    Z, coef, _ = ex.transformed_parts(bundle, X)
    # SHAP subsamples its background to 100 rows by default, which shifts the mean; use all rows.
    masker = shap.maskers.Independent(Z.to_numpy(), max_samples=len(Z))
    expl = shap.LinearExplainer((coef.to_numpy(), 0.0), masker)
    url = "http://verify-bank99.example3.com/login/update/x"
    z_row, _, _ = ex.transformed_parts(bundle, pd.DataFrame([extract_url_features(url)]))
    sv = expl.shap_values(z_row.to_numpy())[0]
    mine = (z_row.iloc[0] * coef).to_numpy()
    # Equal up to the (near-zero) background mean of standardised features.
    np.testing.assert_allclose(sv, mine, atol=1e-6)


def test_analyse_payload_outcomes(synth):
    bundle, _, _ = synth
    und = ex.analyse_payload(None, "none", bundle)
    assert und["status"] == ex.STATUS_UNDECODABLE and und["explanation"] is None and "band" not in und
    non = ex.analyse_payload("WIFI:S:x;;", "opencv", bundle)
    assert non["status"] == ex.STATUS_NO_URL and non["explanation"] is None and non["payload_display"] == "WIFI:S:x;;"
    ok = ex.analyse_payload("https://example.com/a", "opencv", bundle)
    assert ok["status"] == ex.STATUS_COMPLETE and ok["explanation"].band in config.RISK_RECOMMENDATIONS
    fb = ex.analyse_payload("https://example.com/a", "opencv_aruco", bundle)
    assert fb["status"] == ex.STATUS_FALLBACK
    assert ok["caveat"] == ex.EXPLANATION_CAVEAT and ok["disclaimer"] == config.DISCLAIMER


def test_no_network_or_browser_imports():
    import pathlib
    src = pathlib.Path(ex.__file__).read_text()
    for bad in ("import requests", "import urllib.request", "import webbrowser", "import socket", "import http"):
        assert bad not in src


def test_offsetting_feature_is_not_shown_against_its_group():
    contrib = pd.Series({f: 0.0 for f in URL_FEATURES})
    z = pd.Series({f: 1.0 for f in URL_FEATURES})
    feats = {f: 0.0 for f in URL_FEATURES}
    # keyword group: count raises strongly, one flag lowers -> ONE combined sentence, net raises
    contrib["url_suspicious_token_count"] = 1.5
    contrib["url_has_token_account"] = -0.4
    feats.update(url_suspicious_token_count=5, url_has_token_login=1, url_has_token_account=1)
    # character-mix group: net positive, special-char count opposes -> hidden
    contrib["url_digit_count"] = 1.0
    contrib["url_special_char_count"] = -0.3
    # group whose net effect is ~0 -> its features hidden
    contrib["url_path_depth"] = 0.5
    contrib["url_query_param_count"] = -0.48
    raising, lowering = ex.build_factor_sentences(feats, z, contrib)
    text = " ".join(raising + lowering)
    assert sum("Watch-list words" in r for r in raising) == 1 and "'account'" in text and not lowering
    assert "special characters" not in text and "digits" in text
    assert "Path depth" not in text and "query parameters" not in text

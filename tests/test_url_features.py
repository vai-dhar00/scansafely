"""URL feature tests. Run:  python -m pytest -q"""
import math

from src.url_features import URL_FEATURES, extract_url_features, shannon_entropy


def test_all_features_present_and_numeric():
    f = extract_url_features("https://example.com/")
    assert list(f) == URL_FEATURES
    assert all(isinstance(v, (int, float)) for v in f.values())


def test_known_values():
    f = extract_url_features("http://login.secure.example.co.uk:8080/a/b//c?x=1&y=2")
    assert f["url_subdomain_count"] == 2          # 'login.secure' (public suffix = co.uk)
    assert f["url_path_depth"] == 3
    assert f["url_query_param_count"] == 2
    assert f["url_has_https"] == 0
    assert f["url_has_nonstandard_port"] == 1
    assert f["url_has_double_slash_path"] == 1
    assert f["url_has_token_login"] == 1 and f["url_has_token_secure"] == 1
    assert f["url_suspicious_token_count"] == 2


def test_ip_and_punycode():
    assert extract_url_features("http://192.168.0.1/x")["url_has_ip"] == 1
    assert extract_url_features("http://192.168.0.1/x")["url_subdomain_count"] == 0
    assert extract_url_features("https://xn--80ak6aa92e.com")["url_has_punycode"] == 1


def test_entropy():
    assert shannon_entropy("aaaa") == 0.0
    assert math.isclose(shannon_entropy("ab"), 1.0)


def test_malformed_input_never_raises():
    for bad in ["", "not a url", "http://[::1", "http://host:abc/", "::::"]:
        f = extract_url_features(bad)
        assert list(f) == URL_FEATURES


def test_log1p_policy_covers_exactly_the_count_and_length_features():
    from src.url_features import URL_LOG1P_FEATURES
    not_logged = set(URL_FEATURES) - set(URL_LOG1P_FEATURES)
    assert set(URL_LOG1P_FEATURES) <= set(URL_FEATURES)
    # Everything not log-transformed must be a flag, the digit ratio, or entropy.
    assert all(f.startswith("url_has_") or f in ("url_digit_ratio", "url_entropy") for f in not_logged)

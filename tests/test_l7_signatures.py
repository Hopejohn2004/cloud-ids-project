"""Unit tests for the layer-7 signature overlay (src/l7_signatures.py)."""

from src.l7_signatures import (
    BRUTE_FORCE,
    XSS,
    OVERRIDE_CONFIDENCE,
    fuse,
    hints_from_payloads,
    hints_from_snippets,
)


def test_hints_detect_xss():
    h = hints_from_payloads([b"GET /?q=<script>alert(1)</script> HTTP/1.1"])
    assert h["xss"] is True
    assert h["brute_force"] is False


def test_hints_detect_svg_marker():
    h = hints_from_payloads([b"<svg onload=alert('x')></svg>"])
    assert h["xss"] is True


def test_hints_detect_brute_force():
    h = hints_from_payloads([b"POST /wp-login.php HTTP/1.1\r\nlog=admin&pwd=hunter2"])
    assert h["brute_force"] is True
    assert h["xss"] is False


def test_hints_ignore_plain_http():
    h = hints_from_payloads([b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n"])
    assert h == {"xss": False, "brute_force": False}


def test_hints_consider_multiple_snippets():
    h = hints_from_snippets([b"GET / HTTP/1.1", b"", b"<script>alert(1)</script>"])
    assert h["xss"] is True


# ── Fusion ────────────────────────────────────────────────────────────────────
def test_fuse_xss_promotes_benign():
    assert fuse("BENIGN", 99.9, {"xss": True}) == (XSS, OVERRIDE_CONFIDENCE)


def test_fuse_xss_promotes_uncertain():
    assert fuse("UNCERTAIN", 30.0, {"xss": True}) == (XSS, OVERRIDE_CONFIDENCE)


def test_fuse_xss_promotes_bruteforce_base():
    # When statistics look like brute force but the payload is clearly a script,
    # the L7 evidence wins (they share one statistical manifold).
    assert fuse(BRUTE_FORCE, 58.0, {"xss": True, "brute_force": True}) == (XSS, OVERRIDE_CONFIDENCE)


def test_fuse_bruteforce_promotes_benign():
    assert fuse("BENIGN", 70.0, {"brute_force": True}) == (BRUTE_FORCE, OVERRIDE_CONFIDENCE)


def test_fuse_keeps_other_statistical_classes():
    # A volumetric DDoS flow stays DDoS even if one request contains a marker —
    # the statistical story (huge packet volume) is the dominant evidence.
    assert fuse("DDoS", 93.0, {"xss": True, "brute_force": True}) == ("DDoS", 93.0)


def test_fuse_no_markers_unchanged():
    assert fuse("BENIGN", 99.0, {"xss": False, "brute_force": False}) == ("BENIGN", 99.0)


def test_fuse_rejects_non_dict_hints():
    assert fuse("DDoS", 90.0, None) == ("DDoS", 90.0)
    assert fuse("DDoS", 90.0, [1, 2, 3]) == ("DDoS", 90.0)
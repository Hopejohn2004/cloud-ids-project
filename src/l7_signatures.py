"""
Layer-7 signature overlay for the classification verdict.

The XGBoost model consumes 70 packet-statistical features. Several classes have
nearly identical single-flow statistics — most notably CIC's "Web Attack -
Brute Force" vs "Web Attack - XSS", which both look like a burst of small
HTTP exchanges. The model alone cannot reliably separate them, but the actual
payload bytes can. This module derives a small set of boolean "hints" from the
captured L4 payload and provides a CONSERVATIVE fusion rule:

  * an attack marker found in the payload can only RAISE a benign / ambiguous /
    benchtop-web verdict into the matching attack class;
  * it never downgrades a verdict, and it never overrides classes whose
    statistical story is genuinely different (e.g. DDoS is still DDoS even if
    one request in the flood happens to contain "alert(").

The capture agent computes the hints per flow from the packets it saw and posts
them as {"l7": {...}} alongside the feature vector; "app.py" calls fuse().

Class names match "models/label_encoder.pkl" exactly.
"""

from __future__ import annotations

BRUTE_FORCE = "Web Attack - Brute Force"
XSS = "Web Attack - XSS"

# Override confidence used when a payload marker is decisive. 72% is well above
# the 55% confidence gate (so the verdict is committed) while still honest about
# the fact this is a signature-based call, not a model call.
OVERRIDE_CONFIDENCE = 72.0

#: Case-insensitive byte markers looked for in the aggregated payload.
XSS_MARKERS = (
    b"<script", b"alert(", b"onerror=", b"onload=", b"onclick=",
    b"onmouseover", b"javascript:", b"document.cookie", b"<svg", b"<iframe",
    b"fromcharcode",
)
BRUTE_FORCE_MARKERS = (
    b"password=", b"username=", b"user_login", b"pwd=", b"wp-login",
    b"login.php", b"&user=", b"user=", b"pass=", b"log=",
)

_SNIPPET_MAX = 512          # bytes of payload per packet kept for inspection
_MAX_SNIPPETS = 64          # packet count cap when aggregating a flow


def hints_from_snippets(snippets):
    """Reduce an iterable of payload-byte snippets to {xss, brute_force} bools."""
    blob = b"".join((s or b"") for s in snippets)[:_MAX_SNIPPETS * _SNIPPET_MAX]
    blob_lower = blob.lower()
    return {
        "xss": any(m in blob_lower for m in XSS_MARKERS),
        "brute_force": any(m in blob_lower for m in BRUTE_FORCE_MARKERS),
    }


def hints_from_payloads(payload_list):
    """Same as hints_from_snippets but accepts raw payload bytes (truncates)."""
    return hints_from_snippets(p[: _SNIPPET_MAX] for p in (payload_list or []))


def fuse(base_attack, base_conf_pct, hints):
    """
    Conservative L7 overlay over a raw model verdict.

    Returns (attack_type, confidence_pct). The saved confidence is the boosted
    override value so the response engine logs an attributable number; when no
    marker fires (or the base class is not the web/benign manifold) the base
    verdict is returned untouched.
    """
    if not isinstance(hints, dict):
        return base_attack, base_conf_pct

    target = None
    if hints.get("xss"):
        target = XSS
    elif hints.get("brute_force"):
        target = BRUTE_FORCE

    if target is None:
        return base_attack, base_conf_pct

    if base_attack in (XSS, BRUTE_FORCE, "BENIGN", "UNCERTAIN"):
        return target, OVERRIDE_CONFIDENCE
    return base_attack, base_conf_pct
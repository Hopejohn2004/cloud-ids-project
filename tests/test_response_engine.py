"""
Unit tests for src.response_engine (in-memory fallback + store-backed paths).
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

from src.response_engine import ResponseEngine
from src.storage import SQLiteStore


def _temp_store():
    tmp = os.path.join(tempfile.mkdtemp(), "test_engine.db")
    return SQLiteStore(tmp)


# ── In-memory fallback (store=None) ──────────────────────────────────────────
class TestInMemoryEngine:
    def test_block_adds_ip_and_records_action(self):
        eng = ResponseEngine()
        rec = eng.handle("DDoS", "HIGH", "BLOCK", source_ip="1.2.3.4")

        assert rec["action"] == "BLOCK"
        assert rec["executed"] is True
        assert rec["command"] == "iptables -A INPUT -s 1.2.3.4 -j DROP"
        assert rec["blocked_ip"] == "1.2.3.4"
        assert rec["blocked_ips_total"] == 1

    def test_block_wildcard(self):
        eng = ResponseEngine()
        rec = eng.handle("DDoS", "HIGH", "BLOCK", source_ip=None)
        assert rec["blocked_ip"] is None
        assert rec["command"] is None
        assert eng.blocked_ips == set()

    def test_alert_increments_counts(self):
        eng = ResponseEngine()
        eng.handle("PortScan", "MEDIUM", "ALERT", source_ip="10.0.0.1")
        assert eng.counts["alert"] == 1
        assert eng.counts["block"] == 0
        assert eng.counts["allow"] == 0

    def test_allow_records_no_block(self):
        eng = ResponseEngine()
        rec = eng.handle("BENIGN", "NONE", "ALLOW", source_ip="10.0.0.2")
        assert rec["action"] == "ALLOW"
        assert rec["command"] is None
        assert eng.counts["allow"] == 1

    def test_snapshot_format(self):
        eng = ResponseEngine()
        snap = eng.snapshot()
        assert set(snap.keys()) == {"blocked_ips", "blocked_ip_count", "counts", "recent"}
        assert snap["blocked_ips"] == []
        assert snap["blocked_ip_count"] == 0
        assert isinstance(snap["counts"], dict)

    def test_action_log_capped_at_100(self):
        eng = ResponseEngine()
        for _ in range(120):
            eng.handle("DDoS", "HIGH", "BLOCK", source_ip="1.2.3.4")
        assert len(eng.action_log) == 100


# ── Store-backed engine ──────────────────────────────────────────────────────
class TestStoreBackedEngine:
    def test_block_persisted_across_instances(self):
        store = _temp_store()
        store.register_client("c1")
        eng1 = ResponseEngine(store=store, client_id="c1")
        eng1.handle("DDoS", "HIGH", "BLOCK", source_ip="5.5.5.5")

        # New engine instance on the SAME store sees the blocked IP
        eng2 = ResponseEngine(store=store, client_id="c1")
        snap = eng2.snapshot()
        assert "5.5.5.5" in snap["blocked_ips"]
        assert snap["blocked_ip_count"] == 1

    def test_counts_returned_from_store(self):
        store = _temp_store()
        store.register_client("c2")
        eng = ResponseEngine(store=store, client_id="c2")
        eng.handle("PortScan", "MEDIUM", "ALERT", source_ip="1.1.1.1")
        eng.handle("BENIGN", "NONE", "ALLOW", source_ip="2.2.2.2")
        counts = eng.snapshot()["counts"]
        assert counts["alert"] == 1
        assert counts["allow"] == 1
        assert counts["block"] == 0

    def test_recent_actions_limited_to_20(self):
        store = _temp_store()
        store.register_client("c3")
        eng = ResponseEngine(store=store, client_id="c3")
        for _ in range(30):
            eng.handle("PortScan", "MEDIUM", "ALERT", source_ip="1.1.1.1")
        assert len(eng.snapshot()["recent"]) == 20

    def test_separate_clients_do_not_bleed(self):
        store = _temp_store()
        store.register_client("ca")
        store.register_client("cb")
        ResponseEngine(store=store, client_id="ca").handle(
            "DDoS", "HIGH", "BLOCK", source_ip="9.9.9.9"
        )
        snap_a = ResponseEngine(store=store, client_id="ca").snapshot()
        snap_b = ResponseEngine(store=store, client_id="cb").snapshot()
        assert "9.9.9.9" in snap_a["blocked_ips"]
        assert snap_b["blocked_ips"] == []

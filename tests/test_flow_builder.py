"""
Unit tests for src.flow_builder — bidirectional grouping & expiry.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

from src.flow_builder import FlowAccumulator, make_key
from src.flow_features import FlowPacket


def test_make_key_is_direction_agnostic():
    assert make_key("A", 1, "B", 2, 6) == make_key("B", 2, "A", 1, 6)
    both = make_key("A", 1, "B", 2, 6)
    assert both == ("A", 1, "B", 2, 6)     # lexically smaller side first


def test_forward_direction_follows_first_packet():
    acc = FlowAccumulator()
    acc.add(0.0, "10.0.0.1", 12345, "93.184.216.34", 443, 6,
            ip_len=60, flags="S")
    acc.add(0.1, "93.184.216.34", 443, "10.0.0.1", 12345, 6,
            ip_len=60, flags="SA")

    flows = acc.flush()
    assert len(flows) == 1                      # both directions merged
    flow = flows[0]
    assert flow.forward_src == "10.0.0.1"
    assert flow.forward_dport == 443
    assert flow.forward_sport == 12345
    fwd, bwd = acc.split_directions(flow)
    assert [p.is_forward for p in flow.packets] == [True, False]
    assert len(fwd) == 1
    assert len(bwd) == 1


def test_forward_direction_follows_the_lexically_larger_initiator():
    # The attacker (198.51.100.1) is lexically greater than the target
    # (10.0.0.6), so make_key would sort it second — the direction logic must
    # still pick the true FIRST-SEEN packet as forward.
    acc = FlowAccumulator()
    acc.add(0.0, "198.51.100.1", 40000, "10.0.0.6", 443, 6, ip_len=60, flags="S")
    acc.add(0.01, "10.0.0.6", 443, "198.51.100.1", 40000, 6, ip_len=60, flags="SA")

    flow = acc.flush()[0]
    assert flow.forward_src == "198.51.100.1"
    assert flow.forward_sport == 40000
    assert flow.forward_dport == 443                       # the attacker's target


def test_flow_expires_after_idle_timeout():
    acc = FlowAccumulator(idle_timeout=5.0)
    acc.add(1000.0, "A", 1, "B", 2, 6, ip_len=60)

    # a new packet at a later time from a DIFFERENT flow triggers the sweep
    expired = acc.add(1010.0, "C", 3, "D", 4, 6, ip_len=60)
    assert len(expired) == 1
    assert expired[0].key[0] == "A"

    # the second flow is still active (recent) until its own timeout passes
    assert acc.active_count() == 1
    expired2 = acc.expire(now=1020.0)
    assert len(expired2) == 1
    assert expired2[0].key[0] == "C"
    assert acc.active_count() == 0


def test_flush_returns_everything():
    acc = FlowAccumulator()
    acc.add(0.0, "A", 1, "B", 2, 6, ip_len=60)
    acc.add(0.1, "X", 9, "Y", 8, 17, ip_len=60)
    assert len(acc.flush()) == 2
    assert acc.active_count() == 0


def test_packet_stats_round_trip():
    acc = FlowAccumulator()
    acc.add(0.0, "A", 1, "B", 2, 6, ip_len=100, l4_header_len=20,
            tcp_window=3000, payload_len=50, flags="S")
    flow = acc.flush()[0]
    p = flow.packets[0]
    assert isinstance(p, FlowPacket)
    assert (p.ip_len, p.tcp_window, p.payload_len, p.flags) == (100, 3000, 50, "S")
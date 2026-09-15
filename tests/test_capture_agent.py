"""
Regression tests for the live capture agent's packet extraction.  These cover
the IPv6 field differences that used to crash the live sniffer (IPv6 uses
`nh`/`plen`, not `proto`/`len`).  Packets are round-tripped through bytes so
they carry the fields Scapy sets on real captures.
"""

import pytest
from scapy.packet import raw


@pytest.fixture(scope="module")
def extract_fn():
    from src.capture_agent import extract
    return extract


def test_ipv4_tcp_packet(extract_fn):
    from scapy.layers.inet import IP, TCP

    pkt = IP(raw(IP(src="1.2.3.4", dst="5.6.7.8") /
                 TCP(sport=12345, dport=443, window=4096)))
    pkt.time = 123.0
    f = extract_fn(pkt)
    assert f["src_ip"] == "1.2.3.4"
    assert f["dst_ip"] == "5.6.7.8"
    assert f["protocol"] == 6
    assert f["src_port"] == 12345
    assert f["dst_port"] == 443
    assert f["tcp_window"] == 4096
    assert f["l4_header_len"] == 20
    assert f["ip_len"] > 0


def test_ipv6_packet_does_not_crash(extract_fn):
    from scapy.layers.inet import UDP
    from scapy.layers.inet6 import IPv6

    pkt = IPv6(raw(IPv6(src="2001:db8::1", dst="2001:db8::2") /
                   UDP(sport=5353, dport=5353)))
    pkt.time = 123.0
    f = extract_fn(pkt)
    assert f["src_ip"] == "2001:db8::1"
    assert f["dst_ip"] == "2001:db8::2"
    assert f["protocol"] == 17
    assert f["src_port"] == 5353
    assert f["ip_len"] == 40 + pkt.plen         # fixed header + payload length


def test_non_ip_packet_skipped(extract_fn):
    from scapy.layers.l2 import ARP

    assert extract_fn(ARP(psrc="1.2.3.4", pdst="5.6.7.8")) is None


def test_multicast_dst_skipped(extract_fn):
    """Multicast/broadcast SSDP/mDNS/LAN chatter must not be classified."""
    from scapy.layers.inet import IP, TCP
    from scapy.packet import raw

    # IPv4 multicast
    pkt = IP(raw(IP(src="192.168.1.100", dst="224.0.0.251") /
                 TCP(sport=5353, dport=5353, window=1024)))
    pkt.time = 100.0
    assert extract_fn(pkt) is None
    # IPv6 multicast
    from scapy.layers.inet6 import IPv6
    pkt6 = IPv6(raw(IPv6(src="fe80::1", dst="ff02::fb") /
                    TCP(sport=5353, dport=5353, window=1024)))
    pkt6.time = 100.0
    assert extract_fn(pkt6) is None


def test_min_pkts_skips_short_flow():
    from src.capture_agent import FlowReporter

    class FakeFlow:
        packets = [b"x"]               # only 1 packet < min_pkts
        forward_dport = 80

    class FakeBuilder:
        def split_directions(self, f):
            raise AssertionError("split_directions should not be called on a short flow")
        def add(self, **kw):
            return []

    reporter = FlowReporter(FakeBuilder(), api="http://x", client_id="test",
                            min_pkts=2, dry_run=True)
    reporter.on_flow(FakeFlow())
    assert reporter.total_flows == 0                     # skipped


def test_malformed_l4_ignored_in_live_ingest():
    """_ingest must never propagate an exception out of sniff()."""
    from src.capture_agent import _ingest
    from scapy.layers.inet6 import IPv6

    class FakeBuilder:
        def __init__(self):
            self.added = []

        def add(self, **fields):
            self.added.append(fields)
            return []

    pkt = IPv6(src="2001:db8::1", dst="2001:db8::2")     # L4-less IPv6 is fine
    pkt.time = 123.0
    builder = FakeBuilder()
    _ingest(builder, None, pkt)
    assert len(builder.added) == 1
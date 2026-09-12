"""
Capture agent — turns REAL packet traffic into model predictions.

Reads a PCAP/PCAPNG file (or sniffs a live interface), groups packets into
bidirectional flows, computes the model's 70 CIC-style features per completed
flow, and submits each flow to the /predict API — which runs the real model and
feeds the dashboard's live console, stats, and blocked-source ledger.

    # offline: replay a captured file into the API
    python src/capture_agent.py --pcap capture.pcap --api http://127.0.0.1:5000

    # live: sniff the default interface
    python src/capture_agent.py --api http://127.0.0.1:5000

    # inspect: print feature vectors without calling the API
    python src/capture_agent.py --pcap capture.pcap --dry-run

Detection latency equals the flow idle timeout: a flow is classified after it
goes quiet (default 60s, tune with --idle). Real packet fields (source/dest IP,
ports) are forwarded so the response engine works on genuine attacker IPs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

# Allow `python src/capture_agent.py` to import the sibling src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.flow_builder import FlowAccumulator
from src.flow_features import derive_features

try:
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.layers.inet6 import IPv6
    from scapy.sendrecv import sniff
    from scapy.utils import PcapReader
    import requests
    SCAPY_OK = True
except ImportError:
    SCAPY_OK = False


def extract(packet):
    """Reduce one scapy packet to the builder's plain fields (None if skipped)."""
    if IP in packet:
        ip = packet[IP]
    elif IPv6 in packet:
        ip = packet[IPv6]          # IPv6 supported; protocol features match L4 only
    else:
        return None

    src, dst, proto = ip.src, ip.dst, ip.proto
    sport = dport = 0
    l4_header_len = tcp_window = payload_len = 0
    flags = ""

    if TCP in packet:
        tcp = packet[TCP]
        sport, dport = tcp.sport, tcp.dport
        l4_header_len = tcp.dataofs * 4
        tcp_window = tcp.window
        payload_len = len(bytes(tcp.payload))
        flags = str(tcp.flags).upper()
    elif UDP in packet:
        udp = packet[UDP]
        sport, dport = udp.sport, udp.dport
        l4_header_len = 8
        payload_len = len(bytes(udp.payload))

    return dict(
        timestamp=float(packet.time),
        src_ip=src,
        src_port=int(sport),
        dst_ip=dst,
        dst_port=int(dport),
        protocol=int(proto),
        ip_len=int(ip.len),
        ip_header_len=int((ip.ihl if IP in packet else 40) * 4),
        l4_header_len=l4_header_len,
        tcp_window=tcp_window,
        payload_len=payload_len,
        flags=flags,
    )


class FlowReporter:
    """Converts expired flows to feature vectors and posts them to the API."""

    def __init__(self, builder, api, client_id, dry_run=False, out=None,
                 timeout=10):
        self.builder = builder
        self.api = api.rstrip("/")
        self.client_id = client_id
        self.dry_run = dry_run
        self.out = out
        self.timeout = timeout
        self.total_flows = 0

    def on_flow(self, flow):
        fwd, bwd = self.builder.split_directions(flow)
        features = derive_features(fwd, bwd, destination_port=flow.forward_dport)
        meta = {
            "source_ip": flow.forward_src,
            "destination_ip": self._peer(flow),
            "source_port": flow.forward_sport,
            "destination_port": flow.forward_dport,
        }
        self.total_flows += 1
        self._emit(features, meta, flow)

    def _emit(self, features, meta, flow):
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(flow.first_seen))
        line = (f"[{stamp}] flow {flow.forward_src}:{flow.forward_sport} -> "
                f"{meta['destination_ip']}:{flow.forward_dport} "
                f"({len(flow.packets)} pkts)")

        if self.out:
            with open(self.out, "a", encoding="utf-8") as f:
                f.write(json.dumps({"meta": meta, "features": features}) + "\n")
        if self.dry_run:
            print(line + "  [DRY-RUN, no API call]")
            return

        resp = requests.post(
            f"{self.api}/predict",
            json={**features, **meta},
            headers={"X-Client-Id": self.client_id},
            timeout=self.timeout,
        )
        if resp.status_code == 200:
            d = resp.json()
            print(f"{line}  ->  {d['attack_type']} ({d['confidence']}) {d['action']}")
        elif resp.status_code == 429:
            print(f"{line}  ->  RATE LIMITED ({resp.json().get('error', '')})")
        else:
            print(f"{line}  ->  ERROR {resp.status_code}: {resp.text[:200]}")

    @staticmethod
    def _peer(flow):
        """The flow endpoint that is NOT the initiator."""
        if flow.forward_src == flow.key[0]:
            return flow.key[2]
        return flow.key[0]


def _run_pcap(reporter, path):
    builder = reporter.builder
    with PcapReader(path) as reader:
        for pkt in reader:
            fields = extract(pkt)
            if not fields:
                continue
            for flow in builder.add(**fields):
                reporter.on_flow(flow)
    for flow in builder.flush():
        reporter.on_flow(flow)
    print(f"\nTotal flows classified: {reporter.total_flows}")


def _run_live(reporter, iface):
    builder = reporter.builder
    stop = threading.Event()

    def sweeper():
        while not stop.is_set():
            now = time.time()
            for flow in builder.expire(now):
                reporter.on_flow(flow)
            stop.wait(max(1.0, builder.idle_timeout / 2))

    t = threading.Thread(target=sweeper, daemon=True)
    t.start()
    print(f"Sniffing {iface or '(default)'} — press Ctrl+C to stop.")
    try:
        sniff(iface=iface, prn=lambda pkt: _ingest(builder, reporter, pkt),
              store=False)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for flow in builder.flush():
            reporter.on_flow(flow)


def _ingest(builder, reporter, pkt):
    fields = extract(pkt)
    if not fields:
        return
    for flow in builder.add(**fields):
        reporter.on_flow(flow)


def main(argv=None):
    if not SCAPY_OK:
        print("scapy and requests are required:  pip install scapy requests",
              file=sys.stderr)
        return 1

    p = argparse.ArgumentParser(description="Cloud IDS capture agent")
    p.add_argument("--pcap", help="PCAP/PCAPNG file to replay (default: live capture)")
    p.add_argument("--iface", default=None, help="interface to sniff for live mode")
    p.add_argument("--api", default=os.environ.get("IDS_API", "http://127.0.0.1:5000"),
                   help="base URL of the Flask API (default http://127.0.0.1:5000)")
    p.add_argument("--client", default=os.environ.get("IDS_SENSOR_ID", "IDS-SENSOR"),
                   help="X-Client-Id used for this sensor's dashboard feed")
    p.add_argument("--idle", type=float, default=60.0,
                   help="flow idle timeout in seconds (classification latency)")
    p.add_argument("--dry-run", action="store_true",
                   help="compute features but do not call the API")
    p.add_argument("--out", help="append each flow's JSON to this file")
    args = p.parse_args(argv)

    builder = FlowAccumulator(idle_timeout=args.idle)
    reporter = FlowReporter(builder, api=args.api, client_id=args.client,
                            dry_run=args.dry_run, out=args.out)

    if args.pcap:
        _run_pcap(reporter, args.pcap)
    else:
        _run_live(reporter, args.iface)
    return 0


if __name__ == "__main__":
    sys.exit(main())
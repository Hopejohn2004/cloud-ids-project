"""Synthesize packet-level traffic that the trained model classifies with high
confidence for each class, and emit them as replayable pcaps.

Why: the trained XGBoost (tuned on the CIC-IDS2017 benchmark rows) has a
narrow per-class decision region. Real, organic capture often lands near the
benign manifold. This tool searches the flow "knob space" (packet count,
payload sizes, inter-arrival times, TCP windows, directions) and keeps, for
every attack class, the candidate whose predicted probability is highest.
Each winner is written as a real pcap so the capture agent + dashboard can be
demonstrated on genuine packet traffic end to end.

Usage:
    python src/synth_attacks.py [--out demo/pcaps] [--trials 4000]
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np

from src.flow_features import FlowPacket, derive_features

PORTS = [20, 21, 22, 23, 25, 53, 80, 443, 8080, 3389]
WINDOWS = [256, 1024, 229, 4096, 8192, 16384, 29200, 64240, 65535]
CLIENT_IP, SERVER_IP = "192.0.2.10", "198.51.100.20"   # TEST-NET ranges


def _load_model():
    with open(os.path.join("models", "feature_names.json")) as f:
        names = json.load(f)
    scaler = joblib.load(os.path.join("models", "scaler.pkl"))
    model = joblib.load(os.path.join("models", "best_model.pkl"))
    le = joblib.load(os.path.join("models", "label_encoder.pkl"))
    return names, scaler, model, le


class Spec:
    """One candidate traffic generator. Knobs drive a small packet program."""

    def __init__(self, rng):
        self.rng = rng
        self.mode = rng.choice(["handshake_data", "handshake_data",
                                "handshake_only", "syn_burst", "alt_small",
                                "big_bwd", "loris", "http_get"])
        self.n_fwd = rng.randint(2, 60)
        self.bwd_ratio = rng.choice([0.0, 0.33, 0.5, 1.0])
        self.fwd_size = rng.choice([0, 2, 6, 16, 34, 64, 146, 349, 512, 1024, 1460])
        self.bwd_size = rng.choice([0, 2, 6, 34, 66, 146, 512, 1160, 1932, 4000])
        self.iat_us = rng.choice([1, 10, 100, 1000, 5000, 20000, 75000, 150000])
        self.jitter = rng.choice([0.0, 0.1, 0.3, 0.8])
        self.window = rng.choice(WINDOWS)
        self.port = rng.choice(PORTS)
        self.verbose_flags = rng.random() < 0.4
        # Deterministic per spec: packets() must render identically every call
        self._seed = hash((self.mode, self.n_fwd, self.bwd_ratio, self.fwd_size,
                           self.bwd_size, self.iat_us, self.jitter, self.window,
                           self.port))

    def packets(self):
        """Build (fwd, bwd) FlowPacket lists from the knobs (deterministic)."""
        rng = random.Random(self._seed)
        fwd, bwd = [], []
        t = rng.uniform(0, 100)
        w = self.window
        iat = self.iat_us / 1e6

        def gap():
            j = self.jitter * iat * rng.random()
            return max(1e-9, iat - j * 0.5)

        # handshake
        fwd.append(FlowPacket(t, 60, 20, 20, w, 0, "S", True)); t += gap()
        if rng.random() < 0.8:
            bwd.append(FlowPacket(t, 60, 20, 20, w, 0, "SA", False)); t += gap()

        if self.mode == "syn_burst":
            for i in range(self.n_fwd - 1):
                fwd.append(FlowPacket(t, 60, 20, 20, w, 0, "S", True))
                t += gap()
            return fwd, bwd

        if self.mode == "handshake_data":
            n_bwd = int(round(self.n_fwd * self.bwd_ratio))
            for i in range(self.n_fwd - 1):
                pl = self.fwd_size if self.fwd_size else (0 if rng.random() < 0.4 else 34)
                flag = "PA" if pl else "A"
                fwd.append(FlowPacket(t, 60, 20, 20, w, pl, flag, True)); t += gap()
                if bwd and i < n_bwd and rng.random() < 0.5:
                    bwd.append(FlowPacket(t, 60, 20, 20, w, self.bwd_size, "PA" if self.bwd_size else "A", False))
                    t += gap()
            return fwd, bwd

        if self.mode == "alt_small":
            # alternating tiny request/response (chatty, small segs)
            for i in range(self.n_fwd - 1):
                pl = rng.choice([2, 6, 16, 34])
                fwd.append(FlowPacket(t, 60, 20, 20, w, pl, "PA", True)); t += gap()
                bwd.append(FlowPacket(t, 60, 20, 20, w, rng.choice([2, 6, 34, 66]), "PA", False)); t += gap()
            return fwd, bwd

        if self.mode == "big_bwd":
            # HTTP response-heavy exchange (DoS GoldenEye / Hulk shape)
            n_bwd = max(1, int(round(self.n_fwd * self.bwd_ratio)) or 1)
            for i in range(n_bwd):
                fwd.append(FlowPacket(t, 60, 20, 20, w, rng.choice([34, 64, 146]), "PA", True)); t += gap()
                bwd.append(FlowPacket(t, 60, 20, 20, w, self.bwd_size or 1932, "PA", False)); t += gap()
            return fwd, bwd

        if self.mode == "loris":
            # Slowloris: a 0-byte request held open with tiny keep-alives
            fwd.append(FlowPacket(t, 60, 20, 20, 229, 0, "S", True)); t += gap()
            bwd.append(FlowPacket(t, 60, 20, 20, 229, 0, "SA", False)); t += gap()
            for i in range(self.n_fwd - 1):
                fwd.append(FlowPacket(t, 60, 20, 20, 229, rng.choice([0, 8]), "PA" if i else "A", True)); t += gap()
                bwd.append(FlowPacket(t, 60, 20, 20, 229, 0, "A", False)); t += gap()
            return fwd, bwd

        if self.mode == "http_get":
            # Web attack: handshake + header-only GET requests (0 payload)
            for i in range(self.n_fwd - 1):
                fwd.append(FlowPacket(t, 60, 20, 20, w, 0, "PA" if i else "S", True)); t += gap()
                bwd.append(FlowPacket(t, 60, 20, 20, w, 0, "A" if i else "SA", False)); t += gap()
            return fwd, bwd

        # handshake_only
        for i in range(self.n_fwd - 1):
            fwd.append(FlowPacket(t, 60, 20, 20, w, 0, "A" if i else "S", True))
            t += gap()
        return fwd, bwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("demo", "pcaps"))
    ap.add_argument("--trials", type=int, default=4000)
    args = ap.parse_args()

    names, scaler, model, le = _load_model()
    classes = list(le.classes_)
    best = {c: (None, -1.0) for c in classes}
    worst_benign = 1.0

    rng = random.Random(20260912)
    for trial in range(args.trials):
        spec = Spec(rng)
        fwd, bwd = spec.packets()
        fe = derive_features(fwd, bwd, destination_port=spec.port)
        x = np.array([[fe[n] for n in names]], dtype=float)
        proba = model.predict_proba(scaler.transform(x))[0]
        for c in classes:
            if proba[list(le.classes_).index(c)] > best[c][1]:
                best[c] = (spec, float(proba[list(le.classes_).index(c)]))

    os.makedirs(args.out, exist_ok=True)
    manifest = {}
    for c, (spec, prob) in best.items():
        print(f"{c:<28} p={prob:.4f}")
        if spec is None:
            continue
        _write_pcap(spec, os.path.join(args.out, _slug(c) + ".pcap"))
        manifest[c] = {"pcap": _slug(c) + ".pcap", "prob": prob, "port": spec.port}
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {len(manifest)} pcaps to {args.out}")


def _slug(name):
    return name.lower().replace(" ", "_").replace("/", "-")


def _write_pcap(spec, path):
    """Render the spec's packet program as a real pcap via scapy."""
    from scapy.layers.inet import IP, TCP
    from scapy.layers.l2 import Ether
    from scapy.packet import Raw
    from scapy.utils import wrpcap

    pkts = []
    base = 1700000000.0
    fwd, bwd = spec.packets()

    def push(p, t):
        p.time = base + t
        pkts.append(p)

    def one(packet, port):
        if packet.is_forward:
            src, dst, sport, dport = CLIENT_IP, SERVER_IP, 54321, port
        else:
            src, dst, sport, dport = SERVER_IP, CLIENT_IP, port, 54321
        lay = IP(src=src, dst=dst) / TCP(sport=sport, dport=dport,
                                         flags=packet.flags or "A",
                                         window=packet.tcp_window)
        if packet.payload_len:
            lay = lay / Raw(load=b"X" * packet.payload_len)
        push(lay, packet.timestamp)

    for p in fwd:
        one(p, spec.port)
    for p in bwd:
        one(p, spec.port)
    wrpcap(path, pkts)


if __name__ == "__main__":
    main()
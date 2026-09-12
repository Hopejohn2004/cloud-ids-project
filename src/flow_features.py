"""
CIC-style flow feature extraction.

Turns the raw packets of one bidirectional network flow into the exact 70
features the trained model expects (see models/feature_names.json). The math
mirrors CICFlowMeter's output as closely as is practical from raw packets:

  - forward = direction of the FIRST packet seen on the flow (the initiator)
  - lengths/segments/subflows use the L4 payload bytes (CIC convention;
    IP total length is NOT used)
  - header lengths use the L4 (TCP/UDP) header bytes including options
  - IATs and active/idle bursts are in microseconds; durations in µs too
  - means/stds are population statistics (ddof=0)
  - Down/Up Ratio is the rounded bwd→fwd packet-count ratio, and flag counts
    follow the CIC-IDS2017 publishing artifact (only SYN Flag Count is real;
    see CIC2017_FLAG_QUIRK below)


Designed to be dependency-light and unit-testable: the per-packet data is
passed in as simple FlowPacket dataclasses (no scapy import here).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean

# Idle/active split threshold (seconds). Gaps larger than this are "idle".
# Configurable for experimentation; only affects the 6 Active/Idle features.
ACTIVE_IDLE_T = 2.0

# Number of packets captured per direction into the "Subflow" features.
SUBFLOW_COUNT = 4

# CIC-IDS2017 quirk: the published CSV's FIN/RST/PSH/ACK/URG/CWE/ECE flag-count
# columns are ~always 0 (a capture/parsing artifact) even for data-heavy flows;
# only "SYN Flag Count" is populated. The model was trained on those values, so
# an extractor that counts real flags produces extreme outliers (z > 1000) and
# corrupts predictions. We mirror the dataset's convention; set to False to
# emit true packet-level counts instead.
CIC2017_FLAG_QUIRK = True


@dataclass
class FlowPacket:
    """One packet of a flow, reduced to the fields the features need."""
    timestamp: float          # seconds (epoch, fine)
    ip_len: int = 0           # total L3 (IP) packet length in bytes
    ip_header_len: int = 0    # L3 header bytes
    l4_header_len: int = 0    # L4 (TCP/UDP) header bytes
    tcp_window: int = 0       # raw TCP window size (0 for non-TCP)
    payload_len: int = 0      # bytes of L4 payload
    flags: str = ""           # TCP flag letters e.g. "FS" (S,SYN A,ACK F,FIN ...)
    is_forward: bool = True   # True if packet belongs to flow initiator


# ── The exact model feature order (models/feature_names.json) ────────────────
_TEMPLATE = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s",
    "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Fwd URG Flags", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size",
    "Avg Bwd Segment Size", "Fwd Header Length.1", "Subflow Fwd Packets",
    "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd",
    "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]


def derive_features(fwd_packets, bwd_packets, destination_port):
    """
    Compute the 70 model features for one flow.

    Args:
        fwd_packets: FlowPacket list in the initiator->responder direction
                     (arrival order).
        bwd_packets: FlowPacket list in the responder->initiator direction.
        destination_port: numeric destination port (from the forward direction).

    Returns a dict with exactly the 70 keys/order of feature_names.json.
    """
    fwd = list(fwd_packets)
    bwd = list(bwd_packets)
    n_fwd, n_bwd = len(fwd), len(bwd)
    n_all = n_fwd + n_bwd

    # Re-export via helper so math is kept in one place
    all_pkts = sorted(fwd + bwd, key=lambda p: p.timestamp)
    fwd_lens = [p.payload_len for p in fwd]
    bwd_lens = [p.payload_len for p in bwd]
    all_lens = [p.payload_len for p in all_pkts]

    # ── Timing ────────────────────────────────────────────────────────────
    if n_all == 0:
        duration_s = 0.0
    else:
        duration_s = max(0.0, all_pkts[-1].timestamp - all_pkts[0].timestamp)
    duration_us = duration_s * 1e6

    fwd_iat  = [d * 1e6 for d in _iat_list(fwd)]
    bwd_iat  = [d * 1e6 for d in _iat_list(bwd)]
    all_iat  = [d * 1e6 for d in _iat_list(all_pkts)]

    # ── Length stats ──────────────────────────────────────────────────────
    fwd_max  = _max0(fwd_lens)
    fwd_min  = _min0(fwd_lens)
    fwd_mean = _mean0(fwd_lens)
    fwd_std  = _std0(fwd_lens)
    bwd_max  = _max0(bwd_lens)
    bwd_min  = _min0(bwd_lens)
    bwd_mean = _mean0(bwd_lens)
    bwd_std  = _std0(bwd_lens)
    min_len  = _min0(all_lens)
    max_len  = _max0(all_lens)
    mean_len = _mean0(all_lens)
    std_len  = _std0(all_lens)
    var_len  = std_len * std_len

    # ── Bytes & flows-per-second ──────────────────────────────────────────
    total_fwd_bytes = sum(fwd_lens)
    total_bwd_bytes = sum(bwd_lens)
    total_bytes = total_fwd_bytes + total_bwd_bytes
    flow_bytes_s = total_bytes / duration_s if duration_s > 0 else 0.0
    flow_pkts_s  = n_all / duration_s if duration_s > 0 else 0.0
    fwd_pkts_s   = n_fwd / duration_s if duration_s > 0 else 0.0
    bwd_pkts_s   = n_bwd / duration_s if duration_s > 0 else 0.0

    # ── Directional byte counts + segment sizes (CIC payload semantics) ───
    down_up_ratio = math.floor(n_bwd / n_fwd) if n_fwd > 0 else 0.0
    avg_pkt_size  = total_bytes / n_all if n_all > 0 else 0.0
    avg_fwd_seg   = total_fwd_bytes / n_fwd if n_fwd > 0 else 0.0
    avg_bwd_seg   = total_bwd_bytes / n_bwd if n_bwd > 0 else 0.0

    # ── Header lengths ────────────────────────────────────────────────────
    fwd_header_len = sum(p.l4_header_len for p in fwd)
    bwd_header_len = sum(p.l4_header_len for p in bwd)

    # ── Flag counts ───────────────────────────────────────────────────────
    fwd_flags  = "".join(p.flags for p in fwd)
    bwd_flags  = "".join(p.flags for p in bwd)
    all_flags  = fwd_flags + bwd_flags

    fin, syn, rst = _count(all_flags, "F"), _count(all_flags, "S"), _count(all_flags, "R")
    psh = _count(all_flags, "P")
    ack = _count(all_flags, "A")
    urg = _count(all_flags, "U")
    cwe = _count(all_flags, "C")   # CWR
    ece = _count(all_flags, "E")

    if CIC2017_FLAG_QUIRK:
        fin = rst = psh = ack = urg = cwe = ece = 0

    # ── TCP window / payload markers ──────────────────────────────────────
    fwd_win  = next((p.tcp_window for p in fwd if p.tcp_window > 0), 0)
    bwd_win  = next((p.tcp_window for p in bwd if p.tcp_window > 0), 0)
    act_data_fwd = sum(1 for p in fwd if p.payload_len > 0)
    min_seg_fwd = min((p.l4_header_len for p in fwd), default=0)

    # ── Subflows (first SUBFLOW_COUNT packets each direction) ─────────────
    sfwd = fwd[:SUBFLOW_COUNT]
    sbwd = bwd[:SUBFLOW_COUNT]

    # ── Active / idle periods ─────────────────────────────────────────────
    active, idle = _active_idle_stats(all_pkts)

    out = {
        "Destination Port": float(destination_port),
        "Flow Duration": duration_us,
        "Total Fwd Packets": float(n_fwd),
        "Total Backward Packets": float(n_bwd),
        "Total Length of Fwd Packets": float(total_fwd_bytes),
        "Total Length of Bwd Packets": float(total_bwd_bytes),
        "Fwd Packet Length Max": fwd_max,
        "Fwd Packet Length Min": fwd_min,
        "Fwd Packet Length Mean": fwd_mean,
        "Fwd Packet Length Std": fwd_std,
        "Bwd Packet Length Max": bwd_max,
        "Bwd Packet Length Min": bwd_min,
        "Bwd Packet Length Mean": bwd_mean,
        "Bwd Packet Length Std": bwd_std,
        "Flow Bytes/s": flow_bytes_s,
        "Flow Packets/s": flow_pkts_s,
        "Flow IAT Mean": _mean0(all_iat),
        "Flow IAT Std": _std0(all_iat),
        "Flow IAT Max": _max0(all_iat),
        "Flow IAT Min": _min0(all_iat),
        "Fwd IAT Total": float(sum(fwd_iat)),
        "Fwd IAT Mean": _mean0(fwd_iat),
        "Fwd IAT Std": _std0(fwd_iat),
        "Fwd IAT Max": _max0(fwd_iat),
        "Fwd IAT Min": _min0(fwd_iat),
        "Bwd IAT Total": float(sum(bwd_iat)),
        "Bwd IAT Mean": _mean0(bwd_iat),
        "Bwd IAT Std": _std0(bwd_iat),
        "Bwd IAT Max": _max0(bwd_iat),
        "Bwd IAT Min": _min0(bwd_iat),
        "Fwd PSH Flags": float(0 if CIC2017_FLAG_QUIRK else _count(fwd_flags, "P")),
        "Fwd URG Flags": float(0 if CIC2017_FLAG_QUIRK else _count(fwd_flags, "U")),
        "Fwd Header Length": float(fwd_header_len),
        "Bwd Header Length": float(bwd_header_len),
        "Fwd Packets/s": fwd_pkts_s,
        "Bwd Packets/s": bwd_pkts_s,
        "Min Packet Length": min_len,
        "Max Packet Length": max_len,
        "Packet Length Mean": mean_len,
        "Packet Length Std": std_len,
        "Packet Length Variance": var_len,
        "FIN Flag Count": float(fin),
        "SYN Flag Count": float(syn),
        "RST Flag Count": float(rst),
        "PSH Flag Count": float(psh),
        "ACK Flag Count": float(ack),
        "URG Flag Count": float(urg),
        "CWE Flag Count": float(cwe),
        "ECE Flag Count": float(ece),
        "Down/Up Ratio": round(down_up_ratio, 8),
        "Average Packet Size": round(avg_pkt_size, 8),
        "Avg Fwd Segment Size": avg_fwd_seg,
        "Avg Bwd Segment Size": avg_bwd_seg,
        "Fwd Header Length.1": float(fwd_header_len),
        "Subflow Fwd Packets": float(len(sfwd)),
        "Subflow Fwd Bytes": float(sum(p.payload_len for p in sfwd)),
        "Subflow Bwd Packets": float(len(sbwd)),
        "Subflow Bwd Bytes": float(sum(p.payload_len for p in sbwd)),
        "Init_Win_bytes_forward": float(fwd_win),
        "Init_Win_bytes_backward": float(bwd_win),
        "act_data_pkt_fwd": float(act_data_fwd),
        "min_seg_size_forward": float(min_seg_fwd),
        "Active Mean": active["mean"],
        "Active Std": active["std"],
        "Active Max": active["max"],
        "Active Min": active["min"],
        "Idle Mean": idle["mean"],
        "Idle Std": idle["std"],
        "Idle Max": idle["max"],
        "Idle Min": idle["min"],
    }
    return {k: out[k] for k in _TEMPLATE}


# ── Statistics helpers (population stats; empty-safe) ────────────────────────
def _iat_list(packets):
    """Seconds between consecutive packets of a list (sorted ascending)."""
    if len(packets) < 2:
        return []
    return [b.timestamp - a.timestamp for a, b in zip(packets, packets[1:])]


def _active_idle_stats(packets):
    """Split the flow timeline into active bursts and idle gaps, then stats
    in microseconds (CICFlowMeter convention)."""
    empty = {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0}
    if len(packets) < 1:
        return empty, empty

    times = [p.timestamp for p in packets]
    bursts = []                       # (start, end)
    cur_start = cur_end = times[0]
    for t in times[1:]:
        if t - cur_end > ACTIVE_IDLE_T:
            bursts.append((cur_start, cur_end))
            cur_start = cur_end = t
        else:
            cur_end = t
    bursts.append((cur_start, cur_end))

    active_durs = [max(0.0, end - start) * 1e6 for start, end in bursts]
    idle_durs = [(bursts[i + 1][0] - bursts[i][1]) * 1e6
                 for i in range(len(bursts) - 1)]

    return (_stats(active_durs), _stats(idle_durs))


def _stats(values):
    return {
        "mean": _mean0(values),
        "std": _std0(values),
        "max": _max0(values),
        "min": _min0(values),
    }


def _mean0(vals):
    return round(mean(vals), 8) if vals else 0.0


def _std0(vals):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return round(math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals)), 8)


def _max0(vals):
    return round(float(max(vals)), 8) if vals else 0.0


def _min0(vals):
    return round(float(min(vals)), 8) if vals else 0.0


def _count(flags, ch):
    return flags.count(ch)
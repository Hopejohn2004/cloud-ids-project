"""
Bidirectional flow accumulator.

Groups raw packets into flows keyed by the normalised 5-tuple, remembers which
side sent the first packet (the "forward" initiator) so feature extraction can
split directions, and expires flows that have been idle too long.

Each `add()` call returns a list of completed flows so the caller can classify
them immediately. Pure Python — no scapy import here; the caller converts its
own packet objects into the fields the builder needs.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import List, Tuple

from src.flow_features import FlowPacket

FlowId = Tuple[str, int, str, int, int]  # (ipA, portA, ipB, portB, protocol)


@dataclass
class Flow:
    """A captured flow: its key, origin & packets in arrival order."""
    key: FlowId
    forward_src: str                       # IP of the initiator (first packet)
    forward_sport: int
    forward_dport: int
    protocol: int
    first_seen: float
    last_seen: float
    packets: List[FlowPacket] = field(default_factory=list)


def make_key(src_ip, src_port, dst_ip, dst_port, protocol):
    """5-tuple with the two endpoints normalised so both directions collide."""
    if (src_ip, src_port) <= (dst_ip, dst_port):
        return (src_ip, src_port, dst_ip, dst_port, protocol)
    return (dst_ip, dst_port, src_ip, src_port, protocol)


class FlowAccumulator:
    """Collects packets into bidirectional flows; yields expired ones."""

    def __init__(self, idle_timeout=60.0):
        self.idle_timeout = idle_timeout
        self._flows = {}              # FlowId -> Flow
        self._lock = threading.Lock()   # safe for concurrent add + sweeper thread

    def add(self, timestamp, src_ip, src_port, dst_ip, dst_port, protocol,
            *, ip_len=0, ip_header_len=0, l4_header_len=0, tcp_window=0,
            payload_len=0, flags="", payload_snippet=b""):
        """
        Ingest one packet. Returns a list of flows that expired as a result
        (flows idle > idle_timeout). The newly-added flow is never returned.
        """
        with self._lock:
            flow_id = make_key(src_ip, src_port, dst_ip, dst_port, protocol)

            flow = self._flows.get(flow_id)
            if flow is None:
                # The FIRST packet defines the forward (initiator) direction —
                # regardless of how make_key lexically normalised the endpoints.
                flow = Flow(
                    key=flow_id,
                    forward_src=src_ip,
                    forward_sport=src_port,
                    forward_dport=dst_port,
                    protocol=protocol,
                    first_seen=timestamp,
                    last_seen=timestamp,
                )
                self._flows[flow_id] = flow

            is_forward = src_ip == flow.forward_src
            flow.packets.append(
                FlowPacket(
                    timestamp=timestamp,
                    ip_len=ip_len,
                    ip_header_len=ip_header_len,
                    l4_header_len=l4_header_len,
                    tcp_window=tcp_window,
                    payload_len=payload_len,
                    flags=flags,
                    is_forward=is_forward,
                    payload_snippet=payload_snippet,
                )
            )
            flow.last_seen = max(flow.last_seen, timestamp)
            return self._expire_old(timestamp)

    def expire(self, now):
        """Return flows idle longer than idle_timeout (and drop them)."""
        with self._lock:
            return self._expire_old(now)

    def flush(self):
        """Return and drop ALL currently-held flows."""
        with self._lock:
            flows = list(self._flows.values())
            self._flows.clear()
            return flows

    def active_count(self):
        with self._lock:
            return len(self._flows)

    def split_directions(self, flow):
        """Return (fwd, bwd) FlowPacket lists for a Flow."""
        fwd, bwd = [], []
        for p in flow.packets:
            (fwd if p.is_forward else bwd).append(p)
        return fwd, bwd

    def _expire_old(self, now):
        expired = []
        for flow_id, flow in list(self._flows.items()):
            if now - flow.last_seen > self.idle_timeout:
                expired.append(flow)
                del self._flows[flow_id]
        return expired
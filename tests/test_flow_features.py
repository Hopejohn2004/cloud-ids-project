"""
Unit tests for src.flow_features — synthetic packets, no scapy needed.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

from src import flow_features as _ff
from src.flow_features import FlowPacket, derive_features, ACTIVE_IDLE_T, CIC2017_FLAG_QUIRK


class TestModelCompatibility:
    """Derived features must reproduce the model's expected class on a flow
    that mirrors the real BENIGN benchmark row (guard against semantic drift
    between the extractor, the scaler and the XGBoost model)."""

    def test_benign_flow_scores_as_benign(self):
        import joblib
        import numpy as np

        with open(os.path.join("models", "feature_names.json")) as f:
            names = json.load(f)
        scaler = joblib.load(os.path.join("models", "scaler.pkl"))
        model = joblib.load(os.path.join("models", "best_model.pkl"))
        le = joblib.load(os.path.join("models", "label_encoder.pkl"))

        pkts = [
            P(0.0, l4_header_len=32, tcp_window=29200, flags="S"),
            P(0.002, l4_header_len=32, tcp_window=29200, flags="A"),
            # forward data (34-byte payload, TCP header 32 like the benchmark row)
            P(0.003, l4_header_len=32, tcp_window=29200, payload_len=34, flags="PA"),
        ]
        bwd = [
            P(0.001, l4_header_len=32, tcp_window=29200, flags="SA", is_forward=False),
            # backward data (66-byte payload)
            P(0.004, l4_header_len=32, tcp_window=29200, payload_len=66, flags="PA",
              is_forward=False),
        ]

        fe = derive_features(pkts, bwd, destination_port=443)
        x = np.array([[fe[n] for n in names]], dtype=float)
        label = le.classes_[int(np.argmax(model.predict_proba(scaler.transform(x))[0]))]
        assert label == "BENIGN"


def P(t, **kw):
    """FlowPacket helper with sensible TCP defaults."""
    defaults = dict(ip_len=60, ip_header_len=20, l4_header_len=20,
                    tcp_window=0, payload_len=0, flags="")
    defaults.update(kw)
    return FlowPacket(timestamp=t, **defaults)


def model_feature_names():
    with open(os.path.join("models", "feature_names.json")) as f:
        return json.load(f)


class TestFeatureCompleteness:
    def test_contains_exactly_the_70_model_features_in_order(self):
        feats = derive_features([], [], destination_port=443)
        assert list(feats.keys()) == model_feature_names()

    def test_no_nan_or_inf(self):
        import math
        feats = derive_features([P(0.0), P(0.1)], [P(0.05)], destination_port=80)
        for k, v in feats.items():
            assert math.isfinite(v), f"{k} is not finite: {v}"


class TestSinglePacketFlow:
    def test_empty_flow_all_zeros(self):
        feats = derive_features([], [], destination_port=443)
        assert feats["Total Fwd Packets"] == 0
        assert feats["Total Backward Packets"] == 0
        assert feats["Destination Port"] == 443
        assert feats["Flow Duration"] == 0.0
        assert feats["Flow Bytes/s"] == 0.0
        assert feats["Flow Packets/s"] == 0.0

    def test_one_syn_packet(self):
        feats = derive_features(
            [P(1000.0, ip_len=60, tcp_window=65535, flags="S")],
            [], destination_port=443,
        )
        assert feats["Total Fwd Packets"] == 1
        assert feats["Total Backward Packets"] == 0
        assert feats["Flow Duration"] == 0
        assert feats["SYN Flag Count"] == 1
        assert feats["Init_Win_bytes_forward"] == 65535
        assert feats["Fwd Header Length"] == 20
        assert feats["Destination Port"] == 443


class TestBidirectionalFlow:
    def setup_method(self, method):
        self.fwd = [P(0.0, ip_len=100), P(0.1, ip_len=120)]
        self.bwd = [P(0.05, ip_len=60)]

    def test_counts_and_lengths(self):
        self.fwd = [P(0.0, ip_len=100, payload_len=100), P(0.1, ip_len=120, payload_len=120)]
        self.bwd = [P(0.05, ip_len=60, payload_len=60)]
        feats = derive_features(self.fwd, self.bwd, destination_port=443)
        assert feats["Total Fwd Packets"] == 2
        assert feats["Total Backward Packets"] == 1
        assert feats["Total Length of Fwd Packets"] == 220      # L4 payload bytes
        assert feats["Total Length of Bwd Packets"] == 60
        assert feats["Flow Duration"] == 100_000.0            # 0.1s in us
        assert feats["Down/Up Ratio"] == 0.0                  # floor(1 / 2) packet ratio
        assert feats["Average Packet Size"] == round(280 / 3, 8)

    def test_iat_per_direction(self):
        self.fwd = [P(0.0, ip_len=100), P(0.1, ip_len=120)]
        self.bwd = [P(0.05, ip_len=60)]
        feats = derive_features(self.fwd, self.bwd, destination_port=443)
        # all packets sorted: 0, 0.05, 0.1 -> IATs [0.05, 0.05] in microseconds
        assert feats["Flow IAT Mean"] == 50_000.0
        assert feats["Flow IAT Std"] == 0.0
        assert feats["Flow IAT Min"] == 50_000.0
        assert feats["Flow IAT Max"] == 50_000.0
        # forward only: 0, 0.1 -> [0.1] in microseconds
        assert feats["Fwd IAT Total"] == 100_000.0
        assert feats["Fwd IAT Mean"] == 100_000.0
        assert feats["Bwd IAT Total"] == 0.0                  # single bwd packet

    def test_header_lengths(self):
        feats = derive_features(self.fwd, self.bwd, destination_port=443)
        assert feats["Fwd Header Length"] == 40               # 2 x 20
        assert feats["Bwd Header Length"] == 20


class TestFlagsAndPayload:
    def test_flag_counts(self):
        feats = derive_features(
            [P(0.0, flags="SA"), P(0.1, flags="PA", payload_len=10)],
            [P(0.05, flags="A")],
            destination_port=443,
        )
        assert feats["SYN Flag Count"] == 1
        # CIC-IDS2017 artifact: ACK/PSH/etc flag columns are ~zero in the CSV
        assert feats["ACK Flag Count"] == 0
        assert feats["PSH Flag Count"] == 0
        assert feats["FIN Flag Count"] == 0
        assert feats["Fwd PSH Flags"] == 0
        assert feats["act_data_pkt_fwd"] == 1
        assert feats["min_seg_size_forward"] == 20            # min forward TCP header

    def test_true_flag_counts_when_quirk_disabled(self):
        old = _ff.CIC2017_FLAG_QUIRK
        _ff.CIC2017_FLAG_QUIRK = False
        feats = _ff.derive_features(
            [P(0.0, flags="SA"), P(0.1, flags="PA", payload_len=10)],
            [P(0.05, flags="A")],
            destination_port=443,
        )
        _ff.CIC2017_FLAG_QUIRK = old
        assert feats["ACK Flag Count"] == 3
        assert feats["PSH Flag Count"] == 1
        assert feats["Fwd PSH Flags"] == 1

    def test_cwr_and_ece(self):
        feats = derive_features(
            [P(0.0, flags="CE")], [], destination_port=443,
        )
        assert feats["CWE Flag Count"] == 0                  # quirk zeros them
        assert feats["ECE Flag Count"] == 0


class TestActiveIdle:
    def test_splits_bursts_on_gap(self):
        # burst A at 0,0.1 ; burst B at 10,10.05 -> gap of 9.9s is idle
        pkts = [P(0.0), P(0.1), P(10.0), P(10.05)]
        feats = derive_features(pkts, [], destination_port=443)

        assert feats["Active Mean"] == 75_000.0              # (0.1 + 0.05)/2 in us
        assert feats["Active Max"] == 100_000.0
        assert feats["Active Min"] == 50_000.0
        assert feats["Idle Max"] == 9_900_000.0
        assert feats["Idle Mean"] == 9_900_000.0
"""
Simulated Response Engine for the Cloud IDS.

Turns a detection verdict into a concrete, "executed" mitigation for the demo:
block offending source IPs, raise alerts, or allow traffic. Nothing touches a
real network — this models the decision/hand-off layer that a production IDS
would send to a firewall/EDR so the dashboard can report response_executed=true.

Design:
  - Pure Python, no Flask dependency -> reusable and independently testable.
  - Keeps session state: blocked source IPs + a rolling action log. When an
    optional SQLiteStore is provided, that state is persisted durably and
    shared across app workers; otherwise it stays in-memory (fallback).
  - The generated `command` strings mirror real-world firewall/EDR actions for
    realism but are logged/simulated only, never actually run.
"""

from datetime import datetime


class ResponseEngine:
    """Maintains blocked IPs and an action log across detections."""

    LOG_LIMIT = 100

    def __init__(self, store=None, client_id="shared"):
        self.store = store
        self.client_id = client_id
        if store is None:
            self.blocked_ips = set()
            self.action_log = []          # newest first
            self.counts = {"block": 0, "alert": 0, "allow": 0}

    # ── Public API ────────────────────────────────────────────────────────
    def handle(self, attack_name, severity, action, source_ip=None, confidence=None):
        """
        Execute (simulated) the response for one detection verdict.

        Parameters mirror what /predict produces, so app.py can forward the
        detection entry in one call.

        Returns a dict describing what the engine "did":
            {
                "executed": bool,
                "action":   "BLOCK" | "ALERT" | "ALLOW",
                "command":  str | None,
                "blocked_ip": str | None,
                "detail":   human-readable summary,
                "blocked_ips_total": int,
                "timestamp": "..."
            }
        """
        action = (action or "ALLOW").upper()
        ip = self._normalise_ip(source_ip)

        if action == "BLOCK":
            if ip != "*":
                self._block_ip(ip)
            command = self._block_command(ip)
            detail = (f"Source {ip} added to deny list "
                      f"({self._blocked_ip_count()} sources currently blocked).")
        elif action == "ALERT":
            self._count("alert")
            command = f"notify:alert severity={severity or 'MEDIUM'} source={ip}"
            detail = f"Alert raised to security console for source {ip}."
        else:  # ALLOW
            self._count("allow")
            command = None
            detail = "Traffic allowed. No action required."

        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attack_type": str(attack_name),
            "severity": str(severity),
            "action": action,
            "command": command,
            "blocked_ip": (ip if action in ("BLOCK", "ALERT") and ip != "*" else None),
            "source_ip": ip,
            "confidence": confidence,
            "detail": detail,
            "executed": True,
            "blocked_ips_total": self._blocked_ip_count(),
        }

        self._log_action(record)
        return record

    def snapshot(self):
        """Complete engine state for the dashboard /responses panel."""
        return {
            "blocked_ips": sorted(self._blocked_ips()),
            "blocked_ip_count": self._blocked_ip_count(),
            "counts": dict(self._action_counts()),
            "recent": self._recent_actions(),
        }

    # ── Persistence delegation (store-backed vs in-memory fallback) ───────
    def _block_ip(self, ip):
        if self.store is not None:
            self.store.block_ip(self.client_id, ip)
            return
        self.blocked_ips.add(ip)

    def _blocked_ips(self):
        if self.store is not None:
            return self.store.blocked_ips(self.client_id)
        return list(self.blocked_ips)

    def _blocked_ip_count(self):
        return len(self._blocked_ips())

    def _count(self, action):
        if self.store is None:
            self.counts[action] = self.counts.get(action, 0) + 1

    def _action_counts(self):
        if self.store is not None:
            return self.store.action_counts(self.client_id)
        return dict(self.counts)

    def _log_action(self, record):
        if self.store is not None:
            self.store.add_action(self.client_id, record)
            return
        self.action_log.insert(0, record)
        if len(self.action_log) > self.LOG_LIMIT:
            self.action_log.pop()

    def _recent_actions(self):
        if self.store is not None:
            return self.store.recent_actions(self.client_id, 20)
        return self.action_log[:20]

    # ── Internals ─────────────────────────────────────────────────────────
    @staticmethod
    def _normalise_ip(ip):
        """Return a sane string for the offending host; fall back to a wildcard."""
        if ip is None or str(ip).strip() in ("", "None"):
            return "*"
        return str(ip).strip()

    @staticmethod
    def _block_command(ip):
        """A realistic-looking mitigation command. Simulated only — never executed."""
        if ip == "*":
            return None
        return f"iptables -A INPUT -s {ip} -j DROP"
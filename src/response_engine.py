"""
Simulated Response Engine for the Cloud IDS.

Turns a detection verdict into a concrete, "executed" mitigation for the demo:
block offending source IPs, raise alerts, or allow traffic. Nothing touches a
real network — this models the decision/hand-off layer that a production IDS
would send to a firewall/EDR so the dashboard can report response_executed=true.

Design:
  - Pure Python, no Flask dependency -> reusable and independently testable.
  - Keeps session state: blocked source IPs + a rolling action log.
  - The generated `command` strings mirror real-world firewall/EDR actions for
    realism but are logged/simulated only, never actually run.
"""

from datetime import datetime


class ResponseEngine:
    """Maintains blocked IPs and an action log across detections."""

    LOG_LIMIT = 100

    def __init__(self):
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
            self.blocked_ips.add(ip)
            self.counts["block"] += 1
            command = self._block_command(ip)
            detail = (f"Source {ip} added to deny list ({len(self.blocked_ips)} "
                      f"sources currently blocked).")
        elif action == "ALERT":
            self.counts["alert"] += 1
            command = f"notify:alert severity={severity or 'MEDIUM'} source={ip}"
            detail = f"Alert raised to security console for source {ip}."
        else:  # ALLOW
            self.counts["allow"] += 1
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
        }

        self.action_log.insert(0, record)
        if len(self.action_log) > self.LOG_LIMIT:
            self.action_log.pop()

        return record

    def snapshot(self):
        """Complete engine state for the dashboard /responses panel."""
        return {
            "blocked_ips": sorted(self.blocked_ips),
            "blocked_ip_count": len(self.blocked_ips),
            "counts": dict(self.counts),
            "recent": self.action_log[:20],
        }

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
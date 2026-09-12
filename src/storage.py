"""
SQLite persistence layer for the Cloud IDS.

Replaces the entirely in-memory detection log / stats / blocked-IP deny list
with a durable SQLite database, so state survives app restarts and is safe
across app servers sharing a filesystem.

Every method is keyed by a `client_id` so each dashboard device keeps its own
feed the same way the earlier in-memory version did. A connection-per-call
approach (plus WAL mode) keeps this safe for multi-worker deployments.

All Python types are stored (bools -> int, dicts -> JSON); read-back methods
rehydrate them so calling code sees plain Python objects.
"""

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime

DB_PATH = os.environ.get(
    "IDS_DB_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ids_state.db"),
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS clients (
    client_id TEXT PRIMARY KEY,
    last_seen TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    attack_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    action TEXT NOT NULL,
    recommended_action TEXT NOT NULL,
    response_executed INTEGER NOT NULL,
    response_detail TEXT NOT NULL,
    confidence TEXT NOT NULL,
    is_threat INTEGER NOT NULL,
    source_ip TEXT,
    destination_ip TEXT,
    source_port TEXT,
    destination_port TEXT
);
CREATE INDEX IF NOT EXISTS idx_detections_client ON detections (client_id, id);
CREATE TABLE IF NOT EXISTS stats (
    client_id TEXT PRIMARY KEY,
    total INTEGER NOT NULL DEFAULT 0,
    threats INTEGER NOT NULL DEFAULT 0,
    benign INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS blocked_ips (
    client_id TEXT NOT NULL,
    ip TEXT NOT NULL,
    blocked_at TEXT NOT NULL,
    PRIMARY KEY (client_id, ip)
);
CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    attack_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    action TEXT NOT NULL,
    command TEXT,
    blocked_ip TEXT,
    source_ip TEXT,
    confidence TEXT,
    detail TEXT NOT NULL,
    executed INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_actions_client ON actions (client_id, id);
"""


class SQLiteStore:
    """Durably persists per-client detections, stats & response state."""

    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        # ":memory:" needs a single shared connection (each new connect() would
        # otherwise create a fresh empty database); everything else gets a
        # fresh connection per call so threads/workers never share a handle.
        self._memory_conn = None
        self._lock = threading.Lock()
        self._init_schema()

    # ── Connection helpers ────────────────────────────────────────────────
    def _init_schema(self):
        with self._cursor() as (conn, cur):
            cur.executescript(_SCHEMA)
            conn.commit()

    @contextmanager
    def _cursor(self):
        """Yield an open (connection, cursor); the caller manages commit."""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        try:
            yield conn, cur
        finally:
            # Never close the shared in-memory connection (that would drop the DB)
            if self.db_path != ":memory:":
                conn.close()

    def _connect(self):
        if self.db_path == ":memory:":
            if self._memory_conn is None:
                self._memory_conn = sqlite3.connect(
                    ":memory:", check_same_thread=False, timeout=30
                )
            return self._memory_conn
        return sqlite3.connect(self.db_path, timeout=30)

    # ── Clients ───────────────────────────────────────────────────────────
    def register_client(self, client_id):
        with self._cursor() as (conn, cur):
            cur.execute(
                "INSERT OR REPLACE INTO clients (client_id, last_seen) VALUES (?, ?)",
                (client_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )
            conn.commit()

    # ── Detections / stats / distribution ─────────────────────────────────
    def add_detection(self, client_id, entry):
        """Insert one detection and bump the client's counters in one tx."""
        self.register_client(client_id)
        is_threat = int(bool(entry.get("is_threat")))
        with self._cursor() as (conn, cur):
            cur.execute(
                """INSERT INTO detections
                   (client_id, timestamp, attack_type, severity, action,
                    recommended_action, response_executed, response_detail,
                    confidence, is_threat, source_ip, destination_ip,
                    source_port, destination_port)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    client_id,
                    entry.get("timestamp"),
                    entry.get("attack_type"),
                    entry.get("severity"),
                    entry.get("action"),
                    entry.get("recommended_action"),
                    int(bool(entry.get("response_executed"))),
                    json.dumps(entry.get("response_detail")),
                    entry.get("confidence"),
                    is_threat,
                    entry.get("source_ip"),
                    entry.get("destination_ip"),
                    str(entry.get("source_port")) if entry.get("source_port") is not None else None,
                    str(entry.get("destination_port")) if entry.get("destination_port") is not None else None,
                ),
            )
            cur.execute(
                """INSERT INTO stats (client_id, total, threats, benign)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(client_id) DO UPDATE SET
                       total  = total + excluded.total,
                       threats  = threats + excluded.threats,
                       benign = benign + excluded.benign""",
                (client_id, 1, is_threat, 0 if is_threat else 1),
            )
            conn.commit()

    def recent_detections(self, client_id, limit=20):
        with self._cursor() as (conn, cur):
            cur.execute(
                """SELECT * FROM detections WHERE client_id=?
                   ORDER BY id DESC LIMIT ?""",
                (client_id, limit),
            )
            rows = cur.fetchall()
            conn.commit()
        return [self._row_to_detection(r) for r in rows]

    @staticmethod
    def _row_to_detection(row):
        return {
            "timestamp":           row["timestamp"],
            "attack_type":         row["attack_type"],
            "severity":            row["severity"],
            "action":              row["action"],
            "recommended_action":  row["recommended_action"],
            "response_executed":   bool(row["response_executed"]),
            "response_detail":     json.loads(row["response_detail"] or "{}"),
            "confidence":          row["confidence"],
            "is_threat":           bool(row["is_threat"]),
            "source_ip":           row["source_ip"],
            "destination_ip":      row["destination_ip"],
            "source_port":         row["source_port"],
            "destination_port":    row["destination_port"],
        }

    def stats(self, client_id):
        defaults = {"total": 0, "threats": 0, "benign": 0}
        with self._cursor() as (conn, cur):
            cur.execute(
                "SELECT total, threats, benign FROM stats WHERE client_id=?",
                (client_id,),
            )
            row = cur.fetchone()
            conn.commit()
        if row is None:
            return defaults
        return {"total": row["total"], "threats": row["threats"], "benign": row["benign"]}

    def distribution(self, client_id):
        with self._cursor() as (conn, cur):
            cur.execute(
                """SELECT attack_type AS name, COUNT(*) AS n
                   FROM detections WHERE client_id=?
                   GROUP BY attack_type ORDER BY n DESC""",
                (client_id,),
            )
            rows = cur.fetchall()
            conn.commit()
        return {r["name"]: r["n"] for r in rows}

    # ── Response engine state ─────────────────────────────────────────────
    def blocked_ips(self, client_id):
        with self._cursor() as (conn, cur):
            cur.execute(
                "SELECT ip FROM blocked_ips WHERE client_id=? ORDER BY ip",
                (client_id,),
            )
            rows = cur.fetchall()
            conn.commit()
        return [r["ip"] for r in rows]

    def block_ip(self, client_id, ip):
        with self._cursor() as (conn, cur):
            cur.execute(
                "INSERT OR IGNORE INTO blocked_ips (client_id, ip, blocked_at) VALUES (?, ?, ?)",
                (client_id, ip, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )
            conn.commit()

    def add_action(self, client_id, record):
        with self._cursor() as (conn, cur):
            cur.execute(
                """INSERT INTO actions
                   (client_id, timestamp, attack_type, severity, action,
                    command, blocked_ip, source_ip, confidence, detail, executed)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    client_id,
                    record.get("timestamp"),
                    record.get("attack_type"),
                    record.get("severity"),
                    record.get("action"),
                    record.get("command"),
                    record.get("blocked_ip"),
                    record.get("source_ip"),
                    record.get("confidence"),
                    record.get("detail"),
                    int(bool(record.get("executed"))),
                ),
            )
            conn.commit()

    def recent_actions(self, client_id, limit=20):
        with self._cursor() as (conn, cur):
            cur.execute(
                """SELECT * FROM actions WHERE client_id=?
                   ORDER BY id DESC LIMIT ?""",
                (client_id, limit),
            )
            rows = cur.fetchall()
            conn.commit()
        return [
            {
                "timestamp":      r["timestamp"],
                "attack_type":    r["attack_type"],
                "severity":       r["severity"],
                "action":         r["action"],
                "command":        r["command"],
                "blocked_ip":     r["blocked_ip"],
                "source_ip":      r["source_ip"],
                "confidence":     r["confidence"],
                "detail":         r["detail"],
                "executed":       bool(r["executed"]),
            }
            for r in rows
        ]

    def action_counts(self, client_id):
        counts = {"block": 0, "alert": 0, "allow": 0}
        with self._cursor() as (conn, cur):
            cur.execute(
                "SELECT action, COUNT(*) AS n FROM actions WHERE client_id=? "
                "GROUP BY action",
                (client_id,),
            )
            rows = cur.fetchall()
            conn.commit()
        for r in rows:
            key = r["action"].lower()
            if key in counts:
                counts[key] = r["n"]
        return counts

    # ── Maintenance / tests ───────────────────────────────────────────────
    def reset_client(self, client_id):
        with self._cursor() as (conn, cur):
            for table in ("detections", "stats", "blocked_ips", "actions"):
                cur.execute(f"DELETE FROM {table} WHERE client_id=?", (client_id,))
            conn.commit()
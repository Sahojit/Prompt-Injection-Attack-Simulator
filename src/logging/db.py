"""
SQLite-based logging system for all prompts, responses, defense results, and scores.
Uses a single connection pool via thread-local storage for safety.
"""
import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from src.config import get_settings

logger = logging.getLogger(__name__)

DDL = """
CREATE TABLE IF NOT EXISTS attack_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    model           TEXT NOT NULL,
    system_mode     TEXT NOT NULL,
    attack_name     TEXT NOT NULL,
    attack_type     TEXT NOT NULL,
    severity        TEXT,
    prompt          TEXT NOT NULL,
    response        TEXT NOT NULL,
    attack_succeeded INTEGER NOT NULL,
    defense_blocked  INTEGER NOT NULL,
    defense_risk_score REAL NOT NULL,
    defense_strategies TEXT,      -- JSON list of strategies that blocked
    rule_based_score    REAL,
    llm_guard_score     REAL,
    latency_ms          REAL,
    metadata            TEXT       -- JSON blob
);

CREATE INDEX IF NOT EXISTS idx_run_id ON attack_logs(run_id);
CREATE INDEX IF NOT EXISTS idx_attack_type ON attack_logs(attack_type);
CREATE INDEX IF NOT EXISTS idx_model ON attack_logs(model);
CREATE INDEX IF NOT EXISTS idx_timestamp ON attack_logs(timestamp);

CREATE TABLE IF NOT EXISTS run_summaries (
    run_id          TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    total_attacks   INTEGER,
    attack_success_rate REAL,
    defense_detection_rate REAL,
    f1_score        REAL,
    models_used     TEXT,   -- JSON list
    metadata        TEXT    -- JSON blob
);
"""


class SimulatorDB:
    _local = threading.local()

    def __init__(self, db_path: Optional[str] = None):
        cfg = get_settings()
        path = db_path or cfg.database.path
        self.db_path = Path(path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript(DDL)
        conn.commit()

    def log_attack(
        self,
        run_id: str,
        model: str,
        system_mode: str,
        attack_name: str,
        attack_type: str,
        severity: str,
        prompt: str,
        response: str,
        attack_succeeded: bool,
        defense_blocked: bool,
        defense_risk_score: float,
        defense_strategies: Optional[list[str]] = None,
        rule_based_score: Optional[float] = None,
        llm_guard_score: Optional[float] = None,
        latency_ms: float = 0.0,
        metadata: Optional[dict] = None,
    ) -> int:
        conn = self._get_conn()
        cursor = conn.execute(
            """INSERT INTO attack_logs
               (run_id, timestamp, model, system_mode, attack_name, attack_type,
                severity, prompt, response, attack_succeeded, defense_blocked,
                defense_risk_score, defense_strategies, rule_based_score,
                llm_guard_score, latency_ms, metadata)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id,
                datetime.now(timezone.utc).isoformat(),
                model, system_mode, attack_name, attack_type, severity,
                prompt, response,
                int(attack_succeeded), int(defense_blocked),
                defense_risk_score,
                json.dumps(defense_strategies or []),
                rule_based_score, llm_guard_score,
                latency_ms,
                json.dumps(metadata or {}),
            ),
        )
        conn.commit()
        return cursor.lastrowid

    def log_run_summary(self, run_id: str, summary: dict, models_used: list[str]):
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO run_summaries
               (run_id, timestamp, total_attacks, attack_success_rate,
                defense_detection_rate, f1_score, models_used, metadata)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                run_id,
                datetime.now(timezone.utc).isoformat(),
                summary.get("total_attacks"),
                summary.get("attack_success_rate"),
                summary.get("defense_detection_rate"),
                summary.get("f1_score"),
                json.dumps(models_used),
                json.dumps(summary),
            ),
        )
        conn.commit()

    def get_logs(
        self,
        run_id: Optional[str] = None,
        model: Optional[str] = None,
        attack_type: Optional[str] = None,
        limit: int = 500,
    ) -> list[dict]:
        conditions = []
        params: list = []
        if run_id:
            conditions.append("run_id = ?")
            params.append(run_id)
        if model:
            conditions.append("model = ?")
            params.append(model)
        if attack_type:
            conditions.append("attack_type = ?")
            params.append(attack_type)

        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        rows = self._get_conn().execute(
            f"SELECT * FROM attack_logs {where} ORDER BY timestamp DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        return [dict(r) for r in rows]

    def get_run_summaries(self, limit: int = 50) -> list[dict]:
        rows = self._get_conn().execute(
            "SELECT * FROM run_summaries ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

"""
SQLite-based logging system — extended with ensemble_score, defense_action,
risk_tier columns, replay support, and analytics queries.
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
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL,
    timestamp           TEXT NOT NULL,
    model               TEXT NOT NULL,
    system_mode         TEXT NOT NULL,
    attack_name         TEXT NOT NULL,
    attack_type         TEXT NOT NULL,
    severity            TEXT,
    prompt              TEXT NOT NULL,
    response            TEXT NOT NULL,
    attack_succeeded    INTEGER NOT NULL,
    defense_blocked     INTEGER NOT NULL,
    defense_risk_score  REAL NOT NULL,
    ensemble_score      REAL,
    risk_tier           TEXT DEFAULT 'allow',
    defense_action      TEXT DEFAULT 'none',
    defense_strategies  TEXT,
    rule_based_score    REAL,
    ml_classifier_score REAL,
    llm_guard_score     REAL,
    latency_ms          REAL,
    metadata            TEXT
);

CREATE INDEX IF NOT EXISTS idx_run_id      ON attack_logs(run_id);
CREATE INDEX IF NOT EXISTS idx_attack_type ON attack_logs(attack_type);
CREATE INDEX IF NOT EXISTS idx_model       ON attack_logs(model);
CREATE INDEX IF NOT EXISTS idx_timestamp   ON attack_logs(timestamp);

CREATE TABLE IF NOT EXISTS run_summaries (
    run_id                  TEXT PRIMARY KEY,
    timestamp               TEXT NOT NULL,
    total_attacks           INTEGER,
    attack_success_rate     REAL,
    defense_detection_rate  REAL,
    false_negative_rate     REAL DEFAULT 0.0,
    false_positive_rate     REAL DEFAULT 0.0,
    f1_score                REAL,
    models_used             TEXT,
    defense_mode            TEXT DEFAULT 'full',
    metadata                TEXT
);
"""

# Migration: add new columns to existing DBs without wiping data
MIGRATIONS = [
    "ALTER TABLE attack_logs ADD COLUMN ensemble_score REAL",
    "ALTER TABLE attack_logs ADD COLUMN risk_tier TEXT DEFAULT 'allow'",
    "ALTER TABLE attack_logs ADD COLUMN defense_action TEXT DEFAULT 'none'",
    "ALTER TABLE attack_logs ADD COLUMN ml_classifier_score REAL",
    "ALTER TABLE run_summaries ADD COLUMN false_negative_rate REAL DEFAULT 0.0",
    "ALTER TABLE run_summaries ADD COLUMN false_positive_rate REAL DEFAULT 0.0",
    "ALTER TABLE run_summaries ADD COLUMN defense_mode TEXT DEFAULT 'full'",
    # Index on risk_tier must come after the column is added
    "CREATE INDEX IF NOT EXISTS idx_risk_tier ON attack_logs(risk_tier)",
]


class SimulatorDB:
    _local = threading.local()

    def __init__(self, db_path: Optional[str] = None):
        cfg = get_settings()
        path = db_path or cfg.database.path
        self.db_path = Path(path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._run_migrations()

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

    def _run_migrations(self):
        conn = self._get_conn()
        for sql in MIGRATIONS:
            try:
                conn.execute(sql)
                conn.commit()
            except sqlite3.OperationalError:
                pass  # column already exists

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
        ensemble_score: Optional[float] = None,
        risk_tier: str = "allow",
        defense_action: str = "none",
        defense_strategies: Optional[list[str]] = None,
        rule_based_score: Optional[float] = None,
        ml_classifier_score: Optional[float] = None,
        llm_guard_score: Optional[float] = None,
        latency_ms: float = 0.0,
        metadata: Optional[dict] = None,
    ) -> int:
        conn = self._get_conn()
        cursor = conn.execute(
            """INSERT INTO attack_logs
               (run_id, timestamp, model, system_mode, attack_name, attack_type,
                severity, prompt, response, attack_succeeded, defense_blocked,
                defense_risk_score, ensemble_score, risk_tier, defense_action,
                defense_strategies, rule_based_score, ml_classifier_score,
                llm_guard_score, latency_ms, metadata)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id,
                datetime.now(timezone.utc).isoformat(),
                model, system_mode, attack_name, attack_type, severity,
                prompt, response,
                int(attack_succeeded), int(defense_blocked),
                defense_risk_score,
                ensemble_score,
                risk_tier,
                defense_action,
                json.dumps(defense_strategies or []),
                rule_based_score, ml_classifier_score, llm_guard_score,
                latency_ms,
                json.dumps(metadata or {}),
            ),
        )
        conn.commit()
        return cursor.lastrowid

    def log_run_summary(
        self,
        run_id: str,
        summary: dict,
        models_used: list[str],
        defense_mode: str = "full",
    ):
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO run_summaries
               (run_id, timestamp, total_attacks, attack_success_rate,
                defense_detection_rate, false_negative_rate, false_positive_rate,
                f1_score, models_used, defense_mode, metadata)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id,
                datetime.now(timezone.utc).isoformat(),
                summary.get("total_attacks"),
                summary.get("attack_success_rate"),
                summary.get("defense_detection_rate"),
                summary.get("false_negative_rate", 0.0),
                summary.get("false_positive_rate", 0.0),
                summary.get("f1_score"),
                json.dumps(models_used),
                defense_mode,
                json.dumps(summary),
            ),
        )
        conn.commit()

    def get_logs(
        self,
        run_id: Optional[str] = None,
        model: Optional[str] = None,
        attack_type: Optional[str] = None,
        risk_tier: Optional[str] = None,
        limit: int = 500,
    ) -> list[dict]:
        conditions, params = [], []
        if run_id:
            conditions.append("run_id = ?"); params.append(run_id)
        if model:
            conditions.append("model = ?"); params.append(model)
        if attack_type:
            conditions.append("attack_type = ?"); params.append(attack_type)
        if risk_tier:
            conditions.append("risk_tier = ?"); params.append(risk_tier)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
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

    def replay_attack(self, log_id: int) -> Optional[dict]:
        """Fetch a logged attack by ID for replaying."""
        row = self._get_conn().execute(
            "SELECT * FROM attack_logs WHERE id = ?", (log_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_insights(self) -> dict:
        """Aggregate statistics across all runs for the insights page."""
        conn = self._get_conn()

        total = conn.execute("SELECT COUNT(*) FROM attack_logs").fetchone()[0]
        if not total:
            return {"total_attacks": 0}

        succeeded = conn.execute(
            "SELECT COUNT(*) FROM attack_logs WHERE attack_succeeded = 1"
        ).fetchone()[0]

        blocked = conn.execute(
            "SELECT COUNT(*) FROM attack_logs WHERE defense_blocked = 1"
        ).fetchone()[0]

        # ASR by attack type
        rows = conn.execute(
            """SELECT attack_type,
                      COUNT(*) as total,
                      SUM(attack_succeeded) as succeeded,
                      AVG(defense_risk_score) as avg_risk
               FROM attack_logs
               GROUP BY attack_type"""
        ).fetchall()
        by_type = {
            r["attack_type"]: {
                "total": r["total"],
                "asr": round(r["succeeded"] / r["total"], 3) if r["total"] else 0,
                "avg_risk": round(r["avg_risk"] or 0, 3),
            }
            for r in rows
        }

        # Tier distribution
        tier_rows = conn.execute(
            "SELECT risk_tier, COUNT(*) as cnt FROM attack_logs GROUP BY risk_tier"
        ).fetchall()
        tier_dist = {r["risk_tier"]: r["cnt"] for r in tier_rows}

        # Model vulnerability comparison
        model_rows = conn.execute(
            """SELECT model, COUNT(*) as total, SUM(attack_succeeded) as succeeded
               FROM attack_logs GROUP BY model"""
        ).fetchall()
        by_model = {
            r["model"]: {
                "total": r["total"],
                "asr": round(r["succeeded"] / r["total"], 3) if r["total"] else 0,
            }
            for r in model_rows
        }

        # Top bypassing attacks (succeeded despite defense)
        bypass_rows = conn.execute(
            """SELECT attack_name, COUNT(*) as bypass_count
               FROM attack_logs
               WHERE attack_succeeded = 1 AND defense_blocked = 0
               GROUP BY attack_name
               ORDER BY bypass_count DESC
               LIMIT 5"""
        ).fetchall()
        top_bypasses = [{"attack": r["attack_name"], "count": r["bypass_count"]} for r in bypass_rows]

        return {
            "total_attacks": total,
            "overall_asr": round(succeeded / total, 3),
            "overall_block_rate": round(blocked / total, 3),
            "by_attack_type": by_type,
            "tier_distribution": tier_dist,
            "by_model": by_model,
            "top_bypassing_attacks": top_bypasses,
        }

    def close(self):
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

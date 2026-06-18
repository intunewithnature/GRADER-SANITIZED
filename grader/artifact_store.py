from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .models import BlindResult
from .util import now_iso, state_dir


class ArtifactStore:
    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.db_path = state_dir(base_dir) / "artifacts.sqlite"
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    safe_text TEXT NOT NULL,
                    hidden_field_types TEXT NOT NULL,
                    canary_count INTEGER NOT NULL,
                    verification_status TEXT NOT NULL,
                    verification_summary TEXT NOT NULL,
                    input_sha256 TEXT NOT NULL,
                    safe_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS judgments (
                    judgment_id TEXT PRIMARY KEY,
                    artifact_id TEXT NOT NULL,
                    judgment_text TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def save_artifact(self, result: BlindResult) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO artifacts (
                    artifact_id, safe_text, hidden_field_types, canary_count,
                    verification_status, verification_summary, input_sha256,
                    safe_sha256, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.artifact_id,
                    result.safe_text,
                    json.dumps(result.hidden_field_types),
                    result.canary_count,
                    result.verification_status,
                    result.verification_summary,
                    result.input_sha256,
                    result.safe_sha256,
                    result.created_at,
                ),
            )

    def delete_artifact(self, artifact_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM artifacts WHERE artifact_id = ?", (artifact_id,))

    def get_artifact(self, artifact_id: str) -> BlindResult | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)
            ).fetchone()
        if not row:
            return None
        return BlindResult(
            artifact_id=row["artifact_id"],
            safe_text=row["safe_text"],
            hidden_field_types=json.loads(row["hidden_field_types"]),
            canary_count=row["canary_count"],
            verification_status=row["verification_status"],
            verification_summary=row["verification_summary"],
            input_sha256=row["input_sha256"],
            safe_sha256=row["safe_sha256"],
            created_at=row["created_at"],
        )

    def list_artifacts(self, limit: int = 50) -> list[BlindResult]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM artifacts ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        artifacts = []
        for row in rows:
            artifacts.append(
                BlindResult(
                    artifact_id=row["artifact_id"],
                    safe_text=row["safe_text"],
                    hidden_field_types=json.loads(row["hidden_field_types"]),
                    canary_count=row["canary_count"],
                    verification_status=row["verification_status"],
                    verification_summary=row["verification_summary"],
                    input_sha256=row["input_sha256"],
                    safe_sha256=row["safe_sha256"],
                    created_at=row["created_at"],
                )
            )
        return artifacts

    def save_judgment(self, artifact_id: str, judgment_id: str, judgment_text: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO judgments (judgment_id, artifact_id, judgment_text, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (judgment_id, artifact_id, judgment_text, now_iso()),
            )

    def get_judgment(self, judgment_id: str) -> tuple[str, str] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT artifact_id, judgment_text FROM judgments WHERE judgment_id = ?",
                (judgment_id,),
            ).fetchone()
        return (row["artifact_id"], row["judgment_text"]) if row else None

from __future__ import annotations

import sqlite3
from pathlib import Path

from .models import RevealMapEntry
from .util import now_iso, sha256_text, state_dir


class RevealStore:
    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.db_path = state_dir(base_dir) / "reveal_map.sqlite"
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reveal_map (
                    artifact_id TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    field_type TEXT NOT NULL,
                    placeholder TEXT NOT NULL,
                    original_value TEXT NOT NULL,
                    original_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (artifact_id, scope_id, field_type, original_sha256)
                )
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_reveal_placeholder
                ON reveal_map (artifact_id, scope_id, field_type, placeholder)
                """
            )

    def get_by_original(
        self, artifact_id: str, scope_id: str, field_type: str, original_value: str
    ) -> RevealMapEntry | None:
        original_hash = sha256_text(original_value)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM reveal_map
                WHERE artifact_id = ? AND scope_id = ? AND field_type = ? AND original_sha256 = ?
                """,
                (artifact_id, scope_id, field_type, original_hash),
            ).fetchone()
        return self._row_to_entry(row) if row else None

    def get_or_create(
        self,
        artifact_id: str,
        scope_id: str,
        field_type: str,
        original_value: str,
        placeholder_format: str,
    ) -> RevealMapEntry:
        existing = self.get_by_original(artifact_id, scope_id, field_type, original_value)
        if existing:
            return existing
        with self._connect() as conn:
            count = conn.execute(
                """
                SELECT COUNT(*) FROM reveal_map
                WHERE artifact_id = ? AND scope_id = ? AND field_type = ?
                """,
                (artifact_id, scope_id, field_type),
            ).fetchone()[0]
            placeholder = placeholder_format.format(n=count + 1)
            entry = RevealMapEntry(
                artifact_id=artifact_id,
                scope_id=scope_id,
                field_type=field_type,
                placeholder=placeholder,
                original_value=original_value,
                original_sha256=sha256_text(original_value),
                created_at=now_iso(),
            )
            conn.execute(
                """
                INSERT INTO reveal_map (
                    artifact_id, scope_id, field_type, placeholder,
                    original_value, original_sha256, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.artifact_id,
                    entry.scope_id,
                    entry.field_type,
                    entry.placeholder,
                    entry.original_value,
                    entry.original_sha256,
                    entry.created_at,
                ),
            )
            return entry

    def list_entries(self, artifact_id: str) -> list[RevealMapEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM reveal_map WHERE artifact_id = ? ORDER BY field_type, placeholder",
                (artifact_id,),
            ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def delete_artifact(self, artifact_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM reveal_map WHERE artifact_id = ?", (artifact_id,))

    def replace_placeholders(self, artifact_id: str, text: str) -> str:
        revealed = text
        for entry in self.list_entries(artifact_id):
            revealed = revealed.replace(entry.placeholder, entry.original_value)
        return revealed

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> RevealMapEntry:
        return RevealMapEntry(
            artifact_id=row["artifact_id"],
            scope_id=row["scope_id"],
            field_type=row["field_type"],
            placeholder=row["placeholder"],
            original_value=row["original_value"],
            original_sha256=row["original_sha256"],
            created_at=row["created_at"],
        )

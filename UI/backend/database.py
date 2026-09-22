"""SQLite persistence for pipeline jobs and human reviews."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    source_kind TEXT NOT NULL,
                    source_path TEXT,
                    original_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    status_message TEXT,
                    output_dir TEXT NOT NULL,
                    output_video TEXT,
                    analysis_path TEXT UNIQUE,
                    log_path TEXT,
                    error_message TEXT,
                    exit_code INTEGER,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT
                );

                CREATE INDEX IF NOT EXISTS jobs_status_created_idx
                    ON jobs(status, created_at);

                CREATE TABLE IF NOT EXISTS reviews (
                    job_id TEXT NOT NULL,
                    event_key TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    reviewer TEXT NOT NULL DEFAULT '',
                    reviewed_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, event_key),
                    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS plate_corrections (
                    job_id TEXT NOT NULL,
                    event_key TEXT NOT NULL,
                    corrected_plate TEXT NOT NULL,
                    corrected_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, event_key),
                    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
                );
                """
            )

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row is not None else None

    def insert_job(self, values: dict[str, Any], *, ignore_existing: bool = False) -> bool:
        columns = (
            "id",
            "source_kind",
            "source_path",
            "original_name",
            "status",
            "status_message",
            "output_dir",
            "output_video",
            "analysis_path",
            "log_path",
            "error_message",
            "exit_code",
            "created_at",
            "started_at",
            "finished_at",
        )
        verb = "INSERT OR IGNORE" if ignore_existing else "INSERT"
        placeholders = ", ".join("?" for _ in columns)
        sql = f"{verb} INTO jobs ({', '.join(columns)}) VALUES ({placeholders})"
        with self.connect() as connection:
            cursor = connection.execute(sql, [values.get(column) for column in columns])
            return cursor.rowcount > 0

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row(row)

    def list_jobs(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC, id DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def recover_interrupted_jobs(self) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status = 'queued', status_message = '伺服器重啟，已重新排入佇列',
                    started_at = NULL
                WHERE status = 'running'
                """
            )
            return cursor.rowcount

    def claim_next_job(self) -> dict[str, Any] | None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at, id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            started_at = utc_now()
            connection.execute(
                """
                UPDATE jobs
                SET status = 'running', status_message = 'AI pipeline 執行中',
                    started_at = ?, finished_at = NULL, error_message = NULL,
                    exit_code = NULL
                WHERE id = ? AND status = 'queued'
                """,
                (started_at, row["id"]),
            )
            claimed = connection.execute(
                "SELECT * FROM jobs WHERE id = ?", (row["id"],)
            ).fetchone()
        return self._row(claimed)

    def update_job(self, job_id: str, **values: Any) -> None:
        allowed = {
            "status",
            "status_message",
            "output_video",
            "analysis_path",
            "log_path",
            "error_message",
            "exit_code",
            "started_at",
            "finished_at",
        }
        updates = {key: value for key, value in values.items() if key in allowed}
        if not updates:
            return
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self.connect() as connection:
            connection.execute(
                f"UPDATE jobs SET {assignments} WHERE id = ?",
                [*updates.values(), job_id],
            )

    def retry_job(self, job_id: str) -> bool:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status = 'queued', status_message = '已重新排入佇列',
                    error_message = NULL, exit_code = NULL, started_at = NULL,
                    finished_at = NULL
                WHERE id = ? AND status = 'failed'
                """,
                (job_id,),
            )
            return cursor.rowcount > 0

    def get_reviews(self, job_id: str) -> dict[str, dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM reviews WHERE job_id = ?", (job_id,)
            ).fetchall()
        return {row["event_key"]: dict(row) for row in rows}

    def get_plate_corrections(self, job_id: str) -> dict[str, dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM plate_corrections WHERE job_id = ?", (job_id,)
            ).fetchall()
        return {row["event_key"]: dict(row) for row in rows}

    def save_plate_correction(
        self,
        job_id: str,
        event_key: str,
        corrected_plate: str,
    ) -> dict[str, Any]:
        corrected_at = utc_now()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO plate_corrections (
                    job_id, event_key, corrected_plate, corrected_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(job_id, event_key) DO UPDATE SET
                    corrected_plate = excluded.corrected_plate,
                    corrected_at = excluded.corrected_at
                """,
                (job_id, event_key, corrected_plate, corrected_at),
            )
            row = connection.execute(
                """
                SELECT * FROM plate_corrections
                WHERE job_id = ? AND event_key = ?
                """,
                (job_id, event_key),
            ).fetchone()
        return dict(row)

    def delete_plate_correction(self, job_id: str, event_key: str) -> bool:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM plate_corrections
                WHERE job_id = ? AND event_key = ?
                """,
                (job_id, event_key),
            )
            return cursor.rowcount > 0

    def save_review(
        self,
        job_id: str,
        event_key: str,
        verdict: str,
        note: str,
        reviewer: str,
    ) -> dict[str, Any]:
        reviewed_at = utc_now()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO reviews (job_id, event_key, verdict, note, reviewer, reviewed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id, event_key) DO UPDATE SET
                    verdict = excluded.verdict,
                    note = excluded.note,
                    reviewer = excluded.reviewer,
                    reviewed_at = excluded.reviewed_at
                """,
                (job_id, event_key, verdict, note, reviewer, reviewed_at),
            )
            row = connection.execute(
                "SELECT * FROM reviews WHERE job_id = ? AND event_key = ?",
                (job_id, event_key),
            ).fetchone()
        return dict(row)

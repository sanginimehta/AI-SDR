"""SQLite-backed job store.

Results survive a process restart. A job left queued or running when the
server starts is marked failed so the results page does not poll forever.
"""

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone

# Re-entrant: public methods hold the lock and then open the database.
_lock = threading.RLock()
_conn: sqlite3.Connection | None = None
_path: str | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def db_path() -> str:
    return os.getenv("JOBS_DB", "jobs.db")


def init_db() -> None:
    """Open the jobs database and create tables if needed."""
    global _conn, _path
    path = db_path()
    with _lock:
        if _conn is not None and _path == path:
            return
        if _conn is not None:
            _conn.close()
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                mode TEXT NOT NULL,
                total INTEGER NOT NULL DEFAULT 0,
                completed INTEGER NOT NULL DEFAULT 0,
                current_prospect TEXT NOT NULL DEFAULT '',
                error TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        conn.commit()
        _conn = conn
        _path = path


def _connection() -> sqlite3.Connection:
    if _conn is None or _path != db_path():
        init_db()
    assert _conn is not None
    return _conn


def fail_interrupted_jobs() -> int:
    """Mark leftover queued/running jobs failed after a restart."""
    message = "Server restarted before this run finished."
    with _lock:
        conn = _connection()
        cursor = conn.execute(
            """
            UPDATE jobs
            SET status = 'failed',
                error = ?,
                current_prospect = '',
                updated_at = ?
            WHERE status IN ('queued', 'running')
            """,
            (message, _now()),
        )
        conn.commit()
        return cursor.rowcount


def create_job(mode: str, total: int) -> str:
    job_id = str(uuid.uuid4())
    timestamp = _now()
    with _lock:
        conn = _connection()
        conn.execute(
            """
            INSERT INTO jobs (
                id, status, mode, total, completed, current_prospect, error, created_at, updated_at
            ) VALUES (?, 'queued', ?, ?, 0, '', '', ?, ?)
            """,
            (job_id, mode, total, timestamp, timestamp),
        )
        conn.commit()
    return job_id


def mark_running(job_id: str) -> None:
    _update(job_id, "UPDATE jobs SET status = 'running', updated_at = ? WHERE id = ?", (_now(), job_id))


def mark_done(job_id: str) -> None:
    _update(
        job_id,
        "UPDATE jobs SET status = 'done', current_prospect = '', updated_at = ? WHERE id = ?",
        (_now(), job_id),
    )


def mark_failed(job_id: str, error: str) -> None:
    _update(
        job_id,
        """
        UPDATE jobs
        SET status = 'failed', error = ?, current_prospect = '', updated_at = ?
        WHERE id = ?
        """,
        (error, _now(), job_id),
    )


def set_total(job_id: str, total: int) -> None:
    _update(job_id, "UPDATE jobs SET total = ?, updated_at = ? WHERE id = ?", (total, _now(), job_id))


def set_current(job_id: str, name: str) -> None:
    _update(
        job_id,
        "UPDATE jobs SET current_prospect = ?, updated_at = ? WHERE id = ?",
        (name, _now(), job_id),
    )


def append_result(job_id: str, payload: dict, position: int) -> None:
    with _lock:
        conn = _connection()
        conn.execute(
            "INSERT INTO results (job_id, position, payload) VALUES (?, ?, ?)",
            (job_id, position, json.dumps(payload)),
        )
        conn.execute(
            "UPDATE jobs SET completed = completed + 1, updated_at = ? WHERE id = ?",
            (_now(), job_id),
        )
        conn.commit()


def get_job(job_id: str) -> dict | None:
    with _lock:
        conn = _connection()
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        result_rows = conn.execute(
            "SELECT payload FROM results WHERE job_id = ? ORDER BY position, id",
            (job_id,),
        ).fetchall()
    job = dict(row)
    job["results"] = [json.loads(item["payload"]) for item in result_rows]
    return job


def _update(job_id: str, sql: str, params: tuple) -> None:
    with _lock:
        conn = _connection()
        conn.execute(sql, params)
        conn.commit()

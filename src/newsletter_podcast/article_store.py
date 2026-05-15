from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

FINAL_STATUSES = {"published", "skipped_source", "too_short"}


class ArticleStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS article_jobs (
                    url_hash TEXT PRIMARY KEY,
                    source_email_uid TEXT NOT NULL,
                    url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    publisher TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    processed_at TEXT
                )
                """
            )

    def should_attempt(self, url: str, max_attempts: int = 3) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status, attempts FROM article_jobs WHERE url_hash = ?",
                (url_hash(url),),
            ).fetchone()
        if row is None:
            return True
        status, attempts = row
        return str(status) not in FINAL_STATUSES and int(attempts) < max_attempts

    def start(self, source_email_uid: str, url: str, title: str, publisher: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO article_jobs (
                    url_hash, source_email_uid, url, title, publisher, status, attempts
                )
                VALUES (?, ?, ?, ?, ?, 'queued', 0)
                ON CONFLICT(url_hash) DO UPDATE SET
                    title = excluded.title,
                    publisher = excluded.publisher
                """,
                (url_hash(url), source_email_uid, url, title, publisher),
            )

    def mark_attempt(self, url: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE article_jobs
                SET attempts = attempts + 1, status = 'fetching'
                WHERE url_hash = ?
                """,
                (url_hash(url),),
            )

    def mark_done(self, url: str, status: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE article_jobs
                SET status = ?, processed_at = CURRENT_TIMESTAMP, last_error = NULL
                WHERE url_hash = ?
                """,
                (status, url_hash(url)),
            )

    def mark_failed(self, url: str, status: str, error: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE article_jobs
                SET status = ?, last_error = ?
                WHERE url_hash = ?
                """,
                (status, error[:1000], url_hash(url)),
            )


def url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()

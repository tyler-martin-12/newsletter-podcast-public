from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from email.utils import parseaddr
from pathlib import Path

from imap_tools import AND, MailBox, MailMessageFlags

from newsletter_podcast.config import ImapConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NewsletterEmail:
    uid: str
    subject: str
    sender: str
    html: str


class ProcessedStore:
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
                CREATE TABLE IF NOT EXISTS processed_messages (
                    uid TEXT PRIMARY KEY,
                    processed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def has(self, uid: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM processed_messages WHERE uid = ?",
                (uid,),
            ).fetchone()
        return row is not None

    def add(self, uid: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO processed_messages (uid) VALUES (?)",
                (uid,),
            )


def fetch_messages(
    config: ImapConfig,
    store: ProcessedStore,
    include_seen: bool = False,
) -> list[NewsletterEmail]:
    messages: list[NewsletterEmail] = []
    criteria = AND(all=True) if include_seen else AND(seen=False)
    with _mailbox(config) as mailbox:
        for message in mailbox.fetch(criteria, mark_seen=False):
            uid = str(message.uid)
            if store.has(uid):
                logger.info("Skipping already processed message", extra={"uid": uid})
                continue
            body = _best_body(message.html, message.text)
            if not body.strip():
                logger.warning("Skipping message without text body", extra={"uid": uid})
                continue
            messages.append(
                NewsletterEmail(
                    uid=uid,
                    subject=message.subject or "Untitled newsletter",
                    sender=parseaddr(message.from_ or "")[1] or message.from_ or "unknown",
                    html=body,
                )
            )
    return messages


def fetch_unread(config: ImapConfig, store: ProcessedStore) -> list[NewsletterEmail]:
    return fetch_messages(config, store, include_seen=False)


def mark_read(config: ImapConfig, uid: str) -> None:
    with _mailbox(config) as mailbox:
        mailbox.flag(uid, MailMessageFlags.SEEN, True)


def finish_message(config: ImapConfig, uid: str) -> None:
    with _mailbox(config) as mailbox:
        mailbox.flag(uid, MailMessageFlags.SEEN, True)
        if config.move_to_folder and config.move_to_folder != config.folder:
            mailbox.move(uid, config.move_to_folder)


@contextmanager
def _mailbox(config: ImapConfig) -> Iterator[MailBox]:
    mailbox = MailBox(config.host, port=config.port)
    try:
        mailbox.login(config.username, config.password, initial_folder=config.folder)
        yield mailbox
    finally:
        with suppress(Exception):
            mailbox.logout()


def _best_body(html: str | None, text: str | None) -> str:
    if html and len(html.strip()) >= len((text or "").strip()):
        return html
    return text or html or ""

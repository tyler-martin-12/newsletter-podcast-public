from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from newsletter_podcast.article_fetch import (
    ArticleFetchError,
    ArticleTextTooShortError,
    fetch_article,
)
from newsletter_podcast.article_store import ArticleStore
from newsletter_podcast.clean import TextTooShortError, clean_article
from newsletter_podcast.config import AppConfig, TtsConfig, load_config
from newsletter_podcast.feed import prune_old_episodes, write_feed
from newsletter_podcast.fetch import (
    NewsletterEmail,
    ProcessedStore,
    fetch_messages,
    finish_message,
)
from newsletter_podcast.weekly import (
    extract_skipped_links,
    extract_supported_links,
    is_weekly_picks,
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in _LOG_RECORD_KEYS:
                continue
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, separators=(",", ":"))


_LOG_RECORD_KEYS = set(logging.makeLogRecord({}).__dict__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, help="Path to config.yaml")
    parser.add_argument("--folder", help="Override the IMAP folder or Gmail label to read")
    parser.add_argument(
        "--move-to-folder",
        help="Move handled messages to this IMAP folder or label",
    )
    parser.add_argument(
        "--include-seen",
        action="store_true",
        help="Include read messages as well as unread messages",
    )
    args = parser.parse_args(argv)
    configure_logging()

    config = load_config(args.config)
    if args.folder or args.move_to_folder:
        config = config.model_copy(
            update={
                "imap": config.imap.model_copy(
                    update={
                        "folder": args.folder or config.imap.folder,
                        "move_to_folder": args.move_to_folder or config.imap.move_to_folder,
                    }
                )
            }
        )
    config.output.lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(config.output.lock_path), timeout=0)
    try:
        with lock:
            run_pipeline(config, include_seen=args.include_seen)
    except Timeout:
        logging.getLogger(__name__).warning("Pipeline already running")
        return 0
    except Exception:
        logging.getLogger(__name__).exception("Pipeline failed")
        return 1
    return 0


def run_pipeline(config: AppConfig, include_seen: bool = False) -> None:
    store = ProcessedStore(config.output.state_db)
    logger = logging.getLogger(__name__)
    prune_old_episodes(config.output.episodes_dir, config.feed.retention_days)
    messages = _fetch_with_retry(config, store, include_seen)
    logger.info(
        "Fetched messages",
        extra={"count": len(messages), "include_seen": include_seen},
    )

    for message in messages:
        if is_weekly_picks(message.subject, message.sender):
            if _process_weekly_picks(
                config,
                message,
                article_store=ArticleStore(config.output.state_db),
            ):
                store.add(message.uid)
                finish_message(config.imap, message.uid)
            write_feed(config)
            continue

        try:
            article = clean_article(message.subject, message.sender, message.html)
        except TextTooShortError:
            store.add(message.uid)
            finish_message(config.imap, message.uid)
            continue

        _synthesize_in_subprocess(
            title=article.podcast_title,
            artist=article.author or message.sender,
            text=article.text,
            config=config.tts,
            episodes_dir=config.output.episodes_dir,
        )
        store.add(message.uid)
        finish_message(config.imap, message.uid)
        write_feed(config)
        logger.info("Processed message", extra={"uid": message.uid, "title": article.title})

    write_feed(config)


def _process_weekly_picks(
    config: AppConfig,
    message: NewsletterEmail,
    article_store: ArticleStore,
) -> bool:
    logger = logging.getLogger(__name__)
    retry_needed = False
    supported_links = extract_supported_links(message.html)
    skipped_links = extract_skipped_links(message.html)
    logger.info(
        "Processing Weekly Picks links",
        extra={
            "uid": message.uid,
            "supported_count": len(supported_links),
            "skipped_count": len(skipped_links),
        },
    )

    for skipped_url in skipped_links:
        article_store.start(message.uid, skipped_url, "", "audio app source")
        article_store.mark_done(skipped_url, "skipped_source")

    for link in supported_links:
        article_store.start(message.uid, link.url, link.title, link.source)
        if not article_store.should_attempt(link.url):
            logger.info("Skipping article link already handled", extra={"url": link.url})
            continue
        article_store.mark_attempt(link.url)
        try:
            article = fetch_article(link.url, fallback_title=link.title, publisher=link.source)
        except ArticleTextTooShortError:
            article_store.mark_done(link.url, "too_short")
            logger.warning("Rejecting linked article below threshold", extra={"url": link.url})
            continue
        except ArticleFetchError as exc:
            article_store.mark_failed(link.url, "failed_fetch", str(exc))
            retry_needed = retry_needed or article_store.should_attempt(link.url)
            logger.warning(
                "Failed to fetch linked article",
                extra={"url": link.url, "retry_needed": retry_needed},
            )
            continue
        except Exception as exc:
            article_store.mark_failed(link.url, "failed_extract", str(exc))
            retry_needed = retry_needed or article_store.should_attempt(link.url)
            logger.warning(
                "Failed to process linked article",
                exc_info=True,
                extra={"url": link.url, "retry_needed": retry_needed},
            )
            continue

        _synthesize_in_subprocess(
            title=article.podcast_title,
            artist=article.publisher,
            text=article.text,
            config=config.tts,
            episodes_dir=config.output.episodes_dir,
        )
        article_store.mark_done(link.url, "published")
        write_feed(config)
        logger.info(
            "Processed linked article",
            extra={"url": link.url, "title": article.podcast_title, "uid": message.uid},
        )

    return not retry_needed


def _fetch_with_retry(
    config: AppConfig,
    store: ProcessedStore,
    include_seen: bool,
    attempts: int = 3,
) -> list[NewsletterEmail]:
    logger = logging.getLogger(__name__)
    for attempt in range(1, attempts + 1):
        try:
            return fetch_messages(config.imap, store, include_seen=include_seen)
        except Exception:
            if attempt == attempts:
                raise
            logger.warning(
                "Fetch failed, retrying",
                exc_info=True,
                extra={"attempt": attempt, "attempts": attempts},
            )
            time.sleep(5 * attempt)
    raise RuntimeError("Unreachable fetch retry state")


def _synthesize_in_subprocess(
    title: str,
    artist: str,
    text: str,
    config: TtsConfig,
    episodes_dir: Path,
) -> None:
    with tempfile.TemporaryDirectory(prefix="newsletter-podcast-job-") as temp_name:
        job_path = Path(temp_name) / "job.json"
        job = {
            "title": title,
            "artist": artist,
            "text": text,
            "episodes_dir": str(episodes_dir),
            "tts": config.model_dump(mode="json"),
        }
        job_path.write_text(json.dumps(job), encoding="utf-8")
        environment = os.environ.copy()
        environment.setdefault("PYTHONUNBUFFERED", "1")
        subprocess.run(
            [sys.executable, "-m", "newsletter_podcast.synthesize_worker", "--job", str(job_path)],
            check=True,
            env=environment,
        )


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


if __name__ == "__main__":
    raise SystemExit(main())

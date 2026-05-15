from __future__ import annotations

import os
import re
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from urllib.parse import urljoin

from feedgen.feed import FeedGenerator
from mutagen.mp3 import MP3

from newsletter_podcast.config import AppConfig

DATE_RE = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})_.+\.mp3$")


@dataclass(frozen=True)
class EpisodeFile:
    path: Path
    pub_date: date
    duration_seconds: int
    size_bytes: int


def write_feed(config: AppConfig) -> None:
    prune_old_episodes(config.output.episodes_dir, config.feed.retention_days)
    episodes = scan_episodes(config.output.episodes_dir)

    feed = FeedGenerator()
    feed.load_extension("podcast")
    feed.title(config.feed.title)
    feed.description(config.feed.description)
    feed.link(href=str(config.output.base_url), rel="alternate")
    feed.id(str(config.output.base_url))
    feed.language("en")
    feed.generator("newsletter-podcast")
    if config.feed.cover_image:
        feed.image(
            url=_public_url(config, config.feed.cover_image.name),
            title=config.feed.title,
            link=str(config.output.base_url),
        )

    for episode in episodes:
        entry = feed.add_entry()
        title = _title_from_filename(episode.path)
        episode_url = _public_url(config, f"episodes/{episode.path.name}")
        entry.id(episode_url)
        entry.title(title)
        entry.description(title)
        pub_datetime = datetime.combine(episode.pub_date, datetime.min.time(), tzinfo=UTC)
        entry.pubDate(format_datetime(pub_datetime))
        entry.enclosure(episode_url, str(episode.size_bytes), "audio/mpeg")
        with suppress(AttributeError):
            entry.podcast.itunes_duration(_duration_string(episode.duration_seconds))

    config.output.feed_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = config.output.feed_path.with_suffix(config.output.feed_path.suffix + ".tmp")
    feed.rss_file(str(temp_path), pretty=True)
    os.replace(temp_path, config.output.feed_path)


def scan_episodes(episodes_dir: Path) -> list[EpisodeFile]:
    if not episodes_dir.exists():
        return []
    episodes: list[EpisodeFile] = []
    for path in episodes_dir.glob("*.mp3"):
        match = DATE_RE.match(path.name)
        if not match:
            continue
        audio = MP3(path)
        episodes.append(
            EpisodeFile(
                path=path,
                pub_date=date.fromisoformat(match.group("date")),
                duration_seconds=round(audio.info.length),
                size_bytes=path.stat().st_size,
            )
        )
    return sorted(episodes, key=lambda episode: (episode.pub_date, episode.path.name), reverse=True)


def prune_old_episodes(episodes_dir: Path, retention_days: int) -> None:
    if not episodes_dir.exists():
        return
    cutoff = date.today() - timedelta(days=retention_days)
    for episode in scan_episodes(episodes_dir):
        if episode.pub_date < cutoff:
            episode.path.unlink()


def _public_url(config: AppConfig, relative: str) -> str:
    base = str(config.output.base_url).rstrip("/") + "/"
    return urljoin(base, relative)


def _title_from_filename(path: Path) -> str:
    stem = DATE_RE.sub("", path.name).removesuffix(".mp3")
    if "_" in path.stem:
        stem = path.stem.split("_", 1)[1]
    return stem.replace("-", " ").title()


def _duration_string(seconds: int) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

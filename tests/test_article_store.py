from __future__ import annotations

from pathlib import Path

from newsletter_podcast.article_store import ArticleStore


def test_article_store_tracks_attempts_and_final_statuses(tmp_path: Path) -> None:
    store = ArticleStore(tmp_path / "state.sqlite3")
    url = "https://www.theatlantic.com/story"

    assert store.should_attempt(url)

    store.start("123", url, "Story", "The Atlantic")
    store.mark_attempt(url)
    store.mark_failed(url, "failed_fetch", "timeout")

    assert store.should_attempt(url)

    store.mark_attempt(url)
    store.mark_failed(url, "failed_fetch", "timeout")
    store.mark_attempt(url)
    store.mark_failed(url, "failed_fetch", "timeout")

    assert not store.should_attempt(url)

    store.mark_done(url, "published")
    store.start("123", url, "Story", "The Atlantic")

    assert not store.should_attempt(url)

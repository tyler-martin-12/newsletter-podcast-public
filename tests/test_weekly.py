from __future__ import annotations

from newsletter_podcast.weekly import (
    extract_skipped_links,
    extract_supported_links,
    is_weekly_picks,
    normalize_url,
)


def test_weekly_picks_detection(monkeypatch) -> None:
    monkeypatch.setenv("WEEKLY_PICKS_SENDER", "me@example.com")
    assert is_weekly_picks("Weekly Picks - May 1", "me@example.com")
    assert is_weekly_picks("Weekly Pics - May 1", "me@example.com")
    assert not is_weekly_picks("Weekly Picks - May 1", "someone@example.com")


def test_normalize_url_removes_tracking_params() -> None:
    assert normalize_url(
        "https://www.technologyreview.com/story?utm_source=email&x=1#section"
    ) == "https://www.technologyreview.com/story?x=1"


def test_extract_supported_links_keeps_only_sources_to_synthesize() -> None:
    html = """
    <a href="https://www.technologyreview.com/2026/05/01/ai/?utm_source=email">MIT</a>
    <a href="https://www.theatlantic.com/technology/archive/2026/05/story/">Atlantic</a>
    <a href="https://www.nytimes.com/2026/05/01/technology/story.html">NYT</a>
    <a href="https://www.economist.com/science-and-technology/2026/05/01/story">Econ</a>
    """

    links = extract_supported_links(html)

    assert [link.source for link in links] == ["MIT Technology Review", "The Atlantic"]
    assert links[0].url == "https://www.technologyreview.com/2026/05/01/ai"


def test_extract_skipped_links_finds_nyt_and_economist() -> None:
    html = """
    <a href="https://www.nytimes.com/2026/05/01/technology/story.html">NYT</a>
    <a href="https://www.economist.com/science-and-technology/2026/05/01/story">Econ</a>
    <a href="https://www.theatlantic.com/technology/archive/2026/05/story/">Atlantic</a>
    """

    assert extract_skipped_links(html) == [
        "https://www.nytimes.com/2026/05/01/technology/story.html",
        "https://www.economist.com/science-and-technology/2026/05/01/story",
    ]

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import trafilatura

from newsletter_podcast.clean import format_podcast_title

logger = logging.getLogger(__name__)


class ArticleFetchError(RuntimeError):
    pass


class ArticleTextTooShortError(ValueError):
    pass


@dataclass(frozen=True)
class FetchedArticle:
    title: str
    publisher: str
    text: str

    @property
    def podcast_title(self) -> str:
        return format_podcast_title(self.title, self.publisher)


def fetch_article(url: str, fallback_title: str, publisher: str) -> FetchedArticle:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ArticleFetchError(f"Could not fetch article URL: {url}")
    metadata = trafilatura.extract_metadata(downloaded)
    extracted = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        favor_recall=True,
    )
    text = _clean_text(extracted or "")
    word_count = len(re.findall(r"\b\w+\b", text))
    if word_count < 200:
        raise ArticleTextTooShortError(f"Article text has {word_count} words")
    title = (metadata.title if metadata and metadata.title else fallback_title).strip()
    return FetchedArticle(
        title=title or fallback_title or "Untitled article",
        publisher=publisher,
        text=text,
    )


def _clean_text(text: str) -> str:
    text = re.sub(r"\u200b|\ufeff", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

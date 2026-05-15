from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

TRACKING_PARAMS = {
    "campaign_id",
    "emc",
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "smid",
    "utm_campaign",
    "utm_content",
    "utm_medium",
    "utm_source",
    "utm_term",
}

SYNTHESIZED_SOURCES = {
    "technologyreview.com": "MIT Technology Review",
    "www.technologyreview.com": "MIT Technology Review",
    "theatlantic.com": "The Atlantic",
    "www.theatlantic.com": "The Atlantic",
}

SKIPPED_AUDIO_APP_SOURCES = {
    "economist.com",
    "www.economist.com",
    "nytimes.com",
    "www.nytimes.com",
}


@dataclass(frozen=True)
class ArticleLink:
    url: str
    title: str
    source: str


def is_weekly_picks(subject: str, sender: str) -> bool:
    lower_subject = subject.lower()
    is_weekly = "weekly pick" in lower_subject or "weekly pic" in lower_subject
    configured_sender = os.environ.get("WEEKLY_PICKS_SENDER", "").strip().lower()
    return is_weekly and bool(configured_sender) and configured_sender in sender.lower()


def extract_supported_links(html_body: str) -> list[ArticleLink]:
    parser = _LinkParser()
    parser.feed(html_body)
    links: list[ArticleLink] = []
    seen: set[str] = set()
    for url, text in parser.links:
        normalized_url = normalize_url(url)
        if not normalized_url or normalized_url in seen:
            continue
        source = source_for_url(normalized_url)
        if not source:
            continue
        seen.add(normalized_url)
        links.append(
            ArticleLink(
                url=normalized_url,
                title=clean_link_title(text),
                source=source,
            )
        )
    return links


def source_for_url(url: str) -> str | None:
    host = urlparse(url).netloc.lower()
    return SYNTHESIZED_SOURCES.get(host)


def skipped_source_for_url(url: str) -> str | None:
    host = urlparse(url).netloc.lower()
    if host in SKIPPED_AUDIO_APP_SOURCES:
        return host.removeprefix("www.")
    return None


def extract_skipped_links(html_body: str) -> list[str]:
    parser = _LinkParser()
    parser.feed(html_body)
    skipped: list[str] = []
    seen: set[str] = set()
    for url, _text in parser.links:
        normalized_url = normalize_url(url)
        if not normalized_url or normalized_url in seen:
            continue
        if skipped_source_for_url(normalized_url):
            skipped.append(normalized_url)
            seen.add(normalized_url)
    return skipped


def normalize_url(url: str) -> str | None:
    url = html.unescape(url).strip()
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    query = urlencode(
        [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if key.lower() not in TRACKING_PARAMS and not key.lower().startswith("utm_")
        ],
        doseq=True,
    )
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/") or "/",
            "",
            query,
            "",
        )
    )


def clean_link_title(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", html.unescape(text)).strip()
    return cleaned


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._current_href: str | None = None
        self._current_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attrs_by_name = {name.lower(): value for name, value in attrs}
        href = attrs_by_name.get("href")
        if href:
            self._current_href = href
            self._current_text = []

    def handle_data(self, data: str) -> None:
        if self._current_href:
            self._current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or not self._current_href:
            return
        self.links.append((self._current_href, "".join(self._current_text)))
        self._current_href = None
        self._current_text = []

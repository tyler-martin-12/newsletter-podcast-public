from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from email.utils import parseaddr

import trafilatura

logger = logging.getLogger(__name__)


class TextTooShortError(ValueError):
    pass


@dataclass(frozen=True)
class CleanArticle:
    title: str
    author: str
    text: str
    newsletter: str

    @property
    def podcast_title(self) -> str:
        return format_podcast_title(self.title, self.newsletter)


COMMON_PREFIX_RE = re.compile(r"^(\[[^\]]{1,80}\]\s*|(?:newsletter|post):\s*)+", re.I)
SUBJECT_NOISE_RE = re.compile(r"^(fwd:\s*|re:\s*)+", re.I)
SPACE_RE = re.compile(r"[ \t]+")
BLANK_RE = re.compile(r"\n{3,}")
URL_RE = re.compile(r"https?://\S+")

NEWSLETTER_BY_SENDER = {
    "casey@platformer.news": "Platformer",
    "imagine@theconversation.com": "Imagine",
    "importai@substack.com": "Import AI",
    "info@editorial.theguardian.com": "The Guardian",
    "londoncentric@substack.com": "London Centric",
    "newsletters@technologyreview.com": "MIT Technology Review",
    "noreply@news.bloomberg.com": "Money Stuff",
    "noreply@wheresyoured.at": "Where's Your Ed At",
    "uk.newsletter@theconversation.com": "The Conversation",
    "wired@newsletters.wired.com": "WIRED",
}

ARTICLE_TRUNCATION_MARKERS = [
    "\nKeep Reading\n",
    "\nMost Popular\n",
    "\nStay connected\n",
    "\nSponsored\n",
    "\nFollowing\n",
    "\nSide Quests\n",
    "\nThose good posts\n",
    "\nTalk to us\n",
    "\nDeep Dive\n",
    "\nExplore more\n",
    "\nAbout the Author\n",
    "\nRelated:\n",
    "\nHere are three new stories from The Atlantic:\n",
    "\nThe wider TechScape\n",
    "\nLondon Centric has got lots of original stories",
    "\nPS ",
    "\nLet me know yours by replying",
    "\nIf you have any questions or comments about any of our newsletters",
    "\nThis week I want to hear",
    "\nThis week we want you to",
    "\nLast week we asked",
    "\nLast week, we asked",
    "\nLast week, environment editor",
    "\nIf you think we missed one",
    "\nIf you enjoy Imagine",
    "\nYou are receiving this email because",
    "\nYou're receiving this email because",
    "\nYou’re receiving this email because",
    "\nIf you'd like to get Money Stuff",
    "\nLike getting this newsletter?",
    "\nP.S. If you've ever wanted",
    "\nIn this week’s Big Story",
    "\nIn this week's Big Story",
    "\nWhat did you think about today’s newsletter?",
    "\nWhat do you think of",
    "\nTell us about your favorite WIRED",
    "\nOn TikTok,",
    "\n● ",
    "\nThanks to Andrew Sullivan",
    "\nIn the Harvard Business Review Analytic Services session",
    "\nExclusive offer for MIT Technology Review readers",
    "\nSubscribe today to lock in",
    "\nGet a Presale ticket",
]

ARTICLE_METADATA_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in [
        r"^\s*$",
        r"^By\s+",
        r"^Follow\s*$",
        r"^Published:\s+",
        r"^Authors?\s*$",
        r"^Partners\s*$",
        r"^Disclosure statement\s*$",
        r"^DOI\s*$",
        r"^CC BY",
        r"^Share article\s*$",
        r"^Print article\s*$",
        r"^Subscribe\s*$",
        r"^Sign In\s*$",
        r"^English Edition\s*$",
        r"^Print Edition\s*$",
        r"^Latest Headlines\s*$",
        r"^Puzzles\s*$",
        r"^More\s*$",
        r"^Opinion\s*$",
        r"^Commentary\s*$",
        r"^View the full list\s*$",
        r"^Republish this article\s*$",
        r"^Explore more on these topics\s*$",
        r"^Most viewed\s*$",
        r"^Related Story\s*$",
        r"^Popular\s*$",
        r"^Keep Reading\s*$",
        r"^Most Popular\s*$",
        r"^Stay connected\s*$",
        r"^Skip to Content\s*$",
        r"^Share story on (?:linkedin|facebook|email)\s*$",
        r"^\d+\s+(minutes?|hours?|days?)\s+ago$",
    ]
]

ARTICLE_DROP_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in [
        r"^Before it.?s here, it.?s on the Bloomberg Terminal\.",
        r"^Read more:",
        r"^Appeared in the .* print edition",
        r"^Copyright",
        r"^Get the Newsletter$",
        r"^To dig further into the archive and support the Guardian",
        r"^About Free Expression$",
        r"^Editor:\s+.*$",
        r"^Copyeditor:\s+.*$",
        r"^Tagged:.*$",
        r"^Support Longreads$",
        r"^Follow us on (?:Twitter|Tumblr|Facebook|Instagram)$",
        r"^You.?re receiving this email because",
        r"^You are receiving this email because",
        r"^If you have any questions or comments about any of our newsletters",
        r"^Let me know yours by replying",
        r"^This week I want to hear",
        r"^This week we want you to",
        r"^A clear-eyed view of the tech news",
        r"^WIRED editor at large",
        r"^In the Harvard Business Review Analytic Services session",
        r"^Exclusive offer for MIT Technology Review readers",
        r"^Subscribe today to lock in",
        r"^Get a Presale ticket",
        r"^London Centric.?s paying subscribers fund all our journalism",
        r"^Want to get in touch with London Centric",
        r"^Click here to claim",
        r"^Thank you to everyone who supports original local journalism",
        r"^You.?re currently a free subscriber to London Centric",
        r"^If you liked this piece, please subscribe",
        r"^I also just did a piece",
        r"^Subscribing to premium is both great value",
        r"^What did you think about today.?s newsletter",
        r"^Tell us about your favorite WIRED",
        r"^What do you think of .* Send an email",
        r"^Welcome to Import AI,",
        r"^Welcome to Something Good,",
        r"^You.?re reading the Imagine newsletter",
        r"^The Conversation is made up of",
        r"^Together, we cover the planet",
        r"^Sidebar: If you.?re interested .* please subscribe",
        r"^This email features references to books",
    ]
]

IMAGE_INDICATOR_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in [
        r"^image$",
        r"^View image in fullscreen",
        r"^View full image",
        r"^Photograph[:\s]",
        r"^Photo[:\s]",
        r"^Image[:\s]",
        r"^Caption[:\s]",
        r"^Credit[:\s]",
        r"^Courtesy of .*$",
        r".*/Author provided.*$",
        r".*\b(?:Getty Images|Shutterstock|Wikimedia Commons|Zuma Press)\b.*$",
    ]
]

IMAGE_CAPTION_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in [
        r"^Image of ",
        r"^Photo of ",
        r"^Photograph of ",
        r"^Picture of ",
        r"^A photo of ",
        r"^An image of ",
        r"^A photo illustration ",
        r"^An illustration ",
        r"^Illustration by ",
    ]
]

MONEY_STUFF_HEADER_EXCLUDE_STARTS = {
    "also",
    "and",
    "anyway",
    "because",
    "but",
    "for",
    "here",
    "i",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "there",
    "these",
    "this",
    "we",
    "what",
    "when",
    "where",
    "why",
    "you",
}

STRIP_LINE_PATTERNS = [
    re.compile(r"^\s*view (this )?(email )?in (your )?browser\s*$", re.I),
    re.compile(r"^\s*(unsubscribe|manage preferences|update your preferences)\b.*$", re.I),
    re.compile(r"^\s*sponsored by\b.*$", re.I),
    re.compile(r"^\s*\[?image\]?\s*$", re.I),
    re.compile(r"^\s*(follow us|follow me|share this|forwarded this email)\b.*$", re.I),
    re.compile(r"^\s*(facebook|twitter|x|linkedin|instagram|threads)\s*$", re.I),
    re.compile(r"^\s*open tracking pixel\s*$", re.I),
]

DOMAIN_LINE_PATTERNS = {
    "substack.com": [
        re.compile(r"^\s*you('re| are) receiving this because\b.*$", re.I),
        re.compile(r"^\s*like this post\?\s*$", re.I),
    ],
    "stratechery.com": [
        re.compile(r"^\s*this daily update is for subscribers\b.*$", re.I),
        re.compile(r"^\s*stratechery update\b.*$", re.I),
    ],
    "bloomberg.com": [
        re.compile(r"^\s*before it'?s here, it'?s on the bloomberg terminal\b.*$", re.I),
        re.compile(r"^\s*bloomberg may send me offers\b.*$", re.I),
    ],
    "platformer.news": [
        re.compile(r"^\s*sent to you by platformer\b.*$", re.I),
        re.compile(r"^\s*thanks for reading platformer\b.*$", re.I),
    ],
}


def clean_article(subject: str, sender: str, html: str) -> CleanArticle:
    extracted = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        favor_recall=True,
    )
    text = extracted or _html_fallback(html)
    text = _post_process(text, sender)
    text = _clean_article_text(text)
    word_count = len(re.findall(r"\b\w+\b", text))
    if word_count < 200:
        logger.warning(
            "Rejecting newsletter below word threshold",
            extra={"subject": subject, "sender": sender, "word_count": word_count},
        )
        raise TextTooShortError(f"Cleaned text has {word_count} words")
    return CleanArticle(
        title=derive_title(subject),
        author=derive_author(sender),
        text=text,
        newsletter=derive_newsletter_title(subject, sender),
    )


def derive_title(subject: str) -> str:
    title = SUBJECT_NOISE_RE.sub("", subject).strip()
    title = COMMON_PREFIX_RE.sub("", title).strip()
    title = re.sub(r"\s+", " ", title)
    return title or "Untitled newsletter"


def derive_newsletter_title(subject: str, sender: str) -> str:
    bracket_match = re.match(r"^\[([^\]]{1,80})\]", SUBJECT_NOISE_RE.sub("", subject).strip())
    if bracket_match:
        return bracket_match.group(1).strip()
    sender_name, sender_email = parseaddr(sender)
    mapped = NEWSLETTER_BY_SENDER.get(sender_email.lower())
    if mapped:
        return mapped
    if sender_name:
        return sender_name.strip()
    domain = sender_email.split("@", 1)[1] if "@" in sender_email else sender_email
    domain_root = domain.split(".", 1)[0] if domain else "Newsletter"
    return domain_root.replace("-", " ").replace("_", " ").title()


def format_podcast_title(title: str, newsletter: str) -> str:
    clean_title = title.strip() or "Untitled newsletter"
    clean_newsletter = newsletter.strip()
    if not clean_newsletter:
        return clean_title
    if _contains_newsletter_title(clean_title, clean_newsletter):
        return clean_title
    return f"{clean_newsletter}: {clean_title}"


def _contains_newsletter_title(title: str, newsletter: str) -> bool:
    normalized_title = _normalize_title_token(title)
    normalized_newsletter = _normalize_title_token(newsletter)
    if not normalized_newsletter:
        return True
    return normalized_newsletter in normalized_title


def _normalize_title_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def derive_author(sender: str) -> str:
    name, email = parseaddr(sender)
    if name:
        return name
    local = email.split("@", 1)[0] if email else sender
    return local.replace(".", " ").replace("_", " ").title() or "Newsletter"


def _post_process(text: str, sender: str) -> str:
    domain_patterns = _patterns_for_sender(sender)
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = SPACE_RE.sub(" ", raw_line).strip()
        if not line:
            lines.append("")
            continue
        line = URL_RE.sub("", line).strip()
        if _should_strip(line, STRIP_LINE_PATTERNS + domain_patterns):
            continue
        if len(line) <= 3 and line.lower() in {"ad", "x"}:
            continue
        lines.append(line)
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\u200b|\ufeff", "", cleaned)
    cleaned = BLANK_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def _clean_article_text(text: str) -> str:
    is_money_stuff = "Money Stuff" in text
    text = _truncate_after(text, ARTICLE_TRUNCATION_MARKERS)
    lines = [line.strip() for line in text.splitlines()]
    lines = _remove_image_lines(lines)
    lines = _remove_related_story_teasers(lines)
    lines = [
        line
        for line in lines
        if _keep_article_line(line, is_money_stuff=is_money_stuff)
    ]
    lines = _trim_article_edges(lines, is_money_stuff=is_money_stuff)
    if is_money_stuff:
        lines = [
            f"HEADER - {line.strip()}" if _is_money_stuff_header(line) else line
            for line in lines
        ]
    lines = _dedupe_lines(lines)
    cleaned = "\n".join(lines)
    cleaned = URL_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+Read Kelly’s piece.*?comment below the article\.", "", cleaned)
    cleaned = re.sub(r"■|…|\.{3}|\[\d+\]", "", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = BLANK_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def _truncate_after(text: str, markers: list[str]) -> str:
    earliest = None
    for marker in markers:
        index = text.find(marker)
        if index != -1:
            earliest = index if earliest is None else min(earliest, index)
    return text[:earliest].rstrip() if earliest is not None else text


def _remove_related_story_teasers(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    index = 0
    while index < len(lines):
        if lines[index].strip().lower() == "related story":
            index += 1
            while index < len(lines):
                candidate = lines[index].strip()
                if not candidate:
                    index += 1
                    continue
                if len(candidate) >= 100:
                    break
                index += 1
            continue
        cleaned.append(lines[index])
        index += 1
    return cleaned


def _remove_image_lines(lines: list[str]) -> list[str]:
    indicator_indices = {
        index for index, line in enumerate(lines) if _is_image_indicator(line)
    }
    caption_indices: set[int] = set()
    for index in indicator_indices:
        previous = index - 1
        while previous >= 0 and not lines[previous].strip():
            previous -= 1
        if previous >= 0:
            caption_indices.add(previous)
    drop = indicator_indices | caption_indices
    return [line for index, line in enumerate(lines) if index not in drop]


def _keep_article_line(line: str, is_money_stuff: bool) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _matches_any(stripped, ARTICLE_METADATA_PATTERNS):
        return False
    if _matches_any(stripped, ARTICLE_DROP_PATTERNS):
        return False
    if _is_image_caption(stripped):
        return False
    if re.match(r"^\[\d+\]", stripped):
        return False
    is_short_money_header = is_money_stuff and _is_money_stuff_header(stripped)
    if len(stripped) < 60 and not is_short_money_header:
        return False
    if is_short_money_header:
        return True
    if _looks_like_caption_or_promo(stripped):
        return False
    return not (is_money_stuff and _looks_like_headline_digest(stripped))


def _trim_article_edges(lines: list[str], is_money_stuff: bool) -> list[str]:
    trimmed = list(lines)
    while trimmed and not _is_body_candidate(trimmed[0], is_money_stuff):
        trimmed.pop(0)
    while trimmed and not _is_body_candidate(trimmed[-1], is_money_stuff):
        trimmed.pop()
    return trimmed


def _is_body_candidate(line: str, is_money_stuff: bool) -> bool:
    stripped = line.strip()
    if is_money_stuff and _is_money_stuff_header(stripped):
        return True
    if len(stripped) < 60:
        return False
    if _matches_any(stripped, ARTICLE_METADATA_PATTERNS + ARTICLE_DROP_PATTERNS):
        return False
    return stripped.endswith((".", "!", "?", "\"", "”", "'", "’")) or stripped.count(". ") >= 1


def _looks_like_caption_or_promo(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) >= 120:
        return False
    if re.match(r"^[a-z][a-z\s,-]+$", stripped) and len(stripped.split()) >= 6:
        return True
    if any(character in stripped for character in ['"', "“", "”", ":", ";"]):
        return False
    words = stripped.split()
    if len(words) < 5:
        return True
    alpha_words = [
        word.strip(".,'’()")
        for word in words
        if any(character.isalpha() for character in word)
    ]
    capitalized_words = sum(1 for word in alpha_words if word[:1].isupper())
    if stripped.endswith((".", "!", "?")):
        return len(alpha_words) >= 6 and capitalized_words >= len(alpha_words) * 0.5
    return capitalized_words >= max(3, len(alpha_words) // 2)


def _looks_like_headline_digest(line: str) -> bool:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", line)
        if sentence.strip()
    ]
    if len(sentences) < 6:
        return False
    word_counts = [len(sentence.split()) for sentence in sentences]
    short_sentence_ratio = sum(count <= 14 for count in word_counts) / len(word_counts)
    return short_sentence_ratio >= 0.7


def _is_image_indicator(line: str) -> bool:
    stripped = line.strip()
    return _matches_any(stripped, IMAGE_INDICATOR_PATTERNS) or _is_uppercase_credit_line(stripped)


def _is_image_caption(line: str) -> bool:
    return _matches_any(line.strip(), IMAGE_CAPTION_PATTERNS)


def _is_uppercase_credit_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped.split()) > 4:
        return False
    letters = [character for character in stripped if character.isalpha()]
    return bool(letters) and stripped == stripped.upper()


def _is_money_stuff_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) < 3 or len(stripped) > 60:
        return False
    if stripped.endswith((".", ":", "?", "!", "\"", "”", "'", "’")):
        return False
    blocked_characters = [",", ";", "(", ")", "[", "]", "/", "“", "”", "\""]
    if any(character in stripped for character in blocked_characters):
        return False
    words = re.findall(r"[A-Za-z0-9&'-]+", stripped)
    if not words or len(words) > 5:
        return False
    if words[0].lower() in MONEY_STUFF_HEADER_EXCLUDE_STARTS:
        return False
    return not any(len(word) > 20 for word in words)


def _matches_any(line: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(line) for pattern in patterns)


def _dedupe_lines(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for line in lines:
        if line in seen:
            continue
        deduped.append(line)
        seen.add(line)
    return deduped


def _patterns_for_sender(sender: str) -> list[re.Pattern[str]]:
    lower = sender.lower()
    patterns: list[re.Pattern[str]] = []
    for domain, domain_patterns in DOMAIN_LINE_PATTERNS.items():
        if domain in lower:
            patterns.extend(domain_patterns)
    return patterns


def _should_strip(line: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(line) for pattern in patterns)


def _html_fallback(html: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p>", "\n\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text)

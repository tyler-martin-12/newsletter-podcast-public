from __future__ import annotations

from pathlib import Path

import pytest

from newsletter_podcast.clean import (
    TextTooShortError,
    clean_article,
    derive_newsletter_title,
    derive_title,
    format_podcast_title,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    ("fixture", "sender", "subject"),
    [
        ("substack.html", "Markets <notes@substack.com>", "[Markets] The Week in Markets"),
        ("beehiiv.html", "Product Notes <hello@beehiiv.com>", "Product Notes: Roadmap Review"),
        ("mailchimp.html", "City Desk <news@mailchimpapp.net>", "[City Desk] City Briefing"),
    ],
)
def test_clean_article_strips_common_newsletter_chrome(
    fixture: str,
    sender: str,
    subject: str,
) -> None:
    html = (FIXTURES / fixture).read_text(encoding="utf-8")
    article = clean_article(subject, sender, html)

    assert len(article.text.split()) >= 200
    assert "Unsubscribe" not in article.text
    assert "Manage preferences" not in article.text
    assert "Sponsored by" not in article.text
    assert "[image]" not in article.text
    assert article.title
    assert article.author


def test_derive_title_removes_common_prefixes() -> None:
    assert derive_title("Fwd: [Platformer] A Better Internet") == "A Better Internet"


def test_derive_newsletter_title_prefers_subject_prefix() -> None:
    assert (
        derive_newsletter_title(
            "Fwd: [Platformer] A Better Internet",
            "Casey Newton <casey@platformer.news>",
        )
        == "Platformer"
    )


def test_format_podcast_title_adds_newsletter_when_missing() -> None:
    assert format_podcast_title("A Better Internet", "Platformer") == (
        "Platformer: A Better Internet"
    )


def test_format_podcast_title_does_not_duplicate_newsletter() -> None:
    assert format_podcast_title("Money Stuff: Private Credit", "Money Stuff") == (
        "Money Stuff: Private Credit"
    )


def test_clean_article_rejects_short_text() -> None:
    with pytest.raises(TextTooShortError):
        clean_article("Short", "Brief <brief@example.com>", "<p>Unsubscribe</p><p>Tiny note.</p>")


def test_article_cleaner_removes_captions_teasers_and_duplicate_lines() -> None:
    body = " ".join(["This paragraph is clearly part of the article body."] * 18)
    duplicate = " ".join(["This duplicate paragraph should only appear one time."] * 12)
    html = f"""
    <article>
      <p>Navigation</p>
      <p>A photo of a trader standing near a screen of market prices</p>
      <p>Getty Images</p>
      <p>{body}</p>
      <p>Related Story</p>
      <p>Short teaser headline</p>
      <p>{duplicate}</p>
      <p>{duplicate}</p>
      <p>Keep Reading</p>
      <p>This footer should not remain in the final text.</p>
    </article>
    """

    article = clean_article("Article", "Editor <editor@example.com>", html)

    assert "Getty Images" not in article.text
    assert "Short teaser headline" not in article.text
    assert article.text.count(duplicate) == 1
    assert "This footer should not remain" not in article.text


def test_money_stuff_keeps_short_section_headers() -> None:
    paragraph = " ".join(
        [
            "Money Stuff explains this market structure problem with enough context and detail "
            "that it reads like a real paragraph rather than a compact headline digest."
        ]
        * 10
    )
    html = f"""
    <article>
      <p>Money Stuff</p>
      <p>Private Funds</p>
      <p>{paragraph}</p>
      <p>Before it's here, it's on the Bloomberg Terminal.</p>
      <p>{paragraph} Second body paragraph with enough substance to pass threshold.</p>
    </article>
    """

    article = clean_article(
        "Money Stuff: Private Funds",
        "Matt Levine <newsletter@bloomberg.com>",
        html,
    )

    assert "HEADER - Private Funds" in article.text
    assert "Bloomberg Terminal" not in article.text


def test_article_cleaner_truncates_reader_prompts_and_roundups() -> None:
    body = " ".join(["The main article paragraph has enough substance for spoken audio."] * 24)
    html = f"""
    <article>
      <p>{body}</p>
      <p>Last week, we asked readers to send us their thoughts.</p>
      <p>This reader response section should be removed.</p>
    </article>
    """

    article = clean_article("Main story", "Editor <editor@example.com>", html)

    assert "main article paragraph" in article.text
    assert "Last week" not in article.text
    assert "reader response section" not in article.text


def test_article_cleaner_removes_import_ai_intro() -> None:
    body = " ".join(["AI systems are becoming more capable across research workflows."] * 26)
    intro = (
        "Welcome to Import AI, a newsletter about AI research. Import AI runs on arXiv "
        "and feedback from readers. If you'd like to support this, please subscribe."
    )
    html = f"""
    <article>
      <p>{intro}</p>
      <p>{body}</p>
    </article>
    """

    article = clean_article(
        "Import AI 455: AI systems are about to start building themselves.",
        "importai@substack.com",
        html,
    )

    assert "Welcome to Import AI" not in article.text
    assert "AI systems are becoming more capable" in article.text


def test_article_cleaner_removes_inline_comment_prompt() -> None:
    body = " ".join(["The main article paragraph has enough substance for spoken audio."] * 24)
    prompt = (
        "Read Kelly’s piece and then share your thoughts. Will we continue to find "
        "new endeavors, or will we learn how to be jobless humans on planet Earth? "
        "Send me an email or comment below the article."
    )
    html = f"<article><p>{body} {prompt}</p></article>"

    article = clean_article("Robots", "Wired <wired@example.com>", html)

    assert "main article paragraph" in article.text
    assert "Read Kelly" not in article.text
    assert "comment below" not in article.text

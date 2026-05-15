from __future__ import annotations

from newsletter_podcast.synthesize import chunk_text


def test_chunk_text_keeps_paragraph_boundaries() -> None:
    text = "\n\n".join(["first paragraph " * 20, "second paragraph " * 20, "third paragraph " * 20])

    chunks = chunk_text(text, max_chars=400)

    assert len(chunks) == 3
    assert all(len(chunk) <= 400 for chunk in chunks)
    assert chunks[0].startswith("first paragraph")


def test_chunk_text_splits_long_paragraph() -> None:
    text = "word " * 1000

    chunks = chunk_text(text, max_chars=500)

    assert len(chunks) > 1
    assert all(len(chunk) <= 500 for chunk in chunks)

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def load_eval_module() -> Any:
    script_path = Path(__file__).parents[1] / "tools" / "eval_whisper_wer.py"
    spec = importlib.util.spec_from_file_location("eval_whisper_wer", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load eval_whisper_wer.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_whisper_wer"] = module
    spec.loader.exec_module(module)
    return module


eval_wer = load_eval_module()


def test_ordinal_digits_match_words() -> None:
    left = eval_wer.normalize_text("4th and 5th generation", "standard")
    right = eval_wer.normalize_text("fourth and fifth generation", "standard")

    assert left.tokens == right.tokens
    assert eval_wer.align_tokens(left.tokens, right.tokens).wer == 0


def test_unicode_punctuation_does_not_create_substitutions() -> None:
    left = eval_wer.normalize_text("“people” said it’s next-generation", "standard")
    right = eval_wer.normalize_text("people said it's next generation", "standard")

    assert left.tokens == right.tokens
    assert eval_wer.align_tokens(left.tokens, right.tokens).wer == 0


def test_window_search_does_not_keep_long_reference_tail() -> None:
    prefix = " ".join(f"prefix{i}" for i in range(80))
    article = " ".join(f"article{i}" for i in range(80))
    tail = " ".join(f"tail{i}" for i in range(120))
    source = f"{prefix} {article} {tail}"
    transcript = article

    normalized_source = eval_wer.normalize_text(source, "standard")
    normalized_transcript = eval_wer.normalize_text(transcript, "standard")
    window = eval_wer.find_best_reference_window(
        source,
        normalized_source,
        normalized_transcript,
        "standard",
    )
    alignment = eval_wer.align_tokens(window.normalized.tokens, normalized_transcript.tokens)

    assert alignment.deletions == 0
    assert alignment.wer == 0
    assert window.normalized.tokens == normalized_transcript.tokens


def test_boundary_trim_removes_excerpt_edge_deletions() -> None:
    source = "alpha beta gamma delta extra tail words"
    window = eval_wer.WindowResult(
        start_word=0,
        end_word=7,
        raw_text=source,
        normalized=eval_wer.normalize_text(source, "standard"),
    )
    transcript = eval_wer.normalize_text("alpha beta gamma delta", "standard")
    alignment = eval_wer.align_tokens(window.normalized.tokens, transcript.tokens)

    trimmed_window, trimmed_alignment = eval_wer.trim_window_boundaries(
        source,
        window,
        alignment,
    )

    assert trimmed_window.normalized.tokens == transcript.tokens
    assert trimmed_alignment.deletions == 0
    assert trimmed_alignment.wer == 0

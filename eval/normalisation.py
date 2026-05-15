from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import ssl
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROMPT_VERSION = "normalisation-v5"
TARGETED_PROMPT_VERSION = "targeted-normalisation-v2"
DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_PROFILES_DIR = Path(__file__).parent / "source_profiles"
TOKEN_RE = re.compile(r"\S+")
INITIALISM_RE = re.compile(r"^(?:[A-Z]\.){2,}$|^[A-Z]{2,}$")
UNIT_RE = re.compile(
    r"^(?:GW|MW|kW|Wh|kWh|mg|g|kg|km|m|cm|mm|mph|mpg|bn|m|tn|bbl|CO2|CO₂)$",
    re.IGNORECASE,
)
DATE_RE = re.compile(
    r"\b(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|"
    r"Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)\b",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"\d")
CURRENCY_RE = re.compile(r"^(?:USD|EUR|GBP|JPY|AUD|CAD|CHF)$")
CATEGORIES = ("initialism", "unit", "date", "number", "currency")
SPOKEN_AS_LETTERS = {
    "AI",
    "AIs",
    "CEO",
    "CEO/CTO",
    "CTO",
    "DLP",
    "DSPM",
    "IDS/IPS",
    "IMF",
    "IRS",
    "LAPD",
    "M.B.A.",
    "MBA",
    "MIT",
    "SSL/IPSec",
    "UC",
    "USC",
    "USPTO",
}


@dataclass(frozen=True)
class TokenSpan:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class ArticleInput:
    episode_id: str
    source: str
    title: str
    source_text_path: Path
    raw_transcript: str


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate LLM text normalisation headroom")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--mode", choices=["generic", "targeted"], default="generic")
    parser.add_argument("--model", default=os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL))
    parser.add_argument("--article-id", action="append", dest="article_ids")
    parser.add_argument("--existing-results", type=Path)
    parser.add_argument("--use-profiles", action="store_true")
    parser.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES_DIR)
    parser.add_argument("--generic-results", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-api", action="store_true")
    args = parser.parse_args(argv)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if args.out_dir is None:
        args.out_dir = Path(
            "results/normalisation_targeted"
            if args.mode == "targeted"
            else "results/normalisation"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    articles = load_articles(args.manifest, args.details)
    if args.article_ids:
        wanted = set(args.article_ids)
        articles = [article for article in articles if article.episode_id in wanted]
    if args.limit is not None:
        articles = articles[: args.limit]

    results = run_normalisation_eval(
        articles=articles,
        out_dir=args.out_dir,
        model=args.model,
        api_key=api_key,
        skip_api=args.skip_api,
        mode=args.mode,
        use_profiles=args.use_profiles,
        profiles_dir=args.profiles_dir,
    )
    if args.generic_results and args.mode == "targeted":
        attach_generic_results(results, args.generic_results)
    write_article_summary(args.out_dir / "article_summary.csv", results)
    write_augmented_results_csv(
        source_path=args.existing_results or args.details.with_name("results.csv"),
        output_path=args.out_dir / "results.csv",
        rows=results,
    )
    write_source_summary(args.out_dir / "summary.csv", results)
    write_category_contribution(args.out_dir / "category_contribution.csv", results)
    if args.generic_results and args.mode == "generic":
        write_profile_comparison(
            path=args.out_dir / "profile_comparison.csv",
            generic_path=args.generic_results,
            profiled_rows=results,
        )
    if args.generic_results and args.mode == "targeted":
        write_targeted_comparison(
            path=args.out_dir / "targeted_comparison.csv",
            generic_path=args.generic_results,
            targeted_rows=results,
        )
    flagged = sum(1 for result in results if result["flagged"])
    print(f"Processed {len(results)} articles")
    print(f"Flagged {flagged} articles above 15 percent token difference")
    print(f"Wrote {args.out_dir}")
    return 0


def run_normalisation_eval(
    articles: list[ArticleInput],
    out_dir: Path,
    model: str,
    api_key: str | None,
    skip_api: bool = False,
    mode: str = "generic",
    use_profiles: bool = False,
    profiles_dir: Path = DEFAULT_PROFILES_DIR,
) -> list[dict[str, Any]]:
    eval_wer = load_eval_wer_module()
    results: list[dict[str, Any]] = []
    for article in articles:
        article_dir = out_dir / article.episode_id
        article_dir.mkdir(parents=True, exist_ok=True)
        raw_text = article.source_text_path.read_text(encoding="utf-8")
        if mode == "targeted":
            normalised_text = load_or_create_targeted_text(
                raw_text=raw_text,
                source=article.source,
                article_dir=article_dir,
                model=model,
                api_key=api_key,
                skip_api=skip_api,
                use_profile=use_profiles,
                profiles_dir=profiles_dir,
            )
        else:
            normalised_text = load_or_create_normalised_text(
                raw_text=raw_text,
                source=article.source,
                article_dir=article_dir,
                model=model,
                api_key=api_key,
                skip_api=skip_api,
                use_profile=use_profiles,
                profiles_dir=profiles_dir,
            )
        diff_records = diff_texts(raw_text, normalised_text)
        write_jsonl(article_dir / "diff.jsonl", diff_records)

        raw_eval = evaluate_reference(eval_wer, raw_text, article.raw_transcript)
        normalised_eval = evaluate_reference(eval_wer, normalised_text, article.raw_transcript)
        category_results = {
            category: evaluate_reference(
                eval_wer,
                apply_category(raw_text, diff_records, {category}),
                article.raw_transcript,
            )
            for category in CATEGORIES
        }
        token_delta = token_difference_ratio(raw_text, normalised_text)
        raw_wer = raw_eval["wer"]
        normalised_wer = normalised_eval["wer"]
        delta = raw_wer - normalised_wer
        result = {
            "episode_id": article.episode_id,
            "source": article.source,
            "title": article.title,
            "raw_wer": raw_wer,
            "normalised_wer": normalised_wer,
            "delta": delta,
            "delta_pct_of_raw": delta / raw_wer if raw_wer else 0.0,
            "flagged": token_delta > 0.15,
            "token_difference_ratio": token_delta,
            "total_substitutions": len(diff_records),
            "substitutions_by_category": dict(
                Counter(record["category"] for record in diff_records)
            ),
            "diff_path": str(article_dir / "diff.jsonl"),
            "raw_path": str(article_dir / "raw.txt"),
            "normalised_path": str(
                article_dir / ("targeted.txt" if mode == "targeted" else "normalised.txt")
            ),
            "targeted_path": str(article_dir / "targeted.txt") if mode == "targeted" else "",
            "targeted_wer": normalised_wer if mode == "targeted" else "",
            "targeted_delta": delta if mode == "targeted" else "",
            "category_wer": {
                category: value["wer"] for category, value in category_results.items()
            },
            "category_delta": {
                category: raw_wer - value["wer"] for category, value in category_results.items()
            },
        }
        results.append(result)
    return results


def load_articles(manifest_path: Path, details_path: Path) -> list[ArticleInput]:
    detail_rows = (
        json.loads(line)
        for line in details_path.read_text(encoding="utf-8").splitlines()
    )
    details = {row["episode_id"]: row for row in detail_rows if "error" not in row}
    articles: list[ArticleInput] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            episode_id = row["episode_id"]
            if episode_id not in details:
                continue
            source_path = Path(row["source_text_path"]).expanduser()
            articles.append(
                ArticleInput(
                    episode_id=episode_id,
                    source=row["source"],
                    title=row["title"],
                    source_text_path=source_path,
                    raw_transcript=str(details[episode_id]["raw_transcript"]),
                )
            )
    return articles


def load_or_create_normalised_text(
    raw_text: str,
    source: str,
    article_dir: Path,
    model: str,
    api_key: str | None,
    skip_api: bool,
    use_profile: bool = False,
    profiles_dir: Path = DEFAULT_PROFILES_DIR,
) -> str:
    raw_path = article_dir / "raw.txt"
    normalised_path = article_dir / "normalised.txt"
    raw_path.write_text(raw_text, encoding="utf-8")
    if normalised_path.exists():
        return normalised_path.read_text(encoding="utf-8")
    if skip_api:
        normalised_path.write_text(raw_text, encoding="utf-8")
        return raw_text
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required when normalised text is not cached")

    chunks = split_text(raw_text)
    profile = profile_for_prompt(load_source_profile(source, profiles_dir)) if use_profile else None
    normalised_chunks = [
        call_anthropic_cached(
            chunk=chunk,
            cache_dir=article_dir / "cache",
            model=model,
            api_key=api_key or "",
            profile=profile,
        )
        for chunk in chunks
    ]
    normalised_text = "\n\n".join(normalised_chunks).strip()
    normalised_path.write_text(normalised_text, encoding="utf-8")
    return normalised_text


def load_or_create_targeted_text(
    raw_text: str,
    source: str,
    article_dir: Path,
    model: str,
    api_key: str | None,
    skip_api: bool,
    use_profile: bool = False,
    profiles_dir: Path = DEFAULT_PROFILES_DIR,
) -> str:
    raw_path = article_dir / "raw.txt"
    targeted_path = article_dir / "targeted.txt"
    raw_path.write_text(raw_text, encoding="utf-8")
    if targeted_path.exists():
        return targeted_path.read_text(encoding="utf-8")
    if skip_api:
        replacements: list[dict[str, Any]] = []
    else:
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required when targeted text is not cached")
        profile = (
            profile_for_prompt(load_source_profile(source, profiles_dir))
            if use_profile
            else None
        )
        replacements = call_targeted_replacements_cached(
            text=raw_text,
            source=source,
            cache_dir=article_dir / "cache",
            model=model,
            api_key=api_key,
            profile=profile,
        )
    targeted_text, replacement_log = apply_targeted_replacements(raw_text, replacements)
    targeted_path.write_text(targeted_text, encoding="utf-8")
    write_jsonl(article_dir / "replacements.jsonl", replacement_log)
    return targeted_text


def call_targeted_replacements_cached(
    text: str,
    source: str,
    cache_dir: Path,
    model: str,
    api_key: str,
    profile: str | None = None,
) -> list[dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    profile_key = profile or ""
    key = hashlib.sha256(
        f"{TARGETED_PROMPT_VERSION}\n{model}\n{source}\n{profile_key}\n{text}".encode()
    ).hexdigest()
    cache_path = cache_dir / f"{key}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    prompt = build_targeted_prompt(text=text, source=source, profile=profile)
    response = call_anthropic_prompt(prompt, model, api_key, max_tokens=4096)
    replacements = parse_replacement_response(response)
    cache_path.write_text(json.dumps(replacements, ensure_ascii=False, indent=2), encoding="utf-8")
    return replacements


def parse_replacement_response(response: str) -> list[dict[str, Any]]:
    stripped = response.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Targeted replacement response was not valid JSON: {response}") from exc
    if not isinstance(data, list):
        raise RuntimeError("Targeted replacement response must be a JSON list")
    replacements: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        source_span = str(item.get("source_span", "")).strip()
        replacement = str(item.get("replacement", "")).strip()
        category = str(item.get("category", "")).strip()
        if not source_span or not replacement:
            continue
        if category not in {"unit", "initialism", "currency"}:
            category = "other"
        replacements.append(
            {
                "source_span": source_span,
                "replacement": replacement,
                "category": category,
                "context_before": str(item.get("context_before", "")),
                "context_after": str(item.get("context_after", "")),
                "rationale": str(item.get("rationale", "")),
            }
        )
    return replacements


def apply_targeted_replacements(
    raw_text: str,
    replacements: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    for replacement in replacements:
        skip_reason = targeted_skip_reason(replacement)
        if skip_reason:
            record = dict(replacement)
            record.update({"status": "skipped", "skip_reason": skip_reason})
            skipped.append(record)
            continue
        match = find_targeted_match(raw_text, replacement, occupied)
        record = dict(replacement)
        if match is None:
            record.update({"status": "skipped", "skip_reason": "not found unambiguously"})
            skipped.append(record)
        else:
            start, end = match
            record.update(
                {
                    "status": "applied",
                    "start": start,
                    "end": end,
                    "applied_source_span": raw_text[start:end],
                }
            )
            applied.append(record)
            occupied.append((start, end))
        occupied.sort()

    pieces: list[str] = []
    cursor = 0
    for record in sorted(applied, key=lambda item: int(item["start"])):
        start = int(record["start"])
        end = int(record["end"])
        pieces.append(raw_text[cursor:start])
        pieces.append(str(record["replacement"]))
        cursor = end
    pieces.append(raw_text[cursor:])
    return "".join(pieces), [*applied, *skipped]


def targeted_skip_reason(replacement: dict[str, Any]) -> str | None:
    source_span = str(replacement.get("source_span", "")).strip()
    replacement_text = str(replacement.get("replacement", "")).strip()
    category = str(replacement.get("category", "")).strip()
    if not source_span or not replacement_text:
        return "empty source or replacement"
    if source_span == replacement_text:
        return "replacement is identical to source span"
    if category == "initialism" and source_span in SPOKEN_AS_LETTERS:
        return "initialism is conventionally spoken as letters"
    if category not in {"unit", "initialism", "currency"}:
        return "category is outside targeted scope"
    return None


def find_targeted_match(
    raw_text: str,
    replacement: dict[str, Any],
    occupied: list[tuple[int, int]],
) -> tuple[int, int] | None:
    source_span = str(replacement["source_span"])
    context_before = str(replacement.get("context_before", ""))
    context_after = str(replacement.get("context_after", ""))
    matches: list[tuple[int, int]] = []
    start = 0
    while True:
        index = raw_text.find(source_span, start)
        if index == -1:
            break
        end = index + len(source_span)
        start = end
        if any(index < used_end and end > used_start for used_start, used_end in occupied):
            continue
        if context_before and not raw_text[:index].endswith(context_before):
            continue
        if context_after and not raw_text[end:].startswith(context_after):
            continue
        matches.append((index, end))
    if len(matches) == 1:
        return matches[0]
    if context_before or context_after:
        return None
    loose_matches = find_loose_matches(raw_text, source_span, occupied)
    return loose_matches[0] if len(loose_matches) == 1 else None


def find_loose_matches(
    raw_text: str,
    source_span: str,
    occupied: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    pattern = re.compile(rf"(?<!\w){re.escape(source_span)}(?!\w)")
    matches: list[tuple[int, int]] = []
    for match in pattern.finditer(raw_text):
        start, end = match.span()
        if any(start < used_end and end > used_start for used_start, used_end in occupied):
            continue
        matches.append((start, end))
    return matches


def normalise_with_profile(
    text: str,
    source: str,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    profiles_dir: Path = DEFAULT_PROFILES_DIR,
    cache_dir: Path | None = None,
) -> str:
    profile = profile_for_prompt(load_source_profile(source, profiles_dir))
    target_cache_dir = (
        cache_dir or Path("results/normalisation_per_source/cache") / safe_name(source)
    )
    chunks = split_text(text)
    return "\n\n".join(
        call_anthropic_cached(
            chunk=chunk,
            cache_dir=target_cache_dir,
            model=model,
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
            profile=profile,
        )
        for chunk in chunks
    ).strip()


def load_source_profile(source: str, profiles_dir: Path = DEFAULT_PROFILES_DIR) -> str:
    profile_path = profiles_dir / f"{source}.md"
    if not profile_path.exists():
        raise FileNotFoundError(f"Missing source profile: {profile_path}")
    return profile_path.read_text(encoding="utf-8").strip()


def profile_for_prompt(profile: str) -> str:
    return re.split(r"\nBoilerplate to strip:\n", profile, maxsplit=1)[0].strip()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def split_text(text: str, max_chars: int = 10_000) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text.strip())
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        addition = len(paragraph) + 2
        if current and current_len + addition > max_chars:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_len = 0
        current.append(paragraph)
        current_len += addition
    if current:
        chunks.append("\n\n".join(current).strip())
    return chunks


def call_anthropic_cached(
    chunk: str,
    cache_dir: Path,
    model: str,
    api_key: str,
    profile: str | None = None,
) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    profile_key = profile or ""
    key = hashlib.sha256(
        f"{PROMPT_VERSION}\n{model}\n{profile_key}\n{chunk}".encode()
    ).hexdigest()
    cache_path = cache_dir / f"{key}.txt"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    response = call_anthropic(chunk, model, api_key, profile)
    cache_path.write_text(response, encoding="utf-8")
    return response


def call_anthropic(text: str, model: str, api_key: str, profile: str | None = None) -> str:
    return call_anthropic_prompt(
        prompt=build_prompt(text, profile),
        model=model,
        api_key=api_key,
        max_tokens=max(1024, min(8192, len(text) // 3 + 1024)),
    )


def call_anthropic_prompt(prompt: str, model: str, api_key: str, max_tokens: int) -> str:
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required when normalised text is not cached")
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ],
    }
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120, context=tls_context()) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Anthropic request failed: {exc.code} {body}") from exc
    parts = [
        block.get("text", "")
        for block in data.get("content", [])
        if block.get("type") == "text"
    ]
    return "\n".join(parts).strip()


def tls_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def build_prompt(text: str, profile: str | None = None) -> str:
    profile_section = ""
    if profile:
        profile_section = (
            "Source-specific reading profile:\n"
            f"{profile}\n\n"
            "Use the profile only to resolve pronunciation, units, and "
            "source-specific conventions. The core rules below still control.\n\n"
            "Do not remove boilerplate, footers, links, headings, or sentences from the "
            "supplied text.\n\n"
        )
    return profile_section + (
        "Rewrite the following article text exactly as a competent human newsreader would "
        "read it aloud for text to speech.\n\n"
        "Rules:\n"
        "- Do not paraphrase.\n"
        "- Do not reorder text.\n"
        "- Do not add or remove sentences.\n"
        "- Preserve all proper nouns exactly.\n"
        "- Only substitute tokens where a human reader would say something different from "
        "the literal text.\n"
        "- Expand initialisms and unit abbreviations when context makes the expansion "
        "unambiguous, such as GW to gigawatts, EV to electric vehicle, and mg to milligrams.\n"
        "- Keep initialisms conventionally spoken as letters, such as BBC, FBI, CEO, AI.\n"
        "- Expand dates to spoken form, such as April 2, 2025 to April second, twenty "
        "twenty-five.\n"
        "- Expand numerics where convention demands, such as 1,000 to one thousand.\n"
        "- Keep digits where digit reading is natural, such as phone numbers or model numbers.\n"
        "- Return only the rewritten text, with no preamble.\n\n"
        "Text:\n"
        f"{text}"
    )


def build_targeted_prompt(text: str, source: str, profile: str | None = None) -> str:
    profile_section = ""
    if profile:
        profile_section = (
            "Optional source profile for pronunciation context:\n"
            f"{profile}\n\n"
        )
    return profile_section + (
        "You are preparing a narrow text-to-speech normalisation eval for "
        f"{source}.\n\n"
        "Return only a JSON list of replacement objects. Do not return prose, markdown, "
        "or code fences.\n\n"
        "Each object must have these fields:\n"
        "- source_span: the exact text span copied from the article.\n"
        "- replacement: the spoken-form replacement.\n"
        "- category: one of unit, initialism, currency.\n"
        "- context_before: exact nearby text before source_span, ideally 10 to 40 chars.\n"
        "- context_after: exact nearby text after source_span, ideally 10 to 40 chars.\n"
        "- rationale: a short reason.\n\n"
        "Only propose replacements for:\n"
        "- Unit abbreviations where context is unambiguous, such as GW, MW, kg, mg, km, "
        "kWh, CO2, or CO₂.\n"
        "- Initialisms that a newsreader would expand to words for speech, not ones "
        "conventionally spoken as letters.\n"
        "- Currency abbreviations in unusual forms.\n\n"
        "Do not touch:\n"
        "- Contractions.\n"
        "- AI, AIs, CEO, CTO, DLP, DSPM, IDS/IPS, IRS, LAPD, MIT, SSL/IPSec, USC, "
        "USPTO, or similar initialisms normally spoken as letters.\n"
        "- Standard dates and numbers.\n"
        "- Punctuation.\n"
        "- Quote marks.\n"
        "- Company names, people, places, or brands unless the replacement is only the "
        "spoken form of an abbreviation and the meaning is unambiguous.\n"
        "- Anything that already reads naturally as text.\n\n"
        "If there are no targeted replacements, return [].\n\n"
        "Article text:\n"
        f"{text}"
    )


def diff_texts(raw_text: str, normalised_text: str) -> list[dict[str, Any]]:
    import difflib

    raw_tokens = token_spans(raw_text)
    normalised_tokens = token_spans(normalised_text)
    matcher = difflib.SequenceMatcher(
        None,
        [token.text for token in raw_tokens],
        [token.text for token in normalised_tokens],
        autojunk=False,
    )
    records: list[dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        source_span = span_text(raw_text, raw_tokens, i1, i2)
        normalised_span = span_text(normalised_text, normalised_tokens, j1, j2)
        source_start = raw_tokens[i1].start if i1 < len(raw_tokens) else len(raw_text)
        source_end = raw_tokens[i2 - 1].end if i2 > i1 else source_start
        records.append(
            {
                "source_span": source_span,
                "normalised_span": normalised_span,
                "category": classify_substitution(source_span, normalised_span),
                "source_start": source_start,
                "source_end": source_end,
                "context": raw_text[
                    max(0, source_start - 20) : min(len(raw_text), source_end + 20)
                ],
            }
        )
    return records


def token_spans(text: str) -> list[TokenSpan]:
    return [
        TokenSpan(match.group(0), match.start(), match.end())
        for match in TOKEN_RE.finditer(text)
    ]


def span_text(text: str, tokens: list[TokenSpan], start: int, end: int) -> str:
    if end <= start:
        return ""
    return text[tokens[start].start : tokens[end - 1].end]


def classify_substitution(source_span: str, normalised_span: str) -> str:
    stripped = source_span.strip()
    if CURRENCY_RE.search(stripped):
        return "currency"
    if UNIT_RE.search(source_span.strip()):
        return "unit"
    if INITIALISM_RE.search(stripped):
        return "initialism"
    if DATE_RE.search(source_span):
        return "date"
    if NUMBER_RE.search(source_span):
        return "number"
    return "other"


def evaluate_reference(eval_wer: Any, reference_text: str, transcript: str) -> dict[str, Any]:
    normalised_source = eval_wer.normalize_text(reference_text, "standard")
    normalised_transcript = eval_wer.normalize_text(transcript, "standard")
    window = eval_wer.find_best_reference_window(
        reference_text,
        normalised_source,
        normalised_transcript,
        "standard",
    )
    alignment = eval_wer.align_tokens(window.normalized.tokens, normalised_transcript.tokens)
    window, alignment = eval_wer.trim_window_boundaries(reference_text, window, alignment)
    return {
        "wer": alignment.wer,
        "substitutions": alignment.substitutions,
        "insertions": alignment.insertions,
        "deletions": alignment.deletions,
        "num_ref_words": len(window.normalized.tokens),
    }


def apply_category(
    raw_text: str,
    records: list[dict[str, Any]],
    categories: set[str],
) -> str:
    selected = [record for record in records if record["category"] in categories]
    pieces: list[str] = []
    cursor = 0
    for record in sorted(selected, key=lambda item: int(item["source_start"])):
        start = int(record["source_start"])
        end = int(record["source_end"])
        if start < cursor:
            continue
        pieces.append(raw_text[cursor:start])
        pieces.append(str(record["normalised_span"]))
        cursor = end
    pieces.append(raw_text[cursor:])
    return "".join(pieces)


def token_difference_ratio(raw_text: str, normalised_text: str) -> float:
    import difflib

    raw = [token.text for token in token_spans(raw_text)]
    normalised = [token.text for token in token_spans(normalised_text)]
    if not raw:
        return 0.0
    matcher = difflib.SequenceMatcher(None, raw, normalised, autojunk=False)
    equal = sum(block.size for block in matcher.get_matching_blocks())
    return 1 - equal / len(raw)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_article_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "episode_id",
        "source",
        "title",
        "raw_wer",
        "normalised_wer",
        "delta",
        "delta_pct_of_raw",
        "flagged",
        "token_difference_ratio",
        "total_substitutions",
        "substitutions_by_category",
        "diff_path",
        "raw_path",
        "normalised_path",
        "generic_wer",
        "generic_delta",
        "targeted_wer",
        "targeted_delta",
        "targeted_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = {field: row.get(field, "") for field in fieldnames}
            output["substitutions_by_category"] = json.dumps(
                output["substitutions_by_category"],
                sort_keys=True,
            )
            writer.writerow(output)


def write_augmented_results_csv(
    source_path: Path,
    output_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not source_path.exists():
        return
    result_by_id = {row["episode_id"]: row for row in rows}
    with source_path.open("r", encoding="utf-8", newline="") as source_handle:
        reader = csv.DictReader(source_handle)
        original_fieldnames = reader.fieldnames or []
        extra_fieldnames = [
            "normalisation_raw_wer",
            "normalisation_wer",
            "normalisation_delta",
            "normalisation_delta_pct_of_raw",
            "normalisation_flagged",
            "normalisation_token_difference_ratio",
            "normalisation_total_substitutions",
            "normalisation_substitutions_by_category",
            "normalisation_diff_path",
            "normalisation_raw_path",
            "normalisation_normalised_path",
            "targeted_wer",
            "targeted_delta",
            "targeted_path",
        ]
        with output_path.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.DictWriter(
                output_handle,
                fieldnames=[*original_fieldnames, *extra_fieldnames],
            )
            writer.writeheader()
            for original_row in reader:
                result = result_by_id.get(original_row["episode_id"])
                if result:
                    original_row.update(
                        {
                            "normalisation_raw_wer": result["raw_wer"],
                            "normalisation_wer": result["normalised_wer"],
                            "normalisation_delta": result["delta"],
                            "normalisation_delta_pct_of_raw": result["delta_pct_of_raw"],
                            "normalisation_flagged": result["flagged"],
                            "normalisation_token_difference_ratio": result[
                                "token_difference_ratio"
                            ],
                            "normalisation_total_substitutions": result[
                                "total_substitutions"
                            ],
                            "normalisation_substitutions_by_category": json.dumps(
                                result["substitutions_by_category"],
                                sort_keys=True,
                            ),
                            "normalisation_diff_path": result["diff_path"],
                            "normalisation_raw_path": result["raw_path"],
                            "normalisation_normalised_path": result["normalised_path"],
                            "targeted_wer": result.get("targeted_wer", ""),
                            "targeted_delta": result.get("targeted_delta", ""),
                            "targeted_path": result.get("targeted_path", ""),
                        }
                    )
                writer.writerow(original_row)


def write_source_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    valid_rows = [row for row in rows if not row["flagged"]]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid_rows:
        grouped[row["source"]].append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "source",
            "n_articles",
            "mean_raw_wer",
            "mean_normalised_wer",
            "mean_delta",
            "total_substitutions",
            "substitutions_by_category",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for source, source_rows in sorted(grouped.items()):
            category_counts: Counter[str] = Counter()
            for row in source_rows:
                category_counts.update(row["substitutions_by_category"])
            writer.writerow(
                {
                    "source": source,
                    "n_articles": len(source_rows),
                    "mean_raw_wer": mean(row["raw_wer"] for row in source_rows),
                    "mean_normalised_wer": mean(row["normalised_wer"] for row in source_rows),
                    "mean_delta": mean(row["delta"] for row in source_rows),
                    "total_substitutions": sum(row["total_substitutions"] for row in source_rows),
                    "substitutions_by_category": json.dumps(category_counts, sort_keys=True),
                }
            )


def write_category_contribution(path: Path, rows: list[dict[str, Any]]) -> None:
    valid_rows = [row for row in rows if not row["flagged"]]
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["category", "mean_delta", "total_delta"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for category in CATEGORIES:
            deltas = [row["category_delta"][category] for row in valid_rows]
            writer.writerow(
                {
                    "category": category,
                    "mean_delta": mean(deltas),
                    "total_delta": sum(deltas),
                }
            )


def write_profile_comparison(
    path: Path,
    generic_path: Path,
    profiled_rows: list[dict[str, Any]],
) -> None:
    if not generic_path.exists():
        raise FileNotFoundError(f"Missing generic results: {generic_path}")
    generic_rows = read_article_summary(generic_path)
    generic_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    profiled_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in generic_rows:
        if not row["flagged"]:
            generic_by_source[row["source"]].append(row)
    for row in profiled_rows:
        if not row["flagged"]:
            profiled_by_source[row["source"]].append(row)

    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "source",
            "generic_wer_delta",
            "profiled_wer_delta",
            "profile_marginal_gain",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for source in sorted(set(generic_by_source) | set(profiled_by_source)):
            generic_delta = mean(row["delta"] for row in generic_by_source.get(source, []))
            profiled_delta = mean(row["delta"] for row in profiled_by_source.get(source, []))
            writer.writerow(
                {
                    "source": source,
                    "generic_wer_delta": generic_delta,
                    "profiled_wer_delta": profiled_delta,
                    "profile_marginal_gain": profiled_delta - generic_delta,
                }
            )


def attach_generic_results(rows: list[dict[str, Any]], generic_path: Path) -> None:
    generic_by_id = {
        row["episode_id"]: row
        for row in read_article_summary(generic_path)
        if "episode_id" in row
    }
    for row in rows:
        generic = generic_by_id.get(row["episode_id"])
        if generic:
            row["generic_wer"] = generic["normalised_wer"]
            row["generic_delta"] = generic["delta"]


def write_targeted_comparison(
    path: Path,
    generic_path: Path,
    targeted_rows: list[dict[str, Any]],
) -> None:
    if not generic_path.exists():
        raise FileNotFoundError(f"Missing generic results: {generic_path}")
    generic_rows = read_article_summary(generic_path)
    generic_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    targeted_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in generic_rows:
        if not row["flagged"]:
            generic_by_source[row["source"]].append(row)
    for row in targeted_rows:
        if not row["flagged"]:
            targeted_by_source[row["source"]].append(row)

    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "source",
            "generic_wer_delta",
            "targeted_wer_delta",
            "targeted_marginal_gain",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for source in sorted(set(generic_by_source) | set(targeted_by_source)):
            generic_delta = mean(row["delta"] for row in generic_by_source.get(source, []))
            targeted_delta = mean(row["delta"] for row in targeted_by_source.get(source, []))
            writer.writerow(
                {
                    "source": source,
                    "generic_wer_delta": generic_delta,
                    "targeted_wer_delta": targeted_delta,
                    "targeted_marginal_gain": targeted_delta - generic_delta,
                }
            )


def read_article_summary(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "episode_id": row["episode_id"],
                    "source": row["source"],
                    "delta": float(row["delta"]),
                    "normalised_wer": float(row["normalised_wer"]),
                    "flagged": row["flagged"].lower() == "true",
                }
            )
    return rows


def mean(values: Any) -> float:
    values_list = list(values)
    return sum(values_list) / len(values_list) if values_list else 0.0


def load_eval_wer_module() -> Any:
    script_path = Path(__file__).parents[1] / "tools" / "eval_whisper_wer.py"
    spec = importlib.util.spec_from_file_location("eval_whisper_wer", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load tools/eval_whisper_wer.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_whisper_wer"] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(main())

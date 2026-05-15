#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import string
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ATTRIBUTION_WORDS = {
    "according",
    "asked",
    "reported",
    "said",
    "says",
    "told",
    "wrote",
}

UNICODE_PUNCTUATION = "\u201c\u201d\u201e\u201f\u00ab\u00bb\u2014\u2013\u2026"

SMALL_NUMBER_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
    "11": "eleven",
    "12": "twelve",
    "13": "thirteen",
    "14": "fourteen",
    "15": "fifteen",
    "16": "sixteen",
    "17": "seventeen",
    "18": "eighteen",
    "19": "nineteen",
    "20": "twenty",
}

SMALL_ORDINAL_WORDS = {
    "1": "first",
    "2": "second",
    "3": "third",
    "4": "fourth",
    "5": "fifth",
    "6": "sixth",
    "7": "seventh",
    "8": "eighth",
    "9": "ninth",
    "10": "tenth",
    "11": "eleventh",
    "12": "twelfth",
    "13": "thirteenth",
    "14": "fourteenth",
    "15": "fifteenth",
    "16": "sixteenth",
    "17": "seventeenth",
    "18": "eighteenth",
    "19": "nineteenth",
    "20": "twentieth",
}


@dataclass(frozen=True)
class ManifestRow:
    episode_id: str
    source: str
    title: str
    audio_path: Path
    source_text_path: Path
    start_sec: float | None
    end_sec: float | None
    notes: str


@dataclass(frozen=True)
class NormalizedText:
    text: str
    tokens: list[str]


@dataclass(frozen=True)
class AlignmentResult:
    substitutions: int
    insertions: int
    deletions: int
    operations: list[dict[str, Any]]

    @property
    def wer(self) -> float:
        ref_words = sum(1 for op in self.operations if op["op"] != "insert")
        if ref_words == 0:
            return 0.0
        return (self.substitutions + self.insertions + self.deletions) / ref_words


@dataclass(frozen=True)
class WindowResult:
    start_word: int
    end_word: int
    raw_text: str
    normalized: NormalizedText


@dataclass(frozen=True)
class TranscriptSegment:
    start_sec: float
    end_sec: float
    text: str


@dataclass(frozen=True)
class TranscriptResult:
    text: str
    segments: list[TranscriptSegment]


class Transcriber:
    def __init__(self, model_size: str, device: str, compute_type: str) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.backend = ""
        self._model: Any = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from faster_whisper import WhisperModel

            model_device = self.device if self.device != "auto" else "auto"
            self._model = WhisperModel(
                self.model_size,
                device=model_device,
                compute_type=self.compute_type,
            )
            self.backend = "faster-whisper"
            return
        except ImportError:
            pass

        try:
            import whisper

            model_device = None if self.device == "auto" else self.device
            self._model = whisper.load_model(self.model_size, device=model_device)
            self.backend = "openai-whisper"
            return
        except ImportError as exc:
            raise RuntimeError(
                "Install faster-whisper or openai-whisper to run transcription"
            ) from exc

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        if self.backend == "faster-whisper":
            segments, _info = self._model.transcribe(str(audio_path))
            result_segments = [
                TranscriptSegment(
                    start_sec=float(segment.start),
                    end_sec=float(segment.end),
                    text=segment.text.strip(),
                )
                for segment in segments
            ]
            return TranscriptResult(
                text=" ".join(segment.text for segment in result_segments).strip(),
                segments=result_segments,
            )

        kwargs: dict[str, Any] = {}
        if self.device == "cpu":
            kwargs["fp16"] = False
        result = self._model.transcribe(str(audio_path), **kwargs)
        result_segments = [
            TranscriptSegment(
                start_sec=float(segment.get("start", 0.0)),
                end_sec=float(segment.get("end", 0.0)),
                text=str(segment.get("text", "")).strip(),
            )
            for segment in result.get("segments", [])
        ]
        return TranscriptResult(
            text=str(result.get("text", "")).strip(),
            segments=result_segments,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pragmatic Whisper WER eval for generated newsletter podcast episodes"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="small")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument(
        "--number-mode",
        default="standard",
        choices=["standard", "digits-to-words"],
    )
    parser.add_argument("--window-search", action="store_true")
    parser.add_argument("--keep-excerpts", action="store_true")
    parser.add_argument(
        "--error-clips",
        action="store_true",
        help="Write short audio clips around alignment errors for inspection",
    )
    parser.add_argument("--error-clip-window", type=float, default=8.0)
    parser.add_argument("--max-error-clips", type=int, default=40)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    details_path = args.out_dir / "details.jsonl"
    results_path = args.out_dir / "results.csv"
    summary_path = args.out_dir / "summary_by_source.csv"
    refs_dir = args.out_dir / "selected_references"
    transcripts_dir = args.out_dir / "transcripts"
    excerpts_dir = args.out_dir / "audio_excerpts"
    error_clips_dir = args.out_dir / "error_clips"
    refs_dir.mkdir(exist_ok=True)
    transcripts_dir.mkdir(exist_ok=True)
    if args.keep_excerpts:
        excerpts_dir.mkdir(exist_ok=True)
    if args.error_clips:
        error_clips_dir.mkdir(exist_ok=True)

    rows = read_manifest(args.manifest)
    transcriber = Transcriber(args.model, args.device, args.compute_type)

    result_rows: list[dict[str, Any]] = []
    with details_path.open("w", encoding="utf-8") as details_file:
        for row in rows:
            try:
                result = evaluate_row(
                    row=row,
                    transcriber=transcriber,
                    whisper_model=f"{transcriber.backend}:{args.model}",
                    out_dir=args.out_dir,
                    refs_dir=refs_dir,
                    transcripts_dir=transcripts_dir,
                    excerpts_dir=excerpts_dir,
                    error_clips_dir=error_clips_dir,
                    window_search=args.window_search,
                    number_mode=args.number_mode,
                    keep_excerpts=args.keep_excerpts,
                    error_clips=args.error_clips,
                    error_clip_window=args.error_clip_window,
                    max_error_clips=args.max_error_clips,
                )
            except Exception as exc:
                print(f"[warn] {row.episode_id}: {exc}", file=sys.stderr)
                result = failed_result_row(row, transcriber.backend, args.model, str(exc))
                details_file.write(
                    json.dumps({"episode_id": row.episode_id, "error": str(exc)}) + "\n"
                )
            result_rows.append(result["summary"])
            details_file.write(json.dumps(result["details"], ensure_ascii=False) + "\n")

    write_results_csv(results_path, result_rows)
    write_summary_csv(summary_path, result_rows)
    print(f"Wrote {results_path}")
    print(f"Wrote {details_path}")
    print(f"Wrote {summary_path}")
    return 0


def evaluate_row(
    row: ManifestRow,
    transcriber: Transcriber,
    whisper_model: str,
    out_dir: Path,
    refs_dir: Path,
    transcripts_dir: Path,
    excerpts_dir: Path,
    error_clips_dir: Path,
    window_search: bool,
    number_mode: str,
    keep_excerpts: bool,
    error_clips: bool,
    error_clip_window: float,
    max_error_clips: int,
) -> dict[str, Any]:
    ensure_file(row.audio_path)
    ensure_file(row.source_text_path)

    with tempfile.TemporaryDirectory(prefix="whisper-wer-") as temp_name:
        temp_dir = Path(temp_name)
        eval_audio = prepare_audio(row, temp_dir, excerpts_dir, keep_excerpts)
        transcript_result = transcriber.transcribe(eval_audio)
        raw_transcript = transcript_result.text
        duration = audio_duration(eval_audio)

        source_text = row.source_text_path.read_text(encoding="utf-8")
        normalized_source = normalize_text(source_text, number_mode)
        normalized_transcript = normalize_text(raw_transcript, number_mode)

        if window_search:
            window = find_best_reference_window(
                source_text,
                normalized_source,
                normalized_transcript,
                number_mode,
            )
        else:
            window = WindowResult(
                start_word=0,
                end_word=len(normalized_source.tokens),
                raw_text=source_text,
                normalized=normalized_source,
            )

        alignment = align_tokens(window.normalized.tokens, normalized_transcript.tokens)
        if window_search:
            window, alignment = trim_window_boundaries(source_text, window, alignment)

        token_timings = transcript_token_timings(transcript_result.segments, number_mode)
        error_contexts = build_error_contexts(
            row=row,
            alignment=alignment,
            token_timings=token_timings,
            eval_audio=eval_audio,
            error_clips_dir=error_clips_dir,
            enabled=error_clips,
            clip_window=error_clip_window,
            max_clips=max_error_clips,
            duration=duration,
        )

    reference_features = reference_features_for_window(
        source_text,
        window.start_word,
        window.end_word,
    )
    slices = stratified_error_rates(alignment.operations, reference_features)

    selected_ref_path = refs_dir / f"{safe_name(row.episode_id)}.txt"
    transcript_path = transcripts_dir / f"{safe_name(row.episode_id)}.txt"
    selected_ref_path.write_text(window.raw_text, encoding="utf-8")
    transcript_path.write_text(raw_transcript, encoding="utf-8")

    summary = {
        "episode_id": row.episode_id,
        "source": row.source,
        "title": row.title,
        "duration_sec": round(duration, 2),
        "whisper_model": whisper_model,
        "overall_wer": round(alignment.wer, 6),
        "substitutions": alignment.substitutions,
        "insertions": alignment.insertions,
        "deletions": alignment.deletions,
        "num_ref_words": len(window.normalized.tokens),
        "number_error_rate": round(slices["number"], 6),
        "proper_noun_error_rate": round(slices["proper_noun"], 6),
        "attribution_error_rate": round(slices["attribution"], 6),
        "acronym_error_rate": round(slices["acronym"], 6),
        "selected_reference_start_word": window.start_word,
        "selected_reference_end_word": window.end_word,
    }
    details = {
        **asdict(row),
        "audio_path": str(row.audio_path),
        "source_text_path": str(row.source_text_path),
        "normalised_reference_window": window.normalized.text,
        "normalised_transcript": normalized_transcript.text,
        "raw_transcript": raw_transcript,
        "raw_selected_reference_window": window.raw_text,
        "alignment_operations": alignment.operations,
        "error_contexts": error_contexts,
        "selected_reference_file": str(selected_ref_path),
        "transcript_file": str(transcript_path),
    }
    return {"summary": summary, "details": details}


def read_manifest(path: Path) -> list[ManifestRow]:
    ensure_file(path)
    if path.suffix.lower() == ".jsonl":
        return [
            manifest_row_from_mapping(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    with path.open("r", encoding="utf-8", newline="") as handle:
        return [manifest_row_from_mapping(row) for row in csv.DictReader(handle)]


def manifest_row_from_mapping(raw: dict[str, Any]) -> ManifestRow:
    return ManifestRow(
        episode_id=str(raw["episode_id"]),
        source=str(raw["source"]),
        title=str(raw["title"]),
        audio_path=Path(str(raw["audio_path"])).expanduser(),
        source_text_path=Path(str(raw["source_text_path"])).expanduser(),
        start_sec=parse_optional_float(raw.get("start_sec")),
        end_sec=parse_optional_float(raw.get("end_sec")),
        notes=str(raw.get("notes") or ""),
    )


def prepare_audio(
    row: ManifestRow,
    temp_dir: Path,
    excerpts_dir: Path,
    keep_excerpts: bool,
) -> Path:
    if row.start_sec is None and row.end_sec is None:
        return row.audio_path
    output = temp_dir / f"{safe_name(row.episode_id)}.wav"
    ffmpeg_clip(row.audio_path, output, row.start_sec, row.end_sec)
    if keep_excerpts:
        kept = excerpts_dir / output.name
        kept.write_bytes(output.read_bytes())
    return output


def ffmpeg_clip(
    input_path: Path,
    output_path: Path,
    start: float | None,
    end: float | None,
) -> None:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start is not None:
        command.extend(["-ss", str(start)])
    command.extend(["-i", str(input_path)])
    if end is not None:
        if start is not None:
            command.extend(["-t", str(max(0.0, end - start))])
        else:
            command.extend(["-to", str(end)])
    command.extend(["-vn", "-ac", "1", "-ar", "16000", str(output_path)])
    subprocess.run(command, check=True)


def audio_duration(path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return float(completed.stdout.strip())


def normalize_text(text: str, number_mode: str) -> NormalizedText:
    text = text.lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(
        r"\b(\d+)(st|nd|rd|th)\b",
        lambda match: ordinal_token_to_words(match.group(1), match.group(2)),
        text,
    )
    if number_mode == "digits-to-words":
        text = re.sub(r"\b\d+\b", lambda match: digit_token_to_words(match.group(0)), text)
    punctuation = string.punctuation.replace("'", "") + UNICODE_PUNCTUATION
    text = text.translate(str.maketrans({character: " " for character in punctuation}))
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split() if text else []
    return NormalizedText(text=text, tokens=tokens)


def digit_token_to_words(token: str) -> str:
    if token in SMALL_NUMBER_WORDS:
        return SMALL_NUMBER_WORDS[token]
    return " ".join(SMALL_NUMBER_WORDS.get(character, character) for character in token)


def ordinal_token_to_words(number: str, suffix: str) -> str:
    if number in SMALL_ORDINAL_WORDS:
        return SMALL_ORDINAL_WORDS[number]
    return f"{number}{suffix}"


def find_best_reference_window(
    raw_source: str,
    normalized_source: NormalizedText,
    normalized_transcript: NormalizedText,
    number_mode: str,
) -> WindowResult:
    source_tokens = normalized_source.tokens
    transcript_tokens = normalized_transcript.tokens
    if not transcript_tokens or len(source_tokens) <= math.ceil(len(transcript_tokens) * 1.1):
        return WindowResult(0, len(source_tokens), raw_source, normalized_source)

    transcript_counts = Counter(transcript_tokens)
    candidates: list[tuple[float, int, int]] = []
    target_lengths = {
        max(20, min(len(source_tokens), math.ceil(len(transcript_tokens) * ratio)))
        for ratio in (0.9, 1.0, 1.1)
    }
    for target_len in sorted(target_lengths):
        step = max(8, target_len // 10)
        for start in range(0, max(1, len(source_tokens) - target_len + 1), step):
            end = min(len(source_tokens), start + target_len)
            score = overlap_score(Counter(source_tokens[start:end]), transcript_counts)
            candidates.append((score, start, end))
    candidates.sort(reverse=True)

    best_start, best_end = 0, min(len(source_tokens), max(target_lengths))
    best_wer = float("inf")
    for _score, start, end in candidates[:16]:
        alignment = align_tokens(source_tokens[start:end], transcript_tokens)
        if alignment.wer < best_wer:
            best_wer = alignment.wer
            best_start, best_end = start, end

    raw_words = raw_source.split()
    raw_window = " ".join(raw_words[best_start:best_end])
    normalized_tokens = source_tokens[best_start:best_end]
    normalized_window = NormalizedText(
        text=" ".join(normalized_tokens),
        tokens=normalized_tokens,
    )
    return WindowResult(best_start, best_end, raw_window, normalized_window)


def trim_window_boundaries(
    raw_source: str,
    window: WindowResult,
    alignment: AlignmentResult,
) -> tuple[WindowResult, AlignmentResult]:
    operations = alignment.operations
    start = 0
    end = len(operations)
    while start < end and operations[start]["op"] in {"insert", "delete"}:
        start += 1
    while end > start and operations[end - 1]["op"] in {"insert", "delete"}:
        end -= 1
    if start == 0 and end == len(operations):
        return window, alignment

    trimmed_operations = operations[start:end]
    ref_indices = [
        int(operation["ref_index"])
        for operation in trimmed_operations
        if operation["ref_index"] is not None
    ]
    hyp_tokens = [
        str(operation["hyp"])
        for operation in trimmed_operations
        if operation["op"] != "delete"
    ]
    if not ref_indices or not hyp_tokens:
        return window, alignment

    relative_start = min(ref_indices)
    relative_end = max(ref_indices) + 1
    normalized_tokens = window.normalized.tokens[relative_start:relative_end]
    trimmed_alignment = alignment_from_operations(
        trimmed_operations,
        relative_start,
    )

    raw_words = raw_source.split()
    absolute_start = window.start_word + relative_start
    absolute_end = window.start_word + relative_end
    raw_window = " ".join(raw_words[absolute_start:absolute_end])
    trimmed_window = WindowResult(
        start_word=absolute_start,
        end_word=absolute_end,
        raw_text=raw_window,
        normalized=NormalizedText(
            text=" ".join(normalized_tokens),
            tokens=normalized_tokens,
        ),
    )
    return trimmed_window, trimmed_alignment


def alignment_from_operations(
    operations: list[dict[str, Any]],
    ref_offset: int,
) -> AlignmentResult:
    substitutions = insertions = deletions = 0
    adjusted: list[dict[str, Any]] = []
    for operation in operations:
        copied = dict(operation)
        if copied["ref_index"] is not None:
            copied["ref_index"] = int(copied["ref_index"]) - ref_offset
        if copied["hyp_index"] is not None:
            copied["original_hyp_index"] = copied["hyp_index"]
        if copied["op"] == "substitute":
            substitutions += 1
        elif copied["op"] == "insert":
            insertions += 1
        elif copied["op"] == "delete":
            deletions += 1
        adjusted.append(copied)
    return AlignmentResult(substitutions, insertions, deletions, adjusted)


def transcript_token_timings(
    segments: list[TranscriptSegment],
    number_mode: str,
) -> dict[int, tuple[float, float]]:
    timings: dict[int, tuple[float, float]] = {}
    token_index = 0
    for segment in segments:
        tokens = normalize_text(segment.text, number_mode).tokens
        if not tokens:
            continue
        duration = max(0.0, segment.end_sec - segment.start_sec)
        token_duration = duration / len(tokens) if duration else 0.0
        for offset, _token in enumerate(tokens):
            start = segment.start_sec + offset * token_duration
            end = segment.start_sec + (offset + 1) * token_duration
            timings[token_index] = (start, end)
            token_index += 1
    return timings


def build_error_contexts(
    row: ManifestRow,
    alignment: AlignmentResult,
    token_timings: dict[int, tuple[float, float]],
    eval_audio: Path,
    error_clips_dir: Path,
    enabled: bool,
    clip_window: float,
    max_clips: int,
    duration: float,
) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    error_count = 0
    for index, operation in enumerate(alignment.operations):
        if operation["op"] == "equal":
            continue
        if error_count >= max_clips:
            break
        timing = timing_for_operation(operation, alignment.operations, token_timings, index)
        if timing is None:
            continue
        center = sum(timing) / 2
        clip_start = max(0.0, center - clip_window)
        clip_end = min(duration, center + clip_window)
        context_id = f"{safe_name(row.episode_id)}_{error_count:03d}"
        clip_path = error_clips_dir / f"{context_id}.wav"
        if enabled:
            ffmpeg_clip(eval_audio, clip_path, clip_start, clip_end)
        ref_context, hyp_context = operation_context(alignment.operations, index)
        contexts.append(
            {
                "id": context_id,
                "operation_index": index,
                "op": operation["op"],
                "ref": operation["ref"],
                "hyp": operation["hyp"],
                "ref_context": ref_context,
                "hyp_context": hyp_context,
                "start_sec": round(clip_start, 3),
                "end_sec": round(clip_end, 3),
                "audio_clip_file": str(clip_path) if enabled else "",
            }
        )
        error_count += 1
    return contexts


def timing_for_operation(
    operation: dict[str, Any],
    operations: list[dict[str, Any]],
    token_timings: dict[int, tuple[float, float]],
    operation_index: int,
) -> tuple[float, float] | None:
    hyp_index = operation.get("original_hyp_index", operation.get("hyp_index"))
    if hyp_index is not None and int(hyp_index) in token_timings:
        return token_timings[int(hyp_index)]
    for neighbor in range(1, 8):
        for candidate_index in (operation_index - neighbor, operation_index + neighbor):
            if not 0 <= candidate_index < len(operations):
                continue
            candidate = operations[candidate_index]
            candidate_hyp = candidate.get("original_hyp_index", candidate.get("hyp_index"))
            if candidate_hyp is not None and int(candidate_hyp) in token_timings:
                return token_timings[int(candidate_hyp)]
    return None


def operation_context(
    operations: list[dict[str, Any]],
    operation_index: int,
    radius: int = 10,
) -> tuple[str, str]:
    start = max(0, operation_index - radius)
    end = min(len(operations), operation_index + radius + 1)
    ref_words: list[str] = []
    hyp_words: list[str] = []
    for operation in operations[start:end]:
        if operation["op"] != "insert":
            ref_words.append(mark_error_token(str(operation["ref"]), operation["op"]))
        if operation["op"] != "delete":
            hyp_words.append(mark_error_token(str(operation["hyp"]), operation["op"]))
    return " ".join(ref_words), " ".join(hyp_words)


def mark_error_token(token: str, operation: str) -> str:
    if operation == "equal":
        return token
    return f"[{token}]"


def overlap_score(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = sum(min(left[token], right[token]) for token in left.keys() & right.keys())
    return overlap / max(1, sum(right.values()))


def align_tokens(ref: list[str], hyp: list[str]) -> AlignmentResult:
    rows = len(ref) + 1
    cols = len(hyp) + 1
    costs = [[0] * cols for _ in range(rows)]
    back = [[""] * cols for _ in range(rows)]
    for i in range(1, rows):
        costs[i][0] = i
        back[i][0] = "delete"
    for j in range(1, cols):
        costs[0][j] = j
        back[0][j] = "insert"
    for i in range(1, rows):
        for j in range(1, cols):
            if ref[i - 1] == hyp[j - 1]:
                choices = [(costs[i - 1][j - 1], "equal")]
            else:
                choices = [(costs[i - 1][j - 1] + 1, "substitute")]
            choices.extend(
                [
                    (costs[i - 1][j] + 1, "delete"),
                    (costs[i][j - 1] + 1, "insert"),
                ]
            )
            costs[i][j], back[i][j] = min(choices, key=lambda item: item[0])

    operations: list[dict[str, Any]] = []
    substitutions = insertions = deletions = 0
    i, j = len(ref), len(hyp)
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "equal":
            operations.append(
                {
                    "op": "equal",
                    "ref_index": i - 1,
                    "hyp_index": j - 1,
                    "ref": ref[i - 1],
                    "hyp": hyp[j - 1],
                }
            )
            i -= 1
            j -= 1
        elif op == "substitute":
            substitutions += 1
            operations.append(
                {
                    "op": "substitute",
                    "ref_index": i - 1,
                    "hyp_index": j - 1,
                    "ref": ref[i - 1],
                    "hyp": hyp[j - 1],
                }
            )
            i -= 1
            j -= 1
        elif op == "delete":
            deletions += 1
            operations.append(
                {
                    "op": "delete",
                    "ref_index": i - 1,
                    "hyp_index": None,
                    "ref": ref[i - 1],
                    "hyp": "",
                }
            )
            i -= 1
        else:
            insertions += 1
            operations.append(
                {
                    "op": "insert",
                    "ref_index": None,
                    "hyp_index": j - 1,
                    "ref": "",
                    "hyp": hyp[j - 1],
                }
            )
            j -= 1
    operations.reverse()
    return AlignmentResult(substitutions, insertions, deletions, operations)


def reference_features_for_window(
    raw_source: str,
    start_word: int,
    end_word: int,
) -> dict[int, set[str]]:
    words = raw_source.split()[start_word:end_word]
    features: dict[int, set[str]] = {}
    for index, word in enumerate(words):
        cleaned = clean_feature_token(word)
        normalized = normalize_text(cleaned, "standard").tokens
        if not normalized:
            continue
        token_features: set[str] = set()
        if any(character.isdigit() for character in cleaned):
            token_features.add("number")
        if cleaned.lower() in ATTRIBUTION_WORDS:
            token_features.add("attribution")
        if re.fullmatch(r"[A-Z][A-Z0-9&.-]{1,}", cleaned):
            token_features.add("acronym")
        if looks_like_proper_noun(cleaned, index):
            token_features.add("proper_noun")
        if token_features:
            features[index] = token_features
    return features


def clean_feature_token(token: str) -> str:
    return token.strip(string.punctuation + "“”‘’")


def looks_like_proper_noun(token: str, index: int) -> bool:
    if index == 0:
        return False
    if not token or not token[0].isupper():
        return False
    if token.isupper():
        return False
    return any(character.islower() for character in token)


def stratified_error_rates(
    operations: list[dict[str, Any]],
    features: dict[int, set[str]],
) -> dict[str, float]:
    totals = {"number": 0, "proper_noun": 0, "attribution": 0, "acronym": 0}
    errors = {"number": 0, "proper_noun": 0, "attribution": 0, "acronym": 0}
    for operation in operations:
        ref_index = operation["ref_index"]
        if ref_index is None or ref_index not in features:
            continue
        for feature in features[ref_index]:
            totals[feature] += 1
            if operation["op"] != "equal":
                errors[feature] += 1
    return {
        feature: (errors[feature] / totals[feature] if totals[feature] else 0.0)
        for feature in totals
    }


def write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "episode_id",
        "source",
        "title",
        "duration_sec",
        "whisper_model",
        "overall_wer",
        "substitutions",
        "insertions",
        "deletions",
        "num_ref_words",
        "number_error_rate",
        "proper_noun_error_rate",
        "attribution_error_rate",
        "acronym_error_rate",
        "selected_reference_start_word",
        "selected_reference_end_word",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["source"]), []).append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "samples", "mean_wer"])
        writer.writeheader()
        for source, source_rows in sorted(grouped.items()):
            wers = [
                float(row["overall_wer"])
                for row in source_rows
                if row.get("overall_wer") != ""
            ]
            writer.writerow(
                {
                    "source": source,
                    "samples": len(source_rows),
                    "mean_wer": round(sum(wers) / len(wers), 6) if wers else "",
                }
            )


def failed_result_row(
    row: ManifestRow,
    backend: str,
    model: str,
    error: str,
) -> dict[str, Any]:
    summary = {
        "episode_id": row.episode_id,
        "source": row.source,
        "title": row.title,
        "duration_sec": "",
        "whisper_model": f"{backend}:{model}",
        "overall_wer": "",
        "substitutions": "",
        "insertions": "",
        "deletions": "",
        "num_ref_words": "",
        "number_error_rate": "",
        "proper_noun_error_rate": "",
        "attribution_error_rate": "",
        "acronym_error_rate": "",
        "selected_reference_start_word": "",
        "selected_reference_end_word": "",
    }
    return {"summary": summary, "details": {"episode_id": row.episode_id, "error": error}}


def parse_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "sample"


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import logging
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import Any, cast

from mutagen.easyid3 import EasyID3
from pydub import AudioSegment
from slugify import slugify

from newsletter_podcast.config import TtsConfig

logger = logging.getLogger(__name__)

MAX_CHARS = 4000


def synthesize_episode(
    title: str,
    artist: str,
    text: str,
    episodes_dir: Path,
    config: TtsConfig,
    published: date | None = None,
) -> Path:
    published = published or date.today()
    episodes_dir.mkdir(parents=True, exist_ok=True)
    output_path = _episode_path(episodes_dir, published, title)
    chunks = chunk_text(text, max_chars=MAX_CHARS)
    started = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="newsletter-podcast-tts-") as temp_name:
        temp_dir = Path(temp_name)
        wav_paths = _generate_wavs(chunks, temp_dir, config)
        combined = AudioSegment.silent(duration=0)
        for wav_path in wav_paths:
            combined += AudioSegment.from_wav(wav_path)
        combined = combined.set_channels(1)
        combined.export(output_path, format="mp3", bitrate="64k")

    elapsed = time.monotonic() - started
    duration = AudioSegment.from_mp3(output_path).duration_seconds
    realtime_factor = elapsed / duration if duration else 0.0
    _write_id3(output_path, title, artist, published)
    logger.info(
        "Synthesized episode",
        extra={
            "title": title,
            "path": str(output_path),
            "duration_seconds": round(duration, 2),
            "elapsed_seconds": round(elapsed, 2),
            "realtime_factor": round(realtime_factor, 3),
        },
    )
    return output_path


def chunk_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_long_paragraph(paragraph, max_chars))
            continue
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current.strip())
            current = paragraph
    if current:
        chunks.append(current.strip())
    return chunks


def _generate_wavs(chunks: list[str], temp_dir: Path, config: TtsConfig) -> list[Path]:
    import soundfile as sf
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code=_lang_code(config.voice), device=config.device)
    wav_paths: list[Path] = []
    for index, chunk in enumerate(chunks):
        wav_path = temp_dir / f"chunk-{index:04d}.wav"
        audio = _synthesize_chunk(pipeline, chunk, config)
        sf.write(wav_path, audio, 24000)
        wav_paths.append(wav_path)
    return wav_paths


def _synthesize_chunk(pipeline: Any, text: str, config: TtsConfig) -> Any:
    generator = pipeline(text, voice=config.voice, speed=config.speed, split_pattern=None)
    audio_parts = []
    for item in generator:
        audio_parts.append(_audio_from_result(item))
    if len(audio_parts) == 1:
        return audio_parts[0]

    import numpy as np

    return np.concatenate(audio_parts)


def _audio_from_result(item: Any) -> Any:
    if isinstance(item, tuple) and len(item) >= 3:
        audio = item[2]
    elif hasattr(item, "audio"):
        audio = item.audio
    else:
        audio = item
    if hasattr(audio, "detach"):
        return audio.detach().cpu().numpy()
    return audio


def _split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    sentences = paragraph.replace(". ", ".\n").splitlines()
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            pieces.extend(
                sentence[start : start + max_chars]
                for start in range(0, len(sentence), max_chars)
            )
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)
    return pieces


def _episode_path(episodes_dir: Path, published: date, title: str) -> Path:
    slug = slugify(title, max_length=80) or "newsletter"
    return episodes_dir / f"{published.isoformat()}_{slug}.mp3"


def _write_id3(path: Path, title: str, artist: str, published: date) -> None:
    easy_id3 = cast(Any, EasyID3)
    try:
        tags = easy_id3(path)
    except Exception:
        tags = easy_id3()
    tags["title"] = title
    tags["artist"] = artist
    tags["album"] = "Newsletters"
    tags["date"] = published.isoformat()
    tags.save(path)


def _lang_code(voice: str) -> str:
    return voice.split("_", 1)[0][:1] or "a"

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TypedDict

from newsletter_podcast.config import TtsConfig
from newsletter_podcast.synthesize import synthesize_episode


class TtsJobConfig(TypedDict):
    voice: str
    speed: float
    device: str


class SynthesisJob(TypedDict):
    title: str
    artist: str
    text: str
    episodes_dir: str
    tts: TtsJobConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True, help="Path to synthesis job JSON")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    job = _load_job(args.job)
    synthesize_episode(
        title=job["title"],
        artist=job["artist"],
        text=job["text"],
        episodes_dir=Path(job["episodes_dir"]),
        config=TtsConfig.model_validate(job["tts"]),
    )
    return 0


def _load_job(path: Path) -> SynthesisJob:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Synthesis job must be a JSON object: {path}")
    return SynthesisJob(
        title=str(raw["title"]),
        artist=str(raw["artist"]),
        text=str(raw["text"]),
        episodes_dir=str(raw["episodes_dir"]),
        tts=TtsJobConfig(
            voice=str(raw["tts"]["voice"]),
            speed=float(raw["tts"]["speed"]),
            device=str(raw["tts"]["device"]),
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

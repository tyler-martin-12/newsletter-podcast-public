# Newsletter Podcast

<img src="docs/assets/newsletter-stack.png" alt="A pen-and-ink drawing of a tall stack of newspapers with headphones resting on top." width="320">

Self-hosted pipeline for turning email newsletters into a private podcast feed. It reads an IMAP folder, extracts clean article text, generates Kokoro-82M speech, writes MP3 episodes, and builds an RSS feed for podcast apps.

Includes a small Whisper-WER eval workflow for checking generated audio against the cleaned source text. This is a personal project shared for reference, not a supported product.

Blog post: https://tyler-alexander-martin.com/blog/newsletter-podcast/

## Stack

Python 3.11+, `imap-tools`, `trafilatura`, `kokoro`, `pydub`, `ffmpeg`, `feedgen`, `mutagen`, `faster-whisper`, Caddy, and systemd.

## Install

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv ffmpeg

python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev,eval]"
```

## Configure

```bash
cp config.example.yaml config.yaml
editor config.yaml
export NEWSLETTER_PODCAST_CONFIG="$PWD/config.yaml"
export IMAP_PASSWORD="your_app_password"
```

Key fields in `config.yaml`:

- `imap.*`: host, port, username, password env var, source folder, optional destination folder.
- `tts.*`: Kokoro voice, speed, and device.
- `output.*`: episode directory, feed path, base URL, state DB, lock path.
- `feed.*`: title, description, cover image, retention days.

Optional env vars:

- `WEEKLY_PICKS_SENDER`: sender used to detect personal weekly-picks emails.
- `ANTHROPIC_API_KEY`: only needed for normalisation eval.
- `ANTHROPIC_MODEL`: optional normalisation model override.

## Run

```bash
.venv/bin/newsletter-podcast
```

The systemd templates in `systemd/` run the pipeline on a timer. `Caddyfile.example` shows how to serve `feed.xml` and episodes behind basic auth.

## Eval

Run the included synthetic eval sample:

```bash
.venv/bin/python tools/eval_whisper_wer.py \
  --manifest examples/synthetic/manifest.csv \
  --out-dir examples/synthetic/eval_results \
  --model tiny \
  --window-search
```

Run WER on your own generated episodes with a CSV or JSONL manifest containing:

```text
episode_id,source,title,audio_path,source_text_path,start_sec,end_sec,notes
```

```bash
.venv/bin/python tools/eval_whisper_wer.py \
  --manifest my_eval_manifest.csv \
  --out-dir eval_results \
  --model small \
  --window-search \
  --error-clips
```

Targeted text normalisation eval reuses existing Whisper details:

```bash
export ANTHROPIC_API_KEY="your_key"
.venv/bin/python eval/normalisation.py \
  --manifest my_eval_manifest.csv \
  --details eval_results/details.jsonl \
  --mode targeted \
  --out-dir eval_results/normalisation_targeted
```

The eval is pragmatic and inspectable. It is not a formal benchmark.

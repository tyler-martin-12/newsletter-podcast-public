# Newsletter Podcast

Newsletter Podcast is a self-hosted Python pipeline that turns email newsletters into a private podcast feed. It reads messages from an IMAP folder, extracts article text, synthesises speech with Kokoro-82M, writes MP3 episodes, and builds an RSS feed that can be served behind basic authentication.

This repository also includes a pragmatic Whisper-WER evaluation workflow. The eval compares generated audio transcripts against the cleaned text that was sent to TTS, then optionally estimates whether narrow text normalisation would improve the output. This is a personal project shared for reference, not a supported product.

Blog post: https://example.com/newsletter-podcast-blog-post

## Stack

- Python 3.11+
- `imap-tools` for IMAP
- `trafilatura` for HTML and article text extraction
- `kokoro` for Kokoro-82M TTS
- `pydub` and `ffmpeg` for audio stitching and MP3 output
- `feedgen` for RSS
- `mutagen` for audio metadata and duration
- `faster-whisper` or `openai-whisper` for eval
- Caddy for HTTPS and basic auth
- systemd timer for scheduled runs

## Quickstart

Install system dependencies:

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv ffmpeg
```

Create a virtual environment and install the project:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev,eval]"
```

Create a local config:

```bash
cp config.example.yaml config.yaml
editor config.yaml
```

Run the synthetic examples:

```bash
.venv/bin/python tools/eval_whisper_wer.py \
  --manifest examples/synthetic/manifest.csv \
  --out-dir examples/synthetic/eval_results \
  --model tiny \
  --window-search \
  --number-mode standard
```

To run the email-to-podcast pipeline on your own mailbox:

```bash
export NEWSLETTER_PODCAST_CONFIG="$PWD/config.yaml"
export IMAP_PASSWORD="your_app_password"
.venv/bin/newsletter-podcast
```

## Configuration

The app reads `config.yaml`, either from `--config` or from `NEWSLETTER_PODCAST_CONFIG`.

Environment variables:

- `NEWSLETTER_PODCAST_CONFIG`: path to `config.yaml`.
- `IMAP_PASSWORD`: app password or mailbox password. The name is configurable through `imap.password_env`.
- `WEEKLY_PICKS_SENDER`: optional sender address used to detect personal weekly-picks emails.
- `ANTHROPIC_API_KEY`: optional, only needed for `eval/normalisation.py`.
- `ANTHROPIC_MODEL`: optional model override for normalisation eval.

Config fields:

- `imap.host`: IMAP host.
- `imap.port`: IMAP TLS port, usually `993`.
- `imap.username`: mailbox username.
- `imap.password_env`: environment variable that contains the mailbox password.
- `imap.folder`: folder or Gmail label to read.
- `imap.move_to_folder`: optional folder or label to move handled messages into.
- `tts.voice`: Kokoro voice, for example `af_bella`.
- `tts.speed`: speech speed.
- `tts.device`: Kokoro device, usually `cpu`.
- `output.episodes_dir`: directory for generated MP3 files.
- `output.feed_path`: RSS output path.
- `output.base_url`: public base URL used in enclosure links.
- `output.state_db`: SQLite state database.
- `output.lock_path`: lock file preventing concurrent runs.
- `feed.title`: podcast title.
- `feed.description`: podcast description.
- `feed.cover_image`: optional cover image path.
- `feed.retention_days`: delete MP3 episodes older than this many days.

## Eval

The WER eval expects a CSV or JSONL manifest with:

```text
episode_id,source,title,audio_path,source_text_path,start_sec,end_sec,notes
```

Run it on your own generated episodes:

```bash
.venv/bin/python tools/eval_whisper_wer.py \
  --manifest my_eval_manifest.csv \
  --out-dir eval_results \
  --model small \
  --window-search \
  --error-clips
```

Run the targeted normalisation eval without rerunning Whisper:

```bash
export ANTHROPIC_API_KEY="your_key"
.venv/bin/python eval/normalisation.py \
  --manifest my_eval_manifest.csv \
  --details eval_results/details.jsonl \
  --mode targeted \
  --out-dir eval_results/normalisation_targeted
```

The normalisation eval reads existing transcripts from `details.jsonl`. It does not rerun Whisper. Generic mode asks for rewritten text, while targeted mode asks for JSON replacement spans and applies them mechanically.

## Deployment

The `systemd/` directory contains a timer and service template. The default service assumes:

- project installed at `/opt/newsletter-podcast`
- config at `/etc/newsletter-podcast/config.yaml`
- environment file at `/etc/newsletter-podcast/env`

The `Caddyfile.example` serves `feed.xml`, `cover.jpg`, and generated episodes with basic auth. Replace `podcast.example.com`, the username, and the password hash before using it.

## Notes

Kokoro may download model assets on first use. CPU inference works, but long newsletters can take several minutes. The eval workflow is intended to catch obvious issues and support manual inspection. It is not a formal benchmark.

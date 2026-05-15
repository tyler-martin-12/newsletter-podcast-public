# Whisper WER Eval

This is a one-session, pragmatic evaluation script for checking whether generated Kokoro audio preserves the cleaned article text. It is not a formal benchmark. Use it to spot obvious transcription, number, name, attribution, and acronym problems.

## Dependencies

Required system tools:

```bash
ffmpeg
ffprobe
```

Install one Whisper backend:

```bash
pip install faster-whisper
```

or:

```bash
pip install openai-whisper
```

`faster-whisper` is tried first. If it is not installed, the script falls back to `openai-whisper`.

For notebook inspection:

```bash
pip install -e ".[eval]"
```

## Manifest

Use CSV or JSONL. CSV columns:

```text
episode_id,source,title,audio_path,source_text_path,start_sec,end_sec,notes
```

`start_sec` and `end_sec` are optional. If present, the script clips that audio range before transcription. The source text file can still contain the full article because the script searches for the best matching source window.

See `eval_manifest.sample.csv`.

## Run

```bash
python tools/eval_whisper_wer.py \
  --manifest eval_manifest.sample.csv \
  --out-dir eval_results \
  --model small \
  --window-search \
  --number-mode standard \
  --error-clips
```

For a larger model:

```bash
python tools/eval_whisper_wer.py \
  --manifest eval_manifest.sample.csv \
  --out-dir eval_results_medium \
  --model medium \
  --window-search
```

For simple digit normalization:

```bash
python tools/eval_whisper_wer.py \
  --manifest eval_manifest.sample.csv \
  --out-dir eval_results_digits \
  --model small \
  --window-search \
  --number-mode digits-to-words
```

## Outputs

The output directory contains:

```text
results.csv
details.jsonl
summary_by_source.csv
selected_references/
transcripts/
audio_excerpts/  # only when --keep-excerpts is used
error_clips/     # only when --error-clips is used
```

`results.csv` has one row per sample with overall WER, edit counts, and slice error rates for numbers, proper nouns, attribution words, and acronyms.

`details.jsonl` includes the raw transcript, normalized transcript, selected reference window, and edit operations. Inspect this before drawing conclusions, especially for excerpts.

When `--error-clips` is enabled, `details.jsonl` also includes `error_contexts` entries with surrounding reference text, surrounding transcript text, approximate timestamps, and a short audio clip path for each highlighted error.

## Notebook

Open `notebooks/whisper_wer_eval.ipynb` to inspect and plot a run. By default it reads `eval_results/latest`. You can point it at another run with:

```bash
EVAL_RESULTS_DIR=/path/to/results jupyter notebook notebooks/whisper_wer_eval.ipynb
```

## How To Interpret

Use WER as a rough alarm bell, not as a final truth. Whisper is both the evaluator and another speech model, so it can miss audio issues or invent text errors. The selected reference window can also be wrong for short excerpts or repeated phrasing.

For a blog post, report:

- model used, for example `faster-whisper:small`
- number of samples
- whether samples were full episodes or excerpts
- mean WER by source
- number and proper noun error rates
- examples of representative mistakes from `details.jsonl`

The most useful finding is usually qualitative: whether errors change meaning, damage names, numbers, or attributions, or are merely small wording differences.

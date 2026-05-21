"""Convert raw LibriSpeech FLAC files to WebDataset shards.

Per-sample processing:
  1. Load FLAC audio with torchaudio, resample to 16 kHz if needed.
  2. Compute log-mel spectrogram → (80, T) float16 .npy via model/whisper_encoder.py.
  3. Apply whisper.normalizers.BasicTextNormalizer  → unformatted transcript.
  4. Four-pass formatted transcript:
       a. whisper.normalizers.EnglishTextNormalizer  → numbers/symbols to spoken form
       b. deepmultilingualpunctuation.PunctuationModel → punctuation + casing restored
       c. _capitalize_sentences → sentence-initial capitalisation
       d. spaCy en_core_web_sm PROPN tagger → proper nouns capitalised

When --labels_file is provided (output of scripts/precompute_labels.py), steps 3 and 4
are skipped entirely and labels are looked up from the precomputed JSONL file instead.
This avoids loading PunctuationModel and spaCy and reduces sharding time from ~1 hour
to a few minutes for large splits.

Two invocation modes:

  Single split (--input_dir + --split):
    python scripts/preprocess.py \\
      --input_dir  data/librispeech/LibriSpeech/train-clean-100 \\
      --output_dir data/shards/ \\
      --split      train-clean-100

  Full dataset in one run (--librispeech_dir):
    python scripts/preprocess.py \\
      --librispeech_dir data/librispeech/LibriSpeech/ \\
      --output_dir      data/shards/ \\
      --labels_file     data/labels.jsonl \\
      --seed            42

  The full-dataset mode discovers all train-* splits under --librispeech_dir,
  processes them sequentially, and writes a single unified manifest.jsonl.
  Labels and NLP models are loaded once and reused across all splits.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tarfile
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torchaudio

# Allow running from any working directory by ensuring the repo root is on sys.path.
# This lets us import model.whisper_encoder without requiring pip install -e .
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.whisper_encoder import SAMPLE_RATE, log_mel_spectrogram

# openai-whisper is permitted here (normalizers only — not used in model code)
from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer


def _capitalize_sentences(text: str) -> str:
    """Capitalise the first character and any character following sentence-ending punctuation.

    deepmultilingualpunctuation's decode step does not reliably capitalise with
    current transformers versions, so we apply this as a post-processing pass.

    Args:
        text: punctuation-restored string from PunctuationModel.restore_punctuation()

    Returns:
        string with sentence-initial letters capitalised
    """
    if not text:
        return text
    text = text[0].upper() + text[1:]
    return re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)


def _capitalize_proper_nouns(text: str, nlp) -> str:
    """Capitalise the first letter of every PROPN-tagged token detected by spaCy.

    Runs after _capitalize_sentences so punctuation and sentence boundaries are
    already in place, giving the POS tagger better context.

    Args:
        text: sentence-capitalised string
        nlp:  loaded spacy.Language model (en_core_web_sm)

    Returns:
        string with proper nouns capitalised in addition to sentence starts
    """
    if not text:
        return text
    doc   = nlp(text)
    chars = list(text)
    for token in doc:
        if token.pos_ == "PROPN":
            chars[token.idx] = chars[token.idx].upper()
    return "".join(chars)


def _iter_librispeech(
    input_dir: Path,
    seed: int | None = None,
) -> Iterator[tuple[str, Path, str]]:
    """Yield (key, flac_path, raw_transcript) for every utterance in input_dir.

    LibriSpeech directory layout:
        {input_dir}/{speaker_id}/{chapter_id}/{speaker_id}-{chapter_id}.trans.txt
        {input_dir}/{speaker_id}/{chapter_id}/{speaker_id}-{chapter_id}-{utt_id}.flac

    When seed is provided the entries are globally shuffled before yielding.
    This ensures every shard contains a cross-speaker mix so the WebDataset
    shuffle buffer does not need to be large to get diverse batches.

    Args:
        input_dir: root of a single LibriSpeech split (e.g. LibriSpeech/train-clean-100)
        seed:      RNG seed for global shuffle; None keeps sorted order

    Yields:
        (key, flac_path, raw_transcript) triples
    """
    import random

    entries: list[tuple[str, Path, str]] = []

    for trans_file in sorted(input_dir.rglob("*.trans.txt")):
        chapter_dir = trans_file.parent

        with trans_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                key, _, raw_text = line.partition(" ")
                flac_path = chapter_dir / f"{key}.flac"
                if not flac_path.exists():
                    print(f"Warning: missing FLAC for {key} — skipping", flush=True)
                    continue
                entries.append((key, flac_path, raw_text))

    if seed is not None:
        random.Random(seed).shuffle(entries)
    else:
        entries.sort(key=lambda t: t[0])

    yield from entries


def _load_audio(flac_path: Path) -> tuple[torch.Tensor, float]:
    """Load a FLAC file and return a mono 16 kHz waveform and its duration.

    Uses soundfile for I/O (reliably supports FLAC without torchcodec) and
    torchaudio only for resampling when the native sample rate differs from 16 kHz.

    Args:
        flac_path: path to a .flac file

    Returns:
        (waveform, duration_s) where waveform is (T,) float32 in [-1, 1]
    """
    import soundfile as sf

    data, sr = sf.read(str(flac_path), dtype="float32", always_2d=False)
    waveform = torch.from_numpy(data)

    if waveform.ndim > 1:
        waveform = waveform.mean(dim=-1)  # mix multi-channel to mono

    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

    duration_s = waveform.shape[0] / SAMPLE_RATE
    return waveform, duration_s


def _add_file(
    tar: tarfile.TarFile,
    name: str,
    data: bytes,
) -> None:
    """Write a bytes blob into an open TarFile under the given name.

    Args:
        tar:  open TarFile in write mode
        name: filename to write inside the archive
        data: raw bytes content
    """
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _load_labels(labels_file: Path) -> dict[str, dict]:
    """Load a precomputed labels JSONL file into a key → record dict.

    Args:
        labels_file: JSONL file produced by scripts/precompute_labels.py

    Returns:
        dict mapping utterance key → {"key": ..., "unformatted": ..., "formatted": ...}
    """
    print(f"Loading precomputed labels from {labels_file} …")
    precomputed: dict[str, dict] = {}
    with labels_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                precomputed[record["key"]] = record
    print(f"  Loaded {len(precomputed):,} precomputed label entries.")
    return precomputed


def _load_inline_models():
    """Load PunctuationModel and spaCy for inline (non-precomputed) label generation.

    Returns:
        (basic_norm, english_norm, punct_model, nlp)
    """
    import deepmultilingualpunctuation.punctuationmodel as _pmod
    from transformers import pipeline as _hf_pipeline

    def _fixed_punct_init(
        self,
        model: str = "oliverguhr/fullstop-punctuation-multilang-large",
    ) -> None:
        # aggregation_strategy=None avoids the deprecated grouped_entities kwarg
        self.pipe = _hf_pipeline(
            "token-classification",
            model,
            aggregation_strategy=None,
        )

    _pmod.PunctuationModel.__init__ = _fixed_punct_init

    from deepmultilingualpunctuation import PunctuationModel
    import spacy as _spacy

    print("Loading PunctuationModel …")
    punct_model = PunctuationModel()
    print("Loading spaCy en_core_web_sm …")
    nlp = _spacy.load("en_core_web_sm")
    return punct_model, nlp


def preprocess_split(
    input_dir: Path,
    output_dir: Path,
    split: str,
    manifest_f,
    shard_duration_mins: float = 30.0,
    max_duration_s: float = 30.0,
    seed: int | None = None,
    precomputed: dict[str, dict] | None = None,
    punct_model=None,
    nlp=None,
) -> tuple[int, int]:
    """Convert one LibriSpeech split directory to WebDataset shards.

    Shards are sized by total audio duration, not sample count. A new shard is
    opened once the cumulative duration of already-written samples meets or
    exceeds `shard_duration_mins`. This gives approximately uniform compute
    load per shard, which pairs well with the GCS prefetch window.

    Mels are stored at their natural length (no zero-padding). The DataLoader
    collation function pads to the batch maximum at training time.

    Samples longer than `max_duration_s` are skipped entirely: trimming audio
    without trimming the transcript would introduce label noise.

    Each sample key has three files inside the shard .tar:
        {key}.mel.npy          float16 numpy array, shape (80, T)  T = duration_s * 100
        {key}.unformatted.txt  BasicTextNormalizer output (UTF-8)
        {key}.formatted.txt    four-pass formatted transcript (UTF-8)

    Args:
        input_dir:           root of the extracted LibriSpeech split
        output_dir:          destination directory for .tar shards
        split:               shard filename prefix (e.g. "train-clean-100")
        manifest_f:          open writable file handle for manifest.jsonl lines
        shard_duration_mins: target total audio duration per shard, in minutes
        max_duration_s:      samples longer than this are skipped (default 30 s)
        seed:                global shuffle seed for speaker diversity; None = sorted order
        precomputed:         dict from _load_labels(); when provided, punct_model and nlp
                             are unused — labels are looked up by key instead
        punct_model:         PunctuationModel instance (only needed when precomputed is None)
        nlp:                 spaCy Language model (only needed when precomputed is None)

    Returns:
        (sample_count, shard_count) for this split
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_duration_secs = shard_duration_mins * 60.0
    basic_norm          = BasicTextNormalizer()
    english_norm        = EnglishTextNormalizer()

    shard_idx          = 0
    sample_count       = 0
    skipped_long       = 0
    current_tar: tarfile.TarFile | None = None
    current_shard_name = ""
    shard_duration_acc = 0.0   # cumulative audio seconds in the current shard

    def _open_next_shard() -> tarfile.TarFile:
        nonlocal shard_idx, current_shard_name, shard_duration_acc
        name = f"{split}-{shard_idx:06d}.tar"
        current_shard_name = name
        shard_idx         += 1
        shard_duration_acc = 0.0
        print(f"  → opening shard {name}", flush=True)
        return tarfile.open(output_dir / name, "w")

    try:
        for key, flac_path, raw_text in _iter_librispeech(input_dir, seed=seed):
            try:
                waveform, duration_s = _load_audio(flac_path)
            except Exception as exc:
                print(f"Warning: failed to load {flac_path}: {exc} — skipping", flush=True)
                continue

            # Skip without trimming — truncating audio mismatches the transcript label.
            if duration_s > max_duration_s:
                skipped_long += 1
                continue

            mel     = log_mel_spectrogram(waveform)   # (80, T) float32, T = duration_s * 100
            mel_f16 = mel.numpy().astype(np.float16)

            if precomputed is not None:
                record = precomputed.get(key)
                if record is None:
                    print(f"Warning: no precomputed label for {key} — skipping", flush=True)
                    continue
                unformatted = record["unformatted"]
                formatted   = record["formatted"]
            else:
                unformatted = basic_norm(raw_text)
                formatted   = _capitalize_proper_nouns(
                    _capitalize_sentences(
                        punct_model.restore_punctuation(english_norm(raw_text))
                    ),
                    nlp,
                )

            # Open the first shard, or a new one when the duration threshold is met.
            if current_tar is None or shard_duration_acc >= shard_duration_secs:
                if current_tar is not None:
                    current_tar.close()
                current_tar = _open_next_shard()

            buf = io.BytesIO()
            np.save(buf, mel_f16)
            _add_file(current_tar, f"{key}.mel.npy",         buf.getvalue())
            _add_file(current_tar, f"{key}.unformatted.txt", unformatted.encode("utf-8"))
            _add_file(current_tar, f"{key}.formatted.txt",   formatted.encode("utf-8"))

            manifest_f.write(json.dumps({
                "key":          key,
                "split":        split,
                "shard":        current_shard_name,
                "duration_s":   round(duration_s, 4),
                "n_mel_frames": mel_f16.shape[1],
                "unformatted":  unformatted,
                "formatted":    formatted,
            }) + "\n")

            shard_duration_acc += duration_s
            sample_count       += 1
            if sample_count % 500 == 0:
                print(f"  Processed {sample_count} samples …", flush=True)

    finally:
        if current_tar is not None:
            current_tar.close()

    skipped_msg = f"  ({skipped_long} skipped, duration > {max_duration_s}s)" if skipped_long else ""
    print(f"  {split}: {sample_count} samples → {shard_idx} shard(s){skipped_msg}", flush=True)
    return sample_count, shard_idx


def preprocess_all(
    librispeech_dir: Path,
    output_dir: Path,
    shard_duration_mins: float = 30.0,
    max_duration_s: float = 30.0,
    seed: int | None = None,
    labels_file: Path | None = None,
) -> None:
    """Process all train-* splits under librispeech_dir into a single output directory.

    Splits are discovered by scanning for subdirectories matching 'train-*' directly
    under librispeech_dir (e.g. train-clean-100, train-clean-360, train-other-500).
    They are processed in sorted order.

    Labels and NLP models are loaded once and shared across all splits. A single
    manifest.jsonl covering all splits is written to output_dir. Each manifest
    record includes a 'split' field so downstream tools can filter by split.

    Args:
        librispeech_dir:     root LibriSpeech directory containing train-* subdirs
        output_dir:          destination for all .tar shards and unified manifest.jsonl
        shard_duration_mins: target total audio duration per shard, in minutes
        max_duration_s:      samples longer than this are skipped
        seed:                global shuffle seed applied to every split independently
        labels_file:         JSONL from precompute_labels.py; skips NLP model loading
    """
    splits = sorted(
        d for d in librispeech_dir.iterdir()
        if d.is_dir() and d.name.startswith("train-")
    )
    if not splits:
        raise FileNotFoundError(
            f"No train-* subdirectories found under {librispeech_dir}"
        )

    print(f"Found {len(splits)} split(s): {[s.name for s in splits]}")

    precomputed  = _load_labels(labels_file) if labels_file is not None else None
    punct_model  = None
    nlp          = None
    if precomputed is None:
        punct_model, nlp = _load_inline_models()

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    total_samples = 0
    total_shards  = 0

    with manifest_path.open("w", encoding="utf-8") as manifest_f:
        for split_dir in splits:
            print(f"\n── Processing {split_dir.name} ──", flush=True)
            n_samples, n_shards = preprocess_split(
                input_dir=split_dir,
                output_dir=output_dir,
                split=split_dir.name,
                manifest_f=manifest_f,
                shard_duration_mins=shard_duration_mins,
                max_duration_s=max_duration_s,
                seed=seed,
                precomputed=precomputed,
                punct_model=punct_model,
                nlp=nlp,
            )
            total_samples += n_samples
            total_shards  += n_shards

    print(f"\nAll splits done. {total_samples:,} samples → {total_shards} shards")
    print(f"Manifest: {manifest_path}")


def main() -> None:
    """Parse CLI arguments and run preprocessing.

    Two modes (mutually exclusive):
      --librispeech_dir  process all train-* splits under that directory (full dataset)
      --input_dir + --split  process a single split
    """
    parser = argparse.ArgumentParser(
        description="Convert LibriSpeech FLAC files to WebDataset shards.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Full dataset (recommended):\n"
            "  python scripts/preprocess.py \\\n"
            "    --librispeech_dir data/librispeech/LibriSpeech/ \\\n"
            "    --output_dir data/shards/ \\\n"
            "    --labels_file data/labels.jsonl --seed 42\n\n"
            "  # Single split:\n"
            "  python scripts/preprocess.py \\\n"
            "    --input_dir data/librispeech/LibriSpeech/train-clean-100 \\\n"
            "    --output_dir data/shards/ --split train-clean-100 --seed 42\n"
        ),
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--librispeech_dir",
        type=Path,
        metavar="DIR",
        help=(
            "root LibriSpeech directory; all train-* subdirs are processed "
            "in order with a single shared manifest.jsonl"
        ),
    )
    mode.add_argument(
        "--input_dir",
        type=Path,
        metavar="DIR",
        help="root of a single LibriSpeech split (requires --split)",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="directory to write .tar shards and manifest.jsonl",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="shard filename prefix, e.g. train-clean-100 (required with --input_dir)",
    )
    parser.add_argument(
        "--shard_duration_mins",
        type=float,
        default=30.0,
        help="target total audio duration per shard in minutes  (default: 30)",
    )
    parser.add_argument(
        "--max_duration_s",
        type=float,
        default=30.0,
        help="skip samples longer than this many seconds  (default: 30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="global shuffle seed for speaker diversity; omit to keep sorted order",
    )
    parser.add_argument(
        "--labels_file",
        type=Path,
        default=None,
        help=(
            "JSONL file from scripts/precompute_labels.py; skips loading "
            "PunctuationModel and spaCy — labels are looked up by key"
        ),
    )
    args = parser.parse_args()

    if args.librispeech_dir is not None:
        preprocess_all(
            librispeech_dir=args.librispeech_dir,
            output_dir=args.output_dir,
            shard_duration_mins=args.shard_duration_mins,
            max_duration_s=args.max_duration_s,
            seed=args.seed,
            labels_file=args.labels_file,
        )
    else:
        if args.split is None:
            parser.error("--split is required when using --input_dir")

        precomputed = _load_labels(args.labels_file) if args.labels_file else None
        punct_model, nlp = (None, None) if precomputed is not None else _load_inline_models()

        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = args.output_dir / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as manifest_f:
            n_samples, n_shards = preprocess_split(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                split=args.split,
                manifest_f=manifest_f,
                shard_duration_mins=args.shard_duration_mins,
                max_duration_s=args.max_duration_s,
                seed=args.seed,
                precomputed=precomputed,
                punct_model=punct_model,
                nlp=nlp,
            )
        print(f"Manifest ({n_samples:,} samples): {manifest_path}")


if __name__ == "__main__":
    main()

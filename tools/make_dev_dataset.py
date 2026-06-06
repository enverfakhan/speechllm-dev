"""Create a minimal development dataset for single-speaker overfitting experiments.

Selects a LibriSpeech speaker with many chapters and ≤ 30 minutes of total audio,
or accepts an explicit --speaker_id. The smallest chapter(s) are held out as a
diagnostic shard; the remaining chapters become the training shards.

Shard format matches scripts/preprocess.py exactly, so the output can be used
directly with train.py (--shards_file for training, --diag_shard for diagnostics).

When --labels_file is omitted, BasicTextNormalizer is applied inline for the
unformatted label and the formatted label is set equal to it. This is acceptable
for --instruction_mode unformatted (the default dev mode) where formatted.txt is
never read during training.

Usage:
    # Auto-select speaker:
    python scripts/make_dev_dataset.py \\
      --librispeech_dir data/librispeech/LibriSpeech/train-clean-100 \\
      --output_dir      data/dev_shards/ \\
      --output_train    data/dev_train_shards.txt \\
      --output_diag     data/dev_diag_shard.txt

    # Explicit speaker, two holdout chapters, with precomputed labels:
    python scripts/make_dev_dataset.py \\
      --librispeech_dir data/librispeech/LibriSpeech/train-clean-100 \\
      --speaker_id      2196 \\
      --output_dir      data/dev_shards/ \\
      --output_train    data/dev_train_shards.txt \\
      --output_diag     data/dev_diag_shard.txt \\
      --holdout_chapters 2 \\
      --labels_file     data/labels.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.whisper_encoder import SAMPLE_RATE, log_mel_spectrogram
from whisper.normalizers import BasicTextNormalizer

# Skip utterances longer than this (matches preprocess.py)
_MAX_DURATION_S: float = 30.0


def _flac_duration(flac_path: Path) -> float:
    """Return the duration of a FLAC file in seconds by reading its header only.

    Uses soundfile.info() which reads the metadata block without decoding the
    audio stream — fast enough for scanning thousands of files during speaker
    selection.

    Args:
        flac_path: path to a .flac file

    Returns:
        duration in seconds
    """
    return float(sf.info(str(flac_path)).duration)


def _load_audio(flac_path: Path) -> tuple[torch.Tensor, float]:
    """Load a FLAC file and return a mono 16 kHz float32 waveform and its duration.

    Args:
        flac_path: path to a .flac file

    Returns:
        (waveform, duration_s) — waveform is (T,) float32 in [-1, 1]
    """
    data, sr = sf.read(str(flac_path), dtype="float32", always_2d=False)
    waveform  = torch.from_numpy(data)

    if waveform.ndim > 1:
        waveform = waveform.mean(dim=-1)

    if sr != SAMPLE_RATE:
        import torchaudio
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform  = resampler(waveform.unsqueeze(0)).squeeze(0)

    return waveform, float(waveform.shape[0]) / SAMPLE_RATE


def _add_to_tar(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    """Write a bytes blob into an open TarFile under the given name.

    Args:
        tar:  open TarFile in write mode
        name: filename inside the archive
        data: raw bytes content
    """
    info      = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _select_speaker(
    librispeech_dir: Path,
    max_total_duration_s: float = 1800.0,
) -> str:
    """Find the speaker with the most chapters whose total audio fits within budget.

    Scans all speaker subdirectories; uses header-only FLAC reads to measure
    total duration without decoding audio.

    Args:
        librispeech_dir:      root of a LibriSpeech split (e.g. train-clean-100)
        max_total_duration_s: upper bound on total audio in seconds (default 30 min)

    Returns:
        speaker_id string (the speaker subdirectory name)

    Raises:
        ValueError if no qualifying speaker is found
    """
    speaker_dirs = sorted(d for d in librispeech_dir.iterdir() if d.is_dir())
    print(f"Scanning {len(speaker_dirs)} speaker dirs …")

    best_id:       str | None = None
    best_chapters: int        = -1
    best_duration: float      = 0.0

    for speaker_dir in speaker_dirs:
        chapter_dirs = sorted(d for d in speaker_dir.iterdir() if d.is_dir())
        if not chapter_dirs:
            continue

        total_dur = sum(
            _flac_duration(flac)
            for ch in chapter_dirs
            for flac in ch.glob("*.flac")
        )

        if total_dur > max_total_duration_s:
            continue

        n_ch = len(chapter_dirs)
        if n_ch > best_chapters:
            best_chapters = n_ch
            best_id       = speaker_dir.name
            best_duration = total_dur

    if best_id is None:
        raise ValueError(
            f"No speaker found with total audio ≤ {max_total_duration_s / 60:.0f} min "
            f"under {librispeech_dir}. Try increasing --max_total_duration_s."
        )

    print(
        f"Selected speaker {best_id}: {best_chapters} chapters, "
        f"{best_duration / 60:.1f} min total audio."
    )
    return best_id


def _scan_chapters(speaker_dir: Path) -> dict[str, dict]:
    """Scan a speaker directory and return per-chapter metadata and utterance lists.

    LibriSpeech chapter layout:
        {speaker_dir}/{chapter_id}/{speaker_id}-{chapter_id}.trans.txt
        {speaker_dir}/{chapter_id}/{speaker_id}-{chapter_id}-{utt_id}.flac

    Args:
        speaker_dir: path to one speaker subdirectory

    Returns:
        dict mapping chapter_id → {
            "duration_s": float,
            "utterances": list of (key, flac_path, raw_text),
        }
    """
    chapters: dict[str, dict] = {}

    for chapter_dir in sorted(speaker_dir.iterdir()):
        if not chapter_dir.is_dir():
            continue

        trans_files = list(chapter_dir.glob("*.trans.txt"))
        if not trans_files:
            continue

        utterances: list[tuple[str, Path, str]] = []
        with trans_files[0].open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                key, _, raw_text = line.partition(" ")
                flac_path = chapter_dir / f"{key}.flac"
                if flac_path.exists():
                    utterances.append((key, flac_path, raw_text))

        total_dur = sum(_flac_duration(u[1]) for u in utterances)
        chapters[chapter_dir.name] = {
            "duration_s": total_dur,
            "utterances": utterances,
        }

    return chapters


def _write_shards(
    utterances: list[tuple[str, Path, str]],
    output_dir: Path,
    shard_prefix: str,
    labels: dict[str, dict] | None,
    basic_norm: BasicTextNormalizer,
    shard_duration_mins: float = 5.0,
    force_single: bool = False,
) -> list[Path]:
    """Write utterances to WebDataset .tar shards in the same format as preprocess.py.

    Each sample key is stored as three files inside the .tar:
        {key}.mel.npy          float16 numpy array, shape (80, T)
        {key}.unformatted.txt  BasicTextNormalizer output
        {key}.formatted.txt    precomputed formatted label or equal to unformatted

    When force_single is True all utterances go into exactly one .tar regardless of
    duration — used for the diagnostic shard.

    When labels is None, formatted is set equal to unformatted. This is fine for
    dev runs that use --instruction_mode unformatted.

    Args:
        utterances:          list of (key, flac_path, raw_text) triples
        output_dir:          directory to write .tar files
        shard_prefix:        prefix for shard filenames (e.g. "dev-train-1284")
        labels:              precomputed dict from labels.jsonl; None = inline BasicTextNorm
        basic_norm:          BasicTextNormalizer instance
        shard_duration_mins: target audio duration per shard; ignored when force_single
        force_single:        write all utterances to exactly one shard

    Returns:
        list of written shard paths (in creation order)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dur_s = shard_duration_mins * 60.0
    shard_idx   = 0
    written:    list[Path] = []
    cur_tar:    tarfile.TarFile | None = None
    cur_dur     = 0.0
    n_written   = 0
    n_skipped   = 0

    def _open_next() -> tarfile.TarFile:
        nonlocal shard_idx, cur_dur
        path = output_dir / f"{shard_prefix}-{shard_idx:06d}.tar"
        print(f"  → {path.name}", flush=True)
        written.append(path)
        shard_idx += 1
        cur_dur    = 0.0
        return tarfile.open(path, "w")

    try:
        for key, flac_path, raw_text in utterances:
            try:
                waveform, duration_s = _load_audio(flac_path)
            except Exception as exc:
                print(f"Warning: {flac_path.name}: {exc} — skipping", flush=True)
                n_skipped += 1
                continue

            if duration_s > _MAX_DURATION_S:
                n_skipped += 1
                continue

            mel     = log_mel_spectrogram(waveform)
            mel_f16 = mel.numpy().astype(np.float16)

            if labels is not None:
                record = labels.get(key)
                if record is None:
                    print(f"Warning: no precomputed label for {key} — skipping", flush=True)
                    n_skipped += 1
                    continue
                unformatted = record["unformatted"]
                formatted   = record["formatted"]
            else:
                unformatted = basic_norm(raw_text)
                formatted   = unformatted   # dev mode: formatted.txt is not accessed

            if cur_tar is None or (not force_single and cur_dur >= shard_dur_s):
                if cur_tar is not None:
                    cur_tar.close()
                cur_tar = _open_next()

            buf = io.BytesIO()
            np.save(buf, mel_f16)
            _add_to_tar(cur_tar, f"{key}.mel.npy",         buf.getvalue())
            _add_to_tar(cur_tar, f"{key}.unformatted.txt", unformatted.encode("utf-8"))
            _add_to_tar(cur_tar, f"{key}.formatted.txt",   formatted.encode("utf-8"))

            cur_dur  += duration_s
            n_written += 1

    finally:
        if cur_tar is not None:
            cur_tar.close()

    print(
        f"  {n_written} samples written"
        + (f", {n_skipped} skipped" if n_skipped else "")
        + f" → {len(written)} shard(s)",
        flush=True,
    )
    return written


def main() -> None:
    """Parse CLI arguments and create the development dataset."""
    p = argparse.ArgumentParser(
        description=(
            "Create a single-speaker development dataset for overfitting experiments. "
            "Selects a LibriSpeech speaker with many chapters and ≤ 30 min total audio, "
            "or uses an explicit --speaker_id. Holds out the smallest chapter(s) for "
            "diagnostics and writes the rest as training shards."
        ),
    )
    p.add_argument(
        "--librispeech_dir", type=Path, required=True,
        help=(
            "Root directory of a LibriSpeech split "
            "(e.g. data/librispeech/LibriSpeech/train-clean-100)."
        ),
    )
    p.add_argument(
        "--speaker_id", type=str, default=None,
        help=(
            "LibriSpeech speaker ID to use (the subdirectory name, e.g. '1284'). "
            "If omitted, automatically selects the speaker with the most chapters "
            "whose total audio is ≤ --max_total_duration_s."
        ),
    )
    p.add_argument(
        "--output_dir", type=Path, required=True,
        help="Directory to write .tar shard files.",
    )
    p.add_argument(
        "--output_train", type=Path, required=True,
        help="Text file to write training shard paths (one per line).",
    )
    p.add_argument(
        "--output_diag", type=Path, required=True,
        help="Text file to write the single diagnostic shard path (one line).",
    )
    p.add_argument(
        "--holdout_chapters", type=int, default=1,
        help=(
            "Number of chapters to hold out for the diagnostic shard, selected by "
            "ascending duration (smallest first). Default 1."
        ),
    )
    p.add_argument(
        "--shard_duration_mins", type=float, default=5.0,
        help="Target audio duration per training shard in minutes. Default 5.",
    )
    p.add_argument(
        "--labels_file", type=Path, default=None,
        help=(
            "Optional JSONL file of precomputed labels from scripts/precompute_labels.py. "
            "When omitted, BasicTextNormalizer is applied inline and the formatted label "
            "equals the unformatted label (suitable for --instruction_mode unformatted)."
        ),
    )
    p.add_argument(
        "--max_total_duration_s", type=float, default=1800.0,
        help=(
            "Maximum total audio duration in seconds for auto-speaker selection "
            "(default 1800 = 30 min). Ignored when --speaker_id is provided."
        ),
    )
    args = p.parse_args()

    # ── Resolve speaker ────────────────────────────────────────────────────────
    if args.speaker_id is not None:
        speaker_id  = args.speaker_id
        speaker_dir = args.librispeech_dir / speaker_id
        if not speaker_dir.is_dir():
            raise FileNotFoundError(f"Speaker directory not found: {speaker_dir}")
        print(f"Using speaker {speaker_id}.")
    else:
        speaker_id  = _select_speaker(args.librispeech_dir, args.max_total_duration_s)
        speaker_dir = args.librispeech_dir / speaker_id

    # ── Scan chapters ──────────────────────────────────────────────────────────
    chapters = _scan_chapters(speaker_dir)
    if not chapters:
        raise RuntimeError(f"No chapters found for speaker {speaker_id} in {speaker_dir}")

    sorted_chapters = sorted(chapters.items(), key=lambda kv: kv[1]["duration_s"])
    print(f"\nChapters for speaker {speaker_id} (sorted by duration):")
    for ch_id, info in sorted_chapters:
        n = len(info["utterances"])
        print(f"  chapter {ch_id}:  {info['duration_s'] / 60:.1f} min  ({n} utterances)")

    # ── Split train / holdout ──────────────────────────────────────────────────
    n_holdout = min(args.holdout_chapters, len(sorted_chapters) - 1)
    if n_holdout < 1:
        raise ValueError(
            f"Speaker {speaker_id} has only {len(sorted_chapters)} chapter(s); "
            "cannot hold out any while keeping at least one for training."
        )

    holdout_chapters = sorted_chapters[:n_holdout]
    train_chapters   = sorted_chapters[n_holdout:]

    holdout_utts = [u for _, info in holdout_chapters for u in info["utterances"]]
    train_utts   = [u for _, info in train_chapters   for u in info["utterances"]]
    holdout_dur  = sum(info["duration_s"] for _, info in holdout_chapters)
    train_dur    = sum(info["duration_s"] for _, info in train_chapters)

    print(
        f"\nHoldout ({n_holdout} chapter(s)):   "
        f"{holdout_dur / 60:.1f} min  ({len(holdout_utts)} utterances)"
    )
    print(
        f"Training ({len(train_chapters)} chapter(s)): "
        f"{train_dur / 60:.1f} min  ({len(train_utts)} utterances)"
    )

    # ── Load precomputed labels if provided ────────────────────────────────────
    labels: dict[str, dict] | None = None
    if args.labels_file is not None:
        print(f"\nLoading precomputed labels from {args.labels_file} …")
        labels = {}
        with args.labels_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record        = json.loads(line)
                    labels[record["key"]] = record
        print(f"  Loaded {len(labels):,} entries.")

    basic_norm = BasicTextNormalizer()

    # ── Write diagnostic shard (single .tar, all holdout utterances) ───────────
    print(f"\nWriting diagnostic shard …")
    diag_paths = _write_shards(
        holdout_utts,
        args.output_dir,
        shard_prefix=f"dev-diag-{speaker_id}",
        labels=labels,
        basic_norm=basic_norm,
        force_single=True,
    )
    if not diag_paths:
        raise RuntimeError("Diagnostic shard is empty — no valid utterances in holdout chapters.")

    # ── Write training shards (duration-split) ─────────────────────────────────
    print(f"\nWriting training shards …")
    train_paths = _write_shards(
        train_utts,
        args.output_dir,
        shard_prefix=f"dev-train-{speaker_id}",
        labels=labels,
        basic_norm=basic_norm,
        shard_duration_mins=args.shard_duration_mins,
    )
    if not train_paths:
        raise RuntimeError("No training shards written — no valid utterances in training chapters.")

    # ── Write shard list files ─────────────────────────────────────────────────
    args.output_train.parent.mkdir(parents=True, exist_ok=True)
    args.output_diag.parent.mkdir(parents=True, exist_ok=True)

    args.output_train.write_text(
        "\n".join(str(p) for p in train_paths) + "\n", encoding="utf-8"
    )
    args.output_diag.write_text(str(diag_paths[0]) + "\n", encoding="utf-8")

    print(f"\nDone.")
    print(f"  Training shards ({len(train_paths)}): {args.output_train}")
    print(f"  Diagnostic shard:          {args.output_diag}")
    print(f"\nSuggested train.py flags:")
    print(f"  --shards_file {args.output_train} \\")
    print(f"  --diag_shard  {diag_paths[0]}")


if __name__ == "__main__":
    main()

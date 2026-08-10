"""WebDataset-based dataloader for preprocessed LibriSpeech shards.

Consumes shards written by scripts/preprocess.py. Each shard sample contains:
  {key}.mel.npy          float16 numpy array, shape (80, T); T varies by utterance
  {key}.unformatted.txt  plain text, BasicTextNormalizer output
  {key}.formatted.txt    plain text, two-pass formatted transcript

Does not perform any preprocessing — mel computation and transcript normalisation
happen offline in scripts/preprocess.py.

Batch format returned by the DataLoader:
  mel                (B, 80, T_max)    float32  — zero-padded to T_max (multiple of 8)
  audio_lengths      (B,)              int64    — adapter output token count per sample
  instruction_ids    (B, T_inst_max)   int64    — tokenised instruction in pruned vocab
  instruction_lengths (B,)             int64    — real instruction token count per sample
  transcript_ids     (B, T_trans_max)  int64    — tokenised transcript in pruned vocab
  transcript_lengths (B,)              int64    — real transcript token count per sample

Pass this tuple directly to model.adapter.prepare_input().

In PAIRED mode (build_dataloader(..., paired=True)) each audio contributes both
prompt variants, so mel/audio_lengths keep one row per audio (B = n) while the
instruction/transcript tensors carry 2n interleaved sequence rows — see the
build_dataloader docstring for the full contract.

Epoch-level shard shuffling is the caller's responsibility:

    all_shards = list_shards("data/shards/train-{000000..000127}.tar")
    for epoch in range(n_epochs):
        epoch_shards = list(all_shards)
        random.Random(base_seed + epoch).shuffle(epoch_shards)
        loader = build_dataloader(epoch_shards, ...)
        for batch in loader:
            ...

This gives reproducible per-epoch diversity without relying on WebDataset's
internal shard shuffling, so the same list can be shared with a GCS prefetch
subprocess for overlapped I/O.
"""

from __future__ import annotations

import glob
import io
import json
import math
import random
import re
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.utils.data
import webdataset as wds


INSTRUCTION_VARIANTS: list[str] = [
    "Transcribe the following audio without formatting.",
    "Transcribe the following audio with proper formatting.",
]


class PrunedTokenizer:
    """Loads a HuggingFace tokenizer and remaps IDs to the pruned vocabulary space.

    Used in data.py for encoding during training and in train.py for decoding
    generated token IDs back to text during WER evaluation.
    """

    def __init__(self, tokenizer_path: Path) -> None:
        """Load tokenizer files and vocab_map from tokenizer_path.

        Args:
            tokenizer_path: directory produced by scripts/build_vocab.py, containing
                            the original Llama tokenizer files plus vocab_map.json
        """
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(str(tokenizer_path))

        with (Path(tokenizer_path) / "vocab_map.json").open(encoding="utf-8") as f:
            raw = json.load(f)
        # JSON keys are strings; convert to int → int
        self._vocab_map: dict[int, int] = {int(k): v for k, v in raw.items()}
        # Reverse map for decode(): pruned_id → original_id
        self._reverse_map: dict[int, int] = {v: k for k, v in self._vocab_map.items()}

    def encode(self, text: str) -> list[int]:
        """Tokenize text and remap to pruned vocabulary IDs.

        Tokens whose original IDs are absent from the pruned vocab (e.g. tokens
        unique to dev/test splits that were not seen during vocab building) are
        silently dropped, consistent with how decode() handles unknown IDs.

        Args:
            text: plain-text string

        Returns:
            list of token IDs in the pruned vocabulary space (unknown tokens omitted)
        """
        old_ids = self._tok.encode(text, add_special_tokens=False)
        return [self._vocab_map[i] for i in old_ids if i in self._vocab_map]

    def decode(self, pruned_ids: list[int]) -> str:
        """Convert pruned vocabulary IDs back to a text string.

        Silently skips any ID not present in the reverse map (e.g. the SEP
        token 40147, which is never a valid transcript token).

        Args:
            pruned_ids: list of token IDs in the pruned vocabulary space

        Returns:
            decoded text string
        """
        old_ids = [self._reverse_map[i] for i in pruned_ids if i in self._reverse_map]
        return self._tok.decode(old_ids, skip_special_tokens=True)


def list_shards(pattern: str) -> list[str]:
    """Expand a brace/glob pattern to a sorted list of existing local shard paths.

    Handles WebDataset-style brace patterns:
        "data/shards/train-{000000..000127}.tar"

    Falls back to shell glob if no brace range is found.

    Args:
        pattern: brace pattern or shell glob; GCS paths (gs://) are not supported
                 here — the two-stage architecture keeps GCS handling in the
                 prefetch subprocess and the DataLoader always reads local disk

    Returns:
        Sorted list of shard paths that exist on disk
    """
    m = re.search(r'\{(\d+)\.\.(\d+)\}', pattern)
    if m:
        lo, hi   = int(m.group(1)), int(m.group(2))
        width    = len(m.group(1))
        prefix   = pattern[:m.start()]
        suffix   = pattern[m.end():]
        candidates = [f"{prefix}{i:0{width}d}{suffix}" for i in range(lo, hi + 1)]
        return [p for p in candidates if Path(p).exists()]
    return sorted(glob.glob(pattern))


def build_dataloader(
    shard_pattern: str | list[str],
    tokenizer_path: Path,
    sep_token_id: int,
    batch_size: int,
    num_workers: int,
    instruction_variants: list[tuple[str, str]],
    shuffle_buffer: int = 1000,
    partial: bool = False,
    *,
    paired: bool = False,
) -> torch.utils.data.DataLoader:
    """Build a DataLoader over preprocessed WebDataset shards.

    Shards are consumed in the order provided — no internal shard shuffling.
    For per-epoch diversity, shuffle the shard list before calling this function
    (see module docstring for the canonical pattern).

    Args:
        shard_pattern:        brace/glob pattern string, or an explicit list of
                              shard paths already in the desired consumption order
        tokenizer_path:       path to pruned tokenizer directory (build_vocab.py output)
        sep_token_id:         SEP token ID in the pruned vocabulary; stored in
                              pruned_config.json and forwarded to prepare_input()
        batch_size:           SEQUENCES per batch; incomplete final batches are dropped
        num_workers:          DataLoader worker processes
        instruction_variants: list of (instruction_text, transcript_key) pairs.
                              Pass one pair to train on a single mode, or two for
                              joint training. transcript_key must be one of
                              "unformatted.txt" or "formatted.txt".
                              One pair is chosen uniformly at random per sample
                              (unpaired mode); paired mode uses both.
        shuffle_buffer:       in-flight sample buffer; 2–3× batch_size is sufficient
                              when shards are pre-shuffled on disk
        partial:              if True, include the final incomplete batch (useful for
                              small diagnostic shards where dropping it may leave no
                              batches at all)
        paired:               if True, every audio sample contributes BOTH prompt
                              variants (requires exactly two instruction_variants
                              with distinct transcript keys, and an even batch_size)

    Returns:
        DataLoader yielding 6-tuples:
            (mel, audio_lengths, instruction_ids, instruction_lengths,
             transcript_ids, transcript_lengths)

        Unpaired (paired=False) — every tensor has batch dim B = batch_size, one
        row per audio sample, one instruction variant drawn at random per sample.

        Paired (paired=True) — the batch holds n = batch_size // 2 unique audios
        and 2n sequences, so the two families of tensors have DIFFERENT batch dims:
            mel            (n, 80, T_max)   audio_lengths       (n,)
            instruction_*  (2n, …)          transcript_*        (2n, …)
        Sequence rows 2i and 2i+1 both belong to audio i, in instruction_variants
        order (row 2i = variants[0], row 2i+1 = variants[1]).  The caller is
        responsible for expanding the audio side to 2n rows before sequence
        assembly (training.py does this with adapter_out.repeat_interleave(2, 0),
        whose backward sums both rows' gradients back into the shared audio row).
    """
    if isinstance(shard_pattern, str):
        shards = list_shards(shard_pattern)
    else:
        shards = list(shard_pattern)

    if not shards:
        raise FileNotFoundError(f"No shards found for pattern: {shard_pattern!r}")

    if paired:
        if len(instruction_variants) != 2:
            raise ValueError(
                "paired=True requires exactly two instruction_variants (one per "
                f"prompt mode), got {len(instruction_variants)}"
            )
        if instruction_variants[0][1] == instruction_variants[1][1]:
            raise ValueError(
                "paired=True requires two DISTINCT transcript keys, got "
                f"{instruction_variants[0][1]!r} twice"
            )
        if batch_size % 2 != 0:
            raise ValueError(
                f"paired=True requires an even batch_size (it counts sequences, "
                f"two per audio), got {batch_size}"
            )

    tokenizer = PrunedTokenizer(tokenizer_path)

    def _process(sample: dict[str, Any]) -> tuple:
        mel = np.load(io.BytesIO(sample["mel.npy"])).astype(np.float32)  # (80, T)

        instruction_text, trans_key = random.choice(instruction_variants)
        transcript = sample[trans_key].decode("utf-8")

        return (mel, tokenizer.encode(instruction_text), tokenizer.encode(transcript))

    def _collate(samples: list[tuple]) -> tuple[torch.Tensor, ...]:
        mels, instr_lists, trans_lists = zip(*samples)
        B = len(mels)

        # ── Mel: pad to T_max (multiple of 8) ─────────────────────────────
        T_list = [m.shape[1] for m in mels]
        T_max  = math.ceil(max(T_list) / 8) * 8
        mel_batch = torch.zeros(B, 80, T_max, dtype=torch.float32)
        for i, m in enumerate(mels):
            mel_batch[i, :, : m.shape[1]] = torch.from_numpy(m)

        # audio_lengths[i] = adapter output tokens for sample i
        # encoder conv stride-2 → T_enc_i = T_mel_i // 2
        # adapter ceil-pool-4   → T_adapt_i = (T_enc_i + 3) // 4
        audio_lengths = torch.tensor(
            [(T // 2 + 3) // 4 for T in T_list], dtype=torch.long
        )

        # ── Instruction IDs ────────────────────────────────────────────────
        I_max      = max(len(ids) for ids in instr_lists)
        instr_ids  = torch.zeros(B, I_max, dtype=torch.long)
        instr_lens = torch.zeros(B, dtype=torch.long)
        for i, ids in enumerate(instr_lists):
            instr_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            instr_lens[i]            = len(ids)

        # ── Transcript IDs ─────────────────────────────────────────────────
        L_max      = max(len(ids) for ids in trans_lists)
        trans_ids  = torch.zeros(B, L_max, dtype=torch.long)
        trans_lens = torch.zeros(B, dtype=torch.long)
        for i, ids in enumerate(trans_lists):
            trans_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            trans_lens[i]            = len(ids)

        return (mel_batch, audio_lengths, instr_ids, instr_lens, trans_ids, trans_lens)

    def _process_paired(sample: dict[str, Any]) -> tuple:
        """Emit BOTH prompt variants for one audio sample.

        Returns (mel, inst_ids_a, trans_ids_a, inst_ids_b, trans_ids_b) where
        a/b follow instruction_variants order — the transcript key comes from
        the pair itself, so no key string is hardcoded here.
        """
        mel = np.load(io.BytesIO(sample["mel.npy"])).astype(np.float32)  # (80, T)

        out: list[Any] = [mel]
        for instruction_text, trans_key in instruction_variants:
            out.append(tokenizer.encode(instruction_text))
            out.append(tokenizer.encode(sample[trans_key].decode("utf-8")))
        return tuple(out)

    def _collate_paired(samples: list[tuple]) -> tuple[torch.Tensor, ...]:
        """Collate paired samples: n audio rows, 2n interleaved sequence rows."""
        mels, instr_a, trans_a, instr_b, trans_b = zip(*samples)
        n = len(mels)

        # ── Mel: one row per UNIQUE audio, padded to T_max (multiple of 8) ─────
        T_list = [m.shape[1] for m in mels]
        T_max  = math.ceil(max(T_list) / 8) * 8
        mel_batch = torch.zeros(n, 80, T_max, dtype=torch.float32)
        for i, m in enumerate(mels):
            mel_batch[i, :, : m.shape[1]] = torch.from_numpy(m)

        # Same formula as the unpaired path; one entry per unique audio.
        audio_lengths = torch.tensor(
            [(T // 2 + 3) // 4 for T in T_list], dtype=torch.long
        )

        # ── Interleave the two variants: row 2i = variant a, row 2i+1 = b ──────
        instr_lists = [ids for pair in zip(instr_a, instr_b) for ids in pair]
        trans_lists = [ids for pair in zip(trans_a, trans_b) for ids in pair]

        def _pad(id_lists: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
            L_max   = max(len(ids) for ids in id_lists)
            out     = torch.zeros(2 * n, L_max, dtype=torch.long)
            lengths = torch.zeros(2 * n, dtype=torch.long)
            for i, ids in enumerate(id_lists):
                out[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                lengths[i]         = len(ids)
            return out, lengths

        instr_ids, instr_lens = _pad(instr_lists)
        trans_ids, trans_lens = _pad(trans_lists)

        return (mel_batch, audio_lengths, instr_ids, instr_lens, trans_ids, trans_lens)

    dataset = wds.WebDataset(shards, shardshuffle=False, nodesplitter=wds.split_by_node)
    if paired:
        # batch_size counts sequences; each audio yields two of them.
        dataset = (
            dataset
            .map(_process_paired)
            .shuffle(shuffle_buffer)
            .batched(batch_size // 2, collation_fn=_collate_paired, partial=partial)
        )
    else:
        dataset = (
            dataset
            .map(_process)
            .shuffle(shuffle_buffer)
            .batched(batch_size, collation_fn=_collate, partial=partial)
        )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=None,   # batching already done by .batched() above
        num_workers=num_workers,
        pin_memory=True,
    )


def _eval_collate_batch(samples: list[tuple]) -> tuple:
    """Collate a list of eval samples into a single 8-tuple batch.

    Each sample is (mel_arr, ids_unfmt, ids_fmt, ref_unfmt, ref_fmt) where
    mel_arr is a float32 np.ndarray of shape (80, T).  Used by both
    build_eval_dataloader and build_sorted_eval_dataloader so the two produce
    identical tensor layouts.

    Returns:
        (mel, audio_lengths, unfmt_ids, unfmt_lens,
         fmt_ids, fmt_lens, refs_unfmt, refs_fmt)
    """
    mels, unfmt_lists, fmt_lists, refs_unfmt, refs_fmt = zip(*samples)
    B = len(mels)

    T_list = [m.shape[1] for m in mels]
    T_max  = math.ceil(max(T_list) / 8) * 8
    mel_batch = torch.zeros(B, 80, T_max, dtype=torch.float32)
    for i, m in enumerate(mels):
        mel_batch[i, :, : m.shape[1]] = torch.from_numpy(m)

    audio_lengths = torch.tensor(
        [(T // 2 + 3) // 4 for T in T_list], dtype=torch.long
    )

    def _pad_ids(id_lists: tuple) -> tuple[torch.Tensor, torch.Tensor]:
        I_max   = max(len(ids) for ids in id_lists)
        out     = torch.zeros(B, I_max, dtype=torch.long)
        lengths = torch.zeros(B, dtype=torch.long)
        for i, ids in enumerate(id_lists):
            out[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            lengths[i]         = len(ids)
        return out, lengths

    unfmt_ids, unfmt_lens = _pad_ids(unfmt_lists)
    fmt_ids,   fmt_lens   = _pad_ids(fmt_lists)

    return (
        mel_batch, audio_lengths,
        unfmt_ids, unfmt_lens,
        fmt_ids,   fmt_lens,
        list(refs_unfmt), list(refs_fmt),
    )


def build_eval_dataloader(
    shard_path: str | Path,
    tokenizer_path: Path,
    instruction_variants: list[str],
    batch_size: int = 8,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """Build a DataLoader for WER evaluation over a single eval shard.

    Returns both instruction variants and both reference transcripts per batch
    so the caller can evaluate how well the model follows each instruction.

    Args:
        shard_path:           path to a single .tar shard
        tokenizer_path:       path to pruned tokenizer directory
        instruction_variants: [unformatted_instruction, formatted_instruction];
                              must match the same two strings used in training
        batch_size:           samples per batch; last (partial) batch is included
        num_workers:          DataLoader worker processes

    Returns:
        DataLoader yielding 8-tuples:
            (mel, audio_lengths,
             unformatted_ids, unformatted_lens,
             formatted_ids,   formatted_lens,
             refs_unformatted, refs_formatted)
        where refs_* are list[str] of raw reference transcripts
    """
    tokenizer = PrunedTokenizer(tokenizer_path)
    ids_unfmt = tokenizer.encode(instruction_variants[0])
    ids_fmt   = tokenizer.encode(instruction_variants[1])

    def _eval_process(sample: dict[str, Any]) -> tuple:
        mel       = np.load(io.BytesIO(sample["mel.npy"])).astype(np.float32)
        ref_unfmt = sample["unformatted.txt"].decode("utf-8")
        ref_fmt   = sample["formatted.txt"].decode("utf-8")
        return (mel, ids_unfmt, ids_fmt, ref_unfmt, ref_fmt)

    dataset = (
        wds.WebDataset(str(shard_path), shardshuffle=False, nodesplitter=wds.split_by_node)
        .map(_eval_process)
        .batched(batch_size, collation_fn=_eval_collate_batch, partial=True)
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_sorted_eval_dataloader(
    shard_path: str | Path,
    tokenizer_path: Path,
    instruction_variants: list[str],
    batch_size: int = 8,
) -> list[tuple]:
    """Build a finite, length-sorted list of eval batches from a single .tar shard.

    Reads the shard fully via stdlib tarfile (deterministic, finite — NOT
    webdataset, which can loop), sorts samples ascending by audio_length, then
    chunks into contiguous batches of batch_size including the final partial batch.

    Args:
        shard_path:           path to a single .tar shard
        tokenizer_path:       path to pruned tokenizer directory
        instruction_variants: [unformatted_instruction, formatted_instruction]
        batch_size:           samples per batch; final partial batch is included

    Returns:
        List of 8-tuples in the same format as build_eval_dataloader:
            (mel, audio_lengths,
             unformatted_ids, unformatted_lens,
             formatted_ids,   formatted_lens,
             refs_unformatted: list[str], refs_formatted: list[str])
        Batches are sorted ascending by audio_length; CPU tensors.
    """
    tokenizer = PrunedTokenizer(tokenizer_path)
    ids_unfmt = tokenizer.encode(instruction_variants[0])
    ids_fmt   = tokenizer.encode(instruction_variants[1])

    # Read every member of the shard and group by key (split on first ".").
    groups: dict[str, dict[str, bytes]] = {}
    with tarfile.open(Path(shard_path), "r") as tf:
        for member in tf.getmembers():
            dot = member.name.find(".")
            if dot < 0:
                continue
            key = member.name[:dot]
            ext = member.name[dot + 1:]
            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            groups.setdefault(key, {})[ext] = fobj.read()

    _REQUIRED = {"mel.npy", "unformatted.txt", "formatted.txt"}
    complete = {k: v for k, v in groups.items() if _REQUIRED <= set(v.keys())}

    # Decode mel arrays and compute audio_lengths; sort ascending.
    raw: list[tuple[int, Any, str, str]] = []  # (audio_len, mel_arr, ref_unfmt, ref_fmt)
    for members in complete.values():
        mel = np.load(io.BytesIO(members["mel.npy"])).astype(np.float32)
        T   = mel.shape[1]
        raw.append(
            ((T // 2 + 3) // 4, mel,
             members["unformatted.txt"].decode("utf-8"),
             members["formatted.txt"].decode("utf-8")),
        )
    raw.sort(key=lambda x: x[0])

    # Chunk into contiguous batches and collate each one.
    batches: list[tuple] = []
    for start in range(0, len(raw), batch_size):
        chunk = raw[start : start + batch_size]
        samples = [
            (mel, ids_unfmt, ids_fmt, ref_unfmt, ref_fmt)
            for (_, mel, ref_unfmt, ref_fmt) in chunk
        ]
        batches.append(_eval_collate_batch(samples))

    return batches


if __name__ == "__main__":
    import argparse
    import sys
    import tempfile

    _p = argparse.ArgumentParser()
    _p.add_argument("--self-test", action="store_true")
    _args = _p.parse_args()

    if not _args.self_test:
        _p.print_help()
        sys.exit(0)

    import tarfile as _tf_mod

    # Synthetic shard written by both self-tests below: sample {i} carries a
    # zeroed mel of the requested length plus the transcripts "unfmt {i}" /
    # "fmt {i}", so a mock tokenizer can encode the sample index into the IDs.
    _MEL_LENGTHS = [40, 120, 80, 200, 160, 60, 100, 180, 140, 220, 240]

    def _write_synthetic_shard(path: Path, mel_lengths: list[int]) -> None:
        with _tf_mod.open(path, "w") as _tar:
            for _i, _T in enumerate(mel_lengths):
                _key = f"sample-{_i:04d}"
                _mel = np.zeros((80, _T), dtype=np.float16)
                _buf = io.BytesIO()
                np.save(_buf, _mel)
                _mel_bytes = _buf.getvalue()
                for _name, _data in [
                    (f"{_key}.mel.npy",         _mel_bytes),
                    (f"{_key}.unformatted.txt",  f"unfmt {_i}".encode()),
                    (f"{_key}.formatted.txt",    f"fmt {_i}".encode()),
                ]:
                    _info = _tf_mod.TarInfo(name=_name)
                    _info.size = len(_data)
                    _tar.addfile(_info, io.BytesIO(_data))

    # ── build_sorted_eval_dataloader self-test ────────────────────────────────
    with tempfile.TemporaryDirectory() as _tmp:
        _tmp_dir = Path(_tmp)
        _shard   = _tmp_dir / "test.tar"

        # Fake tokenizer dir with minimal vocab_map.json + pruned_config.json
        _tok_dir = _tmp_dir / "tokenizer"
        _tok_dir.mkdir()
        # We only need instruction IDs to be non-empty lists — mock PrunedTokenizer.
        _IDS_UNFMT = [1, 2, 3]
        _IDS_FMT   = [4, 5, 6, 7]
        _VARIANTS  = ["unfmt instruction", "fmt instruction"]

        _write_synthetic_shard(_shard, _MEL_LENGTHS)

        # Patch PrunedTokenizer to avoid needing a real tokenizer on disk.
        class _MockTokenizer:
            def encode(self, text: str) -> list[int]:
                return _IDS_UNFMT if "unfmt" in text else _IDS_FMT

        # When running as __main__, build_sorted_eval_dataloader looks up
        # PrunedTokenizer in __main__'s globals, so patch there directly.
        _orig_cls = globals()["PrunedTokenizer"]
        globals()["PrunedTokenizer"] = type(
            "_PatchedTokenizer", (_MockTokenizer,), {"__init__": lambda self, p: None}
        )

        try:
            batches = build_sorted_eval_dataloader(
                _shard, _tok_dir, _VARIANTS, batch_size=4
            )

            # Should have ceil(11/4) = 3 batches
            assert len(batches) == 3, f"Expected 3 batches, got {len(batches)}"

            # Collect all audio_lengths in order to verify ascending sort.
            all_lengths: list[int] = []
            for _b in batches:
                all_lengths.extend(_b[1].tolist())
            assert all_lengths == sorted(all_lengths), (
                f"Batches not sorted ascending: {all_lengths}"
            )

            # All 11 samples covered; no duplicates.
            assert len(all_lengths) == 11, f"Expected 11 samples total, got {len(all_lengths)}"

            # Final batch has 3 samples (11 % 4 == 3), not 4.
            assert batches[-1][0].shape[0] == 3, (
                f"Last batch should be partial (3), got {batches[-1][0].shape[0]}"
            )

            # Check 8-tuple structure.
            for _b in batches:
                _mel, _al, _ui, _ul, _fi, _fl, _ru, _rf = _b
                _B = _mel.shape[0]
                assert _mel.shape[1] == 80,               "mel dim 1 should be 80"
                assert _al.shape == (_B,),                "audio_lengths shape"
                assert _ui.shape[0] == _B,                "unfmt_ids batch dim"
                assert _fi.shape[0] == _B,                "fmt_ids batch dim"
                assert _ul.shape == (_B,),                "unfmt_lens shape"
                assert _fl.shape == (_B,),                "fmt_lens shape"
                assert len(_ru) == _B,                    "refs_unfmt length"
                assert len(_rf) == _B,                    "refs_fmt length"

            # Reference strings match what was written.
            _all_unfmt = []
            _all_fmt   = []
            for _b in batches:
                _all_unfmt.extend(_b[6])
                _all_fmt.extend(_b[7])
            assert all(_s.startswith("unfmt ") for _s in _all_unfmt), "unfmt refs wrong"
            assert all(_s.startswith("fmt ")   for _s in _all_fmt),   "fmt refs wrong"

        finally:
            globals()["PrunedTokenizer"] = _orig_cls

    print("[OK] build_sorted_eval_dataloader")

    # ── build_dataloader(paired=True) self-test ───────────────────────────────
    with tempfile.TemporaryDirectory() as _tmp:
        _tmp_dir = Path(_tmp)
        _shard   = _tmp_dir / "paired.tar"
        _tok_dir = _tmp_dir / "tokenizer"
        _tok_dir.mkdir()

        _write_synthetic_shard(_shard, _MEL_LENGTHS)

        # Instruction pairs in the canonical order: index 0 unformatted, 1 formatted.
        _PAIRS = [
            ("unfmt instruction", "unformatted.txt"),
            ("fmt instruction",   "formatted.txt"),
        ]

        class _PairedMockTokenizer:
            """Encodes the sample index into the IDs so pairing is checkable.

            "unfmt instruction" → [1, 2, 3]        "fmt instruction" → [4, 5, 6, 7]
            "unfmt {i}"         → [10, i]          "fmt {i}"         → [20, i]
            """
            def encode(self, text: str) -> list[int]:
                if text == "unfmt instruction":
                    return [1, 2, 3]
                if text == "fmt instruction":
                    return [4, 5, 6, 7]
                _kind, _idx = text.split()
                return [10 if _kind == "unfmt" else 20, int(_idx)]

        _orig_cls = globals()["PrunedTokenizer"]
        globals()["PrunedTokenizer"] = type(
            "_PatchedPairedTokenizer", (_PairedMockTokenizer,),
            {"__init__": lambda self, p: None},
        )

        try:
            # batch_size counts SEQUENCES: 4 → 2 audios × 2 prompt variants.
            _loader = build_dataloader(
                [str(_shard)],
                tokenizer_path       = _tok_dir,
                sep_token_id         = 40147,
                batch_size           = 4,
                num_workers          = 0,
                instruction_variants = _PAIRS,
                shuffle_buffer       = 4,
                partial              = False,
                paired               = True,
            )
            _batch = next(iter(_loader))
            _mel, _al, _ii, _il, _ti, _tl = _batch

            # Audio side keeps one row per unique audio; sequence side is 2n.
            assert _mel.shape[0] == 2,  f"mel batch dim should be 2, got {_mel.shape[0]}"
            assert _mel.shape[1] == 80, f"mel dim 1 should be 80, got {_mel.shape[1]}"
            assert _al.shape == (2,),   f"audio_lengths shape should be (2,), got {_al.shape}"
            assert _ii.shape[0] == 4,   f"instruction_ids batch dim should be 4, got {_ii.shape[0]}"
            assert _ti.shape[0] == 4,   f"transcript_ids batch dim should be 4, got {_ti.shape[0]}"
            assert _il.shape == (4,),   f"instruction_lengths shape, got {_il.shape}"
            assert _tl.shape == (4,),   f"transcript_lengths shape, got {_tl.shape}"

            # Rows 2i / 2i+1 are the unformatted / formatted variant of audio i.
            for _i in range(2):
                _u, _f = _ti[2 * _i].tolist(), _ti[2 * _i + 1].tolist()
                assert _u[0] == 10, f"row {2*_i} should be the unformatted variant, got {_u}"
                assert _f[0] == 20, f"row {2*_i+1} should be the formatted variant, got {_f}"
                assert _u[1] == _f[1], (
                    f"rows {2*_i}/{2*_i+1} must share one audio: {_u} vs {_f}"
                )
            # The two audios in the batch are distinct samples.
            assert _ti[0, 1].item() != _ti[2, 1].item(), "rows 0/1 and 2/3 must differ"

            # Instructions follow the same interleave (lengths 3 / 4 / 3 / 4).
            assert _il.tolist() == [3, 4, 3, 4], f"instruction lengths: {_il.tolist()}"
            assert _ii[0, :3].tolist() == [1, 2, 3],    _ii[0].tolist()
            assert _ii[1, :4].tolist() == [4, 5, 6, 7], _ii[1].tolist()

            # A single instruction variant cannot be paired.
            _raised = False
            try:
                build_dataloader(
                    [str(_shard)],
                    tokenizer_path       = _tok_dir,
                    sep_token_id         = 40147,
                    batch_size           = 4,
                    num_workers          = 0,
                    instruction_variants = [_PAIRS[0]],
                    paired               = True,
                )
            except ValueError as exc:
                _raised = True
                assert "two instruction_variants" in str(exc), str(exc)
            assert _raised, "paired=True with one instruction variant must raise ValueError"

            # Odd batch_size cannot be split into audio pairs.
            _raised = False
            try:
                build_dataloader(
                    [str(_shard)],
                    tokenizer_path       = _tok_dir,
                    sep_token_id         = 40147,
                    batch_size           = 5,
                    num_workers          = 0,
                    instruction_variants = _PAIRS,
                    paired               = True,
                )
            except ValueError as exc:
                _raised = True
                assert "even batch_size" in str(exc), str(exc)
            assert _raised, "paired=True with an odd batch_size must raise ValueError"

        finally:
            globals()["PrunedTokenizer"] = _orig_cls

    print("[OK] build_dataloader paired collate")

    print("\nPASSED")

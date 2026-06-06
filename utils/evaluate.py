"""WER evaluation engine for speech-llm.

evaluate_all_splits is the shared WER engine used by train.py (--eval_at_end)
and will be reused by tools/run_wer.py in a later step.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch

from utils.generate import greedy_generate

if TYPE_CHECKING:
    from data import PrunedTokenizer
    from model.adapter import AudioAdapter
    from model.llama import Llama
    from model.whisper_encoder import WhisperEncoder


def compute_wer(refs: list[str], hyps: list[str]) -> float:
    """Word error rate via jiwer.  Returns NaN when hyps is empty.

    No text normalisation is applied so scores reflect whether the model
    actually follows the formatting instruction.
    """
    import jiwer
    return jiwer.wer(refs, hyps) if hyps else float("nan")


def evaluate_all_splits(
    encoder:      "WhisperEncoder",
    adapter:      "AudioAdapter",
    llama:        "Llama",
    eval_loaders: dict[str, torch.utils.data.DataLoader],
    tokenizer:    "PrunedTokenizer",
    sep_token_id: int,
    device:       torch.device,
    max_batches:  int | None = None,
    n_samples:    int = 20,
    sample_seed:  int = 0,
) -> tuple[dict[str, float], list[dict]]:
    """Run batched greedy WER evaluation on every eval split with both instructions.

    For each split, generation is run twice per batch — once with the unformatted
    instruction and once with the formatted instruction — and WER is reported
    separately for each.  No text normalisation is applied so the scores reflect
    whether the model actually follows the formatting instruction.

    Also randomly samples up to n_samples (reference, hypothesis) pairs per split
    per instruction type for qualitative inspection.

    Args:
        eval_loaders:  split name → DataLoader (from build_eval_dataloader)
        tokenizer:     PrunedTokenizer for decoding generated IDs to text
        max_batches:   cap per split (None = full eval)
        n_samples:     number of (ref, hyp) pairs to sample per split per type
        sample_seed:   RNG seed for reproducible sampling

    Returns:
        wer_dict:    keys like "dev-clean/unformatted" and "dev-clean/formatted"
        sample_rows: list of dicts with keys split, type, reference, hypothesis
    """
    encoder.eval()
    adapter.eval()
    llama.eval()

    results:     dict[str, float] = {}
    sample_rows: list[dict]       = []
    rng = random.Random(sample_seed)

    for split_name, loader in eval_loaders.items():
        pairs_unfmt: list[tuple[str, str]] = []   # (ref, hyp)
        pairs_fmt:   list[tuple[str, str]] = []

        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            (mel, audio_lengths,
             unfmt_ids, unfmt_lens,
             fmt_ids,   fmt_lens,
             refs_unformatted, refs_formatted) = batch

            mel           = mel.to(device)
            audio_lengths = audio_lengths.to(device)
            unfmt_ids     = unfmt_ids.to(device)
            unfmt_lens    = unfmt_lens.to(device)
            fmt_ids       = fmt_ids.to(device)
            fmt_lens      = fmt_lens.to(device)

            batch_hyps_unfmt = greedy_generate(
                encoder, adapter, llama,
                mel, audio_lengths, unfmt_ids, unfmt_lens,
                sep_token_id=sep_token_id,
            )
            batch_hyps_fmt = greedy_generate(
                encoder, adapter, llama,
                mel, audio_lengths, fmt_ids, fmt_lens,
                sep_token_id=sep_token_id,
            )

            for i in range(len(batch_hyps_unfmt)):
                pairs_unfmt.append((refs_unformatted[i], tokenizer.decode(batch_hyps_unfmt[i])))
                pairs_fmt.append((refs_formatted[i],     tokenizer.decode(batch_hyps_fmt[i])))

        refs_unfmt = [r for r, _ in pairs_unfmt]
        hyps_unfmt = [h for _, h in pairs_unfmt]
        refs_fmt   = [r for r, _ in pairs_fmt]
        hyps_fmt   = [h for _, h in pairs_fmt]

        wer_unfmt = compute_wer(refs_unfmt, hyps_unfmt)
        wer_fmt   = compute_wer(refs_fmt,   hyps_fmt)

        n = len(hyps_unfmt)
        print(f"  WER {split_name}/unformatted: {wer_unfmt:.1%}  ({n} samples)")
        print(f"  WER {split_name}/formatted:   {wer_fmt:.1%}")

        results[f"{split_name}/unformatted"] = wer_unfmt
        results[f"{split_name}/formatted"]   = wer_fmt

        # Random sample for qualitative table
        for ref, hyp in rng.sample(pairs_unfmt, min(n_samples, len(pairs_unfmt))):
            sample_rows.append({
                "split": split_name, "type": "unformatted",
                "reference": ref, "hypothesis": hyp,
            })
        for ref, hyp in rng.sample(pairs_fmt, min(n_samples, len(pairs_fmt))):
            sample_rows.append({
                "split": split_name, "type": "formatted",
                "reference": ref, "hypothesis": hyp,
            })

    encoder.train()
    adapter.train()
    llama.train()

    return results, sample_rows

"""WER evaluation engine for speech-llm.

evaluate_all_splits is the shared WER engine used by train.py (--eval_at_end)
and will be reused by tools/run_wer.py in a later step.
"""

from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING, Any

import torch

from utils.generate import greedy_generate

if TYPE_CHECKING:
    from data import PrunedTokenizer
    from model.adapter import BridgeAdapter
    from model.llama import Llama
    from model.whisper_encoder import WhisperEncoder


def compute_wer(refs: list[str], hyps: list[str]) -> float:
    """Word error rate via jiwer.  Returns NaN when hyps is empty.

    No text normalisation is applied so scores reflect whether the model
    actually follows the formatting instruction.
    """
    import jiwer
    return jiwer.wer(refs, hyps) if hyps else float("nan")


def _transcription_row(
    split_name: str, fmt_name: str, triple: tuple[str, str, str],
) -> dict:
    """Build one transcription record from a (key, reference, hypothesis) triple.

    Shared by the sampled and the full return paths so both emit exactly the
    same field names — tools/count_degeneracies.py consumes either.
    """
    key, ref, hyp = triple
    return {
        "key": key, "split": split_name, "type": fmt_name,
        "reference": ref, "hypothesis": hyp,
    }


def evaluate_all_splits(
    encoder:      "WhisperEncoder",
    adapter:      "BridgeAdapter",
    llama:        "Llama",
    eval_loaders: dict[str, Any],
    tokenizer:    "PrunedTokenizer",
    sep_token_id: int,
    device:       torch.device,
    max_batches:       int | None = None,
    n_samples:         int = 20,
    sample_seed:       int = 0,
    formats:           list[str] | None = None,
    progress_interval: float | None = None,
    return_all_transcriptions: bool = False,
) -> tuple[dict[str, float], list[dict]] | tuple[dict[str, float], list[dict], list[dict]]:
    """Run batched greedy WER evaluation on every eval split.

    Assumes a finite, single-pass loader — use build_sorted_eval_dataloader
    from data.py, which reads via tarfile and never loops.

    For each split, generation is run once per requested format and WER is
    reported separately for each.  No text normalisation is applied so the
    scores reflect whether the model actually follows the formatting instruction.

    Also randomly samples up to n_samples (reference, hypothesis) pairs per
    split per format for qualitative inspection.

    Args:
        eval_loaders:      split name → finite iterable of 9-tuples
                           (from build_sorted_eval_dataloader)
        tokenizer:         PrunedTokenizer for decoding generated IDs to text
        max_batches:       cap per split (None = full eval)
        n_samples:         number of (ref, hyp) pairs to sample per split per format
        sample_seed:       RNG seed for reproducible sampling
        formats:           which instruction variants to run; None (default) runs both.
                           Pass ["unformatted"] or ["formatted"] to restrict.
        progress_interval: print a progress line every this many seconds; None = silent.
        return_all_transcriptions:
                           when True, ALSO return every (ref, hyp) pair evaluated,
                           not just the n_samples subset — for post-hoc analysis
                           tools that must not require a re-run to see a sample
                           they did not happen to draw.  Off by default because
                           the sampled rows feed a W&B table, which must stay small.

    Returns:
        wer_dict:    keys like "dev-clean/unformatted" and/or "dev-clean/formatted"
        sample_rows: list of dicts with keys key, split, type, reference, hypothesis
                     (up to n_samples per split per format)
        all_rows:    ONLY when return_all_transcriptions=True — the same dict
                     shape, one entry per evaluated sample per format, in
                     loader (length-sorted) order.  The return is a 3-tuple in
                     that case and a 2-tuple otherwise.
    """
    run_unfmt = formats is None or "unformatted" in formats
    run_fmt   = formats is None or "formatted"   in formats

    encoder.eval()
    adapter.eval()
    llama.eval()

    results:     dict[str, float] = {}
    sample_rows: list[dict]       = []
    all_rows:    list[dict]       = []
    rng = random.Random(sample_seed)

    for split_name, loader in eval_loaders.items():
        # (key, ref, hyp) — the key lets a post-hoc tool attribute a bad
        # hypothesis back to its utterance.
        pairs_unfmt: list[tuple[str, str, str]] = []
        pairs_fmt:   list[tuple[str, str, str]] = []
        n_processed   = 0
        t_split_start = time.perf_counter()
        t_last_print  = t_split_start

        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            (mel, audio_lengths,
             unfmt_ids, unfmt_lens,
             fmt_ids,   fmt_lens,
             refs_unformatted, refs_formatted, keys) = batch

            mel           = mel.to(device)
            audio_lengths = audio_lengths.to(device)
            B = audio_lengths.shape[0]

            # Cap generation at the longest reference in this batch (in tokens) + 10
            # so a single runaway sample can't stall the whole eval run.
            all_refs = (
                (refs_unformatted if run_unfmt else []) +
                (refs_formatted   if run_fmt   else [])
            )
            max_new_tokens = max(len(tokenizer.encode(r)) for r in all_refs) + 10

            if run_unfmt:
                unfmt_ids  = unfmt_ids.to(device)
                unfmt_lens = unfmt_lens.to(device)
                batch_hyps_unfmt = greedy_generate(
                    encoder, adapter, llama,
                    mel, audio_lengths, unfmt_ids, unfmt_lens,
                    sep_token_id=sep_token_id,
                    max_new_tokens=max_new_tokens,
                )
                for i, hyp_ids in enumerate(batch_hyps_unfmt):
                    pairs_unfmt.append(
                        (keys[i], refs_unformatted[i], tokenizer.decode(hyp_ids))
                    )

            if run_fmt:
                fmt_ids  = fmt_ids.to(device)
                fmt_lens = fmt_lens.to(device)
                batch_hyps_fmt = greedy_generate(
                    encoder, adapter, llama,
                    mel, audio_lengths, fmt_ids, fmt_lens,
                    sep_token_id=sep_token_id,
                    max_new_tokens=max_new_tokens,
                )
                for i, hyp_ids in enumerate(batch_hyps_fmt):
                    pairs_fmt.append(
                        (keys[i], refs_formatted[i], tokenizer.decode(hyp_ids))
                    )

            n_processed += B

            if progress_interval is not None:
                t_now = time.perf_counter()
                if t_now - t_last_print >= progress_interval:
                    elapsed = t_now - t_split_start
                    rate    = n_processed / elapsed if elapsed > 0 else 0.0
                    mins, secs = divmod(int(elapsed), 60)
                    elapsed_str = f"{mins}m{secs:02d}s" if mins else f"{secs}s"
                    print(f"  [{split_name}] batch {batch_idx + 1}  "
                          f"{n_processed:,} samples  "
                          f"{rate:.1f} samples/s  "
                          f"{elapsed_str} elapsed")
                    t_last_print = t_now

        for fmt_name, pairs, run_it in (
            ("unformatted", pairs_unfmt, run_unfmt),
            ("formatted",   pairs_fmt,   run_fmt),
        ):
            if not run_it:
                continue
            refs = [r for _, r, _ in pairs]
            hyps = [h for _, _, h in pairs]
            wer  = compute_wer(refs, hyps)
            label = f"{split_name}/{fmt_name}:"
            print(f"  WER {label:<28} {wer:.1%}  ({len(hyps)} samples)")
            results[f"{split_name}/{fmt_name}"] = wer

            for triple in rng.sample(pairs, min(n_samples, len(pairs))):
                sample_rows.append(_transcription_row(split_name, fmt_name, triple))
            if return_all_transcriptions:
                all_rows.extend(
                    _transcription_row(split_name, fmt_name, t) for t in pairs
                )

    encoder.train()
    adapter.train()
    llama.train()

    if return_all_transcriptions:
        return results, sample_rows, all_rows
    return results, sample_rows

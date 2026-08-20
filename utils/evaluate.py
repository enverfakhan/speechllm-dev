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
    from model.sequence import ChatTemplate
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


def evaluate_all_splits(
    encoder:      "WhisperEncoder",
    adapter:      "BridgeAdapter",
    llama:        "Llama",
    eval_loaders: dict[str, Any],
    tokenizer:    "PrunedTokenizer",
    terminator_id: int,
    device:       torch.device,
    max_batches:       int | None = None,
    n_samples:         int = 20,
    sample_seed:       int = 0,
    formats:           list[str] | None = None,
    progress_interval: float | None = None,
    chat:              "ChatTemplate | None" = None,
) -> tuple[dict[str, float], list[dict]]:
    """Run batched greedy WER evaluation on every eval split.

    Assumes a finite, single-pass loader — use build_sorted_eval_dataloader
    from data.py, which reads via tarfile and never loops.

    For each split, generation is run once per requested format and WER is
    reported separately for each.  No text normalisation is applied so the
    scores reflect whether the model actually follows the formatting instruction.

    Also randomly samples up to n_samples (reference, hypothesis) pairs per
    split per format for qualitative inspection.

    Args:
        eval_loaders:      split name → finite iterable of 8-tuples
                           (from build_sorted_eval_dataloader)
        tokenizer:         PrunedTokenizer for decoding generated IDs to text
        terminator_id:     EOS token generation stops on — SEP under the flat
                           convention, <|eot_id|> under the chat one
        max_batches:       cap per split (None = full eval)
        n_samples:         number of (ref, hyp) pairs to sample per split per format
        sample_seed:       RNG seed for reproducible sampling
        formats:           which instruction variants to run; None (default) runs both.
                           Pass ["unformatted"] or ["formatted"] to restrict.
        progress_interval: print a progress line every this many seconds; None = silent.
        chat:              ChatTemplate when the run uses the chat input
                           convention, None for the flat one

    Returns:
        wer_dict:    keys like "dev-clean/unformatted" and/or "dev-clean/formatted"
        sample_rows: list of dicts with keys split, type, reference, hypothesis
    """
    run_unfmt = formats is None or "unformatted" in formats
    run_fmt   = formats is None or "formatted"   in formats

    encoder.eval()
    adapter.eval()
    llama.eval()

    results:     dict[str, float] = {}
    sample_rows: list[dict]       = []
    rng = random.Random(sample_seed)

    for split_name, loader in eval_loaders.items():
        pairs_unfmt: list[tuple[str, str]] = []   # (ref, hyp)
        pairs_fmt:   list[tuple[str, str]] = []
        n_processed   = 0
        t_split_start = time.perf_counter()
        t_last_print  = t_split_start

        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            (mel, audio_lengths,
             unfmt_ids, unfmt_lens,
             fmt_ids,   fmt_lens,
             refs_unformatted, refs_formatted) = batch

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
                    stop_token_id=terminator_id,
                    max_new_tokens=max_new_tokens,
                    chat=chat,
                )
                for i, hyp_ids in enumerate(batch_hyps_unfmt):
                    pairs_unfmt.append(
                        (refs_unformatted[i], tokenizer.decode(hyp_ids))
                    )

            if run_fmt:
                fmt_ids  = fmt_ids.to(device)
                fmt_lens = fmt_lens.to(device)
                batch_hyps_fmt = greedy_generate(
                    encoder, adapter, llama,
                    mel, audio_lengths, fmt_ids, fmt_lens,
                    stop_token_id=terminator_id,
                    max_new_tokens=max_new_tokens,
                    chat=chat,
                )
                for i, hyp_ids in enumerate(batch_hyps_fmt):
                    pairs_fmt.append(
                        (refs_formatted[i], tokenizer.decode(hyp_ids))
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

        if run_unfmt:
            refs_u = [r for r, _ in pairs_unfmt]
            hyps_u = [h for _, h in pairs_unfmt]
            wer_u  = compute_wer(refs_u, hyps_u)
            print(f"  WER {split_name}/unformatted: {wer_u:.1%}  ({len(hyps_u)} samples)")
            results[f"{split_name}/unformatted"] = wer_u
            for ref, hyp in rng.sample(pairs_unfmt, min(n_samples, len(pairs_unfmt))):
                sample_rows.append({
                    "split": split_name, "type": "unformatted",
                    "reference": ref, "hypothesis": hyp,
                })

        if run_fmt:
            refs_f = [r for r, _ in pairs_fmt]
            hyps_f = [h for _, h in pairs_fmt]
            wer_f  = compute_wer(refs_f, hyps_f)
            print(f"  WER {split_name}/formatted:   {wer_f:.1%}  ({len(hyps_f)} samples)")
            results[f"{split_name}/formatted"] = wer_f
            for ref, hyp in rng.sample(pairs_fmt, min(n_samples, len(pairs_fmt))):
                sample_rows.append({
                    "split": split_name, "type": "formatted",
                    "reference": ref, "hypothesis": hyp,
                })

    encoder.train()
    adapter.train()
    llama.train()

    return results, sample_rows

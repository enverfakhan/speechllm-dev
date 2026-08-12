"""Compute corpus-level loss baselines from preprocessed WebDataset shards.

Baselines computed:
  - Uniform:     log(vocab_size)  — theoretical maximum, model knows nothing
  - Unigram:     H(token freq)    — model knows marginal word frequencies
  - Bigram:      H(token | prev)  — model knows pairwise transition frequencies
  - First-token: H(first-word freq) — best possible without audio signal

Interpretation ladder for loss/eval_rest:
  loss > uniform_loss     → worse than random (broken)
  loss < uniform_loss     → knows something
  loss < unigram_loss     → using sequential context (not just frequencies)
  loss < bigram_loss      → using context longer than one token
  loss < first_token_loss → (only applies to loss/eval_first_token)
                            model is using the audio signal

Usage:
python scripts/compute_baselines.py \
    --shards data/full_training_shards.txt \
    --tokenizer data/pruned_tokenizer/ \
    --output baselines.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import webdataset as wds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import PrunedTokenizer


def compute_baselines(
    shard_paths: list[str],
    tokenizer: PrunedTokenizer,
    vocab_size: int,
) -> dict[str, float]:
    """Compute unigram, bigram, and first-token loss baselines from shards.

    All losses are in nats (natural log). Converting to bits: divide by log(2).

    The bigram loss is the average negative log-likelihood under the empirical
    bigram model: for each token w_i with previous token w_{i-1}, the model
    assigns p(w_i | w_{i-1}) = count(w_{i-1}, w_i) / count(w_{i-1}).

    This is computed efficiently without storing the full bigram matrix:
    bigram_loss = -1/N * sum_i log[ count(w_{i-1}, w_i) / count(w_{i-1}) ]
               = -1/N * sum_i [ log count(w_{i-1}, w_i) - log count(w_{i-1}) ]

    We store only the bigram counts as a dict-of-Counters, which is sparse
    and memory-efficient for natural language (most bigrams never appear).

    Args:
        shard_paths: list of .tar shard paths to scan
        tokenizer:   PrunedTokenizer for encoding transcripts
        vocab_size:  pruned vocabulary size, from the tokenizer's
                     pruned_config.json — sets the uniform baseline log(V)

    Returns:
        dict of baseline values and metadata
    """
    # Unigram counts
    token_counts: Counter[int] = Counter()

    # Bigram counts: bigram_counts[w_{i-1}][w_i] = count
    # Using defaultdict(Counter) keeps memory proportional to observed bigrams
    bigram_counts: defaultdict[int, Counter[int]] = defaultdict(Counter)

    # First-token counts
    first_token_counts: Counter[int] = Counter()

    total_tokens      = 0
    total_utterances  = 0
    # Bigram loss is computed over tokens 1..N (not token 0, which has no predecessor)
    total_bigram_tokens = 0

    dataset = wds.WebDataset(shard_paths, shardshuffle=False)

    for sample in dataset:
        try:
            transcript = sample["unformatted.txt"].decode("utf-8").strip()
        except KeyError:
            continue

        ids = tokenizer.encode(transcript)
        if not ids:
            continue

        # Unigram
        token_counts.update(ids)
        total_tokens += len(ids)

        # First token
        first_token_counts[ids[0]] += 1
        total_utterances += 1

        # Bigram — iterate over consecutive pairs within each utterance.
        # We do NOT create a bigram across utterance boundaries because
        # there is no linguistic relationship between the last word of one
        # utterance and the first word of the next.
        for prev, curr in zip(ids[:-1], ids[1:]):
            bigram_counts[prev][curr] += 1
            total_bigram_tokens += 1

    if total_tokens == 0:
        raise ValueError("No tokens found — check shard paths and transcript keys")

    # ── Uniform baseline ──────────────────────────────────────────────────
    uniform_loss = math.log(vocab_size)

    # ── Unigram loss: H = -sum_v p(v) log p(v) ───────────────────────────
    unigram_loss = 0.0
    for count in token_counts.values():
        p = count / total_tokens
        unigram_loss -= p * math.log(p)

    # ── Bigram loss: -1/N * sum_i log p(w_i | w_{i-1}) ───────────────────
    #
    # Computed as the average NLL under the empirical bigram model.
    # Equivalent to the conditional entropy H(W_i | W_{i-1}).
    #
    # Note: this is the *training* bigram loss — it uses the same corpus to
    # both estimate and evaluate the model, so it is a lower bound on what
    # a bigram model would achieve on held-out data. On a small single-speaker
    # corpus it will be optimistic. Use it as a directional reference, not
    # an absolute target.
    bigram_nll_sum = 0.0
    for prev_token, next_counter in bigram_counts.items():
        prev_count  = token_counts[prev_token]
        for curr_token, pair_count in next_counter.items():
            p_conditional = pair_count / prev_count
            # Each (prev, curr) pair contributes pair_count times
            bigram_nll_sum += pair_count * math.log(p_conditional)
    bigram_loss = -bigram_nll_sum / total_bigram_tokens

    # ── First-token loss: H of first-word distribution ────────────────────
    first_token_loss = 0.0
    for count in first_token_counts.values():
        p = count / total_utterances
        first_token_loss -= p * math.log(p)

    # ── Vocab coverage ────────────────────────────────────────────────────
    vocab_coverage      = len(token_counts) / vocab_size
    bigram_types_seen   = sum(len(v) for v in bigram_counts.values())

    return {
        # Baseline loss values (nats)
        "uniform_loss":           uniform_loss,
        "unigram_loss":           unigram_loss,
        "bigram_loss":            bigram_loss,
        "first_token_loss":       first_token_loss,

        # Perplexities
        "uniform_perplexity":     math.exp(uniform_loss),
        "unigram_perplexity":     math.exp(unigram_loss),
        "bigram_perplexity":      math.exp(bigram_loss),
        "first_token_perplexity": math.exp(first_token_loss),

        # Corpus statistics
        "total_tokens":           total_tokens,
        "total_utterances":       total_utterances,
        "total_bigram_tokens":    total_bigram_tokens,
        "vocab_coverage":         vocab_coverage,
        "unique_bigrams_seen":    bigram_types_seen,

        # Interpretation notes
        "_note_bigram": (
            "Computed on training corpus — optimistic lower bound. "
            "loss/eval_rest below this means model uses context > 1 token."
        ),
        "_note_first_token": (
            "loss/eval_first_token below this means model uses audio signal, "
            "not just which words tend to start utterances."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards",    required=True,
                        help="path to shard list .txt file or glob pattern")
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output",    required=True, type=Path,
                        help="path to write baselines.json")
    args = parser.parse_args()

    shard_list_path = Path(args.shards)
    if shard_list_path.suffix == ".txt":
        shards = [
            line.strip()
            for line in shard_list_path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    else:
        from data import list_shards
        shards = list_shards(args.shards)

    print(f"Scanning {len(shards)} shard(s) ...")
    tokenizer  = PrunedTokenizer(args.tokenizer)
    # Read the size from the tokenizer that produced these ids rather than
    # hardcoding it: the uniform baseline is log(vocab_size), so a stale
    # constant silently shifts the reference line every W&B run is judged
    # against. Rebuilding the vocab (tools/build_vocab.py) changes this number.
    with (args.tokenizer / "pruned_config.json").open() as f:
        vocab_size = json.load(f)["vocab_size"]
    baselines  = compute_baselines(shards, tokenizer, vocab_size)

    sep = "─" * 56
    print(f"\n{sep}")
    print(f"  CORPUS BASELINES")
    print(sep)
    print(f"  Utterances        : {baselines['total_utterances']:,}")
    print(f"  Total tokens      : {baselines['total_tokens']:,}")
    print(f"  Unique bigrams    : {baselines['unique_bigrams_seen']:,}")
    print(f"  Vocab coverage    : {baselines['vocab_coverage']:.1%} of pruned vocab")
    print(sep)
    print(f"  {'Baseline':<18}  {'Loss':>7}  {'PPL':>10}  Interpretation")
    print(sep)
    print(f"  {'Uniform':<18}  {baselines['uniform_loss']:>7.4f}  "
          f"{baselines['uniform_perplexity']:>10,.0f}  "
          f"model knows nothing")
    print(f"  {'Unigram':<18}  {baselines['unigram_loss']:>7.4f}  "
          f"{baselines['unigram_perplexity']:>10.1f}  "
          f"knows word frequencies")
    print(f"  {'Bigram':<18}  {baselines['bigram_loss']:>7.4f}  "
          f"{baselines['bigram_perplexity']:>10.1f}  "
          f"knows pairwise transitions (train corpus, optimistic)")
    print(f"  {'First-token':<18}  {baselines['first_token_loss']:>7.4f}  "
          f"{baselines['first_token_perplexity']:>10.1f}  "
          f"best possible without audio")
    print(sep)
    print(f"\n  Training milestones for loss/eval_rest:")
    print(f"    < {baselines['uniform_loss']:.2f}  model knows something")
    print(f"    < {baselines['unigram_loss']:.2f}  model uses sequential context")
    print(f"    < {baselines['bigram_loss']:.2f}  model uses context > 1 token")
    print(f"\n  Training milestone for loss/eval_first_token:")
    print(f"    < {baselines['first_token_loss']:.2f}  model uses audio signal")
    print(sep + "\n")

    args.output.write_text(json.dumps(baselines, indent=2))
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
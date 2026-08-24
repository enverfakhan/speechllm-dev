#!/usr/bin/env python3
"""Build a word -> training-frequency table for tools/analyze_slices.py.

WHY THIS EXISTS
---------------
`analyze_slices.py` slices eval WER by whether an utterance contains a RARE
word. Without a frequency table it falls back to eval-set hapaxes, which is a
badly degenerate proxy: on LibriSpeech dev-clean that fallback flags 893 of 958
utterances as "rare" (WER 7.03 vs 7.12 overall — no discrimination at all).
Against real training counts the same slice flags 162 utterances at WER 11.09
vs 6.18 for common ones. The table is what makes the slice mean anything.

Counts come from the UNFORMATTED labels, which are the corpus's canonical
lowercase word forms, normalized with the same `normalize()` the analyzer uses
for its lookups — the two sides must agree or every lookup misses.

Records marked `validation: failed` are skipped, matching what
tools/build_vocab.py and tools/preprocess.py drop (CLAUDE.md Decision 005), so
the frequencies describe the corpus that was actually trained on.

Pure standard library. Runs anywhere, takes seconds.

Usage:
    python tools/build_vocab_freq.py \
        --labels_file data/labels.jsonl \
        --output data/vocab_freq.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_slices import normalize  # noqa: E402  (stdlib-only, same tools/ dir)


def build(labels_file: Path, field: str = "unformatted") -> tuple[Counter, int, int]:
    """Count normalized word frequencies over the labels file.

    Args:
        labels_file: labels.jsonl, one record per utterance
        field:       which label text to count ("unformatted" by default)

    Returns:
        (counter, n_used, n_skipped)
    """
    counts: Counter = Counter()
    used = skipped = 0
    with labels_file.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            # Same exclusion the vocabulary and the shards use.
            if rec.get("validation") == "failed":
                skipped += 1
                continue
            counts.update(normalize(rec[field]).split())
            used += 1
    return counts, used, skipped


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels_file", type=Path, default=Path("data/labels.jsonl"))
    p.add_argument("--output", type=Path, default=Path("data/vocab_freq.json"))
    p.add_argument("--field", type=str, default="unformatted",
                   choices=("unformatted", "formatted"))
    args = p.parse_args(argv)

    if not args.labels_file.exists():
        raise SystemExit(f"{args.labels_file}: not found")

    counts, used, skipped = build(args.labels_file, args.field)
    if not counts:
        raise SystemExit(f"{args.labels_file}: no usable records found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(counts, sort_keys=True))

    print(f"utterances counted : {used}")
    print(f"skipped (failed)   : {skipped}")
    print(f"distinct words     : {len(counts)}")
    print(f"total tokens       : {sum(counts.values())}")
    print(f"count<=5 (rare)    : {sum(1 for c in counts.values() if c <= 5)}")
    print(f"hapax (count==1)   : {sum(1 for c in counts.values() if c == 1)}")
    print(f"written            : {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

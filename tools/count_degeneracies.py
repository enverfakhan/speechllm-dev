"""Count decoding degeneracies in run_wer.py sample transcriptions.

run_wer.py writes one JSONL file per (step, split) holding sampled
(reference, hypothesis) pairs:

    {"checkpoint": "...", "step": 4680, "split": "dev-clean",
     "type": "unformatted", "reference": "...", "hypothesis": "..."}

This tool reads any number of those files and reports, per
(checkpoint/step, split, format), how often the decoder failed in a
characteristic way.  WER alone cannot tell a model that is uniformly a bit
wrong from one that empties out on half the corpus and loops on the rest;
these counts split those apart.

Categories (a hypothesis may fall into several):

    empty          hypothesis strips to ""
    truncation     len(hyp_words) / len(ref_words) < 0.7   (and not empty)
    hallucination  len(hyp_words) / len(ref_words) > 1.3
    loop           some word n-gram (n = 1..4) repeats >= 3 times in a row
    style_flip     UNFORMATTED only: the hypothesis is capitalised or
                   punctuated, i.e. the model ignored the instruction

Outputs a CSV of counts and rates, plus a companion .txt holding the worst
example per category per group (largest deviation).

Pure standard library — no torch, no jiwer.  Runs anywhere.

Usage:
    python tools/count_degeneracies.py \\
        --inputs 'results/*.jsonl' \\
        --output results/degeneracies.csv

    python tools/count_degeneracies.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ── Thresholds ────────────────────────────────────────────────────────────────
# Length ratios bracketing "the right amount of text".  Below TRUNCATION_RATIO
# the decoder stopped early (usually an early SEP); above HALLUCINATION_RATIO it
# kept going past the audio.
TRUNCATION_RATIO:    float = 0.7
HALLUCINATION_RATIO: float = 1.3

# A loop is an n-gram repeated at least this many times back to back.  Three is
# the smallest count that is not plausibly natural English ("very very" is; "very
# very very" is not).
LOOP_MIN_REPEATS: int = 3
LOOP_MAX_N:       int = 4

# Characters that only ever appear in the formatted variant.  The unformatted
# labels are upper-case-free and punctuation-free, so any of these in an
# "unformatted" hypothesis means the instruction was ignored.
STYLE_PUNCTUATION: str = '.,!?;:"'

CATEGORIES: tuple[str, ...] = ("empty", "truncation", "hallucination", "loop", "style_flip")


# ── Detectors ─────────────────────────────────────────────────────────────────

def longest_ngram_repeat(words: list[str], max_n: int = LOOP_MAX_N) -> tuple[int, int, str]:
    """Find the longest run of a consecutively repeated word n-gram.

    Sliding start, greedy repeat count: for every n and every start position the
    block words[start:start+n] is extended as long as the next n words equal it.

    Args:
        words: word-level tokens of the hypothesis
        max_n: largest n-gram size to consider

    Returns:
        (repeats, n, ngram_text) for the winning run; (1, 0, "") when nothing
        repeats at all.  repeats counts the block itself, so "a a a" → 3.
    """
    best_reps, best_n, best_text = 1, 0, ""

    for n in range(1, max_n + 1):
        if len(words) < 2 * n:
            continue
        for start in range(len(words) - n + 1):
            block = words[start:start + n]
            reps  = 1
            nxt   = start + n
            while words[nxt:nxt + n] == block:
                reps += 1
                nxt  += n
            if reps > best_reps:
                best_reps, best_n, best_text = reps, n, " ".join(block)

    return best_reps, best_n, best_text


def style_markers(hypothesis: str) -> int:
    """Count formatting markers in a hypothesis that should carry none.

    Args:
        hypothesis: decoded text produced under the *unformatted* instruction

    Returns:
        Number of markers: 1 if the first character is upper case, plus one per
        occurrence of a character in STYLE_PUNCTUATION.  0 means clean.
    """
    stripped = hypothesis.strip()
    if not stripped:
        return 0
    leading = 1 if stripped[0].isupper() else 0
    return leading + sum(1 for ch in stripped if ch in STYLE_PUNCTUATION)


def classify(reference: str, hypothesis: str, fmt: str) -> dict[str, float]:
    """Classify one (reference, hypothesis) pair.

    Args:
        reference:  ground-truth transcript
        hypothesis: decoded transcript
        fmt:        "unformatted" or "formatted"; style_flip applies to the
                    former only

    Returns:
        category name → deviation magnitude, for the categories that fired.
        The deviation is "how badly", used to pick the worst example per group:
        empty → reference length, truncation/hallucination → distance past the
        ratio threshold, loop → repeat count, style_flip → marker count.
    """
    hits: dict[str, float] = {}

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if not hypothesis.strip():
        hits["empty"] = float(len(ref_words))
    elif ref_words:
        # Ratio categories need a non-empty reference to divide by, and an empty
        # hypothesis is already accounted for above (its ratio of 0 would double
        # count as a truncation).
        ratio = len(hyp_words) / len(ref_words)
        if ratio < TRUNCATION_RATIO:
            hits["truncation"] = TRUNCATION_RATIO - ratio
        elif ratio > HALLUCINATION_RATIO:
            hits["hallucination"] = ratio - HALLUCINATION_RATIO

    reps, _, _ = longest_ngram_repeat(hyp_words)
    if reps >= LOOP_MIN_REPEATS:
        hits["loop"] = float(reps)

    if fmt == "unformatted":
        markers = style_markers(hypothesis)
        if markers > 0:
            hits["style_flip"] = float(markers)

    return hits


# ── Aggregation ───────────────────────────────────────────────────────────────

@dataclass
class Group:
    """Counts and worst examples for one (checkpoint/step, split, format)."""
    checkpoint: str
    step:       int
    split:      str
    fmt:        str
    n:          int                  = 0
    counts:     dict[str, int]       = field(default_factory=lambda: {c: 0 for c in CATEGORIES})
    # category → (deviation, reference, hypothesis)
    worst:      dict[str, tuple[float, str, str]] = field(default_factory=dict)

    def add(self, reference: str, hypothesis: str) -> None:
        """Classify one utterance and fold it into this group's tallies."""
        self.n += 1
        for category, deviation in classify(reference, hypothesis, self.fmt).items():
            self.counts[category] += 1
            if category not in self.worst or deviation > self.worst[category][0]:
                self.worst[category] = (deviation, reference, hypothesis)

    def rate(self, category: str) -> float:
        """Fraction of utterances in this group that hit `category`."""
        return self.counts[category] / self.n if self.n else 0.0

    def applies(self, category: str) -> bool:
        """False for categories that are undefined for this group's format."""
        return category != "style_flip" or self.fmt == "unformatted"


def read_rows(paths: list[Path]) -> list[dict]:
    """Read every JSONL file into a flat list of sample dicts.

    Args:
        paths: JSONL files as written by run_wer.py

    Returns:
        One dict per line, in file order.

    Raises:
        KeyError: a line is missing one of the required fields.
    """
    required = ("split", "type", "reference", "hypothesis")
    rows: list[dict] = []
    for path in paths:
        with path.open() as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                missing = [k for k in required if k not in row]
                if missing:
                    raise KeyError(f"{path}:{lineno} missing field(s) {missing}")
                rows.append(row)
    return rows


def aggregate(rows: list[dict]) -> list[Group]:
    """Bucket sample rows by (checkpoint, step, split, format) and classify each.

    Args:
        rows: sample dicts from read_rows()

    Returns:
        Groups sorted by (step, split, format).
    """
    groups: dict[tuple[str, int, str, str], Group] = {}
    for row in rows:
        key = (
            str(row.get("checkpoint", "")),
            int(row.get("step", 0)),
            str(row["split"]),
            str(row["type"]),
        )
        if key not in groups:
            groups[key] = Group(checkpoint=key[0], step=key[1], split=key[2], fmt=key[3])
        groups[key].add(str(row["reference"]), str(row["hypothesis"]))

    return sorted(groups.values(), key=lambda g: (g.step, g.split, g.fmt))


# ── Output ────────────────────────────────────────────────────────────────────

def _csv_fieldnames() -> list[str]:
    names = ["checkpoint", "step", "split", "format", "n_utterances"]
    for category in CATEGORIES:
        names += [category, f"{category}_rate"]
    return names


def write_csv(groups: list[Group], output: Path) -> None:
    """Write per-group counts and rates.

    Columns that do not apply to a group (style_flip on a formatted split) are
    left blank rather than zero, so a genuine zero stays distinguishable from
    "not measured here".
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_csv_fieldnames())
        writer.writeheader()
        for g in groups:
            row: dict = {
                "checkpoint":   g.checkpoint,
                "step":         g.step,
                "split":        g.split,
                "format":       g.fmt,
                "n_utterances": g.n,
            }
            for category in CATEGORIES:
                if g.applies(category):
                    row[category]              = g.counts[category]
                    row[f"{category}_rate"]    = f"{g.rate(category):.4f}"
                else:
                    row[category]              = ""
                    row[f"{category}_rate"]    = ""
            writer.writerow(row)


def write_worst(groups: list[Group], output: Path) -> None:
    """Write the worst example per category per group to a companion .txt."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for g in groups:
            f.write(f"{'=' * 78}\nstep {g.step}  {g.split}  {g.fmt}  "
                    f"({g.n} utterances)  [{g.checkpoint}]\n{'=' * 78}\n")
            for category in CATEGORIES:
                if not g.applies(category):
                    continue
                if category not in g.worst:
                    f.write(f"\n-- {category}: none\n")
                    continue
                deviation, ref, hyp = g.worst[category]
                f.write(
                    f"\n-- {category}: {g.counts[category]}/{g.n} "
                    f"({g.rate(category):.1%})  worst deviation {deviation:.3f}\n"
                    f"   REF: {ref}\n"
                    f"   HYP: {hyp}\n"
                )
            f.write("\n")


def print_summary(groups: list[Group]) -> None:
    """Print a readable counts-and-rates table to stdout."""
    hdr = (f"{'Step':>8}  {'Split':<12}  {'Format':<12}  {'N':>5}  "
           + "  ".join(f"{c:>13}" for c in CATEGORIES))
    print(f"\n── Degeneracy rates ──\n{hdr}\n{'-' * len(hdr)}")
    for g in groups:
        cells = []
        for category in CATEGORIES:
            cells.append(
                f"{g.counts[category]:>4d} {g.rate(category):>7.1%}"
                if g.applies(category) else f"{'—':>13}"
            )
        print(f"{g.step:>8}  {g.split:<12}  {g.fmt:<12}  {g.n:>5}  " + "  ".join(cells))


# ── CLI ───────────────────────────────────────────────────────────────────────

def _expand_inputs(patterns: list[str]) -> list[Path]:
    """Glob-expand each pattern; deduplicate and sort the union."""
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(pattern))   # literal path; fails loudly on open
    return sorted(set(paths))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count decoding degeneracies in run_wer.py sample JSONL files.",
    )
    parser.add_argument(
        "--inputs", nargs="+", metavar="GLOB",
        help="One or more JSONL paths or shell globs (e.g. 'results/*.jsonl'). "
             "Quote globs so this tool expands them, not the shell.",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Output CSV path.  The worst-example report is written alongside it "
             "as <output stem>.worst.txt.",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Run the self-test with synthetic data and exit.",
    )
    args = parser.parse_args(argv)
    if not args.self_test:
        if not args.inputs:
            parser.error("--inputs is required unless --self-test is passed")
        if args.output is None:
            parser.error("--output is required unless --self-test is passed")
    return args


def main(argv: list[str] | None = None) -> None:
    """Read sample JSONL files, count degeneracies, write CSV + worst-example txt."""
    args = _parse_args(argv)
    if args.self_test:
        _self_test()
        return

    paths = _expand_inputs(args.inputs)
    print(f"Reading {len(paths)} JSONL file(s).")
    rows = read_rows(paths)
    if not rows:
        print("[warn] No sample rows found — nothing to count.")
        return

    groups = aggregate(rows)
    print(f"{len(rows)} utterances in {len(groups)} group(s).")

    print_summary(groups)

    write_csv(groups, args.output)
    worst_path = args.output.with_suffix(".worst.txt")
    write_worst(groups, worst_path)
    print(f"\nCounts  → {args.output}\nExamples → {worst_path}")


# ── Self-test ──────────────────────────────────────────────────────────────────

def _self_test() -> None:
    """Five synthetic hypotheses, one per category, end to end through the CLI."""
    import tempfile

    ref10 = "the quick brown fox jumps over the lazy dog again"   # 10 words

    # One utterance per category.  Each is built so it triggers exactly the
    # category it is named for — the assertions below pin that down.
    samples = [
        # empty: decoder emitted SEP immediately
        (ref10, "   "),
        # truncation: 5/10 = 0.50 < 0.70
        (ref10, "the quick brown fox jumps"),
        # hallucination: 15/10 = 1.50 > 1.30, all words distinct (no loop)
        (ref10, "the quick brown fox jumps over a lazy dog again and then some more words"),
        # loop: "sat" three times in a row; ratio 11/10 = 1.10 is in range
        (ref10, "the cat sat sat sat on the mat by the door"),
        # style_flip: leading capital + full stop; ratio 10/10 = 1.00
        (ref10, "The quick brown fox jumps over the lazy dog again."),
    ]

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        jsonl = tmp / "0004680_dev-clean.jsonl"
        with jsonl.open("w") as f:
            for ref, hyp in samples:
                f.write(json.dumps({
                    "checkpoint": "checkpoints/test/step0004680.pt",
                    "step": 4680, "split": "dev-clean", "type": "unformatted",
                    "reference": ref, "hypothesis": hyp,
                }) + "\n")
            # A formatted row: style_flip must NOT be counted for it.
            f.write(json.dumps({
                "checkpoint": "checkpoints/test/step0004680.pt",
                "step": 4680, "split": "dev-clean", "type": "formatted",
                "reference": ref10, "hypothesis": "The quick brown fox jumps over the lazy dog again.",
            }) + "\n")

        out = tmp / "degeneracies.csv"
        main(argv=["--inputs", str(jsonl), "--output", str(out)])

        # ── Per-utterance classification ──────────────────────────────────────
        expected = [
            {"empty"},
            {"truncation"},
            {"hallucination"},
            {"loop"},
            {"style_flip"},
        ]
        for (ref, hyp), want in zip(samples, expected):
            got = set(classify(ref, hyp, "unformatted"))
            assert got == want, f"{hyp!r}: expected {want}, got {got}"
        print("[OK] each synthetic hypothesis hits exactly its own category")

        # ── Loop detector detail ──────────────────────────────────────────────
        reps, n, text = longest_ngram_repeat("the cat sat sat sat on the mat".split())
        assert (reps, n, text) == (3, 1, "sat"), (reps, n, text)
        reps2, n2, text2 = longest_ngram_repeat("go home go home go home now".split())
        assert (reps2, n2, text2) == (3, 2, "go home"), (reps2, n2, text2)
        assert longest_ngram_repeat("no repeats at all here".split())[0] == 1
        print("[OK] longest_ngram_repeat: unigram, bigram, and none")

        # ── Aggregated counts ─────────────────────────────────────────────────
        groups = aggregate(read_rows([jsonl]))
        assert len(groups) == 2, [(g.split, g.fmt) for g in groups]
        unfmt = next(g for g in groups if g.fmt == "unformatted")
        fmt   = next(g for g in groups if g.fmt == "formatted")

        assert unfmt.n == 5, unfmt.n
        for category in CATEGORIES:
            assert unfmt.counts[category] == 1, (
                f"{category}: expected 1, got {unfmt.counts[category]}"
            )
            assert abs(unfmt.rate(category) - 0.2) < 1e-9, unfmt.rate(category)
        assert fmt.counts["style_flip"] == 0 and not fmt.applies("style_flip"), (
            "style_flip must not be counted on a formatted group"
        )
        print("[OK] group counts: 1 of each category over 5 utterances (rate 0.2)")

        # ── Worst examples ────────────────────────────────────────────────────
        assert set(unfmt.worst) == set(CATEGORIES), set(unfmt.worst)
        assert unfmt.worst["loop"][0] == 3.0, unfmt.worst["loop"]
        assert abs(unfmt.worst["truncation"][0] - 0.2) < 1e-9, unfmt.worst["truncation"]
        assert abs(unfmt.worst["hallucination"][0] - 0.2) < 1e-9, unfmt.worst["hallucination"]
        print("[OK] worst example recorded per category")

        # ── Files on disk ─────────────────────────────────────────────────────
        worst_path = out.with_suffix(".worst.txt")
        assert out.exists() and worst_path.exists(), "CSV and worst-example txt must exist"
        with out.open() as f:
            csv_rows = list(csv.DictReader(f))
        assert len(csv_rows) == 2, csv_rows
        unfmt_row = next(r for r in csv_rows if r["format"] == "unformatted")
        fmt_row   = next(r for r in csv_rows if r["format"] == "formatted")
        assert unfmt_row["n_utterances"] == "5", unfmt_row
        assert unfmt_row["loop"] == "1" and unfmt_row["loop_rate"] == "0.2000", unfmt_row
        assert unfmt_row["style_flip"] == "1", unfmt_row
        assert fmt_row["style_flip"] == "" and fmt_row["style_flip_rate"] == "", fmt_row
        worst_text = worst_path.read_text()
        for category in CATEGORIES:
            assert f"-- {category}:" in worst_text, f"{category} missing from worst report"
        print("[OK] CSV and worst-example report written")

    print("\nPASSED")


if __name__ == "__main__":
    main(argv=sys.argv[1:])

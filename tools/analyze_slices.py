#!/usr/bin/env python3
"""Fine-grained WER / degeneracy / instruction-adherence slicing over run_wer.py output.

WHY THIS EXISTS
---------------
`run_wer.py` reports one aggregate WER per (split, format). Aggregate WER cannot
tell a model that *mishears a rare word* from one that *drops into a subword
loop* or *truncates early* — different failures that call for opposite fixes
(DEGENERACY_TAXONOMY.md). Nor can it show whether the model actually *obeyed the
format instruction* rather than emitting the style the audio suggested
(FORMATTING_SPEC.md §5 style-flip). This tool slices the same per-utterance
JSONL that `run_wer.py` already writes, so every headline number in the report
can be backed by an error decomposition rather than an average.

It is deliberately **stdlib-only** — same discipline as `count_degeneracies.py`.
It must run on a laptop against a downloaded JSONL with no dependency install.

WHAT IT PRODUCES
----------------
For one checkpoint's eval (one or more per-utterance JSONL files):
  1. WER (un-normalized, house convention) with S/D/I decomposition, sliced by
       - instruction mode (unformatted / formatted),
       - reference length bucket (probes truncation),
       - numeral-bearing utterances (where formatted vs unformatted diverges),
       - rare/OOV-bearing utterances (the known loop trigger),
  2. Degeneracy category rates per slice (empty / truncation / loop /
     hallucination / style-flip), using the DEGENERACY_TAXONOMY detection rules,
  3. An instruction-adherence rate per mode (the headline instructability number),
  4. A curated set of *example candidates* per category (best format-switch,
     worst rare-word loop, a style-flip, a clean long utterance) — the raw
     material for the report's behavioural gallery.

Output is emitted both as a machine-readable JSON (for the report to ingest)
and as a paste-ready Markdown block.

SCHEMA WIRING (confirmed against run_wer.py)
--------------------------------------------
`run_wer.py::_write_transcriptions` writes ONE JSONL per (step, split), holding
BOTH instruction modes — one row per utterance per mode. Confirmed field names:

    id         : "key"          e.g. "3081-166546-0063"
    reference  : "reference"
    hypothesis : "hypothesis"
    mode       : "type"         "unformatted" | "formatted"
    split      : "split"        e.g. "dev-clean"

Because the mode rides on every row, the single `--jsonl` form is the one to use
against this repo's output; `--unformatted/--formatted` remains for hand-made
files that split the modes. The loader FAILS LOUDLY naming the keys it did see
if a required field is missing — no silent guessing.

WER IS UN-NORMALIZED, by house convention: `utils/evaluate.py::compute_wer`
applies no normalisation "so scores reflect whether the model actually follows
the formatting instruction". This tool scores raw whitespace tokens for exactly
that reason, and its aggregate reconciles with run_wer.py's wer.csv to 2dp.
Normalisation is used only for rarity/numeral LOOKUPS, never for error counts.

DEGENERACY RULES ARE IMPORTED, not reimplemented — see the note on the
count_degeneracies import below.

USAGE
-----
    # one file that carries the mode field per row:
    python tools/analyze_slices.py --jsonl results/<run>/dev-clean.jsonl

    # or one file per mode (joined on id so the gallery can show both):
    python tools/analyze_slices.py \
        --unformatted results/<run>/dev-clean.unformatted.jsonl \
        --formatted   results/<run>/dev-clean.formatted.jsonl \
        --split dev-clean

    # rare-word slice is far sharper with a training-vocab frequency table:
    python tools/analyze_slices.py --jsonl ... --train-vocab data/vocab_freq.json
    #   (word -> training count; build once from labels.jsonl. Absent -> the tool
    #    falls back to eval-set hapaxes as a weaker rarity proxy and says so.)

    # write artifacts instead of stdout:
    python tools/analyze_slices.py --jsonl ... --out-json out/slices.json \
                                              --out-md   out/slices.md \
                                              --examples out/examples.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


# --------------------------------------------------------------------------- #
# Schema wiring — edit if run_wer.py uses different field names.
# --------------------------------------------------------------------------- #
# CONFIRMED against tools/run_wer.py::_write_transcriptions (which forwards the
# rows utils/evaluate.py::_transcription_row builds). One JSONL per (step, split)
# carrying BOTH modes, one row per utterance per mode:
#
#   {"checkpoint": ..., "step": 18720, "key": "3081-166546-0063",
#    "split": "dev-clean", "type": "unformatted",
#    "reference": ..., "hypothesis": ..., "wer": 0.0}
#
# The real name is listed FIRST in every tuple so there is no ambiguity; the
# remaining alternates are kept only so a hand-made JSONL still loads. Note the
# mode field is "type" — NOT one of the names this tool originally guessed, so
# every row used to load as mode="unknown", silently zeroing the adherence rate
# and the format-switch gallery.
FIELDS = {
    "id": ("key", "id", "utt_id", "uttid"),
    "reference": ("reference", "ref", "target", "label"),
    "hypothesis": ("hypothesis", "hyp", "prediction", "output"),
    "mode": ("type", "format", "mode", "instruction", "style"),
    "split": ("split", "subset"),
}

# Reference length buckets, by whitespace word count. The eval loader filters
# references to <=41 tokens (CLAUDE.md Decision 010 / DATA_PROVENANCE), so the
# top bucket is open-ended but shallow.
LENGTH_BUCKETS = [(0, 8), (9, 16), (17, 25), (26, 10_000)]
_BUCKET_LABELS = [f"{lo}-{hi if hi < 10_000 else '+'}" for lo, hi in LENGTH_BUCKETS]

# Degeneracy detection is NOT reimplemented here. tools/count_degeneracies.py
# owns the rules and is the tool the rest of the pipeline already reports from;
# duplicating its thresholds guaranteed the two would drift apart on the same
# JSONL. We import its per-utterance `classify` and slice ITS verdicts, so the
# category counts in this report reconcile exactly with a count_degeneracies.py
# run over the same file.
#
# Its rules, for reference (authoritative values live in that module):
#   empty          hypothesis strips to ""
#   truncation     len(hyp)/len(ref) < TRUNCATION_RATIO (0.7)
#   hallucination  len(hyp)/len(ref) > HALLUCINATION_RATIO (1.3)
#   loop           an n-gram (n=1..4) repeats >= 3 times BACK TO BACK
#   style_flip     unformatted only: hypothesis is capitalised or punctuated
# Categories are NOT mutually exclusive there (a hypothesis can loop AND
# hallucinate), so the per-slice counts below can sum past the utterance count.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import count_degeneracies as _cd  # noqa: E402  (stdlib-only, same tools/ dir)

RARE_HAPAX_ONLY = True          # fallback rarity = occurs once across eval refs

# The rare-word set for this run, published once classified so the example
# gallery can name WHICH words tripped an utterance. Set in main(); empty here.
_RARE_CACHE: set[str] = set()

NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "million", "billion",
}
# Titular allowlist inverse map (FORMATTING_SPEC §5), for normalized comparison.
INVERSE_ALLOWLIST = {"mrs.": "missus", "mr.": "mister", "dr.": "doctor", "st.": "saint"}


# --------------------------------------------------------------------------- #
# Text handling
# --------------------------------------------------------------------------- #
_PUNCT_RE = re.compile(r"[^\w\s']|_")
_APOS_RE = re.compile(r"'")
_DIGIT_RE = re.compile(r"\d")


def normalize(text: str) -> str:
    """FORMATTING_SPEC §6 normalization (shared with validator / WER decomp).

    lowercase -> inverse allowlist -> delete apostrophes (keep token intact) ->
    strip remaining punctuation to spaces -> collapse whitespace. Digit-bearing
    tokens are NOT expanded (expansion is ambiguous); callers that need the
    digit-wildcard alignment must handle it separately.
    """
    text = text.lower()
    for k, v in INVERSE_ALLOWLIST.items():
        text = text.replace(k, v)
    text = _APOS_RE.sub("", text)          # delete apostrophes: boy's -> boys
    text = _PUNCT_RE.sub(" ", text)        # other punctuation -> space
    return " ".join(text.split())


def words(text: str) -> list[str]:
    return text.split()


# --------------------------------------------------------------------------- #
# WER with edit-operation decomposition (Levenshtein on word tokens)
# --------------------------------------------------------------------------- #
@dataclass
class EditCounts:
    sub: int = 0
    dele: int = 0   # deletions: reference words the hypothesis dropped
    ins: int = 0
    ref_len: int = 0

    @property
    def errors(self) -> int:
        return self.sub + self.dele + self.ins

    @property
    def wer(self) -> float:
        return self.errors / self.ref_len if self.ref_len else 0.0

    def __iadd__(self, other: "EditCounts") -> "EditCounts":
        self.sub += other.sub
        self.dele += other.dele
        self.ins += other.ins
        self.ref_len += other.ref_len
        return self


def edit_ops(ref: list[str], hyp: list[str]) -> EditCounts:
    """Standard Levenshtein alignment, returning S/D/I counts against `ref`.

    Deletion = a reference word with no hypothesis match (drives truncation
    signal). Insertion = an extra hypothesis word (drives loop/hallucination).
    """
    n, m = len(ref), len(hyp)
    # dp[i][j] = (cost, back-op) ; op in {"m","s","d","i"}
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bk = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        bk[i][0] = "d"
    for j in range(1, m + 1):
        dp[0][j] = j
        bk[0][j] = "i"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                bk[i][j] = "m"
                continue
            sub_c = dp[i - 1][j - 1] + 1
            del_c = dp[i - 1][j] + 1
            ins_c = dp[i][j - 1] + 1
            best = min(sub_c, del_c, ins_c)
            dp[i][j] = best
            bk[i][j] = "s" if best == sub_c else ("d" if best == del_c else "i")
    ec = EditCounts(ref_len=n)
    i, j = n, m
    while i > 0 or j > 0:
        op = bk[i][j]
        if op == "m":
            i, j = i - 1, j - 1
        elif op == "s":
            ec.sub += 1; i, j = i - 1, j - 1
        elif op == "d":
            ec.dele += 1; i -= 1
        else:
            ec.ins += 1; j -= 1
    return ec


# --------------------------------------------------------------------------- #
# Per-utterance record
# --------------------------------------------------------------------------- #
@dataclass
class Utt:
    uid: str
    mode: str                    # "unformatted" | "formatted" | "unknown"
    split: str
    ref_raw: str
    hyp_raw: str
    ref: list[str] = field(default_factory=list)   # RAW tokens (WER is un-normalized)
    hyp: list[str] = field(default_factory=list)
    ref_norm: list[str] = field(default_factory=list)  # normalized, for rarity/numeral lookups only
    ec: EditCounts = field(default_factory=EditCounts)

    # original row, kept so --group-by can slice on any field the JSONL carries
    raw: dict = field(default_factory=dict)

    # slice flags
    has_numeral: bool = False
    has_rare: bool = False
    length_bucket: str = ""

    # degeneracy flags
    empty: bool = False
    truncation: bool = False
    loop: bool = False
    hallucination: bool = False
    style_flip: bool = False
    adheres: bool = True


def _pick(row: dict, alts: tuple[str, ...]) -> Optional[str]:
    for a in alts:
        if a in row:
            return a
    return None


def load_rows(path: Path, forced_mode: Optional[str], forced_split: Optional[str]) -> list[Utt]:
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    if not rows:
        raise SystemExit(f"{path}: no JSON lines found")
    sample = rows[0]
    id_k = _pick(sample, FIELDS["id"])
    ref_k = _pick(sample, FIELDS["reference"])
    hyp_k = _pick(sample, FIELDS["hypothesis"])
    mode_k = _pick(sample, FIELDS["mode"])
    split_k = _pick(sample, FIELDS["split"])
    if ref_k is None or hyp_k is None:
        raise SystemExit(
            f"{path}: could not find reference/hypothesis fields. "
            f"Saw keys: {sorted(sample)}. Edit FIELDS at the top of this script."
        )
    if mode_k is None and forced_mode is None:
        raise SystemExit(
            f"{path}: no instruction-mode field found (tried {FIELDS['mode']}). "
            f"Saw keys: {sorted(sample)}. Either edit FIELDS, or pass the files "
            f"as --unformatted/--formatted so the mode is supplied explicitly. "
            f"Adherence and the format-switch gallery need it."
        )
    out: list[Utt] = []
    for idx, r in enumerate(rows):
        mode = forced_mode or (str(r.get(mode_k)).lower() if mode_k else "unknown")
        # normalize common encodings of the mode field
        if mode in ("unformatted", "raw", "verbatim", "0", "false"):
            mode = "unformatted"
        elif mode in ("formatted", "cased", "punctuated", "1", "true"):
            mode = "formatted"
        out.append(
            Utt(
                uid=str(r.get(id_k, f"{path.stem}:{idx}")),
                mode=mode,
                split=forced_split or (str(r.get(split_k)) if split_k else path.stem),
                ref_raw=str(r.get(ref_k, "")),
                hyp_raw=str(r.get(hyp_k, "")),
                raw=r,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Flagging
# --------------------------------------------------------------------------- #
def has_numeral(u: Utt) -> bool:
    if _DIGIT_RE.search(u.ref_raw) or _DIGIT_RE.search(u.hyp_raw):
        return True
    return any(w in NUMBER_WORDS for w in u.ref_norm)


def bucket_for(n: int) -> str:
    for lo, hi in LENGTH_BUCKETS:
        if lo <= n <= hi:
            return f"{lo}-{hi if hi < 10_000 else '+'}"
    return "?"


def looks_sentence_cased(raw: str) -> bool:
    s = raw.strip()
    return bool(s) and s[0].isupper()


def has_terminal_punct(raw: str) -> bool:
    return raw.strip().endswith((".", "?", "!"))


def classify(u: Utt, rare_words: set[str]) -> None:
    """Score one utterance: WER decomposition, slice flags, degeneracy verdicts.

    WER is computed on RAW whitespace tokens, deliberately UN-normalized: that
    is the house convention (utils/evaluate.py::compute_wer applies no
    normalisation, "so scores reflect whether the model actually follows the
    formatting instruction"), and it makes this tool's aggregate reconcile with
    run_wer.py's headline number instead of silently reporting a lower one.

    `normalize()` is still used, but only for LOOKUPS — matching reference words
    against the rarity table and the number-word list, where casing and trailing
    punctuation are noise. It never touches the error counts.
    """
    u.ref = words(u.ref_raw)
    u.hyp = words(u.hyp_raw)
    u.ec = edit_ops(u.ref, u.hyp)
    u.ref_norm = words(normalize(u.ref_raw))
    u.has_numeral = has_numeral(u)
    u.has_rare = bool(rare_words & set(u.ref_norm))
    u.length_bucket = bucket_for(len(u.ref))

    # Degeneracy verdicts come from count_degeneracies.py so the two tools can
    # never disagree on the same JSONL. Its categories overlap by design.
    hits = _cd.classify(u.ref_raw, u.hyp_raw, u.mode)
    u.empty = "empty" in hits
    u.truncation = "truncation" in hits
    u.loop = "loop" in hits
    u.hallucination = "hallucination" in hits
    u.style_flip = "style_flip" in hits

    # Instruction adherence. The shared style_flip rule is UNFORMATTED-only (the
    # unformatted labels carry no case or punctuation, so any marker is a flip).
    # The formatted direction has no counterpart there and is judged here: a
    # formatted hypothesis should be sentence-cased or terminally punctuated.
    # Reported separately from style_flip precisely so the shared count stays
    # shared.
    if u.mode == "unformatted":
        u.adheres = not u.style_flip
    elif u.mode == "formatted":
        u.adheres = looks_sentence_cased(u.hyp_raw) or has_terminal_punct(u.hyp_raw)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def rare_word_set(utts: list[Utt], vocab_freq: Optional[dict]) -> set[str]:
    if vocab_freq is not None:
        thr = 5
        return {w for w, c in vocab_freq.items() if c <= thr}
    # fallback: hapaxes across the eval references (weaker proxy). Count each
    # utterance ONCE (dedupe by id) so the unformatted/formatted duplication of
    # every reference word does not push genuine hapaxes to count 2.
    seen_ids: set[str] = set()
    cnt: Counter = Counter()
    for u in utts:
        if u.uid in seen_ids:
            continue
        seen_ids.add(u.uid)
        cnt.update(words(normalize(u.ref_raw)))
    return {w for w, c in cnt.items() if c == 1} if RARE_HAPAX_ONLY else set()


def agg(utts: Iterable[Utt]) -> dict:
    ec = EditCounts()
    n = 0
    deg = Counter()
    any_deg = 0
    adhere = 0
    adhere_n = 0
    for u in utts:
        ec += u.ec
        n += 1
        for k in ("empty", "truncation", "loop", "hallucination", "style_flip"):
            if getattr(u, k):
                deg[k] += 1
        # any-of, NOT a sum: the shared categories overlap, so summing them
        # double-counts a hypothesis that both loops and hallucinates.
        if u.empty or u.truncation or u.loop or u.hallucination:
            any_deg += 1
        if u.mode in ("unformatted", "formatted"):
            adhere_n += 1
            adhere += int(u.adheres)
    return {
        "n": n,
        "wer": round(100 * ec.wer, 2),
        "sub": ec.sub, "del": ec.dele, "ins": ec.ins, "ref_words": ec.ref_len,
        "del_frac_of_err": round(ec.dele / ec.errors, 3) if ec.errors else 0.0,
        "degeneracy": dict(deg),
        "degeneracy_frac": round(any_deg / n, 4) if n else 0.0,
        "adherence": round(100 * adhere / adhere_n, 2) if adhere_n else None,
    }


def sliced(utts: list[Utt], key, order: Optional[list[str]] = None) -> dict:
    """Group utterances by `key` and aggregate each group.

    `order` fixes the row order for slices whose labels do not sort sensibly as
    strings (the length buckets: "9-16" sorts after "26-+" lexicographically).
    Labels in `order` come first, in that order; anything unexpected follows,
    sorted, rather than being dropped.
    """
    groups: dict[str, list[Utt]] = {}
    for u in utts:
        groups.setdefault(key(u), []).append(u)
    if order:
        keys = [k for k in order if k in groups] + sorted(set(groups) - set(order))
    else:
        keys = sorted(groups)
    return {k: agg(groups[k]) for k in keys}


# --------------------------------------------------------------------------- #
# Example candidates for the behavioural gallery
# --------------------------------------------------------------------------- #
def example_candidates(utts: list[Utt]) -> dict:
    by_id: dict[str, dict[str, Utt]] = {}
    for u in utts:
        by_id.setdefault(u.uid, {})[u.mode] = u

    def pack(u: Utt) -> dict:
        return {
            "id": u.uid, "mode": u.mode, "split": u.split,
            "reference": u.ref_raw, "hypothesis": u.hyp_raw,
            "wer": round(100 * u.ec.wer, 1),
            "flags": [k for k in ("empty", "truncation", "loop", "hallucination", "style_flip") if getattr(u, k)],
            "has_rare": u.has_rare, "has_numeral": u.has_numeral,
            "rare_words": sorted(set(u.ref_norm) & _RARE_CACHE) if u.has_rare else [],
        }

    # rare-bearing loops first (the report's weakness example), then by severity
    loops = sorted((u for u in utts if u.loop),
                   key=lambda u: (not u.has_rare, -u.ec.wer))[:5]
    flips = [u for u in utts if u.style_flip][:5]
    truncs = sorted((u for u in utts if u.truncation), key=lambda u: -u.ec.dele)[:5]

    # The headline instructability example: the SAME audio rendered as a number
    # word under "unformatted" and as a DIGIT under "formatted". Requiring a
    # digit in the formatted reference (and none in the unformatted one) is what
    # makes it a real switch — merely containing the word "two" is not, since
    # both references then read "two" and nothing has switched.
    switches, missed = [], []
    for uid, d in by_id.items():
        if "unformatted" not in d or "formatted" not in d:
            continue
        a, b = d["unformatted"], d["formatted"]
        wants_digit = bool(_DIGIT_RE.search(b.ref_raw)) and not _DIGIT_RE.search(a.ref_raw)
        if not wants_digit:
            continue
        got_digit = bool(_DIGIT_RE.search(b.hyp_raw))
        entry = {
            "id": uid,
            "combined_wer": round(100 * (a.ec.wer + b.ec.wer) / 2, 1),
            "unformatted": pack(a), "formatted": pack(b),
        }
        if got_digit and a.adheres and b.adheres and not _DIGIT_RE.search(a.hyp_raw):
            switches.append(entry)      # clean: word under one mode, digit under the other
        else:
            missed.append(entry)        # the instruction did not move the rendering
    switches.sort(key=lambda e: e["combined_wer"])   # cleanest pair first
    missed.sort(key=lambda e: e["combined_wer"])

    return {
        "format_switch_numeral": switches[:5],
        "format_switch_numeral_missed": missed[:5],
        "rare_word_loops": [pack(u) for u in loops],
        "style_flips": [pack(u) for u in flips],
        "truncations": [pack(u) for u in truncs],
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def md_table(title: str, rows: dict, cols: list[str]) -> str:
    lines = [f"#### {title}", "", "| slice | " + " | ".join(cols) + " |",
             "|---|" + "|".join(["---"] * len(cols)) + "|"]
    for slice_name, a in rows.items():
        cells = []
        for c in cols:
            v = a.get(c)
            cells.append("—" if v is None else str(v))
        lines.append(f"| {slice_name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_md(report: dict) -> str:
    cols = ["n", "wer", "del_frac_of_err", "adherence", "degeneracy_frac"]
    out = ["# Fine-grained eval breakdown", "",
           f"_source: {report['meta']['source']}_  ",
           f"_rarity: {report['meta']['rarity_source']}_  ",
           f"_WER is un-normalized (house convention); del_frac_of_err flags truncation-heavy slices._",
           ""]
    out.append(md_table("By instruction mode", report["by_mode"], cols))
    out.append(md_table("By reference length", report["by_length"], cols))
    out.append(md_table("By numeral presence", report["by_numeral"], cols))
    out.append(md_table("By rare/OOV word presence", report["by_rare"], cols))
    known = {"by_mode", "by_length", "by_numeral", "by_rare"}
    for key in report:
        if key.startswith("by_") and key not in known:
            out.append(md_table(f"By {key[3:]}", report[key], cols))
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl", type=Path, help="single per-utterance JSONL (mode read from a field)")
    p.add_argument("--unformatted", type=Path, help="per-utterance JSONL, unformatted mode")
    p.add_argument("--formatted", type=Path, help="per-utterance JSONL, formatted mode")
    p.add_argument("--split", type=str, default=None, help="override split label")
    p.add_argument("--train-vocab", type=Path, default=None, help="word->count JSON for rarity")
    p.add_argument("--group-by", type=str, default=None,
                   help="extra slice on any field the JSONL rows carry (e.g. turn_index, context_mode)")
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--out-md", type=Path, default=None)
    p.add_argument("--examples", type=Path, default=None)
    args = p.parse_args(argv)

    utts: list[Utt] = []
    sources: list[str] = []
    if args.jsonl:
        utts += load_rows(args.jsonl, None, args.split)
        sources.append(str(args.jsonl))
    if args.unformatted:
        utts += load_rows(args.unformatted, "unformatted", args.split)
        sources.append(str(args.unformatted))
    if args.formatted:
        utts += load_rows(args.formatted, "formatted", args.split)
        sources.append(str(args.formatted))
    if not utts:
        p.error("provide --jsonl, or --unformatted/--formatted")

    vocab_freq = json.loads(args.train_vocab.read_text()) if args.train_vocab else None
    rare = rare_word_set(utts, vocab_freq)
    global _RARE_CACHE
    _RARE_CACHE = rare
    for u in utts:
        classify(u, rare)

    report = {
        "meta": {
            "source": ", ".join(sources),
            "n_utts": len(utts),
            "rarity_source": "train-vocab (count<=5)" if vocab_freq else "eval-set hapaxes (fallback)",
            "length_buckets": _BUCKET_LABELS,
        },
        "overall": agg(utts),
        "by_mode": sliced(utts, lambda u: u.mode),
        "by_length": sliced(utts, lambda u: u.length_bucket, order=_BUCKET_LABELS),
        "by_numeral": sliced(utts, lambda u: "numeral" if u.has_numeral else "no-numeral"),
        "by_rare": sliced(utts, lambda u: "rare/OOV" if u.has_rare else "common"),
    }
    if args.group_by:
        gb = args.group_by
        report[f"by_{gb}"] = sliced(utts, lambda u: str(u.raw.get(gb, "?")))

    md = render_md(report)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2))
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md)
    if args.examples:
        args.examples.parent.mkdir(parents=True, exist_ok=True)
        args.examples.write_text(json.dumps(example_candidates(utts), indent=2))
    if not (args.out_json or args.out_md):
        print(md)
        print("\n```json\n" + json.dumps(report["overall"], indent=2) + "\n```")
    return 0


if __name__ == "__main__":
    sys.exit(main())

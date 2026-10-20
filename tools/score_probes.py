"""Instruction-generalisation probe suite — scoring half.

Reads the JSONL files tools/run_probes.py writes and emits ONE Markdown report:
a summary table (probe | n | pass rate | headline sub-metric | two verbatim
example failures) plus a per-probe detail section.

Pure standard library — no torch, no jiwer — like tools/count_degeneracies.py,
so scoring a pod run needs nothing but the files.

WHAT IT REUSES, AND WHY
-----------------------
WER, text normalisation and the adherence rule are IMPORTED from
tools/analyze_slices.py, and the degeneracy verdicts from
tools/count_degeneracies.py, rather than reimplemented.  P5's canonical variant
is the trained instruction over dev-clean utterances — the same measurement the
banked slice reports hold — so it must be computed by the same code or the
comparison is worthless.  WER here is therefore UN-normalized raw-token WER,
the house convention (utils/evaluate.py applies no normalisation "so scores
reflect whether the model actually follows the formatting instruction").

THE TWO NUMBERS PER PROBE
-------------------------
    Following Rate  fraction of outputs that are even the right KIND of
                    response — a transcript where a transcript was asked for, a
                    single word where a single word was asked for, a count line
                    where a count line was asked for.  This is the summary
                    number for the report: it asks whether the model understood
                    the SHAPE of the request at all.
    Pass rate       fraction that additionally got it RIGHT — the constraint
                    actually satisfied, the answer actually correct.

They come apart in the informative direction: a model that answers "the last
word is today" has a Following Rate of 0 on P3 (it did not answer with one
word) even when the word it names is correct, and one that emits a fluent
transcript under every P5 paraphrase has a Following Rate of 1 whatever its WER.

USAGE
-----
    python tools/score_probes.py --dir results/probes/ \\
        --out-md results/probes/probe-report.md \\
        --out-json results/probes/probe-report.json

    python tools/score_probes.py --self-test
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import count_degeneracies as _cd      # noqa: E402  (stdlib-only, same tools/ dir)
import probe_spec                     # noqa: E402
from analyze_slices import (          # noqa: E402
    EditCounts, edit_ops, has_terminal_punct, looks_sentence_cased, normalize,
)
from probe_spec import P4_QUESTIONS, P5_VARIANTS, PROBES, PROBE_IDS, probe_filename


# A response counts as "the right kind" of transcript when it is non-empty and
# within this factor of the reference length either way.  Deliberately loose:
# the question is whether the model produced a transcript AT ALL, not whether it
# is a good one — the pass rate and the WER answer that.
TRANSCRIPT_MIN_RATIO: float = 0.5
TRANSCRIPT_MAX_RATIO: float = 2.0

# P1's constraint: at least this fraction of alphabetic characters upper case.
UPPERCASE_PASS_FRACTION: float = 0.95

# P3/P4 answer shape: an answer of at most this many words is "one word"-ish.
# Not 1: a model that answers "the word today" has still answered in the right
# SHAPE and failed the constraint, which is a different finding from dumping the
# entire transcript.  Anything longer is not an answer, it is a transcript.
ANSWER_MAX_WORDS: int = 3

# ...and at most this many characters.  The word guard alone is not enough: a
# degenerate run-on with no spaces ("liftliftlift...") is ONE whitespace token
# and would otherwise be scored as a well-shaped one-word answer.  Caught on the
# stub run, where exactly that came out of a random-init model.
ANSWER_MAX_CHARS: int = 40

# P3: a hypothesis holding at least this fraction of the full transcript's words
# is a transcript dump — the ASR-collapse signature this probe exists to catch.
DUMP_MIN_RATIO: float = 0.7

# A control condition is only a control if it produced usable output.  Above
# this fraction of degenerate (looping) responses, P4's text-only control is
# reported as UNUSABLE and no leakage gap is computed from it — a gap between a
# broken control and a real condition is not a measurement.
CONTROL_MAX_DEGENERATE: float = 0.5

# Examples shown per probe in the summary table.
N_EXAMPLES: int = 2


@dataclass
class ProbeReport:
    """One probe's scored result."""

    probe_id:       str
    name:           str
    n:              int
    following_rate: float           # 0..1
    pass_rate:      float           # 0..1
    headline:       str             # the sub-metric, formatted for the table
    failures:       list[dict] = field(default_factory=list)
    detail:         dict = field(default_factory=dict)
    note:           str = ""

    def to_dict(self) -> dict:
        return {
            "probe": self.probe_id, "name": self.name, "n": self.n,
            "following_rate": round(100 * self.following_rate, 2),
            "pass_rate":      round(100 * self.pass_rate, 2),
            "headline": self.headline, "note": self.note,
            "failures": self.failures, "detail": self.detail,
        }


# ── Shared helpers ────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> list[dict]:
    """Read a probe JSONL, sorted by key so example selection is deterministic."""
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    return sorted(rows, key=lambda r: (r.get("key", ""), r.get("variant_id", "")))


def _looks_like_transcript(reference: str, hypothesis: str) -> bool:
    """Right KIND of response for a transcription probe: non-empty, right size."""
    hyp = hypothesis.split()
    ref = reference.split()
    if not hyp:
        return False
    if not ref:
        return True
    ratio = len(hyp) / len(ref)
    return TRANSCRIPT_MIN_RATIO <= ratio <= TRANSCRIPT_MAX_RATIO


def is_answer_shaped(hypothesis: str) -> bool:
    """Right KIND of response for a probe that asked for a one-word answer.

    Short in WORDS and short in CHARACTERS — see ANSWER_MAX_CHARS.  Says nothing
    about whether the answer is right; that is the pass rate's job.
    """
    stripped = hypothesis.strip()
    n_words  = len(normalize(hypothesis).split())
    return 0 < n_words <= ANSWER_MAX_WORDS and len(stripped) <= ANSWER_MAX_CHARS


def _fail(row: dict, why: str, extra: dict | None = None) -> dict:
    """One example-failure record, verbatim — never trimmed or tidied."""
    out = {
        "key":        row.get("key", ""),
        "why":        why,
        "reference":  row.get("reference", ""),
        "hypothesis": row.get("hypothesis", ""),
    }
    if extra:
        out.update(extra)
    return out


def _rate(n_ok: int, n: int) -> float:
    return n_ok / n if n else 0.0


def _wer_of(rows: list[dict]) -> float:
    """Corpus WER over rows, raw-token and un-normalized (the house convention)."""
    ec = EditCounts()
    for r in rows:
        ec += edit_ops(r.get("reference", "").split(), r.get("hypothesis", "").split())
    return 100 * ec.wer


# ── P1 · uppercase ────────────────────────────────────────────────────────────

def uppercase_fraction(text: str) -> float:
    """Fraction of ALPHABETIC characters that are upper case; 0.0 if none."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(c.isupper() for c in alpha) / len(alpha)


def score_p1(rows: list[dict]) -> ProbeReport:
    """P1 — the case constraint, with perception held constant.

    WER is case-INSENSITIVE here on purpose: it answers "did asking for capitals
    damage the transcription?", which is a different question from "did it obey",
    and the constraint itself is measured by the upper-case character fraction.
    """
    n = len(rows)
    following = sum(_looks_like_transcript(r["reference"], r["hypothesis"]) for r in rows)
    fracs = [uppercase_fraction(r["hypothesis"]) for r in rows]
    passed = sum(f >= UPPERCASE_PASS_FRACTION for f in fracs)

    ci = EditCounts()
    for r in rows:
        ci += edit_ops(r["reference"].upper().split(), r["hypothesis"].upper().split())

    failures = [
        _fail(r, f"{100 * f:.0f}% of alphabetic characters upper case "
                 f"(needs {100 * UPPERCASE_PASS_FRACTION:.0f}%)",
              {"uppercase_fraction": round(f, 4)})
        for r, f in zip(rows, fracs) if f < UPPERCASE_PASS_FRACTION
    ]
    mean_frac = sum(fracs) / n if n else 0.0
    return ProbeReport(
        "P1", "uppercase", n, _rate(following, n), _rate(passed, n),
        headline=f"{100 * mean_frac:.1f}% mean upper-case chars; "
                 f"{100 * ci.wer:.2f}% case-insensitive WER",
        failures=failures[:N_EXAMPLES],
        detail={"mean_uppercase_fraction": round(100 * mean_frac, 2),
                "case_insensitive_wer": round(100 * ci.wer, 2)},
    )


# ── P2 · word count ───────────────────────────────────────────────────────────

_INT_RE = re.compile(r"^\s*(\d+)\s*$")


def parse_count_line(hypothesis: str) -> tuple[str, int | None]:
    """Split a P2 hypothesis into (transcript, count).

    The instruction says the count goes on a NEW LINE, so the last line is the
    candidate and everything before it is the transcript.  A hypothesis with one
    line, or whose last line is not a bare integer, did not follow the format —
    count is None and the transcript is the whole thing.

    Returns:
        (transcript text, parsed count or None)
    """
    lines = [ln for ln in hypothesis.splitlines()]
    if len(lines) < 2:
        return hypothesis.strip(), None
    m = _INT_RE.match(lines[-1])
    if not m:
        return hypothesis.strip(), None
    return "\n".join(lines[:-1]).strip(), int(m.group(1))


def score_p2(rows: list[dict]) -> ProbeReport:
    """P2 — a second structured field, consistent with what the model itself wrote.

    The pass criterion is SELF-consistency, because that is the instruction as
    written: "the number of words in your transcript".  Agreement with the
    reference's own word count is reported separately — it folds in transcription
    errors and answers a different question.
    """
    n = len(rows)
    following = 0
    self_ok   = 0
    ref_ok    = 0
    failures: list[dict] = []

    for r in rows:
        transcript, count = parse_count_line(r["hypothesis"])
        if count is None:
            failures.append(_fail(
                r, "no parseable count on a final line",
                {"parsed_transcript": transcript, "parsed_count": None}))
            continue
        following += 1
        own = len(transcript.split())
        ref = len(r["reference"].split())
        self_ok += (count == own)
        ref_ok  += (count == ref)
        if count != own:
            failures.append(_fail(
                r, f"count {count} != {own} words in its own transcript",
                {"parsed_transcript": transcript, "parsed_count": count,
                 "own_words": own, "reference_words": ref}))

    return ProbeReport(
        "P2", "word_count", n, _rate(following, n), _rate(self_ok, n),
        headline=f"{100 * _rate(ref_ok, n):.1f}% also match the REFERENCE word count",
        failures=failures[:N_EXAMPLES],
        detail={"self_consistent_pct":     round(100 * _rate(self_ok, n), 2),
                "matches_reference_pct":   round(100 * _rate(ref_ok, n), 2),
                "emitted_count_line_pct":  round(100 * _rate(following, n), 2)},
    )


# ── P3 · last word ────────────────────────────────────────────────────────────

def score_p3(rows: list[dict]) -> ProbeReport:
    """P3 — answer ABOUT the audio rather than transcribing it.

    Two rates that must be read together: exact match to the reference's last
    word (the pass), and how often the model dumped the whole transcript instead
    of answering (the ASR-collapse signature this probe exists to catch).

    Following Rate here is "short AND not a transcript dump".  A short
    utterance's entire transcript can fit inside ANSWER_MAX_WORDS, so shortness
    alone would score a collapse as compliance.
    """
    n = len(rows)
    following = 0
    correct   = 0
    dumps     = 0
    single    = 0
    failures: list[dict] = []

    for r in rows:
        hyp_words = normalize(r["hypothesis"]).split()
        expected  = normalize(r["reference"]).strip()
        full_ref  = r.get("full_reference", "")
        full_words = normalize(full_ref).split()

        is_single = len(hyp_words) == 1
        single   += is_single

        is_dump = bool(full_words) and len(hyp_words) >= DUMP_MIN_RATIO * len(full_words)
        dumps  += is_dump

        # Answer-SHAPED means short AND not simply the transcript.  Both halves
        # are needed: a short utterance's whole transcript can be under
        # ANSWER_MAX_WORDS, and counting that as "the right kind of response"
        # would hide the exact collapse this probe exists to catch.
        following += is_answer_shaped(r["hypothesis"]) and not is_dump

        hit = bool(hyp_words) and " ".join(hyp_words) == expected
        correct += hit
        if not hit:
            failures.append(_fail(
                r, "transcript dump instead of an answer" if is_dump
                   else f"answered {' '.join(hyp_words[:6])!r}, expected {expected!r}",
                {"full_reference": full_ref, "is_dump": is_dump,
                 "hyp_word_count": len(hyp_words)}))

    return ProbeReport(
        "P3", "last_word", n, _rate(following, n), _rate(correct, n),
        headline=f"{100 * _rate(single, n):.1f}% answered with a single word; "
                 f"{100 * _rate(dumps, n):.1f}% dumped the transcript",
        failures=failures[:N_EXAMPLES],
        detail={"single_word_pct":     round(100 * _rate(single, n), 2),
                "transcript_dump_pct": round(100 * _rate(dumps, n), 2),
                "exact_match_pct":     round(100 * _rate(correct, n), 2)},
    )


# ── P4 · ignore the audio ─────────────────────────────────────────────────────

def _p4_verdict(hypothesis: str, q) -> tuple[bool, bool, bool]:
    """Score one P4 answer.

    Returns:
        (right_kind, correct, right_answer_wrong_form)

    right_kind:  answered in at most ANSWER_MAX_WORDS — did not ramble or
                 transcribe
    correct:     matches the expected answer or one of its aliases,
                 case-insensitively and after §6 normalisation
    wrong_form:  the answer is RIGHT but in the form the instruction did not ask
                 for ("five" where "5" was demanded) — a different finding from
                 not knowing it, so it is never silently counted as correct
    """
    hyp_words  = normalize(hypothesis).split()
    right_kind = is_answer_shaped(hypothesis)
    hyp_norm   = " ".join(hyp_words)

    accepted = {normalize(a) for a in (q.answer, *q.aliases)}
    wrong    = {normalize(a) for a in q.wrong_form}

    # A one-word answer may arrive wrapped ("the answer is Paris"); accept it
    # when the expected token is present AND the response is answer-shaped, so
    # a full transcript containing the word by chance cannot score.
    correct    = right_kind and (hyp_norm in accepted
                                 or any(a in hyp_words for a in accepted))
    wrong_form = right_kind and not correct and (
        hyp_norm in wrong or any(w in hyp_words for w in wrong))
    return right_kind, correct, wrong_form


def score_p4(rows: list[dict]) -> ProbeReport:
    """P4 — disregard the audio, plus the text-only leakage control.

    The two conditions are scored separately and the GAP between them is the
    headline.  The backbone is frozen, so the text-only rate is what this LLM
    can do with the question and nothing else; anything the with-audio condition
    loses against it is what the audio conditioning costs.
    """
    by_qid = {q.qid: q for q in P4_QUESTIONS}
    audio  = [r for r in rows if r.get("context") == "audio"]
    text   = [r for r in rows if r.get("context") == "text_only"]

    def score(subset: list[dict]) -> dict:
        n = len(subset)
        kind = corr = wf = transcribed = degenerate = 0
        fails: list[dict] = []
        per_q: dict[str, dict] = {}
        for r in subset:
            q = by_qid.get(r.get("qid", ""))
            if q is None:
                continue
            k, c, w = _p4_verdict(r["hypothesis"], q)
            kind += k; corr += c; wf += w

            # Did it transcribe the audio instead of answering?  This is the
            # dominant failure mode and it is invisible without the utterance's
            # own transcript to compare against, which run_probes.py attaches.
            full = normalize(r.get("full_reference", ""))
            hyp  = normalize(r["hypothesis"])
            is_transcription = bool(full) and hyp == full
            transcribed += is_transcription

            # A looping response is degenerate, not an answer — tracked so a
            # control made of word salad cannot masquerade as a 0% score.
            degenerate += "loop" in _cd.classify(
                r.get("full_reference", "") or r.get("reference", ""),
                r["hypothesis"], "ignore_audio")

            slot = per_q.setdefault(r["qid"], {"n": 0, "correct": 0, "wrong_form": 0})
            slot["n"] += 1; slot["correct"] += c; slot["wrong_form"] += w
            if not c:
                fails.append(_fail(
                    r, "right answer, wrong form (words where digits were asked for)"
                       if w else
                       "transcribed the audio instead of answering" if is_transcription
                       else "did not answer the question",
                    {"question": r.get("question", ""), "expected": q.answer}))
        return {"n": n, "following": _rate(kind, n), "correct": _rate(corr, n),
                "wrong_form": _rate(wf, n),
                "transcribed_instead": _rate(transcribed, n),
                "degenerate": _rate(degenerate, n),
                "failures": fails, "per_question": per_q}

    a = score(audio)
    t = score(text)

    # A gap is only meaningful against a control that produced usable output.
    control_usable = bool(t["n"]) and t["degenerate"] <= CONTROL_MAX_DEGENERATE
    gap = (a["correct"] - t["correct"]) if control_usable else None

    if not t["n"]:
        headline = (f"with-audio {100 * a['correct']:.1f}% correct; "
                    "no text-only control in these rows")
        note = ""
    elif not control_usable:
        headline = (f"with-audio {100 * a['correct']:.1f}% correct; "
                    f"{100 * a['transcribed_instead']:.1f}% transcribed the audio "
                    "instead. **Text-only control UNUSABLE** "
                    f"({100 * t['degenerate']:.0f}% degenerate) — no leakage gap")
        note = ("⚠ The text-only control did not produce usable output: "
                f"{100 * t['degenerate']:.0f}% of its responses are degenerate "
                "loops rather than answers. The frozen backbone, given this "
                "scaffold with no audio in it, does not answer coherently, so "
                "the with-audio versus text-only comparison CANNOT be read as a "
                "leakage measurement and no gap is reported. What the "
                "with-audio condition still shows on its own is unaffected.")
    else:
        headline = (f"text-only {100 * t['correct']:.1f}% vs with-audio "
                    f"{100 * a['correct']:.1f}% → leakage gap {100 * gap:+.1f} pts")
        note = ("The text-only control is 1 generation per question (10 rows); "
                "its rate is a coarse estimate and a single question moves it "
                "10 points.")

    return ProbeReport(
        "P4", "ignore_audio", a["n"], a["following"], a["correct"],
        headline=headline,
        failures=a["failures"][:N_EXAMPLES],
        detail={
            "with_audio":  {k: v for k, v in a.items() if k != "failures"},
            "text_only":   {k: v for k, v in t.items() if k != "failures"},
            "control_usable":  control_usable,
            "leakage_gap_pts": None if gap is None else round(100 * gap, 2),
            "text_only_failures": t["failures"][:N_EXAMPLES],
        },
        note=note,
    )


# ── P5 · prompt variation ─────────────────────────────────────────────────────

def _adheres(row: dict) -> bool:
    """analyze_slices.py's adherence rule, imported wholesale.

    unformatted: no case/punctuation markers (count_degeneracies' style_flip)
    formatted:   sentence-cased or terminally punctuated
    """
    mode = row.get("mode") or row.get("type", "")
    hyp  = row.get("hypothesis", "")
    if mode == "unformatted":
        return "style_flip" not in _cd.classify(row.get("reference", ""), hyp, mode)
    if mode == "formatted":
        return looks_sentence_cased(hyp) or has_terminal_punct(hyp)
    return True


def score_p5(rows: list[dict]) -> ProbeReport:
    """P5 — is the trained behaviour attached to the meaning or the surface string?

    Every variant is measured with the SAME code as the banked slice reports
    (imported WER and adherence), and the canonical string is one of the
    variants, so the reported spread is prompt sensitivity alone.
    """
    n = len(rows)
    by_variant: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by_variant.setdefault((r.get("mode", "?"), r.get("variant_id", "?")), []).append(r)

    per_variant: list[dict] = []
    for v in P5_VARIANTS:
        subset = by_variant.get((v.mode, v.vid), [])
        if not subset:
            continue
        per_variant.append({
            "mode": v.mode, "variant_id": v.vid, "is_canonical": v.is_canonical,
            "instruction": v.text, "n": len(subset),
            "wer": round(_wer_of(subset), 2),
            "adherence_pct": round(100 * _rate(sum(_adheres(r) for r in subset),
                                               len(subset)), 2),
            "following_pct": round(100 * _rate(
                sum(_looks_like_transcript(r["reference"], r["hypothesis"])
                    for r in subset), len(subset)), 2),
        })

    # Spread is measured per MODE: the two modes have different WERs by design
    # (the formatted task is harder), so pooling them would report that
    # difference as prompt sensitivity.
    spreads: dict[str, dict] = {}
    for mode in ("unformatted", "formatted"):
        vs = [v for v in per_variant if v["mode"] == mode]
        if not vs:
            continue
        canon = next((v for v in vs if v["is_canonical"]), None)
        wers  = [v["wer"] for v in vs]
        adhs  = [v["adherence_pct"] for v in vs]
        spreads[mode] = {
            "canonical_wer": canon["wer"] if canon else None,
            "wer_min": min(wers), "wer_max": max(wers),
            "wer_spread": round(max(wers) - min(wers), 2),
            "worst_variant": max(vs, key=lambda v: v["wer"])["variant_id"],
            "adherence_min": min(adhs), "adherence_max": max(adhs),
            "adherence_spread": round(max(adhs) - min(adhs), 2),
        }

    following = sum(_looks_like_transcript(r["reference"], r["hypothesis"]) for r in rows)
    passed    = sum(_adheres(r) for r in rows)

    parts = []
    for mode, s in spreads.items():
        canon = f"{s['canonical_wer']:.2f}" if s["canonical_wer"] is not None else "n/a"
        parts.append(f"{mode}: WER {s['wer_min']:.2f}–{s['wer_max']:.2f} "
                     f"(canonical {canon}, spread {s['wer_spread']:.2f})")

    # The worst individual paraphrase, so the table shows a concrete failure.
    failures: list[dict] = []
    worst_ids = {s["worst_variant"] for s in spreads.values()}
    for r in rows:
        if r.get("variant_id") in worst_ids and not _adheres(r):
            failures.append(_fail(
                r, f"variant {r.get('variant_id')} ({r.get('mode')}): "
                   "instruction not adhered to",
                {"instruction": r.get("instruction", "")}))

    return ProbeReport(
        "P5", "prompt_variation", n, _rate(following, n), _rate(passed, n),
        headline="; ".join(parts) if parts else "no variants scored",
        failures=failures[:N_EXAMPLES],
        detail={"per_variant": per_variant, "spread_by_mode": spreads},
    )


SCORERS = {"P1": score_p1, "P2": score_p2, "P3": score_p3,
           "P4": score_p4, "P5": score_p5}


# ── Report ────────────────────────────────────────────────────────────────────

def _truncate(text: str, n: int = 110) -> str:
    """One-line, escaped, length-capped — a Markdown table cell must survive it."""
    flat = text.replace("\n", "\\n").replace("|", "\\|")
    return flat if len(flat) <= n else flat[: n - 1] + "…"


def render_md(reports: list[ProbeReport], meta: dict) -> str:
    """The single Markdown report."""
    out: list[str] = [
        "# Instruction-generalisation probe suite",
        "",
        f"Checkpoint: `{meta.get('checkpoint', '<unknown>')}`"
        + (f" (step {meta['step']})" if meta.get("step") is not None else ""),
        f"Utterances: {meta.get('n_selected', '?')} from "
        f"`{meta.get('shard', '?')}` ({meta.get('split', '?')}), "
        f"seed {meta.get('seed', '?')}.",
        "",
        "**Every probe below is zero-shot.** Training only ever saw the two "
        "instruction strings in `data.INSTRUCTION_VARIANTS`, in one fixed "
        "single-turn layout. None of these instructions, output formats or "
        "question texts appears anywhere in the training data.",
        "",
        "**Following Rate** is the summary number: the fraction of outputs that "
        "are even the right *kind* of response. **Pass rate** is the fraction "
        "that additionally got it right. They are reported per probe and are "
        "not comparable across probes — the probes ask for different things, "
        "and a suite-level mean would hide exactly the variation that matters.",
        "",
    ]

    # ── Feasibility gate ─────────────────────────────────────────────────────
    gate = meta.get("gate", {})
    dropped = meta.get("probes_dropped", [])
    if gate or dropped:
        out += ["## Feasibility gate", "",
                "The vocabulary is pruned to the LibriSpeech label ids "
                "(Decision 005). A probe whose expected output cannot be "
                "tokenized inside it is impossible **by construction**, and "
                "reporting it as a model failure would be wrong. "
                "`tools/check_vocab_feasibility.py` decides this before any "
                "decoding.", ""]
        if gate:
            out += ["| probe | status | reason |", "|---|---|---|"]
            for pid, v in gate.items():
                out.append(f"| {pid} | {v.get('status', '?')} | {v.get('reason', '')} |")
            out.append("")
        if dropped:
            out += ["**Dropped, and not scored below:**", ""]
            for d in dropped:
                out.append(f"- **{d['probe']}** — {d.get('status')}: {d.get('reason')}"
                           + ("  ⚠ RUN ANYWAY via `--force`" if d.get("forced") else ""))
            out.append("")

    # ── Summary table ────────────────────────────────────────────────────────
    out += ["## Summary", "",
            "| probe | n | Following Rate | pass rate | headline sub-metric | example failures |",
            "|---|---:|---:|---:|---|---|"]
    for rep in reports:
        ex = "<br>".join(
            f"`{_truncate(f['hypothesis'])}` — {f['why']}" for f in rep.failures
        ) or "—"
        out.append(
            f"| **{rep.probe_id}** {rep.name} | {rep.n:,} | "
            f"{100 * rep.following_rate:.1f}% | {100 * rep.pass_rate:.1f}% | "
            f"{rep.headline} | {ex} |"
        )
    out.append("")

    # ── Per-probe detail ─────────────────────────────────────────────────────
    for rep in reports:
        out += [f"## {rep.probe_id} · {rep.name}", "",
                f"*{PROBES[rep.probe_id].measures}*", ""]
        if rep.note:
            out += [f"> {rep.note}", ""]

        if rep.probe_id == "P4":
            d = rep.detail
            out += ["| condition | n | Following Rate | correct | right answer, "
                    "wrong form | transcribed instead | degenerate |",
                    "|---|---:|---:|---:|---:|---:|---:|"]
            for label, key in (("with audio", "with_audio"), ("text-only control", "text_only")):
                c = d.get(key, {})
                if not c.get("n"):
                    continue
                out.append(
                    f"| {label} | {c['n']} | {100 * c['following']:.1f}% | "
                    f"{100 * c['correct']:.1f}% | {100 * c['wrong_form']:.1f}% | "
                    f"{100 * c.get('transcribed_instead', 0):.1f}% | "
                    f"{100 * c.get('degenerate', 0):.1f}% |")
            if d.get("leakage_gap_pts") is not None:
                out += ["", f"**Leakage gap: {d['leakage_gap_pts']:+.1f} points.** "
                            "The backbone is frozen, so the text-only rate is what "
                            "this LLM does with the question alone; the drop under "
                            "audio is what the audio conditioning costs.", ""]
            elif d.get("text_only", {}).get("n") and not d.get("control_usable"):
                out += ["", "**No leakage gap is reported.** The control is not "
                            "usable (see the note above); a gap between a broken "
                            "control and a real condition would not be a "
                            "measurement.", ""]
            per_q = d.get("with_audio", {}).get("per_question", {})
            if per_q:
                out += ["", "| question | n | correct | wrong form |", "|---|---:|---:|---:|"]
                for q in P4_QUESTIONS:
                    s = per_q.get(q.qid)
                    if not s:
                        continue
                    out.append(f"| {q.text} → `{q.answer}` | {s['n']} | "
                               f"{s['correct']} | {s['wrong_form']} |")
            for f in d.get("text_only_failures", []):
                out.append(f"\n- text-only failure — {f['question']}: "
                           f"`{_truncate(f['hypothesis'])}` ({f['why']})")
            out.append("")

        elif rep.probe_id == "P5":
            out += ["| mode | variant | canonical | n | WER | adherence | Following Rate |",
                    "|---|---|---|---:|---:|---:|---:|"]
            for v in rep.detail.get("per_variant", []):
                out.append(
                    f"| {v['mode']} | `{v['variant_id']}` | "
                    f"{'**yes**' if v['is_canonical'] else ''} | {v['n']} | "
                    f"{v['wer']:.2f} | {v['adherence_pct']:.1f}% | "
                    f"{v['following_pct']:.1f}% |")
            out.append("")
            for mode, s in rep.detail.get("spread_by_mode", {}).items():
                out.append(
                    f"- **{mode}** — WER spread {s['wer_spread']:.2f} points across "
                    f"wordings (worst: `{s['worst_variant']}`), adherence spread "
                    f"{s['adherence_spread']:.1f} points.")
            out += ["", "Instructions, verbatim:", ""]
            for v in rep.detail.get("per_variant", []):
                out.append(f"- `{v['variant_id']}` ({v['mode']}): {v['instruction']}")
            out.append("")

        else:
            for k, val in rep.detail.items():
                out.append(f"- {k}: {val}")
            out.append("")

        if rep.failures:
            out += ["Example failures, verbatim:", ""]
            for f in rep.failures:
                out += [f"- **{f['key']}** — {f['why']}",
                        f"  - reference:  `{_truncate(f['reference'])}`",
                        f"  - hypothesis: `{_truncate(f['hypothesis'])}`"]
            out.append("")

    return "\n".join(out) + "\n"


def _self_test() -> None:
    """Every scorer against hand-built rows with known answers."""

    def row(**kw) -> dict:
        base = {"key": "k", "reference": "", "hypothesis": "", "probe": "P?"}
        base.update(kw)
        return base

    # ── P1 ────────────────────────────────────────────────────────────────────
    assert uppercase_fraction("ABC") == 1.0
    assert uppercase_fraction("AbC") == 2 / 3
    assert uppercase_fraction("123 !?") == 0.0, "no letters → 0, never a crash"
    r1 = score_p1([
        row(reference="hello there", hypothesis="HELLO THERE"),
        row(reference="hello there", hypothesis="hello there"),
    ])
    assert r1.n == 2 and r1.pass_rate == 0.5, (r1.n, r1.pass_rate)
    assert r1.following_rate == 1.0, "both ARE transcripts, one just ignores the case"
    assert r1.detail["case_insensitive_wer"] == 0.0, r1.detail
    print("  [OK] P1: case constraint separated from perception")

    # ── P2 ────────────────────────────────────────────────────────────────────
    assert parse_count_line("a b c\n3") == ("a b c", 3)
    assert parse_count_line("a b c") == ("a b c", None), "one line → no count"
    assert parse_count_line("a b\nthree") == ("a b\nthree", None), "not an integer"
    r2 = score_p2([
        row(reference="one two three", hypothesis="one two three\n3"),   # self+ref ok
        row(reference="one two three", hypothesis="one two\n2"),         # self ok, ref no
        row(reference="one two three", hypothesis="one two three\n7"),   # self no
        row(reference="one two three", hypothesis="one two three"),      # no count line
    ])
    assert r2.following_rate == 0.75, r2.following_rate
    assert r2.pass_rate == 0.5, r2.pass_rate
    assert r2.detail["matches_reference_pct"] == 25.0, r2.detail
    print("  [OK] P2: self-consistency is the pass, reference agreement is separate")

    # ── P3 ────────────────────────────────────────────────────────────────────
    r3 = score_p3([
        row(reference="today", hypothesis="today", full_reference="it is today"),
        row(reference="today", hypothesis="Today.", full_reference="it is today"),
        row(reference="today", hypothesis="it is today", full_reference="it is today"),
        row(reference="today", hypothesis="the word is today",
            full_reference="it is today"),
    ])
    assert r3.pass_rate == 0.5, "normalisation must make 'Today.' match 'today'"
    assert r3.detail["transcript_dump_pct"] == 50.0, r3.detail
    assert r3.detail["single_word_pct"] == 50.0, r3.detail
    assert r3.following_rate == 0.5, "3-word dump is answer-shaped, 4-word is not"
    print("  [OK] P3: exact match, single-word rate and transcript dumps split apart")

    # ── P4 ────────────────────────────────────────────────────────────────────
    p4_rows = [
        row(key="a", qid="capital_france", question="q", expected="Paris",
            reference="Paris", hypothesis="Paris", context="audio"),
        row(key="b", qid="capital_france", question="q", expected="Paris",
            reference="Paris", hypothesis="the answer is paris", context="audio"),
        row(key="c", qid="two_plus_three", question="q", expected="5",
            reference="5", hypothesis="five", context="audio"),
        row(key="d", qid="two_plus_three", question="q", expected="5",
            reference="5", hypothesis="and then he went to the market today",
            context="audio"),
        row(key="e", qid="capital_france", question="q", expected="Paris",
            reference="Paris", hypothesis="Paris", context="text_only"),
    ]
    r4 = score_p4(p4_rows)
    assert r4.n == 4, r4.n
    assert r4.pass_rate == 0.25, r4.pass_rate     # only "Paris" is right AND in form
    assert r4.detail["with_audio"]["wrong_form"] == 0.25, r4.detail
    assert r4.following_rate == 0.5, "two of four are answer-shaped"
    assert r4.detail["text_only"]["correct"] == 1.0
    assert r4.detail["control_usable"] is True
    assert r4.detail["leakage_gap_pts"] == -75.0, r4.detail["leakage_gap_pts"]
    print("  [OK] P4: wrong-form never counts as correct; leakage gap is signed")

    # A control made of degenerate loops must NOT yield a gap.
    salad = "mushrooms mushrooms mushrooms mushrooms hostility hostility hostility"
    broken = [r for r in p4_rows if r["context"] == "audio"] + [
        row(key=f"t{i}", qid=q.qid, question="q", expected=q.answer,
            reference=q.answer, hypothesis=salad, context="text_only")
        for i, q in enumerate(P4_QUESTIONS)
    ]
    r4b = score_p4(broken)
    assert r4b.detail["control_usable"] is False, r4b.detail
    assert r4b.detail["leakage_gap_pts"] is None, r4b.detail
    assert "UNUSABLE" in r4b.headline, r4b.headline
    assert "cannot be read as a leakage measurement" in r4b.note.lower(), r4b.note
    print("  [OK] P4: a degenerate control is refused, not reported as a 0-point gap")

    # Transcription collapse is counted when the utterance's transcript is present.
    collapsed = [row(key="x", qid="capital_france", question="q", expected="Paris",
                     reference="Paris", hypothesis="are you a giant",
                     full_reference="are you a giant", context="audio")]
    r4c = score_p4(collapsed)
    assert r4c.detail["with_audio"]["transcribed_instead"] == 1.0, r4c.detail
    assert "transcribed the audio instead" in r4c.failures[0]["why"], r4c.failures
    print("  [OK] P4: 'transcribed the audio instead of answering' is counted, not lumped in")

    assert is_answer_shaped("Paris")
    assert not is_answer_shaped(""), "empty is not an answer"
    assert not is_answer_shaped("lift" * 30), \
        "a run-on with no spaces is one whitespace token but not a one-word answer"
    assert not is_answer_shaped("one two three four")
    print("  [OK] is_answer_shaped guards words AND characters")

    # A four-word 'answer' containing the word must NOT score.
    _, corr, _ = _p4_verdict("well the capital is paris i think",
                             P4_QUESTIONS[0])
    assert not corr, "a rambling response containing the word is not an answer"
    print("  [OK] P4: a transcript that happens to contain the answer does not pass")

    # ── P5 ────────────────────────────────────────────────────────────────────
    p5_rows = []
    for v in P5_VARIANTS:
        for i in range(2):
            hyp = ("hello there" if v.mode == "unformatted" else "Hello there.")
            if v.vid == "u3":
                hyp = "Hello there."          # a paraphrase that breaks adherence
            p5_rows.append(row(key=f"k{i}", mode=v.mode, variant_id=v.vid,
                               type=v.mode,
                               reference="hello there" if v.mode == "unformatted"
                                         else "Hello there.",
                               hypothesis=hyp))
    r5 = score_p5(p5_rows)
    pv = {(v["mode"], v["variant_id"]): v for v in r5.detail["per_variant"]}
    assert pv[("unformatted", "canonical")]["adherence_pct"] == 100.0
    assert pv[("unformatted", "u3")]["adherence_pct"] == 0.0, pv[("unformatted", "u3")]
    assert pv[("unformatted", "u3")]["wer"] > 0, "cased hypothesis costs raw-token WER"
    spread = r5.detail["spread_by_mode"]["unformatted"]
    assert spread["worst_variant"] == "u3" and spread["adherence_spread"] == 100.0, spread
    assert r5.detail["spread_by_mode"]["formatted"]["wer_spread"] == 0.0
    print("  [OK] P5: per-variant WER/adherence, spread measured within a mode")

    md = render_md([r1, r2, r3, r4, r5], {
        "checkpoint": "ck.pt", "step": 15040, "n_selected": 200,
        "shard": "s.tar", "split": "dev-clean", "seed": 17,
        "gate": {"P1": {"status": "infeasible", "reason": "uppercase unreachable"}},
        "probes_dropped": [{"probe": "P1", "status": "infeasible",
                            "reason": "uppercase unreachable"}],
    })
    assert "zero-shot" in md and "Feasibility gate" in md and "Leakage gap" in md, md[:400]
    assert "| **P5** prompt_variation |" in md
    print("  [OK] render_md: gate, summary table and per-probe detail")
    print("PASSED")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dir", type=Path, default=None,
                   help="Directory of probe JSONLs written by tools/run_probes.py "
                        "(also read for probe-run-manifest.json).")
    p.add_argument("--jsonl", type=Path, nargs="+", default=None, metavar="PATH",
                   help="Explicit probe JSONL paths instead of --dir.")
    p.add_argument("--manifest", type=Path, default=None,
                   help="Run manifest (default: <dir>/probe-run-manifest.json).")
    p.add_argument("--feasibility", type=Path, default=None,
                   help="Gate JSON, so the report states what was dropped and why. "
                        "Default: the path recorded in the manifest.")
    p.add_argument("--out-md", type=Path, default=None, dest="out_md")
    p.add_argument("--out-json", type=Path, default=None, dest="out_json")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)

    if args.self_test:
        _self_test()
        return 0
    if not args.dir and not args.jsonl:
        p.error("pass --dir or --jsonl")

    # ── Locate the per-probe files ───────────────────────────────────────────
    paths: dict[str, Path] = {}
    if args.dir:
        for pid in PROBE_IDS:
            cand = args.dir / probe_filename(pid)
            if cand.exists():
                paths[pid] = cand
    for path in (args.jsonl or []):
        rows = read_jsonl(path)
        if not rows:
            print(f"[warn] {path} is empty — skipping")
            continue
        pid = rows[0].get("probe")
        if pid not in SCORERS:
            print(f"[warn] {path}: unrecognised probe {pid!r} — skipping")
            continue
        paths[pid] = path
    if not paths:
        print("[error] no probe JSONLs found.")
        return 1

    # ── Manifest + gate, so the report can state what did NOT run ────────────
    manifest: dict = {}
    man_path = args.manifest or (args.dir / "probe-run-manifest.json" if args.dir else None)
    if man_path and Path(man_path).exists():
        manifest = json.loads(Path(man_path).read_text())
    elif man_path:
        print(f"[warn] no manifest at {man_path}: the report cannot state which "
              "probes were dropped by the feasibility gate, which is part of "
              "reporting this suite honestly.")

    gate: dict = {}
    gate_path = args.feasibility or manifest.get("feasibility")
    if gate_path and Path(gate_path).exists():
        gate = json.loads(Path(gate_path).read_text()).get("probes", {})

    meta = {
        "checkpoint":      manifest.get("checkpoint"),
        "step":            manifest.get("step"),
        "shard":           manifest.get("shard"),
        "split":           manifest.get("split"),
        "seed":            manifest.get("seed"),
        "n_selected":      (manifest.get("selection") or {}).get("n_selected"),
        "gate":            gate,
        "probes_dropped":  manifest.get("probes_dropped", []),
    }

    reports: list[ProbeReport] = []
    for pid in PROBE_IDS:
        if pid not in paths:
            continue
        rows = read_jsonl(paths[pid])
        if not rows:
            print(f"[warn] {paths[pid]} is empty — skipping")
            continue
        if any(r.get("gate_forced") for r in rows):
            print(f"[warn] {pid} rows carry gate_forced=true: this probe was run "
                  "against the feasibility gate's refusal. Its numbers describe "
                  "the vocabulary at least as much as the model.")
        rep = SCORERS[pid](rows)
        if any(r.get("gate_forced") for r in rows):
            rep.note = ("⚠ Run with --force against the feasibility gate. "
                        "The expected output is not fully reachable under the "
                        "pruned vocabulary, so this is not a clean measurement "
                        "of instruction-following. " + rep.note).strip()
        reports.append(rep)
        print(f"{pid} {rep.name:<17} n={rep.n:<5} "
              f"following {100 * rep.following_rate:5.1f}%   "
              f"pass {100 * rep.pass_rate:5.1f}%   {rep.headline}")

    md = render_md(reports, meta)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md)
        print(f"\nwrote {args.out_md}")
    else:
        print("\n" + md)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(
            {"meta": meta, "probes": [r.to_dict() for r in reports]}, indent=2))
        print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

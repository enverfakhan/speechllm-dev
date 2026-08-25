"""The probe suite's definitions — instructions, questions, paraphrases.

WHY THIS IS ITS OWN MODULE
--------------------------
Three tools must agree, exactly, on what each probe asks:

    tools/check_vocab_feasibility.py   decides whether a probe is POSSIBLE
                                       under the pruned vocabulary
    tools/run_probes.py                sends the instruction to the model
    tools/score_probes.py              scores the answer it expected

A second copy of any prompt string in any of them is a silent drift risk of the
worst kind: PrunedTokenizer.encode DROPS ids it has no mapping for (Decision
005), so a prompt that drifted — or that was never checked — does not raise.  It
just reaches the model with words missing, and the resulting "instruction
failure" is the harness's, not the model's.  Same pattern as
count_degeneracies.classify, which analyze_slices imports rather than reimplements.

Standard library only, so the feasibility gate and the scorer stay runnable with
no torch anywhere.

WHAT THE PROBES ARE FOR
-----------------------
Speech-IFEval style: every probe is an OUTPUT CONSTRAINT that is independent of
what the audio says.  A model can transcribe perfectly and still fail all of
them, and that separation is the point — instruction-following is measured apart
from perception.  All five are ZERO-SHOT: none of these instructions, question
texts or output formats appears anywhere in training, which only ever saw the
two strings in data.INSTRUCTION_VARIANTS.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── The two instructions the model was actually trained on ────────────────────
# Duplicated from data.INSTRUCTION_VARIANTS deliberately: importing data.py here
# would drag in torch and transformers, and this module is the one every
# stdlib-only tool leans on.  The copy is guarded — tools/score_probes.py's
# self-test asserts the two lists are identical when data.py is importable, and
# run_probes.py (which imports both) asserts it at run time.
TRAINED_INSTRUCTIONS: dict[str, str] = {
    "unformatted": "Transcribe the audio exactly as spoken, in lowercase with no punctuation.",
    "formatted":   "Transcribe the audio as written text, with capitalization, punctuation, and numbers as digits.",
}


@dataclass(frozen=True)
class Probe:
    """One probe: what it asks, what it measures, and what its output must be.

    Attributes:
        probe_id:    "P1".."P5"
        name:        short slug used in filenames and report rows
        instruction: the instruction string sent to the model.  For P4 it is a
                     TEMPLATE carrying "{question}" — see p4_instruction().
        measures:    one line for the report, saying what a failure means
        needs_audio: False only for the P4 text-only control condition
    """

    probe_id:    str
    name:        str
    instruction: str
    measures:    str


@dataclass(frozen=True)
class Question:
    """One P4 question with its accepted answers.

    Attributes:
        qid:     stable id, so a per-question rate survives a reordering
        text:    the question as asked
        answer:  the expected answer, in the form the instruction demands
        aliases: other spellings scored as CORRECT (case/format-insensitive
                 matching happens in the scorer, so these are only for genuinely
                 different words)
        wrong_form:
                 answers that are RIGHT but in the form the instruction did not
                 ask for — "five" where "5" was demanded.  Scored as a failure
                 of the constraint, reported separately as a sub-metric, because
                 the two failure modes are different findings: the model knows
                 the fact but ignores the format, versus it does not know it.
    """

    qid:        str
    text:       str
    answer:     str
    aliases:    tuple[str, ...] = ()
    wrong_form: tuple[str, ...] = ()


# ── P4: ten questions whose answer is one word and has nothing to do with audio
# Chosen to be answerable by a frozen Llama 3.1 8B Instruct with no context, and
# to sit inside a vocabulary frozen on 19th-century audiobooks — which is not a
# given, hence the feasibility gate.  Three ask for digits, which is also the
# only place this suite exercises the numeral rule.
P4_QUESTIONS: tuple[Question, ...] = (
    Question("capital_france",  "What is the capital of France?",  "Paris"),
    Question("capital_italy",   "What is the capital of Italy?",   "Rome"),
    Question("capital_england", "What is the capital of England?", "London"),
    Question("two_plus_three",  "What is two plus three? Answer in digits.", "5",
             wrong_form=("five",)),
    Question("ten_minus_four",  "What is ten minus four? Answer in digits.", "6",
             wrong_form=("six",)),
    Question("days_in_week",    "How many days are in a week? Answer in digits.", "7",
             wrong_form=("seven",)),
    Question("color_sky",       "What color is the sky on a clear day?", "blue"),
    Question("color_grass",     "What color is grass?",                  "green"),
    Question("opposite_hot",    "What is the opposite of hot?",          "cold"),
    Question("animal_meow",     "What animal says meow?",                "cat",
             aliases=("kitten",)),
)

# The P4 instruction is a template; the question is spliced in.  Kept as one
# string so the feasibility gate checks the exact prompt that will be sent.
P4_INSTRUCTION_TEMPLATE: str = "Ignore the audio entirely. Answer in one word: {question}"


def p4_instruction(question: Question) -> str:
    """The full instruction string for one P4 question."""
    return P4_INSTRUCTION_TEMPLATE.format(question=question.text)


# ── P5: paraphrases of the two trained instructions ───────────────────────────
# Each is meant to be SEMANTICALLY EXACT — same output contract, different
# wording — so any WER or adherence spread across them is prompt sensitivity and
# not a different task.  The canonical string is included as a variant so the
# spread is measured inside one run, over the same utterances, in the same
# batches, under the same generation cap.  (Comparing against a banked WER sweep
# instead would confound prompt sensitivity with batch composition: the Whisper
# encoder attends across mel padding, so a different batch perturbs the audio
# embeddings.)
@dataclass(frozen=True)
class Variant:
    """One P5 instruction wording.

    Attributes:
        vid:         stable id ("canonical", "u1"...)
        mode:        "unformatted" | "formatted" — which output contract it states
        text:        the instruction string
        is_canonical: True for the exact string used in training
    """

    vid:          str
    mode:         str
    text:         str
    is_canonical: bool = False


P5_VARIANTS: tuple[Variant, ...] = (
    Variant("canonical", "unformatted", TRAINED_INSTRUCTIONS["unformatted"], True),
    Variant("u1", "unformatted",
            "Write out the words in the audio exactly as they are spoken, using only "
            "lowercase letters and no punctuation."),
    Variant("u2", "unformatted",
            "Give a word for word transcript of the audio in lowercase, without any "
            "punctuation marks."),
    Variant("u3", "unformatted",
            "Transcribe what you hear exactly, all in lowercase and with no punctuation."),
    Variant("canonical", "formatted", TRAINED_INSTRUCTIONS["formatted"], True),
    Variant("f1", "formatted",
            "Transcribe the audio into written text, using capital letters where "
            "appropriate, punctuation, and digits for numbers."),
    Variant("f2", "formatted",
            "Write the audio out as properly written text, with capitalization, "
            "punctuation, and numbers written as digits."),
    Variant("f3", "formatted",
            "Produce a written transcript of the audio, including capitalization and "
            "punctuation, and write numbers as digits."),
)


# ── P1-P3 ─────────────────────────────────────────────────────────────────────
PROBES: dict[str, Probe] = {
    "P1": Probe(
        "P1", "uppercase",
        "Transcribe the audio exactly as spoken, in all capital letters.",
        "Can it hold an output constraint (case) it was never trained on, "
        "while transcribing the same audio?",
    ),
    "P2": Probe(
        "P2", "word_count",
        # "transcription" is UNREACHABLE in this pruned vocabulary (the gate
        # caught it); "transcript" is fine.  A word missing from a prompt does
        # not raise — it is simply deleted on the way in — so the wording here
        # is dictated by tools/check_vocab_feasibility.py, not by taste.
        "Transcribe the audio exactly as spoken. Then on a new line write the "
        "number of words in your transcript, as digits.",
        "Can it emit a second, structured field after the transcript and keep "
        "it consistent with what it just wrote?",
    ),
    "P3": Probe(
        "P3", "last_word",
        "Listen to the audio and answer with only the last word spoken.",
        "Can it answer ABOUT the audio instead of transcribing it? Dumping the "
        "whole transcript is the ASR-collapse signature.",
    ),
    "P4": Probe(
        "P4", "ignore_audio",
        P4_INSTRUCTION_TEMPLATE,
        "Can it disregard the audio entirely on request? The gap against the "
        "same question asked with no audio is the leakage measurement.",
    ),
    "P5": Probe(
        "P5", "prompt_variation",
        "<one of P5_VARIANTS>",
        "Is the trained behaviour attached to the instruction's MEANING or to "
        "its exact surface string?",
    ),
}

PROBE_IDS: tuple[str, ...] = ("P1", "P2", "P3", "P4", "P5")

# Probe id → the JSONL basename run_probes.py writes and score_probes.py reads.
def probe_filename(probe_id: str) -> str:
    """Canonical JSONL basename for one probe's generations."""
    return f"probe-{probe_id}-{PROBES[probe_id].name}.jsonl"


# ── Feasibility candidate groups ──────────────────────────────────────────────
# What tools/check_vocab_feasibility.py must prove reachable before a probe is
# allowed to run.  A probe whose expected OUTPUT cannot be tokenized inside the
# pruned vocabulary is impossible by construction, and reporting it as a model
# failure would be wrong.  A probe whose PROMPT cannot be tokenized is worse: it
# does not fail loudly, it silently reaches the model with words deleted.

@dataclass(frozen=True)
class CandidateGroup:
    """One set of strings whose reachability decides a probe's fate.

    Attributes:
        name:      report row name
        kind:      "output" — strings the model must be able to EMIT
                   "prompt" — strings that must survive encoding on the way IN
        probes:    probe ids that this group gates
        strings:   the candidates
        min_pct:   fraction of strings that must be fully reachable to pass.
                   Prompts are 100 by definition: one dropped token mangles the
                   instruction.  Outputs allow a margin because a single exotic
                   reference does not make a probe impossible.
        note:      one line for the report
    """

    name:    str
    kind:    str
    probes:  tuple[str, ...]
    strings: tuple[str, ...]
    min_pct: float = 100.0
    note:    str = ""
    # Filled in by the gate, not by the caller.
    results: list = field(default_factory=list, compare=False)


def digit_candidates() -> tuple[str, ...]:
    """0-99, bare and newline-prefixed — P2's second line and P4's digit answers.

    Both forms are checked because BPE is context-sensitive: "7" after a newline
    is not necessarily the same token as "7" at the start of a string, and P2
    demands the count on a NEW LINE.
    """
    bare = [str(n) for n in range(100)]
    return tuple(bare + [f"\n{n}" for n in range(100)])


def prompt_candidates() -> tuple[str, ...]:
    """Every instruction string this suite will send to the model."""
    out = [PROBES[p].instruction for p in ("P1", "P2", "P3")]
    out += [p4_instruction(q) for q in P4_QUESTIONS]
    out += [v.text for v in P5_VARIANTS]
    return tuple(out)


def p4_answer_candidates() -> tuple[str, ...]:
    """Every accepted P4 answer, in the two positions a generation can start from.

    A one-word answer is generated at the start of the assistant turn, so the
    bare form is the one that matters; the space-prefixed form is checked too
    because it is a different BPE token and a model that emits a leading space
    would be scored on it.

    ALL-CAPS forms are deliberately NOT candidates.  Upper case is unreachable
    corpus-wide in this vocabulary (the same finding that makes P1 impossible),
    and P4's instruction never asks for capitals — requiring a form the model
    provably cannot emit would fail the probe for the vocabulary's reasons
    rather than the model's.  The scorer matches case-insensitively regardless.
    """
    out: list[str] = []
    for q in P4_QUESTIONS:
        for a in (q.answer, *q.aliases, *q.wrong_form):
            out.extend([a, f" {a}", a.lower(), f" {a.lower()}"])
    return tuple(dict.fromkeys(out))   # de-duplicate, preserve order


def build_groups(uppercase_samples: tuple[str, ...]) -> list[CandidateGroup]:
    """Assemble every candidate group the gate should measure.

    Args:
        uppercase_samples: real dev-clean references, upper-cased — the P1
                           expected outputs.  Empty when no shard was supplied,
                           in which case the P1 group is omitted and P1 stays
                           UNDECIDED rather than being silently passed.

    Returns:
        list of CandidateGroup, in report order
    """
    groups = [
        CandidateGroup(
            "probe_prompts", "prompt", PROBE_IDS, prompt_candidates(), 100.0,
            "Every instruction string the suite sends. PrunedTokenizer.encode "
            "drops unmapped ids silently, so an unreachable word here reaches "
            "the model as a hole in the prompt.",
        ),
        CandidateGroup(
            "digits_0_99", "output", ("P2", "P4"), digit_candidates(), 100.0,
            "P2's word count and P4's three arithmetic answers are digit "
            "strings; bare and newline-prefixed forms both checked.",
        ),
        CandidateGroup(
            "p4_answers", "output", ("P4",), p4_answer_candidates(), 100.0,
            "Every accepted P4 answer. An unreachable answer makes its question "
            "unanswerable by construction.",
        ),
    ]
    if uppercase_samples:
        groups.insert(0, CandidateGroup(
            "uppercase_transcripts", "output", ("P1",), uppercase_samples, 95.0,
            "Real dev-clean references upper-cased — P1's expected output. "
            "Upper-case subwords are the likeliest casualty of a vocabulary "
            "pruned to lower-cased and sentence-cased audiobook text.",
        ))
    return groups

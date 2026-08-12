"""Generate formatted transcript labels for LibriSpeech via the Anthropic Message Batches API.

The formatted label convention is defined by FORMATTING_SPEC.md; the labelling
prompt (passed verbatim as the system prompt) lives in prompts/formatting_v1.txt.
This tool drives the whole labelling run: it turns LibriSpeech chapters into
batch requests, submits them, polls for results, validates every returned
utterance against the spec's word-preservation contract, retries only the
utterances that failed, and merges everything into a labels JSONL.

Every subcommand reads and writes ONE state manifest (`manifest.json` in
--out-dir), so every command is idempotent and the whole run is resumable after
an interruption at any point.  Raw model responses are kept as individual files
under `<out-dir>/responses/` so a re-validate never needs the API.

    prepare   walk *.trans.txt, chunk chapters, write requests to the manifest
    submit    create Message Batch(es) from every not-yet-submitted request
    poll      download results for finished batches, exit when all are terminal
    validate  parse responses, run the FORMATTING_SPEC.md §6/§7 check per
              utterance; --show-failures also sorts each failure into
              `model-edit` (a real §1 violation) or `probable-reference-defect`
              (a corpus typo the model read through)
    retry     re-ask ONLY failed utterances, with valid neighbours as context;
              reports failures it did NOT queue because an attempt is already
              outstanding (queued-but-unsent → `submit`; in flight → `poll`)
    finalize  merge to JSONL + print a totals / failure / cost summary

Typical run:

    export ANTHROPIC_API_KEY=...
    python tools/label_formatted.py prepare \\
        --librispeech-root data/librispeech/LibriSpeech/dev-clean \\
        --out-dir          runs/label-haiku \\
        --prompt-file      prompts/formatting_v1.txt \\
        --chapters         20                      # pilot subset; omit for all
    python tools/label_formatted.py submit   --out-dir runs/label-haiku
    python tools/label_formatted.py poll     --out-dir runs/label-haiku
    python tools/label_formatted.py validate --out-dir runs/label-haiku
    python tools/label_formatted.py retry    --out-dir runs/label-haiku   # repeat
    python tools/label_formatted.py finalize --out-dir runs/label-haiku \\
        --labels-out data/labels_formatted.jsonl

Comparing two models is two out-dirs: run prepare/submit into each with a
different --model, then diff the finalize summaries.

Standard library only, except the `anthropic` SDK, which is imported lazily so
that --self-test runs in an environment without it:

    python tools/label_formatted.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ── Batch API limits ──────────────────────────────────────────────────────────
# The API caps a batch at 100,000 requests and 256 MB.  We cut well below both:
# a rejected batch would mean re-serialising and re-submitting everything.
MAX_REQUESTS_PER_BATCH: int = 50_000
MAX_BATCH_BYTES:        int = 200 * 1024 * 1024

# ── Request sizing ────────────────────────────────────────────────────────────
# max_tokens is derived per request from the text being formatted, never fixed
# globally: a 70-utterance chunk and a single retried utterance need budgets two
# orders of magnitude apart, and a too-small budget truncates the JSON object
# (stop_reason="max_tokens"), which surfaces as an unparseable response.
#
# The multiplier is measured rather than guessed. Over the 6,778 answered
# requests of the first full-corpus run, output tokens came to 1.08x the
# estimated input at the median, 1.34x at p99, and 2.25x at the worst; 1.5x left
# the tail with no head-room at all.  Head-room is close to free — max_tokens is
# a ceiling, and only the tokens the model actually emits are billed — so this is
# sized for the tail, not the median.
#
# Every attempt beyond the first multiplies the budget again by
# RETRY_MAX_TOKENS_GROWTH.  A retry costs a whole attempt out of the cap, and
# spending one to truncate a second time buys nothing.
CHARS_PER_TOKEN:          float = 4.0   # English heuristic; no tokenizer here
OUTPUT_TOKEN_MULTIPLIER:  float = 2.5   # formatting inflates: casing, punctuation, digits
MAX_TOKENS_MARGIN:        int   = 512
MAX_TOKENS_FLOOR:         int   = 1024
MAX_TOKENS_CEILING:       int   = 16_384
JSON_CHARS_PER_ENTRY:     int   = 10    # `"": "",\n` around each id/text pair
RETRY_MAX_TOKENS_GROWTH:  float = 2.0   # budget factor per attempt after the first

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL:                str = "claude-haiku-4-5"
DEFAULT_MAX_UTTS_PER_REQUEST: int = 70
DEFAULT_MAX_ATTEMPTS:         int = 3
DEFAULT_POLL_INTERVAL:        int = 300
ERRORS_PRINTED_PER_BATCH:     int = 5   # failed entries shown verbatim during poll
CHUNK_OVERLAP:                int = 3   # context utterances carried across a chunk seam
RETRY_CONTEXT_NEIGHBOURS:     int = 2   # valid neighbours per side when retrying one utterance

# ── Pricing (USD per 1M tokens, standard list price) ──────────────────────────
# The Batches API bills at 50% of these.  Unlisted models print token totals
# only rather than a wrong number.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5":  (3.00, 15.00),
    "claude-opus-5":    (5.00, 25.00),
}
BATCH_DISCOUNT:    float = 0.50
CACHE_WRITE_MULT:  float = 1.25
CACHE_READ_MULT:   float = 0.10

MANIFEST_VERSION: int = 1

# A request in one of these states has no verdict from the API yet: it is either
# not sent or still running. Nothing may be concluded about its utterances, and
# they must not be queued for a retry behind its back.
IN_FLIGHT_STATES: frozenset[str] = frozenset({"pending", "submitted"})


# ══════════════════════════════════════════════════════════════════════════════
# FORMATTING_SPEC.md §5–§7 — normalisation and validation
# ══════════════════════════════════════════════════════════════════════════════

# §5 inverse allowlist, keyed without the period.
INVERSE_ALLOWLIST: dict[str, str] = {
    "mrs": "missus",
    "mr":  "mister",
    "dr":  "doctor",
    "st":  "saint",
}

# The trailing period is OPTIONAL: "Mrs Rachel" and "Mrs. Rachel" both normalise
# to "missus rachel". A missing period is a punctuation slip, not a changed
# word, so failing on it would burn a retry over something §1 does not protect.
#
# Two details are load-bearing:
#   \b after the abbreviation — without it "st" matches inside "street"
#                               (1,820 occurrences in LibriSpeech) and yields
#                               "sainteet".
#   longest alternative first  — otherwise "mr" wins inside "mrs".
# The reference side is unaffected either way: bare "st"/"mr"/"mrs"/"dr" occur
# zero times in the corpus, which always spells out saint/mister/missus/doctor.
_ALLOWLIST_RE = re.compile(
    r"\b(" + "|".join(sorted(INVERSE_ALLOWLIST, key=len, reverse=True)) + r")\b\.?",
)

# A wildcard (digit-bearing) token may stand for this many spoken words.
# "1895" is three ("eighteen ninety five"); the ceiling covers the longest
# renderings the spec permits without letting a wildcard swallow a clause.
WILDCARD_MIN_WORDS: int = 1
WILDCARD_MAX_WORDS: int = 5

_DIGIT_RE = re.compile(r"\d")


def normalize(text: str) -> str:
    """FORMATTING_SPEC.md §6 normalisation, shared by the validator and WER.

    lowercase → inverse allowlist → strip all punctuation → collapse whitespace.
    Digit-bearing tokens are left as digits; expanding them back to words is
    ambiguous, so §7 handles them by alignment instead.

    Punctuation is handled two ways, and the difference is the point:

    apostrophes are DELETED, so "boy's" → "boys" and "didn't" → "didnt" stay
    single tokens. Reference apostrophe placement is unreliable in both
    directions (it drops "boy's", adds "abalone's"), and adding or removing a
    possessive is exactly the punctuation the formatted mode is supposed to
    supply. Expanding a contraction is still caught, because that changes token
    COUNT, which is orthogonal to the apostrophe character.

    everything else becomes a SPACE, so a join the spec forbids ("cotton-warp")
    normalises back to two tokens instead of silently fusing into one.
    """
    text = text.lower()
    text = _ALLOWLIST_RE.sub(lambda m: INVERSE_ALLOWLIST[m.group(1)], text)

    chars: list[str] = []
    for ch in text:
        if ch.isalnum() or ch.isspace():
            chars.append(ch)
        elif ch != "'":
            chars.append(" ")
    return " ".join("".join(chars).split())


def is_wildcard(token: str) -> bool:
    """A digit-bearing token stands for 1–5 spoken reference words (§7)."""
    return bool(_DIGIT_RE.search(token))


def align(hyp: list[str], ref: list[str]) -> tuple[bool, list[tuple[str, list[str]]]]:
    """FORMATTING_SPEC.md §7 alignment.

    Non-wildcard hypothesis tokens must match reference words exactly, in order;
    each wildcard consumes 1–5 consecutive reference words.  Simple reachability
    DP with a backtrace, so a success also yields the numeric spans for the
    spot-check log.

    Returns (ok, spans) where spans is [(digit_token, [reference words])].
    """
    n_hyp, n_ref = len(hyp), len(ref)
    reach = [[False] * (n_ref + 1) for _ in range(n_hyp + 1)]
    back: dict[tuple[int, int], tuple[int, int]] = {}
    reach[0][0] = True

    for i in range(n_hyp):
        token = hyp[i]
        wildcard = is_wildcard(token)
        for j in range(n_ref + 1):
            if not reach[i][j]:
                continue
            if wildcard:
                for k in range(WILDCARD_MIN_WORDS, WILDCARD_MAX_WORDS + 1):
                    if j + k <= n_ref and not reach[i + 1][j + k]:
                        reach[i + 1][j + k] = True
                        back[(i + 1, j + k)] = (i, j)
            elif j < n_ref and token == ref[j] and not reach[i + 1][j + 1]:
                reach[i + 1][j + 1] = True
                back[(i + 1, j + 1)] = (i, j)

    if not reach[n_hyp][n_ref]:
        return False, []

    spans: list[tuple[str, list[str]]] = []
    node = (n_hyp, n_ref)
    while node != (0, 0):
        i, j = node
        pi, pj = back[node]
        if is_wildcard(hyp[pi]):
            spans.append((hyp[pi], ref[pj:j]))
        node = (pi, pj)
    spans.reverse()
    return True, spans


def validate_utterance(
    formatted: str,
    unformatted: str,
) -> tuple[bool, list[tuple[str, list[str]]]]:
    """Run the §6/§7 check for one utterance. Returns (ok, numeric spans)."""
    return align(normalize(formatted).split(), normalize(unformatted).split())


def minimal_divergence(
    ref: list[str],
    hyp: list[str],
) -> tuple[list[str], list[str]]:
    """The smallest differing window between two normalised token lists.

    Trims the common prefix and suffix so a report shows the offending edit
    ("an" → "and") instead of two forty-word sentences the reader has to diff by
    eye.  Purely a presentation aid — validity is decided by align().
    """
    limit = min(len(ref), len(hyp))
    prefix = 0
    while prefix < limit and ref[prefix] == hyp[prefix]:
        prefix += 1
    suffix = 0
    while suffix < limit - prefix and ref[-1 - suffix] == hyp[-1 - suffix]:
        suffix += 1
    return ref[prefix : len(ref) - suffix], hyp[prefix : len(hyp) - suffix]


# ══════════════════════════════════════════════════════════════════════════════
# Reference-defect heuristic (FORMATTING_SPEC.md §7)
#
# §7 sorts alignment failures into two piles by whether the reference side of
# the failing edit is a real word: `model-edit` is a genuine §1 violation and is
# routed to retry; `probable-reference-defect` is a corpus typo ("bui", "canyou")
# that the model is correctly reading through, which no conformant formatting
# can ever align to.
#
# The test is deliberately conservative — hence "probable". A reference-side
# token counts as suspect only if nothing vouches for it: not a dictionary word,
# not attested elsewhere in the corpus, not digit-bearing, not a §5 expansion.
# ══════════════════════════════════════════════════════════════════════════════

TAG_MODEL_EDIT:       str = "model-edit"
TAG_REFERENCE_DEFECT: str = "probable-reference-defect"

SYSTEM_WORDLISTS: tuple[Path, ...] = (
    Path("/usr/share/dict/words"),
    Path("/usr/share/dict/american-english"),
    Path("/usr/share/dict/british-english"),
    Path("/usr/dict/words"),
)

# LibriSpeech is human-verified, so a real word recurs across the corpus while a
# transcription defect is essentially unique. This signal needs no dictionary at
# all and sharpens as the corpus grows — at full-corpus scale it carries most of
# the weight, and on a small pilot it carries little.
CORPUS_ATTESTATION_MIN: int = 2

# Last-resort lexicon: the few hundred commonest English words, which is what
# trimmed edit windows overwhelmingly consist of ("to day", "any thing",
# "arm chair"). Small on purpose — it exists so the heuristic degrades to
# something useful without a dictionary, not to be a dictionary.
FALLBACK_WORDS: frozenset[str] = frozenset("""
a able about above across act after again against age ago air all almost alone
along already also although always am among an and another answer any anything
appear are arm around arrived art as ask at away back bad be bear beautiful
became because become bed been before began begin behind being believe below
beside best better between beyond big bird black blood blue boat body book both
box boy bread break bring brother brought brown but buy by call came can cannot
car care carry case cat catch caught cause certain chair chance change chief
child children church city class clear close cold colour come common company
could country course cover cried cry cut dark daughter day dead dear death deep
did die different do does dog done door doubt down draw dream dress drink drive
drop dry during each ear early earth east easy eat effect eight either else end
enough enter even evening ever every everything eye face fact fail fall family
far fast father fear feel feet fell felt few field fight fill find fine fire
first fish five fly follow food foot for force form found four free friend from
front full gave general get girl give glad go god gold gone good got great green
grew ground grow had hair half hand happy hard has hat have he head hear heard
heart heavy held help her here herself high hill him himself his hold home hope
horse hot hour house how however human hundred husband i idea if ill in indeed
instead interest into is it its itself john join journey joy just keep kept kind
king knew know known lady land large last late laugh law lay lead learn least
leave led left leg less let letter lie life light like line lips listen little
live long look lord lost lot love low made make man manner many march mark
master matter may me mean men met might mile mind mine minute miss moment money
month moon more morning morrow most mother mountain mouth move much must my
myself name near necessary need never new news next night nine no nor north not
nothing now number obliged observed of off often oh old on once one only open or
order other our out over own page paper part party pass past pay people perhaps
person picture piece place plain play please point poor position possible power
present pretty probably promise pull purpose put quality question quick quiet
quite rain raised ran rather reach read ready real reason receive red remain
remember reply rest result return rich ride right rise river road rock roll room
round run said sail same sat save saw say school sea season second see seem seen
self send sense sent serve set seven several shall shape she ship shore short
should show side sight silence simple since sing single sir sister sit six size
sky sleep slow small smile snow so soft some son song soon sort sound south
speak special spirit spoke spring stairs stand star start state stay step still
stone stood stop story straight strange street strength strike strong such
sudden summer sun suppose sure surface sweet table take taken talk teach tell
ten than that the their them themselves then there these they thing think third
this those though thought three through throw thus till time to today together
told tomorrow too took top touch toward town travel tree true try turn twelve
twenty two under understand until up upon us use usual very village visit voice
wait walk wall want war warm was watch water way we wear week well went were
west what when where whether which while white who whole whom whose why wide
wife wild will wind window wine winter wish with within without woman women
wonder wood word work world would write written wrong wrote yard year yes yet
you young your yourself
""".split())

# Contractions and §5 expansions a plain wordlist will not carry. §6 deletes
# apostrophes, so the forms that actually reach this test are the apostrophe-less
# ones ("didnt", "oclock"); the apostrophe spellings are kept as a cheap hedge.
ALWAYS_KNOWN: frozenset[str] = frozenset(INVERSE_ALLOWLIST.values()) | frozenset("""
aint arent cant couldnt darent didnt doesnt dont hadnt hasnt havent isnt
mustnt neednt oclock shant shouldnt theres wasnt werent wont wouldnt
im ive ill id hes shes its theyre theyve weve youre youve youll
ain't aren't can't couldn't daren't didn't doesn't don't hadn't hasn't haven't
isn't mustn't needn't o'clock shan't shouldn't wasn't weren't won't wouldn't
""".split())


@dataclass(frozen=True)
class Lexicon:
    """What this machine knows about English, plus what this corpus attests."""

    words: frozenset[str]
    source: str
    comprehensive: bool                 # a real dictionary, not the fallback list
    corpus_counts: dict[str, int]
    attestation: str = "unknown"        # which text the counts were built from

    def vouches_for(self, token: str) -> bool:
        """True when something independent says this reference token is real."""
        if not token or is_wildcard(token):
            return True                                  # §7 handles digits
        if token in ALWAYS_KNOWN or token in self.words:
            return True
        # Possessives and contractions: check the base form. §6 strips the
        # apostrophe, so "lynde's" arrives as "lyndes" and a bare trailing "s"
        # has to be tried alongside the apostrophe spellings.
        for suffix in ("s", "'s", "'ll", "'re", "'ve", "'d", "'m", "n't"):
            if token.endswith(suffix) and token[: -len(suffix)] in self.words:
                return True
        return self.corpus_counts.get(token, 0) >= CORPUS_ATTESTATION_MIN


def _load_wordlist() -> tuple[frozenset[str], str, bool]:
    """Best available English wordlist. Never raises, never a hard dependency."""
    for path in SYSTEM_WORDLISTS:
        try:
            if path.is_file():
                words = {
                    line.strip().lower()
                    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
                    if line.strip()
                }
                if words:
                    return frozenset(words), f"{path} ({len(words):,} words)", True
        except OSError:
            continue

    try:                                                       # pragma: no cover
        from wordfreq import top_n_list

        words = {w.lower() for w in top_n_list("en", 100_000)}
        return frozenset(words), f"wordfreq ({len(words):,} words)", True
    except Exception:
        pass

    try:                                                       # pragma: no cover
        from spellchecker import SpellChecker

        words = {w.lower() for w in SpellChecker().word_frequency.dictionary}
        return frozenset(words), f"pyspellchecker ({len(words):,} words)", True
    except Exception:
        pass

    return FALLBACK_WORDS, f"bundled fallback ({len(FALLBACK_WORDS)} words)", False


def build_lexicon(
    corpus_counts: dict[str, int],
    attestation: str = "unknown",
) -> Lexicon:
    words, source, comprehensive = _load_wordlist()
    return Lexicon(
        words=words,
        source=source,
        comprehensive=comprehensive,
        corpus_counts=corpus_counts,
        attestation=attestation,
    )


def attestation_counts(
    manifest: Manifest,
    vocab_root: str | None = None,
) -> tuple[dict[str, int], str]:
    """Token counts for the attestation signal, from the widest text available.

    Prefer the WHOLE LibriSpeech tree over the labelled subset: it is the same
    domain, it is already on disk, and it costs a few seconds to scan. Breadth
    is what makes this signal work — an ordinary word like "rejection" occurs
    once in a 2.7k-utterance pilot (and would look like a defect) but 28 times
    across the full corpus, while a real defect like "bui" stays at one however
    much text you add.

    Defaults to the tree this run was prepared from, which for a single-split
    run is only that split. Point --vocab-root at the whole LibriSpeech tree to
    widen it; nothing else about the run changes, since this signal is computed
    at report time and never stored.

    Falls back to the labelled subset when neither is reachable, e.g. a run
    directory synced to a machine without the corpus.
    """
    root = Path(vocab_root or manifest.config["librispeech_root"])
    if root.is_dir():
        texts = [text for _, utterances in iter_chapters(root) for _, text in utterances]
        if texts:
            return corpus_token_counts(texts), f"{len(texts):,} utterances under {root}"
    return (
        corpus_token_counts([u["unformatted"] for u in manifest.utterances.values()]),
        f"{len(manifest.utterances):,} labelled utterances only (corpus not reachable)",
    )


def corpus_token_counts(texts: Iterator[str] | list[str]) -> dict[str, int]:
    """How many distinct utterances each normalised token appears in."""
    counts: dict[str, int] = {}
    for text in texts:
        for token in set(normalize(text).split()):
            counts[token] = counts.get(token, 0) + 1
    return counts


def classify_alignment_failure(
    ref_edit: list[str],
    lexicon: Lexicon,
) -> tuple[str, list[str]]:
    """Tag one alignment failure from the reference side of its trimmed edit.

    Returns (tag, suspect tokens). Suspect tokens are reported so a misfire is
    visible rather than silent — with a small lexicon, a rare-but-real word can
    land here and the reader can see exactly which token caused it.
    """
    suspects = [token for token in ref_edit if not lexicon.vouches_for(token)]
    return (TAG_REFERENCE_DEFECT if suspects else TAG_MODEL_EDIT), suspects


# ══════════════════════════════════════════════════════════════════════════════
# Response parsing
# ══════════════════════════════════════════════════════════════════════════════

def strip_code_fences(text: str) -> str:
    """Drop a ```json ... ``` wrapper the prompt asked the model not to emit."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()[1:]                       # opening fence + lang tag
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def iter_json_objects(text: str) -> Iterator[str]:
    """Yield every balanced top-level {...} span, in order of appearance.

    Brace-counting with string/escape awareness, so a brace inside a formatted
    utterance cannot close the object early.  A span left unbalanced at the end
    of the text (the max_tokens case) is simply never yielded.
    """
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0:
                yield text[start : index + 1]


def parse_response(text: str) -> dict[str, str] | None:
    """Parse a response into {utterance_id: formatted}, or None if unparseable.

    Tolerates accidental code fences and prose wrapped around the object.
    Anything that is not a flat string→string mapping is rejected rather than
    half-accepted.

    A response may also contain SEVERAL objects: a model that miscounts talks
    itself into a do-over ("Wait, let me recount the utterances. Let me restart")
    and emits a second, complete object after the first.  Scanning for balanced
    spans keeps both readable, where taking the text from the first `{` to the
    last `}` spans the gap between them and parses as nothing at all — 38
    utterances of the first full-corpus run were written off as `unparseable`
    over exactly this.  The largest object wins, later beating earlier on a tie,
    so the model's own correction is the one that counts.
    """
    stripped = strip_code_fences(text)

    best: dict[str, str] | None = None
    for candidate in (stripped, *iter_json_objects(stripped)):
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if (
            isinstance(parsed, dict)
            and all(isinstance(k, str) and isinstance(v, str) for k, v in parsed.items())
            and (best is None or len(parsed) >= len(best))
        ):
            best = parsed
    return best


# ══════════════════════════════════════════════════════════════════════════════
# LibriSpeech walking and chapter chunking
# ══════════════════════════════════════════════════════════════════════════════

def iter_chapters(root: Path) -> Iterator[tuple[str, list[tuple[str, str]]]]:
    """Yield (chapter_id, [(utterance_id, lowercased text)]) per *.trans.txt.

    One trans.txt is one chapter; utterances stay in file order, which is the
    order they are spoken, because the prompt relies on consecutive slices.
    """
    for trans_file in sorted(root.rglob("*.trans.txt")):
        chapter = trans_file.name.removesuffix(".trans.txt")
        utterances: list[tuple[str, str]] = []
        with trans_file.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                key, _, raw = line.partition(" ")
                if raw:
                    # Lowercase only — apostrophes are part of the words (§1).
                    utterances.append((key, raw.lower()))
        if utterances:
            yield chapter, utterances


def select_chapters(chapters: list[str], n: int) -> list[str]:
    """Deterministic pilot subset: n chapters spread evenly over sorted order.

    Even spacing rather than a prefix, because sorted LibriSpeech ids cluster by
    speaker and a prefix would pilot on one or two voices.
    """
    if n >= len(chapters):
        return list(chapters)
    if n <= 1:
        return chapters[:1]
    step = (len(chapters) - 1) / (n - 1)
    # Half-up rather than round(), whose banker's rounding makes the picked
    # indices needlessly hard to predict by hand.
    return [chapters[int(i * step + 0.5)] for i in range(n)]


def chunk_chapter(
    utt_ids: list[str],
    max_utts: int,
    overlap: int = CHUNK_OVERLAP,
) -> list[tuple[list[str], list[str]]]:
    """Split a chapter into [(target_ids, context_ids)].

    Targets partition the chapter exactly once; context ids are the `overlap`
    utterances on either side of the seam, repeated into the neighbouring chunk
    so a proper noun or a mid-sentence construction is not judged blind at the
    boundary.  Context ids are never formatted — they are marked context-only in
    the request.

    Per §3 the context does NOT decide an utterance's opening capital or
    terminal mark: every utterance is framed as a self-contained sentence, so
    those are fixed regardless of where the chunk seam happens to fall.
    """
    if len(utt_ids) <= max_utts:
        return [(list(utt_ids), [])]

    chunks: list[tuple[list[str], list[str]]] = []
    for start in range(0, len(utt_ids), max_utts):
        targets = utt_ids[start : start + max_utts]
        end = start + len(targets)
        before = utt_ids[max(0, start - overlap) : start]
        after = utt_ids[end : end + overlap]
        chunks.append((targets, before + after))
    return chunks


def build_user_message(
    context: list[tuple[str, str]],
    targets: list[tuple[str, str]],
) -> str:
    """Render the user turn: labelled context block, then the utterances to format."""
    parts: list[str] = []
    if context:
        parts.append(
            "CONTEXT ONLY — surrounding utterances, provided for proper-noun "
            "casing and for punctuation inside an utterance. Each utterance is "
            "still formatted as a self-contained sentence, so do NOT use this "
            "context to decide an opening capital or a terminal mark. Do NOT "
            "format these lines and do NOT include their IDs in your output."
        )
        parts.append("\n".join(f"{key}\t{text}" for key, text in context))
        parts.append("")
    noun = "ID" if len(targets) == 1 else "IDs"
    parts.append(
        f"FORMAT THESE — your JSON object must contain exactly these "
        f"{len(targets)} {noun}, and no others."
    )
    parts.append("\n".join(f"{key}\t{text}" for key, text in targets))
    return "\n".join(parts)


def estimate_max_tokens(targets: list[tuple[str, str]], attempt: int = 1) -> int:
    """Size max_tokens from the text being formatted (see §Request sizing).

    `attempt` is 1-based and counts the original try, so it matches an
    utterance's `attempts` field plus one.  Each attempt past the first scales
    the budget — and the floor with it, since the floor is what actually binds a
    single-utterance retry — so an utterance that truncated once is not sent
    back with the budget that already proved too small.
    """
    payload_chars = sum(
        len(key) + len(text) + JSON_CHARS_PER_ENTRY for key, text in targets
    )
    input_tokens = math.ceil(payload_chars / CHARS_PER_TOKEN)
    budget = int(input_tokens * OUTPUT_TOKEN_MULTIPLIER) + MAX_TOKENS_MARGIN

    growth = RETRY_MAX_TOKENS_GROWTH ** max(0, attempt - 1)
    floor = int(MAX_TOKENS_FLOOR * growth)
    return min(MAX_TOKENS_CEILING, max(floor, int(budget * growth)))


# ══════════════════════════════════════════════════════════════════════════════
# Manifest
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Manifest:
    """The single source of truth for a labelling run.

    Structure:
        config      run parameters, checked for compatibility on re-prepare
        requests    custom_id → {chapter, target_ids, context_ids, state, ...}
        utterances  utt_id    → {unformatted, attempts, formatted, validation, ...}
        batches     batch_id  → {state, request_ids, results_downloaded, ...}

    Written atomically (tmp + rename) after every state change, so a kill at any
    point leaves a loadable manifest.
    """

    out_dir: Path
    data: dict[str, Any]

    @property
    def path(self) -> Path:
        return self.out_dir / "manifest.json"

    @property
    def responses_dir(self) -> Path:
        return self.out_dir / "responses"

    @property
    def config(self) -> dict[str, Any]:
        return self.data["config"]

    def prompt_text(self) -> str:
        """The system prompt, read from the run's own copy (see cmd_prepare).

        Falls back to the original --prompt-file only for manifests prepared
        before the copy existed, and verifies its digest either way: silently
        labelling half a corpus under a different prompt than the manifest
        records would be undetectable downstream.
        """
        local = self.out_dir / "prompt.txt"
        source = local if local.exists() else Path(self.config["prompt_file"])
        if not source.is_file():
            raise SystemExit(
                f"Prompt file missing: neither {local} nor {self.config['prompt_file']} exists."
            )
        text = source.read_text(encoding="utf-8")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != self.config["prompt_sha256"]:
            raise SystemExit(
                f"{source} no longer matches the prompt this run was prepared with "
                f"(sha256 {digest[:12]} vs {self.config['prompt_sha256'][:12]}).\n"
                "Restore it, or start a new run in a fresh --out-dir."
            )
        return text

    @property
    def requests(self) -> dict[str, Any]:
        return self.data["requests"]

    @property
    def utterances(self) -> dict[str, Any]:
        return self.data["utterances"]

    @property
    def batches(self) -> dict[str, Any]:
        return self.data["batches"]

    @classmethod
    def create(cls, out_dir: Path, config: dict[str, Any]) -> "Manifest":
        return cls(
            out_dir=out_dir,
            data={
                "version": MANIFEST_VERSION,
                "config": config,
                "requests": {},
                "utterances": {},
                "batches": {},
            },
        )

    @classmethod
    def load(cls, out_dir: Path) -> "Manifest":
        path = out_dir / "manifest.json"
        if not path.exists():
            raise SystemExit(
                f"No manifest at {path}. Run `prepare --out-dir {out_dir}` first."
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        version = data.get("version")
        if version != MANIFEST_VERSION:
            raise SystemExit(
                f"{path} has manifest version {version}, this tool writes "
                f"version {MANIFEST_VERSION}. Use a fresh --out-dir."
            )
        return cls(out_dir=out_dir, data=data)

    def save(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
# Anthropic client (imported lazily so --self-test needs no SDK)
# ══════════════════════════════════════════════════════════════════════════════

def anthropic_client() -> Any:
    try:
        import anthropic
    except ImportError as exc:                                   # pragma: no cover
        raise SystemExit(
            "The `anthropic` SDK is required for submit/poll/retry.\n"
            "    pip install anthropic"
        ) from exc
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set in the environment.")
    return anthropic.Anthropic()


def batch_request(custom_id: str, model: str, system: str, user: str, max_tokens: int) -> Any:
    """Build one Message Batches request.

    No temperature / thinking / effort parameters: the accepted set varies by
    model and this tool must work unchanged across --model choices.
    """
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    return Request(
        custom_id=custom_id,
        params=MessageCreateParamsNonStreaming(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# prepare
# ══════════════════════════════════════════════════════════════════════════════

def cmd_prepare(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    root = Path(args.librispeech_root)
    prompt_file = Path(args.prompt_file)

    if not root.is_dir():
        raise SystemExit(f"--librispeech-root is not a directory: {root}")
    if not prompt_file.is_file():
        raise SystemExit(f"--prompt-file not found: {prompt_file}")

    config = {
        "librispeech_root":     str(root.resolve()),
        "prompt_file":          str(prompt_file.resolve()),
        "prompt_sha256":        sha256_file(prompt_file),
        "max_utts_per_request": args.max_utts_per_request,
        "chunk_overlap":        CHUNK_OVERLAP,
        "chapters":             args.chapters,
    }

    if (out_dir / "manifest.json").exists():
        manifest = Manifest.load(out_dir)
        mismatched = [
            key for key, value in config.items() if manifest.config.get(key) != value
        ]
        if mismatched:
            raise SystemExit(
                "An existing manifest in this --out-dir was prepared with "
                f"different settings ({', '.join(mismatched)}).\n"
                "Re-preparing would invalidate already-submitted requests. "
                "Use a fresh --out-dir."
            )
        print(f"Resuming existing manifest at {manifest.path}")
    else:
        manifest = Manifest.create(out_dir, config)

    chapters = list(iter_chapters(root))
    if not chapters:
        raise SystemExit(f"No *.trans.txt found under {root}")

    if args.chapters is not None:
        keep = set(select_chapters([chapter for chapter, _ in chapters], args.chapters))
        chapters = [(c, u) for c, u in chapters if c in keep]

    new_requests = 0
    new_utterances = 0
    for chapter, utterances in chapters:
        texts = dict(utterances)
        utt_ids = [key for key, _ in utterances]

        for key, text in utterances:
            if key not in manifest.utterances:
                manifest.utterances[key] = {
                    "chapter":       chapter,
                    "unformatted":   text,
                    "attempts":      0,
                    "formatted":     None,
                    "validation":    "pending",
                    "failure":       None,
                    "numeric_spans": [],
                    "source_request": None,
                }
                new_utterances += 1

        chunks = chunk_chapter(utt_ids, args.max_utts_per_request)
        for index, (targets, context) in enumerate(chunks):
            custom_id = chapter if len(chunks) == 1 else f"{chapter}_c{index}"
            if custom_id in manifest.requests:
                continue
            manifest.requests[custom_id] = {
                "kind":          "initial",
                "chapter":       chapter,
                "chunk":         index,
                "target_ids":    targets,
                "context_ids":   context,
                "max_tokens":    estimate_max_tokens([(k, texts[k]) for k in targets]),
                "state":         "pending",
                "batch_id":      None,
                "model":         None,
                "response_path": None,
                "usage":         None,
                "stop_reason":   None,
                "error":         None,
                "extra_ids":     [],
            }
            for key in targets:
                manifest.utterances[key]["source_request"] = custom_id
            new_requests += 1

    # Copy the prompt into the run directory so the run is self-contained: it
    # gets synced to GCS / a RunPod volume between prepare and submit, where the
    # original --prompt-file path does not exist.
    manifest.responses_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompt.txt").write_text(
        prompt_file.read_text(encoding="utf-8"), encoding="utf-8"
    )
    manifest.responses_dir.mkdir(parents=True, exist_ok=True)
    manifest.save()

    print(
        f"prepare: {len(chapters)} chapters, "
        f"{new_utterances} new utterances ({len(manifest.utterances)} total), "
        f"{new_requests} new requests ({len(manifest.requests)} total)"
    )
    print(f"  manifest: {manifest.path}")
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# submit
# ══════════════════════════════════════════════════════════════════════════════

def _request_payload(manifest: Manifest, custom_id: str) -> tuple[str, int]:
    """Render (user message, max_tokens) for a manifest request record."""
    record = manifest.requests[custom_id]
    utterances = manifest.utterances

    if record["kind"] == "retry":
        # Retry context is other utterances' VALIDATED formatted text, so the
        # model sees the house style rather than raw lowercase input.
        context = [
            (key, utterances[key]["formatted"] or utterances[key]["unformatted"])
            for key in record["context_ids"]
        ]
    else:
        context = [(key, utterances[key]["unformatted"]) for key in record["context_ids"]]

    targets = [(key, utterances[key]["unformatted"]) for key in record["target_ids"]]
    return build_user_message(context, targets), record["max_tokens"]


def submit_pending(manifest: Manifest, model: str, max_requests_per_batch: int) -> list[str]:
    """Create batches from every request in state "pending". Returns batch ids.

    Attempt counters are incremented here, once per target per submitted
    request, so both the initial pass and retries are counted the same way.
    """
    pending = sorted(
        cid for cid, rec in manifest.requests.items() if rec["state"] == "pending"
    )
    if not pending:
        return []

    system = manifest.prompt_text()
    client = anthropic_client()

    batch_ids: list[str] = []
    chunk: list[Any] = []
    chunk_ids: list[str] = []
    chunk_bytes = 0

    def flush() -> None:
        nonlocal chunk, chunk_ids, chunk_bytes
        if not chunk:
            return
        batch = client.messages.batches.create(requests=chunk)
        manifest.batches[batch.id] = {
            "state":              batch.processing_status,
            "model":              model,
            "request_ids":        list(chunk_ids),
            "results_downloaded": False,
        }
        for cid in chunk_ids:
            record = manifest.requests[cid]
            record["state"] = "submitted"
            record["batch_id"] = batch.id
            record["model"] = model
            for key in record["target_ids"]:
                manifest.utterances[key]["attempts"] += 1
        manifest.save()
        batch_ids.append(batch.id)
        print(f"  batch {batch.id}: {len(chunk_ids)} requests ({batch.processing_status})")
        chunk, chunk_ids, chunk_bytes = [], [], 0

    for custom_id in pending:
        user, max_tokens = _request_payload(manifest, custom_id)
        request = batch_request(custom_id, model, system, user, max_tokens)
        size = len(json.dumps(request, default=str).encode("utf-8"))

        if chunk and (
            len(chunk) >= max_requests_per_batch or chunk_bytes + size > MAX_BATCH_BYTES
        ):
            flush()

        chunk.append(request)
        chunk_ids.append(custom_id)
        chunk_bytes += size

    flush()
    return batch_ids


def cmd_submit(args: argparse.Namespace) -> int:
    manifest = Manifest.load(Path(args.out_dir))
    batch_ids = submit_pending(manifest, args.model, args.max_requests_per_batch)
    if not batch_ids:
        print("submit: nothing pending — every request is already submitted.")
        return 0
    print(f"submit: created {len(batch_ids)} batch(es) with model {args.model}")
    print(f"  next: python {Path(sys.argv[0]).name} poll --out-dir {args.out_dir}")
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# poll
# ══════════════════════════════════════════════════════════════════════════════

def _error_dict(error: Any, fallback_type: str) -> dict[str, str]:
    """Flatten a batch result error into {type, message}.

    A failed batch entry carries an error *response envelope*, not the error:
    `{"type": "error", "error": {"type": "invalid_request_error", "message": ...}}`.
    Reading `.type` / `.message` off the outer object yields "error" and "" — so
    the first full-corpus run recorded two failed requests (119 utterances) with
    no trace of why they failed.  Unwrap one level, keeping the envelope as a
    fallback in case a future shape is flat.
    """
    inner = getattr(error, "error", None)
    for candidate in (inner, error):
        message = str(getattr(candidate, "message", "") or "")
        if message:
            return {
                "type": str(getattr(candidate, "type", "") or fallback_type),
                "message": message,
            }
    return {
        "type": str(getattr(inner, "type", None) or getattr(error, "type", None) or fallback_type),
        "message": "",
    }


def _usage_dict(usage: Any) -> dict[str, int]:
    return {
        field: int(getattr(usage, field, 0) or 0)
        for field in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        )
    }


def download_results(manifest: Manifest, client: Any, batch_id: str) -> None:
    """Stream one finished batch's results into the manifest and response files.

    Results arrive in arbitrary order, so everything is keyed by custom_id.
    """
    manifest.responses_dir.mkdir(parents=True, exist_ok=True)
    counts = {"succeeded": 0, "errored": 0, "canceled": 0, "expired": 0}

    for entry in client.messages.batches.results(batch_id):
        custom_id = entry.custom_id
        record = manifest.requests.get(custom_id)
        if record is None:
            print(f"  warning: result for unknown custom_id {custom_id!r}, ignored")
            continue

        result_type = entry.result.type
        counts[result_type] = counts.get(result_type, 0) + 1
        record["state"] = result_type

        if result_type != "succeeded":
            record["error"] = _error_dict(getattr(entry.result, "error", None), result_type)
            # Print the first few verbatim: a run that dies on quota or a bad
            # parameter says so once, here, and nowhere else — `validate` only
            # ever sees these utterances as "no_response".
            if counts[result_type] <= ERRORS_PRINTED_PER_BATCH:
                detail = record["error"]
                print(
                    f"    {custom_id}: {result_type} — "
                    f"{detail['type']}: {detail['message'][:200] or '<no message>'}"
                )
            continue

        message = entry.result.message
        text = "".join(block.text for block in message.content if block.type == "text")
        usage = _usage_dict(message.usage)
        response_path = manifest.responses_dir / f"{custom_id}.json"
        response_path.write_text(
            json.dumps(
                {
                    "custom_id":   custom_id,
                    "batch_id":    batch_id,
                    "model":       message.model,
                    "stop_reason": message.stop_reason,
                    "usage":       usage,
                    "text":        text,
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        record["response_path"] = str(response_path.relative_to(manifest.out_dir))
        record["usage"] = usage
        record["stop_reason"] = message.stop_reason
        record["error"] = None

    manifest.batches[batch_id]["results_downloaded"] = True
    manifest.batches[batch_id]["result_counts"] = counts
    manifest.save()
    summary = ", ".join(f"{k}={v}" for k, v in counts.items() if v)
    print(f"  batch {batch_id}: downloaded ({summary or 'no results'})")


def cmd_poll(args: argparse.Namespace) -> int:
    manifest = Manifest.load(Path(args.out_dir))
    if not manifest.batches:
        print("poll: no batches submitted yet.")
        return 0

    client = anthropic_client()
    try:
        while True:
            outstanding = [
                batch_id
                for batch_id, record in manifest.batches.items()
                if not record.get("results_downloaded")
            ]
            if not outstanding:
                print("poll: all batches terminal and downloaded.")
                print(
                    f"  next: python {Path(sys.argv[0]).name} validate "
                    f"--out-dir {args.out_dir}"
                )
                return 0

            still_running = 0
            for batch_id in outstanding:
                batch = client.messages.batches.retrieve(batch_id)
                manifest.batches[batch_id]["state"] = batch.processing_status
                if batch.processing_status == "ended":
                    download_results(manifest, client, batch_id)
                else:
                    still_running += 1
                    counts = batch.request_counts
                    print(
                        f"  batch {batch_id}: {batch.processing_status} "
                        f"(processing={counts.processing}, succeeded={counts.succeeded}, "
                        f"errored={counts.errored})"
                    )
            manifest.save()

            if still_running == 0:
                continue
            print(f"poll: {still_running} batch(es) still running; sleeping {args.interval}s")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        manifest.save()
        print("\npoll: interrupted — manifest saved, safe to re-run `poll`.")
        return 130


# ══════════════════════════════════════════════════════════════════════════════
# validate
# ══════════════════════════════════════════════════════════════════════════════

def _mark_failed(utterance: dict[str, Any], failure: str, formatted: str | None = None) -> None:
    utterance["validation"] = "failed"
    utterance["failure"] = failure
    if formatted is not None:
        utterance["formatted"] = formatted


def validate_request(manifest: Manifest, custom_id: str) -> None:
    """Validate one request's response, writing per-utterance status."""
    record = manifest.requests[custom_id]
    targets = record["target_ids"]

    # Only requests the API has finished with can be judged; marking an in-flight
    # request's utterances failed would let `retry` resubmit them behind its back.
    if record["state"] in IN_FLIGHT_STATES:
        return
    if record["state"] != "succeeded" or not record["response_path"]:
        for key in targets:
            _mark_failed(manifest.utterances[key], "no_response")
        return

    payload = json.loads((manifest.out_dir / record["response_path"]).read_text(encoding="utf-8"))
    parsed = parse_response(payload["text"])

    if parsed is None:
        # A max_tokens stop means the JSON object was cut off mid-object; that
        # is a budget failure, not a model failure, and it is worth separating
        # in the summary because the retry (one utterance, same floor budget)
        # reliably fixes it.
        failure = "truncated" if payload.get("stop_reason") == "max_tokens" else "unparseable"
        for key in targets:
            _mark_failed(manifest.utterances[key], failure)
        record["extra_ids"] = []
        return

    # §7: unrequested IDs are logged and discarded, not failed. Each target is
    # checked against its own reference, so a stray ID cannot corrupt one
    # without failing that target's own alignment.
    record["extra_ids"] = sorted(set(parsed) - set(targets))

    for key in targets:
        utterance = manifest.utterances[key]
        formatted = parsed.get(key)
        if formatted is None:
            _mark_failed(utterance, "missing")
            continue
        ok, spans = validate_utterance(formatted, utterance["unformatted"])
        utterance["formatted"] = formatted
        utterance["numeric_spans"] = [
            {"digits": digits, "words": words} for digits, words in spans
        ]
        if ok:
            utterance["validation"] = "ok"
            utterance["failure"] = None
        else:
            _mark_failed(utterance, "alignment")


def failure_report(
    manifest: Manifest,
    limit: int,
    vocab_root: str | None = None,
) -> list[str]:
    """Render failing utterances for spot-checking: reference, hypothesis, edit.

    The trimmed divergence is the point of this: most alignment failures are a
    single word ("an" → "and", "up stairs" → "upstairs"), and seeing it next to
    the reference is what tells a genuine model violation apart from a corrupt
    LibriSpeech transcript that no formatting can ever satisfy.
    """
    failed = [
        (key, utterance)
        for key, utterance in sorted(manifest.utterances.items())
        if utterance["validation"] == "failed"
    ]
    if not failed:
        return []

    counts, attestation = attestation_counts(manifest, vocab_root)
    lexicon = build_lexicon(counts, attestation)

    # Classify, then order so the probable-reference-defect pile prints last:
    # it is the one a human actually has to eyeball, so it belongs next to the
    # summary rather than scrolled off the top.
    rows: list[tuple[int, str, dict[str, Any], str, list[str], list[str]]] = []
    for key, utterance in failed:
        if utterance["failure"] != "alignment" or not utterance["formatted"]:
            rows.append((0, key, utterance, "", [], []))
            continue
        ref_diff, hyp_diff = minimal_divergence(
            normalize(utterance["unformatted"]).split(),
            normalize(utterance["formatted"]).split(),
        )
        tag, suspects = classify_alignment_failure(ref_diff, lexicon)
        rows.append((
            2 if tag == TAG_REFERENCE_DEFECT else 1, key, utterance, tag, suspects,
            [" ".join(ref_diff), " ".join(hyp_diff)],
        ))
    rows.sort(key=lambda row: (row[0], row[1]))

    lines = [
        f"  failing utterances ({len(failed)})",
        f"    wordlist:    {lexicon.source}",
        f"    attestation: {lexicon.attestation}",
    ]
    if not lexicon.comprehensive:
        lines.append(
            "    note: no system wordlist found, so the tag rests on corpus "
            "attestation alone. Install one (apt install wamerican) for a"
        )
        lines.append(
            "          sharper tag, and treat a lone rare word as unproven."
        )

    for _, key, utterance, tag, suspects, edit in rows[:limit]:
        label = f"  <{tag}>" if tag else ""
        lines.append(
            f"    {key}  [{utterance['failure']}]  "
            f"attempts={utterance['attempts']}  request={utterance['source_request']}{label}"
        )
        lines.append(f"      ref: {utterance['unformatted']}")
        lines.append(f"      hyp: {utterance['formatted'] or '<no response>'}")
        if edit:
            lines.append(
                f"      edit: {edit[0] or '<nothing>'!r}  ->  {edit[1] or '<nothing>'!r}"
            )
        if suspects:
            lines.append(f"      unrecognised reference token(s): {', '.join(suspects)}")
    if len(failed) > limit:
        lines.append(f"    ... and {len(failed) - limit} more (raise --show-failures)")

    tally: dict[str, int] = {}
    for _, _, _, tag, _, _ in rows:
        tally[tag or "other"] = tally.get(tag or "other", 0) + 1
    lines.append("  failure classification:")
    for tag in (TAG_MODEL_EDIT, TAG_REFERENCE_DEFECT, "other"):
        if tally.get(tag):
            lines.append(f"    {tag:26s} {tally[tag]:6d}")
    return lines


def cmd_validate(args: argparse.Namespace) -> int:
    manifest = Manifest.load(Path(args.out_dir))

    for custom_id in sorted(manifest.requests):
        validate_request(manifest, custom_id)
    manifest.save()

    counts: dict[str, int] = {}
    for utterance in manifest.utterances.values():
        key = utterance["validation"]
        if key == "failed":
            key = f"failed:{utterance['failure']}"
        counts[key] = counts.get(key, 0) + 1

    total = len(manifest.utterances)
    failed = sum(v for k, v in counts.items() if k.startswith("failed"))
    extra = sum(len(r["extra_ids"]) for r in manifest.requests.values())

    print(f"validate: {total} utterances")
    for key in sorted(counts):
        print(f"  {key:24s} {counts[key]:7d}")
    if extra:
        print(f"  {'(extra ids returned)':24s} {extra:7d}  — recorded, not failed")
    print(f"  failure rate: {failed / total:.2%}" if total else "  failure rate: n/a")

    if failed and args.show_failures:
        for line in failure_report(manifest, args.show_failures, args.vocab_root):
            print(line)

    if failed:
        if not args.show_failures:
            print(
                f"  inspect: python {Path(sys.argv[0]).name} validate "
                f"--out-dir {args.out_dir} --show-failures"
            )
        print(
            f"  next: python {Path(sys.argv[0]).name} retry --out-dir {args.out_dir}"
        )
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# retry
# ══════════════════════════════════════════════════════════════════════════════

def valid_neighbours(
    manifest: Manifest,
    chapter_ids: list[str],
    index: int,
    per_side: int = RETRY_CONTEXT_NEIGHBOURS,
) -> list[str]:
    """Up to `per_side` validated utterances on each side of chapter_ids[index].

    Only utterances that passed validation qualify, so the retry never learns
    style from a response that was itself rejected.
    """
    picked: list[str] = []
    for step in (-1, 1):
        found = 0
        cursor = index + step
        while 0 <= cursor < len(chapter_ids) and found < per_side:
            key = chapter_ids[cursor]
            if manifest.utterances[key]["validation"] == "ok":
                picked.append(key)
                found += 1
            cursor += step
    # Chapter order keeps the context readable as running text.
    order = {key: i for i, key in enumerate(chapter_ids)}
    return sorted(picked, key=lambda key: order[key])


@dataclass(frozen=True)
class RetryPlan:
    """What one `retry` pass decided about every still-failing utterance.

    Every failed utterance lands in exactly one bucket, and the three it is NOT
    queued into mean very different things — which is the whole reason they are
    counted separately rather than skipped in silence:

        queued       a fresh request was written to the manifest; `retry`
                     submits these itself
        capped       out of attempts; ships as failed at finalize (terminal)
        unsubmitted  a request exists but never reached the API — a previous
                     `retry` died between the manifest save and the first batch
                     create. Recovered by `submit`, NOT by another `retry`.
        in_flight    a request is at the API and its result is not collected
                     yet. Recovered by `poll` + `validate`.

    The last two look identical from the outside (both "failed", both skipped)
    but are one command away from being retried, so reporting them as "nothing
    to retry" would read as terminal and invite a premature finalize.
    """

    queued: int
    capped: int
    unsubmitted: int
    in_flight: int

    @property
    def recoverable(self) -> int:
        """Failed utterances that still have an unconsumed attempt in the system."""
        return self.unsubmitted + self.in_flight


def build_retry_requests(manifest: Manifest, max_attempts: int) -> RetryPlan:
    """Queue one pending retry request per retryable failed utterance.

    Never resubmits a whole chapter: each request carries exactly one target
    utterance plus its valid neighbours as context.
    """
    by_chapter: dict[str, list[str]] = {}
    for key, utterance in manifest.utterances.items():
        by_chapter.setdefault(utterance["chapter"], []).append(key)
    for keys in by_chapter.values():
        keys.sort()

    # An utterance with a request still in flight must not get a second one.
    # This covers both halves of a re-run: a `retry` interrupted after queuing
    # but before submitting (pending), and a `retry` re-run before `poll` has
    # collected the previous attempt (submitted).  The two are kept apart rather
    # than merged into one skip-set because they need different fixes.
    awaiting: dict[str, str] = {}
    for record in manifest.requests.values():
        if record["state"] not in IN_FLIGHT_STATES:
            continue
        for key in record["target_ids"]:
            # "submitted" wins a tie: an utterance with any request already at
            # the API is waiting on `poll`, whatever else is queued beside it.
            if awaiting.get(key) != "submitted":
                awaiting[key] = record["state"]

    queued = 0
    capped = 0
    unsubmitted = 0
    in_flight = 0
    for chapter, chapter_ids in sorted(by_chapter.items()):
        for index, key in enumerate(chapter_ids):
            utterance = manifest.utterances[key]
            if utterance["validation"] != "failed":
                continue
            # Checked BEFORE the cap: an utterance whose attempt is still in the
            # system has not spent it yet (`attempts` increments at submit), so
            # calling it capped would be wrong as well as unhelpful.
            state = awaiting.get(key)
            if state == "pending":
                unsubmitted += 1
                continue
            if state == "submitted":
                in_flight += 1
                continue
            if utterance["attempts"] >= max_attempts:
                capped += 1
                continue

            context = valid_neighbours(manifest, chapter_ids, index)
            custom_id = f"retry_{key}_a{utterance['attempts'] + 1}"
            manifest.requests[custom_id] = {
                "kind":          "retry",
                "chapter":       chapter,
                "chunk":         0,
                "target_ids":    [key],
                "context_ids":   context,
                "max_tokens":    estimate_max_tokens(
                    [(key, utterance["unformatted"])],
                    attempt=utterance["attempts"] + 1,
                ),
                "state":         "pending",
                "batch_id":      None,
                "model":         None,
                "response_path": None,
                "usage":         None,
                "stop_reason":   None,
                "error":         None,
                "extra_ids":     [],
            }
            utterance["source_request"] = custom_id
            queued += 1

    return RetryPlan(
        queued=queued, capped=capped, unsubmitted=unsubmitted, in_flight=in_flight
    )


def cmd_retry(args: argparse.Namespace) -> int:
    manifest = Manifest.load(Path(args.out_dir))
    tool = Path(sys.argv[0]).name

    plan = build_retry_requests(manifest, args.max_attempts)
    manifest.save()

    if plan.capped:
        print(
            f"retry: {plan.capped} utterance(s) at the {args.max_attempts}-attempt cap "
            "— not resubmitted; they keep their last response and finalize as failed."
        )
    # Reported before the queue result: these utterances are NOT terminal, and a
    # run that stops here would finalize them as failed while a paid-for attempt
    # was still sitting in the manifest unused.
    if plan.unsubmitted:
        print(
            f"retry: {plan.unsubmitted} utterance(s) already have a retry request queued "
            "that was never submitted — a previous `retry` was interrupted between "
            "writing the manifest and reaching the API."
        )
        print("  these do NOT need another retry; send the existing ones:")
        print(f"    python {tool} submit --out-dir {args.out_dir}")
    if plan.in_flight:
        print(
            f"retry: {plan.in_flight} utterance(s) have a retry in flight whose result "
            "has not been collected yet."
        )
        print("  collect it before retrying again:")
        print(f"    python {tool} poll     --out-dir {args.out_dir}")
        print(f"    python {tool} validate --out-dir {args.out_dir}")

    if not plan.queued:
        if not plan.recoverable:
            print("retry: nothing to retry.")
        else:
            print(
                f"retry: queued 0 new request(s) — {plan.recoverable} still-failing "
                "utterance(s) are waiting on the command(s) above, not on a retry."
            )
        return 0

    print(f"retry: queued {plan.queued} single-utterance request(s)")
    model = args.model or _last_model(manifest)
    batch_ids = submit_pending(manifest, model, args.max_requests_per_batch)
    print(f"retry: created {len(batch_ids)} batch(es) with model {model}")
    print(f"  next: python {tool} poll --out-dir {args.out_dir}")
    return 0


def _last_model(manifest: Manifest) -> str:
    """Reuse the model the run has been using, so a retry is comparable."""
    for record in reversed(list(manifest.batches.values())):
        if record.get("model"):
            return str(record["model"])
    return DEFAULT_MODEL


# ══════════════════════════════════════════════════════════════════════════════
# finalize
# ══════════════════════════════════════════════════════════════════════════════

def cost_summary(manifest: Manifest) -> list[str]:
    """Token totals per model plus batch-priced cost where the price is known."""
    per_model: dict[str, dict[str, int]] = {}
    for record in manifest.requests.values():
        usage = record.get("usage")
        if not usage:
            continue
        model = record.get("model") or "unknown"
        totals = per_model.setdefault(
            model,
            {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        )
        totals["requests"] += 1
        for field, value in usage.items():
            totals[field] = totals.get(field, 0) + int(value)

    lines: list[str] = []
    grand_total = 0.0
    priced = True
    for model in sorted(per_model):
        totals = per_model[model]
        lines.append(f"  {model}")
        lines.append(f"    requests        {totals['requests']:12,d}")
        lines.append(f"    input tokens    {totals['input_tokens']:12,d}")
        lines.append(f"    output tokens   {totals['output_tokens']:12,d}")
        if totals["cache_creation_input_tokens"] or totals["cache_read_input_tokens"]:
            lines.append(f"    cache write     {totals['cache_creation_input_tokens']:12,d}")
            lines.append(f"    cache read      {totals['cache_read_input_tokens']:12,d}")

        if model not in PRICES:
            lines.append("    cost            (no list price on file for this model)")
            priced = False
            continue
        in_price, out_price = PRICES[model]
        cost = BATCH_DISCOUNT * (
            in_price * totals["input_tokens"] / 1e6
            + out_price * totals["output_tokens"] / 1e6
            + in_price * CACHE_WRITE_MULT * totals["cache_creation_input_tokens"] / 1e6
            + in_price * CACHE_READ_MULT * totals["cache_read_input_tokens"] / 1e6
        )
        grand_total += cost
        lines.append(f"    cost (batch)    ${cost:11,.4f}")

    if priced and len(per_model) > 1:
        lines.append(f"  total cost        ${grand_total:11,.4f}")
    return lines


def cmd_finalize(args: argparse.Namespace) -> int:
    manifest = Manifest.load(Path(args.out_dir))
    labels_out = Path(args.labels_out)
    labels_out.parent.mkdir(parents=True, exist_ok=True)

    failures_by_chapter: dict[str, int] = {}
    failure_kinds: dict[str, int] = {}
    written = 0
    ok = 0

    with labels_out.open("w", encoding="utf-8") as handle:
        for key in sorted(manifest.utterances):
            utterance = manifest.utterances[key]
            valid = utterance["validation"] == "ok"
            if valid:
                ok += 1
            else:
                failures_by_chapter[utterance["chapter"]] = (
                    failures_by_chapter.get(utterance["chapter"], 0) + 1
                )
                kind = utterance["failure"] or "unvalidated"
                failure_kinds[kind] = failure_kinds.get(kind, 0) + 1

            # A capped or never-answered utterance still ships, marked failed.
            # Falling back to the unformatted text keeps every consumer's
            # `formatted` field a usable string rather than null.
            #
            # NOT YET IMPLEMENTED — FORMATTING_SPEC.md §7 requires confirmed
            # reference defects to be DROPPED from both label files here, since
            # they are the one case where the unformatted and formatted labels
            # disagree on a spoken word. Nothing currently distinguishes a
            # *confirmed* defect from the `probable-reference-defect` the report
            # heuristic tags, so the drop needs that confirmation step first.
            # Until then every utterance ships and the tag is advisory only.
            handle.write(
                json.dumps(
                    {
                        "key":         key,
                        "unformatted": utterance["unformatted"],
                        "formatted":   utterance["formatted"] or utterance["unformatted"],
                        "validation":  "ok" if valid else "failed",
                        "attempts":    utterance["attempts"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    failed = written - ok
    print(f"finalize: wrote {written} records to {labels_out}")
    print(f"  ok            {ok:12,d}")
    print(f"  failed        {failed:12,d}" + (f"   ({failed / written:.2%})" if written else ""))
    for kind in sorted(failure_kinds):
        print(f"    {kind:24s} {failure_kinds[kind]:8,d}")

    if failures_by_chapter:
        print(f"  failures by chapter ({len(failures_by_chapter)} chapters affected):")
        worst = sorted(failures_by_chapter.items(), key=lambda kv: (-kv[1], kv[0]))
        for chapter, count in worst[: args.top_chapters]:
            print(f"    {chapter:24s} {count:8,d}")
        if len(worst) > args.top_chapters:
            print(f"    ... and {len(worst) - args.top_chapters} more")

    print("  token / cost accounting:")
    for line in cost_summary(manifest):
        print(line)
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# Self-tests (CPU-only, API mocked)
# ══════════════════════════════════════════════════════════════════════════════

def _test_normalize() -> None:
    assert normalize("Hello, World!") == "hello world"
    # §6: apostrophes are deleted, so possessives and contractions stay single
    # tokens and the unreliable reference apostrophe never decides a failure.
    assert normalize("He didn't take Lynde's hat.") == "he didnt take lyndes hat"
    # Expanding a contraction changes the token count, so it is still caught.
    assert normalize("He did not take it") == "he did not take it"

    # §5 inverse allowlist, with the period optional on every abbreviation.
    assert normalize("Mrs. Rachel") == "missus rachel"
    assert normalize("Mrs Rachel") == "missus rachel"
    assert normalize("Mr. and Dr. Blair of St. Louis") == \
        "mister and doctor blair of saint louis"
    assert normalize("Mr and Dr Blair of St Louis") == \
        "mister and doctor blair of saint louis"
    # A missing period must not change the verdict either way.
    assert normalize("Mrs Rachel") == normalize("missus rachel")

    # The abbreviations must not fire inside longer words. "st" is the dangerous
    # one: 1,820 "street" and 1,274 "stone" tokens in the corpus.
    assert normalize("down the street past the stone") == "down the street past the stone"
    assert normalize("Storm on Sunday") == "storm on sunday"
    assert normalize("a drab drill") == "a drab drill"
    # "first." must not be read as an "st." abbreviation.
    assert normalize("the first. then") == "the first then"
    # Digits are left alone (§6) and whitespace collapses.
    assert normalize("It was  1895;\n  truly") == "it was 1895 truly"
    # §6 strips all punctuation, so a clock colon goes too and "5:30" becomes
    # two wildcard tokens. Alignment handles that fine (one word each); pinned
    # here because the spec text, not convenience, decides it.
    assert normalize("the 5:30 train") == "the 5 30 train"
    # A forbidden hyphen-join degrades to two tokens rather than fusing.
    assert normalize("cotton-warp") == "cotton warp"
    # A leading quote is deleted like any other apostrophe (§6).
    assert normalize("'tis") == "tis"
    print("  normalize                    ok")


def _test_alignment() -> None:
    ref = "it was eighteen ninety five".split()
    ok, spans = align("it was 1895".split(), ref)
    assert ok and spans == [("1895", ["eighteen", "ninety", "five"])]

    # Full pipeline: casing, punctuation, allowlist, digits, apostrophes.
    ok, spans = validate_utterance(
        "Mrs. Rachel didn't arrive until 5:30 on the sixteenth.",
        "missus rachel didn't arrive until five thirty on the sixteenth",
    )
    assert ok, "spec-conformant formatting must validate"
    assert spans == [("5", ["five"]), ("30", ["thirty"])]
    # ...and the same utterance with the §5 period dropped.
    assert validate_utterance(
        "Mrs Rachel didn't arrive until 5:30 on the sixteenth.",
        "missus rachel didn't arrive until five thirty on the sixteenth",
    )[0], "a missing abbreviation period must not fail an utterance"

    # §4: the number becomes digits and the unit stays an ordinary token.
    assert validate_utterance("It cost 10 pounds.", "it cost ten pounds")[0]

    # CHARACTERISATION, not endorsement: the symbol forms §4 bans are NOT
    # caught here. "£10" normalises to the single token "10", and a §7 wildcard
    # may consume 1-5 reference words, so it silently swallows "ten pounds" —
    # exactly the invariant break §4's RATIONALE predicts. Only the prompt
    # prevents symbol output; the validator is blind to it.
    assert validate_utterance("It cost £10.", "it cost ten pounds")[0]
    assert validate_utterance("It was 50%.", "it was fifty percent")[0]

    # A wildcard may consume one word ("sixteen" → 16).
    ok, _ = validate_utterance("There were 16 of them.", "there were sixteen of them")
    assert ok

    # Word substitution, insertion, and deletion are all rejected (§1).
    assert not validate_utterance("It was not 1895.", "it was eighteen ninety five")[0]
    assert not validate_utterance("He did not take it.", "he didn't take it")[0]
    assert not validate_utterance("He took it.", "he took it quickly")[0]

    # A wildcard must not swallow more than five reference words.
    assert not align(["1"], ["a", "b", "c", "d", "e", "f"])[0]
    assert align(["1"], ["a", "b", "c", "d", "e"])[0]

    # An empty hypothesis fails against a non-empty reference.
    assert not validate_utterance("", "some words here")[0]

    # Real failures seen on dev-clean, kept as regression cases. §1 forbids
    # joining words, changing spelling, adding diacritics, and "correcting"
    # dialect — every one of these must stay a failure.
    for hyp, ref in [
        ("we live it today.", "we live it to day"),                 # join
        ("He remained upstairs.", "he remained up stairs"),         # join
        ("Told them goodbye.", "told them good bye"),               # join
        ("A note from McGregor.", "a note from mc gregor"),         # join
        ("Bill Skelly and his gang.", "bill skelly an his gang"),   # dialect
        ("Families of laborers.", "families of labourers"),         # spelling
        ("I took it to Trübner.", "i took it to trubner"),          # diacritic
    ]:
        assert not validate_utterance(hyp, ref)[0], f"must reject: {hyp!r}"
    print("  alignment                    ok")


def _test_minimal_divergence() -> None:
    # The report must localise the edit, not dump both sentences.
    assert minimal_divergence(
        "we live it to day".split(), "we live it today".split()
    ) == (["to", "day"], ["today"])
    assert minimal_divergence(
        "bill skelly an his gang".split(), "bill skelly and his gang".split()
    ) == (["an"], ["and"])
    # Divergence at the very first token (a corrupt reference word).
    assert minimal_divergence(
        "bui we should".split(), "but we should".split()
    ) == (["bui"], ["but"])
    # Pure insertion: prefix and suffix must not overlap and double-count.
    assert minimal_divergence("a c".split(), "a b c".split()) == ([], ["b"])
    assert minimal_divergence("a b".split(), "a b".split()) == ([], [])
    assert minimal_divergence([], ["x"]) == ([], ["x"])
    print("  minimal_divergence           ok")


def _test_reference_defect_tag() -> None:
    # A corpus where every word recurs except the corrupt one — the shape of
    # real LibriSpeech, where a defect is unique and real words repeat.
    corpus = [
        "can you see the canyon from here",
        "we walked to the canyon and back",
        "it seems to me we live it to day",
        "another day and another walk to the hills",
        "canyou see the canyon",                      # the defective transcript
    ]
    lexicon = build_lexicon(corpus_token_counts(corpus))

    # Known defect: "canyou" is in no wordlist and occurs once in the corpus.
    tag, suspects = classify_alignment_failure(["canyou"], lexicon)
    assert tag == TAG_REFERENCE_DEFECT, tag
    assert suspects == ["canyou"]

    # Known model edit: the reference "to day" is two ordinary words, so the
    # join to "today" is the model's doing, not a bad transcript.
    tag, suspects = classify_alignment_failure(["to", "day"], lexicon)
    assert tag == TAG_MODEL_EDIT, tag
    assert suspects == []

    # Guards on the "not digit-bearing / not §5 allowlist" carve-outs.
    assert classify_alignment_failure(["1895"], lexicon)[0] == TAG_MODEL_EDIT
    assert classify_alignment_failure(["missus"], lexicon)[0] == TAG_MODEL_EDIT
    # normalize() deletes apostrophes, so contractions arrive bare and must not
    # look like defects just because no wordlist spells them that way.
    assert classify_alignment_failure(["didnt"], lexicon)[0] == TAG_MODEL_EDIT
    assert classify_alignment_failure(["oclock"], lexicon)[0] == TAG_MODEL_EDIT
    assert classify_alignment_failure([], lexicon)[0] == TAG_MODEL_EDIT

    # A word absent from the fallback list but attested across the corpus is
    # vouched for by attestation alone — this is what carries the heuristic on
    # machines with no system dictionary.
    assert not FALLBACK_WORDS & {"canyon"}
    assert lexicon.vouches_for("canyon")
    assert not lexicon.vouches_for("canyou")
    print("  reference-defect tag         ok")


def _test_parse_response() -> None:
    assert parse_response('{"a": "x"}') == {"a": "x"}
    assert parse_response('```json\n{"a": "x"}\n```') == {"a": "x"}
    assert parse_response('```\n{"a": "x"}\n```') == {"a": "x"}
    assert parse_response('Here you go:\n{"a": "x"}\nDone.') == {"a": "x"}
    assert parse_response('{"a": "x"') is None            # truncated at max_tokens
    assert parse_response('{"a": 3}') is None             # not a string map
    assert parse_response("not json at all") is None

    # A model that miscounts restarts and emits a second complete object. Taking
    # first-`{`-to-last-`}` spans the prose between them and parses as nothing;
    # the correction is the answer that counts.
    restarted = (
        '```json\n{"a": "first", "b": "first"}\n```\n\n'
        "Wait, let me recount the utterances. Let me restart:\n\n"
        '```json\n{"a": "second", "b": "second"}\n```'
    )
    assert parse_response(restarted) == {"a": "second", "b": "second"}

    # ...and when the do-over is itself cut off at max_tokens, the complete
    # first object is still recovered rather than the whole response being lost.
    assert parse_response(
        '{"a": "first", "b": "first"}\n\nLet me restart:\n\n{"a": "seco'
    ) == {"a": "first", "b": "first"}

    # A brace inside a formatted utterance must not close the object early.
    assert parse_response('{"a": "a } brace", "b": "y"}') == {"a": "a } brace", "b": "y"}
    assert parse_response(r'{"a": "quote \" and } brace"}') == {"a": 'quote " and } brace'}

    # The largest object wins, so a stray fragment cannot displace the real one.
    assert parse_response('{"a": "x", "b": "y"}\n{"a": "z"}') == {"a": "x", "b": "y"}
    print("  parse_response               ok")


def _test_error_dict() -> None:
    """A batch error arrives wrapped in an error-response envelope."""

    class _Obj:
        def __init__(self, **fields: Any) -> None:
            self.__dict__.update(fields)

    envelope = _Obj(type="error", error=_Obj(type="rate_limit_error", message="quota"))
    assert _error_dict(envelope, "errored") == {"type": "rate_limit_error", "message": "quota"}

    # A flat shape (or a future one) still yields something usable.
    assert _error_dict(_Obj(type="api_error", message="boom"), "errored") == {
        "type": "api_error", "message": "boom",
    }
    # No detail at all: fall back to the result type rather than inventing one.
    assert _error_dict(None, "expired") == {"type": "expired", "message": ""}
    # A bare envelope carries nothing but its own type; report that, not "".
    assert _error_dict(_Obj(type="error"), "errored") == {"type": "error", "message": ""}
    print("  batch error unwrapping       ok")


def _test_chunking() -> None:
    ids = [f"u{i}" for i in range(10)]

    # A chapter under the cap is one request with no context.
    assert chunk_chapter(ids, 20) == [(ids, [])]

    chunks = chunk_chapter(ids, 4, overlap=3)
    assert [targets for targets, _ in chunks] == [
        ["u0", "u1", "u2", "u3"],
        ["u4", "u5", "u6", "u7"],
        ["u8", "u9"],
    ], "targets must partition the chapter exactly once"

    # Seam context: 3 utterances on each available side, never a target itself.
    assert chunks[0][1] == ["u4", "u5", "u6"]
    assert chunks[1][1] == ["u1", "u2", "u3", "u8", "u9"]
    assert chunks[2][1] == ["u5", "u6", "u7"]
    for targets, context in chunks:
        assert not set(targets) & set(context)

    # Every utterance is a target in exactly one chunk.
    flat = [key for targets, _ in chunks for key in targets]
    assert flat == ids and len(set(flat)) == len(ids)

    # The rendered request labels context and names the target count.
    message = build_user_message([("u1", "one")], [("u4", "four"), ("u5", "five")])
    assert "CONTEXT ONLY" in message and "u1\tone" in message
    assert "exactly these 2 IDs" in message and "u4\tfour" in message
    # A single-utterance retry request must not read "these 1 IDs".
    assert "exactly these 1 ID," in build_user_message([], [("u4", "four")])

    # Pilot selection is deterministic and spread over sorted order.
    chapters = [f"c{i:02d}" for i in range(10)]
    assert select_chapters(chapters, 3) == ["c00", "c05", "c09"]
    assert select_chapters(chapters, 20) == chapters
    assert select_chapters(chapters, 1) == ["c00"]

    # max_tokens scales with content and respects the floor.
    small = estimate_max_tokens([("u0", "a short line")])
    large = estimate_max_tokens([(f"u{i}", "a much longer line of text " * 6) for i in range(70)])
    assert small == MAX_TOKENS_FLOOR < large <= MAX_TOKENS_CEILING

    # Every attempt past the first buys head-room, floor included — the floor is
    # what binds a single-utterance retry, so leaving it fixed would make the
    # escalation a no-op in exactly the case it exists for.
    assert estimate_max_tokens([("u0", "a short line")], attempt=1) == MAX_TOKENS_FLOOR
    assert estimate_max_tokens([("u0", "a short line")], attempt=2) == 2 * MAX_TOKENS_FLOOR
    assert estimate_max_tokens([("u0", "a short line")], attempt=3) == 4 * MAX_TOKENS_FLOOR
    assert estimate_max_tokens(
        [(f"u{i}", "a much longer line of text " * 6) for i in range(70)], attempt=2
    ) > large
    # ...but never past the ceiling, however many attempts are configured.
    assert estimate_max_tokens(
        [(f"u{i}", "a much longer line of text " * 6) for i in range(70)], attempt=9
    ) == MAX_TOKENS_CEILING
    print("  chunking / sizing            ok")


def _fake_prepared_run(tmp: Path, n_utts: int = 5) -> Manifest:
    """Build a small prepared manifest on disk without touching the API."""
    root = tmp / "LibriSpeech" / "dev-clean" / "1272" / "128104"
    root.mkdir(parents=True)
    (root / "1272-128104.trans.txt").write_text(
        "\n".join(
            f"1272-128104-{i:04d} MISSUS RACHEL DIDN'T ARRIVE UNTIL FIVE THIRTY"
            for i in range(n_utts)
        ),
        encoding="utf-8",
    )
    prompt = tmp / "prompt.txt"
    prompt.write_text("You convert transcripts.", encoding="utf-8")

    out_dir = tmp / "run"
    cmd_prepare(
        argparse.Namespace(
            librispeech_root=str(tmp / "LibriSpeech"),
            out_dir=str(out_dir),
            prompt_file=str(prompt),
            max_utts_per_request=DEFAULT_MAX_UTTS_PER_REQUEST,
            chapters=None,
        )
    )
    return Manifest.load(out_dir)


def _apply_fake_response(manifest: Manifest, custom_id: str, mapping: dict[str, str]) -> None:
    """Stand in for poll(): write a response file and mark the request succeeded."""
    record = manifest.requests[custom_id]
    path = manifest.responses_dir / f"{custom_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "custom_id":   custom_id,
                "stop_reason": "end_turn",
                "usage":       {"input_tokens": 100, "output_tokens": 50},
                "text":        json.dumps(mapping),
            }
        ),
        encoding="utf-8",
    )
    record["state"] = "succeeded"
    record["response_path"] = str(path.relative_to(manifest.out_dir))
    record["usage"] = {"input_tokens": 100, "output_tokens": 50}
    record["stop_reason"] = "end_turn"
    record["model"] = DEFAULT_MODEL
    manifest.save()


def _test_manifest_resume() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        manifest = _fake_prepared_run(tmp, n_utts=5)
        out_dir = manifest.out_dir
        assert len(manifest.utterances) == 5
        assert len(manifest.requests) == 1
        custom_id = next(iter(manifest.requests))

        # Re-running prepare on the same out-dir is a no-op, not a duplicate.
        cmd_prepare(
            argparse.Namespace(
                librispeech_root=str(tmp / "LibriSpeech"),
                out_dir=str(out_dir),
                prompt_file=str(tmp / "prompt.txt"),
                max_utts_per_request=DEFAULT_MAX_UTTS_PER_REQUEST,
                chapters=None,
            )
        )
        reloaded = Manifest.load(out_dir)
        assert len(reloaded.requests) == 1 and len(reloaded.utterances) == 5

        # Prepared with different settings → refuse rather than corrupt state.
        try:
            cmd_prepare(
                argparse.Namespace(
                    librispeech_root=str(tmp / "LibriSpeech"),
                    out_dir=str(out_dir),
                    prompt_file=str(tmp / "prompt.txt"),
                    max_utts_per_request=7,
                    chapters=None,
                )
            )
        except SystemExit:
            pass
        else:
            raise AssertionError("prepare must refuse incompatible settings")

        # The run carries its own prompt copy, so submit works after the run
        # dir is synced to a machine where --prompt-file never existed.
        manifest = Manifest.load(out_dir)
        assert (out_dir / "prompt.txt").exists()
        assert manifest.prompt_text() == "You convert transcripts."
        (tmp / "prompt.txt").unlink()
        assert manifest.prompt_text() == "You convert transcripts.", \
            "prompt must resolve from the run copy, not the original path"

        # A prompt edited underneath a live run is refused, not silently used.
        (out_dir / "prompt.txt").write_text("a different prompt", encoding="utf-8")
        try:
            manifest.prompt_text()
        except SystemExit:
            pass
        else:
            raise AssertionError("prompt_text must reject a changed prompt")
        (out_dir / "prompt.txt").write_text("You convert transcripts.", encoding="utf-8")

        # Simulate: submitted, then killed mid-flight (state on disk only).
        manifest = Manifest.load(out_dir)
        manifest.requests[custom_id]["state"] = "submitted"
        for key in manifest.requests[custom_id]["target_ids"]:
            manifest.utterances[key]["attempts"] += 1
        manifest.save()

        # A fresh process picks the run up exactly where it stopped.
        manifest = Manifest.load(out_dir)
        assert manifest.requests[custom_id]["state"] == "submitted"
        assert all(u["attempts"] == 1 for u in manifest.utterances.values())

        keys = sorted(manifest.utterances)
        good = "Missus Rachel didn't arrive until 5:30."
        _apply_fake_response(
            manifest,
            custom_id,
            {
                keys[0]: good,
                keys[1]: "Missus Rachel did not arrive until 5:30.",   # expanded → fails §1
                keys[2]: good,
                # keys[3] omitted entirely                             → "missing"
                keys[4]: good,
                "1272-128104-9999": "an id we never asked for",        # extra
            },
        )
        cmd_validate(argparse.Namespace(out_dir=str(out_dir), show_failures=0, vocab_root=None))

        manifest = Manifest.load(out_dir)
        assert manifest.utterances[keys[0]]["validation"] == "ok"
        assert manifest.utterances[keys[0]]["numeric_spans"] == [
            {"digits": "5", "words": ["five"]},
            {"digits": "30", "words": ["thirty"]},
        ]
        assert manifest.utterances[keys[1]]["failure"] == "alignment"
        assert manifest.utterances[keys[3]]["failure"] == "missing"
        # Extra ids are recorded but never fail an otherwise-valid target.
        assert manifest.requests[custom_id]["extra_ids"] == ["1272-128104-9999"]
        assert manifest.utterances[keys[4]]["validation"] == "ok"

        # Retry queues one request per failed utterance, with valid neighbours.
        plan = build_retry_requests(manifest, DEFAULT_MAX_ATTEMPTS)
        manifest.save()
        assert (plan.queued, plan.capped) == (2, 0)
        assert plan.recoverable == 0
        retry_ids = [c for c, r in manifest.requests.items() if r["kind"] == "retry"]
        assert len(retry_ids) == 2
        for cid in retry_ids:
            record = manifest.requests[cid]
            assert len(record["target_ids"]) == 1, "never resubmit a whole chapter"
            assert record["context_ids"], "retry must carry neighbour context"
            # Second attempt, so a budget above the first attempt's floor: an
            # utterance that truncated once must not be sent back with the
            # budget that already proved too small.
            assert record["max_tokens"] > MAX_TOKENS_FLOOR, "a retry must buy head-room"
            assert record["max_tokens"] == estimate_max_tokens(
                [(record["target_ids"][0],
                  manifest.utterances[record["target_ids"][0]]["unformatted"])],
                attempt=2,
            )
            assert all(
                manifest.utterances[k]["validation"] == "ok" for k in record["context_ids"]
            ), "only validated neighbours may be used as context"
            assert len(record["context_ids"]) <= 2 * RETRY_CONTEXT_NEIGHBOURS

        # Re-running retry after an interruption must not double-queue — and
        # must SAY that the queued-but-unsent requests exist. Reporting these as
        # "nothing to retry" is what makes an interrupted submit look terminal:
        # the fix is `submit`, and nothing else in the run says so.
        plan = build_retry_requests(manifest, DEFAULT_MAX_ATTEMPTS)
        assert plan.queued == 0
        assert (plan.unsubmitted, plan.in_flight) == (2, 0), plan
        assert plan.capped == 0, "an unsent attempt is not a spent attempt"

        # ...nor once those retries are submitted but not yet polled: the
        # utterances are still marked failed, but their retry is in flight.
        for cid in retry_ids:
            manifest.requests[cid]["state"] = "submitted"
        plan = build_retry_requests(manifest, DEFAULT_MAX_ATTEMPTS)
        assert plan.queued == 0, "must not retry behind an in-flight attempt"
        assert (plan.unsubmitted, plan.in_flight) == (0, 2), plan

        # validate() must not touch an in-flight request's utterances either —
        # a submitted retry is not evidence that anything failed permanently.
        before = {k: dict(v) for k, v in manifest.utterances.items()}
        cmd_validate(argparse.Namespace(out_dir=str(out_dir), show_failures=0, vocab_root=None))
        manifest = Manifest.load(out_dir)
        for cid in retry_ids:
            key = manifest.requests[cid]["target_ids"][0]
            assert manifest.utterances[key]["failure"] == before[key]["failure"], \
                "an in-flight request must not be judged"
    print("  manifest resume / validate   ok")


def _test_retry_cap() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        manifest = _fake_prepared_run(tmp, n_utts=3)
        keys = sorted(manifest.utterances)

        # The original request is terminal — as it would be after poll+validate.
        # (While it is still "pending" its targets are deliberately not retried.)
        for record in manifest.requests.values():
            record["state"] = "succeeded"

        # One utterance has burned every attempt; the other has one left.
        for key in keys:
            manifest.utterances[key]["validation"] = "failed"
            manifest.utterances[key]["failure"] = "alignment"
            manifest.utterances[key]["formatted"] = "He did not take it."
        manifest.utterances[keys[0]]["attempts"] = DEFAULT_MAX_ATTEMPTS
        manifest.utterances[keys[1]]["attempts"] = DEFAULT_MAX_ATTEMPTS - 1
        manifest.utterances[keys[2]]["attempts"] = DEFAULT_MAX_ATTEMPTS + 1  # cannot exceed cap
        manifest.save()

        plan = build_retry_requests(manifest, DEFAULT_MAX_ATTEMPTS)
        manifest.save()
        assert (plan.queued, plan.capped) == (1, 2), plan
        assert plan.recoverable == 0, "a capped utterance is terminal, not recoverable"
        targets = [
            r["target_ids"][0] for r in manifest.requests.values() if r["kind"] == "retry"
        ]
        assert targets == [keys[1]], "only the under-cap utterance may be resubmitted"

        # An at-cap utterance that ALSO has an unsent request must report as
        # recoverable, not capped: `attempts` increments at submit, so that
        # attempt has not been spent and `submit` still has to send it.
        manifest.requests["retry_probe"] = {
            "kind": "retry", "chapter": manifest.utterances[keys[0]]["chapter"],
            "chunk": 0, "target_ids": [keys[0]], "context_ids": [],
            "max_tokens": MAX_TOKENS_FLOOR, "state": "pending", "batch_id": None,
            "model": None, "response_path": None, "usage": None,
            "stop_reason": None, "error": None, "extra_ids": [],
        }
        probed = build_retry_requests(manifest, DEFAULT_MAX_ATTEMPTS)
        # keys[0] via the probe and keys[1] via the request queued just above;
        # only keys[2], with nothing outstanding, is still genuinely capped.
        assert (probed.queued, probed.unsubmitted, probed.capped) == (0, 2, 1), probed

        # Capped utterances still reach the final JSONL, marked failed.
        labels = tmp / "labels.jsonl"
        cmd_finalize(
            argparse.Namespace(
                out_dir=str(manifest.out_dir),
                labels_out=str(labels),
                top_chapters=10,
            )
        )
        records = [json.loads(line) for line in labels.read_text(encoding="utf-8").splitlines()]
        assert len(records) == 3, "every utterance must appear, capped ones included"
        assert {r["key"] for r in records} == set(keys)
        for record in records:
            assert record["validation"] == "failed"
            assert record["formatted"], "formatted must never be empty"
            assert record["attempts"] >= DEFAULT_MAX_ATTEMPTS - 1
        capped_record = next(r for r in records if r["key"] == keys[0])
        assert capped_record["attempts"] == DEFAULT_MAX_ATTEMPTS
        assert capped_record["formatted"] == "He did not take it."
    print("  retry cap / finalize         ok")


def run_self_tests() -> int:
    print("label_formatted self-tests")
    _test_normalize()
    _test_alignment()
    _test_minimal_divergence()
    _test_reference_defect_tag()
    _test_parse_response()
    _test_error_dict()
    _test_chunking()
    _test_manifest_resume()
    _test_retry_cap()
    print("all self-tests passed")
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--self-test", action="store_true", help="run offline self-tests")
    subparsers = parser.add_subparsers(dest="command")

    prepare = subparsers.add_parser("prepare", help="build the request manifest")
    prepare.add_argument("--librispeech-root", required=True,
                         help="directory scanned recursively for *.trans.txt")
    prepare.add_argument("--out-dir", required=True)
    prepare.add_argument("--prompt-file", required=True,
                         help="sent verbatim as the system prompt")
    prepare.add_argument("--chapters", type=int, default=None,
                         help="pilot mode: use N chapters, spread over sorted order")
    prepare.add_argument("--max-utts-per-request", type=int,
                         default=DEFAULT_MAX_UTTS_PER_REQUEST)
    prepare.set_defaults(func=cmd_prepare)

    submit = subparsers.add_parser("submit", help="create batches from pending requests")
    submit.add_argument("--out-dir", required=True)
    submit.add_argument("--model", default=DEFAULT_MODEL)
    submit.add_argument("--max-requests-per-batch", type=int,
                        default=MAX_REQUESTS_PER_BATCH)
    submit.set_defaults(func=cmd_submit)

    poll = subparsers.add_parser("poll", help="download results until all batches end")
    poll.add_argument("--out-dir", required=True)
    poll.add_argument("--interval", type=int, default=DEFAULT_POLL_INTERVAL,
                      help="seconds between status passes")
    poll.set_defaults(func=cmd_poll)

    validate = subparsers.add_parser("validate", help="run the spec check per utterance")
    validate.add_argument("--out-dir", required=True)
    validate.add_argument("--show-failures", type=int, nargs="?", const=20, default=0,
                          metavar="N",
                          help="print up to N failing utterances with the offending "
                               "edit (bare flag = 20)")
    validate.add_argument("--vocab-root", default=None, metavar="DIR",
                          help="LibriSpeech tree scanned to decide which reference "
                               "words are real (default: this run's root). Point at "
                               "the whole corpus for a sharper reference-defect tag")
    validate.set_defaults(func=cmd_validate)

    retry = subparsers.add_parser("retry", help="resubmit only failed utterances")
    retry.add_argument("--out-dir", required=True)
    retry.add_argument("--model", default=None,
                       help="defaults to the model already used by this run")
    retry.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS,
                       help="hard cap, counting the original attempt")
    retry.add_argument("--max-requests-per-batch", type=int,
                       default=MAX_REQUESTS_PER_BATCH)
    retry.set_defaults(func=cmd_retry)

    finalize = subparsers.add_parser("finalize", help="merge to JSONL and summarise")
    finalize.add_argument("--out-dir", required=True)
    finalize.add_argument("--labels-out", required=True)
    finalize.add_argument("--top-chapters", type=int, default=20,
                          help="how many worst chapters to list")
    finalize.set_defaults(func=cmd_finalize)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return run_self_tests()
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

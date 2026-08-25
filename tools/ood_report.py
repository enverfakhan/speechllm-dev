"""Join the OOD decodes into the one paired table: ours vs Whisper-small, per set.

WHY THIS EXISTS
---------------
The out-of-distribution claim this project can defend is not "our WER on
TED-LIUM is X".  X is measured against this project's own references, with its
own un-normalized house convention, on a length-filtered subset — it is not
comparable to any published number and pretending otherwise would be the
headline mistake.  What IS defensible is the Δ against a known system decoded on
the SAME segments under the SAME scoring, reported next to the vocabulary
coverage that bounds how well either system could possibly have done.

This tool produces exactly that table and nothing else.

SCORING, AND WHY THE VERBATIM PAIRING IS NORMALISED ON BOTH SIDES
------------------------------------------------------------------
    formatted pairing   raw hypothesis vs the native cased/punctuated reference,
                        no normalisation — the house convention, and already
                        symmetric: both systems emit cased, punctuated text and
                        both are judged on it.

    verbatim pairing    FORMATTING_SPEC §6 applied to the reference AND to both
                        systems' hypotheses.

That second choice is a deliberate departure from "no normalisation ever", and
the reason is that the house convention is only symmetric in domain.  Whisper
writes cased, punctuated text always; scoring it raw against a verbatim
reference would measure punctuation, not transcription.  Normalising only the
control (and not us) is worse still — it hands the control a free pass on
apostrophes and casing that we pay for.  §6 on both sides of both systems is the
only arrangement under which the Δ means "who got the words right".  §6 is
idempotent on already-normalised text, so this is a no-op wherever it is not
needed.  The un-normalized house numbers are computed too and kept in the JSON
under ``house_convention``.

THE DIGIT CAVEAT, QUANTIFIED RATHER THAN ARGUED
------------------------------------------------
§6 deliberately does NOT expand digit-bearing tokens back to words (the
expansion is ambiguous: 1895 → "eighteen ninety five" | "one thousand eight
hundred ninety five"), so a system that writes "60,000" where the reference says
"sixty thousand" takes three errors it did not earn.  Whisper renders numerals
as digits; a model trained on this project's verbatim convention renders them as
words.  That asymmetry favours US, so it must be surfaced, not buried: every
verbatim row is scored a second time over the digit-free subset, and both
numbers are reported.  Read the digit-free Δ as the honest one.

COVERAGE IS NOT DECORATION
--------------------------
Each row carries the utterance-level reachability from
tools/check_vocab_coverage.py and a WER FLOOR: the share of reference words the
pruned vocabulary cannot represent at all (Decision 005).  Our model cannot beat
that floor however well it hears; the control is unconstrained by it.  A row
whose floor is a large fraction of its Δ is flagged, because that Δ is measuring
the pruning decision, not the model.

USAGE
-----
    python tools/ood_report.py \\
        --ours    tedlium3-test-le41=results/ood/0015040_tedlium3-test-le41.jsonl \\
        --control tedlium3-test-le41=out/ood-tedlium-le41-whisper.jsonl \\
        --ours    commonvoice-en-test=results/ood/0015040_commonvoice-en-test.jsonl \\
        --control commonvoice-en-test=out/ood-commonvoice-whisper.jsonl \\
        --coverage out/ood-vocab-coverage.json \\
        --vocab    data/pruned_tokenizer/ \\
        --out-md out/ood-report.md --out-json out/ood-report.json

``--vocab`` is optional and adds the fully-covered-utterances slice: the same
table re-scored over the utterances whose reference the pruned vocabulary can
actually emit, which is the only subset where our WER is a statement about the
model rather than about Decision 005.

``--ours`` may be omitted while the model decode is still pending; the table is
then a control-only baseline with the model columns left as "—".

    python tools/ood_report.py --self-test
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_formatted import normalize as spec6_normalize  # noqa: E402

MODE_LABEL: dict[str, str] = {"unformatted": "verbatim", "formatted": "formatted"}
_DIGIT_RE = re.compile(r"\d")

# A row is flagged when the pruning floor eats this share of the measured Δ:
# below it, "worse than the control" is partly just "cannot say the word".
_FLOOR_SHARE_FLAG = 0.25


def corpus_wer(pairs: list[tuple[str, str]]) -> float:
    """Corpus-level WER through jiwer — the same engine as utils/evaluate.py.

    Corpus level, not the mean of per-utterance WERs: a mean over utterances
    weights a three-word reference like a thirty-word one.
    """
    import jiwer

    pairs = [(r, h) for r, h in pairs if r.strip()]
    if not pairs:
        return float("nan")
    return jiwer.wer([r for r, _ in pairs], [h for _, h in pairs])


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def keys_for(rows: list[dict], mode: str) -> frozenset[str]:
    """Utterance keys a system actually produced for one mode."""
    return frozenset(
        r["key"] for r in rows if r.get("type") == mode and r.get("key") is not None
    )


def score(
    rows: list[dict], mode: str, normalise: bool,
    exclude_keys: frozenset[str] | None = None,
    only_keys: frozenset[str] | None = None,
) -> tuple[float, int]:
    """Score one system on one mode.

    Args:
        rows:         per-utterance JSONL rows from either system
        mode:         "unformatted" (verbatim) or "formatted"
        normalise:    apply §6 to reference and hypothesis alike
        exclude_keys: utterance keys to skip — used for the digit-free slice,
                      which must exclude the SAME utterances from both systems
                      or the comparison stops being paired
        only_keys:    restrict to these keys — the intersection both systems
                      decoded, so a partial or interrupted decode narrows the
                      comparison instead of silently comparing two corpora

    Returns:
        (wer, n_utterances_scored)
    """
    pairs: list[tuple[str, str]] = []
    for row in rows:
        if row.get("type") != mode:
            continue
        if only_keys is not None and row.get("key") not in only_keys:
            continue
        if exclude_keys and row.get("key") in exclude_keys:
            continue
        ref, hyp = row["reference"], row["hypothesis"]
        if normalise:
            ref, hyp = spec6_normalize(ref), spec6_normalize(hyp)
        pairs.append((ref, hyp))
    return corpus_wer(pairs), len(pairs)


def digit_keys(rows: list[dict], mode: str) -> frozenset[str]:
    """Keys whose reference or hypothesis carries a digit-bearing token.

    Collected per system and UNIONED by the caller: an utterance where either
    system wrote a numeral as digits is unscoreable under §6, which does not
    expand them, so it must leave BOTH sides of the comparison.
    """
    return frozenset(
        row["key"] for row in rows
        if row.get("type") == mode and row.get("key") is not None
        and (_DIGIT_RE.search(row["reference"]) or _DIGIT_RE.search(row["hypothesis"]))
    )


def uncovered_keys(rows: list[dict], mode: str, reach) -> frozenset[str]:
    """Keys whose REFERENCE contains a token the pruned vocabulary cannot emit.

    Unlike digit_keys this is a property of the reference alone, so both systems
    necessarily produce the same set — but it is still collected per system and
    unioned by the caller, for the same reason: an utterance that leaves one side
    of the comparison must leave the other or the pairing breaks.

    Reachability is decided by check_vocab_coverage.FullTokenizer, which OWNS the
    definition — measuring it a second time here would let the subset drift away
    from the coverage percentage printed beside it in the same table.  The test
    is on the whole reference string, matching that tool's n_utterances_full.

    Args:
        rows:  per-utterance JSONL rows from either system
        mode:  "unformatted" | "formatted"
        reach: FullTokenizer over the vocabulary the model was trained on

    Returns:
        frozenset of utterance keys that are NOT fully reachable
    """
    out: set[str] = set()
    memo: dict[str, bool] = {}
    for row in rows:
        if row.get("type") != mode or row.get("key") is None:
            continue
        ref = row["reference"]
        full = memo.get(ref)
        if full is None:
            full = not reach.unreachable(ref)
            memo[ref] = full
        if not full:
            out.add(row["key"])
    return frozenset(out)


def coverage_for(coverage: dict, dataset: str, mode: str) -> dict | None:
    """Find the coverage row for a dataset tag, allowing a shard-variant suffix.

    Decode tags name the shard ("tedlium3-test-le41") while coverage is measured
    once per corpus ("tedlium3-test"), so the longest coverage dataset name that
    prefixes the tag wins.  Longest, not first: "tedlium3-test-le41" must not be
    matched by a shorter unrelated corpus that happens to share a prefix.
    """
    want = MODE_LABEL.get(mode, mode)
    best: dict | None = None
    for row in coverage.get("coverage", []):
        if row["reference_form"] != want:
            continue
        if not dataset.startswith(row["dataset"]):
            continue
        if best is None or len(row["dataset"]) > len(best["dataset"]):
            best = row
    return best


def build_report(
    ours: dict[str, list[dict]],
    control: dict[str, list[dict]],
    coverage: dict,
    reach=None,
) -> list[dict]:
    """One record per (dataset, mode) present in either system's decodes.

    Args:
        reach: optional FullTokenizer; when given, every record also carries a
               "covered_only" slice scored over the fully-reachable utterances
    """
    records: list[dict] = []
    for dataset in sorted(set(ours) | set(control)):
        our_rows  = ours.get(dataset, [])
        ctl_rows  = control.get(dataset, [])
        modes = sorted(
            {r.get("type") for r in our_rows + ctl_rows} & set(MODE_LABEL),
            key=lambda m: list(MODE_LABEL).index(m),
        )
        for mode in modes:
            normalise = mode == "unformatted"
            our_keys, ctl_keys = keys_for(our_rows, mode), keys_for(ctl_rows, mode)

            # When both systems are present the Δ must be paired: score both over
            # the utterances BOTH decoded.  Otherwise an interrupted decode on one
            # side quietly compares two different corpora and calls it a delta.
            common: frozenset[str] | None = None
            if our_keys and ctl_keys and our_keys != ctl_keys:
                common = our_keys & ctl_keys
                print(f"[warn] {dataset}/{MODE_LABEL[mode]}: ours has "
                      f"{len(our_keys):,} utterances, control has {len(ctl_keys):,}; "
                      f"scoring both over the {len(common):,} they share.")
            elif our_keys and ctl_keys:
                common = our_keys

            our_wer, our_n = score(our_rows, mode, normalise, only_keys=common) if our_rows else (None, 0)
            ctl_wer, ctl_n = score(ctl_rows, mode, normalise, only_keys=common) if ctl_rows else (None, 0)
            delta = (our_wer - ctl_wer) if (our_wer is not None and ctl_wer is not None) else None
            if delta is not None and delta != delta:      # NaN: one side had nothing
                delta = None

            rec: dict = {
                "dataset":     dataset,
                "mode":        MODE_LABEL[mode],
                "scoring":     "§6 both sides" if normalise else "un-normalized (house)",
                "n":           our_n or ctl_n,
                "paired":      common is not None,
                "ours_wer":    our_wer,
                "control_wer": ctl_wer,
                "delta":       delta,
            }

            if normalise:
                skip = digit_keys(our_rows, mode) | digit_keys(ctl_rows, mode)
                if common is not None:
                    skip &= common
                our_df, our_dn = score(our_rows, mode, True, skip, common) if our_rows else (None, 0)
                ctl_df, ctl_dn = score(ctl_rows, mode, True, skip, common) if ctl_rows else (None, 0)
                df_delta = ((our_df - ctl_df)
                            if (our_df is not None and ctl_df is not None) else None)
                if df_delta is not None and df_delta != df_delta:
                    df_delta = None
                rec["digit_free"] = {
                    "n":           our_dn or ctl_dn,
                    "n_dropped":   len(skip),
                    "ours_wer":    our_df,
                    "control_wer": ctl_df,
                    "delta":       df_delta,
                }

            # The fully-covered subset: drop every utterance whose reference
            # contains a word the model CANNOT emit.  On those the WER measures
            # the vocabulary, not the model — a perfect transcription still
            # scores an error — so this is the cleanest read of the two systems.
            # It cuts both ways: the control has the full Llama-free vocabulary
            # and pays no such floor, so restricting to covered utterances takes
            # away OUR handicap, and the Δ here is the honest one.
            if reach is not None:
                skip = uncovered_keys(our_rows, mode, reach) | uncovered_keys(ctl_rows, mode, reach)
                if common is not None:
                    skip &= common
                our_cv, our_cn = score(our_rows, mode, normalise, skip, common) if our_rows else (None, 0)
                ctl_cv, ctl_cn = score(ctl_rows, mode, normalise, skip, common) if ctl_rows else (None, 0)
                cv_delta = ((our_cv - ctl_cv)
                            if (our_cv is not None and ctl_cv is not None) else None)
                if cv_delta is not None and cv_delta != cv_delta:
                    cv_delta = None
                rec["covered_only"] = {
                    "n":           our_cn or ctl_cn,
                    "n_dropped":   len(skip),
                    "ours_wer":    our_cv,
                    "control_wer": ctl_cv,
                    "delta":       cv_delta,
                }

            rec["house_convention"] = {
                "note": "no normalisation on either side — comparable in kind to "
                        "the in-domain LibriSpeech tables, not to the control",
                "ours_wer":    score(our_rows, mode, False, only_keys=common)[0] if our_rows else None,
                "control_wer": score(ctl_rows, mode, False, only_keys=common)[0] if ctl_rows else None,
            }

            cov = coverage_for(coverage, dataset, mode)
            if cov:
                floor = (100.0 - cov["words_reachable_pct"]) / 100.0
                rec["coverage"] = {
                    "utterances_fully_reachable_pct": cov["utterances_fully_reachable_pct"],
                    "words_reachable_pct":            cov["words_reachable_pct"],
                    "wer_floor_from_pruning":         round(floor, 5),
                }
                delta = rec["delta"]
                if delta is not None and delta > 0 and floor >= _FLOOR_SHARE_FLAG * delta:
                    rec["flag"] = (
                        f"vocabulary pruning alone guarantees {floor:.2%} WER on this "
                        f"set — {floor / delta:.0%} of the {delta:.2%} gap. The gap "
                        "measures Decision 005 as much as the model."
                    )
            records.append(rec)
    return records


def _fmt_pct(v: float | None) -> str:
    return "—" if v is None or v != v else f"{v:.2%}"


def render_md(records: list[dict], coverage: dict) -> str:
    """Paste-ready Markdown: the headline table, then the caveat tables."""
    out = [
        "## Out-of-distribution WER — paired against Whisper-small",
        "",
        "Every number is measured on this project's own references with its own "
        "scoring, so **only the Δ is meaningful**; the absolute WER is not "
        "comparable to published figures. The control is `openai/whisper-small` "
        "(encoder + decoder), greedy, no LM, decoded on byte-identical segments.",
        "",
        "| dataset | mode | n | coverage (utt) | ours | whisper-small | Δ |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        cov = r.get("coverage") or {}
        cov_s = (f"{cov['utterances_fully_reachable_pct']:.1f}%"
                 if cov else "—")
        delta = r["delta"]
        delta_s = "—" if delta is None else f"{delta:+.2%}"
        out.append(
            f"| {r['dataset']} | {r['mode']} | {r['n']:,} | {cov_s} | "
            f"{_fmt_pct(r['ours_wer'])} | {_fmt_pct(r['control_wer'])} | {delta_s} |"
        )

    out += [
        "",
        "Scoring per mode: verbatim rows apply FORMATTING_SPEC §6 to the reference "
        "and to **both** systems' hypotheses (the only symmetric arrangement — "
        "Whisper always writes cased, punctuated text); formatted rows are "
        "un-normalized on both sides, the house convention.",
        "",
        "### Digit rendering, isolated",
        "",
        "§6 does not expand digit-bearing tokens, so a control that writes "
        "`60,000` where the reference says `sixty thousand` takes three errors it "
        "did not earn — an asymmetry that flatters us. Re-scored over the "
        "digit-free subset:",
        "",
        "| dataset | mode | n (digit-free) | dropped | ours | whisper-small | Δ |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        df = r.get("digit_free")
        if not df:
            continue
        delta = df["delta"]
        out.append(
            f"| {r['dataset']} | {r['mode']} | {df['n']:,} | {df['n_dropped']:,} | "
            f"{_fmt_pct(df['ours_wer'])} | {_fmt_pct(df['control_wer'])} | "
            f"{'—' if delta is None else f'{delta:+.2%}'} |"
        )

    if any(r.get("covered_only") for r in records):
        out += [
            "",
            "### Fully-covered utterances only",
            "",
            "An utterance whose reference contains a word the pruned vocabulary "
            "cannot emit is unscoreable for us at any model quality — a perfect "
            "transcription still takes the error. The control carries no such "
            "floor, so those rows measure Decision 005, not the model. Re-scored "
            "over the utterances whose reference is fully reachable, with the "
            "same utterances excluded from both systems:",
            "",
            "| dataset | mode | n (covered) | dropped | ours | whisper-small | Δ |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for r in records:
            cv = r.get("covered_only")
            if not cv:
                continue
            delta = cv["delta"]
            out.append(
                f"| {r['dataset']} | {r['mode']} | {cv['n']:,} | {cv['n_dropped']:,} | "
                f"{_fmt_pct(cv['ours_wer'])} | {_fmt_pct(cv['control_wer'])} | "
                f"{'—' if delta is None else f'{delta:+.2%}'} |"
            )
        out += [
            "",
            "This slice removes the pruning floor, not the domain shift, and it is "
            "independent of the digit-free slice above — verbatim rows here still "
            "carry the numeral asymmetry. Note also that it is not a random subset: "
            "unreachable words cluster in the long, topical, modern-vocabulary "
            "utterances, so what survives is easier for both systems.",
        ]

    flagged = [r for r in records if r.get("flag")]
    if flagged:
        out += ["", "### ⚠ Coverage makes these rows hard to read", ""]
        for r in flagged:
            out.append(f"- **{r['dataset']} / {r['mode']}** — {r['flag']}")

    out += [
        "",
        "### Un-normalized house scoring (secondary)",
        "",
        "Comparable in kind to the in-domain LibriSpeech tables, NOT between "
        "systems — the control is penalised here for punctuation it was never "
        "asked to drop.",
        "",
        "| dataset | mode | ours | whisper-small |",
        "|---|---|---:|---:|",
    ]
    for r in records:
        h = r["house_convention"]
        out.append(f"| {r['dataset']} | {r['mode']} | {_fmt_pct(h['ours_wer'])} | "
                   f"{_fmt_pct(h['control_wer'])} |")

    if coverage:
        out += ["", f"Vocabulary: `{coverage.get('tokenizer')}` "
                    f"({coverage.get('vocab_size', 0):,} tokens). "
                    "Full coverage breakdown in the coverage report."]
    return "\n".join(out) + "\n"


def _parse_pairs(specs: list[str] | None) -> dict[str, list[dict]]:
    """Parse NAME=PATH.jsonl arguments into {name: rows}."""
    out: dict[str, list[dict]] = {}
    for spec in specs or []:
        name, sep, raw = spec.partition("=")
        if not sep or not name or not raw:
            raise ValueError(f"expected NAME=PATH.jsonl, got {spec!r}")
        path = Path(raw)
        if not path.exists():
            raise ValueError(f"{name}: {path} does not exist")
        out[name] = read_jsonl(path)
    return out


def _self_test() -> None:
    """Scoring symmetry, the digit slice, coverage matching and the flag."""
    ctl = [
        {"type": "unformatted", "key": "1", "reference": "it's sixty thousand",
         "hypothesis": "its 60 000"},
        {"type": "unformatted", "key": "2", "reference": "hello there",
         "hypothesis": "hello there"},
    ]
    ours = [
        {"type": "unformatted", "key": "1", "reference": "it's sixty thousand",
         "hypothesis": "it's sixty thousand"},
        {"type": "unformatted", "key": "2", "reference": "hello there",
         "hypothesis": "hello world"},
    ]

    # §6 on both sides: "it's" vs "its" must NOT be an error.
    w, n = score([ours[0]], "unformatted", normalise=True)
    assert w == 0.0 and n == 1, (w, n)
    w_raw, _ = score([{"type": "unformatted", "reference": "it's a", "hypothesis": "its a"}],
                     "unformatted", normalise=False)
    assert w_raw == 0.5, w_raw
    print("  [OK] §6 on both sides neutralises the apostrophe convention")

    # Digit slice drops exactly the numeral utterance.
    _, n_all = score(ctl, "unformatted", True)
    skip = digit_keys(ours, "unformatted") | digit_keys(ctl, "unformatted")
    assert skip == {"1"}, skip          # only the control wrote "60 000"
    _, n_df_ctl = score(ctl, "unformatted", True, skip)
    _, n_df_our = score(ours, "unformatted", True, skip)
    assert (n_all, n_df_ctl, n_df_our) == (2, 1, 1), (n_all, n_df_ctl, n_df_our)
    print("  [OK] digit-free slice drops the SAME utterances from both systems")

    coverage = {
        "tokenizer": "data/pruned_tokenizer/", "vocab_size": 40034,
        "coverage": [
            {"dataset": "tedlium3-test", "reference_form": "verbatim",
             "utterances_fully_reachable_pct": 80.3, "words_reachable_pct": 98.96},
            {"dataset": "ted", "reference_form": "verbatim",
             "utterances_fully_reachable_pct": 1.0, "words_reachable_pct": 1.0},
        ],
    }
    cov = coverage_for(coverage, "tedlium3-test-le41", "unformatted")
    assert cov is not None and cov["dataset"] == "tedlium3-test", cov
    assert coverage_for(coverage, "tedlium3-test-le41", "formatted") is None
    print("  [OK] coverage_for prefers the longest matching corpus name")

    recs = build_report({"tedlium3-test-le41": ours}, {"tedlium3-test-le41": ctl}, coverage)
    assert len(recs) == 1, recs
    rec = recs[0]
    assert rec["mode"] == "verbatim" and rec["n"] == 2, rec
    # ours: 1 sub over 5 ref words. control: "sixty thousand" -> "60 000", 2 subs.
    assert rec["ours_wer"] == 0.2 and rec["control_wer"] == 0.4, rec
    assert abs(rec["delta"] - (0.2 - 0.4)) < 1e-9, rec
    # Dropping the numeral utterance leaves one pair each and flips the sign:
    # the control's only error was the digit rendering §6 cannot score.
    assert rec["digit_free"]["n"] == 1 and rec["digit_free"]["n_dropped"] == 1, rec
    assert rec["digit_free"]["control_wer"] == 0.0, rec
    assert rec["digit_free"]["delta"] > 0, rec
    assert rec["house_convention"]["ours_wer"] == 0.2, rec
    assert "coverage" in rec and rec["coverage"]["wer_floor_from_pruning"] > 0
    assert "flag" not in rec, "a Δ in our favour must not be flagged"
    print("  [OK] build_report: per-mode records, digit slice, house column")

    # A losing Δ smaller than four times the pruning floor gets flagged.
    worse = [{"type": "unformatted", "key": "2", "reference": "hello there",
              "hypothesis": "goodbye world"}]
    ctl_ok = [{"type": "unformatted", "key": "2", "reference": "hello there",
               "hypothesis": "hello there"}]
    flagged = build_report({"tedlium3-test": worse}, {"tedlium3-test": ctl_ok}, coverage)[0]
    assert "flag" not in flagged, "Δ=100% dwarfs a 1% floor — must not flag"
    # Floor 40% against a 100% gap: pruning accounts for 40% of the difference.
    tiny_gap_cov = {"coverage": [{"dataset": "tedlium3-test", "reference_form": "verbatim",
                                  "utterances_fully_reachable_pct": 20.0,
                                  "words_reachable_pct": 60.0}]}
    flagged2 = build_report({"tedlium3-test": worse}, {"tedlium3-test": ctl_ok},
                            tiny_gap_cov)[0]
    assert "flag" in flagged2, flagged2
    print("  [OK] flag fires when the pruning floor is a large share of the gap")

    # Unequal key sets: the Δ is scored over the intersection, not over two corpora.
    partial = build_report(
        {"d": ours[:1]},                       # only key "1"
        {"d": ctl},                            # keys "1" and "2"
        {},
    )[0]
    assert partial["n"] == 1 and partial["paired"] is True, partial
    assert partial["ours_wer"] == 0.0, partial   # key "1" alone, and we got it right
    # Control-only rows keep delta as None rather than a NaN that renders as "+nan%".
    none_mode = build_report({"d": [dict(ours[0], type="formatted")]}, {"d": ctl}, {})
    fmt_rec = next(r for r in none_mode if r["mode"] == "formatted")
    assert fmt_rec["delta"] is None, fmt_rec
    assert "+nan" not in render_md(none_mode, {}), render_md(none_mode, {})
    print("  [OK] unequal key sets score over the intersection; no NaN deltas render")

    # Fully-covered slice, with a stub reachability oracle: "sixty" is the only
    # unreachable word, so key "1" leaves BOTH systems and key "0" survives.
    class _StubReach:
        def unreachable(self, text: str) -> list[int]:
            return [1] if "sixty" in text else []

    cov_recs = build_report({"tedlium3-test-le41": ours}, {"tedlium3-test-le41": ctl},
                            coverage, _StubReach())
    cv = cov_recs[0]["covered_only"]
    assert cv["n"] == 1 and cv["n_dropped"] == 1, cv
    # The dropped utterance was the control's only error, so the covered-only Δ
    # flips against us — the slice must be able to move the sign either way.
    assert cv["control_wer"] == 0.0 and cv["delta"] > 0, cv
    # Reference-derived, so it must not matter which system is asked.
    assert (uncovered_keys(ours, "unformatted", _StubReach())
            == uncovered_keys(ctl, "unformatted", _StubReach()) == {"1"})
    # The formatted mode gets the slice too (digit_free is verbatim-only).
    fmt = build_report({"d": [dict(r, type="formatted") for r in ours]}, {}, {}, _StubReach())[0]
    assert "covered_only" in fmt and "digit_free" not in fmt, fmt
    assert "Fully-covered utterances only" in render_md(cov_recs, coverage)
    # Without --vocab the section stays out entirely.
    assert "Fully-covered" not in render_md(recs, coverage)
    print("  [OK] covered-only slice: same drops both sides, both modes, opt-in")

    md = render_md(recs, coverage)
    for needle in ("Whisper-small", "tedlium3-test-le41", "verbatim", "Digit rendering"):
        assert needle in md, needle
    # Control-only mode: no --ours yet.
    ctl_only = build_report({}, {"tedlium3-test-le41": ctl}, coverage)[0]
    assert ctl_only["ours_wer"] is None and ctl_only["delta"] is None, ctl_only
    assert "—" in render_md([ctl_only], coverage)
    print("  [OK] render_md, and a control-only table while the model decode is pending")
    print("PASSED")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ours", nargs="+", default=None, metavar="NAME=PATH",
                   help="Our model's per-utterance JSONL per dataset; repeatable.")
    p.add_argument("--control", nargs="+", default=None, metavar="NAME=PATH",
                   help="Whisper-small control JSONL per dataset; repeatable.")
    p.add_argument("--coverage", type=Path, default=None,
                   help="tools/check_vocab_coverage.py --out-json output.")
    p.add_argument("--vocab", type=Path, default=None,
                   help="Pruned tokenizer dir (data/pruned_tokenizer/). When given, "
                        "the report adds a WER slice over fully-reachable utterances.")
    p.add_argument("--out-md", type=Path, default=None, dest="out_md")
    p.add_argument("--out-json", type=Path, default=None, dest="out_json")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)

    if args.self_test:
        _self_test()
        return 0
    if not args.control and not args.ours:
        p.error("pass --ours and/or --control")

    ours     = _parse_pairs(args.ours)
    control  = _parse_pairs(args.control)
    coverage = json.loads(args.coverage.read_text()) if args.coverage else {}

    reach = None
    if args.vocab:
        from check_vocab_coverage import FullTokenizer
        reach = FullTokenizer(args.vocab)

    records = build_report(ours, control, coverage, reach)
    md = render_md(records, coverage)
    print(md)

    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md)
        print(f"wrote {args.out_md}")
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(
            {"coverage_source": str(args.coverage), "rows": records}, indent=2))
        print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

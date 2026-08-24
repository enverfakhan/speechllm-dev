"""Vocabulary reachability of a reference set under the pruned Llama vocabulary.

WHY THIS RUNS BEFORE ANY DECODING
---------------------------------
Decision 005 prunes the Llama tokenizer to the ~40k ids that appear in the
LibriSpeech labels.  Everything outside that set is unreachable: no logit row
exists for it, so the model CANNOT emit it however good its audio understanding
is.  In-domain that is free — the vocabulary was built from those very labels.
Out of distribution it is not, and an OOD WER that is really a pruning artefact
would be read as a generalisation failure.

So: measure it first, report it beside every WER, and refuse to interpret a WER
whose coverage number is bad.  An unreachable word is a guaranteed error that
measures the PRUNING DECISION, not the model.

WHAT "REACHABLE" MEANS HERE
---------------------------
The full (unpruned) tokenizer encodes the text; a token is reachable when its
original id survives in ``vocab_map.json``.  Two granularities, because they
answer different questions:

  utterance : the whole reference string is encoded ONCE, in context, and is
              "fully reachable" when every one of its ids survives.  This is
              exactly what the model faces, so it is the headline number.
  word      : each whitespace word is encoded as " word" — with the leading
              space, because Llama BPE makes " the" and "the" different tokens
              and the space-prefixed form is what a word inside a sentence
              actually becomes.  This granularity exists to NAME the offenders.

Note that PrunedTokenizer.encode silently DROPS unmapped ids (Decision 005), so
an unreachable word does not raise anywhere — it just quietly vanishes from a
reference or a decoded hypothesis.  That silence is the reason this tool exists.

USAGE
-----
    python tools/check_vocab_coverage.py \\
        --manifest data/ood_shards/tedlium3-test/manifest.jsonl \\
        --manifest data/ood_shards/commonvoice-en-test/manifest.jsonl \\
        --manifest data/eval_shards/manifest.jsonl --split dev-clean \\
        --tokenizer data/pruned_tokenizer/ \\
        --out-json out/ood-vocab-coverage.json \\
        --out-md   out/ood-vocab-coverage.md

    python tools/check_vocab_coverage.py --self-test

``--split`` restricts records of a manifest that carries a ``split`` field (the
LibriSpeech eval manifest holds all four splits in one file); it is applied to
every manifest that has the field, and ignored by those that do not.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The two reference forms a manifest may carry, in report order.
REF_FORMS: tuple[str, ...] = ("unformatted", "formatted")

# Human names, so the report does not say "unformatted" where the protocol says
# "verbatim".
FORM_LABEL: dict[str, str] = {"unformatted": "verbatim", "formatted": "formatted"}


class FullTokenizer:
    """The unpruned tokenizer plus the pruned id set, for reachability tests.

    PrunedTokenizer cannot answer this question: its encode() already dropped
    the unmapped ids by the time a caller sees the output.  This one keeps them.
    """

    def __init__(self, tokenizer_path: Path) -> None:
        from transformers import AutoTokenizer

        self.path = Path(tokenizer_path)
        self._tok = AutoTokenizer.from_pretrained(str(self.path))
        with (self.path / "vocab_map.json").open(encoding="utf-8") as f:
            raw = json.load(f)
        self.kept_ids: frozenset[int] = frozenset(int(k) for k in raw)
        with (self.path / "pruned_config.json").open(encoding="utf-8") as f:
            self.pruned_config = json.load(f)

    def encode(self, text: str) -> list[int]:
        """Original-space token ids for text (no specials, nothing dropped)."""
        return self._tok.encode(text, add_special_tokens=False)

    def unreachable(self, text: str) -> list[int]:
        """Original ids in text that the pruned vocabulary does not contain."""
        return [i for i in self.encode(text) if i not in self.kept_ids]

    def decode_ids(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    def ceiling(self, text: str) -> str:
        """The best output the pruned vocabulary can represent for this text.

        Encodes with the full tokenizer, drops every id the pruned vocabulary
        lacks — exactly what PrunedTokenizer.encode does silently — and decodes
        what survives.  A perfect model, hearing perfectly, cannot beat this
        string, which turns an abstract coverage percentage into the concrete
        sentence the model is allowed to say.
        """
        return self._tok.decode([i for i in self.encode(text) if i in self.kept_ids])


@dataclass
class Coverage:
    """Reachability counters for one (dataset, reference form)."""

    dataset: str
    form: str
    n_utterances: int = 0
    n_utterances_full: int = 0
    n_words: int = 0
    n_words_reachable: int = 0
    n_tokens: int = 0
    n_tokens_reachable: int = 0
    unreachable_words: Counter = field(default_factory=Counter)

    @property
    def utterance_pct(self) -> float:
        return 100.0 * self.n_utterances_full / self.n_utterances if self.n_utterances else 0.0

    @property
    def word_pct(self) -> float:
        return 100.0 * self.n_words_reachable / self.n_words if self.n_words else 0.0

    @property
    def token_pct(self) -> float:
        return 100.0 * self.n_tokens_reachable / self.n_tokens if self.n_tokens else 0.0

    def to_dict(self, top_n: int = 20) -> dict:
        return {
            "dataset":              self.dataset,
            "reference_form":       FORM_LABEL.get(self.form, self.form),
            "n_utterances":         self.n_utterances,
            "utterances_fully_reachable_pct": round(self.utterance_pct, 3),
            "n_words":              self.n_words,
            "words_reachable_pct":  round(self.word_pct, 3),
            "n_tokens":             self.n_tokens,
            "tokens_reachable_pct": round(self.token_pct, 3),
            "n_distinct_unreachable_words": len(self.unreachable_words),
            "top_unreachable_words": [
                {"word": w, "count": c} for w, c in self.unreachable_words.most_common(top_n)
            ],
        }


def measure(
    texts: list[str], tok: FullTokenizer, dataset: str, form: str,
    word_cache: dict[str, bool] | None = None,
) -> Coverage:
    """Count utterance-, word- and token-level reachability over texts.

    Args:
        texts:      reference strings for one dataset and one reference form
        tok:        FullTokenizer over the vocabulary being judged
        dataset:    dataset name for the report row
        form:       reference form key ("unformatted" | "formatted")
        word_cache: shared word → reachable memo; words repeat heavily across a
                    corpus and each miss is a tokenizer call

    Returns:
        Coverage
    """
    cov = Coverage(dataset=dataset, form=form)
    cache = word_cache if word_cache is not None else {}

    for text in texts:
        ids = tok.encode(text)
        n_bad = sum(1 for i in ids if i not in tok.kept_ids)
        cov.n_utterances += 1
        cov.n_utterances_full += (n_bad == 0)
        cov.n_tokens += len(ids)
        cov.n_tokens_reachable += len(ids) - n_bad

        for word in text.split():
            reachable = cache.get(word)
            if reachable is None:
                # Leading space: " the" and "the" are different tokens in Llama
                # BPE, and the space-prefixed form is what a word in a sentence
                # actually becomes.
                reachable = not tok.unreachable(" " + word)
                cache[word] = reachable
            cov.n_words += 1
            if reachable:
                cov.n_words_reachable += 1
            else:
                cov.unreachable_words[word] += 1

    return cov


def load_manifest(path: Path, split: str | None) -> tuple[str, dict[str, list[str]]]:
    """Read a manifest and group its reference strings by form.

    Args:
        path:  manifest.jsonl written by a prepare_ood_* tool or preprocess.py
        split: when given, keep only records whose "split" field matches; a
               manifest without that field is unaffected

    Returns:
        (dataset_name, {form: [text, ...]})
    """
    names: Counter = Counter()
    texts: dict[str, list[str]] = {f: [] for f in REF_FORMS}
    has_formatted_ref = True

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if split is not None and "split" in rec and rec["split"] != split:
                continue
            names[rec.get("dataset") or rec.get("split") or path.parent.name] += 1
            # A set with no genuine formatted reference writes a COPY of the
            # verbatim text into formatted.txt so the shard is well formed
            # (see tools/prepare_ood_tedlium.py). Scoring that copy, or
            # reporting coverage for it, would invent a reference form.
            if rec.get("has_formatted_ref") is False:
                has_formatted_ref = False
            for form in REF_FORMS:
                if rec.get(form):
                    texts[form].append(rec[form])

    if not has_formatted_ref:
        texts["formatted"] = []

    dataset = names.most_common(1)[0][0] if names else path.parent.name
    return dataset, {f: t for f, t in texts.items() if t}


def ceiling_examples(
    texts: list[str], tok: FullTokenizer, dataset: str, form: str, limit: int,
) -> list[dict]:
    """Sample references the pruned vocabulary cannot represent, with the ceiling.

    Ordered by how much the vocabulary eats (most-damaged first), because those
    are the ones a reader needs to see before trusting a WER on this set.
    """
    rows: list[dict] = []
    for text in texts:
        ceiling = tok.ceiling(text)
        if ceiling.strip() == text.strip():
            continue
        lost = len(text.split()) - len(ceiling.split())
        rows.append({
            "dataset":        dataset,
            "reference_form": FORM_LABEL.get(form, form),
            "reference":      text,
            "vocabulary_ceiling": ceiling.strip(),
            "words_lost":     lost,
        })
    rows.sort(key=lambda r: -r["words_lost"])
    return rows[:limit]


def load_shard(path: Path) -> tuple[str, dict[str, list[str]]]:
    """Read reference strings straight out of an eval shard .tar.

    The manifest is the better input — it carries has_formatted_ref, so a set
    whose formatted.txt is a placeholder copy is not reported as having two
    reference forms.  This path exists for shards whose manifest was overwritten
    or never kept (data/eval_subset_shards/, data/eval_shards/), where the tar
    is the only surviving record of what a WER run actually scored.

    Args:
        path: shard .tar in the house format

    Returns:
        (dataset_name from the tar stem, {form: [text, ...]})
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ood_shard import read_shard

    groups = read_shard(path)
    texts: dict[str, list[str]] = {f: [] for f in REF_FORMS}
    for key in sorted(groups):
        for form in REF_FORMS:
            blob = groups[key].get(f"{form}.txt")
            if blob:
                texts[form].append(blob.decode("utf-8"))
    return path.stem, {f: t for f, t in texts.items() if t}


def render_md(rows: list[dict], tokenizer_path: str, vocab_size: int) -> str:
    """Paste-ready Markdown for the report."""
    out = [
        "### Vocabulary coverage",
        "",
        f"Pruned vocabulary: `{tokenizer_path}` ({vocab_size:,} tokens).",
        "A word outside it cannot be emitted at all, so it is a guaranteed error "
        "that measures the pruning decision (Decision 005), not the model.",
        "",
        "The **utterance** column is the headline: it encodes each reference once, "
        "in context, exactly as the model faces it. The **word** column probes each "
        "word as `\" word\"` to name the offenders, which can flag a word that is "
        "reachable in its real position (a sentence-initial `Forgotten` has no "
        "leading space), so it reads slightly pessimistic by design.",
        "",
        "| dataset | ref form | n utt | utt fully reachable | words reachable | tokens reachable |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        out.append(
            f"| {r['dataset']} | {r['reference_form']} | {r['n_utterances']:,} | "
            f"{r['utterances_fully_reachable_pct']:.2f}% | "
            f"{r['words_reachable_pct']:.3f}% | {r['tokens_reachable_pct']:.3f}% |"
        )
    out.append("")
    for r in rows:
        if not r["top_unreachable_words"]:
            continue
        top = ", ".join(f"`{d['word']}` ×{d['count']}" for d in r["top_unreachable_words"])
        out.append(f"- **{r['dataset']} / {r['reference_form']}** — "
                   f"{r['n_distinct_unreachable_words']:,} distinct unreachable words. "
                   f"Top: {top}")
    return "\n".join(out) + "\n"


def _self_test() -> None:
    """Coverage arithmetic against a stub tokenizer — no real vocabulary needed."""
    import tempfile

    class StubTok:
        """Character-id tokenizer; ids for the letters in BANNED are 'pruned'."""

        BANNED = set("qz")

        def __init__(self) -> None:
            self.kept_ids = frozenset(
                ord(c) for c in "abcdefghijklmnopqrstuvwxyz " if c not in self.BANNED
            )

        def encode(self, text: str) -> list[int]:
            return [ord(c) for c in text]

        def unreachable(self, text: str) -> list[int]:
            return [i for i in self.encode(text) if i not in self.kept_ids]

    tok = StubTok()
    cov = measure(["the cat", "the quiz", "a zebra"], tok, "stub", "unformatted")
    assert cov.n_utterances == 3 and cov.n_utterances_full == 1, (
        cov.n_utterances_full, "only 'the cat' is fully reachable"
    )
    assert cov.n_words == 6 and cov.n_words_reachable == 4, (cov.n_words_reachable,)
    assert dict(cov.unreachable_words) == {"quiz": 1, "zebra": 1}, cov.unreachable_words
    assert abs(cov.utterance_pct - 100 / 3) < 1e-6
    assert cov.n_tokens == len("the cat") + len("the quiz") + len("a zebra")
    assert cov.n_tokens_reachable == cov.n_tokens - 3, "q+z in 'quiz', z in 'zebra'"
    print("  [OK] measure: utterance / word / token counts and offender list")

    class CeilingStub(StubTok):
        def ceiling(self, text: str) -> str:
            return "".join(c for c in text if ord(c) in self.kept_ids)

    ex = ceiling_examples(["the cat", "the quiz", "a zz kills"],
                          CeilingStub(), "stub", "unformatted", limit=5)
    assert len(ex) == 2, ex
    # "zz" vanishes entirely (one word lost); "quiz" only loses two letters.
    assert ex[0]["reference"] == "a zz kills", ex[0]
    assert ex[0]["vocabulary_ceiling"] == "a  kills", ex[0]
    assert ex[0]["words_lost"] == 1 and ex[1]["words_lost"] == 0, ex
    assert all(e["reference_form"] == "verbatim" for e in ex), ex
    assert ceiling_examples(["the cat"], CeilingStub(), "s", "unformatted", 5) == []
    print("  [OK] ceiling_examples: only damaged references, worst first")

    cache: dict[str, bool] = {}
    measure(["the cat"], tok, "s", "unformatted", cache)
    assert cache == {"the": True, "cat": True}, cache
    print("  [OK] word cache memoises reachability")

    with tempfile.TemporaryDirectory() as tmp:
        # TED-style manifest: has_formatted_ref False → the copied formatted
        # text must NOT appear as a second reference form.
        p = Path(tmp) / "manifest.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in [
            {"key": "a", "dataset": "ted", "unformatted": "hello there",
             "formatted": "hello there", "has_formatted_ref": False},
            {"key": "b", "dataset": "ted", "unformatted": "bye", "formatted": "bye",
             "has_formatted_ref": False},
        ]))
        name, texts = load_manifest(p, None)
        assert name == "ted" and set(texts) == {"unformatted"}, (name, texts.keys())

        # LibriSpeech-style manifest: split filter, both forms real.
        p2 = Path(tmp) / "ls.jsonl"
        p2.write_text("\n".join(json.dumps(r) for r in [
            {"key": "a", "split": "dev-clean", "unformatted": "one", "formatted": "One."},
            {"key": "b", "split": "dev-other", "unformatted": "two", "formatted": "Two."},
        ]))
        name2, texts2 = load_manifest(p2, "dev-clean")
        assert name2 == "dev-clean", name2
        assert texts2 == {"unformatted": ["one"], "formatted": ["One."]}, texts2
        print("  [OK] load_manifest: split filter and has_formatted_ref suppression")

    with tempfile.TemporaryDirectory() as tmp:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from ood_shard import ShardWriter
        import numpy as _np

        shard = Path(tmp) / "dev-clean-480.tar"
        w = ShardWriter(shard)
        w.add_sample("k-1", _np.zeros((80, 8), _np.float16), "hello there", "Hello there.")
        w.add_sample("k-2", _np.zeros((80, 8), _np.float16), "bye", "Bye.")
        w.close()
        name3, texts3 = load_shard(shard)
        assert name3 == "dev-clean-480", name3
        assert texts3 == {"unformatted": ["hello there", "bye"],
                          "formatted": ["Hello there.", "Bye."]}, texts3
        print("  [OK] load_shard reads both reference forms out of a tar")

    md = render_md([cov.to_dict()], "stub/", 26)
    assert "quiz" in md and "stub" in md
    print("  [OK] render_md")
    print("PASSED")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--manifest", type=Path, action="append", default=None, metavar="PATH",
                   help="manifest.jsonl to measure; repeatable.")
    p.add_argument("--shard", type=Path, action="append", default=None, metavar="PATH",
                   help="Eval shard .tar to measure directly; repeatable.  Use "
                        "--manifest instead where one exists: a shard cannot say "
                        "whether its formatted.txt is a real reference or a placeholder.")
    p.add_argument("--tokenizer", type=Path, default=Path("data/pruned_tokenizer/"))
    p.add_argument("--split", type=str, default=None,
                   help="Restrict manifests that carry a 'split' field to this split.")
    p.add_argument("--top", type=int, default=20, help="Unreachable words to list (default 20).")
    p.add_argument("--out-json", type=Path, default=None, dest="out_json")
    p.add_argument("--out-examples", type=Path, default=None, dest="out_examples",
                   metavar="PATH",
                   help="Write reference/vocabulary-ceiling pairs for the "
                        "utterances the pruned vocabulary cannot represent.")
    p.add_argument("--n-examples", type=int, default=10, dest="n_examples")
    p.add_argument("--out-md", type=Path, default=None, dest="out_md")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)

    if args.self_test:
        _self_test()
        return 0
    if not args.manifest and not args.shard:
        p.error("pass --manifest and/or --shard (both repeatable) unless --self-test")

    tok = FullTokenizer(args.tokenizer)
    vocab_size = int(tok.pruned_config["vocab_size"])
    print(f"Vocabulary: {args.tokenizer}  ({vocab_size:,} pruned tokens, "
          f"{len(tok.kept_ids):,} original ids kept)\n")

    sources: list[tuple[Path, str]] = (
        [(m, "manifest") for m in (args.manifest or [])]
        + [(s, "shard") for s in (args.shard or [])]
    )

    rows: list[dict] = []
    examples: list[dict] = []
    for source_path, kind in sources:
        dataset, texts = (
            load_manifest(source_path, args.split) if kind == "manifest"
            else load_shard(source_path)
        )
        if not texts:
            print(f"[warn] {source_path} yielded no references"
                  + (f" for split {args.split}" if args.split else "")
                  + " — skipping")
            continue
        cache: dict[str, bool] = {}
        for form in REF_FORMS:
            if form not in texts:
                continue
            cov = measure(texts[form], tok, dataset, form, cache)
            rows.append(cov.to_dict(args.top))
            if args.out_examples:
                examples.extend(
                    ceiling_examples(texts[form], tok, dataset, form, args.n_examples)
                )
            print(f"{dataset:<24} {FORM_LABEL[form]:<11} "
                  f"utt {cov.utterance_pct:6.2f}%   "
                  f"words {cov.word_pct:7.3f}%   tokens {cov.token_pct:7.3f}%   "
                  f"({cov.n_utterances:,} utt)")
            if cov.unreachable_words:
                top = ", ".join(f"{w}×{c}" for w, c in cov.unreachable_words.most_common(8))
                print(f"{'':24} unreachable: {top}")

    report = {
        "tokenizer":  str(args.tokenizer),
        "vocab_size": vocab_size,
        "split_filter": args.split,
        "sources":    [str(p) for p, _ in sources],
        "coverage":   rows,
    }
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out_json}")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_md(rows, str(args.tokenizer), vocab_size))
        print(f"wrote {args.out_md}")
    if args.out_examples:
        args.out_examples.parent.mkdir(parents=True, exist_ok=True)
        args.out_examples.write_text(json.dumps(examples, indent=2))
        print(f"wrote {args.out_examples}  ({len(examples)} ceiling examples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Prune the Llama tokenizer to the vocabulary needed for LibriSpeech training.

Scans raw transcript files from all LibriSpeech splits directly — no preprocessed
shards or manifest required. This lets vocabulary building run in parallel with
(or before) shard preprocessing, and guarantees the pruned vocabulary covers the
full 960h corpus regardless of which splits have been processed.

Two sources of label text, and they must match whatever the shards actually hold:

  --labels_file  read the label text straight out of a labels JSONL
                 (tools/label_formatted.py finalize, or tools/precompute_labels.py).
                 Records marked "validation": "failed" are skipped, exactly as
                 tools/preprocess.py and tools/relabel_shards.py skip them, so the
                 vocabulary covers the shipped corpus and nothing else.

  --librispeech_dir  re-derive the labels by running the same normalisation
                 pipeline tools/preprocess.py uses when it has no labels file:
                   Unformatted:  BasicTextNormalizer
                   Formatted:    EnglishTextNormalizer
                                 → deepmultilingualpunctuation PunctuationModel
                                 → _capitalize_sentences  (sentence-initial caps)
                                 → spaCy en_core_web_sm PROPN tagger

USE THE SAME SOURCE THE SHARDS WERE BUILT FROM. PrunedTokenizer.encode silently
DROPS ids missing from the map (data.py), so a vocabulary built from a different
labelling pipeline does not fail — it quietly deletes tokens from every affected
label. Rebuilding from --librispeech_dir against LLM-written labels would lose,
for example, every apostrophe-carrying token ("'s", "'t", "'ll") and every ';',
because the old BasicTextNormalizer stripped them.

PunctuationModel inference is batched across texts for efficiency: all transcripts
in a chunk are joined into a list and processed in a single pipeline call rather
than one text at a time. This reduces vocab-build time from ~40 min to ~3 min.

Steps:
  1. Collect every label string (from --labels_file, or by scanning and
     normalising all *.trans.txt under --librispeech_dir).
  2. Tokenize all resulting strings; collect the union of token IDs.
  3. Tokenize both instruction prompt strings.
  4. Tokenize the chat scaffold words ("user", "assistant", "\n\n", "\n") and
     FORCE IN the four Llama 3.1 chat specials by their canonical ids
     (<|begin_of_text|> 128000, <|start_header_id|> 128006,
      <|end_header_id|> 128007, <|eot_id|> 128009).
  5. Verify the assembled chat scaffold against the tokenizer's own
     apply_chat_template rendering (see _verify_chat_scaffold).
  6. Save vocab_map.json, pruned_config.json, and tokenizer files to --output_dir.

Why the specials are forced in by ID rather than looked up by name: a Meta-format
Llama directory loads through transformers' TikToken fallback, which exposes NO
special tokens at all (all_special_ids == [], convert_tokens_to_ids returns None,
len(tokenizer) == 128000).  The previous version of this tool scanned downward
from vocab_size-1 for an unused id and landed on 127999 — the ordinary CJK token
"锦" — which then served as the SEP/EOS target for the whole base-model
experiment line.  <|eot_id|> carries a real termination prior from Instruct
pretraining; an arbitrary CJK token carries none.

Rebuilding the vocabulary REASSIGNS every id, so `vocab_map.json` and any
checkpoint carrying a trained `llama` state_dict must be regenerated together:
embedding rows are indexed by the new id. Checkpoints that never trained llama
are unaffected — build.py rebuilds their embedding from the pretrained Llama
weights through vocab_map on every load.

Output directory layout:
    output_dir/
        vocab_map.json       — {old_str_id: new_int_id, ...}
        pruned_config.json   — {vocab_size, terminator ids; see _PRUNED_CONFIG_KEYS}
        tokenizer.*          — original tokenizer files (for text → token ID lookup)

pruned_config.json carries the pruned-space ids of the four chat specials
(bos_token_id, start_header_id, end_header_id, eot_token_id) plus vocab_size.
eot_token_id is THE terminator: the only emitted stop token and the only trailing
EOS target.  sep_token_id is written as an alias of it so that a run still using
the flat convention against this vocabulary terminates on the same token.

Usage:
    python tools/build_vocab.py \\
      --labels_file      data/labels.jsonl \\
      --llama_dir        weights/llama3.1-8b/ \\
      --output_dir       data/pruned_tokenizer/

    python tools/build_vocab.py \\
      --librispeech_dir  data/librispeech/LibriSpeech/ \\
      --llama_dir        weights/llama3.1-8b/ \\
      --output_dir       data/pruned_tokenizer/
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.sequence import ChatTemplate


# The two instruction variants — must match data.py verbatim. These strings are
# tokenized into the pruned vocabulary, so changing them here without rebuilding
# (or rebuilding without changing them) leaves the training-time prompt with
# tokens the map does not cover, which encode() drops in silence.
INSTRUCTION_VARIANTS: list[str] = [
    "Transcribe the audio exactly as spoken, in lowercase with no punctuation.",
    "Transcribe the audio as written text, with capitalization, punctuation, and numbers as digits.",
]

_WEIGHT_SUFFIXES = frozenset({".pth", ".safetensors", ".bin", ".gguf", ".pt"})

# Ordinary (non-special) text the chat scaffold is made of.  Every one of these
# MUST survive pruning or ChatTemplate.from_tokenizer would assemble a scaffold
# with a silently missing header word — PrunedTokenizer.encode drops unmapped ids
# without raising.  "user"/"assistant" happen to occur in LibriSpeech anyway;
# the newline runs do not.
CHAT_SCAFFOLD_TEXTS: tuple[str, ...] = ("user", "assistant", "\n\n", "\n")

# The four Llama 3.1 chat specials, by pruned_config.json key → (token, canonical
# original id).  Forced in BY ID: a Meta-format tokenizer directory loads through
# the TikToken fallback, which knows no special tokens at all, so a name lookup
# returns None (see the module docstring for what that cost the base-model line).
LLAMA_CHAT_SPECIALS: dict[str, tuple[str, int]] = {
    "bos_token_id":    ("<|begin_of_text|>",   128000),
    "start_header_id": ("<|start_header_id|>", 128006),
    "end_header_id":   ("<|end_header_id|>",   128007),
    "eot_token_id":    ("<|eot_id|>",          128009),
}

# Full Llama 3.1 embedding-table size: 128,000 BPE tokens + 256 reserved special
# slots.  The specials live in the tail, so a checkpoint row exists for each of
# them even when the tokenizer object cannot name them.
_FULL_TABLE_SIZE = 128_256


def _resolve_special_ids(tokenizer) -> dict[str, int]:
    """Return {pruned_config key: ORIGINAL token id} for the four chat specials.

    The canonical ids are used as ground truth.  When the tokenizer DOES know a
    special by name (an HF-format directory with tokenizer_config.json), the two
    must agree — a disagreement means this is not a Llama 3.1 tokenizer and every
    downstream assumption about the scaffold is void, so it raises.

    Args:
        tokenizer: HuggingFace tokenizer loaded from the Llama directory

    Returns:
        {"bos_token_id": 128000, "start_header_id": 128006, ...}
    """
    resolved: dict[str, int] = {}
    for key, (token, canonical_id) in LLAMA_CHAT_SPECIALS.items():
        named_id = tokenizer.convert_tokens_to_ids(token)
        if named_id is not None and named_id != tokenizer.unk_token_id:
            if named_id != canonical_id:
                raise ValueError(
                    f"tokenizer maps {token} to id {named_id}, not the canonical "
                    f"Llama 3.1 id {canonical_id} — this is not the expected "
                    "tokenizer, refusing to build a vocabulary from it"
                )
        if canonical_id >= _FULL_TABLE_SIZE:
            raise ValueError(f"{token}: id {canonical_id} outside the embedding table")
        resolved[key] = canonical_id
    return resolved


def _find_subsequence(haystack: list[int], needle: tuple[int, ...]) -> int:
    """Index of the first contiguous occurrence of *needle* in *haystack*, or -1."""
    n = len(needle)
    if n == 0:
        return -1
    for i in range(len(haystack) - n + 1):
        if tuple(haystack[i : i + n]) == needle:
            return i
    return -1


def _verify_chat_scaffold(tokenizer, special_ids: dict[str, int]) -> None:
    """Anchor our assembled scaffold against the tokenizer's apply_chat_template.

    This is the test that catches a mis-tokenized "user"/"assistant"/newline, or
    a BPE merge that crosses one of our splice boundaries: we build the scaffold
    segment by segment, the reference renders the whole turn in one go, and the
    two must agree token-for-token around every splice point.

    Only LOCAL windows are compared, not the whole rendering: Llama 3.1's
    template always emits a system block carrying the knowledge-cutoff and
    today's date, which our convention deliberately omits.  What must match is
    every boundary the audio and the transcript are spliced between.

    Args:
        tokenizer:   HuggingFace tokenizer from the Llama directory
        special_ids: ORIGINAL-space ids from _resolve_special_ids

    Raises:
        ValueError: when a segment is not present, contiguous, in the reference
    """
    class _OriginalSpaceTokenizer:
        """ChatTemplate source in ORIGINAL id space (see model/sequence.py)."""

        bos_token_id    = special_ids["bos_token_id"]
        start_header_id = special_ids["start_header_id"]
        end_header_id   = special_ids["end_header_id"]
        eot_token_id    = special_ids["eot_token_id"]

        @staticmethod
        def encode(text: str) -> list[int]:
            return tokenizer.encode(text, add_special_tokens=False)

    chat = ChatTemplate.from_tokenizer(_OriginalSpaceTokenizer())

    placeholder = "AUDIO"
    instruction = INSTRUCTION_VARIANTS[0]
    transcript  = "the quick brown fox"
    reference: list[int] = tokenizer.apply_chat_template(
        [
            {"role": "user",      "content": f"{placeholder}\n{instruction}"},
            {"role": "assistant", "content": transcript},
        ],
        tokenize             = True,
        add_generation_prompt = False,
    )

    enc = _OriginalSpaceTokenizer.encode
    eot = special_ids["eot_token_id"]

    checks: list[tuple[str, tuple[int, ...]]] = [
        # User header, up to and including the "\n\n" the audio is spliced after.
        # (The leading bos is checked separately: the reference puts the system
        # block between it and the user header.)
        ("user header + audio splice point",
         chat.seg_pre_audio[1:] + tuple(enc(placeholder))),
        # Tail of the user turn: "\n" separator, instruction, end of turn.
        ("audio → instruction → end of user turn",
         chat.seg_pre_instruction + tuple(enc(instruction)) + (eot,)),
        # The whole assistant turn: header, transcript, terminating eot.
        ("assistant header + transcript + trailing eot",
         chat.seg_pre_transcript + tuple(enc(transcript)) + (eot,)),
    ]
    for label, segment in checks:
        if _find_subsequence(reference, segment) < 0:
            raise ValueError(
                f"chat scaffold mismatch [{label}]: the assembled ids {segment} do "
                f"not occur contiguously in apply_chat_template's rendering "
                f"{reference}.  Our segment-by-segment assembly disagrees with the "
                "tokenizer — do NOT train against this vocabulary."
            )

    if reference[0] != chat.seg_pre_audio[0]:
        raise ValueError(
            f"reference rendering starts with id {reference[0]}, expected "
            f"<|begin_of_text|> ({chat.seg_pre_audio[0]})"
        )
    if reference[-1] != eot:
        raise ValueError(
            f"reference rendering ends with id {reference[-1]}, expected "
            f"<|eot_id|> ({eot})"
        )
    print("  chat scaffold verified against apply_chat_template "
          f"({len(checks)} splice windows + bos/eot anchors)")


# ── Text helpers (mirrors preprocess.py) ──────────────────────────────────────

def _capitalize_sentences(text: str) -> str:
    if not text:
        return text
    text = text[0].upper() + text[1:]
    return re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)


def _capitalize_proper_nouns(text: str, nlp) -> str:
    if not text:
        return text
    doc   = nlp(text)
    chars = list(text)
    for token in doc:
        if token.pos_ == "PROPN":
            chars[token.idx] = chars[token.idx].upper()
    return "".join(chars)


# ── Transcript iterator ───────────────────────────────────────────────────────

def _iter_label_texts(labels_file: Path) -> Iterator[str]:
    """Yield both label strings of every usable record in a labels JSONL.

    A record marked "validation": "failed" is skipped: tools/preprocess.py and
    tools/relabel_shards.py drop those samples, so their tokens are not in the
    corpus and have no business reserving embedding rows. Records with no
    `validation` field (tools/precompute_labels.py output) count as usable.
    """
    n_ok = n_failed = 0
    with labels_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("validation", "ok") != "ok":
                n_failed += 1
                continue
            n_ok += 1
            yield record["unformatted"]
            yield record["formatted"]
    print(f"  {n_ok:,} usable records, {n_failed:,} failed records skipped")


def _collect_ids(tokenizer, texts: Iterator[str], chunk_size: int) -> set[int]:
    """Union of token ids over a stream of strings.

    Tokenizes in batches rather than one string at a time: the fast tokenizer
    parallelises a batch, and this runs over ~584k strings.
    """
    used_ids: set[int] = set()
    n_done = 0
    batch: list[str] = []

    def _flush() -> None:
        nonlocal n_done
        if not batch:
            return
        for ids in tokenizer(batch, add_special_tokens=False)["input_ids"]:
            used_ids.update(ids)
        n_done += len(batch)
        if n_done % 50_000 < len(batch):
            print(f"  {n_done:,} strings, {len(used_ids):,} unique token IDs …", flush=True)
        batch.clear()

    for text in texts:
        batch.append(text)
        if len(batch) >= chunk_size:
            _flush()
    _flush()
    print(f"Scanned {n_done:,} strings. Unique token IDs: {len(used_ids):,}")
    return used_ids


def _iter_transcripts(librispeech_dir: Path) -> Iterator[str]:
    """Yield raw transcript strings from all *.trans.txt files."""
    for trans_file in sorted(librispeech_dir.rglob("*.trans.txt")):
        with trans_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                _, _, raw_text = line.partition(" ")
                if raw_text:
                    yield raw_text


# ── Batched full-pipeline formatter ───────────────────────────────────────────

def _format_batch(
    raw_texts: list[str],
    english_norm,
    punct_model,
    nlp,
    punct_batch_size: int = 128,
    spacy_batch_size: int = 256,
) -> list[str]:
    """Apply the full formatting pipeline to a list of raw transcripts.

    Mirrors the per-sample logic in preprocess.py but processes multiple texts
    in a single PunctuationModel and spaCy batch call for throughput.

    Args:
        raw_texts:        list of raw LibriSpeech transcript strings
        english_norm:     EnglishTextNormalizer instance
        punct_model:      PunctuationModel instance (monkey-patched init)
        nlp:              spaCy Language model (en_core_web_sm)
        punct_batch_size: HuggingFace pipeline batch size for BERT inference
        spacy_batch_size: spaCy nlp.pipe batch size

    Returns:
        list of formatted transcript strings, same order as raw_texts
    """
    # ── Step 1: EnglishTextNormalizer (fast, rule-based) ──────────────────
    english_texts = [english_norm(raw) for raw in raw_texts]

    # ── Step 2: Preprocess for PunctuationModel (split into word lists) ───
    # PunctuationModel.preprocess removes any residual punctuation and splits
    # into words — for LibriSpeech (which has no punctuation after english_norm)
    # this is just a split.
    word_lists = [punct_model.preprocess(text) for text in english_texts]

    # Filter out empty inputs (can happen after preprocess on very short texts)
    joined = [" ".join(wl) if wl else " " for wl in word_lists]

    # ── Step 3: Batch BERT NER through PunctuationModel ───────────────────
    # Calling punct_model.pipe([text1, text2, ...], batch_size=N) processes
    # all texts in one batched forward pass instead of N sequential calls.
    all_ner: list[list[dict]] = punct_model.pipe(joined, batch_size=punct_batch_size)

    # ── Step 4: Map NER results back to word-level labels → punctuated text ─
    punct_texts: list[str] = []
    for words, ner_results in zip(word_lists, all_ner):
        if not words:
            punct_texts.append("")
            continue

        tagged_words = []
        char_index   = 0
        result_index = 0

        for word in words:
            char_index += len(word) + 1   # +1 for the space separator
            label = 0
            while (
                result_index < len(ner_results)
                and char_index > ner_results[result_index]["end"]
            ):
                label = ner_results[result_index]["entity"]
                result_index += 1
            tagged_words.append([word, label, 0])

        punct_texts.append(punct_model.prediction_to_text(tagged_words))

    # ── Step 5: Sentence capitalisation ───────────────────────────────────
    cap_texts = [_capitalize_sentences(t) for t in punct_texts]

    # ── Step 6: Batch spaCy PROPN capitalisation ──────────────────────────
    formatted: list[str] = []
    for doc, text in zip(nlp.pipe(cap_texts, batch_size=spacy_batch_size), cap_texts):
        chars = list(text)
        for token in doc:
            if token.pos_ == "PROPN":
                chars[token.idx] = chars[token.idx].upper()
        formatted.append("".join(chars))

    return formatted


# ── Main ──────────────────────────────────────────────────────────────────────

def _corpus_label_texts(librispeech_dir: Path, chunk_size: int) -> Iterator[str]:
    """Re-derive both label strings per transcript with the old NLP pipeline.

    Only used by --librispeech_dir. The heavy models are imported and loaded
    here rather than in build_vocab so that a --labels_file run needs neither
    PunctuationModel nor spaCy installed.
    """
    from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer

    basic_norm   = BasicTextNormalizer()
    english_norm = EnglishTextNormalizer()

    print("Loading PunctuationModel …")
    import deepmultilingualpunctuation.punctuationmodel as _pmod
    from transformers import pipeline as _hf_pipeline

    def _fixed_punct_init(
        self,
        model: str = "oliverguhr/fullstop-punctuation-multilang-large",
    ) -> None:
        self.pipe = _hf_pipeline(
            "token-classification",
            model,
            aggregation_strategy=None,
        )

    _pmod.PunctuationModel.__init__ = _fixed_punct_init
    from deepmultilingualpunctuation import PunctuationModel
    punct_model = PunctuationModel()

    print("Loading spaCy en_core_web_sm …")
    import spacy as _spacy
    nlp = _spacy.load("en_core_web_sm")

    print(f"Scanning and formatting transcripts under {librispeech_dir} …")
    buf: list[str] = []

    def _drain(buf: list[str]) -> Iterator[str]:
        for raw, fmt in zip(buf, _format_batch(buf, english_norm, punct_model, nlp)):
            yield basic_norm(raw)
            yield fmt

    for raw in _iter_transcripts(librispeech_dir):
        buf.append(raw)
        if len(buf) >= chunk_size:
            yield from _drain(buf)
            buf = []
    if buf:
        yield from _drain(buf)


def build_vocab(
    llama_dir: Path,
    output_dir: Path,
    librispeech_dir: Path | None = None,
    labels_file: Path | None = None,
    chunk_size: int = 512,
    allow_missing_chat_template: bool = False,
) -> None:
    """Prune the Llama tokenizer and save the result.

    Exactly one of librispeech_dir / labels_file selects where the label text
    comes from; see the module docstring on why that choice must match the
    shards.

    Args:
        llama_dir:       directory containing the Llama 3.1 8B tokenizer files
        output_dir:      directory to write vocab_map.json and pruned tokenizer
        librispeech_dir: root LibriSpeech directory, labels re-derived by the
                         BasicTextNormalizer / PunctuationModel / spaCy pipeline
        labels_file:     labels JSONL to read the label text from directly
        chunk_size:      strings processed per batch (for memory control)
        allow_missing_chat_template: build even when the tokenizer has no chat
                         template to anchor the scaffold against (prints a
                         warning instead of raising)
    """
    from transformers import AutoTokenizer

    if (librispeech_dir is None) == (labels_file is None):
        raise ValueError("Pass exactly one of librispeech_dir / labels_file.")

    print(f"Loading tokenizer from {llama_dir} …")
    tokenizer = AutoTokenizer.from_pretrained(str(llama_dir))

    original_vocab_size = tokenizer.vocab_size
    print(f"Original vocab size: {original_vocab_size:,}")

    if labels_file is not None:
        print(f"Reading label text from {labels_file} …")
        texts = _iter_label_texts(labels_file)
    else:
        texts = _corpus_label_texts(librispeech_dir, chunk_size)

    used_ids = _collect_ids(tokenizer, texts, chunk_size)

    # ── Tokenize both instruction strings ─────────────────────────────────
    for variant in INSTRUCTION_VARIANTS:
        used_ids.update(tokenizer.encode(variant, add_special_tokens=False))

    print(f"After instruction strings: {len(used_ids):,} unique token IDs")

    # ── Chat scaffold: ordinary words, then the four specials ─────────────
    for text in CHAT_SCAFFOLD_TEXTS:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError(f"chat scaffold text {text!r} tokenizes to nothing")
        used_ids.update(ids)
    print(f"After chat scaffold words: {len(used_ids):,} unique token IDs")

    special_ids = _resolve_special_ids(tokenizer)
    used_ids.update(special_ids.values())

    # Anchor our assembly against the tokenizer's own rendering BEFORE writing
    # anything: a scaffold that disagrees must not reach a training run.
    if getattr(tokenizer, "chat_template", None) is None:
        message = (
            f"the tokenizer in {llama_dir} carries no chat_template, so the "
            "assembled chat scaffold cannot be anchored against "
            "apply_chat_template.  Point --llama_dir at the Instruct model's "
            "HuggingFace directory (its tokenizer_config.json carries the "
            "template), or pass --allow_missing_chat_template to build the "
            "vocabulary unverified."
        )
        if not allow_missing_chat_template:
            raise ValueError(message)
        print(f"  WARNING: {message}")
    else:
        _verify_chat_scaffold(tokenizer, special_ids)

    # ── Build contiguous re-indexing ──────────────────────────────────────
    vocab_ids   = sorted(used_ids)
    vocab_map   = {old: new for new, old in enumerate(vocab_ids)}
    pruned_size = len(vocab_ids)

    # Every special must have survived pruning — they were just forced into
    # used_ids, so a failure here means the re-indexing itself is broken.
    pruned_special_ids = {}
    for key, old_id in special_ids.items():
        assert old_id in vocab_map, f"{key} ({old_id}) missing from vocab_map"
        pruned_special_ids[key] = vocab_map[old_id]
    eot_new_id = pruned_special_ids["eot_token_id"]

    # ── Write outputs ─────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy the tokenizer files FIRST, then write our own outputs on top.  The two
    # generated files are also explicitly excluded from the copy: pointing
    # --llama_dir at a previously built pruned_tokenizer directory (which holds a
    # vocab_map.json of its own) would otherwise overwrite the freshly computed
    # map with the stale one and leave it inconsistent with pruned_config.json —
    # a silent corruption that mis-indexes every embedding row.
    _GENERATED = {"vocab_map.json", "pruned_config.json"}
    copied = []
    for item in sorted(llama_dir.iterdir()):
        if item.is_file() and item.suffix not in _WEIGHT_SUFFIXES:
            if item.name in _GENERATED:
                print(f"  not copying {item.name} from {llama_dir} — this build writes it")
                continue
            shutil.copy2(item, output_dir / item.name)
            copied.append(item.name)

    with (output_dir / "vocab_map.json").open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in vocab_map.items()}, f)

    pruned_config = {
        "vocab_size": pruned_size,
        # <|eot_id|> is THE terminator: the only emitted stop token and the only
        # trailing EOS target.  sep_token_id is written as an alias of it so a
        # flat-convention run against this vocabulary stops on the same token
        # (there is no separate SEP any more).
        "eot_token_id":     eot_new_id,
        "sep_token_id":     eot_new_id,
        "bos_token_id":     pruned_special_ids["bos_token_id"],
        "start_header_id":  pruned_special_ids["start_header_id"],
        "end_header_id":    pruned_special_ids["end_header_id"],
        # Original-space ids, for auditing a built vocabulary against the
        # checkpoint's embedding table.
        "special_old_ids":  special_ids,
    }
    with (output_dir / "pruned_config.json").open("w", encoding="utf-8") as f:
        json.dump(pruned_config, f, indent=2)

    print()
    print(f"Original vocab size  : {original_vocab_size:,}")
    print(f"Pruned vocab size    : {pruned_size:,}")
    print("Chat specials (pruned id ← original id):")
    for key, (token, _) in LLAMA_CHAT_SPECIALS.items():
        print(f"  {token:<22} {key:<16} {pruned_special_ids[key]:>6} ← {special_ids[key]}")
    print(f"Terminator (eot_token_id, pruned): {eot_new_id}")
    print(f"Tokenizer files      : {copied}")
    print(f"Output directory     : {output_dir}")
    print()
    print("Remember: rebuilding REASSIGNS every id. Update metrics.py:_VOCAB_SIZE "
          "and tools/probe_accum.py:_VOCAB_SIZE to "
          f"{pruned_size}, and rebuild baselines.json (tools/compute_baselines.py).")


def main() -> None:
    """Parse CLI arguments and run vocab pruning."""
    parser = argparse.ArgumentParser(
        description="Prune the Llama tokenizer to the LibriSpeech vocabulary.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--labels_file",
        type=Path,
        help=(
            "labels JSONL to read label text from directly (finalize output); "
            "records marked validation=failed are skipped. Use this whenever the "
            "shards were built from a labels file"
        ),
    )
    source.add_argument(
        "--librispeech_dir",
        type=Path,
        help=(
            "root LibriSpeech directory; all *.trans.txt files found recursively "
            "are included (e.g. data/librispeech/LibriSpeech/ to cover all splits). "
            "Labels are re-derived with the BasicTextNormalizer / PunctuationModel "
            "/ spaCy pipeline"
        ),
    )
    parser.add_argument(
        "--llama_dir",
        type=Path,
        required=True,
        help="directory containing the Llama 3.1 8B tokenizer files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="directory to write vocab_map.json and pruned tokenizer",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="strings per processing batch (default: 512)",
    )
    parser.add_argument(
        "--allow_missing_chat_template",
        action="store_true",
        help=(
            "build even when the tokenizer directory has no chat template to "
            "anchor the assembled scaffold against (warns instead of raising)"
        ),
    )
    args = parser.parse_args()

    build_vocab(
        llama_dir=args.llama_dir,
        output_dir=args.output_dir,
        librispeech_dir=args.librispeech_dir,
        labels_file=args.labels_file,
        chunk_size=args.chunk_size,
        allow_missing_chat_template=args.allow_missing_chat_template,
    )


# ── Self-test ─────────────────────────────────────────────────────────────────

def _self_test() -> None:
    """CPU-only checks of the chat-scaffold machinery (no tokenizer files needed).

    Uses a hand-written fake tokenizer that renders the real Llama 3.1 chat
    layout — system block included — so the verification logic is exercised
    against the same shape of input it will see on the pod, and against a broken
    renderer that it must reject.
    """
    # Ordinary-token id table for the fake tokenizer.  Values are arbitrary but
    # must be distinct from the specials.
    _WORDS = {
        "user": [100], "assistant": [101], "\n\n": [102], "\n": [103],
        "AUDIO": [104], "the quick brown fox": [105, 106, 107],
        "system": [108], "Cutting Knowledge": [109],
        INSTRUCTION_VARIANTS[0]: [110, 111, 112],
    }
    _BOS, _START, _END, _EOT = (
        LLAMA_CHAT_SPECIALS[k][1]
        for k in ("bos_token_id", "start_header_id", "end_header_id", "eot_token_id")
    )

    class _FakeTokenizer:
        chat_template  = "not None — presence is all build_vocab checks"
        unk_token_id   = None

        def convert_tokens_to_ids(self, token: str):
            # Name-unaware, exactly like a Meta-format directory loaded through
            # the TikToken fallback.  This is the case that forced the by-id path.
            return None

        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            if text not in _WORDS:
                raise KeyError(f"fake tokenizer has no entry for {text!r}")
            return list(_WORDS[text])

        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
            # The real template always emits a system block first; our convention
            # omits it, which is exactly why verification compares local windows.
            ids = [_BOS, _START, *_WORDS["system"], _END, *_WORDS["\n\n"],
                   *_WORDS["Cutting Knowledge"], _EOT]
            for msg in messages:
                ids += [_START, *_WORDS[msg["role"]], _END, *_WORDS["\n\n"]]
                for chunk in msg["content"].split("\n"):
                    if chunk:
                        ids += _WORDS[chunk]
                    else:
                        continue
                    if chunk != msg["content"].split("\n")[-1]:
                        ids += _WORDS["\n"]
                ids.append(_EOT)
            return ids

    tok = _FakeTokenizer()

    # ── _resolve_special_ids: canonical ids used when names resolve to nothing ─
    resolved = _resolve_special_ids(tok)
    assert resolved == {k: v for k, (_, v) in LLAMA_CHAT_SPECIALS.items()}, resolved
    assert resolved["eot_token_id"] == 128009
    print("[OK] chat specials resolved by canonical id on a name-unaware tokenizer")

    # A tokenizer that names a special DIFFERENTLY is a different tokenizer.
    class _WrongTokenizer(_FakeTokenizer):
        def convert_tokens_to_ids(self, token: str):
            return 999 if token == "<|eot_id|>" else None

    try:
        _resolve_special_ids(_WrongTokenizer())
    except ValueError:
        pass
    else:
        raise AssertionError("a tokenizer disagreeing on a special id must raise")
    print("[OK] disagreeing special id rejected")

    # ── _find_subsequence ─────────────────────────────────────────────────────
    assert _find_subsequence([1, 2, 3, 4], (2, 3)) == 1
    assert _find_subsequence([1, 2, 3, 4], (2, 4)) == -1
    assert _find_subsequence([1, 2], (1, 2, 3)) == -1

    # ── _verify_chat_scaffold: passes on a faithful renderer ──────────────────
    _verify_chat_scaffold(tok, resolved)
    print("[OK] scaffold verification passes against a faithful renderer")

    # ── … and FAILS when the renderer disagrees ──────────────────────────────
    # A template that puts the header words in a different order (or tokenizes
    # the newline differently) must be caught — that is the whole point.
    class _DriftedTokenizer(_FakeTokenizer):
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
            ids = super().apply_chat_template(messages, tokenize, add_generation_prompt)
            # Drop the "\n\n" that follows the assistant header: our scaffold
            # emits it, so the assistant window no longer matches.
            head = [_START, *_WORDS["assistant"], _END, *_WORDS["\n\n"]]
            at   = _find_subsequence(ids, tuple(head))
            assert at >= 0
            return ids[: at + len(head) - 1] + ids[at + len(head):]

    try:
        _verify_chat_scaffold(_DriftedTokenizer(), resolved)
    except ValueError:
        pass
    else:
        raise AssertionError("a drifted chat template must be rejected")
    print("[OK] scaffold verification rejects a drifted template")

    # ── ChatTemplate offsets, in original id space ───────────────────────────
    class _Shim:
        bos_token_id, start_header_id = _BOS, _START
        end_header_id,  eot_token_id  = _END, _EOT
        encode = staticmethod(lambda text: list(_WORDS[text]))

    chat = ChatTemplate.from_tokenizer(_Shim())
    assert chat.seg_pre_audio == (_BOS, _START, 100, _END, 102), chat.seg_pre_audio
    assert chat.audio_offset == 6, chat.audio_offset
    print(f"[OK] ChatTemplate assembled in original id space (C = {chat.audio_offset})")

    print("\nPASSED")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
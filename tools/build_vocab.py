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
  4. Select SEP: a Llama special token absent from the collected set.
  5. Save vocab_map.json, pruned_config.json, and tokenizer files to --output_dir.

Rebuilding the vocabulary REASSIGNS every id, so `vocab_map.json` and any
checkpoint carrying a trained `llama` state_dict must be regenerated together:
embedding rows are indexed by the new id. Checkpoints that never trained llama
are unaffected — build.py rebuilds their embedding from the pretrained Llama
weights through vocab_map on every load.

Output directory layout:
    output_dir/
        vocab_map.json       — {old_str_id: new_int_id, ...}
        pruned_config.json   — {vocab_size, sep_token_id, sep_old_token_id}
        tokenizer.*          — original tokenizer files (for text → token ID lookup)

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
from pathlib import Path
from typing import Iterator


# The two instruction variants — must match data.py verbatim. These strings are
# tokenized into the pruned vocabulary, so changing them here without rebuilding
# (or rebuilding without changing them) leaves the training-time prompt with
# tokens the map does not cover, which encode() drops in silence.
INSTRUCTION_VARIANTS: list[str] = [
    "Transcribe the audio exactly as spoken, in lowercase with no punctuation.",
    "Transcribe the audio as written text, with capitalization, punctuation, and numbers as digits.",
]

_WEIGHT_SUFFIXES = frozenset({".pth", ".safetensors", ".bin", ".gguf", ".pt"})


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

    # ── Choose SEP token ───────────────────────────────────────────────────
    sep_old_id: int | None = None
    for tok_id in sorted(tokenizer.all_special_ids):
        if tok_id not in used_ids:
            sep_old_id = tok_id
            break

    if sep_old_id is None:
        for tok_id in range(original_vocab_size - 1, -1, -1):
            if tok_id not in used_ids:
                sep_old_id = tok_id
                break

    if sep_old_id is None:
        raise RuntimeError("Cannot find a free SEP token — check the manifest.")

    # ── Build contiguous re-indexing ──────────────────────────────────────
    vocab_ids   = sorted(used_ids | {sep_old_id})
    vocab_map   = {old: new for new, old in enumerate(vocab_ids)}
    sep_new_id  = vocab_map[sep_old_id]
    pruned_size = len(vocab_ids)

    # ── Write outputs ─────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "vocab_map.json").open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in vocab_map.items()}, f)

    copied = []
    for item in sorted(llama_dir.iterdir()):
        if item.is_file() and item.suffix not in _WEIGHT_SUFFIXES:
            shutil.copy2(item, output_dir / item.name)
            copied.append(item.name)

    pruned_config = {
        "vocab_size":       pruned_size,
        "sep_token_id":     sep_new_id,
        "sep_old_token_id": sep_old_id,
    }
    with (output_dir / "pruned_config.json").open("w", encoding="utf-8") as f:
        json.dump(pruned_config, f, indent=2)

    sep_token_str = tokenizer.convert_ids_to_tokens(sep_old_id)
    print()
    print(f"Original vocab size  : {original_vocab_size:,}")
    print(f"Pruned vocab size    : {pruned_size:,}")
    print(f"SEP token (old id)   : {sep_old_id}  ({sep_token_str!r})")
    print(f"SEP token (new id)   : {sep_new_id}")
    print(f"Tokenizer files      : {copied}")
    print(f"Output directory     : {output_dir}")


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
    args = parser.parse_args()

    build_vocab(
        llama_dir=args.llama_dir,
        output_dir=args.output_dir,
        librispeech_dir=args.librispeech_dir,
        labels_file=args.labels_file,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
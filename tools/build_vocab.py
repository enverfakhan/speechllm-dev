"""Prune the Llama tokenizer to the vocabulary needed for LibriSpeech training.

Scans raw transcript files from all LibriSpeech splits directly — no preprocessed
shards or manifest required. This lets vocabulary building run in parallel with
(or before) shard preprocessing, and guarantees the pruned vocabulary covers the
full 960h corpus regardless of which splits have been processed.

Text normalisation runs the exact same pipeline as scripts/preprocess.py:
  Unformatted:  BasicTextNormalizer
  Formatted:    EnglishTextNormalizer
                → deepmultilingualpunctuation PunctuationModel (BERT NER)
                → _capitalize_sentences  (sentence-initial caps)
                → spaCy en_core_web_sm PROPN tagger (proper-noun caps)

PunctuationModel inference is batched across texts for efficiency: all transcripts
in a chunk are joined into a list and processed in a single pipeline call rather
than one text at a time. This reduces vocab-build time from ~40 min to ~3 min.

Steps:
  1. Scan all *.trans.txt files under --librispeech_dir (all splits).
  2. Apply both normalisation pipelines to every transcript.
  3. Tokenize all resulting strings; collect the union of token IDs.
  4. Tokenize both instruction prompt strings.
  5. Select SEP: a Llama special token absent from the collected set.
  6. Save vocab_map.json, pruned_config.json, and tokenizer files to --output_dir.

Output directory layout:
    output_dir/
        vocab_map.json       — {old_str_id: new_int_id, ...}
        pruned_config.json   — {vocab_size, sep_token_id, sep_old_token_id}
        tokenizer.*          — original tokenizer files (for text → token ID lookup)

Usage:
    python scripts/build_vocab.py \\
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


# The two instruction variants — must match data.py and preprocess.py
INSTRUCTION_VARIANTS: list[str] = [
    "Transcribe the following audio without formatting.",
    "Transcribe the following audio with proper formatting.",
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

def build_vocab(
    librispeech_dir: Path,
    llama_dir: Path,
    output_dir: Path,
    chunk_size: int = 512,
) -> None:
    """Prune the Llama tokenizer and save the result.

    Args:
        librispeech_dir: root LibriSpeech directory containing all splits
        llama_dir:       directory containing the Llama 3.1 8B tokenizer files
        output_dir:      directory to write vocab_map.json and pruned tokenizer
        chunk_size:      number of transcripts processed per batch (for memory control)
    """
    from transformers import AutoTokenizer
    from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer

    print(f"Loading tokenizer from {llama_dir} …")
    tokenizer    = AutoTokenizer.from_pretrained(str(llama_dir))
    basic_norm   = BasicTextNormalizer()
    english_norm = EnglishTextNormalizer()

    original_vocab_size = tokenizer.vocab_size
    print(f"Original vocab size: {original_vocab_size:,}")

    # ── Load formatting models ─────────────────────────────────────────────
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

    # ── Stream all transcripts and process in chunks ───────────────────────
    print(f"Scanning and formatting transcripts under {librispeech_dir} …")
    used_ids: set[int] = set()
    n_done      = 0
    chunk_buf:  list[str] = []

    def _flush(buf: list[str]) -> None:
        nonlocal n_done
        unformatted_texts = [basic_norm(raw) for raw in buf]
        formatted_texts   = _format_batch(buf, english_norm, punct_model, nlp)

        for un, fmt in zip(unformatted_texts, formatted_texts):
            used_ids.update(tokenizer.encode(un,  add_special_tokens=False))
            used_ids.update(tokenizer.encode(fmt, add_special_tokens=False))

        n_done += len(buf)
        if n_done % 50_000 < len(buf):
            print(f"  {n_done:,} transcripts, {len(used_ids):,} unique token IDs …")

    for raw in _iter_transcripts(librispeech_dir):
        chunk_buf.append(raw)
        if len(chunk_buf) >= chunk_size:
            _flush(chunk_buf)
            chunk_buf.clear()

    if chunk_buf:
        _flush(chunk_buf)

    print(f"Scanned {n_done:,} transcripts. Unique token IDs: {len(used_ids):,}")

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
    parser.add_argument(
        "--librispeech_dir",
        type=Path,
        required=True,
        help=(
            "root LibriSpeech directory; all *.trans.txt files found recursively "
            "are included (e.g. data/librispeech/LibriSpeech/ to cover all splits)"
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
        help="transcripts per processing batch (default: 512)",
    )
    args = parser.parse_args()

    build_vocab(
        librispeech_dir=args.librispeech_dir,
        llama_dir=args.llama_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
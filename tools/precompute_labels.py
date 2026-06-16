"""Pre-compute unformatted and formatted transcript labels for all LibriSpeech samples.

Running the full formatting pipeline (PunctuationModel + spaCy) inside the sharding
loop makes preprocessing prohibitively slow for large splits (train-clean-360,
train-other-500). This script runs the pipeline once across all splits, saves the
results to a JSONL file, and lets preprocess.py do a fast O(1) label lookup instead.

The output file has one JSON record per line:
    {"key": "1234-56789-0001", "unformatted": "...", "formatted": "..."}

Usage:
    python scripts/precompute_labels.py \\
      --librispeech_dir  data/librispeech/LibriSpeech/ \\
      --output           data/labels.jsonl

Then pass --labels_file to preprocess.py:
    python scripts/preprocess.py \\
      --input_dir  data/librispeech/LibriSpeech/train-other-500 \\
      --output_dir data/shards/ \\
      --split      train-other-500 \\
      --labels_file data/labels.jsonl \\
      --seed 42
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterator


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

def _iter_transcripts(librispeech_dir: Path) -> Iterator[tuple[str, str]]:
    """Yield (key, raw_text) for every utterance in librispeech_dir."""
    for trans_file in sorted(librispeech_dir.rglob("*.trans.txt")):
        with trans_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                key, _, raw_text = line.partition(" ")
                if raw_text:
                    yield key, raw_text


# ── Batched full-pipeline formatter (mirrors build_vocab.py) ─────────────────

def _format_batch(
    raw_texts: list[str],
    english_norm,
    punct_model,
    nlp,
    punct_batch_size: int = 128,
    spacy_batch_size: int = 256,
) -> list[str]:
    """Apply the full formatting pipeline to a list of raw transcripts."""
    english_texts = [english_norm(raw) for raw in raw_texts]
    word_lists    = [punct_model.preprocess(text) for text in english_texts]
    joined        = [" ".join(wl) if wl else " " for wl in word_lists]

    all_ner: list[list[dict]] = punct_model.pipe(joined, batch_size=punct_batch_size)

    punct_texts: list[str] = []
    for words, ner_results in zip(word_lists, all_ner):
        if not words:
            punct_texts.append("")
            continue
        tagged_words = []
        char_index   = 0
        result_index = 0
        for word in words:
            char_index += len(word) + 1
            label = 0
            while (
                result_index < len(ner_results)
                and char_index > ner_results[result_index]["end"]
            ):
                label = ner_results[result_index]["entity"]
                result_index += 1
            tagged_words.append([word, label, 0])
        punct_texts.append(punct_model.prediction_to_text(tagged_words))

    cap_texts = [_capitalize_sentences(t) for t in punct_texts]

    formatted: list[str] = []
    for doc, text in zip(nlp.pipe(cap_texts, batch_size=spacy_batch_size), cap_texts):
        chars = list(text)
        for token in doc:
            if token.pos_ == "PROPN":
                chars[token.idx] = chars[token.idx].upper()
        formatted.append("".join(chars))

    return formatted


# ── Main ──────────────────────────────────────────────────────────────────────

def precompute_labels(
    librispeech_dir: Path,
    output_path: Path,
    chunk_size: int = 512,
) -> None:
    """Scan all trans.txt files, run the full formatting pipeline, write labels JSONL.

    Args:
        librispeech_dir: root LibriSpeech directory (all splits scanned recursively)
        output_path:     destination JSONL file
        chunk_size:      transcripts processed per batch
    """
    from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer

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

    basic_norm   = BasicTextNormalizer()
    english_norm = EnglishTextNormalizer()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Scanning transcripts under {librispeech_dir} …")
    n_done = 0
    chunk_keys: list[str] = []
    chunk_raws: list[str] = []

    with output_path.open("w", encoding="utf-8") as out_f:

        def _flush() -> None:
            nonlocal n_done
            unformatted_texts = [basic_norm(raw) for raw in chunk_raws]
            formatted_texts   = _format_batch(chunk_raws, english_norm, punct_model, nlp)
            for key, un, fmt in zip(chunk_keys, unformatted_texts, formatted_texts):
                out_f.write(json.dumps({"key": key, "unformatted": un, "formatted": fmt}) + "\n")
            n_done += len(chunk_keys)
            if n_done % 50_000 < len(chunk_keys):
                print(f"  {n_done:,} samples written …")

        for key, raw in _iter_transcripts(librispeech_dir):
            chunk_keys.append(key)
            chunk_raws.append(raw)
            if len(chunk_keys) >= chunk_size:
                _flush()
                chunk_keys.clear()
                chunk_raws.clear()

        if chunk_keys:
            _flush()

    print(f"Done. {n_done:,} labels written → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute transcript labels for all LibriSpeech samples.",
    )
    parser.add_argument(
        "--librispeech_dir",
        type=Path,
        required=True,
        help="root LibriSpeech directory (all splits scanned recursively)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="destination JSONL file, e.g. data/labels.jsonl",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="transcripts processed per batch  (default: 512)",
    )
    args = parser.parse_args()

    precompute_labels(
        librispeech_dir=args.librispeech_dir,
        output_path=args.output,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()

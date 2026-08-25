"""Build house-format eval shards from an Earnings-22 subset (far-domain audio).

WHY THIS EXISTS
---------------
TED-LIUM varies speaking style and Common Voice varies speaker and channel, but
both are still one person talking clearly into a microphone about a prepared
subject.  Earnings-22 is 119 hours of real earnings calls: telephone-band
audio, multiple speakers, global accents, and financial jargon dense with
numerals.  It is the far-domain end of the protocol — where an in-domain-tuned
model is expected to fail, and the question is only whether it fails GRACEFULLY
(substitutions) or IN KIND (loops, hallucinations — DEGENERACY_TAXONOMY.md).

SEGMENTATION IS REUSED, NOT REBUILT
------------------------------------
Earnings-22 ships as whole calls with a full-file transcript, and chunking those
against the transcript is genuine forced-alignment work this protocol explicitly
declines to do.  It is not needed: the ESB baseline release
(``anton-l/earnings22_baseline_5_gram``, the source the official
``revdotcom/earnings22`` loader itself downloads) publishes the segmentation —
``metadata.csv`` gives every segment's ``start_ts``/``end_ts`` and sentence, and
``data/chunked/{source_id}.tar.gz`` holds the already-cut per-segment wavs.  This
tool consumes that, so no aligner exists here to be fragile.

SUBSET, AND WHAT IT COSTS
--------------------------
The chunked release is ~20 GB across 125 calls.  Rather than pull all of it for
a couple of thousand segments, a deterministic subset of CALLS is drawn first
and only those tarballs are fetched, then segments are drawn within them.

That trade is worth naming: Earnings-22's whole point is accent breadth across
global companies, and sampling 40 of 125 calls samples that breadth too.  The
resulting number is "far-domain audio from 40 calls", not "Earnings-22".  Widen
``--sources`` (at ~80 MB per call) if the accent axis is the claim being made.
Both figures land in stats.json.

REFERENCES
----------
Rev's human transcripts — genuine transcriptions of what was said, unlike Common
Voice's prompt text.  They are cased and punctuated, so both reference forms are
real here: ``formatted.txt`` is the ASCII-folded sentence and
``unformatted.txt`` is its FORMATTING_SPEC §6 normalisation.

**22% of Earnings-22 sentences carry digits** — by far the highest of the three
sets, and §6 does not expand digit-bearing tokens back to words.  Every segment
is flagged ``has_digits`` so tools/ood_report.py can score the digit-free slice,
which is the number to read on this corpus.

USAGE
-----
    python tools/prepare_ood_earnings22.py --download \\
        --output_dir data/ood_shards/earnings22/ \\
        --tokenizer  data/pruned_tokenizer/ \\
        --sources 40 --n 2000 --seed 42

    python tools/prepare_ood_earnings22.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.whisper_encoder import SAMPLE_RATE, log_mel_spectrogram

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_formatted import normalize as spec6_normalize  # noqa: E402
from ood_shard import ShardWriter, load_audio_bytes, resample  # noqa: E402
from prepare_ood_commonvoice import fold_ascii, out_of_convention  # noqa: E402

_BASE_URL = (
    "https://huggingface.co/datasets/anton-l/earnings22_baseline_5_gram/resolve/main/"
)
_METADATA_URL = _BASE_URL + "metadata.csv"

_DIGIT_RE = re.compile(r"\d")

# Spontaneous-speech markers Rev's transcripts keep and LibriSpeech labels have
# never contained: filled pauses, and the marker for audio the transcriber could
# not make out.  Flagged per segment because they are a reference-convention
# floor, not an ASR error — a model trained on read audiobooks will not emit
# "um", and neither will most ASR systems, so both sides of the Δ pay for them.
_DISFLUENCY_RE = re.compile(r"\b(um|uh|erm|mm)\b", re.IGNORECASE)
_INAUDIBLE_RE  = re.compile(r"<?\binaudible\b>?", re.IGNORECASE)


def select_sources(source_ids: list[str], n_sources: int, seed: int) -> list[str]:
    """Draw a reproducible subset of calls.  Sorted first, so tsv order is irrelevant."""
    ordered = sorted(set(source_ids))
    return sorted(random.Random(seed).sample(ordered, min(n_sources, len(ordered))))


def select_segments(
    by_source: dict[str, list[dict]], sources: list[str],
    per_source: int, n: int, seed: int,
) -> list[dict]:
    """Draw up to per_source segments from each selected call, capped at n total.

    Per-call rather than global so one long call cannot dominate the subset —
    the accent axis lives across calls, not within them.

    Args:
        by_source:  source_id → its metadata rows
        sources:    selected call ids, in sorted order
        per_source: cap per call
        n:          global cap
        seed:       RNG seed

    Returns:
        selected rows, ordered by (source_id, segment_id)
    """
    rng = random.Random(seed)
    picked: list[dict] = []
    for source in sources:
        rows = sorted(by_source[source], key=lambda r: int(r["segment_id"]))
        picked.extend(rng.sample(rows, min(per_source, len(rows))))
    picked.sort(key=lambda r: (r["source_id"], int(r["segment_id"])))
    return picked[:n]


def fetch_source(source_id: str, cache_dir: Path) -> Path:
    """Download one call's chunked tarball into cache_dir, if it is not there."""
    import urllib.request

    dest = cache_dir / f"{source_id}.tar.gz"
    if dest.exists():
        return dest
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = f"{_BASE_URL}data/chunked/{source_id}.tar.gz"
    print(f"  fetching {source_id}.tar.gz …", flush=True)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)          # rename last, so an interrupted fetch is not cached
    return dest


def build(
    metadata_path: Path,
    cache_dir: Path,
    output_dir: Path,
    tokenizer_path: Path | None,
    n: int = 2000,
    n_sources: int = 40,
    per_source: int = 50,
    seed: int = 42,
    max_label_len: int | None = 41,
    max_duration_s: float = 30.0,
    download: bool = True,
) -> dict:
    """Select calls and segments, fetch the needed tarballs, write one shard.

    Args:
        metadata_path:  ESB baseline metadata.csv
        cache_dir:      where the per-call tarballs live
        output_dir:     destination for the shard, manifest and stats
        tokenizer_path: pruned tokenizer dir; needed when max_label_len is set
        n:              target segment count
        n_sources:      calls to draw from
        per_source:     segment cap per call
        seed:           selection seed
        max_label_len:  house eval filter, in tokens
        max_duration_s: drop segments longer than this (never trimmed)
        download:       fetch missing tarballs (False = use only what is cached)

    Returns:
        stats dict (also written to output_dir/stats.json)
    """
    encode = None
    if max_label_len is not None:
        if tokenizer_path is None:
            raise ValueError("--tokenizer is required unless --max-label-len 0")
        from data import PrunedTokenizer
        encode = PrunedTokenizer(tokenizer_path).encode

    with metadata_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[row["source_id"]].append(row)
    print(f"Metadata: {len(rows):,} segments across {len(by_source)} calls")

    sources = select_sources(list(by_source), n_sources, seed)
    selected = select_segments(by_source, sources, per_source, n, seed)
    print(f"Selected {len(selected):,} segments from {len(sources)} calls "
          f"(seed {seed})")

    writer = ShardWriter(output_dir / f"earnings22-{n}.tar")
    manifest: list[dict] = []
    n_long = n_too_many_tokens = n_nonascii = n_empty = 0
    n_out_of_conv = n_digits = n_disfluency = n_inaudible = 0
    total_duration = 0.0

    current_source: str | None = None
    members: dict[str, bytes] = {}

    for rec in selected:
        source = rec["source_id"]
        if source != current_source:
            # One tarball open at a time: the segments are ordered by call, so a
            # streaming read never revisits an archive.
            path = fetch_source(source, cache_dir) if download else cache_dir / f"{source}.tar.gz"
            if not path.exists():
                print(f"  [warn] {path} missing and --no-download — skipping call")
                current_source, members = source, {}
                continue
            with tarfile.open(path, "r:gz") as tf:
                members = {
                    m.name.rsplit("/", 1)[-1]: tf.extractfile(m).read()
                    for m in tf.getmembers()
                    if m.isfile() and m.name.endswith(".wav")
                }
            current_source = source
            print(f"  {source}: {len(members)} segments in archive", flush=True)

        wav_name = f"{rec['segment_id']}.wav"
        if wav_name not in members:
            continue

        formatted = fold_ascii(rec["sentence"])
        if not formatted:
            n_empty += 1
            continue
        if not formatted.isascii():
            n_nonascii += 1
            continue
        verbatim = spec6_normalize(formatted)
        if not verbatim:
            n_empty += 1
            continue

        if encode is not None:
            n_tok = max(len(encode(verbatim)), len(encode(formatted)))
            if n_tok > max_label_len:
                n_too_many_tokens += 1
                continue
        else:
            n_tok = 0

        audio, sr = load_audio_bytes(members[wav_name])
        audio = resample(audio, sr)
        duration_s = len(audio) / SAMPLE_RATE
        if duration_s > max_duration_s:
            n_long += 1
            continue

        key = f"e22-{source}-{rec['segment_id']}"
        mel = log_mel_spectrogram(torch.from_numpy(audio)).numpy().astype(np.float16)
        writer.add_sample(key, mel=mel, unformatted=verbatim,
                          formatted=formatted, audio=audio)

        ooc = out_of_convention(formatted)
        digits = bool(_DIGIT_RE.search(formatted))
        disfl = bool(_DISFLUENCY_RE.search(formatted))
        inaud = bool(_INAUDIBLE_RE.search(formatted))
        n_out_of_conv += ooc
        n_digits += digits
        n_disfluency += disfl
        n_inaudible += inaud
        total_duration += duration_s
        manifest.append({
            "key":               key,
            "dataset":           "earnings22",
            "source_id":         source,
            "segment_id":        rec["segment_id"],
            "start_s":           float(rec["start_ts"]),
            "end_s":             float(rec["end_ts"]),
            "duration_s":        round(duration_s, 4),
            "n_mel_frames":      int(mel.shape[1]),
            "unformatted":       verbatim,
            "formatted":         formatted,
            "n_label_tokens":    n_tok,
            "reference_source":  "human transcript",
            "has_formatted_ref": True,
            "out_of_convention": ooc,
            "has_digits":        digits,
            "has_disfluency":    disfl,
            "has_inaudible":     inaud,
        })

    writer.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for rec in manifest:
            f.write(json.dumps(rec) + "\n")

    stats = {
        "dataset":     "earnings22",
        "corpus":      "Earnings-22 (arXiv:2203.15591), ESB baseline segmentation",
        "source":      {"metadata": str(metadata_path), "cache": str(cache_dir)},
        "segmentation": (
            "reused from anton-l/earnings22_baseline_5_gram (per-segment wavs + "
            "start/end timestamps); no forced alignment performed here"
        ),
        "selection":   {
            "n_requested": n, "seed": seed,
            "calls_sampled": len(sources), "calls_in_corpus": len(by_source),
            "per_source_cap": per_source,
            "caveat": (
                f"{len(sources)} of {len(by_source)} calls. Earnings-22's accent "
                "breadth lives across calls, so this subset samples that breadth "
                "too — the number is 'far-domain audio from these calls', not "
                "'Earnings-22'."
            ),
        },
        "segments":    len(manifest),
        "total_hours": round(total_duration / 3600, 3),
        "shard":       str(writer.path),
        "reference_forms": ["verbatim", "formatted"],
        "reference_source": "human transcript",
        "corpus_notes": [
            f"{n_digits} of {len(manifest)} kept segments carry digits "
            f"({n_digits / max(len(manifest), 1):.1%}). FORMATTING_SPEC §6 does not "
            "expand digit-bearing tokens, so read the digit-free slice from "
            "tools/ood_report.py as the honest verbatim number on this corpus.",
            f"{n_out_of_conv} segments use punctuation outside the house formatted "
            "convention; flagged as out_of_convention, not dropped.",
            f"{n_disfluency} of {len(manifest)} segments carry filled pauses "
            f"({n_disfluency / max(len(manifest), 1):.1%}) and {n_inaudible} carry an "
            "<inaudible> marker. Neither exists in LibriSpeech's labels; both are a "
            "reference-convention floor, flagged (has_disfluency / has_inaudible) "
            "rather than cleaned out of a human transcript.",
        ],
        "dropped": {
            "over_max_duration":       n_long,
            "over_max_label_len":      n_too_many_tokens,
            "non_ascii_after_folding": n_nonascii,
            "empty_after_normalise":   n_empty,
        },
        "has_digits":        n_digits,
        "has_disfluency":    n_disfluency,
        "has_inaudible":     n_inaudible,
        "out_of_convention": n_out_of_conv,
    }
    with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSegments written: {len(manifest):,}  ({stats['total_hours']} h)  "
          f"→ {writer.path.name}")
    print(f"  dropped: {n_long} > {max_duration_s:g}s, "
          f"{n_too_many_tokens} > {max_label_len} tokens, "
          f"{n_nonascii} non-ascii, {n_empty} empty")
    print(f"  digit-bearing: {n_digits} ({n_digits / max(len(manifest), 1):.1%}) — "
          "score the digit-free slice")
    print(f"  filled pauses: {n_disfluency} ({n_disfluency / max(len(manifest), 1):.1%}), "
          f"<inaudible>: {n_inaudible} — reference-convention floor, flagged not cleaned")
    print(f"  from {len(sources)} of {len(by_source)} calls (see stats.json caveat)")
    return stats


def _self_test() -> None:
    """Selection determinism and an end-to-end build off a synthetic tarball."""
    rows = [
        {"source_id": f"call{s}", "segment_id": str(i),
         "sentence": f"Sentence {i}.", "start_ts": "0", "end_ts": "1"}
        for s in range(6) for i in range(10)
    ]
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_source[r["source_id"]].append(r)

    a = select_sources(list(by_source), 3, 42)
    assert a == select_sources(list(by_source)[::-1], 3, 42), "order of input mattered"
    assert a != select_sources(list(by_source), 3, 7), "seed had no effect"
    assert len(a) == 3 and a == sorted(a), a
    assert len(select_sources(list(by_source), 99, 42)) == 6, "must clamp to what exists"
    print("  [OK] select_sources: sorted, reproducible, seed-sensitive, clamped")

    segs = select_segments(by_source, a, per_source=4, n=100, seed=42)
    assert len(segs) == 12, len(segs)
    assert {r["source_id"] for r in segs} == set(a), segs
    from collections import Counter
    assert set(Counter(r["source_id"] for r in segs).values()) == {4}, "per-source cap"
    capped = select_segments(by_source, a, per_source=4, n=5, seed=42)
    assert len(capped) == 5, len(capped)
    assert segs == select_segments(by_source, a, per_source=4, n=100, seed=42)
    print("  [OK] select_segments: per-call cap, global cap, reproducible")

    import soundfile as sf
    from ood_shard import read_shard

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        cache = tmp_dir / "cache"
        cache.mkdir()
        sentences = {
            "0": "We expect 15 percent growth.",     # digits
            "1": 'He said "no comment" today.',      # out of convention
            "2": "Thank you for joining us.",
            "3": "And, uh, revenue was <inaudible> strong.",   # disfluency + inaudible
        }
        with tarfile.open(cache / "callX.tar.gz", "w:gz") as tf:
            for seg in sentences:
                wav = (np.sin(np.arange(24000) * 0.01) * 0.1).astype(np.float32)
                buf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(buf.name, wav, 24000, subtype="PCM_16")
                tf.add(buf.name, arcname=f"./callX/{seg}.wav")

        meta = tmp_dir / "metadata.csv"
        with meta.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["source_id", "segment_id", "file", "start_ts", "end_ts", "sentence"])
            for seg, text in sentences.items():
                w.writerow(["callX", seg, f"callX/{seg}.wav", "0", "1", text])

        out = tmp_dir / "shards"
        stats = build(meta, cache, out, tokenizer_path=None, n=4, n_sources=1,
                      per_source=4, seed=42, max_label_len=None, download=False)
        assert stats["segments"] == 4, stats
        assert stats["has_digits"] == 1, stats
        # 2, not 1: the <inaudible> marker's angle brackets are themselves
        # outside the house charset, so that segment is flagged both ways.
        assert stats["out_of_convention"] == 2, stats
        assert stats["has_disfluency"] == 1, stats
        assert stats["has_inaudible"] == 1, stats

        got = read_shard(out / "earnings22-4.tar")
        assert len(got) == 4, got.keys()
        for members in got.values():
            assert {"mel.npy", "unformatted.txt", "formatted.txt", "flac"} == set(members)
        recs = {json.loads(l)["segment_id"]: json.loads(l)
                for l in (out / "manifest.jsonl").read_text().splitlines()}
        assert recs["0"]["unformatted"] == "we expect 15 percent growth", recs["0"]
        assert recs["0"]["has_digits"] is True
        assert recs["1"]["out_of_convention"] is True
        assert recs["3"]["has_disfluency"] is True and recs["3"]["has_inaudible"] is True
        assert recs["2"]["has_disfluency"] is False
        assert all(r["reference_source"] == "human transcript" for r in recs.values())
        # 24 kHz source must land at 16 kHz in the mel and the flac alike.
        assert recs["0"]["n_mel_frames"] == 100, recs["0"]["n_mel_frames"]
        print("  [OK] end-to-end build: resample, both reference forms, digit flag")

    print("PASSED")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--metadata", type=Path, default=Path("data/ood/raw/e22-metadata.csv"))
    p.add_argument("--cache_dir", type=Path, default=Path("data/ood/raw/e22/"),
                   help="Where per-call tarballs are downloaded and kept.")
    p.add_argument("--download", action=argparse.BooleanOptionalAction, default=True,
                   help="Fetch missing metadata and per-call tarballs (default: on).")
    p.add_argument("--output_dir", type=Path, default=Path("data/ood_shards/earnings22/"))
    p.add_argument("--tokenizer", type=Path, default=Path("data/pruned_tokenizer/"))
    p.add_argument("--n", type=int, default=2000, help="Target segment count.")
    p.add_argument("--sources", type=int, default=40, dest="n_sources",
                   help="Calls to draw from, at ~80 MB each (default: 40 of 125).")
    p.add_argument("--per-source", type=int, default=50, dest="per_source")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-label-len", type=int, default=41, dest="max_label_len")
    p.add_argument("--max-duration", type=float, default=30.0, dest="max_duration_s")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)

    if args.self_test:
        _self_test()
        return

    if args.download and not args.metadata.exists():
        import urllib.request
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {_METADATA_URL}\n         → {args.metadata}")
        urllib.request.urlretrieve(_METADATA_URL, args.metadata)
    if not args.metadata.exists():
        p.error(f"{args.metadata} not found — pass --download")

    build(
        args.metadata, args.cache_dir, args.output_dir, args.tokenizer,
        n=args.n, n_sources=args.n_sources, per_source=args.per_source, seed=args.seed,
        max_label_len=args.max_label_len if args.max_label_len > 0 else None,
        max_duration_s=args.max_duration_s, download=args.download,
    )


if __name__ == "__main__":
    main()

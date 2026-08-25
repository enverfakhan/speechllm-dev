"""Build house-format eval shards from the TED-LIUM 3 test set (STM segments).

WHY THIS EXISTS
---------------
LibriSpeech is read speech from a single, narrow domain.  TED-LIUM 3 is
spontaneous lecture speech recorded in a hall: the same language, a different
speaking style, channel and vocabulary.  It is the cheapest available probe of
"does this model generalise off its training distribution at all".

SEGMENTATION
------------
TED-LIUM ships an STM file per talk giving the utterance boundaries; the
official test set is the 1,155 scoreable segments those boundaries define.  We
do not re-derive them.  The source read here is the HuggingFace mirror
``distil-whisper/tedlium-dev-test``, which is the ``LIUM/tedlium`` release-3
loader's output: audio already cut at the STM boundaries, with the segment
boundaries preserved in the ``id`` field
(``{talk}-{start_s}-{end_s}-<o,f0,{gender}>``).  Those boundaries land in the
manifest so any segment can be traced back to its talk and offset.

The 314 ``inter_segment_gap`` rows carry the STM's ``ignore_time_segment_in_scoring``
marker and are dropped, exactly as every published TED-LIUM number does — they
are the unlabelled audio *between* scoreable segments, not utterances.

REFERENCE TEXT
--------------
TED-LIUM references are already lowercase and unpunctuated, so this corpus can
only score the VERBATIM (unformatted) instruction.  One de-tokenisation is
applied: the STM writes clitics detached (``it 's``, ``we 're``, ``do n't``),
which no ASR system emits and which would otherwise count as an insertion on
every contraction in the set.  Joining them (``" '"`` → ``"'"``) is the standard
TED-LIUM recipe and matches this project's own unformatted convention, which
keeps apostrophes inside the word.

``{key}.formatted.txt`` is written as a COPY of the verbatim text purely so the
shard satisfies the three-member house format that data.build_sorted_eval_dataloader
requires.  **It is not a formatted reference and must never be scored** — the
manifest says so per sample (``has_formatted_ref: false``) and the summary
printed at the end repeats it.  Decode this set with ``--formats unformatted``.

OUTPUT
------
One .tar per label-length cap, in the house shard format plus one addition:

    {key}.mel.npy          float16 (80, T), model/whisper_encoder.log_mel_spectrogram
    {key}.unformatted.txt  verbatim reference (UTF-8)
    {key}.formatted.txt    copy of the above — NOT a formatted reference
    {key}.flac             16 kHz mono source audio  <- addition

The .flac member is what lets tools/run_whisper_control.py score the control
system on byte-identical audio through Whisper's own front end, instead of
re-cutting the segments itself.  Readers that group members by key and require
only the three house members (data.py, tools/make_eval_subset.py) ignore it.

Keys are ``ted-{talk}-{seq:04d}``: shard readers split a member name on the
FIRST ".", and TED-LIUM's own ids carry decimal timestamps, which would truncate
every key at the first offset digit.

USAGE
-----
    # fetch the source parquet (352 MB) and build both caps
    python tools/prepare_ood_tedlium.py \\
        --download \\
        --output_dir data/ood_shards/tedlium3-test/ \\
        --tokenizer  data/pruned_tokenizer/ \\
        --max-label-len 41 64

    # 50-utterance smoke shard
    python tools/prepare_ood_tedlium.py --parquet data/ood/raw/tedlium3-test.parquet \\
        --output_dir data/ood_shards/smoke-tedlium/ --limit 50

    python tools/prepare_ood_tedlium.py --self-test
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.whisper_encoder import SAMPLE_RATE, log_mel_spectrogram

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ood_shard import ShardWriter, load_audio_bytes, resample  # noqa: E402

# The HF mirror of LIUM/tedlium release-3 dev+test, already STM-segmented.
_SOURCE_URL = (
    "https://huggingface.co/datasets/distil-whisper/tedlium-dev-test/resolve/main/"
    "data/test-00000-of-00001-a778c14684971c41.parquet"
)

# STM marker on the unlabelled audio between scoreable segments.
_IGNORE_MARKER = "ignore_time_segment_in_scoring"

# id = "{talk}-{start}-{end}-<o,f0,{gender}>"
_ID_RE = re.compile(r"^(?P<talk>.+?)-(?P<start>[0-9.]+)-(?P<end>[0-9.]+)-<")


def clean_reference(text: str) -> str:
    """De-tokenise an STM reference into this project's verbatim convention.

    TED-LIUM writes clitics as separate tokens ("it 's", "we 're", "do n't").
    No ASR system emits that, and this project's unformatted labels keep the
    apostrophe inside the word, so leaving it would score one insertion per
    contraction against every system alike — noise that swamps the domain
    signal this set exists to measure.  Joining is the standard TED-LIUM recipe.

    Args:
        text: raw STM segment text (already lowercase and unpunctuated)

    Returns:
        whitespace-collapsed reference with clitics rejoined
    """
    return " ".join(text.replace(" '", "'").split())


def parse_segment_id(seg_id: str) -> tuple[str, float, float]:
    """Split a TED-LIUM segment id into (talk, start_s, end_s).

    Raises:
        ValueError: the id does not carry the expected boundary encoding — a
            silent fallback would put unattributable segments in the manifest.
    """
    m = _ID_RE.match(seg_id)
    if not m:
        raise ValueError(f"Unrecognised TED-LIUM segment id: {seg_id!r}")
    return m.group("talk"), float(m.group("start")), float(m.group("end"))


def build(
    parquet_path: Path,
    output_dir: Path,
    tokenizer_path: Path | None,
    max_label_lens: list[int],
    max_duration_s: float = 30.0,
    limit: int | None = None,
) -> dict:
    """Read the source parquet and write one shard per label-length cap.

    Args:
        parquet_path:    downloaded distil-whisper/tedlium-dev-test test parquet
        output_dir:      destination directory for the .tar shards + manifest
        tokenizer_path:  pruned tokenizer dir; required unless max_label_lens is empty
        max_label_lens:  one shard per cap, in tokens (e.g. [41, 64]); [] = no cap
        max_duration_s:  segments longer than this are dropped, never trimmed
        limit:           stop after this many kept segments (smoke runs)

    Returns:
        stats dict (also written to output_dir/stats.json)
    """
    import pyarrow.parquet as pq

    encode = None
    if max_label_lens:
        if tokenizer_path is None:
            raise ValueError("--tokenizer is required unless --max-label-len is empty")
        from data import PrunedTokenizer
        encode = PrunedTokenizer(tokenizer_path).encode

    table = pq.read_table(parquet_path).to_pydict()
    n_source = len(table["id"])
    print(f"Source segments: {n_source:,}")

    caps: list[int | None] = [*sorted(max_label_lens)] or [None]
    writers = {
        cap: ShardWriter(
            output_dir / f"tedlium3-test{'' if cap is None else f'-le{cap}'}.tar"
        )
        for cap in caps
    }

    n_ignore = n_long = n_kept = 0
    per_cap_kept = {cap: 0 for cap in caps}
    label_token_lens: list[int] = []
    manifest: list[dict] = []
    seq_per_talk: dict[str, int] = {}

    for i in range(n_source):
        raw_text = table["text"][i]
        if raw_text.strip() == _IGNORE_MARKER:
            n_ignore += 1
            continue

        seg_id = table["id"][i]
        talk, start_s, end_s = parse_segment_id(seg_id)

        audio, sr = load_audio_bytes(table["audio"][i]["bytes"])
        audio = resample(audio, sr)
        duration_s = len(audio) / SAMPLE_RATE

        # Skip, never trim: cutting audio without cutting the reference is label noise.
        if duration_s > max_duration_s:
            n_long += 1
            continue

        reference = clean_reference(raw_text)
        if not reference:
            continue

        seq = seq_per_talk.get(talk, 0)
        seq_per_talk[talk] = seq + 1
        key = f"ted-{talk}-{seq:04d}"

        n_tok = len(encode(reference)) if encode is not None else 0
        label_token_lens.append(n_tok)

        mel = log_mel_spectrogram(torch.from_numpy(audio)).numpy().astype(np.float16)

        wrote_into: list[int | None] = []
        for cap in caps:
            if cap is not None and n_tok > cap:
                continue
            writers[cap].add_sample(
                key,
                mel=mel,
                unformatted=reference,
                # NOT a formatted reference — see the module docstring.
                formatted=reference,
                audio=audio,
            )
            per_cap_kept[cap] += 1
            wrote_into.append(cap)

        n_kept += 1
        manifest.append({
            "key":              key,
            "dataset":          "tedlium3-test",
            "source_id":        seg_id,
            "talk":             talk,
            "start_s":          start_s,
            "end_s":            end_s,
            "duration_s":       round(duration_s, 4),
            "n_mel_frames":     int(mel.shape[1]),
            "unformatted":      reference,
            "n_label_tokens":   n_tok,
            "has_formatted_ref": False,
            "shards":           [("uncapped" if c is None else f"le{c}") for c in wrote_into],
        })

        if n_kept % 200 == 0:
            print(f"  processed {n_kept:,} segments …", flush=True)
        if limit is not None and n_kept >= limit:
            break

    for w in writers.values():
        w.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for rec in manifest:
            f.write(json.dumps(rec) + "\n")

    stats = {
        "dataset":            "tedlium3-test",
        "source":             str(parquet_path),
        "source_segments":    n_source,
        "dropped_ignore_marker": n_ignore,
        "dropped_over_30s":   n_long,
        "scoreable_segments": n_kept,
        "total_hours":        round(sum(r["duration_s"] for r in manifest) / 3600, 3),
        "reference_forms":    ["verbatim"],
        "formatted_scoreable": False,
        "label_token_len": {
            "mean": round(sum(label_token_lens) / len(label_token_lens), 1)
                    if label_token_lens else None,
            "max":  max(label_token_lens) if label_token_lens else None,
        },
        "shards": {
            ("uncapped" if cap is None else f"le{cap}"): {
                "path": str(writers[cap].path),
                "kept": per_cap_kept[cap],
                "kept_fraction": round(per_cap_kept[cap] / n_kept, 4) if n_kept else 0.0,
            }
            for cap in caps
        },
    }
    with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nScoreable segments:      {n_kept:,}  ({stats['total_hours']} h)")
    print(f"  dropped (ignore marker): {n_ignore:,}")
    print(f"  dropped (> {max_duration_s:g}s):        {n_long:,}")
    for cap in caps:
        name = "uncapped" if cap is None else f"≤{cap} tokens"
        frac = per_cap_kept[cap] / n_kept if n_kept else 0.0
        print(f"  {name:<14} {per_cap_kept[cap]:>5,} kept  ({frac:.1%})  "
              f"→ {writers[cap].path.name}")
    print("\nNOTE: TED-LIUM references are normalised — VERBATIM MODE ONLY.")
    print("      formatted.txt is a copy of the verbatim text; do not score it.")
    print("      Decode with:  --formats unformatted")
    return stats


def _self_test() -> None:
    """Reference cleaning, id parsing, and a synthetic end-to-end shard build."""
    assert clean_reference("if i can leave you it 's the whole") == "if i can leave you it's the whole"
    assert clean_reference("we 're  at  their most  frail") == "we're at their most frail"
    assert clean_reference("  spaced   out  ") == "spaced out"
    print("  [OK] clean_reference joins clitics and collapses whitespace")

    talk, s, e = parse_segment_id("GaryFlake-16.06-27.12-<o,f0,male>")
    assert (talk, s, e) == ("GaryFlake", 16.06, 27.12), (talk, s, e)
    talk, s, e = parse_segment_id("Al-Gore_2009-1.5-9.25-<o,f0,male>")
    assert talk == "Al-Gore_2009", talk
    try:
        parse_segment_id("no-boundaries-here")
    except ValueError:
        pass
    else:
        raise AssertionError("parse_segment_id accepted a malformed id")
    print("  [OK] parse_segment_id handles hyphenated talk names and fails loudly")

    # ── Synthetic parquet → shard round trip ─────────────────────────────────
    import pyarrow as pa
    import pyarrow.parquet as pq
    import soundfile as sf
    import tarfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        rows = []
        for i in range(4):
            n = SAMPLE_RATE * (1 + i)
            wav = (np.sin(np.arange(n) * 0.01) * 0.1).astype(np.float32)
            buf = io.BytesIO()
            sf.write(buf, wav, SAMPLE_RATE, format="WAV", subtype="PCM_16")
            rows.append({
                "audio": {"bytes": buf.getvalue(), "path": None},
                "text":  "one two three four five six" if i else _IGNORE_MARKER,
                "id":    f"Talk{i}-0.0-{1 + i}.0-<o,f0,male>",
            })
        pa.parquet = pq
        tbl = pa.Table.from_pylist(rows)
        pq_path = tmp_dir / "src.parquet"
        pq.write_table(tbl, pq_path)

        out = tmp_dir / "shards"
        # no tokenizer needed: an empty cap list skips the label-length filter
        stats = build(
            pq_path, out, tokenizer_path=None, max_label_lens=[], limit=None,
        )
        assert stats["dropped_ignore_marker"] == 1, stats
        assert stats["scoreable_segments"] == 3, stats

        shard = out / "tedlium3-test.tar"
        with tarfile.open(shard) as tf:
            names = tf.getnames()
        keys = {n.split(".", 1)[0] for n in names}
        assert len(keys) == 3, keys
        for k in keys:
            for ext in ("mel.npy", "unformatted.txt", "formatted.txt", "flac"):
                assert f"{k}.{ext}" in names, f"missing {k}.{ext}"
        assert all("." not in k.split("ted-", 1)[1] for k in keys), (
            f"key carries a '.', which shard readers truncate on: {keys}"
        )
        manifest = [json.loads(l) for l in (out / "manifest.jsonl").read_text().splitlines()]
        assert len(manifest) == 3 and all(not r["has_formatted_ref"] for r in manifest)
        print("  [OK] end-to-end build: ignore rows dropped, 4 members per key, dot-free keys")

    print("PASSED")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--parquet", type=Path, default=Path("data/ood/raw/tedlium3-test.parquet"),
                   help="Source parquet (default: data/ood/raw/tedlium3-test.parquet).")
    p.add_argument("--download", action="store_true",
                   help="Fetch the source parquet to --parquet if it is not already there.")
    p.add_argument("--output_dir", type=Path, default=Path("data/ood_shards/tedlium3-test/"))
    p.add_argument("--tokenizer", type=Path, default=Path("data/pruned_tokenizer/"),
                   help="Pruned tokenizer dir, for the label-length filter.")
    p.add_argument("--max-label-len", type=int, nargs="*", default=[41, 64],
                   dest="max_label_lens", metavar="L",
                   help="Write one shard per cap (default: 41 64). Empty = one uncapped shard.")
    p.add_argument("--max-duration", type=float, default=30.0, dest="max_duration_s")
    p.add_argument("--limit", type=int, default=None, help="Stop after N segments (smoke runs).")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)

    if args.self_test:
        _self_test()
        return

    if args.download and not args.parquet.exists():
        import urllib.request
        args.parquet.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {_SOURCE_URL}\n         → {args.parquet}")
        urllib.request.urlretrieve(_SOURCE_URL, args.parquet)

    if not args.parquet.exists():
        p.error(f"{args.parquet} not found — pass --download or --parquet")

    build(
        args.parquet, args.output_dir, args.tokenizer,
        args.max_label_lens, args.max_duration_s, args.limit,
    )


if __name__ == "__main__":
    main()

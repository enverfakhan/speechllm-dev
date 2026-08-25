"""Build house-format eval shards from a Common Voice EN test subset.

WHY THIS EXISTS
---------------
LibriSpeech is studio-read audiobook English from a small pool of volunteers.
Common Voice is crowd-recorded on consumer microphones by speakers with a wide
range of accents.  It varies the two axes LibriSpeech holds fixed — speaker
population and channel — while keeping the task identical.

It is also the only OOD set here with a natively CASED AND PUNCTUATED reference,
which makes it the one out-of-distribution test of the FORMATTED instruction.
TED-LIUM and Earnings-22 references are normalised and can score verbatim only.

TWO REFERENCE FORMS PER CLIP
----------------------------
    formatted.txt    the native sentence, ASCII-folded (see below)
    unformatted.txt  FORMATTING_SPEC.md §6 normalisation of the same sentence

§6 is imported from tools/label_formatted.py rather than reimplemented — it is
the same function the labelling validator and the WER decomposition use, and a
third copy would be a third thing to drift.

REFERENCES ARE PROMPT TEXT, NOT TRANSCRIPTS
-------------------------------------------
A Common Voice clip is someone reading a sentence off a screen.  The stored
sentence is the PROMPT, not a transcription of what was said, so it diverges
from the audio whenever a reader misreads, self-corrects, or reads a variant.
Every manifest record carries ``reference_source: "prompt"`` to keep that
visible downstream; a few points of WER on this set are the corpus, not the
model, and the same floor applies to the paired control system.

TWO CORPUS FACTS THAT SHAPE WHAT THIS SET CAN MEASURE
-----------------------------------------------------
1. Common Voice forbids digits in its prompts — the en test split contains ZERO
   digit-bearing sentences.  So the formatted-mode test here probes casing and
   punctuation only; the FORMATTING_SPEC §4 numeral rule is untested by it.
2. About 3.6% of sentences use punctuation outside the house formatted
   convention (§3 permits ``. , ? ! ; : '`` and the dash; the corpus also uses
   quotation marks and parentheses).  Those clips are KEPT — dropping them would
   quietly select the corpus toward the model — and flagged per record as
   ``out_of_convention: true`` so the slice can be separated in the report.  A
   model trained on the house convention cannot emit a quotation mark, so those
   clips carry a guaranteed error that measures the convention, not the ASR.

Non-ASCII typography (curly quotes, en/em dashes, the ellipsis character) is
folded to its ASCII equivalent, matching §1's "labels stay ASCII".  Clips still
non-ASCII after folding (5 of 16,393) are dropped as out-of-scope for an
English ASCII vocabulary.

OUTPUT
------
The house shard format plus a ``{key}.flac`` member — see tools/ood_shard.py for
the layout and why the audio rides along.

USAGE
-----
    # download the sources (711 MB tar + 5 MB tsv), then build
    python tools/prepare_ood_commonvoice.py --download \\
        --output_dir data/ood_shards/commonvoice-en-test/ \\
        --tokenizer  data/pruned_tokenizer/ --n 5000 --seed 42

    python tools/prepare_ood_commonvoice.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import sys
import tarfile
import tempfile
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

_BASE_URL = (
    "https://huggingface.co/datasets/fsicoli/common_voice_17_0/resolve/main/"
)
_AUDIO_URL = _BASE_URL + "audio/en/test/en_test_0.tar"
_TSV_URL   = _BASE_URL + "transcript/en/test.tsv"

# Typography → ASCII.  FORMATTING_SPEC §1: labels stay ASCII.
_FOLD: dict[str, str] = {
    "’": "'", "‘": "'", "ʼ": "'",   # curly / modifier apostrophes
    "“": '"', "”": '"',                   # curly double quotes
    "–": "-", "—": "-",                   # en / em dash
    "…": "...",                                 # ellipsis
    " ": " ",                                   # nbsp
}

# FORMATTING_SPEC §2-§4: the characters a house formatted label may contain.
_HOUSE_CHARSET = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,?!;:'-"
)


def fold_ascii(text: str) -> str:
    """Replace non-ASCII typography with its ASCII equivalent."""
    for src, dst in _FOLD.items():
        text = text.replace(src, dst)
    return " ".join(text.split())


def out_of_convention(text: str) -> bool:
    """True when the sentence uses punctuation the house formatted spec forbids.

    Not a rejection — a label.  §3 permits ``. , ? ! ; : '`` and the dash;
    Common Voice also uses quotation marks and parentheses, which a model trained
    on the house convention will never produce.
    """
    return any(ch not in _HOUSE_CHARSET for ch in text)


def select_clips(
    rows: list[dict],
    available: set[str],
    n: int,
    seed: int,
    oversample: float = 1.5,
) -> list[dict]:
    """Draw a reproducible candidate pool of clips present in the audio tar.

    Sorted before sampling so the draw depends only on (corpus, n, seed) and not
    on tsv row order.  Oversampled because clips are still dropped downstream for
    duration and label length; the caller takes the first n survivors IN SAMPLED
    ORDER, which keeps the final set deterministic without a second sampling pass.

    Args:
        rows:       parsed test.tsv records
        available:  clip filenames actually present in the audio tar
        n:          target subset size
        seed:       RNG seed
        oversample: pool size multiplier

    Returns:
        candidate records, in sampled order
    """
    by_path = {r["path"]: r for r in rows if r["path"] in available}
    ordered = sorted(by_path)
    pool = min(len(ordered), math.ceil(n * oversample))
    return [by_path[p] for p in random.Random(seed).sample(ordered, pool)]


def build(
    tar_path: Path,
    tsv_path: Path,
    output_dir: Path,
    tokenizer_path: Path | None,
    n: int = 5000,
    seed: int = 42,
    max_label_len: int | None = 41,
    max_duration_s: float = 30.0,
) -> dict:
    """Build one shard holding a deterministic n-clip subset of CV EN test.

    Args:
        tar_path:       en_test_0.tar of mp3 clips
        tsv_path:       test.tsv transcript file
        output_dir:     destination for the shard, manifest and stats
        tokenizer_path: pruned tokenizer dir; needed when max_label_len is set
        n:              subset size
        seed:           selection seed
        max_label_len:  drop clips whose longer reference exceeds this many tokens
        max_duration_s: drop clips longer than this (never trimmed)

    Returns:
        stats dict (also written to output_dir/stats.json)
    """
    encode = None
    if max_label_len is not None:
        if tokenizer_path is None:
            raise ValueError("--tokenizer is required unless --max-label-len 0")
        from data import PrunedTokenizer
        encode = PrunedTokenizer(tokenizer_path).encode

    with tsv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    print(f"Transcript rows: {len(rows):,}")

    # Read the audio tar once into memory-mapped members; 711 MB of mp3 is
    # small enough to index by name and random-access without re-scanning.
    tf = tarfile.open(tar_path, "r")
    members = {m.name.rsplit("/", 1)[-1]: m for m in tf.getmembers() if m.name.endswith(".mp3")}
    print(f"Clips in audio tar: {len(members):,}")

    candidates = select_clips(rows, set(members), n, seed)
    print(f"Candidate pool: {len(candidates):,} (target {n:,})")

    writer = ShardWriter(output_dir / f"commonvoice-en-test-{n}.tar")
    manifest: list[dict] = []
    n_nonascii = n_long = n_too_many_tokens = n_empty = 0
    n_out_of_conv = 0
    total_duration = 0.0

    for rec in candidates:
        if len(manifest) >= n:
            break

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

        fobj = tf.extractfile(members[rec["path"]])
        assert fobj is not None, rec["path"]
        audio, sr = load_audio_bytes(fobj.read())
        audio = resample(audio, sr)
        duration_s = len(audio) / SAMPLE_RATE
        if duration_s > max_duration_s:
            n_long += 1
            continue

        key = f"cv-{rec['path'].removesuffix('.mp3')}"
        mel = log_mel_spectrogram(torch.from_numpy(audio)).numpy().astype(np.float16)
        writer.add_sample(key, mel=mel, unformatted=verbatim,
                          formatted=formatted, audio=audio)

        ooc = out_of_convention(formatted)
        n_out_of_conv += ooc
        total_duration += duration_s
        manifest.append({
            "key":               key,
            "dataset":           "commonvoice-en-test",
            "source_path":       rec["path"],
            "client_id":         rec["client_id"],
            "duration_s":        round(duration_s, 4),
            "n_mel_frames":      int(mel.shape[1]),
            "unformatted":       verbatim,
            "formatted":         formatted,
            "n_label_tokens":    n_tok,
            # The stored sentence is the prompt the speaker READ, not a
            # transcript of what they said; misreads show up as WER floor.
            "reference_source":  "prompt",
            "has_formatted_ref": True,
            "out_of_convention": ooc,
            "accents":           rec.get("accents", ""),
            "age":               rec.get("age", ""),
            "gender":            rec.get("gender", ""),
            "up_votes":          rec.get("up_votes", ""),
            "down_votes":        rec.get("down_votes", ""),
        })
        if len(manifest) % 500 == 0:
            print(f"  built {len(manifest):,} clips …", flush=True)

    writer.close()
    tf.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for rec in manifest:
            f.write(json.dumps(rec) + "\n")

    stats = {
        "dataset":         "commonvoice-en-test",
        "source":          {"audio": str(tar_path), "transcript": str(tsv_path)},
        "corpus":          "Common Voice 17.0 en / test (CC-0)",
        "selection":       {"n_requested": n, "seed": seed, "pool": len(candidates)},
        "clips":           len(manifest),
        "total_hours":     round(total_duration / 3600, 3),
        "shard":           str(writer.path),
        "reference_forms": ["verbatim", "formatted"],
        "reference_source": "prompt",
        "manifest_note": (
            "Common Voice references are the PROMPT TEXT the speaker was asked to "
            "read, not a transcript of the audio. They diverge on misreads and "
            "self-corrections; both the model and the control system pay that floor."
        ),
        "corpus_notes": [
            "Zero digit-bearing sentences in en/test: the FORMATTING_SPEC §4 "
            "numeral rule is NOT exercised by this set.",
            f"{n_out_of_conv} of {len(manifest)} kept clips use punctuation outside "
            "the house formatted convention (mostly quotation marks); flagged per "
            "record as out_of_convention.",
        ],
        "dropped": {
            "non_ascii_after_folding": n_nonascii,
            "over_max_duration":       n_long,
            "over_max_label_len":      n_too_many_tokens,
            "empty_after_normalise":   n_empty,
        },
        "out_of_convention": n_out_of_conv,
    }
    with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nClips written:  {len(manifest):,}  ({stats['total_hours']} h)  "
          f"→ {writer.path.name}")
    print(f"  dropped: {n_nonascii} non-ascii, {n_long} > {max_duration_s:g}s, "
          f"{n_too_many_tokens} > {max_label_len} tokens, {n_empty} empty")
    print(f"  out-of-convention punctuation: {n_out_of_conv} "
          f"({n_out_of_conv / max(len(manifest), 1):.1%}) — flagged, not dropped")
    print("\nNOTE: references are PROMPT text, not transcripts (see stats.json).")
    return stats


def _self_test() -> None:
    """Folding, convention flagging, deterministic selection, and a build."""
    assert fold_ascii("Don’t “stop” – now…") == "Don't \"stop\" - now..."
    assert fold_ascii("  a   b  ") == "a b"
    print("  [OK] fold_ascii maps typography to ASCII")

    assert out_of_convention('He said "hi".') is True
    assert out_of_convention("Don't stop; go - now!") is False
    assert out_of_convention("A (parenthetical).") is True
    print("  [OK] out_of_convention flags quotes and parens, allows house punctuation")

    # §6 comes from label_formatted — spot-check the contract this tool relies on.
    assert spec6_normalize("Don't stop; go - now!") == "dont stop go now"
    print("  [OK] §6 normalisation imported from label_formatted")

    rows = [{"path": f"c{i}.mp3", "sentence": f"s{i}", "client_id": "x"} for i in range(20)]
    avail = {r["path"] for r in rows}
    a = select_clips(rows, avail, n=5, seed=42)
    b = select_clips(rows, avail, n=5, seed=42)
    c = select_clips(rows, avail, n=5, seed=7)
    assert [r["path"] for r in a] == [r["path"] for r in b], "selection not reproducible"
    assert [r["path"] for r in a] != [r["path"] for r in c], "seed had no effect"
    assert len(a) == math.ceil(5 * 1.5), len(a)
    missing = select_clips(rows, {"c0.mp3", "c1.mp3"}, n=5, seed=42)
    assert {r["path"] for r in missing} <= {"c0.mp3", "c1.mp3"}, missing
    print("  [OK] select_clips is reproducible, seed-sensitive, tar-aware")

    # ── Synthetic sources → shard round trip ─────────────────────────────────
    import soundfile as sf
    from ood_shard import read_shard

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        tar_path = tmp_dir / "audio.tar"
        sentences = [
            "Joe Keaton disapproved of films.",
            "Don’t you understand me?",
            'She said "no" and left.',
            "A short one.",
        ]
        with tarfile.open(tar_path, "w") as tf:
            for i in range(len(sentences)):
                wav = (np.sin(np.arange(SAMPLE_RATE) * 0.01) * 0.1).astype(np.float32)
                buf = io.BytesIO()
                sf.write(buf, wav, SAMPLE_RATE, format="WAV", subtype="PCM_16")
                data = buf.getvalue()
                info = tarfile.TarInfo(name=f"en_test_0/c{i}.mp3")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))

        tsv_path = tmp_dir / "test.tsv"
        with tsv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["client_id", "path", "sentence", "up_votes", "down_votes",
                        "age", "gender", "accents"])
            for i, s in enumerate(sentences):
                w.writerow([f"cid{i}", f"c{i}.mp3", s, "2", "0", "", "", ""])

        out = tmp_dir / "shards"
        stats = build(tar_path, tsv_path, out, tokenizer_path=None,
                      n=4, seed=42, max_label_len=None)
        assert stats["clips"] == 4, stats
        assert stats["out_of_convention"] == 1, stats

        got = read_shard(out / "commonvoice-en-test-4.tar")
        assert len(got) == 4, got.keys()
        for members in got.values():
            assert {"mel.npy", "unformatted.txt", "formatted.txt", "flac"} == set(members)
        recs = {json.loads(l)["source_path"]: json.loads(l)
                for l in (out / "manifest.jsonl").read_text().splitlines()}
        assert recs["c1.mp3"]["formatted"] == "Don't you understand me?"
        assert recs["c1.mp3"]["unformatted"] == "dont you understand me"
        assert recs["c2.mp3"]["out_of_convention"] is True
        assert all(r["reference_source"] == "prompt" for r in recs.values())
        print("  [OK] end-to-end build: both reference forms, prompt flag, 4 members")

    print("PASSED")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tar", type=Path, default=Path("data/ood/raw/cv17-en-test.tar"))
    p.add_argument("--tsv", type=Path, default=Path("data/ood/raw/cv17-en-test.tsv"))
    p.add_argument("--download", action="store_true",
                   help="Fetch the sources to --tar/--tsv if they are not already there.")
    p.add_argument("--output_dir", type=Path,
                   default=Path("data/ood_shards/commonvoice-en-test/"))
    p.add_argument("--tokenizer", type=Path, default=Path("data/pruned_tokenizer/"))
    p.add_argument("--n", type=int, default=5000, help="Subset size (default: 5000).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-label-len", type=int, default=41, dest="max_label_len",
                   help="House eval filter, in tokens; 0 disables (default: 41).")
    p.add_argument("--max-duration", type=float, default=30.0, dest="max_duration_s")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)

    if args.self_test:
        _self_test()
        return

    if args.download:
        import urllib.request
        for url, dest in ((_AUDIO_URL, args.tar), (_TSV_URL, args.tsv)):
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {url}\n         → {dest}")
            urllib.request.urlretrieve(url, dest)

    for path, flag in ((args.tar, "--tar"), (args.tsv, "--tsv")):
        if not path.exists():
            p.error(f"{path} not found — pass --download or point {flag} at it")

    build(
        args.tar, args.tsv, args.output_dir, args.tokenizer,
        n=args.n, seed=args.seed,
        max_label_len=args.max_label_len if args.max_label_len > 0 else None,
        max_duration_s=args.max_duration_s,
    )


if __name__ == "__main__":
    main()

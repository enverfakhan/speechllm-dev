"""Paired Whisper-small control decode over the OOD eval shards.

WHY THIS EXISTS
---------------
An absolute WER on this project's un-normalized house convention is not
comparable to any published number: the references are its own, the scoring
applies no normalisation, and the eval sets are filtered by label length.  The
only defensible out-of-distribution claim is therefore a **Δ against a known
system measured on exactly the same segments under exactly the same scoring**.

openai/whisper-small is the right control: same 88M encoder this project
borrows, plus the 153M decoder it discarded, so the Δ isolates "what the
SpeechLLM stack does with those audio features" rather than "how good is the
front end".  Greedy, no external LM, no beam search — the same decoding
discipline utils/generate.py uses.

SAME SEGMENTS, NOT SAME MEL
---------------------------
The control reads the ``{key}.flac`` member the prepare_ood_* tools store in
each shard — streamed one batch at a time, never the whole corpus at once — so
it scores byte-identical audio.  It does NOT reuse the stored
mel: Whisper's own front end computes its log-mel over the 30-second padded
signal, so its ``log_spec.max() - 8`` floor is taken over a different support
than this project's natural-length mel.  Feeding our mel would decode Whisper
slightly off its training distribution and understate the control — a bias in
the wrong direction, since a weaker control flatters the model under test.

TWO SCORINGS, BECAUSE WHISPER EMITS ONE STYLE
----------------------------------------------
Whisper writes cased, punctuated text.  Each hypothesis is therefore paired
twice, and one row is written per pairing:

    type="formatted"    raw output vs the formatted reference, un-normalized —
                        the house convention, already symmetric because both
                        systems are judged on the style they emit.
    type="unformatted"  §6 applied to the hypothesis AND to the reference.

The reference is normalised too, not just the hypothesis.  Normalising one side
only is the trap: this project's verbatim references keep apostrophes inside the
word ("it\'s"), §6 deletes them ("its"), and a control whose hypothesis went
through §6 while the reference did not would take one error per contraction it
transcribed correctly — three whole points of WER on TED-LIUM.  §6 on both sides
is the only arrangement under which the pairing measures words.

§6 is FORMATTING_SPEC.md's normalisation, imported from tools/label_formatted.py
so the control is normalised by the very function that defines the convention.
It is idempotent, so tools/ood_report.py re-applying it when it scores our model
against the same references leaves these rows unchanged — the two systems stay
comparable.  ``hypothesis_raw`` and ``reference_raw`` keep the originals.

Rows are written in tools/run_wer.py's JSONL schema (checkpoint, step, key,
split, type, reference, hypothesis, wer) plus ``dataset``, ``system``,
``hypothesis_raw`` and ``reference_raw``, so tools/analyze_slices.py and tools/count_degeneracies.py
read the control's output with no changes — the control gets the same
degeneracy and slice treatment as the model under test.

USAGE
-----
    python tools/run_whisper_control.py \\
        --shard data/ood_shards/tedlium3-test/tedlium3-test-le41.tar \\
        --output out/ood-tedlium-le41-whisper.jsonl --formats unformatted

    python tools/run_whisper_control.py \\
        --shard data/ood_shards/commonvoice-en-test/commonvoice-en-test-5000.tar \\
        --output out/ood-commonvoice-whisper.jsonl

    # 50-clip smoke run
    python tools/run_whisper_control.py --shard ... --output ... --limit 50

    python tools/run_whisper_control.py --self-test
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_formatted import normalize as spec6_normalize  # noqa: E402

DEFAULT_MODEL = "openai/whisper-small"

# Reference member ← pairing name.  "unformatted" is this repo's name for the
# verbatim mode; keeping it makes the rows drop straight into analyze_slices.
_REF_MEMBER = {"unformatted": "unformatted.txt", "formatted": "formatted.txt"}


def compute_wer(reference: str, hypothesis: str) -> float:
    """Single-pair WER through the same jiwer path as utils/evaluate.py.

    A zero-length reference makes WER undefined, so it yields NaN rather than
    killing a decode whose cost is already sunk.
    """
    import jiwer

    if not reference.strip():
        return float("nan")
    return jiwer.wer([reference], [hypothesis])


class ShardAudio:
    """Lazy reader over one eval shard: metadata up front, audio per batch.

    Deliberately NOT ood_shard.read_shard, which pulls every member into memory.
    A shard's mel arrays are ~10x its audio and this tool never touches them, so
    reading the whole tar costs gigabytes to use a fraction: the 5,000-clip
    Common Voice shard OOM-killed this process the first time it was written that
    way.  Here the tar stays open, only member handles and reference text are
    held, and audio is decoded one batch at a time.

    Sample order mirrors data.build_sorted_eval_dataloader — ascending by audio
    length, so a batch holds similar-length clips and the order is reproducible.
    """

    def __init__(self, shard: Path, limit: int | None = None) -> None:
        import soundfile as sf

        self.path = Path(shard)
        self._tf = tarfile.open(self.path, "r")

        members: dict[str, tarfile.TarInfo] = {}
        text: dict[str, dict[str, str]] = {}
        for member in self._tf.getmembers():
            dot = member.name.find(".")
            if dot < 0:
                continue
            key, ext = member.name[:dot], member.name[dot + 1:]
            if ext == "flac":
                members[key] = member
            elif ext in ("unformatted.txt", "formatted.txt"):
                fobj = self._tf.extractfile(member)
                if fobj is not None:
                    text.setdefault(key, {})[ext[:-4]] = fobj.read().decode("utf-8")

        if not members:
            self._tf.close()
            raise ValueError(
                f"{self.path}: no .flac members. The control must score the same "
                "audio the model does; rebuild the shard with a "
                "tools/prepare_ood_*.py tool, which stores it."
            )

        samples: list[dict] = []
        for key in sorted(members):
            fobj = self._tf.extractfile(members[key])
            assert fobj is not None, key
            # Header read only — the frames count is what the sort needs, and
            # the decoded signal would be the expensive part to keep.
            info = sf.info(io.BytesIO(fobj.read()))
            samples.append({
                "key":         key,
                "unformatted": text.get(key, {}).get("unformatted", ""),
                "formatted":   text.get(key, {}).get("formatted", ""),
                "n_frames":    info.frames,
                "sample_rate": int(info.samplerate),
                "_member":     members[key],
            })

        samples.sort(key=lambda s: s["n_frames"])
        self.samples = samples[:limit] if limit is not None else samples

    def audio(self, sample: dict) -> "np.ndarray":
        """Decode one sample's audio; nothing is cached."""
        import numpy as np
        import soundfile as sf

        fobj = self._tf.extractfile(sample["_member"])
        assert fobj is not None, sample["key"]
        data, _ = sf.read(io.BytesIO(fobj.read()), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=-1)
        return np.ascontiguousarray(data)

    def close(self) -> None:
        self._tf.close()

    def __len__(self) -> int:
        return len(self.samples)


def transcribe(
    shard: "ShardAudio",
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 16,
    device: str | None = None,
    progress_interval: float | None = 30.0,
) -> list[str]:
    """Greedy-decode every sample with the full Whisper model.

    Audio is decoded one batch at a time straight off the shard, so peak host
    memory is a batch of waveforms rather than the whole corpus.

    Args:
        shard:             open ShardAudio over the eval shard
        model_name:        HF model id (default openai/whisper-small)
        batch_size:        clips per forward pass
        device:            torch device string; None picks cuda when available
        progress_interval: seconds between progress lines; None silences them

    Returns:
        raw hypothesis strings, one per sample, in shard.samples order
    """
    samples = shard.samples
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    print(f"Loading {model_name} on {dev} ({dtype})")

    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name, dtype=dtype)
    model.to(dev).eval()

    hyps: list[str] = []
    t0 = t_last = time.perf_counter()
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        feats = processor(
            [shard.audio(s) for s in chunk],
            sampling_rate=chunk[0]["sample_rate"],
            return_tensors="pt",
        ).input_features.to(dev, dtype)

        with torch.no_grad():
            # Greedy, no beam search, no external LM — matching utils/generate.py.
            ids = model.generate(
                feats, language="en", task="transcribe",
                num_beams=1, do_sample=False,
            )
        hyps.extend(
            t.strip() for t in processor.batch_decode(ids, skip_special_tokens=True)
        )

        now = time.perf_counter()
        if progress_interval is not None and now - t_last >= progress_interval:
            rate = len(hyps) / (now - t0)
            print(f"  {len(hyps):,}/{len(samples):,} clips  {rate:.1f} clips/s",
                  flush=True)
            t_last = now

    elapsed = time.perf_counter() - t0
    print(f"  decoded {len(hyps):,} clips in {elapsed:.0f}s "
          f"({len(hyps) / max(elapsed, 1e-9):.1f} clips/s)", flush=True)
    return hyps


def build_rows(
    samples: list[dict], hyps: list[str], dataset: str, formats: list[str],
    model_name: str,
) -> list[dict]:
    """Pair each raw hypothesis against every requested reference form.

    Args:
        samples:    decode inputs, in the same order as hyps
        hyps:       raw Whisper output
        dataset:    dataset tag written on every row
        formats:    subset of ("unformatted", "formatted")
        model_name: recorded as the row's `checkpoint`, so run_wer.py-shaped
                    consumers show something meaningful in that column

    Returns:
        JSONL rows in run_wer.py's schema, plus dataset/system/hypothesis_raw
    """
    rows: list[dict] = []
    for sample, raw in zip(samples, hyps, strict=True):
        for fmt in formats:
            ref_raw = sample[fmt]
            # Verbatim pairing normalises BOTH sides down to §6; formatted
            # pairing compares raw against raw.  See the module docstring for
            # why normalising the hypothesis alone would be worth ~3 WER.
            if fmt == "unformatted":
                reference, hypothesis = spec6_normalize(ref_raw), spec6_normalize(raw)
            else:
                reference, hypothesis = ref_raw, raw
            rows.append({
                "checkpoint":     model_name,
                "step":           0,
                "key":            sample["key"],
                "dataset":        dataset,
                "split":          dataset,
                "system":         "whisper-small",
                "type":           fmt,
                "reference":      reference,
                "hypothesis":     hypothesis,
                "reference_raw":  ref_raw,
                "hypothesis_raw": raw,
                "wer":            compute_wer(reference, hypothesis),
            })
    return rows


def summarise(rows: list[dict]) -> dict[str, float]:
    """Corpus-level WER per pairing — not the mean of the per-row numbers."""
    import jiwer

    out: dict[str, float] = {}
    for fmt in ("unformatted", "formatted"):
        pairs = [(r["reference"], r["hypothesis"]) for r in rows if r["type"] == fmt]
        if not pairs:
            continue
        out[fmt] = jiwer.wer([r for r, _ in pairs], [h for _, h in pairs])
    return out


def _self_test() -> None:
    """Row construction, pairing and scoring — mocked decode, no model download."""
    import tempfile

    import numpy as np
    from ood_shard import ShardWriter

    samples = [
        {"key": "cv-1", "audio": np.zeros(16000, np.float32), "sample_rate": 16000,
         "unformatted": "don't you understand me", "formatted": "Don't you understand me?"},
    ]
    rows = build_rows(samples, ["Don't you understand me?"], "cv", 
                      ["unformatted", "formatted"], "openai/whisper-small")
    assert len(rows) == 2, rows
    unf = next(r for r in rows if r["type"] == "unformatted")
    fmt = next(r for r in rows if r["type"] == "formatted")
    # Both sides of the verbatim pairing land on §6: the apostrophe the house
    # reference keeps must not become an error for a control that got it right.
    assert unf["hypothesis"] == "dont you understand me", unf["hypothesis"]
    assert unf["reference"] == "dont you understand me", unf["reference"]
    assert unf["reference_raw"] == "don't you understand me", unf["reference_raw"]
    assert fmt["hypothesis"] == "Don't you understand me?", fmt["hypothesis"]
    assert fmt["reference"] == "Don't you understand me?", fmt["reference"]
    assert unf["hypothesis_raw"] == fmt["hypothesis_raw"] == "Don't you understand me?"
    assert unf["wer"] == 0.0 and fmt["wer"] == 0.0, (unf["wer"], fmt["wer"])
    print("  [OK] build_rows: §6 on BOTH sides of verbatim, raw on formatted")

    # A control that is right about the WORDS but writes them Whisper's way must
    # score 0 on the verbatim pairing — that is the whole point of normalising it.
    rows2 = build_rows(
        [{"key": "k", "audio": None, "sample_rate": 16000,
          "unformatted": "mister smith paid ten dollars", "formatted": "x"}],
        ["Mr. Smith paid ten dollars."], "s", ["unformatted"], "m",
    )
    assert rows2[0]["wer"] == 0.0, rows2[0]
    print("  [OK] §6 allowlist inversion lands the control on the house convention")

    agg = summarise(rows + rows2)
    assert agg["unformatted"] == 0.0 and agg["formatted"] == 0.0, agg
    bad = build_rows(samples, ["totally different words here"], "cv", ["unformatted"], "m")
    assert summarise(bad)["unformatted"] > 0.5, summarise(bad)
    print("  [OK] summarise aggregates at corpus level")

    with tempfile.TemporaryDirectory() as tmp:
        shard_path = Path(tmp) / "s.tar"
        w = ShardWriter(shard_path)
        audio = (np.sin(np.arange(8000) * 0.01) * 0.1).astype(np.float32)
        w.add_sample("b-long", np.zeros((80, 8), np.float16), "b", "B.", audio)
        w.add_sample("a-short", np.zeros((80, 8), np.float16), "a", "A.", audio[:4000])
        w.close()

        shard = ShardAudio(shard_path)
        assert [s["key"] for s in shard.samples] == ["a-short", "b-long"], "not length-sorted"
        assert shard.samples[0]["formatted"] == "A." and shard.samples[0]["sample_rate"] == 16000
        assert shard.samples[0]["n_frames"] == 4000, shard.samples[0]["n_frames"]
        got = shard.audio(shard.samples[1])
        assert abs(len(got) - 8000) <= 1 and np.abs(got - audio[: len(got)]).max() < 1e-3
        shard.close()
        assert len(ShardAudio(shard_path, limit=1)) == 1

        noaudio = Path(tmp) / "n.tar"
        w2 = ShardWriter(noaudio)
        w2.add_sample("k", np.zeros((80, 8), np.float16), "a", "A.")
        w2.close()
        try:
            ShardAudio(noaudio)
        except ValueError as exc:
            assert "flac" in str(exc)
        else:
            raise AssertionError("ShardAudio accepted a shard with no audio")
        print("  [OK] ShardAudio: length sort, lazy decode, limit, loud failure without .flac")

    print("PASSED")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--shard", type=Path, help="Eval shard .tar carrying {key}.flac.")
    p.add_argument("--output", type=Path, help="Per-utterance JSONL to write.")
    p.add_argument("--dataset", type=str, default=None,
                   help="Dataset tag for the rows (default: the shard stem).")
    p.add_argument("--formats", nargs="+", choices=["unformatted", "formatted"],
                   default=["unformatted", "formatted"],
                   help="Pairings to score.  Sets with no genuine formatted "
                        "reference (TED-LIUM) must pass --formats unformatted.")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--limit", type=int, default=None, help="Decode only N clips (smoke runs).")
    p.add_argument("--progress-interval", type=float, default=30.0, dest="progress_interval")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)

    if args.self_test:
        _self_test()
        return 0
    if args.shard is None or args.output is None:
        p.error("--shard and --output are required unless --self-test")

    dataset = args.dataset or args.shard.stem
    shard = ShardAudio(args.shard, args.limit)
    print(f"{args.shard.name}: {len(shard):,} clips  (dataset tag: {dataset})")

    try:
        hyps = transcribe(
            shard, args.model, args.batch_size, args.device,
            args.progress_interval if args.progress_interval > 0 else None,
        )
        rows = build_rows(shard.samples, hyps, dataset, args.formats, args.model)
    finally:
        shard.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {args.output}  ({len(rows):,} rows)")

    for fmt, wer in summarise(rows).items():
        print(f"  whisper-small  {dataset}/{fmt:<12} WER {wer:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

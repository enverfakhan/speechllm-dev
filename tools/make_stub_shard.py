"""Generate self-contained stub WebDataset shards for local stub runs.

The stub configs (configs/stub*.yaml) used to point at data_dev/, a directory
of REAL dev shards built from LibriSpeech audio (a since-removed tool) —
so a fresh checkout could not run a stub at all.  A stub model is a tiny
random Llama: it learns nothing from the audio, so the mel can be noise.  What
must be real is the LABEL text, because PrunedTokenizer.encode silently drops
ids missing from the vocabulary (Decision 005) and a made-up label would be
quietly truncated.  The labels therefore come from a committed fixture copied
out of data/labels.jsonl.

Output (default --out-dir data_stub/, gitignored):

    stub-train-000000.tar  stub-train-000001.tar   training shards, half each
    stub-diag-000000.tar                            every record, fresh noise

Each sample is the house shard format (Decision 006):

    {key}.mel.npy          float16 (80, T), T varies with the label length
    {key}.unformatted.txt
    {key}.formatted.txt

The diag shard doubles as the eval shard for the chat-convention stubs
(run_probes.py, run_longform.py): it is a plain tarfile read in member order.

Fixture: tests/fixtures/stub_labels.jsonl — 64 records = 8 runs of 8
CONSECUTIVE utterances of one (speaker, chapter), each <= 20 words, chosen
from chapters spread through data/labels.jsonl.  Consecutive runs are what
let tools/run_longform.py form multi-turn conversations from this shard; the
word cap keeps every reference under the 41-token eval filter.

T is ~40 mel frames (0.4 s) per word plus a 50-frame lead-in, jittered x0.8-1.25
and clamped to [100, 3000], so audio-token / transcript-token ratios look like
real read speech and batches carry genuinely varied T.  Deterministic given
--seed.

Usage:
    python tools/make_stub_shard.py                    # writes data_stub/
    python tools/make_stub_shard.py --self-test
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

_REPO_ROOT       = Path(__file__).resolve().parent.parent
_DEFAULT_LABELS  = _REPO_ROOT / "tests" / "fixtures" / "stub_labels.jsonl"
_DEFAULT_OUT_DIR = Path("data_stub")

N_MELS          = 80
_FRAMES_PER_WORD = 40      # 10 ms hop → 0.4 s per word, ~2.5 words/s read speech
_LEAD_FRAMES     = 50
_T_MIN, _T_MAX   = 100, 3000   # 3000 frames = Whisper's 30 s ceiling


def load_labels(path: Path) -> list[dict]:
    """Read the fixture JSONL; every record needs key / unformatted / formatted."""
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for r in records:
        missing = {"key", "unformatted", "formatted"} - set(r)
        if missing:
            raise ValueError(f"{path}: record {r.get('key')!r} lacks {sorted(missing)}")
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def stub_mel(n_words: int, rng: np.random.Generator) -> np.ndarray:
    """Random float16 log-mel-like noise, (80, T), T scaled by the label length."""
    t = int((_FRAMES_PER_WORD * n_words + _LEAD_FRAMES) * rng.uniform(0.8, 1.25))
    t = min(max(t, _T_MIN), _T_MAX)
    # Whisper's log-mel is normalised to roughly [-1, 1.5]; match the range so
    # the encoder sees inputs of the usual scale.
    mel = np.clip(rng.normal(0.0, 0.5, size=(N_MELS, t)), -1.0, 1.5)
    return mel.astype(np.float16)


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def write_shard(path: Path, records: list[dict], rng: np.random.Generator) -> list[int]:
    """Write one shard; returns the T of every sample written (for reporting)."""
    ts: list[int] = []
    with tarfile.open(path, "w") as tar:
        for r in records:
            mel = stub_mel(len(r["unformatted"].split()), rng)
            buf = io.BytesIO()
            np.save(buf, mel)
            _add(tar, f"{r['key']}.mel.npy",         buf.getvalue())
            _add(tar, f"{r['key']}.unformatted.txt", r["unformatted"].encode("utf-8"))
            _add(tar, f"{r['key']}.formatted.txt",   r["formatted"].encode("utf-8"))
            ts.append(mel.shape[1])
    return ts


def make_stub_shards(labels: Path, out_dir: Path, seed: int = 0) -> dict[str, list[int]]:
    """Write the two training shards and the diag shard; returns {name: [T, ...]}."""
    records = load_labels(labels)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng  = np.random.default_rng(seed)
    half = (len(records) + 1) // 2
    plan = {
        "stub-train-000000.tar": records[:half],
        "stub-train-000001.tar": records[half:],
        # Same labels, fresh noise: the diag pass must not see the exact
        # training tensors, or "eval" loss would just be train loss.
        "stub-diag-000000.tar":  records,
    }
    return {name: write_shard(out_dir / name, recs, rng) for name, recs in plan.items()}


def _self_test() -> None:
    import tempfile

    sys.path.insert(0, str(_REPO_ROOT))
    from data import read_eval_shard

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        labels = tmp / "labels.jsonl"
        recs = [
            {"key": f"1-2-{i:04d}", "unformatted": "a b c " * (i + 1),
             "formatted": f"A b c {i}."}
            for i in range(5)
        ]
        labels.write_text("".join(json.dumps(r) + "\n" for r in recs))

        ts1 = make_stub_shards(labels, tmp / "a", seed=3)
        ts2 = make_stub_shards(labels, tmp / "b", seed=3)
        assert ts1 == ts2, "same seed must reproduce the same shapes"
        assert (tmp / "a" / "stub-diag-000000.tar").read_bytes() == \
               (tmp / "b" / "stub-diag-000000.tar").read_bytes(), "not deterministic"
        assert [len(v) for v in ts1.values()] == [3, 2, 5], ts1
        assert len(set(ts1["stub-diag-000000.tar"])) > 1, "T must vary"

        samples = read_eval_shard(tmp / "a" / "stub-diag-000000.tar")
        assert [s.key for s in samples] == [r["key"] for r in recs]
        for s, r in zip(samples, recs):
            assert s.mel.shape[0] == N_MELS and _T_MIN <= s.mel.shape[1] <= _T_MAX
            assert s.ref_unformatted == r["unformatted"]
            assert s.ref_formatted   == r["formatted"]

        bad = tmp / "bad.jsonl"
        bad.write_text(json.dumps({"key": "x", "unformatted": "y"}) + "\n")
        try:
            load_labels(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("a record missing 'formatted' must be rejected")

    print("[OK] deterministic, varied T, three-member format, read back by read_eval_shard")
    print("PASSED")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--labels",  type=Path, default=_DEFAULT_LABELS,
                   help="label fixture JSONL (default: tests/fixtures/stub_labels.jsonl)")
    p.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR,
                   help="output directory (default: data_stub/, gitignored)")
    p.add_argument("--seed",    type=int,  default=0)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return

    shards = make_stub_shards(args.labels, args.out_dir, args.seed)
    for name, ts in shards.items():
        print(f"{args.out_dir / name}: {len(ts)} samples, T {min(ts)}..{max(ts)} frames")


if __name__ == "__main__":
    main()

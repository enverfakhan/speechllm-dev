"""Write a reproducible random subset of eval samples to a single .tar file.

Reads one or more source shards using stdlib tarfile (deterministic, finite —
NOT webdataset, which can loop in this project), samples n keys with a fixed
seed, and writes them to --output.

Filtering: samples whose longest label (unformatted or formatted) exceeds
--max-label-len tokens are removed from the candidate pool before the final
selection.  This caps worst-case generation time during eval.

Two-stage selection (when filtering is active):
  1. Sample 2×n candidates from the full sorted key list.
  2. Drop any candidate whose max(len(unfmt_ids), len(fmt_ids)) > max_label_len.
  3. Sample n from the survivors.

Usage:
    python tools/make_eval_subset.py \\
        --input      data/shards/dev-clean-*.tar \\
        --output     data/eval_subsets/dev-clean-480.tar \\
        --n          480 \\
        --seed       42 \\
        --max-label-len 41 \\
        --tokenizer  data/pruned_tokenizer/

Self-test (no real data needed):
    python tools/make_eval_subset.py --self-test
"""

from __future__ import annotations

import argparse
import glob
import io
import random
import sys
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_REQUIRED_EXTS = {"mel.npy", "unformatted.txt", "formatted.txt"}


def _read_shard(tar_path: Path) -> dict[str, dict[str, bytes]]:
    """Read one .tar shard and group members by sample key.

    Splits each member name on the first "." to get (key, ext).  Only
    samples that have all three expected members are returned.

    Returns:
        dict mapping key → {ext: raw_bytes}
    """
    groups: dict[str, dict[str, bytes]] = {}
    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            dot = member.name.find(".")
            if dot < 0:
                continue
            key = member.name[:dot]
            ext = member.name[dot + 1:]
            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            groups.setdefault(key, {})[ext] = fobj.read()
    return {k: v for k, v in groups.items() if _REQUIRED_EXTS <= set(v.keys())}


def _expand_inputs(patterns: list[str]) -> list[Path]:
    """Glob-expand each pattern and return a sorted, deduplicated list of Paths."""
    paths: set[Path] = set()
    for pat in patterns:
        matched = glob.glob(pat)
        if matched:
            for m in matched:
                paths.add(Path(m))
        else:
            paths.add(Path(pat))  # treat as literal; will fail loudly on open
    return sorted(paths)


def build_subset(
    input_paths: list[Path],
    output_path: Path,
    n: int = 480,
    seed: int = 42,
    max_label_len: int | None = 41,
    tokenizer_path: Path | None = None,
    *,
    _encode_fn: Callable[[str], list[int]] | None = None,
) -> int:
    """Read all source shards, filter by label length, sample n keys, write to output.

    Selection is reproducible: keys are sorted before each sampling step.

    When max_label_len is set, a two-stage selection is performed:
      1. Sample min(2*n, total) candidate keys.
      2. Keep only those where max(len(unfmt_ids), len(fmt_ids)) <= max_label_len.
      3. Sample n from the survivors.

    Args:
        input_paths:    list of source .tar paths (read in sorted order)
        output_path:    destination .tar file
        n:              number of samples to select
        seed:           random seed
        max_label_len:  maximum token length of either label variant; None disables
        tokenizer_path: path to pruned tokenizer dir; required when max_label_len is set
        _encode_fn:     inject a custom encode function (testing only); takes
                        precedence over tokenizer_path when provided

    Returns:
        Number of samples written (may be < n if fewer survive filtering)
    """
    all_samples: dict[str, dict[str, bytes]] = {}
    for shard_path in input_paths:
        all_samples.update(_read_shard(shard_path))

    total = len(all_samples)
    if total == 0:
        raise ValueError(f"No complete samples found in: {input_paths}")

    sorted_keys = sorted(all_samples.keys())

    # ── Stage 1: draw a 2×n candidate pool ───────────────────────────────────
    pool_size = min(2 * n, total)
    pool_keys = random.Random(seed).sample(sorted_keys, pool_size)

    # ── Stage 2: filter by max label token length ─────────────────────────────
    if max_label_len is not None:
        encode: Callable[[str], list[int]]
        if _encode_fn is not None:
            encode = _encode_fn
        elif tokenizer_path is not None:
            from data import PrunedTokenizer
            tok    = PrunedTokenizer(tokenizer_path)
            encode = tok.encode
        else:
            raise ValueError(
                "tokenizer_path (or --tokenizer) is required when max_label_len is set"
            )

        filtered_keys: list[str] = []
        n_removed = 0
        for key in pool_keys:
            members = all_samples[key]
            unfmt   = members["unformatted.txt"].decode("utf-8")
            fmt     = members["formatted.txt"].decode("utf-8")
            if max(len(encode(unfmt)), len(encode(fmt))) <= max_label_len:
                filtered_keys.append(key)
            else:
                n_removed += 1

        n_pass = len(filtered_keys)
        print(
            f"Pool (2×{n}={pool_size}):   {pool_size} candidates  "
            f"→ filter (max_label_len={max_label_len}): "
            f"{n_pass} passed, {n_removed} removed"
        )
    else:
        filtered_keys = pool_keys
        n_pass        = pool_size

    # ── Stage 3: sample n from the survivors ─────────────────────────────────
    if n_pass == 0:
        raise ValueError(
            f"No samples passed the max_label_len={max_label_len} filter."
        )

    n_select = min(n, n_pass)
    if n_select < n:
        print(
            f"[warn] Only {n_pass} samples passed filter; "
            f"requested {n} — taking all {n_pass}.",
            file=sys.stderr,
        )

    # Sort survivors before sampling so stage-3 selection is also reproducible.
    selected_keys = random.Random(seed).sample(sorted(filtered_keys), n_select)

    # ── Write ─────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, "w") as tf:
        for key in selected_keys:
            for ext, data in all_samples[key].items():
                name = f"{key}.{ext}"
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))

    print(f"Source samples: {total}")
    print(f"Selected:       {n_select}")
    print(f"Output:         {output_path}")
    return n_select


def _self_test() -> None:
    """Synthetic round-trip test — no real data or tokenizer needed."""
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # ── Build source shard: 12 fake samples ──────────────────────────────
        source_path = tmp_dir / "source.tar"
        fake_keys   = [f"fake-{i:04d}-key" for i in range(12)]
        with tarfile.open(source_path, "w") as tf:
            for i, key in enumerate(fake_keys):
                mel       = np.random.randn(80, 100 + i * 20).astype(np.float16)
                buf       = io.BytesIO()
                np.save(buf, mel)
                mel_bytes = buf.getvalue()
                for name, data in [
                    (f"{key}.mel.npy",         mel_bytes),
                    (f"{key}.unformatted.txt",  f"unfmt {i}".encode()),
                    (f"{key}.formatted.txt",    f"Fmt {i}.".encode()),
                ]:
                    info      = tarfile.TarInfo(name=name)
                    info.size = len(data)
                    tf.addfile(info, io.BytesIO(data))

        # ── Basic: n=5 no filter ─────────────────────────────────────────────
        out1 = tmp_dir / "s42.tar"
        n_written = build_subset([source_path], out1, n=5, seed=42, max_label_len=None)
        assert n_written == 5, f"Expected 5, got {n_written}"
        back1 = _read_shard(out1)
        assert len(back1) == 5, f"Expected 5 in output, got {len(back1)}"
        for key, members in back1.items():
            assert set(members) >= _REQUIRED_EXTS, f"{key} missing members"

        # ── Reproducibility: same seed → same key set ────────────────────────
        out2 = tmp_dir / "s42b.tar"
        build_subset([source_path], out2, n=5, seed=42, max_label_len=None)
        assert set(_read_shard(out2)) == set(back1), "Same seed gave different keys"

        # ── Different seed → different key set ──────────────────────────────
        out3 = tmp_dir / "s99.tar"
        build_subset([source_path], out3, n=5, seed=99, max_label_len=None)
        assert set(_read_shard(out3)) != set(back1), (
            "Different seeds produced identical key sets — statistically improbable"
        )

        # ── Byte-identical member contents ───────────────────────────────────
        orig = _read_shard(source_path)
        for key, members in back1.items():
            assert members["unformatted.txt"] == orig[key]["unformatted.txt"], (
                f"{key}: unformatted.txt bytes differ"
            )
            assert members["mel.npy"] == orig[key]["mel.npy"], (
                f"{key}: mel.npy bytes differ"
            )

        # ── Filter test ───────────────────────────────────────────────────────
        # Build a shard where samples 0-5 have short labels (2 words) and
        # samples 6-11 have long labels (10 words).  Use word count as a
        # stand-in for token count so the test needs no real tokenizer.
        filter_path = tmp_dir / "filter_source.tar"
        with tarfile.open(filter_path, "w") as tf:
            for i in range(12):
                key = f"flt-{i:04d}"
                mel = np.zeros((80, 80), dtype=np.float16)
                buf = io.BytesIO()
                np.save(buf, mel)
                unfmt = ("short label" if i < 6 else "one two three four five six seven eight nine ten")
                for name, data in [
                    (f"{key}.mel.npy",         buf.getvalue()),
                    (f"{key}.unformatted.txt",  unfmt.encode()),
                    (f"{key}.formatted.txt",    unfmt.encode()),
                ]:
                    info      = tarfile.TarInfo(name=name)
                    info.size = len(data)
                    tf.addfile(info, io.BytesIO(data))

        # word-count mock: max_label_len=4 passes short (2 words), drops long (10 words)
        def word_count(s: str) -> list[str]:  # type: ignore[return]
            return s.split()

        out_flt = tmp_dir / "filtered.tar"
        n_flt   = build_subset(
            [filter_path], out_flt,
            n=4, seed=42, max_label_len=4, _encode_fn=word_count,
        )
        assert n_flt == 4, f"Expected 4 filtered samples, got {n_flt}"
        back_flt = _read_shard(out_flt)
        # All selected keys must come from the short-label half (i < 6)
        assert all(int(k.split("-")[1]) < 6 for k in back_flt), (
            f"Long-label sample slipped through filter: {set(back_flt)}"
        )

        # Requesting more than what survives the filter → warn and take all 6
        out_flt2 = tmp_dir / "filtered_all.tar"
        n_flt2   = build_subset(
            [filter_path], out_flt2,
            n=10, seed=42, max_label_len=4, _encode_fn=word_count,
        )
        assert n_flt2 == 6, f"Expected 6 (all that pass filter), got {n_flt2}"

    print("PASSED")


def main(argv: list[str] | None = None) -> None:
    """Parse CLI and run subset selection."""
    parser = argparse.ArgumentParser(
        description=(
            "Write a reproducible random subset of eval samples to a single .tar. "
            "Uses stdlib tarfile for deterministic, finite iteration over source shards."
        ),
    )
    parser.add_argument(
        "--input", nargs="+", default=None, metavar="PATH",
        help="One or more source shard paths or globs. Required unless --self-test.",
    )
    parser.add_argument(
        "--output", type=Path, default=None, metavar="PATH",
        help="Output .tar path.",
    )
    parser.add_argument(
        "--n", type=int, default=480,
        help="Number of samples to select (default: 480).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--max-label-len", type=int, default=41, dest="max_label_len",
        metavar="L",
        help=(
            "Remove candidates whose longest label (unformatted or formatted) exceeds "
            "L tokens before final selection.  L=41 ≈ 90th percentile of LibriSpeech "
            "transcript lengths.  Set to 0 to disable filtering (default: 41)."
        ),
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("data/pruned_tokenizer/"),
        metavar="PATH",
        help=(
            "Path to the pruned tokenizer directory (default: data/pruned_tokenizer/). "
            "Required when --max-label-len > 0."
        ),
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Run self-test with synthetic data and exit.",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        _self_test()
        return

    if args.input is None:
        parser.error("--input is required unless --self-test is passed")
    if args.output is None:
        parser.error("--output is required unless --self-test is passed")

    max_label_len  = args.max_label_len if args.max_label_len > 0 else None
    tokenizer_path = args.tokenizer if max_label_len is not None else None

    input_paths = _expand_inputs(args.input)
    if not input_paths:
        print(f"[error] No files matched: {args.input}", file=sys.stderr)
        sys.exit(1)

    build_subset(
        input_paths, args.output,
        n=args.n, seed=args.seed,
        max_label_len=max_label_len,
        tokenizer_path=tokenizer_path,
    )


if __name__ == "__main__":
    main()

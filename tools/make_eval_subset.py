"""Write a reproducible random subset of eval samples to a single .tar file.

Reads one or more source shards using stdlib tarfile (deterministic, finite —
NOT webdataset, which can loop in this project), samples n keys with a fixed
seed, and writes them to --output.

Usage:
    python tools/make_eval_subset.py \\
        --input  data/shards/dev-clean-*.tar \\
        --output data/eval_subsets/dev-clean-480.tar \\
        --n      480 \\
        --seed   42

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
) -> int:
    """Read all source shards, sample n keys deterministically, write to output.

    Selection is reproducible: keys are sorted before sampling, so the result
    is independent of tar iteration order.

    Args:
        input_paths: list of source .tar paths (read in sorted order)
        output_path: destination .tar file
        n:           number of samples to select
        seed:        random seed

    Returns:
        Number of samples written (may be < n if fewer available)
    """
    all_samples: dict[str, dict[str, bytes]] = {}
    for shard_path in input_paths:
        all_samples.update(_read_shard(shard_path))

    total = len(all_samples)
    if total == 0:
        raise ValueError(f"No complete samples found in: {input_paths}")

    sorted_keys = sorted(all_samples.keys())
    n_select = min(n, total)
    if n_select < n:
        print(
            f"[warn] Only {total} samples available; requested {n} — taking all {total}.",
            file=sys.stderr,
        )

    selected_keys = random.Random(seed).sample(sorted_keys, n_select)

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

        # Build a source shard with 12 fake samples of varying mel length.
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

        # ── n=5, seed=42: exactly 5 samples written ──────────────────────────
        out1 = tmp_dir / "s42.tar"
        n_written = build_subset([source_path], out1, n=5, seed=42)
        assert n_written == 5, f"Expected 5, got {n_written}"
        back1 = _read_shard(out1)
        assert len(back1) == 5, f"Expected 5 in output, got {len(back1)}"
        for key, members in back1.items():
            assert set(members) >= _REQUIRED_EXTS, f"{key} missing members"

        # ── Reproducibility: same seed → same key set ────────────────────────
        out2 = tmp_dir / "s42b.tar"
        build_subset([source_path], out2, n=5, seed=42)
        assert set(_read_shard(out2)) == set(back1), "Same seed gave different keys"

        # ── Different seed → different key set ──────────────────────────────
        out3 = tmp_dir / "s99.tar"
        build_subset([source_path], out3, n=5, seed=99)
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

    input_paths = _expand_inputs(args.input)
    if not input_paths:
        print(f"[error] No files matched: {args.input}", file=sys.stderr)
        sys.exit(1)

    build_subset(input_paths, args.output, n=args.n, seed=args.seed)


if __name__ == "__main__":
    main()

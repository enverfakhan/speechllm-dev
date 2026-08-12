"""Rewrite existing WebDataset shards with new labels, dropping unusable samples.

The mel spectrogram in a shard is a pure function of the audio, so a relabelling
run has no reason to recompute it: this tool streams every source .tar, copies
each `{key}.mel.npy` member through byte-for-byte, and replaces only the two
label members from a labels JSONL.  Re-running tools/preprocess.py would give the
identical result at the cost of decoding 960 h of FLAC and recomputing every mel
— hours of CPU against minutes of I/O here.

Samples whose label record is not `"validation": "ok"` are dropped ENTIRELY (all
three members, plus their manifest row).  A failed record is one where the
labelling model's output could not be aligned to the reference transcript, so its
two labels disagree about which words were spoken; keeping it would train the
formatted and unformatted instructions toward contradictory targets on the same
audio.  Dropping is applied to eval shards as well as training shards — a sample
that is unfit to train on is equally unfit to score against.

Shard membership is otherwise preserved: a sample stays in the shard it was
already in, and shard filenames are unchanged, so existing shard-list files
(subset_shards.txt and friends) stay valid without regeneration.

    # a directory of shards + its manifest.jsonl
    python tools/relabel_shards.py \\
        --labels data/labels.jsonl \\
        --in     data_old/shards \\
        --out    data/shards

    # a single loose .tar (eval subsets, the packed full-eval tar)
    python tools/relabel_shards.py \\
        --labels data/labels.jsonl \\
        --in     data_old/full-eval-test-dev-clean-other.tar \\
        --out    data/

    python tools/relabel_shards.py --self-test     # offline, no data needed

Standard library only.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import tarfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# The three members preprocess.py writes per sample. Order inside the tar is
# preserved as-is; WebDataset groups by the key prefix, not by position.
MEL_SUFFIX:         str = ".mel.npy"
UNFORMATTED_SUFFIX: str = ".unformatted.txt"
FORMATTED_SUFFIX:   str = ".formatted.txt"
LABEL_SUFFIXES: tuple[str, str] = (UNFORMATTED_SUFFIX, FORMATTED_SUFFIX)
ALL_SUFFIXES: tuple[str, ...] = (MEL_SUFFIX, *LABEL_SUFFIXES)

MANIFEST_NAME: str = "manifest.jsonl"

# 4 MB: large enough that a ~1 MB mel copies in one or two reads, small enough
# that the working set stays trivial next to a 53 GB pass.
COPY_BUFSIZE: int = 4 * 1024 * 1024


@dataclass
class Labels:
    """New label text per key, plus the keys that must not be shipped at all."""

    ok: dict[str, tuple[str, str]] = field(default_factory=dict)   # key → (unf, fmt)
    failed: set[str] = field(default_factory=set)

    def __len__(self) -> int:
        return len(self.ok) + len(self.failed)


def load_labels(path: Path) -> Labels:
    """Read a labels JSONL into ok-text and failed-key sets.

    Accepts the output of tools/label_formatted.py finalize (which carries a
    `validation` field) and of tools/precompute_labels.py (which does not — every
    record is then treated as usable, matching that tool's own semantics).
    """
    labels = Labels()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = record["key"]
            if record.get("validation", "ok") != "ok":
                labels.failed.add(key)
            else:
                labels.ok[key] = (record["unformatted"], record["formatted"])
    return labels


def member_key(name: str) -> tuple[str, str] | None:
    """Split a tar member name into (key, suffix), or None if it is not a sample."""
    for suffix in ALL_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix
    return None


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    """Write a bytes blob as a tar member — mirrors preprocess.py's _add_file."""
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def rewrite_tar(
    src: Path,
    dst: Path,
    labels: Labels,
    drop_unlabelled: bool,
    dry_run: bool = False,
) -> Counter:
    """Stream one shard, swapping label members and dropping unusable samples.

    Read in stream mode ("r|") rather than random-access: these archives are read
    exactly once, front to back, and streaming skips building the member index
    for a 30 MB shard's worth of entries.
    """
    counts: Counter = Counter()
    out: tarfile.TarFile | None = None
    # Write to a sidecar and rename on success, so an interrupted run can never
    # leave a truncated .tar that looks finished. With --skip-existing this is
    # what makes a resume safe: every .tar present is one that completed.
    tmp = dst.with_name(dst.name + ".tmp")
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        out = tarfile.open(tmp, "w")

    try:
        with tarfile.open(src, "r|") as inp:
            for info in inp:
                if not info.isfile():
                    continue
                split = member_key(info.name)
                if split is None:
                    counts["member_unknown"] += 1
                    if out is not None:
                        source = inp.extractfile(info)
                        if source is not None:
                            out.addfile(info, source)
                    continue

                key, suffix = split
                if key in labels.failed:
                    counts["member_dropped_failed"] += 1
                    if suffix == MEL_SUFFIX:
                        counts["sample_dropped_failed"] += 1
                    continue
                if key not in labels.ok:
                    if not drop_unlabelled:
                        raise SystemExit(
                            f"{src}: sample {key!r} has no record in the labels file.\n"
                            "Every sample in a shard must be labelled, or the rewrite "
                            "would silently change the corpus. Re-check the labels "
                            "file covers this split, or pass --drop-unlabelled to "
                            "drop these samples deliberately."
                        )
                    counts["member_dropped_unlabelled"] += 1
                    if suffix == MEL_SUFFIX:
                        counts["sample_dropped_unlabelled"] += 1
                    continue

                if suffix == MEL_SUFFIX:
                    counts["sample_kept"] += 1
                    if out is not None:
                        source = inp.extractfile(info)
                        if source is None:                       # pragma: no cover
                            raise SystemExit(f"{src}: unreadable member {info.name}")
                        # copyfileobj streams: a mel is never held whole in memory.
                        buffer = io.BytesIO()
                        shutil.copyfileobj(source, buffer, COPY_BUFSIZE)
                        payload = buffer.getvalue()
                        if len(payload) != info.size:            # pragma: no cover
                            raise SystemExit(
                                f"{src}: short read on {info.name} "
                                f"({len(payload)} of {info.size} bytes)"
                            )
                        out.addfile(info, io.BytesIO(payload))
                    continue

                unformatted, formatted = labels.ok[key]
                text = unformatted if suffix == UNFORMATTED_SUFFIX else formatted
                counts["member_relabelled"] += 1
                if out is not None:
                    _add_bytes(out, info.name, text.encode("utf-8"))
    except BaseException:
        # Includes KeyboardInterrupt: a half-written sidecar must never be
        # promoted to a real shard name.
        if out is not None:
            out.close()
            tmp.unlink(missing_ok=True)
        raise
    else:
        if out is not None:
            out.close()
            os.replace(tmp, dst)

    return counts


def rewrite_manifest(
    src: Path,
    dst: Path,
    labels: Labels,
    drop_unlabelled: bool,
    dry_run: bool = False,
) -> Counter:
    """Filter and relabel a manifest.jsonl alongside its shards.

    duration_s / n_mel_frames / shard are properties of the audio and of shard
    placement, neither of which this tool changes, so they are carried through
    untouched and only the two text fields are replaced.
    """
    counts: Counter = Counter()
    lines: list[str] = []
    with src.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = record["key"]
            if key in labels.failed:
                counts["manifest_dropped_failed"] += 1
                continue
            if key not in labels.ok:
                if not drop_unlabelled:
                    raise SystemExit(
                        f"{src}: manifest row {key!r} has no record in the labels file."
                    )
                counts["manifest_dropped_unlabelled"] += 1
                continue
            record["unformatted"], record["formatted"] = labels.ok[key]
            counts["manifest_kept"] += 1
            lines.append(json.dumps(record))

    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return counts


def run(
    sources: list[Path],
    out_dir: Path,
    labels: Labels,
    drop_unlabelled: bool,
    dry_run: bool,
    skip_existing: bool = False,
) -> Counter:
    """Rewrite every source .tar (and manifest) into out_dir. Returns totals.

    With skip_existing, a destination that already exists is left alone. This is
    only sound because shards are written to a sidecar and renamed on success:
    a present .tar is therefore a complete one. The totals then cover just the
    work this invocation did, so a resumed run reports fewer samples than the
    corpus holds — verify the finished corpus against the manifest, not against
    a resumed run's totals.
    """
    totals: Counter = Counter()
    for index, src in enumerate(sources, start=1):
        dst = out_dir / src.name
        if src.resolve() == dst.resolve():
            raise SystemExit(
                f"Refusing to rewrite {src} onto itself — a streamed rewrite "
                "cannot read and write the same file. Choose a different --out."
            )
        if skip_existing and dst.exists():
            totals["skipped_existing"] += 1
            continue
        if src.name == MANIFEST_NAME:
            counts = rewrite_manifest(src, dst, labels, drop_unlabelled, dry_run)
        else:
            counts = rewrite_tar(src, dst, labels, drop_unlabelled, dry_run)
        totals.update(counts)
        dropped = counts["sample_dropped_failed"] + counts["manifest_dropped_failed"]
        print(
            f"  [{index:>4d}/{len(sources)}] {src.name:<40s} "
            f"kept={counts['sample_kept'] or counts['manifest_kept']:>6d} "
            f"dropped={dropped:>3d}",
            flush=True,
        )
    return totals


def collect_sources(paths: list[Path]) -> list[Path]:
    """Expand each --in into the concrete files to rewrite.

    A directory contributes its *.tar plus its manifest.jsonl if present; the
    manifest goes last so a crash mid-run never leaves a manifest describing
    shards that were not written.
    """
    sources: list[Path] = []
    for path in paths:
        if path.is_dir():
            sources.extend(sorted(path.glob("*.tar")))
            manifest = path / MANIFEST_NAME
            if manifest.is_file():
                sources.append(manifest)
        elif path.is_file():
            sources.append(path)
        else:
            raise SystemExit(f"--in not found: {path}")
    if not sources:
        raise SystemExit("No .tar shards found under the given --in path(s).")
    return sources


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rewrite WebDataset shards with new labels, dropping failed samples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--self-test", action="store_true", help="run offline self-tests")
    parser.add_argument("--labels", type=Path, help="labels JSONL (finalize output)")
    parser.add_argument("--in", dest="inputs", type=Path, nargs="+", metavar="PATH",
                        help="shard directory and/or individual .tar files")
    parser.add_argument("--out", type=Path, metavar="DIR",
                        help="destination directory (shard filenames are preserved)")
    parser.add_argument("--drop-unlabelled", action="store_true",
                        help="drop samples absent from the labels file instead of failing")
    parser.add_argument("--dry-run", action="store_true",
                        help="count everything, write nothing")
    parser.add_argument("--skip-existing", action="store_true",
                        help="leave already-written outputs alone (resume after a crash)")
    args = parser.parse_args(argv)

    if args.self_test:
        _self_test()
        return 0
    if not (args.labels and args.inputs and args.out):
        parser.error("--labels, --in and --out are required (or use --self-test)")

    labels = load_labels(args.labels)
    print(
        f"labels: {len(labels.ok):,} usable, {len(labels.failed):,} failed "
        f"(dropped) from {args.labels}"
    )
    sources = collect_sources(args.inputs)
    print(f"rewriting {len(sources)} file(s) → {args.out}" + ("  [dry run]" if args.dry_run else ""))

    totals = run(
        sources, args.out, labels, args.drop_unlabelled, args.dry_run, args.skip_existing
    )

    print("\ntotals")
    for name in (
        "sample_kept", "sample_dropped_failed", "sample_dropped_unlabelled",
        "member_relabelled", "member_unknown", "skipped_existing",
        "manifest_kept", "manifest_dropped_failed", "manifest_dropped_unlabelled",
    ):
        if totals[name]:
            print(f"  {name:28s} {totals[name]:>10,d}")
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# Self-tests (CPU-only, no corpus needed)
# ══════════════════════════════════════════════════════════════════════════════

def _write_fake_shard(path: Path, keys: list[str]) -> None:
    with tarfile.open(path, "w") as tar:
        for key in keys:
            _add_bytes(tar, f"{key}{MEL_SUFFIX}", f"MEL:{key}".encode())
            _add_bytes(tar, f"{key}{UNFORMATTED_SUFFIX}", f"old unformatted {key}".encode())
            _add_bytes(tar, f"{key}{FORMATTED_SUFFIX}", f"Old formatted {key}.".encode())


def _self_test() -> None:
    import tempfile

    assert member_key("1-2-3.mel.npy") == ("1-2-3", MEL_SUFFIX)
    assert member_key("1-2-3.unformatted.txt") == ("1-2-3", UNFORMATTED_SUFFIX)
    assert member_key("nonsense.bin") is None

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        src_dir, out_dir = tmp / "old", tmp / "new"
        src_dir.mkdir()

        keys = ["a-1-0", "a-1-1", "a-1-2"]
        _write_fake_shard(src_dir / "train-000000.tar", keys)
        with (src_dir / MANIFEST_NAME).open("w", encoding="utf-8") as handle:
            for key in keys:
                handle.write(json.dumps({
                    "key": key, "split": "train", "shard": "train-000000.tar",
                    "duration_s": 1.5, "n_mel_frames": 150,
                    "unformatted": f"old unformatted {key}",
                    "formatted": f"Old formatted {key}.",
                }) + "\n")

        labels_path = tmp / "labels.jsonl"
        with labels_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": keys[0], "unformatted": "new text zero",
                                     "formatted": "New text zero.", "validation": "ok"}) + "\n")
            # The failed one keeps a plausible `formatted` — the drop must be
            # driven by `validation`, never by the text looking wrong.
            handle.write(json.dumps({"key": keys[1], "unformatted": "new text one",
                                     "formatted": "New text one.", "validation": "failed"}) + "\n")
            handle.write(json.dumps({"key": keys[2], "unformatted": "new text two",
                                     "formatted": "New text two.", "validation": "ok"}) + "\n")

        labels = load_labels(labels_path)
        assert len(labels.ok) == 2 and labels.failed == {keys[1]}

        totals = run(collect_sources([src_dir]), out_dir, labels, False, False)
        assert totals["sample_kept"] == 2, totals
        assert totals["sample_dropped_failed"] == 1, totals
        assert totals["member_relabelled"] == 4, totals
        assert totals["manifest_kept"] == 2 and totals["manifest_dropped_failed"] == 1

        # Every member of a dropped sample must be gone — a stray .mel.npy with
        # no labels beside it would surface as a KeyError deep in collation.
        with tarfile.open(out_dir / "train-000000.tar") as tar:
            names = tar.getnames()
            assert not any(name.startswith(keys[1]) for name in names), names
            assert len(names) == 6, names
            # Mel bytes pass through untouched; labels are the new text.
            assert tar.extractfile(f"{keys[0]}{MEL_SUFFIX}").read() == b"MEL:a-1-0"
            assert tar.extractfile(f"{keys[0]}{UNFORMATTED_SUFFIX}").read() == b"new text zero"
            assert tar.extractfile(f"{keys[2]}{FORMATTED_SUFFIX}").read() == b"New text two."

        rows = [
            json.loads(line)
            for line in (out_dir / MANIFEST_NAME).read_text(encoding="utf-8").splitlines()
        ]
        assert [r["key"] for r in rows] == [keys[0], keys[2]]
        assert rows[0]["unformatted"] == "new text zero"
        assert rows[0]["formatted"] == "New text zero."
        # Audio-derived fields are not this tool's business to change.
        assert rows[0]["duration_s"] == 1.5 and rows[0]["n_mel_frames"] == 150

        # An unlabelled sample is a corpus mismatch: loud by default, droppable
        # only on request.
        _write_fake_shard(src_dir / "train-000001.tar", ["b-9-9"])
        try:
            run([src_dir / "train-000001.tar"], out_dir, labels, False, True)
        except SystemExit as exc:
            assert "no record in the labels file" in str(exc)
        else:                                                    # pragma: no cover
            raise AssertionError("an unlabelled sample must fail by default")
        counts = run([src_dir / "train-000001.tar"], out_dir, labels, True, True)
        assert counts["sample_dropped_unlabelled"] == 1, counts

        # Reading and writing the same path cannot work in stream mode.
        try:
            run([src_dir / "train-000000.tar"], src_dir, labels, False, False)
        except SystemExit as exc:
            assert "onto itself" in str(exc)
        else:                                                    # pragma: no cover
            raise AssertionError("in-place rewrite must be refused")

        # Crash safety: a run that dies mid-shard must leave NO .tar under the
        # real name, so a --skip-existing resume cannot mistake a truncated
        # shard for a finished one.
        crash_src = src_dir / "train-000002.tar"
        _write_fake_shard(crash_src, ["c-1-0", "c-1-1"])
        labels.ok["c-1-0"] = ("c zero", "C zero.")
        labels.ok["c-1-1"] = ("c one", "C one.")
        boom = out_dir / "boom"

        real_add = tarfile.TarFile.addfile
        calls = {"n": 0}

        def exploding_addfile(self, info, fileobj=None):         # pragma: no cover
            calls["n"] += 1
            if calls["n"] > 2:
                raise OSError("simulated disk failure")
            return real_add(self, info, fileobj)

        tarfile.TarFile.addfile = exploding_addfile
        try:
            run([crash_src], boom, labels, False, False)
        except OSError:
            pass
        else:                                                    # pragma: no cover
            raise AssertionError("the simulated failure must propagate")
        finally:
            tarfile.TarFile.addfile = real_add

        assert not (boom / crash_src.name).exists(), "a crashed shard must not be named .tar"
        assert not list(boom.glob("*.tmp")), "the sidecar must be cleaned up"

        # ...and the resume then rewrites it, leaving finished shards untouched.
        before = (out_dir / "train-000000.tar").read_bytes()
        counts = run(
            [crash_src, src_dir / "train-000000.tar"], out_dir, labels, False, False,
            skip_existing=True,
        )
        assert counts["skipped_existing"] == 1, counts
        assert counts["sample_kept"] == 2, counts
        assert (out_dir / "train-000000.tar").read_bytes() == before
        assert (out_dir / crash_src.name).exists()

    print("  relabel shards                ok")
    print("all self-tests passed")


if __name__ == "__main__":
    sys.exit(main())

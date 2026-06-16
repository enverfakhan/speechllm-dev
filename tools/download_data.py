"""Download LibriSpeech splits (training and/or evaluation).

Downloads from OpenSLR using urllib (no HuggingFace datasets dependency).
Verifies MD5 checksums after each download, then extracts in-place.

Both the download step and the extraction step are idempotent: an existing
archive with a valid checksum is not re-downloaded, and an already-extracted
split directory is not re-extracted.

Usage:
    # Download all three training splits (default):
    python scripts/download_data.py --output_dir data/librispeech/

    # Download eval splits only:
    python scripts/download_data.py --output_dir data/librispeech/ \\
        --splits dev-clean dev-other test-clean test-other

    # Download a specific split:
    python scripts/download_data.py --output_dir data/librispeech/ \\
        --splits train-clean-100
"""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

# Published MD5 checksums from openslr.org/12
SPLITS: dict[str, dict[str, str]] = {
    # ── Training ──────────────────────────────────────────────────────────────
    "train-clean-100": {
        "url": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
        "md5": "2a93770f6d5c6c964bc36631d331a522",
        "extracted_subdir": "LibriSpeech/train-clean-100",
    },
    "train-clean-360": {
        "url": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
        "md5": "c0e676e450a7ff2f54aeade5171606fa",
        "extracted_subdir": "LibriSpeech/train-clean-360",
    },
    "train-other-500": {
        "url": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
        "md5": "d1a0fd59409feb2c614ce4d30c387708",
        "extracted_subdir": "LibriSpeech/train-other-500",
    },
    # ── Evaluation ────────────────────────────────────────────────────────────
    "dev-clean": {
        "url": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
        "md5": "42e2234ba48799c1f50f24a7926300a1",
        "extracted_subdir": "LibriSpeech/dev-clean",
    },
    "dev-other": {
        "url": "https://www.openslr.org/resources/12/dev-other.tar.gz",
        "md5": "c8d0bcc9cca99d4f8b62fcc847357931",
        "extracted_subdir": "LibriSpeech/dev-other",
    },
    "test-clean": {
        "url": "https://www.openslr.org/resources/12/test-clean.tar.gz",
        "md5": "32fa31d27d2e1cad72775fee3f4849a9",
        "extracted_subdir": "LibriSpeech/test-clean",
    },
    "test-other": {
        "url": "https://www.openslr.org/resources/12/test-other.tar.gz",
        "md5": "fb5a50374b501bb3bac4815ee91d3135",
        "extracted_subdir": "LibriSpeech/test-other",
    },
}


class _ProgressHook:
    """urllib reporthook that drives a tqdm progress bar."""

    def __init__(self, desc: str) -> None:
        self._desc = desc
        self._bar: tqdm | None = None
        self._prev = 0

    def __call__(self, block_num: int, block_size: int, total_size: int) -> None:
        if self._bar is None:
            self._bar = tqdm(
                total=total_size if total_size > 0 else None,
                unit="B",
                unit_scale=True,
                desc=self._desc,
            )
        current = block_num * block_size
        self._bar.update(current - self._prev)
        self._prev = current

    def close(self) -> None:
        """Close the tqdm bar if it was opened."""
        if self._bar is not None:
            self._bar.close()


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    """Compute the MD5 hex digest of a file.

    Args:
        path:  file to hash
        chunk: read chunk size in bytes

    Returns:
        lowercase hex digest string
    """
    h = hashlib.md5()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(chunk), b""):
            h.update(data)
    return h.hexdigest()


def download_split(output_dir: Path, split_name: str, info: dict[str, str]) -> None:
    """Download one LibriSpeech split, verify its checksum, and extract it.

    Args:
        output_dir:  directory to save the .tar.gz archive
        split_name:  e.g. "train-clean-100"
        info:        dict with keys "url", "md5", "extracted_subdir"
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_name = info["url"].rsplit("/", 1)[-1]
    archive_path = output_dir / archive_name
    extracted_dir = output_dir / info["extracted_subdir"]

    # ── Download ──────────────────────────────────────────────────────────
    if archive_path.exists():
        print(f"[{split_name}] Found {archive_path.name} — verifying MD5 …")
        if _md5(archive_path) == info["md5"]:
            print(f"[{split_name}] MD5 OK — skipping download.")
        else:
            print(f"[{split_name}] MD5 mismatch — re-downloading.")
            archive_path.unlink()

    if not archive_path.exists():
        print(f"[{split_name}] Downloading {archive_name} …")
        hook = _ProgressHook(archive_name)
        try:
            urllib.request.urlretrieve(info["url"], archive_path, reporthook=hook)
        except Exception:
            if archive_path.exists():
                archive_path.unlink()
            raise
        finally:
            hook.close()

        actual_md5 = _md5(archive_path)
        if actual_md5 != info["md5"]:
            archive_path.unlink()
            raise RuntimeError(
                f"[{split_name}] MD5 mismatch after download.\n"
                f"  expected : {info['md5']}\n"
                f"  got      : {actual_md5}\n"
                "Archive removed — please retry."
            )
        print(f"[{split_name}] MD5 verified: {actual_md5}")

    # ── Extract ───────────────────────────────────────────────────────────
    if extracted_dir.exists():
        print(f"[{split_name}] {extracted_dir} already exists — skipping extraction.")
        return

    print(f"[{split_name}] Extracting {archive_path.name} → {output_dir} …")
    with tarfile.open(archive_path, "r:gz") as tf:
        members = tf.getmembers()
        for member in tqdm(members, desc=f"Extracting {split_name}", unit="file"):
            tf.extract(member, path=output_dir)
    print(f"[{split_name}] Extracted → {extracted_dir}")


def download_splits(output_dir: Path, splits: list[str]) -> None:
    """Download and verify a list of LibriSpeech training splits.

    Args:
        output_dir: directory to save downloaded archives and extracted data
        splits:     list of split names to download (keys of SPLITS)
    """
    for name in splits:
        if name not in SPLITS:
            raise ValueError(
                f"Unknown split '{name}'. "
                f"Available: {list(SPLITS.keys())}"
            )
        download_split(output_dir, name, SPLITS[name])


def main() -> None:
    """Parse CLI arguments and run the download."""
    parser = argparse.ArgumentParser(
        description="Download LibriSpeech training splits from OpenSLR.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="directory to save archives and extracted data",
    )
    _TRAIN_SPLITS = ["train-clean-100", "train-clean-360", "train-other-500"]
    parser.add_argument(
        "--splits",
        nargs="+",
        default=_TRAIN_SPLITS,
        choices=list(SPLITS.keys()),
        help=(
            "splits to download  "
            "(default: all three training splits; "
            "pass dev-clean dev-other test-clean test-other for eval splits)"
        ),
    )
    args = parser.parse_args()

    download_splits(args.output_dir, args.splits)
    print("\nAll requested splits downloaded and extracted.")


if __name__ == "__main__":
    main()

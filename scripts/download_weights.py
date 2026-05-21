"""Download model weights required for training.

1. Downloads OpenAI Whisper small checkpoint directly from OpenAI's CDN
   (no openai-whisper dependency — direct URL fetch). Verifies SHA256.
   Saves to --output_dir/whisper_small.pt.

2. Prints step-by-step instructions for downloading Llama 3.1 8B from
   Meta/HuggingFace (requires licence acceptance — cannot be automated).

Usage:
    python scripts/download_weights.py --output_dir weights/
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

from tqdm import tqdm

WHISPER_SMALL_URL = (
    "https://openaipublic.azureedge.net/main/whisper/models/"
    "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt"
)
WHISPER_SMALL_SHA256 = (
    "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794"
)


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


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Compute the SHA256 hex digest of a file.

    Args:
        path:  file to hash
        chunk: read chunk size in bytes

    Returns:
        lowercase hex digest string
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(chunk), b""):
            h.update(data)
    return h.hexdigest()


def download_whisper_small(output_dir: Path) -> None:
    """Fetch the Whisper small checkpoint and verify its SHA256 hash.

    If a file already exists at the destination with the correct hash,
    the download is skipped. A corrupted existing file is removed and
    re-downloaded.

    Args:
        output_dir: directory to write whisper_small.pt
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / "whisper_small.pt"

    if dest.exists():
        print(f"Found {dest} — verifying SHA256 …")
        if _sha256(dest) == WHISPER_SMALL_SHA256:
            print("SHA256 OK — skipping download.")
            return
        print("SHA256 mismatch on existing file — re-downloading.")
        dest.unlink()

    print(f"Downloading Whisper small → {dest}")
    hook = _ProgressHook("whisper_small.pt")
    try:
        urllib.request.urlretrieve(WHISPER_SMALL_URL, dest, reporthook=hook)
    except Exception:
        if dest.exists():
            dest.unlink()
        raise
    finally:
        hook.close()

    actual = _sha256(dest)
    if actual != WHISPER_SMALL_SHA256:
        dest.unlink()
        raise RuntimeError(
            f"SHA256 mismatch after download.\n"
            f"  expected : {WHISPER_SMALL_SHA256}\n"
            f"  got      : {actual}\n"
            "File removed — please retry."
        )
    print(f"SHA256 verified: {actual}")
    print(f"Saved → {dest}")


def print_llama_instructions() -> None:
    """Print manual download instructions for Llama 3.1 8B."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║           Llama 3.1 8B — Manual Download Instructions                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Meta requires licence acceptance before downloading Llama weights.          ║
║  This cannot be automated — follow the steps below once per machine.         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Option A — HuggingFace Hub (recommended):
  1. Accept the licence at:
       https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
  2. Generate an HF access token at:
       https://huggingface.co/settings/tokens
  3. Install huggingface_hub if not already present:
       pip install huggingface_hub
  4. Download the checkpoint (original Meta format):
       huggingface-cli download meta-llama/Meta-Llama-3.1-8B \\
           --local-dir weights/llama3.1-8b \\
           --include "original/*"

Option B — Meta's direct download:
  1. Request access at:
       https://ai.meta.com/resources/models-and-libraries/llama-downloads/
  2. Follow the download link in the approval e-mail.
  3. Place the resulting files under weights/llama3.1-8b/

Expected final layout:
  weights/
    whisper_small.pt              ← downloaded by this script
    llama3.1-8b/
      params.json                 ← model hyperparameters
      tokenizer.model             ← SentencePiece model
      consolidated.00.pth         ← weight shard(s)
      …
""")


def main() -> None:
    """Parse CLI arguments and run the download."""
    parser = argparse.ArgumentParser(
        description="Download Whisper small checkpoint and print Llama instructions.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("weights"),
        help="directory to save whisper_small.pt  (default: weights/)",
    )
    args = parser.parse_args()

    download_whisper_small(args.output_dir)
    print_llama_instructions()


if __name__ == "__main__":
    main()

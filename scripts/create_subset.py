"""Write a shard list for a reproducible training subset.

Simulates the shard assignment that the training loop performs for a specific
rank and epoch, using the same shuffling formula:

    random.Random(base_seed + epoch).shuffle(all_shards)

For single-GPU training (num_ranks=1, rank=0), all shards are assigned to
rank 0. For multi-GPU (num_ranks > 1), shards are split interleaved across
ranks — rank r receives all_shards[r::num_ranks] — matching WebDataset's
split_by_node behaviour.

The output is a plain-text file listing one absolute shard path per line,
ready to be loaded by data.list_shards() or passed directly to
data.build_dataloader().

Usage:
    python scripts/create_subset.py \\
      --shard_dir   data/shards/ \\
      --split       train-clean-100 \\
      --n_shards    10 \\
      --base_seed   42 \\
      --epoch       0 \\
      --rank        0 \\
      --num_ranks   1 \\
      --output      data/subset_shards.txt

Then in the training loop:
    with open("data/subset_shards.txt") as f:
        shards = [line.strip() for line in f if line.strip()]
    loader = build_dataloader(shards, ...)
"""

from __future__ import annotations

import argparse
import glob
import random
from pathlib import Path


def create_shard_subset(
    shard_dir: Path,
    split: str,
    n_shards: int | None,
    base_seed: int,
    epoch: int,
    rank: int,
    num_ranks: int,
    output_path: Path,
) -> None:
    """Compute the shard list for a given rank/epoch and write it to a file.

    Args:
        shard_dir:   directory containing the .tar shard files
        split:       shard filename prefix (e.g. "train-clean-100")
        n_shards:    how many shards to include; None = all shards for this rank
        base_seed:   base random seed; epoch seed = base_seed + epoch
        epoch:       epoch number (0-indexed); determines the shuffle permutation
        rank:        this rank's index (0-indexed)
        num_ranks:   total number of ranks (GPUs); 1 for single-GPU
        output_path: destination text file (one shard path per line)
    """
    pattern = str(shard_dir / f"{split}-*.tar")
    all_shards = sorted(glob.glob(pattern))

    if not all_shards:
        raise FileNotFoundError(
            f"No shards matching '{pattern}'. "
            "Run scripts/preprocess.py first."
        )

    # Apply the same epoch shuffle as the training loop
    epoch_shards = list(all_shards)
    random.Random(base_seed + epoch).shuffle(epoch_shards)

    # Interleaved rank split — rank r gets indices r, r+num_ranks, r+2*num_ranks, …
    rank_shards = epoch_shards[rank::num_ranks]

    if n_shards is not None:
        rank_shards = rank_shards[:n_shards]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for path in rank_shards:
            f.write(path + "\n")

    total = len(all_shards)
    print(
        f"Wrote {len(rank_shards)} shard path(s) → {output_path}\n"
        f"  total shards available : {total}\n"
        f"  epoch shuffle seed     : {base_seed + epoch}  (base_seed={base_seed}, epoch={epoch})\n"
        f"  rank                   : {rank} / {num_ranks}\n"
        f"  shards assigned before slice : {len(epoch_shards[rank::num_ranks])}"
    )


def main() -> None:
    """Parse CLI arguments and create the shard subset list."""
    parser = argparse.ArgumentParser(
        description="Generate a shard list for a specific rank and epoch.",
    )
    parser.add_argument(
        "--shard_dir",
        type=Path,
        required=True,
        help="directory containing the .tar shard files",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="shard filename prefix, e.g. train-clean-100",
    )
    parser.add_argument(
        "--n_shards",
        type=int,
        default=None,
        help="number of shards to include; omit to include all shards for this rank",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="base random seed; epoch seed = base_seed + epoch  (default: 42)",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=0,
        help="epoch number, 0-indexed  (default: 0)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="rank index, 0-indexed  (default: 0)",
    )
    parser.add_argument(
        "--num_ranks",
        type=int,
        default=1,
        help="total number of ranks/GPUs  (default: 1 for single-GPU)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="destination text file (one shard path per line)",
    )
    args = parser.parse_args()

    create_shard_subset(
        shard_dir=args.shard_dir,
        split=args.split,
        n_shards=args.n_shards,
        base_seed=args.base_seed,
        epoch=args.epoch,
        rank=args.rank,
        num_ranks=args.num_ranks,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

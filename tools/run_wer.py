"""Multi-checkpoint post-hoc WER evaluation tool.

Sweeps one or more checkpoints against the eval splits defined in a training
config and writes a WER summary to a CSV or JSON file.

Usage:
    python tools/run_wer.py \\
        --config  configs/example.yaml \\
        --checkpoints checkpoints/run/step_*.pt \\
        --output  results/wer.csv

    # Full eval (no max_batches cap)
    python tools/run_wer.py --config ... --checkpoints ... --output ... --full

    # W&B logging
    python tools/run_wer.py --config ... --checkpoints ... --output ... --wandb

NOTE: The config must have data.shards_file or data.shards set (to pass
validation) even though run_wer.py does not consume training shards.
Any training run config is suitable as-is.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build import build_models
from data import INSTRUCTION_VARIANTS, PrunedTokenizer, build_eval_dataloader
from utils.checkpoint import load_weights
from utils.config import load_config
from utils.evaluate import evaluate_all_splits


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-hoc WER evaluation across multiple checkpoints.",
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Training config YAML (provides model dims, tokenizer, eval splits, "
             "eval_batch_size, wer.max_batches, wer.sample_transcriptions).",
    )
    parser.add_argument(
        "--checkpoints", nargs="+", required=True, metavar="PATH",
        help="One or more checkpoint paths or shell globs (e.g. checkpoints/run/step_*.pt). "
             "Each argument is glob-expanded and the union sorted.",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output file path.  Extension determines format: .csv (default) or .json.",
    )
    parser.add_argument(
        "--max-batches", type=int, default=None, dest="max_batches",
        help="Override cfg.metrics.wer.max_batches (batches per split per checkpoint).",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Evaluate entire splits with no batch cap (overrides --max-batches "
             "and cfg.metrics.wer.max_batches).",
    )
    parser.add_argument(
        "--sample-transcriptions", type=int, default=None, dest="sample_transcriptions",
        help="Override cfg.metrics.wer.sample_transcriptions.",
    )
    parser.add_argument(
        "--formats", nargs="+", choices=["unformatted", "formatted"],
        default=None, metavar="FORMAT",
        help="Instruction variant(s) to evaluate: 'unformatted', 'formatted', or both "
             "(default: both).  Pass a single value to halve generation time.",
    )
    parser.add_argument(
        "--wandb", action=argparse.BooleanOptionalAction, default=False,
        help="Log WER-vs-step to W&B (requires WANDB_API_KEY env var).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device string (default: cuda if available, else cpu).",
    )
    return parser.parse_args(argv)


def _expand_checkpoints(patterns: list[str]) -> list[Path]:
    """Glob-expand each pattern and return a deduplicated sorted list of Paths."""
    paths: list[Path] = []
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(pat))   # treat as literal path; will fail loudly later
    # deduplicate, preserving sort
    seen: set[Path] = set()
    result: list[Path] = []
    for p in sorted(set(paths)):
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def _parse_step(stem: str, fallback: int) -> int:
    """Extract the optimizer step from a filename stem (e.g. 'step0001234').

    Returns fallback (checkpoint index) when no numeric suffix is found.
    """
    m = re.search(r'step[_-]?(\d+)', stem, re.IGNORECASE)
    return int(m.group(1)) if m else fallback


def main(argv: list[str] | None = None) -> None:
    """Load config, sweep checkpoints, and write WER summary."""
    args = _parse_args(argv)

    cfg = load_config(args.config)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # ── Tokenizer + sep_token_id ──────────────────────────────────────────────
    with (cfg.data.tokenizer / "pruned_config.json").open() as f:
        pc = json.load(f)
    sep_token_id = pc["sep_token_id"]
    tokenizer    = PrunedTokenizer(cfg.data.tokenizer)

    # ── Eval loaders (one per configured split) ───────────────────────────────
    eval_cfg = cfg.data.eval
    eval_shard_map: list[tuple[str, Path]] = [
        ("dev-clean",  eval_cfg.dev_clean),
        ("dev-other",  eval_cfg.dev_other),
        ("test-clean", eval_cfg.test_clean),
        ("test-other", eval_cfg.test_other),
    ]

    eval_loaders: dict[str, torch.utils.data.DataLoader] = {}
    for split_name, shard_path in eval_shard_map:
        if shard_path is None:
            continue
        if not Path(shard_path).exists():
            print(f"[warn] eval shard for {split_name} not found: {shard_path} — skipping")
            continue
        eval_loaders[split_name] = build_eval_dataloader(
            shard_path,
            tokenizer_path        = cfg.data.tokenizer,
            instruction_variants  = INSTRUCTION_VARIANTS,
            batch_size            = cfg.metrics.eval_batch_size,
        )
        print(f"Eval loader: {split_name} → {shard_path}")

    if not eval_loaders:
        print(
            "[warn] No eval loaders available — either no eval shards are configured "
            "(data.eval.*) or none of the configured shard files exist on disk.  Exiting."
        )
        return

    # ── Build base model (pretrained weights, no checkpoint overlay yet) ──────
    encoder, adapter, llama = build_models(cfg, device, train=False, apply_init_from=False)

    # ── Resolve evaluation parameters ─────────────────────────────────────────
    if args.full:
        max_batches: int | None = None
    elif args.max_batches is not None:
        max_batches = args.max_batches
    else:
        max_batches = cfg.metrics.wer.max_batches

    n_samples = args.sample_transcriptions or cfg.metrics.wer.sample_transcriptions

    # ── W&B initialisation ────────────────────────────────────────────────────
    use_wandb = args.wandb
    if use_wandb:
        import os
        import wandb as _wandb
        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "WANDB_API_KEY is not set. "
                "Add 'export WANDB_API_KEY=...' to ~/.bashrc and reload your shell."
            )
        run_name = (cfg.logging.run_name or "run_wer") + "_wer"
        _wandb.init(project=cfg.logging.project, name=run_name)

    # ── Expand + sort checkpoints ─────────────────────────────────────────────
    checkpoints = _expand_checkpoints(args.checkpoints)
    if not checkpoints:
        print("[error] No checkpoint paths resolved from --checkpoints.")
        sys.exit(1)
    print(f"Evaluating {len(checkpoints)} checkpoint(s).")

    # ── Checkpoint sweep ──────────────────────────────────────────────────────
    seen_key_sets: set[frozenset[str]] = set()
    all_rows:    list[dict] = []
    all_samples: list[dict] = []

    for ckpt_idx, ckpt_path in enumerate(checkpoints):
        if not ckpt_path.exists():
            print(f"[warn] checkpoint not found: {ckpt_path} — skipping")
            continue

        step = _parse_step(ckpt_path.stem, fallback=ckpt_idx)
        print(f"\n[{ckpt_idx + 1}/{len(checkpoints)}] {ckpt_path.name}  (step={step})")

        # Overlay weights from this checkpoint
        loaded = load_weights(ckpt_path, encoder=encoder, adapter=adapter, llama=llama)
        key_set = frozenset(loaded)

        # Heterogeneity guard
        if seen_key_sets and key_set not in seen_key_sets:
            print(
                f"[warn] Checkpoint {ckpt_path.name} provides modules {sorted(key_set)} "
                f"but earlier checkpoint(s) provided "
                f"{[sorted(s) for s in seen_key_sets]}.  "
                "Mixed-module sweeps can leave stale encoder/llama weights between "
                "checkpoints and may produce incorrect WER numbers.  "
                "Recommend grouping checkpoints by module set."
            )
        seen_key_sets.add(key_set)

        # Ensure eval mode (evaluate_all_splits also sets it, but be explicit)
        encoder.eval()
        adapter.eval()
        llama.eval()

        # Run WER evaluation
        wer_results, sample_rows = evaluate_all_splits(
            encoder, adapter, llama,
            eval_loaders, tokenizer, sep_token_id, device,
            max_batches  = max_batches,
            n_samples    = n_samples,
            sample_seed  = step,
            formats      = args.formats,
        )

        # Accumulate rows
        for key, wer_val in wer_results.items():
            split, fmt = key.rsplit("/", 1)
            all_rows.append({
                "checkpoint": str(ckpt_path),
                "step":       step,
                "split":      split,
                "format":     fmt,
                "wer":        wer_val,
                "n_samples":  n_samples,
            })

        for sr in sample_rows:
            all_samples.append({
                "checkpoint": str(ckpt_path),
                "step":       step,
                **sr,
            })

        # W&B per-step logging
        if use_wandb:
            import wandb as _wandb
            wandb_payload: dict = {}
            for key, wer_val in wer_results.items():
                split, fmt = key.rsplit("/", 1)
                wandb_payload[f"wer/{split}/{fmt}"] = wer_val

            trans_rows = [
                (sr["split"], sr["type"], sr["reference"], sr["hypothesis"])
                for sr in sample_rows
            ]
            if trans_rows:
                tbl = _wandb.Table(columns=["split", "type", "reference", "hypothesis"])
                for r in trans_rows:
                    tbl.add_data(*r)
                wandb_payload["transcriptions"] = tbl

            _wandb.log(wandb_payload, step=step)

    # ── Print readable summary table ──────────────────────────────────────────
    col_w = min(40, max((len(Path(r["checkpoint"]).name) for r in all_rows), default=10))
    hdr = f"{'Checkpoint':<{col_w}}  {'Step':>8}  {'Split':<12}  {'Format':<14}  {'WER':>7}"
    print(f"\n── WER Summary ──\n{hdr}\n{'-' * len(hdr)}")
    for row in all_rows:
        name = Path(row["checkpoint"]).name
        print(
            f"{name:<{col_w}}  {row['step']:>8}  {row['split']:<12}  "
            f"{row['format']:<14}  {row['wer']:>6.1%}"
        )

    # ── Write CSV / JSON output ───────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".json":
        with args.output.open("w") as f:
            json.dump(all_rows, f, indent=2)
    else:
        with args.output.open("w", newline="") as f:
            if all_rows:
                writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
                writer.writeheader()
                writer.writerows(all_rows)
    print(f"\nWER summary → {args.output}")

    # ── Write sample transcriptions ───────────────────────────────────────────
    samples_path = args.output.parent / (args.output.stem + "_samples.jsonl")
    with samples_path.open("w") as f:
        for sr in all_samples:
            f.write(json.dumps(sr) + "\n")
    print(f"Sample transcriptions → {samples_path}")

    if use_wandb:
        import wandb as _wandb
        _wandb.finish()


if __name__ == "__main__":
    main(argv=sys.argv[1:])

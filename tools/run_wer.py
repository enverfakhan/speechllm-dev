"""Multi-checkpoint post-hoc WER evaluation tool.

Sweeps one or more checkpoints against the eval splits defined in a training
config and writes a WER summary to a CSV or JSON file, plus one JSONL file per
(checkpoint, split) holding EVERY transcription it produced.

Usage:
    python tools/run_wer.py \\
        --config  configs/example.yaml \\
        --checkpoints checkpoints/run/step_*.pt \\
        --output  results/wer.csv

    # Full eval (no max_batches cap)
    python tools/run_wer.py --config ... --checkpoints ... --output ... --full

    # Only the dev splits
    python tools/run_wer.py --config ... --checkpoints ... --output ... \\
        --splits dev-clean dev-other

    # Out-of-distribution sets (shards outside cfg.data.eval)
    python tools/run_wer.py --config ... --checkpoints ... \\
        --eval-tar tedlium3-test-le41=data/ood_shards/tedlium3-test/tedlium3-test-le41.tar \\
        --dataset tedlium3-test-le41 --formats unformatted --full \\
        --output results/ood/wer.csv

    # W&B logging
    python tools/run_wer.py --config ... --checkpoints ... --output ... --wandb

    # Plumbing self-test (mocked model + evaluator; no GPU, no data)
    python tools/run_wer.py --self-test

Outputs, all under the --output directory:
    <output>                        summary rows (CSV or JSON), one per
                                    (checkpoint, split, format), carrying the
                                    checkpoint's stage/epoch metadata
    {step:07d}_{split}.jsonl        one row per evaluated sample per format —
                                    reference, hypothesis, per-sample WER and
                                    the utterance key.  Generation is the
                                    expensive part of this tool, so nothing is
                                    thrown away: later analyses (see
                                    tools/count_degeneracies.py) read these
                                    files instead of re-running the sweep.

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
from data import (
    build_sorted_eval_dataloader,
    load_pruned_config,
    INSTRUCTION_VARIANTS,
    PrunedTokenizer,
)
from model.sequence import ChatTemplate
from utils.checkpoint import apply_weights, read_checkpoint
from utils.config import Config, load_config
from utils.evaluate import compute_wer, evaluate_all_splits

# Every eval split this tool knows about, in report order.
SPLIT_NAMES: tuple[str, ...] = ("dev-clean", "dev-other", "test-clean", "test-other")


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
        "--splits", nargs="+", choices=list(SPLIT_NAMES),
        default=None, metavar="SPLIT",
        help="Eval split(s) to run (default: all four configured splits).  "
             "Restricting to the dev splits roughly halves wall-clock; the "
             "per-(step, split) JSONL naming keeps a later test-split run from "
             "colliding with this one's outputs.",
    )
    parser.add_argument(
        "--eval-tar", nargs="+", default=None, dest="eval_tars", metavar="NAME=PATH",
        help="Evaluate arbitrary eval shard .tar files instead of the four splits "
             "in cfg.data.eval — the out-of-distribution protocol (TED-LIUM, "
             "Common Voice, Earnings-22) points here.  NAME replaces the split "
             "name on every summary and JSONL row, and names the per-checkpoint "
             "JSONL ({step:07d}_{NAME}.jsonl), so an OOD sweep never collides "
             "with a LibriSpeech one.  Mutually exclusive with --splits.",
    )
    parser.add_argument(
        "--dataset", type=str, default=None, metavar="TAG",
        help="Dataset tag written onto every summary and JSONL row.  Lets one "
             "analysis read several corpora's dumps without inferring the corpus "
             "from a filename.",
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
        "--progress-interval", type=float, default=30.0, dest="progress_interval",
        metavar="SECONDS",
        help="Print a progress line every N seconds per split (default: 30). "
             "Pass 0 to disable.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device string (default: cuda if available, else cpu).",
    )
    return parser.parse_args(argv)


def _parse_eval_tars(specs: list[str]) -> list[tuple[str, Path]]:
    """Parse --eval-tar NAME=PATH arguments into (name, path) pairs.

    The name is not cosmetic: it lands on every summary and JSONL row and names
    the output files, so a typo that silently became a path stem would attribute
    a corpus's results to the wrong dataset.  Hence the explicit form and the
    loud failure.

    Args:
        specs: raw "NAME=PATH" strings

    Returns:
        (name, path) pairs, in the order given

    Raises:
        ValueError: a spec is missing "=", names a duplicate, or points nowhere
    """
    pairs: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for spec in specs:
        name, sep, raw = spec.partition("=")
        if not sep or not name or not raw:
            raise ValueError(
                f"--eval-tar expects NAME=PATH, got {spec!r} "
                "(e.g. tedlium3-test=data/ood_shards/tedlium3-test/tedlium3-test-le41.tar)"
            )
        if name in seen:
            raise ValueError(f"--eval-tar name {name!r} given twice")
        path = Path(raw)
        if not path.exists():
            raise ValueError(f"--eval-tar {name}: {path} does not exist")
        seen.add(name)
        pairs.append((name, path))
    return pairs


def _expand_checkpoints(patterns: list[str]) -> list[Path]:
    """Glob-expand each pattern and return a deduplicated sorted list of Paths.

    Raises:
        ValueError: any resolved path is an adapter sidecar (``*-adapter.pt``).
            training.py writes one of those next to every handoff checkpoint and
            next to each periodic save of an adapter-only stage, so the obvious
            glob ``step*.pt`` sweeps them up alongside the real checkpoints.  A
            sidecar carries only the bridge (plus audio adapters) — loading one
            would overlay a fraction of the delta and silently report WER for a
            model that never existed.  Fail here instead.
    """
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

    sidecars = [p for p in result if p.name.endswith("-adapter.pt")]
    if sidecars:
        raise ValueError(
            "--checkpoints resolved to adapter sidecar file(s):\n  "
            + "\n  ".join(str(p) for p in sidecars)
            + "\nA '-adapter.pt' file is a weights-only sidecar holding the bridge "
              "(and audio adapters) alone, not a full checkpoint delta — evaluating "
              "one would leave every other trained module at its pretrained value "
              "and report a WER for a model that was never trained.  List the "
              "'step*.pt' / '*-stage-handoff.pt' files explicitly, or use a glob "
              "that excludes the sidecars."
        )
    return result


def _parse_step(stem: str, fallback: int) -> int:
    """Extract the optimizer step from a filename stem (e.g. 'step0001234').

    Returns fallback (checkpoint index) when no numeric suffix is found.
    """
    m = re.search(r'step[_-]?(\d+)', stem, re.IGNORECASE)
    return int(m.group(1)) if m else fallback


def _checkpoint_metadata(ckpt: dict, cfg: Config) -> dict:
    """Extract the stage/epoch fields that tag every summary row.

    training.py stores these on both checkpoint kinds; legacy files predate some
    of them, hence the defaults.  ``epoch`` is 0-based and PER STAGE (the epoch
    that was running when the file was written), so it is renamed here to
    epoch_in_stage — plotting tools must not read it as a global epoch counter.

    Args:
        ckpt: checkpoint dict from read_checkpoint()
        cfg:  the training Config, for resolving stage_index → stage name

    Returns:
        {stage_index, stage_name, epoch_in_stage, step_in_stage, kind}
    """
    stage_index = int(ckpt.get("stage_index", 0))
    # A checkpoint swept with a config whose stage list has since been edited
    # would index out of range; report that rather than crashing the sweep.
    if 0 <= stage_index < len(cfg.stages):
        stage_name = cfg.stages[stage_index].name
    else:
        stage_name = f"stage_{stage_index}?"

    return {
        "stage_index":    stage_index,
        "stage_name":     stage_name,
        "epoch_in_stage": int(ckpt.get("epoch", 0)),
        "step_in_stage":  int(ckpt.get("step_in_stage", ckpt.get("step", 0))),
        # Untagged legacy checkpoints are periodic by definition (Decision 004).
        "kind":           str(ckpt.get("kind", "periodic")),
    }


def _sample_wer(reference: str, hypothesis: str) -> float:
    """WER of a single (reference, hypothesis) pair, via the aggregate's jiwer path.

    Same engine and no text normalisation, so a per-sample number is directly
    comparable to the split aggregate (which is a corpus-level WER, not the mean
    of these).  A zero-length reference makes WER undefined — jiwer's behaviour
    there varies by version — so it yields NaN.  LibriSpeech references are never
    empty; this is defensive, and one bad row must not kill a sweep whose
    generation cost is already sunk.
    """
    if not reference.strip():
        return float("nan")
    return compute_wer([reference], [hypothesis])


def _write_transcriptions(
    path: Path, rows: list[dict], checkpoint: str, step: int,
    dataset: str | None = None,
) -> None:
    """Write one JSONL row per evaluated sample per format.

    Field names match what tools/count_degeneracies.py consumes (split, type,
    reference, hypothesis, plus checkpoint/step for grouping); ``key`` (the
    utterance id) and ``wer`` (per-sample) are additions it ignores.

    Args:
        path:       output .jsonl (overwritten)
        rows:       transcription dicts from evaluate_all_splits(
                        return_all_transcriptions=True)
        checkpoint: checkpoint path string, recorded on every row
        step:       global optimizer step of that checkpoint
        dataset:    optional corpus tag (--dataset), recorded on every row so an
                    analysis spanning several corpora need not parse filenames
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps({
                "checkpoint": checkpoint,
                "step":       step,
                **({"dataset": dataset} if dataset else {}),
                **row,
                "wer":        _sample_wer(row["reference"], row["hypothesis"]),
            }) + "\n")


def main(argv: list[str] | None = None) -> None:
    """Load config, sweep checkpoints, and write WER summary."""
    args = _parse_args(argv)

    cfg = load_config(args.config)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # ── Tokenizer + terminator + input convention ─────────────────────────────
    terminator_id = load_pruned_config(cfg.data.tokenizer).terminator_id
    tokenizer     = PrunedTokenizer(cfg.data.tokenizer)

    # Decoding must use the SAME sequence convention the checkpoints were trained
    # under, or every hypothesis is generated from a prompt the model never saw.
    chat: ChatTemplate | None = None
    if cfg.model.input_convention == "chat":
        chat = ChatTemplate.from_tokenizer(tokenizer)
        print(f"Chat convention: audio offset {chat.audio_offset}, "
              f"stop token <|eot_id|> = {chat.eot_token_id}")

    # ── Eval loaders (one per configured split, or per --eval-tar) ────────────
    eval_shard_map: list[tuple[str, Path]]
    if args.eval_tars:
        if args.splits is not None:
            print("[error] --eval-tar and --splits are mutually exclusive: --splits "
                  "names the configured LibriSpeech splits, which --eval-tar replaces.")
            sys.exit(1)
        eval_shard_map = _parse_eval_tars(args.eval_tars)
        print("Eval sets from --eval-tar: "
              + ", ".join(f"{n} → {p}" for n, p in eval_shard_map))
    else:
        eval_cfg = cfg.data.eval
        eval_shard_map = [
            ("dev-clean",  eval_cfg.dev_clean),
            ("dev-other",  eval_cfg.dev_other),
            ("test-clean", eval_cfg.test_clean),
            ("test-other", eval_cfg.test_other),
        ]
        if args.splits is not None:
            requested = set(args.splits)
            eval_shard_map = [(n, p) for n, p in eval_shard_map if n in requested]
            print(f"Splits restricted to: {', '.join(n for n, _ in eval_shard_map)}")

    eval_loaders: dict[str, list[tuple]] = {}
    for split_name, shard_path in eval_shard_map:
        if shard_path is None:
            continue
        if not Path(shard_path).exists():
            print(f"[warn] eval shard for {split_name} not found: {shard_path} — skipping")
            continue
        eval_loaders[split_name] = build_sorted_eval_dataloader(
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
    # apply_init_from=False is correct under the delta invariant: every swept
    # checkpoint is a COMPLETE delta over the pretrained base, so the base model
    # must be pretrained-only and each checkpoint overlays its own delta below
    # (load_weights per checkpoint).  Applying cfg.model.init_from here would
    # leave stale warm-start weights bleeding across checkpoints.  The 4th return
    # (init_from-loaded names) is empty here and unused.
    encoder, adapter, llama, _ = build_models(cfg, device, train=False, apply_init_from=False)

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

    # ── Output directory (mirrors --output location) ─────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    progress_interval = args.progress_interval if args.progress_interval > 0 else None

    # ── Checkpoint sweep ──────────────────────────────────────────────────────
    seen_key_sets: set[frozenset[str]] = set()
    all_rows: list[dict] = []

    for ckpt_idx, ckpt_path in enumerate(checkpoints):
        if not ckpt_path.exists():
            print(f"[warn] checkpoint not found: {ckpt_path} — skipping")
            continue

        step = _parse_step(ckpt_path.stem, fallback=ckpt_idx)
        print(f"\n[{ckpt_idx + 1}/{len(checkpoints)}] {ckpt_path.name}  (step={step})")

        # One read gives both halves: the weight states to overlay and the
        # stage/epoch metadata that tags this checkpoint's summary rows.
        ckpt   = read_checkpoint(ckpt_path)
        meta   = _checkpoint_metadata(ckpt, cfg)
        loaded = apply_weights(ckpt, encoder=encoder, adapter=adapter, llama=llama)
        del ckpt   # a full-llama checkpoint is ~32 GB of host RAM; drop it now
        key_set = frozenset(loaded)

        # Heterogeneity guard
        if seen_key_sets and key_set not in seen_key_sets:
            grew = all(s <= key_set for s in seen_key_sets)
            print(
                f"[warn] Checkpoint {ckpt_path.name} provides modules {sorted(key_set)} "
                f"but earlier checkpoint(s) provided "
                f"{[sorted(s) for s in seen_key_sets]}.  "
                "Mixed-module sweeps can leave stale weights between checkpoints: a "
                "module present in an EARLIER checkpoint but absent from a later one "
                "keeps the earlier checkpoint's trained values instead of falling back "
                "to pretrained, so the later WER is measured on a model that never "
                "existed.\n"
                + (
                    "         This sweep is the safe case: the module set only GREW, "
                    "and checkpoints run in ascending step order, so every module "
                    "each checkpoint omits is still at its pretrained value — exactly "
                    "what that checkpoint was trained against (a cumulative-unfreeze "
                    "run reaches this warning once, at its first encoder-bearing "
                    "checkpoint)."
                    if grew else
                    "         The module set did NOT simply grow here — group the "
                    "checkpoints by module set and sweep each group separately."
                )
            )
        seen_key_sets.add(key_set)

        encoder.eval()
        adapter.eval()
        llama.eval()

        # Evaluate one split at a time so we can flush the sample file immediately.
        ckpt_wer_results: dict[str, float] = {}
        ckpt_sample_rows: list[dict] = []

        for split_name, loader in eval_loaders.items():
            split_wer, split_samples, split_all = evaluate_all_splits(
                encoder, adapter, llama,
                {split_name: loader},
                tokenizer, terminator_id, device,
                max_batches       = max_batches,
                n_samples         = n_samples,
                sample_seed       = step,
                formats           = args.formats,
                progress_interval = progress_interval,
                chat                      = chat,
                return_all_transcriptions = True,
            )
            ckpt_wer_results.update(split_wer)

            # Write EVERY transcription for this (step, split) immediately, so a
            # sweep that dies at checkpoint 9 still leaves 8 usable dumps behind.
            trans_path = args.output.parent / f"{step:07d}_{split_name}.jsonl"
            _write_transcriptions(trans_path, split_all, str(ckpt_path), step,
                                  dataset=args.dataset)
            print(f"  transcriptions → {trans_path.name}  ({len(split_all)} rows)")

            # The sampled subset stays capped at n_samples: it feeds a W&B table.
            ckpt_sample_rows.extend(split_samples)

        # Accumulate summary rows
        for key, wer_val in ckpt_wer_results.items():
            split, fmt = key.rsplit("/", 1)
            all_rows.append({
                "checkpoint": str(ckpt_path),
                "step":       step,
                **({"dataset": args.dataset} if args.dataset else {}),
                "split":      split,
                "format":     fmt,
                "wer":        wer_val,
                "n_samples":  n_samples,
                **meta,
            })

        # W&B per-step logging
        if use_wandb:
            import wandb as _wandb
            wandb_payload: dict = {}
            for key, wer_val in ckpt_wer_results.items():
                split, fmt = key.rsplit("/", 1)
                wandb_payload[f"wer/{split}/{fmt}"] = wer_val

            trans_rows = [
                (sr["split"], sr["type"], sr["reference"], sr["hypothesis"])
                for sr in ckpt_sample_rows
            ]
            if trans_rows:
                tbl = _wandb.Table(columns=["split", "type", "reference", "hypothesis"])
                for r in trans_rows:
                    tbl.add_data(*r)
                wandb_payload["transcriptions"] = tbl

            _wandb.log(wandb_payload, step=step)

    # ── Print readable summary table ──────────────────────────────────────────
    col_w = min(40, max((len(Path(r["checkpoint"]).name) for r in all_rows), default=10))
    stage_w = min(22, max((len(str(r["stage_name"])) for r in all_rows), default=5))
    hdr = (f"{'Checkpoint':<{col_w}}  {'Step':>8}  {'Stage':<{stage_w}}  "
           f"{'Split':<12}  {'Format':<14}  {'WER':>7}")
    print(f"\n── WER Summary ──\n{hdr}\n{'-' * len(hdr)}")
    for row in all_rows:
        name = Path(row["checkpoint"]).name
        print(
            f"{name:<{col_w}}  {row['step']:>8}  {str(row['stage_name']):<{stage_w}}  "
            f"{row['split']:<12}  {row['format']:<14}  {row['wer']:>6.1%}"
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

    if use_wandb:
        import wandb as _wandb
        _wandb.finish()


# ── Self-test ─────────────────────────────────────────────────────────────────

def _self_test() -> None:
    """Exercise the CLI plumbing with a mocked model, evaluator and checkpoints.

    Covers the parts that are cheap to get wrong and expensive to discover on a
    pod: the sidecar guard, --splits filtering, and the JSONL row schema (which
    tools/count_degeneracies.py reads).  No torch model is built — build_models,
    the eval loaders and evaluate_all_splits are all replaced in module globals.
    """
    import tempfile

    from utils.config import (
        Config, DataConfig, EvalShards, LoggingConfig, MetricsConfig, StageConfig, WerConfig,
    )

    print("run_wer.py self-test")

    # ── 1. Sidecar guard ──────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as _td:
        ck = Path(_td)
        for name in ("step0000360.pt", "step0000360-adapter.pt",
                     "step0004680-stage-handoff.pt", "step0004680-stage-handoff-adapter.pt"):
            (ck / name).touch()

        try:
            _expand_checkpoints([str(ck / "step*.pt")])
            raise AssertionError("a glob matching sidecars must raise")
        except ValueError as exc:
            assert "-adapter.pt" in str(exc), exc
            assert "step0000360-adapter.pt" in str(exc), exc
        print("  [OK] _expand_checkpoints rejects '-adapter.pt' sidecars")

        explicit = _expand_checkpoints(
            [str(ck / "step0000360.pt"), str(ck / "step0004680-stage-handoff.pt")]
        )
        assert [p.name for p in explicit] == [
            "step0000360.pt", "step0004680-stage-handoff.pt"
        ], explicit
        print("  [OK] _expand_checkpoints accepts explicit non-sidecar paths")

    # ── 1b. --eval-tar parsing ────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as _td:
        tar_a = Path(_td) / "a.tar"
        tar_b = Path(_td) / "b.tar"
        tar_a.touch()
        tar_b.touch()

        pairs = _parse_eval_tars([f"ted={tar_a}", f"cv={tar_b}"])
        assert pairs == [("ted", tar_a), ("cv", tar_b)], pairs

        for spec, why in (
            (f"{tar_a}",          "no NAME= prefix"),
            ("ted=",              "empty path"),
            (f"=/{tar_a}",        "empty name"),
            ("ted=/nope/x.tar",   "path does not exist"),
        ):
            try:
                _parse_eval_tars([spec])
            except ValueError:
                continue
            raise AssertionError(f"_parse_eval_tars accepted {why}: {spec!r}")

        try:
            _parse_eval_tars([f"ted={tar_a}", f"ted={tar_b}"])
        except ValueError as exc:
            assert "twice" in str(exc), exc
        else:
            raise AssertionError("_parse_eval_tars accepted a duplicate name")
        print("  [OK] _parse_eval_tars: pairs, and fails loudly on every malformed spec")

    # ── 2. Checkpoint metadata → summary columns ──────────────────────────────
    cfg_stages = [StageConfig(name=n) for n in
                  ("bridge_only", "audio_adapters_only", "bridge_plus_adapters", "full_stack")]
    meta = _checkpoint_metadata(
        {"step": 9360, "epoch": 1, "step_in_stage": 4680, "stage_index": 1, "kind": "handoff"},
        Config(stages=cfg_stages),
    )
    assert meta == {
        "stage_index": 1, "stage_name": "audio_adapters_only",
        "epoch_in_stage": 1, "step_in_stage": 4680, "kind": "handoff",
    }, meta
    # Legacy file: no kind / stage fields at all.
    legacy = _checkpoint_metadata({"step": 100}, Config(stages=cfg_stages))
    assert legacy["kind"] == "periodic" and legacy["stage_index"] == 0, legacy
    assert legacy["step_in_stage"] == 100, legacy
    # stage_index past the configured list must not crash the sweep.
    oob = _checkpoint_metadata({"step": 1, "stage_index": 9}, Config(stages=cfg_stages))
    assert oob["stage_name"] == "stage_9?", oob
    print("  [OK] _checkpoint_metadata: full, legacy and out-of-range stage_index")

    # ── 3. Per-sample WER ─────────────────────────────────────────────────────
    assert _sample_wer("a b c d", "a b c d") == 0.0
    assert _sample_wer("a b c d", "a b c x") == 0.25
    assert _sample_wer("a b c d", "") == 1.0
    import math as _math
    assert _math.isnan(_sample_wer("", "anything")), "empty reference → NaN"
    print("  [OK] _sample_wer: exact, one substitution, empty hyp, empty ref")

    # ── 4. End-to-end main() with everything heavy mocked ─────────────────────
    globals_backup = {
        name: globals()[name]
        for name in ("load_config", "build_models", "PrunedTokenizer",
                     "build_sorted_eval_dataloader", "read_checkpoint",
                     "apply_weights", "evaluate_all_splits")
    }
    requested_splits: list[str] = []

    with tempfile.TemporaryDirectory() as _td:
        tmp     = Path(_td)
        tok_dir = tmp / "tokenizer"
        tok_dir.mkdir()
        (tok_dir / "pruned_config.json").write_text(
            json.dumps({"vocab_size": 40034, "sep_token_id": 40033})
        )

        shard_dir = tmp / "shards"
        shard_dir.mkdir()
        shard_paths = {s: shard_dir / f"{s}.tar" for s in SPLIT_NAMES}
        for p in shard_paths.values():
            p.touch()

        ckpt_dir = tmp / "checkpoints"
        ckpt_dir.mkdir()
        ckpt_paths = [ckpt_dir / "step0004680-stage-handoff.pt", ckpt_dir / "step0009360.pt"]
        for p in ckpt_paths:
            p.touch()

        fake_cfg = Config(
            data=DataConfig(
                shards="data/*.tar", tokenizer=tok_dir,
                eval=EvalShards(
                    dev_clean  = shard_paths["dev-clean"],
                    dev_other  = shard_paths["dev-other"],
                    test_clean = shard_paths["test-clean"],
                    test_other = shard_paths["test-other"],
                ),
            ),
            logging=LoggingConfig(run_name="selftest"),
            metrics=MetricsConfig(wer=WerConfig(max_batches=2, sample_transcriptions=2)),
            stages=cfg_stages,
        )

        # Two utterances per (split, format) so the sampled cap (2) and the full
        # dump (4 rows per split) are distinguishable.
        def _fake_evaluate(   # mirrors evaluate_all_splits' signature
            encoder, adapter, llama, eval_loaders, tokenizer, terminator_id, device,
            max_batches=None, n_samples=20, sample_seed=0, formats=None,
            progress_interval=None, chat=None, return_all_transcriptions=False,
        ):
            # The fake config leaves input_convention at its "flat" default, so
            # the tool must pass chat=None; a chat run is covered end-to-end by
            # the stub smoke test instead.
            assert chat is None, "flat config must not build a ChatTemplate"
            split = next(iter(eval_loaders))
            requested_splits.append(split)
            wer_dict, rows = {}, []
            for fmt in ("unformatted", "formatted"):
                wer_dict[f"{split}/{fmt}"] = 0.5
                for i in range(2):
                    rows.append({
                        "key": f"{split}-utt{i}", "split": split, "type": fmt,
                        "reference": "the quick brown fox",
                        "hypothesis": "the quick brown fox" if i == 0 else "the quick",
                    })
            sampled = rows[:n_samples]
            return (wer_dict, sampled, rows) if return_all_transcriptions else (wer_dict, sampled)

        globals().update(
            load_config                 = lambda *a, **k: fake_cfg,
            build_models                = lambda *a, **k: (None, None, None, []),
            PrunedTokenizer             = lambda *a, **k: None,
            build_sorted_eval_dataloader= lambda *a, **k: [],
            read_checkpoint             = lambda p: {
                "step": 4680, "epoch": 0, "step_in_stage": 4680,
                "stage_index": 0, "kind": "handoff",
            },
            apply_weights               = lambda ckpt, **k: ["adapter", "audio_adapters"],
            evaluate_all_splits         = _fake_evaluate,
        )
        # eval() / train() are called on the mocked modules; a bare object won't do.
        class _NoOpModule:
            def eval(self) -> None: ...
            def train(self) -> None: ...
        globals()["build_models"] = lambda *a, **k: (
            _NoOpModule(), _NoOpModule(), _NoOpModule(), []
        )

        try:
            out_csv = tmp / "results" / "wer.csv"
            main(argv=[
                "--config", "unused.yaml",
                "--checkpoints", *[str(p) for p in ckpt_paths],
                "--splits", "dev-clean", "dev-other",
                "--output", str(out_csv),
                "--device", "cpu",
                "--progress-interval", "0",
            ])

            # --splits: exactly the two dev loaders, once per checkpoint.
            assert requested_splits == ["dev-clean", "dev-other"] * 2, requested_splits
            print("  [OK] --splits evaluates exactly the requested loaders")

            # Summary CSV: metadata columns present on every row.
            with out_csv.open() as f:
                csv_rows = list(csv.DictReader(f))
            assert len(csv_rows) == 2 * 2 * 2, csv_rows      # 2 ckpts × 2 splits × 2 formats
            for r in csv_rows:
                for col in ("stage_index", "stage_name", "epoch_in_stage",
                            "step_in_stage", "kind"):
                    assert r[col] != "", f"{col} missing on {r}"
                assert r["stage_name"] == "bridge_only", r
                assert r["kind"] == "handoff", r
            assert {r["split"] for r in csv_rows} == {"dev-clean", "dev-other"}, csv_rows
            print("  [OK] summary CSV carries stage/epoch metadata columns")

            # JSONL: full dump, one row per sample per format, with per-sample WER.
            jsonl_paths = sorted((tmp / "results").glob("*.jsonl"))
            assert [p.name for p in jsonl_paths] == [
                "0004680_dev-clean.jsonl", "0004680_dev-other.jsonl",
                "0009360_dev-clean.jsonl", "0009360_dev-other.jsonl",
            ], jsonl_paths
            rows = [json.loads(line) for line in jsonl_paths[0].read_text().splitlines()]
            assert len(rows) == 4, rows          # 2 samples × 2 formats, NOT the n_samples cap
            for r in rows:
                for field_name in ("checkpoint", "step", "split", "type",
                                   "key", "reference", "hypothesis", "wer"):
                    assert field_name in r, f"{field_name} missing from {r}"
            assert rows[0]["wer"] == 0.0 and rows[1]["wer"] == 0.5, rows[:2]
            print("  [OK] JSONL holds every transcription with a per-sample WER")

            # Field compatibility: count_degeneracies.py must run unchanged.
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import count_degeneracies as _cd

            _cd.main(argv=["--inputs", str(jsonl_paths[0]),
                           "--output", str(tmp / "results" / "degeneracies.csv")])
            groups = _cd.aggregate(_cd.read_rows([jsonl_paths[0]]))
            assert len(groups) == 2, groups                   # unformatted + formatted
            assert all(g.n == 2 for g in groups), [g.n for g in groups]
            unfmt = next(g for g in groups if g.fmt == "unformatted")
            assert unfmt.counts["truncation"] == 1, unfmt.counts
            print("  [OK] count_degeneracies.py consumes the new JSONL unchanged")

            # ── --eval-tar + --dataset: the out-of-distribution path ─────────
            ood_tar = tmp / "tedlium3-test-le41.tar"
            ood_tar.touch()
            requested_splits.clear()
            ood_csv = tmp / "ood" / "wer.csv"
            main(argv=[
                "--config", "unused.yaml",
                "--checkpoints", str(ckpt_paths[0]),
                "--eval-tar", f"tedlium3-test={ood_tar}",
                "--dataset", "tedlium3-test",
                "--formats", "unformatted",
                "--output", str(ood_csv),
                "--device", "cpu",
                "--progress-interval", "0",
            ])
            assert requested_splits == ["tedlium3-test"], requested_splits
            ood_jsonl = tmp / "ood" / "0004680_tedlium3-test.jsonl"
            assert ood_jsonl.exists(), sorted((tmp / "ood").iterdir())
            ood_rows = [json.loads(l) for l in ood_jsonl.read_text().splitlines()]
            assert all(r["dataset"] == "tedlium3-test" for r in ood_rows), ood_rows[0]
            assert all(r["split"] == "tedlium3-test" for r in ood_rows), ood_rows[0]
            with ood_csv.open() as f:
                ood_summary = list(csv.DictReader(f))
            assert all(r["dataset"] == "tedlium3-test" for r in ood_summary), ood_summary
            print("  [OK] --eval-tar names the loader, the JSONL and the dataset column")

            # --eval-tar and --splits must not be combined: one replaces the other.
            try:
                main(argv=[
                    "--config", "unused.yaml",
                    "--checkpoints", str(ckpt_paths[0]),
                    "--eval-tar", f"x={ood_tar}", "--splits", "dev-clean",
                    "--output", str(tmp / "ood" / "x.csv"),
                    "--device", "cpu", "--progress-interval", "0",
                ])
            except SystemExit as exc:
                assert exc.code == 1, exc
            else:
                raise AssertionError("--eval-tar with --splits must exit")
            print("  [OK] --eval-tar refuses to be combined with --splits")
        finally:
            globals().update(globals_backup)

    print("\nPASSED")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main(argv=sys.argv[1:])

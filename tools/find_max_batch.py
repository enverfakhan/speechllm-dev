"""Binary-search the largest batch size that fits in VRAM, for one config stage.

The probe is CONFIG-DRIVEN: it builds the exact stack the run will build, through
the same two entry points training.py uses —

    build.build_models(cfg)      architecture: bridge_type, audio_adapter_r/type,
                                 audio_adapter_zero_writer_layers, gradient
                                 checkpointing on Llama AND on the Whisper encoder
    stages.Stage.setup(...)      trainability + optimizer: per-stage `trainable`
                                 set (including the name-selected "audio_adapters"
                                 subset of llama), per-group LRs, wd=0 for the
                                 gated adapters

so the answer is for the model the config actually trains.  Point it at a stage
whose architecture is untested (e.g. the first stage that unfreezes the encoder
while in-layer adapters are in the optimizer) and the number is trustworthy.

Training mode (--stage N / --stage-name NAME):
  Mirrors the training step: fp16 autocast, accum_steps micro-batches of
  forward+backward, gradient clipping when optim.grad_clip.enabled, one 8-bit
  AdamW step.  Sequence shape is the worst case the corpus can produce — a full
  30 s of audio (375 adapter tokens) for every sample in the batch.

  bitsandbytes AdamW8bit defers GPU state allocation until the 4th
  optimizer.step() call, so a surprise allocation mid-search would make the
  binary search unreliable.  The script runs 4 warm-up steps at batch=1 first.

Eval mode (--eval):
  Forward-only under torch.no_grad(), models in .eval(), no optimizer and no
  gradient checkpointing.  Use this for metrics.eval_batch_size / WER runs.

Usage:
    python tools/find_max_batch.py --config configs/staged_full_stack.yaml --stage 4
    python tools/find_max_batch.py --config configs/staged_full_stack.yaml --stage-name full_stack
    python tools/find_max_batch.py --config configs/staged_full_stack.yaml --stage 4 --accum 1 2 3
    python tools/find_max_batch.py --config configs/staged_full_stack.yaml --eval
"""


from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build import build_models
from data import INSTRUCTION_VARIANTS
from model.sequence import prepare_input
from stages import Stage, StageContext
from utils.config import Config, load_config


# ── Constants ─────────────────────────────────────────────────────────────────
# Worst-case sequence: the corpus caps utterances at 30 s, so 3000 mel frames.
_MEL_FRAMES  = 3000
_INST_TOKS   = 50
_TRANS_TOKS  = 50
_WARMUP_MEL  = 800    # short clip for the bnb warm-up steps (batch=1)
_TOTAL_VRAM_GB = (
    torch.cuda.get_device_properties(0).total_memory / 1e9
    if torch.cuda.is_available() else 80.0
)


def _audio_tokens(mel_frames: int) -> int:
    """Adapter-token count for a mel of mel_frames (Decision 007)."""
    return (mel_frames // 2 + 3) // 4


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Binary-search max batch size for one stage of a run config."
    )
    p.add_argument("--config", type=Path, required=True,
                   help="Run config YAML (merged over configs/base.yaml).")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stage", type=int,
                      help="1-based index into the config's `stages:` list.")
    mode.add_argument("--stage-name", dest="stage_name",
                      help="Stage `name:` from the config's `stages:` list.")
    mode.add_argument("--eval", action="store_true",
                      help="Eval mode: no optimizer, no gradients, no grad checkpointing.")
    p.add_argument("--accum", type=int, nargs="+", default=None,
                   help="accum_steps values to probe (default: the stage's own accum_steps).")
    p.add_argument("--min_batch", type=int, default=1)
    p.add_argument("--max_batch", type=int, default=128)
    return p.parse_args()


# ── Config plumbing ───────────────────────────────────────────────────────────

def _vocab_meta(cfg: Config) -> tuple[int, int]:
    """Read (vocab_size, sep_token_id) from the pruned tokenizer — ground truth."""
    with (cfg.data.tokenizer / "pruned_config.json").open() as f:
        pruned = json.load(f)
    return int(pruned["vocab_size"]), int(pruned["sep_token_id"])


def _stage_context(cfg: Config, sep_token_id: int) -> StageContext:
    """Minimal StageContext for Stage.setup().

    Only betas/weight_decay reach the optimizer; the data fields exist because
    StageContext owns them for make_loader(), which this tool never calls — hence
    the empty shard list (the probe needs no shards on disk, only weights and the
    pruned tokenizer).
    """
    return StageContext(
        shards               = [],
        tokenizer_path       = cfg.data.tokenizer,
        sep_token_id         = sep_token_id,
        num_workers          = 0,
        seed                 = cfg.seed,
        instruction_pairs    = [
            (INSTRUCTION_VARIANTS[0], "unformatted.txt"),
            (INSTRUCTION_VARIANTS[1], "formatted.txt"),
        ],
        run_instruction_mode = cfg.run.instruction_mode,
        betas                = cfg.optim.betas,
        weight_decay         = cfg.optim.weight_decay,
    )


def _resolve_stage_index(cfg: Config, args: argparse.Namespace) -> int:
    """Map --stage / --stage-name onto a 0-based index into cfg.stages."""
    names = [s.name for s in cfg.stages]
    if args.stage_name is not None:
        if args.stage_name not in names:
            raise SystemExit(
                f"--stage-name {args.stage_name!r} not in {args.config}: {names}"
            )
        return names.index(args.stage_name)

    if not 1 <= args.stage <= len(cfg.stages):
        raise SystemExit(
            f"--stage {args.stage} out of range: {args.config} has {len(cfg.stages)} "
            f"stages (1..{len(cfg.stages)}) — {names}"
        )
    return args.stage - 1


# ── Synthetic batch ───────────────────────────────────────────────────────────

def _make_batch(
    batch_size: int,
    mel_frames: int,
    vocab_size: int,
    device:     torch.device,
) -> tuple[torch.Tensor, ...]:
    """Worst-case batch in the dataloader's dtypes (mel is float32 there)."""
    a_lens = _audio_tokens(mel_frames)
    return (
        torch.randn(batch_size, 80, mel_frames, device=device, dtype=torch.float32),
        torch.full((batch_size,), a_lens, dtype=torch.long, device=device),
        torch.randint(0, vocab_size, (batch_size, _INST_TOKS), device=device),
        torch.full((batch_size,), _INST_TOKS, dtype=torch.long, device=device),
        torch.randint(0, vocab_size, (batch_size, _TRANS_TOKS), device=device),
        torch.full((batch_size,), _TRANS_TOKS, dtype=torch.long, device=device),
    )


def _is_oom(exc: BaseException) -> bool:
    """OOM surfaces as torch.cuda.OutOfMemoryError, or a plain RuntimeError."""
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


# ── Probes ────────────────────────────────────────────────────────────────────

def _try_batch(
    cfg:          Config,
    encoder:      torch.nn.Module,
    adapter:      torch.nn.Module,
    llama:        torch.nn.Module,
    optimizer:    torch.optim.Optimizer,
    batch_size:   int,
    accum_steps:  int,
    vocab_size:   int,
    sep_token_id: int,
    mel_frames:   int = _MEL_FRAMES,
) -> tuple[bool, float, float]:
    """One training step: accum_steps micro-batches + clip + optimizer step.

    GradScaler is deliberately absent.  Its own footprint is a handful of scalars,
    but a fresh scaler starts at scale 2**16 and skips its first optimizer steps on
    overflow — which with random inputs would skip exactly the allocation this
    probe exists to measure.  Everything else mirrors training.py's step.

    Returns (fits, peak_allocated_gb, peak_reserved_gb).
    """
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    try:
        optimizer.zero_grad(set_to_none=True)

        for _ in range(accum_steps):
            (mel, a_lens, inst_ids, inst_lens,
             tr_ids, tr_lens) = _make_batch(batch_size, mel_frames, vocab_size, device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                enc_out = encoder(mel)
                adp_out = adapter(enc_out)
                inputs, labels = prepare_input(
                    adp_out, a_lens, inst_ids, inst_lens, tr_ids, tr_lens,
                    llama.embed_tokens, sep_token_id,
                )
                _, loss = llama(inputs, labels, audio_lengths=a_lens)

            (loss / accum_steps).backward()

            del mel, a_lens, inst_ids, inst_lens, tr_ids, tr_lens
            del enc_out, adp_out, inputs, labels, loss

        if cfg.optim.grad_clip.enabled:
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]],
                cfg.optim.grad_clip.max_norm,
            )
        optimizer.step()

        return (True,
                torch.cuda.max_memory_allocated(device) / 1e9,
                torch.cuda.max_memory_reserved(device) / 1e9)

    except Exception as exc:                      # re-raised unless it is an OOM
        if not _is_oom(exc):
            raise
        return False, 0.0, 0.0
    finally:
        # Drop the half-accumulated graph/grads so the next probe starts clean.
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()


def _try_batch_eval(
    encoder:      torch.nn.Module,
    adapter:      torch.nn.Module,
    llama:        torch.nn.Module,
    batch_size:   int,
    vocab_size:   int,
    sep_token_id: int,
) -> tuple[bool, float, float]:
    """Forward-only pass under torch.no_grad().  Returns (fits, alloc_gb, reserved_gb)."""
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    try:
        (mel, a_lens, inst_ids, inst_lens,
         tr_ids, tr_lens) = _make_batch(batch_size, _MEL_FRAMES, vocab_size, device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            enc_out = encoder(mel)
            adp_out = adapter(enc_out)
            inputs, labels = prepare_input(
                adp_out, a_lens, inst_ids, inst_lens, tr_ids, tr_lens,
                llama.embed_tokens, sep_token_id,
            )
            llama(inputs, labels, audio_lengths=a_lens)

        return (True,
                torch.cuda.max_memory_allocated(device) / 1e9,
                torch.cuda.max_memory_reserved(device) / 1e9)

    except Exception as exc:                      # re-raised unless it is an OOM
        if not _is_oom(exc):
            raise
        return False, 0.0, 0.0
    finally:
        gc.collect()
        torch.cuda.empty_cache()


# ── bnb warm-up ───────────────────────────────────────────────────────────────

def _warmup_optimizer(
    cfg:          Config,
    encoder:      torch.nn.Module,
    adapter:      torch.nn.Module,
    llama:        torch.nn.Module,
    optimizer:    torch.optim.Optimizer,
    vocab_size:   int,
    sep_token_id: int,
    n_steps:      int = 4,
) -> None:
    """Run n_steps real optimizer steps at batch=1 to materialise bnb 8-bit state.

    bitsandbytes AdamW8bit defers GPU allocation of exp_avg/exp_avg_sq until the
    4th optimizer.step(); pre-materialising it means the binary search always
    measures against a stable baseline instead of tripping over a late allocation.
    """
    for _ in range(n_steps):
        ok, _, _ = _try_batch(
            cfg, encoder, adapter, llama, optimizer,
            batch_size=1, accum_steps=1,
            vocab_size=vocab_size, sep_token_id=sep_token_id,
            mel_frames=_WARMUP_MEL,
        )
        if not ok:
            raise SystemExit(
                "OOM during optimizer warm-up at batch=1 — this stage does not fit "
                "on this GPU at any batch size."
            )

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(torch.device("cuda"))


def _optimizer_state_gb(optimizer: torch.optim.Optimizer) -> float:
    """Bytes held by optimizer state tensors (bnb 8-bit moments + qmaps)."""
    total = 0
    for state in optimizer.state.values():
        for v in state.values():
            if torch.is_tensor(v):
                total += v.numel() * v.element_size()
    return total / 1e9


# ── Binary search ─────────────────────────────────────────────────────────────

def _search(probe, min_batch: int, max_batch: int) -> tuple[int, float, float]:
    """Binary search the largest batch `probe(bs)` accepts.

    probe(batch_size) -> (fits, alloc_gb, reserved_gb).
    Returns (max_batch_size, alloc_gb_at_max, reserved_gb_at_max); 0 when even
    batch=1 does not fit.
    """
    ok, alloc, reserved = probe(1)
    if not ok:
        print(f"  batch={1:4d}  ✗ OOM")
        return 0, 0.0, 0.0
    print(f"  batch={1:4d}  ✓ OK   peak={alloc:.1f}GB  reserved={reserved:.1f}GB  "
          f"util={alloc / _TOTAL_VRAM_GB * 100:.0f}%")
    best, best_alloc, best_reserved = 1, alloc, reserved

    lo, hi = max(2, min_batch), max_batch
    while lo <= hi:
        mid = (lo + hi) // 2
        ok, alloc, reserved = probe(mid)
        if ok:
            print(f"  batch={mid:4d}  ✓ OK   peak={alloc:.1f}GB  reserved={reserved:.1f}GB  "
                  f"util={alloc / _TOTAL_VRAM_GB * 100:.0f}%")
            best, best_alloc, best_reserved = mid, alloc, reserved
            lo = mid + 1
        else:
            print(f"  batch={mid:4d}  ✗ OOM")
            hi = mid - 1

    return best, best_alloc, best_reserved


def _suggested(max_bs: int) -> int:
    """~90 % of the measured ceiling, rounded down to a multiple of 4.

    The measured max is the fragmentation-fragile edge: real batches vary in
    length, and the allocator is not as tidy after hours of training as it is in
    a fresh probe.  Project practice is to run below it (Decision 009: measured
    72 → ran 64).
    """
    return max(1, (int(max_bs * 0.9) // 4) * 4) if max_bs >= 4 else max_bs


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(args.config)
    vocab_size, sep_token_id = _vocab_meta(cfg)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {_TOTAL_VRAM_GB:.1f} GB")
    print(f"Config: {args.config}")
    print(f"Vocab: {vocab_size} tokens  SEP id: {sep_token_id}")
    print(f"Sequence: {_MEL_FRAMES} mel frames → {_audio_tokens(_MEL_FRAMES)} audio tokens "
          f"+ {_INST_TOKS} instruction + {_TRANS_TOKS} transcript")

    device = torch.device("cuda")

    # ── Eval path ─────────────────────────────────────────────────────────────
    if args.eval:
        print("Mode: eval (no optimizer, no gradients, no grad checkpointing)")
        print()
        # apply_init_from=False: warm-start weights change no allocation, only load time.
        encoder, adapter, llama, _ = build_models(
            cfg, device, train=False, apply_init_from=False
        )
        print()
        best, alloc, reserved = _search(
            lambda bs: _try_batch_eval(
                encoder, adapter, llama, bs, vocab_size, sep_token_id
            ),
            args.min_batch, args.max_batch,
        )
        print()
        print("═" * 62)
        print("SUMMARY  eval (forward-only, fp16 autocast)")
        print("═" * 62)
        if best > 0:
            print(f"  max batch size : {best}   (suggested: {_suggested(best)})")
            print(f"  peak VRAM      : {alloc:.1f} GB allocated / {reserved:.1f} GB reserved "
                  f"({alloc / _TOTAL_VRAM_GB * 100:.0f}% of {_TOTAL_VRAM_GB:.1f} GB)")
            print(f"  config value   : metrics.eval_batch_size = {cfg.metrics.eval_batch_size}")
        else:
            print("  OOM at batch=1 — cannot fit even a single sample.")
        return

    # ── Training path ─────────────────────────────────────────────────────────
    stage_idx = _resolve_stage_index(cfg, args)
    stage_cfg = cfg.stages[stage_idx]
    accums    = args.accum if args.accum is not None else [stage_cfg.accum_steps]

    print(f"Stage {stage_idx + 1}/{len(cfg.stages)}: {stage_cfg.name}  "
          f"trainable={list(stage_cfg.trainable)}")
    print(f"Config batch_size={stage_cfg.batch_size}  accum_steps={stage_cfg.accum_steps}")
    print(f"Probing accum_steps: {accums}")
    print()

    encoder, adapter, llama, _ = build_models(
        cfg, device, train=True, apply_init_from=False
    )
    stage = Stage(stage_cfg, _stage_context(cfg, sep_token_id))
    optimizer, _ = stage.setup(encoder, adapter, llama)   # prints the stage header

    print(f"Pre-warming optimizer (4 steps at batch=1, mel_t={_WARMUP_MEL}) …")
    _warmup_optimizer(cfg, encoder, adapter, llama, optimizer, vocab_size, sep_token_id)
    n_total = sum(len(g["params"]) for g in optimizer.param_groups)
    print(f"  bnb state = {_optimizer_state_gb(optimizer):.2f} GB  "
          f"entries = {len(optimizer.state)}/{n_total}")
    print()

    results: list[tuple[int, int, float, float]] = []   # (accum, max_bs, alloc, reserved)

    for accum in accums:
        print(f"─── accum_steps={accum}  (eff. batch = micro_batch × {accum}) ───")
        best, alloc, reserved = _search(
            lambda bs, _a=accum: _try_batch(
                cfg, encoder, adapter, llama, optimizer, bs, _a,
                vocab_size, sep_token_id,
            ),
            args.min_batch, args.max_batch,
        )
        results.append((accum, best, alloc, reserved))
        print(f"  → max micro-batch={best}  eff. batch={best * accum}  "
              f"(suggested micro-batch {_suggested(best)})\n")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("═" * 72)
    print(f"SUMMARY  {args.config}  stage {stage_idx + 1} ({stage_cfg.name})  "
          f"trainable={list(stage_cfg.trainable)}")
    print("═" * 72)
    print(f"  {'accum':>6}  {'micro-bs':>9}  {'suggest':>8}  {'eff-bs':>7}  "
          f"{'peak':>9}  {'reserved':>10}  {'util':>6}")
    print("  " + "─" * 66)
    for accum, max_bs, alloc, reserved in results:
        if max_bs > 0:
            util = alloc / _TOTAL_VRAM_GB * 100
            print(f"  {accum:6d}  {max_bs:9d}  {_suggested(max_bs):8d}  {max_bs * accum:7d}  "
                  f"{alloc:6.1f} GB  {reserved:7.1f} GB  {util:5.0f}%")
        else:
            print(f"  {accum:6d}  {'OOM':>9}  {'-':>8}  {'-':>7}  {'-':>9}  {'-':>10}  {'-':>6}")
    print()
    print("  'suggest' = 90 % of the measured ceiling, rounded to a multiple of 4 —")
    print("  real batches vary in length and the allocator fragments over a long run.")


if __name__ == "__main__":
    main()

"""Probe VRAM under gradient accumulation for stage-2 training.

Replicates train.py exactly (fp32 params, fp16 autocast + GradScaler) to
reproduce and diagnose the OOM with accum_steps > 1.

Diagnosis
─────────
The crash at step 540 (not step 1) is classic CUDA allocator fragmentation.
Gradient tensors stay live across micro-batches, raising the steady-state
pressure. After ~500 steps, the allocator's free cache (146 MB at crash time)
is split into fragments < 58 MB, so it cannot satisfy the next allocation.

Two fixes to try, in order of preference:

  1. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   ← try this first
     Lets the allocator grow segments on demand instead of needing a
     contiguous 58 MB hole in a fixed pool.

  2. --empty_cache   (flag to this script, matches --empty_cache in train.py
     if you add it there)
     Calls torch.cuda.empty_cache() between micro-batches to release the
     fragmented cache back to CUDA.  Trades ~5% throughput for stability.

Usage
─────
  # Reproduce the OOM (matches current train.py exactly):
  python scripts/probe_accum.py --accum 2

  # Test fix 1 — expandable segments (no code change needed):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
      python scripts/probe_accum.py --accum 2

  # Test fix 2 — empty_cache between micro-batches:
  python scripts/probe_accum.py --accum 2 --empty_cache

  # Sweep accum 1..4:
  python scripts/probe_accum.py --max_accum 4

  # Long run to expose fragmentation build-up (matches ~500-step failure window):
  python scripts/probe_accum.py --accum 2 --n_steps 20
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.adapter import AudioAdapter, prepare_input
from model.llama import Llama, LlamaConfig
from model.whisper_encoder import WhisperEncoder

# ── Constants matching the real training setup ────────────────────────────────
_VOCAB_SIZE = 40148
_AUDIO_TOKS = 375    # adapter output for 30 s audio after pool-4
_INST_TOKS  = 14     # "Transcribe the following audio without formatting." tokenised
_TRANS_TOKS = 50     # representative transcript length
_SEP_ID     = _VOCAB_SIZE - 1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VRAM probe: gradient accumulation (stage 2, fp32 model).")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--accum",     type=int, default=None,
                      help="Run a single accum_steps value.")
    mode.add_argument("--max_accum", type=int, default=4,
                      help="Sweep accum_steps 1 .. max_accum (default 4). Ignored if --accum is set.")
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--n_steps",      type=int, default=5,
                   help="Optimizer steps to run per accum value (default 5). "
                        "Use 20+ to expose fragmentation build-up.")
    p.add_argument("--whisper_ckpt", type=str, default="weights/whisper_small.pt")
    p.add_argument("--llama_ckpt",   type=str,
                   default="weights/Llama3.1-8B/Llama3.1-8B/Llama3.1-8B/")
    p.add_argument("--empty_cache",  action="store_true",
                   help="Call torch.cuda.empty_cache() between micro-batches "
                        "(tests fix 2 — trades ~5%% throughput for stability).")
    return p.parse_args()


def _mem(device: torch.device) -> str:
    alloc = torch.cuda.memory_allocated(device) / 1e9
    resv  = torch.cuda.memory_reserved(device)  / 1e9
    return f"alloc={alloc:.2f}GB  resv={resv:.2f}GB"


def _build_models(
    whisper_ckpt: str,
    llama_ckpt: str,
    device: torch.device,
) -> tuple[WhisperEncoder, AudioAdapter, Llama, torch.optim.Optimizer, torch.amp.GradScaler]:
    """Build fp32 models + fp16 GradScaler, exactly as train.py does."""
    print("Loading model weights …")
    cfg = LlamaConfig(vocab_size=_VOCAB_SIZE)

    encoder = WhisperEncoder()
    encoder.load_openai_weights(whisper_ckpt)

    adapter = AudioAdapter(llama_dim=4096)

    llama = Llama(cfg)
    llama.load_meta_weights(llama_ckpt)

    # fp32 on device — no dtype cast, matching current train.py
    encoder = encoder.to(device)
    adapter = adapter.to(device)
    llama   = llama.to(device)

    llama.enable_gradient_checkpointing()
    encoder.train(); adapter.train(); llama.train()

    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(
        [
            {"params": list(encoder.parameters()), "lr": 1e-6},
            {"params": list(adapter.parameters()), "lr": 5e-5},
            {"params": list(llama.parameters()),   "lr": 1.5e-5},
        ],
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    scaler = torch.amp.GradScaler("cuda")

    n_total = (sum(p.numel() for p in encoder.parameters()) +
               sum(p.numel() for p in adapter.parameters()) +
               sum(p.numel() for p in llama.parameters()))
    print(f"  fp32 params: {n_total * 4 / 1e9:.1f} GB  "
          f"fp32 grads (when alive): {n_total * 4 / 1e9:.1f} GB  "
          f"8-bit Adam states: {n_total * 2 / 1e9:.1f} GB")
    print(f"  baseline (params+Adam, no grads): {n_total * 6 / 1e9:.1f} GB")
    print(f"  baseline (params+grads+Adam):     {n_total * 10 / 1e9:.1f} GB")
    return encoder, adapter, llama, optimizer, scaler


def _make_batch(batch_size: int, device: torch.device) -> tuple:
    """Worst-case batch: 30 s audio (max mel length)."""
    mel           = torch.randn(batch_size, 80, 3000, device=device)
    audio_lengths = torch.full((batch_size,), _AUDIO_TOKS, dtype=torch.long, device=device)
    # Fixed IDs — values don't affect memory, only shape does.
    inst_ids  = torch.full((batch_size, _INST_TOKS),  4, dtype=torch.long, device=device)
    inst_lens = torch.full((batch_size,), _INST_TOKS,    dtype=torch.long, device=device)
    trans_ids = torch.full((batch_size, _TRANS_TOKS), 4, dtype=torch.long, device=device)
    trans_lens = torch.full((batch_size,), _TRANS_TOKS,  dtype=torch.long, device=device)
    return mel, audio_lengths, inst_ids, inst_lens, trans_ids, trans_lens


def _run_accum(
    encoder: WhisperEncoder,
    adapter: AudioAdapter,
    llama: Llama,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    batch_size: int,
    accum_steps: int,
    n_steps: int,
    device: torch.device,
    empty_cache: bool,
    verbose_step: int = 1,    # print per-micro-batch memory only for this step
) -> tuple[bool, int]:
    """Run n_steps optimizer steps, each consisting of accum_steps micro-batches.

    Returns (success, step_that_failed).  step_that_failed is -1 on success.
    """
    optimizer.zero_grad()
    torch.cuda.reset_peak_memory_stats(device)

    for step in range(1, n_steps + 1):
        for k in range(accum_steps):
            try:
                mel, audio_lengths, inst_ids, inst_lens, trans_ids, trans_lens = (
                    _make_batch(batch_size, device)
                )

                if step == verbose_step:
                    alloc_pre = torch.cuda.memory_allocated(device) / 1e9
                    print(f"  step {step}  micro {k+1}/{accum_steps}  "
                          f"[before fwd]  {_mem(device)}")

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    enc_out     = encoder(mel)
                    adapter_out = adapter(enc_out)
                    inputs, labels = prepare_input(
                        adapter_out, audio_lengths,
                        inst_ids, inst_lens,
                        trans_ids, trans_lens,
                        llama.embed_tokens,
                        sep_token_id=_SEP_ID,
                    )
                    _, loss = llama(inputs, labels)

                if step == verbose_step:
                    print(f"  step {step}  micro {k+1}/{accum_steps}  "
                          f"[after  fwd]  {_mem(device)}")

                scaler.scale(loss / accum_steps).backward()

                if step == verbose_step:
                    print(f"  step {step}  micro {k+1}/{accum_steps}  "
                          f"[after  bwd]  {_mem(device)}")

                if empty_cache:
                    torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                return False, step

        # Optimizer step
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for g in optimizer.param_groups for p in g["params"]], 1.0
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if empty_cache:
            torch.cuda.empty_cache()

        peak = torch.cuda.max_memory_allocated(device) / 1e9
        if step == verbose_step or step == n_steps:
            print(f"  step {step}  [after opt.step+zero_grad]  "
                  f"{_mem(device)}  peak_so_far={peak:.2f}GB")
        torch.cuda.reset_peak_memory_stats(device)

    return True, -1


def _reset_models(encoder, adapter, llama, optimizer, scaler, device):
    """Zero grads and reset peak stats for a clean re-run."""
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)


def main() -> None:
    args   = _parse_args()
    device = torch.device("cuda")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.", file=sys.stderr)
        sys.exit(1)

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    exp_seg  = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    exp_on   = "expandable_segments:True" in exp_seg

    print(f"GPU: {torch.cuda.get_device_name(0)}  ({total_gb:.1f} GB)")
    print(f"PYTORCH_CUDA_ALLOC_CONF: {exp_seg!r}  "
          f"({'expandable_segments ON' if exp_on else 'expandable_segments OFF — default'})")
    print(f"batch_size={args.batch_size}  n_steps={args.n_steps}  "
          f"empty_cache={args.empty_cache}")
    print()

    encoder, adapter, llama, optimizer, scaler = _build_models(
        args.whisper_ckpt, args.llama_ckpt, device,
    )
    print(f"After load:  {_mem(device)}")
    print()

    accum_values = [args.accum] if args.accum is not None else list(range(1, args.max_accum + 1))

    results: list[tuple[int, bool, int]] = []   # (accum, ok, fail_step)
    for accum in accum_values:
        _reset_models(encoder, adapter, llama, optimizer, scaler, device)
        print(f"─── accum_steps={accum}  effective_batch={args.batch_size * accum} ───")
        ok, fail_step = _run_accum(
            encoder, adapter, llama, optimizer, scaler,
            batch_size=args.batch_size,
            accum_steps=accum,
            n_steps=args.n_steps,
            device=device,
            empty_cache=args.empty_cache,
            verbose_step=1,
        )
        status = "OK" if ok else f"OOM at step {fail_step}"
        print(f"  → {status}")
        print()
        results.append((accum, ok, fail_step))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("═" * 50)
    print("SUMMARY")
    print("═" * 50)
    print(f"  {'accum':>6}  {'eff_bs':>7}  result")
    print("  " + "─" * 30)
    any_oom = False
    for accum, ok, fail_step in results:
        eff = args.batch_size * accum
        status = f"OK  ({args.n_steps}/{args.n_steps} steps)" if ok else f"OOM at step {fail_step}"
        print(f"  {accum:6d}  {eff:7d}  {status}")
        if not ok:
            any_oom = True
    print()

    if any_oom:
        print("Recommended fixes (try in order):")
        print()
        if not exp_on:
            print("  1. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
            print("     Lets the allocator grow segments on demand instead of")
            print("     needing a contiguous block in a fragmented fixed pool.")
            print("     Add to ~/.bashrc or prepend to the train.py invocation.")
            print()
        print("  2. --empty_cache flag (add to this script / train.py)")
        print("     Calls torch.cuda.empty_cache() between micro-batches.")
        print("     Costs ~5% throughput but releases fragmented cache blocks.")
        print()
        print("  3. If OOM only appears after many steps (fragmentation build-up),")
        print("     fix 1 is almost certainly sufficient on its own.")
    else:
        if exp_on:
            print("All accum values fit WITH expandable_segments:True.")
            print("Add PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to batch_sweep.sh.")
        elif args.empty_cache:
            print("All accum values fit WITH --empty_cache.")
        else:
            print("All accum values fit with default allocator settings.")


if __name__ == "__main__":
    main()

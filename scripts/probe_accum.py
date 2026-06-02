"""Probe VRAM under gradient accumulation for stage-2 training.

Why accum_steps > 1 OOMs on step ~5 even with expandable_segments + empty_cache
────────────────────────────────────────────────────────────────────────────────
From the step-1 probe:
  after fwd (micro 2..4):  alloc = 66.41 GB
  peak_so_far:             80.99 GB  (backward recomputation adds ~14.6 GB peak)
  GPU:                     85.10 GB
  Headroom at step 1:       4.11 GB

Both expandable_segments and empty_cache address fragmentation. The OOM at step 5
is NOT fragmentation — it is real memory growth across steps, almost certainly
bitsandbytes 8-bit Adam initialising its optimizer state tensors (≈14.5 GB for
7.25 B params) lazily over the first several steps. Each new param-state pair
allocated eats into the 4.11 GB headroom until step 5 pushes the total over 85 GB.

torch.cuda.memory_allocated() does NOT include bnb's internal cudaMalloc
tensors. This probe reports cuda_used = total − cuda_free (which DOES include
bnb), so you can see the step-by-step growth of the untracked component.

Fix
───
Cast models to bfloat16 (as find_max_batch.py does) and use bfloat16 autocast.
This halves both param memory (29 → 14.5 GB) and grad memory (29 → 14.5 GB),
dropping the step-1 peak to ~50 GB and leaving 35 GB of headroom for bnb states
and activation peaks.

Usage
─────
  # Reproduce the problem (fp32, matching current train.py):
  python scripts/probe_accum.py --accum 4 --n_steps 8

  # Test the bf16 fix (matches find_max_batch.py):
  python scripts/probe_accum.py --accum 4 --n_steps 8 --bf16

  # Sweep accum 1..4 in bf16:
  python scripts/probe_accum.py --max_accum 4 --n_steps 5 --bf16

  # Baseline check (accum=1 should always pass):
  python scripts/probe_accum.py --accum 1 --n_steps 5
  python scripts/probe_accum.py --accum 1 --n_steps 5 --bf16
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

_VOCAB_SIZE = 40148
_AUDIO_TOKS = 375    # adapter output tokens for 30 s audio
_INST_TOKS  = 14     # "Transcribe the following audio without formatting."
_TRANS_TOKS = 50
_SEP_ID     = _VOCAB_SIZE - 1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--accum",     type=int, default=None)
    mode.add_argument("--max_accum", type=int, default=4,
                      help="Sweep accum_steps 1 .. N (ignored if --accum given).")
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--n_steps",      type=int, default=8,
                   help="Optimizer steps per accum value (default 8, enough to expose "
                        "bnb lazy-init growth).")
    p.add_argument("--whisper_ckpt", type=str, default="weights/whisper_small.pt")
    p.add_argument("--llama_ckpt",   type=str,
                   default="weights/Llama3.1-8B/Llama3.1-8B/Llama3.1-8B/")
    p.add_argument("--bf16", action="store_true",
                   help="Cast models to bfloat16 and use bfloat16 autocast "
                        "(matches find_max_batch.py — the intended fix).")
    p.add_argument("--empty_cache", action="store_true",
                   help="torch.cuda.empty_cache() between micro-batches.")
    return p.parse_args()


def _cuda_used_gb(device: torch.device) -> float:
    """Total GPU memory in use by this process (torch + bnb + CUDA driver)."""
    free, total = torch.cuda.mem_get_info(device)
    return (total - free) / 1e9


def _build_models(
    whisper_ckpt: str,
    llama_ckpt: str,
    device: torch.device,
    bf16: bool,
) -> tuple[WhisperEncoder, AudioAdapter, Llama, torch.optim.Optimizer, torch.amp.GradScaler]:
    dtype        = torch.bfloat16 if bf16 else torch.float32
    dtype_label  = "bfloat16" if bf16 else "fp32"
    bytes_per    = 2 if bf16 else 4

    print(f"Loading model weights (dtype={dtype_label}) …")
    encoder = WhisperEncoder()
    encoder.load_openai_weights(whisper_ckpt)
    adapter = AudioAdapter(llama_dim=4096)
    llama   = Llama(LlamaConfig(vocab_size=_VOCAB_SIZE))
    llama.load_meta_weights(llama_ckpt)

    encoder = encoder.to(dtype).to(device)
    adapter = adapter.to(dtype).to(device)
    llama   = llama.to(dtype).to(device)

    llama.enable_gradient_checkpointing()
    encoder.train(); adapter.train(); llama.train()

    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(
        [
            {"params": list(encoder.parameters()), "lr": 1e-6},
            {"params": list(adapter.parameters()), "lr": 5e-5},
            {"params": list(llama.parameters()),   "lr": 1.5e-5},
        ],
        betas=(0.9, 0.999), weight_decay=0.01,
    )
    scaler = torch.amp.GradScaler("cuda")

    n = (sum(p.numel() for p in encoder.parameters()) +
         sum(p.numel() for p in adapter.parameters()) +
         sum(p.numel() for p in llama.parameters()))
    p_gb = n * bytes_per / 1e9
    g_gb = p_gb
    a_gb = n * 2 / 1e9   # bnb int8 states: 1 byte × 2 states
    print(f"  params={p_gb:.1f} GB  grads={g_gb:.1f} GB  "
          f"bnb-states≈{a_gb:.1f} GB  "
          f"minimum_peak≈{p_gb+g_gb+a_gb:.1f} GB")
    return encoder, adapter, llama, optimizer, scaler


def _make_batch(batch_size: int, device: torch.device) -> tuple:
    mel           = torch.randn(batch_size, 80, 3000, device=device)
    audio_lengths = torch.full((batch_size,), _AUDIO_TOKS, dtype=torch.long, device=device)
    inst_ids      = torch.full((batch_size, _INST_TOKS),  4, dtype=torch.long, device=device)
    inst_lens     = torch.full((batch_size,), _INST_TOKS,    dtype=torch.long, device=device)
    trans_ids     = torch.full((batch_size, _TRANS_TOKS), 4, dtype=torch.long, device=device)
    trans_lens    = torch.full((batch_size,), _TRANS_TOKS,   dtype=torch.long, device=device)
    return mel, audio_lengths, inst_ids, inst_lens, trans_ids, trans_lens


def _run_accum(
    encoder, adapter, llama, optimizer, scaler,
    batch_size: int, accum_steps: int, n_steps: int,
    device: torch.device, empty_cache: bool, bf16: bool,
) -> tuple[bool, int, list[dict]]:
    """Run n_steps optimizer steps.

    Returns (success, fail_step, per_step_stats).
    fail_step is -1 on success.  per_step_stats has one entry per completed step.
    """
    autocast_dt = torch.bfloat16 if bf16 else torch.float16
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)

    stats: list[dict] = []

    for step in range(1, n_steps + 1):
        for k in range(accum_steps):
            try:
                mel, audio_lengths, inst_ids, inst_lens, trans_ids, trans_lens = (
                    _make_batch(batch_size, device)
                )

                if step == 1:
                    print(f"  step {step}  micro {k+1}/{accum_steps}  "
                          f"[before fwd]  "
                          f"torch={torch.cuda.memory_allocated(device)/1e9:.2f}GB  "
                          f"cuda_total={_cuda_used_gb(device):.2f}GB")

                with torch.amp.autocast("cuda", dtype=autocast_dt):
                    enc_out     = encoder(mel)
                    adapter_out = adapter(enc_out)
                    inputs, labels = prepare_input(
                        adapter_out, audio_lengths,
                        inst_ids, inst_lens,
                        trans_ids, trans_lens,
                        llama.embed_tokens, sep_token_id=_SEP_ID,
                    )
                    _, loss = llama(inputs, labels)

                if step == 1:
                    print(f"  step {step}  micro {k+1}/{accum_steps}  "
                          f"[after  fwd]  "
                          f"torch={torch.cuda.memory_allocated(device)/1e9:.2f}GB  "
                          f"cuda_total={_cuda_used_gb(device):.2f}GB")

                scaler.scale(loss / accum_steps).backward()

                if step == 1:
                    print(f"  step {step}  micro {k+1}/{accum_steps}  "
                          f"[after  bwd]  "
                          f"torch={torch.cuda.memory_allocated(device)/1e9:.2f}GB  "
                          f"cuda_total={_cuda_used_gb(device):.2f}GB")

                if empty_cache:
                    torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                return False, step, stats

        try:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]], 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if empty_cache:
                torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            return False, step, stats

        peak  = torch.cuda.max_memory_allocated(device) / 1e9
        alloc = torch.cuda.memory_allocated(device) / 1e9
        cuda  = _cuda_used_gb(device)
        torch.cuda.reset_peak_memory_stats(device)

        stats.append({
            "step":  step, "peak_torch": peak,
            "alloc": alloc, "cuda_total": cuda,
            "bnb_approx": cuda - alloc,   # memory not tracked by torch (bnb states + driver)
        })

        if step == 1:
            print(f"  step {step}  [after opt+zero_grad]  "
                  f"torch={alloc:.2f}GB  cuda_total={cuda:.2f}GB  "
                  f"untracked≈{cuda-alloc:.2f}GB  peak={peak:.2f}GB")
        else:
            print(f"  step {step:3d}  torch={alloc:.2f}GB  "
                  f"cuda_total={cuda:.2f}GB  untracked≈{cuda-alloc:.2f}GB  "
                  f"peak={peak:.2f}GB")

    return True, -1, stats


def main() -> None:
    args   = _parse_args()
    device = torch.device("cuda")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.", file=sys.stderr)
        sys.exit(1)

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    exp_seg  = "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")

    print(f"GPU: {torch.cuda.get_device_name(0)}  ({total_gb:.1f} GB)")
    print(f"dtype={'bfloat16' if args.bf16 else 'fp32 (current train.py)'}  "
          f"batch_size={args.batch_size}  n_steps={args.n_steps}  "
          f"empty_cache={args.empty_cache}  "
          f"expandable_segments={'ON' if exp_seg else 'off'}")
    print()

    encoder, adapter, llama, optimizer, scaler = _build_models(
        args.whisper_ckpt, args.llama_ckpt, device, args.bf16,
    )
    print(f"After load:  torch={torch.cuda.memory_allocated(device)/1e9:.2f}GB  "
          f"cuda_total={_cuda_used_gb(device):.2f}GB")
    print()

    accum_values = [args.accum] if args.accum is not None else list(range(1, args.max_accum + 1))

    results = []
    for accum in accum_values:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        gc.collect()
        print(f"─── accum_steps={accum}  effective_batch={args.batch_size * accum} ───")

        ok, fail_step, step_stats = _run_accum(
            encoder, adapter, llama, optimizer, scaler,
            batch_size=args.batch_size, accum_steps=accum,
            n_steps=args.n_steps, device=device,
            empty_cache=args.empty_cache, bf16=args.bf16,
        )

        if ok:
            print(f"  → OK  all {args.n_steps} steps passed")
            if len(step_stats) > 1:
                bnb_growth = step_stats[-1]["bnb_approx"] - step_stats[0]["bnb_approx"]
                peak_growth = step_stats[-1]["peak_torch"] - step_stats[0]["peak_torch"]
                if abs(bnb_growth) > 0.1 or abs(peak_growth) > 0.1:
                    print(f"  untracked growth step 1→{args.n_steps}: "
                          f"{bnb_growth:+.2f} GB  "
                          f"peak growth: {peak_growth:+.2f} GB")
        else:
            print(f"  → OOM at step {fail_step}")
            if step_stats:
                bnb_s1 = step_stats[0]["bnb_approx"]
                bnb_last = step_stats[-1]["bnb_approx"]
                print(f"  untracked memory grew from "
                      f"{bnb_s1:.2f} GB (step 1) → "
                      f"{bnb_last:.2f} GB (step {len(step_stats)}) "
                      f"[delta={bnb_last-bnb_s1:+.2f} GB]")
        print()
        results.append((accum, ok, fail_step, step_stats))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("═" * 58)
    print(f"SUMMARY  dtype={'bfloat16' if args.bf16 else 'fp32'}")
    print("═" * 58)
    print(f"  {'accum':>6}  {'eff_bs':>7}  {'peak_step1':>12}  result")
    print("  " + "─" * 50)
    any_oom = False
    for accum, ok, fail_step, step_stats in results:
        eff = args.batch_size * accum
        peak1 = f"{step_stats[0]['peak_torch']:.1f} GB" if step_stats else "—"
        status = "OK" if ok else f"OOM at step {fail_step}"
        print(f"  {accum:6d}  {eff:7d}  {peak1:>12}  {status}")
        if not ok:
            any_oom = True
    print()

    if any_oom and not args.bf16:
        print("Analysis:")
        if step_stats and len(step_stats) >= 2:
            growth = step_stats[-1]["bnb_approx"] - step_stats[0]["bnb_approx"]
            print(f"  Untracked memory (bnb states) grew by {growth:.2f} GB "
                  f"over {len(step_stats)} steps.")
            print("  This is NOT fragmentation — expandable_segments and empty_cache "
                  "cannot help.")
        print()
        print("Recommended fix:")
        print("  Re-run this probe with --bf16")
        print("  Matches find_max_batch.py: models cast to bfloat16, bfloat16 autocast.")
        print("  Halves both param memory and grad memory, dropping the backward peak")
        print("  from ~81 GB to ~50 GB and leaving ~35 GB of headroom for bnb states.")
        print()
        print("  If --bf16 passes here, the change needed in train.py is:")
        print("    encoder/adapter/llama: .to(torch.bfloat16) before .to(device)")
        print("    all torch.amp.autocast(..., dtype=torch.float16)")
        print("      → torch.amp.autocast(..., dtype=torch.bfloat16)")

    elif any_oom and args.bf16:
        print("bf16 also OOMs. Reduce batch_size or increase gradient checkpointing coverage.")

    elif not any_oom and args.bf16:
        print("bf16 passes. This confirms the fix: use bfloat16 in train.py.")

    elif not any_oom and not args.bf16:
        print("fp32 passes for all tested accum values.")


if __name__ == "__main__":
    main()

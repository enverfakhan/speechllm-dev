"""Binary search for the maximum batch size that fits in GPU VRAM.

Stage 1: adapter only (encoder + Llama frozen, excluded from optimizer).
Stage 2: encoder + adapter + Llama, all trainable.

Both stages use:
  - Full Llama 3.1 8B dims (d_model=4096, 32 layers, GQA 32/8 heads)
  - Flash Attention via F.scaled_dot_product_attention
  - torch.utils.checkpoint for gradient checkpointing
  - torch.autocast(bfloat16)
  - 8-bit AdamW (bitsandbytes)

Usage:
    python scripts/find_max_batch.py --stage 1
    python scripts/find_max_batch.py --stage 2
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make parent dir importable so we can reuse model classes
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.whisper_encoder import WhisperEncoder
from model.adapter import AudioAdapter
from model.llama import Llama, LlamaConfig


# ── Constants ─────────────────────────────────────────────────────────────────
# Sequence length: 375 audio tokens + ~50 instruction tokens = 425 total input tokens
# During training the full sequence (audio + instruction + transcript) passes through
# Llama. We use a fixed 425-token prefix + 50 transcript tokens = 475 seq len.
_SEQ_LEN     = 475
_AUDIO_TOKS  = 375   # adapter output tokens for ~30s audio
_INST_TOKS   = 50    # typical instruction token count
_TRANS_TOKS  = 50    # typical transcript token count (loss computed here)
_VOCAB_SIZE  = 40148
_TOTAL_VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 80.0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Binary-search max batch size for a given training stage.")
    p.add_argument("--stage", type=int, choices=[1, 2], required=True,
                   help="Training stage: 1=adapter only, 2=all params trainable.")
    p.add_argument("--min_batch", type=int, default=1)
    p.add_argument("--max_batch", type=int, default=128)
    return p.parse_args()


def _build_models(stage: int) -> tuple[WhisperEncoder, AudioAdapter, Llama]:
    llama_cfg = LlamaConfig(vocab_size=_VOCAB_SIZE)
    encoder   = WhisperEncoder()
    adapter   = AudioAdapter(llama_dim=4096)
    llama     = Llama(llama_cfg)

    device = torch.device("cuda")
    encoder = encoder.to(device)
    adapter = adapter.to(device)
    llama   = llama.to(device)

    if stage == 1:
        encoder.requires_grad_(False)
        llama.requires_grad_(False)

    return encoder, adapter, llama


def _build_optimizer(stage: int, encoder: WhisperEncoder, adapter: AudioAdapter, llama: Llama):
    import bitsandbytes as bnb
    if stage == 1:
        return bnb.optim.AdamW8bit(adapter.parameters(), weight_decay=0.01)
    return bnb.optim.AdamW8bit(
        [
            {"params": encoder.parameters(), "lr": 1e-7},
            {"params": adapter.parameters(), "lr": 1e-5},
            {"params": llama.parameters(),   "lr": 1e-5},
        ],
        weight_decay=0.01,
    )


def _try_batch(
    encoder: WhisperEncoder,
    adapter: AudioAdapter,
    llama: Llama,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    stage: int,
) -> tuple[bool, float]:
    """Attempt a forward + backward pass at the given batch size.

    Returns (success, peak_vram_gb).
    """
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    try:
        # Dummy mel: (B, 80, 3000) — 30s at 100 fps
        mel          = torch.randn(batch_size, 80, 3000, device=device)
        audio_lengths = torch.full((batch_size,), _AUDIO_TOKS, dtype=torch.long, device=device)
        # Dummy instruction and transcript token IDs
        inst_ids     = torch.randint(0, _VOCAB_SIZE, (batch_size, _INST_TOKS), device=device)
        inst_lens    = torch.full((batch_size,), _INST_TOKS, dtype=torch.long, device=device)
        trans_ids    = torch.randint(0, _VOCAB_SIZE, (batch_size, _TRANS_TOKS), device=device)
        trans_lens   = torch.full((batch_size,), _TRANS_TOKS, dtype=torch.long, device=device)

        from model.adapter import prepare_input

        optimizer.zero_grad()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            enc_out     = encoder(mel)
            adapter_out = adapter(enc_out)
            inputs, labels = prepare_input(
                adapter_out, audio_lengths,
                inst_ids, inst_lens,
                trans_ids, trans_lens,
                llama.embed_tokens,
                sep_token_id=_VOCAB_SIZE - 1,  # dummy SEP
            )
            _, loss = llama(inputs, labels)

        loss.backward()
        optimizer.step()

        peak = torch.cuda.max_memory_allocated(device) / 1e9
        return True, peak

    except torch.cuda.OutOfMemoryError:
        return False, 0.0
    finally:
        # Clean up activations so OOM state doesn't persist
        torch.cuda.empty_cache()
        gc.collect()


def main() -> None:
    args = _parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.", file=sys.stderr)
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {_TOTAL_VRAM_GB:.1f} GB")
    print(f"Stage {args.stage}: {'adapter only' if args.stage == 1 else 'all params trainable'}")
    print()

    encoder, adapter, llama = _build_models(args.stage)
    optimizer = _build_optimizer(args.stage, encoder, adapter, llama)

    lo, hi = args.min_batch, args.max_batch
    best   = 0
    best_peak = 0.0

    # First check if even batch=1 works
    ok, peak = _try_batch(encoder, adapter, llama, optimizer, 1, args.stage)
    util = peak / _TOTAL_VRAM_GB * 100
    if ok:
        print(f"batch={1:4d}  ✓ OK   peak={peak:.1f}GB  utilisation={util:.0f}%")
        best = 1
        best_peak = peak
    else:
        print(f"batch={1:4d}  ✗ OOM")
        print(f"\nMAX BATCH SIZE (stage {args.stage}): 0  (OOM at batch=1)")
        return

    # Binary search
    lo = 2
    while lo <= hi:
        mid = (lo + hi) // 2
        ok, peak = _try_batch(encoder, adapter, llama, optimizer, mid, args.stage)
        util = peak / _TOTAL_VRAM_GB * 100
        if ok:
            print(f"batch={mid:4d}  ✓ OK   peak={peak:.1f}GB  utilisation={util:.0f}%")
            best = mid
            best_peak = peak
            lo = mid + 1
        else:
            print(f"batch={mid:4d}  ✗ OOM")
            hi = mid - 1

    util_best = best_peak / _TOTAL_VRAM_GB * 100
    sweep = [best, best * 2, best * 4, best * 8]

    print()
    print(f"MAX BATCH SIZE (stage {args.stage}): {best}")
    print(f"Peak VRAM: {best_peak:.1f}GB / {_TOTAL_VRAM_GB:.1f}GB")
    print(f"Recommended sweep range: {sweep}")


if __name__ == "__main__":
    main()

"""Binary search for the maximum batch size that fits in VRAM.

Eval mode (--eval):
  No optimizer, no gradients, no gradient checkpointing.
  Models run in .eval() under torch.no_grad() + bfloat16 autocast.
  Use this to find the largest batch size for inference / evaluation.

Training modes (--stage 1, 2, or 3):
  Stage 1: adapter only (encoder + Llama frozen, excluded from optimizer).
  Stage 2: encoder + adapter + Llama, all trainable.
  Stage 3: encoder + adapter trainable, Llama frozen (excluded from optimizer).

  All training stages use:
    - Full Llama 3.1 8B dims (d_model=4096, 32 layers, GQA 32/8 heads)
    - Flash Attention via F.scaled_dot_product_attention
    - torch.utils.checkpoint for gradient checkpointing
    - torch.autocast(bfloat16)
    - 8-bit AdamW (bitsandbytes)

  For each accum_steps value the script finds the largest micro-batch size that
  fits in VRAM when running accum_steps forward+backward passes before a single
  optimizer step.

  Stage 2/3 note: bitsandbytes AdamW8bit defers GPU state allocation until the 4th
  optimizer.step() call. Without a warm-up this creates a surprise memory spike
  mid-accumulation that would make the binary search unreliable. The script runs
  4 warm-up steps at batch=1 before searching to pre-materialise the state.

Usage:
    python tools/find_max_batch.py --eval
    python tools/find_max_batch.py --stage 2
    python tools/find_max_batch.py --stage 2 --accum 1 4 8
    python tools/find_max_batch.py --stage 1 --accum 1
    python tools/find_max_batch.py --stage 3 --accum 1 4 8
"""


from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.adapter import AudioAdapter
from model.sequence import prepare_input
from model.llama import Llama, LlamaConfig
from model.whisper_encoder import WhisperEncoder


# ── Constants ─────────────────────────────────────────────────────────────────
_AUDIO_TOKS = 375   # adapter output tokens for ~30s audio (mel_t=3000)
_INST_TOKS  = 50
_TRANS_TOKS = 50
_VOCAB_SIZE = 40034   # data/pruned_tokenizer/pruned_config.json is ground truth
_TOTAL_VRAM_GB = (
    torch.cuda.get_device_properties(0).total_memory / 1e9
    if torch.cuda.is_available() else 80.0
)


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Binary-search max batch size for eval or training stages."
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--eval", action="store_true",
                      help="Eval mode: no optimizer, no gradients, no grad checkpointing.")
    mode.add_argument("--stage", type=int, choices=[1, 2, 3],
                      help="Training stage: 1=adapter only, 2=all params trainable, "
                           "3=encoder+adapter trainable (Llama frozen).")
    p.add_argument("--accum", type=int, nargs="+", default=[1, 2, 4, 8],
                   help="Gradient accumulation steps to probe (training only, default: 1 2 4 8).")
    p.add_argument("--min_batch", type=int, default=1)
    p.add_argument("--max_batch", type=int, default=128)
    return p.parse_args()


# ── Model / optimizer construction ───────────────────────────────────────────

def _build_models(
    stage: int | None,
    for_eval: bool = False,
) -> tuple[WhisperEncoder, AudioAdapter, Llama]:
    llama_cfg = LlamaConfig(vocab_size=_VOCAB_SIZE)
    encoder   = WhisperEncoder()
    encoder.load_openai_weights("./weights/whisper_small.pt")
    adapter   = AudioAdapter(llama_dim=4096)
    llama     = Llama(llama_cfg)
    llama.load_meta_weights("./weights/Llama3.1-8B/Llama3.1-8B/Llama3.1-8B/")

    device = torch.device("cuda")
    encoder = encoder.to(device)
    adapter = adapter.to(device)
    llama   = llama.to(device)

    if for_eval:
        encoder.eval()
        adapter.eval()
        llama.eval()
        return encoder, adapter, llama

    if stage == 1:
        encoder.requires_grad_(False)
        llama.requires_grad_(False)
    elif stage == 3:
        llama.requires_grad_(False)

    llama.enable_gradient_checkpointing()
    encoder.train()
    adapter.train()
    llama.train()

    return encoder, adapter, llama


def _build_optimizer(
    stage: int,
    encoder: WhisperEncoder,
    adapter: AudioAdapter,
    llama: Llama,
) -> torch.optim.Optimizer:
    import bitsandbytes as bnb
    if stage == 1:
        return bnb.optim.AdamW8bit(adapter.parameters(), weight_decay=0.01)
    if stage == 3:
        return bnb.optim.AdamW8bit(
            [
                {"params": list(encoder.parameters()), "lr": 1e-7},
                {"params": list(adapter.parameters()), "lr": 1e-5},
            ],
            weight_decay=0.01,
        )
    return bnb.optim.AdamW8bit(
        [
            {"params": list(encoder.parameters()), "lr": 1e-7},
            {"params": list(adapter.parameters()), "lr": 1e-5},
            {"params": list(llama.parameters()),   "lr": 1e-5},
        ],
        weight_decay=0.01,
    )


# ── bnb warm-up ───────────────────────────────────────────────────────────────

def _warmup_optimizer(
    encoder: WhisperEncoder,
    adapter: AudioAdapter,
    llama: Llama,
    optimizer: torch.optim.Optimizer,
    n_steps: int = 4,
) -> None:
    """Run n_steps optimizer steps at batch=1, mel_t=800 to materialise bnb state.

    bitsandbytes AdamW8bit defers GPU allocation of exp_avg/exp_avg_sq tensors
    until the 4th optimizer.step() call. Pre-materialising ensures the binary
    search always operates against a stable memory baseline.
    """
    device = torch.device("cuda")
    audio_len = (800 // 2 + 3) // 4  # adapter token count for mel_t=800

    for _ in range(n_steps):
        mel       = torch.randn(1, 80, 800, device=device, dtype=torch.bfloat16)
        a_lens    = torch.full((1,), audio_len, dtype=torch.long, device=device)
        inst_ids  = torch.randint(0, _VOCAB_SIZE, (1, _INST_TOKS), device=device)
        inst_lens = torch.full((1,), _INST_TOKS, dtype=torch.long, device=device)
        tr_ids    = torch.randint(0, _VOCAB_SIZE, (1, _TRANS_TOKS), device=device)
        tr_lens   = torch.full((1,), _TRANS_TOKS, dtype=torch.long, device=device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            enc_out = encoder(mel)
            adp_out = adapter(enc_out)
            inputs, labels = prepare_input(
                adp_out, a_lens, inst_ids, inst_lens, tr_ids, tr_lens,
                llama.embed_tokens, sep_token_id=_VOCAB_SIZE - 1,
            )
            _, loss = llama(inputs, labels, audio_lengths=a_lens)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        del mel, a_lens, inst_ids, inst_lens, tr_ids, tr_lens
        del enc_out, adp_out, inputs, labels, loss

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)


# ── Single probe ──────────────────────────────────────────────────────────────

def _try_batch(
    encoder: WhisperEncoder,
    adapter: AudioAdapter,
    llama: Llama,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    accum_steps: int,
) -> tuple[bool, float]:
    """Attempt accum_steps micro-batches + one optimizer step at batch_size.

    Returns (success, peak_vram_gb).
    """
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    try:
        optimizer.zero_grad(set_to_none=True)

        for _ in range(accum_steps):
            mel       = torch.randn(batch_size, 80, 3000, device=device, dtype=torch.bfloat16)
            a_lens    = torch.full((batch_size,), _AUDIO_TOKS, dtype=torch.long, device=device)
            inst_ids  = torch.randint(0, _VOCAB_SIZE, (batch_size, _INST_TOKS), device=device)
            inst_lens = torch.full((batch_size,), _INST_TOKS, dtype=torch.long, device=device)
            tr_ids    = torch.randint(0, _VOCAB_SIZE, (batch_size, _TRANS_TOKS), device=device)
            tr_lens   = torch.full((batch_size,), _TRANS_TOKS, dtype=torch.long, device=device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                enc_out = encoder(mel)
                adp_out = adapter(enc_out)
                inputs, labels = prepare_input(
                    adp_out, a_lens, inst_ids, inst_lens, tr_ids, tr_lens,
                    llama.embed_tokens, sep_token_id=_VOCAB_SIZE - 1,
                )
                _, loss = llama(inputs, labels, audio_lengths=a_lens)

            (loss / accum_steps).backward()

        optimizer.step()

        peak = torch.cuda.max_memory_allocated(device) / 1e9
        return True, peak

    except torch.cuda.OutOfMemoryError:
        return False, 0.0
    finally:
        torch.cuda.empty_cache()
        gc.collect()


# ── Eval probe ───────────────────────────────────────────────────────────────

def _try_batch_eval(
    encoder: WhisperEncoder,
    adapter: AudioAdapter,
    llama: Llama,
    batch_size: int,
) -> tuple[bool, float]:
    """Forward-only pass under torch.no_grad(). No optimizer, no backward.

    Returns (success, peak_vram_gb).
    """
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    try:
        mel       = torch.randn(batch_size, 80, 3000, device=device, dtype=torch.bfloat16)
        a_lens    = torch.full((batch_size,), _AUDIO_TOKS, dtype=torch.long, device=device)
        inst_ids  = torch.randint(0, _VOCAB_SIZE, (batch_size, _INST_TOKS), device=device)
        inst_lens = torch.full((batch_size,), _INST_TOKS, dtype=torch.long, device=device)
        tr_ids    = torch.randint(0, _VOCAB_SIZE, (batch_size, _TRANS_TOKS), device=device)
        tr_lens   = torch.full((batch_size,), _TRANS_TOKS, dtype=torch.long, device=device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            enc_out = encoder(mel)
            adp_out = adapter(enc_out)
            inputs, labels = prepare_input(
                adp_out, a_lens, inst_ids, inst_lens, tr_ids, tr_lens,
                llama.embed_tokens, sep_token_id=_VOCAB_SIZE - 1,
            )
            _, loss = llama(inputs, labels, audio_lengths=a_lens)

        peak = torch.cuda.max_memory_allocated(device) / 1e9
        return True, peak

    except torch.cuda.OutOfMemoryError:
        return False, 0.0
    finally:
        torch.cuda.empty_cache()
        gc.collect()


# ── Binary search ─────────────────────────────────────────────────────────────

def _search(
    encoder: WhisperEncoder,
    adapter: AudioAdapter,
    llama: Llama,
    optimizer: torch.optim.Optimizer,
    accum_steps: int,
    min_batch: int,
    max_batch: int,
) -> tuple[int, float]:
    """Binary search for max micro-batch size at a given accum_steps.

    Returns (max_batch_size, peak_vram_gb_at_max).
    """
    ok, peak = _try_batch(encoder, adapter, llama, optimizer, 1, accum_steps)
    util = peak / _TOTAL_VRAM_GB * 100
    if ok:
        print(f"  batch={1:4d}  ✓ OK   peak={peak:.1f}GB  util={util:.0f}%")
        best, best_peak = 1, peak
    else:
        print(f"  batch={1:4d}  ✗ OOM")
        return 0, 0.0

    lo, hi = max(2, min_batch), max_batch
    while lo <= hi:
        mid = (lo + hi) // 2
        ok, peak = _try_batch(encoder, adapter, llama, optimizer, mid, accum_steps)
        util = peak / _TOTAL_VRAM_GB * 100
        if ok:
            print(f"  batch={mid:4d}  ✓ OK   peak={peak:.1f}GB  util={util:.0f}%")
            best, best_peak = mid, peak
            lo = mid + 1
        else:
            print(f"  batch={mid:4d}  ✗ OOM")
            hi = mid - 1

    return best, best_peak


# ── Eval binary search ────────────────────────────────────────────────────────

def _search_eval(
    encoder: WhisperEncoder,
    adapter: AudioAdapter,
    llama: Llama,
    min_batch: int,
    max_batch: int,
) -> tuple[int, float]:
    """Binary search for max eval batch size. Returns (max_batch_size, peak_vram_gb)."""
    ok, peak = _try_batch_eval(encoder, adapter, llama, 1)
    util = peak / _TOTAL_VRAM_GB * 100
    if ok:
        print(f"  batch={1:4d}  ✓ OK   peak={peak:.1f}GB  util={util:.0f}%")
        best, best_peak = 1, peak
    else:
        print(f"  batch={1:4d}  ✗ OOM")
        return 0, 0.0

    lo, hi = max(2, min_batch), max_batch
    while lo <= hi:
        mid = (lo + hi) // 2
        ok, peak = _try_batch_eval(encoder, adapter, llama, mid)
        util = peak / _TOTAL_VRAM_GB * 100
        if ok:
            print(f"  batch={mid:4d}  ✓ OK   peak={peak:.1f}GB  util={util:.0f}%")
            best, best_peak = mid, peak
            lo = mid + 1
        else:
            print(f"  batch={mid:4d}  ✗ OOM")
            hi = mid - 1

    return best, best_peak


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.", file=sys.stderr)
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {_TOTAL_VRAM_GB:.1f} GB")

    # ── Eval path ─────────────────────────────────────────────────────────────
    if args.eval:
        print("Mode: eval (no optimizer, no gradients, no grad checkpointing)")
        print()
        encoder, adapter, llama = _build_models(stage=None, for_eval=True)
        best, best_peak = _search_eval(
            encoder, adapter, llama,
            min_batch=args.min_batch,
            max_batch=args.max_batch,
        )
        util = best_peak / _TOTAL_VRAM_GB * 100 if best > 0 else 0.0
        print()
        print("═" * 50)
        print("SUMMARY  eval (forward-only, bfloat16)")
        print("═" * 50)
        if best > 0:
            print(f"  max batch size : {best}")
            print(f"  peak VRAM      : {best_peak:.1f} GB  ({util:.0f}% of {_TOTAL_VRAM_GB:.1f} GB)")
        else:
            print("  OOM at batch=1 — cannot fit even a single sample.")
        return

    # ── Training path ─────────────────────────────────────────────────────────
    stage_labels = {
        1: "adapter only (encoder + Llama frozen)",
        2: "all params trainable",
        3: "encoder + adapter trainable (Llama frozen)",
    }
    print(f"Stage {args.stage}: {stage_labels[args.stage]}")
    print(f"Probing accum_steps: {args.accum}")
    print()

    encoder, adapter, llama = _build_models(args.stage, for_eval=False)
    optimizer = _build_optimizer(args.stage, encoder, adapter, llama)

    if args.stage in (2, 3):
        print("Pre-warming optimizer (4 steps at batch=1, mel_t=800) ...")
        _warmup_optimizer(encoder, adapter, llama, optimizer)
        try:
            import scripts.probe_accum as _pa
            state_gb = _pa._opt_state_gb(optimizer)
            n_state  = len(optimizer.state)
            n_total  = sum(len(list(g["params"])) for g in optimizer.param_groups)
            print(f"  bnb state = {state_gb:.2f} GB  entries = {n_state}/{n_total}")
        except Exception:
            pass
        print()

    results: list[tuple[int, int, float]] = []  # (accum, max_bs, peak_gb)

    for accum in args.accum:
        print(f"─── accum_steps={accum}  (eff. batch = micro_batch × {accum}) ───")
        best, best_peak = _search(
            encoder, adapter, llama, optimizer,
            accum_steps=accum,
            min_batch=args.min_batch,
            max_batch=args.max_batch,
        )
        results.append((accum, best, best_peak))
        eff = best * accum
        print(f"  → max micro-batch={best}  eff. batch={eff}\n")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("═" * 62)
    summary_labels = {
        1: "adapter-only",
        2: "all params, bf16",
        3: "encoder+adapter, Llama frozen, bf16",
    }
    print(f"SUMMARY  stage={args.stage}  ({summary_labels[args.stage]})")
    print("═" * 62)
    print(f"  {'accum':>6}  {'micro-bs':>9}  {'eff-bs':>7}  {'peak':>8}  {'util':>6}")
    print("  " + "─" * 50)
    for accum, max_bs, peak in results:
        eff_bs = max_bs * accum
        util   = peak / _TOTAL_VRAM_GB * 100 if max_bs > 0 else 0.0
        status = f"{peak:7.1f} GB  {util:5.0f}%" if max_bs > 0 else "      OOM"
        print(f"  {accum:6d}  {max_bs:9d}  {eff_bs:7d}  {status}")


if __name__ == "__main__":
    main()

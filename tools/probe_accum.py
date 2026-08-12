"""Probe VRAM under gradient accumulation for stage-2 training.

Two live hypotheses for the step-5 OOM (expandable_segments already ON,
graph retention already ruled out by the existing probe):

  (A) cuDNN/cuBLAS workspace autotuning: the Whisper conv stem re-tunes per
      new mel-T shape and caches growing workspaces in non-PyTorch memory.
      Signature: `external` column grows; stops growing with --cudnn_benchmark false.

  (B) bnb 8-bit Adam lazy init: some trainable params receive no gradient on
      step 1, so their optimizer state is deferred to later steps.
      Signature: `optstate` column grows past step 1.

Simple OOM probe (existing behaviour, no --diag flag):
  python scripts/probe_accum.py --accum 4 --n_steps 8 [--bf16]

Diagnostic run (A vs B table, 12 steps, variable mel lengths):
  python scripts/probe_accum.py --accum 4 --diag [--bf16]
  python scripts/probe_accum.py --accum 4 --diag --cudnn_benchmark false [--bf16]
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.adapter import AudioAdapter
from model.sequence import prepare_input
from model.llama import Llama, LlamaConfig
from model.whisper_encoder import WhisperEncoder

# ── Constants ─────────────────────────────────────────────────────────────────
_VOCAB_SIZE = 40034   # data/pruned_tokenizer/pruned_config.json is ground truth
_INST_TOKS  = 19     # the longer of data.py's two INSTRUCTION_VARIANTS
_TRANS_TOKS = 50     # representative transcript length
_SEP_ID     = _VOCAB_SIZE - 1

# GradScaler._unscale_ calls _amp_foreach_non_finite_check_and_unscale_cuda,
# which is not implemented for BFloat16 even in PyTorch 2.8.  Use a fixed
# manual loss scale instead: bf16 has fp32-equivalent dynamic range so it
# cannot overflow, but its 7-bit mantissa needs amplification to prevent
# adapter gradients from rounding away through 32 Llama layers backward.
_BF16_GRAD_SCALE = 4096.0


# ── Memory helpers ────────────────────────────────────────────────────────────

def _opt_state_gb(optimizer: torch.optim.Optimizer) -> float:
    """Sum of all tensor bytes in optimizer.state (bnb states, absmax codes, etc.)."""
    total = 0
    for param_state in optimizer.state.values():
        for v in param_state.values():
            if isinstance(v, torch.Tensor):
                total += v.numel() * v.element_size()
    return total / 1e9


def _mem_snapshot(device: torch.device) -> dict[str, float]:
    """Capture a point-in-time memory breakdown.

    Returns:
        alloc    — torch.cuda.memory_allocated (live tensors tracked by PyTorch)
        resv     — torch.cuda.memory_reserved  (PyTorch pool: alloc + cached free)
        used     — CUDA total − CUDA free       (everything: PyTorch + bnb + driver)
        external — used − resv                  (cuDNN/cuBLAS workspaces, bnb states
                                                 allocated outside PyTorch's pool)
        frag     — resv − alloc                 (PyTorch cache fragmentation)
    """
    alloc = torch.cuda.memory_allocated(device) / 1e9
    resv  = torch.cuda.memory_reserved(device)  / 1e9
    free, total = torch.cuda.mem_get_info(device)
    used = (total - free) / 1e9
    return {
        "alloc":    alloc,
        "resv":     resv,
        "used":     used,
        "external": used - resv,
        "frag":     resv - alloc,
    }


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-2 VRAM probe and A/B diagnostic.")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--accum",     type=int, default=None,
                      help="Single accum_steps value to test.")
    mode.add_argument("--max_accum", type=int, default=4,
                      help="Sweep accum_steps 1..N (ignored if --accum given).")

    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--n_steps",      type=int, default=8,
                   help="Optimizer steps per run (overridden to 12 in --diag mode).")
    p.add_argument("--seed",         type=int, default=42,
                   help="RNG seed for reproducible mel-length sequences.")
    p.add_argument("--whisper_ckpt", type=str, default="weights/whisper_small.pt")
    p.add_argument("--llama_ckpt",   type=str,
                   default="weights/Llama3.1-8B/Llama3.1-8B/Llama3.1-8B/")
    p.add_argument("--bf16",         action="store_true",
                   help="bf16 params + bf16 autocast + manual loss scale (no GradScaler).")
    p.add_argument("--empty_cache",  action="store_true",
                   help="torch.cuda.empty_cache() between micro-batches.")
    p.add_argument("--diag",         action="store_true",
                   help="Run A/B diagnostic: per-step table, bnb lazy-init check, verdict.")
    p.add_argument("--cudnn_benchmark", choices=["true", "false"], default="false",
                   help="Set torch.backends.cudnn.benchmark. "
                        "true may grow external memory via cuDNN autotuning (hypothesis A).")
    p.add_argument("--var_seq_len",  action="store_true",
                   help="Vary mel length per micro-batch (U[800,3000] rounded to ×8) "
                        "to trigger cuDNN re-autotuning. Implied by --diag.")
    return p.parse_args()


# ── Model construction ────────────────────────────────────────────────────────

def _build_models(
    whisper_ckpt: str,
    llama_ckpt: str,
    device: torch.device,
    bf16: bool,
) -> tuple[WhisperEncoder, AudioAdapter, Llama, torch.optim.Optimizer,
           torch.amp.GradScaler | None]:
    dtype      = torch.bfloat16 if bf16 else torch.float32
    bytes_per  = 2 if bf16 else 4

    print(f"Loading model weights (dtype={'bfloat16' if bf16 else 'fp32'}) …")
    encoder = WhisperEncoder()
    encoder.load_openai_weights(whisper_ckpt)
    adapter = AudioAdapter(llama_dim=4096)
    llama   = Llama(LlamaConfig(vocab_size=_VOCAB_SIZE))
    llama.load_meta_weights(llama_ckpt)

    encoder = encoder.to(dtype).to(device)
    adapter = adapter.to(dtype).to(device)
    llama   = llama.to(dtype).to(device)

    llama.enable_gradient_checkpointing()
    encoder.train()
    adapter.train()
    llama.train()

    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(
        [
            {"params": list(encoder.parameters()), "lr": 1e-6},
            {"params": list(adapter.parameters()), "lr": 5e-5},
            {"params": list(llama.parameters()),   "lr": 1.5e-5},
        ],
        betas=(0.9, 0.999), weight_decay=0.01,
    )

    # bf16: GradScaler._unscale_ not implemented for BFloat16 (PyTorch 2.8).
    # fp32: GradScaler needed to prevent fp16 autocast gradient underflow.
    scaler = None if bf16 else torch.amp.GradScaler("cuda")

    n = sum(p.numel() for p in encoder.parameters()) + \
        sum(p.numel() for p in adapter.parameters()) + \
        sum(p.numel() for p in llama.parameters())
    p_gb = n * bytes_per / 1e9
    print(f"  params={p_gb:.1f} GB  grads≈{p_gb:.1f} GB  "
          f"bnb-states≈{n*2/1e9:.1f} GB  "
          f"minimum_peak (params+grads+bnb)≈{p_gb*2 + n*2/1e9:.1f} GB")

    return encoder, adapter, llama, optimizer, scaler


# ── Batch factory ─────────────────────────────────────────────────────────────

def _make_batch(
    batch_size: int,
    device: torch.device,
    mel_t: int = 3000,
) -> tuple[torch.Tensor, ...]:
    """Fixed or variable-length batch. mel_t sets the time dimension of the mel."""
    audio_len  = (mel_t // 2 + 3) // 4          # adapter output token count
    mel         = torch.randn(batch_size, 80, mel_t, device=device)
    audio_lengths = torch.full((batch_size,), audio_len, dtype=torch.long, device=device)
    inst_ids    = torch.full((batch_size, _INST_TOKS),  4, dtype=torch.long, device=device)
    inst_lens   = torch.full((batch_size,), _INST_TOKS,    dtype=torch.long, device=device)
    trans_ids   = torch.full((batch_size, _TRANS_TOKS), 4, dtype=torch.long, device=device)
    trans_lens  = torch.full((batch_size,), _TRANS_TOKS,   dtype=torch.long, device=device)
    return mel, audio_lengths, inst_ids, inst_lens, trans_ids, trans_lens


def _sample_mel_t(rng: random.Random) -> int:
    """Sample a mel time length from a realistic distribution (1–30 s, step 8 frames)."""
    raw = rng.randint(800, 3000)
    return (raw // 8) * 8


# ── Optimizer step helpers ────────────────────────────────────────────────────

def _optimizer_step(
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    all_params: list[torch.nn.Parameter],
    grad_clip: float = 1.0,
) -> None:
    """Unscale + clip + step, handling both GradScaler (fp32) and manual scale (bf16)."""
    if scaler is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        # Manual unscale: bf16 cannot produce inf, so no inf/nan check needed.
        for p in all_params:
            if p.grad is not None:
                p.grad.div_(_BF16_GRAD_SCALE)
        torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
        optimizer.step()


def _backward(
    loss: torch.Tensor,
    accum_steps: int,
    scaler: torch.amp.GradScaler | None,
) -> None:
    if scaler is not None:
        scaler.scale(loss / accum_steps).backward()
    else:
        (loss * _BF16_GRAD_SCALE / accum_steps).backward()


# ── Simple OOM probe (existing mode) ─────────────────────────────────────────

def _run_accum(
    encoder: WhisperEncoder,
    adapter: AudioAdapter,
    llama: Llama,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    batch_size: int,
    accum_steps: int,
    n_steps: int,
    device: torch.device,
    empty_cache: bool,
    bf16: bool,
    verbose_step: int = 1,
) -> tuple[bool, int, list[dict]]:
    """Run n_steps optimizer steps, printing detailed memory for step verbose_step.

    Returns (success, fail_step, per_step_stats).  fail_step is -1 on success.
    """
    autocast_dt = torch.bfloat16 if bf16 else torch.float16
    all_params  = [p for g in optimizer.param_groups for p in g["params"]]
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    stats: list[dict] = []

    def _mem() -> str:
        a = torch.cuda.memory_allocated(device) / 1e9
        free, tot = torch.cuda.mem_get_info(device)
        return f"torch={a:.2f}GB  cuda_total={(tot-free)/1e9:.2f}GB"

    # Same 4-step warm-up as _run_diagnostic: bnb materialises state on call #4.
    for _ in range(4):
        _wb = _make_batch(batch_size, device, 800)
        _wm, _wal, _wii, _wil, _wti, _wtl = _wb
        with torch.amp.autocast("cuda", dtype=autocast_dt):
            _we = encoder(_wm)
            _wa = adapter(_we)
            _wins, _wlbl = prepare_input(
                _wa, _wal, _wii, _wil, _wti, _wtl,
                llama.embed_tokens, sep_token_id=_SEP_ID,
            )
            _, _wloss = llama(_wins, _wlbl, audio_lengths=_wal)
        _backward(_wloss, 1, scaler)
        _optimizer_step(optimizer, scaler, all_params)
        optimizer.zero_grad(set_to_none=True)
        del _wb, _wm, _wal, _wii, _wil, _wti, _wtl, _we, _wa, _wins, _wlbl, _wloss
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)

    for step in range(1, n_steps + 1):
        for k in range(accum_steps):
            try:
                batch = _make_batch(batch_size, device)
                mel, audio_lengths, inst_ids, inst_lens, trans_ids, trans_lens = batch

                if step == verbose_step:
                    print(f"  step {step}  micro {k+1}/{accum_steps}  [before fwd]  {_mem()}")

                with torch.amp.autocast("cuda", dtype=autocast_dt):
                    enc_out     = encoder(mel)
                    adapter_out = adapter(enc_out)
                    inputs, labels = prepare_input(
                        adapter_out, audio_lengths,
                        inst_ids, inst_lens, trans_ids, trans_lens,
                        llama.embed_tokens, sep_token_id=_SEP_ID,
                    )
                    _, loss = llama(inputs, labels, audio_lengths=audio_lengths)

                if step == verbose_step:
                    print(f"  step {step}  micro {k+1}/{accum_steps}  [after  fwd]  {_mem()}")

                _backward(loss, accum_steps, scaler)

                if step == verbose_step:
                    print(f"  step {step}  micro {k+1}/{accum_steps}  [after  bwd]  {_mem()}")

                if empty_cache:
                    torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                return False, step, stats

        try:
            _optimizer_step(optimizer, scaler, all_params)
            optimizer.zero_grad(set_to_none=True)
            if empty_cache:
                torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect()
            return False, step, stats

        peak  = torch.cuda.max_memory_allocated(device) / 1e9
        alloc = torch.cuda.memory_allocated(device) / 1e9
        free, tot = torch.cuda.mem_get_info(device)
        cuda  = (tot - free) / 1e9
        torch.cuda.reset_peak_memory_stats(device)
        stats.append({"step": step, "peak_torch": peak, "alloc": alloc,
                      "cuda_total": cuda, "bnb_approx": cuda - alloc})

        line = (f"  step {step:3d}  torch={alloc:.2f}GB  cuda_total={cuda:.2f}GB  "
                f"untracked≈{cuda-alloc:.2f}GB  peak={peak:.2f}GB")
        print(line)

    return True, -1, stats


# ── Diagnostic run (A vs B) ───────────────────────────────────────────────────

def _run_diagnostic(
    encoder: WhisperEncoder,
    adapter: AudioAdapter,
    llama: Llama,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    batch_size: int,
    accum_steps: int,
    n_steps: int,
    device: torch.device,
    bf16: bool,
    var_seq_len: bool,
    seed: int,
) -> None:
    """Per-step memory table that distinguishes hypothesis A from hypothesis B.

    Columns
    ───────
    step      optimizer step number
    maxT      maximum mel time-dim (frames) seen across micro-batches this step
    toks/µ    transcript tokens per micro-batch (loss denominator)
    alloc     torch.cuda.memory_allocated  (GB)
    resv      torch.cuda.memory_reserved   (GB)
    used      CUDA total − CUDA free        (GB)
    external  used − resv  (cuDNN/cuBLAS workspaces + bnb if outside PyTorch pool)
    frag      resv − alloc (PyTorch cache: reserved but not live)
    optstate  sum of all tensors in optimizer.state  (GB)
    peak      torch peak this step (reset each step)
    Δext/Δopt/Δfrg  step-over-step delta of external / optstate / frag
    """
    autocast_dt = torch.bfloat16 if bf16 else torch.float16
    all_params  = [p for g in optimizer.param_groups for p in g["params"]]
    rng         = random.Random(seed)

    # Named parameters for the bnb lazy-init check
    named_params: list[tuple[str, torch.nn.Parameter]] = (
        list(encoder.named_parameters()) +
        [("adapter." + n, p) for n, p in adapter.named_parameters()] +
        [("llama." + n,   p) for n, p in llama.named_parameters()]
    )

    # ── Table header ──────────────────────────────────────────────────────────
    hdr = (f"{'step':>4}  {'maxT':>5}  {'toks/µ':>6}  "
           f"{'alloc':>6}  {'resv':>6}  {'used':>6}  "
           f"{'ext':>6}  {'frag':>6}  {'opt':>6}  {'peak':>6}  "
           f"{'Δext':>7}  {'Δopt':>7}  {'Δfrg':>7}")
    print()
    print(hdr)
    print("─" * len(hdr))

    prev: dict | None = None
    rows: list[dict]  = []
    oom_info: dict | None = None

    optimizer.zero_grad(set_to_none=True)

    # ── Warm-up: force bnb state materialisation before the timed loop ────────
    # bnb AdamW8bit defers GPU state allocation until the 4th optimizer.step()
    # call — verified empirically: steps 1–3 leave optimizer.state empty, step
    # 4 allocates ~14.7 GB of 8-bit exp_avg/exp_avg_sq tensors for all params.
    # With only 1 warm-up step the spike still hits at main step 3 (4th total).
    # Running 4 warm-up steps materialises state before the timed loop starts,
    # so the baseline is predictable and there is no mid-accumulation spike.
    # mel_t=800 (shortest realistic length) keeps each warm-up peak ~73 GB.
    _N_WARMUP = 4
    for _ in range(_N_WARMUP):
        _wb = _make_batch(batch_size, device, 800)
        _wm, _wal, _wii, _wil, _wti, _wtl = _wb
        with torch.amp.autocast("cuda", dtype=autocast_dt):
            _we = encoder(_wm)
            _wa = adapter(_we)
            _wins, _wlbl = prepare_input(
                _wa, _wal, _wii, _wil, _wti, _wtl,
                llama.embed_tokens, sep_token_id=_SEP_ID,
            )
            _, _wloss = llama(_wins, _wlbl, audio_lengths=_wal)
        _backward(_wloss, 1, scaler)
        _optimizer_step(optimizer, scaler, all_params)
        optimizer.zero_grad(set_to_none=True)
        del _wb, _wm, _wal, _wii, _wil, _wti, _wtl, _we, _wa, _wins, _wlbl, _wloss
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)

    _pre_state_gb = _opt_state_gb(optimizer)
    print(f"  [pre-warm: {_N_WARMUP} steps — bnb state = {_pre_state_gb:.3f} GB, "
          f"entries = {len(optimizer.state)}/{len(all_params)}]")

    for step in range(1, n_steps + 1):
        max_mel_t = 0
        torch.cuda.reset_peak_memory_stats(device)

        try:
            for k in range(accum_steps):
                mel_t = _sample_mel_t(rng) if var_seq_len else 3000
                max_mel_t = max(max_mel_t, mel_t)

                batch = _make_batch(batch_size, device, mel_t)
                mel, audio_lengths, inst_ids, inst_lens, trans_ids, trans_lens = batch

                with torch.amp.autocast("cuda", dtype=autocast_dt):
                    enc_out     = encoder(mel)
                    adapter_out = adapter(enc_out)
                    inputs, labels = prepare_input(
                        adapter_out, audio_lengths,
                        inst_ids, inst_lens, trans_ids, trans_lens,
                        llama.embed_tokens, sep_token_id=_SEP_ID,
                    )
                    _, loss = llama(inputs, labels, audio_lengths=audio_lengths)

                _backward(loss, accum_steps, scaler)

        except torch.cuda.OutOfMemoryError:
            oom_info = {"step": step, "max_mel_t": max_mel_t}
            torch.cuda.empty_cache()
            gc.collect()
            print(f"\n[OOM during accumulation at step {step}, max_mel_t={max_mel_t}]")
            try:
                print(torch.cuda.memory_summary(device, abbreviated=True))
            except Exception:
                pass
            break

        try:
            _optimizer_step(optimizer, scaler, all_params)
        except torch.cuda.OutOfMemoryError:
            oom_info = {"step": step, "max_mel_t": max_mel_t, "during": "opt_step"}
            torch.cuda.empty_cache()
            gc.collect()
            print(f"\n[OOM during optimizer.step() at step {step}]")
            try:
                print(torch.cuda.memory_summary(device, abbreviated=True))
            except Exception:
                pass
            break

        # ── Hypothesis B: bnb lazy-init check (once, after first step) ────────
        if step == 1:
            no_grad = [(name, p) for name, p in named_params
                       if p.requires_grad and p.grad is None]
            n_state = len(optimizer.state)
            n_total = len(all_params)
            print(f"\n  [bnb lazy-init @ step 1]  trainable={n_total}  "
                  f"grad=None: {len(no_grad)}  "
                  f"bnb state entries: {n_state}/{n_total}")
            if no_grad:
                sample = [name for name, _ in no_grad[:5]]
                print(f"    first no-grad params: {sample}")
            print()

        optimizer.zero_grad(set_to_none=True)

        # ── Metrics ───────────────────────────────────────────────────────────
        snap = _mem_snapshot(device)
        snap["optstate"] = _opt_state_gb(optimizer)
        snap["peak"]     = torch.cuda.max_memory_allocated(device) / 1e9
        torch.cuda.reset_peak_memory_stats(device)

        d_ext = snap["external"] - prev["external"] if prev else 0.0
        d_opt = snap["optstate"] - prev["optstate"] if prev else 0.0
        d_frg = snap["frag"]     - prev["frag"]     if prev else 0.0

        row = {**snap, "step": step, "max_mel_t": max_mel_t,
               "d_ext": d_ext, "d_opt": d_opt, "d_frg": d_frg}
        rows.append(row)
        prev = snap

        print(
            f"{step:4d}  {max_mel_t:5d}  {batch_size*_TRANS_TOKS:6d}  "
            f"{snap['alloc']:6.2f}  {snap['resv']:6.2f}  {snap['used']:6.2f}  "
            f"{snap['external']:6.3f}  {snap['frag']:6.3f}  {snap['optstate']:6.3f}  "
            f"{snap['peak']:6.2f}  "
            f"{d_ext:+7.3f}  {d_opt:+7.3f}  {d_frg:+7.3f}"
        )

    # ── Verdict ───────────────────────────────────────────────────────────────
    if len(rows) < 2:
        return

    total_d_ext = sum(r["d_ext"] for r in rows[1:])
    total_d_opt = sum(r["d_opt"] for r in rows[1:])
    total_d_frg = sum(abs(r["d_frg"]) for r in rows[1:])

    candidates = [
        ("external", total_d_ext, "→ hypothesis A: cuDNN/cuBLAS workspace autotuning"),
        ("optstate", total_d_opt, "→ hypothesis B: bnb optimizer state lazy init"),
        ("frag",     total_d_frg, "→ allocator fragmentation"),
    ]
    winner = max(candidates, key=lambda x: x[1])

    print()
    print("═" * 58)
    print("VERDICT")
    print("═" * 58)
    print(f"  Δexternal total: {total_d_ext:+.3f} GB   "
          f"(cuDNN/cuBLAS workspaces outside PyTorch pool)")
    print(f"  Δoptstate total: {total_d_opt:+.3f} GB   "
          f"(bnb state tensors inside optimizer.state)")
    print(f"  Δfrag     total: {total_d_frg:+.3f} GB   "
          f"(PyTorch reserved cache oscillation)")
    print()
    print(f"  Dominant growth: {winner[0]}  (+{winner[1]:.3f} GB  {winner[2]})")
    if oom_info:
        print(f"  OOM at step {oom_info['step']}  max_mel_t={oom_info['max_mel_t']}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # Apply cudnn.benchmark before any CUDA ops.
    cudnn_bm = args.cudnn_benchmark == "true"
    torch.backends.cudnn.benchmark = cudnn_bm

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.", file=sys.stderr)
        sys.exit(1)

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "(not set)")

    print(f"GPU: {torch.cuda.get_device_name(0)}  ({total_gb:.1f} GB)")
    print(f"PYTORCH_CUDA_ALLOC_CONF: {alloc_conf}")
    print(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    print(f"dtype={'bfloat16' if args.bf16 else 'fp32'}  "
          f"batch_size={args.batch_size}  "
          f"empty_cache={args.empty_cache}")
    print()

    encoder, adapter, llama, optimizer, scaler = _build_models(
        args.whisper_ckpt, args.llama_ckpt, device, args.bf16,
    )

    free, tot = torch.cuda.mem_get_info(device)
    print(f"After load:  torch={torch.cuda.memory_allocated(device)/1e9:.2f}GB  "
          f"cuda_total={(tot-free)/1e9:.2f}GB")
    print()

    # ── Diagnostic mode ───────────────────────────────────────────────────────
    if args.diag:
        accum = args.accum if args.accum is not None else args.max_accum
        n_steps = 12  # web Claude's recommendation; enough to see growth pattern
        var_seq = True  # always vary mel lengths in diagnostic mode
        print(f"DIAGNOSTIC MODE  accum_steps={accum}  effective_batch={args.batch_size*accum}  "
              f"n_steps={n_steps}  var_seq_len={var_seq}")
        _run_diagnostic(
            encoder, adapter, llama, optimizer, scaler,
            batch_size=args.batch_size,
            accum_steps=accum,
            n_steps=n_steps,
            device=device,
            bf16=args.bf16,
            var_seq_len=var_seq,
            seed=args.seed,
        )
        return

    # ── Simple OOM sweep ──────────────────────────────────────────────────────
    accum_values = ([args.accum] if args.accum is not None
                    else list(range(1, args.max_accum + 1)))

    results: list[tuple[int, bool, int, list[dict]]] = []
    for accum in accum_values:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        gc.collect()
        print(f"─── accum_steps={accum}  effective_batch={args.batch_size * accum} ───")

        ok, fail_step, step_stats = _run_accum(
            encoder, adapter, llama, optimizer, scaler,
            batch_size=args.batch_size,
            accum_steps=accum,
            n_steps=args.n_steps,
            device=device,
            empty_cache=args.empty_cache,
            bf16=args.bf16,
        )

        if ok:
            print(f"  → OK  all {args.n_steps} steps passed")
            if len(step_stats) > 1:
                d_bnb  = step_stats[-1]["bnb_approx"] - step_stats[0]["bnb_approx"]
                d_peak = step_stats[-1]["peak_torch"]  - step_stats[0]["peak_torch"]
                if abs(d_bnb) > 0.05 or abs(d_peak) > 0.05:
                    print(f"  untracked Δ step 1→{args.n_steps}: {d_bnb:+.2f} GB  "
                          f"peak Δ: {d_peak:+.2f} GB")
        else:
            print(f"  → OOM at step {fail_step}")
            if step_stats:
                s1 = step_stats[0]["bnb_approx"]
                sl = step_stats[-1]["bnb_approx"]
                print(f"  untracked: {s1:.2f} GB (step 1) → "
                      f"{sl:.2f} GB (step {len(step_stats)})  Δ={sl-s1:+.2f} GB")
        print()
        results.append((accum, ok, fail_step, step_stats))

    # Summary
    print("═" * 52)
    print(f"SUMMARY  dtype={'bfloat16' if args.bf16 else 'fp32'}")
    print("═" * 52)
    print(f"  {'accum':>6}  {'eff_bs':>7}  {'peak_step1':>12}  result")
    print("  " + "─" * 44)
    for accum, ok, fail_step, step_stats in results:
        peak1  = f"{step_stats[0]['peak_torch']:.1f} GB" if step_stats else "—"
        status = "OK" if ok else f"OOM at step {fail_step}"
        print(f"  {accum:6d}  {args.batch_size*accum:7d}  {peak1:>12}  {status}")

    any_oom = any(not ok for _, ok, _, _ in results)
    if any_oom and not args.bf16:
        print()
        print("Run with --bf16 to test the bfloat16 fix.")
        print("Run with --bf16 --diag to diagnose A vs B for the bf16 path.")


if __name__ == "__main__":
    main()

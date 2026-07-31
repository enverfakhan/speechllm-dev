"""Config-driven multi-stage training loop for speech-llm.

Replaces train.py with a declarative, stage-based design driven by YAML configs.
A single-stage stub run on the same seed and data produces identical loss
trajectories to train.py.

Usage:
    python training.py --config configs/example.yaml [--stub] [--resume path/to/ckpt.pt]
    python training.py --config configs/example.yaml --wandb --run-name my-run
"""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from build import build_models
from data import build_dataloader, list_shards, PrunedTokenizer, INSTRUCTION_VARIANTS
from model.adapter import AudioAdapter
from model.llama import Llama
from model.sequence import prepare_input
from model.whisper_encoder import WhisperEncoder
from metrics import MetricCollector, render
from stages import Stage, StageContext
from utils.checkpoint import (
    apply_full_checkpoint,
    read_checkpoint,
    ResumeState,
    save_adapter_checkpoint,
    save_checkpoint,
)
from utils.config import Config, load_config
from utils.evaluate import compute_wer
from utils.generate import greedy_generate


# ── Constants ─────────────────────────────────────────────────────────────────

_INSTRUCTION_PAIRS: list[tuple[str, str]] = [
    (INSTRUCTION_VARIANTS[0], "unformatted.txt"),
    (INSTRUCTION_VARIANTS[1], "formatted.txt"),
]

_EMA_ALPHA = 0.98


# ── Run state ─────────────────────────────────────────────────────────────────

@dataclass
class RunState:
    """Mutable counters threaded through the stage loop.

    global_step persists across stage boundaries; step_in_stage and epoch
    are reset at the start of each new (non-resumed) stage.
    """
    global_step:   int = 0   # run-global, persists across stages
    step_in_stage: int = 0   # per-stage
    epoch:         int = 0   # per-stage

    @classmethod
    def fresh(cls) -> "RunState":
        return cls()

    @classmethod
    def from_resume(cls, rs: "ResumeState") -> "RunState":
        return cls(global_step=rs.step, step_in_stage=rs.step_in_stage, epoch=rs.epoch)

    def enter_new_stage(self) -> None:
        self.epoch = 0
        self.step_in_stage = 0


# ── Helpers ───────────────────────────────────────────────────────────────────

def run_eval_pass(
    encoder:      WhisperEncoder,
    adapter:      AudioAdapter,
    llama:        Llama,
    diag_loader:  torch.utils.data.DataLoader,
    diag_iter:    Any,
    metrics:      MetricCollector,
    sep_token_id: int,
    device:       torch.device,
    n_batches:    int,
    global_step:  int,
) -> tuple[dict, list, Any]:
    """Run n_batches of teacher-forced eval on the diag shard.

    Returns (eval_metrics, retained_batches, updated_diag_iter).
    Retained batches are 6-tuples of device tensors, reused by maybe_run_wer.
    """
    encoder.eval()
    adapter.eval()
    llama.eval()

    retained: list[tuple] = []

    for _ in range(n_batches):
        try:
            batch = next(diag_iter)
        except StopIteration:
            diag_iter = iter(diag_loader)
            batch = next(diag_iter)

        (d_mel, d_audio_len,
         d_inst_ids, d_inst_lens,
         d_trans_ids, d_trans_lens) = [t.to(device) for t in batch]

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            d_enc  = encoder(d_mel)
            d_ada  = adapter(d_enc)
            d_inp, d_lbl = prepare_input(
                d_ada, d_audio_len,
                d_inst_ids, d_inst_lens,
                d_trans_ids, d_trans_lens,
                llama.embed_tokens, sep_token_id,
            )
            d_logits, d_loss = llama(d_inp, d_lbl, audio_lengths=d_audio_len)

        metrics.observe("eval", global_step, logits=d_logits, labels=d_lbl, loss=d_loss.detach())
        retained.append((d_mel, d_audio_len, d_inst_ids, d_inst_lens, d_trans_ids, d_trans_lens))

    encoder.train()
    adapter.train()
    llama.train()

    eval_metrics = metrics.flush("eval", global_step)
    return eval_metrics, retained, diag_iter


def maybe_run_wer(
    cfg:          Config,
    eval_metrics: dict,
    global_step:  int,
    eval_subset:  list[tuple],
    encoder:      WhisperEncoder,
    adapter:      AudioAdapter,
    llama:        Llama,
    tokenizer:    PrunedTokenizer,
    sep_token_id: int,
    device:       torch.device,
    use_wandb:    bool,
) -> dict:
    """Optionally run greedy WER decode on retained eval batches.

    Returns {} when conditions are not met, {"wer/diag": float, ...} otherwise.
    Readiness gate: only run when eval_first_token is below the configured threshold.
    """
    wer_cfg = cfg.metrics.wer
    if not wer_cfg.enabled:
        return {}
    if global_step % wer_cfg.period != 0:
        return {}
    ftl = eval_metrics.get("loss/eval_first_token")
    if ftl is None or ftl >= wer_cfg.readiness_first_token_below:
        return {}

    encoder.eval()
    adapter.eval()
    llama.eval()

    all_refs: list[str] = []
    all_hyps: list[str] = []
    sample_pairs: list[tuple[str, str]] = []

    for batch in eval_subset[: wer_cfg.max_batches]:
        (mel, audio_lengths, inst_ids, inst_lens, trans_ids, trans_lens) = batch

        gen_ids = greedy_generate(
            encoder, adapter, llama,
            mel, audio_lengths, inst_ids, inst_lens,
            sep_token_id,
        )
        for i in range(len(gen_ids)):
            hyp = tokenizer.decode(gen_ids[i])
            ref = tokenizer.decode(trans_ids[i, : int(trans_lens[i].item())].tolist())
            all_refs.append(ref)
            all_hyps.append(hyp)
            sample_pairs.append((ref, hyp))

    encoder.train()
    adapter.train()
    llama.train()

    wer_val = compute_wer(all_refs, all_hyps)
    print(f"  WER/diag step {global_step}: {wer_val:.1%}  ({len(all_hyps)} samples)")

    n_show = min(wer_cfg.sample_transcriptions, len(sample_pairs))
    for ref, hyp in sample_pairs[:n_show]:
        print(f"    REF: {ref[:80]}")
        print(f"    HYP: {hyp[:80]}")

    out: dict = {"wer/diag": wer_val}

    if use_wandb:
        import wandb
        rows = sample_pairs[: wer_cfg.sample_transcriptions]
        tbl  = wandb.Table(columns=["step", "reference", "hypothesis"])
        for ref, hyp in rows:
            tbl.add_data(global_step, ref, hyp)
        out["wer/diag_transcriptions"] = tbl

    return out


# ── Per-stage loop ────────────────────────────────────────────────────────────

def run_stage(
    encoder:      WhisperEncoder,
    adapter:      AudioAdapter,
    llama:        Llama,
    optimizer:    torch.optim.Optimizer,
    scheduler:    torch.optim.lr_scheduler.LRScheduler,
    scaler:       torch.amp.GradScaler,
    metrics:      MetricCollector,
    tokenizer:    PrunedTokenizer,
    diag_loader:  torch.utils.data.DataLoader | None,
    diag_iter:    Any,
    stage:        Stage,
    stage_idx:    int,
    run:          RunState,
    cfg:          Config,
    device:       torch.device,
    sep_token_id: int,
    use_wandb:    bool,
    ckpt_dir:     Path,
    baselines_data: dict,
    train_start:  float,
    modules_dirty: set[str],
) -> bool:
    """Run one stage to completion (or until the global step cap is hit).

    Returns True iff training should stop entirely (cfg.run.max_steps reached);
    False means this stage advanced and the caller should move to the next one.

    modules_dirty is the run-level set of heavy modules ("encoder"/"llama") that
    diverge from pretrained and must be written into every checkpoint (delta
    invariant — see utils/checkpoint.save_checkpoint).
    """
    optimizer.zero_grad()
    accum_loss  = 0.0
    loss_ema: float | None = None
    micro_acc   = 0   # accum-boundary counter; never reset across epochs in this stage
    advanced    = False

    step_start    = time.perf_counter()
    step_audio_s  = 0.0
    total_audio_s = 0.0

    max_steps = cfg.run.max_steps

    # ── Epoch loop ────────────────────────────────────────────────────────────
    while not advanced and (max_steps is None or run.global_step < max_steps):
        micro_step_in_epoch = 0
        loader = stage.make_loader(run.epoch, stage_idx)

        # ── Batch loop ────────────────────────────────────────────────────────
        for batch in loader:
            micro_step_in_epoch += 1

            (mel, audio_lengths,
             instruction_ids, instruction_lengths,
             transcript_ids, transcript_lengths) = [t.to(device) for t in batch]

            # audio_lengths[i] = adapter tokens; each = 8 mel frames @ 10 ms
            step_audio_s += audio_lengths.sum().item() * 8 * 0.01

            with torch.amp.autocast("cuda", dtype=torch.float16):
                enc_out     = encoder(mel)
                adapter_out = adapter(enc_out)
                inputs, labels = prepare_input(
                    adapter_out,
                    audio_lengths,
                    instruction_ids,
                    instruction_lengths,
                    transcript_ids,
                    transcript_lengths,
                    llama.embed_tokens,
                    sep_token_id,
                )
                logits, loss = llama(inputs, labels, audio_lengths=audio_lengths)

            metrics.observe("train", run.global_step, logits=logits, labels=labels)
            scaler.scale(loss / stage.accum_steps).backward()
            accum_loss += loss.item()
            micro_acc  += 1

            if micro_acc % stage.accum_steps != 0:
                continue  # accumulate more micro-steps before stepping

            # ── Optimizer step ────────────────────────────────────────────────
            scaler.unscale_(optimizer)
            if cfg.optim.grad_clip.enabled:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g["params"]],
                    cfg.optim.grad_clip.max_norm,
                )
            # Observe gradients (after unscale, before step)
            metrics.observe(
                "train", run.global_step,
                encoder = encoder,
                adapter = adapter,
                llama   = llama,
            )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            run.global_step   += 1
            run.step_in_stage += 1

            avg_loss   = accum_loss / stage.accum_steps
            accum_loss = 0.0
            loss_ema   = (
                avg_loss if loss_ema is None
                else _EMA_ALPHA * loss_ema + (1 - _EMA_ALPHA) * avg_loss
            )

            # ── Periodic checkpoint ───────────────────────────────────────────
            if cfg.checkpoint.save_every and run.global_step % cfg.checkpoint.save_every == 0:
                _save_checkpoint(
                    ckpt_dir, stage, stage_idx,
                    run.global_step, run.epoch, micro_step_in_epoch, run.step_in_stage,
                    encoder, adapter, llama, optimizer, scaler, scheduler,
                    modules_dirty, cfg,
                )

            # ── Eval pass ─────────────────────────────────────────────────────
            eval_metrics: dict = {}
            eval_subset:  list = []
            wer_metrics:  dict = {}

            if diag_loader is not None and run.global_step % cfg.metrics.eval_every == 0:
                eval_metrics, eval_subset, diag_iter = run_eval_pass(
                    encoder, adapter, llama,
                    diag_loader, diag_iter,
                    metrics, sep_token_id, device,
                    cfg.metrics.eval_batches,
                    run.global_step,
                )
                wer_metrics = maybe_run_wer(
                    cfg, eval_metrics, run.global_step, eval_subset,
                    encoder, adapter, llama, tokenizer,
                    sep_token_id, device, use_wandb,
                )

            # ── Train metrics flush ───────────────────────────────────────────
            train_metrics = metrics.flush("train", run.global_step)

            # ── Throughput (reset after checkpoint I/O folds in) ──────────────
            t_now          = time.perf_counter()
            elapsed        = max(t_now - step_start, 1e-9)
            throughput     = step_audio_s / elapsed
            total_audio_s += step_audio_s
            step_audio_s   = 0.0
            step_start     = t_now

            # ── Console output ────────────────────────────────────────────────
            print(
                f"step {run.global_step:6d}  loss {avg_loss:.4f}"
                f"  ema {loss_ema:.4f}"
                f"  {throughput:.2f}× realtime"
                f"  stage={stage.name}"
                f"  epoch={run.epoch}"
            )
            if train_metrics:
                render(train_metrics, "train", run.global_step)
            if eval_metrics:
                render(eval_metrics, "eval", run.global_step)

            # ── W&B ───────────────────────────────────────────────────────────
            if use_wandb:
                # Per-layer gate diagnostic: which depths open their gates, and
                # how fast.  Only when the gated adapters exist AND this stage
                # trains them (cheap: n_layers-1 scalars).
                gate_metrics = (
                    llama.audio_gate_values()
                    if (cfg.model.audio_adapter_r > 0
                        and "audio_adapters" in set(stage.trainable))
                    else {}
                )
                _log_wandb(
                    run.global_step, stage_idx, stage, run.epoch,
                    avg_loss, loss_ema,
                    optimizer, throughput, total_audio_s, t_now, train_start,
                    train_metrics, eval_metrics, wer_metrics,
                    baselines_data, gate_metrics,
                )

            # ── Stage advance check ───────────────────────────────────────────
            if stage.should_advance(eval_metrics, run.step_in_stage):
                print(
                    f"[stage] '{stage.name}' advance criterion met "
                    f"at step {run.global_step} (step_in_stage={run.step_in_stage})."
                )
                advanced = True

            # ── Global exit check ──────────────────────────────────────────────
            if (max_steps is not None and run.global_step >= max_steps) or advanced:
                break

        # end batch loop

        if advanced:
            # Stage handoff: save full checkpoint + adapter sidecar
            _save_checkpoint(
                ckpt_dir, stage, stage_idx,
                run.global_step, run.epoch, micro_step_in_epoch, run.step_in_stage,
                encoder, adapter, llama, optimizer, scaler, scheduler,
                modules_dirty, cfg,
                suffix="stage-handoff",
            )
            run.epoch = 0
            break

        run.epoch += 1
    # end epoch loop

    return not advanced


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    """Load config and run the multi-stage training loop."""
    cfg = load_config(argv=argv)

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Pruned vocab ──────────────────────────────────────────────────────────
    with (cfg.data.tokenizer / "pruned_config.json").open() as f:
        _pc         = json.load(f)
    sep_token_id = _pc["sep_token_id"]
    print(f"Pruned vocab: {_pc['vocab_size']} tokens  SEP id: {sep_token_id}")

    # ── Baselines (for W&B reference lines) ───────────────────────────────────
    baseline_path = Path("baselines.json")
    baselines_data: dict = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}

    # ── Models ────────────────────────────────────────────────────────────────
    encoder, adapter, llama, init_from_loaded = build_models(cfg, device)

    # modules_dirty: the run-level set of heavy modules that DIFFER from the
    # pretrained base and therefore must be written into every checkpoint (delta
    # invariant — see utils/checkpoint.save_checkpoint).  It lives here as run
    # provenance (next to init_from), not on RunState, because it is not a
    # per-stage counter; on resume it is restored explicitly from the checkpoint.
    # Seed: encoder/llama warm-started from init_from already diverge from
    # pretrained.  (adapter/audio_adapters are always saved, so ignore them here.)
    modules_dirty: set[str] = {m for m in init_from_loaded if m in ("encoder", "llama")}

    # ── Persistent scaler (shared across all stages) ──────────────────────────
    scaler = torch.amp.GradScaler("cuda")

    # ── Tokenizer + MetricCollector ───────────────────────────────────────────
    tokenizer = PrunedTokenizer(cfg.data.tokenizer)
    metrics   = MetricCollector(
        tokenizer    = tokenizer,
        sep_token_id = sep_token_id,
        log_every    = cfg.metrics.eval_every,
        top_k        = 5,
    )

    # ── Shard list ────────────────────────────────────────────────────────────
    if cfg.data.shards_file is not None:
        lines     = cfg.data.shards_file.read_text().splitlines()
        all_shards = [ln.strip() for ln in lines if ln.strip()]
    else:
        all_shards = list_shards(cfg.data.shards)
    if not all_shards:
        raise FileNotFoundError("No shards found; check data.shards_file or data.shards.")
    print(f"Training on {len(all_shards)} shards.")

    # ── Instruction pairs for StageContext ────────────────────────────────────
    run_mode = cfg.run.instruction_mode
    if run_mode == "unformatted":
        diag_instruction_pairs = [_INSTRUCTION_PAIRS[0]]
    elif run_mode == "formatted":
        diag_instruction_pairs = [_INSTRUCTION_PAIRS[1]]
    else:
        diag_instruction_pairs = list(_INSTRUCTION_PAIRS)

    # ── Diagnostic shard dataloader ───────────────────────────────────────────
    _diag_loader: torch.utils.data.DataLoader | None = None
    _diag_iter:   Any                                = None

    if cfg.data.diag_shard is not None:
        if not cfg.data.diag_shard.exists():
            raise FileNotFoundError(f"data.diag_shard not found: {cfg.data.diag_shard}")
        _diag_loader = build_dataloader(
            [str(cfg.data.diag_shard)],
            tokenizer_path        = cfg.data.tokenizer,
            sep_token_id          = sep_token_id,
            batch_size            = cfg.metrics.eval_batch_size,
            num_workers           = 0,
            instruction_variants  = diag_instruction_pairs,
            shuffle_buffer        = 100,
            partial               = True,
        )
        _diag_iter = iter(_diag_loader)
        print(f"Diagnostic shard: {cfg.data.diag_shard}")

    # ── StageContext + Stage objects ──────────────────────────────────────────
    ctx = StageContext(
        shards               = all_shards,
        tokenizer_path       = cfg.data.tokenizer,
        sep_token_id         = sep_token_id,
        num_workers          = cfg.data.num_workers,
        seed                 = cfg.seed,
        instruction_pairs    = list(_INSTRUCTION_PAIRS),
        run_instruction_mode = cfg.run.instruction_mode,
        betas                = cfg.optim.betas,
        weight_decay         = cfg.optim.weight_decay,
        shuffle_buffer       = 1000,
    )
    stages = [Stage(sc, ctx) for sc in cfg.stages]

    # ── W&B ───────────────────────────────────────────────────────────────────
    use_wandb = cfg.logging.wandb
    if use_wandb:
        import os
        import wandb as _wandb
        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "WANDB_API_KEY is not set. "
                "Add 'export WANDB_API_KEY=...' to ~/.bashrc and reload your shell."
            )
        _wandb.init(
            project = cfg.logging.project,
            name    = cfg.logging.run_name,
            config  = {
                "seed":                    cfg.seed,
                "n_stages":                len(cfg.stages),
                "instruction_mode":        cfg.run.instruction_mode,
                "stub":                    cfg.model.stub,
                "gradient_checkpointing":  cfg.model.gradient_checkpointing,
                "resume":                  str(cfg.resume) if cfg.resume else None,
                "init_from":               str(cfg.model.init_from) if cfg.model.init_from else None,
            },
        )

    # ── Checkpoint directory ──────────────────────────────────────────────────
    ckpt_dir = cfg.checkpoint.dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_start = time.perf_counter()

    # ── Stage loop ────────────────────────────────────────────────────────────
    run = RunState.fresh()
    start_stage = 0
    ckpt: dict | None = None
    if cfg.resume is not None:
        ckpt = read_checkpoint(cfg.resume)
        ckpt_kind = ckpt.get("kind", "periodic")  # untagged legacy checkpoints → resumable
        if ckpt_kind == "handoff":
            raise ValueError(
                f"{cfg.resume} is a handoff/weights checkpoint (kind='handoff'), not a "
                f"full-state resume point. Use model.init_from to warm-start weights from "
                f"it; use --resume only with a periodic checkpoint."
            )
        start_stage = ckpt["stage_index"]
        if not (0 <= start_stage < len(stages)):
            raise ValueError(
                f"resume checkpoint stage_index={start_stage} out of range "
                f"for {len(stages)} stages"
            )
        print(f"Resuming from {cfg.resume}: stage_index={start_stage}")

    for stage_idx in range(start_stage, len(stages)):
        stage = stages[stage_idx]
        optimizer, scheduler = stage.setup(encoder, adapter, llama)

        if stage_idx == start_stage and ckpt is not None:
            rs = apply_full_checkpoint(
                ckpt, encoder=encoder, adapter=adapter, llama=llama,
                optimizer=optimizer, scaler=scaler, scheduler=scheduler,
                current_init_from=(
                    str(cfg.model.init_from) if cfg.model.init_from is not None else None
                ),
            )
            run = RunState.from_resume(rs)
            # Restore the accumulated dirty set so the resumed run keeps saving
            # every module the original run had already dirtied.
            modules_dirty |= set(rs.modules_dirty)
            del ckpt  # free the CPU-side checkpoint dict
            ckpt = None
            print(
                f"  Restored: global_step={run.global_step} "
                f"step_in_stage={run.step_in_stage} epoch={run.epoch} "
                f"modules_dirty={sorted(modules_dirty)}"
            )
        else:
            run.enter_new_stage()

        # Every stage (fresh or resumed) dirties the heavy modules it trains, and
        # dirtiness persists: a run that finetunes llama in stage 0 then freezes
        # it in stage 1 must still save llama in stage-1 checkpoints.
        modules_dirty |= set(stage.trainable) & {"encoder", "llama"}

        stopped = run_stage(
            encoder, adapter, llama,
            optimizer, scheduler, scaler,
            metrics, tokenizer,
            _diag_loader, _diag_iter,
            stage, stage_idx, run,
            cfg, device, sep_token_id, use_wandb,
            ckpt_dir, baselines_data, train_start,
            modules_dirty,
        )
        if stopped:
            break
    # end stage loop

    # ── Final checkpoint (same delta gating as periodic saves) ────────────────
    _final = ckpt_dir / f"final-step{run.global_step}.pt"
    _final_init_from = str(cfg.model.init_from) if cfg.model.init_from is not None else None
    save_checkpoint(
        _final,
        step                = run.global_step,
        epoch               = run.epoch,
        micro_step_in_epoch = 0,
        step_in_stage       = run.step_in_stage,
        stage_index         = stage_idx,
        batch_size          = stages[stage_idx].batch_size,
        adapter             = adapter,
        optimizer           = optimizer,
        scaler              = scaler,
        scheduler           = scheduler,
        encoder             = encoder if "encoder" in modules_dirty else None,
        llama               = llama   if "llama"   in modules_dirty else None,
        audio_adapters_from = llama,
        modules_dirty       = modules_dirty,
        init_from           = _final_init_from,
        kind                = "periodic",
    )
    print(f"Final checkpoint saved → {_final}")

    _final_ada = ckpt_dir / f"final-adapter-step{run.global_step}.pt"
    save_adapter_checkpoint(
        _final_ada,
        step                = run.global_step,
        epoch               = run.epoch,
        micro_step_in_epoch = 0,
        step_in_stage       = run.step_in_stage,
        stage_index         = stage_idx,
        batch_size          = stages[stage_idx].batch_size,
        adapter             = adapter,
        optimizer           = optimizer,
        llama               = llama,   # rides audio-adapter tensors along, if any
    )
    print(f"Final adapter checkpoint saved → {_final_ada}")

    if use_wandb:
        import wandb as _wandb
        _wandb.finish()


# ── Save helper ───────────────────────────────────────────────────────────────

def _save_checkpoint(
    ckpt_dir:     Path,
    stage:        Stage,
    stage_idx:    int,
    global_step:  int,
    epoch:        int,
    micro_step_in_epoch: int,
    step_in_stage: int,
    encoder:      nn.Module,
    adapter:      nn.Module,
    llama:        nn.Module,
    optimizer:    torch.optim.Optimizer,
    scaler:       torch.amp.GradScaler,
    scheduler:    torch.optim.lr_scheduler.LRScheduler,
    modules_dirty: set[str],
    cfg:          Config,
    suffix:       str = "",
) -> None:
    """Save full checkpoint + optional adapter sidecar for adapter-only stages.

    The full checkpoint is a DELTA over pretrained (see utils/checkpoint): the
    heavy encoder/llama states are written only when in modules_dirty; audio
    adapters are always harvested from llama when full llama is skipped.
    """
    name = f"step{global_step:07d}"
    if suffix:
        name += f"-{suffix}"
    ckpt_path = ckpt_dir / f"{name}.pt"
    init_from = str(cfg.model.init_from) if cfg.model.init_from is not None else None
    save_checkpoint(
        ckpt_path,
        step                = global_step,
        epoch               = epoch,
        micro_step_in_epoch = micro_step_in_epoch,
        step_in_stage       = step_in_stage,
        stage_index         = stage_idx,
        batch_size          = stage.batch_size,
        adapter             = adapter,
        optimizer           = optimizer,
        scaler              = scaler,
        scheduler           = scheduler,
        # Delta gating: write the big frozen states only when they diverge from
        # pretrained; always harvest the small audio-adapter delta from llama.
        encoder             = encoder if "encoder" in modules_dirty else None,
        llama               = llama   if "llama"   in modules_dirty else None,
        audio_adapters_from = llama,
        modules_dirty       = modules_dirty,
        init_from           = init_from,
        kind                = "handoff" if suffix == "stage-handoff" else "periodic",
    )
    print(f"Checkpoint saved → {ckpt_path}")

    # Adapter sidecar for cross-stage loading: always at stage handoff,
    # only for adapter-only stages during periodic saves
    is_adapter_only = set(stage.trainable) == {"adapter"}
    if suffix == "stage-handoff" or is_adapter_only:
        ada_path = ckpt_dir / f"{name}-adapter.pt"
        save_adapter_checkpoint(
            ada_path,
            step                = global_step,
            epoch               = epoch,
            micro_step_in_epoch = micro_step_in_epoch,
            step_in_stage       = step_in_stage,
            stage_index         = stage_idx,
            batch_size          = stage.batch_size,
            adapter             = adapter,
            optimizer           = optimizer,
            llama               = llama,   # rides audio-adapter tensors along, if any
        )
        print(f"Adapter checkpoint saved → {ada_path}")


# ── W&B logging helper ────────────────────────────────────────────────────────

def _log_wandb(
    global_step:   int,
    stage_idx:     int,
    stage:         Stage,
    epoch:         int,
    avg_loss:      float,
    loss_ema:      float,
    optimizer:     torch.optim.Optimizer,
    throughput:    float,
    total_audio_s: float,
    t_now:         float,
    train_start:   float,
    train_metrics: dict,
    eval_metrics:  dict,
    wer_metrics:   dict,
    baselines_data: dict,
    gate_metrics:  dict | None = None,
) -> None:
    import wandb as _wandb

    # Per-group LR using group["name"] — no positional index assumptions
    lr_dict = {
        f"train/lr_{g['name']}": g["lr"]
        for g in optimizer.param_groups
    }

    # Per-module trainable flags
    trainable_set = set(stage.trainable)
    trainable_flags = {
        "train/encoder_trainable": float("encoder" in trainable_set),
        "train/adapter_trainable": float("adapter" in trainable_set),
        "train/llama_trainable":   float("llama"   in trainable_set),
    }

    # Train/eval gap metrics
    gap: dict = {}
    _tr = train_metrics.get("loss/train_rest")
    _er = eval_metrics.get("loss/eval_rest")
    _tf = train_metrics.get("loss/train_first_token")
    _ef = eval_metrics.get("loss/eval_first_token")
    if _tr is not None and _er is not None:
        gap["loss/gap_rest"] = _er - _tr
    if _tf is not None and _ef is not None:
        gap["loss/gap_first_token"] = _ef - _tf

    # Numeric baseline values only (skip string notes and keys starting with "_")
    baseline_payload = {
        f"baseline/{k}": v
        for k, v in baselines_data.items()
        if isinstance(v, (int, float)) and not k.startswith("_")
    }

    payload = {
        "train/loss":       avg_loss,
        "train/loss_ema":   loss_ema,
        "train/stage_index": stage_idx,
        "train/stage_name":  stage.name,
        "train/epoch":       epoch,
        **trainable_flags,
        **lr_dict,
        "runtime/throughput_audio_sec_per_sec": throughput,
        "runtime/cumulative_audio_hours":       total_audio_s / 3600,
        "runtime/wall_time_min":               (t_now - train_start) / 60,
        **train_metrics,
        **eval_metrics,
        **wer_metrics,
        **gap,
        **baseline_payload,
        **(gate_metrics or {}),
    }

    _wandb.log(payload, step=global_step)


if __name__ == "__main__":
    main(argv=sys.argv[1:])

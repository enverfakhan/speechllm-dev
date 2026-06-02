"""Single-GPU training loop for speech-llm.

Trains WhisperEncoder + AudioAdapter + Llama jointly with differential learning
rates (encoder 1e-5, adapter and LLM 1e-4). Supports gradient accumulation,
mixed precision (torch.amp), periodic checkpointing, and optional W&B logging.

Usage:
    python train.py \\
      --shards_file  data/subset_shards.txt \\
      --tokenizer    data/pruned_tokenizer/ \\
      --whisper_ckpt weights/whisper_small.pt \\
      --llama_ckpt   /home/goivagoi/.llama/checkpoints/Llama3.1-8B/ \\
      --batch_size   4 \\
      --accum_steps  8 \\
      --max_steps    100 \\
      --wandb
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from data import build_dataloader, build_eval_dataloader, list_shards, PrunedTokenizer
from model.adapter import AudioAdapter, EvalPrefixBatch, prepare_input
from model.llama import Llama, LlamaConfig
from model.whisper_encoder import WhisperEncoder
from diagnostics import Diagnostics


INSTRUCTION_VARIANTS = [
    "Transcribe the following audio without formatting.",
    "Transcribe the following audio with proper formatting.",
]

# (instruction_text, transcript_key) pairs consumed by build_dataloader.
# build_eval_dataloader always evaluates both variants regardless of training mode.
_INSTRUCTION_PAIRS: list[tuple[str, str]] = [
    (INSTRUCTION_VARIANTS[0], "unformatted.txt"),
    (INSTRUCTION_VARIANTS[1], "formatted.txt"),
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SpeechLLM on LibriSpeech shards.")

    shard_group = p.add_mutually_exclusive_group(required=True)
    shard_group.add_argument(
        "--shards_file", type=Path,
        help="Text file listing shard paths, one per line (e.g. data/subset_shards.txt).",
    )
    shard_group.add_argument(
        "--shards", type=str,
        help="Brace/glob pattern for shards (e.g. 'data/shards/train-{000000..000200}.tar').",
    )

    p.add_argument("--tokenizer",    type=Path, required=True,
                   help="Path to pruned tokenizer directory (build_vocab.py output).")
    p.add_argument("--whisper_ckpt", type=Path, default=None,
                   help="Path to whisper_small.pt checkpoint. Omit with --stub.")
    p.add_argument("--llama_ckpt",   type=Path, default=None,
                   help="Path to Llama 3.1 8B checkpoint directory. Omit with --stub.")

    p.add_argument(
        "--stub", action="store_true",
        help=(
            "Use a tiny randomly-initialised model (no pretrained weights) to "
            "validate the training pipeline on hardware that cannot fit Llama 8B. "
            "Config matches smoke_test.py: n_layers=2, d_model=512."
        ),
    )

    p.add_argument("--batch_size",      type=int, default=4)
    p.add_argument("--num_workers",     type=int, default=4)
    p.add_argument("--accum_steps",     type=int, default=8,
                   help="Gradient accumulation steps before each optimizer update.")
    p.add_argument("--max_steps",       type=int, default=None,
                   help="Stop after this many optimizer steps (for smoke runs).")
    p.add_argument(
        "--save_every", type=int, default=None,
        help=(
            "Save a checkpoint every N optimizer steps. "
            "Default None = no checkpointing. Disable during development to avoid "
            "expensive disk I/O and WER evaluation."
        ),
    )
    p.add_argument(
        "--log_every", type=int, default=10,
        help=(
            "Print and log diagnostic metrics every N optimizer steps. "
            "Also controls how often the --diag_shard eval pass is triggered."
        ),
    )
    p.add_argument(
        "--no_grad_clip", action="store_true",
        help=(
            "Disable gradient norm clipping. Only use during development / "
            "single-speaker overfitting where fast memorisation is desired and "
            "training stability is not a concern."
        ),
    )
    p.add_argument(
        "--grad_clip_max_norm", type=float, default=1.0,
        help="Max gradient norm for clipping (default 1.0). Ignored when --no_grad_clip is set.",
    )
    p.add_argument("--checkpoint_dir",  type=Path, default=Path("checkpoints"))
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--wandb",           action="store_true",
                   help="Enable Weights & Biases logging.")
    p.add_argument("--wandb_project",   type=str, default="speech-llm-dev",
                   help=(
                       "W&B project name. Default 'speech-llm-dev' is for local "
                       "development. Use 'speech-llm' for real training runs."
                   ))

    # ── Evaluation shards (each a single .tar file) ───────────────────────────
    p.add_argument("--eval_dev_clean",  type=Path, default=None,
                   help="Path to dev-clean-000000.tar shard.")
    p.add_argument("--eval_dev_other",  type=Path, default=None,
                   help="Path to dev-other-000000.tar shard.")
    p.add_argument("--eval_test_clean", type=Path, default=None,
                   help="Path to test-clean-000000.tar shard.")
    p.add_argument("--eval_test_other", type=Path, default=None,
                   help="Path to test-other-000000.tar shard.")
    p.add_argument("--eval_batch_size", type=int, default=8,
                   help="Batch size for WER evaluation forward passes.")
    p.add_argument("--max_eval_batches", type=int, default=None,
                   help="Cap eval at N batches per split (useful during development).")
    p.add_argument(
        "--eval_at_end", action="store_true",
        help=(
            "Run WER evaluation once after training finishes (after --max_steps is "
            "reached or the data is exhausted). Evaluation is NOT run mid-training. "
            "Useful during development to avoid expensive eval every checkpoint."
        ),
    )

    # ── Diagnostic shard ─────────────────────────────────────────────────────
    p.add_argument(
        "--diag_shard", type=Path, default=None,
        help=(
            "Path to a single .tar shard for in-loop diagnostic evaluation. "
            "When provided, a small eval pass is run every --log_every steps "
            "and results are logged under the 'diag_eval/' prefix alongside "
            "the training diagnostics. Produced by scripts/make_dev_dataset.py."
        ),
    )
    p.add_argument(
        "--max_diag_batches", type=int, default=3,
        help=(
            "Maximum number of batches to process from --diag_shard per eval pass. "
            "Kept small (default 3) so the diag pass does not stall the training loop."
        ),
    )

    p.add_argument(
        "--instruction_mode",
        choices=["unformatted", "formatted", "both"],
        default="unformatted",
        help=(
            "Which instruction variant(s) to use during training. "
            "'unformatted' trains only on the plain-text instruction+label pair. "
            "'formatted' trains only on the punctuated instruction+label pair. "
            "'both' randomly alternates between the two per sample (original behaviour). "
            "Eval always runs both variants regardless of this setting."
        ),
    )
    p.add_argument(
        "--n_sample_transcriptions", type=int, default=20,
        help=(
            "Number of (reference, hypothesis) pairs to sample per split per "
            "instruction type and log to W&B as a comparison table at each eval."
        ),
    )

    p.add_argument(
        "--adapter_pca_init", type=str, default=None,
        help=(
            "Path to a PCA init file produced by scripts/compute_adapter_pca_init.py. "
            "When provided, AudioAdapter.mlp[0].weight is initialised with the saved "
            "PCA basis instead of the default random init."
        ),
    )

    p.add_argument(
        "--gradient_checkpointing", action="store_true",
        help=(
            "Enable gradient checkpointing on the Llama model to reduce activation "
            "memory at the cost of ~20-30% extra compute during the backward pass. "
            "Recommended for all GCP/RunPod training runs. Not needed for stub runs."
        ),
    )
    p.add_argument(
        "--freeze_encoder", action="store_true",
        help=(
            "Freeze the Whisper encoder: disable gradients and exclude it from the "
            "optimizer. Useful when overfitting on a small dataset and you want to "
            "isolate adapter+LLM training. Logged as train/encoder_frozen=1 in W&B."
        ),
    )
    p.add_argument(
        "--freeze_llama", action="store_true",
        help=(
            "Freeze Llama: set requires_grad=False on all Llama parameters and "
            "exclude them from the optimizer. Logged as train/llama_frozen=1 in W&B."
        ),
    )
    p.add_argument(
        "--staged_encoder", action="store_true",
        help=(
            "Start with the encoder frozen. Once loss/train_first_token drops below "
            "baselines['first_token_loss'] (from baselines.json), the encoder is "
            "unfrozen and its LR linearly warms up from adapter_lr/100 to adapter_lr/10 "
            "over 100 optimizer steps, then stays at adapter_lr/10. "
            "Requires baselines.json to contain first_token_loss. "
            "Mutually exclusive with --freeze_encoder (permanent freeze wins)."
        ),
    )
    p.add_argument(
        "--staged_llama", action="store_true",
        help="Placeholder — raises NotImplementedError if passed.",
    )
    p.add_argument(
        "--adapter_ckpt", type=Path, default=None,
        help=(
            "Path to a saved adapter-only checkpoint (.pt) containing at minimum "
            "{'adapter': state_dict} and optionally {'optimizer_adapter': optimizer "
            "state for adapter params}. Used for cross-stage loading."
        ),
    )
    p.add_argument(
        "--resume_ckpt", type=Path, default=None,
        help=(
            "Path to a full checkpoint (.pt) for same-stage resume. Loads encoder, "
            "adapter, llama, optimizer, scaler, step, epoch, micro_step_in_epoch, "
            "batch_size. When passed, --whisper_ckpt / --llama_ckpt / --adapter_ckpt "
            "are ignored for weight loading."
        ),
    )
    # ── Training stage and stage-2 LR schedule ───────────────────────────────
    p.add_argument(
        "--stage", type=int, default=1, choices=[1, 2],
        help=(
            "Training stage. "
            "1 = adapter-only (encoder + Llama frozen, existing behaviour). "
            "2 = all three modules unfrozen, linear LR warmup then constant. "
            "Stage 2 requires --adapter_ckpt (stage-1 weights) or --resume_ckpt."
        ),
    )
    p.add_argument(
        "--lr_encoder", type=float, default=1e-6,
        help="Stage-2 peak LR for the Whisper encoder (default 1e-6). Pretrained, most fragile.",
    )
    p.add_argument(
        "--lr_adapter", type=float, default=5e-5,
        help="Stage-2 peak LR for the MLP adapter (default 5e-5). Must re-track the moving Llama.",
    )
    p.add_argument(
        "--lr_llama", type=float, default=1.5e-5,
        help="Stage-2 peak LR for the Llama backbone (default 1.5e-5). Gentle full fine-tune.",
    )
    p.add_argument(
        "--warmup_steps", type=int, default=1000,
        help=(
            "Stage-2 linear LR warmup duration in optimizer steps (default 1000). "
            "LR is held constant at peak after warmup — no decay. "
            "Steps once per optimizer.step() so duration is consistent under --accum_steps."
        ),
    )
    p.add_argument(
        "--beta1", type=float, default=0.9,
        help="AdamW beta1 (default 0.9).",
    )
    p.add_argument(
        "--beta2", type=float, default=0.999,
        help=(
            "AdamW beta2 (default 0.999). "
            "Higher than typical 0.995 for fine-tuning second-moment stability."
        ),
    )

    p.add_argument(
        "--wandb_run_name", type=str, default=None,
        help="Explicit W&B run name. If None, W&B auto-generates one.",
    )

    p.add_argument(
        "--early_stop_metric",
        type=str, default=None,
        choices=["eval_first_token_loss", "eval_loss"],
        help=(
            "Metric to monitor for early stopping. Requires --diag_shard. "
            "'eval_first_token_loss' monitors the diag eval first-token loss; "
            "'eval_loss' monitors the diag eval mean loss."
        ),
    )
    p.add_argument(
        "--early_stop_threshold",
        type=float, default=None,
        help=(
            "Stop training when the monitored metric drops below this value. "
            "Required if --early_stop_metric is set."
        ),
    )
    p.add_argument(
        "--early_stop_min_steps",
        type=int, default=500,
        help=(
            "Do not trigger early stopping before this many optimizer steps. "
            "Prevents stopping on a lucky early eval before the model has settled."
        ),
    )

    return p.parse_args()


@torch.no_grad()
def _greedy_generate(
    encoder: WhisperEncoder,
    adapter: AudioAdapter,
    llama: Llama,
    mel: torch.Tensor,
    audio_lengths: torch.Tensor,
    instruction_ids: torch.Tensor,
    instruction_lengths: torch.Tensor,
    sep_token_id: int,
    max_new_tokens: int = 448,
) -> list[list[int]]:
    """Greedy-decode a batch of samples in parallel using EvalPrefixBatch.

    All B sequences generate simultaneously. When a sequence emits the SEP
    token, it is marked finished and subsequent steps append a zero column to
    maintain tensor alignment without polluting its causal attention history.
    Generation stops once every sequence is finished or max_new_tokens is reached.

    Prefix lengths differ across samples (different audio durations). EvalPrefixBatch
    right-pads shorter prefixes with zeros and inserts each generated token at
    gen_pos[i] rather than at the absolute end, so causal attention never sees
    a padding zero in the history of real tokens — matching the training distribution.

    Args:
        mel:                (B, 80, T_mel)
        audio_lengths:      (B,)
        instruction_ids:    (B, T_inst_max)
        instruction_lengths:(B,)
        sep_token_id:       stop token — generation halts when this is emitted
        max_new_tokens:     hard cap; applied per sequence

    Returns:
        list of B lists of pruned token IDs (stop token excluded)
    """
    B      = mel.shape[0]
    device = mel.device

    with torch.amp.autocast("cuda", dtype=torch.float16):
        enc_out     = encoder(mel)
        adapter_out = adapter(enc_out)

    pfx = EvalPrefixBatch(
        adapter_out, audio_lengths,
        instruction_ids, instruction_lengths,
        llama.embed_tokens, sep_token_id,
    )

    finished   = torch.zeros(B, dtype=torch.bool, device=device)
    generated: list[list[int]] = [[] for _ in range(B)]

    for _ in range(max_new_tokens):
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits, _ = llama(pfx.get_batch(), labels=None)  # (B, S, vocab)

        # Read the logit at each sequence's current generation position
        idx_t    = pfx.logit_indices                          # (B,)
        next_ids = logits[torch.arange(B, device=device), idx_t, :].argmax(dim=-1)

        for i in range(B):
            if not finished[i]:
                if int(next_ids[i].item()) == sep_token_id:
                    finished[i] = True
                else:
                    generated[i].append(int(next_ids[i].item()))

        if finished.all():
            break

        safe_ids    = next_ids.masked_fill(finished, 0)
        next_embeds = llama.embed_tokens(safe_ids.unsqueeze(1))  # (B, 1, d)
        pfx.append(next_embeds, finished)

    return generated


def _evaluate_all_splits(
    encoder: WhisperEncoder,
    adapter: AudioAdapter,
    llama: Llama,
    eval_loaders: dict[str, torch.utils.data.DataLoader],
    tokenizer: PrunedTokenizer,
    sep_token_id: int,
    device: torch.device,
    max_batches: int | None = None,
    n_samples: int = 20,
    sample_seed: int = 0,
) -> tuple[dict[str, float], list[dict]]:
    """Run batched greedy WER evaluation on every eval split with both instructions.

    For each split, generation is run twice per batch — once with the unformatted
    instruction and once with the formatted instruction — and WER is reported
    separately for each. No text normalisation is applied so the scores reflect
    whether the model actually follows the formatting instruction.

    Also randomly samples up to n_samples (reference, hypothesis) pairs per split
    per instruction type for qualitative inspection.

    Args:
        eval_loaders:  split name → DataLoader (from build_eval_dataloader)
        tokenizer:     PrunedTokenizer for decoding generated IDs to text
        max_batches:   cap per split (None = full eval)
        n_samples:     number of (ref, hyp) pairs to sample per split per type
        sample_seed:   RNG seed for reproducible sampling

    Returns:
        wer_dict:    keys like "dev-clean/unformatted" and "dev-clean/formatted"
        sample_rows: list of dicts with keys split, type, reference, hypothesis
    """
    import jiwer

    encoder.eval()
    adapter.eval()
    llama.eval()

    results:     dict[str, float] = {}
    sample_rows: list[dict]       = []
    rng = random.Random(sample_seed)

    for split_name, loader in eval_loaders.items():
        pairs_unfmt: list[tuple[str, str]] = []   # (ref, hyp)
        pairs_fmt:   list[tuple[str, str]] = []

        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            (mel, audio_lengths,
             unfmt_ids, unfmt_lens,
             fmt_ids,   fmt_lens,
             refs_unformatted, refs_formatted) = batch

            mel           = mel.to(device)
            audio_lengths = audio_lengths.to(device)
            unfmt_ids     = unfmt_ids.to(device)
            unfmt_lens    = unfmt_lens.to(device)
            fmt_ids       = fmt_ids.to(device)
            fmt_lens      = fmt_lens.to(device)

            batch_hyps_unfmt = _greedy_generate(
                encoder, adapter, llama,
                mel, audio_lengths, unfmt_ids, unfmt_lens,
                sep_token_id=sep_token_id,
            )
            batch_hyps_fmt = _greedy_generate(
                encoder, adapter, llama,
                mel, audio_lengths, fmt_ids, fmt_lens,
                sep_token_id=sep_token_id,
            )

            for i in range(len(batch_hyps_unfmt)):
                pairs_unfmt.append((refs_unformatted[i], tokenizer.decode(batch_hyps_unfmt[i])))
                pairs_fmt.append((refs_formatted[i],     tokenizer.decode(batch_hyps_fmt[i])))

        refs_unfmt = [r for r, _ in pairs_unfmt]
        hyps_unfmt = [h for _, h in pairs_unfmt]
        refs_fmt   = [r for r, _ in pairs_fmt]
        hyps_fmt   = [h for _, h in pairs_fmt]

        wer_unfmt = jiwer.wer(refs_unfmt, hyps_unfmt) if hyps_unfmt else float("nan")
        wer_fmt   = jiwer.wer(refs_fmt,   hyps_fmt)   if hyps_fmt   else float("nan")

        n = len(hyps_unfmt)
        print(f"  WER {split_name}/unformatted: {wer_unfmt:.1%}  ({n} samples)")
        print(f"  WER {split_name}/formatted:   {wer_fmt:.1%}")

        results[f"{split_name}/unformatted"] = wer_unfmt
        results[f"{split_name}/formatted"]   = wer_fmt

        # Random sample for qualitative table
        for ref, hyp in rng.sample(pairs_unfmt, min(n_samples, len(pairs_unfmt))):
            sample_rows.append({
                "split": split_name, "type": "unformatted",
                "reference": ref, "hypothesis": hyp,
            })
        for ref, hyp in rng.sample(pairs_fmt, min(n_samples, len(pairs_fmt))):
            sample_rows.append({
                "split": split_name, "type": "formatted",
                "reference": ref, "hypothesis": hyp,
            })

    encoder.train()
    adapter.train()
    llama.train()

    return results, sample_rows


def _load_adapter_optimizer_state(
    optimizer: torch.optim.Optimizer,
    saved_opt_state: dict,
    adapter: "AudioAdapter",
) -> None:
    """Copy saved adapter optimizer state into the current optimizer.

    Matches adapter parameters by their position within the adapter's own
    parameter list, then inserts matching state entries into optimizer.state
    keyed by the live parameter tensors. param_groups are left untouched.
    """
    adapter_ptrs = {p.data_ptr(): p for p in adapter.parameters()}

    # Build a flat ordered list of live adapter params as they appear in the optimizer
    live_adapter_params: list[torch.Tensor] = []
    for grp in optimizer.param_groups:
        for p in grp["params"]:
            if p.data_ptr() in adapter_ptrs:
                live_adapter_params.append(p)

    # Build a flat ordered list of saved adapter params from saved_opt_state
    saved_params_flat: list[int] = []
    for grp in saved_opt_state.get("param_groups", []):
        saved_params_flat.extend(grp["params"])

    saved_state: dict = saved_opt_state.get("state", {})
    for live_idx, (live_p, saved_idx) in enumerate(
        zip(live_adapter_params, saved_params_flat)
    ):
        if saved_idx in saved_state:
            optimizer.state[live_p] = saved_state[saved_idx]


def _exhaust_dataloader(loader: torch.utils.data.DataLoader, n_steps: int) -> None:
    """Advance the dataloader by n_steps batches without processing them.

    Used to skip data already seen in a prior training stage so stage 2
    does not train on data seen in stage 1.
    """
    it = iter(loader)
    for _ in range(n_steps):
        try:
            next(it)
        except StopIteration:
            break


def _load_shard_list(args: argparse.Namespace) -> list[str]:
    if args.shards_file is not None:
        lines = args.shards_file.read_text().splitlines()
        return [ln.strip() for ln in lines if ln.strip()]
    return list_shards(args.shards)


def _build_stage2_param_groups(
    encoder: WhisperEncoder,
    adapter: AudioAdapter,
    llama: Llama,
    lr_encoder: float,
    lr_adapter: float,
    lr_llama: float,
) -> list[dict]:
    """Three named param groups for stage-2 joint training.

    Each group carries its own peak LR. A shared warmup scheduler scales all
    three by the same factor so relative LR ratios are preserved during warmup.
    """
    groups = [
        {"name": "encoder", "params": list(encoder.parameters()), "lr": lr_encoder},
        {"name": "adapter", "params": list(adapter.parameters()), "lr": lr_adapter},
        {"name": "llama",   "params": list(llama.parameters()),   "lr": lr_llama},
    ]
    for g in groups:
        assert len(g["params"]) > 0, f"Param group '{g['name']}' is empty — check model construction."
    return groups


def _make_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear ramp 0 → peak over warmup_steps, then constant.

    One lambda scales all param groups; per-group peak comes from each group's
    base_lr, so a single factor is correct for all three simultaneously.
    """
    def lr_lambda(step: int) -> float:
        return min(step / max(warmup_steps, 1), 1.0)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _print_stage2_startup(
    param_groups: list[dict],
    warmup_steps: int,
    betas: tuple[float, float],
) -> None:
    """Print stage-2 optimizer config to stdout so the run is self-documenting."""
    print("─" * 60)
    print("Stage 2 — joint fine-tune config:")
    for g in param_groups:
        n = sum(p.numel() for p in g["params"])
        print(f"  {g['name']:8s}  {n / 1e6:8.1f}M params  peak LR {g['lr']:.2e}")
    print(f"  warmup_steps={warmup_steps}  betas={betas}  weight_decay=0.01")
    print("─" * 60)


def main() -> None:
    """Parse CLI arguments and run the training loop."""
    args = _parse_args()
    if args.staged_llama:
        # TODO: mirror --staged_encoder once implemented
        raise NotImplementedError("--staged_llama is not yet implemented.")
    if args.stage == 2 and args.resume_ckpt is None and args.adapter_ckpt is None:
        raise ValueError(
            "--stage 2 requires --adapter_ckpt (stage-1 adapter weights) "
            "or --resume_ckpt (mid-stage-2 resume)."
        )
    if args.early_stop_metric is not None and args.diag_shard is None:
        raise argparse.ArgumentTypeError(
            "--early_stop_metric requires --diag_shard to be set"
        )
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Pruned vocab config ───────────────────────────────────────────────────
    config_path = Path(args.tokenizer) / "pruned_config.json"
    with config_path.open() as f:
        pruned_cfg = json.load(f)
    vocab_size   = pruned_cfg["vocab_size"]    # 40148
    sep_token_id = pruned_cfg["sep_token_id"]  # 40147
    print(f"Pruned vocab: {vocab_size} tokens  SEP id: {sep_token_id}")

    # baseline values for milestone target during training.
    baseline_path = Path("baselines.json")
    if baseline_path.exists():
        baselines_data = json.loads(baseline_path.read_text())
    else:
        baselines_data = {}

    # Staged encoder: freeze encoder initially, auto-thaw when first_token_loss crossed.
    _staged_encoder = args.staged_encoder
    if _staged_encoder:
        if args.freeze_encoder:
            print("WARNING: --freeze_encoder overrides --staged_encoder; encoder permanently frozen.")
            _staged_encoder = False
        elif baselines_data.get("first_token_loss") is None:
            print("WARNING: --staged_encoder set but baselines.json has no first_token_loss; staged training disabled.")
            _staged_encoder = False
    print(
        f"Staged encoder: {'ACTIVE' if _staged_encoder else 'disabled'}"
        + (f"  (threshold first_token_loss={baselines_data['first_token_loss']:.4f})" if _staged_encoder else "")
    )
    # Stage 2 trains all modules jointly — staged encoder would conflict.
    if args.stage == 2 and _staged_encoder:
        print("Stage 2: --staged_encoder incompatible with all-module training — disabling.")
        _staged_encoder = False
    # ── Model instantiation ───────────────────────────────────────────────────
    if args.stub:
        # Tiny model for pipeline validation when full 8B does not fit in VRAM/RAM.
        # Matches smoke_test.py stub config exactly.
        llama_cfg = LlamaConfig(
            n_layers=6, d_model=512, n_heads=8, n_kv_heads=2,
            intermediate_size=1024, vocab_size=vocab_size,
        )
        llama_dim = 512
        print("STUB mode: using tiny randomly-initialised model (no pretrained weights).")
    else:
        llama_cfg = LlamaConfig(vocab_size=vocab_size)
        llama_dim = 4096

    encoder = WhisperEncoder()
    adapter = AudioAdapter(llama_dim=llama_dim, pca_init_path=args.adapter_pca_init)
    llama   = Llama(llama_cfg)

    # ── Load pretrained weights ───────────────────────────────────────────────
    if not args.stub:
        if args.whisper_ckpt is None or args.llama_ckpt is None:
            raise ValueError("--whisper_ckpt and --llama_ckpt are required unless --stub is set.")
        print("Loading Whisper encoder weights …")
        encoder.load_openai_weights(args.whisper_ckpt)
        vocab_map_path = Path(args.tokenizer) / "vocab_map.json"
        with vocab_map_path.open() as f:
            vocab_map = json.load(f)
        print("Loading Llama transformer weights (embedding initialised from pretrained rows) …")
        llama.load_meta_weights(args.llama_ckpt, vocab_map=vocab_map)
    elif args.whisper_ckpt is not None:
        print("Loading Whisper encoder weights …")
        encoder.load_openai_weights(args.whisper_ckpt)

    encoder = encoder.to(device)
    adapter = adapter.to(device)
    llama   = llama.to(device)

    if args.gradient_checkpointing:
        if args.stub:
            print("[warn] --gradient_checkpointing ignored with --stub")
        else:
            llama.enable_gradient_checkpointing()
            print("[info] gradient checkpointing enabled")

    encoder.train()
    adapter.train()
    llama.train()

    # ── Encoder / Llama freeze (must happen before optimizer construction) ────
    if args.stage == 2:
        # All three modules train jointly. --freeze_encoder / --freeze_llama are ignored.
        if args.freeze_encoder or args.freeze_llama:
            print("WARNING: --freeze_encoder / --freeze_llama are ignored in --stage 2.")
        encoder.requires_grad_(True)
        adapter.requires_grad_(True)
        llama.requires_grad_(True)
        print("Encoder: trainable (stage 2)")
        print("Llama:   trainable (stage 2)")
    else:
        if args.freeze_encoder or _staged_encoder:
            encoder.requires_grad_(False)
            print("Encoder: FROZEN" + (" (staged — will auto-thaw)" if _staged_encoder else ""))
        else:
            print("Encoder: trainable")

        if args.freeze_llama:
            for p in llama.parameters():
                p.requires_grad_(False)
            print("Llama: FROZEN")
        else:
            print("Llama: trainable")

    n_enc   = sum(p.numel() for p in encoder.parameters())
    n_ada   = sum(p.numel() for p in adapter.parameters())
    n_llm   = sum(p.numel() for p in llama.parameters())
    print(
        f"Parameters — encoder: {n_enc / 1e6:.1f}M  "
        f"adapter: {n_ada / 1e6:.1f}M  "
        f"llama: {n_llm / 1e6:.0f}M  "
        f"total: {(n_enc + n_ada + n_llm) / 1e9:.2f}B"
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    import bitsandbytes as bnb
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None

    if args.stage == 2:
        # Three named groups, each with its own peak LR; one warmup factor scales all.
        # Fresh optimizer — do not carry over stage-1 second-moment estimates.
        _param_groups = _build_stage2_param_groups(
            encoder, adapter, llama,
            args.lr_encoder, args.lr_adapter, args.lr_llama,
        )
        optimizer = bnb.optim.AdamW8bit(
            _param_groups,
            betas=(args.beta1, args.beta2),
            weight_decay=0.01,
        )
        scheduler = _make_warmup_scheduler(optimizer, args.warmup_steps)
        _print_stage2_startup(_param_groups, args.warmup_steps, (args.beta1, args.beta2))
    else:
        # Stage 1: adapter-only or partial unfreeze (existing behaviour).
        # In staged mode the encoder is excluded here; added via add_param_group on thaw.
        # Adapter is always param_groups[0] when encoder is absent (frozen or staged).
        _param_groups = []
        if not args.freeze_encoder and not _staged_encoder:
            _param_groups.append({"params": encoder.parameters(), "lr": 1e-7})
        _param_groups.append({"params": adapter.parameters(), "lr": 1e-4})
        if not args.freeze_llama:
            _param_groups.append({"params": llama.parameters(), "lr": 1e-5})
        optimizer = bnb.optim.AdamW8bit(
            _param_groups,
            betas=(args.beta1, args.beta2),
            weight_decay=0.01,
        )
    scaler = torch.amp.GradScaler("cuda")

    # ── Staged encoder state ──────────────────────────────────────────────────
    _staged_thawed      = False   # becomes True when trigger fires
    _encoder_thaw_step  = 0       # global_step when encoder was thawed
    _ENCODER_WARMUP_STEPS = 100   # steps to ramp from adapter_lr/100 → adapter_lr/10

    # ── Stage-2 diagnostics state ─────────────────────────────────────────────
    _stage2_enc_grad_checked = False          # one-shot check that encoder receives gradients
    _prev_eval_loss: float | None = None      # for unfreeze-shock detection
    _stage2_eval_count   = 0
    _stage2_shock_warned = False
    _STAGE2_SHOCK_WINDOW = 5                  # warn only within the first N diag evals

    # ── W&B ───────────────────────────────────────────────────────────────────
    if args.wandb:
        import os
        import wandb
        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "WANDB_API_KEY is not set. "
                "Add 'export WANDB_API_KEY=...' to ~/.bashrc and reload your shell."
            )
        # wandb.login(key=api_key, relogin=False)
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "stage":              args.stage,
                "effective_batch":    args.batch_size * args.accum_steps,
                "batch_size":         args.batch_size,
                "accum_steps":        args.accum_steps,
                "max_steps":          args.max_steps,
                "freeze_encoder":     args.freeze_encoder,
                "freeze_llama":       args.freeze_llama,
                "staged_encoder":     args.staged_encoder,
                "lr_encoder":         args.lr_encoder,
                "lr_adapter":         args.lr_adapter,
                "lr_llama":           args.lr_llama,
                "warmup_steps":       args.warmup_steps,
                "beta1":              args.beta1,
                "beta2":              args.beta2,
                "grad_clip_max_norm": args.grad_clip_max_norm,
                "instruction_mode":   args.instruction_mode,
                "seed":               args.seed,
                "resume_ckpt":            str(args.resume_ckpt) if args.resume_ckpt else None,
                "adapter_ckpt":           str(args.adapter_ckpt) if args.adapter_ckpt else None,
                "gradient_checkpointing": args.gradient_checkpointing,
            },
        )

    if args.stage == 2:
        _encoder_frozen_val = 0.0   # stage 2 always unfreezes regardless of --freeze_* flags
        _llama_frozen_val   = 0.0
    else:
        _encoder_frozen_val = 1.0 if (args.freeze_encoder or _staged_encoder) else 0.0
        _llama_frozen_val   = 1.0 if args.freeze_llama else 0.0

    # ── Instruction mode ──────────────────────────────────────────────────────
    if args.instruction_mode == "unformatted":
        train_pairs = [_INSTRUCTION_PAIRS[0]]
    elif args.instruction_mode == "formatted":
        train_pairs = [_INSTRUCTION_PAIRS[1]]
    else:
        train_pairs = list(_INSTRUCTION_PAIRS)
    print(f"Instruction mode: {args.instruction_mode}  ({len(train_pairs)} variant(s))")

    # ── Shards ───────────────────────────────────────────────────────────────
    all_shards = _load_shard_list(args)
    if not all_shards:
        raise FileNotFoundError("No shards found; check --shards_file or --shards.")
    print(f"Training on {len(all_shards)} shards.")

    # ── Eval dataloaders ──────────────────────────────────────────────────────
    tokenizer = PrunedTokenizer(args.tokenizer)
    diag = Diagnostics(
        tokenizer=tokenizer,
        sep_token_id=sep_token_id,
        log_every=args.log_every,
        top_k=5,
    )

    # ── Diagnostic shard dataloader ───────────────────────────────────────────
    # Iterated every --log_every steps for in-loop eval without WER generation.
    # WebDataset is infinite so _diag_iter never raises StopIteration.
    _diag_iter = None
    if args.diag_shard is not None:
        if not args.diag_shard.exists():
            raise FileNotFoundError(f"--diag_shard not found: {args.diag_shard}")
        _diag_loader = build_dataloader(
            [str(args.diag_shard)],
            tokenizer_path=args.tokenizer,
            sep_token_id=sep_token_id,
            batch_size=args.batch_size,
            num_workers=0,
            instruction_variants=train_pairs,
            shuffle_buffer=100,
            partial=True,   # keep the final incomplete batch on small shards
        )
        _diag_iter = iter(_diag_loader)
        print(f"Diagnostic shard: {args.diag_shard}")
    _EVAL_SHARD_ARGS = {
        "dev-clean":  args.eval_dev_clean,
        "dev-other":  args.eval_dev_other,
        "test-clean": args.eval_test_clean,
        "test-other": args.eval_test_other,
    }
    eval_loaders: dict[str, torch.utils.data.DataLoader] = {
        name: build_eval_dataloader(
            shard_path=path,
            tokenizer_path=args.tokenizer,
            instruction_variants=INSTRUCTION_VARIANTS,
            batch_size=args.eval_batch_size,
        )
        for name, path in _EVAL_SHARD_ARGS.items()
        if path is not None
    }
    if eval_loaders:
        print(f"Eval splits: {list(eval_loaders)}")
    else:
        print("No eval splits provided — skipping WER evaluation.")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Checkpoint loading ────────────────────────────────────────────────────
    resume_global_step         = 0
    resume_epoch               = 0
    resume_micro_step_in_epoch = 0
    resume_batch_size          = args.batch_size
    _cross_stage_resume        = False

    if args.resume_ckpt:
        print(f"Resuming from checkpoint: {args.resume_ckpt}")
        _ckpt = torch.load(args.resume_ckpt, map_location="cpu")
        if "encoder" in _ckpt:
            encoder.load_state_dict(_ckpt["encoder"])
        if "adapter" in _ckpt:
            adapter.load_state_dict(_ckpt["adapter"])
        if "llama" in _ckpt:
            llama.load_state_dict(_ckpt["llama"])
        optimizer.load_state_dict(_ckpt["optimizer"])
        scaler.load_state_dict(_ckpt["scaler"])
        resume_global_step         = _ckpt["step"]
        resume_epoch               = _ckpt.get("epoch", 0)
        resume_micro_step_in_epoch = _ckpt.get("micro_step_in_epoch", 0)
        resume_batch_size          = _ckpt.get("batch_size", args.batch_size)
        print(
            f"  Resumed at step={resume_global_step}  epoch={resume_epoch}  "
            f"micro_step_in_epoch={resume_micro_step_in_epoch}"
        )
    elif args.adapter_ckpt:
        print(f"Cross-stage load from adapter checkpoint: {args.adapter_ckpt}")
        _ckpt = torch.load(args.adapter_ckpt, map_location="cpu")
        adapter.load_state_dict(_ckpt["adapter"])
        if args.stage != 2 and "optimizer_adapter" in _ckpt:
            # Stage 1: restore adapter optimizer state for LR continuity.
            # Stage 2: fresh optimizer — stage-1 second moments are stale for the new LR scale.
            _load_adapter_optimizer_state(optimizer, _ckpt["optimizer_adapter"], adapter)
        resume_epoch               = _ckpt.get("epoch", 0)
        resume_micro_step_in_epoch = _ckpt.get("micro_step_in_epoch", 0)
        resume_batch_size          = _ckpt.get("batch_size", args.batch_size)
        _cross_stage_resume        = True
        print(
            f"  Cross-stage: epoch={resume_epoch}  "
            f"micro_step_in_epoch={resume_micro_step_in_epoch}"
        )

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step   = resume_global_step
    micro_step    = 0     # gradient accumulation steps
    accum_loss    = 0.0
    loss_ema: float | None = None
    _EMA_ALPHA = 0.98

    # Throughput tracking — audio-seconds processed per wall-clock second.
    # step_start resets AFTER checkpoint saving so that checkpoint I/O is
    # included in that step's elapsed time, which produces the valley effect.
    train_start   = time.perf_counter()
    step_start    = time.perf_counter()
    step_audio_s  = 0.0   # audio-seconds accumulated in the current step window
    total_audio_s = 0.0   # cumulative audio-seconds for the whole run

    optimizer.zero_grad()

    epoch               = resume_epoch
    micro_step_in_epoch = 0
    done  = False
    # cached_batch = None
    while not done:
        epoch_shards = list(all_shards)
        random.Random(args.seed + epoch).shuffle(epoch_shards)

        loader = build_dataloader(
            epoch_shards,
            tokenizer_path=args.tokenizer,
            sep_token_id=sep_token_id,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            instruction_variants=train_pairs,
        )

        # ── Cross-stage fast-forward (first epoch only) ───────────────────────
        if _cross_stage_resume and epoch == resume_epoch:
            n_skip = math.floor(
                resume_micro_step_in_epoch * resume_batch_size / args.batch_size
            )
            if n_skip > 0:
                print(
                    f"[resume] skipping {n_skip} batches in epoch {epoch} "
                    f"to avoid stage-1 data"
                )
                _exhaust_dataloader(loader, n_skip)
                micro_step_in_epoch = n_skip
            _cross_stage_resume = False  # only fast-forward once

        for batch in loader:
            micro_step_in_epoch += 1
            # if cached_batch is None:
            #     cached_batch = batch
            # else:
            #     batch = cached_batch
            (mel, audio_lengths,
             instruction_ids, instruction_lengths,
             transcript_ids, transcript_lengths) = [t.to(device) for t in batch]

            # audio_lengths[i] = adapter tokens = ceil(T_mel/2/4).
            # Reverse: T_mel ≈ audio_lengths * 8 frames; each frame = 10 ms.
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
                logits, loss = llama(inputs, labels)

            diag.record_micro_with_logits(labels, logits, loss.detach())
            scaler.scale(loss / args.accum_steps).backward()
            accum_loss += loss.item()
            micro_step += 1

            if micro_step % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                if not args.no_grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        [p for grp in optimizer.param_groups for p in grp["params"]],
                        args.grad_clip_max_norm,
                    )
                diag.record_grad_norms(encoder, adapter, llama)

                # One-shot sanity check on the first stage-2 step: confirm the encoder
                # actually has gradients (requires_grad was set before optimizer build).
                if args.stage == 2 and not _stage2_enc_grad_checked:
                    _enc_gn = math.sqrt(sum(
                        p.grad.norm().item() ** 2
                        for p in encoder.parameters()
                        if p.grad is not None
                    ))
                    if _enc_gn == 0.0:
                        print("[WARN] Stage-2 step 1: encoder grad norm is ZERO — check requires_grad.")
                    else:
                        print(f"[info] Stage-2 step 1: encoder grad norm = {_enc_gn:.4e}  (unfreeze confirmed)")
                    _stage2_enc_grad_checked = True

                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    # Step once per optimizer update, not per micro-batch, so warmup
                    # duration is identical regardless of --accum_steps.
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss   = accum_loss / args.accum_steps
                accum_loss = 0.0
                loss_ema   = avg_loss if loss_ema is None else _EMA_ALPHA * loss_ema + (1 - _EMA_ALPHA) * avg_loss

                # ── Checkpoint (optional) ─────────────────────────────────────
                _save_regular = args.save_every is not None and global_step % args.save_every == 0
                _save_staged  = _staged_thawed and global_step % (2 * args.log_every) == 0
                if _save_regular or _save_staged:
                    ckpt_dir  = args.checkpoint_dir
                    ckpt_path = ckpt_dir / f"step_{global_step:07d}.pt"
                    _ckpt_dict = {
                        "step":                global_step,
                        "epoch":               epoch,
                        "micro_step_in_epoch": micro_step_in_epoch,
                        "batch_size":          args.batch_size,
                        "adapter":             adapter.state_dict(),
                        "optimizer":           optimizer.state_dict(),
                        "scaler":              scaler.state_dict(),
                    }
                    if args.stage == 2 or not args.freeze_encoder:
                        _ckpt_dict["encoder"] = encoder.state_dict()
                    if args.stage == 2 or not args.freeze_llama:
                        _ckpt_dict["llama"] = llama.state_dict()
                    torch.save(_ckpt_dict, ckpt_path)
                    print(f"Checkpoint saved → {ckpt_path}")

                    # Adapter-only file for cross-stage loading (stage 1 only)
                    if args.freeze_encoder and args.freeze_llama:
                        adapter_ckpt_path = ckpt_dir / f"adapter-step{global_step}.pt"
                        torch.save(
                            {
                                "step":                global_step,
                                "epoch":               epoch,
                                "micro_step_in_epoch": micro_step_in_epoch,
                                "batch_size":          args.batch_size,
                                "adapter":             adapter.state_dict(),
                                "optimizer_adapter":   optimizer.state_dict(),
                            },
                            adapter_ckpt_path,
                        )
                        print(f"Adapter checkpoint saved → {adapter_ckpt_path}")

                # ── Diagnostic shard eval pass ────────────────────────────────
                eval_diag_metrics: dict = {}
                if _diag_iter is not None and global_step % args.log_every == 0:
                    encoder.eval()
                    adapter.eval()
                    llama.eval()
                    for _ in range(args.max_diag_batches):
                        try:
                            diag_batch = next(_diag_iter)
                        except StopIteration:
                            # shard exhausted — restart from the beginning
                            _diag_iter = iter(_diag_loader)
                            diag_batch = next(_diag_iter)
                        (d_mel, d_audio_len,
                         d_inst_ids, d_inst_lens,
                         d_trans_ids, d_trans_lens) = [t.to(device) for t in diag_batch]
                        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                            d_enc  = encoder(d_mel)
                            d_ada  = adapter(d_enc)
                            d_inp, d_lbl = prepare_input(
                                d_ada, d_audio_len,
                                d_inst_ids, d_inst_lens,
                                d_trans_ids, d_trans_lens,
                                llama.embed_tokens, sep_token_id,
                            )
                            d_logits, d_loss = llama(d_inp, d_lbl)
                        diag.record_eval_micro_with_logits(d_lbl, d_logits, d_loss.detach())
                    encoder.train()
                    adapter.train()
                    llama.train()
                    eval_diag_metrics = diag.flush_eval(global_step)

                    # Unfreeze-shock detection: a rising diag eval loss within the
                    # first few evals is the classic sign that warmup is too short
                    # or lr_encoder / lr_llama is too high.
                    if args.stage == 2 and not _stage2_shock_warned:
                        _curr_el = eval_diag_metrics.get("loss/eval_rest")
                        if _curr_el is not None:
                            _stage2_eval_count += 1
                            if (
                                _prev_eval_loss is not None
                                and _curr_el > _prev_eval_loss
                                and _stage2_eval_count <= _STAGE2_SHOCK_WINDOW
                            ):
                                print(
                                    f"[WARN] Unfreeze shock at step {global_step}: "
                                    f"diag eval loss {_prev_eval_loss:.4f} → {_curr_el:.4f}. "
                                    f"Consider longer --warmup_steps or lower "
                                    f"--lr_encoder / --lr_llama."
                                )
                                _stage2_shock_warned = True
                            _prev_eval_loss = _curr_el

                # ── Throughput: reset AFTER checkpoint so disk I/O folds in ──
                t_now         = time.perf_counter()
                elapsed       = max(t_now - step_start, 1e-9)
                throughput    = step_audio_s / elapsed
                total_audio_s += step_audio_s
                step_audio_s  = 0.0
                step_start    = t_now

                print(
                    f"step {global_step:6d}  loss {avg_loss:.4f}"
                    f"  {throughput:.2f}× realtime"
                )
                diag_metrics = diag.flush(global_step)

                # ── Staged encoder: trigger check ─────────────────────────────
                # Fires at most once: when train_first_token first drops below
                # the corpus first_token_loss baseline.
                if _staged_encoder and not _staged_thawed:
                    _ftl = diag_metrics.get("loss/train_first_token")
                    if _ftl is not None and _ftl < baselines_data["first_token_loss"]:
                        encoder.requires_grad_(True)
                        _adapter_lr = optimizer.param_groups[0]["lr"]
                        optimizer.add_param_group(
                            {"params": list(encoder.parameters()), "lr": _adapter_lr / 100}
                        )
                        _staged_thawed     = True
                        _encoder_thaw_step = global_step
                        _encoder_frozen_val = 0.0
                        print(
                            f"[step {global_step}] Encoder THAWED — "
                            f"train_first_token={_ftl:.4f} < baseline={baselines_data['first_token_loss']:.4f}"
                        )

                # ── Staged encoder: LR warmup (every step after thaw) ─────────
                # Encoder is always the last param group when staged.
                if _staged_thawed:
                    steps_since_thaw = global_step - _encoder_thaw_step
                    _adapter_lr = optimizer.param_groups[0]["lr"]
                    if steps_since_thaw <= _ENCODER_WARMUP_STEPS:
                        t = steps_since_thaw / _ENCODER_WARMUP_STEPS
                        new_enc_lr = _adapter_lr * (0.01 + 0.09 * t)
                    else:
                        new_enc_lr = _adapter_lr / 10
                    optimizer.param_groups[-1]["lr"] = new_enc_lr

                if args.wandb:
                    # Load and log corpus baselines as wandb summary values.
                    # These appear as horizontal reference lines when you use
                    # wandb's "add reference line" feature on any loss chart.
                    if baselines_data:
                        baselines = {
                        "baseline/unigram_loss":     baselines_data["unigram_loss"],
                        "baseline/bigram_loss":      baselines_data["bigram_loss"],
                        "baseline/first_token_loss": baselines_data["first_token_loss"],
                        "baseline/uniform_loss":     baselines_data["uniform_loss"],
                        }
                    else:
                        baselines = {}
                    _gap: dict[str, float] = {}
                    _tr = diag_metrics.get("loss/train_rest")
                    _er = eval_diag_metrics.get("loss/eval_rest")
                    _tf = diag_metrics.get("loss/train_first_token")
                    _ef = eval_diag_metrics.get("loss/eval_first_token")
                    if _tr is not None and _er is not None:
                        _gap["loss/gap_rest"] = _er - _tr
                    if _tf is not None and _ef is not None:
                        _gap["loss/gap_first_token"] = _ef - _tf
                    # Stage 2: fixed group order (enc=0, ada=1, llm=2).
                    # Stage 1: dynamic order depending on freeze flags.
                    if args.stage == 2:
                        _lr_dict = {
                            "train/lr":         optimizer.param_groups[2]["lr"],
                            "train/lr_encoder": optimizer.param_groups[0]["lr"],
                            "train/lr_adapter": optimizer.param_groups[1]["lr"],
                            "train/lr_llama":   optimizer.param_groups[2]["lr"],
                        }
                    else:
                        # In staged mode: adapter=0, llama=1(if unfrozen), encoder=last(after thaw).
                        # In normal mode: encoder=0(if trainable), adapter=next, llama=last.
                        if args.freeze_llama:
                            _llama_pg_idx = 0
                        elif _staged_encoder or args.freeze_encoder:
                            _llama_pg_idx = 1
                        else:
                            _llama_pg_idx = -1
                        _enc_lr_log = (
                            optimizer.param_groups[-1]["lr"] if _staged_thawed
                            else (0.0 if (args.freeze_encoder or _staged_encoder)
                                  else optimizer.param_groups[0]["lr"])
                        )
                        _lr_dict = {
                            "train/lr":         optimizer.param_groups[_llama_pg_idx]["lr"],
                            "train/lr_encoder": _enc_lr_log,
                        }
                    wandb.log(
                        {
                            "train/loss":                            avg_loss,
                            "train/loss_ema":                        loss_ema,
                            "train/encoder_frozen":                  _encoder_frozen_val,
                            "train/llama_frozen":                    _llama_frozen_val,
                            "runtime/throughput_audio_sec_per_sec":  throughput,
                            "runtime/cumulative_audio_hours":         total_audio_s / 3600,
                            "runtime/wall_time_min":                  (t_now - train_start) / 60,
                            **_lr_dict,
                            **diag_metrics,
                            **eval_diag_metrics,
                            **_gap,
                            **baselines,
                        },
                        step=global_step,
                    )
                if args.max_steps is not None and global_step >= args.max_steps:
                    print(f"Reached --max_steps {args.max_steps}. Done.")
                    done = True
                    break

                # ── Early stopping ────────────────────────────────────────────
                if (args.early_stop_metric is not None
                        and args.early_stop_threshold is not None
                        and global_step >= args.early_stop_min_steps):
                    if args.early_stop_metric == "eval_first_token_loss":
                        _es_val = eval_diag_metrics.get("loss/eval_first_token")
                    else:  # eval_loss
                        _es_val = eval_diag_metrics.get("loss/eval_rest")

                    if _es_val is not None and _es_val < args.early_stop_threshold:
                        print(
                            f"[early stop] {args.early_stop_metric}={_es_val:.4f}"
                            f" < {args.early_stop_threshold}"
                            f" at step {global_step} — stopping."
                        )
                        if args.wandb:
                            wandb.log({
                                "early_stop/metric":    _es_val,
                                "early_stop/threshold": args.early_stop_threshold,
                                "early_stop/step":      global_step,
                            })
                        _es_ckpt_dir  = args.checkpoint_dir
                        _es_ckpt_path = _es_ckpt_dir / f"checkpoint-step{global_step}-early-stop.pt"
                        _es_ckpt_dict = {
                            "step":                global_step,
                            "epoch":               epoch,
                            "micro_step_in_epoch": micro_step_in_epoch,
                            "batch_size":          args.batch_size,
                            "adapter":             adapter.state_dict(),
                            "optimizer":           optimizer.state_dict(),
                            "scaler":              scaler.state_dict(),
                        }
                        if args.stage == 2 or not args.freeze_encoder:
                            _es_ckpt_dict["encoder"] = encoder.state_dict()
                        if args.stage == 2 or not args.freeze_llama:
                            _es_ckpt_dict["llama"] = llama.state_dict()
                        torch.save(_es_ckpt_dict, _es_ckpt_path)
                        print(f"Early-stop checkpoint saved → {_es_ckpt_path}")
                        if args.freeze_encoder and args.freeze_llama:
                            _es_adapter_path = _es_ckpt_dir / f"adapter-step{global_step}-early-stop.pt"
                            torch.save(
                                {
                                    "step":                global_step,
                                    "epoch":               epoch,
                                    "micro_step_in_epoch": micro_step_in_epoch,
                                    "batch_size":          args.batch_size,
                                    "adapter":             adapter.state_dict(),
                                    "optimizer_adapter":   optimizer.state_dict(),
                                },
                                _es_adapter_path,
                            )
                            print(f"Early-stop adapter checkpoint saved → {_es_adapter_path}")
                        done = True
                        break

        if not done:
            epoch += 1
            micro_step_in_epoch = 0

    # ── Final checkpoint (always saved at end of training) ───────────────────
    _final_ckpt_path = args.checkpoint_dir / f"checkpoint-step{global_step}-final.pt"
    _final_ckpt_dict = {
        "step":                global_step,
        "epoch":               epoch,
        "micro_step_in_epoch": micro_step_in_epoch,
        "batch_size":          args.batch_size,
        "adapter":             adapter.state_dict(),
        "optimizer":           optimizer.state_dict(),
        "scaler":              scaler.state_dict(),
    }
    if args.stage == 2 or not args.freeze_encoder:
        _final_ckpt_dict["encoder"] = encoder.state_dict()
    if args.stage == 2 or not args.freeze_llama:
        _final_ckpt_dict["llama"] = llama.state_dict()
    torch.save(_final_ckpt_dict, _final_ckpt_path)
    print(f"Final checkpoint saved → {_final_ckpt_path}")
    if args.freeze_encoder and args.freeze_llama:
        _final_adapter_path = args.checkpoint_dir / f"adapter-step{global_step}-final.pt"
        torch.save(
            {
                "step":                global_step,
                "epoch":               epoch,
                "micro_step_in_epoch": micro_step_in_epoch,
                "batch_size":          args.batch_size,
                "adapter":             adapter.state_dict(),
                "optimizer_adapter":   optimizer.state_dict(),
            },
            _final_adapter_path,
        )
        print(f"Final adapter checkpoint saved → {_final_adapter_path}")

    # ── End-of-training WER evaluation ───────────────────────────────────────
    if args.eval_at_end and eval_loaders:
        print("Running end-of-training WER evaluation …")
        wer_results, sample_rows = _evaluate_all_splits(
            encoder, adapter, llama,
            eval_loaders, tokenizer, sep_token_id, device,
            max_batches=args.max_eval_batches,
            n_samples=args.n_sample_transcriptions,
            sample_seed=global_step,
        )
        if args.wandb:
            table = wandb.Table(columns=["split", "type", "reference", "hypothesis"])
            for row in sample_rows:
                table.add_data(
                    row["split"], row["type"],
                    row["reference"], row["hypothesis"],
                )
            wandb.log(
                {
                    **{f"wer/{k}": v for k, v in wer_results.items()},
                    "transcription_samples":   table,
                    "runtime/wall_time_min":   (time.perf_counter() - train_start) / 60,
                    "train/encoder_frozen":    _encoder_frozen_val,
                },
                step=global_step,
            )


if __name__ == "__main__":
    main()

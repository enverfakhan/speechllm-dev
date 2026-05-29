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
        "--freeze_encoder", action="store_true",
        help=(
            "Freeze the Whisper encoder: disable gradients and exclude it from the "
            "optimizer. Useful when overfitting on a small dataset and you want to "
            "isolate adapter+LLM training. Logged as train/encoder_frozen=1 in W&B."
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


def _load_shard_list(args: argparse.Namespace) -> list[str]:
    if args.shards_file is not None:
        lines = args.shards_file.read_text().splitlines()
        return [ln.strip() for ln in lines if ln.strip()]
    return list_shards(args.shards)


def main() -> None:
    """Parse CLI arguments and run the training loop."""
    args = _parse_args()
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
        print("Loading Llama transformer weights (embedding trained from scratch) …")
        llama.load_meta_weights(args.llama_ckpt)
    elif args.whisper_ckpt is not None:
        print("Loading Whisper encoder weights …")
        encoder.load_openai_weights(args.whisper_ckpt)

    encoder = encoder.to(device)
    adapter = adapter.to(device)
    llama   = llama.to(device)

    encoder.train()
    adapter.train()
    llama.train()

    # ── Encoder freeze (must happen before optimizer construction) ────────────
    if args.freeze_encoder or _staged_encoder:
        encoder.requires_grad_(False)
        print("Encoder: FROZEN" + (" (staged — will auto-thaw)" if _staged_encoder else ""))
    else:
        print("Encoder: trainable")

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
    # In staged mode the encoder is excluded here; added via add_param_group on thaw.
    # Adapter is always param_groups[0] when encoder is absent (frozen or staged).
    _param_groups = []
    if not args.freeze_encoder and not _staged_encoder:
        _param_groups.append({"params": encoder.parameters(), "lr": 1e-7})
    _param_groups += [
        {"params": adapter.parameters(), "lr": 1e-5},
        {"params": llama.parameters(),   "lr": 1e-5},
    ]
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(_param_groups, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")

    # ── Staged encoder state ──────────────────────────────────────────────────
    _staged_thawed      = False   # becomes True when trigger fires
    _encoder_thaw_step  = 0       # global_step when encoder was thawed
    _ENCODER_WARMUP_STEPS = 100   # steps to ramp from adapter_lr/100 → adapter_lr/10

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
        wandb.login(key=api_key, relogin=False)
        wandb.init(project=args.wandb_project)

    _encoder_frozen_val = 1.0 if (args.freeze_encoder or _staged_encoder) else 0.0

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
            shuffle_buffer=1,
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

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step   = 0     # optimizer steps
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

    epoch = 0
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

        for batch in loader:
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
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

                avg_loss   = accum_loss / args.accum_steps
                accum_loss = 0.0
                loss_ema   = avg_loss if loss_ema is None else _EMA_ALPHA * loss_ema + (1 - _EMA_ALPHA) * avg_loss

                # ── Checkpoint (optional) ─────────────────────────────────────
                _save_regular = args.save_every is not None and global_step % args.save_every == 0
                _save_staged  = _staged_thawed and global_step % (2 * args.log_every) == 0
                if _save_regular or _save_staged:
                    ckpt_path = args.checkpoint_dir / f"step_{global_step:07d}.pt"
                    torch.save(
                        {
                            "step":      global_step,
                            "encoder":   encoder.state_dict(),
                            "adapter":   adapter.state_dict(),
                            "llama":     llama.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scaler":    scaler.state_dict(),
                        },
                        ckpt_path,
                    )
                    print(f"Checkpoint saved → {ckpt_path}")

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
                    # In staged mode: adapter=0, llama=1, encoder=2(after thaw).
                    # In normal mode: encoder=0(if trainable), adapter=..., llama=-1.
                    _llama_pg_idx = 1 if (_staged_encoder or args.freeze_encoder) else -1
                    _enc_lr_log = (
                        optimizer.param_groups[-1]["lr"] if _staged_thawed
                        else (0.0 if (args.freeze_encoder or _staged_encoder)
                              else optimizer.param_groups[0]["lr"])
                    )
                    wandb.log(
                        {
                            "train/loss":                            avg_loss,
                            "train/loss_ema":                        loss_ema,
                            "train/lr":                              optimizer.param_groups[_llama_pg_idx]["lr"],
                            "train/lr_encoder":                      _enc_lr_log,
                            "train/encoder_frozen":                  _encoder_frozen_val,
                            "runtime/throughput_audio_sec_per_sec":  throughput,
                            "runtime/cumulative_audio_hours":         total_audio_s / 3600,
                            "runtime/wall_time_min":                  (t_now - train_start) / 60,
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

        if not done:
            epoch += 1

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

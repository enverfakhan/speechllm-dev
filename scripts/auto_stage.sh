#!/usr/bin/env bash
# Auto-stage run: stage 1 (adapter-only) then stage 2 (full fine-tune) in a
# single job. The transition fires automatically when eval_first_token_loss from
# --diag_shard drops below the first_token_loss threshold in baselines.json.
#
# Stage 1: encoder + Llama frozen, adapter-only, large effective batch.
# Stage 2: all modules unfrozen, linear LR warmup, smaller effective batch.
#          DataLoader is immediately rebuilt with S2_BATCH_SIZE at transition.
#
# Usage:
#   bash scripts/auto_stage.sh
set -e

# ── Stage-1 hyperparameters ───────────────────────────────────────────────────
S1_BATCH_SIZE=64
S1_ACCUM_STEPS=2
S1_ADAPTER_LR=1e-4

# ── Stage-2 hyperparameters ───────────────────────────────────────────────────
S2_BATCH_SIZE=8
S2_ACCUM_STEPS=1
LR_ENCODER=1e-6
LR_ADAPTER=5e-5
LR_LLAMA=1.5e-5
WARMUP_STEPS=1000

# ── Shared settings ───────────────────────────────────────────────────────────
MAX_STEPS=50000          # hard cap; auto-stage transition fires earlier via eval criterion
GCS_BUCKET="gs://speechllm-data"
WANDB_PROJECT="speechllm-auto-stage"
WANDB_RUN_NAME=""        # leave empty for W&B auto-generated name

SHARDS_FILE="data/full_training_shards.txt"
TOKENIZER="data/pruned_tokenizer/"
WHISPER_CKPT="weights/whisper_small.pt"
LLAMA_CKPT="weights/Llama3.1-8B/Llama3.1-8B/Llama3.1-8B/"
ADAPTER_CKPT="checkpoints/auto-stage/stage1-final-step810.pt" 
DIAG_SHARD="data/full-eval-test-dev-clean-other.tar"
MAX_DIAG_BATCHES=10
INSTRUCTION_MODE="unformatted"
NUM_WORKERS=32
LOG_EVERY=90
CHECKPOINT_DIR="checkpoints/auto-stage"

# ── Helpers ───────────────────────────────────────────────────────────────────
mkdir -p logs "$CHECKPOINT_DIR"

_diag_args() {
    if [[ -n "$DIAG_SHARD" ]]; then
        echo "--diag_shard $DIAG_SHARD --max_diag_batches $MAX_DIAG_BATCHES"
    fi
}

_name_arg() {
    if [[ -n "$WANDB_RUN_NAME" ]]; then
        echo "--wandb_run_name $WANDB_RUN_NAME"
    fi
}

# ── Summary ───────────────────────────────────────────────────────────────────
S1_EFF=$(( S1_BATCH_SIZE * S1_ACCUM_STEPS ))
S2_EFF=$(( S2_BATCH_SIZE * S2_ACCUM_STEPS ))
LOG_FILE="logs/auto-stage-$(date +%Y%m%d-%H%M%S).log"

echo "════════════════════════════════════════════════════════"
echo "  AUTO-STAGE RUN"
echo "  Stage 1  batch=${S1_BATCH_SIZE} × accum=${S1_ACCUM_STEPS} = eff ${S1_EFF}  lr_adapter=${S1_ADAPTER_LR}"
echo "  Stage 2  batch=${S2_BATCH_SIZE} × accum=${S2_ACCUM_STEPS} = eff ${S2_EFF}"
echo "           lr_enc=${LR_ENCODER}  lr_ada=${LR_ADAPTER}  lr_llm=${LR_LLAMA}"
echo "           warmup_steps=${WARMUP_STEPS}"
echo "  checkpoint_dir: ${CHECKPOINT_DIR}"
echo "  log: ${LOG_FILE}"
echo "════════════════════════════════════════════════════════"

# shellcheck disable=SC2046
python train.py \
    --stage 1 \
    --freeze_encoder \
    --freeze_llama \
    --shards_file        data/full_training_shards.txt \
    --tokenizer          data/pruned_tokenizer/ \
    --whisper_ckpt       weights/whisper_small.pt \
    --llama_ckpt         weights/Llama3.1-8B/Llama3.1-8B/ \
    --batch_size         64 \
    --accum_steps        2 \
    --max_steps          5000 \
    --wandb_project      speechllm-auto-stage \
    --wandb_run_name     stage1-overfit-probe \
    --checkpoint_dir     checkpoints/stage1-overfit-probe \
    --save_every         1000 \
    --instruction_mode   unformatted \
    --log_every          90 \
    --num_workers        32 \
    --diag_shard         data/full-eval-test-dev-clean-other.tar \
    --max_diag_batches   10 \
    --wandb \
    --gradient_checkpointing \
    |& tee logs/stage1-overfit-probe.log

echo "[auto-stage] Uploading checkpoints …"
gsutil -m cp -r "$CHECKPOINT_DIR"/ \
    "$GCS_BUCKET/checkpoints/$(basename "$CHECKPOINT_DIR")/" || true

echo "Done. Log: $LOG_FILE"

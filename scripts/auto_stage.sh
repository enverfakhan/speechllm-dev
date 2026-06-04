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
    --auto_stage \
    --shards_file        "$SHARDS_FILE" \
    --tokenizer          "$TOKENIZER" \
    --whisper_ckpt       "$WHISPER_CKPT" \
    --adapter_ckpt       "$ADAPTER_CKPT" \
    --llama_ckpt         "$LLAMA_CKPT" \
    --s1_batch_size      "$S1_BATCH_SIZE" \
    --s1_accum_steps     "$S1_ACCUM_STEPS" \
    --s1_adapter_lr      "$S1_ADAPTER_LR" \
    --batch_size         "$S2_BATCH_SIZE" \
    --accum_steps        "$S2_ACCUM_STEPS" \
    --lr_encoder         "$LR_ENCODER" \
    --lr_adapter         "$LR_ADAPTER" \
    --lr_llama           "$LR_LLAMA" \
    --warmup_steps       "$WARMUP_STEPS" \
    --wandb_project      "$WANDB_PROJECT" \
    --checkpoint_dir     "$CHECKPOINT_DIR" \
    --save_every         1000 \
    --max_steps          "$MAX_STEPS" \
    --instruction_mode   "$INSTRUCTION_MODE" \
    --log_every          "$LOG_EVERY" \
    --num_workers        "$NUM_WORKERS" \
    --wandb \
    --gradient_checkpointing \
    $(_diag_args) \
    $(_name_arg) \
    |& tee "$LOG_FILE"

echo "[auto-stage] Uploading checkpoints …"
gsutil -m cp -r "$CHECKPOINT_DIR"/ \
    "$GCS_BUCKET/checkpoints/$(basename "$CHECKPOINT_DIR")/" || true

echo "Done. Log: $LOG_FILE"

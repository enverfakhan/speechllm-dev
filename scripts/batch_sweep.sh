#!/usr/bin/env bash
# Batch-size sweep: stage 1 (adapter only) and/or stage 2 (full fine-tune).
# Run find_max_batch.py first to set PER_GPU_BATCH_STAGE1/2 below.
#
# Usage:
#   bash scripts/batch_sweep.sh           # run both stages (default)
#   bash scripts/batch_sweep.sh --stage 1 # stage 1 only
#   bash scripts/batch_sweep.sh --stage 2 # stage 2 only (BEST_STAGE1_ADAPTER_CKPT must be set)
set -e

# ── CLI argument parsing ──────────────────────────────────────────────────────
RUN_STAGE=""   # empty = both; "1" = stage 1 only; "2" = stage 2 only
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)
            RUN_STAGE="$2"
            if [[ "$RUN_STAGE" != "1" && "$RUN_STAGE" != "2" ]]; then
                echo "ERROR: --stage must be 1 or 2 (got '$RUN_STAGE')" >&2
                exit 1
            fi
            shift 2
            ;;
        *)
            echo "ERROR: unknown argument '$1'" >&2
            echo "Usage: $0 [--stage 1|2]" >&2
            exit 1
            ;;
    esac
done

# ── Configurable variables ────────────────────────────────────────────────────
GCS_BUCKET="gs://speechllm-data"
PER_GPU_BATCH_STAGE1=64      # set from: python scripts/find_max_batch.py --stage 1
PER_GPU_BATCH_STAGE2=8      # set from: python scripts/find_max_batch.py --stage 2
MAX_STEPS_STAGE1=10
MAX_STEPS_STAGE2=10
BEST_STAGE1_ADAPTER_CKPT=""  # fill in between stages (e.g. checkpoints/stage1-bs0128/adapter-step5000.pt)
WANDB_PROJECT="speechllm-batch-sweep"
EARLY_STOP_MIN_STEPS=1

# Fixed paths — adjust if your layout differs
SHARDS_FILE="data/subset_shards.txt"
TOKENIZER="data/pruned_tokenizer/"
WHISPER_CKPT="weights/whisper_small.pt"
LLAMA_CKPT="weights/Llama3.1-8B/"
DIAG_SHARD="data/full-eval-test-dev-clean-other.tar"           # optional: path to diag shard; leave empty to omit
MAX_DIAG_BATCHES=15
INSTRUCTION_MODE="unformatted"
NUM_WORKERS=4
LOG_EVERY=50

# ── Helpers ───────────────────────────────────────────────────────────────────
mkdir -p logs checkpoints

declare -A STAGE1_TIMES   # run_name → wall-clock seconds

_banner() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $1"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

_diag_args() {
    if [[ -n "$DIAG_SHARD" ]]; then
        echo "--diag_shard $DIAG_SHARD --max_diag_batches $MAX_DIAG_BATCHES"
    fi
}

# ── Stage 1 loop ──────────────────────────────────────────────────────────────
if [[ -z "$RUN_STAGE" || "$RUN_STAGE" == "1" ]]; then
echo ""
echo "════════════════════════════════════════"
echo "  STAGE 1: adapter-only training"
echo "════════════════════════════════════════"

for multiplier in 1; do
    effective_batch=$(( PER_GPU_BATCH_STAGE1 * multiplier ))
    accum=$(( multiplier ))
    run_name="stage1-bs$(printf '%04d' $effective_batch)"
    ckpt_dir="checkpoints/$run_name"
    log_file="logs/$run_name.log"

    mkdir -p "$ckpt_dir"
    _banner "Starting $run_name  (batch=$PER_GPU_BATCH_STAGE1 × accum=$accum = effective $effective_batch)"

    t_start=$(date +%s)

    # shellcheck disable=SC2046
    python train.py \
        --shards_file        "$SHARDS_FILE" \
        --tokenizer          "$TOKENIZER" \
        --whisper_ckpt       "$WHISPER_CKPT" \
        --llama_ckpt         "$LLAMA_CKPT" \
        --freeze_encoder \
        --freeze_llama \
        --batch_size         "$PER_GPU_BATCH_STAGE1" \
        --accum_steps        "$accum" \
        --wandb_run_name     "$run_name" \
        --wandb_project      "$WANDB_PROJECT" \
        --checkpoint_dir     "$ckpt_dir" \
        --save_every         500 \
        --max_steps          "$MAX_STEPS_STAGE1" \
        --instruction_mode   "$INSTRUCTION_MODE" \
        --log_every          "$LOG_EVERY" \
        --num_workers        "$NUM_WORKERS" \
        --wandb \
        --gradient_checkpointing \
        --early_stop_metric    eval_first_token_loss \
        --early_stop_threshold 4.88 \
        --early_stop_min_steps "$EARLY_STOP_MIN_STEPS" \
        $(_diag_args) \
        |& tee "$log_file"

    t_end=$(date +%s)
    elapsed_min=$(( (t_end - t_start) / 60 ))
    STAGE1_TIMES["$run_name"]=$elapsed_min

    echo "[sweep] Uploading adapter checkpoints for $run_name …"
    gsutil -m cp -r "$ckpt_dir"/adapter-*.pt \
        "$GCS_BUCKET/checkpoints/$run_name/" || true
done

# ── Stage 1 summary ───────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo "  STAGE 1 SUMMARY"
echo "════════════════════════════════════════"
printf "%-30s  %-16s  %s\n" "run_name" "effective_batch" "wall_clock_min"
for multiplier in 1; do
    effective_batch=$(( PER_GPU_BATCH_STAGE1 * multiplier ))
    run_name="stage1-bs$(printf '%04d' $effective_batch)"
    printf "%-30s  %-16d  %d\n" "$run_name" "$effective_batch" "${STAGE1_TIMES[$run_name]:-?}"
done
echo ""
echo "Set BEST_STAGE1_ADAPTER_CKPT in this script to the adapter"
echo "checkpoint path of the best run, then press Enter to continue"
echo "to stage 2."
echo ""
echo "Example:"
echo "  BEST_STAGE1_ADAPTER_CKPT=checkpoints/stage1-bs0128/adapter-step5000.pt"
echo ""
read -p "Press Enter when ready (after setting BEST_STAGE1_ADAPTER_CKPT)..." </dev/tty

if [[ -z "$BEST_STAGE1_ADAPTER_CKPT" ]]; then
    echo "ERROR: BEST_STAGE1_ADAPTER_CKPT is not set. Edit this script and re-run from stage 2." >&2
    exit 1
fi

fi  # end stage 1 block

# ── Stage 2 loop ──────────────────────────────────────────────────────────────
if [[ -z "$RUN_STAGE" || "$RUN_STAGE" == "2" ]]; then

# When jumping straight to stage 2 (skipping stage 1 in this run), validate
# that BEST_STAGE1_ADAPTER_CKPT is set before doing any work.
if [[ -z "$BEST_STAGE1_ADAPTER_CKPT" ]]; then
    echo "ERROR: BEST_STAGE1_ADAPTER_CKPT is not set." >&2
    echo "Set it to the adapter checkpoint from your best stage-1 run and re-run with --stage 2." >&2
    exit 1
fi

echo ""
echo "════════════════════════════════════════"
echo "  STAGE 2: full fine-tune"
echo "  Adapter init: $BEST_STAGE1_ADAPTER_CKPT"
echo "════════════════════════════════════════"

declare -A STAGE2_TIMES

for multiplier in 1; do
    effective_batch=$(( PER_GPU_BATCH_STAGE2 * multiplier ))
    accum=$(( multiplier ))
    run_name="stage2-bs$(printf '%04d' $effective_batch)"
    ckpt_dir="checkpoints/$run_name"
    log_file="logs/$run_name.log"

    mkdir -p "$ckpt_dir"
    _banner "Starting $run_name  (batch=$PER_GPU_BATCH_STAGE2 × accum=$accum = effective $effective_batch)"

    t_start=$(date +%s)

    # shellcheck disable=SC2046
    python train.py \
        --shards_file        "$SHARDS_FILE" \
        --tokenizer          "$TOKENIZER" \
        --whisper_ckpt       "$WHISPER_CKPT" \
        --llama_ckpt         "$LLAMA_CKPT" \
        --adapter_ckpt       "$BEST_STAGE1_ADAPTER_CKPT" \
        --batch_size         "$PER_GPU_BATCH_STAGE2" \
        --accum_steps        "$accum" \
        --wandb_run_name     "$run_name" \
        --wandb_project      "$WANDB_PROJECT" \
        --checkpoint_dir     "$ckpt_dir" \
        --save_every         500 \
        --max_steps          "$MAX_STEPS_STAGE2" \
        --instruction_mode   "$INSTRUCTION_MODE" \
        --log_every          "$LOG_EVERY" \
        --num_workers        "$NUM_WORKERS" \
        --wandb \
        --gradient_checkpointing \
        --early_stop_metric    eval_loss \
        --early_stop_threshold 2.30 \
        --early_stop_min_steps "$EARLY_STOP_MIN_STEPS" \
        $(_diag_args) \
        |& tee "$log_file"

    t_end=$(date +%s)
    elapsed_min=$(( (t_end - t_start) / 60 ))
    STAGE2_TIMES["$run_name"]=$elapsed_min

    echo "[sweep] Uploading full checkpoint for $run_name …"
    gsutil -m cp -r "$ckpt_dir/" \
        "$GCS_BUCKET/checkpoints/$run_name/" || true
done

# ── Stage 2 summary ───────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo "  STAGE 2 SUMMARY"
echo "════════════════════════════════════════"
printf "%-30s  %-16s  %s\n" "run_name" "effective_batch" "wall_clock_min"
for multiplier in 1 2 4 8; do
    effective_batch=$(( PER_GPU_BATCH_STAGE2 * multiplier ))
    run_name="stage2-bs$(printf '%04d' $effective_batch)"
    printf "%-30s  %-16d  %d\n" "$run_name" "$effective_batch" "${STAGE2_TIMES[$run_name]:-?}"
done

echo ""
echo "Batch sweep complete."

fi  # end stage 2 block

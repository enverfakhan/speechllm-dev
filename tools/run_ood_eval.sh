#!/usr/bin/env bash
#
# Drive the whole out-of-distribution protocol for ONE checkpoint.
#
# WHY THIS EXISTS
# ---------------
# The protocol is a fixed sequence of eight commands whose arguments have to
# agree with each other exactly: the same shard paths must reach run_wer.py and
# ood_report.py, the per-set JSONL names are derived from the checkpoint's step
# number, and TED-LIUM must be decoded verbatim-only while the other two sets
# take both instruction variants.  Getting one of those wrong produces output
# that looks right — a WER against a placeholder reference, or a Δ computed
# between two different corpora — so the sequence is written down once here
# rather than retyped per run.
#
# It is eval-only and idempotent: rerunning overwrites its own outputs and
# touches nothing else.
#
# WHAT IT RUNS
#   0. preflight   — resolve the config and refuse to decode with the wrong
#                    tokenizer or input convention, which fail SILENTLY
#                    (PrunedTokenizer.encode drops ids it does not have)
#   1. smoke       — one batch per set through the identical code path
#   2. decode      — three run_wer.py invocations (see the note on grouping)
#   3. report      — ood_report.py against the banked Whisper-small control
#   4. slices      — analyze_slices.py + count_degeneracies.py per set
#
# USAGE
#   tools/run_ood_eval.sh --checkpoint checkpoints/staged-full-stack/step0018720.pt
#
#   tools/run_ood_eval.sh --checkpoint ... --dry-run      # print, run nothing
#   tools/run_ood_eval.sh --checkpoint ... --smoke-only   # 1 batch/set, then stop
#   tools/run_ood_eval.sh --checkpoint ... --skip-smoke   # straight to the full decode
#   tools/run_ood_eval.sh --checkpoint ... --max-batches 4  # cap the "full" pass
#
#   The two preflight checks that match on PATH NAMES (chat vocabulary, Instruct
#   backbone) are heuristics.  If your directories are named differently, pass
#   --force to downgrade them to warnings; the existence checks always apply.
#
#   For the chat line, pass its config — everything else is the same:
#   tools/run_ood_eval.sh --config configs/instruct-chat-3stage_resume.yaml \
#       --checkpoint checkpoints/instruct-chat-3stage/step0018720.pt --tag chat
#
set -euo pipefail

CONFIG="configs/staged_full_stack.yaml"
CHECKPOINT=""
TAG="base"
SHARDS_DIR="data/ood_shards"
CONTROL_DIR="out"
REPORT_DIR="out"
COVERAGE="out/ood-vocab-coverage.json"
VOCAB_FREQ="data/vocab_freq.json"
OUT_DIR=""
DEVICE=""
SMOKE=1
SMOKE_ONLY=0
DRY_RUN=0
MAX_BATCHES=""
FORCE=0

usage() { sed -n '3,30p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)      CONFIG="$2";      shift 2 ;;
    --checkpoint)  CHECKPOINT="$2";  shift 2 ;;
    --tag)         TAG="$2";         shift 2 ;;
    --shards-dir)  SHARDS_DIR="$2";  shift 2 ;;
    --control-dir) CONTROL_DIR="$2"; shift 2 ;;
    --report-dir)  REPORT_DIR="$2";  shift 2 ;;
    --coverage)    COVERAGE="$2";    shift 2 ;;
    --vocab-freq)  VOCAB_FREQ="$2";  shift 2 ;;
    --out-dir)     OUT_DIR="$2";     shift 2 ;;
    --device)      DEVICE="$2";      shift 2 ;;
    --max-batches) MAX_BATCHES="$2"; shift 2 ;;
    --skip-smoke)  SMOKE=0;          shift ;;
    --smoke-only)  SMOKE_ONLY=1;     shift ;;
    --dry-run)     DRY_RUN=1;        shift ;;
    --force)       FORCE=1;          shift ;;
    -h|--help)     usage 0 ;;
    *) echo "[error] unknown argument: $1" >&2; usage 1 ;;
  esac
done

[[ -n "$CHECKPOINT" ]] || { echo "[error] --checkpoint is required" >&2; usage 1; }
if [[ $SMOKE_ONLY -eq 1 && $SMOKE -eq 0 ]]; then
  echo "[error] --smoke-only and --skip-smoke contradict each other" >&2
  usage 1
fi
OUT_DIR="${OUT_DIR:-results/ood-$TAG}"

cd "$(dirname "$0")/.."

DEVICE_ARG=()
[[ -n "$DEVICE" ]] && DEVICE_ARG=(--device "$DEVICE")

# ── The four eval sets ───────────────────────────────────────────────────────
# TED-LIUM's formatted.txt is a COPY of its verbatim text (it has no cased,
# punctuated reference), so it is decoded --formats unformatted.  Scoring that
# copy would invent a reference form and report a number for it.
TED41="$SHARDS_DIR/tedlium3-test/tedlium3-test-le41.tar"
TED64="$SHARDS_DIR/tedlium3-test/tedlium3-test-le64.tar"
CV="$SHARDS_DIR/commonvoice-en-test/commonvoice-en-test-5000.tar"
E22="$SHARDS_DIR/earnings22/earnings22-2000.tar"

SET_NAMES=(tedlium3-test-le41 tedlium3-test-le64 commonvoice-en-test earnings22)
SET_PATHS=("$TED41" "$TED64" "$CV" "$E22")
CONTROL_FILES=(
  "$CONTROL_DIR/ood-tedlium-le41-whisper.jsonl"
  "$CONTROL_DIR/ood-tedlium-le64-whisper.jsonl"
  "$CONTROL_DIR/ood-commonvoice-whisper.jsonl"
  "$CONTROL_DIR/ood-earnings22-whisper.jsonl"
)

run() {
  echo "+ $*"
  if [[ $DRY_RUN -eq 0 ]]; then "$@"; fi
}

hr() { printf '\n\033[1m── %s ──\033[0m\n' "$1"; }

# ── 0. Preflight ─────────────────────────────────────────────────────────────
hr "preflight"

for i in "${!SET_NAMES[@]}"; do
  [[ -f "${SET_PATHS[$i]}" ]] || {
    echo "[error] missing shard for ${SET_NAMES[$i]}: ${SET_PATHS[$i]}" >&2
    echo "        build it with tools/prepare_ood_*.py, or pass --shards-dir" >&2
    exit 1
  }
done
[[ -f "$CHECKPOINT" ]] || { echo "[error] no such checkpoint: $CHECKPOINT" >&2; exit 1; }
[[ "$CHECKPOINT" == *-adapter.pt ]] && {
  echo "[error] $CHECKPOINT is an adapter sidecar, not a full checkpoint" >&2; exit 1; }

MISSING_CONTROL=0
for f in "${CONTROL_FILES[@]}"; do
  [[ -f "$f" ]] || { echo "[warn] control decode missing: $f"; MISSING_CONTROL=1; }
done
[[ $MISSING_CONTROL -eq 1 ]] && echo "[warn] the report will have no Δ column for those sets."

# The decode must use the SAME tokenizer and input convention the checkpoint was
# trained under.  Neither mismatch raises: PrunedTokenizer.encode silently drops
# unmapped ids, and the flat/chat assembly difference just produces a prompt the
# model never saw.  Both come out as a plausible-looking bad WER.
python - "$CONFIG" "$CHECKPOINT" "$FORCE" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, ".")
from utils.config import load_config

cfg = load_config(Path(sys.argv[1]))
ckpt = sys.argv[2]
force = sys.argv[3] == "1"

overridden = False

def fail(msg: str) -> None:
    """Abort, or warn when --force says the path-name heuristic is wrong here."""
    global overridden
    if force:
        overridden = True
        print(f"[warn] --force overrode: {msg}")
    else:
        sys.exit(
            f"[error] {msg}\n"
            "        Pass --force if this check is wrong for your layout."
        )

print(f"  config          {sys.argv[1]}")
print(f"  checkpoint      {ckpt}")
print(f"  tokenizer       {cfg.data.tokenizer}")
print(f"  convention      {cfg.model.input_convention}")
print(f"  llama_ckpt      {cfg.model.llama_ckpt}")
print(f"  eval_batch_size {cfg.metrics.eval_batch_size}")

if not Path(cfg.data.tokenizer).exists():
    sys.exit(f"[error] data.tokenizer does not exist: {cfg.data.tokenizer}")

# The chat convention needs the chat vocabulary and vice versa: a flat config
# pointed at pruned_tokenizer_chat/ would index every embedding row wrong.
chat_vocab = "chat" in str(cfg.data.tokenizer)
if (cfg.model.input_convention == "chat") != chat_vocab:
    fail(
        f"input_convention={cfg.model.input_convention!r} but "
        f"data.tokenizer={cfg.data.tokenizer} — these must match "
        "(flat → pruned_tokenizer/, chat → pruned_tokenizer_chat/)."
    )

# The flat line trained against BASE Llama, the chat line against Instruct.
if not cfg.model.stub and cfg.model.llama_ckpt:
    instruct = "instruct" in str(cfg.model.llama_ckpt).lower()
    if instruct != (cfg.model.input_convention == "chat"):
        fail(
            f"llama_ckpt={cfg.model.llama_ckpt} does not match "
            f"input_convention={cfg.model.input_convention!r} — the flat line "
            "trained against the base checkpoint, the chat line against Instruct."
        )
if overridden:
    print("  [warn] preflight overridden — you are responsible for the pairing")
else:
    print("  [ok] tokenizer, convention and backbone agree")
PY

[[ $DRY_RUN -eq 0 ]] && mkdir -p "$OUT_DIR" "$REPORT_DIR"

# ── Decode helper ────────────────────────────────────────────────────────────
# Three invocations rather than one, so every group carries a correct --dataset
# tag (one corpus per invocation) and a failure on the last set does not lose
# the summary CSV of the earlier ones.  The cost is two extra model loads.
decode() {
  local out_csv="$1" dataset="$2"; shift 2
  local formats=()
  if [[ "$1" == "--verbatim-only" ]]; then formats=(--formats unformatted); shift; fi
  run python tools/run_wer.py \
    --config "$CONFIG" \
    --checkpoints "$CHECKPOINT" \
    --eval-tar "$@" \
    --dataset "$dataset" \
    "${formats[@]}" \
    "${EXTENT[@]}" \
    "${DEVICE_ARG[@]}" \
    --output "$out_csv"
}

decode_all() {
  local dir="$1"
  [[ $DRY_RUN -eq 0 ]] && mkdir -p "$dir"
  decode "$dir/wer-tedlium.csv"     tedlium3-test       --verbatim-only \
      "tedlium3-test-le41=$TED41" "tedlium3-test-le64=$TED64"
  decode "$dir/wer-commonvoice.csv" commonvoice-en-test \
      "commonvoice-en-test=$CV"
  decode "$dir/wer-earnings22.csv"  earnings22 \
      "earnings22=$E22"
}

# ── 1. Smoke ─────────────────────────────────────────────────────────────────
# One batch per set, through the identical code path — the point is to fail in
# 2 minutes rather than 40 if a shard, a vocabulary or the checkpoint is wrong.
if [[ $SMOKE -eq 1 ]]; then
  hr "smoke — 1 batch per set"
  EXTENT=(--max-batches 1 --progress-interval 0)
  decode_all "$OUT_DIR/smoke"
  echo "smoke output → $OUT_DIR/smoke/"
  if [[ $SMOKE_ONLY -eq 1 ]]; then
    echo "--smoke-only: stopping before the full decode."
    exit 0
  fi
fi
# ── 2. Full decode ───────────────────────────────────────────────────────────
hr "full decode"
if [[ -n "$MAX_BATCHES" ]]; then
  echo "[warn] --max-batches $MAX_BATCHES: this is a PARTIAL decode. The WER is"
  echo "       over the shortest utterances only (the eval loader sorts by length),"
  echo "       so it is not comparable to the control or to any banked number."
  EXTENT=(--max-batches "$MAX_BATCHES")
else
  EXTENT=(--full)
fi
decode_all "$OUT_DIR"

# ── Locate each set's JSONL ──────────────────────────────────────────────────
# run_wer.py names it {step:07d}_{NAME}.jsonl, with the step parsed from the
# checkpoint filename.  Globbing beats re-deriving that here: it stays correct
# for a checkpoint whose name carries no step at all (run_wer falls back to the
# checkpoint index).
OURS_ARGS=()
REPORT_SETS=()
for i in "${!SET_NAMES[@]}"; do
  name="${SET_NAMES[$i]}"
  if [[ $DRY_RUN -eq 1 ]]; then
    jsonl="$OUT_DIR/<step>_$name.jsonl"
  else
    jsonl="$(ls -1 "$OUT_DIR"/*_"$name".jsonl 2>/dev/null | tail -1 || true)"
    [[ -n "$jsonl" ]] || { echo "[error] no JSONL produced for $name" >&2; exit 1; }
  fi
  OURS_ARGS+=("$name=$jsonl")
  REPORT_SETS+=("$name:$jsonl")
done

CONTROL_ARGS=()
for i in "${!SET_NAMES[@]}"; do
  [[ -f "${CONTROL_FILES[$i]}" ]] && CONTROL_ARGS+=("${SET_NAMES[$i]}=${CONTROL_FILES[$i]}")
done

# ── 3. Paired report ─────────────────────────────────────────────────────────
hr "paired report"
COVERAGE_ARG=()
if [[ -f "$COVERAGE" ]]; then
  COVERAGE_ARG=(--coverage "$COVERAGE")
else
  echo "[warn] $COVERAGE not found — rows will carry no coverage column."
  echo "       build it with tools/check_vocab_coverage.py."
fi

run python tools/ood_report.py \
  --ours "${OURS_ARGS[@]}" \
  ${CONTROL_ARGS[0]+--control} "${CONTROL_ARGS[@]}" \
  "${COVERAGE_ARG[@]}" \
  --out-md   "$REPORT_DIR/ood-$TAG-report.md" \
  --out-json "$REPORT_DIR/ood-$TAG-report.json"

# ── 4. Slices + degeneracies ─────────────────────────────────────────────────
hr "slices"
VOCAB_ARG=()
if [[ -f "$VOCAB_FREQ" ]]; then
  VOCAB_ARG=(--train-vocab "$VOCAB_FREQ")
else
  echo "[warn] $VOCAB_FREQ not found — the rare-word slice falls back to eval-set"
  echo "       hapaxes, a weaker proxy.  Build it with tools/build_vocab_freq.py."
fi

for entry in "${REPORT_SETS[@]}"; do
  name="${entry%%:*}"; jsonl="${entry#*:}"
  run python tools/analyze_slices.py \
    --jsonl "$jsonl" \
    "${VOCAB_ARG[@]}" \
    --out-json "$REPORT_DIR/ood-$TAG-$name-slices.json" \
    --out-md   "$REPORT_DIR/ood-$TAG-$name-slices.md" \
    --examples "$REPORT_DIR/ood-$TAG-$name-examples.json"
done

run python tools/count_degeneracies.py \
  --inputs "${REPORT_SETS[@]#*:}" \
  --output "$REPORT_DIR/ood-$TAG-degeneracies.csv"

# ── Done ─────────────────────────────────────────────────────────────────────
hr "done"
cat <<EOM
transcriptions  $OUT_DIR/
report          $REPORT_DIR/ood-$TAG-report.md
slices          $REPORT_DIR/ood-$TAG-*-slices.md
degeneracies    $REPORT_DIR/ood-$TAG-degeneracies.csv

Read the digit-free rows in the report on Earnings-22 (20.7% of its segments
carry digits, and FORMATTING_SPEC §6 does not expand them), and read every WER
next to its coverage column — at 49.7% utterance coverage the Earnings-22
formatted row is not a model result.
EOM

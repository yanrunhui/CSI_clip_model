#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SEED="${SEED:-0}"
DATA_DIR="${DATA_DIR:-artifacts/d2los_100k_multiconfig_aligned}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/multinumerology_generalization_holdout_nf128_expert_gate}"
TRAIN_SUFFIX="${TRAIN_SUFFIX:-train}"
TEST_SUFFIX="${TEST_SUFFIX:-test}"
SINGLE_WEIGHT="${SINGLE_WEIGHT:-1.0}"
UNCERTAINTY_RANKING_WEIGHT="${UNCERTAINTY_RANKING_WEIGHT:-0.0}"
UNCERTAINTY_RANKING_MARGIN="${UNCERTAINTY_RANKING_MARGIN:-0.25}"
UNCERTAINTY_RANKING_MIN_GAP_NS="${UNCERTAINTY_RANKING_MIN_GAP_NS:-10.0}"
GATE_WEIGHT="${GATE_WEIGHT:-0.1}"
PERIOD_PRIOR_STRENGTH="${PERIOD_PRIOR_STRENGTH:-0.0}"
FALLBACK_CONFIDENCE_THRESHOLD="${FALLBACK_CONFIDENCE_THRESHOLD:-0.5}"
OUTPUT_DIR="$OUTPUT_ROOT/seed_$SEED"
mkdir -p "$OUTPUT_DIR"

OPTIONAL_CALIBRATION_ARGS=()
TRAIN_HELP="$(python scripts/train_multinumerology_generalization.py --help 2>&1)"
if grep -q -- "--uncertainty-ranking-weight" <<<"$TRAIN_HELP"; then
  OPTIONAL_CALIBRATION_ARGS+=(
    --uncertainty-ranking-weight "$UNCERTAINTY_RANKING_WEIGHT"
    --uncertainty-ranking-margin "$UNCERTAINTY_RANKING_MARGIN"
    --uncertainty-ranking-min-gap-ns "$UNCERTAINTY_RANKING_MIN_GAP_NS"
    --period-prior-strength "$PERIOD_PRIOR_STRENGTH"
    --fallback-confidence-threshold "$FALLBACK_CONFIDENCE_THRESHOLD"
  )
elif [[ "$UNCERTAINTY_RANKING_WEIGHT" != "0" && "$UNCERTAINTY_RANKING_WEIGHT" != "0.0" ]] \
  || [[ "$PERIOD_PRIOR_STRENGTH" != "0" && "$PERIOD_PRIOR_STRENGTH" != "0.0" ]]; then
  echo "error: calibrated-gate options require the latest train_multinumerology_generalization.py" >&2
  exit 2
else
  echo "warning: using legacy expert-gate trainer without calibration-only options" >&2
fi

python scripts/train_multinumerology_generalization.py \
  --train-pair nf64 640 \
    "$DATA_DIR/d2los_100k_upa8x8_nf64_los50k_nlos50k_${TRAIN_SUFFIX}.pt" \
    nf96 960 \
    "$DATA_DIR/d2los_100k_upa8x8_nf96_los50k_nlos50k_${TRAIN_SUFFIX}.pt" \
  --train-pair nf64 640 \
    "$DATA_DIR/d2los_100k_upa8x8_nf64_los50k_nlos50k_${TRAIN_SUFFIX}.pt" \
    nf192 1920 \
    "$DATA_DIR/d2los_100k_upa8x8_nf192_los50k_nlos50k_${TRAIN_SUFFIX}.pt" \
  --train-pair nf96 960 \
    "$DATA_DIR/d2los_100k_upa8x8_nf96_los50k_nlos50k_${TRAIN_SUFFIX}.pt" \
    nf192 1920 \
    "$DATA_DIR/d2los_100k_upa8x8_nf192_los50k_nlos50k_${TRAIN_SUFFIX}.pt" \
  --train-pair nf192 1920 \
    "$DATA_DIR/d2los_100k_upa8x8_nf192_los50k_nlos50k_${TRAIN_SUFFIX}.pt" \
    nf256 2560 \
    "$DATA_DIR/d2los_100k_upa8x8_nf256_los50k_nlos50k_${TRAIN_SUFFIX}.pt" \
  --test-pair nf96 960 \
    "$DATA_DIR/d2los_100k_upa8x8_nf96_los50k_nlos50k_${TEST_SUFFIX}.pt" \
    nf128 1280 \
    "$DATA_DIR/d2los_100k_upa8x8_nf128_los50k_nlos50k_${TEST_SUFFIX}.pt" \
  --test-pair nf128 1280 \
    "$DATA_DIR/d2los_100k_upa8x8_nf128_los50k_nlos50k_${TEST_SUFFIX}.pt" \
    nf192 1920 \
    "$DATA_DIR/d2los_100k_upa8x8_nf192_los50k_nlos50k_${TEST_SUFFIX}.pt" \
  --held-out-name nf128 \
  --max-delay-ns 1920 \
  --max-delay-spread-ns 400 \
  --epochs 30 \
  --batch-size 128 \
  --single-weight "$SINGLE_WEIGHT" \
  --residue-weight 0.25 \
  --consistency-weight 0.1 \
  --direct-weight 0.25 \
  --uncertainty-weight 0.05 \
  --gate-weight "$GATE_WEIGHT" \
  --regret-weight 0.25 \
  --residual-weight 0.01 \
  --residual-scale-ns 50 \
  --view-dropout 0.1 \
  "${OPTIONAL_CALIBRATION_ARGS[@]}" \
  --seed "$SEED" \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$OUTPUT_DIR/train_test_output.txt"

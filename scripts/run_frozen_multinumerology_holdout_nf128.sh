#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SEED="${SEED:-0}"
DATA_DIR="${DATA_DIR:-artifacts/d2los_80k_multiconfig_fit70k_val10k}"
TRAIN_SUFFIX="${TRAIN_SUFFIX:-fit}"
TEST_SUFFIX="${TEST_SUFFIX:-val}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/dev_holdout_nf128_expert_gate}"

echo "frozen_final_output=paired_gate_ab_hard"
echo "frozen_training_numerologies=nf64,nf96,nf192,nf256"
echo "frozen_held_out_numerology=nf128"

env \
  SEED="$SEED" \
  DATA_DIR="$DATA_DIR" \
  TRAIN_SUFFIX="$TRAIN_SUFFIX" \
  TEST_SUFFIX="$TEST_SUFFIX" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  SINGLE_WEIGHT=1.0 \
  UNCERTAINTY_RANKING_WEIGHT=0 \
  UNCERTAINTY_RANKING_MARGIN=0.25 \
  UNCERTAINTY_RANKING_MIN_GAP_NS=10 \
  GATE_WEIGHT=0.1 \
  PERIOD_PRIOR_STRENGTH=0 \
  FALLBACK_CONFIDENCE_THRESHOLD=0 \
  bash scripts/run_multinumerology_generalization_holdout_nf128.sh

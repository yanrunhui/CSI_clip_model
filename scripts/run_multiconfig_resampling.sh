#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

D2LOS_ROOT="${D2LOS_ROOT:-$ROOT/deepmimo_scenarios/D2Los_Data}"
CANDIDATE_DIR="${CANDIDATE_DIR:-$ROOT/artifacts/d2los_400k_multiconfig_seed23421}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/artifacts/d2los_100k_multiconfig_aligned}"
SAMPLE_SEED="${SAMPLE_SEED:-23421}"
TARGET_NF="${TARGET_NF:-128}"
BANDWIDTH_HZ="${BANDWIDTH_HZ:-100000000}"

mkdir -p "$CANDIDATE_DIR/logs" "$OUTPUT_DIR"

preprocess_configuration() {
  local name="$1"
  local rows="$2"
  local cols="$3"
  local source_nf="$4"
  local output="$CANDIDATE_DIR/d2los_400k_${name}_seed${SAMPLE_SEED}.pt"

  echo "=== preprocessing ${name}: ${rows}x${cols}, source_nf=${source_nf} ==="
  python scripts/preprocess_all.py \
    --d2los-root "$D2LOS_ROOT" \
    --output "$output" \
    --max-samples 400000 \
    --d2los-sampling map_uniform \
    --d2los-sample-seed "$SAMPLE_SEED" \
    --tx-shape "$rows" "$cols" \
    --bandwidth-hz "$BANDWIDTH_HZ" \
    --total-subcarriers "$source_nf" \
    --target-nf "$TARGET_NF" \
    2>&1 | tee "$CANDIDATE_DIR/logs/${name}.log"
}

preprocess_configuration "upa8x8_nf64" 8 8 64
preprocess_configuration "upa8x8_nf96" 8 8 96
preprocess_configuration "upa8x8_nf128" 8 8 128
preprocess_configuration "upa8x8_nf192" 8 8 192
preprocess_configuration "upa8x8_nf256" 8 8 256
preprocess_configuration "upa4x4_nf128" 4 4 128
preprocess_configuration "ula64_nf128" 1 64 128

python scripts/make_aligned_balanced_splits.py \
  --input "d2los_100k_upa8x8_nf64_los50k_nlos50k=$CANDIDATE_DIR/d2los_400k_upa8x8_nf64_seed${SAMPLE_SEED}.pt" \
  --input "d2los_100k_upa8x8_nf96_los50k_nlos50k=$CANDIDATE_DIR/d2los_400k_upa8x8_nf96_seed${SAMPLE_SEED}.pt" \
  --input "d2los_100k_upa8x8_nf128_los50k_nlos50k=$CANDIDATE_DIR/d2los_400k_upa8x8_nf128_seed${SAMPLE_SEED}.pt" \
  --input "d2los_100k_upa8x8_nf192_los50k_nlos50k=$CANDIDATE_DIR/d2los_400k_upa8x8_nf192_seed${SAMPLE_SEED}.pt" \
  --input "d2los_100k_upa8x8_nf256_los50k_nlos50k=$CANDIDATE_DIR/d2los_400k_upa8x8_nf256_seed${SAMPLE_SEED}.pt" \
  --input "d2los_100k_upa4x4_nf128_los50k_nlos50k=$CANDIDATE_DIR/d2los_400k_upa4x4_nf128_seed${SAMPLE_SEED}.pt" \
  --input "d2los_100k_ula64_nf128_los50k_nlos50k=$CANDIDATE_DIR/d2los_400k_ula64_nf128_seed${SAMPLE_SEED}.pt" \
  --output-dir "$OUTPUT_DIR" \
  --samples-per-status 50000 \
  --test-fraction 0.2 \
  --seed "$SAMPLE_SEED" \
  2>&1 | tee "$OUTPUT_DIR/aligned_sampling.log"

echo "=== completed ==="
echo "candidate_dir=$CANDIDATE_DIR"
echo "output_dir=$OUTPUT_DIR"
echo "manifest=$OUTPUT_DIR/aligned_balanced_manifest.json"

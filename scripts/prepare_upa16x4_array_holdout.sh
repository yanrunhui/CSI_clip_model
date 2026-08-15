#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

D2LOS_ROOT=${D2LOS_ROOT:-$ROOT/deepmimo_scenarios/D2Los_Data}
CANDIDATE_DIR=${CANDIDATE_DIR:-$ROOT/artifacts/d2los_400k_multiconfig_seed23421}
ALIGNED_DIR=${ALIGNED_DIR:-$ROOT/artifacts/d2los_100k_multiconfig_aligned}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/artifacts/d2los_upa16x4_nf128_aligned_test}
SAMPLE_SEED=${SAMPLE_SEED:-23421}

candidate="$CANDIDATE_DIR/d2los_400k_upa16x4_nf128_seed${SAMPLE_SEED}.pt"
reference="$ALIGNED_DIR/d2los_100k_upa8x8_nf128_los50k_nlos50k_test.pt"

mkdir -p "$CANDIDATE_DIR/logs" "$OUTPUT_DIR"
if [[ ! -f "$candidate" ]]; then
  python scripts/preprocess_all.py \
    --d2los-root "$D2LOS_ROOT" \
    --output "$candidate" \
    --max-samples 400000 \
    --d2los-sampling map_uniform \
    --d2los-sample-seed "$SAMPLE_SEED" \
    --tx-shape 16 4 \
    --bandwidth-hz 100000000 \
    --total-subcarriers 128 \
    --target-nf 128 \
    2>&1 | tee "$CANDIDATE_DIR/logs/upa16x4_nf128.log"
fi

python scripts/make_paired_configuration_subset.py \
  --input "upa8x8_reference=$reference" \
  --input "upa16x4=$candidate" \
  --output-dir "$OUTPUT_DIR" \
  --seed "$SAMPLE_SEED"

echo "saved_upa16x4_test=$OUTPUT_DIR/upa16x4_paired.pt"

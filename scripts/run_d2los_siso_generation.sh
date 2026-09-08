#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

D2LOS_ROOT="${D2LOS_ROOT:-/home/yrh/CSI_model/CSI_model/deepmimo_scenarios/D2Los_Data}"
OUTPUT="${OUTPUT:-$ROOT/artifacts/d2los_siso_1x1_bw100m_nf128_seed23421.pt}"
MAX_SAMPLES="${MAX_SAMPLES:-400000}"
SAMPLE_SEED="${SAMPLE_SEED:-23421}"
BANDWIDTH_HZ="${BANDWIDTH_HZ:-100000000}"
SUBCARRIERS="${SUBCARRIERS:-128}"
TARGET_NF="${TARGET_NF:-128}"

mkdir -p "$(dirname "$OUTPUT")"

python scripts/generate_d2los_siso.py \
  --d2los-root "$D2LOS_ROOT" \
  --output "$OUTPUT" \
  --max-samples "$MAX_SAMPLES" \
  --sampling map_uniform \
  --sample-seed "$SAMPLE_SEED" \
  --bandwidth-hz "$BANDWIDTH_HZ" \
  --subcarriers "$SUBCARRIERS" \
  --target-nf "$TARGET_NF" \
  2>&1 | tee "${OUTPUT%.pt}.log"

echo "siso_pt=$OUTPUT"

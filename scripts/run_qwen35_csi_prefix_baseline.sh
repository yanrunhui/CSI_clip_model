#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-smoke}"
GPU="${GPU:-1}"
MODEL="${MODEL:-models/Qwen3.5-2B}"
TRAIN="${TRAIN:-artifacts/d2los_100k_multiconfig_aligned/d2los_100k_upa8x8_nf128_los50k_nlos50k_train.pt}"
TEST="${TEST:-artifacts/d2los_100k_multiconfig_aligned/d2los_100k_upa8x8_nf128_los50k_nlos50k_test.pt}"
REFERENCE_MAPPING="${REFERENCE_MAPPING:-artifacts/qwen3_1p7b_explicit_csi_prefix/seed_0_gpu0/mapping_final.pt}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

case "$MODE" in
  smoke)
    STEPS="${STEPS:-5}"
    LIMIT="${LIMIT:-2}"
    OUT="${OUT:-artifacts/qwen3_5_2b_explicit_csi_prefix/smoke_2samples}"
    ;;
  full)
    STEPS="${STEPS:-500}"
    LIMIT="${LIMIT:-1000}"
    OUT="${OUT:-artifacts/qwen3_5_2b_explicit_csi_prefix/seed_0}"
    ;;
  *)
    echo "MODE must be smoke or full, got: $MODE" >&2
    exit 1
    ;;
esac

for path in "$MODEL/config.json" "$TRAIN" "$TEST" "$REFERENCE_MAPPING"; do
  test -f "$path" || { echo "missing input: $path" >&2; exit 1; }
done

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$OUT/evaluation/text_metrics"
rm -f "$OUT/BENCHMARK_COMPLETE"

python scripts/validate_qwen35_csi_prefix.py --model-path "$MODEL"

python scripts/train_qwen_csi_prefix.py \
  --model-path "$MODEL" \
  --train-data "$TRAIN" \
  --output-dir "$OUT" \
  --controlled-config-from "$REFERENCE_MAPPING" \
  --max-steps "$STEPS" \
  --seed 0 \
  > "$OUT/train_output.txt" 2>&1

test -f "$OUT/mapping_final.pt" || {
  echo "training did not produce mapping_final.pt" >&2
  exit 1
}
test -f "$OUT/qwen_adapter/adapter_config.json" || {
  echo "training did not produce the expected LoRA adapter" >&2
  exit 1
}

python scripts/evaluate_qwen_csi_prefix.py \
  --model-path "$MODEL" \
  --mapping-checkpoint "$OUT/mapping_final.pt" \
  --adapter-path "$OUT/qwen_adapter" \
  --data-path "$TEST" \
  --output-jsonl "$OUT/evaluation/predictions.jsonl" \
  --data-jsonl "$OUT/evaluation/targets.jsonl" \
  --load-in-4bit \
  --compute-dtype bfloat16 \
  --batch-size 1 \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --limit "$LIMIT" \
  --log-every 1 \
  > "$OUT/evaluation/evaluate_output.txt" 2>&1

python scripts/build_qwen_signal_description_payload.py \
  --data-jsonl "$OUT/evaluation/targets.jsonl" \
  --predictions-jsonl "$OUT/evaluation/predictions.jsonl" \
  --output "$OUT/evaluation/signal_descriptions.pt"

python scripts/evaluate_signal_descriptions.py \
  --payload "$OUT/evaluation/signal_descriptions.pt" \
  --output-dir "$OUT/evaluation/text_metrics" \
  > "$OUT/evaluation/text_metrics/evaluate_text_output.txt" 2>&1

python - \
  "$OUT/evaluation/predictions.jsonl.summary.json" \
  "$OUT/evaluation/predictions.jsonl" \
  "$LIMIT" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
prediction_path = Path(sys.argv[2])
expected = int(sys.argv[3])
if int(summary.get("sample_count", -1)) != expected:
    raise SystemExit(f"sample_count mismatch: {summary}")
if float(summary.get("parse_success_rate", -1.0)) != 1.0:
    print("qwen35_json_parse_failures:")
    for line in prediction_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("parse_error") is not None:
            print(
                json.dumps(
                    {
                        "index": row.get("index"),
                        "parse_error": row.get("parse_error"),
                        "generated_text": row.get("generated_text"),
                    },
                    ensure_ascii=True,
                )
            )
    raise SystemExit(f"parse_success_rate is not 1.0: {summary}")
print("qwen35_generation_validation=passed")
PY

touch "$OUT/BENCHMARK_COMPLETE"
echo "qwen35_${MODE}_complete=$OUT"

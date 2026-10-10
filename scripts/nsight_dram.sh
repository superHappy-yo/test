#!/usr/bin/env bash
# Optional DRAM-byte capture for the wide FP16 matmul list.
#
# Requires Nsight Compute (`ncu`) and a CUDA PyTorch. This script does not
# run unless you invoke it, or set TRY_NSIGHT=1 for scripts/run_a40.sh.
#
# Metric names change across Nsight versions. The default is
#   dram__bytes_read.sum,dram__bytes_write.sum
# Override with NCU_METRICS. If ncu rejects the names:
#   ncu --query-metrics | grep dram__bytes
#
# Output: $OUT/nsight_dram.json with status NSIGHT_DRAM_PROCESS_SUM.
# Each shape is its own process, and every kernel in that process is summed,
# including copies. Do not relabel the status to NSIGHT_DRAM unless you have
# filtered to the matmul kernel. torch.profiler memory is a different file
# and is not DRAM.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
OUT="${OUT:-$ROOT/results/a40}"
METRICS="${NCU_METRICS:-dram__bytes_read.sum,dram__bytes_write.sum}"
mkdir -p "$OUT"

if ! command -v ncu >/dev/null 2>&1; then
  echo "ERROR: ncu not on PATH."
  echo "Install Nsight Compute, or leave TRY_NSIGHT unset and use --profile torch."
  echo "torch.profiler memory is not DRAM bytes."
  echo "List metrics once ncu exists: ncu --query-metrics | grep dram__bytes"
  exit 1
fi

if ! "$PYTHON" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  echo "ERROR: PyTorch CUDA is required for the Nsight worker."
  exit 1
fi

JSON="$OUT/nsight_dram.json"
rm -f "$JSON"
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

while read -r label m n k; do
  echo "ncu: $label ${m}x${n}x${k}"
  if ! ncu --csv --metrics "$METRICS" \
      "$PYTHON" -m shapefuse.nsight_worker --label "$label" --m "$m" --n "$n" --k "$k" \
      >"$TMP"; then
    echo "ERROR: ncu failed for $label."
    echo "Check metric names: ncu --query-metrics | grep dram__bytes"
    echo "Override with NCU_METRICS=metric.a,metric.b"
    exit 1
  fi
  "$PYTHON" -m shapefuse.nsight_worker \
    --label "$label" --m "$m" --n "$n" --k "$k" \
    --parse "$TMP" --out "$JSON" --metrics "$METRICS"
done < <("$PYTHON" -m shapefuse.nsight_worker --list-shapes)

echo "wrote $JSON"
echo "status is NSIGHT_DRAM_PROCESS_SUM (process-wide, not an isolated kernel)."

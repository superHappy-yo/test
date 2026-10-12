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
#
# ERR_NVGPUCTRPERM means ncu is installed but this user cannot read GPU
# performance counters. That is not a bad metric name. Admin steps:
# docs/NSIGHT_PERMISSIONS.md (modprobe NVreg_RestrictProfilingToAdminUsers=0
# or /etc/modprobe.d/, then reboot). The failing ncu text is saved to
# $OUT/nsight_error.log.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
OUT="${OUT:-$ROOT/results/a40}"
METRICS="${NCU_METRICS:-dram__bytes_read.sum,dram__bytes_write.sum}"
mkdir -p "$OUT"

if ! command -v ncu >/dev/null 2>&1; then
  echo "ERROR: ncu not on PATH."
  echo "Nsight Compute (ncu) is a separate package from the CUDA driver and from nvidia-smi."
  echo "Install it, then confirm with: ncu --version"
  echo "The matmul sweep does not need ncu. DRAM bytes and achieved AI do."
  echo "torch.profiler memory is not DRAM bytes, and the copy-kernel probe is not Nsight."
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

ERR_LOG="$OUT/nsight_error.log"
rm -f "$ERR_LOG"

while read -r label m n k; do
  echo "ncu: $label ${m}x${n}x${k}"
  set +e
  ncu --csv --metrics "$METRICS" \
      "$PYTHON" -m shapefuse.nsight_worker --label "$label" --m "$m" --n "$n" --k "$k" \
      >"$TMP" 2>"$ERR_LOG"
  ncu_status=$?
  set -e
  if [[ -s "$TMP" ]] && grep -q "ERR_NVGPUCTRPERM" "$TMP"; then
    cat "$TMP" >>"$ERR_LOG"
  fi
  if [[ "$ncu_status" -ne 0 ]]; then
    echo "ncu failed for $label (exit ${ncu_status}). Log: $ERR_LOG"
    if [[ -f "$ERR_LOG" ]]; then
      cat "$ERR_LOG"
    fi
    if [[ -f "$ERR_LOG" ]] && grep -q "ERR_NVGPUCTRPERM" "$ERR_LOG"; then
      echo "ERROR: ERR_NVGPUCTRPERM — this user cannot access NVIDIA GPU performance counters."
      echo "ncu is installed. This is not a missing metric name. achieved_ai was not filled."
      echo "An administrator must allow profiling, then reboot. Steps: docs/NSIGHT_PERMISSIONS.md"
      echo "https://developer.nvidia.com/ERR_NVGPUCTRPERM"
      echo "After reboot, verify ncu and re-run: TRY_NSIGHT=1 bash scripts/run_a40.sh"
      echo "torch.profiler memory is not DRAM bytes."
    else
      echo "ERROR: ncu failed for $label."
      echo "Check metric names: ncu --query-metrics | grep dram__bytes"
      echo "Override with NCU_METRICS=metric.a,metric.b"
    fi
    exit 1
  fi
  "$PYTHON" -m shapefuse.nsight_worker \
    --label "$label" --m "$m" --n "$n" --k "$k" \
    --parse "$TMP" --out "$JSON" --metrics "$METRICS"
done < <("$PYTHON" -m shapefuse.nsight_worker --list-shapes)

echo "wrote $JSON"
echo "status is NSIGHT_DRAM_PROCESS_SUM (process-wide, not an isolated kernel)."

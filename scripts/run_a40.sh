#!/usr/bin/env bash
# NVIDIA A40 entry point. One command for the next server run.
#
# Checks nvidia-smi and PyTorch CUDA, warns if the device name is not an A40,
# then writes analytical tables against the A40 datasheet ridge plus:
#   - torch.matmul latencies (PARTIAL_VENDOR_MATMUL, not fused plans)
#   - torch.profiler allocator-memory proxy (not DRAM, not achieved AI)
#   - matmul+RMSNorm and bias+GELU proxies (not MemMode / ComputeMode)
#   - dual-plan decision_log.csv and agreement.json
#
# Nsight DRAM bytes are still missing from the archived A40 package
# (nsight_status TBD_RUN_PROFILER, quote_as_scic_accuracy false).
# Strongly recommended on the next server run:
#   TRY_NSIGHT=1 bash scripts/run_a40.sh
# ncu is Nsight Compute, separate from the NVIDIA driver. If `ncu` is not on
# PATH the capture is skipped, a note is written to nsight_skipped.txt, and
# the matmul sweep still runs. torch.profiler memory is not a substitute.
# achieved_ai stays empty until ncu succeeds.
#
# A copy-kernel bandwidth estimate is also written when CUDA works. Its
# status is ESTIMATE_COPY_NOT_NSIGHT. It is not a measured ridge.
#
# Datasheet anchor defaults to A40_FP16. Override with MACHINE= only if you
# mean a different datasheet. The physical GPU is whatever nvidia-smi shows.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
OUT="${OUT:-$ROOT/results/a40}"
MACHINE="${MACHINE:-A40_FP16}"

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  "$PYTHON" -m pip install -r requirements.txt
  "$PYTHON" -m pip install -e .
fi

"$PYTHON" -m shapefuse selfcheck

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Refusing to write a results/ that could be mistaken for an A40."
  echo "CPU analytical path: bash scripts/run_cpu.sh"
  exit 1
fi

nvidia-smi

if ! "$PYTHON" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  echo "ERROR: PyTorch with CUDA is not importable."
  echo "Install a CUDA wheel that matches this driver (see https://pytorch.org/get-started/locally/),"
  echo "then rerun with SKIP_INSTALL=1 so this script does not replace that wheel."
  exit 1
fi

GPU_NAME="$("$PYTHON" -c "import torch; print(torch.cuda.get_device_name(0))")"
echo "CUDA device: ${GPU_NAME}"
case "$GPU_NAME" in
  *A40*) ;;
  *)
    echo "WARNING: device name does not contain A40 (${GPU_NAME})."
    echo "The script will continue. Datasheet anchor is ${MACHINE}."
    echo "Do not relabel these rows as H100 or A100 measurements."
    ;;
esac

echo
echo "==== Nsight DRAM: strongly recommended (TRY_NSIGHT=1) ===="
echo "The archived A40 package has no DRAM bytes. SCIC accuracy stays"
echo "quote_as_scic_accuracy=false until an isolated Nsight capture exists."
echo "Recommended command: TRY_NSIGHT=1 bash scripts/run_a40.sh"
mkdir -p "$OUT"
if command -v ncu >/dev/null 2>&1; then
  echo "ncu: found at $(command -v ncu)"
  if [[ "${TRY_NSIGHT:-0}" != "1" ]]; then
    echo "WARNING: ncu is installed but TRY_NSIGHT is not 1, so this run will NOT capture DRAM bytes."
    echo "Re-run with TRY_NSIGHT=1. Metric names: ncu --query-metrics | grep dram__bytes"
  fi
else
  echo "ncu: NOT on PATH."
  echo "Nsight Compute is not the same thing as the CUDA driver or nvidia-smi."
  echo "Install the Nsight Compute package that matches this CUDA toolkit, then check: ncu --version"
  echo "Until ncu exists, DRAM bytes stay empty. Do not fill them from torch.profiler."
  echo "List metrics after install: ncu --query-metrics | grep dram__bytes"
fi

NSIGHT_ARGS=()
if [[ "${TRY_NSIGHT:-0}" == "1" ]]; then
  if ! command -v ncu >/dev/null 2>&1; then
    echo "WARNING: TRY_NSIGHT=1 but ncu is missing. Skipping DRAM capture."
    cat >"$OUT/nsight_skipped.txt" <<EOF
TRY_NSIGHT=1 was set, but ncu was not on PATH.
Nsight Compute is a separate install from the NVIDIA driver.
The matmul sweep continues without DRAM bytes.
torch.profiler memory and the copy-kernel bandwidth estimate are not substitutes.
Install Nsight Compute, confirm with: ncu --version
Then: ncu --query-metrics | grep dram__bytes
Re-run: TRY_NSIGHT=1 bash scripts/run_a40.sh
EOF
    echo "Wrote $OUT/nsight_skipped.txt"
  elif OUT="$OUT" PYTHON="$PYTHON" bash "$ROOT/scripts/nsight_dram.sh"; then
    NSIGHT_ARGS=(--nsight-json "$OUT/nsight_dram.json")
    rm -f "$OUT/nsight_skipped.txt"
  else
    echo "WARNING: Nsight capture failed. Continuing without DRAM bytes."
    echo "torch.profiler output is allocator memory, not achieved AI."
    echo "If ncu rejected metric names: ncu --query-metrics | grep dram__bytes"
    echo "Override with NCU_METRICS=metric.a,metric.b"
    cat >"$OUT/nsight_skipped.txt" <<EOF
TRY_NSIGHT=1 was set and ncu was on PATH, but scripts/nsight_dram.sh failed.
achieved_ai was not filled. See the script output above.
Check: ncu --query-metrics | grep dram__bytes
Override metric names with NCU_METRICS.
torch.profiler memory is not DRAM bytes.
EOF
  fi
else
  echo "TRY_NSIGHT is not 1. nsight_dram.json will not be written."
fi

"$PYTHON" -m shapefuse all \
  --out "$OUT" \
  --device cuda \
  --require-cuda \
  --bench-level a40 \
  --machine "$MACHINE" \
  --profile torch \
  --fusion-proxy \
  "${NSIGHT_ARGS[@]}"

echo
echo "==== Copy-kernel bandwidth estimate (NOT Nsight) ===="
if "$PYTHON" -m shapefuse.bandwidth_probe --out "$OUT/bandwidth_estimate.json" --machine "$MACHINE"; then
  echo "bandwidth_estimate.json status is ESTIMATE_COPY_NOT_NSIGHT."
  echo "Do not copy estimated_ridge_flops_per_byte into measured_ridge_flops_per_byte."
else
  echo "WARNING: bandwidth probe did not write an estimate. Measured ridge stays empty."
fi

echo
echo "==== Send these files back for the paper ===="
echo "Directory: $OUT"
echo "Nsight is still the gap that blocks SCIC accuracy. Prefer TRY_NSIGHT=1 when ncu exists."
cat "$OUT/SEND_BACK.txt"

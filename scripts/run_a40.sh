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
# Nsight DRAM bytes are optional because ncu often needs extra privileges:
#   TRY_NSIGHT=1 bash scripts/run_a40.sh
# If ncu fails, the matmul sweep still runs and achieved_ai stays empty.
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

NSIGHT_ARGS=()
if [[ "${TRY_NSIGHT:-0}" == "1" ]]; then
  if OUT="$OUT" PYTHON="$PYTHON" bash "$ROOT/scripts/nsight_dram.sh"; then
    NSIGHT_ARGS=(--nsight-json "$OUT/nsight_dram.json")
  else
    echo "WARNING: Nsight capture failed. Continuing without DRAM bytes."
    echo "torch.profiler output is allocator memory, not achieved AI."
  fi
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
echo "==== Send these files back for the paper ===="
echo "Directory: $OUT"
cat "$OUT/SEND_BACK.txt"

#!/usr/bin/env bash
# Tiny GDDR bandwidth estimate from a device-to-device copy.
#
# This is ESTIMATE_COPY_NOT_NSIGHT. It is not Nsight Compute, it does not
# read DRAM counters, and it must not be copied into
# measured_ridge_flops_per_byte. The ridge numerator is the datasheet FLOP
# peak of MACHINE (default A40_FP16).
#
#   bash scripts/bandwidth_probe.sh
#   OUT=results/a40 MACHINE=A40_FP16 bash scripts/bandwidth_probe.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
OUT="${OUT:-$ROOT/results/a40}"
MACHINE="${MACHINE:-A40_FP16}"
mkdir -p "$OUT"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Not writing a bandwidth estimate."
  exit 1
fi

nvidia-smi -L || true

"$PYTHON" -m shapefuse.bandwidth_probe --out "$OUT/bandwidth_estimate.json" --machine "$MACHINE"
echo "status in that file is ESTIMATE_COPY_NOT_NSIGHT."
echo "Do not treat estimated_ridge_flops_per_byte as a measured ridge."

#!/usr/bin/env bash
# H100 (or any CUDA) entry point.
# Analytical tables always come from the datasheet Roofline model.
# bench_gemm.csv times torch.matmul only. It does not time fused plans.
# Nsight bytes, measured ridge, and MemMode vs ComputeMode kernels stay TBD.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
OUT="${OUT:-$ROOT/results}"

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  "$PYTHON" -m pip install -r requirements.txt
  "$PYTHON" -m pip install -e .
fi

"$PYTHON" -m shapefuse selfcheck

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Refusing to write a results/ that could be mistaken for H100."
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

"$PYTHON" -m shapefuse all \
  --out "$OUT" \
  --device cuda \
  --require-cuda \
  --bench-level h100 \
  --machine "${MACHINE:-H100_SXM_FP16}"

echo
echo "==== Send these files back for the paper ===="
cat "$OUT/SEND_BACK.txt"

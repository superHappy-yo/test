#!/usr/bin/env bash
# One-command Docker bakeoff. No Nsight.
#
# Times MatMul+RMSNorm and an attention microbench across decode and prefill
# shapes. Baselines: eager unfused, torch.compile default, torch.compile
# max-autotune when it succeeds, library scaled_dot_product_attention,
# always-fuse, always-unfuse, and the path SCIC selects.
#
# The regime comes from the datasheet ridge (--machine, default A40_FP16).
# It is not a measured ridge. quote_as_scic_accuracy stays false.
# SDPA rows are not FlashAttention-3. always-fuse / always-unfuse are not
# custom MemMode or ComputeMode kernels. Stage-Stream is not run.
#
# Library attention scores compile_sdpa_chain (always_fuse) for the
# epilogue. GEMM still unfuses on the compute side of the ridge.
# Run on an idle GPU. The bakeoff raises Dynamo cache_size_limit
# (default 64, override with DYNAMO_CACHE_SIZE_LIMIT). The archived
# A40 run was not idle and hit the default limit of 8.
#
# Requires nvidia-smi and a CUDA build of PyTorch already installed.
# This script does not install PyTorch and does not download a wheel.
#
#   bash scripts/run_baseline_bakeoff.sh
#   SKIP_INSTALL=1 bash scripts/run_baseline_bakeoff.sh
#   SKIP_MAX_AUTOTUNE=1 bash scripts/run_baseline_bakeoff.sh
#   SKIP_ATTENTION=1 bash scripts/run_baseline_bakeoff.sh
#   MACHINE=H100_SXM_FP16 bash scripts/run_baseline_bakeoff.sh
#
# Send back the output directory. The list is results/bakeoff/SEND_BACK.txt.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
OUT="${OUT:-$ROOT/results/bakeoff}"
MACHINE="${MACHINE:-A40_FP16}"
LEVEL="${LEVEL:-decode-prefill}"

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  "$PYTHON" -m pip install -r requirements.txt
  "$PYTHON" -m pip install -e .
fi

"$PYTHON" -m shapefuse selfcheck

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. This bakeoff is the CUDA wall-clock run."
  echo "Schema check without a GPU: python -m shapefuse bakeoff --device cpu --level smoke --attention --skip-max-autotune --out results/bakeoff-smoke"
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
echo "Datasheet anchor: ${MACHINE} (not a measured ridge)."
echo "Idle GPU recommended. A busy device changes these latencies."
echo "The archived A40 run was not idle (see paper/data/bakeoff/RUN_NOTES.txt)."
echo "Dynamo cache_size_limit is raised inside the bakeoff (default 64)."
echo "Override with DYNAMO_CACHE_SIZE_LIMIT. Graphs are still not inspected."
echo "Nsight is not invoked. achieved AI and SCIC accuracy are not outputs of this script."
echo "Attention epilogue policy is compile_sdpa_chain (always_fuse). Re-run after that change before quoting it."

mkdir -p "$OUT"
nvidia-smi >"$OUT/nvidia_smi.txt"
"$PYTHON" -c "import torch, sys; print(torch.__version__); print('cuda', torch.version.cuda); print(torch.cuda.get_device_name(0)); print(sys.version)" \
  >"$OUT/torch_version.txt"

ARGS=(
  bakeoff
  --out "$OUT"
  --device cuda
  --require-cuda
  --machine "$MACHINE"
  --level "$LEVEL"
  --skip-plots
)
if [[ "${SKIP_ATTENTION:-0}" != "1" ]]; then
  ARGS+=(--attention)
fi
if [[ "${SKIP_MAX_AUTOTUNE:-0}" == "1" ]]; then
  echo "max-autotune: skipped (SKIP_MAX_AUTOTUNE=1). The CSV row is marked skipped."
  ARGS+=(--skip-max-autotune)
else
  echo "max-autotune: attempting torch.compile mode=max-autotune."
  echo "If it fails, that baseline is marked failed and the other rows are kept."
  echo "Set SKIP_MAX_AUTOTUNE=1 to skip it."
fi

"$PYTHON" -m shapefuse "${ARGS[@]}"

echo
echo "Bakeoff directory: ${OUT}"
echo "Send that directory back. See ${OUT}/SEND_BACK.txt"
echo "Do not fill empty medians. Do not quote the CSV as SCIC accuracy or as FlashAttention-3."

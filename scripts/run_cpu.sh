#!/usr/bin/env bash
# No-GPU path. Writes analytical CSV/JSON/figures plus a tiny CPU matmul
# placeholder. Do not cite those latencies as H100 numbers.
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
"$PYTHON" -m shapefuse all --out "$OUT" --device cpu --bench-level smoke --cpu-proxy

echo
cat "$OUT/SEND_BACK.txt"

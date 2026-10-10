"""One-matmul process for ``scripts/nsight_dram.sh``.

Nsight Compute should wrap this module. The process does a single FP16
``torch.matmul`` and exits. The shell script sums DRAM counters across that
process; those counters can include copies, so the JSON status is
``NSIGHT_DRAM_PROCESS_SUM`` unless you edit the file by hand to
``NSIGHT_DRAM`` after isolating one kernel.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from shapefuse.bench import WIDE_SHAPES
from shapefuse.profiler_hooks import NSIGHT_PROCESS_SUM, parse_ncu_csv


def run_one(m: int, n: int, k: int) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("nsight worker needs PyTorch with CUDA")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    a = torch.randn(m, k, generator=generator, dtype=torch.float32).to(device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, generator=generator, dtype=torch.float32).to(device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    c = a @ b
    torch.cuda.synchronize()
    if not torch.isfinite(c).all():
        raise SystemExit("matmul produced a non-finite value")


def accumulate(out: Path, label: str, m: int, n: int, k: int, csv_text: str, metrics: list[str]) -> None:
    found = parse_ncu_csv(csv_text, metrics)
    missing = [metric for metric in metrics if metric not in found]
    if missing:
        raise SystemExit(
            "ncu CSV is missing "
            + ", ".join(missing)
            + ". List names with: ncu --query-metrics | grep dram__bytes"
        )
    dram_bytes = float(sum(found.values()))
    payload = {
        "status": NSIGHT_PROCESS_SUM,
        "metrics": metrics,
        "note": (
            "Sum of the named Nsight metrics over every kernel in one short "
            "process per shape. Extra launches (copies) are included. This is "
            "not an isolated kernel counter and not torch allocator memory. "
            "Change status to NSIGHT_DRAM only after you have filtered to the "
            "matmul kernel."
        ),
        "rows": [],
    }
    if out.is_file():
        payload = json.loads(out.read_text(encoding="utf-8"))
    rows = [row for row in payload.get("rows", []) if row.get("label") != label]
    rows.append(
        {
            "label": label,
            "M": m,
            "N": n,
            "K": k,
            "dram_bytes": dram_bytes,
            "status": NSIGHT_PROCESS_SUM,
            "metric_values": found,
        }
    )
    payload["rows"] = rows
    payload["status"] = NSIGHT_PROCESS_SUM
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One matmul, or fold an ncu CSV into nsight_dram.json")
    parser.add_argument("--label")
    parser.add_argument("--m", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--parse", type=Path, help="ncu CSV to fold into --out")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--metrics", default="dram__bytes_read.sum,dram__bytes_write.sum")
    parser.add_argument("--list-shapes", action="store_true")
    args = parser.parse_args(argv)
    if args.list_shapes:
        for label, m, n, k in WIDE_SHAPES:
            print(f"{label} {m} {n} {k}")
        return 0
    if args.label is None or args.m is None or args.n is None or args.k is None:
        parser.error("--label --m --n --k are required unless --list-shapes")
    if args.parse is not None:
        if args.out is None:
            parser.error("--parse requires --out")
        text = args.parse.read_text(encoding="utf-8", errors="replace")
        metrics = [part.strip() for part in args.metrics.split(",") if part.strip()]
        accumulate(args.out, args.label, args.m, args.n, args.k, text, metrics)
        return 0
    run_one(args.m, args.n, args.k)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Command line for analytical sweeps and optional matmul timings."""

from __future__ import annotations

import argparse
import sys

from shapefuse import __version__
from shapefuse.machines import MACHINES


def build_parser() -> argparse.ArgumentParser:
    machines = ", ".join(sorted(MACHINES))
    parser = argparse.ArgumentParser(
        prog="shapefuse",
        description=(
            "Shape-conditioned intensity classification and a dual-plan fusion "
            "selector (MemMode / ComputeMode). Analytical Roofline by default. "
            "Does not implement Stage-Stream and does not emit fused kernels."
        ),
    )
    parser.add_argument(
        "cmd",
        nargs="?",
        default="all",
        choices=("all", "analytical", "bench", "plans", "selfcheck"),
        help="all = tables + plans + figures + a matmul timing (default)",
    )
    parser.add_argument("--out", default="results", help="Output directory")
    parser.add_argument("--machine", default="H100_SXM_FP16", help=f"Ridge anchor. One of: {machines}")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--bench-level",
        default="smoke",
        choices=("smoke", "default", "h100"),
        help="h100 is a 4096-wide matmul sweep and is refused on CPU",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Exit if PyTorch CUDA is not available",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="If CUDA was requested and is missing, time on CPU and mark placeholders",
    )
    parser.add_argument(
        "--cpu-proxy",
        action="store_true",
        help="Also time a short NumPy FP32 M-sweep (proxy, not H100)",
    )
    parser.add_argument(
        "--launch-overhead-us",
        type=float,
        default=5.0,
        help="Sensitivity knob for epilogue speedup. Not a measurement. Use 0 to disable.",
    )
    parser.add_argument("--mem-frac", type=float, default=0.5, help="Memory band is AI < mem_frac * R")
    parser.add_argument(
        "--compute-frac",
        type=float,
        default=2.0,
        help="Compute band is AI > compute_frac * R",
    )
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--version", action="version", version=f"shapefuse {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "selfcheck":
        from shapefuse.selfcheck import run_selfcheck

        run_selfcheck()
        return 0
    from shapefuse.runner import run_command

    return run_command(args)


if __name__ == "__main__":
    sys.exit(main())

"""Command line for analytical sweeps and optional matmul timings."""

from __future__ import annotations

import argparse
import sys

from shapefuse import __version__
from shapefuse.machines import ALIASES, MACHINES


def build_parser() -> argparse.ArgumentParser:
    machines = ", ".join(sorted(MACHINES))
    aliases = ", ".join(f"{alias}->{ALIASES[alias]}" for alias in sorted(ALIASES))
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
        choices=("all", "analytical", "bench", "plans", "selfcheck", "bakeoff"),
        help=(
            "all = tables + plans + figures + a matmul timing (default). "
            "bakeoff = wall-clock subgraph latency and the SCIC-selected path. "
            "No Nsight."
        ),
    )
    parser.add_argument("--out", default="results", help="Output directory")
    parser.add_argument(
        "--machine",
        default="H100_SXM_FP16",
        help=f"Datasheet ridge anchor. Canonical: {machines}. Aliases: {aliases}.",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--bench-level",
        default="smoke",
        choices=("smoke", "default", "h100", "a40"),
        help="h100 and a40 share the 4096-wide matmul list and are refused on CPU",
    )
    parser.add_argument(
        "--profile",
        default="off",
        choices=("off", "torch"),
        help=(
            "torch records allocator memory via torch.profiler and stores it as "
            "proxy_ai. That is not DRAM and not achieved_ai."
        ),
    )
    parser.add_argument(
        "--nsight-json",
        default=None,
        help="JSON from scripts/nsight_dram.sh. Fills achieved_ai. Does not run ncu itself.",
    )
    parser.add_argument(
        "--fusion-proxy",
        action="store_true",
        help=(
            "Time matmul+RMSNorm and bias+GELU proxies. Not MemMode or ComputeMode. "
            "See fusion_proxy.csv honesty column."
        ),
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
    parser.add_argument(
        "--level",
        default="decode-prefill",
        choices=("smoke", "decode-prefill"),
        help=(
            "Bakeoff shape list. smoke is a CPU schema check. "
            "decode-prefill is the CUDA MatMul+RMSNorm and attention list."
        ),
    )
    parser.add_argument(
        "--attention",
        action="store_true",
        help="Also time the attention microbench (materialized eager, SDPA, fuse policy).",
    )
    parser.add_argument(
        "--skip-max-autotune",
        action="store_true",
        help="Do not try torch.compile mode=max-autotune. The row is marked skipped.",
    )
    parser.add_argument(
        "--calibrate",
        default="ridge-compute",
        choices=("ridge-compute", "all-gemm", "off"),
        help=(
            "Few-shot wall-clock of always_fuse vs always_unfuse for GEMM. "
            "ridge-compute (default) probes the ridge band and the compute side. "
            "all-gemm also probes memory-bound MatMul+RMSNorm. "
            "off keeps the datasheet baseline. Not DRAM AI. Attention is unchanged. "
            "A busy GPU can flip the winner."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Timed iterations per baseline. Default 10 on CUDA, 2 otherwise.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Warmup iterations per baseline. Default 3 on CUDA, 1 otherwise.",
    )
    parser.add_argument("--version", action="version", version=f"shapefuse {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "selfcheck":
        from shapefuse.selfcheck import run_selfcheck

        run_selfcheck()
        return 0
    if args.cmd == "bakeoff":
        from shapefuse.bakeoff import main_bakeoff

        return main_bakeoff(args)
    from shapefuse.runner import run_command

    return run_command(args)


if __name__ == "__main__":
    sys.exit(main())

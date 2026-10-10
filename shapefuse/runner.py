"""Write the analytical bundle, optional timings, and summary.json."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

from shapefuse import __version__
from shapefuse.bench import CudaUnavailable, resolve_backend, run_bench, run_cpu_proxy_sweep
from shapefuse.intensity import Thresholds
from shapefuse.io_utils import write_csv, write_json
from shapefuse.machines import get_machine
from shapefuse.sweeps import SweepBundle, build_sweeps


def run_analytical(out: Path, machine_name: str, mem_frac: float, compute_frac: float, launch_us: float) -> SweepBundle:
    machine = get_machine(machine_name)
    bundle = build_sweeps(machine, Thresholds(mem_frac, compute_frac), launch_s=launch_us * 1e-6)
    _write_tables(out, bundle)
    return bundle


def run_command(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "bench":
        return _run_bench_only(args, out)
    try:
        bundle = run_analytical(out, args.machine, args.mem_frac, args.compute_frac, args.launch_overhead_us)
    except KeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    proxy_rows = None
    bench_rows = None
    bench_meta = None
    if args.cmd == "all":
        try:
            bench_rows, bench_meta = _bench(args)
        except CudaUnavailable as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        write_csv(out / "bench_gemm.csv", bench_rows)
        if args.cpu_proxy:
            proxy_rows = run_cpu_proxy_sweep()
            write_csv(out / "gemm_M_sweep_cpu.csv", proxy_rows)
    elif args.cmd == "analytical" and args.cpu_proxy:
        proxy_rows = run_cpu_proxy_sweep()
        write_csv(out / "gemm_M_sweep_cpu.csv", proxy_rows)
    if args.cmd in {"all", "analytical", "plans"} and not args.skip_plots:
        from shapefuse.plots import write_figures

        names = write_figures(bundle, out, proxy_rows)
        print("figures:", ", ".join(names))
    summary = _summary(args, bundle, bench_rows, bench_meta, proxy_rows)
    write_json(out / "summary.json", summary)
    if bench_meta is not None:
        write_json(
            out / "gpu_measurements.json",
            {
                "status": "PLACEHOLDER_NO_GPU" if bench_meta["placeholder"] else "PARTIAL_VENDOR_MATMUL",
                "do_not_treat_as_fused_kernel_study": True,
                "bench": bench_meta,
                "rows_file": "bench_gemm.csv",
                "still_tbd": _still_tbd(),
            },
        )
    elif args.cmd != "bench":
        write_json(
            out / "gpu_measurements.json",
            {
                "status": "PLACEHOLDER_NOT_RUN",
                "do_not_treat_as_fused_kernel_study": True,
                "bench": None,
                "still_tbd": _still_tbd(),
            },
        )
    (out / "SEND_BACK.txt").write_text(_send_back(out, summary), encoding="utf-8")
    print(f"wrote {out.resolve()}")
    print(f"gpu_measurement_status: {summary['gpu_measurement_status']}")
    return 0


def _run_bench_only(args, out: Path) -> int:
    try:
        rows, meta = _bench(args)
    except CudaUnavailable as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    write_csv(out / "bench_gemm.csv", rows)
    write_json(
        out / "gpu_measurements.json",
        {
            "status": "PLACEHOLDER_NO_GPU" if meta["placeholder"] else "PARTIAL_VENDOR_MATMUL",
            "do_not_treat_as_fused_kernel_study": True,
            "bench": meta,
            "rows_file": "bench_gemm.csv",
            "still_tbd": _still_tbd(),
        },
    )
    print(f"wrote {out.resolve()}")
    return 0


def _bench(args):
    ctx = resolve_backend(args.device, args.require_cuda, args.allow_cpu_fallback)
    if ctx.placeholder:
        print(f"NOTE: {ctx.note}")
    rows, meta = run_bench(ctx, args.bench_level)
    if meta["warning"]:
        print(f"WARNING: {meta['warning']}")
    return rows, meta


def _write_tables(out: Path, bundle: SweepBundle) -> None:
    write_csv(out / "gemm_shapes.csv", bundle.gemm)
    write_csv(out / "gemm_M_sweep.csv", bundle.gemm_curve)
    write_csv(out / "attention_intensity.csv", bundle.attention)
    write_csv(out / "attention_decode.csv", bundle.attention_decode)
    write_csv(out / "rmsnorm_epilogue.csv", bundle.rmsnorm)
    write_csv(out / "ew_chains.csv", bundle.ew)
    write_csv(out / "moe_grouped.csv", bundle.moe)
    write_csv(out / "quant_epilogue.csv", bundle.quant)
    write_csv(out / "dual_plans.csv", bundle.plans)
    write_json(out / "crossings.json", bundle.crossings)


def _summary(args, bundle: SweepBundle, bench_rows, bench_meta, proxy_rows) -> dict:
    if bench_meta is None:
        gpu_status = "PLACEHOLDER_NOT_RUN"
    elif bench_meta["placeholder"]:
        gpu_status = "PLACEHOLDER_NO_GPU"
    else:
        gpu_status = "PARTIAL_VENDOR_MATMUL"
    highlights = []
    for row in bundle.gemm:
        highlights.append(
            {
                "label": row["label"],
                "shape": [row["M"], row["N"], row["K"]],
                "AI_fp16": round(row["AI_fp16"], 2),
                "regime": row["regime"],
                "vs_H100": row["vs_H100"],
            }
        )
    return {
        "schema_version": 1,
        "package_version": __version__,
        "thesis": (
            "Operator intensity is shape-dependent; fusion must be plan-polymorphic "
            "across the Roofline ridge."
        ),
        "non_contributions": [
            "Stage-Stream",
            "a new attention algorithm",
            "emitted CUDA kernels for MemMode or ComputeMode",
        ],
        "gpu_measurement_status": gpu_status,
        "machine": bundle.machine.name,
        "ridge_flops_per_byte": bundle.machine.ridge,
        "peak_source": bundle.machine.source,
        "thresholds": {
            "mem_frac": bundle.thresholds.mem_frac,
            "compute_frac": bundle.thresholds.compute_frac,
        },
        "launch_overhead_us": bundle.launch_s * 1e6,
        "launch_status": "ASSUMED_NOT_MEASURED",
        "crossings": bundle.crossings,
        "gemm_highlights": highlights,
        "moe_highlights": [
            {
                "scenario": r["scenario"],
                "AI_aggregate_fp16": r["AI_aggregate_fp16"],
                "AI_min_expert": r["AI_min_expert"],
                "AI_max_expert": r["AI_max_expert"],
                "regime_aggregate": r["regime_aggregate"],
                "regime_min_expert": r["regime_min_expert"],
                "regime_max_expert": r["regime_max_expert"],
                "heterogeneous": r["heterogeneous"],
            }
            for r in bundle.moe
        ],
        "plan_counts": _plan_counts(bundle.plans),
        "bench": bench_meta,
        "bench_rows": 0 if not bench_rows else len(bench_rows),
        "cpu_proxy_rows": len(proxy_rows) if proxy_rows else 0,
        "still_tbd": _still_tbd(),
        "repro": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "command": " ".join(sys.argv),
            "requested_device": args.device,
            "bench_level": args.bench_level,
        },
    }


def _plan_counts(rows: list[dict]) -> dict:
    primary: dict[str, int] = {}
    both = 0
    for row in rows:
        primary[row["primary_mode"]] = primary.get(row["primary_mode"], 0) + 1
        if row["emit_both"]:
            both += 1
    return {"primary": primary, "emit_both": both, "rows": len(rows), "kernels_emitted": 0}


def _still_tbd() -> list[str]:
    return [
        "Nsight DRAM bytes and achieved AI for each bench shape",
        "Measured HBM bandwidth and a measured ridge (datasheet ridge is not a substitute)",
        "MemMode vs ComputeMode kernel latencies (this repo does not emit those kernels)",
        "SCIC agreement rate against measured AI",
        "Plan-switch overhead on a serving trace",
        "End-to-end tokens/s on a model",
        "CUDA launch latency; 5 µs is only a sensitivity knob",
    ]


def _send_back(out: Path, summary: dict) -> str:
    return "\n".join(
        [
            "Send this directory back for the paper.",
            f"gpu_measurement_status: {summary['gpu_measurement_status']}",
            "",
            "Required:",
            f"  {out / 'summary.json'}",
            f"  {out / 'gpu_measurements.json'}",
            f"  {out / 'bench_gemm.csv'}",
            f"  {out / 'dual_plans.csv'}",
            f"  {out / 'gemm_shapes.csv'}",
            f"  {out / 'attention_intensity.csv'}",
            f"  {out / 'attention_decode.csv'}",
            f"  {out / 'moe_grouped.csv'}",
            f"  {out / 'rmsnorm_epilogue.csv'}",
            f"  {out / 'quant_epilogue.csv'}",
            f"  {out / 'ew_chains.csv'}",
            "  fig_*.png",
            "",
            "Still TBD after a successful H100 matmul sweep:",
            *[f"  - {item}" for item in summary["still_tbd"]],
            "",
            "Do not relabel CPU or NumPy rows as H100 measurements.",
            "Do not describe bench_gemm.csv as a fused-kernel comparison.",
            "Stage-Stream is not produced by this run.",
            "",
        ]
    )

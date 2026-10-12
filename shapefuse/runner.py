"""Write the analytical bundle, optional timings, and summary.json."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

from shapefuse import __version__
from shapefuse.agreement import build_decision_log, summarize_agreement
from shapefuse.bench import CudaUnavailable, resolve_backend, run_bench, run_cpu_proxy_sweep
from shapefuse.intensity import Thresholds
from shapefuse.io_utils import write_csv, write_json
from shapefuse.machines import get_machine
from shapefuse.profiler_hooks import load_nsight_json
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
    agreement = None
    fusion_meta = None
    if args.cmd == "all":
        try:
            bench_rows, bench_meta, agreement, fusion_meta = _write_bench_bundle(args, out)
        except CudaUnavailable as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        except (KeyError, ValueError, OSError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
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
    summary = _summary(args, bundle, bench_rows, bench_meta, proxy_rows, agreement, fusion_meta)
    write_json(out / "summary.json", summary)
    if bench_meta is not None:
        write_json(
            out / "gpu_measurements.json",
            _gpu_payload(bench_meta, agreement, fusion_meta),
        )
    elif args.cmd != "bench":
        write_json(
            out / "gpu_measurements.json",
            {
                "status": "PLACEHOLDER_NOT_RUN",
                "do_not_treat_as_fused_kernel_study": True,
                "bench": None,
                "decision_agreement": None,
                "fusion_proxy": None,
                "still_tbd": _still_tbd(None, None, None),
            },
        )
    (out / "SEND_BACK.txt").write_text(_send_back(out, summary), encoding="utf-8")
    print(f"wrote {out.resolve()}")
    print(f"gpu_measurement_status: {summary['gpu_measurement_status']}")
    return 0


def _run_bench_only(args, out: Path) -> int:
    try:
        rows, meta, agreement, fusion_meta = _write_bench_bundle(args, out)
    except CudaUnavailable as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except (KeyError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    write_json(out / "gpu_measurements.json", _gpu_payload(meta, agreement, fusion_meta))
    print(f"wrote {out.resolve()}")
    print(f"scic_accuracy_status: {agreement['scic_accuracy_status']}")
    return 0


def _bench(args):
    ctx = resolve_backend(args.device, args.require_cuda, args.allow_cpu_fallback)
    machine = get_machine(args.machine)
    nsight = None
    nsight_path = getattr(args, "nsight_json", None)
    if nsight_path:
        nsight = load_nsight_json(Path(nsight_path))
    if ctx.placeholder:
        print(f"NOTE: {ctx.note}")
    rows, meta = run_bench(
        ctx,
        args.bench_level,
        machine=machine,
        profile=getattr(args, "profile", "off"),
        nsight_by_label=nsight,
    )
    if meta["warning"]:
        print(f"WARNING: {meta['warning']}")
    return rows, meta, ctx, machine


def _write_bench_bundle(args, out: Path):
    rows, meta, ctx, machine = _bench(args)
    write_csv(out / "bench_gemm.csv", rows)
    log = build_decision_log(rows, machine)
    write_csv(out / "decision_log.csv", log)
    agreement = summarize_agreement(log)
    write_json(out / "agreement.json", agreement)
    fusion_meta = None
    if getattr(args, "fusion_proxy", False):
        from shapefuse.fusion_proxy import run_fusion_proxy

        fusion_rows, fusion_meta = run_fusion_proxy(rows, ctx)
        write_csv(out / "fusion_proxy.csv", fusion_rows)
    if not args.skip_plots:
        from shapefuse.plots import write_vendor_matmul_figure

        write_vendor_matmul_figure(rows, out / "fig_vendor_matmul_tflops.png", machine)
    return rows, meta, agreement, fusion_meta


def _gpu_payload(meta: dict, agreement: dict | None, fusion_meta: dict | None) -> dict:
    return {
        "status": "PLACEHOLDER_NO_GPU" if meta["placeholder"] else "PARTIAL_VENDOR_MATMUL",
        "do_not_treat_as_fused_kernel_study": True,
        "bench": meta,
        "rows_file": "bench_gemm.csv",
        "decision_log_file": "decision_log.csv",
        "decision_agreement": agreement,
        "fusion_proxy": fusion_meta,
        "still_tbd": _still_tbd(agreement, fusion_meta, meta),
    }


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


def _summary(args, bundle: SweepBundle, bench_rows, bench_meta, proxy_rows, agreement, fusion_meta) -> dict:
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
            "claiming FlashAttention-3 kernels",
            "quoting a profiler or process-sum byte count as SCIC accuracy",
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
        "decision_agreement": agreement,
        "fusion_proxy": fusion_meta,
        "still_tbd": _still_tbd(agreement, fusion_meta, bench_meta),
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


def _still_tbd(agreement, fusion_meta, bench_meta) -> list[str]:
    items = []
    nsight_done = bool(bench_meta) and bench_meta.get("nsight_status") in {
        "NSIGHT_DRAM",
        "NSIGHT_DRAM_PROCESS_SUM",
    }
    if not nsight_done:
        items.append(
            "Nsight DRAM bytes and achieved AI for each bench shape "
            "(torch.profiler memory is not a substitute)"
        )
    measured_ridge = None if not bench_meta else bench_meta.get("measured_ridge_flops_per_byte")
    if measured_ridge is None:
        items.append(
            "Measured HBM or GDDR bandwidth and a measured ridge "
            "(a datasheet ridge is not a substitute)"
        )
    items.append(
        "MemMode vs ComputeMode kernel latencies "
        "(fusion_proxy.csv is a proxy and does not emit those kernels)"
    )
    quote = bool(agreement) and agreement.get("quote_as_scic_accuracy") is True
    if not quote:
        items.append(
            "SCIC accuracy against isolated Nsight DRAM AI "
            "(proxy and process-sum agreement rates are not this number)"
        )
    items.append("Plan-switch overhead on a serving trace")
    items.append("End-to-end tokens/s on a model")
    items.append("CUDA launch latency; 5 µs is only a sensitivity knob")
    items.append("Attention epilogue kernels, MoE weight-dispatch policy, and measured FP8 or W8A8 peaks")
    if fusion_meta is None:
        items.append("Fusion proxy was not run. scripts/run_a40.sh passes --fusion-proxy.")
    return items


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
            f"  {out / 'decision_log.csv'}",
            f"  {out / 'agreement.json'}",
            f"  {out / 'fig_vendor_matmul_tflops.png'}",
            "  fig_*.png",
            "",
            "When --fusion-proxy ran (scripts/run_a40.sh does):",
            f"  {out / 'fusion_proxy.csv'}",
            "",
            "Strongly recommended. The third A40 package set TRY_NSIGHT=1;",
            "  ncu was on PATH and failed with ERR_NVGPUCTRPERM.",
            "  Admin steps, reboot, then: TRY_NSIGHT=1 bash scripts/run_a40.sh",
            "  docs/NSIGHT_PERMISSIONS.md",
            "  If `ncu` is not on PATH, Nsight Compute is not installed (it is",
            "  separate from the driver). The matmul sweep still runs;",
            "  achieved_ai stays empty. torch.profiler memory is not DRAM.",
            f"  {out / 'nsight_dram.json'}   (only after ncu succeeds)",
            f"  {out / 'nsight_skipped.txt'}  (written when TRY_NSIGHT=1 but ncu is missing or fails)",
            f"  {out / 'nsight_error.log'}    (ncu stderr; ERR_NVGPUCTRPERM lives here)",
            "",
            "Optional copy-kernel bandwidth estimate (NOT Nsight, NOT a measured ridge):",
            f"  {out / 'bandwidth_estimate.json'}",
            "  status ESTIMATE_COPY_NOT_NSIGHT. Do not copy estimated_ridge into",
            "  measured_ridge_flops_per_byte.",
            "",
            "Still TBD:",
            *[f"  - {item}" for item in summary["still_tbd"]],
            "",
            "Do not relabel CPU or NumPy rows as GPU measurements.",
            "Do not describe bench_gemm.csv as a fused-kernel comparison.",
            "Do not describe fusion_proxy.csv as MemMode versus ComputeMode.",
            "Do not report fusion_proxy time ratios as fusion speedups",
            "  (time_ratio_status is PROXY_TIMING_NOT_A_FUSION_SPEEDUP).",
            "Do not quote agreement.json as SCIC accuracy unless quote_as_scic_accuracy is true.",
            "Regime columns use a datasheet ridge unless measured_ridge_flops_per_byte is filled.",
            "The archived third A40 package used --machine A40_FP16.",
            "  ncu failed with ERR_NVGPUCTRPERM. achieved_ai stayed empty.",
            "proxy_ai status TORCH_PROFILER_MEMORY_NOT_DRAM is profiler memory, not DRAM.",
            "  In that package proxy_ai was stuck at 4096 (output-sized counts).",
            "  TORCH_PROFILER_EMPTY means the profiler table was empty;",
            "  TORCH_ALLOCATOR_DELTA_NOT_DRAM is a caching-allocator fallback, still not DRAM.",
            "Stage-Stream is not produced by this run.",
            "",
        ]
    )

"""Wall-clock bakeoff of subgraph latency and of the plan SCIC selects.

This is the evaluation that can run in Docker without Nsight Compute.
It times MatMul+RMSNorm, and optionally attention plus RMSNorm, across
decode and prefill shapes. It does not invent latencies: a path that did
not run leaves ``median_ms`` empty.

What each baseline actually is:

- ``eager_unfused``: eager PyTorch (or NumPy on the CPU placeholder).
  MatMul+RMSNorm stays two eager ops. Attention materializes scores.
- ``torch_compile_default``: ``torch.compile`` of the fuse-attempt
  subgraph. CUDA only. The graph is not inspected.
- ``torch_compile_max_autotune``: the same with ``mode="max-autotune"``
  when that call succeeds. Skipped or marked failed when it is not feasible.
- ``sdpa``: ``scaled_dot_product_attention`` plus RMSNorm. Library
  operator, not a FlashAttention-3 kernel.
- ``always_fuse``: the compile path (epilogue fusion attempt).
- ``always_unfuse``: the eager path that does not ask the compiler to
  fuse. Attention still uses SDPA; the norm is timed as a second region.
- ``scic_selected``: copies the median of ``always_fuse`` or
  ``always_unfuse`` according to :func:`select_executable_path`. It is
  not a third kernel. Attention copies ``always_fuse``
  (``compile_sdpa_chain``) in every regime, including when a GEMM probe
  would have preferred the other path. ``paper/data/bakeoff_v2`` is the
  wall-clock of that attention policy. The GPU was not idle. GEMM in the
  ridge band and on the compute side (or every GEMM shape, with
  ``--calibrate all-gemm``) takes a few-shot wall-clock of ``always_fuse``
  versus ``always_unfuse``. ``selected_baseline`` follows ``measured_winner``
  when that probe returns one, and otherwise keeps the analytical baseline.
  The probe is median latency on this machine. It is not DRAM AI. A busy
  GPU can flip the winner. ``paper/data/bakeoff_v2`` predates the probe.

``quote_as_scic_accuracy`` is false. Profiler bytes are not read.
Stage-Stream is not involved.
"""

from __future__ import annotations

import math
import os
import platform
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from shapefuse import __version__
from shapefuse.bench import BenchContext, CudaUnavailable, resolve_backend
from shapefuse.intensity import Thresholds
from shapefuse.io_utils import write_csv, write_json
from shapefuse.machines import Machine, get_machine
from shapefuse.plan_paths import (
    ATTENTION_BASELINES,
    BASELINE_ALWAYS_FUSE,
    BASELINE_ALWAYS_UNFUSE,
    BASELINE_COMPILE_DEFAULT,
    BASELINE_COMPILE_MAX_AUTOTUNE,
    BASELINE_EAGER,
    BASELINE_SCIC,
    BASELINE_SDPA,
    CALIBRATION_SCOPE_RIDGE_COMPUTE,
    CALIBRATION_SCOPES,
    GEMM_BASELINES,
    HONESTY_ALWAYS_FUSE_ATTN,
    HONESTY_ALWAYS_FUSE_GEMM,
    HONESTY_ALWAYS_UNFUSE_ATTN,
    HONESTY_ALWAYS_UNFUSE_GEMM,
    HONESTY_ANALYTICAL_FALLBACK,
    HONESTY_ATTN_NOT_CALIBRATED,
    HONESTY_ATTN_OOM,
    HONESTY_CALIBRATION_AGREES,
    HONESTY_COMPILE_DEFAULT,
    HONESTY_COMPILE_FAILED,
    HONESTY_COMPILE_SKIPPED_CPU,
    HONESTY_EAGER_ATTN,
    HONESTY_EAGER_GEMM,
    HONESTY_GPU_CONTENTION,
    HONESTY_MAX_AUTOTUNE,
    HONESTY_MAX_AUTOTUNE_FAILED,
    HONESTY_MAX_AUTOTUNE_NOT_FEASIBLE,
    HONESTY_MAX_AUTOTUNE_SKIPPED,
    HONESTY_LIBRARY_SDPA_CHAIN,
    HONESTY_LIBRARY_SDPA_NOT_PRIMARY,
    HONESTY_MEASURED_OVERRIDE,
    HONESTY_RIDGE_PRIMARY_ONLY,
    HONESTY_SCIC_COPIED,
    HONESTY_SCIC_MISSING,
    HONESTY_SDPA,
    HONESTY_SDPA_UNAVAILABLE,
    HONESTY_WALLCLOCK_NOT_DRAM,
    SELECTION_ANALYTICAL,
    SELECTION_ANALYTICAL_FALLBACK,
    SELECTION_LIBRARY_SDPA,
    SELECTION_MEASURED,
    analytical_executable_path,
    attach_measured_winner,
    common_honesty,
    gemm_calibration_applies,
    join_honesty,
    pick_wallclock_winner,
    select_executable_path,
)
from shapefuse.plans import POLICY_COMPILE_SDPA_CHAIN, PlanCache, select_plan
from shapefuse.scic import IntensityReport, Site, classify_site

OP_GEMM = "matmul_rmsnorm"
OP_ATTN = "attention"


@dataclass(frozen=True)
class GemmShape:
    label: str
    m: int
    n: int
    k: int

    def shape_key(self) -> str:
        return f"M={self.m},N={self.n},K={self.k}"


@dataclass(frozen=True)
class AttnShape:
    """Prefill uses ``s`` as S. Decode uses ``s`` as Sk and ``sq`` as Sq."""

    label: str
    kind: str
    b: int
    h: int
    s: int
    d: int
    sq: int

    def shape_key(self) -> str:
        if self.kind == "attention_decode":
            return f"B={self.b},H={self.h},Sq={self.sq},Sk={self.s},D={self.d}"
        return f"B={self.b},H={self.h},S={self.s},D={self.d}"


@dataclass
class Measurement:
    median_ms: float | None
    ok: bool
    error: str
    honesty: str
    timing_copied_from: str = ""
    extra_fn: object | None = field(default=None, repr=False, compare=False)


# Few-shot, not the bakeoff's 10-repeat sample. Override with the environment.
CALIBRATION_REPEATS = 3
CALIBRATION_WARMUP = 1
CALIBRATION_METRIC = "wallclock_median_ms"


@dataclass(frozen=True)
class WallclockCalibration:
    """Few-shot fuse versus unfuse medians, or an explicit skip.

    ``winner`` is ``always_fuse``, ``always_unfuse``, or empty. Empty means
    the analytical or library baseline stands. Times are milliseconds.
    They are wall-clock on this process, not DRAM bytes.
    """

    attempted: bool
    winner: str
    fuse_ms: float | None
    unfuse_ms: float | None
    repeats: int
    warmup: int
    source: str
    honesty: str
    note: str
    metric: str = CALIBRATION_METRIC


# N = K = 64 keeps the CPU schema check small. Every shape here is
# memory-bound on the datasheet ridges; the polymorphic switch is tested
# on the decode-prefill list, not on this one.
SMOKE_GEMM: tuple[GemmShape, ...] = (
    GemmShape("decode_m1", 1, 64, 64),
    GemmShape("prefill_m128", 128, 64, 64),
)

# Decode M, ridge-band M, and prefill M at N = K = 4096. The same widths
# as the A40 vendor curve. Regimes below are for the A40 FP16 datasheet
# (R ≈ 215), not a measured ridge.
DECODE_PREFILL_GEMM: tuple[GemmShape, ...] = (
    GemmShape("decode_m1", 1, 4096, 4096),
    GemmShape("decode_m32", 32, 4096, 4096),
    GemmShape("ridge_m128", 128, 4096, 4096),
    GemmShape("ridge_m256", 256, 4096, 4096),
    GemmShape("ridge_m512", 512, 4096, 4096),
    GemmShape("prefill_m2048", 2048, 4096, 4096),
    GemmShape("prefill_m4096", 4096, 4096, 4096),
)

SMOKE_ATTN: tuple[AttnShape, ...] = (
    AttnShape("prefill_s32", "attention_prefill", 1, 2, 32, 16, 32),
    AttnShape("decode_sk32", "attention_decode", 1, 2, 32, 16, 1),
)

# Head dim cancels in the FA-style AI (S/2 in FP16), so D = 64 is enough
# for the regime switch and keeps the materialized score tensor small.
# H = 8, S = 2048 is 64 MiB of FP16 scores, not the H = 32 analytical table.
DECODE_PREFILL_ATTN: tuple[AttnShape, ...] = (
    AttnShape("prefill_s128", "attention_prefill", 1, 8, 128, 64, 128),
    AttnShape("prefill_s512", "attention_prefill", 1, 8, 512, 64, 512),
    AttnShape("prefill_s2048", "attention_prefill", 1, 8, 2048, 64, 2048),
    AttnShape("decode_sk128", "attention_decode", 1, 8, 128, 64, 1),
    AttnShape("decode_sk2048", "attention_decode", 1, 8, 2048, 64, 1),
)

GEMM_LEVELS = {
    "smoke": SMOKE_GEMM,
    "decode-prefill": DECODE_PREFILL_GEMM,
}
ATTN_LEVELS = {
    "smoke": SMOKE_ATTN,
    "decode-prefill": DECODE_PREFILL_ATTN,
}


def decide_gemm(shape: GemmShape, machine: Machine, thresholds: Thresholds):
    report = classify_site(
        Site("gemm", shape.label, {"M": shape.m, "N": shape.n, "K": shape.k}),
        machine,
        thresholds,
    )
    decision = select_plan(report, length=shape.m)
    return report, decision, select_executable_path(decision)


def decide_attention(shape: AttnShape, machine: Machine, thresholds: Thresholds):
    if shape.kind == "attention_decode":
        site = Site(
            "attention_decode",
            shape.label,
            {
                "B": shape.b,
                "H": shape.h,
                "Hkv": shape.h,
                "Sk": shape.s,
                "D": shape.d,
                "Sq": shape.sq,
            },
        )
        length = shape.s
    else:
        site = Site(
            "attention_prefill",
            shape.label,
            {"B": shape.b, "H": shape.h, "S": shape.s, "D": shape.d},
        )
        length = shape.s
    report = classify_site(site, machine, thresholds)
    decision = select_plan(report, length=length)
    return report, decision, select_executable_path(decision)


def assemble_latency_rows(
    *,
    op: str,
    label: str,
    shape: str,
    baselines: tuple[str, ...],
    measurements: dict[str, Measurement],
    chosen: str,
    report: IntensityReport,
    emit_both: bool,
    primary_mode: str,
    fuse_epilogue: bool,
    placeholder: bool,
    epilogue_policy: str = "",
    backend: str,
    device_name: str,
    dtype_ran: str,
    selection_honesty: str = "",
) -> list[dict]:
    """Build CSV rows. ``scic_selected`` copies ``chosen`` and does not retime."""
    attention = op == OP_ATTN
    eager = measurements.get(BASELINE_EAGER)
    eager_ms = eager.median_ms if eager is not None and eager.ok else None
    rows: list[dict] = []
    for baseline in baselines:
        if baseline == BASELINE_SCIC:
            # Analytical ComputeMode leaves fuse_epilogue false. Library SDPA
            # still times always_fuse. Do not label that copy as the
            # analytical primary.
            library_override = (
                attention
                and epilogue_policy == POLICY_COMPILE_SDPA_CHAIN
                and chosen == BASELINE_ALWAYS_FUSE
                and not fuse_epilogue
            )
            meas = _scic_measurement(
                measurements.get(chosen),
                chosen,
                emit_both and not library_override,
                attention,
                placeholder,
                library_override=library_override,
                extra_honesty=selection_honesty,
            )
            copied = chosen if meas.ok else ""
        else:
            meas = measurements[baseline]
            copied = meas.timing_copied_from
        speedup = _speedup(eager_ms, meas.median_ms)
        if baseline == BASELINE_EAGER and meas.ok and meas.median_ms is not None:
            speedup = 1.0
        rows.append(
            {
                "baseline": baseline,
                "shape": shape,
                "median_ms": meas.median_ms,
                "speedup_vs_eager": speedup,
                "honesty": meas.honesty,
                "op": op,
                "label": label,
                "ok": meas.ok,
                "error": meas.error,
                "selected_baseline": chosen,
                "timing_copied_from": copied,
                "regime": report.regime,
                "roofline_limit": report.roofline_limit,
                "primary_mode": primary_mode,
                "fuse_epilogue": fuse_epilogue,
                "epilogue_policy": epilogue_policy,
                "emit_both": emit_both,
                "analytical_ai": report.analytical_ai,
                "machine": report.machine,
                "ridge_flops_per_byte": report.ridge,
                "ridge_source": "datasheet",
                "dtype_ran": dtype_ran,
                "backend": backend,
                "device_name": device_name,
                "placeholder": placeholder,
                "nsight_used": False,
                "quote_as_scic_accuracy": False,
                "claims_fa3": False,
                "claims_stage_stream": False,
                "claims_custom_kernel": False,
                "graph_inspected": False,
                "kernel_emitted": False,
            }
        )
    return rows


def decision_row(
    *,
    op: str,
    label: str,
    shape: str,
    report: IntensityReport,
    decision,
    chosen: str,
    analytical: str,
    calibration: WallclockCalibration,
) -> dict:
    alternate = "" if decision.alternate is None else decision.alternate.mode
    return {
        "op": op,
        "label": label,
        "shape": shape,
        "kind": report.kind,
        "analytical_ai": report.analytical_ai,
        "regime": report.regime,
        "roofline_limit": report.roofline_limit,
        "ai_over_ridge": report.ai_over_ridge,
        "machine": report.machine,
        "ridge_flops_per_byte": report.ridge,
        "ridge_source": "datasheet",
        "primary_mode": decision.primary.mode,
        "alternate_mode": alternate,
        "emit_both": decision.emit_both,
        "fuse_epilogue": decision.primary.fuse_epilogue,
        "epilogue_policy": decision.epilogue_policy,
        "materialize_intermediate": decision.primary.materialize_intermediate,
        "analytical_baseline": analytical,
        "measured_winner": calibration.winner,
        "selection_source": calibration.source,
        "selected_baseline": chosen,
        "calibration_attempted": calibration.attempted,
        "calibration_fuse_ms": calibration.fuse_ms,
        "calibration_unfuse_ms": calibration.unfuse_ms,
        "calibration_repeats": calibration.repeats if calibration.attempted else None,
        "calibration_warmup": calibration.warmup if calibration.attempted else None,
        "calibration_metric": calibration.metric if calibration.attempted else "",
        "calibration_honesty": calibration.honesty,
        "calibration_note": calibration.note,
        "cache_key": decision.cache_key,
        "rationale": decision.rationale,
        "quote_as_scic_accuracy": False,
        "claims_fa3": False,
        "claims_stage_stream": False,
        "claims_custom_kernel": False,
        "nsight_used": False,
        "kernel_emitted": False,
    }


def skipped_calibration(source: str, note: str, honesty: str = "") -> WallclockCalibration:
    """No probe ran. ``source`` is ``analytical`` or ``library_sdpa``."""
    return WallclockCalibration(
        attempted=False,
        winner="",
        fuse_ms=None,
        unfuse_ms=None,
        repeats=0,
        warmup=0,
        source=source,
        honesty=honesty,
        note=note,
        metric="",
    )


def calibration_from_medians(
    fuse_ms: float | None,
    unfuse_ms: float | None,
    *,
    repeats: int,
    warmup: int,
    fuse_error: str = "",
    unfuse_error: str = "",
) -> WallclockCalibration:
    """Turn two probe medians into a decision-log record.

    The strictly faster positive finite median becomes ``measured_winner``.
    A miss or a tie keeps the analytical baseline and says so.
    """
    winner = pick_wallclock_winner(fuse_ms, unfuse_ms) or ""
    notes: list[str] = []
    if fuse_ms is None:
        notes.append(fuse_error or "always_fuse probe did not return a median")
    if unfuse_ms is None:
        notes.append(unfuse_error or "always_unfuse probe did not return a median")
    if (
        not winner
        and fuse_ms is not None
        and unfuse_ms is not None
        and fuse_ms == unfuse_ms
        and fuse_ms > 0
    ):
        notes.append("tie; analytical baseline kept")
    if winner:
        source = SELECTION_MEASURED
        honesty = join_honesty(HONESTY_WALLCLOCK_NOT_DRAM, HONESTY_GPU_CONTENTION)
        if not notes:
            notes.append(
                "few-shot wall-clock on this machine; not DRAM AI; "
                "GPU contention can flip the winner"
            )
    else:
        source = SELECTION_ANALYTICAL_FALLBACK
        honesty = join_honesty(
            HONESTY_WALLCLOCK_NOT_DRAM,
            HONESTY_GPU_CONTENTION,
            HONESTY_ANALYTICAL_FALLBACK,
        )
        if not notes:
            notes.append("probe did not produce a strictly faster path")
    return WallclockCalibration(
        attempted=True,
        winner=winner,
        fuse_ms=fuse_ms,
        unfuse_ms=unfuse_ms,
        repeats=repeats,
        warmup=warmup,
        source=source,
        honesty=honesty,
        note="; ".join(notes),
    )


def scic_selection_honesty(calibration: WallclockCalibration, analytical: str, chosen: str) -> str:
    """Flags for the ``scic_selected`` latency row. Empty when no probe ran."""
    if calibration.source == SELECTION_MEASURED:
        relation = (
            HONESTY_MEASURED_OVERRIDE if chosen != analytical else HONESTY_CALIBRATION_AGREES
        )
        return join_honesty(HONESTY_WALLCLOCK_NOT_DRAM, HONESTY_GPU_CONTENTION, relation)
    if calibration.source == SELECTION_ANALYTICAL_FALLBACK:
        return join_honesty(HONESTY_WALLCLOCK_NOT_DRAM, HONESTY_ANALYTICAL_FALLBACK)
    return ""


def calibration_budget() -> tuple[int, int]:
    """``(repeats, warmup)`` for the GEMM probe. Defaults are 3 and 1."""
    return (
        _positive_int(os.environ.get("CALIBRATION_REPEATS"), CALIBRATION_REPEATS),
        _positive_int(os.environ.get("CALIBRATION_WARMUP"), CALIBRATION_WARMUP),
    )


def _positive_int(raw: str | None, default: int) -> int:
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if value < 1:
        return default
    return value


def run_bakeoff(
    *,
    out: Path,
    machine_name: str,
    device: str,
    require_cuda: bool,
    allow_cpu_fallback: bool,
    level: str,
    attention: bool,
    skip_max_autotune: bool,
    repeats: int | None,
    warmup: int | None,
    mem_frac: float,
    compute_frac: float,
    calibrate: str = CALIBRATION_SCOPE_RIDGE_COMPUTE,
) -> dict:
    if level not in GEMM_LEVELS:
        known = ", ".join(sorted(GEMM_LEVELS))
        raise ValueError(f"Unknown bakeoff level {level!r}. Known: {known}")
    if calibrate not in CALIBRATION_SCOPES:
        known = ", ".join(CALIBRATION_SCOPES)
        raise ValueError(f"Unknown calibration scope {calibrate!r}. Known: {known}")
    machine = get_machine(machine_name)
    thresholds = Thresholds(mem_frac, compute_frac)
    dynamo_limit = configure_dynamo_cache_limit()
    if dynamo_limit is None:
        print("Dynamo cache_size_limit: not set (torch._dynamo is not importable).")
    else:
        print(
            f"Dynamo cache_size_limit={dynamo_limit}. "
            "Graphs are not inspected. An idle GPU is recommended."
        )
    cal_repeats, cal_warmup = calibration_budget()
    print(
        f"GEMM calibration scope={calibrate}. "
        f"Probe repeats={cal_repeats} warmup={cal_warmup}. "
        "The probe is wall-clock on this machine, not DRAM AI. "
        "A busy GPU can flip measured_winner. "
        "Attention stays compile_sdpa_chain."
    )
    ctx = resolve_backend(device, require_cuda, allow_cpu_fallback)
    cuda = ctx.backend == "cuda-torch"
    if repeats is None:
        repeats = 10 if cuda else 2
    if warmup is None:
        warmup = 3 if cuda else 1
    out.mkdir(parents=True, exist_ok=True)

    latency_rows: list[dict] = []
    decisions: list[dict] = []
    cache = PlanCache()
    for shape in GEMM_LEVELS[level]:
        report, decision, _ = decide_gemm(shape, machine, thresholds)
        analytical = analytical_executable_path(decision)
        probe = gemm_calibration_applies(decision, calibrate)
        try:
            measured, calibration = _time_gemm(
                shape,
                ctx,
                repeats,
                warmup,
                skip_max_autotune,
                calibrate=probe,
            )
        except Exception as exc:
            measured = _failed_baselines(GEMM_BASELINES, ctx, _exc_text(exc), attention=False)
            if probe:
                calibration = calibration_from_medians(
                    None,
                    None,
                    repeats=cal_repeats,
                    warmup=cal_warmup,
                    fuse_error=_exc_text(exc),
                    unfuse_error=_exc_text(exc),
                )
            else:
                calibration = skipped_calibration(SELECTION_ANALYTICAL, _exc_text(exc))
        if calibration.winner:
            decision = attach_measured_winner(decision, calibration.winner)
            winning_ms = (
                calibration.fuse_ms
                if calibration.winner == BASELINE_ALWAYS_FUSE
                else calibration.unfuse_ms
            )
            if winning_ms is not None:
                # The coarse bucket merges nearby M. Keep the shape label so
                # ridge_m256 does not overwrite ridge_m512.
                cache.record_measured_winner(
                    f"{decision.cache_key}|{shape.label}",
                    calibration.winner,
                    CALIBRATION_METRIC,
                    float(winning_ms),
                )
        chosen = select_executable_path(decision)
        _log_decision(OP_GEMM, shape.label, report, decision, chosen, analytical, calibration)
        decisions.append(
            decision_row(
                op=OP_GEMM,
                label=shape.label,
                shape=shape.shape_key(),
                report=report,
                decision=decision,
                chosen=chosen,
                analytical=analytical,
                calibration=calibration,
            )
        )
        latency_rows.extend(
            assemble_latency_rows(
                op=OP_GEMM,
                label=shape.label,
                shape=shape.shape_key(),
                baselines=GEMM_BASELINES,
                measurements=measured,
                chosen=chosen,
                report=report,
                emit_both=decision.emit_both,
                primary_mode=decision.primary.mode,
                fuse_epilogue=decision.primary.fuse_epilogue,
                epilogue_policy=decision.epilogue_policy,
                placeholder=ctx.placeholder,
                backend=ctx.backend,
                device_name=ctx.device_name,
                dtype_ran="fp16" if cuda else "fp32",
                selection_honesty=scic_selection_honesty(calibration, analytical, chosen),
            )
        )
    if attention:
        library = skipped_calibration(
            SELECTION_LIBRARY_SDPA,
            "library SDPA stays compile_sdpa_chain; the GEMM wall-clock probe is not applied",
            HONESTY_ATTN_NOT_CALIBRATED,
        )
        for shape in ATTN_LEVELS[level]:
            report, decision, chosen = decide_attention(shape, machine, thresholds)
            analytical = analytical_executable_path(decision)
            _log_decision(OP_ATTN, shape.label, report, decision, chosen, analytical, library)
            decisions.append(
                decision_row(
                    op=OP_ATTN,
                    label=shape.label,
                    shape=shape.shape_key(),
                    report=report,
                    decision=decision,
                    chosen=chosen,
                    analytical=analytical,
                    calibration=library,
                )
            )
            try:
                measured = _time_attention(shape, ctx, repeats, warmup, skip_max_autotune)
            except Exception as exc:
                measured = _failed_baselines(ATTENTION_BASELINES, ctx, _exc_text(exc), attention=True)
            latency_rows.extend(
                assemble_latency_rows(
                    op=OP_ATTN,
                    label=shape.label,
                    shape=shape.shape_key(),
                    baselines=ATTENTION_BASELINES,
                    measurements=measured,
                    chosen=chosen,
                    report=report,
                    emit_both=decision.emit_both,
                    primary_mode=decision.primary.mode,
                    fuse_epilogue=decision.primary.fuse_epilogue,
                    epilogue_policy=decision.epilogue_policy,
                    placeholder=ctx.placeholder,
                    backend=ctx.backend,
                    device_name=ctx.device_name,
                    dtype_ran="fp16" if cuda else "fp32",
                )
            )

    write_csv(out / "bakeoff_latency.csv", latency_rows)
    write_csv(out / "bakeoff_decisions.csv", decisions)
    summary = _summary(
        ctx=ctx,
        machine=machine,
        level=level,
        attention=attention,
        skip_max_autotune=skip_max_autotune,
        repeats=repeats,
        warmup=warmup,
        latency_rows=latency_rows,
        decisions=decisions,
        thresholds=thresholds,
        calibrate=calibrate,
        calibration_repeats=cal_repeats,
        calibration_warmup=cal_warmup,
        plan_cache_entries=len(cache.entries),
    )
    write_json(out / "bakeoff_summary.json", summary)
    (out / "SEND_BACK.txt").write_text(_send_back(out, summary), encoding="utf-8")
    print(f"wrote {out.resolve()}")
    print(f"gpu_measurement_status: {summary['gpu_measurement_status']}")
    print(f"nsight_used: {summary['nsight_used']}")
    print(f"quote_as_scic_accuracy: {summary['quote_as_scic_accuracy']}")
    return summary


def main_bakeoff(args) -> int:
    try:
        run_bakeoff(
            out=Path(args.out),
            machine_name=args.machine,
            device=args.device,
            require_cuda=args.require_cuda,
            allow_cpu_fallback=args.allow_cpu_fallback,
            level=getattr(args, "level", "decode-prefill"),
            attention=getattr(args, "attention", False),
            skip_max_autotune=getattr(args, "skip_max_autotune", False),
            repeats=getattr(args, "repeats", None),
            warmup=getattr(args, "warmup", None),
            mem_frac=args.mem_frac,
            compute_frac=args.compute_frac,
            calibrate=getattr(args, "calibrate", CALIBRATION_SCOPE_RIDGE_COMPUTE),
        )
    except CudaUnavailable as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except (KeyError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


def _failed_baselines(baselines: tuple[str, ...], ctx: BenchContext, error: str, attention: bool) -> dict[str, Measurement]:
    """One shape blew up before a median existed. Do not invent one."""
    specific = HONESTY_EAGER_ATTN if attention else HONESTY_EAGER_GEMM
    return {
        name: _fail(ctx, specific, error, attention=attention)
        for name in baselines
        if name != BASELINE_SCIC
    }


def configure_dynamo_cache_limit() -> int | None:
    """Raise ``torch._dynamo.config.cache_size_limit`` when Dynamo imports.

    The archived A40 bakeoff hit the default of 8 on the MatMul chain and
    on ``sdpa_chain``. A higher limit lets each shape specialize. It does
    not inspect the graph, and a compile-labelled row is still not proof
    that compiled code ran. Returns the limit that was set, or ``None``
    when Dynamo is not importable. Override with ``DYNAMO_CACHE_SIZE_LIMIT``.
    """
    raw = os.environ.get("DYNAMO_CACHE_SIZE_LIMIT", "64")
    try:
        limit = int(raw)
    except ValueError:
        limit = 64
    if limit < 1:
        limit = 64
    try:
        import torch._dynamo as dynamo
    except Exception:
        return None
    dynamo.config.cache_size_limit = limit
    return limit


def _scic_measurement(
    source: Measurement | None,
    chosen: str,
    emit_both: bool,
    attention: bool,
    placeholder: bool,
    library_override: bool = False,
    extra_honesty: str = "",
) -> Measurement:
    extra = [HONESTY_SCIC_COPIED, f"TIMING_COPIED_FROM_{chosen}"]
    if extra_honesty:
        extra.append(extra_honesty)
    if library_override:
        extra.append(HONESTY_LIBRARY_SDPA_CHAIN)
        extra.append(HONESTY_LIBRARY_SDPA_NOT_PRIMARY)
    elif attention and chosen == BASELINE_ALWAYS_FUSE:
        extra.append(HONESTY_LIBRARY_SDPA_CHAIN)
    if emit_both:
        extra.append(HONESTY_RIDGE_PRIMARY_ONLY)
    prefix = join_honesty(*common_honesty(placeholder=placeholder, attention=attention), *extra)
    if source is None or not source.ok or source.median_ms is None:
        err = "selected path did not run"
        if source is not None and source.error:
            err = f"{err}: {source.error}"
        return Measurement(
            median_ms=None,
            ok=False,
            error=err,
            honesty=join_honesty(prefix, HONESTY_SCIC_MISSING),
            timing_copied_from="",
        )
    return Measurement(
        median_ms=source.median_ms,
        ok=True,
        error="",
        honesty=join_honesty(prefix, source.honesty),
        timing_copied_from=chosen,
    )


def _speedup(eager_ms: float | None, this_ms: float | None) -> float | None:
    if eager_ms is None or this_ms is None or this_ms <= 0 or eager_ms <= 0:
        return None
    return eager_ms / this_ms


def _log_decision(
    op: str,
    label: str,
    report: IntensityReport,
    decision,
    chosen: str,
    analytical: str,
    calibration: WallclockCalibration,
) -> None:
    print(
        "SCIC"
        f" op={op}"
        f" label={label}"
        f" regime={report.regime}"
        f" mode={decision.primary.mode}"
        f" emit_both={str(decision.emit_both).lower()}"
        f" fuse_epilogue={str(decision.primary.fuse_epilogue).lower()}"
        f" epilogue_policy={decision.epilogue_policy or '-'}"
        f" analytical={analytical}"
        f" measured={calibration.winner or '-'}"
        f" source={calibration.source}"
        f" -> {chosen}"
    )


def _time_gemm(
    shape: GemmShape,
    ctx: BenchContext,
    repeats: int,
    warmup: int,
    skip_max_autotune: bool,
    calibrate: bool,
):
    if ctx.backend == "cpu-numpy":
        return _time_gemm_numpy(shape, ctx, repeats, warmup, calibrate)
    return _time_gemm_torch(shape, ctx, repeats, warmup, skip_max_autotune, calibrate)


def _time_attention(shape: AttnShape, ctx: BenchContext, repeats: int, warmup: int, skip_max_autotune: bool):
    if ctx.backend == "cpu-numpy":
        return _time_attention_numpy(shape, ctx, repeats, warmup)
    return _time_attention_torch(shape, ctx, repeats, warmup, skip_max_autotune)


def _time_gemm_numpy(
    shape: GemmShape,
    ctx: BenchContext,
    repeats: int,
    warmup: int,
    calibrate: bool,
) -> tuple[dict[str, Measurement], WallclockCalibration]:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((shape.m, shape.k), dtype=np.float32)
    b = rng.standard_normal((shape.k, shape.n), dtype=np.float32)
    weight = np.ones((shape.n,), dtype=np.float32)

    def chain():
        c = a @ b
        var = np.mean(c * c, axis=-1, keepdims=True)
        return c * (1.0 / np.sqrt(var + 1e-6)) * weight

    eager = _measure_cpu(chain, repeats, warmup, _spec(ctx, HONESTY_EAGER_GEMM, attention=False))
    unfuse = _measure_cpu(chain, repeats, warmup, _spec(ctx, HONESTY_ALWAYS_UNFUSE_GEMM, attention=False))
    skipped_compile = _skipped_compile(ctx, attention=False)
    skipped_max = _skipped_max_autotune(ctx, skipped_by_flag=True, attention=False)
    measurements = {
        BASELINE_EAGER: eager,
        BASELINE_COMPILE_DEFAULT: skipped_compile,
        BASELINE_COMPILE_MAX_AUTOTUNE: skipped_max,
        BASELINE_ALWAYS_FUSE: _fuse_unavailable(ctx, attention=False, reason=skipped_compile.error),
        BASELINE_ALWAYS_UNFUSE: unfuse,
    }
    if not calibrate:
        calibration = skipped_calibration(
            SELECTION_ANALYTICAL,
            "datasheet baseline; this GEMM shape is outside the calibration scope",
        )
    else:
        # Eager versus eager would be a fake tie. Compile does not run here.
        cal_repeats, cal_warmup = calibration_budget()
        calibration = calibration_from_medians(
            None,
            None,
            repeats=cal_repeats,
            warmup=cal_warmup,
            fuse_error="torch.compile is not run on the NumPy path",
            unfuse_error="wall-clock probe needs both always_fuse and always_unfuse",
        )
    return measurements, calibration


def _time_gemm_torch(
    shape: GemmShape,
    ctx: BenchContext,
    repeats: int,
    warmup: int,
    skip_max_autotune: bool,
    calibrate: bool,
):
    import torch

    cuda = ctx.backend == "cuda-torch"
    device = torch.device("cuda" if cuda else "cpu")
    dtype = torch.float16 if cuda else torch.float32
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    a = torch.randn(shape.m, shape.k, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    b = torch.randn(shape.k, shape.n, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    weight = torch.ones(shape.n, device=device, dtype=dtype)

    def chain():
        c = a @ b
        return _torch_rmsnorm(c, weight)

    compiled_fn = None
    if calibrate:
        calibration, compiled_fn = _probe_gemm_chain(chain, ctx, cuda)
    else:
        calibration = skipped_calibration(
            SELECTION_ANALYTICAL,
            "datasheet baseline; this GEMM shape is outside the calibration scope",
        )

    eager = _measure_torch(chain, repeats, warmup, cuda, _spec(ctx, HONESTY_EAGER_GEMM, attention=False))
    unfuse = _measure_torch(chain, repeats, warmup, cuda, _spec(ctx, HONESTY_ALWAYS_UNFUSE_GEMM, attention=False))
    compiled = _compile_measurement(
        chain,
        mode=None,
        repeats=repeats,
        warmup=warmup,
        cuda=cuda,
        ctx=ctx,
        specific=HONESTY_COMPILE_DEFAULT,
        attention=False,
        compiled_fn=compiled_fn,
    )
    fuse = _compile_measurement(
        chain,
        mode=None,
        repeats=repeats,
        warmup=warmup,
        cuda=cuda,
        ctx=ctx,
        specific=HONESTY_ALWAYS_FUSE_GEMM,
        attention=False,
        compiled_fn=compiled.extra_fn or compiled_fn,
    )
    if not cuda:
        # Reuse the single CPU compile attempt for always_fuse. A second
        # compile of the same eager chain does not become a CUDA result.
        fuse = _retag(compiled, HONESTY_ALWAYS_FUSE_GEMM, ctx, attention=False)
    max_auto = _max_autotune_measurement(
        chain, ctx, cuda, skip_max_autotune, repeats, warmup, attention=False
    )
    return (
        {
            BASELINE_EAGER: eager,
            BASELINE_COMPILE_DEFAULT: compiled,
            BASELINE_COMPILE_MAX_AUTOTUNE: max_auto,
            BASELINE_ALWAYS_FUSE: fuse,
            BASELINE_ALWAYS_UNFUSE: unfuse,
        },
        calibration,
    )


def _probe_gemm_chain(chain, ctx: BenchContext, cuda: bool) -> tuple[WallclockCalibration, object | None]:
    """One warmup and a few timed iterations of fuse versus unfuse.

    The compiled function is returned so the full bakeoff can reuse it.
    Without CUDA, ``torch.compile`` does not run and the winner stays empty.
    """
    repeats, warmup = calibration_budget()
    unfuse = _measure_torch(
        chain,
        repeats,
        warmup,
        cuda,
        _spec(ctx, HONESTY_ALWAYS_UNFUSE_GEMM, attention=False),
    )
    fuse = _compile_measurement(
        chain,
        mode=None,
        repeats=repeats,
        warmup=warmup,
        cuda=cuda,
        ctx=ctx,
        specific=HONESTY_ALWAYS_FUSE_GEMM,
        attention=False,
    )
    calibration = calibration_from_medians(
        fuse.median_ms if fuse.ok else None,
        unfuse.median_ms if unfuse.ok else None,
        repeats=repeats,
        warmup=warmup,
        fuse_error=fuse.error,
        unfuse_error=unfuse.error,
    )
    return calibration, fuse.extra_fn


def _time_attention_numpy(shape: AttnShape, ctx: BenchContext, repeats: int, warmup: int):
    q, k, v, weight = _numpy_attn_tensors(shape)

    def materialized():
        y = _numpy_attention(q, k, v)
        return _numpy_rmsnorm(y, weight)

    eager = _measure_cpu(materialized, repeats, warmup, _spec(ctx, HONESTY_EAGER_ATTN, attention=True))
    unavailable = _fail(
        ctx,
        HONESTY_SDPA_UNAVAILABLE,
        "scaled_dot_product_attention is not used on the NumPy path",
        attention=True,
    )
    skipped_compile = _skipped_compile(ctx, attention=True)
    return {
        BASELINE_EAGER: eager,
        BASELINE_COMPILE_DEFAULT: skipped_compile,
        BASELINE_COMPILE_MAX_AUTOTUNE: _skipped_max_autotune(ctx, skipped_by_flag=True, attention=True),
        BASELINE_SDPA: unavailable,
        BASELINE_ALWAYS_FUSE: _fuse_unavailable(ctx, attention=True, reason=skipped_compile.error),
        BASELINE_ALWAYS_UNFUSE: unavailable,
    }


def _time_attention_torch(shape: AttnShape, ctx: BenchContext, repeats: int, warmup: int, skip_max_autotune: bool):
    import torch
    import torch.nn.functional as F

    cuda = ctx.backend == "cuda-torch"
    device = torch.device("cuda" if cuda else "cpu")
    dtype = torch.float16 if cuda else torch.float32
    q, k, v, weight = _torch_attn_tensors(shape, device, dtype)

    def materialized():
        y = _torch_materialized(q, k, v)
        return _torch_rmsnorm(y, weight)

    def sdpa_chain():
        y = F.scaled_dot_product_attention(q, k, v)
        return _torch_rmsnorm(y, weight)

    def sdpa_only():
        return F.scaled_dot_product_attention(q, k, v)

    eager = _measure_torch(materialized, repeats, warmup, cuda, _spec(ctx, HONESTY_EAGER_ATTN, attention=True))
    sdpa = _measure_torch(sdpa_chain, repeats, warmup, cuda, _spec(ctx, HONESTY_SDPA, attention=True))
    # Two timed regions: the library attention body, then RMSNorm on its output.
    sdpa_ms = _median_only(sdpa_only, repeats, warmup, cuda)
    norm_ms = None
    unfuse_error = ""
    if sdpa_ms is None:
        unfuse_error = "SDPA region did not return a median"
    else:
        try:
            y = sdpa_only()
            norm_ms = _median_only(lambda: _torch_rmsnorm(y, weight), repeats, warmup, cuda)
            if norm_ms is None:
                unfuse_error = "RMSNorm region did not return a median"
        except Exception as exc:
            _release_oom(cuda)
            unfuse_error = _exc_text(exc)
    if sdpa_ms is not None and norm_ms is not None:
        unfuse = Measurement(
            median_ms=sdpa_ms + norm_ms,
            ok=True,
            error="",
            honesty=_spec(ctx, HONESTY_ALWAYS_UNFUSE_ATTN, attention=True),
        )
    else:
        unfuse = _fail(ctx, HONESTY_ALWAYS_UNFUSE_ATTN, unfuse_error or "always_unfuse failed", attention=True)

    compiled = _compile_measurement(
        sdpa_chain,
        mode=None,
        repeats=repeats,
        warmup=warmup,
        cuda=cuda,
        ctx=ctx,
        specific=HONESTY_COMPILE_DEFAULT,
        attention=True,
    )
    if cuda:
        fuse = _compile_measurement(
            sdpa_chain,
            mode=None,
            repeats=repeats,
            warmup=warmup,
            cuda=cuda,
            ctx=ctx,
            specific=HONESTY_ALWAYS_FUSE_ATTN,
            attention=True,
            compiled_fn=compiled.extra_fn,
        )
    else:
        fuse = _retag(compiled, HONESTY_ALWAYS_FUSE_ATTN, ctx, attention=True)
    max_auto = _max_autotune_measurement(
        sdpa_chain, ctx, cuda, skip_max_autotune, repeats, warmup, attention=True
    )
    return {
        BASELINE_EAGER: eager,
        BASELINE_COMPILE_DEFAULT: compiled,
        BASELINE_COMPILE_MAX_AUTOTUNE: max_auto,
        BASELINE_SDPA: sdpa,
        BASELINE_ALWAYS_FUSE: fuse,
        BASELINE_ALWAYS_UNFUSE: unfuse,
    }


def _numpy_attn_tensors(shape: AttnShape):
    rng = np.random.default_rng(0)
    sq = shape.s if shape.kind == "attention_prefill" else shape.sq
    q = rng.standard_normal((shape.b, shape.h, sq, shape.d), dtype=np.float32)
    k = rng.standard_normal((shape.b, shape.h, shape.s, shape.d), dtype=np.float32)
    v = rng.standard_normal((shape.b, shape.h, shape.s, shape.d), dtype=np.float32)
    weight = np.ones((shape.d,), dtype=np.float32)
    return q, k, v, weight


def _torch_attn_tensors(shape: AttnShape, device, dtype):
    import torch

    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    sq = shape.s if shape.kind == "attention_prefill" else shape.sq
    q = torch.randn(shape.b, shape.h, sq, shape.d, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    k = torch.randn(shape.b, shape.h, shape.s, shape.d, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    v = torch.randn(shape.b, shape.h, shape.s, shape.d, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    weight = torch.ones(shape.d, device=device, dtype=dtype)
    return q, k, v, weight


def _numpy_attention(q, k, v):
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    exp = np.exp(scores)
    probs = exp / np.sum(exp, axis=-1, keepdims=True)
    return np.matmul(probs, v)


def _numpy_rmsnorm(y, weight):
    var = np.mean(y * y, axis=-1, keepdims=True)
    return y * (1.0 / np.sqrt(var + 1e-6)) * weight


def _torch_materialized(q, k, v):
    import torch

    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def _torch_rmsnorm(y, weight):
    import torch

    # FP16 GEMM outputs can overflow when squared in FP16. Inductor may
    # retain FP32 intermediates, making the old eager and compiled chains
    # numerically different. Use the same explicit FP32 normalization in
    # both paths and restore the original output dtype only at the end.
    values = y.float()
    var = (values * values).mean(dim=-1, keepdim=True)
    result = values * torch.rsqrt(var + 1e-6) * weight.float()
    return result.to(dtype=y.dtype)


def _spec(ctx: BenchContext, specific: str, attention: bool) -> str:
    return join_honesty(*common_honesty(placeholder=ctx.placeholder, attention=attention), specific)


def _fail(ctx: BenchContext, specific: str, error: str, attention: bool) -> Measurement:
    text = error
    if "out of memory" in error.lower():
        specific = join_honesty(specific, HONESTY_ATTN_OOM)
    return Measurement(None, False, _short(text), _spec(ctx, specific, attention))


def _skipped_compile(ctx: BenchContext, attention: bool) -> Measurement:
    return _fail(
        ctx,
        HONESTY_COMPILE_SKIPPED_CPU,
        "torch.compile is not run without CUDA; this row is not a GPU compile result",
        attention=attention,
    )


def _fuse_unavailable(ctx: BenchContext, attention: bool, reason: str) -> Measurement:
    specific = HONESTY_ALWAYS_FUSE_ATTN if attention else HONESTY_ALWAYS_FUSE_GEMM
    return _fail(ctx, specific, reason or "always_fuse did not run", attention=attention)


def _skipped_max_autotune(ctx: BenchContext, skipped_by_flag: bool, attention: bool) -> Measurement:
    if ctx.backend != "cuda-torch":
        flag = HONESTY_MAX_AUTOTUNE_NOT_FEASIBLE
        reason = "max-autotune is not run without CUDA"
    elif skipped_by_flag:
        flag = HONESTY_MAX_AUTOTUNE_SKIPPED
        reason = "SKIP_MAX_AUTOTUNE or --skip-max-autotune"
    else:
        flag = HONESTY_MAX_AUTOTUNE_NOT_FEASIBLE
        reason = "max-autotune was not attempted"
    return _fail(ctx, flag, reason, attention=attention)


def _compile_measurement(
    fn,
    *,
    mode: str | None,
    repeats: int,
    warmup: int,
    cuda: bool,
    ctx: BenchContext,
    specific: str,
    attention: bool,
    compiled_fn=None,
) -> Measurement:
    if not cuda:
        meas = _skipped_compile(ctx, attention=attention)
        meas.honesty = join_honesty(meas.honesty, specific)
        meas.extra_fn = None
        return meas
    import torch

    if not hasattr(torch, "compile"):
        meas = _fail(ctx, HONESTY_COMPILE_FAILED, "torch.compile is missing", attention=attention)
        meas.honesty = join_honesty(meas.honesty, specific)
        meas.extra_fn = None
        return meas
    try:
        compiled = compiled_fn if compiled_fn is not None else torch.compile(fn, dynamic=False)
        if compiled_fn is None:
            # Surface compile and runtime errors before the timed loop.
            out = compiled()
            if cuda:
                torch.cuda.synchronize()
            _ = out.reshape(-1)[0].item()
        meas = _measure_torch(compiled, repeats, warmup, cuda, _spec(ctx, specific, attention))
        meas.extra_fn = compiled
        return meas
    except Exception as exc:
        _release_oom(cuda)
        flag = HONESTY_COMPILE_FAILED if mode is None else HONESTY_MAX_AUTOTUNE_FAILED
        meas = _fail(ctx, flag, _exc_text(exc), attention=attention)
        meas.honesty = join_honesty(meas.honesty, specific)
        meas.extra_fn = None
        return meas


def _max_autotune_measurement(fn, ctx, cuda: bool, skip: bool, repeats: int, warmup: int, attention: bool):
    if not cuda or skip:
        return _skipped_max_autotune(ctx, skipped_by_flag=skip or not cuda, attention=attention)
    import torch

    try:
        compiled = torch.compile(fn, mode="max-autotune", dynamic=False)
        out = compiled()
        torch.cuda.synchronize()
        _ = out.reshape(-1)[0].item()
        return _measure_torch(
            compiled,
            repeats,
            max(1, warmup),
            cuda,
            _spec(ctx, HONESTY_MAX_AUTOTUNE, attention),
        )
    except Exception as exc:
        _release_oom(True)
        return _fail(ctx, HONESTY_MAX_AUTOTUNE_FAILED, _exc_text(exc), attention=attention)


def _retag(meas: Measurement, specific: str, ctx: BenchContext, attention: bool) -> Measurement:
    """CPU placeholder: always_fuse is the compile attempt, which did not run."""
    if meas.ok:
        return Measurement(
            meas.median_ms,
            True,
            "",
            join_honesty(_spec(ctx, specific, attention), meas.honesty),
            meas.timing_copied_from,
        )
    return _fail(ctx, specific, meas.error or "path did not run", attention=attention)


def _measure_cpu(fn, repeats: int, warmup: int, honesty: str) -> Measurement:
    try:
        median_s = _cpu_median_s(fn, repeats, warmup)
    except Exception as exc:
        return Measurement(None, False, _exc_text(exc), honesty)
    return Measurement(median_s * 1e3, True, "", honesty)


def _measure_torch(fn, repeats: int, warmup: int, cuda: bool, honesty: str) -> Measurement:
    try:
        median_s = _torch_median_s(fn, repeats, warmup, cuda)
    except Exception as exc:
        _release_oom(cuda)
        text = _exc_text(exc)
        if "out of memory" in text.lower():
            honesty = join_honesty(honesty, HONESTY_ATTN_OOM)
        return Measurement(None, False, text, honesty)
    return Measurement(median_s * 1e3, True, "", honesty)


def _median_only(fn, repeats: int, warmup: int, cuda: bool) -> float | None:
    try:
        return _torch_median_s(fn, repeats, warmup, cuda) * 1e3
    except Exception:
        _release_oom(cuda)
        return None


def _cpu_median_s(fn, repeats: int, warmup: int) -> float:
    import time

    for _ in range(warmup):
        out = fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        _touch(out)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _torch_median_s(fn, repeats: int, warmup: int, cuda: bool) -> float:
    import time

    import torch

    if cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(warmup):
            out = fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            start.record()
            out = fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / 1e3)
        _touch(out)
        return float(np.median(times))
    for _ in range(warmup):
        out = fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        _touch(out)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _touch(out) -> None:
    if isinstance(out, np.ndarray):
        if not np.isfinite(out.reshape(-1)[0]):
            raise RuntimeError("non-finite output")
        return
    value = out.reshape(-1)[0]
    item = value.item() if hasattr(value, "item") else float(value)
    if not math.isfinite(float(item)):
        raise RuntimeError("non-finite output")


def _release_oom(cuda: bool) -> None:
    if not cuda:
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _exc_text(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if not str(exc):
        text = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    return _short(text)


def _short(text: str, limit: int = 400) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _summary(
    *,
    ctx: BenchContext,
    machine: Machine,
    level: str,
    attention: bool,
    skip_max_autotune: bool,
    repeats: int,
    warmup: int,
    latency_rows: list[dict],
    decisions: list[dict],
    thresholds: Thresholds,
    calibrate: str,
    calibration_repeats: int,
    calibration_warmup: int,
    plan_cache_entries: int,
) -> dict:
    ok = sum(1 for row in latency_rows if row["ok"])
    status = "PLACEHOLDER_NO_GPU" if ctx.placeholder else "BAKEOFF_WALLCLOCK"
    return {
        "schema_version": 1,
        "package_version": __version__,
        "eval": "wall_clock_subgraph_latency_and_plan_selection",
        "gpu_measurement_status": status,
        "nsight_used": False,
        "quote_as_scic_accuracy": False,
        "claims_fa3": False,
        "claims_stage_stream": False,
        "claims_custom_memmode_kernel": False,
        "claims_custom_computemode_kernel": False,
        "graph_inspected": False,
        "do_not_invent_numbers": True,
        "note": (
            "Latencies are whatever bakeoff_latency.csv recorded for this process. "
            "Do not type them into the paper from memory, and do not fill a failed "
            "row. quote_as_scic_accuracy is false. sdpa is not FlashAttention-3. "
            "always_fuse and always_unfuse are not custom MemMode or ComputeMode kernels. "
            "The regime column uses the requested datasheet ridge. "
            "Library attention scores compile_sdpa_chain (always_fuse) for the "
            "epilogue, including in ridge and compute. paper/data/bakeoff is the "
            "pre-fix selector. paper/data/bakeoff_v2 is the attention-fix "
            "wall-clock on a busy GPU and predates GEMM calibration. "
            "Ridge and compute GEMM (scope ridge-compute) take a few-shot "
            "wall-clock of always_fuse versus always_unfuse. selected_baseline "
            "follows measured_winner when that probe returns one, else the "
            "analytical baseline. The probe is wall-clock on this machine, not "
            "DRAM AI. A busy GPU can flip the winner. It is not SCIC accuracy."
        ),
        "machine": machine.name,
        "ridge_flops_per_byte": machine.ridge,
        "ridge_source": "datasheet",
        "thresholds": {"mem_frac": thresholds.mem_frac, "compute_frac": thresholds.compute_frac},
        "level": level,
        "attention": attention,
        "skip_max_autotune": skip_max_autotune,
        "repeats": repeats,
        "warmup": warmup,
        "gemm_calibration": {
            "scope": calibrate,
            "metric": CALIBRATION_METRIC,
            "repeats": calibration_repeats,
            "warmup": calibration_warmup,
            "not_dram_ai": True,
            "gpu_contention_can_flip_winner": True,
            "quote_as_scic_accuracy": False,
            "attention_policy": "compile_sdpa_chain",
            "plan_cache_entries": plan_cache_entries,
        },
        "backend": ctx.backend,
        "device_name": ctx.device_name,
        "torch_version": ctx.torch_version,
        "cuda_version": ctx.cuda_version,
        "placeholder": ctx.placeholder,
        "rows": len(latency_rows),
        "rows_ok": ok,
        "decisions": len(decisions),
        "selected_baseline_counts": _counts(decisions, "selected_baseline"),
        "selection_source_counts": _counts(decisions, "selection_source"),
        "rows_file": "bakeoff_latency.csv",
        "decisions_file": "bakeoff_decisions.csv",
        "baselines_gemm": list(GEMM_BASELINES),
        "baselines_attention": list(ATTENTION_BASELINES) if attention else [],
        "still_tbd": [
            "Nsight DRAM bytes and achieved AI (not collected; Docker bakeoff does not run Nsight)",
            "SCIC accuracy against isolated Nsight AI (quote_as_scic_accuracy is false)",
            "Custom MemMode and ComputeMode CUDA kernels",
            "FlashAttention-3 kernels (sdpa rows are the PyTorch library operator)",
            "Stage-Stream",
            "A measured ridge on this GPU",
            "End-to-end tokens/s",
            "Plan-switch overhead on a serving trace",
            "Nsight DRAM AI for the GEMM probe. measured_winner is wall-clock "
            "on this machine. GPU contention can flip it. Not SCIC accuracy",
            "A re-run of scripts/run_baseline_bakeoff.sh. paper/data/bakeoff_v2 "
            "predates the probe and still copies always_unfuse on ridge and compute GEMM",
            "paper/data/bakeoff is the pre-fix attention selector. "
            "paper/data/bakeoff_v2 is the compile_sdpa_chain wall-clock on a busy GPU",
        ],
        "repro": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "command": " ".join(sys.argv),
        },
    }


def _counts(rows: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, ""))
        out[value] = out.get(value, 0) + 1
    return out


def _send_back(out: Path, summary: dict) -> str:
    return "\n".join(
        [
            "Send this directory back. It is the Docker baseline bakeoff.",
            "Nsight was not run. Do not add profiler numbers on the way out.",
            f"gpu_measurement_status: {summary['gpu_measurement_status']}",
            "",
            "Required:",
            f"  {out / 'bakeoff_latency.csv'}",
            f"  {out / 'bakeoff_decisions.csv'}",
            f"  {out / 'bakeoff_summary.json'}",
            f"  {out / 'nvidia_smi.txt'}   (written by scripts/run_baseline_bakeoff.sh)",
            f"  {out / 'torch_version.txt'}  (written by scripts/run_baseline_bakeoff.sh)",
            "",
            "CSV columns that the paper reads:",
            "  baseline, shape, median_ms, speedup_vs_eager, honesty",
            "",
            "Do not:",
            "  - invent a median for a row whose ok field is false",
            "  - quote any row as SCIC accuracy (quote_as_scic_accuracy is false)",
            "  - call sdpa a FlashAttention-3 kernel (claims_fa3 is false)",
            "  - call always_fuse or always_unfuse a custom MemMode or ComputeMode kernel",
            "  - treat the datasheet regime column as a measured ridge",
            "  - describe this directory as an Nsight capture",
            "  - claim Stage-Stream ran",
            "",
            "scic_selected copies median_ms from timing_copied_from.",
            "That column is always_fuse or always_unfuse.",
            "bakeoff_decisions.csv logs analytical_baseline, measured_winner, and selection_source.",
            "measured_winner is a few-shot wall-clock on this machine, not DRAM AI.",
            "A busy GPU can flip it. It is not SCIC accuracy.",
            "Attention stays compile_sdpa_chain and is not wall-clock calibrated.",
            "",
            "Still TBD:",
            *[f"  - {item}" for item in summary["still_tbd"]],
            "",
        ]
    )


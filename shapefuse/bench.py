"""Vendor-matmul microbenchmark.

CUDA uses PyTorch (cuBLAS). If CUDA is missing, the timer falls back to
PyTorch CPU or NumPy FP32. Either fallback is a placeholder: it is not an
H100 result.

The timer does not measure fused MemMode versus ComputeMode kernels. Those
kernels are not generated. Achieved AI needs DRAM bytes from a profiler and
is left null.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from shapefuse.intensity import classify, gemm_ai, gemm_flops
from shapefuse.machines import H100_SXM_FP16

SMOKE_SHAPES: list[tuple[str, int, int, int]] = [
    ("smoke_m64", 64, 256, 256),
    ("smoke_m256", 256, 256, 256),
]

DEFAULT_SHAPES: list[tuple[str, int, int, int]] = [
    ("decode_m1", 1, 1024, 1024),
    ("decode_m32", 32, 1024, 1024),
    ("square_512", 512, 512, 512),
    ("square_1024", 1024, 1024, 1024),
]

# Multiples of 8 so a Tensor-Core kernel is allowed to run. 344 is the
# integer next to the analytical M* (~344.9) that is 8-aligned.
H100_SHAPES: list[tuple[str, int, int, int]] = [
    ("decode_gemv", 1, 4096, 4096),
    ("micro_batch", 4, 4096, 4096),
    ("decode_batch32", 32, 4096, 4096),
    ("near_mstar", 344, 4096, 4096),
    ("ridge_m512", 512, 4096, 4096),
    ("prefill_m2048", 2048, 4096, 4096),
    ("square_large", 4096, 4096, 4096),
]

LEVELS = {
    "smoke": SMOKE_SHAPES,
    "default": DEFAULT_SHAPES,
    "h100": H100_SHAPES,
}


class CudaUnavailable(RuntimeError):
    pass


@dataclass
class BenchContext:
    backend: str
    device_name: str
    torch_version: str | None
    cuda_version: str | None
    placeholder: bool
    note: str


def try_import_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def resolve_backend(device: str, require_cuda: bool, allow_cpu_fallback: bool) -> BenchContext:
    requested = device.lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"device must be auto, cpu, or cuda, got {device!r}")
    torch = try_import_torch()
    cuda_ok = bool(torch is not None and torch.cuda.is_available())
    if requested == "auto":
        requested = "cuda" if cuda_ok else "cpu"
    if requested == "cuda" and not cuda_ok:
        message = (
            "CUDA was requested but PyTorch with CUDA is not available. "
            "Install a CUDA build of PyTorch, or rerun with --device cpu."
        )
        if require_cuda or not allow_cpu_fallback:
            raise CudaUnavailable(message)
        requested = "cpu"
    if requested == "cuda":
        assert torch is not None
        return BenchContext(
            backend="cuda-torch",
            device_name=torch.cuda.get_device_name(0),
            torch_version=torch.__version__,
            cuda_version=getattr(torch.version, "cuda", None),
            placeholder=False,
            note=(
                "Latency of torch.matmul only. Not a fused-kernel comparison. "
                "nsight_dram_bytes and achieved_ai are TBD."
            ),
        )
    if torch is not None:
        return BenchContext(
            backend="cpu-torch",
            device_name="cpu",
            torch_version=torch.__version__,
            cuda_version=None,
            placeholder=True,
            note="CPU fallback. Do not report these latencies as H100 numbers.",
        )
    return BenchContext(
        backend="cpu-numpy",
        device_name="cpu",
        torch_version=None,
        cuda_version=None,
        placeholder=True,
        note="NumPy FP32 proxy. Do not report these latencies as H100 numbers.",
    )


def shapes_for_level(level: str, backend: str) -> tuple[list[tuple[str, int, int, int]], str | None]:
    if level not in LEVELS:
        known = ", ".join(LEVELS)
        raise ValueError(f"Unknown bench level {level!r}. Known: {known}")
    warning = None
    if backend != "cuda-torch" and level == "h100":
        warning = "Downgraded bench-level h100 to smoke because the backend is not CUDA."
        level = "smoke"
    return list(LEVELS[level]), warning


def run_bench(
    ctx: BenchContext,
    level: str,
    repeats: int | None = None,
    warmup: int | None = None,
) -> tuple[list[dict], dict]:
    shapes, warning = shapes_for_level(level, ctx.backend)
    if ctx.backend == "cuda-torch":
        warmup_n = 5 if warmup is None else warmup
        repeat_n = 10 if repeats is None else repeats
    else:
        warmup_n = 1 if warmup is None else warmup
        repeat_n = 3 if repeats is None else repeats
    rows = []
    for label, m, n, k in shapes:
        rows.append(_time_one(ctx, label, m, n, k, repeat_n, warmup_n))
    measured = [r for r in rows if r["median_s"] is not None]
    peak = max((r["achieved_tflops"] for r in measured), default=None)
    meta = {
        "backend": ctx.backend,
        "device_name": ctx.device_name,
        "torch_version": ctx.torch_version,
        "cuda_version": ctx.cuda_version,
        "placeholder": ctx.placeholder,
        "note": ctx.note,
        "warning": warning,
        "bench_level": "smoke" if warning else level,
        "what_was_timed": "torch.matmul or numpy matmul; not MemMode/ComputeMode kernels",
        "nsight_dram_bytes": None,
        "nsight_status": "TBD_RUN_PROFILER",
        "measured_hbm_bandwidth_bytes_per_s": None,
        "measured_ridge_flops_per_byte": None,
        "achieved_tflops_max": peak,
        "datasheet_h100_fp16_tflops": H100_SXM_FP16.peak_tflops,
        "fraction_of_h100_fp16_datasheet_peak": (
            None if peak is None or ctx.placeholder else peak / H100_SXM_FP16.peak_tflops
        ),
    }
    return rows, meta


def run_cpu_proxy_sweep(max_m: int = 256, n: int = 512, k: int = 512) -> list[dict]:
    """Short NumPy FP32 sweep. Qualitative only; the ridge it implies is the CPU's."""
    rows = []
    m = 1
    while m <= max_m:
        med = _numpy_median(m, n, k, repeats=3, warmup=1)
        flops = gemm_flops(m, n, k)
        bytes_fp32 = (m * k + k * n + m * n) * 4
        ai16 = gemm_ai(m, n, k, 2, 2, 2)
        rows.append(
            {
                "M": m,
                "N": n,
                "K": k,
                "AI_fp16_analytical": ai16,
                "regime_vs_H100_datasheet": classify(ai16, H100_SXM_FP16.ridge),
                "cpu_median_s": med,
                "cpu_GFLOPS": flops / med / 1e9,
                "cpu_GBps_proxy": bytes_fp32 / med / 1e9,
                "timing_status": "NUMPY_FP32_PROXY",
            }
        )
        m *= 2
    return rows


def _time_one(ctx: BenchContext, label: str, m: int, n: int, k: int, repeats: int, warmup: int) -> dict:
    ai16 = gemm_ai(m, n, k, 2, 2, 2)
    base = {
        "label": label,
        "M": m,
        "N": n,
        "K": k,
        "backend": ctx.backend,
        "device_name": ctx.device_name,
        "dtype_requested": "fp16",
        "analytical_ai_fp16": ai16,
        "regime_vs_h100_datasheet": classify(ai16, H100_SXM_FP16.ridge),
        "nsight_dram_bytes": None,
        "achieved_ai": None,
        "achieved_ai_status": "TBD_NSIGHT",
        "kernel": "pytorch_matmul" if ctx.backend != "cpu-numpy" else "numpy_matmul",
        "fused_plan_timed": False,
        "placeholder": ctx.placeholder,
    }
    if ctx.backend == "cpu-numpy":
        med, dtype_ran = _numpy_median(m, n, k, repeats, warmup), "fp32"
    elif ctx.backend == "cpu-torch":
        med, dtype_ran = _torch_median(m, n, k, repeats, warmup, cuda=False)
    else:
        med, dtype_ran = _torch_median(m, n, k, repeats, warmup, cuda=True)
    flops = gemm_flops(m, n, k)
    base["dtype_ran"] = dtype_ran
    base["median_s"] = med
    base["achieved_tflops"] = flops / med / 1e12
    return base


def _numpy_median(m: int, n: int, k: int, repeats: int, warmup: int) -> float:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((m, k), dtype=np.float32)
    b = rng.standard_normal((k, n), dtype=np.float32)
    for _ in range(warmup):
        c = a @ b
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        c = a @ b
        _ = float(c.reshape(-1)[0])
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _torch_median(m: int, n: int, k: int, repeats: int, warmup: int, cuda: bool) -> tuple[float, str]:
    import torch

    device = torch.device("cuda") if cuda else torch.device("cpu")
    dtype = torch.float16 if cuda else torch.float32
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    a = torch.randn(m, k, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    b = torch.randn(k, n, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    if cuda:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        for _ in range(warmup):
            c = a @ b
        torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            starter.record()
            c = a @ b
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender) / 1e3)
        _ = c.reshape(-1)[0].item()
        return float(np.median(times)), "fp16"
    for _ in range(warmup):
        c = a @ b
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        c = a @ b
        _ = c.reshape(-1)[0].item()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), "fp32"

"""Unfused versus fused-proxy timings. Not MemMode or ComputeMode.

Honesty levels, in the order a CUDA run attempts them:

- ``TWO_KERNEL_BASELINE_NOT_A_FUSED_GEMM``: time ``matmul`` and RMSNorm as
  two regions and add the medians.
- ``EAGER_CHAIN_STILL_SEPARATE_OPS``: time the same two ops inside one
  Python region. Eager mode still launches both.
- ``TORCH_COMPILE_PROXY_FUSION_NOT_VERIFIED``: ``torch.compile`` on that
  chain. Inductor may or may not fuse. This repo does not inspect the graph,
  so a speedup is not evidence of fusion.
- ``ELEMENTWISE_TWO_KERNEL_NOT_GEMM_FUSION`` versus
  ``ELEMENTWISE_HAND_FUSED_STUB_NOT_GEMM_FUSION``: bias+GELU only. The GEMM
  output is already materialized. The hand stub is one Python function of
  add and a tanh GELU. On eager CUDA that can still be more than one kernel.
- ``TRITON_ELEMENTWISE_STUB_NOT_GEMM_FUSION``: a Triton kernel for
  bias+approximate-GELU when Triton imports. It does not fuse the GEMM.

No row sets a MemMode or ComputeMode kernel as timed. Stage-Stream is not
involved. CPU and NumPy rows are placeholders.
"""

from __future__ import annotations

import time

import numpy as np

from shapefuse.bench import BenchContext

HONESTY_TWO_KERNEL = "TWO_KERNEL_BASELINE_NOT_A_FUSED_GEMM"
HONESTY_EAGER = "EAGER_CHAIN_STILL_SEPARATE_OPS"
HONESTY_COMPILE = "TORCH_COMPILE_PROXY_FUSION_NOT_VERIFIED"
HONESTY_COMPILE_UNAVAILABLE = "TORCH_COMPILE_UNAVAILABLE"
HONESTY_BIAS_UNFUSED = "ELEMENTWISE_TWO_KERNEL_NOT_GEMM_FUSION"
HONESTY_BIAS_HAND = "ELEMENTWISE_HAND_FUSED_STUB_NOT_GEMM_FUSION"
HONESTY_TRITON = "TRITON_ELEMENTWISE_STUB_NOT_GEMM_FUSION"
HONESTY_TRITON_UNAVAILABLE = "TRITON_UNAVAILABLE"
HONESTY_TRITON_FAILED = "TRITON_KERNEL_FAILED"

_GELU_K0 = 0.7978845608028654  # sqrt(2/pi)
_GELU_K1 = 0.044715


def run_fusion_proxy(
    bench_rows: list[dict],
    ctx: BenchContext,
    repeats: int | None = None,
    warmup: int | None = None,
) -> tuple[list[dict], dict]:
    shapes = [(str(r["label"]), int(r["M"]), int(r["N"]), int(r["K"])) for r in bench_rows]
    cuda = ctx.backend == "cuda-torch"
    if cuda:
        warmup_n = 3 if warmup is None else warmup
        repeat_n = 5 if repeats is None else repeats
    else:
        warmup_n = 1 if warmup is None else warmup
        repeat_n = 2 if repeats is None else repeats
    rows: list[dict] = []
    triton_used = False
    compile_used = False
    for label, m, n, k in shapes:
        if ctx.backend == "cpu-numpy":
            produced = _numpy_shapes(label, m, n, k, repeat_n, warmup_n, ctx)
        else:
            produced, flags = _torch_shapes(label, m, n, k, repeat_n, warmup_n, ctx, cuda=cuda)
            triton_used = triton_used or flags["triton"]
            compile_used = compile_used or flags["compile"]
        rows.extend(produced)
    meta = {
        "ran": True,
        "status": "PROXY_NOT_FUSED_GEMM",
        "claims_memmode_kernel": False,
        "claims_computemode_kernel": False,
        "claims_stage_stream": False,
        "triton_used": triton_used,
        "torch_compile_used": compile_used,
        "placeholder": ctx.placeholder,
        "backend": ctx.backend,
        "device_name": ctx.device_name,
        "note": (
            "Proxy only. Two-kernel matmul+RMSNorm versus an eager chain and, "
            "when it runs, torch.compile. Bias+GELU compares two eager ops "
            "with one Python expression and, if importable, one Triton "
            "elementwise kernel. None of these is a fused GEMM, MemMode, "
            "ComputeMode, or Stage-Stream. Do not report the time ratio as a "
            "fusion speedup."
        ),
    }
    return rows, meta


def _row(
    ctx: BenchContext,
    label: str,
    m: int,
    n: int,
    k: int,
    op: str,
    variant: str,
    honesty: str,
    median_s: float | None,
    unfused_s: float | None,
    dtype_ran: str,
    ok: bool,
    error: str | None,
) -> dict:
    ratio = None
    if median_s is not None and unfused_s is not None and median_s > 0:
        ratio = unfused_s / median_s
    return {
        "label": label,
        "M": m,
        "N": n,
        "K": k,
        "op": op,
        "variant": variant,
        "honesty": honesty,
        "median_s": median_s,
        "unfused_reference_s": unfused_s,
        "time_ratio_unfused_over_this": ratio,
        "time_ratio_status": "PROXY_TIMING_NOT_A_FUSION_SPEEDUP",
        "backend": ctx.backend,
        "device_name": ctx.device_name,
        "placeholder": ctx.placeholder,
        "dtype_ran": dtype_ran,
        "memmode_kernel": False,
        "computemode_kernel": False,
        "kernel_emitted": False,
        "ok": ok,
        "error": error or "",
    }


def _numpy_shapes(label, m, n, k, repeats, warmup, ctx) -> list[dict]:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((m, k), dtype=np.float32)
    b = rng.standard_normal((k, n), dtype=np.float32)
    # RMSNorm runs on the matmul output [M, N], so the weight length is N.
    weight = np.ones((n,), dtype=np.float32)

    def matmul():
        return a @ b

    def norm(c):
        var = np.mean(c * c, axis=-1, keepdims=True)
        return c * (1.0 / np.sqrt(var + 1e-6)) * weight

    med_mm = _cpu_median(matmul, repeats, warmup)
    c = matmul()
    med_norm = _cpu_median(lambda: norm(c), repeats, warmup)
    unfused = med_mm + med_norm
    med_chain = _cpu_median(lambda: norm(matmul()), repeats, warmup)
    bias = np.ones((n,), dtype=np.float32)
    biased = c + bias
    med_bias = _cpu_median(lambda: c + bias, repeats, warmup)
    med_gelu = _cpu_median(lambda: _gelu_np(biased), repeats, warmup)
    med_hand = _cpu_median(lambda: _gelu_np(c + bias), repeats, warmup)
    bias_unfused = med_bias + med_gelu
    base = dict(label=label, m=m, n=n, k=k, ctx=ctx, dtype="fp32")
    return [
        _row(ctx, label, m, n, k, "matmul_rmsnorm", "unfused_two_regions", HONESTY_TWO_KERNEL, unfused, unfused, "fp32", True, None),
        _row(ctx, label, m, n, k, "matmul_rmsnorm", "eager_chain", HONESTY_EAGER, med_chain, unfused, "fp32", True, None),
        _row(
            ctx, label, m, n, k, "matmul_rmsnorm", "torch_compile", HONESTY_COMPILE_UNAVAILABLE,
            None, unfused, "fp32", False, "torch.compile is not used on the NumPy path",
        ),
        _row(
            ctx, label, m, n, k, "bias_gelu", "unfused_two_regions", HONESTY_BIAS_UNFUSED,
            bias_unfused, bias_unfused, "fp32", True, None,
        ),
        _row(ctx, label, m, n, k, "bias_gelu", "hand_elementwise_stub", HONESTY_BIAS_HAND, med_hand, bias_unfused, "fp32", True, None),
        _row(
            ctx, label, m, n, k, "bias_gelu", "triton", HONESTY_TRITON_UNAVAILABLE,
            None, bias_unfused, "fp32", False, "Triton is not used on the NumPy path",
        ),
    ]


def _torch_shapes(label, m, n, k, repeats, warmup, ctx, cuda: bool):
    import torch

    device = torch.device("cuda" if cuda else "cpu")
    dtype = torch.float16 if cuda else torch.float32
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    a = torch.randn(m, k, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    b = torch.randn(k, n, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    weight = torch.ones(n, device=device, dtype=dtype)
    bias = torch.ones(n, device=device, dtype=dtype)
    flags = {"triton": False, "compile": False}
    dtype_name = "fp16" if cuda else "fp32"

    def rmsnorm(c):
        var = (c * c).mean(dim=-1, keepdim=True)
        return c * torch.rsqrt(var + 1e-6) * weight

    def chain():
        return rmsnorm(a @ b)

    med_mm = _torch_median(lambda: a @ b, repeats, warmup, cuda)
    c = a @ b
    med_norm = _torch_median(lambda: rmsnorm(c), repeats, warmup, cuda)
    unfused = med_mm + med_norm
    med_eager = _torch_median(chain, repeats, warmup, cuda)
    rows = [
        _row(ctx, label, m, n, k, "matmul_rmsnorm", "unfused_two_regions", HONESTY_TWO_KERNEL, unfused, unfused, dtype_name, True, None),
        _row(ctx, label, m, n, k, "matmul_rmsnorm", "eager_chain", HONESTY_EAGER, med_eager, unfused, dtype_name, True, None),
    ]
    compile_row, compiled = _compile_row(ctx, label, m, n, k, chain, unfused, repeats, warmup, cuda, dtype)
    rows.append(compile_row)
    flags["compile"] = compiled
    biased = c + bias
    med_add = _torch_median(lambda: c + bias, repeats, warmup, cuda)
    med_act = _torch_median(lambda: _gelu_torch(biased), repeats, warmup, cuda)
    bias_unfused = med_add + med_act
    med_hand = _torch_median(lambda: _gelu_torch(c + bias), repeats, warmup, cuda)
    rows.append(
        _row(ctx, label, m, n, k, "bias_gelu", "unfused_two_regions", HONESTY_BIAS_UNFUSED, bias_unfused, bias_unfused, dtype_name, True, None)
    )
    rows.append(
        _row(ctx, label, m, n, k, "bias_gelu", "hand_elementwise_stub", HONESTY_BIAS_HAND, med_hand, bias_unfused, dtype_name, True, None)
    )
    tri_row, used = _triton_row(ctx, label, m, n, k, c, bias, bias_unfused, repeats, warmup, cuda, dtype_name)
    rows.append(tri_row)
    flags["triton"] = used
    return rows, flags


def _compile_row(ctx, label, m, n, k, chain, unfused, repeats, warmup, cuda, dtype):
    import torch

    if not hasattr(torch, "compile"):
        return (
            _row(
                ctx, label, m, n, k, "matmul_rmsnorm", "torch_compile", HONESTY_COMPILE_UNAVAILABLE,
                None, unfused, "fp16" if cuda else "fp32", False, "torch.compile is missing",
            ),
            False,
        )
    try:
        compiled = torch.compile(chain)
        median = _torch_median(compiled, repeats, warmup, cuda)
    except Exception as exc:
        return (
            _row(
                ctx, label, m, n, k, "matmul_rmsnorm", "torch_compile", HONESTY_COMPILE_UNAVAILABLE,
                None, unfused, "fp16" if cuda else "fp32", False, str(exc),
            ),
            False,
        )
    return (
        _row(
            ctx, label, m, n, k, "matmul_rmsnorm", "torch_compile", HONESTY_COMPILE,
            median, unfused, "fp16" if cuda else "fp32", True, None,
        ),
        True,
    )


def _triton_row(ctx, label, m, n, k, c, bias, unfused, repeats, warmup, cuda, dtype_name):
    if not cuda:
        return (
            _row(
                ctx, label, m, n, k, "bias_gelu", "triton", HONESTY_TRITON_UNAVAILABLE,
                None, unfused, dtype_name, False, "Triton elementwise stub runs only on CUDA",
            ),
            False,
        )
    try:
        import triton  # noqa: F401
    except ImportError as exc:
        return (
            _row(
                ctx, label, m, n, k, "bias_gelu", "triton", HONESTY_TRITON_UNAVAILABLE,
                None, unfused, dtype_name, False, str(exc),
            ),
            False,
        )
    try:
        launch = _triton_bias_act_launcher()
        median = _torch_median(lambda: launch(c, bias), repeats, warmup, cuda=True)
    except Exception as exc:
        return (
            _row(
                ctx, label, m, n, k, "bias_gelu", "triton", HONESTY_TRITON_FAILED,
                None, unfused, dtype_name, False, str(exc),
            ),
            False,
        )
    return (
        _row(
            ctx, label, m, n, k, "bias_gelu", "triton", HONESTY_TRITON,
            median, unfused, dtype_name, True, None,
        ),
        True,
    )


def _triton_bias_act_launcher():
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _kernel(x_ptr, bias_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        z = tl.load(x_ptr + offs, mask=mask, other=0.0) + tl.load(bias_ptr + offs, mask=mask, other=0.0)
        # Approximate GELU via sigmoid. One elementwise kernel, not a GEMM.
        s = 1.0 / (1.0 + tl.exp(-1.702 * z))
        tl.store(out_ptr + offs, z * s, mask=mask)

    def launch(x, bias):
        flat_x = x.reshape(-1)
        flat_b = bias.reshape(-1)
        if flat_b.numel() == 1:
            flat_b = flat_b.expand(flat_x.numel())
        elif flat_b.numel() != x.shape[-1]:
            raise RuntimeError("bias length does not match N")
        # Broadcast the row bias across M without claiming a fused GEMM.
        bias_full = bias.view(1, -1).expand(x.shape[0], x.shape[1]).reshape(-1)
        out = torch.empty_like(flat_x)
        n = flat_x.numel()
        block = 1024
        grid = (triton.cdiv(n, block),)
        _kernel[grid](flat_x, bias_full, out, n, BLOCK=block)
        return out.view_as(x)

    return launch


def _gelu_np(z: np.ndarray) -> np.ndarray:
    inner = _GELU_K0 * (z + _GELU_K1 * z * z * z)
    return 0.5 * z * (1.0 + np.tanh(inner))


def _gelu_torch(z):
    import torch

    inner = _GELU_K0 * (z + _GELU_K1 * z * z * z)
    return 0.5 * z * (1.0 + torch.tanh(inner))


def _cpu_median(fn, repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        out = fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        if isinstance(out, np.ndarray):
            _ = float(out.reshape(-1)[0])
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _torch_median(fn, repeats: int, warmup: int, cuda: bool) -> float:
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
        _ = out.reshape(-1)[0].item()
        return float(np.median(times))
    for _ in range(warmup):
        out = fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        _ = out.reshape(-1)[0].item()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))

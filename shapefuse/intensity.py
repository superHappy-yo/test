"""Closed-form arithmetic intensity and Roofline time.

Traffic is a cold-HBM lower bound: each live tensor moves once, with no L2
reuse. FLOP counts follow the usual 2-for-a-MAC convention. Nothing here is
a measured byte count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shapefuse.machines import Machine

# Classifier band from the project note. These are policy knobs, not physical
# constants: inside [mem_frac, compute_frac] * R the site is ridge-region and
# the selector emits both plans.
DEFAULT_MEM_FRAC = 0.5
DEFAULT_COMPUTE_FRAC = 2.0

ELEM_BYTES: dict[str, float] = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "fp8": 1,
    "int8": 1,
    "int32": 4,
}

# Project-note conventions for elementwise-style ops. They fix the O(1)
# intensity; they are not instruction traces.
SOFTMAX_FLOPS_PER_ELEM = 5.0
SOFTMAX_MEM_PASSES = 3.0
RMSNORM_FLOPS_PER_ELEM = 4.0
LAYERNORM_FLOPS_PER_ELEM = 7.0


def elem_bytes(dtype: str) -> float:
    key = dtype.lower()
    if key not in ELEM_BYTES:
        known = ", ".join(sorted(ELEM_BYTES))
        raise KeyError(f"Unknown dtype {dtype!r}. Known: {known}")
    return ELEM_BYTES[key]


@dataclass(frozen=True)
class Thresholds:
    mem_frac: float = DEFAULT_MEM_FRAC
    compute_frac: float = DEFAULT_COMPUTE_FRAC

    def __post_init__(self) -> None:
        if self.mem_frac <= 0 or self.compute_frac <= self.mem_frac:
            raise ValueError("Need 0 < mem_frac < compute_frac")


@dataclass(frozen=True)
class Intensity:
    flops: float
    nbytes: float

    @property
    def ai(self) -> float:
        if self.nbytes <= 0:
            return math.inf
        return self.flops / self.nbytes


def classify(ai: float, ridge: float, thresholds: Thresholds | None = None) -> str:
    """Return memory-bound, ridge-region, or compute-bound."""
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    band = thresholds or Thresholds()
    if ai < band.mem_frac * ridge:
        return "memory-bound"
    if ai > band.compute_frac * ridge:
        return "compute-bound"
    return "ridge-region"


def limiting_resource(ai: float, ridge: float) -> str:
    """Strict Roofline knee at AI = R, distinct from the three-way band."""
    if ai < ridge:
        return "bandwidth"
    if ai > ridge:
        return "compute"
    return "balanced"


def roofline_seconds(flops: float, nbytes: float, machine: Machine) -> float:
    """max(FLOPs/peak, bytes/bandwidth). No launch term."""
    t_comp = flops / machine.peak_flops_per_s
    t_mem = nbytes / machine.peak_bytes_per_s
    return max(t_comp, t_mem)


def gemm_flops(m: int, n: int, k: int) -> float:
    _positive(m, n, k)
    return 2.0 * m * n * k


def gemm_bytes(
    m: int,
    n: int,
    k: int,
    elem_a: float = 2.0,
    elem_b: float = 2.0,
    elem_c: float = 2.0,
) -> float:
    """Read A and B, write C, each once."""
    _positive(m, n, k)
    return m * k * elem_a + k * n * elem_b + m * n * elem_c


def gemm_ai(
    m: int,
    n: int,
    k: int,
    elem_a: float = 2.0,
    elem_b: float = 2.0,
    elem_c: float = 2.0,
) -> float:
    return gemm_flops(m, n, k) / gemm_bytes(m, n, k, elem_a, elem_b, elem_c)


def gemm_m_for_target_ai(d: float, target_ai: float, elem: float = 2.0) -> float:
    """M where N = K = d has arithmetic intensity ``target_ai``.

    AI = 2 M d / ((d + 2 M) elem). Solved:
    M = target * elem * d / (2 * (d - target * elem)), for d > target * elem.
    FP16 (elem = 2) reduces to M = target * d / (d - 2 * target), which is
    the project-note formula M* = R d / (d - 2 R) when target = R.
    """
    if d <= 0 or target_ai <= 0 or elem <= 0:
        raise ValueError("d, target_ai, and elem must be positive")
    denom = 2.0 * (d - target_ai * elem)
    if denom <= 0:
        return math.inf
    return (target_ai * elem * d) / denom


def square_proj_ai(m: float, d: float, elem: float = 2.0) -> float:
    """AI of [M, d] x [d, d] with uniform element size."""
    return (2.0 * m * d) / ((d + 2.0 * m) * elem)


def attn_qkt(b: int, h: int, s: int, d: int, elem: float = 2.0) -> Intensity:
    """QK^T that writes the score matrix to HBM."""
    _positive(b, h, s, d)
    flops = 2.0 * b * h * s * s * d
    nbytes = (2.0 * b * h * s * d + b * h * s * s) * elem
    return Intensity(flops, nbytes)


def softmax_scores(b: int, h: int, s: int, elem: float = 2.0) -> Intensity:
    """Row softmax over an S x S score matrix. AI is independent of S."""
    _positive(b, h, s)
    n = float(b * h * s * s)
    return Intensity(SOFTMAX_FLOPS_PER_ELEM * n, SOFTMAX_MEM_PASSES * n * elem)


def attn_fa_prefill(b: int, h: int, s: int, d: int, elem: float = 2.0) -> Intensity:
    """QK^T + PV with scores kept off HBM. Traffic is Q, K, V, O once.

    With this traffic model AI = S / elem, independent of head dim.
    """
    _positive(b, h, s, d)
    flops = 4.0 * b * h * s * s * d
    nbytes = 4.0 * b * h * s * d * elem
    return Intensity(flops, nbytes)


def attn_unfused_prefill(b: int, h: int, s: int, d: int, elem: float = 2.0) -> Intensity:
    """Same FLOPs as FA, but the score tensor is written and reread.

    Bytes = (Q + K + V + O + 3 score passes) * elem. As S grows, AI
    saturates at 4 D / (3 elem) and does not track S the way FA does.
    """
    _positive(b, h, s, d)
    flops = 4.0 * b * h * s * s * d
    nbytes = (4.0 * b * h * s * d + 3.0 * b * h * s * s) * elem
    return Intensity(flops, nbytes)


def attn_fa_decode(
    b: int,
    h_q: int,
    h_kv: int,
    s_kv: int,
    d: int,
    elem: float = 2.0,
    s_q: int = 1,
) -> Intensity:
    """Decode (or short-query) attention. K and V are the cache.

    As S_kv grows with S_q fixed, AI -> 2 H_q / (H_kv * elem). MHA FP16
    therefore sits near 1 FLOP/byte for any cache length.
    """
    _positive(b, h_q, h_kv, s_kv, d, s_q)
    flops = 4.0 * b * h_q * s_q * s_kv * d
    nbytes = (2.0 * b * h_q * s_q * d + 2.0 * b * h_kv * s_kv * d) * elem
    return Intensity(flops, nbytes)


def rmsnorm(m: int, k: int, elem: float = 2.0) -> Intensity:
    """RMSNorm over the last dimension of an [M, K] activation."""
    _positive(m, k)
    flops = RMSNORM_FLOPS_PER_ELEM * m * k
    nbytes = (2.0 * m * k + k) * elem
    return Intensity(flops, nbytes)


def layernorm(m: int, k: int, elem: float = 2.0) -> Intensity:
    _positive(m, k)
    flops = LAYERNORM_FLOPS_PER_ELEM * m * k
    nbytes = (2.0 * m * k + 2.0 * k) * elem
    return Intensity(flops, nbytes)


def reduction_sum(m: int, k: int, elem: float = 2.0) -> Intensity:
    """Sum-reduce the last dimension. One pass read, one scalar write per row."""
    _positive(m, k)
    flops = float(m * k)
    nbytes = (m * k + m) * elem
    return Intensity(flops, nbytes)


def elementwise(
    n_elem: int,
    flops_per_elem: float,
    n_reads: int,
    n_writes: int,
    elem: float = 2.0,
) -> Intensity:
    if n_elem <= 0:
        raise ValueError("n_elem must be positive")
    if flops_per_elem < 0 or n_reads < 0 or n_writes < 0:
        raise ValueError("traffic counts must be non-negative")
    flops = flops_per_elem * n_elem
    nbytes = (n_reads + n_writes) * n_elem * elem
    return Intensity(flops, nbytes)


def moe_grouped(
    token_counts: list[int],
    hidden: int,
    intermediate: int,
    elem_a: float = 2.0,
    elem_b: float = 2.0,
    elem_c: float = 2.0,
    load_empty_experts: int = 0,
) -> tuple[Intensity, list[float]]:
    """Grouped GEMM, one [t, H] x [H, I] per occupied expert.

    ``load_empty_experts`` adds that many cold weight matrices with no FLOPs
    (a dispatch that touches inactive experts). Per-expert AIs follow the
    occupied experts only.
    """
    if hidden <= 0 or intermediate <= 0:
        raise ValueError("hidden and intermediate must be positive")
    flops = 0.0
    nbytes = 0.0
    per: list[float] = []
    occupied = [t for t in token_counts if t > 0]
    if not occupied:
        raise ValueError("need at least one expert with tokens")
    for t in occupied:
        e_flops = 2.0 * t * hidden * intermediate
        e_bytes = (
            t * hidden * elem_a
            + hidden * intermediate * elem_b
            + t * intermediate * elem_c
        )
        flops += e_flops
        nbytes += e_bytes
        per.append(e_flops / e_bytes)
    if load_empty_experts < 0:
        raise ValueError("load_empty_experts must be >= 0")
    nbytes += load_empty_experts * hidden * intermediate * elem_b
    return Intensity(flops, nbytes), per


def gemm_plus_rmsnorm_traffic(
    m: int,
    n: int,
    k: int,
    elem: float = 2.0,
) -> dict[str, float]:
    """Cold-cache GEMM followed by RMSNorm on C, fused or not.

    Unfused moves A, B, write C, reread C, read w, write Y.
    Fused moves A, B, read w, write Y. FLOPs are the same either way.
    The saved fraction of unfused traffic *grows* with M for N = K = d.
    """
    _positive(m, n, k)
    flops = gemm_flops(m, n, k) + RMSNORM_FLOPS_PER_ELEM * m * n
    unfused = (m * k + k * n + 3.0 * m * n + n) * elem
    fused = (m * k + k * n + n + m * n) * elem
    saved = unfused - fused
    return {
        "flops": flops,
        "bytes_unfused": unfused,
        "bytes_fused": fused,
        "bytes_saved": saved,
        "byte_save_fraction": saved / unfused,
        "byte_ratio_unfused_over_fused": unfused / fused,
    }


def roofline_fusion_speedup(
    flops: float,
    bytes_unfused: float,
    bytes_fused: float,
    machine: Machine,
    launch_s: float = 0.0,
    kernels_unfused: int = 2,
    kernels_fused: int = 1,
) -> dict[str, float | str]:
    """Roofline time ratio. ``launch_s`` is an assumption, not a measurement."""
    t_u = roofline_seconds(flops, bytes_unfused, machine)
    t_f = roofline_seconds(flops, bytes_fused, machine)
    t_u_l = t_u + kernels_unfused * launch_s
    t_f_l = t_f + kernels_fused * launch_s
    return {
        "seconds_unfused": t_u,
        "seconds_fused": t_f,
        "speedup": t_u / t_f if t_f > 0 else math.inf,
        "seconds_unfused_with_launch": t_u_l,
        "seconds_fused_with_launch": t_f_l,
        "speedup_with_launch": t_u_l / t_f_l if t_f_l > 0 else math.inf,
        "launch_s": launch_s,
        "launch_status": "ASSUMED_NOT_MEASURED" if launch_s else "DISABLED",
    }


def _positive(*dims: int) -> None:
    for d in dims:
        if d <= 0:
            raise ValueError(f"dimensions must be positive, got {dims}")

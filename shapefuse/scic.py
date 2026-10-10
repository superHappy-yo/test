"""Shape-conditioned intensity classifier (SCIC).

SCIC turns a site (kind, shape, dtypes) into an analytical AI and a regime
relative to a machine ridge. Measured AI can be passed in later; when it is
absent the report says ``source=analytical`` and ``calibrated=False``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shapefuse.intensity import (
    Intensity,
    Thresholds,
    attn_fa_decode,
    attn_fa_prefill,
    attn_qkt,
    attn_unfused_prefill,
    classify,
    elementwise,
    elem_bytes,
    gemm_ai,
    gemm_bytes,
    gemm_flops,
    layernorm,
    limiting_resource,
    moe_grouped,
    reduction_sum,
    rmsnorm,
    softmax_scores,
)
from shapefuse.machines import Machine


@dataclass(frozen=True)
class Site:
    kind: str
    label: str
    shape: dict
    dtype: str = "fp16"
    # Mixed GEMM operands. None means "use dtype" for every tensor.
    dtype_a: str | None = None
    dtype_b: str | None = None
    dtype_c: str | None = None


@dataclass(frozen=True)
class IntensityReport:
    kind: str
    label: str
    ai: float
    flops: float
    nbytes: float
    regime: str
    ai_over_ridge: float
    roofline_limit: str
    machine: str
    ridge: float
    dtype: str
    source: str
    calibrated: bool
    analytical_ai: float
    measured_ai: float | None
    abs_error: float | None
    detail: dict = field(default_factory=dict)


def classify_site(
    site: Site,
    machine: Machine,
    thresholds: Thresholds | None = None,
    measured_ai: float | None = None,
) -> IntensityReport:
    """Classify one site. ``measured_ai`` overrides the regime when provided."""
    band = thresholds or Thresholds()
    analytical, detail = _analytical(site)
    ai = analytical.ai if measured_ai is None else measured_ai
    regime = classify(ai, machine.ridge, band)
    err = None if measured_ai is None else abs(measured_ai - analytical.ai)
    return IntensityReport(
        kind=site.kind,
        label=site.label,
        ai=ai,
        flops=analytical.flops,
        nbytes=analytical.nbytes,
        regime=regime,
        ai_over_ridge=ai / machine.ridge,
        roofline_limit=limiting_resource(ai, machine.ridge),
        machine=machine.name,
        ridge=machine.ridge,
        dtype=site.dtype,
        source="measured" if measured_ai is not None else "analytical",
        calibrated=measured_ai is not None,
        analytical_ai=analytical.ai,
        measured_ai=measured_ai,
        abs_error=err,
        detail=detail,
    )


def _analytical(site: Site) -> tuple[Intensity, dict]:
    kind = site.kind
    s = site.shape
    if kind == "gemm":
        ea, eb, ec = _gemm_elems(site)
        m, n, k = int(s["M"]), int(s["N"]), int(s["K"])
        return Intensity(gemm_flops(m, n, k), gemm_bytes(m, n, k, ea, eb, ec)), {
            "M": m,
            "N": n,
            "K": k,
            "elem_a": ea,
            "elem_b": eb,
            "elem_c": ec,
        }
    if kind == "attn_qkt":
        b, h, seq, d = int(s["B"]), int(s["H"]), int(s["S"]), int(s["D"])
        e = elem_bytes(site.dtype)
        return attn_qkt(b, h, seq, d, e), {"B": b, "H": h, "S": seq, "D": d}
    if kind == "softmax":
        b, h, seq = int(s["B"]), int(s["H"]), int(s["S"])
        return softmax_scores(b, h, seq, elem_bytes(site.dtype)), {"B": b, "H": h, "S": seq}
    if kind == "attention_prefill":
        b, h, seq, d = int(s["B"]), int(s["H"]), int(s["S"]), int(s["D"])
        e = elem_bytes(site.dtype)
        fa = attn_fa_prefill(b, h, seq, d, e)
        unfused = attn_unfused_prefill(b, h, seq, d, e)
        qk = attn_qkt(b, h, seq, d, e)
        sm = softmax_scores(b, h, seq, e)
        # Regime follows the IO-aware variant. Materialized scores are reported
        # beside it and are not a plan this selector emits.
        return fa, {
            "B": b,
            "H": h,
            "S": seq,
            "D": d,
            "ai_fa": fa.ai,
            "ai_unfused": unfused.ai,
            "ai_qkt": qk.ai,
            "ai_softmax": sm.ai,
            "flops_fa": fa.flops,
            "bytes_fa": fa.nbytes,
            "bytes_unfused": unfused.nbytes,
        }
    if kind == "attention_decode":
        e = elem_bytes(site.dtype)
        dec = attn_fa_decode(
            int(s["B"]),
            int(s["H"]),
            int(s["Hkv"]),
            int(s["Sk"]),
            int(s["D"]),
            e,
            int(s.get("Sq", 1)),
        )
        return dec, {
            "B": int(s["B"]),
            "H": int(s["H"]),
            "Hkv": int(s["Hkv"]),
            "Sk": int(s["Sk"]),
            "Sq": int(s.get("Sq", 1)),
            "D": int(s["D"]),
        }
    if kind == "rmsnorm":
        m, k = int(s["M"]), int(s["K"])
        return rmsnorm(m, k, elem_bytes(site.dtype)), {"M": m, "K": k}
    if kind == "layernorm":
        m, k = int(s["M"]), int(s["K"])
        return layernorm(m, k, elem_bytes(site.dtype)), {"M": m, "K": k}
    if kind == "reduction":
        m, k = int(s["M"]), int(s["K"])
        return reduction_sum(m, k, elem_bytes(site.dtype)), {"M": m, "K": k}
    if kind == "ew_chain":
        n = int(s["n"])
        ew = elementwise(
            n,
            float(s["flops_per_elem"]),
            int(s["reads"]),
            int(s["writes"]),
            elem_bytes(site.dtype),
        )
        return ew, {
            "n": n,
            "flops_per_elem": float(s["flops_per_elem"]),
            "reads": int(s["reads"]),
            "writes": int(s["writes"]),
        }
    if kind == "moe_expert":
        ea, eb, ec = _gemm_elems(site)
        t, hidden, inter = int(s["tokens"]), int(s["H"]), int(s["I"])
        total, per = moe_grouped([t], hidden, inter, ea, eb, ec)
        return total, {"tokens": t, "H": hidden, "I": inter, "ai_expert": per[0]}
    if kind == "moe_grouped":
        ea, eb, ec = _gemm_elems(site)
        counts = [int(t) for t in s["token_counts"]]
        total, per = moe_grouped(
            counts,
            int(s["H"]),
            int(s["I"]),
            ea,
            eb,
            ec,
            load_empty_experts=int(s.get("load_empty_experts", 0)),
        )
        return total, {
            "token_counts": counts,
            "H": int(s["H"]),
            "I": int(s["I"]),
            "per_expert_ai": per,
            "load_empty_experts": int(s.get("load_empty_experts", 0)),
        }
    raise KeyError(f"Unknown site kind {kind!r}")


def _gemm_elems(site: Site) -> tuple[float, float, float]:
    base = site.dtype
    return (
        elem_bytes(site.dtype_a or base),
        elem_bytes(site.dtype_b or base),
        elem_bytes(site.dtype_c or base),
    )

"""Shape catalogs and analytical tables.

Rows are pure functions of the datasheet ridge and the traffic model. CPU
timings are not filled here; see :mod:`shapefuse.bench`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shapefuse.intensity import (
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
    gemm_m_for_target_ai,
    gemm_plus_rmsnorm_traffic,
    limiting_resource,
    moe_grouped,
    rmsnorm,
    roofline_fusion_speedup,
    softmax_scores,
)
from shapefuse.machines import A100_SXM80_FP16, A40_FP16, H100_SXM_FP8, Machine
from shapefuse.plans import moe_is_heterogeneous, select_plan
from shapefuse.scic import Site, classify_site

# GEMM shapes from the project note, plus the H100 FP16 ridge crossing.
GEMM_SHAPES: list[tuple[str, int, int, int, str]] = [
    ("square_large", 4096, 4096, 4096, "classic large square"),
    ("square_med", 1024, 1024, 1024, "medium square"),
    ("square_small", 256, 256, 256, "small square"),
    ("decode_gemv", 1, 4096, 4096, "LLM decode M=1"),
    ("decode_batch8", 8, 4096, 4096, "batched decode"),
    ("decode_batch32", 32, 4096, 4096, "batched decode"),
    ("prefill_long", 2048, 4096, 4096, "prefill-like M"),
    ("tall_skinny_K", 4096, 4096, 64, "small K"),
    ("wide_N", 128, 12288, 4096, "FFN up-projection"),
    ("ffn_down", 128, 4096, 12288, "FFN down-projection"),
    ("micro_batch", 4, 4096, 4096, "serving micro-batch"),
    ("moe_expert_small", 16, 4096, 14336, "one expert, few tokens"),
    ("moe_expert_med", 256, 4096, 14336, "one expert, medium tokens"),
    ("moe_expert_large", 2048, 4096, 14336, "one expert, many tokens"),
    ("near_mstar_h100", 345, 4096, 4096, "near H100 FP16 M* for N=K=4096"),
]

EW_OPS: list[tuple[str, float, int, int, str]] = [
    ("residual_add", 1.0, 2, 1, "y = a + b"),
    ("mul", 1.0, 2, 1, "y = a * b"),
    ("silu", 5.0, 1, 1, "y = x * sigmoid(x), 5 FLOP/elem convention"),
    ("gelu", 8.0, 1, 1, "GELU, 8 FLOP/elem convention"),
    ("swiglu_ew", 6.0, 2, 1, "silu(gate) * up, after the two GEMMs"),
]

MOE_SCENARIOS: list[tuple[str, list[int], int]] = [
    ("balanced_8e", [256] * 8, 0),
    ("skewed_8e", [1024, 512, 256, 128, 64, 32, 16, 8], 0),
    ("decode_tiny", [2] * 8, 0),
    ("prefill_fat", [2048] * 8, 0),
    ("one_hot_expert", [2048] + [0] * 7, 0),
    # Same tokens as one_hot, but the other seven weight matrices still move.
    ("one_hot_load_all_weights", [2048], 7),
]


@dataclass
class SweepBundle:
    machine: Machine
    thresholds: Thresholds
    launch_s: float
    gemm: list[dict] = field(default_factory=list)
    gemm_curve: list[dict] = field(default_factory=list)
    attention: list[dict] = field(default_factory=list)
    attention_decode: list[dict] = field(default_factory=list)
    rmsnorm: list[dict] = field(default_factory=list)
    ew: list[dict] = field(default_factory=list)
    moe: list[dict] = field(default_factory=list)
    quant: list[dict] = field(default_factory=list)
    plans: list[dict] = field(default_factory=list)
    crossings: dict = field(default_factory=dict)


def build_sweeps(
    machine: Machine,
    thresholds: Thresholds | None = None,
    launch_s: float = 5e-6,
) -> SweepBundle:
    band = thresholds or Thresholds()
    bundle = SweepBundle(machine=machine, thresholds=band, launch_s=launch_s)
    bundle.crossings = crossing_table(band)
    bundle.gemm = [_gemm_row(label, m, n, k, note, machine, band) for label, m, n, k, note in GEMM_SHAPES]
    bundle.gemm_curve = gemm_m_curve(machine, band)
    bundle.attention = attention_prefill_rows(machine, band)
    bundle.attention_decode = attention_decode_rows(machine, band)
    bundle.rmsnorm = rmsnorm_rows(machine, band, launch_s)
    bundle.ew = ew_rows(machine, band)
    bundle.moe = moe_rows(machine, band)
    bundle.quant = quant_rows(machine, band)
    bundle.plans = plan_rows(bundle)
    return bundle


def crossing_table(thresholds: Thresholds) -> dict:
    """M* and the 0.5 R / 2 R edges for N = K = 4096."""
    from shapefuse.machines import H100_SXM_FP16

    specs = (
        (H100_SXM_FP16, 2.0),
        (A100_SXM80_FP16, 2.0),
        (A40_FP16, 2.0),
        (H100_SXM_FP8, 1.0),
    )
    out: dict = {}
    d = 4096.0
    for mach, elem in specs:
        r = mach.ridge
        out[mach.name] = {
            "N": d,
            "K": d,
            "elem_bytes": elem,
            "ridge": r,
            "M_memory_edge_0.5R": gemm_m_for_target_ai(d, thresholds.mem_frac * r, elem),
            "M_star_AI_equals_ridge": gemm_m_for_target_ai(d, r, elem),
            "M_compute_edge_2R": gemm_m_for_target_ai(d, thresholds.compute_frac * r, elem),
            "formula": "M = t * elem * d / (2 * (d - t * elem)), N=K=d",
            "peak_source": "datasheet",
        }
    return out


def gemm_m_curve(machine: Machine, thresholds: Thresholds, d: int = 4096) -> list[dict]:
    rows = []
    m = 1
    while m <= 8192:
        ai = gemm_ai(m, d, d, 2, 2, 2)
        rows.append(
            {
                "M": m,
                "N": d,
                "K": d,
                "AI_fp16": ai,
                "ai_over_ridge": ai / machine.ridge,
                "regime": classify(ai, machine.ridge, thresholds),
                "roofline_limit": limiting_resource(ai, machine.ridge),
                "cpu_median_s": None,
                "cpu_GFLOPS": None,
                "cpu_GBps_proxy": None,
            }
        )
        m *= 2
    return rows


def _gemm_row(
    label: str,
    m: int,
    n: int,
    k: int,
    note: str,
    machine: Machine,
    thresholds: Thresholds,
) -> dict:
    ai16 = gemm_ai(m, n, k, 2, 2, 2)
    ai32 = gemm_ai(m, n, k, 4, 4, 4)
    site = Site("gemm", label, {"M": m, "N": n, "K": k})
    report = classify_site(site, machine, thresholds)
    return {
        "op": "GEMM",
        "label": label,
        "M": m,
        "N": n,
        "K": k,
        "note": note,
        "FLOPs": gemm_flops(m, n, k),
        "bytes_fp16": gemm_bytes(m, n, k, 2, 2, 2),
        "AI_fp16": ai16,
        "AI_fp32": ai32,
        "ai_over_ridge": ai16 / machine.ridge,
        "regime": report.regime,
        "roofline_limit": report.roofline_limit,
        "vs_H100": classify(ai16, H100_ridge(), thresholds),
        "vs_A100": classify(ai16, A100_SXM80_FP16.ridge, thresholds),
        "vs_A40": classify(ai16, A40_FP16.ridge, thresholds),
    }


def attention_prefill_rows(machine: Machine, thresholds: Thresholds) -> list[dict]:
    rows = []
    b, h, d = 1, 32, 128
    e = 2.0
    for s in (128, 256, 512, 1024, 2048, 4096, 8192):
        qk = attn_qkt(b, h, s, d, e)
        sm = softmax_scores(b, h, s, e)
        fa = attn_fa_prefill(b, h, s, d, e)
        unf = attn_unfused_prefill(b, h, s, d, e)
        speed = roofline_fusion_speedup(fa.flops, unf.nbytes, fa.nbytes, machine, launch_s=0.0)
        rows.append(
            {
                "op": "AttnPrefill",
                "S": s,
                "B": b,
                "H": h,
                "D": d,
                "AI_QKT": qk.ai,
                "AI_Softmax": sm.ai,
                "AI_FA_style_IO": fa.ai,
                "AI_unfused_scores": unf.ai,
                "QKT_regime": classify(qk.ai, machine.ridge, thresholds),
                "Softmax_regime": classify(sm.ai, machine.ridge, thresholds),
                "FA_regime": classify(fa.ai, machine.ridge, thresholds),
                "unfused_regime": classify(unf.ai, machine.ridge, thresholds),
                "bytes_fa": fa.nbytes,
                "bytes_unfused": unf.nbytes,
                "roofline_speedup_fa_vs_unfused": speed["speedup"],
                "roofline_limit_fa": limiting_resource(fa.ai, machine.ridge),
                "roofline_limit_unfused": limiting_resource(unf.ai, machine.ridge),
                "model_note": "roofline datasheet model, not a kernel measurement",
            }
        )
    return rows


def attention_decode_rows(machine: Machine, thresholds: Thresholds) -> list[dict]:
    rows = []
    b, h, d = 1, 32, 128
    for name, hkv in (("mha", h), ("gqa4", h // 4), ("gqa8", h // 8)):
        for sk in (128, 512, 2048, 8192, 32768):
            dec = attn_fa_decode(b, h, hkv, sk, d, 2.0, 1)
            rows.append(
                {
                    "op": "AttnDecode",
                    "variant": name,
                    "B": b,
                    "H": h,
                    "Hkv": hkv,
                    "Sq": 1,
                    "Sk": sk,
                    "D": d,
                    "AI_fp16": dec.ai,
                    "regime": classify(dec.ai, machine.ridge, thresholds),
                    "roofline_limit": limiting_resource(dec.ai, machine.ridge),
                    "asymptotic_note": "as Sk grows, AI -> 2*H/(Hkv*elem)",
                }
            )
    return rows


def rmsnorm_rows(machine: Machine, thresholds: Thresholds, launch_s: float) -> list[dict]:
    rows = []
    shapes = (
        (1, 4096),
        (8, 4096),
        (32, 4096),
        (128, 4096),
        (256, 4096),
        (512, 4096),
        (1024, 4096),
        (2048, 4096),
        (4096, 4096),
        (128, 12288),
    )
    for m, k in shapes:
        # Projection N = K, matching the project note's square-proj epilogue.
        n = k
        norm = rmsnorm(m, k, 2.0)
        traffic = gemm_plus_rmsnorm_traffic(m, n, k, 2.0)
        speed = roofline_fusion_speedup(
            traffic["flops"],
            traffic["bytes_unfused"],
            traffic["bytes_fused"],
            machine,
            launch_s=launch_s,
        )
        g_ai = gemm_ai(m, n, k, 2, 2, 2)
        rows.append(
            {
                "op": "RMSNorm",
                "M": m,
                "N": n,
                "K": k,
                "AI_rmsnorm_fp16": norm.ai,
                "rmsnorm_regime": classify(norm.ai, machine.ridge, thresholds),
                "AI_gemm_fp16": g_ai,
                "gemm_regime": classify(g_ai, machine.ridge, thresholds),
                "bytes_unfused": traffic["bytes_unfused"],
                "bytes_fused": traffic["bytes_fused"],
                "byte_save_fraction": traffic["byte_save_fraction"],
                "roofline_speedup": speed["speedup"],
                "roofline_speedup_with_launch": speed["speedup_with_launch"],
                "launch_overhead_us": launch_s * 1e6,
                "launch_status": "ASSUMED_NOT_MEASURED",
                "note": (
                    "byte_save_fraction grows with M on this model; "
                    "strict roofline speedup collapses to 1 once both sides are compute-bound"
                ),
            }
        )
    return rows


def ew_rows(machine: Machine, thresholds: Thresholds) -> list[dict]:
    rows = []
    # n cancels in AI; a fixed n documents the byte formula.
    n = 4096 * 4096
    for name, fpe, reads, writes, note in EW_OPS:
        ew = elementwise(n, fpe, reads, writes, 2.0)
        rows.append(
            {
                "op": "EW",
                "label": name,
                "note": note,
                "flops_per_elem": fpe,
                "reads": reads,
                "writes": writes,
                "n_example": n,
                "AI_fp16": ew.ai,
                "regime": classify(ew.ai, machine.ridge, thresholds),
            }
        )
    return rows


def moe_rows(machine: Machine, thresholds: Thresholds) -> list[dict]:
    rows = []
    hidden, inter = 4096, 14336
    for name, counts, empty in MOE_SCENARIOS:
        total, per = moe_grouped(counts, hidden, inter, 2, 2, 2, load_empty_experts=empty)
        occupied = [t for t in counts if t > 0]
        regimes = [classify(ai, machine.ridge, thresholds) for ai in per]
        # Empty-weight loads do not create extra expert regimes.
        rows.append(
            {
                "op": "MoE_GroupedGEMM",
                "scenario": name,
                "H": hidden,
                "I": inter,
                "token_counts": occupied,
                "load_empty_experts": empty,
                "AI_aggregate_fp16": total.ai,
                "AI_min_expert": min(per),
                "AI_max_expert": max(per),
                "regime_aggregate": classify(total.ai, machine.ridge, thresholds),
                "regime_min_expert": classify(min(per), machine.ridge, thresholds),
                "regime_max_expert": classify(max(per), machine.ridge, thresholds),
                "expert_regimes": regimes,
                "heterogeneous": moe_is_heterogeneous(regimes) if empty == 0 else False,
                "vs_H100_agg": classify(total.ai, H100_ridge(), thresholds),
                "note": (
                    "inactive experts skipped"
                    if empty == 0
                    else "extra cold weight matrices with zero FLOPs"
                ),
            }
        )
    return rows


def quant_rows(machine: Machine, thresholds: Thresholds) -> list[dict]:
    """Byte-width what-ifs. The FP16 machine ridge is not an FP8/INT8 peak."""
    variants = (
        ("fp16", "fp16", "fp16", "fp16", True),
        ("w8a16_out_fp16", "fp16", "int8", "fp16", False),
        ("w8a8_out_fp16", "int8", "int8", "fp16", False),
        ("w8a8_out_fp8", "int8", "int8", "fp8", False),
    )
    rows = []
    fp8_ridge = H100_SXM_FP8.ridge
    for m in (1, 32, 128, 256, 2048):
        for name, da, db, dc, matched in variants:
            ea, eb, ec = elem_bytes(da), elem_bytes(db), elem_bytes(dc)
            ai = gemm_ai(m, 4096, 4096, ea, eb, ec)
            rows.append(
                {
                    "op": "QuantGEMM",
                    "label": f"{name}_M{m}",
                    "variant": name,
                    "M": m,
                    "N": 4096,
                    "K": 4096,
                    "dtype_a": da,
                    "dtype_b": db,
                    "dtype_c": dc,
                    "AI": ai,
                    "regime_vs_requested_machine": classify(ai, machine.ridge, thresholds),
                    "regime_vs_H100_FP8_datasheet": classify(ai, fp8_ridge, thresholds),
                    "peak_dtype_matches_operands": matched and machine.peak_dtype in {da, db, dc},
                    "note": (
                        "consistent FP16 datasheet pair"
                        if matched
                        else "byte-only what-if; W8A8/FP8 peak is not this ridge unless the machine says so"
                    ),
                }
            )
    return rows


def plan_rows(bundle: SweepBundle) -> list[dict]:
    machine = bundle.machine
    band = bundle.thresholds
    rows: list[dict] = []

    for g in bundle.gemm:
        site = Site("gemm", g["label"], {"M": g["M"], "N": g["N"], "K": g["K"]})
        rows.append(_plan_dict(classify_site(site, machine, band), length=g["M"]))

    for a in bundle.attention:
        site = Site(
            "attention_prefill",
            f"prefill_S{a['S']}",
            {"B": a["B"], "H": a["H"], "S": a["S"], "D": a["D"]},
        )
        rows.append(_plan_dict(classify_site(site, machine, band), length=a["S"]))

    for a in bundle.attention_decode:
        if a["Sk"] not in (128, 8192) or a["variant"] not in ("mha", "gqa4"):
            continue
        site = Site(
            "attention_decode",
            f"decode_{a['variant']}_Sk{a['Sk']}",
            {
                "B": a["B"],
                "H": a["H"],
                "Hkv": a["Hkv"],
                "Sk": a["Sk"],
                "Sq": 1,
                "D": a["D"],
            },
        )
        rows.append(_plan_dict(classify_site(site, machine, band), length=a["Sk"]))

    for r in bundle.rmsnorm:
        if r["K"] != 4096 or r["M"] not in (1, 256, 2048):
            continue
        site = Site("rmsnorm", f"rmsnorm_M{r['M']}", {"M": r["M"], "K": r["K"]})
        rows.append(_plan_dict(classify_site(site, machine, band), length=r["M"]))

    for ew in bundle.ew:
        site = Site(
            "ew_chain",
            ew["label"],
            {
                "n": ew["n_example"],
                "flops_per_elem": ew["flops_per_elem"],
                "reads": ew["reads"],
                "writes": ew["writes"],
            },
        )
        rows.append(_plan_dict(classify_site(site, machine, band), length=ew["n_example"]))

    site = Site("softmax", "softmax_S2048", {"B": 1, "H": 32, "S": 2048})
    rows.append(_plan_dict(classify_site(site, machine, band), length=2048))
    site = Site("reduction", "rowsum_M2048", {"M": 2048, "K": 4096})
    rows.append(_plan_dict(classify_site(site, machine, band), length=2048))

    hidden, inter = 4096, 14336
    skewed = [1024, 512, 256, 128, 64, 32, 16, 8]
    for t in skewed:
        site = Site(
            "moe_expert",
            f"skewed_tokens_{t}",
            {"tokens": t, "H": hidden, "I": inter},
        )
        rows.append(_plan_dict(classify_site(site, machine, band), length=t))
    site = Site(
        "moe_grouped",
        "skewed_8e_aggregate",
        {"token_counts": skewed, "H": hidden, "I": inter},
    )
    rows.append(_plan_dict(classify_site(site, machine, band), length=max(skewed)))

    for m, da, db, dc, label in (
        (1, "int8", "int8", "fp16", "w8a8_out_fp16_M1"),
        (2048, "int8", "int8", "fp16", "w8a8_out_fp16_M2048"),
    ):
        site = Site(
            "gemm",
            label,
            {"M": m, "N": 4096, "K": 4096},
            dtype="fp16",
            dtype_a=da,
            dtype_b=db,
            dtype_c=dc,
        )
        # Kind name on the plan row: the classifier site is a GEMM with mixed
        # dtypes. Relabel the exported kind so the table shows the scope.
        decision = select_plan(classify_site(site, machine, band), length=m)
        row = _decision_to_row(decision)
        row["kind"] = "quantized_epilogue"
        row["peak_dtype_matches_operands"] = False
        rows.append(row)
    return rows


def _plan_dict(report, length: int) -> dict:
    return _decision_to_row(select_plan(report, length=length))


def _decision_to_row(decision) -> dict:
    alt = decision.alternate.mode if decision.alternate else ""
    return {
        "kind": decision.kind,
        "label": decision.label,
        "regime": decision.regime,
        "AI": decision.ai,
        "ai_over_ridge": decision.ai / decision.ridge,
        "primary_mode": decision.primary.mode,
        "alternate_mode": alt,
        "emit_both": decision.emit_both,
        "fuse_epilogue": decision.primary.fuse_epilogue,
        "epilogue_policy": decision.epilogue_policy,
        "materialize_intermediate": decision.primary.materialize_intermediate,
        "tile": decision.primary.tile,
        "protect_mma_pipeline": decision.primary.protect_mma_pipeline,
        "cache_key": decision.cache_key,
        "measured_winner": decision.measured_winner if decision.measured_winner else "",
        "kernel_emitted": decision.kernel_emitted,
        "shape_bucket": decision.detail.get("shape_bucket", ""),
        "plan_note": decision.primary.note,
        "rationale": decision.rationale,
    }


def H100_ridge() -> float:
    from shapefuse.machines import H100_SXM_FP16

    return H100_SXM_FP16.ridge

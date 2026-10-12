"""Golden checks for the closed forms used in the paper draft."""

from __future__ import annotations

from dataclasses import replace

from shapefuse.intensity import (
    attn_fa_prefill,
    attn_qkt,
    gemm_ai,
    gemm_m_for_target_ai,
    gemm_plus_rmsnorm_traffic,
    moe_grouped,
    softmax_scores,
)
from shapefuse.machines import A100_SXM80_FP16, A40_FP16, H100_SXM_FP16, H100_SXM_FP8, get_machine
from shapefuse.plan_paths import (
    BASELINE_ALWAYS_FUSE,
    BASELINE_ALWAYS_UNFUSE,
    analytical_executable_path,
    attach_measured_winner,
    gemm_calibration_applies,
    pick_wallclock_winner,
    select_executable_path,
)
from shapefuse.plans import POLICY_COMPILE_SDPA_CHAIN, select_plan
from shapefuse.scic import Site, classify_site


def assert_invariants() -> None:
    r_h = H100_SXM_FP16.ridge
    r_a = A100_SXM80_FP16.ridge
    r_a40 = A40_FP16.ridge
    _close(r_h, 989e12 / 3.35e12, 1e-12)
    _close(r_a, 312e12 / 2.039e12, 1e-12)
    _close(r_a40, 149.7e12 / 696e9, 1e-12)
    _close(gemm_m_for_target_ai(4096, r_h, 2), 344.94907864709285, 1e-9)
    _close(gemm_m_for_target_ai(4096, r_a, 2), 165.37193778675896, 1e-9)
    _close(gemm_m_for_target_ai(4096, r_a40, 2), 240.32584259093773, 1e-9)
    if get_machine("a40").name != "A40_FP16":
        raise AssertionError("alias a40 should resolve to the A40 FP16 datasheet")
    if get_machine("h100").name != "H100_SXM_FP16" or get_machine("a100").name != "A100_SXM80_FP16":
        raise AssertionError("h100 and a100 aliases should resolve to datasheet anchors")
    # FP8 doubles AI and doubles the ridge, so M* stays put.
    _close(
        gemm_m_for_target_ai(4096, H100_SXM_FP8.ridge, 1),
        gemm_m_for_target_ai(4096, r_h, 2),
        0.5,
    )

    _close(gemm_ai(1, 4096, 4096, 2, 2, 2), 0.9995119570522206, 1e-12)
    _close(gemm_ai(2048, 4096, 4096, 2, 2, 2), 1024.0, 1e-9)
    _close(attn_qkt(1, 32, 128, 128, 2).ai, 42.666666666666664, 1e-9)
    _close(softmax_scores(1, 32, 2048, 2).ai, 5.0 / 6.0, 1e-12)
    _close(attn_fa_prefill(1, 32, 2048, 128, 2).ai, 1024.0, 1e-9)
    _close(attn_fa_prefill(1, 32, 8192, 128, 2).ai, 4096.0, 1e-9)

    tiny, _ = moe_grouped([2] * 8, 4096, 14336, 2, 2, 2)
    _close(tiny.ai, 1.9987452073893344, 1e-9)
    skew, per = moe_grouped([1024, 512, 256, 128, 64, 32, 16, 8], 4096, 14336)
    _close(min(per), 7.979961035346507, 1e-9)
    _close(skew.ai, 236.10165660218942, 1e-9)

    traffic = gemm_plus_rmsnorm_traffic(1, 4096, 4096, 2)
    if not traffic["byte_save_fraction"] < 0.001:
        raise AssertionError("small-M epilogue byte fraction should be tiny")
    traffic_big = gemm_plus_rmsnorm_traffic(2048, 4096, 4096, 2)
    if not traffic_big["byte_save_fraction"] > traffic["byte_save_fraction"]:
        raise AssertionError("byte-save fraction should grow with M")

    mem = classify_site(Site("gemm", "m1", {"M": 1, "N": 4096, "K": 4096}), H100_SXM_FP16)
    comp = classify_site(Site("gemm", "m2048", {"M": 2048, "N": 4096, "K": 4096}), H100_SXM_FP16)
    ridge = classify_site(Site("gemm", "m256", {"M": 256, "N": 4096, "K": 4096}), H100_SXM_FP16)
    if mem.regime != "memory-bound" or select_plan(mem, length=1).primary.mode != "MemMode":
        raise AssertionError("M=1 should be MemMode")
    if comp.regime != "compute-bound" or select_plan(comp, length=2048).primary.mode != "ComputeMode":
        raise AssertionError("M=2048 should be ComputeMode")
    mid = select_plan(ridge, length=256)
    if ridge.regime != "ridge-region" or not mid.emit_both:
        raise AssertionError("M=256 should emit both plans")

    short = classify_site(
        Site("attention_prefill", "s128", {"B": 1, "H": 32, "S": 128, "D": 128}),
        H100_SXM_FP16,
    )
    long = classify_site(
        Site("attention_prefill", "s8192", {"B": 1, "H": 32, "S": 8192, "D": 128}),
        H100_SXM_FP16,
    )
    d_short = select_plan(short, length=128)
    d_long = select_plan(long, length=8192)
    if d_short.primary.mode != "MemMode" or d_short.primary.materialize_intermediate:
        raise AssertionError("short attention should be MemMode and keep scores off HBM")
    if d_long.primary.mode != "ComputeMode" or d_long.primary.materialize_intermediate:
        raise AssertionError("long attention should be ComputeMode and keep scores off HBM")
    if d_long.kernel_emitted:
        raise AssertionError("selector must not claim a kernel was emitted")
    if select_executable_path(select_plan(mem, length=1)) != BASELINE_ALWAYS_FUSE:
        raise AssertionError("memory-bound GEMM must select always_fuse")
    comp_plan = select_plan(comp, length=2048)
    if select_executable_path(comp_plan) != BASELINE_ALWAYS_UNFUSE:
        raise AssertionError("compute-bound GEMM must select always_unfuse without a probe")
    if analytical_executable_path(comp_plan) != BASELINE_ALWAYS_UNFUSE:
        raise AssertionError("analytical compute GEMM stays always_unfuse")
    if not gemm_calibration_applies(comp_plan, "ridge-compute"):
        raise AssertionError("compute GEMM is in the ridge-compute calibration scope")
    if gemm_calibration_applies(select_plan(mem, length=1), "ridge-compute"):
        raise AssertionError("memory-bound GEMM stays analytical unless scope is all-gemm")
    if not gemm_calibration_applies(select_plan(mem, length=1), "all-gemm"):
        raise AssertionError("all-gemm probes every GEMM shape")
    if gemm_calibration_applies(d_long, "all-gemm"):
        raise AssertionError("attention is not wall-clock calibrated")
    overridden = attach_measured_winner(comp_plan, BASELINE_ALWAYS_FUSE)
    if overridden.measured_winner != BASELINE_ALWAYS_FUSE:
        raise AssertionError("probe must fill measured_winner")
    if select_executable_path(overridden) != BASELINE_ALWAYS_FUSE:
        raise AssertionError("measured always_fuse must override analytical always_unfuse")
    if analytical_executable_path(overridden) != BASELINE_ALWAYS_UNFUSE:
        raise AssertionError("analytical baseline ignores measured_winner")
    if pick_wallclock_winner(0.7, 0.9) != BASELINE_ALWAYS_FUSE:
        raise AssertionError("faster fuse median must win")
    if pick_wallclock_winner(0.9, 0.7) != BASELINE_ALWAYS_UNFUSE:
        raise AssertionError("faster unfuse median must win")
    if pick_wallclock_winner(1.0, 1.0) is not None or pick_wallclock_winner(None, 1.0) is not None:
        raise AssertionError("a tie or a missing median keeps the analytical baseline")
    if select_executable_path(attach_measured_winner(d_long, BASELINE_ALWAYS_UNFUSE)) != BASELINE_ALWAYS_FUSE:
        raise AssertionError("attention ignores a measured unfuse winner")
    stuffed = replace(d_long, measured_winner=BASELINE_ALWAYS_UNFUSE)
    if select_executable_path(stuffed) != BASELINE_ALWAYS_FUSE:
        raise AssertionError("library SDPA stays always_fuse even if measured_winner says unfuse")
    if select_executable_path(d_short) != BASELINE_ALWAYS_FUSE:
        raise AssertionError("short attention must select always_fuse (epilogue on SDPA)")
    if d_long.epilogue_policy != POLICY_COMPILE_SDPA_CHAIN:
        raise AssertionError("long attention must score compile_sdpa_chain")
    if select_executable_path(d_long) != BASELINE_ALWAYS_FUSE:
        raise AssertionError(
            "library SDPA must select always_fuse (compile_sdpa_chain) "
            "even when the analytical primary is ComputeMode"
        )
    if d_long.primary.fuse_epilogue:
        raise AssertionError("analytical ComputeMode still leaves fuse_epilogue false")


def run_selfcheck() -> None:
    assert_invariants()
    print("selfcheck: ok")


def _close(got: float, want: float, tol: float) -> None:
    if abs(got - want) > tol:
        raise AssertionError(f"{got} != {want} (tol {tol})")

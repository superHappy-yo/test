"""Map an SCIC decision onto a path the Docker bakeoff can actually run.

MemMode and ComputeMode are not CUDA kernels in this repository. The
bakeoff therefore picks between two existing PyTorch paths:

- ``always_fuse``: ``torch.compile`` of the subgraph, an attempt to fuse the
  epilogue. The graph is not inspected, so a speedup is not proof of fusion.
- ``always_unfuse``: the eager subgraph, so the epilogue stays a separate
  launch. On attention this is library SDPA followed by a separate RMSNorm,
  not a return to writing scores.

Attention never selects a path that materializes scores. The SDPA call is
``torch.nn.functional.scaled_dot_product_attention``. It is not a
FlashAttention-3 kernel written for this paper. Stage-Stream is not selected.

Library SDPA already fuses scores. The attention executable path is
therefore ``compile_sdpa_chain``, timed as ``always_fuse``, even when the
analytical primary is ComputeMode. The wall-clock for that choice is
``paper/data/bakeoff_v2/`` (not an idle GPU). Attention does not consult
``measured_winner``.

GEMM's analytical mapping is unchanged: MemMode is ``always_fuse`` and
ComputeMode is ``always_unfuse``. On the second bakeoff, ridge and compute
GEMMs copied ``always_unfuse`` near 1× versus eager while ``always_fuse``
(``torch.compile``) was faster. The bakeoff can fill ``measured_winner``
from a few-shot wall-clock of those two paths. ``select_executable_path``
follows that winner when it is ``always_fuse`` or ``always_unfuse``, and
otherwise keeps the analytical baseline. The probe is median latency on
the machine that runs it. It is not DRAM arithmetic intensity. A busy GPU
can flip the winner. It is not SCIC accuracy.
"""

from __future__ import annotations

import math

from shapefuse.plans import (
    ALWAYS_MEM_KINDS,
    ATTENTION_KINDS,
    POLICY_COMPILE_SDPA_CHAIN,
    PlanDecision,
)

BASELINE_EAGER = "eager_unfused"
BASELINE_COMPILE_DEFAULT = "torch_compile_default"
BASELINE_COMPILE_MAX_AUTOTUNE = "torch_compile_max_autotune"
BASELINE_SDPA = "sdpa"
BASELINE_ALWAYS_FUSE = "always_fuse"
BASELINE_ALWAYS_UNFUSE = "always_unfuse"
BASELINE_SCIC = "scic_selected"

GEMM_BASELINES: tuple[str, ...] = (
    BASELINE_EAGER,
    BASELINE_COMPILE_DEFAULT,
    BASELINE_COMPILE_MAX_AUTOTUNE,
    BASELINE_ALWAYS_FUSE,
    BASELINE_ALWAYS_UNFUSE,
    BASELINE_SCIC,
)

ATTENTION_BASELINES: tuple[str, ...] = (
    BASELINE_EAGER,
    BASELINE_COMPILE_DEFAULT,
    BASELINE_COMPILE_MAX_AUTOTUNE,
    BASELINE_SDPA,
    BASELINE_ALWAYS_FUSE,
    BASELINE_ALWAYS_UNFUSE,
    BASELINE_SCIC,
)

# Shared honesty. Every bakeoff row carries these so a later reader cannot
# relabel the CSV as Nsight accuracy, FA-3, or Stage-Stream.
HONESTY_NO_NSIGHT = "NO_NSIGHT"
HONESTY_NOT_SCIC_ACCURACY = "NOT_SCIC_ACCURACY"
HONESTY_NOT_FA3 = "NOT_AN_FA3_KERNEL"
HONESTY_NOT_STAGE_STREAM = "NOT_STAGE_STREAM"
HONESTY_NOT_CUSTOM_KERNEL = "NOT_A_CUSTOM_MEM_OR_COMPUTE_KERNEL"
HONESTY_GRAPH_NOT_INSPECTED = "GRAPH_NOT_INSPECTED"
HONESTY_PLACEHOLDER = "PLACEHOLDER_NO_GPU"
HONESTY_DATASHEET_RIDGE = "REGIME_USES_DATASHEET_RIDGE"

HONESTY_EAGER_GEMM = "EAGER_UNFUSED_MATMUL_PLUS_RMSNORM"
HONESTY_EAGER_ATTN = "EAGER_MATERIALIZED_SCORES_PLUS_RMSNORM"
HONESTY_COMPILE_DEFAULT = "TORCH_COMPILE_DEFAULT_GRAPH_NOT_INSPECTED"
HONESTY_COMPILE_FAILED = "TORCH_COMPILE_DEFAULT_FAILED"
HONESTY_COMPILE_SKIPPED_CPU = "TORCH_COMPILE_NOT_RUN_WITHOUT_CUDA"
HONESTY_MAX_AUTOTUNE = "TORCH_COMPILE_MAX_AUTOTUNE_GRAPH_NOT_INSPECTED"
HONESTY_MAX_AUTOTUNE_FAILED = "TORCH_COMPILE_MAX_AUTOTUNE_FAILED"
HONESTY_MAX_AUTOTUNE_SKIPPED = "TORCH_COMPILE_MAX_AUTOTUNE_SKIPPED"
HONESTY_MAX_AUTOTUNE_NOT_FEASIBLE = "TORCH_COMPILE_MAX_AUTOTUNE_NOT_FEASIBLE_WITHOUT_CUDA"
HONESTY_ALWAYS_FUSE_GEMM = "ALWAYS_FUSE_IS_TORCH_COMPILE_DEFAULT_NOT_A_MEMMODE_KERNEL"
HONESTY_ALWAYS_UNFUSE_GEMM = "ALWAYS_UNFUSE_IS_EAGER_CHAIN_NOT_A_COMPUTEMODE_KERNEL"
HONESTY_ALWAYS_FUSE_ATTN = "ALWAYS_FUSE_COMPILES_SDPA_PLUS_RMSNORM_NOT_FA3"
HONESTY_ALWAYS_UNFUSE_ATTN = "ALWAYS_UNFUSE_SDPA_THEN_SEPARATE_RMSNORM"
HONESTY_SDPA = "LIBRARY_SDPA_PLUS_RMSNORM_NOT_AN_FA3_KERNEL"
HONESTY_SDPA_UNAVAILABLE = "SDPA_UNAVAILABLE"
HONESTY_ATTN_OOM = "ATTENTION_OOM"
HONESTY_SPEEDUP_VS_MATERIALIZED = "SPEEDUP_VS_EAGER_COMPARES_TO_MATERIALIZED_ATTENTION"
HONESTY_SCIC_COPIED = "SCIC_SELECTED_EXISTING_PATH_TIMING_COPIED"
HONESTY_SCIC_MISSING = "SCIC_SELECTED_PATH_DID_NOT_RUN"
HONESTY_RIDGE_PRIMARY_ONLY = "RIDGE_BAND_EMIT_BOTH_TIMED_PRIMARY_ONLY"
HONESTY_ATTN_NO_TILE_KERNEL = "TILE_SWITCH_NOT_A_SEPARATE_KERNEL"
# Library SDPA epilogue. Not a claim that the graph fused.
# paper/data/bakeoff_v2 is the wall-clock. That GPU was not idle.
HONESTY_LIBRARY_SDPA_CHAIN = "LIBRARY_SDPA_EPILOGUE_COMPILE_SDPA_CHAIN"
HONESTY_LIBRARY_SDPA_NOT_PRIMARY = "LIBRARY_SDPA_CHAIN_NOT_ANALYTICAL_PRIMARY"
# Few-shot median on this process. Not FLOPs / DRAM bytes.
HONESTY_WALLCLOCK_NOT_DRAM = "WALLCLOCK_CALIBRATION_NOT_DRAM_AI"
HONESTY_GPU_CONTENTION = "GPU_CONTENTION_CAN_FLIP_WINNER"
HONESTY_MEASURED_OVERRIDE = "MEASURED_WINNER_OVERRIDES_ANALYTICAL"
HONESTY_CALIBRATION_AGREES = "MEASURED_WINNER_AGREES_WITH_ANALYTICAL"
HONESTY_ANALYTICAL_FALLBACK = "ANALYTICAL_FALLBACK_NO_MEASURED_WINNER"
HONESTY_ATTN_NOT_CALIBRATED = "LIBRARY_SDPA_NOT_WALLCLOCK_CALIBRATED"

# Bakeoff scopes. ``ridge-compute`` is the GEMM miss on bakeoff_v2.
# ``all-gemm`` also probes memory-bound MatMul+RMSNorm. ``off`` is datasheet only.
CALIBRATION_SCOPES: tuple[str, ...] = ("ridge-compute", "all-gemm", "off")
CALIBRATION_SCOPE_RIDGE_COMPUTE = "ridge-compute"
CALIBRATION_SCOPE_ALL_GEMM = "all-gemm"
CALIBRATION_SCOPE_OFF = "off"

SELECTION_MEASURED = "measured_wallclock"
SELECTION_ANALYTICAL_FALLBACK = "analytical_fallback"
SELECTION_ANALYTICAL = "analytical"
SELECTION_LIBRARY_SDPA = "library_sdpa"

_EXECUTABLE_WINNERS = frozenset({BASELINE_ALWAYS_FUSE, BASELINE_ALWAYS_UNFUSE})


def select_executable_path(decision: PlanDecision) -> str:
    """Return ``always_fuse`` or ``always_unfuse`` for this decision.

    Attention that calls library SDPA does not follow ComputeMode into
    ``always_unfuse``, and it does not follow ``measured_winner``. SDPA
    already keeps scores off HBM. The unfuse path only splits RMSNorm off
    that kernel. The pre-fix archive (``paper/data/bakeoff``) was slower
    there at prefill S=512 and S=2048. The executable policy is
    :data:`POLICY_COMPILE_SDPA_CHAIN`, timed as ``always_fuse``, in every
    attention regime. ``paper/data/bakeoff_v2`` measured that choice. The
    analytical primary mode is unchanged.

    GEMM uses ``measured_winner`` when it is ``always_fuse`` or
    ``always_unfuse``. Otherwise it follows ``fuse_epilogue``: MemMode is
    ``always_fuse``, ComputeMode is ``always_unfuse``. The ridge band emits
    both analytical plans. The bakeoff times one path and records
    ``emit_both`` in the decision log. It does not emit a second kernel.
    A wall-clock winner is not DRAM AI. ``paper/data/bakeoff_v2`` predates
    the probe and still copies ``always_unfuse`` on ridge and compute GEMM.
    """
    if decision.kind in ATTENTION_KINDS and decision.primary.materialize_intermediate:
        raise RuntimeError(
            "attention plan set materialize_intermediate; the selector must keep scores off HBM"
        )
    if decision.kind in ATTENTION_KINDS:
        if decision.epilogue_policy not in {"", POLICY_COMPILE_SDPA_CHAIN}:
            raise RuntimeError(
                f"unknown attention epilogue policy {decision.epilogue_policy!r}"
            )
        return BASELINE_ALWAYS_FUSE
    winner = decision.measured_winner or ""
    if winner in _EXECUTABLE_WINNERS:
        return winner
    return analytical_executable_path(decision)


def analytical_executable_path(decision: PlanDecision) -> str:
    """Datasheet or library mapping, ignoring ``measured_winner``.

    Attention is ``always_fuse`` (``compile_sdpa_chain``). GEMM follows
    ``fuse_epilogue``.
    """
    if decision.kind in ATTENTION_KINDS and decision.primary.materialize_intermediate:
        raise RuntimeError(
            "attention plan set materialize_intermediate; the selector must keep scores off HBM"
        )
    if decision.kind in ATTENTION_KINDS:
        if decision.epilogue_policy not in {"", POLICY_COMPILE_SDPA_CHAIN}:
            raise RuntimeError(
                f"unknown attention epilogue policy {decision.epilogue_policy!r}"
            )
        return BASELINE_ALWAYS_FUSE
    if decision.primary.fuse_epilogue:
        return BASELINE_ALWAYS_FUSE
    return BASELINE_ALWAYS_UNFUSE


def gemm_calibration_applies(decision: PlanDecision, scope: str) -> bool:
    """Whether this decision gets a few-shot fuse-versus-unfuse probe.

    Attention never does. Intrinsic memory ops never do. ``ridge-compute``
    probes GEMM in the ridge band and on the compute side, which is where
    datasheet AI selected ``always_unfuse`` on the second A40 bakeoff.
    ``all-gemm`` also probes memory-bound MatMul+RMSNorm. ``off`` probes
    nothing.
    """
    if scope not in CALIBRATION_SCOPES:
        known = ", ".join(CALIBRATION_SCOPES)
        raise ValueError(f"unknown calibration scope {scope!r}. Known: {known}")
    if scope == CALIBRATION_SCOPE_OFF:
        return False
    if decision.kind in ATTENTION_KINDS or decision.kind in ALWAYS_MEM_KINDS:
        return False
    if decision.kind != "gemm":
        return False
    if scope == CALIBRATION_SCOPE_ALL_GEMM:
        return True
    return decision.regime in {"ridge-region", "compute-bound"}


def pick_wallclock_winner(fuse_ms: float | None, unfuse_ms: float | None) -> str | None:
    """Return the strictly faster of the two medians.

    A missing time, a non-finite time, a non-positive time, or a tie
    returns ``None``. The caller then keeps the analytical baseline.
    Smaller wall-clock wins. This is not a comparison of DRAM intensity.
    """
    if fuse_ms is None or unfuse_ms is None:
        return None
    if not math.isfinite(fuse_ms) or not math.isfinite(unfuse_ms):
        return None
    if fuse_ms <= 0 or unfuse_ms <= 0:
        return None
    if fuse_ms < unfuse_ms:
        return BASELINE_ALWAYS_FUSE
    if unfuse_ms < fuse_ms:
        return BASELINE_ALWAYS_UNFUSE
    return None


def attach_measured_winner(decision: PlanDecision, winner: str | None) -> PlanDecision:
    """Copy ``decision`` with ``measured_winner`` set to an executable path.

    Attention is returned unchanged. An empty winner clears nothing that
    was already empty. ``MemMode`` and ``ComputeMode`` are not executable
    paths and are rejected.
    """
    if decision.kind in ATTENTION_KINDS:
        return decision
    if not winner:
        return decision
    if winner not in _EXECUTABLE_WINNERS:
        raise ValueError(
            "measured_winner must be always_fuse or always_unfuse, "
            f"got {winner!r}"
        )
    from dataclasses import replace

    return replace(decision, measured_winner=winner)


def common_honesty(*, placeholder: bool, attention: bool) -> list[str]:
    flags = [
        HONESTY_NO_NSIGHT,
        HONESTY_NOT_SCIC_ACCURACY,
        HONESTY_NOT_FA3,
        HONESTY_NOT_STAGE_STREAM,
        HONESTY_NOT_CUSTOM_KERNEL,
        HONESTY_GRAPH_NOT_INSPECTED,
        HONESTY_DATASHEET_RIDGE,
    ]
    if placeholder:
        flags.append(HONESTY_PLACEHOLDER)
    if attention:
        flags.append(HONESTY_SPEEDUP_VS_MATERIALIZED)
        flags.append(HONESTY_ATTN_NO_TILE_KERNEL)
    return flags


def join_honesty(*parts: str) -> str:
    seen: list[str] = []
    for part in parts:
        for flag in part.split(";"):
            flag = flag.strip()
            if flag and flag not in seen:
                seen.append(flag)
    return ";".join(seen)

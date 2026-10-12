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
analytical primary is ComputeMode. GEMM still maps ComputeMode to
``always_unfuse``. The attention change needs a new bakeoff before its
latency is quoted.
"""

from __future__ import annotations

from shapefuse.plans import ATTENTION_KINDS, POLICY_COMPILE_SDPA_CHAIN, PlanDecision

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
# Library SDPA epilogue. Not a claim that the graph fused, and not a
# re-measured latency until the bakeoff is run again.
HONESTY_LIBRARY_SDPA_CHAIN = "LIBRARY_SDPA_EPILOGUE_COMPILE_SDPA_CHAIN"
HONESTY_LIBRARY_SDPA_NOT_PRIMARY = "LIBRARY_SDPA_CHAIN_NOT_ANALYTICAL_PRIMARY"


def select_executable_path(decision: PlanDecision) -> str:
    """Return ``always_fuse`` or ``always_unfuse`` for this decision.

    GEMM follows ``fuse_epilogue``: MemMode is ``always_fuse``, ComputeMode
    is ``always_unfuse``. The ridge band emits both analytical plans. The
    bakeoff times one path and records ``emit_both`` in the decision log.
    It does not emit a second kernel.

    Attention that calls library SDPA does not follow ComputeMode into
    ``always_unfuse``. SDPA already keeps scores off HBM. The unfuse path
    only splits RMSNorm off that kernel, and the archived A40 wall-clock
    was slower there at prefill S=512 and S=2048. The executable policy is
    :data:`POLICY_COMPILE_SDPA_CHAIN`, timed as ``always_fuse``, in every
    attention regime. The analytical primary mode is unchanged. This
    default has not been re-measured.
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

"""Shape-adaptive operator fusion: SCIC and dual-plan selection.

Operator intensity is shape-dependent. The same GEMM, attention, or MoE
site can sit on opposite sides of a GPU's Roofline ridge, so fusion is
emitted as two plans (MemMode and ComputeMode) rather than one static
decision.

This package is an analytical classifier and a plan selector stub. It does
not implement Stage-Stream, and it does not emit CUDA kernels.
"""

from shapefuse.intensity import (
    ELEM_BYTES,
    Intensity,
    classify,
    elem_bytes,
    gemm_ai,
    gemm_bytes,
    gemm_flops,
    gemm_m_for_target_ai,
    limiting_resource,
    roofline_seconds,
)
from shapefuse.machines import (
    A100_SXM80_FP16,
    A40_FP16,
    H100_SXM_FP8,
    H100_SXM_FP16,
    MACHINES,
    Machine,
    get_machine,
)
from shapefuse.plans import Plan, PlanCache, PlanDecision, select_plan
from shapefuse.scic import IntensityReport, classify_site

__all__ = [
    "A100_SXM80_FP16",
    "A40_FP16",
    "ELEM_BYTES",
    "H100_SXM_FP16",
    "H100_SXM_FP8",
    "Intensity",
    "IntensityReport",
    "MACHINES",
    "Machine",
    "Plan",
    "PlanCache",
    "PlanDecision",
    "classify",
    "classify_site",
    "elem_bytes",
    "gemm_ai",
    "gemm_bytes",
    "gemm_flops",
    "gemm_m_for_target_ai",
    "get_machine",
    "limiting_resource",
    "roofline_seconds",
    "select_plan",
]

__version__ = "0.2.0"

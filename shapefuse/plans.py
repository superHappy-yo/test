"""Dual-plan selector stub.

MemMode spends the decision on HBM traffic and launch count (fuse the
epilogue, smaller tiles). ComputeMode protects the MMA pipeline (large
tiles, epilogue fission) and, for GEMM-like producers, materializes the
output. Attention plans never materialize scores: that choice is forced by
the intensity model, and it is not an implementation of Stage-Stream or of
FlashAttention.

No CUDA is emitted. ``kernel_emitted`` is always false. A :class:`PlanCache`
can record a measured winner. Analytical sweeps leave it empty. The bakeoff
records a wall-clock GEMM path when the few-shot probe returns one. That
record is not DRAM arithmetic intensity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shapefuse.scic import IntensityReport

# Ops whose own AI is O(1) FLOP/byte. They do not get a compute-mode fission.
ALWAYS_MEM_KINDS = frozenset({"ew_chain", "reduction", "softmax", "rmsnorm", "layernorm"})

ATTENTION_KINDS = frozenset({"attention_prefill", "attention_decode"})

# Executable epilogue when the attention body is library SDPA. The bakeoff
# times this name as ``always_fuse`` (compile SDPA and RMSNorm together).
# It is not a new kernel, and it is not a measured winner in the plan cache.
POLICY_COMPILE_SDPA_CHAIN = "compile_sdpa_chain"

# Producers whose ComputeMode materializes the GEMM output.
MATERIALIZE_ON_COMPUTE = frozenset({"gemm", "moe_expert", "moe_grouped", "quantized_epilogue"})


@dataclass(frozen=True)
class Plan:
    mode: str
    fuse_epilogue: bool
    materialize_intermediate: bool
    tile: str
    protect_mma_pipeline: bool
    note: str


@dataclass(frozen=True)
class PlanDecision:
    kind: str
    label: str
    regime: str
    ai: float
    ridge: float
    primary: Plan
    alternate: Plan | None
    emit_both: bool
    cache_key: str
    measured_winner: str | None
    kernel_emitted: bool
    rationale: str
    # Empty for GEMM. Attention sets ``compile_sdpa_chain``: library SDPA
    # already fuses scores, so the bakeoff epilogue prefers ``always_fuse``.
    epilogue_policy: str = ""
    detail: dict = field(default_factory=dict)

    @property
    def stub(self) -> bool:
        return not self.kernel_emitted


@dataclass
class PlanCache:
    """In-memory (op, shape-bucket) cache. Not durable, not a runtime."""

    entries: dict[str, dict] = field(default_factory=dict)

    def lookup(self, key: str) -> dict | None:
        return self.entries.get(key)

    def record_measured_winner(self, key: str, mode: str, metric: str, value: float) -> None:
        # MemMode / ComputeMode are the analytical plan names. always_fuse /
        # always_unfuse are bakeoff paths. The wall-clock probe records the
        # path. It does not claim a custom kernel won.
        allowed = {"MemMode", "ComputeMode", "always_fuse", "always_unfuse"}
        if mode not in allowed:
            raise ValueError(f"mode must be one of {sorted(allowed)}, got {mode!r}")
        source = "measured_wallclock" if mode in {"always_fuse", "always_unfuse"} else "measured"
        self.entries[key] = {
            "mode": mode,
            "metric": metric,
            "value": value,
            "source": source,
        }


def shape_bucket(n: int) -> str:
    """Coarse bucket so a plan cache does not key on every integer M."""
    if n <= 0:
        return "0"
    edges = (1, 8, 32, 128, 512, 2048, 8192)
    labels = ("1", "2-8", "9-32", "33-128", "129-512", "513-2048", "2049-8192")
    for edge, label in zip(edges, labels):
        if n <= edge:
            return label
    return "8193+"


def select_plan(report: IntensityReport, *, length: int | None = None) -> PlanDecision:
    """Pick MemMode, ComputeMode, or both, from an SCIC report."""
    kind = report.kind
    bucket_n = length if length is not None else _length_hint(report)
    key = f"{kind}|{report.dtype}|{report.machine}|bucket={shape_bucket(bucket_n)}"

    if kind in ALWAYS_MEM_KINDS:
        plan = _template("MemMode", kind)
        return PlanDecision(
            kind=kind,
            label=report.label,
            regime=report.regime,
            ai=report.ai,
            ridge=report.ridge,
            primary=plan,
            alternate=None,
            emit_both=False,
            cache_key=key,
            measured_winner=None,
            kernel_emitted=False,
            rationale=(
                "Intrinsic memory op (AI is O(1) FLOP/byte). Only MemMode is "
                "emitted: fuse the chain, reduction, norm, or softmax into its "
                "producer. No kernel is generated. Stage-Stream is not emitted."
            ),
            detail={"shape_bucket": shape_bucket(bucket_n)},
        )

    mem = _template("MemMode", kind)
    comp = _template("ComputeMode", kind)
    if report.regime == "memory-bound":
        primary, alternate, both = mem, None, False
        why = "AI is below 0.5 R, so the primary plan is MemMode."
    elif report.regime == "compute-bound":
        primary, alternate, both = comp, None, False
        why = "AI is above 2 R, so the primary plan is ComputeMode."
    else:
        if report.ai < report.ridge:
            primary, alternate = mem, comp
            side = "below"
        else:
            primary, alternate = comp, mem
            side = "above"
        both = True
        why = (
            f"AI is inside the ridge band and {side} the ridge point. "
            "Emit both plans. The measured winner is empty."
        )
    epilogue_policy = ""
    if kind in ATTENTION_KINDS:
        # Analytical primary stays MemMode or ComputeMode (tiles, and whether
        # a custom mainloop would also absorb a heavy epilogue). The bakeoff
        # cannot emit that mainloop. Both of its attention paths call library
        # SDPA, which already keeps scores off HBM. ``always_unfuse`` only
        # splits RMSNorm into a second timed region. The pre-fix A40
        # wall-clock (paper/data/bakeoff) lost at prefill S=512 and S=2048.
        # Score compile_sdpa_chain, timed as always_fuse, in every attention
        # regime, including ridge and compute. paper/data/bakeoff_v2 is that
        # wall-clock. The GPU was not idle. GEMM's analytical ComputeMode
        # still leaves fuse_epilogue false. The bakeoff may set
        # measured_winner from a wall-clock probe; select_plan does not time it.
        epilogue_policy = POLICY_COMPILE_SDPA_CHAIN
        why += (
            " Scores stay off HBM in both plans. Tile size and extra epilogue "
            "fusion are what switch on a custom kernel. This is not a "
            "Stage-Stream implementation and not a new attention algorithm. "
            "Library SDPA already fuses scores, so the executable epilogue "
            f"policy is {POLICY_COMPILE_SDPA_CHAIN} (timed as always_fuse), "
            "including when the analytical primary is ComputeMode. "
            "always_unfuse only splits RMSNorm off that library kernel. "
            "paper/data/bakeoff_v2 records the wall-clock for this policy. "
            "That GPU was not idle. "
            "GEMM's analytical ComputeMode still leaves fuse_epilogue false. "
            "A wall-clock probe may override the GEMM executable path later. "
            "select_plan does not time that probe, and the probe is not DRAM AI."
        )
    elif kind == "moe_grouped":
        why += " Grouped-GEMM aggregate only; split experts with moe_expert rows."
    else:
        why += " No kernel is generated; this row is a selector decision."
    return PlanDecision(
        kind=kind,
        label=report.label,
        regime=report.regime,
        ai=report.ai,
        ridge=report.ridge,
        primary=primary,
        alternate=alternate,
        emit_both=both,
        cache_key=key,
        measured_winner=None,
        kernel_emitted=False,
        rationale=why,
        epilogue_policy=epilogue_policy,
        detail={"shape_bucket": shape_bucket(bucket_n)},
    )


def moe_is_heterogeneous(expert_regimes: list[str]) -> bool:
    return len(set(expert_regimes)) > 1


def _template(mode: str, kind: str) -> Plan:
    if mode == "MemMode":
        return Plan(
            mode="MemMode",
            fuse_epilogue=True,
            materialize_intermediate=False,
            tile="small",
            protect_mma_pipeline=False,
            note=(
                "MemMode: fuse the epilogue, keep intermediates off HBM, smaller "
                "tiles. Plan choice only; Stage-Stream is not emitted."
            ),
        )
    materialize = kind in MATERIALIZE_ON_COMPUTE
    return Plan(
        mode="ComputeMode",
        fuse_epilogue=False,
        materialize_intermediate=materialize,
        tile="large",
        protect_mma_pipeline=True,
        note=(
            "ComputeMode: large tiles, do not fuse extra epilogue into the "
            "mainloop, protect the MMA pipeline. Attention still keeps scores "
            "off HBM. Stage-Stream is not emitted."
        ),
    )


def _length_hint(report: IntensityReport) -> int:
    d = report.detail
    for key in ("M", "S", "Sk", "tokens", "n"):
        if key in d:
            return int(d[key])
    counts = d.get("token_counts")
    if counts:
        return int(max(counts))
    return 0

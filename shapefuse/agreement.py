"""Dual-plan decision log and regime-agreement rates.

The log always records the analytical SCIC decision for each timed shape.
A second regime is computed only when a byte-derived AI is present.

``achieved_ai`` from Nsight can fill that column. ``proxy_ai`` from
``torch.profiler`` memory can too, but the summary marks that rate as not
valid SCIC accuracy: allocator bytes are not DRAM traffic. Agreement against
the analytical cold-cache byte count is intentionally not computed, because
it would be 1 by construction.
"""

from __future__ import annotations

from shapefuse.intensity import gemm_flops
from shapefuse.machines import Machine
from shapefuse.plans import select_plan
from shapefuse.profiler_hooks import NSIGHT_AI_STATUSES, STRICT_NSIGHT_STATUSES, TORCH_MEMORY
from shapefuse.scic import Site, classify_site


def build_decision_log(bench_rows: list[dict], machine: Machine) -> list[dict]:
    rows = []
    for bench in bench_rows:
        m, n, k = int(bench["M"]), int(bench["N"]), int(bench["K"])
        site = Site("gemm", str(bench["label"]), {"M": m, "N": n, "K": k})
        analytical = classify_site(site, machine)
        analytical_plan = select_plan(analytical, length=m)
        byte_ai, byte_status = _byte_ai(bench)
        if byte_ai is None:
            byte_regime = ""
            byte_mode = ""
            byte_both = ""
            regime_agree = ""
            decision_agree = ""
            decision_flipped = ""
        else:
            measured = classify_site(site, machine, measured_ai=byte_ai)
            measured_plan = select_plan(measured, length=m)
            byte_regime = measured.regime
            byte_mode = measured_plan.primary.mode
            byte_both = measured_plan.emit_both
            regime_agree = analytical.regime == measured.regime
            decision_agree = (
                analytical_plan.primary.mode == measured_plan.primary.mode
                and analytical_plan.emit_both == measured_plan.emit_both
            )
            decision_flipped = not decision_agree
        valid = byte_status in STRICT_NSIGHT_STATUSES
        rows.append(
            {
                "label": bench["label"],
                "M": m,
                "N": n,
                "K": k,
                "machine": machine.name,
                "ridge_flops_per_byte": machine.ridge,
                "ridge_source": "datasheet",
                "FLOPs": gemm_flops(m, n, k),
                "analytical_ai": analytical.ai,
                "analytical_regime": analytical.regime,
                "analytical_roofline_limit": analytical.roofline_limit,
                "analytical_primary_mode": analytical_plan.primary.mode,
                "analytical_emit_both": analytical_plan.emit_both,
                "byte_ai": byte_ai,
                "byte_ai_status": byte_status,
                "byte_regime": byte_regime,
                "byte_primary_mode": byte_mode,
                "byte_emit_both": byte_both,
                "regime_agree": regime_agree,
                "decision_agree": decision_agree,
                "decision_flipped": decision_flipped,
                "agreement_valid_for_scic_accuracy": valid if byte_ai is not None else False,
                "kernel_emitted": False,
                "note": (
                    "Analytical plan is SCIC on the datasheet ridge. "
                    "byte_* uses Nsight DRAM AI when status is NSIGHT_DRAM or "
                    "NSIGHT_DRAM_PROCESS_SUM, else torch.profiler memory AI. "
                    "Only NSIGHT_DRAM is valid SCIC accuracy. "
                    "MemMode and ComputeMode kernels are not timed."
                ),
            }
        )
    return rows


def summarize_agreement(log_rows: list[dict]) -> dict:
    comparable = [row for row in log_rows if row["byte_ai"] is not None and row["byte_ai"] != ""]
    strict = [row for row in comparable if row["byte_ai_status"] in STRICT_NSIGHT_STATUSES]
    process = [row for row in comparable if row["byte_ai_status"] in NSIGHT_AI_STATUSES]
    proxy = [row for row in comparable if row["byte_ai_status"] == TORCH_MEMORY]
    if strict:
        status = "NSIGHT_DRAM"
        quote = True
    elif process:
        status = "NSIGHT_DRAM_PROCESS_SUM_NOT_ISOLATED_KERNEL"
        quote = False
    elif proxy:
        status = "PROXY_TORCH_MEMORY_NOT_VALID_SCIC_ACCURACY"
        quote = False
    elif comparable:
        status = "BYTE_AI_PRESENT_NOT_VALID_SCIC_ACCURACY"
        quote = False
    else:
        status = "TBD_NO_MEASURED_AI"
        quote = False
    return {
        "n_shapes": len(log_rows),
        "n_with_byte_ai": len(comparable),
        "n_nsight_dram_isolated": len(strict),
        "n_nsight_process_sum": len(process),
        "n_torch_profiler_memory": len(proxy),
        "regime_agreement_rate_any_byte_source": _rate(comparable, "regime_agree"),
        "decision_agreement_rate_any_byte_source": _rate(comparable, "decision_agree"),
        "n_decision_flips_any_byte_source": _count_flips(comparable),
        "regime_agreement_rate_nsight_isolated": _rate(strict, "regime_agree"),
        "decision_agreement_rate_nsight_isolated": _rate(strict, "decision_agree"),
        "regime_agreement_rate_nsight_process_sum": _rate(process, "regime_agree"),
        "decision_agreement_rate_nsight_process_sum": _rate(process, "decision_agree"),
        "scic_accuracy_status": status,
        "quote_as_scic_accuracy": quote,
        "kernel_emitted": 0,
        "note": (
            "quote_as_scic_accuracy is true only for isolated Nsight DRAM "
            "(status NSIGHT_DRAM). A process-sum Nsight capture and a torch "
            "profiler memory estimate still produce rates so the next run has "
            "a number to inspect; do not put those rates in the paper as "
            "SCIC accuracy. The ridge is still the datasheet ridge."
        ),
    }


def _byte_ai(bench: dict) -> tuple[float | None, str]:
    status = str(bench.get("achieved_ai_status") or "")
    achieved = bench.get("achieved_ai")
    if status in NSIGHT_AI_STATUSES and achieved not in (None, ""):
        return float(achieved), status
    proxy = bench.get("proxy_ai")
    proxy_status = str(bench.get("proxy_ai_status") or "")
    if proxy not in (None, "") and proxy_status and proxy_status != "PROFILER_OFF":
        return float(proxy), proxy_status
    return None, "TBD_NO_MEASURED_AI"


def _rate(rows: list[dict], key: str) -> float | None:
    if not rows:
        return None
    agreed = 0
    for row in rows:
        value = row[key]
        if value is True or value == "true":
            agreed += 1
    return agreed / len(rows)


def _count_flips(rows: list[dict]) -> int:
    flips = 0
    for row in rows:
        value = row["decision_flipped"]
        if value is True or value == "true":
            flips += 1
    return flips

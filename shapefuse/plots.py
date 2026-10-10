"""Analytical figures. Titles say they are models, not device traces."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shapefuse.intensity import classify, gemm_ai
from shapefuse.machines import A100_SXM80_FP16, A40_FP16, H100_SXM_FP16, Machine
from shapefuse.sweeps import SweepBundle

MEM = "#4C78A8"
RIDGE = "#F58518"
COMPUTE = "#E45756"
FA = "#54A24B"


def write_figures(bundle: SweepBundle, out: Path, proxy_rows: list[dict] | None = None) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    paths = [
        _gemm(bundle, out / "fig_gemm_ai_vs_M.png"),
        _attn(bundle, out / "fig_attn_ai_vs_S.png"),
        _moe(bundle, out / "fig_moe_skew_ai.png"),
        _epilogue(bundle, out / "fig_epilogue_tradeoff.png"),
        _plans(bundle, out / "fig_plan_regions.png"),
    ]
    if proxy_rows:
        paths.append(_cpu(proxy_rows, out / "fig_cpu_gflops_vs_M.png"))
    return [str(p.name) for p in paths]


def _gemm(bundle: SweepBundle, path: Path) -> Path:
    d = 4096
    ms = [2**i for i in range(0, 14)]
    ais = [gemm_ai(m, d, d, 2, 2, 2) for m in ms]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(ms, ais, "o-", color="#1F4E79", label="GEMM AI, N=K=4096, FP16 bytes")
    ax.axhline(H100_SXM_FP16.ridge, color=COMPUTE, ls="--", label=f"H100 datasheet ridge {H100_SXM_FP16.ridge:.0f}")
    ax.axhline(A40_FP16.ridge, color=RIDGE, ls="--", label=f"A40 datasheet ridge {A40_FP16.ridge:.0f}")
    ax.axhline(A100_SXM80_FP16.ridge, color=FA, ls="--", label=f"A100 datasheet ridge {A100_SXM80_FP16.ridge:.0f}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("M")
    ax.set_ylabel("Arithmetic intensity (FLOP/byte)")
    ax.set_title("Same GEMM, different side of the ridge (analytical)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _attn(bundle: SweepBundle, path: Path) -> Path:
    rows = bundle.attention
    ss = [r["S"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(ss, [r["AI_QKT"] for r in rows], "o-", label="QK^T writes scores")
    ax.plot(ss, [r["AI_Softmax"] for r in rows], "s-", label="Softmax alone")
    ax.plot(ss, [r["AI_unfused_scores"] for r in rows], "v-", label="QK+softmax+PV, scores in HBM")
    ax.plot(ss, [r["AI_FA_style_IO"] for r in rows], "^-", color=FA, label="FA-style IO (Q,K,V,O only)")
    decode = [r for r in bundle.attention_decode if r["variant"] == "mha"]
    if decode:
        ax.axhline(decode[-1]["AI_fp16"], color=MEM, ls=":", label="Decode MHA AI (long cache)")
    ax.axhline(bundle.machine.ridge, color=COMPUTE, ls="--", label="Requested-machine ridge")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length S (prefill)")
    ax.set_ylabel("Arithmetic intensity (FLOP/byte)")
    ax.set_title("Attention intensity depends on whether scores hit HBM")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _moe(bundle: SweepBundle, path: Path) -> Path:
    skewed = next(r for r in bundle.moe if r["scenario"] == "skewed_8e")
    tokens = skewed["token_counts"]
    ais = [gemm_ai(t, skewed["I"], skewed["H"], 2, 2, 2) for t in tokens]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    colors = []
    for ai in ais:
        regime = classify(ai, bundle.machine.ridge, bundle.thresholds)
        colors.append({"memory-bound": MEM, "ridge-region": RIDGE, "compute-bound": COMPUTE}[regime])
    labels = [str(t) for t in tokens]
    ax.barh(labels, ais, color=colors)
    ax.axvline(bundle.machine.ridge, color=COMPUTE, ls="--", label="Ridge R")
    ax.axvline(0.5 * bundle.machine.ridge, color=MEM, ls=":", label="0.5 R")
    ax.axvline(2.0 * bundle.machine.ridge, color=RIDGE, ls=":", label="2 R")
    ax.set_xlabel("Per-expert AI (FLOP/byte, FP16)")
    ax.set_ylabel("Tokens on that expert")
    ax.set_title("Skewed MoE: one layer, more than one regime")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _epilogue(bundle: SweepBundle, path: Path) -> Path:
    rows = [r for r in bundle.rmsnorm if r["K"] == 4096]
    ms = [r["M"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    axes[0].plot(ms, [r["byte_save_fraction"] for r in rows], "o-", color=MEM)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("M (N=K=4096)")
    axes[0].set_ylabel("Fraction of unfused bytes removed")
    axes[0].set_title("Byte savings grow with M")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(ms, [r["roofline_speedup"] for r in rows], "o-", label="Roofline, no launch")
    axes[1].plot(
        ms,
        [r["roofline_speedup_with_launch"] for r in rows],
        "s--",
        label=f"Plus assumed {rows[0]['launch_overhead_us']:.0f} µs launch",
    )
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("M (N=K=4096)")
    axes[1].set_ylabel("Unfused time / fused time")
    axes[1].set_title("Launch term is not a measurement")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _plans(bundle: SweepBundle, path: Path) -> Path:
    d = 4096.0
    cross = bundle.crossings.get(bundle.machine.name)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ms = [2**i for i in range(0, 14)]
    ais = [gemm_ai(m, int(d), int(d), 2, 2, 2) for m in ms]
    ax.plot(ms, ais, "o-", color="#1F4E79", label="GEMM AI")
    if cross and bundle.machine.peak_dtype == "fp16":
        m0 = cross["M_memory_edge_0.5R"]
        m1 = cross["M_compute_edge_2R"]
        ax.axvspan(1, m0, color=MEM, alpha=0.15, label="MemMode (AI < 0.5 R)")
        ax.axvspan(m0, m1, color=RIDGE, alpha=0.15, label="Emit both (ridge band)")
        ax.axvspan(m1, 8192, color=COMPUTE, alpha=0.12, label="ComputeMode (AI > 2 R)")
    ax.axhline(bundle.machine.ridge, color=COMPUTE, ls="--", label="Ridge R")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(1, 8192)
    ax.set_xlabel("M (N=K=4096, FP16 model)")
    ax.set_ylabel("Arithmetic intensity (FLOP/byte)")
    ax.set_title("Dual-plan regions for a square projection (selector stub)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def write_vendor_matmul_figure(rows: list[dict], path: Path, machine: Machine | None = None) -> Path:
    """Bar chart of timed matmul throughput.

    The series is vendor ``matmul`` (or a CPU placeholder). It is not a
    MemMode versus ComputeMode comparison. A datasheet peak is drawn only
    when it sits near the measured bars; an H100 peak above an A40 trace
    would otherwise flatten the figure.
    """
    usable = [r for r in rows if r.get("achieved_tflops") not in (None, "")]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    if not usable:
        ax.set_title("No matmul timings")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return path
    values = [float(r["achieved_tflops"]) for r in usable]
    labels = [f"{r['label']}\nM={r['M']}" for r in usable]
    ax.bar(range(len(usable)), values, color="#1F4E79")
    ax.set_xticks(range(len(usable)), labels, fontsize=8)
    if min(values) > 0 and max(values) / min(values) > 10:
        ax.set_yscale("log")
    placeholder = _as_bool(usable[0].get("placeholder"))
    device = str(usable[0].get("device_name") or "device")
    if placeholder:
        title = "PLACEHOLDER matmul timing — not a GPU Roofline"
    else:
        title = f"{device}: torch.matmul only (PARTIAL_VENDOR_MATMUL)"
    ax.set_ylabel("Achieved TFLOP/s")
    ax.set_xlabel("Not MemMode vs ComputeMode. This figure has no DRAM bytes.")
    if machine is not None and not placeholder:
        peak = machine.peak_tflops
        if peak <= max(values) * 3.0:
            ax.axhline(
                peak,
                color=COMPUTE,
                ls="--",
                label=f"{machine.name} datasheet peak {peak:.1f} (not measured)",
            )
            ax.legend(fontsize=8)
        else:
            title += f"\n{machine.name} datasheet peak {peak:.1f} TFLOP/s is off-scale and not drawn"
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _cpu(rows: list[dict], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot([r["M"] for r in rows], [r["cpu_GFLOPS"] for r in rows], "s-", color=MEM)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(f"M (N=K={rows[0]['N']})")
    ax.set_ylabel("NumPy FP32 GFLOP/s (proxy)")
    ax.set_title("CPU proxy only — not an H100 curve")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path

#!/usr/bin/env python3
"""Regenerate Journal of Systems Architecture figures from archived CSVs.

Reads only files already in the repository:

- paper/data/a40/bench_gemm.csv
- paper/data/a40/summary.json          (datasheet crossings)
- paper/data/bakeoff_v4/bakeoff_latency.csv
- paper/data/bakeoff_v4/bakeoff_decisions.csv
- paper/data/bakeoff_v4/bakeoff_summary.json

The Roofline curve and the GEMM AI(M) curve use the same closed forms as
shapefuse.intensity (cold-HBM bytes, 2 FLOPs per MAC). Markers on those
curves are analytical_ai cells from bakeoff_decisions.csv. No latency is
invented. Achieved arithmetic intensity is not plotted: Nsight DRAM bytes
were not collected.

Run from the repository root:

    python paper/jsa/scripts/plot_jsa_figures.py

Writes PDF and PNG into paper/jsa/figures/.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from shapefuse.intensity import square_proj_ai  # noqa: E402

A40_DIR = REPO / "paper" / "data" / "a40"
V4_DIR = REPO / "paper" / "data" / "bakeoff_v4"
OUT = REPO / "paper" / "jsa" / "figures"

# Okabe–Ito, print-friendly.
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
BLACK = "#222222"
GRAY = "#666666"
BAND = "#F4E4C4"

GEMM_ORDER = [
    "decode_m1",
    "decode_m32",
    "ridge_m128",
    "ridge_m256",
    "ridge_m512",
    "prefill_m2048",
    "prefill_m4096",
]
ATTN_ORDER = [
    "prefill_s128",
    "prefill_s512",
    "prefill_s2048",
    "decode_sk128",
    "decode_sk2048",
]
ATTN_TICKS = [
    "Prefill\n$S{=}128$",
    "Prefill\n$S{=}512$",
    "Prefill\n$S{=}2048$",
    "Decode\n$S_k{=}128$",
    "Decode\n$S_k{=}2048$",
]


def _load_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _latency_index(rows: list[dict]) -> dict[tuple[str, str, str], dict]:
    return {(row["op"], row["label"], row["baseline"]): row for row in rows}


def _f(row: dict, key: str) -> float:
    return float(row[key])


def plot_roofline(decisions: list[dict], summary_a40: dict) -> None:
    """Datasheet Roofline schematic and the analytical GEMM intensity curve."""
    ridge = float(summary_a40["ridge_flops_per_byte"])
    # Datasheet anchors stored with the A40 package, not a measured peak.
    peak_tflops = 149.7
    bandwidth_tb = 0.696  # 696 GB/s
    crossings = summary_a40["crossings"]["A40_FP16"]
    m_star = float(crossings["M_star_AI_equals_ridge"])
    m_lo = float(crossings["M_memory_edge_0.5R"])
    m_hi = float(crossings["M_compute_edge_2R"])

    gemm = [row for row in decisions if row["op"] == "matmul_rmsnorm"]
    by_label = {row["label"]: row for row in gemm}

    fig, axes = plt.subplots(1, 2, figsize=(7.55, 3.35), constrained_layout=True)
    ax = axes[0]
    ais = [10 ** (x / 80) for x in range(-40, 280)]  # ~0.3 to ~3000
    roofs = [min(peak_tflops, ai * bandwidth_tb) for ai in ais]
    ax.plot(ais, roofs, color=BLACK, lw=1.6, zorder=3)
    ax.axvline(ridge, color=VERMILION, ls="--", lw=0.8, zorder=2)
    ax.axvspan(0.5 * ridge, 2.0 * ridge, color=BAND, zorder=0)
    # Analytical sites from the decision log, placed on the datasheet roof.
    # Decode AI sits on the same point as GEMM M=1, so it is not drawn twice.
    attn = {row["label"]: row for row in decisions if row["op"] == "attention"}
    marked = [
        (float(by_label["decode_m1"]["analytical_ai"]), "GEMM $M{=}1$", BLUE, "o", (10, 7)),
        (float(attn["prefill_s128"]["analytical_ai"]), "Attn. $S{=}128$", PURPLE, "D", (9, 7)),
        (float(by_label["ridge_m512"]["analytical_ai"]), "GEMM $M{=}512$", BLUE, "o", (8, -14)),
        (float(by_label["prefill_m2048"]["analytical_ai"]), "GEMM $M{=}2048$", BLUE, "o", (8, 8)),
    ]
    for ai, text, color, marker, (dx, dy) in marked:
        y = min(peak_tflops, ai * bandwidth_tb)
        ax.scatter([ai], [y], s=26, color=color, marker=marker, zorder=4)
        ha = "right" if dx < 0 else "left"
        ax.annotate(
            text,
            (ai, y),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=7,
            color=BLACK,
            ha=ha,
        )
    ax.annotate(
        f"$R{{=}}{ridge:.0f}$",
        (ridge, 3.2),
        textcoords="offset points",
        xytext=(-4, 0),
        fontsize=7.5,
        color=VERMILION,
        ha="right",
        va="center",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.4, 5000)
    ax.set_ylim(0.3, 400)
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Datasheet attainable (TFLOP/s)")
    ax.set_title("(a) A40 datasheet Roofline")
    ax.legend(
        handles=[
            Line2D([0], [0], color=BLACK, lw=1.6, label="Datasheet roof"),
            Patch(facecolor=BAND, edgecolor="none", label="Policy band $[0.5R,\\,2R]$"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markersize=5, label="GEMM site"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor=PURPLE, markersize=5, label="Attention site"),
        ],
        loc="upper left",
        frameon=False,
    )
    ax.grid(True, which="major", alpha=0.25, lw=0.4)

    ax = axes[1]
    ms = [2 ** (i / 4) for i in range(0, 49)]  # 1 .. 4096
    curve = [square_proj_ai(m, 4096, 2.0) for m in ms]
    ax.plot(ms, curve, color=BLACK, lw=1.5, label="Analytical AI, $N{=}K{=}4096$")
    ax.axhspan(0.5 * ridge, 2.0 * ridge, color=BAND, zorder=0)
    ax.axhline(ridge, color=VERMILION, ls="--", lw=0.8, label=f"A40 ridge $R{{=}}{ridge:.0f}$")
    ax.axhline(0.5 * ridge, color=BLUE, ls=":", lw=0.7)
    ax.axhline(2.0 * ridge, color=ORANGE, ls=":", lw=0.7)
    ax.axvline(m_star, color=GRAY, ls=":", lw=0.7)
    for label in GEMM_ORDER:
        row = by_label[label]
        # M is the first field of the shape string "M=1,N=4096,K=4096".
        m_tok = row["shape"].split(",")[0]
        m_val = int(m_tok.split("=")[1])
        ai = float(row["analytical_ai"])
        color = BLUE if row["selected_baseline"] == "always_fuse" else ORANGE
        ax.scatter([m_val], [ai], s=28, color=color, zorder=4, edgecolors="white", linewidths=0.4)
    ax.annotate(
        f"$M^\\star{{=}}{m_star:.0f}$",
        (m_star, 8),
        fontsize=7,
        color=GRAY,
        ha="center",
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(1, 5000)
    ax.set_ylim(0.5, 3000)
    ax.set_xlabel("$M$")
    ax.set_ylabel("Arithmetic intensity (FLOP/byte)")
    ax.set_title("(b) GEMM intensity and the selected path")
    handles = [
        Line2D([0], [0], color=BLACK, lw=1.5, label="Closed-form AI"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE, markeredgecolor="white", markersize=6, label="Probe selects unfuse"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor="white", markersize=6, label="Probe selects fuse"),
        Patch(facecolor=BAND, edgecolor="none", label="Ridge band"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False)
    ax.grid(True, which="major", alpha=0.25, lw=0.4)
    # Silence unused locals that document the band edges in the CSV.
    _ = (m_lo, m_hi)
    _save(fig, "fig_roofline_gemm")


def plot_matmul(bench: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(4.9, 3.25), constrained_layout=True)
    regime_color = {
        "memory-bound": BLUE,
        "ridge-region": ORANGE,
        "compute-bound": VERMILION,
    }
    ms = [int(row["M"]) for row in bench]
    tflops = [float(row["achieved_tflops"]) for row in bench]
    colors = [regime_color[row["regime_vs_requested_datasheet"]] for row in bench]
    ax.plot(ms, tflops, color=GRAY, lw=0.8, zorder=1)
    ax.scatter(ms, tflops, c=colors, s=36, zorder=3, edgecolors="white", linewidths=0.4)
    ax.axhline(149.7, color=BLACK, ls="--", lw=0.8, label="A40 datasheet peak 149.7 TFLOP/s")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("$M$  ($N{=}K{=}4096$, FP16)")
    ax.set_ylabel("Achieved TFLOP/s")
    ax.set_title("Stock torch.matmul on an NVIDIA A40")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markersize=6, label="Memory band"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE, markersize=6, label="Ridge band"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=VERMILION, markersize=6, label="Compute band"),
        Line2D([0], [0], color=BLACK, ls="--", lw=0.8, label="Datasheet peak"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    ax.grid(True, which="major", alpha=0.25, lw=0.4)
    ax.set_ylim(0.3, 250)
    _save(fig, "fig_a40_torch_matmul_tflops")


def plot_gemm_speedup(latency: dict[tuple[str, str, str], dict], decisions: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.35), constrained_layout=True)
    by_label = {row["label"]: row for row in decisions if row["op"] == "matmul_rmsnorm"}
    ms = []
    series = {
        "always_fuse": [],
        "always_unfuse": [],
        "torch_compile_default": [],
        "scic_selected": [],
    }
    for label in GEMM_ORDER:
        shape_m = int(by_label[label]["shape"].split(",")[0].split("=")[1])
        ms.append(shape_m)
        for name in series:
            series[name].append(_f(latency[("matmul_rmsnorm", label, name)], "speedup_vs_eager"))
    ax.axhline(1.0, color=GRAY, lw=0.7, zorder=0)
    ax.plot(ms, series["always_fuse"], "o-", color=BLUE, ms=5, lw=1.1, label="always-fuse (torch.compile)")
    ax.plot(ms, series["torch_compile_default"], "s--", color=SKY, ms=4, lw=0.9, label="torch.compile, separate timing")
    ax.plot(ms, series["always_unfuse"], "^-", color=ORANGE, ms=5, lw=1.1, label="always-unfuse (eager chain)")
    ax.plot(ms, series["scic_selected"], "D", color=GREEN, ms=6, lw=0, label="SCIC-selected (copied median)", zorder=4)
    ax.axvspan(256, 512, color=BAND, zorder=0, alpha=0.9)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("$M$  ($N{=}K{=}4096$, MatMul+RMSNorm)")
    ax.set_ylabel("Speedup versus eager")
    ax.set_title("GEMM path selection on the idle A40")
    ax.set_ylim(0.7, 1.32)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, which="major", axis="y", alpha=0.25, lw=0.4)
    _save(fig, "fig_gemm_plan_speedup")


def plot_attention(latency: dict[tuple[str, str, str], dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.55, 3.35), constrained_layout=True)
    import numpy as np

    x = np.arange(len(ATTN_ORDER))
    width = 0.24
    paths = [
        ("sdpa", "Eager SDPA", PURPLE),
        ("always_unfuse", "SDPA, norm split", ORANGE),
        ("always_fuse", "Compiled chain (selected)", BLUE),
    ]
    ax = axes[0]
    for i, (key, label, color) in enumerate(paths):
        vals = [_f(latency[("attention", lab, key)], "median_ms") for lab in ATTN_ORDER]
        ax.bar(x + (i - 1) * width, vals, width=width, color=color, label=label, zorder=2)
    ax.set_xticks(x, ATTN_TICKS)
    ax.set_ylabel("Median latency (ms)")
    ax.set_title("(a) Library attention paths")
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    ax.grid(True, axis="y", alpha=0.25, lw=0.4)
    ax.set_ylim(0, 0.26)

    ax = axes[1]
    eager_sp = []
    for key, label, color in (
        ("sdpa", "Eager SDPA", PURPLE),
        ("always_unfuse", "Norm split", ORANGE),
        ("scic_selected", "SCIC (compiled chain)", GREEN),
    ):
        vals = [_f(latency[("attention", lab, key)], "speedup_vs_eager") for lab in ATTN_ORDER]
        eager_sp.append(vals)
        ax.plot(x, vals, "o-", color=color, ms=5, lw=1.15, label=label)
    ax.axhline(1.0, color=GRAY, lw=0.7)
    ax.set_xticks(x, ATTN_TICKS)
    ax.set_ylabel("Speedup versus materialized eager")
    ax.set_title("(b) Against score materialization")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, axis="y", alpha=0.25, lw=0.4)
    ax.set_ylim(0, 8.0)
    _save(fig, "fig_attn_library_paths")


def plot_oracle(latency: dict[tuple[str, str, str], dict]) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.7), constrained_layout=True)
    nice = {
        "decode_m1": "GEMM $M{=}1$",
        "decode_m32": "GEMM $M{=}32$",
        "ridge_m128": "GEMM $M{=}128$",
        "ridge_m256": "GEMM $M{=}256$",
        "ridge_m512": "GEMM $M{=}512$",
        "prefill_m2048": "GEMM $M{=}2048$",
        "prefill_m4096": "GEMM $M{=}4096$",
        "prefill_s128": "Attn. $S{=}128$",
        "prefill_s512": "Attn. $S{=}512$",
        "prefill_s2048": "Attn. $S{=}2048$",
        "decode_sk128": "Attn. $S_k{=}128$",
        "decode_sk2048": "Attn. $S_k{=}2048$",
    }
    rows = []
    for label in GEMM_ORDER:
        fuse = _f(latency[("matmul_rmsnorm", label, "always_fuse")], "median_ms")
        unfuse = _f(latency[("matmul_rmsnorm", label, "always_unfuse")], "median_ms")
        scic = _f(latency[("matmul_rmsnorm", label, "scic_selected")], "median_ms")
        rows.append((nice[label], scic / min(fuse, unfuse), "gemm"))
    for label in ATTN_ORDER:
        fuse = _f(latency[("attention", label, "always_fuse")], "median_ms")
        unfuse = _f(latency[("attention", label, "always_unfuse")], "median_ms")
        scic = _f(latency[("attention", label, "scic_selected")], "median_ms")
        rows.append((nice[label], scic / min(fuse, unfuse), "attn"))
    # Largest gap at the top.
    rows_plot = list(reversed(rows))
    y = range(len(rows_plot))
    colors = [BLUE if kind == "gemm" else PURPLE for _, _, kind in rows_plot]
    ax.barh(list(y), [gap for _, gap, _ in rows_plot], color=colors, height=0.72, zorder=2)
    ax.axvline(1.0, color=BLACK, lw=0.8)
    ax.set_yticks(list(y), [name for name, _, _ in rows_plot])
    ax.set_xlabel("SCIC latency / faster of fuse and unfuse")
    ax.set_xlim(0.9, 1.85)
    ax.set_title("Gap to the two-path oracle")
    handles = [
        Patch(facecolor=BLUE, label="MatMul+RMSNorm"),
        Patch(facecolor=PURPLE, label="Attention+RMSNorm"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False)
    ax.grid(True, axis="x", alpha=0.25, lw=0.4)
    _save(fig, "fig_oracle_gap")


def main() -> None:
    _style()
    bench = _load_csv(A40_DIR / "bench_gemm.csv")
    decisions = _load_csv(V4_DIR / "bakeoff_decisions.csv")
    latency_rows = _load_csv(V4_DIR / "bakeoff_latency.csv")
    summary_a40 = json.loads((A40_DIR / "summary.json").read_text())
    latency = _latency_index(latency_rows)
    plot_roofline(decisions, summary_a40)
    plot_matmul(bench)
    plot_gemm_speedup(latency, decisions)
    plot_attention(latency)
    plot_oracle(latency)
    written = sorted(p.name for p in OUT.iterdir() if p.suffix in {".pdf", ".png"})
    print(f"wrote {len(written)} files under {OUT}")
    for name in written:
        print(f"  {name}")


if __name__ == "__main__":
    main()

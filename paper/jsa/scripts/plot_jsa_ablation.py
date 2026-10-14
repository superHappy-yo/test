#!/usr/bin/env python3
"""Export complete-model adapter diagnostics with process-cluster intervals."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def main():
    rows = list(csv.DictReader((ROOT/"paper/data/jsa_ablation/analysis/summary.csv").open()))
    policies = ["fixed_compile", "fixed_compile_single", "fixed_compile_bound", "inline_strict", "scic", "scic_bound"]
    labels = ["Fixed\n2 lookups", "Fixed\n1 lookup", "Fixed\nprebound",
              "Inline\nstatic", "SCIC\n2 lookups", "SCIC\nprebound"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.8), sharey=True)
    for ax, phase, title in zip(axes, ["prefill", "decode64"], ["Prefill", "64-step decode"]):
        selected = [next(r for r in rows if r["stage"]=="system" and r["baseline"]=="original"
                         and r["case"]=="all" and r["phase"]==phase and r["policy"]==p) for p in policies]
        values = np.array([float(r["geomean_host_speedup"]) for r in selected])
        low = np.array([float(r["ci95_low"]) for r in selected])
        high = np.array([float(r["ci95_high"]) for r in selected])
        # Percentile intervals need not contain the all-pair point estimate in
        # an incomplete matrix. Plot only the completed, balanced experiment.
        if any(int(r["pairs"]) != 24 or int(r["clusters"]) != 6 for r in selected):
            raise RuntimeError("Figure requires the complete two-device matrix")
        x = np.arange(len(policies))
        ax.bar(x, values, color=["#8ba5b5", "#6e91a7", "#4d788f", "#367a55", "#bf9568", "#a67345"], width=.72)
        ax.errorbar(x, values, yerr=[values-low, high-values], fmt="none", ecolor="black", capsize=3, linewidth=1)
        ax.axhline(1, color="#555555", linestyle="--", linewidth=1)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0, max(1.15, max(high)*1.15))
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        for pos, value, cap in zip(x, values, high):
            ax.text(pos, cap+.025, f"{value:.3f}", ha="center", fontsize=8)
    axes[0].set_ylabel("Host-time speed ratio vs original compiled model")
    fig.tight_layout()
    out = ROOT/"paper/jsa/figures/fig_adapter_ablation"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

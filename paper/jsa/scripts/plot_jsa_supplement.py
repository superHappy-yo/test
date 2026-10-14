#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(summary,out):
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":8,"axes.spines.top":False,"axes.spines.right":False,
                         "pdf.fonttype":42})
    rows=summary["system_summary"]
    fig,axes=plt.subplots(1,2,figsize=(7.55,2.65),sharey=True)
    for ax,phase in zip(axes,("prefill","decode64")):
        cases=[("1","128"),("1","2048"),("8","128"),("8","2048")]
        x=np.arange(len(cases))
        for i,(policy,label,color) in enumerate([
            ("fixed_native","Native adapter","#a6bddb"),("fixed_compile","Static compiled","#74a9cf"),
            ("scic","SCIC","#045a8d")]):
            selected=[next((r for r in rows if r["phase"]==phase and r["policy"]==policy
                           and (r["batch"],r["prompt_length"])==case),None) for case in cases]
            values=np.array([r["geomean_host_speedup"] if r else np.nan for r in selected])
            lower=np.array([r["ci95_low"] if r else np.nan for r in selected])
            upper=np.array([r["ci95_high"] if r else np.nan for r in selected])
            ax.bar(x+(i-1)*.24,values,width=.23,label=label,color=color,
                   yerr=[np.maximum(0,values-lower),np.maximum(0,upper-values)],
                   capsize=2,error_kw={"elinewidth":.6})
        ax.axhline(1,color="black",linestyle="--",linewidth=.7)
        ax.set_xticks(x,["B1/S128","B1/S2048","B8/S128","B8/S2048"],rotation=15)
        ax.set_title("Prefill" if phase=="prefill" else "64 teacher-forced decode steps")
        ax.set_ylim(0,1.25)
        ax.grid(axis="y",alpha=.2)
    axes[0].set_ylabel("Speedup vs original compiled model")
    axes[1].legend(loc="upper right",fontsize=6)
    fig.tight_layout()
    for ext in ("pdf","png"):
        fig.savefig(out/f"fig_supplement_system.{ext}",dpi=220)
    plt.close(fig)
    rows=summary["operators"]
    fig,axes=plt.subplots(1,2,figsize=(7.55,2.65))
    for ax,op in zip(axes,("gemm","attention")):
        methods=["scic_off","scic_ridge-only","scic_ridge-compute","scic_all-gemm"]
        x=np.arange(len(methods))
        for i,(family,graph,label,color) in enumerate([
             ("torch251","none","2.5.1 / direct","#a6bddb"),
             ("torch251","manual","2.5.1 / graph","#74a9cf"),
             ("torch214","none","2.14.1 / direct","#045a8d"),
             ("torch214","manual","2.14.1 / graph","#023858")]):
            selected=[next((r for r in rows if r["suite"]==family and r["graph"]==graph
                              and r["op"]==op and r["method"]==m),None) for m in methods]
            vals=[r["geomean_speedup_vs_compile_default"] if r else np.nan for r in selected]
            lows=[r.get("ci95_low_vs_compile_default",np.nan) if r else np.nan for r in selected]
            highs=[r.get("ci95_high_vs_compile_default",np.nan) if r else np.nan for r in selected]
            ax.bar(x+(i-1.5)*.2,vals,width=.19,label=label,color=color,
                   yerr=[np.maximum(0,np.asarray(vals)-lows),np.maximum(0,np.asarray(highs)-vals)],
                   capsize=2,error_kw={"elinewidth":.6})
        ax.axhline(1,color="black",linestyle="--",linewidth=.7)
        ax.set_xticks(x,["Analytical","Ridge only","Ridge+compute","All GEMM"],rotation=15)
        ax.set_title("MatMul + RMSNorm" if op=="gemm" else "SDPA + RMSNorm")
        ax.set_ylabel("Speedup vs static default compilation")
        ax.grid(axis="y",alpha=.2)
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    for ext in ("pdf","png"):fig.savefig(out/f"fig_supplement_selector.{ext}",dpi=220)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--summary",default="paper/data/jsa_supplement/analysis/summary.json")
    p.add_argument("--out",default="paper/jsa/figures")
    args=p.parse_args()
    plot(json.loads(Path(args.summary).read_text()),Path(args.out))

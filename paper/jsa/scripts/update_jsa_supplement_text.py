#!/usr/bin/env python3
"""Write matching English/Chinese tables from admitted, paired measurements."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--analysis",default="paper/data/jsa_supplement/analysis")
    p.add_argument("--paper",default="paper/jsa")
    p.add_argument("--allow-incomplete",action="store_true")
    args=p.parse_args();analysis=Path(args.analysis);paper=Path(args.paper)
    summary=json.loads((analysis/"summary.json").read_text())
    operators=summary["operators"]
    def find(suite,graph,op,method):
        return next(r for r in operators if (r["suite"],r["graph"],r["op"],r["method"])==(suite,graph,op,method))
    records=[]
    for suite,version in (("torch251","2.5.1"),("torch214","2.14.1")):
        for graph in ("none","manual"):
            for op in ("gemm","attention"):
                r=find(suite,graph,op,"scic_ridge-compute")
                expected=126 if op=="gemm" else 54
                if not args.allow_incomplete and (r["rows"]!=expected or r["clusters"]!=6):
                    raise RuntimeError(f"Incomplete formal coverage: {suite}/{graph}/{op}: {r['rows']}/{expected}")
                records.append((version,graph,op,r))
    tex=[r"\subsection{Measured optimized-baseline and research comparisons}",r"\label{sec:supplement-results}",
         "Table~\\ref{tab:supplement-selector} reports separately timed default SCIC against in-environment static paths. "
         "Ratios below one are slowdowns. Eager-relative gains do not imply an improvement over default compilation. "
         "The attention native reference is SDPA plus RMSNorm, not materialized attention. "
         "Max-autotune ratios use only common numerically admitted samples, whose counts are shown in parentheses.",
         r"\begin{table*}[t]",r"\centering\small",
         r"\caption{Executed ridge-plus-compute SCIC. Each ratio is a geometric mean of paired CUDA-event latencies. Direct calls and controlled whole-chain graph replay are separate. $n$ counts process/shape pairs; max-autotune uses common admitted pairs.}",
         r"\label{tab:supplement-selector}",r"\begin{tabular}{@{}lllrrrr@{}}",r"\toprule",
         r"PyTorch & Workload & Timing & $n$ & / native & / default & / max-no-graph ($n$) \\",r"\midrule"]
    zh=["### 5.5 优化基线与近期研究的实测比较", "",
        "下表报告独立计时的默认 ridge-plus-compute SCIC；数值小于 1 表示退化。注意力 native 是 SDPA+RMSNorm，不是物化注意力。与 max-autotune 的比值只使用共同通过数值验证的配对样本，括号注明样本量。相对 eager 的收益不能证明优于默认编译。", "",
        "| PyTorch | 负载 | 计时 | 配对数 | / native | / default | / max-no-graph（配对数） |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for version,graph,op,r in records:
        mode="Direct" if graph=="none" else "Graph"
        kind="GEMM" if op=="gemm" else "Attention"
        values=[r['geomean_speedup_vs_native'],r['geomean_speedup_vs_compile_default'],r['geomean_speedup_vs_compile_max_no_graph']]
        formatted=[f"{v:.3f}" if v is not None else "--" for v in values]
        tail=f"{formatted[-1]} ({r['paired_rows_vs_compile_max_no_graph']})"
        cells=[version,kind,mode,str(r['rows']),*formatted[:2],tail]
        tex.append(" & ".join(cells)+r" \\")
        zh.append("| "+" | ".join(cells)+" |")
    tex += [r"\bottomrule",r"\end{tabular}",r"\end{table*}",
            r"\begin{figure*}[t]",r"\centering",r"\includegraphics[width=\linewidth]{figures/fig_supplement_selector.pdf}",
            r"\caption{Independently timed selection relative to static default compilation, with process-bootstrap intervals. The analytical, ridge-only, default ridge-plus-compute, and all-GEMM scopes share the original $0.5R/2R$ rule. Attention always uses its fixed compiled-SDPA binding.}",
            r"\label{fig:supplement-selector}",r"\end{figure*}"]
    zh += ["", "![SCIC 相对默认编译](figures/fig_supplement_selector.png)", "",
           "图：解析、ridge-only、默认 ridge-plus-compute 和 all-GEMM 范围的独立实测速度比及进程 bootstrap 区间。0.5R/2R 规则保持不变；注意力始终采用固定 compiled-SDPA 绑定。"]
    cal=list(csv.DictReader((analysis/"calibration_summary.csv").open()))
    tex.append("For default-scope GEMM calibration, the following elapsed costs include probe calls and synchronization, after candidate preparation; they exclude compiler preparation and numerical auditing.")
    zh += ["", "默认范围 GEMM 校准的耗时包含候选准备之后的探测调用和同步；不包含编译准备与数值审计。"]
    for suite,version in (("torch251","2.5.1"),("torch214","2.14.1")):
        for graph in ("none","manual"):
            r=next(r for r in cal if (r["suite"],r["graph"],r["method"])==(suite,graph,"scic_ridge-compute"))
            text=f"PyTorch {version} / {graph}: {r['probed']} probes, median {float(r['median_ms']):.3f} ms, p95 {float(r['p95_ms']):.3f} ms."
            tex.append(text)
            zh.append(f"PyTorch {version} / {graph}：{r['probed']} 次探测，中位数 {float(r['median_ms']):.3f} ms，p95 {float(r['p95_ms']):.3f} ms。")
    tex += ["Table~\\ref{tab:supplement-research} compares the admitted research artifacts against their matching compiler baselines. "
            "These are three process repetitions on one A40 per artifact, not a cross-environment ranking. "
            "RedFuser denotes the separately parameterized committed kernel, not successful regeneration.",
            r"\begin{table*}[t]",r"\centering\small",
            r"\caption{Complete attention-plus-compiled-RMSNorm research adapters. Default official schedules are used. Unsupported shapes have no timing. Max-no-graph uses the paired count in parentheses.}",
            r"\label{tab:supplement-research}",r"\begin{tabular}{@{}llrrrr@{}}",r"\toprule",
            r"Artifact & Timing & Shapes & Pairs & / default & / max-no-graph ($n$) \\",r"\midrule"]
    zh += ["", "近期研究各使用一张 A40 上 3 个进程重复，与各自环境的编译基线比较，不进行跨环境排名。RedFuser 表示独立参数化的仓库预生成内核，不表示自动再生成成功。采用官方默认 schedule；不支持的形状没有性能值。", "",
           "| 工件 | 计时 | 形状数 | 配对数 | / default | / max-no-graph（配对数） |",
           "| --- | --- | ---: | ---: | ---: | ---: |"]
    artifacts=[("redfuser_shipped_gpu0","redfuser_shipped","RedFuser committed"),
               ("neptune_gpu1","neptune","Neptune TVM")]
    if any(r["suite"]=="neptune_torch_gpu1" for r in operators):
        artifacts.append(("neptune_torch_gpu1","neptune_torch","Neptune Torch"))
    for suite,method,label in artifacts:
        for graph in ("none","manual"):
            if method=="neptune" and graph=="manual":
                tex.append(r"Neptune TVM & Graph & -- & 0 & unsupported & unsupported \\")
                zh.append("| Neptune TVM | Graph | — | 0 | 当前桥接不支持 | 当前桥接不支持 |")
                continue
            r=find(suite,graph,"attention",method)
            if not args.allow_incomplete and r["clusters"]!=3: raise RuntimeError(f"Incomplete research suite: {suite}/{graph}")
            cells=[label,"Direct" if graph=="none" else "Graph",str(r['cases']),str(r['rows']),
                   f"{r['geomean_speedup_vs_compile_default']:.3f}",
                   f"{r['geomean_speedup_vs_compile_max_no_graph']:.3f} ({r['paired_rows_vs_compile_max_no_graph']})"]
            tex.append(" & ".join(cells)+r" \\");zh.append("| "+" | ".join(cells)+" |")
    tex += [r"\bottomrule",r"\end{tabular}",r"\end{table*}",
            "The Neptune TVM/Triton bridge terminated with SIGSEGV during whole-chain CUDA Graph capture. "
            "Those process failures are retained; later captures are explicitly rejected rather than retried. "
            "This is an adapter compatibility boundary and does not invalidate its admitted direct calls.",
            "Per-shape timings, process-bootstrap intervals, candidate admission status, available backend traces, and calibration costs remain in the supplementary analysis archive. "
            "The measured candidate minimum is limited to admitted executable paths and is not a global schedule optimum."]
    zh += ["", "Neptune 的 TVM/Triton 桥接在完整链 CUDA Graph capture 中发生 SIGSEGV，相关进程失败保留在归档中；后续捕获明确拒绝，避免重复崩溃。此结果是当前适配器的兼容性边界，不影响通过校验的直接调用。", "", "逐形状时延、进程 bootstrap 区间、候选接纳状态、可获得的后端轨迹及校准耗时保存在补充分析目录。测得的候选最小值仅涵盖通过验证的执行路径，不代表全局最优 schedule。"]
    if any(r["suite"]=="neptune_torch_gpu1" for r in operators):
        tex.append("The official extracted Torch-callable Triton route requires Python integers for its launch configuration; the retained patch converts the TVM integer values without changing their values or schedules. The initial failed series is preserved separately. The admitted Torch-bridge cases exclude multi-kernel decode composition and GQA.")
        zh += ["", "Neptune 官方导出的 Torch 可调用 Triton 接口要求启动配置使用 Python 整数；归档补丁将 TVM 整数转换为相同数值的 Python 整数，不改变参数值或 schedule。最初失败序列另行保留；Torch 桥接的接纳样本不包括多内核 decode 组合和 GQA。"]
    compiler=[r for r in operators if r['suite']=='torch214' and r['graph']=='compiler'
              and r['method'] in ('compile_reduce_overhead','compile_max_autotune')]
    if compiler:
        tex += [r"\paragraph{Framework-managed graphs}",
                "The compiler-managed group retains its own direct default baseline. It measures reduce-overhead and max-autotune with the framework's internal graphs and ordinary, unmarked tensor arguments. Profiler traces contain input-copy kernels. Controlled manual replay instead captures fixed storage, and model parameters have their own framework treatment; these protocols are reported separately.",
                r"\begin{table}[t]",r"\centering\small",
                r"\caption{PyTorch 2.14.1 framework-managed graph modes relative to default compilation in the same process. Only admitted pairs enter each ratio.}",
                r"\label{tab:compiler-graphs}",r"\begin{tabular}{@{}llrr@{}}",r"\toprule",
                r"Workload & Mode & Pairs & / default \\",r"\midrule"]
        zh += ["", "**框架自管 Graph。** 本组使用自身的直接默认编译基线，对普通、未标记静态地址的张量参数测量 reduce-overhead 与 max-autotune 内部 Graph；profiler 轨迹包含输入复制内核。受控手动 replay 捕获固定存储，模型参数则由框架另行处理，各协议分开报告。", "",
               "| 负载 | 模式 | 配对数 | / default |", "| --- | --- | ---: | ---: |"]
        for r in sorted(compiler,key=lambda r:(r['op'],r['method'])):
            label='reduce-overhead' if r['method']=='compile_reduce_overhead' else 'max-autotune'
            cells=[r['op'],label,str(r['paired_rows_vs_compile_default']),f"{r['geomean_speedup_vs_compile_default']:.3f}"]
            tex.append(' & '.join(cells)+r" \\");zh.append('| '+' | '.join(cells)+' |')
        tex += [r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    fa2=[r for r in operators if r['suite']=='torch251' and r['method']=='fa2_compiled_norm']
    if fa2:
        tex += [r"\paragraph{FlashAttention-2}",
                "FlashAttention-2 with the same compiled FP32 normalization epilogue is available in the PyTorch 2.5.1 GPU-1 environment. Its admitted coverage is reported explicitly; the CUDA extension is ABI-specific and is not reused in the independent PyTorch 2.14.1 environment.",
                r"\begin{table}[t]",r"\centering\small",
                r"\caption{FlashAttention-2 plus compiled RMSNorm relative to the compiled SDPA chain in the same process.}",
                r"\label{tab:fa2-supplement}",r"\begin{tabular}{@{}lrrr@{}}",r"\toprule",
                r"Timing & Shapes & Pairs & / default \\",r"\midrule"]
        zh += ["", "**FlashAttention-2。** 在 PyTorch 2.5.1 的 GPU-1 环境测量 FA2 与同一 FP32 编译归一化后处理。接纳覆盖明确列出；CUDA 扩展依赖 PyTorch ABI，没有跨用到独立的 2.14.1 环境。", "",
               "| 计时 | 形状数 | 配对数 | / compiled-SDPA |", "| --- | ---: | ---: | ---: |"]
        for r in sorted(fa2,key=lambda r:r['graph']):
            cells=['Direct' if r['graph']=='none' else 'Graph',str(r['cases']),str(r['paired_rows_vs_compile_default']),f"{r['geomean_speedup_vs_compile_default']:.3f}"]
            tex.append(' & '.join(cells)+r" \\");zh.append('| '+' | '.join(cells)+' |')
        tex += [r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    (paper/"supplement_results.tex").write_text("\n\n".join(tex)+"\n",encoding="utf-8")
    target=paper/"jsa_main_zh.md";source=target.read_text(encoding="utf-8")
    begin="<!-- SUPPLEMENT_RESULTS_START -->";end="<!-- SUPPLEMENT_RESULTS_END -->"
    left,rest=source.split(begin,1);_,right=rest.split(end,1)
    target.write_text(left+begin+"\n\n"+"\n".join(zh)+"\n\n"+end+right,encoding="utf-8")


if __name__=="__main__": main()

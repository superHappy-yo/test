#!/usr/bin/env python3
"""Publish only a complete, numerically admitted adapter ablation matrix."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT/"paper/data/jsa_ablation"
PAPER = ROOT/"paper/jsa"


def main():
    evidence = json.loads((DATA/"analysis/summary.json").read_text())
    if evidence["rows"] != 588 or evidence["failed"]:
        raise RuntimeError("Manuscript requires all 588 admitted rows (24 model and 6 projection processes)")
    rows = evidence["summaries"]
    def find(policy, phase, baseline="original"):
        row = next(r for r in rows if (r["stage"],r["baseline"],r["phase"],r["policy"],r["case"]) ==
                   ("system",baseline,phase,policy,"all"))
        if row["pairs"] != 24 or row["clusters"] != 6:
            raise RuntimeError("Incomplete paired model matrix")
        return row
    order = [("fixed_compile","Fixed compiled, two lookups","固定编译，两次查找"),
             ("fixed_compile_single","Fixed compiled, one lookup","固定编译，一次查找"),
             ("fixed_compile_bound","Fixed compiled, prebound","固定编译，预绑定"),
             ("inline_strict","Inline strict, static","严格内联，静态方案"),
             ("scic","SCIC, two lookups","SCIC，两次查找"),
             ("scic_bound","SCIC, prebound","SCIC，预绑定")]
    inline = [find("inline_strict", p) for p in ("prefill", "decode64")]
    bound = [find("fixed_compile_bound", p) for p in ("prefill", "decode64")]
    single_gain = [find("fixed_compile_single", p, "fixed_compile")["geomean_host_speedup"] for p in ("prefill", "decode64")]
    single_prefill = find("fixed_compile_single", "prefill", "fixed_compile")
    bound_gain = [find("fixed_compile_bound", p, "fixed_compile")["geomean_host_speedup"] for p in ("prefill", "decode64")]
    scic_bound = [find("scic_bound", p) for p in ("prefill", "decode64")]
    ceiling = next(r for r in evidence["candidate_summary"] if r["case"]=="all")
    if ceiling["pairs"] != 36 or ceiling["clusters"] != 6:
        raise RuntimeError("Incomplete projection candidate matrix")
    cv = next(r for r in evidence["candidate_cv_summary"] if r["case"]=="all")
    if cv["pairs"] != 36 or cv["clusters"] != 6:
        raise RuntimeError("Incomplete held-out ranking check")
    with (DATA/"analysis/profile_counts.csv").open() as handle:
        profiles = [r for r in csv.DictReader(handle) if r["stage"] == "system"]
    if len(profiles) != 112:
        raise RuntimeError("Incomplete untimed profiler matrix")
    matched = 0
    for row in profiles:
        if row["policy"] == "original":
            continue
        ref = next(r for r in profiles if r["policy"] == "original" and
                   (r["process"],r["case"],r["phase"]) == (row["process"],row["case"],row["phase"]))
        matched += row["cuda_events"] == ref["cuda_events"]
    if matched != 96:
        raise RuntimeError("Review changed CUDA event counts before using the structure interpretation")
    for row in profiles:
        expected = (1 if row["phase"] == "prefill" else 64) * (1 if row["policy"] in ("original","inline_strict") else 7)
        if int(row["compiled_regions"]) != expected or int(row["dynamo_lookups"]) != expected:
            raise RuntimeError("Review unexpected profiler region/lookup counts")
    effects = [json.loads((DATA/f"diagnostic_profiler_effect/gpu{g}/profiler_effect.json").read_text()) for g in (0,1)]
    if any(r["model_status"] for r in effects):
        raise RuntimeError("Profiler intervention failed model validation")
    effect_ratios = [r["geomean_host_speed_ratio_after_vs_before"] for r in effects]
    tex = [r"\subsection{Follow-up diagnosis of adapter overhead}",r"\label{sec:adapter-ablation}",
        "The model slowdowns motivate a controlled diagnostic rather than a claim that threshold tuning will restore an advantage. "
        "We repeat all four HF workloads on two A40s, with three independent processes per device/workload, using PyTorch 2.5.1 and Transformers 4.51.3. "
        "The two device streams run concurrently on the same host; comparisons pair policies within a process and workload. "
        "The 95\\% intervals resample six device/repetition clusters over the fixed workload matrix.",
        "Initial trials interleaved profiler sessions and subsequent policy timings within a process. "
        "A separate intervention measures the same warmed original-model decode callable in three batches before and three after a CUDA-profiler session. "
        f"After/before host-time speed ratios are ${effect_ratios[0]:.3f}\\times$ on GPU 0 and ${effect_ratios[1]:.3f}\\times$ on GPU 1, even after the profiler context exits. "
        "These within-process batches are descriptive checks, not independent repetitions. "
        "The initial trials are retained but excluded from every final latency and candidate summary; seed 0 is remeasured in processes that never initialize the profiler. "
        "The apparent inline advantage in those initial trials is therefore not used as acceleration evidence.",
        "All seven arms use static outer compilation with internal CUDA Graphs disabled and precision-cast emulation enabled. "
        "Decode position tensors are allocated before timing. This common configuration differs from the earlier model protocol; its ratios are not pooled with those measurements. "
        "Every arm passes eager and compiled full-logit validation for the prefill and all 64 teacher-forced decode outputs at the unchanged $10^{-2}$ tolerances. "
        "Preparation, calibration, and profiler samples are excluded from the five-warmup/20-sample prefill and two-warmup/20-sample decode measurements; event samples and a separate synchronized host batch are retained.",
        "The archived adapter constructs an exact signature twice per call. One-lookup dispatch reuses the cached entry directly. "
        "Prebinding freezes a separately validated callable for each layer's fixed prefill/decode phase; the caller checks the phase shape and the outer compiler supplies static guards. "
        "The prebound SCIC arm reuses the corresponding legacy pre-compilation SCIC decisions, avoiding a second selection experiment. "
        "These endpoints are fixed-workload specializations, not a general dynamic-shape runtime. "
        "The static inline arm removes the outer projection--add--norm custom operation and nested compiled callable, while retaining the opaque FP16 linear boundary and cast-before-affine semantics. "
        "It contains no adaptive selector or new MemMode/ComputeMode schedule. "
        "The one-lookup implementation is promoted to the general adapter while retaining the explicit two-lookup diagnostic control.",
        r"\begin{table*}[t]",r"\centering\small",
        r"\caption{Follow-up adapter ablation. Each entry is a geometric mean of 24 paired host-time speed ratios. Original denotes the original compiled model under this follow-up protocol. Fixed denotes the two-lookup fixed-compiled adapter. Values below one are slowdowns.}",
        r"\label{tab:adapter-ablation}",r"\begin{tabular}{@{}lrrrr@{}}",r"\toprule",
        r"Execution path & Prefill / original & Decode / original & Prefill / fixed & Decode / fixed \\",r"\midrule"]
    zh = ["### 5.7 接入开销的后续诊断", "",
        "针对完整模型退化，进一步进行受控消融，而不预设调整分类阈值就能恢复优势。使用 PyTorch 2.5.1、Transformers 4.51.3，在两张 A40 上重复四个 HF 负载，每个设备/负载 3 个独立进程。两卡实验流在同一主机上并行运行；策略在同一进程、负载内配对。95% 区间对固定负载矩阵中的 6 个设备/轮次簇进行 bootstrap。", "",
        f"最初试验在同一进程中交替运行 profiler 和后续策略计时。单独的受控检查对同一个已预热原始模型 decode 函数，在一次 CUDA profiler 会话前后各测量 3 个批次；即使 profiler 上下文已经退出，GPU 0、GPU 1 的后/前主机时间速度比仍分别为 {effect_ratios[0]:.3f}×、{effect_ratios[1]:.3f}×。这些进程内批次只作为描述性检查，不计为独立重复。最初试验保留，但从全部最终时延和候选汇总中排除；seed 0 在完全不初始化 profiler 的进程中重新测量。因此，初始试验的内联表面优势不作为加速证据。", "",
        "七个变体统一采用静态外层编译，关闭内部 CUDA Graph，并启用精度 cast 模拟；decode 位置张量在计时前分配。该共同配置与先前模型实验不同，速度比不合并。每个变体的 eager、compiled 输出都通过 prefill 和全部 64 步 teacher-forced decode 的完整 logits 检查，保持相对/绝对容差 1e-2。Prefill 预热 5 次、测量 20 次，decode 循环预热 2 次、测量 20 次；分别保留 CUDA-event 样本及单独同步的主机批次。准备、校准和 profiler 样本不进入常规时延。", "",
        "归档适配器每次调用构造两次精确签名；单次查找直接调用已缓存的执行入口。预绑定为每层的固定 prefill/decode 阶段冻结经过验证的函数，调用方检查阶段形状，外层编译器提供静态 guards。SCIC 预绑定复用同进程旧路径在外层编译前的选择，避免重新选择造成干扰。这是固定负载特化，尚未实现通用动态形状运行时。静态内联路径移除外层投影—残差—归一化自定义算子及嵌套编译函数，同时保留不透明 FP16 线性边界和 affine 前的 cast 语义；该路径不包含自适应选择，也没有新 MemMode/ComputeMode 调度。单次查找实现已接入通用适配器，同时保留显式的两次查找诊断对照。", "",
        "| 执行路径 | Prefill / 原始 | Decode / 原始 | Prefill / 固定 | Decode / 固定 |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for policy, label, label_zh in order:
        values = [find(policy, phase, baseline)["geomean_host_speedup"]
                  for baseline in ("original", "fixed_compile") for phase in ("prefill", "decode64")]
        formatted = [f"{v:.3f}" for v in values]
        tex.append(" & ".join([label,*formatted])+r" \\")
        zh.append("| "+" | ".join([label_zh,*formatted])+" |")
    tex += [r"\bottomrule",r"\end{tabular}",r"\end{table*}",
        f"Relative to the two-lookup fixed adapter, one lookup achieves ${single_gain[0]:.3f}\\times$ for prefill and ${single_gain[1]:.3f}\\times$ for decode; prebinding achieves ${bound_gain[0]:.3f}\\times$ and ${bound_gain[1]:.3f}\\times$. "
        f"The prebound fixed adapter remains at ${bound[0]['geomean_host_speedup']:.3f}\\times$ and ${bound[1]['geomean_host_speedup']:.3f}\\times$ relative to the original model. "
        f"Prebound SCIC reaches ${scic_bound[0]['geomean_host_speedup']:.3f}\\times$ and ${scic_bound[1]['geomean_host_speedup']:.3f}\\times$. "
        "Removing signature lookup alone therefore does not recover the complete-model loss. "
        f"The one-lookup prefill interval relative to fixed is [{single_prefill['ci95_low']:.3f}, {single_prefill['ci95_high']:.3f}]; it includes one and retains an unusually slow host batch, so no stable prefill gain is claimed.",
        f"Static strict inlining reaches ${inline[0]['geomean_host_speedup']:.3f}\\times$ (95\\% interval [{inline[0]['ci95_low']:.3f}, {inline[0]['ci95_high']:.3f}]) for prefill and "
        f"${inline[1]['geomean_host_speedup']:.3f}\\times$ ([{inline[1]['ci95_low']:.3f}, {inline[1]['ci95_high']:.3f}]) for decode relative to the original model. "
        "This result concerns integration of a static compiled path and does not establish an adaptive-selection advantage.",
        r"\begin{figure*}[t]",r"\centering",r"\includegraphics[width=\linewidth]{figures/fig_adapter_ablation.pdf}",
        r"\caption{Adapter diagnostics under a common follow-up protocol. Error bars resample device/repetition clusters; the original compiled model is the unit reference. Strict inlining is static, while the prebound paths specialize fixed prefill/decode signatures.}",r"\label{fig:adapter-ablation}",r"\end{figure*}",
        "Profiler observations are retained in separate diagnostic processes; no process contributing to the final latency matrix starts a profiler. "
        "All 96 variant/workload/phase comparisons have the same CUDA-event count as their original-model reference. "
        "The opaque adapter paths enter seven compiled regions and perform seven Dynamo lookups per prefill, versus one for the original and inline paths; the 64-step decode counts are 448 versus 64. "
        "Prebinding eliminates exact-signature dispatch but retains these nested compiled calls. "
        "The archive retains CUDA-event names/counts and CPU operation counts. These are evidence about execution structure, not DRAM traffic or an isolated causal decomposition of kernel time.",
        "A separate complete projection--residual--RMSNorm diagnostic measures six actual row counts: $M=1,8,128$ and $M=1024,2048,16384$, at width 1024 on the same six device/repetition clusters. "
        "It uses the initialized model's weight scale and unit normalization weights, preserving the $10^{-3}$ component gate. "
        f"Of 36 paired shape/process measurements, native wins {ceiling['native_host_wins']} host batches and {ceiling['native_gpu_wins']} event medians. "
        f"The optimistic per-shape minimum of native and compiled yields ${ceiling['geomean_host_speedup']:.3f}\\times$ host and ${ceiling['geomean_gpu_speedup']:.3f}\\times$ event speed ratios over fixed compilation. "
        "This minimum uses regular samples and is a diagnostic ceiling for this candidate set, not a measured selector or global schedule optimum. "
        "An initial arbitrary-normalization-weight smoke case failed the unchanged component gate; its rejection is retained and supplies no timing evidence for that signature.",
        "A leave-one-seed-out check holds out both devices of each seed and selects each known shape's candidate using only event medians from the other two seeds. "
        f"Its held-out ratios are ${cv['geomean_host_speedup']:.3f}\\times$ for host batches and ${cv['geomean_gpu_speedup']:.3f}\\times$ for events relative to fixed compilation; the best fixed candidate selected on training data is compiled in every fold. "
        "The held-out policy chooses native at $M=1,8,128,1024$ and compiled at $M=2048,16384$. "
        "This is offline evidence of ranking stability on known shapes, excluding dispatcher and calibration costs; it is not measured adaptive execution or unseen-shape generalization. "
        "The default analytical binding chooses compiled without probing memory-regime shapes, so it can miss the observed small-row native advantage. "
        "The crossover involves the strict adapter's existing execution paths and does not establish a transition between concrete resource-oriented kernel schedules.",
        "The follow-up separates recoverable integration overhead from the unestablished value of analytical selection. "
        "The existing strict projection candidates have measurable local selection headroom, but adapter overhead prevents treating it as system acceleration. "
        "The next executable comparison should test all-shape empirical calibration against default SCIC and the best fixed adapter under the same invocation path. "
        "Demonstrating an intensity-guided scheduling advantage additionally requires concrete alternative schedules and evidence of the analytical feature's incremental value over empirical calibration. "
        "The diagnostic does not extend the optimized adapter to vLLM or validate pretrained normalization weights."]
    zh += ["",
        f"相对两次查找的固定适配器，单次查找的 prefill/decode 速度比为 {single_gain[0]:.3f}×、{single_gain[1]:.3f}×；预绑定为 {bound_gain[0]:.3f}×、{bound_gain[1]:.3f}×。预绑定固定方案相对原始模型仍为 {bound[0]['geomean_host_speedup']:.3f}×、{bound[1]['geomean_host_speedup']:.3f}×；SCIC 预绑定为 {scic_bound[0]['geomean_host_speedup']:.3f}×、{scic_bound[1]['geomean_host_speedup']:.3f}×。仅移除签名查找不足以恢复完整模型损失。单次查找相对固定方案的 prefill 区间为 [{single_prefill['ci95_low']:.3f}, {single_prefill['ci95_high']:.3f}]，包含 1，并保留一个异常缓慢的主机批次，因此不声称稳定 prefill 收益。", "",
        f"严格静态内联相对原始模型的 prefill 速度比为 {inline[0]['geomean_host_speedup']:.3f}×（95% 区间 [{inline[0]['ci95_low']:.3f}, {inline[0]['ci95_high']:.3f}]），decode 为 {inline[1]['geomean_host_speedup']:.3f}×（[{inline[1]['ci95_low']:.3f}, {inline[1]['ci95_high']:.3f}]）。该结果属于静态编译路径的接入优化，不能作为自适应选择优势。", "",
        "![接入开销消融](figures/fig_adapter_ablation.png)", "",
        "图：共同后续协议下的接入消融。误差条对设备/轮次簇重采样；原始编译模型为单位参考。严格内联是静态方案，预绑定特化固定 prefill/decode 签名。", "",
        "Profiler 轨迹来自单独归档的诊断进程；最终时延矩阵的进程不启动 profiler。96 个变体/负载/阶段配对的 CUDA 事件数均与对应原始模型一致；不透明适配器每次 prefill 进入 7 个 compiled region、执行 7 次 Dynamo 查找，原始与内联路径为 1 次，64 步 decode 分别为 448 次与 64 次。预绑定移除了精确签名调度，但保留嵌套编译调用。归档保留 CUDA 事件名称/次数与 CPU 操作次数；这些信息描述执行结构，不是 DRAM 流量，也不提供内核时间的独立因果分解。", "",
        f"另对实际六种行数 M∈{{1,8,128,1024,2048,16384}}、宽度 1024 的完整投影—残差—RMSNorm 进行诊断，共 6 个设备/轮次簇。采用与初始化模型一致的权重幅值及单位归一化权重，组件容差保持 1e-3。36 个形状/进程配对中，native 在 {ceiling['native_host_wins']} 个主机批次、{ceiling['native_gpu_wins']} 个 event 中位数上获胜；逐形状取 native/compiled 最小值，相对固定编译的乐观速度比为主机 {ceiling['geomean_host_speedup']:.3f}×、event {ceiling['geomean_gpu_speedup']:.3f}×。该最小值使用常规样本，是当前候选集的诊断上限，不是实际选择器性能或全局调度最优值。最初使用任意归一化权重的 smoke 样本未通过未改变的组件门槛，拒绝记录保留，该签名不提供性能证据。", "",
        f"进一步留出每个 seed 的两张卡，仅根据另外两个 seed 的 event 中位数为已知形状选择候选。留出样本相对固定编译的速度比为主机 {cv['geomean_host_speedup']:.3f}×、event {cv['geomean_gpu_speedup']:.3f}×；每个折中，训练数据选择的最佳固定候选均为 compiled。该离线策略在 M=1/8/128/1024 选择 native，在 M=2048/16384 选择 compiled。此检查只验证已知形状的排序稳定性，不包含分发和校准成本，也不是实际自适应执行或未见形状泛化。默认解析绑定在访存区间不探测而直接选 compiled，可能遗漏观测到的小行数 native 优势。交叉点涉及严格适配器的现有执行路径，尚未证明具体资源导向内核调度发生转变。", "",
        "后续诊断区分了可以恢复的接入开销与尚未确立的解析选择价值。严格投影候选存在可测的局部选择余量，但接入开销使其不能直接解释为系统加速。下一轮实际执行应在相同调用路径下比较全形状实测校准、默认 SCIC 与最佳固定适配器。要证明强度引导调度优势，还需要具体备选调度，以及解析特征相对纯实测校准的增量价值。这轮诊断未将优化适配器接入 vLLM，也未验证预训练归一化权重。"]
    (PAPER/"adapter_ablation.tex").write_text("\n\n".join(tex)+"\n", encoding="utf-8")
    path = PAPER/"jsa_main_zh.md"
    content = path.read_text(encoding="utf-8")
    start, end = "<!-- ADAPTER_ABLATION_START -->", "<!-- ADAPTER_ABLATION_END -->"
    block = start+"\n\n"+"\n".join(zh)+"\n\n"+end+"\n\n"
    if start in content:
        first, tail = content.split(start, 1)
        _, last = tail.split(end, 1)
        content = first+block+last.lstrip("\n")
    else:
        index = content.index("## 6 ")
        content = content[:index]+block+content[index:]
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()

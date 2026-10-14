#!/usr/bin/env python3
"""Synchronize the serving section and plot from token-validated measurements."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',default='paper/data/jsa_supplement/analysis')
    p.add_argument('--paper',default='paper/jsa');args=p.parse_args()
    paper=Path(args.paper);data=json.loads((Path(args.analysis)/'serving.json').read_text())
    result=next(r for r in data['global_summary'] if r['policy']=='scic')
    if result['clusters']<3 or result['cases']!=6 or data['rejected_pairs']:
        raise RuntimeError('Serving publication requires three complete, token-identical repetitions')
    rows=sorted([r for r in data['summaries'] if r['policy']=='scic'],key=lambda r:(int(r['input_length']),int(r['concurrency'])))
    if any(r['pairs']<3 for r in rows):raise RuntimeError('Incomplete per-case serving coverage')
    ratio=f"{result['geomean_throughput_ratio']:.3f}"
    low=f"{result['ci95_low']:.3f}";high=f"{result['ci95_high']:.3f}"
    gpus=set(','.join(r['gpus'] for r in rows).split(','))
    reports=[]
    for path in sorted(Path(args.analysis).parent.glob('vllm_gpu*/gpu*/scic/projection_report.json')):
        value=json.loads(path.read_text())
        if isinstance(value,list) and value and 'prepared' in value[0]:reports.append(value[0])
    stats=reports[-1] if reports else None
    stats_en=(f"The available final SCIC worker report records {len(stats['prepared'])} prepared layer/signature entries and {stats['calibration_ms']:.3f} ms of calibration; " if stats else "SCIC worker statistics are unavailable; ")+"early statistics RPC failures are retained and are not replaced by inferred values."
    stats_zh=(f"可获得的最终 SCIC worker 记录 {len(stats['prepared'])} 个层/签名准备项、{stats['calibration_ms']:.3f} ms 校准；" if stats else "SCIC worker 统计不可用；")+"早期统计 RPC 失败另行保留，不以推断值替代。"
    tex=[r"\subsection{Actual vLLM serving-engine integration}",r"\label{sec:vllm-system}",
         r"We additionally run the adapter inside vLLM 0.31.0's V1 AsyncLLM engine~\cite{Kwon2023vLLM}. The environment uses the CUDA-12.9 vLLM wheel and PyTorch 2.13.0+cu126 on the existing driver. Matching official torchvision/torchaudio CUDA-12.6 wheels and CPU torchcodec remove incidental CUDA-13 multimedia imports; package checks and the actual engine startup both pass.",
         f"The serving study uses {len(gpus)} A40 device and {result['clusters']} independent process repetitions per policy. GPU 0's sustained occupancy prevented its serving sequence from starting under the idle gate; the two-device operator and PyTorch-model studies remain separate.",
         "The same complete small Qwen3 architecture uses seeded dummy weights, FP16, TP=1, and token-ID inputs. Each policy has its own worker process. All policies retain prefix caching, chunked prefill, the framework's default compilation/graphs, a 4096-token context limit, 16 maximum sequences, and 4096 batched tokens. Prompt lengths are 128 and 2048, concurrency is 1, 8, or 16, and each measured request generates 64 tokens. Two warmup batches precede 128 measured requests per workload. Unique deterministic prefixes prevent cross-request prefix reuse.",
         "The comparison includes the original engine, fixed-native and fixed-compiled adapters, and SCIC. Every candidate passes component checks, and every admitted request's complete generated-token hash matches the original engine at the same seed, workload, and request ID. This verifies generated tokens rather than independently exported full-vocabulary logits. The PyTorch model experiment separately checks full logits.",
         f"Across six workloads and {result['clusters']} paired process repetitions, SCIC's geometric-mean output-throughput ratio is ${ratio}\\times$ relative to the original engine, with a process-bootstrap 95\\% interval of $[{low},{high}]$. This is an aggregate regression under the tested serving configuration; individual workloads differ. The fixed-compiled adapter's corresponding ratio is ${next(r['geomean_throughput_ratio'] for r in data['global_summary'] if r['policy']=='fixed_compile'):.3f}\\times$. Table~\\ref{{tab:vllm-serving}} reports absolute rates and tail TTFT alongside paired ratios.",
         r"\begin{table*}[t]",r"\centering\small",
         r"\caption{Actual vLLM V1 generation, 128 measured requests per process/workload, 64 output tokens per request. Rates are tokens/s and p95 TTFT is in milliseconds; displayed values are medians of per-process values. Ratios and intervals use paired processes. All candidate request token hashes match the original.}",
         r"\label{tab:vllm-serving}",r"\begin{tabular}{@{}rrrrrrr@{}}",r"\toprule",
         r"Input & Conc. & Orig. tok/s & SCIC tok/s & Ratio [95\% CI] & Orig. TTFT$_{95}$ & SCIC TTFT$_{95}$ \\",r"\midrule"]
    zh=["### 5.6 实际 vLLM 服务引擎接入", "",
        "进一步将适配器接入 vLLM 0.31.0 的 V1 AsyncLLM 引擎。环境采用 CUDA 12.9 vLLM 轮包与 PyTorch 2.13.0+cu126，沿用现有驱动；官方匹配的 torchvision/torchaudio CUDA 12.6 轮包与 CPU torchcodec 消除了无关多媒体依赖的 CUDA 13 导入要求，依赖检查和实际引擎启动均通过。", "",
        f"服务实验使用 {len(gpus)} 张 A40，每种策略 {result['clusters']} 个独立进程重复。0 号卡持续占用使该卡服务序列未通过空闲启动门槛；两卡算子和 PyTorch 完整模型实验单独报告。", "",
        "采用同一完整小型 Qwen3 架构、固定种子 dummy 权重、FP16、TP=1 和 token-ID 输入，不下载预训练权重。每种策略独立工作进程；统一保留 prefix caching、chunked prefill、框架默认编译/Graph，最大上下文 4096、最大序列 16、批处理 token 上限 4096。输入长 128/2048，并发 1/8/16，每请求生成 64 token；先预热两批，再测量 128 请求。确定且唯一的随机前缀避免跨请求 prefix reuse。", "",
        "比较原始引擎、固定 native、固定 compile 与 SCIC 适配器。候选通过接入点数值检查后，每个接纳请求的完整生成 token 哈希还与同种子、负载、请求 ID 的原始引擎完全一致。此检查针对生成 token；未单独导出 vLLM 完整词表 logits。PyTorch 模型实验另行检查完整 logits。", "",
        f"六种负载与 {result['clusters']} 个配对进程重复中，SCIC 相对原始引擎的输出吞吐几何平均速度比为 {ratio}×，进程 bootstrap 95% 区间 [{low}, {high}]，在本配置下为总体退化，单个负载结果有差异。固定 compile 适配器为 {next(r['geomean_throughput_ratio'] for r in data['global_summary'] if r['policy']=='fixed_compile'):.3f}×。", "",
        "| 输入长 | 并发 | 原始 tok/s | SCIC tok/s | 速度比 [95% CI] | 原始 TTFT p95 (ms) | SCIC TTFT p95 (ms) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        cells=[r['input_length'],r['concurrency'],f"{r['native_output_tokens_per_second']:.1f}",f"{r['output_tokens_per_second']:.1f}",
               f"{r['geomean_throughput_ratio']:.3f} [{r['ci95_low']:.3f}, {r['ci95_high']:.3f}]",
               f"{r['native_ttft_p95_ms']:.2f}",f"{r['ttft_p95_ms']:.2f}"]
        tex.append(' & '.join(cells)+r" \\");zh.append('| '+' | '.join(cells)+' |')
    tex += [r"\bottomrule",r"\end{tabular}",r"\end{table*}",
            "The client benchmark includes IPC, scheduling, KV-cache management, sampling, and streaming. Aggregate throughput also includes deterministic prompt construction; per-request TTFT starts after construction. Tokenizer and HTTP transport are excluded. ITL uses client-observed arrivals, so coalesced tokens share timestamps. Preparation outside regular measurement and any new-signature work inside measured requests retain their actual placement. "+stats_en,
            r"\begin{figure*}[t]",r"\centering",r"\includegraphics[width=\linewidth]{figures/fig_vllm_serving.pdf}",
            r"\caption{Token-validated vLLM serving throughput relative to the original optimized engine. Error bars resample independent process repetitions. The random-weight architecture and fixed batch arrivals do not establish pretrained task quality or production workload behavior.}",r"\label{fig:vllm-serving}",r"\end{figure*}"]
    zh += ["", "客户端测量包含 IPC、调度、KV cache、采样和流式返回；总吞吐也包含确定性 prompt 构造，而单请求 TTFT 从构造之后计时。排除 tokenizer 与 HTTP 传输。ITL 使用客户端观察到的到达时间，合并返回的 token 共享时间戳。常规计时外的准备与请求中出现的新签名工作保留其实际计时位置。"+stats_zh, "",
           "![vLLM 实测吞吐](figures/fig_vllm_serving.png)", "", "图：token 校验后的实际 vLLM 吞吐相对原始优化引擎的速度比；误差线按独立进程重复 bootstrap。随机权重和固定批次到达不验证预训练质量或生产请求分布。"]
    (paper/'vllm_results.tex').write_text('\n\n'.join(tex)+'\n',encoding='utf-8')
    target=paper/'jsa_main_zh.md';source=target.read_text(encoding='utf-8')
    begin='<!-- VLLM_RESULTS_START -->';end='<!-- VLLM_RESULTS_END -->'
    if begin not in source:
        anchor='<!-- SUPPLEMENT_RESULTS_END -->';source=source.replace(anchor,anchor+'\n\n'+begin+'\n'+end,1)
    left,rest=source.split(begin,1);_,right=rest.split(end,1)
    target.write_text(left+begin+'\n\n'+'\n'.join(zh)+'\n\n'+end+right,encoding='utf-8')
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(7.55,2.5),sharey=True)
    for ax,length in zip(axes,('128','2048')):
        for i,(policy,label,color) in enumerate((('fixed_native','Native adapter','#a6bddb'),('fixed_compile','Static compiled','#74a9cf'),('scic','SCIC','#045a8d'))):
            selected=[next(r for r in data['summaries'] if r['policy']==policy and r['input_length']==length and r['concurrency']==str(c)) for c in (1,8,16)]
            values=np.array([r['geomean_throughput_ratio'] for r in selected]);lower=np.array([r['ci95_low'] for r in selected]);upper=np.array([r['ci95_high'] for r in selected])
            ax.errorbar(np.arange(3)+(i-1)*.18,values,color=color,label=label,fmt='o',markersize=3,
                        yerr=[np.maximum(0,values-lower),np.maximum(0,upper-values)],capsize=2,elinewidth=.6)
        ax.axhline(1,color='black',linestyle='--',linewidth=.7);ax.set_xticks(range(3),['1','8','16'])
        ax.set_xlabel('Concurrent requests');ax.set_title(f'{length} input / 64 output tokens');ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Throughput ratio vs original vLLM');axes[1].legend(fontsize=6)
    fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(paper/'figures'/f'fig_vllm_serving.{ext}',dpi=220)
    plt.close(fig)


if __name__=='__main__':main()

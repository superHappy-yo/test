#!/usr/bin/env python3
"""Render only observed results; incomplete stages are explicitly marked."""
import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
data=ROOT/'paper/data/jsa_v2';paper=ROOT/'paper/jsa'
s=json.loads((data/'summary.json').read_text())

def f(stat,ci=False):
 if stat.get('geomean') is None:return '--'
 val=f"{stat['geomean']:.4f}" if ci else f"{stat['geomean']:.3f}"
 if ci:val+=f" [{stat['ci95'][0]:.4f}, {stat['ci95'][1]:.4f}]"
 return val

tex=[r'\subsection{Concrete candidates and held-out SCIC-v2 evaluation}',r'\label{sec:v2-evaluation}',
 'The revised study is archived separately from all earlier measurements. It uses PyTorch 2.13.0+cu126, Triton 3.7.1, Transformers 5.17.0, and vLLM 0.31.0 on A40 GPUs, with one CPU thread. No primary worker initializes a profiler.',
 'The operator matrix contains 32 geometries: 23 training geometries and nine held-out square-width-2048/8192 geometries. Direct execution and static CUDA Graph replay, and one repeated weight versus six rotated weights, form separate domains. Each candidate has 50 warmups, 200 event samples, and five synchronized host batches of 40 calls; host-batch medians are the primary objective. Selection uses fresh three-warmup probes with five eight-call host batches.',
 'Component validation uses three input seeds, unit and nonunit affine weights, and stress inputs. Both normalized output and residual require relative and absolute tolerances of $10^{-3}$ and relative L2 error at most $10^{-3}$. Numerical rejection and native fallback remain in the artifact. The pool minimum is an oracle over admitted implementations, not evidence that a rejected schedule is usable.']
counts=s['counts']
tex.append('The HF operator series uses six independent processes on two A40 devices. After the other device failed the idle gate, the remaining random-model domains use five processes on physical GPU 1 and pretrained domains use three. These repetitions assess process variation on the available hardware; they are not five independent devices. Registration-failure control trials are archived and excluded; corrected controls are paired only where the same GPU/run and geometry have a completed optimized baseline.')
tex.append(f"Completed primary workers are: {counts.get('operator',0)} HF-semantic operator workers, {counts.get('operator-vllm',0)} vLLM-semantic operator workers, {counts.get('operator-legacy',0)} unmodified legacy-dispatch workers, {counts.get('hf',0)} random-model HF workers, and {counts.get('hf-pretrained',0)} pretrained HF workers. Stages without completed workers have no measured result.")
tex+=[r'\begin{table*}[t]',r'\centering\small',r'\caption{SCIC-v2 held-out comparison. Steady speed is full-calibration latency divided by revised-policy latency; preparation speed includes common compilation/validation and policy probes. Achieving a 30\% reduction requires preparation speed at least $1/0.7$. Domains are not pooled.}',r'\label{tab:v2-selection}',r'\begin{tabular}{llllrrr}',r'\toprule',r'Semantics & Execution & Weight reuse & Policy & Steady & Preparation & AI/no-AI \\',r'\midrule']
for r in s['selection']:
 if not r['coverage']:continue
 tex.append(f"{'HF' if r['domain']=='operator' else 'vLLM'} & {r['mode']} & {r['reuse']} & {'v2' if r['policy']=='scic-v2' else 'no-AI'} & {f(r['steady_vs_full'])} & {f(r['total_preparation_speed'])} & {f(r['ai_vs_no_ai'])} "+r'\\')
tex += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
accepted=sum(r['acceptance'] for r in s['selection']);measured=sum(r['coverage']>0 for r in s['selection'])
tex.append('The operator scan prepares every candidate to audit the pool. Its preparation column therefore accounts for a common exhaustive compilation/validation cost and measures search savings conditional on that protocol. It is not a measurement of cold, lazy deployment preparation. Full-model preparation records separately report the actual candidates compiled, validated, and probed, under the observed compiler-cache state.')
audit_path=data/'prior_audit.json'
if audit_path.exists():
 audit=json.loads(audit_path.read_text())['rows']
 support={name:sorted({r['training_geometries'] for r in audit if r['worker'].endswith('/operator') and r['candidate']==name}) for name in ('split4','split8')}
 tex.append('In the HF-semantic training-only audit, split factors four and eight each retain '+'/'.join(map(str,support['split4']))+' and '+'/'.join(map(str,support['split8']))+' distinct numerically admitted geometries, respectively. Each candidate is fitted separately, so constant candidate-identity features do not supply additional shape information. Large leave-one-geometry-out residual margins often force expanded probing. This is a training-support and model-identifiability limitation, not evidence that arithmetic intensity determines the best schedule. The audit does not tune features or hyperparameters on held-out timings.')
for domain in ('operator','operator-vllm'):
 rows=[r for r in s['selection'] if r['domain']==domain and r['policy']=='scic-v2' and r['coverage']]
 if rows:tex.append(('HF' if domain=='operator' else 'vLLM')+r'-semantic v2/full-calibration steady ratios (95\% process-bootstrap intervals), in direct/same, direct/rotated, graph/same, and graph/rotated order, are '+', '.join(f(r['steady_vs_full'],True) for r in rows)+'.')
if (paper/'figures/fig_v2_selection.pdf').exists():
 tex += [r'\begin{figure*}[t]',r'\centering',r'\includegraphics[width=\textwidth]{figures/fig_v2_selection.pdf}',r'\caption{Held-out steady performance and exhaustive-scan preparation accounting. Error bars are 95\% GPU/run bootstrap intervals. The horizontal thresholds apply separately to each execution/reuse domain.}',r'\label{fig:v2-selection}',r'\end{figure*}']
tex.append(f"Of {measured} measured execution/reuse/semantic/policy domains, {accepted} meet both the 98\% steady-performance requirement and the 30\% total-preparation reduction requirement. Training-only analytical features are assessed against the otherwise identical no-AI policy. Probe reductions alone cannot establish that preparation is cheaper.")
tex += [r'\paragraph{Full-model and serving validation}',
 'The pretrained checkpoint is Qwen/Qwen3-0.6B at revision \\texttt{c1899de289a04d12100db\\allowbreak370d81485cdf75e47ca}, loaded in FP16 for every arm. Its attention output projection is rectangular, with $K=2048$ and $N=1024$, and is an external geometry validation rather than additional fitting data. Random-model HF workloads retain four batch/prompt combinations; pretrained HF uses batch-one/128 and batch-eight/2048. Every prefill and all 64 teacher-forced decode outputs are checked against the optimized original. The elementwise logits tolerance is $10^{-2}$; the additional strict relative-L2 gate is $10^{-3}$. A strict failure has no primary acceleration claim, even when elementwise checks pass and exploratory latency is recorded.',
 'A separate diagnostic finds that compiling the random-weight original model itself can exceed the strict relative-L2 threshold against its eager form. This does not justify changing the threshold after observing candidate results. The artifact retains both the strict verdict and exploratory timing.',
 'Actual vLLM generation uses 128 requests with 64 greedy output tokens each, six random-model workloads and four pretrained workloads. Full request token hashes, TTFT, ITL, and startup/preparation costs are recorded. To preserve vLLM\'s unbounded dynamic token dimension, serving freezes one candidate per layer from a declared 64-decode/one-prefill calibration mixture. This restriction prevents treating serving as validation of arbitrary per-shape dynamic routing. Matching token hashes establishes generation equivalence on these requests, not language-task accuracy.']
tex += [r'\begin{table*}[t]',r'\centering\small',r'\caption{Full-system ratios against the optimized original. HF columns use strictly admitted logits; parenthesized counts give valid paired observations. Generation columns include only matching complete request hashes. Incomplete numerical or request coverage cannot support an aggregate acceleration claim.}',r'\label{tab:v2-systems}',r'\begin{tabular}{lrrrrrr}',r'\toprule',r'Policy & HF rand. prefill & HF rand. decode & HF trained prefill & HF trained decode & vLLM rand. & vLLM trained \\',r'\midrule']
for policy in ('best-fixed','legacy','full-calibration','scic-v2','scic-v2-no-ai'):
 vals=[]
 for domain in ('hf','hf-pretrained'):
  for phase in ('prefill','decode64'):
   r=next((r for r in s['system'] if r['domain']==domain and r['phase']==phase and r['policy']==policy),None)
   vals.append(f(r['strict'])+f" ({r['strict']['n']})" if r else '--')
 for domain in ('serving','serving-pretrained'):
  r=next((r for r in s['serving'] if r['domain']==domain and r['policy']==policy),None)
  vals.append(f(r['throughput'])+f" ({r['throughput']['n']})" if r else '--')
 tex.append(policy.replace('full-calibration','full calibration')+' & '+' & '.join(vals)+' '+r'\\')
tex += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
tex += ['Intervals use 5000 bootstrap resamples of independent GPU/run clusters, after averaging log ratios within each cluster; timer iterations are not independent replicates. A system acceleration claim additionally requires a ratio of at least $1.02\\times$, a 95\\% lower bound above one, and complete numerical/request coverage. Analytic cold traffic is not measured DRAM traffic, and pretrained inference is not a task-accuracy evaluation.']
for domain in ('hf','hf-pretrained'):
 for policy in ('best-fixed','scic-v2','scic-v2-no-ai','full-calibration'):
  deployments=[r for r in s.get('deployment',[]) if r['domain']==domain and r['policy']==policy]
  all_native=sum(r['native']==r['endpoints'] for r in deployments)
  strict_native=sum(r['strict'] and r['native']==r['endpoints'] for r in deployments)
  if all_native:tex.append(f"{domain}/{policy} restores native functional endpoints in all layers/phases for {all_native} of {len(deployments)} measured cases; {strict_native} of these cases pass strict logits. Their timings do not establish a fused-candidate gain.")
 r=next((r for r in s.get('preparation',[]) if r['domain']==domain and r['policy']=='scic-v2'),None)
 if r and r['component_preparation_speed']['n']:tex.append(('Random-model' if domain=='hf' else 'Pretrained')+' HF actual component-preparation speed relative to full calibration is '+f(r['component_preparation_speed'],True)+f"; {r['strict_valid_pairs']} paired cases also pass strict full-model logits in both arms. This descriptive preparation ratio does not establish valid system acceleration, and the observed compiler-cache state does not represent cold startup.")
for domain in ('serving','serving-pretrained'):
 r=next((r for r in s['serving'] if r['domain']==domain and r['policy']=='scic-v2'),None)
 if r and r['throughput']['n']:tex.append(('Random-model' if domain=='serving' else 'Pretrained')+' vLLM v2 throughput/original is '+f(r['throughput'],True)+f" over {r['throughput']['n']} matched paired workloads. "+('It meets the prespecified generation-speed criterion.' if r['generation_speed_claim'] else 'It does not meet the prespecified generation-speed criterion.'))
if (paper/'figures/fig_v2_serving.pdf').exists():
 tex += [r'\begin{figure*}[t]',r'\centering',r'\includegraphics[width=\textwidth]{figures/fig_v2_serving.pdf}',r'\caption{Actual vLLM output throughput relative to its original engine. Only complete matching request-token hashes contribute; an aggregate acceleration claim additionally requires complete workload coverage. Error bars are 95\% process-bootstrap intervals.}',r'\label{fig:v2-serving}',r'\end{figure*}']
rendered='\n\n'.join(tex)+'\n'
rendered=re.sub(r'\\begin\{table\*\}.*?\\end\{table\*\}',lambda m:m.group().replace('\n\n','\n'),rendered,flags=re.S)
rendered=re.sub(r'\\begin\{figure\*\}.*?\\end\{figure\*\}',lambda m:m.group().replace('\n\n','\n'),rendered,flags=re.S)
(paper/'v2_evaluation.tex').write_text(rendered,encoding='utf-8')
p=paper/'main.tex';text=p.read_text();text=text.replace('\\input{supplement_evaluation}', '\\input{supplement_evaluation}\n\n\\input{v2_evaluation}') if '\\input{v2_evaluation}' not in text else text;p.write_text(text,encoding='utf-8')
zh=['<!-- SCIC_V2_RESULTS_START -->','### 5.8 具体内核、留出选择与真实权重验证',
 '新版结果单独存入 `paper/data/jsa_v2/`，不与旧环境数据合并。环境为 PyTorch 2.13.0+cu126、Triton 3.7.1、Transformers 5.17.0、vLLM 0.31.0，A40、FP16、TP=1、CPU 单线程。正式计时进程不初始化 profiler。32 个投影几何中，23 个用于训练，平方宽度 2048/8192 的九个几何用于留出测试；直接执行/CUDA Graph、单权重重复/六权重轮换分别统计。每候选预热 50 次，采集 200 个事件样本和五个 40 调用同步主机批；选择使用独立的三次预热、五个八调用主机批。',
 '组件对三组随机输入、单位/非单位归一化权重和压力输入检查两个完整输出，要求 `atol=rtol=1e-3` 且相对 L2≤`1e-3`。保留候选拒绝及原生回退，不能用候选池的乐观最小值代替真实可用方法。',
 f"已完成正式进程：HF 算子 {counts.get('operator',0)}、vLLM 语义算子 {counts.get('operator-vllm',0)}、未改动旧 SCIC {counts.get('operator-legacy',0)}、随机 HF 模型 {counts.get('hf',0)}、真实权重 HF 模型 {counts.get('hf-pretrained',0)}；未完成阶段没有实测结论。",'| 语义 | 执行 | 权重 | 策略 | 相对完整校准的稳态比 | 总准备速度比 | AI/no-AI |','|---|---|---|---|---:|---:|---:|']
for r in s['selection']:
 if r['coverage']:zh.append(f"| {'HF' if r['domain']=='operator' else 'vLLM'} | {r['mode']} | {r['reuse']} | {r['policy']} | {f(r['steady_vs_full'])} | {f(r['total_preparation_speed'])} | {f(r['ai_vs_no_ai'])} |")
zh += [f"{measured} 个已测域中，{accepted} 个同时达到稳态性能不低于完整校准的 98%、总准备成本降低至少 30%。总成本包含编译、验证和探测，不能仅根据探测次数减少认定成功。分析特征的增量价值须相对 no-AI 独立判定。",
 '真实权重为固定 revision `c1899de289a04d12100db370d81485cdf75e47ca` 的 Qwen/Qwen3-0.6B，所有对照均使用 FP16。其投影为 K=2048、N=1024，属于外部几何验证，不能加入主训练集。随机 HF 保留四个工作负载，真实模型使用 B=1/S=128 与 B=8/S=2048。完整 logits 的逐元素容差为 `1e-2`，额外严格相对 L2 门槛为 `1e-3`，检查 prefill 及全部 64 个 teacher-forced decode 输出。严格失败即使通过逐元素容差，也只能保留探索性计时，不能支持主加速结论。独立诊断中随机权重原模型从 eager 转为编译也可能超过严格相对误差门槛，这不是看过结果后调整门槛的理由。',
 '实际 vLLM 使用每工作负载 128 个请求、每请求 64 个贪心生成 token，随机模型六个、真实模型四个工作负载，保留完整输出哈希、TTFT、ITL 和准备成本。动态 token 维度不能被 Python 形状条件专门化，因此部署时按 64 次 decode/一次 prefill 的声明负载比例固定每层一个候选；不能据此声称验证了任意形状的动态路由。哈希一致仅说明这些请求生成等价，不代表语言任务准确率。',
 '| 策略 | 随机 HF prefill | 随机 HF decode | 真实 HF prefill | 真实 HF decode | 随机 vLLM | 真实 vLLM |','|---|---:|---:|---:|---:|---:|---:|']
for policy in ('best-fixed','legacy','full-calibration','scic-v2','scic-v2-no-ai'):
 vals=[]
 for domain in ('hf','hf-pretrained'):
  for phase in ('prefill','decode64'):
   r=next((r for r in s['system'] if r['domain']==domain and r['phase']==phase and r['policy']==policy),None);vals.append(f(r['strict'])+f" ({r['strict']['n']})" if r else '--')
 for domain in ('serving','serving-pretrained'):
  r=next((r for r in s['serving'] if r['domain']==domain and r['policy']==policy),None);vals.append(f(r['throughput'])+f" ({r['throughput']['n']})" if r else '--')
 zh.append('| '+policy+' | '+' | '.join(vals)+' |')
zh.append('HF 算子采用两张 A40 上六个独立进程；另一张卡未通过空闲门槛后，其余随机模型域在物理 GPU 1 上重复五次，真实权重域重复三次。这衡量进程变化，并非五张独立设备。旧控制的注册失败运行保留并排除，修正后仅与相同 GPU/轮次/几何的优化基线配对。算子扫描已准备全部候选，表中准备成本属于包含共同编译/验证开销的扫描协议，不能直接推断冷启动的按需准备成本。模型工件另行记录实际准备成本，且保留编译缓存状态的限制。')
if audit_path.exists():zh.append('HF 语义下，split-K=4 与 8 分别只有 '+'/'.join(map(str,support['split4']))+'、'+'/'.join(map(str,support['split8']))+' 个不同的数值合格训练几何。每候选单独拟合，恒定的候选身份特征不会提供额外形状信息；较大的留一几何残差经常迫使扩大探测。这反映训练覆盖与模型可识别性的限制，不能证明算术强度能决定最优调度；审计不使用留出时延调整特征或超参数。')
for domain in ('hf','hf-pretrained'):
 for policy in ('best-fixed','scic-v2','scic-v2-no-ai','full-calibration'):
  deployments=[r for r in s.get('deployment',[]) if r['domain']==domain and r['policy']==policy]
  all_native=sum(r['native']==r['endpoints'] for r in deployments)
  strict_native=sum(r['strict'] and r['native']==r['endpoints'] for r in deployments)
  if all_native:zh.append(f"{domain}/{policy} 的 {len(deployments)} 个实测工作负载中，{all_native} 个在所有层/阶段回退到原生函数端点，其中 {strict_native} 个通过严格 logits；这些计时不能证明融合候选收益。")
 r=next((r for r in s.get('preparation',[]) if r['domain']==domain and r['policy']=='scic-v2'),None)
 if r and r['component_preparation_speed']['n']:zh.append(('随机模型' if domain=='hf' else '真实权重')+' HF 的实际组件准备速度比为 '+f(r['component_preparation_speed'],True)+f"，其中 {r['strict_valid_pairs']} 个配对工作负载的两种策略均通过严格完整 logits。该描述性比值不能证明有效系统加速，也不能代表冷缓存启动。")
for domain in ('operator','operator-vllm'):
 rows=[r for r in s['selection'] if r['domain']==domain and r['policy']=='scic-v2' and r['coverage']]
 if rows:zh.append(('HF' if domain=='operator' else 'vLLM')+' 语义下，按 direct/same、direct/rotated、graph/same、graph/rotated 排列的 v2/完整校准稳态比及 95% 进程区间为：'+'；'.join(f(r['steady_vs_full'],True) for r in rows)+'。')
for domain in ('serving','serving-pretrained'):
 r=next((r for r in s['serving'] if r['domain']==domain and r['policy']=='scic-v2'),None)
 if r and r['throughput']['n']:zh.append(('随机模型' if domain=='serving' else '真实权重')+' vLLM 中，v2/原引擎吞吐比为 '+f(r['throughput'],True)+f"，共有 {r['throughput']['n']} 个完整哈希匹配配对工作负载，"+('达到预设生成速度标准。' if r['generation_speed_claim'] else '未达到预设生成速度标准。'))
zh += ['括号为严格数值或完整生成哈希通过的配对观测数。按 GPU/独立进程聚类进行 5000 次 bootstrap，先在进程内平均对数比值，不把单次计时当作独立重复。系统主加速结论还要求至少 1.02×、95% 区间下界大于一、完整数值/请求覆盖。理论冷流量不等于实测 DRAM 流量，真实权重推理也不等于任务准确率评估。','<!-- SCIC_V2_RESULTS_END -->']
p=paper/'jsa_main_zh.md';text=p.read_text(encoding='utf-8');block='\n\n'.join(zh)+'\n\n'
block=re.sub(r'(\|[^\n]+\|)\n\n(?=\|)',r'\1\n',block)
if '<!-- SCIC_V2_RESULTS_START -->' in text:text=re.sub(r'<!-- SCIC_V2_RESULTS_START -->.*?<!-- SCIC_V2_RESULTS_END -->\s*',lambda _:block,text,flags=re.S)
else:text=text.replace('## 6 相关工作',block+'## 6 相关工作')
p.write_text(text,encoding='utf-8')

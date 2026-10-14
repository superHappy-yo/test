#!/usr/bin/env python3
"""Synchronize conclusions and validate/package measured manuscript artifacts."""
import hashlib,json,re,shutil,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];paper=ROOT/'paper/jsa';data=ROOT/'paper/data/jsa_v2'
s=json.loads((data/'summary.json').read_text())
for stage,minimum in [('operator',6),('operator-vllm',5),('operator-legacy',5),('hf',5),('hf-pretrained',3)]:
 assert s['counts'].get(stage,0)>=minimum,f'Unfinished primary stage: {stage}'
for stage,minimum in [('serving',5),('serving-pretrained',3)]:
 references=list(data.glob(f'gpu1_run*/{stage}/original/completed.json'))
 groups=list(data.glob(f'gpu1_run*/{stage}/completed.json'))
 assert len(references)>=minimum and len(groups)>=minimum,f'Unfinished reference or policy matrix: {stage}'
selection=[r for r in s['selection'] if r['coverage']]
accepted=sum(r['acceptance'] for r in selection)
hf_claims=[r for r in s['system'] if r['acceleration_claim']]
serving_claims=[r for r in s['serving'] if r['generation_speed_claim']]
hf_strict=all(r['complete_numerical_coverage'] for r in s['system'])
if serving_claims:
 serving_sentence='Generation-speed gains occur in '+', '.join(r['domain']+'/'+r['policy'] for r in serving_claims)+', with complete matching request-token hashes under the tested workloads.'
 serving_zh='在所测工作负载中，'+ '、'.join(r['domain']+'/'+r['policy'] for r in serving_claims)+' 达到生成速度标准，且完整请求 token 哈希一致。'
else:
 serving_sentence='No tested serving policy meets the prespecified aggregate generation-speed criterion.'
 serving_zh='没有被测服务策略达到预设的总体生成速度标准。'
selection_sentence=f'Across {len(selection)} held-out execution/reuse/semantic/policy domains, {accepted} satisfy both 98\\% of full-calibration steady performance and a 30\\% reduction in preparation under the exhaustive-scan accounting.'
selection_zh=f'{len(selection)} 个留出执行/权重复用/语义/策略域中，{accepted} 个同时满足稳态达到完整校准的 98% 和扫描协议准备成本降低至少 30%。'
abstract='''Operator fusion performance depends on shape, numerical semantics, and execution overhead.
We present a shape-conditioned intensity classifier (SCIC) and dual-plan interface, then test a revised candidate-cost prior rather than treating Roofline regimes as implementation rankings.
The revision uses five concrete Triton or cuBLAS/Triton projection--residual--RMSNorm schedules, training-only latency fitting, empirical uncertainty, correctness-gated probes, and preparation before outer compilation or CUDA Graph capture.
Its 32-geometry operator matrix separates held-out shapes, weight reuse, direct execution, and graph replay.
'''+selection_sentence+'''
A matched no-AI ablation isolates the analytical features; sparse numerically admitted training support limits prediction reliability.
Complete inference tests include a six-layer random-weight Qwen3 architecture and a pinned pretrained Qwen3-0.6B checkpoint, plus actual vLLM generation with complete request-token comparisons.
'''+('Strict full-model logits validation is incomplete; exploratory timings cannot establish valid model acceleration.\n' if not hf_strict else '')+serving_sentence+'''
The study provides an auditable planning interface and concrete negative and conditional evidence, rather than inferring system gains from eager-relative subgraph timings.
Measured DRAM intensity and language-task accuracy remain outside its scope.'''
p=paper/'main.tex';text=p.read_text(encoding='utf-8')
text=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda _: '\\begin{abstract}\n'+abstract+'\n\\end{abstract}',text,flags=re.S)
conclusion='''SCIC expresses shape-dependent resource recommendations, but a Roofline regime does not establish which executable schedule minimizes latency.
Earlier stock-path experiments improve over eager while remaining slower than optimized complete-model and serving baselines.
The concrete SCIC-v2 revision removes hard intensity-band exclusions, makes the candidates visible to outer compilation, and records actual preparation before deployment.
'''+selection_sentence+'''
The exhaustive operator scan does not measure cold lazy startup, so actual full-model preparation is reported separately with its observed compiler-cache state.
The training-only audit exposes limited valid split-K geometry support and large empirical prediction errors; expanding probes protects candidate coverage but weakens the intended search-cost savings.
'''+('Strict full-model logits failures remain excluded from primary acceleration claims.\n' if not hf_strict else '')+serving_sentence+'''
Any serving gain must be attributed through the fixed-candidate and no-AI controls before it is credited to the analytical prior.
The current evidence supports a bounded preparation interface and a testable cost-feature hypothesis, with neither realized resource classification nor task-accuracy generalization established.
Further improvements require broader numerically valid training coverage, schedule/resource features, and validation on independent production workloads.'''
text=re.sub(r'(\\section\{Conclusion\}\n\\label\{sec:conclusion\}\n).*?(?=\\section\*\{Data availability\})',lambda m:m[1]+'\n'+conclusion+'\n\n',text,flags=re.S)
p.write_text(text,encoding='utf-8')
zh_abstract='''算子融合性能依赖形状、数值语义及执行开销。本文提出形状条件算术强度分类器 SCIC 与双计划接口，并将修订方法定位为候选成本先验，而不是把 Roofline 资源区间直接作为实现排序。修订包含五种 Triton 或 cuBLAS/Triton 投影—残差—RMSNorm 调度，仅用训练几何拟合时延，结合经验不确定性和正确性筛选，在外层编译或 CUDA Graph 捕获前完成准备。32 个算子几何分别评估留出形状、权重复用、直接执行和图重放。'''+selection_zh+''' 匹配的 no-AI 消融检验解析特征的增量价值；数值合格训练几何不足限制了预测可靠性。完整推理包含六层随机权重 Qwen3 与固定 revision 的预训练 Qwen3-0.6B，并接入实际 vLLM 生成和完整请求 token 对比。'''+(' 严格完整 logits 验证存在未通过项，探索性计时不能证明有效模型加速。' if not hf_strict else '')+serving_zh+''' 本文提供可审计的规划接口及具体否定性或条件性证据，不从相对 eager 的子图时延推断系统加速。实际 DRAM 强度及语言任务准确率不在测量范围内。'''
zh_conclusion='''SCIC 表达形状相关的资源建议，但 Roofline 区间不能证明哪个具体调度的时延最小。早期现有路径实验相对 eager 有收益，在优化后的完整模型和服务基线面前仍出现退化。具体 SCIC-v2 修订去除了强度区间的硬排除，使候选对外层编译可见，并记录部署前的实际准备。'''+selection_zh+''' 全量算子扫描不等于冷启动的按需准备，因此另行报告完整模型的实际准备成本及编译缓存限制。仅使用训练数据的审计发现合格 split-K 几何不足、经验预测误差较大；扩大探测保护候选覆盖，却削弱了降低搜索成本的目标。'''+(' 严格完整 logits 失败继续从主加速结论中排除。' if not hf_strict else '')+serving_zh+''' 任何服务收益都需通过固定候选与 no-AI 对照归因，才能认定为解析先验的贡献。当前证据支持有边界的准备接口及可检验的成本特征假设，尚未证明实际资源分类或任务准确率的可推广性。进一步改进应扩大数值合格的训练覆盖，引入调度与资源特征，并在独立生产负载上验证。'''
p=paper/'jsa_main_zh.md';text=p.read_text(encoding='utf-8')
text=re.sub(r'(?<=## 摘要\n).*?(?=## 关键词)',lambda _:'\n'+zh_abstract+'\n\n',text,flags=re.S)
text=re.sub(r'(?<=## 8 结论\n).*?(?=## 数据可用性)',lambda _:'\n'+zh_conclusion+'\n\n',text,flags=re.S)
p.write_text(text,encoding='utf-8')
# Validate the original equations byte-for-byte and all source UTF-8.
frozen=json.loads((data/'provenance/original_equations.json').read_text())
current=re.findall(r'\\begin\{equation\}.*?\\end\{equation\}',(paper/'main.tex').read_text(),re.S)
assert current==frozen,'Original displayed equations changed'
for path in [*paper.glob('*.tex'),paper/'jsa_main_zh.md']:
 assert '\ufffd' not in path.read_text(encoding='utf-8'),str(path)
inputs=re.findall(r'\\input\{([^}]+)\}',(paper/'main.tex').read_text())
assert all((paper/(v+'.tex')).exists() for v in inputs)
snapshot=data/'provenance/final_measured_sources';snapshot.mkdir(exist_ok=True)
hashes={}
for source in [*ROOT.glob('shapefuse/v2_*.py'),*paper.glob('scripts/*jsa_v2*.py')]:
 relative=str(source.relative_to(ROOT));hashes[relative]=hashlib.sha256(source.read_bytes()).hexdigest()
 target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
(data/'provenance/final_source_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
(data/'provenance/manuscript_validation.json').write_text(json.dumps(dict(original_equations_unchanged=True,equation_count=len(current),utf8_valid=True,latex_inputs_present=True,selection_domains=len(selection),accepted_selection_domains=accepted,hf_claims=hf_claims,serving_claims=serving_claims),indent=2)+'\n')
print('Validated manuscript source; equations',len(current),'accepted selection domains',accepted)

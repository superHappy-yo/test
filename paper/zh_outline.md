# 中文提纲：按 shape 分类算子强度，并做双计划融合

对应英文草稿 `paper/draft.md`。这是提纲，不是译文。数字与 `shapefuse` 的闭式模型一致；凡是要 GPU 实测的地方写成 TBD，不要用数据手册或 CPU 代理填上。

## 题目（暂定）

按 shape 的强度分类，服务于多态算子融合。
英文：Shape-Conditioned Intensity Classification for Plan-Polymorphic Operator Fusion。

一句话：算子强度随 shape 越过 Roofline ridge；融合必须能在 MemMode 与 ComputeMode 之间切换。

## 0. 投稿定位与边界

- 目标 venue：MLSys 或 ASPLOS。系统论文的写法：有机器模型、有决策算法、有消融计划。当前稿只有前两样的解析版本。
- 相对 ComFuse：他们做 MatMul + epilogue 的联合融合。我们加的是 SCIC（编译期可查询的强度分类）和双计划切换。
- **明确不写进贡献：** Stage-Stream。它是对方用来承载联合调度的机制。本文不实现、不改写、不宣称。
- **也不写进贡献：** 新的 Attention 算法。IO-aware Attention（FlashAttention 一类）是已有计划，不是本文的结果。
- 本文 artifact 停在决策表。没有融合 kernel，没有 tokens/s。

## 1. 问题

服务栈里同一个 GEMM / Attention / MoE 会遇到差别极大的 shape：

- decode：M 或 \(S_q\) 很小；
- prefill：M 或 S 到几千；
- MoE：同一层里专家的 token 数可以差两个数量级。

「这个算子是计算密集的」这种写法会把融合决策冻住。要论证的是：强度是 (代数, shape, dtype, 哪些中间结果落 HBM) 的函数。

## 2. 背景（Roofline）

- \(\mathrm{AI}=\mathrm{FLOPs}/\mathrm{bytes}\)，\(R=\mathrm{峰值FLOP/s}/\mathrm{峰值byte/s}\)。
- 数据手册锚点（不是实测，论文里必须加字）：
  - H100 SXM FP16：989 TFLOP/s，HBM3 3.35 TB/s，R ≈ 295.2
  - H100 SXM FP8：1979 TFLOP/s，同一 HBM，R ≈ 590.7
  - A100 80GB SXM FP16：312 TFLOP/s，HBM2e 2.039 TB/s，R ≈ 153.0
- 不用稀疏（2×）峰值。
- 三档是策略带，不是物理常数：AI < 0.5R 访存，AI > 2R 计算，中间是 ridge。严格拐点 AI = R 另报，字段 `roofline_limit`。
- 时间模型：\(\max(\mathrm{FLOPs}/P,\ \mathrm{bytes}/B)\)。没有 cache、没有 launch、没有重叠。它是分类特征，不是时延预测。
- **TBD：** 用评估机器上的实测峰值替换 P 和 B。

流量约定：冷 HBM，每个活张量搬一次。GEMM 的 MAC 计 2 FLOP。Softmax 按项目笔记：每元素 5 FLOP、3 次访存。RMSNorm 每元素 4 FLOP。这些常数只固定 O(1) 强度的量级。

## 3. 解析证据（正文主体）

### 3.1 GEMM 穿越 ridge

- FP16、N = K = d 时 \(\mathrm{AI}=2Md/((d+2M)e)\)，\(e=2\)。
- \(M^\star = Rd/(d-2R)\)。d = 4096、H100 FP16：**M\* ≈ 344.9**。
- 同一条曲线的三档边界：M ≈ 159 以下 memory，M ≈ 830 以上 compute。
- A100 的 M\* ≈ 165。
- FP8 把峰值和 AI 一起大约翻倍，M\* 几乎不动（≈ 345.2）。只量化权重、峰值仍按 FP16 时，AI 会升高，但 M = 1 的 W8A8（FP16 写出）AI 仍 ≈ 2，相对 R ≈ 295 还是访存。
- 表（FP16，相对 H100 三档）：

| 场景 | shape | AI | 档 |
| --- | --- | --- | --- |
| decode M=1 | 1×4096×4096 | 1.00 | 访存 |
| M=32 | 32×4096×4096 | 31.5 | 访存 |
| M=256 | 256×4096×4096 | 228 | ridge |
| M=345 | 345×4096×4096 | 295 | ridge（正在过 R） |
| prefill M=2048 | 2048×4096×4096 | 1024 | 计算 |
| K=64 | 4096×4096×64 | 62 | 访存 |
| 专家 16 token | 16×4096×14336 | 16 | 访存 |
| 专家 2048 token | 2048×4096×14336 | 1247 | 计算 |

结论句：线性层不是「计算 kernel」。同一层的 decode 和 prefill 在同一张 GPU 上是两种 regime。

### 3.2 Attention：融合与否决定能不能进入计算侧

- Softmax 单独：AI = 5/6 ≈ 0.83，与 S 无关，永远访存。不该默认成独立 kernel。
- QKᵀ 写回 score：AI 随 S 缓增，在扫描范围内仍低于 H100 的 0.5R。
- FA 式 IO（只搬 Q、K、V、O）：**AI = S/2**（FP16），与 head 维无关。H100 上大约 S ≈ 295 过 0.5R，S ≈ 590 过 R，S ≈ 1181 过 2R。
- 同样 FLOP、但 score 落 HBM：AI 饱和在 \(4D/(3e)\)。D = 128、FP16 时约 **85**，低于 H100 的 0.5R（≈ 148）。**所以在这个模型里，未融合 Attention 在任何序列长度上都是访存。** 融合不是给已经计算受限的 kernel 省一个常数，而是让计算受限这个 regime 变得可达。
- 解析 Roofline 加速（数据手册，不是 kernel 测量）：S = 128 约 1.75×；S = 512 约 4.0×；S = 2048 约 3.75×。S = 1024 以后 FA 侧已经计算受限，未融合侧仍然访存，加速比不再往上走。
- Decode：\(S_q=1\) 时 AI → \(2H_q/(H_{kv} e)\)。MHA FP16 → 1；GQA 4:1 → 4；GQA 8:1 → 8。相对 H100 全是访存，KV 多长都一样。「Attention 是计算密集」只对长 prefill 的 IO-aware 计划成立。

计划上的读法（写进方法，不要写成我们发明了 FA）：

- 两套计划都不把 score 写回 HBM。
- 短序列（如 S = 128，FA 的 AI 仍 < 0.5R）：MemMode，小 tile，可以再融便宜的输出 epilogue。
- 长序列（如 S = 8192，AI > 2R）：ComputeMode，大 tile，不要把输出投影 / residual 再塞进主循环。这就是 ComFuse 在 Self-Attn 上输给 FA 的那个反例，收成一个开关。
- S = 512（AI = 256，落在带内）：两套都发，等实测缓存选。

### 3.3 Norm、归约、逐元素链

RMSNorm、LayerNorm、行归约、Add、Mul、SiLU、GELU、SwiGLU 的逐元素乘，AI 都是 O(1) FLOP/byte。分类恒为访存。选择器只发 MemMode：融进 producer 或融成一条链。没有「把 residual add 拆成 ComputeMode」这种计划。

### 3.4 GEMM+RMSNorm：改正旧笔记里的字节比例

冷缓存下，不融合要写 C、再读 C。省下的字节占未融合流量的比例约为 \(2M/(d+4M)\)，**随 M 增大**：

| M | 省下的字节比例 | 严格 Roofline 加速 | 若每次 launch 假设 5 µs |
| --- | --- | --- | --- |
| 1 | 0.05% | ≈ 1.00× | ≈ 1.33× |
| 256 | 10% | ≈ 1.11× | ≈ 1.38× |
| 2048 | 33% | 1.00× | ≈ 1.07× |

M = 2048 时两边都已经计算受限，再省字节也不动 Roofline 时间。5 µs 是敏感度旋钮，CSV 里标 `ASSUMED_NOT_MEASURED`，**不能写进正式表**，除非换成 H100 实测 launch。

旧笔记「小 M 时 epilogue 往返占比更高」与这个代数相反。小 M 的 GEMM 是访存，是因为权重 KN 占满流量，不是因为 epilogue。epilogue 融合仍然属于 MemMode（launch 次数、一串逐元素），但数量级上改变 regime 的是 score 和专家 shape。勘误见 `notes/draft_report_zh.md`。

**TBD：** H100 上 GEMM+RMSNorm 的 kernel 时间和真实 launch。

### 3.5 Ragged MoE

H = 4096，I = 14336，FP16。不加载空专家时：

| 场景 | 聚合 AI | 最瘦 | 最胖 | 档 |
| --- | --- | --- | --- | --- |
| decode 8×2 | 2.0 | 2.0 | 2.0 | 全访存 |
| 均衡 8×256 | 237 | 237 | 237 | 全 ridge |
| 倾斜 1024…8 | 236 | 8.0 | 775 | 聚合 ridge；最瘦访存；最胖计算（775 > 2R≈590） |
| prefill 8×2048 | 1247 | 1247 | 1247 | 全计算 |
| one-hot 2048，只加载活跃权重 | 1247 | | | 计算 |
| one-hot 2048，八个权重都搬 | 237 | | | 聚合掉回 ridge，与均衡层同流量 |

倾斜行是政策点：一个 epilogue 模板覆盖不了同一层里的两种 regime。one-hot 两行是实现对齐点：空专家的权重搬不搬，会把同一个专家从计算侧拉回 ridge。框架里实际怎么做是 **TBD**。

选择器对每个专家单独分类。倾斜层的主计划集合里同时有 MemMode 和 ComputeMode。

### 3.6 量化 epilogue

权重变窄会抬 AI，不改变选择逻辑。注意两点，表 `quant_epilogue.csv` 里有标记：

- INT8 字节配 FP16 峰值是 what-if，`peak_dtype_matches_operands=false`。
- dequant / 写回本身是 O(1) 强度。MemMode 把它折进 producer；ComputeMode 在主循环已经计算受限时把宽累加器的写回拆出去。

**TBD：** 实测 W8A8 / FP8 峰值后再重跑这张表。

## 4. 方法：SCIC

输入：种类、shape、每个张量的 dtype。输出：FLOPs、bytes、AI、AI/R、三档、严格拐点。来源默认 `analytical`。传入 `measured_ai` 后，档位改按测量值算，解析 AI 留下来做绝对误差。

它回答的不是「图上相不相邻」，而是「这个 shape 在这台机器的哪一侧」。相邻仍是融合的必要条件，不是充分条件。

校准计划（没做）：

- **TBD：** 解析 AI 对 Nsight DRAM 字节算出来的 AI，分类准确率。
- **TBD：** 改用测量 regime 之后，有多少融合决策会翻。
- **TBD：** 冷缓存最失真的站点（权重在 L2、因果掩码、paged KV）。

## 5. 方法：双计划

规则就四条，实现在 `shapefuse/plans.py`，不发 CUDA。

1. 逐元素、归约、Softmax、RMSNorm、LayerNorm：只发 MemMode。
2. memory-bound 只发 MemMode；compute-bound 只发 ComputeMode。
3. ridge 带内两套都发。AI < R 时主计划是 MemMode，否则是 ComputeMode。`measured_winner` 为空。
4. Attention 在两种模式下都不物化 score。ComputeMode 的意思是：保持 FA 式 IO、大 tile、不要再把额外 epilogue 融进主循环。

缓存键：`(kind, dtype, machine, shape 桶)`。桶是 1、2–8、9–32、33–128、129–512、513–2048、2049–8192、8193+。`PlanCache` 可以记下实测赢家，本仓库没有调用它。

运行时本该做、这里没做的事：命中桶之后，只发一套就跑那一套；两套都发就用缓存或做一次短校准。Inductor 的 dynamic shape 有符号维度，但融合决策大体仍是冻住的。缺的输入是 regime，不是再多一个符号维。我们没有改 Inductor。接到编译器上是 **TBD**。

**TBD：** 服务 trace 上的切换开销；相对 oracle 的 gap；Triton / CUTLASS / 编译器 IR 里的 kernel 本体。

## 6. 和联合融合怎么比（正文里用一小段，不要写成已跑完的实验）

| 问题 | 联合融合 | 本文 |
| --- | --- | --- |
| 便宜 epilogue 融不融 | 一套联合调度 | AI > 2R 时 ComputeMode 不融 |
| Attention score 写不写 | 联合调度可能输给 IO-aware kernel | 两套计划都不写；模式只选 tile 和额外 epilogue |
| 每个专家同一模板？ | 常见做法 | 按专家分档，一层里可以异构 |
| Stage-Stream | 对方的机制 | 不用、不声称、不实现 |
| 本文证据 | | 闭式 AI 和决策表；kernel 时间 TBD |

预期真正会分出胜负的，是 ridge 带和 Attention 的额外 epilogue：两边政策不一致的地方。远在访存侧时大家都融；远在计算侧时双计划会拒绝联合融合可能仍会尝试的融合。这个对比**还没跑**。

## 7. 实验计划与 TBD 清单

已经完成、可复现：

```bash
pip install -r requirements.txt && pip install -e .
python -m shapefuse selfcheck
bash scripts/run_cpu.sh          # 无 GPU
bash scripts/run_h100.sh         # 有 CUDA 的服务器
```

`selfcheck` 锁住第 3 节用到的闭式：ridge、M\*、M=1 与 M=2048 的 GEMM AI、Softmax、FA 的 AI=S/2、MoE 聚合、字节比例随 M 增大、短/长 Attention 的计划方向。

H100 脚本计时的是 `torch.matmul`，用来看小 M 吞吐是否下降。`fused_plan_timed=false`，`achieved_ai` 留空。不要把它写成融合实验。

完整实验表（写进论文第 7 节，状态保持诚实）：

| 实验 | 状态 |
| --- | --- |
| 解析 AI、分档、M\*、计划表、图 | 完成 |
| H100 上 matmul 时延 / TFLOP/s | TBD，脚本已有 |
| Nsight DRAM 字节、achieved AI | TBD |
| 机器实测 ridge | TBD |
| SCIC 对实测 AI 的准确率 | TBD |
| MemMode / ComputeMode / 总是融 / 从不融 | TBD，没有 kernel |
| 短 S 与长 S 的 Attention 主循环，加或不加额外 epilogue | TBD |
| MoE 倾斜：按专家计划 vs 单一模板；空专家权重搬不搬 | TBD |
| 量化 epilogue 对实测峰值 | TBD |
| 计划缓存命中率和切换开销 | TBD |
| 端到端 tokens/s | TBD |
| 用实测替换 5 µs launch | TBD |

占位表（正文里就这样空着）：

| shape | H100 中位 ms | TFLOP/s | DRAM 字节 | achieved AI |
| --- | --- | --- | --- | --- |
| 1×4096×4096 | TBD | TBD | TBD | TBD |
| 32×4096×4096 | TBD | TBD | TBD | TBD |
| 344×4096×4096 | TBD | TBD | TBD | TBD |
| 2048×4096×4096 | TBD | TBD | TBD | TBD |
| 4096×4096×4096 | TBD | TBD | TBD | TBD |

Achieved AI = FLOPs / 实测 DRAM 字节，不是 FLOPs / 时延。

## 8. 相关工作（写的时候核对文献）

- Roofline：Williams, Waterman, Patterson, CACM 2009。
- FlashAttention 及其后续：当作可选计划，不当作本文贡献。
- ComFuse：文献条目 **TBD**（作者、年份、venue 投稿前核对）。只使用项目笔记里已经说清楚的两点：联合 MatMul+epilogue；Self-Attn 上输给 FA。Stage-Stream 归他们。
- TorchInductor、Triton、CUTLASS、cuBLAS：已有融合与特化。动态 shape 不等于 regime 一变就改融合。接到其中一家是 TBD。
- TensorRT-LLM 一类的 grouped GEMM：是否按专家强度分桶，投稿前对着实现核对，**TBD**。
- 笔记里点过名、文献未核对：FuseFlow、Neptune、RedFuser。不要编作者和年份。**TBD。**

## 9. 局限（必须留在稿子里）

- 冷 HBM，无 L2，无因果掩码，无 paged KV，无重叠。
- 0.5R / 2R 是旋钮，应该用测量误差分布来定。
- Softmax / RMSNorm 的 FLOP 常数是约定。
- 数据手册峰值不是机柜里那张卡。FP8 的 M\* 几乎不变，依赖于峰值和字节同比例变化。
- 双计划没有寄存器 spill 模型。ComputeMode 拒绝融合，是对照 Self-Attn 反例的政策，不是 spill 代价。AI × spill × SMEM 的联合代价是后续工作，本文不声称已经做了。
- 没有 kernel，没有 trace，没有 tokens/s。

## 10. 结论句（可直接用）

在 H100 这一档的 Roofline 上，宽度 4096 的 FP16 方阵投影大约在 M\* ≈ 345 换边；IO-aware Attention 的强度随 S 换边，把 score 写回 HBM 的实现则不会；倾斜的 MoE 一层之内就同时有两种 regime。按算子种类做死的融合跟不上这些穿越。SCIC 把 regime 从 shape 和流量里算出来，双计划把它变成 MemMode、ComputeMode 或两者都要。Stage-Stream 不属于这个设计。本稿的数字论证是解析的，设备上的论证仍然是 TBD。

## 11. 图表与文件对照

| 论点 | CLI 产物 |
| --- | --- |
| GEMM 与 M\* | `gemm_shapes.csv`，`crossings.json`，`fig_gemm_ai_vs_M.png` |
| Attention | `attention_intensity.csv`，`attention_decode.csv`，`fig_attn_ai_vs_S.png` |
| epilogue 字节比例 | `rmsnorm_epilogue.csv`，`fig_epilogue_tradeoff.png` |
| MoE | `moe_grouped.csv`，`fig_moe_skew_ai.png` |
| 量化 what-if | `quant_epilogue.csv` |
| 逐元素 | `ew_chains.csv` |
| 计划 | `dual_plans.csv`，`fig_plan_regions.png` |
| 设备 matmul | `bench_gemm.csv`，`gpu_measurements.json`（GPU 上仍有空字段） |

## 12. 写作时不要出现的句子

- 「我们提出了 Stage-Stream。」
- 「小 M 时 epilogue 占 GEMM 流量的比例更高。」（冷缓存模型不支持。）
- 「H100 上融合加速了 X 倍。」（在 kernel 测出来之前。）
- 把 `run_cpu.sh` 的 NumPy 时延标成 H100。
- 把 `torch.matmul` 的扫参写成 MemMode 对 ComputeMode 的对比。

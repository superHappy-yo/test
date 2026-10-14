> Historical notes: the original eager FP16 RMSNorm can overflow. Do not use the legacy GEMM ratios below as equivalent-computation speedups. The validated submission sources are [main.tex](jsa/main.tex) and [jsa_main_zh.md](jsa/jsa_main_zh.md); current evidence is in [data/jsa_replication](data/jsa_replication/README.md).

# 中文提纲：按 shape 分类算子强度，并做双计划融合

对应英文草稿 `paper/draft.md`。这是提纲，不是译文。解析数字与 `shapefuse` 的闭式模型一致。已有的设备数字来自第三次 A40 上传，归档在 `paper/data/a40/`（`--machine A40_FP16`，`PARTIAL_VENDOR_MATMUL`）。`TRY_NSIGHT=1` 时 `ncu` 在 PATH 上，但因 `ERR_NVGPUCTRPERM` 失败，`achieved_ai` 仍为空。`proxy_ai` 是 `TORCH_PROFILER_MEMORY_NOT_DRAM`，不能当 SCIC 准确率。`fusion_proxy.csv` 的时间比不是融合加速。copy 带宽是 `ESTIMATE_COPY_NOT_NSIGHT`，不是实测 ridge。Docker 墙钟 bakeoff 有三份归档，数字写在 §7。`paper/data/bakeoff/` 是 Attention epilogue 修复之前的选择器（`BAKEOFF_WALLCLOCK`，Nsight 关闭）。`paper/data/bakeoff_v2/` 是 `compile_sdpa_chain` 之后的第二次运行（忙 GPU：脚本启动时物理 GPU 1 利用率 92%，13490 MiB，`gpu_selection.json` 里 `idle_found` 为 false）。`paper/data/bakeoff_v4/` 是 `--calibrate all-gemm` 的第四次运行（`RUN_NOTES.txt` 记录等到连续空闲样本；启动时 0% 利用率、4 MiB 已分配）。没有的测量保持 TBD。不要用数据手册、CPU 代理、`proxy_ai`、copy 估计，或 CSV 里没有的格子填数字。不要把第一份归档的 `always_fuse` 改标成第二次或第四次的 `scic_selected`。`paper/data/bakeoff_v2/` 里 GEMM 的 ComputeMode 仍选 `always_unfuse`，不要把那份归档改写成已经选了 `always_fuse`。校准后的 GEMM `scic_selected` 以 `paper/data/bakeoff_v4/` 的 CSV 为准：decode 抄 `always_unfuse`，prefill M ∈ {2048, 4096} 抄 `always_fuse`，五条 Attention 都是 `library_sdpa` / `always_fuse`。探针是本机墙钟，不是 DRAM AI，忙 GPU 可能翻赢家，也不是 Nsight SCIC 准确率。

## 题目（暂定）

按 shape 的强度分类，服务于多态算子融合。
英文：Shape-Conditioned Intensity Classification for Plan-Polymorphic Operator Fusion。

一句话：算子强度随 shape 越过 Roofline ridge；融合必须能在 MemMode 与 ComputeMode 之间切换。

## 0. 投稿定位与边界

- 目标 venue：MLSys 或 ASPLOS。系统论文的写法：有机器模型、有决策算法、有消融计划。当前稿只有前两样的解析版本。
- 相对 ComFuse（Mu 等，arXiv:2608.03537）：他们做 MatMul + epilogue 的联合融合。我们加的是 SCIC（编译期可查询的强度分类）、双计划切换、库 SDPA 的 `compile_sdpa_chain` epilogue 策略，以及 bakeoff 里可选的 GEMM 墙钟校准（`--calibrate all-gemm`）。校准后的时延只引 `paper/data/bakeoff_v4/`。
- **明确不写进贡献：** Stage-Stream。它是对方用来承载联合调度的机制。本文不实现、不改写、不宣称。
- **也不写进贡献：** 新的 Attention 算法，也不宣称写了 FlashAttention-3 kernel。IO-aware Attention（FlashAttention 一类）是已有计划。bakeoff 里的 `sdpa` 是 `scaled_dot_product_attention`，不是 FA-3。
- **也不写进贡献：** 把 `torch.profiler`、缓存分配器差分或进程合计的 Nsight 字节写成 SCIC 准确率。`quote_as_scic_accuracy` 只有孤立 Nsight DRAM 才为真。
- 本文 artifact 仍没有 MemMode / ComputeMode kernel，没有 tokens/s，没有 Nsight DRAM。第三次 A40 上传有 `torch.matmul` 时延、一份融合代理和一份 copy 带宽估计（§7），状态 `PARTIAL_VENDOR_MATMUL`，`quote_as_scic_accuracy` 为 false。Nsight 被 `ERR_NVGPUCTRPERM` 挡住。Docker bakeoff 计的是 eager 和 `torch.compile` 的墙钟，不是这两套 kernel。修复前归档是 `paper/data/bakeoff/`。修复后的墙钟是 `paper/data/bakeoff_v2/`，GPU 不空闲。GEMM 的校准路径已经接到 bakeoff：`--calibrate all-gemm` 对每个 MatMul+RMSNorm shape 做短墙钟，`selected_baseline` 在有 `measured_winner` 时跟着它。`paper/data/bakeoff_v4/` 是在空闲 GPU 上跑出的归档；v2 的 CSV 仍是校准之前的。

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
  - A40 FP16：密集 Tensor Core 149.7 TFLOP/s（稀疏 299.4 不用），GDDR6 696 GB/s，R ≈ 215.09
- 不用稀疏（2×）峰值。A40 的 149.7 是 FP16 累加的密集峰值。
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
- A40 数据手册：M(0.5R) = 113.50330752177987，M\* = 240.32584259093773，M(2R) = 544.5531470469126，R = 215.08620689655172。第三次上传的 `summary.json` 和与之相同的 `crossings.json` 用 `--machine A40_FP16`，里面有 `A40_FP16` 这一项。这是数据手册闭式，不是卡上测到的 ridge。
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

§7 那七个 shape 上，`regime_vs_h100_datasheet` 和 `regime_vs_requested_datasheet` 一致（M∈{1,4,32} 访存，M∈{344,512} ridge，M∈{2048,4096} 计算）。严格拐点不一致：M=344 的解析 AI=294.5284281，低于 H100 ridge 295.22，高于 A40 数据手册 ridge 215.08620689655172，所以三档仍是 ridge，严格 Roofline 是计算。两列都不是实测 ridge。第三次上传没有 `decision_log.csv`。`agreement.json` 是 profiler 字节的检查用一致率，不是 SCIC 准确率（§7）。

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

上面是解析计划。Docker bakeoff 没有那条自定义主循环。Attention 主体是库 SDPA，score 已经融合，所以可执行 epilogue 策略是 `compile_sdpa_chain`（§5、§7）。

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

第三次上传带了 `agreement.json`，没有 `decision_log.csv`：`achieved_ai` 为空（`TBD_NSIGHT`），`proxy_ai` 被 `torch.profiler` 填上且七行都是 4096，状态 `TORCH_PROFILER_MEMORY_NOT_DRAM`。`scic_accuracy_status=PROXY_TORCH_MEMORY_NOT_VALID_SCIC_ACCURACY`，`quote_as_scic_accuracy=false`，`n_with_byte_ai=7`，`n_torch_profiler_memory=7`，两条检查用一致率都是 0.2857142857142857（`n_decision_flips_any_byte_source=5`）。上面三条准确率仍然是 TBD。不要把 0.2857142857142857 写成 SCIC 准确率。`quote_as_scic_accuracy` 只有状态 `NSIGHT_DRAM` 才为真。`torch.profiler` 内存、缓存分配器差分和 `NSIGHT_DRAM_PROCESS_SUM` 的一致率不能填进这三条。

## 5. 方法：双计划

规则就四条，实现在 `shapefuse/plans.py`，不发 CUDA。

1. 逐元素、归约、Softmax、RMSNorm、LayerNorm：只发 MemMode。
2. memory-bound 只发 MemMode；compute-bound 只发 ComputeMode。
3. ridge 带内两套都发。AI < R 时主计划是 MemMode，否则是 ComputeMode。解析行的 `measured_winner` 为空。bakeoff 可以随后用墙钟填上 GEMM 的赢家。
4. Attention 在两种模式下都不物化 score。ComputeMode 的意思是：保持 FA 式 IO、大 tile、不要再把额外 epilogue 融进主循环。解析主计划不变：长 prefill 仍是 ComputeMode，`fuse_epilogue` 仍为 false。

库 SDPA 已经把 score 融进 kernel。bakeoff 的 `always_unfuse` 只是把 RMSNorm 拆成第二段计时，不是在保护 MMA 主循环。修复前的 A40 归档里 S = 512 和 S = 2048 因此选慢了。可执行策略改为 `compile_sdpa_chain`，计时为 `always_fuse`，ridge 和 compute 也包括在内。`paper/data/bakeoff_v2/` 是这条策略的墙钟：五条 Attention 决策都抄 `always_fuse`。Attention 不参加墙钟校准。GEMM 在那份归档里 ComputeMode 仍选 `always_unfuse`。同一次运行里，ridge 和计算侧 GEMM 抄到约 1×，而 `always_fuse`（`torch.compile`）更快。这是纯数据手册 AI、没有测量反馈的限制。

校准路径合上这个 GEMM 失误。ridge 和计算侧的 MatMul+RMSNorm 做一次短墙钟（默认 1 次热身、3 次计时），严格更快的一边写入 `measured_winner`。有赢家时 `selected_baseline` 跟着它，否则留在解析基线。`--calibrate all-gemm` 探测每个 GEMM shape，`--calibrate off` 保持数据手册映射。`PlanCache.record_measured_winner` 按 shape 记下赢家。这是本机墙钟，不是 DRAM AI，也不是 Nsight SCIC 准确率。忙 GPU 可能把赢家翻掉；v2 那张卡利用率是 92%。`paper/data/bakeoff_v2/` 是校准之前的 CSV，计算侧仍抄 `always_unfuse`。校准后的决策和 `scic_selected` 时延在 `paper/data/bakeoff_v4/`（只引 CSV 格子）。

缓存键：`(kind, dtype, machine, shape 桶)`。桶是 1、2–8、9–32、33–128、129–512、513–2048、2049–8192、8193+。bakeoff 在键后面加上 shape 标签，避免 M = 256 和 M = 512 互相覆盖。

运行时本该做的事：命中桶之后，只发一套就跑那一套；两套都发就用缓存或做一次短校准。bakeoff 现在对 ridge 和计算侧 GEMM 做这次短校准，并记下墙钟赢家。服务 trace 上的守卫仍然没有。Inductor 的 dynamic shape 有符号维度，但融合决策大体仍是冻住的。缺的输入是 regime，不是再多一个符号维。我们没有改 Inductor。接到编译器上是 **TBD**。

**TBD：** 服务 trace 上的切换开销；相对 oracle 的 gap；Triton / CUTLASS / 编译器 IR 里的 kernel 本体。

## 6. 和联合融合怎么比（正文里用一小段，不要写成已跑完的实验）

| 问题 | 联合融合 | 本文 |
| --- | --- | --- |
| 便宜 epilogue 融不融 | 一套联合调度 | AI > 2R 时 ComputeMode 不融 |
| Attention score 写不写 | 联合调度可能输给 IO-aware kernel | 两套计划都不写；模式只选 tile 和额外 epilogue |
| 每个专家同一模板？ | 常见做法 | 按专家分档，一层里可以异构 |
| Stage-Stream | 对方的机制 | 不用、不声称、不实现 |
| 本文证据 | | 闭式 AI、决策表，第三次 A40 `torch.matmul` 曲线作动机（§7），修复前的 Docker bakeoff（`paper/data/bakeoff/`），`compile_sdpa_chain` 之后的第二次 bakeoff（`paper/data/bakeoff_v2/`），以及空闲 GPU 上 `CALIBRATE=all-gemm` 的第四次 bakeoff（`paper/data/bakeoff_v4/`）。copy 带宽是 `ESTIMATE_COPY_NOT_NSIGHT`。fusion proxy 的时间比不是融合加速。前两次不是空闲 GPU。 |

GEMM 上，访存侧 MemMode 选 `always_fuse`。两份归档里 M = 1、32、128 都比 eager 快。第二次运行里，计算侧和 ridge 上侧仍选 `always_unfuse`（M ∈ {256, 512, 2048, 4096}），贴近 1×，而 `always_fuse` 是 1.234890808× 到 1.327495649×（§7）。第一份归档只在 M = 256 上把这个缺口写清楚。Attention 不同：库 SDPA 已经融合 score。修复前把 ComputeMode 映射成 `always_unfuse`，S = 512 和 S = 2048 更慢。第二次归档打 `compile_sdpa_chain`，五个 Attention shape 都抄 `always_fuse`。两条路径都是 PyTorch（eager 或 `torch.compile`），不是手写的 MemMode / ComputeMode kernel，也不是 Stage-Stream。校准后的 GEMM 路径在短墙钟有赢家时跟着赢家；`paper/data/bakeoff_v4/` 在空闲 GPU 上记录了这一选择（decode 抄 `always_unfuse`，prefill 抄 `always_fuse`）。探针不是 DRAM AI，也不是 Nsight SCIC 准确率。`paper/data/bakeoff_v2/` 仍是未校准的抄录。

## 7. 实验：解析 → vendor matmul → bakeoff（含校准）

叙事三层：（1）§7.1 解析 selfcheck；（2）§7.2 A40 `torch.matmul` 只作 shape 动机；（3）§7.3 Docker 墙钟 bakeoff。没有 Nsight 时能跑的主实验是墙钟 bakeoff，不是 DRAM。Docker 里 Nsight Compute 不可用（`ERR_NVGPUCTRPERM`）。`quote_as_scic_accuracy` 全是 false。空闲 GPU 上校准后的 `scic_selected` **只**引 `paper/data/bakeoff_v4/`；`bakeoff/` 与 `bakeoff_v2/` 是忙 GPU 上的选择器演变，不能改标成 v4 时延。

它在 CUDA 上计两件事：子图时延，以及 SCIC 选了哪条路径。

- **MatMul+RMSNorm**，N = K = 4096，M 取 decode 和 prefill：1、32、128、256、512、2048、4096。和 §7.1 的 vendor 曲线同宽，并补上 ridge 带上的 M，好让计划能切换。
- **Attention + RMSNorm**（脚本默认开）。prefill 的 S ∈ {128, 512, 2048}，decode 的 Sk ∈ {128, 2048}，B = 1，H = 8，D = 64，MHA。FP16 的 FA 式强度是 S/2，头维约掉，档位按被计时的 shape 算。H = 8 是为了物化 score 不要太大，不是新 Attention kernel。

基线写在 `bakeoff_latency.csv`：

| baseline | MatMul+RMSNorm | Attention |
| --- | --- | --- |
| `eager_unfused` | eager 的 matmul 再 RMSNorm | 物化 score，再 RMSNorm |
| `torch_compile_default` | `torch.compile` 这条链 | `torch.compile`（库 SDPA 再 RMSNorm） |
| `torch_compile_max_autotune` | `mode="max-autotune"`，调用成功才有数 | 同一模式，作用在 SDPA 链上 |
| `sdpa` | 不用 | `scaled_dot_product_attention` 再 RMSNorm，eager。不是 FA-3 |
| `always_fuse` | 默认 compile，当作「总是融」 | compile(SDPA + RMSNorm) |
| `always_unfuse` | eager 链，当作「总是不融」 | SDPA 和 RMSNorm 分两段计时再相加 |
| `scic_selected` | 抄 `always_fuse` 或 `always_unfuse` 的 median | 同上 |

SCIC 用请求的数据手册 ridge 给 GEMM 或 IO-aware Attention 分档（`--machine`，默认 `A40_FP16`）。这不是卡上测到的 ridge。GEMM 上 MemMode（`fuse_epilogue`）是解析的 `always_fuse`，ComputeMode 是解析的 `always_unfuse`。ridge 带内两套都记在 `emit_both`。ridge 和计算侧再做短墙钟。有 `measured_winner` 时 `selected_baseline` 跟着它，否则留解析基线。`bakeoff_decisions.csv` 有 regime、mode、`epilogue_policy`、`analytical_baseline`、`measured_winner`、`selection_source`、`selected_baseline`。日志里每个 shape 一行：`SCIC op=... analytical=... measured=... source=... -> always_fuse` 或 `always_unfuse`。探针是本机墙钟，不是 DRAM AI。忙 GPU 可能翻赢家。这不是 Nsight SCIC 准确率。

Attention 两条路径都不把 score 写回 HBM，都调用库 SDPA。差别只在 RMSNorm 是进 compile 还是单独一段。tile 切换没有单独的 kernel。Stage-Stream 不在路径里。库 SDPA 已经融合 score，所以可执行 epilogue 策略是 `compile_sdpa_chain`（计时为 `always_fuse`），即使解析主计划是 ComputeMode。下面第一张表是改这条规则之前的归档：Attention 的 `scic_selected` 仍跟着解析的 `fuse_epilogue`。后面的 `paper/data/bakeoff_v2/` 是改完之后的墙钟（忙 GPU，GEMM 未校准）。`paper/data/bakeoff_v4/` 在同一策略上加 `--calibrate all-gemm`（空闲 GPU）。

每一行的诚实标签至少包括 `NO_NSIGHT`、`NOT_SCIC_ACCURACY`、`NOT_AN_FA3_KERNEL`、`NOT_STAGE_STREAM`、`NOT_A_CUSTOM_MEM_OR_COMPUTE_KERNEL`、`GRAPH_NOT_INSPECTED`、`REGIME_USES_DATASHEET_RIDGE`。`scic_selected` 还有 `TIMING_COPIED_FROM_always_fuse` 或 `TIMING_COPIED_FROM_always_unfuse`。没跑成的路径 `median_ms` 留空，`ok=false`。不要补数字。Attention 的加速比分母是物化 score 的 eager，标签 `SPEEDUP_VS_EAGER_COMPARES_TO_MATERIALIZED_ATTENTION` 说明这一点。

论文要读的列：`baseline`、`shape`、`median_ms`、`speedup_vs_eager`、`honesty`。

```bash
bash scripts/run_baseline_bakeoff.sh
SKIP_INSTALL=1 bash scripts/run_baseline_bakeoff.sh
SKIP_MAX_AUTOTUNE=1 bash scripts/run_baseline_bakeoff.sh
SKIP_ATTENTION=1 bash scripts/run_baseline_bakeoff.sh
```

脚本要求 `nvidia-smi` 和 CUDA 版 PyTorch，不下载 wheel，不调用 Nsight。产物在 `results/bakeoff/`。寄回 `SEND_BACK.txt` 里的文件：`bakeoff_latency.csv`、`bakeoff_decisions.csv`、`bakeoff_summary.json`、`nvidia_smi.txt`、`torch_version.txt`。建议空闲 GPU。bakeoff 会把 `torch._dynamo.config.cache_size_limit` 提到 64（可用 `DYNAMO_CACHE_SIZE_LIMIT` 覆盖），因为这次归档打到了默认值 8。提高上限不会检查图。

#### 修复前的运行（`paper/data/bakeoff/`）

下面的表是 `compile_sdpa_chain` 之前的选择器。不要把这个目录里的 `always_fuse` 改标成第二次的 `scic_selected`。

`bakeoff_summary.json`：`BAKEOFF_WALLCLOCK`，`nsight_used=false`，`quote_as_scic_accuracy=false`，`graph_inspected=false`，包版本 `0.3.0`，机器 `A40_FP16`，数据手册 ridge 215.08620689655172，repeats 10，warmup 3。设备 NVIDIA A40，PyTorch 2.5.1+cu121，CUDA 12.1。77 行时延全部 `ok`，12 条决策。`selected_baseline`：`always_fuse` 6，`always_unfuse` 6。`RUN_NOTES.txt` 记录分支 `cursor/docker-baseline-bakeoff-d966`、提交 `81ec1c5`。下表每个数字都是 CSV 格子。`scic_selected` 抄 `timing_copied_from`，不含切换开销。

**MatMul+RMSNorm**，N = K = 4096。加速比是 `speedup_vs_eager`。M = 1 的 `always_fuse` median 是 0.07160000131 ms，SCIC 抄的就是它。

| label | 档 / 解析主计划 | SCIC 抄了 | eager ms | fuse × | unfuse × | scic × | max-autotune × |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decode_m1 | 访存 / MemMode | always_fuse | 0.08276800066 | 1.155977642 | 1.005442218 | 1.155977642 | 0.4102950515 |
| decode_m32 | 访存 / MemMode | always_fuse | 0.09244799986 | 1.253906204 | 0.9921016283 | 1.253906204 | 0.4536746182 |
| ridge_m128 | ridge / MemMode，两套 | always_fuse | 0.09523200244 | 1.28469674 | 1 | 1.28469674 | 0.4547677376 |
| ridge_m256 | ridge / ComputeMode，两套 | always_unfuse | 0.1464319974 | 1.222222167 | 1.007042204 | 1.007042204 | 0.6299559449 |
| ridge_m512 | ridge / ComputeMode，两套 | always_unfuse | 0.2307999954 | 1.006208113 | 1.001527439 | 1.001527439 | 1.006208113 |
| prefill_m2048 | 计算 / ComputeMode | always_unfuse | 0.8734719753 | 0.9861271294 | 1 | 1 | 0.987268489 |
| prefill_m4096 | 计算 / ComputeMode | always_unfuse | 1.626352012 | 0.9960704628 | 0.99419993 | 0.99419993 | 1.000482317 |

访存侧和 ridge M = 128：SCIC 选 `always_fuse`，比 eager 快（1.155977642×、1.253906204×、1.28469674×）。M = 1 的 `torch_compile_default` 是同一种 compile 的另一次计时，0.07126400247 ms（1.161427899×），和 `always_fuse` 的 median 不是同一个数。max-autotune 在小 M 上更慢（M = 1 为 0.4102950515×）。M = 512、2048、4096 上 fuse 和 unfuse 都贴近 eager，选 `always_unfuse` 不会比 eager 差一截。M = 256 不是平局：`always_fuse` 为 1.222222167×，SCIC 抄的 unfuse 为 1.007042204×。unfuse 贴着 eager，没有拿到更快的 compile。这个缺口比下面的 Attention 失误小。

**Attention + RMSNorm**，B = 1，H = 8，D = 64。eager 分母物化了 score，所以 SDPA 的加速比混着库算法和 norm。

| label | 档 / 解析主计划 | SCIC 抄了 | eager ms | fuse × | unfuse × | scic × | sdpa × | compile default × |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prefill_s128 | 访存 / MemMode | always_fuse | 0.03276799992 | 2.452694651 | 1.183815013 | 2.452694651 | 1.18518518 | 2.461538483 |
| prefill_s512 | ridge / ComputeMode，两套 | always_unfuse | 0.06337599829 | 2.628400711 | 1.44456598 | 1.44456598 | 1.547265622 | 2.578775881 |
| prefill_s2048 | 计算 / ComputeMode | always_unfuse | 1.049600005 | 9.192825045 | 7.167832147 | 7.167832147 | 7.48175211 | 9.23423418 |
| decode_sk128 | 访存 / MemMode | always_fuse | 0.02961600013 | 2.22475965 | 0.9640624702 | 2.22475965 | 1.032924098 | 2.22475965 |
| decode_sk2048 | 访存 / MemMode | always_fuse | 0.03993599862 | 1.083333258 | 1.000400758 | 1.083333258 | 1.083333258 | 1.083333258 |

decode Sk = 128 和 prefill S = 128 是 MemMode，SCIC 抄 `always_fuse`（2.22475965×、2.452694651×），这是对的。prefill S = 512 和 S = 2048 是失误：解析 AI 把 S = 512 放在 ridge 带、脊点之上（AI 256，主计划 ComputeMode），S = 2048 放在计算侧（AI 1024），于是抄了 `always_unfuse`（1.44456598× 和 7.167832147×）。同形状的 `always_fuse` 是 2.628400711× 和 9.192825045×。S = 2048 的 `torch_compile_default` 是 9.23423418×。eager 的 `sdpa`（一段区域里的 SDPA 再 RMSNorm，没有 compile）在 S = 512 为 1.547265622×，在 S = 2048 为 7.48175211×，贴近 `always_unfuse`，落后于编译链。decode Sk = 128 的 max-autotune 是 0.9181547732×，比 eager 慢。库 kernel 已经不写 score。ComputeMode「不要再融额外 epilogue」用在这条基线上是错的：它只是把 RMSNorm 从 SDPA 上拆开。

`decode_sk2048` 仍是访存（解析 AI 0.9995119571），SCIC 抄 `always_fuse`，1.083333258×，和 `sdpa`、`torch_compile_default` 同一个 median（0.0368640013 ms）。

#### 和上传一起记下的限制

- `RUN_NOTES.txt`：脚本启动时 GPU 0 利用率 83%，已分配 13092 MiB；GPU 1 为 77%，12948 MiB。不是空闲 GPU。并发负载会挪这些时延。
- 同一归档里的 `nvidia_smi.txt` 是稍后的快照（Mon Oct 12 02:10:34 2026，驱动 535.129.03）。GPU 0 为 81%（13092 MiB，255 W / 300 W），GPU 1 为 78%（12948 MiB，231 W / 300 W）。两份文件的利用率不一样。都不是空闲卡。
- Dynamo 对 MatMul 链和 `sdpa_chain` 报告 `cache_size_limit` 为 8。77 行 `ok` 不能证明每个 compile 行跑的是编译后的代码。`graph_inspected` 为 false。
- `scic_selected` 抄所选基线，不含分发或计划切换开销。
- 没有跑 Nsight。`quote_as_scic_accuracy` 为 false。regime 列是数据手册 ridge。

#### 第二次运行：`compile_sdpa_chain` 之后（`paper/data/bakeoff_v2/`）

这个目录是库 SDPA epilogue 策略的 bakeoff。文件按寄回原样复制，数字没有改。`bakeoff_summary.json` 里仍有生成器写的那句「请重跑」，并把 `paper/data/bakeoff` 称作修复前选择器。那句话是写文件的代码发出的（同一对象的 `still_tbd`）。旁边的 CSV 才是修复后的墙钟。

`RUN_NOTES.txt`：分支 `cursor/bakeoff-numbers-attention-fix-0187`，提交 `cb3992f`。命令是 `CUDA_VISIBLE_DEVICES=1 SKIP_INSTALL=1 bash scripts/run_baseline_bakeoff.sh`。摘要里的 CLI 是：

```text
/root/test/shapefuse/__main__.py bakeoff --out /root/test/results/bakeoff --device cuda --require-cuda --machine A40_FP16 --level decode-prefill --skip-plots --attention
```

`BAKEOFF_WALLCLOCK`，`nsight_used=false`，`quote_as_scic_accuracy=false`，`graph_inspected=false`，包版本 `0.3.0`，机器 `A40_FP16`，数据手册 ridge 215.08620689655172，repeats 10，warmup 3。设备 NVIDIA A40，PyTorch 2.5.1+cu121，CUDA 12.1，Python 3.10.12。77 行全部 `ok`，12 条决策。`selected_baseline`：`always_fuse` 8，`always_unfuse` 4。下表每个数字都是这个目录的 CSV 格子。`scic_selected` 抄 `timing_copied_from`。

**不是空闲 GPU。** `gpu_selection.json` 里 `"gpu": 1`，`"idle_found": false`。跑前采了六次，没有找到空闲设备。`RUN_NOTES.txt`：脚本启动时 GPU 1 利用率 92%，其他负载已占用 13490 MiB。并发会挪这些时延。这次上传没有 `nvidia_smi.txt`、`torch_version.txt`、`run.log`。Dynamo 的 `cache_size_limit` 是 64，`RUN_NOTES.txt` 写 `run.log` 里没有 cache 上限告警。一些 max-autotune 候选超过了 A40 共享内存上限被拒绝。最终 CSV 每一行都是 `ok=true`。图没有检查。没有跑 Nsight。

**Attention + RMSNorm。** 五条决策都抄 `always_fuse`。解析主计划不变，所以 ComputeMode 上 `fuse_epilogue` 仍是 false。被覆盖的抄写带 `LIBRARY_SDPA_EPILOGUE_COMPILE_SDPA_CHAIN`。prefill S = 512 和 S = 2048 还带 `LIBRARY_SDPA_CHAIN_NOT_ANALYTICAL_PRIMARY`，因为解析主计划是 ComputeMode。decode 和 prefill S = 128 不带这条：它们的解析主计划已经是 MemMode。

| label | 档 / 解析主计划 | SCIC 抄了 | eager ms | fuse × | unfuse × | scic × | sdpa × | compile default × |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prefill_s128 | 访存 / MemMode | always_fuse | 0.2073599994 | 15.57692321 | 6.882634006 | 15.57692321 | 7.499999966 | 15.57692321 |
| prefill_s512 | ridge / ComputeMode，两套 | always_fuse | 0.1720000058 | 6.966947518 | 1.493885526 | 6.966947518 | 1.562500053 | 6.718750398 |
| prefill_s2048 | 计算 / ComputeMode | always_fuse | 1.04499197 | 9.193693331 | 5.262004369 | 9.193693331 | 7.448905144 | 9.189812469 |
| decode_sk128 | 访存 / MemMode | always_fuse | 0.02867200039 | 2.153846208 | 0.9927977776 | 2.153846208 | 1.076923104 | 1.908413251 |
| decode_sk2048 | 访存 / MemMode | always_fuse | 0.03584000096 | 1.521739137 | 0.9499576291 | 1.521739137 | 0.9876543299 | 1.489361756 |

修复前的失误在这次墙钟上合上了。prefill S = 512 抄 `always_fuse`，0.02468800079 ms（6.966947518×）。同形状 `always_unfuse` 是 0.1151360013 ms（1.493885526×）。prefill S = 2048 抄 `always_fuse`，0.1136640012 ms（9.193693331×）。`always_unfuse` 是 0.1985919997 ms（5.262004369×）。S = 2048 的 `torch_compile_default` 是 0.1137120016 ms（9.189812469×），同一种 compile 的另一次计时。decode 也抄 `always_fuse`，并且快于 `always_unfuse`：Sk = 128 是 0.01331199985 ms（2.153846208×），对照 0.02888000058 ms（0.9927977776×）；Sk = 2048 是 0.02355200052 ms（1.521739137×），对照 0.0377279995 ms（0.9499576291×）。prefill S = 128 抄 0.01331199985 ms，和 `torch_compile_default` 同一个 median。这一行的 eager 分母是 0.2073599994 ms。修复前同一 label 的 eager median 是 0.03276799992 ms，所以 15.57692321× 不能拿去和修复前的 2.452694651× 比。两次是不同的机器状态。eager 分母仍然物化了 score。

**MatMul+RMSNorm，第二次失误。** GEMM 规则没变。M = 1 的 `always_fuse` 是 0.07168000191 ms，SCIC 抄的就是它。

| label | 档 / 解析主计划 | SCIC 抄了 | eager ms | fuse × | unfuse × | scic × | max-autotune × |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decode_m1 | 访存 / MemMode | always_fuse | 0.08348800242 | 1.164732146 | 1.000191682 | 1.164732146 | 0.4097369612 |
| decode_m32 | 访存 / MemMode | always_fuse | 0.09291199967 | 1.267627163 | 1.001897835 | 1.267627163 | 0.4519066112 |
| ridge_m128 | ridge / MemMode，两套 | always_fuse | 0.09583999962 | 1.285407694 | 1.001839787 | 1.285407694 | 0.3391653976 |
| ridge_m256 | ridge / ComputeMode，两套 | always_unfuse | 0.1585920006 | 1.307134385 | 1.005987998 | 1.005987998 | 0.6851455047 |
| ridge_m512 | ridge / ComputeMode，两套 | always_unfuse | 0.2304320037 | 1.327495649 | 1.003134367 | 1.003134367 | 0.4087761186 |
| prefill_m2048 | 计算 / ComputeMode | always_unfuse | 0.8811199963 | 1.245223289 | 1.014703715 | 1.014703715 | 1.047436091 |
| prefill_m4096 | 计算 / ComputeMode | always_unfuse | 1.622527957 | 1.234890808 | 0.9974916777 | 0.9974916777 | 1.087951901 |

访存侧和 ridge M = 128：SCIC 仍选 `always_fuse`（1.164732146×、1.267627163×、1.285407694×）。max-autotune 在小 M 上更慢（M = 1 为 0.4097369612×）。失误是另外四个 shape。解析 AI 把 M = 256 和 M = 512 放在 ridge 带、脊点之上（主计划 ComputeMode，`emit_both`），把 M = 2048 和 M = 4096 放在计算侧，于是抄了 `always_unfuse`：1.005987998×、1.003134367×、1.014703715×、0.9974916777×。同形状的 `always_fuse` 是 1.307134385×、1.327495649×、1.245223289×、1.234890808×。另一次 `torch_compile_default` 方向一致：1.310937741×、1.325174806×、1.248866091×、1.243339321×。修复前只在 M = 256 上把缺口写清楚（1.222222167× 对 1.007042204×），M = 512、2048、4096 上 fuse 和 unfuse 都贴近 1×。这一次不是。matmul 加 RMSNorm 的 `torch.compile` 在解析计算侧仍然更快。

这是纯数据手册 AI、没有测量反馈的限制。标签说计算受限，这份归档的选择器就拒绝 epilogue。bakeoff 的 `always_fuse` 是 `torch.compile`，不是会挤占 MMA 主循环的手写 kernel。在这张忙卡上，compile 相对 eager 仍大约快 1.23× 到 1.33×，抄到的路径贴近 1×。这不是 SCIC 准确率，不是实测 ridge，也不能据此改 regime 的名字。这份 CSV 的 GEMM 策略仍是解析映射。

本目录采集于墙钟 GEMM 探针之前，不要改标 ridge M = 512 或 prefill M ∈ {2048, 4096} 上的 `always_unfuse`。Attention 仍是 `compile_sdpa_chain`。

#### 第四次运行：空闲 GPU 上 `CALIBRATE=all-gemm`（`paper/data/bakeoff_v4/`）

文件按寄回原样复制。`RUN_NOTES.txt`：分支 `cursor/calibrated-scic-gemm-f3e1`，提交 `361366c`；命令 `CALIBRATE=all-gemm SKIP_INSTALL=1 bash scripts/run_baseline_bakeoff.sh`。摘要 CLI 含 `--calibrate all-gemm --attention`。包版本 `0.3.1`，77 行全部 `ok`，12 条决策。`selected_baseline`：`always_fuse` 8，`always_unfuse` 4。`selection_source`：`measured_wallclock` 7，`library_sdpa` 5。下表数字只来自该目录的 CSV。`scic_selected` 抄 `timing_copied_from`。

**空闲 GPU。** `RUN_NOTES.txt` 记录物理 GPU 1 上连续两次空闲样本；启动时 0% 利用率、4 MiB 已分配。未包含 `gpu_selection.json` 或 `nvidia_smi.txt`。Dynamo `cache_size_limit` 64，无 cache 上限告警。未跑 Nsight。

**决策（`bakeoff_decisions.csv`）。** 五条 Attention：`selection_source=library_sdpa`，`selected_baseline=always_fuse`，`epilogue_policy=compile_sdpa_chain`。七条 GEMM：`selection_source=measured_wallclock`。decode M ∈ {1, 32}、ridge M ∈ {128, 256} 选 `always_unfuse`；ridge M = 512、prefill M ∈ {2048, 4096} 选 `always_fuse`。

**Attention + RMSNorm。** 五条都抄 `always_fuse`。

| label | 档 / 解析主计划 | SCIC 抄了 | eager ms | fuse × | unfuse × | scic × | sdpa × | compile default × |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prefill_s128 | 访存 / MemMode | always_fuse | 0.1531839967 | 1.187251899 | 1.527928474 | 1.187251899 | 1.598864371 | 1.178483423 |
| prefill_s512 | ridge / ComputeMode，两套 | always_fuse | 0.1572639942 | 0.9196294263 | 1.455932412 | 0.9196294263 | 1.507053034 | 0.91415544 |
| prefill_s2048 | 计算 / ComputeMode | always_fuse | 1.072128057 | 5.587725429 | 5.584000359 | 5.587725429 | 6.864870734 | 5.569149074 |
| decode_sk128 | 访存 / MemMode | always_fuse | 0.1495360062 | 1.122103559 | 1.48940244 | 1.122103559 | 1.578716243 | 1.14985235 |
| decode_sk2048 | 访存 / MemMode | always_fuse | 0.1500160024 | 0.8798798842 | 1.427527438 | 0.8798798842 | 1.465687106 | 0.8796322189 |

**MatMul+RMSNorm，校准后 GEMM。**

| label | 档 / 解析主计划 | SCIC 抄了 | eager ms | fuse × | unfuse × | scic × | max-autotune × |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decode_m1 | 访存 / MemMode | always_unfuse | 0.08750399947 | 0.7629743153 | 0.9936409823 | 0.9936409823 | 0.3274065968 |
| decode_m32 | 访存 / MemMode | always_unfuse | 0.1002399996 | 0.8403755622 | 0.9990432446 | 0.9990432446 | 0.3680963526 |
| ridge_m128 | ridge / MemMode，两套 | always_unfuse | 0.1043679975 | 0.8637446566 | 0.9990810161 | 0.9990810161 | 0.3775102631 |
| ridge_m256 | ridge / ComputeMode，两套 | always_unfuse | 0.1607680023 | 0.9401197422 | 1.022801335 | 1.022801335 | 0.5304894191 |
| ridge_m512 | ridge / ComputeMode，两套 | always_fuse | 0.2396160066 | 1.066059265 | 1 | 1.066059265 | 0.6157894709 |
| prefill_m2048 | 计算 / ComputeMode | always_fuse | 0.8744959831 | 1.134130124 | 1 | 1.134130124 | 0.9542898743 |
| prefill_m4096 | 计算 / ComputeMode | always_fuse | 1.627135992 | 1.186761891 | 0.9910055616 | 1.186761891 | 1.043335547 |

空闲 GPU 上 decode 与 ridge M ∈ {128, 256} 跟探针选 `always_unfuse`；prefill M ∈ {2048, 4096} 与 ridge M = 512 跟探针选 `always_fuse`，合上 v2 里可见的 GEMM 缺口。这是本机墙钟，不是 DRAM AI，也不是 Nsight SCIC 准确率。Attention 仍是 `library_sdpa`。

三份归档里填好的 CSV 能支持：每个 shape 选了哪条计划，以及相对 eager / always-fuse / always-unfuse 的墙钟。不能支持：achieved AI、SCIC 准确率、实测 ridge、手写融合 kernel、FlashAttention-3、Stage-Stream。只有 `paper/data/bakeoff_v4/` 是本文引用的空闲 GPU 时延。

A40 的 `torch.matmul` 曲线留在下面，只作「吞吐随 M 变」的动机，不是这次 bakeoff。

解析部分：

```bash
pip install -r requirements.txt && pip install -e .
python -m shapefuse selfcheck
bash scripts/run_cpu.sh          # 无 GPU，PLACEHOLDER_NO_GPU
# 管理员放开性能计数器并重启之后。DRAM 字节仍是 TBD：
TRY_NSIGHT=1 bash scripts/run_a40.sh
bash scripts/run_h100.sh         # 仍是 H100 锚点的 matmul 扫描，默认不开 profiler
```

`ncu` 是 Nsight Compute，不是驱动，也不是 `nvidia-smi`。PATH 上没有 `ncu` 时，脚本会说明；若设了 `TRY_NSIGHT=1`，会写 `nsight_skipped.txt` 并继续跑 matmul。第三次包里 `ncu` 在 PATH 上，但失败码是 `ERR_NVGPUCTRPERM`。管理员步骤见 `docs/NSIGHT_PERMISSIONS.md`（`modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0`，或写入 `/etc/modprobe.d/`，然后重启，再用 `ncu` 验证）。DRAM 字节保持为空。

`selfcheck` 锁住第 3 节用到的闭式，包括 A40 数据手册 ridge 和 M\*。

### 7.1 动机：第三次寄回的 A40 vendor 曲线（`PARTIAL_VENDOR_MATMUL`）

这一小节不是 bakeoff。它只说明 `torch.matmul` 的吞吐随 M 变化。

归档：`paper/data/a40/`（`bench_gemm.csv`、`summary.json`、`gpu_measurements.json`、`crossings.json`、`agreement.json`、`fusion_proxy.csv`、`bandwidth_estimate.json`、`nsight_skipped.txt`、`nsight_error.log`）。这次上传没有 `decision_log.csv`。图：`paper/fig_a40_torch_matmul_tflops.png`。

- 设备：NVIDIA A40。PyTorch 2.5.1+cu121，CUDA 12.1，Python 3.10.12。
- 命令：`--bench-level a40 --machine A40_FP16 --profile torch --fusion-proxy`。ridge = 215.08620689655172，M\* = 240.32584259093773。`summary.json` 与 `crossings.json` 含 `A40_FP16`。
- `nsight_skipped.txt`：`TRY_NSIGHT=1`，`ncu` 在 PATH 上，`scripts/nsight_dram.sh` 失败。`nsight_error.log` 是 `ERR_NVGPUCTRPERM`（进程 30939，设备 0 无权读 GPU 性能计数器）。
- 计时的是 `torch.matmul`，FP16，N=K=4096。`kernel=pytorch_matmul`，`fused_plan_timed=false`。
- `nsight_status=TBD_RUN_PROFILER`（没有 `nsight_dram.json`）。`nsight_dram_bytes` 和 `achieved_ai` 为空（`TBD_NSIGHT`）。
- `proxy_ai_status=TORCH_PROFILER_MEMORY_NOT_DRAM`，`proxy_byte_source=key_averages.self_device_memory_usage`，七行 `proxy_ai` 都是 4096。字节数 8192、32768、262144、2818048、4194304、16777216、33554432，等于 FP16 输出矩阵 `M×4096×2`。`2MNK` 除以这个计数后 M 约掉，剩下 K=4096。
- `agreement.json`：`quote_as_scic_accuracy=false`，`scic_accuracy_status=PROXY_TORCH_MEMORY_NOT_VALID_SCIC_ACCURACY`，`n_with_byte_ai=7`，`n_torch_profiler_memory=7`，`n_nsight_dram_isolated=0`。`regime_agreement_rate_any_byte_source` 和 `decision_agreement_rate_any_byte_source` 都是 0.2857142857142857，`n_decision_flips_any_byte_source=5`。4096 在 A40 和 H100 数据手册上都不低于 2R，所以 profiler 把七个 shape 都标成计算；解析档只在 M∈{2048,4096} 是计算。不要把 0.2857142857142857 写成 SCIC 准确率。
- 下表的 median_s 和 TFLOP/s 是 CSV 原文。两列 regime 都是数据手册三档，不是实测 ridge。

| label | M | median_s | TFLOP/s | 解析 AI | 相对 A40 数据手册 | 相对 H100 数据手册 |
| --- | --- | --- | --- | --- | --- | --- |
| decode_gemv | 1 | 7.072000206e-05 | 0.4744687645 | 0.9995 | 访存 | 访存 |
| micro_batch | 4 | 7.864000276e-05 | 1.706736054 | 3.992 | 访存 | 访存 |
| decode_batch32 | 32 | 7.822399959e-05 | 13.7265012 | 31.51 | 访存 | 访存 |
| near_mstar | 344 | 0.000141967997 | 81.30511704 | 294.5 | ridge | ridge |
| ridge_m512 | 512 | 0.0001788479984 | 96.05849291 | 409.6 | ridge | ridge |
| prefill_m2048 | 2048 | 0.0006588959992 | 104.2948763 | 1024 | 计算 | 计算 |
| square_large | 4096 | 0.00119116801 | 115.3816693 | 1365 | 计算 | 计算 |

`achieved_tflops_max` = 115.38166932883347。`fraction_of_requested_datasheet_peak` = 0.7707526341271441，即相对 A40 数据手册 149.7 约 77.1%。`fraction_of_h100_fp16_datasheet_peak` = 0.11666498415453333，是同一吞吐相对 H100 数据手册 989。`measured_ridge_flops_per_byte` 仍是 null。

可以写的句子：吞吐从 M=1 的 0.4744687645 TFLOP/s 升到 M=4096 的 115.3816693；M≤32 时中位时间停在约 7.07e-5–7.86e-5 s，FLOP 已经增加 32 倍，这是 launch 主导的一端。三档上 A40 数据手册和 H100 数据手册一致。

不要写的句子：融合加速、MemMode 对 ComputeMode、Nsight achieved AI、H100 时延、A40 上测到了 ridge、把 0.2857142857142857 当成 SCIC 准确率。

H100 matmul 表继续空着。Achieved AI = FLOPs / 实测 DRAM 字节，不是 FLOPs / 时延，也不是 FLOPs / profiler 显存。

#### copy 带宽（`ESTIMATE_COPY_NOT_NSIGHT`）

`bandwidth_estimate.json` 的 `status` 是 `ESTIMATE_COPY_NOT_NSIGHT`。`not_nsight`、`not_a_measured_ridge`、`do_not_write_measured_ridge_flops_per_byte` 都是 true。10 次 device-to-device copy，268435456 个 FP16 元素，读写都计，10737418240 字节，用时 0.01883420753479004 s。`estimated_bandwidth_gb_per_s` = 570.1019392595165（约 570.1 GB/s）。`estimated_ridge_flops_per_byte` = 262.5846181025793，分子是数据手册 FLOP 峰值 149.7 TFLOP/s，分母是这次 copy 带宽。`measured_hbm_bandwidth_bytes_per_s` 和 `measured_ridge_flops_per_byte` 都是 null。不要用 262.5846181025793 替换数据手册 ridge 215.08620689655172。

#### 融合代理（`PROXY_NOT_FUSION_SPEEDUP`）

`fusion_proxy.ran=true`，`status=PROXY_NOT_FUSED_GEMM`，`triton_used=true`，`torch_compile_used=true`。`claims_memmode_kernel`、`claims_computemode_kernel`、`claims_stage_stream` 都是 false。每一行 `ok=true`，`time_ratio_status=PROXY_TIMING_NOT_A_FUSION_SPEEDUP`。时间比是两段基线除以该变体，不是融合加速。

matmul+RMSNorm：

- `unfused_two_regions`：两段分开计时再相加。
- `eager_chain`：仍是两次 eager launch。CSV 时间比：M=1 为 1.398085063，M=4 为 1.328853438，M=32 为 1.393103491，M=344 为 1.332332706，M=512 为 1.143115217，M=2048 为 1.016681221，M=4096 为 1.012774456。
- `torch_compile`：没有检查图，不能当成已经融合。M=1 为 1.378291104，M=4 为 1.509902163，M=4096 为 1.211543482（中间各档见 CSV）。

bias+GELU（GEMM 输出已经物化，不是 GEMM 融合）：

- hand stub 时间比从 M=1 的 1.132994609 到 M=4096 的 1.013019741。
- Triton 七行全部 `ok=true`，诚实标签 `TRITON_ELEMENTWISE_STUB_NOT_GEMM_FUSION`。时间比：M=1 为 1.993904752，M=4 为 1.51406735，M=32 为 1.505031954，M=344 为 1.557734173，M=512 为 1.879358184，M=2048 为 4.484666033，M=4096 为 5.375718689。这是已经物化矩阵上的一个逐元素 kernel，不是融合 GEMM。上一包的 `NameError('tl is not defined')` 不是这一包的结果。

### 7.2 Nsight 不在 Docker bakeoff 里

Docker 的下一轮是本节开头的 `scripts/run_baseline_bakeoff.sh`，不调用 Nsight。第三次已经设了 `TRY_NSIGHT=1`，`ncu` 也在 PATH 上，失败原因是 `ERR_NVGPUCTRPERM`。若以后还要 DRAM 字节，再让管理员按 `docs/NSIGHT_PERMISSIONS.md` 放开计数器并重启，然后 `TRY_NSIGHT=1 bash scripts/run_a40.sh`。那条命令不是 bakeoff。没有 `ncu` 才是安装缺口：用 `ncu --version` 确认，再用 `ncu --query-metrics | grep dram__bytes` 看指标名。权限错误改 `NCU_METRICS` 解决不了。`achieved_ai` 在 bakeoff 里保持为空。

这一包已经说明的钩子（仍受上面的限制）：Triton bias+GELU 启动成功，状态 `TRITON_ELEMENTWISE_STUB_NOT_GEMM_FUSION`；profiler 读到了 `self_device_memory_usage`，状态 `TORCH_PROFILER_MEMORY_NOT_DRAM`，`proxy_ai` 卡在 4096。两者都不是 DRAM，也不能当 SCIC 准确率。

脚本末尾会跑 copy kernel 带宽探针，写入 `bandwidth_estimate.json`，状态 `ESTIMATE_COPY_NOT_NSIGHT`。第三次已经写出 570.1019392595165 GB/s 和估计 ridge 262.5846181025793。ridge 的分子是数据手册 FLOP 峰值。不要写进 `measured_ridge_flops_per_byte`。

| 产物 | 诚实等级 |
| --- | --- |
| `bench_gemm.csv` | 仍是 `torch.matmul`。没有 Nsight JSON 时 `achieved_ai` 留空 |
| `proxy_ai` | 这一包是 `TORCH_PROFILER_MEMORY_NOT_DRAM`，值 4096，字节像输出矩阵。只有分配器差分时为 `TORCH_ALLOCATOR_DELTA_NOT_DRAM`；都没有则 `TORCH_PROFILER_EMPTY`。不是 DRAM |
| `fusion_proxy.csv` | 时间比状态固定 `PROXY_TIMING_NOT_A_FUSION_SPEEDUP`。Triton 行是 `TRITON_ELEMENTWISE_STUB_NOT_GEMM_FUSION`。不是融合 GEMM。`memmode_kernel=false` |
| `agreement.json` | 这一包 `quote_as_scic_accuracy=false`，状态 `PROXY_TORCH_MEMORY_NOT_VALID_SCIC_ACCURACY`，一致率 0.2857142857142857。不能当准确率 |
| `nsight_dram.json` | 仅 `TRY_NSIGHT=1` 且 `ncu` 成功。状态 `NSIGHT_DRAM_PROCESS_SUM` |
| `nsight_error.log` | `ncu` 失败时的 stderr。这一包是 `ERR_NVGPUCTRPERM` |
| `nsight_skipped.txt` | `TRY_NSIGHT=1` 但没有 `ncu`，或 ncu 失败 |
| `bandwidth_estimate.json` | `ESTIMATE_COPY_NOT_NSIGHT`。不是 Nsight，不是实测 ridge |

### 7.3 状态表

| 实验 | 状态 |
| --- | --- |
| 解析 AI、分档、M\*、计划表、图 | 完成 |
| A40 上 matmul 时延 / TFLOP/s，A40 数据手册三档 | 完成，§7.1，第三次包，`PARTIAL_VENDOR_MATMUL`，`--machine A40_FP16` |
| A40 fusion proxy | 跑过。Triton 行 `ok=true`，`TRITON_ELEMENTWISE_STUB_NOT_GEMM_FUSION`。时间比不是融合加速 |
| A40 copy 带宽估计 | 已写出，`ESTIMATE_COPY_NOT_NSIGHT`，570.1019392595165 GB/s，估计 ridge 262.5846181025793。`measured_ridge_flops_per_byte` 为 null |
| H100 上 matmul 时延 / TFLOP/s | TBD，脚本 `scripts/run_h100.sh` |
| Nsight DRAM 字节、achieved AI | TBD。不在 Docker bakeoff 里。`nsight_status=TBD_RUN_PROFILER`。第三次 `TRY_NSIGHT=1` 且 `ncu` 在 PATH 上，失败码 `ERR_NVGPUCTRPERM`。若以后要字节：`docs/NSIGHT_PERMISSIONS.md`，重启，再 `TRY_NSIGHT=1`。这不是测得 0 |
| 机器实测 ridge | TBD。copy 探针的状态是 `ESTIMATE_COPY_NOT_NSIGHT`，不是这一行 |
| SCIC 对孤立 Nsight AI 的准确率 | TBD。归档 `agreement.json` 是 `PROXY_TORCH_MEMORY_NOT_VALID_SCIC_ACCURACY`，一致率 0.2857142857142857，`quote_as_scic_accuracy` 为 false |
| Docker bakeoff，修复前的 Attention 选择器 | 已归档，§7，`paper/data/bakeoff/`，`BAKEOFF_WALLCLOCK`。无 Nsight。不是自定义 kernel。不是 FA-3。`quote_as_scic_accuracy` 为 false。GPU 不空闲（启动时 83% 和 77%）。Dynamo cache 上限是 8 |
| `compile_sdpa_chain` 之后的第二次 bakeoff | 已归档，§7，`paper/data/bakeoff_v2/`。五条 Attention 都抄 `always_fuse`。GEMM 在 ridge M ∈ {256, 512} 和 prefill M ∈ {2048, 4096} 仍抄 `always_unfuse`，而 `always_fuse` 更快。不是空闲 GPU（GPU 1 为 92%，13490 MiB）。Dynamo cache 上限 64，没有记下 cache 上限告警。不是 SCIC 准确率 |
| `CALIBRATE=all-gemm` 第四次 bakeoff（空闲 GPU） | 已归档，§7，`paper/data/bakeoff_v4/`。七条 GEMM 来自 `measured_wallclock`（decode 与 ridge M ∈ {128, 256} → `always_unfuse`；ridge M = 512 与 prefill M ∈ {2048, 4096} → `always_fuse`）。五条 Attention 来自 `library_sdpa` → `always_fuse`。不是 SCIC 准确率。`paper/data/bakeoff_v2/` 早于探针 |
| 手写 MemMode / ComputeMode kernel | TBD。bakeoff 不生成它们 |
| 短 S 与长 S 的 Attention：库 SDPA 加或不加进 compile 的 epilogue | bakeoff 的 Attention 行。median TBD。不是 FA-3 |
| MoE 倾斜与空专家权重 | TBD |
| 量化 epilogue 对实测峰值 | TBD |
| 计划缓存命中率和切换开销 | TBD |
| 端到端 tokens/s | TBD |
| 用实测替换 5 µs launch | TBD |

## 8. 相关工作（已核对条目；接到 Inductor 仍 TBD）

- Roofline：Williams, Waterman, Patterson, CACM 2009。
- IO-aware Attention：FlashAttention（Dao 等，NeurIPS 2022）当作可选计划，不当作本文贡献；bakeoff 里是库 SDPA，不是 FA-3。
- ComFuse：Mu 等，arXiv:2608.03537；Stage-Stream 归他们，本文不实现。
- RedFuser：Tang 等，ASPLOS 2026（arXiv:2603.10026）；Neptune：Zhao 等，PLDI 2026（arXiv:2510.08726）。它们发融合 kernel；本文停在分类与路径选择。
- FuseFlow：Lacouture 等，ASPLOS 2026（arXiv:2511.04768）；稀疏/数据流上的融合粒度，与本文 dense GPU regime 标签正交。
- TorchInductor / `torch.compile`（Ansel 等，2024）：已有融合；把 SCIC regime 喂进启发式是 **TBD**。
- TensorRT-LLM 一类 MoE：是否按专家强度分桶，对着实现核对，**TBD**。

## 9. 局限（必须留在稿子里）

- 冷 HBM，无 L2，无因果掩码，无 paged KV，无重叠。
- 0.5R / 2R 是旋钮，应该用测量误差分布来定。
- Softmax / RMSNorm 的 FLOP 常数是约定。
- 数据手册峰值不是机柜里那张卡。FP8 的 M\* 几乎不变，依赖于峰值和字节同比例变化。
- 双计划没有寄存器 spill 模型。ComputeMode 拒绝融合，是对照 Self-Attn 反例的政策，不是 spill 代价。AI × spill × SMEM 的联合代价是后续工作，本文不声称已经做了。
- 没有 MemMode / ComputeMode kernel，没有 tokens/s。§7 的第三次 A40 数字是 `torch.matmul` 的时延和 TFLOP/s，加上不能当成融合加速的 fusion proxy，以及 copy 估计（`ESTIMATE_COPY_NOT_NSIGHT`，570.1019392595165 GB/s，估计 ridge 262.5846181025793）。`measured_ridge_flops_per_byte` 为 null。`quote_as_scic_accuracy` 为 false。profiler 一致率 0.2857142857142857 的状态是 `PROXY_TORCH_MEMORY_NOT_VALID_SCIC_ACCURACY`，不是 SCIC 准确率。
- Nsight 被 `ERR_NVGPUCTRPERM` 挡住。`TRY_NSIGHT=1` 时 `ncu` 在 PATH 上，但读不了 GPU 性能计数器（`nsight_error.log`）。`achieved_ai` 仍为空，`nsight_status` 仍是 `TBD_RUN_PROFILER`。管理员步骤在 `docs/NSIGHT_PERMISSIONS.md`；重启后再跑一次 `TRY_NSIGHT=1`。
- Docker bakeoff 不跑 Nsight。指标是子图墙钟和日志里的计划选择，不能写成 SCIC 准确率。`always_fuse` 是 `torch.compile` 尝试。`always_unfuse` 是 eager（Attention 的 epilogue 分两段计时）。都不是手写的 MemMode / ComputeMode kernel。图没有被检查。`scic_selected` 抄基线 median。修复前与第二次归档的 GPU 不空闲。第四次 `paper/data/bakeoff_v4/RUN_NOTES.txt` 记录等到空闲样本。修复前 Dynamo 的 `cache_size_limit` 是 8。第二次与第四次是 64，没有记下 cache 上限告警。
- `compile_sdpa_chain` 的墙钟在 `paper/data/bakeoff_v2/`。五条 Attention 都抄 `always_fuse`，包括 prefill S = 512（0.02468800079 ms，6.966947518×）和 S = 2048（0.1136640012 ms，9.193693331×）。这是 `torch.compile` 下的库 SDPA 加 RMSNorm，不是新 kernel，也不是空闲 GPU。
- `paper/data/bakeoff_v2/` 里 GEMM 的 ComputeMode 仍映射到 `always_unfuse`。`paper/data/bakeoff_v4/` 在空闲 GPU 上跟探针选路：prefill M ∈ {2048, 4096} 与 ridge M = 512 抄 `always_fuse`（例如 prefill M = 2048 的 `scic_selected` 0.7710720003 ms，1.134130124×）；decode 抄 `always_unfuse`。探针不是 DRAM AI，也不是 Nsight SCIC 准确率。忙 GPU 可能翻赢家。不要改标 v2 的格子。
- 调用 `scaled_dot_product_attention` 的行是 PyTorch 库算子，不是本文的 FlashAttention-3 kernel。Stage-Stream 不在 bakeoff 里。
- 这一包的 Triton bias+GELU 行已经跑起来（`ok=true`，`TRITON_ELEMENTWISE_STUB_NOT_GEMM_FUSION`）。它是已经物化矩阵上的逐元素 stub，不是融合 GEMM。

## 10. 结论句（可直接用）

在 H100 这一档的 Roofline 上，宽度 4096 的 FP16 方阵投影大约在 M\* ≈ 345 换边；同一闭式放到 A40 数据手册（149.7 TFLOP/s，696 GB/s，R = 215.08620689655172）上，M\* = 240.32584259093773，第三次上传的 `summary.json` 和 `crossings.json` 里有这一项。IO-aware Attention 的强度随 S 换边，把 score 写回 HBM 的实现则不会；倾斜的 MoE 一层之内就同时有两种 regime。按算子种类做死的融合跟不上这些穿越。SCIC 把 regime 从 shape 和流量里算出来，双计划把它变成 MemMode、ComputeMode 或两者都要。Stage-Stream 不属于这个设计。设备上目前是 `--machine A40_FP16` 的第三次 `torch.matmul` 曲线，状态 `PARTIAL_VENDOR_MATMUL`：M=1 为 0.4744687645 TFLOP/s，M=4096 为 115.3816693 TFLOP/s，相对 A40 数据手册峰值的比例是 0.7707526341271441，不是实测 ridge。`TRY_NSIGHT=1` 找到了 `ncu`，随后 `ERR_NVGPUCTRPERM`，`achieved_ai` 仍为空（`TBD_RUN_PROFILER`）。`proxy_ai` 是 `TORCH_PROFILER_MEMORY_NOT_DRAM`，卡在 4096。`quote_as_scic_accuracy` 为 false；检查用一致率 0.2857142857142857 的状态是 `PROXY_TORCH_MEMORY_NOT_VALID_SCIC_ACCURACY`，不是 SCIC 准确率。copy 估计是 `ESTIMATE_COPY_NOT_NSIGHT`（570.1019392595165 GB/s，估计 ridge 262.5846181025793），`measured_ridge_flops_per_byte` 为 null。fusion proxy 跑过，时间比是 `PROXY_TIMING_NOT_A_FUSION_SPEEDUP`。Triton bias+GELU 是 `TRITON_ELEMENTWISE_STUB_NOT_GEMM_FUSION` 且 `ok=true`，这是逐元素 stub，不是融合 GEMM。Docker 上的实验是 bakeoff。修复前的归档是 `paper/data/bakeoff/`（`BAKEOFF_WALLCLOCK`，Nsight 关闭）。那一次 Attention 的 ComputeMode 在 prefill S = 512 失误（抄到 1.44456598×，`always_fuse` 为 2.628400711×），在 S = 2048 也失误（抄到 7.167832147×，`always_fuse` 为 9.192825045×）。第二次归档 `paper/data/bakeoff_v2/` 打 `compile_sdpa_chain`，五个 Attention shape 都抄 `always_fuse`。prefill S = 512 是 0.02468800079 ms（6.966947518×），对照 unfuse 的 0.1151360013 ms。prefill S = 2048 是 0.1136640012 ms（9.193693331×），对照 unfuse 的 0.1985919997 ms。decode 也抄 fuse。GEMM 访存侧仍然对上更快的 epilogue（M = 1、32、128：1.164732146×、1.267627163×、1.285407694×）。第二次失误是 ridge 和计算侧 GEMM：SCIC 抄 `always_unfuse`，贴近 1×，而 `always_fuse` 是 1.307134385×、1.327495649×、1.245223289×、1.234890808×。这是没有测量反馈的数据手册 AI；v2 的 GEMM 仍抄 `always_unfuse`。第四次归档 `paper/data/bakeoff_v4/` 在空闲 GPU 上用 `--calibrate all-gemm` 跟墙钟赢家（prefill 与 ridge M=512 抄 `always_fuse`，decode 与 ridge M∈{128,256} 抄 `always_unfuse`）；探针不是 DRAM AI。前两次 bakeoff 在忙 GPU 上。图没有检查，`scic_selected` 抄基线 median。不宣称 FlashAttention-3，不把 profiler 写成 SCIC 准确率。融合 kernel、实测 ridge、tokens/s 仍是 TBD。

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
| 设备 matmul | `bench_gemm.csv`，`gpu_measurements.json`；第三次 A40 归档在 `paper/data/a40/`，图 `paper/fig_a40_torch_matmul_tflops.png` |
| A40 crossings | 归档里 `crossings.json` 的 `A40_FP16`，与 `summary.json` 一致（数据手册，不是实测） |
| 一致率 | `agreement.json`。`quote_as_scic_accuracy` 为 false（`PROXY_TORCH_MEMORY_NOT_VALID_SCIC_ACCURACY`）。这次上传没有 `decision_log.csv` |
| 融合代理 | `fusion_proxy.csv`（`PROXY_TIMING_NOT_A_FUSION_SPEEDUP`；Triton 行 `TRITON_ELEMENTWISE_STUB_NOT_GEMM_FUSION`；不是 MemMode / ComputeMode） |
| Nsight | `nsight_dram.json` 仅 `TRY_NSIGHT=1` 且 `ncu` 成功。这一包是 `nsight_error.log`（`ERR_NVGPUCTRPERM`）和 `nsight_skipped.txt`。管理员步骤：`docs/NSIGHT_PERMISSIONS.md` |
| copy 估计 | `bandwidth_estimate.json`，`ESTIMATE_COPY_NOT_NSIGHT`，估计 ridge 262.5846181025793，`measured_ridge_flops_per_byte` 为 null |
| Docker bakeoff，修复前 | `paper/data/bakeoff/`（`bakeoff_latency.csv`、`bakeoff_decisions.csv`、`bakeoff_summary.json`、`RUN_NOTES.txt`、`nvidia_smi.txt`）。`BAKEOFF_WALLCLOCK`。无 Nsight。修复前的 Attention 选择器。`quote_as_scic_accuracy` 为 false。不是 FA-3。不是 Stage-Stream |
| Docker bakeoff，`compile_sdpa_chain` 之后 | `paper/data/bakeoff_v2/`（`bakeoff_latency.csv`、`bakeoff_decisions.csv`、`bakeoff_summary.json`、`RUN_NOTES.txt`、`gpu_selection.json`）。Attention 抄 `always_fuse`。GEMM 的 ComputeMode 仍抄 `always_unfuse`。忙 GPU（所选设备 92%）。不是 SCIC 准确率。这份 CSV 早于 GEMM 墙钟校准 |
| Docker bakeoff，`CALIBRATE=all-gemm` | `paper/data/bakeoff_v4/`（`bakeoff_latency.csv`、`bakeoff_decisions.csv`、`bakeoff_summary.json`、`RUN_NOTES.txt`）。空闲 GPU。GEMM `measured_wallclock`；Attention `library_sdpa`。不是 SCIC 准确率 |

## 12. 写作时不要出现的句子

- 「我们提出了 Stage-Stream。」
- 「小 M 时 epilogue 占 GEMM 流量的比例更高。」（冷缓存模型不支持。）
- 「H100 上融合加速了 X 倍。」（在 kernel 测出来之前。）
- 把 `run_cpu.sh` 的 NumPy 时延标成 H100。
- 把 `torch.matmul` 的扫参写成 MemMode 对 ComputeMode 的对比。
- 把 A40 上 115.38166932883347 / 989 说成「这张卡只跑到峰值的 12%」。那是相对 H100 数据手册（`fraction_of_h100_fp16_datasheet_peak` = 0.11666498415453333）。相对 A40 数据手册 149.7 的比例是 0.7707526341271441，而且仍不是实测 ridge。
- 把 `torch.profiler` 的显存、缓存分配器差分或 `NSIGHT_DRAM_PROCESS_SUM` 的一致率写成 SCIC 准确率。这次归档的 `quote_as_scic_accuracy` 是 false，状态是 `PROXY_TORCH_MEMORY_NOT_VALID_SCIC_ACCURACY`。0.2857142857142857 只是检查用数字。
- 把 `fusion_proxy.csv` 的时间比写成融合加速。状态是 `PROXY_TIMING_NOT_A_FUSION_SPEEDUP`。Triton 行是 `TRITON_ELEMENTWISE_STUB_NOT_GEMM_FUSION`，不是融合 GEMM。
- 把 copy kernel 的 `ESTIMATE_COPY_NOT_NSIGHT` 写成 Nsight 或实测 ridge。570.1019392595165 GB/s 和 262.5846181025793 都是估计，`measured_ridge_flops_per_byte` 保持 null。
- 把 `ERR_NVGPUCTRPERM` 写成「DRAM 字节测得为 0」。`ncu` 没有读到计数器，`achieved_ai` 仍是 TBD。
- 把上一包 Triton 的 `NameError('tl is not defined')` 写成这一包的结果。这一包七行 `ok=true`，但时间比仍不是融合加速。
- 把 bakeoff 的 `sdpa` 行写成 FlashAttention-3 kernel。它是 `scaled_dot_product_attention`。
- 把 bakeoff 的 `always_fuse` / `always_unfuse` 写成手写的 MemMode / ComputeMode kernel。它们是 `torch.compile` 或 eager。
- 写出 `paper/data/bakeoff/`、`paper/data/bakeoff_v2/` 或 `paper/data/bakeoff_v4/` 的 CSV 里没有的 median 或加速比。不要把第一份归档的 `always_fuse` 改标成第二次或第四次的 `scic_selected`。
- 把 bakeoff 写成 Nsight 实验或 SCIC 准确率。`NO_NSIGHT` 和 `quote_as_scic_accuracy=false` 必须留着。
- 把修复前或第二次归档写成空闲 GPU。第四次才是空闲 GPU 时延（`paper/data/bakeoff_v4/`）。`graph_inspected` 在各归档里均为 false。
- 把第二次的 GEMM 计算侧写成已经改选 `always_fuse`。那份 CSV 里 SCIC 抄的是 `always_unfuse`。校准结果在 `paper/data/bakeoff_v4/`。不要把 v2 的格子改标。

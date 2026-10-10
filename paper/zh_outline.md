# 中文提纲：按 shape 分类算子强度，并做双计划融合

对应英文草稿 `paper/draft.md`。这是提纲，不是译文。解析数字与 `shapefuse` 的闭式模型一致。已有的设备数字来自第二次 A40 上传，归档在 `paper/data/a40/`（`--machine A40_FP16`，`PARTIAL_VENDOR_MATMUL`）。没有 Nsight。`fusion_proxy.csv` 的时间比不是融合加速。没有的测量保持 TBD，不要用数据手册、CPU 代理或 `proxy_ai` 填上。

## 题目（暂定）

按 shape 的强度分类，服务于多态算子融合。
英文：Shape-Conditioned Intensity Classification for Plan-Polymorphic Operator Fusion。

一句话：算子强度随 shape 越过 Roofline ridge；融合必须能在 MemMode 与 ComputeMode 之间切换。

## 0. 投稿定位与边界

- 目标 venue：MLSys 或 ASPLOS。系统论文的写法：有机器模型、有决策算法、有消融计划。当前稿只有前两样的解析版本。
- 相对 ComFuse：他们做 MatMul + epilogue 的联合融合。我们加的是 SCIC（编译期可查询的强度分类）和双计划切换。
- **明确不写进贡献：** Stage-Stream。它是对方用来承载联合调度的机制。本文不实现、不改写、不宣称。
- **也不写进贡献：** 新的 Attention 算法。IO-aware Attention（FlashAttention 一类）是已有计划，不是本文的结果。
- 本文 artifact 仍没有 MemMode / ComputeMode kernel，没有 tokens/s，没有 Nsight。第二次 A40 上传有 `torch.matmul` 时延和一份融合代理（§7），状态 `PARTIAL_VENDOR_MATMUL`，`quote_as_scic_accuracy` 为 false。

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
- A40 数据手册：M(0.5R) = 113.50330752177987，M\* = 240.32584259093773，M(2R) = 544.5531470469126，R = 215.08620689655172。第二次上传的 `crossings.json` 用 `--machine A40_FP16`，里面有 `A40_FP16` 这一项。这是数据手册闭式，不是卡上测到的 ridge。
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

§7 那七个 shape 上，`regime_vs_h100_datasheet` 和 `regime_vs_requested_datasheet` 一致（M∈{1,4,32} 访存，M∈{344,512} ridge，M∈{2048,4096} 计算）。严格拐点不一致：M=344 的解析 AI=294.5284281，低于 H100 ridge 295.22，高于 A40 数据手册 ridge 215.08620689655172。`decision_log.csv` 里 `near_mstar` 的 `analytical_roofline_limit=compute`，三档仍是 ridge，`analytical_emit_both=true`。两列都不是实测 ridge。

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

第二次上传已经带了 `decision_log.csv` 和 `agreement.json`：`achieved_ai` 为空，`proxy_ai_status=TORCH_PROFILER_EMPTY`，`scic_accuracy_status=TBD_NO_MEASURED_AI`，`quote_as_scic_accuracy=false`，`n_with_byte_ai=0`。上面三条仍然是 TBD。`quote_as_scic_accuracy` 只有状态 `NSIGHT_DRAM` 才为真。`torch.profiler` 内存、缓存分配器差分和 `NSIGHT_DRAM_PROCESS_SUM` 的一致率不能填进这三条。

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
| 本文证据 | | 闭式 AI、决策表，一条 A40 `torch.matmul` 曲线，以及不能当成融合加速的 fusion proxy（§7）。融合 kernel 时间 TBD |

预期真正会分出胜负的，是 ridge 带和 Attention 的额外 epilogue：两边政策不一致的地方。远在访存侧时大家都融；远在计算侧时双计划会拒绝联合融合可能仍会尝试的融合。这个对比**还没跑**。

## 7. 实验计划与第二次 A40 包

解析部分：

```bash
pip install -r requirements.txt && pip install -e .
python -m shapefuse selfcheck
bash scripts/run_cpu.sh          # 无 GPU，PLACEHOLDER_NO_GPU
# 强烈建议。这次归档没有 Nsight DRAM：
TRY_NSIGHT=1 bash scripts/run_a40.sh
bash scripts/run_h100.sh         # 仍是 H100 锚点的 matmul 扫描，默认不开 profiler
```

`ncu` 是 Nsight Compute，不是驱动，也不是 `nvidia-smi`。PATH 上没有 `ncu` 时，脚本会说明；若设了 `TRY_NSIGHT=1`，会写 `nsight_skipped.txt` 并继续跑 matmul。DRAM 字节保持为空。

`selfcheck` 锁住第 3 节用到的闭式，包括 A40 数据手册 ridge 和 M\*。

### 7.1 第二次寄回的 A40 结果（`PARTIAL_VENDOR_MATMUL`）

归档：`paper/data/a40/`（`bench_gemm.csv`、`summary.json`、`gpu_measurements.json`、`crossings.json`、`decision_log.csv`、`agreement.json`、`fusion_proxy.csv`）。图：`paper/fig_a40_torch_matmul_tflops.png`。

- 设备：NVIDIA A40。PyTorch 2.5.1+cu121，CUDA 12.1，Python 3.10.12。
- 命令：`--bench-level a40 --machine A40_FP16 --profile torch --fusion-proxy`。ridge = 215.08620689655172，M\* = 240.32584259093773。`crossings.json` 含 `A40_FP16`。
- 计时的是 `torch.matmul`，FP16，N=K=4096。`kernel=pytorch_matmul`，`fused_plan_timed=false`。
- `nsight_status=TBD_RUN_PROFILER`。`nsight_dram_bytes` 和 `achieved_ai` 为空（`TBD_NSIGHT`）。`proxy_ai_status=TORCH_PROFILER_EMPTY`。
- `agreement.json`：`quote_as_scic_accuracy=false`，`scic_accuracy_status=TBD_NO_MEASURED_AI`，`n_with_byte_ai=0`。
- 下表的 median_s 和 TFLOP/s 是 CSV 原文。两列 regime 都是数据手册三档，不是实测 ridge。

| label | M | median_s | TFLOP/s | 解析 AI | 相对 A40 数据手册 | 相对 H100 数据手册 |
| --- | --- | --- | --- | --- | --- | --- |
| decode_gemv | 1 | 7.160000131e-05 | 0.4686373099 | 0.9995 | 访存 | 访存 |
| micro_batch | 4 | 7.948799804e-05 | 1.688528222 | 3.992 | 访存 | 访存 |
| decode_batch32 | 32 | 7.948800176e-05 | 13.50822514 | 31.51 | 访存 | 访存 |
| near_mstar | 344 | 0.000143648006 | 80.35422786 | 294.5 | ridge | ridge |
| ridge_m512 | 512 | 0.0001795520037 | 95.68185721 | 409.6 | ridge | ridge |
| prefill_m2048 | 2048 | 0.0006596159935 | 104.1810347 | 1024 | 计算 | 计算 |
| square_large | 4096 | 0.001194959998 | 115.0155266 | 1365 | 计算 | 计算 |

`achieved_tflops_max` = 115.01552661761667。`fraction_of_requested_datasheet_peak` = 0.7683067910328436，即相对 A40 数据手册 149.7 约 76.8%。`fraction_of_h100_fp16_datasheet_peak` = 0.11629476907746883，是同一吞吐相对 H100 数据手册 989。`measured_ridge_flops_per_byte` 仍是 null。

可以写的句子：吞吐从 M=1 的 0.4686373099 TFLOP/s 升到 M=4096 的 115.0155266；M≤32 时中位时间停在约 7.2e-5–7.9e-5 s，FLOP 已经增加 32 倍，这是 launch 主导的一端。三档上 A40 数据手册和 H100 数据手册一致。

不要写的句子：融合加速、MemMode 对 ComputeMode、Nsight achieved AI、H100 时延、A40 上测到了 ridge、SCIC 准确率。

H100 matmul 表继续空着。Achieved AI = FLOPs / 实测 DRAM 字节，不是 FLOPs / 时延。

#### 融合代理（`PROXY_NOT_FUSION_SPEEDUP`）

`fusion_proxy.ran=true`，`status=PROXY_NOT_FUSED_GEMM`，`triton_used=false`，`torch_compile_used=true`。`claims_memmode_kernel`、`claims_computemode_kernel`、`claims_stage_stream` 都是 false。每一行 `time_ratio_status=PROXY_TIMING_NOT_A_FUSION_SPEEDUP`。时间比是两段基线除以该变体，不是融合加速。

matmul+RMSNorm：

- `unfused_two_regions`：两段分开计时再相加。
- `eager_chain`：仍是两次 eager launch。CSV 时间比：M=1 为 1.388005185，M=4 为 1.46125371，M=4096 为 1.006475964（中间各档见 CSV）。
- `torch_compile`：没有检查图，不能当成已经融合。M=1 的时间比是 0.8089665914（比两段之和更慢），M=4096 是 1.201995369。

bias+GELU（GEMM 输出已经物化，不是 GEMM 融合）：

- hand stub 时间比从 M=1 的 1.162706454 到 M=4096 的 1.011971539。
- Triton 七行全部 `ok=false`，`TRITON_KERNEL_FAILED`，错误是 `NameError('tl is not defined')`，出在 `BLOCK: tl.constexpr`。没有 Triton 时间。

### 7.2 下一轮脚本

强烈建议 `TRY_NSIGHT=1`。没有 `ncu` 就先安装 Nsight Compute，用 `ncu --version` 确认，再用 `ncu --query-metrics | grep dram__bytes` 看指标名。

相对这次归档，代码侧已改、但还不是测量结果：Triton bias+GELU 在模块作用域 `import triton.language as tl`；profiler 改读 `self_device_memory_usage`，事件仍为空时才用缓存分配器差分，状态 `TORCH_ALLOCATOR_DELTA_NOT_DRAM`，仍然不是 DRAM，也不能当 SCIC 准确率。`TORCH_PROFILER_EMPTY` 表示两个来源都没有字节。

脚本末尾会跑 copy kernel 带宽探针，写入 `bandwidth_estimate.json`，状态 `ESTIMATE_COPY_NOT_NSIGHT`。ridge 的分子是数据手册 FLOP 峰值。不要写进 `measured_ridge_flops_per_byte`。

| 产物 | 诚实等级 |
| --- | --- |
| `bench_gemm.csv` | 仍是 `torch.matmul`。没有 Nsight JSON 时 `achieved_ai` 留空 |
| `proxy_ai` | profiler 事件为 `TORCH_PROFILER_MEMORY_NOT_DRAM`；只有分配器差分时为 `TORCH_ALLOCATOR_DELTA_NOT_DRAM`；都没有则 `TORCH_PROFILER_EMPTY`。不是 DRAM |
| `fusion_proxy.csv` | 时间比状态固定 `PROXY_TIMING_NOT_A_FUSION_SPEEDUP`。不是融合 GEMM。`memmode_kernel=false` |
| `decision_log.csv`、`agreement.json` | 只有 `NSIGHT_DRAM` 时 `quote_as_scic_accuracy=true` |
| `nsight_dram.json` | 仅 `TRY_NSIGHT=1` 且 `ncu` 成功。状态 `NSIGHT_DRAM_PROCESS_SUM` |
| `nsight_skipped.txt` | `TRY_NSIGHT=1` 但没有 `ncu`，或 ncu 失败 |
| `bandwidth_estimate.json` | `ESTIMATE_COPY_NOT_NSIGHT`。不是 Nsight，不是实测 ridge |

### 7.3 状态表

| 实验 | 状态 |
| --- | --- |
| 解析 AI、分档、M\*、计划表、图 | 完成 |
| A40 上 matmul 时延 / TFLOP/s，A40 数据手册三档 | 完成，§7.1，`PARTIAL_VENDOR_MATMUL`，`--machine A40_FP16` |
| A40 fusion proxy | 跑过。Triton 行 `NameError('tl is not defined')`。时间比不是融合加速 |
| H100 上 matmul 时延 / TFLOP/s | TBD，脚本 `scripts/run_h100.sh` |
| Nsight DRAM 字节、achieved AI | TBD。`nsight_status=TBD_RUN_PROFILER`。下一轮 `TRY_NSIGHT=1`。没有 `ncu` 是安装缺口，不是测得 0 |
| 机器实测 ridge | TBD。copy 探针若写出，状态是 `ESTIMATE_COPY_NOT_NSIGHT`，不是这一行 |
| SCIC 对孤立 Nsight AI 的准确率 | TBD。归档 `agreement.json` 是 `TBD_NO_MEASURED_AI`，`quote_as_scic_accuracy` 为 false |
| MemMode / ComputeMode / 总是融 / 从不融 | TBD；fusion proxy 不是这一行 |
| 短 S 与长 S 的 Attention 主循环 | TBD |
| MoE 倾斜与空专家权重 | TBD |
| 量化 epilogue 对实测峰值 | TBD |
| 计划缓存命中率和切换开销 | TBD |
| 端到端 tokens/s | TBD |
| 用实测替换 5 µs launch | TBD |

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
- 没有 MemMode / ComputeMode kernel，没有 Nsight trace，没有 tokens/s。§7 的 A40 数字是 `torch.matmul` 的时延和 TFLOP/s，加上不能当成融合加速的 fusion proxy。`quote_as_scic_accuracy` 为 false。Triton 行在出时间之前就失败了。

## 10. 结论句（可直接用）

在 H100 这一档的 Roofline 上，宽度 4096 的 FP16 方阵投影大约在 M\* ≈ 345 换边；同一闭式放到 A40 数据手册（149.7 TFLOP/s，696 GB/s，R = 215.08620689655172）上，M\* = 240.32584259093773，第二次上传的 `crossings.json` 里有这一项。IO-aware Attention 的强度随 S 换边，把 score 写回 HBM 的实现则不会；倾斜的 MoE 一层之内就同时有两种 regime。按算子种类做死的融合跟不上这些穿越。SCIC 把 regime 从 shape 和流量里算出来，双计划把它变成 MemMode、ComputeMode 或两者都要。Stage-Stream 不属于这个设计。设备上目前是 `--machine A40_FP16` 的一条 `torch.matmul` 曲线，状态 `PARTIAL_VENDOR_MATMUL`：M=1 为 0.4686373099 TFLOP/s，M=4096 为 115.0155266 TFLOP/s，相对 A40 数据手册峰值的比例是 0.7683067910328436，不是实测 ridge。Nsight 没抓（`TBD_RUN_PROFILER`），`proxy_ai` 是 `TORCH_PROFILER_EMPTY`，`quote_as_scic_accuracy` 为 false。fusion proxy 跑过，时间比是 `PROXY_TIMING_NOT_A_FUSION_SPEEDUP`，Triton bias+GELU 因 `NameError('tl is not defined')` 失败。融合 kernel、实测 ridge、tokens/s 仍然是 TBD。

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
| 设备 matmul | `bench_gemm.csv`，`gpu_measurements.json`；第二次 A40 归档在 `paper/data/a40/`，图 `paper/fig_a40_torch_matmul_tflops.png` |
| A40 crossings | 归档里 `crossings.json` 的 `A40_FP16`（数据手册，不是实测） |
| 决策日志 | `decision_log.csv`，`agreement.json`（仅 `quote_as_scic_accuracy` 为真时才能当准确率；这次归档为 false） |
| 融合代理 | `fusion_proxy.csv`（`PROXY_TIMING_NOT_A_FUSION_SPEEDUP`；不是 MemMode / ComputeMode） |
| Nsight | `nsight_dram.json`，仅 `TRY_NSIGHT=1` 且 `ncu` 成功。没有 `ncu` 时是 `nsight_skipped.txt` |
| copy 估计 | `bandwidth_estimate.json`，`ESTIMATE_COPY_NOT_NSIGHT`，不是实测 ridge |

## 12. 写作时不要出现的句子

- 「我们提出了 Stage-Stream。」
- 「小 M 时 epilogue 占 GEMM 流量的比例更高。」（冷缓存模型不支持。）
- 「H100 上融合加速了 X 倍。」（在 kernel 测出来之前。）
- 把 `run_cpu.sh` 的 NumPy 时延标成 H100。
- 把 `torch.matmul` 的扫参写成 MemMode 对 ComputeMode 的对比。
- 把 A40 上 115.01552661761667 / 989 说成「这张卡只跑到峰值的 12%」。那是相对 H100 数据手册（`fraction_of_h100_fp16_datasheet_peak` = 0.11629476907746883）。相对 A40 数据手册 149.7 的比例是 0.7683067910328436，而且仍不是实测 ridge。
- 把 `torch.profiler` 的显存、缓存分配器差分或 `NSIGHT_DRAM_PROCESS_SUM` 的一致率写成 SCIC 准确率。这次归档的 `quote_as_scic_accuracy` 是 false。
- 把 `fusion_proxy.csv` 的时间比写成融合加速。状态是 `PROXY_TIMING_NOT_A_FUSION_SPEEDUP`。
- 把 copy kernel 的 `ESTIMATE_COPY_NOT_NSIGHT` 写成 Nsight 或实测 ridge。
- 把 Triton 的 `NameError('tl is not defined')` 写成融合失败的性能结论。那一行没有时间。

# 勘误（相对下文原稿）

下文是研究笔记原稿，保留以便对照。仓库里的 `shapefuse` 和 `paper/draft.md` 以代码中的公式为准。

2026-10 补记：服务器上有一次 NVIDIA A40 的 `torch.matmul` 扫描，状态 `PARTIAL_VENDOR_MATMUL`，归档在 `paper/data/a40/`。计时的不是融合 kernel。那次命令的 `--machine` 是 `H100_SXM_FP16`，所以 regime 列用的是 H100 数据手册，不是 A40 实测 ridge。数字和边界见 `paper/draft.md` §7.2 与 `paper/zh_outline.md` §7。下文「本机无 GPU」是原稿当时的标注。

原稿 §1.4 和 §4 写「小 M 时，不融合的 epilogue 往返占 GEMM 流量的比例更高」。冷缓存模型不是这样：

- 融合省掉的是 C 的写回再读入。N = K = d、FP16 时，这部分占未融合流量的比例约为 \(2M/(d+4M)\)，随 M **增大**。M = 1、d = 4096 时大约只有 0.05%；M = 2048 时大约 33%。
- 小 M 的 GEMM 落在访存侧，是因为权重矩阵 KN 主导流量，不是因为 epilogue。
- 严格 Roofline（没有 launch）上，GEMM+RMSNorm 的融合加速在带宽区只有几个百分点到大约 1.1×；两边都变成计算瓶颈之后降为 1.00×。
- 若额外假设每次 kernel launch 5 µs（`ASSUMED_NOT_MEASURED`），小 M 会重新出现大约 1.3× 的收益。这是敏感度，不是测量。
- 融合真正改变绑定类型的解析证据是 Attention（score 落不落 HBM）和 ragged MoE（同一层里专家分属不同档），不是 RMSNorm 那一次往返。

原稿里下列数字与本仓库一致，可以继续用：H100 / A100 的 ridge、d = 4096 时 H100 的 M\* ≈ 345、A100 的 M\* ≈ 165、Softmax AI ≈ 0.83、FA 式 IO 的 AI = S/2、MoE 四张场景表、GEMM shape 表里的 FP16 AI。

CPU 时延仍只是代理。原稿已经标明无 GPU；正式论文不要把那些 GFLOP/s 写成 H100。

---

# Shape 决定算子算力/访存属性：论据 · 创新点 · 实验证据

> 方法学标注：**本机无 GPU**。算术强度（AI）为 **解析 Roofline（相对 H100/A100 ridge）**；时延为 **NumPy FP32 CPU `@` 代理**，用于定性验证「小 M 吞吐崩塌」，**不能**直接外推到 Hopper Tensor Core。原稿当时的产物目录是 `/workspace/op-fusion-shape-intensity/`。现在的入口是 `python -m shapefuse`，产物在 `results/`。

## 1. 核心论据（Roofline / Arithmetic Intensity）

### 1.1 定义与机器平衡点

算术强度：

\[
\mathrm{AI}=\frac{\mathrm{FLOPs}}{\mathrm{Bytes\ moved}}
\quad[\mathrm{FLOPs/byte}]
\]

机器 ridge（计算/带宽平衡点）：

\[
R=\frac{\mathrm{Peak\ FLOPs/s}}{\mathrm{Peak\ Bytes/s}}
\]

| 机器 | Peak (FP16 TC dense) | HBM BW | Ridge \(R\) |
|------|----------------------|--------|-------------|
| H100 SXM | ≈989 TFLOPS | ≈3.35 TB/s | **≈295 FLOPs/byte** |
| A100 SXM | ≈312 TFLOPS | ≈2.04 TB/s | **≈153 FLOPs/byte** |

判据（本报告所用阈值）：

- \(\mathrm{AI}<0.5R\) → **memory-bound**
- \(\mathrm{AI}>2R\) → **compute-bound**
- 否则 → **ridge-region**（对融合/调度极敏感）

**关键主张：** 「算子是计算密集还是访存密集」不是算子类型的固有标签，而是 **(算子代数 × shape × dtype × 是否融合写回 HBM)** 的函数。同一套 GEMM/Attention/MoE Group-GEMM，在 decode 微 batch / 稀疏 expert / 短序列上可从 compute-bound **跌落** 到 memory-bound；融合策略必须 **shape-adaptive**。

### 1.2 GEMM：显式穿越 ridge 的公式

\(C_{[M,N]}=A_{[M,K]}B_{[K,N]}\)，冷缓存下界流量（读 A、B，写 C 各一次），FP16（`elem=2`）：

\[
\mathrm{FLOPs}=2MNK,\quad
\mathrm{Bytes}=(MK+KN+MN)\cdot 2,\quad
\mathrm{AI}=\frac{MNK}{MK+KN+MN}
\]

特殊情形 \(N=K=d\)（LLM 方阵投影常见）：

\[
\mathrm{AI}(M)=\frac{Md}{d+2M}
\qquad
\text{令 }\mathrm{AI}=R \Rightarrow
M^{\star}=\frac{R\,d}{d-2R}\ (d>2R)
\]

**本实验数值（\(d=4096\)）：**

| 目标 ridge | \(M^{\star}\)（AI 恰好过 ridge） |
|------------|----------------------------------|
| H100 (≈295) | **\(M^{\star}\approx 345\)** |
| A100 (≈153) | **\(M^{\star}\approx 165\)** |

含义：同样是「线性层 GEMM」，**LLM decode（\(M=1\)–几十）在 H100 上系统性 memory-bound**；prefill \(M\gtrsim 2\times 10^3\) 才稳进 compute-bound。融合 epilogue（RMSNorm/SiLU）在小 \(M\) 时几乎总是 **省 HBM 流量的赢家**；大 \(M\) 时则要和 Tensor Core 流水线占位、寄存器压力权衡（ComFuse 在 Self-Attn 上输给 FA 的同一逻辑）。

> 勘误：上一段「小 M 时几乎总是省 HBM 流量的赢家」过强。见文首。小 M 的收益如果存在，主要来自 launch / 尾延迟，而不是字节比例。字节比例随 M 增大。ComFuse 在 Self-Attn 上输给 FA 的逻辑仍然成立，对应本文 ComputeMode。

### 1.3 Attention：QKᵀ / Softmax / FA-IO

设 \(Q,K,V\in\mathbb{R}^{B\times H\times S\times D}\)。

| 子图 | FLOPs（量级） | Bytes（量级） | AI 行为 |
|------|---------------|---------------|---------|
| QKᵀ 写回 scores | \(2BHS^{2}D\) | \((2BHS D + BHS^{2})\cdot 2\) | AI 随 \(S\) 缓增，但仍常 \(<R_{\mathrm{H100}}\)（未融合） |
| Softmax | \(\sim 5BHS^{2}\) | \(\sim 3BHS^{2}\cdot 2\) | **AI≈0.83，恒 memory-bound** |
| FA-style（scores 不落 HBM） | \(\sim 4BHS^{2}D\) | \(\sim 4BHS D\cdot 2\)（Q,K,V,O） | AI \(\propto S\)，长序列变 compute-bound |

**实验解析结果（B=1,H=32,D=128，FP16 模型）：**

| S | AI(QKᵀ→HBM) | AI(Softmax) | AI(FA-IO) | 相对 H100 |
|---|-------------|-------------|-----------|-----------|
| 128 | 42.7 | 0.83 | 64 | 三者偏 mem / FA 仍 mem |
| 512 | 85.3 | 0.83 | 256 | Softmax mem；FA 进 ridge |
| 2048 | 113.8 | 0.83 | **1024** | 未融合 QK 仍 mem；**FA compute** |
| 8192 | 124.1 | 0.83 | **4096** | 融合与否决定整层绑定类型 |

结论：**同一 Attention「算子」，短序列 + 写回 scores → 访存型；长序列 + IO-aware 融合 → 计算型。** Softmax 单独永远不该成为独立 kernel 的默认选择。

### 1.4 RMSNorm epilogue

\[
\mathrm{AI}_{\mathrm{RMS}}\approx\frac{4MK}{(2MK+K)\cdot 2}\approx 1\ \mathrm{FLOPs/byte}
\]

实验各 shape 下 RMSNorm 均为 **memory-bound**。若 MatMul 与 RMSNorm **不融合**：额外付出约 \(2\cdot MN\cdot 2\) 字节的「写 C 再读 C」。

> 勘误：原稿下一句「小 M 时该残差相对 GEMM 主体流量占比更高」不成立，已删除其推论。正确的比例和 Roofline 加速见 `paper/draft.md` §3.4。RMSNorm 本身 AI≈1、恒为访存，这一点仍然成立。

### 1.5 MoE Grouped-GEMM（ragged）

专家 \(e\)：token 数 \(t_e\)，\([t_e,H]\!\times\![H,I]\)。聚合 AI：

\[
\mathrm{AI}=\frac{\sum_e 2 t_e H I}{\sum_e (t_e H + H I + t_e I)\cdot 2}
\]

**实验（H=4096, I=14336，FP16）：**

| 场景 | token 分布 | AI_agg | 最瘦专家 AI | vs H100 |
|------|------------|--------|-------------|---------|
| decode_tiny | 8×2 | **2.0** | 2.0 | **全 memory** |
| skewed_8e | 1024…8 | 236 | **8.0** | 聚合 ridge，**尾部专家 memory** |
| balanced_8e | 8×256 | 237 | 237 | ridge |
| prefill_fat | 8×2048 | **1247** | 1247 | **compute** |

**关键：** 静态「MoE = 计算密集 Group-GEMM」在 decode / 负载倾斜下不成立；**同一 layer 内不同 expert 可处于不同绑定类型** → 需要 per-plan / per-expert 的融合与 tile 策略，而非单一 epilogue 模板。

## 2. 创新点（论文贡献草案，3–5 条）

均从「**intensity 随 shape 穿越 ridge**」推出。和第 1 节勘误一致的取舍是：I1（SCIC）和 I2（双计划）是本文要做的；Stage-Stream 不是。

### I1 — Shape-Conditioned Intensity Classifier（SCIC）

- **What：** 在融合前对每个 producer–consumer 边估计 \(\widehat{\mathrm{AI}}(shape,dtype,layout)\)，并相对目标 GPU 的 \(R\) 分类为 `{mem, ridge, compute}`；分类结果进入代价模型，而不是只看「是否算子相邻」。
- **Why novel：** ComFuse/Inductor 用静态启发式或 residual-time，**不显式对 ridge 做 shape 穿越预测**；FA 模板假定 Attention 已 IO-aware。SCIC 把 Roofline 变成 **可查询的编译期特征**。
- **Eval：** 用解析 AI vs Nsight `dram_bytes` / achieved FLOPs 校准；报告分类准确率与错误分类导致的错误融合比例。**尚未测量。**

### I2 — Dual-Mode（或 Multi-Plan）Fusion 发射

- **What：** 对同一子图生成 ≥2 套可切换 plan：
  - **MemMode：** 激进 epilogue 融合、偏小 tile、优先减 HBM；
  - **ComputeMode：** 保守融合或 fission、保 WGMMA 流水与寄存器、对齐 FA/CuBLAS。
  运行时按实际 \(M/S/t_e\)（或 Inductor guard）选 plan。
- **Why novel：** ComFuse 证明 joint fusion 有益，但 **Self-Attn 反例**说明缺 shape 开关；Inductor `dynamic=True` 有符号 shape **却几乎固定融合决策**。
- **Eval：** decode vs prefill、短/长 S、MoE skew；对比 always-fuse / never-fuse / oracle；指标：tokens/s、HBM bytes、plan-switch 开销。**尚未测量。选择器本身已在 `shapefuse/plans.py`。**

原稿 I2 写过「Stage-Stream 融合」。那不是本仓库的计划内容。MemMode 只表示「融合 epilogue、减小 tile、少碰 HBM」，不表示实现了 Stage-Stream。

### I3 — Intensity × Register/SMEM 联合代价模型

- **现状：** 未做。本文不声称有一个已经标定的 spill 模型。ComputeMode 拒绝把重 epilogue 融进主循环，是对照 Self-Attn 反例的政策，不是 spill 代价。
- **Eval（将来）：** 在 ridge-region shape（如 \(M\approx M^{\star}\)）上做消融：仅 AI、仅 spill、联合模型；看是否避免「融了反而慢」。

### I4 — 在线 plan 缓存（服务侧）

- **What：** 不重做 Stage-Stream，而是在现有 kernel 之上按 `(op, shape_bucket)` 记下 MemMode/ComputeMode 谁赢。`PlanCache` 是空的内存桩。
- **Eval：** 真实 trace 上的切换频率、相对 oracle 的 gap。**TBD。**

### I5 — 按专家分箱的 Group-GEMM

- **What：** 瘦专家走 MemMode，胖专家走 ComputeMode。允许一层里计划异构。解析分类已做；kernel 方言没有。
- **Eval：** Mixtral/Qwen-MoE 类层；W8A8/FP8；相对单一融合模板。**TBD。**

**投稿叙事一句：** *Operator intensity is shape-dependent; fusion must be plan-polymorphic across the Roofline ridge.*

## 3. 实验证据（原稿当时的本机实跑）

### 3.1 方法备注

| 项 | 说明 |
|----|------|
| GPU | **无**（无 CUDA / nvidia-smi） |
| AI | 解析模型，FP16 `elem_bytes=2`，冷缓存下界流量 |
| 时延 | NumPy FP32 CPU `matmul`，中位数；**PROXY** |
| Ridge | H100≈**295.2**，A100≈**153.0** FLOPs/byte |

现在复现解析表不需要重跑 NumPy：`python -m shapefuse selfcheck` 和 `bash scripts/run_cpu.sh`。CPU 代理要加 `--cpu-proxy`，并且仍然不能外推到 Hopper。

### 3.2 GEMM：同一算子，shape 改绑定类型

| label | M×N×K | AI_fp16 | vs H100 | CPU GFLOPS (proxy) |
|-------|-------|---------|---------|---------------------|
| decode_gemv | 1×4096×4096 | **1.00** | memory-bound | 64.9 |
| micro_batch | 4×4096×4096 | **3.99** | memory-bound | 131.0 |
| decode_batch8 | 8×4096×4096 | **7.97** | memory-bound | 221.7 |
| decode_batch32 | 32×4096×4096 | **31.5** | memory-bound | 507.8 |
| square_small | 256³ | **85.3** | memory-bound | 428.4 |
| moe_expert_med | 256×4096×14336 | **237** | ridge-region | — |
| square_med | 1024³ | **341** | ridge-region | 953.7 |
| prefill_long | 2048×4096×4096 | **1024** | compute-bound | — |
| square_large | 4096³ | **1365** | compute-bound | — (过大未测 CPU) |
| tall_skinny_K | 4096×4096×64 | **62.1** | memory-bound | 393.7 |

CPU GFLOPS 来自原稿当时的 NumPy 运行，不是本仓库新测的，也不是 H100。解析 AI 一列可以由当前代码复现。

**穿越点：** \(N=K=4096\) 时，H100 上 \(M^{\star}\!\approx\!345\)：服务微 batch 在左侧（mem），长 prefill 在右侧（compute）。

### 3.3–3.5

Softmax AI≈0.83，恒为访存。FA-IO 在本配置下 S≥2048 进入 compute（AI=1024 > 2R）。RMSNorm AI≈1，恒为访存。MoE：`decode_tiny` AI=2.0；`skewed` 最瘦专家 AI≈8，最胖可达数百并越过 2R；`prefill_fat` AI≈1247。

## 4. 对研究叙事的直接含义

1. 相关工作里不要把 GEMM/MoE 一律写成 compute-intensive；写 *regime-switching under shape*。
2. ComFuse 的 joint fusion 是必要非充分：缺 shape 多态 plan。不要把 Stage-Stream 写成我们的方法。
3. 最短的设备实验：在已有 kernel 上实现 MemMode 与 ComputeMode 两套融合，用 M\* 和 MoE skew 做消融；用 Nsight 校准 SCIC。本仓库还没有这些 kernel。
4. 原稿证据 = 解析 ridge 穿越 + 当时的 CPU 形态验证。正式论文必须在 H100/A100 上重测 achieved AI 与端到端，并保持 TBD 直到那些数字存在。

## 5. 一句话摘要

> 在 H100 的数据手册 Roofline 上，\(N=K=4096\) 的 FP16 GEMM 于 \(M^{\star}\approx 345\) 穿越 ridge；decode（\(M\sim 1\)–32）AI 仅 1–30 FLOP/byte（memory-bound），prefill（\(M\sim 2\)k）AI > 1000（compute-bound）。因此 epilogue / Attention / MoE 融合必须按 shape 切换 plan。切换的依据是 SCIC，不是 Stage-Stream。

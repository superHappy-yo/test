# Shape-adaptive operator fusion (SCIC + dual-plan)

Operator intensity is shape-dependent. The same GEMM, attention block, or MoE grouped GEMM crosses a GPU's Roofline ridge as the shape changes, so a fusion compiler has to emit more than one plan.

This repository is the analytical model and the experiment harness for that claim:

- **SCIC** classifies a site as memory-bound, ridge-region, or compute-bound from closed-form arithmetic intensity and a datasheet ridge.
- **Dual-plan selection** emits **MemMode** (fuse the epilogue, smaller tiles) or **ComputeMode** (keep the MMA pipeline, larger tiles), and emits both inside the ridge band.
- Scope is wider than a single GEMM epilogue: elementwise chains, reductions, norms, softmax, attention prefill versus decode, ragged MoE, and quantized epilogues.

ComFuse jointly fuses MatMul and epilogue. The piece added here is the shape-conditioned classification and the MemMode / ComputeMode switch. **Stage-Stream is not implemented and is not a contribution.** No fused CUDA kernel is generated. `dual_plans.csv` is a selector stub (`kernel_emitted=false`).

Datasheet ridges (not measured on your machine):

| anchor | dense peak | HBM | ridge |
| --- | --- | --- | --- |
| H100 SXM FP16 | 989 TFLOP/s | 3.35 TB/s | ≈ 295 FLOP/byte |
| H100 SXM FP8 | 1979 TFLOP/s | 3.35 TB/s | ≈ 591 FLOP/byte |
| A100 80GB SXM FP16 | 312 TFLOP/s | 2.039 TB/s | ≈ 153 FLOP/byte |

Sparsity peaks are not used. For N = K = 4096, FP16, the H100 M where AI equals the ridge is **M\* ≈ 345**. AI < 0.5 R for M ≲ 159 (memory-bound) and AI > 2 R for M ≳ 830 (compute-bound).

## Layout

```
shapefuse/            package: intensity, SCIC, plans, sweeps, bench, CLI
scripts/run_cpu.sh    analytical run, safe without a GPU
scripts/run_h100.sh   same tables plus a CUDA matmul sweep; refuses to run with no GPU
paper/draft.md        English MLSys/ASPLOS-style draft, with TBD markers
paper/zh_outline.md   Chinese outline
notes/draft_report_zh.md
results/SCHEMA.md     output contract; generated files are gitignored
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m shapefuse selfcheck
```

PyTorch is optional. The analytical path does not import it. On the GPU server, install a CUDA wheel that matches the driver ([pytorch.org](https://pytorch.org/get-started/locally/)) before `scripts/run_h100.sh`. The script will not download that wheel for you.

## Run without a GPU

```bash
bash scripts/run_cpu.sh
# equivalent:
python -m shapefuse all --out results --device cpu --bench-level smoke --cpu-proxy
```

This writes `results/`. `summary.json` will say `gpu_measurement_status: PLACEHOLDER_NO_GPU`. The NumPy and CPU latencies are a qualitative proxy. Do not put them in a table labeled H100.

Useful flags:

```bash
python -m shapefuse analytical --out results --skip-plots
python -m shapefuse plans --out results
python -m shapefuse bench --device cpu --bench-level smoke --out results
python -m shapefuse all --machine A100_SXM80_FP16 --out results/a100 --device cpu --skip-plots
python -m unittest discover -s tests -v
```

`--launch-overhead-us` (default 5) is a sensitivity knob on the GEMM+RMSNorm time ratio. It is not a measured launch latency. Set it to `0` to turn the term off.

## Run on H100

```bash
bash scripts/run_h100.sh
# if PyTorch is already installed and you do not want pip to touch it:
SKIP_INSTALL=1 bash scripts/run_h100.sh
# another datasheet anchor (still your physical GPU):
MACHINE=H100_SXM_FP8 SKIP_INSTALL=1 bash scripts/run_h100.sh
```

The script requires `nvidia-smi` and `torch.cuda.is_available()`. It then runs:

```bash
python -m shapefuse all --out results --device cuda --require-cuda --bench-level h100
```

What that sweep actually times is stock `torch.matmul` at M ∈ {1, 4, 32, 344, 512, 2048, 4096}, N = K = 4096, FP16. That supports the "small M loses throughput" curve. It does **not** compare MemMode against ComputeMode, because those kernels are not generated.

## What to send back for the paper

After the H100 script, send the whole `results/` directory. `results/SEND_BACK.txt` repeats the list. Minimum:

- `summary.json`, `gpu_measurements.json`, `bench_gemm.csv`
- `gemm_shapes.csv`, `attention_intensity.csv`, `attention_decode.csv`
- `rmsnorm_epilogue.csv`, `ew_chains.csv`, `moe_grouped.csv`, `quant_epilogue.csv`, `dual_plans.csv`
- `fig_*.png`
- the `nvidia-smi` header (driver, GPU name) and `torch.__version__` (already copied into `gpu_measurements.json` when CUDA runs)

Leave the following as TBD unless you measure them. Do not invent numbers.

| quantity | where it goes |
| --- | --- |
| DRAM bytes and achieved AI | `nsight_dram_bytes`, `achieved_ai` in `bench_gemm.csv` / `gpu_measurements.json` |
| measured HBM bandwidth and ridge | `measured_hbm_bandwidth_bytes_per_s`, `measured_ridge_flops_per_byte` |
| MemMode vs ComputeMode kernel time | not produced; needs a real kernel study |
| SCIC accuracy vs measured AI | paper evaluation, TBD |
| plan-switch overhead, tokens/s | paper evaluation, TBD |
| real CUDA launch latency | replace the 5 µs assumption |

Profiler metric names depend on the Nsight Compute version. List them before choosing one:

```bash
ncu --query-metrics | grep dram__bytes
```

Achieved AI is FLOPs / measured DRAM bytes. FLOPs / latency is throughput, not intensity.

## Reading the model

- Regime bands are policy knobs: memory if AI < 0.5 R, compute if AI > 2 R, otherwise ridge (both plans). `--mem-frac` and `--compute-frac` change them.
- The strict Roofline knee is AI = R and is reported separately as `roofline_limit`.
- Cold-cache GEMM+RMSNorm **byte-save fraction grows with M**. Small-M GEMMs are memory-bound because the weight matrix dominates, not because the epilogue round-trip is a large share of bytes. The strict roofline speedup of that fusion is about 1× once both versions are compute-bound. A 5 µs launch assumption brings a small-M gain back; that assumption is marked `ASSUMED_NOT_MEASURED`. The earlier Chinese note stated the opposite byte-fraction trend; see the erratum in `notes/draft_report_zh.md`.
- Attention is the case where fusion changes the regime. With D = 128 and FP16, materialized scores stay under 0.5 R for every S in the sweep. FA-style IO (Q, K, V, O only) has AI = S / 2 and becomes compute-bound for long S. Both plans keep scores off HBM. They differ in tile size and whether an extra epilogue is fused into the mainloop.
- Decode attention (MHA, FP16) sits near 1 FLOP/byte for any cache length. GQA raises that constant and it is still far below the H100 ridge.
- A skewed MoE layer has memory-bound and compute-bound experts at the same time. The selector's per-expert rows are the ragged plan. An inactive expert whose weight is still loaded pulls a one-hot dispatch from compute-bound back to the ridge; both accountings are in `moe_grouped.csv`.

## Python API

```python
from shapefuse import H100_SXM_FP16, classify_site, select_plan
from shapefuse.scic import Site

report = classify_site(
    Site("gemm", "decode", {"M": 32, "N": 4096, "K": 4096}),
    H100_SXM_FP16,
)
decision = select_plan(report, length=32)
print(report.regime, decision.primary.mode, decision.emit_both, decision.kernel_emitted)
```

## Paper

- English draft: [`paper/draft.md`](paper/draft.md). Every missing GPU number is marked **TBD**.
- Chinese outline: [`paper/zh_outline.md`](paper/zh_outline.md).

---

# 中文说明

算子是计算受限还是访存受限，取决于 shape，不是算子种类的固定标签。同一 GEMM、Attention、MoE Group-GEMM 会越过 GPU Roofline 的 ridge，融合决策必须是多态的：至少发出 **MemMode** 和 **ComputeMode** 两套计划。

本仓库是这条论断的解析模型和可在服务器上跑的实验入口。

- **SCIC**：用闭式算术强度相对机器 ridge，把站点分成 memory-bound、ridge-region、compute-bound。
- **双计划**：访存侧融合 epilogue、用较小 tile；计算侧保住 MMA 流水、用较大 tile、把重 epilogue 拆出去。落在 ridge 带内则两套都发。
- 范围不限于 GEMM epilogue：逐元素链、归约、Norm、Softmax、Attention 的 prefill/decode、ragged MoE、量化 epilogue。

相对 ComFuse：对方做的是 MatMul 与 epilogue 的联合融合。这里增加的是按 shape 的强度分类，以及 MemMode / ComputeMode 切换。**Stage-Stream 不是我们的贡献，代码里也没有实现。** 不会生成融合 CUDA kernel。`dual_plans.csv` 里 `kernel_emitted=false`。

上表的 H100 / A100 ridge 是数据手册锚点，不是你这台机器的实测峰值。N = K = 4096、FP16、H100 上 AI 等于 ridge 的点是 **M\* ≈ 345**。M ≲ 159 为 memory-bound（AI < 0.5 R），M ≳ 830 为 compute-bound（AI > 2 R）。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m shapefuse selfcheck
```

解析路径不依赖 PyTorch。GPU 服务器上请先按驱动安装 CUDA 版 PyTorch（见 [pytorch.org](https://pytorch.org/get-started/locally/)），再跑下面的 H100 脚本。脚本不会替你下载 wheel。

## 没有 GPU 时

```bash
bash scripts/run_cpu.sh
```

产物在 `results/`。`summary.json` 的 `gpu_measurement_status` 会是 `PLACEHOLDER_NO_GPU`。NumPy / CPU 时延只是定性代理，不能写进标成 H100 的表。

## 在 H100 上

```bash
bash scripts/run_h100.sh
# PyTorch 已经装好、不希望 pip 再改环境时：
SKIP_INSTALL=1 bash scripts/run_h100.sh
```

脚本检查 `nvidia-smi` 和 `torch.cuda.is_available()`，然后跑 `--bench-level h100`。这次计时的是 `torch.matmul`（M 取 1、4、32、344、512、2048、4096，N = K = 4096，FP16），用来看小 M 吞吐是否塌掉。它**不是** MemMode 对 ComputeMode 的对比，因为这两套 kernel 没有生成。

## 论文需要你寄回的东西

跑完后把整个 `results/` 目录寄回。`results/SEND_BACK.txt` 有清单。至少包括：

- `summary.json`、`gpu_measurements.json`、`bench_gemm.csv`
- `gemm_shapes.csv`、`attention_intensity.csv`、`attention_decode.csv`
- `rmsnorm_epilogue.csv`、`ew_chains.csv`、`moe_grouped.csv`、`quant_epilogue.csv`、`dual_plans.csv`
- `fig_*.png`
- `nvidia-smi` 抬头（驱动、GPU 名）和 PyTorch 版本（CUDA 跑通时会写进 `gpu_measurements.json`）

下面这些没有实测就保持 TBD，不要填估计数充数：

| 量 | 填到哪里 |
| --- | --- |
| DRAM 字节和 achieved AI | `bench_gemm.csv` / `gpu_measurements.json` 的 `nsight_dram_bytes`、`achieved_ai` |
| 实测 HBM 带宽和 ridge | `measured_hbm_bandwidth_bytes_per_s`、`measured_ridge_flops_per_byte` |
| MemMode 与 ComputeMode 的 kernel 时延 | 本仓库不产生，需要单独的 kernel 实验 |
| SCIC 相对实测 AI 的分类准确率 | 论文实验，TBD |
| 计划切换开销、tokens/s | 论文实验，TBD |
| 真实 CUDA launch 时延 | 替换 5 µs 的假设 |

Nsight 指标名随版本变化，先列出再选：

```bash
ncu --query-metrics | grep dram__bytes
```

Achieved AI = FLOPs / 实测 DRAM 字节。FLOPs / 时延是吞吐，不是强度。

## 读结果时注意

- 三档阈值是策略参数：AI < 0.5 R 为访存，AI > 2 R 为计算，中间是 ridge（两套计划都发）。可用 `--mem-frac`、`--compute-frac` 修改。
- 严格 Roofline 拐点是 AI = R，字段名 `roofline_limit`，和三档分类不是一回事。
- 冷缓存下 GEMM+RMSNorm **省下的字节比例随 M 增大**。小 M 的访存瓶颈来自权重，不是 epilogue 往返。两边都变成计算瓶颈后，严格 Roofline 加速比约为 1。加上未测量的 5 µs launch 假设后，小 M 才会重新出现收益，该列标记为 `ASSUMED_NOT_MEASURED`。旧中文笔记里「小 M 占比更高」的说法是错的，勘误在 `notes/draft_report_zh.md`。
- 融合真正改变绑定类型的是 Attention：D = 128、FP16 时，score 落 HBM 的实现在扫描的所有 S 上都低于 0.5 R；只搬 Q、K、V、O 时 AI = S / 2，长序列进入计算侧。两套计划都不把 score 写回 HBM，切换的是 tile 和额外 epilogue。
- Decode Attention（MHA、FP16）在任意 KV 长度上 AI 都接近 1。GQA 只把这个常数抬高，相对 H100 ridge 仍然是访存。
- 倾斜的 MoE 层里，瘦专家和胖专家可以分别是访存和计算。`moe_grouped.csv` 里还有「one-hot 但仍加载全部专家权重」的对照：会从 compute-bound 掉回 ridge。

英文草稿：[`paper/draft.md`](paper/draft.md)（缺的 GPU 数字都标了 TBD）。中文提纲：[`paper/zh_outline.md`](paper/zh_outline.md)。

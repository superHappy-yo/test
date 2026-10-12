# Shape-adaptive operator fusion (SCIC + dual-plan)

Operator intensity is shape-dependent. The same GEMM, attention block, or MoE grouped GEMM crosses a GPU's Roofline ridge as the shape changes, so a fusion compiler has to emit more than one plan.

This repository is the analytical model and the experiment harness for that claim:

- **SCIC** classifies a site as memory-bound, ridge-region, or compute-bound from closed-form arithmetic intensity and a datasheet ridge.
- **Dual-plan selection** emits **MemMode** (fuse the epilogue, smaller tiles) or **ComputeMode** (keep the MMA pipeline, larger tiles), and emits both inside the ridge band.
- Scope is wider than a single GEMM epilogue: elementwise chains, reductions, norms, softmax, attention prefill versus decode, ragged MoE, and quantized epilogues.

ComFuse jointly fuses MatMul and epilogue. The piece added here is the shape-conditioned classification and the MemMode / ComputeMode switch. **Stage-Stream is not implemented and is not a contribution.** No fused CUDA kernel is generated. `dual_plans.csv` is a selector stub (`kernel_emitted=false`).

Datasheet ridges (not measured on your machine):

| anchor | dense peak | memory | ridge |
| --- | --- | --- | --- |
| H100 SXM FP16 | 989 TFLOP/s | 3.35 TB/s | ≈ 295 FLOP/byte |
| H100 SXM FP8 | 1979 TFLOP/s | 3.35 TB/s | ≈ 591 FLOP/byte |
| A100 80GB SXM FP16 | 312 TFLOP/s | 2.039 TB/s | ≈ 153 FLOP/byte |
| A40 FP16 | 149.7 TFLOP/s | GDDR6 696 GB/s | ≈ 215 FLOP/byte |

Sparsity peaks are not used. The A40 149.7 figure is dense FP16 with FP16 accumulate; the datasheet sparsity number 299.4 is not used. For N = K = 4096, FP16, the H100 M where AI equals the ridge is **M\* ≈ 345**. The same formula on the A40 datasheet gives **M\* ≈ 240**. AI < 0.5 R for M ≲ 159 on H100 (memory-bound) and AI > 2 R for M ≳ 830 (compute-bound). Short names: `--machine a40`, `--machine h100`, `--machine a100`, `--machine h100_fp8`.

## Layout

```
shapefuse/            package: intensity, SCIC, plans, sweeps, bench, CLI
scripts/run_cpu.sh    analytical run, safe without a GPU
scripts/run_h100.sh   same tables plus a CUDA matmul sweep; refuses to run with no GPU
scripts/run_a40.sh    A40 datasheet anchor, matmul sweep, profiler proxy, fusion proxy
scripts/nsight_dram.sh  optional ncu capture; strongly recommended via TRY_NSIGHT=1
scripts/bandwidth_probe.sh  copy-kernel GDDR estimate; ESTIMATE_COPY_NOT_NSIGHT, not a measured ridge
docs/NSIGHT_PERMISSIONS.md  fix ERR_NVGPUCTRPERM, then TRY_NSIGHT=1 again
paper/draft.md        English MLSys/ASPLOS-style draft, with TBD markers
paper/zh_outline.md   Chinese outline
notes/draft_report_zh.md
results/SCHEMA.md     output contract; generated files are gitignored
scripts/run_baseline_bakeoff.sh   Docker wall-clock bakeoff, no Nsight
```

## Docker bakeoff (one command, no Nsight)

This is the run to send back. It times MatMul+RMSNorm and an attention microbench on CUDA. GEMM follows the datasheet regime: MemMode selects `always_fuse`, ComputeMode selects `always_unfuse`. Library attention scores `compile_sdpa_chain`, timed as `always_fuse`, in every regime, because SDPA already fuses scores. `paper/data/bakeoff/` is the pre-fix selector. `paper/data/bakeoff_v2/` is the wall-clock after that attention change. It is not an idle GPU, and it does not change the GEMM rule. On that CSV, ridge and compute GEMM still copy `always_unfuse` while `torch.compile` is faster. A calibrated selector is TODO. The script does not run Nsight, does not claim FlashAttention-3, and does not quote a profiler as SCIC accuracy.

**Idle GPU recommended.** The pre-fix A40 run was not idle: `paper/data/bakeoff/RUN_NOTES.txt` records GPU 0 at 83% utilization and GPU 1 at 77% when the script started. The second run was not idle either: `paper/data/bakeoff_v2/RUN_NOTES.txt` records the selected GPU at 92% utilization with 13490 MiB already allocated. A busy device changes the medians.

**Raise Dynamo's cache limit.** That same run hit `torch._dynamo.config.cache_size_limit` at the default of 8 for the MatMul chain and for `sdpa_chain`. All 77 rows were `ok`, and the graphs were not inspected, so a compile-labelled row is not proof that compiled code ran. The bakeoff now sets the limit to 64 before compiling. Override with `DYNAMO_CACHE_SIZE_LIMIT`. A higher limit does not inspect the graph.

```bash
bash scripts/run_baseline_bakeoff.sh
# CUDA PyTorch already installed:
SKIP_INSTALL=1 bash scripts/run_baseline_bakeoff.sh
# optional:
SKIP_MAX_AUTOTUNE=1 bash scripts/run_baseline_bakeoff.sh
SKIP_ATTENTION=1 bash scripts/run_baseline_bakeoff.sh
```

The script requires `nvidia-smi` and `torch.cuda.is_available()`. It does not install a CUDA wheel. Output: `results/bakeoff/`.

Send that directory back. `results/bakeoff/SEND_BACK.txt` repeats the list:

- `bakeoff_latency.csv` — columns `baseline`, `shape`, `median_ms`, `speedup_vs_eager`, `honesty`
- `bakeoff_decisions.csv` — regime, MemMode or ComputeMode, `selected_baseline`
- `bakeoff_summary.json`
- `nvidia_smi.txt`
- `torch_version.txt`

Baselines in the CSV: `eager_unfused`, `torch_compile_default`, `torch_compile_max_autotune` (or a skipped/failed honesty flag), `sdpa` on the attention rows, `always_fuse`, `always_unfuse`, `scic_selected`. `scic_selected` copies the median of the path it picked. Leave failed rows empty. Do not invent numbers.

A schema check without a GPU writes placeholders, not a paper table:

```bash
python -m shapefuse bakeoff --device cpu --level smoke --attention --skip-max-autotune --out results/bakeoff-smoke
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

What that sweep actually times is stock `torch.matmul` at M ∈ {1, 4, 32, 344, 512, 2048, 4096}, N = K = 4096, FP16. That supports the "small M loses throughput" curve. It does **not** compare MemMode against ComputeMode, because those kernels are not generated. Optional: `PROFILE=torch FUSION_PROXY=1 bash scripts/run_h100.sh`.

## Run on A40

```bash
bash scripts/run_a40.sh
# PyTorch already installed:
SKIP_INSTALL=1 bash scripts/run_a40.sh
# After an admin allows GPU performance counters and the machine reboots.
# The third package already set TRY_NSIGHT=1; ncu failed with ERR_NVGPUCTRPERM.
TRY_NSIGHT=1 SKIP_INSTALL=1 bash scripts/run_a40.sh
```

Default output directory is `results/a40`. Default `--machine` is `A40_FP16`. The script requires `nvidia-smi` and CUDA PyTorch, prints the device name, and warns if that name does not contain `A40`. It prints whether `ncu` is on `PATH`. **`ncu` is Nsight Compute, not the driver.** If it is missing, the script says so and, when `TRY_NSIGHT=1`, writes `nsight_skipped.txt` and continues. If `ncu` is present and fails with `ERR_NVGPUCTRPERM`, the script says so, keeps `nsight_error.log`, and points at [`docs/NSIGHT_PERMISSIONS.md`](docs/NSIGHT_PERMISSIONS.md). DRAM bytes stay empty. Do not fill them from the profiler.

The archived package in the paper (`paper/data/a40/`) is the third A40 upload: `--machine A40_FP16`, status `PARTIAL_VENDOR_MATMUL`, `nsight_status` `TBD_RUN_PROFILER`. `TRY_NSIGHT=1` found `ncu` on `PATH`; `ncu` failed with `ERR_NVGPUCTRPERM` (`nsight_error.log`). `achieved_ai` is empty. `proxy_ai` is `TORCH_PROFILER_MEMORY_NOT_DRAM` and stuck at 4096 (output-sized allocator counts). `agreement.json` has `quote_as_scic_accuracy` false and status `PROXY_TORCH_MEMORY_NOT_VALID_SCIC_ACCURACY` (inspection rate 0.2857142857142857, not SCIC accuracy). Triton bias+GELU rows are `ok=true` with honesty `TRITON_ELEMENTWISE_STUB_NOT_GEMM_FUSION`. That stub is still not a fused GEMM. `bandwidth_estimate.json` is `ESTIMATE_COPY_NOT_NSIGHT` (about 570.1 GB/s, estimated ridge 262.5846181025793); `measured_ridge_flops_per_byte` is null. The script still does not emit MemMode or ComputeMode kernels.

To capture DRAM bytes, an administrator follows [`docs/NSIGHT_PERMISSIONS.md`](docs/NSIGHT_PERMISSIONS.md) (`modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0`, or the same line in `/etc/modprobe.d/`, then reboot, then verify `ncu`) and re-runs `TRY_NSIGHT=1`.

`scripts/bandwidth_probe.sh` (also invoked at the end of `run_a40.sh`) times a device-to-device copy and writes `bandwidth_estimate.json` with status `ESTIMATE_COPY_NOT_NSIGHT`. The ridge numerator is the datasheet FLOP peak. It does not set `measured_ridge_flops_per_byte`.

What the new script adds, and what it is not:

| output | meaning |
| --- | --- |
| `bench_gemm.csv` | `torch.matmul` latency and TFLOP/s. `achieved_ai` stays empty without Nsight |
| `proxy_ai` | FLOPs / allocator bytes. `TORCH_PROFILER_MEMORY_NOT_DRAM` when profiler events have bytes. `TORCH_ALLOCATOR_DELTA_NOT_DRAM` when the profiler table is empty and the caching allocator moved. `TORCH_PROFILER_EMPTY` when neither did. Not DRAM |
| `fusion_proxy.csv` | matmul+RMSNorm two-region vs eager chain vs `torch.compile` if it runs; bias+GELU two ops vs one Python expression; Triton elementwise stub when Triton imports. Time ratios are `PROXY_TIMING_NOT_A_FUSION_SPEEDUP`. Not a fused GEMM |
| `decision_log.csv`, `agreement.json` | analytical plan vs regime from whatever byte AI exists. Quote as SCIC accuracy only when `quote_as_scic_accuracy` is true (`NSIGHT_DRAM`) |
| `nsight_dram.json` | only after `TRY_NSIGHT=1` and `ncu` succeed. Status `NSIGHT_DRAM_PROCESS_SUM` (process-wide sum, copies included). If `ncu` is missing or fails, see `nsight_skipped.txt`. `ERR_NVGPUCTRPERM` also leaves `nsight_error.log`; fix in [`docs/NSIGHT_PERMISSIONS.md`](docs/NSIGHT_PERMISSIONS.md) |
| `bandwidth_estimate.json` | copy-kernel estimate, `ESTIMATE_COPY_NOT_NSIGHT`. Not a measured ridge |

`scripts/nsight_dram.sh` wraps `ncu`. Metric names vary; the default is `dram__bytes_read.sum,dram__bytes_write.sum`. Override with `NCU_METRICS`. List them with `ncu --query-metrics | grep dram__bytes`.

## What to send back for the paper

For the Docker bakeoff, send `results/bakeoff/` after `bash scripts/run_baseline_bakeoff.sh`. The list is in the section above and in `results/bakeoff/SEND_BACK.txt`. Do not add Nsight numbers to that directory.

The A40 vendor-curve package is separate. After `scripts/run_a40.sh` (or the H100 script), send that output directory too if you still have it. `SEND_BACK.txt` there repeats the list. Minimum:

- `summary.json`, `gpu_measurements.json`, `bench_gemm.csv`
- `decision_log.csv`, `agreement.json`, `fusion_proxy.csv`
- `gemm_shapes.csv`, `attention_intensity.csv`, `attention_decode.csv`
- `rmsnorm_epilogue.csv`, `ew_chains.csv`, `moe_grouped.csv`, `quant_epilogue.csv`, `dual_plans.csv`
- `fig_*.png` (including `fig_vendor_matmul_tflops.png`)
- `nsight_dram.json` if `TRY_NSIGHT=1` succeeded, or `nsight_skipped.txt` if `ncu` was missing or `ncu` failed
- `bandwidth_estimate.json` if the copy probe ran (`ESTIMATE_COPY_NOT_NSIGHT`)
- the `nvidia-smi` header (driver, GPU name) and `torch.__version__` (already copied into `gpu_measurements.json` when CUDA runs)

Leave the following as TBD unless the file actually contains them. Do not invent numbers. Do not quote `proxy_ai` or a process-sum agreement rate as SCIC accuracy.

| quantity | where it goes |
| --- | --- |
| DRAM bytes and achieved AI | `nsight_dram_bytes`, `achieved_ai` after Nsight JSON is ingested. Status must be `NSIGHT_DRAM` or `NSIGHT_DRAM_PROCESS_SUM` |
| profiler memory (not DRAM) | `torch_profiler_memory_bytes`, `proxy_ai`, status `TORCH_PROFILER_MEMORY_NOT_DRAM`. Allocator-delta fallback: status `TORCH_ALLOCATOR_DELTA_NOT_DRAM`, column `proxy_bytes` |
| measured bandwidth and ridge | `measured_hbm_bandwidth_bytes_per_s`, `measured_ridge_flops_per_byte`. A copy-kernel file does not fill these |
| MemMode vs ComputeMode kernel time | not produced. `fusion_proxy.csv` is a proxy with an `honesty` column |
| SCIC accuracy vs isolated Nsight AI | `agreement.json` only if `quote_as_scic_accuracy` is true |
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

- English draft: [`paper/draft.md`](paper/draft.md). §7.3 quotes `paper/data/bakeoff/` (pre-fix attention selector) and `paper/data/bakeoff_v2/` (after `compile_sdpa_chain`, busy GPU). GEMM calibration is still **TODO**. Every other missing GPU number is marked **TBD**.
- Chinese outline: [`paper/zh_outline.md`](paper/zh_outline.md).

---

# 中文说明

算子是计算受限还是访存受限，取决于 shape，不是算子种类的固定标签。同一 GEMM、Attention、MoE Group-GEMM 会越过 GPU Roofline 的 ridge，融合决策必须是多态的：至少发出 **MemMode** 和 **ComputeMode** 两套计划。

本仓库是这条论断的解析模型和可在服务器上跑的实验入口。

- **SCIC**：用闭式算术强度相对机器 ridge，把站点分成 memory-bound、ridge-region、compute-bound。
- **双计划**：访存侧融合 epilogue、用较小 tile；计算侧保住 MMA 流水、用较大 tile、把重 epilogue 拆出去。落在 ridge 带内则两套都发。
- 范围不限于 GEMM epilogue：逐元素链、归约、Norm、Softmax、Attention 的 prefill/decode、ragged MoE、量化 epilogue。

相对 ComFuse：对方做的是 MatMul 与 epilogue 的联合融合。这里增加的是按 shape 的强度分类，以及 MemMode / ComputeMode 切换。**Stage-Stream 不是我们的贡献，代码里也没有实现。** 不会生成融合 CUDA kernel。`dual_plans.csv` 里 `kernel_emitted=false`。也不宣称 FlashAttention-3，也不把 profiler 当成 SCIC 准确率。

## Docker bakeoff（一条命令，无 Nsight）

请跑这一条，并把目录寄回。它在 CUDA 上计 MatMul+RMSNorm，以及可选的 Attention 微基准。GEMM 按数据手册分档：MemMode 选 `always_fuse`，ComputeMode 选 `always_unfuse`。库 Attention 的 epilogue 在所有档位都记 `compile_sdpa_chain`（计时为 `always_fuse`），因为 SDPA 已经融合了 score。`paper/data/bakeoff/` 是改这条规则之前的归档。`paper/data/bakeoff_v2/` 是改完之后的墙钟，不是空闲 GPU，也没有改 GEMM 规则。那份 CSV 里 ridge 和计算侧 GEMM 仍抄 `always_unfuse`，而 `torch.compile` 更快。校准选择器是 TODO。不跑 Nsight，不宣称 FA-3，不把 profiler 写成准确率。

**建议空闲 GPU。** 修复前的 A40 运行不是空闲的：`paper/data/bakeoff/RUN_NOTES.txt` 记录脚本启动时 GPU 0 利用率 83%，GPU 1 为 77%。第二次也不是：`paper/data/bakeoff_v2/RUN_NOTES.txt` 记录所选 GPU 利用率 92%，已分配 13490 MiB。忙设备会改变 median。

**请提高 Dynamo 的 cache 上限。** 同一次运行里 MatMul 链和 `sdpa_chain` 打到了 `torch._dynamo.config.cache_size_limit` 的默认值 8。77 行都是 `ok`，图没有被检查，所以标成 compile 的行不能证明跑的是编译后的代码。bakeoff 现在在编译前把上限设为 64。可用 `DYNAMO_CACHE_SIZE_LIMIT` 覆盖。提高上限不会检查图。

```bash
bash scripts/run_baseline_bakeoff.sh
SKIP_INSTALL=1 bash scripts/run_baseline_bakeoff.sh
SKIP_MAX_AUTOTUNE=1 bash scripts/run_baseline_bakeoff.sh
SKIP_ATTENTION=1 bash scripts/run_baseline_bakeoff.sh
```

需要 `nvidia-smi` 和 CUDA 版 PyTorch。脚本不下载 wheel。产物在 `results/bakeoff/`。

寄回（`results/bakeoff/SEND_BACK.txt` 有同一份清单）：

- `bakeoff_latency.csv`（列：`baseline`、`shape`、`median_ms`、`speedup_vs_eager`、`honesty`）
- `bakeoff_decisions.csv`
- `bakeoff_summary.json`
- `nvidia_smi.txt`
- `torch_version.txt`

失败的行留空。不要编造数字。没有 GPU 时只能做 schema 检查，状态是 `PLACEHOLDER_NO_GPU`，不能写进论文表：

```bash
python -m shapefuse bakeoff --device cpu --level smoke --attention --skip-max-autotune --out results/bakeoff-smoke
```

上表的 H100 / A100 / A40 ridge 是数据手册锚点，不是你这台机器的实测峰值。A40 的 149.7 是密集 FP16（FP16 累加），不用稀疏 299.4。N = K = 4096、FP16、H100 上 AI 等于 ridge 的点是 **M\* ≈ 345**；A40 数据手册上 **M\* ≈ 240**。H100 上 M ≲ 159 为 memory-bound（AI < 0.5 R），M ≳ 830 为 compute-bound（AI > 2 R）。短名字：`--machine a40`、`h100`、`a100`、`h100_fp8`。

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

## 在 A40 上

```bash
bash scripts/run_a40.sh
SKIP_INSTALL=1 bash scripts/run_a40.sh
# 管理员放开 GPU 性能计数器并重启之后。第三次包已经设过 TRY_NSIGHT=1，ncu 报 ERR_NVGPUCTRPERM。
TRY_NSIGHT=1 SKIP_INSTALL=1 bash scripts/run_a40.sh
```

默认输出 `results/a40`，数据手册锚点 `A40_FP16`。脚本检查 `nvidia-smi` 和 CUDA，设备名不含 A40 时只警告。它会打印 `ncu` 是否在 `PATH` 上。**`ncu` 是 Nsight Compute，不是驱动。** 找不到时会说明；若同时设了 `TRY_NSIGHT=1`，会写 `nsight_skipped.txt` 并继续。若 `ncu` 在 PATH 上但报 `ERR_NVGPUCTRPERM`，脚本会说明，保留 `nsight_error.log`，并指向 [`docs/NSIGHT_PERMISSIONS.md`](docs/NSIGHT_PERMISSIONS.md)。DRAM 字节保持为空，不要用 profiler 填。

论文归档（`paper/data/a40/`）是第三次 A40 上传：`--machine A40_FP16`，`PARTIAL_VENDOR_MATMUL`，`nsight_status` 为 `TBD_RUN_PROFILER`。`TRY_NSIGHT=1` 时 `ncu` 在 PATH 上，随后 `ERR_NVGPUCTRPERM`（`nsight_error.log`）。`achieved_ai` 为空。`proxy_ai` 为 `TORCH_PROFILER_MEMORY_NOT_DRAM`，卡在 4096（字节数像输出矩阵）。`agreement.json` 的 `quote_as_scic_accuracy` 为 false，状态 `PROXY_TORCH_MEMORY_NOT_VALID_SCIC_ACCURACY`（检查用一致率 0.2857142857142857，不是 SCIC 准确率）。Triton bias+GELU 行 `ok=true`，诚实标签 `TRITON_ELEMENTWISE_STUB_NOT_GEMM_FUSION`。这个 stub 仍然不是融合 GEMM。`bandwidth_estimate.json` 是 `ESTIMATE_COPY_NOT_NSIGHT`（约 570.1 GB/s，估计 ridge 262.5846181025793）；`measured_ridge_flops_per_byte` 为 null。脚本仍然不生成 MemMode / ComputeMode kernel。

要抓 DRAM 字节，管理员按 [`docs/NSIGHT_PERMISSIONS.md`](docs/NSIGHT_PERMISSIONS.md) 操作（`modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0`，或写入 `/etc/modprobe.d/`，重启，再用 `ncu` 验证），然后重新 `TRY_NSIGHT=1`。

`scripts/bandwidth_probe.sh`（`run_a40.sh` 末尾也会跑）用一次 device-to-device copy 估计带宽，写入 `bandwidth_estimate.json`，状态 `ESTIMATE_COPY_NOT_NSIGHT`。ridge 的分子是数据手册 FLOP 峰值。它不会写入 `measured_ridge_flops_per_byte`。

| 产物 | 含义 |
| --- | --- |
| `bench_gemm.csv` | `torch.matmul` 时延和 TFLOP/s。没有 Nsight 时 `achieved_ai` 为空 |
| `proxy_ai` | 分配器字节。profiler 事件为 `TORCH_PROFILER_MEMORY_NOT_DRAM`；事件为空时的缓存分配器差分为 `TORCH_ALLOCATOR_DELTA_NOT_DRAM`；两者都没有则 `TORCH_PROFILER_EMPTY`。不是 DRAM |
| `fusion_proxy.csv` | matmul+RMSNorm 与 bias+GELU 的代理。时间比的状态是 `PROXY_TIMING_NOT_A_FUSION_SPEEDUP`。不是融合 GEMM |
| `decision_log.csv`、`agreement.json` | 解析计划对字节 AI。只有 `quote_as_scic_accuracy=true`（`NSIGHT_DRAM`）才能当准确率 |
| `nsight_dram.json` | 仅 `TRY_NSIGHT=1` 且 `ncu` 成功。状态 `NSIGHT_DRAM_PROCESS_SUM`。没有 `ncu` 或 ncu 失败时看 `nsight_skipped.txt`。`ERR_NVGPUCTRPERM` 还会留下 `nsight_error.log`，步骤见 [`docs/NSIGHT_PERMISSIONS.md`](docs/NSIGHT_PERMISSIONS.md) |
| `bandwidth_estimate.json` | copy 估计，`ESTIMATE_COPY_NOT_NSIGHT`。不是实测 ridge |

## 论文需要你寄回的东西

Docker bakeoff 请寄 `results/bakeoff/`（跑 `bash scripts/run_baseline_bakeoff.sh` 之后）。清单在上面的中文小节和 `results/bakeoff/SEND_BACK.txt`。不要往这个目录里补 Nsight 数字。

A40 vendor 曲线是另一份。跑完 `scripts/run_a40.sh`（或 H100 脚本）后，如果还留着那个目录，也一并寄回。该目录里的 `SEND_BACK.txt` 有清单。至少包括：

- `summary.json`、`gpu_measurements.json`、`bench_gemm.csv`
- `decision_log.csv`、`agreement.json`、`fusion_proxy.csv`
- `gemm_shapes.csv`、`attention_intensity.csv`、`attention_decode.csv`
- `rmsnorm_epilogue.csv`、`ew_chains.csv`、`moe_grouped.csv`、`quant_epilogue.csv`、`dual_plans.csv`
- `fig_*.png`（含 `fig_vendor_matmul_tflops.png`）
- 若 `TRY_NSIGHT=1` 成功：`nsight_dram.json`；若 `ncu` 缺失或失败：`nsight_skipped.txt`
- 若 copy 探针跑过：`bandwidth_estimate.json`（`ESTIMATE_COPY_NOT_NSIGHT`）
- `nvidia-smi` 抬头（驱动、GPU 名）和 PyTorch 版本（CUDA 跑通时会写进 `gpu_measurements.json`）

下面这些没有写进文件就保持 TBD。不要把 `proxy_ai` 或进程合计的一致率当成 SCIC 准确率。

| 量 | 填到哪里 |
| --- | --- |
| DRAM 字节和 achieved AI | 摄入 Nsight JSON 之后的 `nsight_dram_bytes`、`achieved_ai`。状态应为 `NSIGHT_DRAM` 或 `NSIGHT_DRAM_PROCESS_SUM` |
| profiler 显存（不是 DRAM） | `torch_profiler_memory_bytes`、`proxy_ai`，状态 `TORCH_PROFILER_MEMORY_NOT_DRAM`。分配器差分回退：`TORCH_ALLOCATOR_DELTA_NOT_DRAM`，列 `proxy_bytes` |
| 实测带宽和 ridge | `measured_hbm_bandwidth_bytes_per_s`、`measured_ridge_flops_per_byte`。copy 估计文件不填这两列 |
| MemMode 与 ComputeMode 的 kernel 时延 | 本仓库不产生。`fusion_proxy.csv` 是带 `honesty` 列的代理 |
| SCIC 相对孤立 Nsight AI 的准确率 | 仅当 `agreement.json` 里 `quote_as_scic_accuracy` 为 true |
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

英文草稿：[`paper/draft.md`](paper/draft.md)。§7.3 引用 `paper/data/bakeoff/`，那是 Attention 修复前的归档。其余缺的 GPU 数字标了 TBD。中文提纲：[`paper/zh_outline.md`](paper/zh_outline.md)。

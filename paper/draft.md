# Shape-Conditioned Intensity Classification for Plan-Polymorphic Operator Fusion

**Target venues:** MLSys or ASPLOS.
**Status:** draft. Analytical results are produced by this repository. The third NVIDIA A40 package is archived under `paper/data/a40/` (`PARTIAL_VENDOR_MATMUL`, `--machine A40_FP16`, datasheet ridge ≈ 215.09, M\* ≈ 240.3). It times `torch.matmul`, a fusion proxy, and a copy-kernel bandwidth estimate. `TRY_NSIGHT=1` found `ncu` on `PATH`; `ncu` then failed with `ERR_NVGPUCTRPERM` (`nsight_error.log`, `nsight_skipped.txt`). `achieved_ai` is still empty. `proxy_ai` is `TORCH_PROFILER_MEMORY_NOT_DRAM` (`quote_as_scic_accuracy` false). The copy-kernel file is `ESTIMATE_COPY_NOT_NSIGHT`; `measured_ridge_flops_per_byte` is null. Fused MemMode/ComputeMode kernels are not in that run.
**Authors:** TBD.

> Remaining **TBD** markers are empty on purpose. Do not fill them with datasheet arithmetic, with the CPU proxy, with `proxy_ai`, with the copy-kernel estimate, or by relabeling the A40 `torch.matmul` sweep or `fusion_proxy.csv` as a fused-kernel result. This package's three-way regime column uses the A40 FP16 datasheet because the run passed `--machine A40_FP16`. That ridge was not measured on the card. Nsight DRAM bytes are still missing: the third run's `ncu` was blocked by `ERR_NVGPUCTRPERM`.

## Abstract

Whether a tensor operator is compute-bound or memory-bound is a function of shape, dtype, and which intermediates touch HBM, not a label of the operator's name. On an H100-class Roofline (datasheet dense FP16 Tensor Core 989 TFLOP/s, HBM3 3.35 TB/s, ridge R ≈ 295 FLOP/byte), a single FP16 GEMM with N = K = 4096 has arithmetic intensity AI ≈ 1 at M = 1 and AI = 1024 at M = 2048. The same attention block is memory-bound for every sequence length we sweep if it materializes scores, and compute-bound for long sequences if it keeps scores off HBM. A skewed mixture-of-experts layer holds memory-bound and compute-bound experts at once.

Fusion that is decided once per operator kind cannot track that crossing. We describe a shape-conditioned intensity classifier (SCIC) that places each producer–consumer site on the Roofline before scheduling, and a dual-plan policy that emits a memory-oriented plan (MemMode), a compute-oriented plan (ComputeMode), or both when the site sits in a band around the ridge. Relative to joint MatMul–epilogue fusion, the added mechanism is the classification and the switch. We do not implement Stage-Stream, and we do not claim it.

The classifier, the plan table, and the closed-form figures are analytical. The third NVIDIA A40 package (`--machine A40_FP16`) times stock `torch.matmul` at seven shapes (`PARTIAL_VENDOR_MATMUL` in §7.2). The largest rate in that CSV is 115.3816693 TFLOP/s, and `fraction_of_requested_datasheet_peak` is 0.7707526341271441 (about 77.1% of the A40 datasheet dense FP16 peak, 149.7). That fraction is not a measured ridge. A fusion proxy ran, including a Triton bias+GELU elementwise stub (`TRITON_ELEMENTWISE_STUB_NOT_GEMM_FUSION`); its time ratios are `PROXY_TIMING_NOT_A_FUSION_SPEEDUP`. Nsight failed with `ERR_NVGPUCTRPERM`, so achieved AI, a measured ridge, MemMode versus ComputeMode, and end-to-end tokens/s are **TBD**.

## 1. Introduction

Compilers for GPU kernels already fuse a matrix multiply with the elementwise and normalization code that consumes it. The usual justification is traffic: the product matrix is expensive to write and to read back. That justification is real, and it is incomplete. The same multiply is a different machine regime at decode (a handful of rows) and at prefill (thousands of rows). A fusion that pays for itself on one side of the ridge can crowd registers and the MMA pipeline on the other. Attention makes the gap sharper. FlashAttention-style IO and a score-materializing implementation are not two tunings of one intensity; they are different intensities, and for a 128-wide head in FP16 the materialized version never reaches the H100 ridge no matter how long the sequence is.

This draft argues for plan-polymorphic fusion:

1. Intensity is shape-dependent. GEMM, attention, and grouped GEMM each cross or fail to cross the ridge inside the shape range of a serving stack.
2. A classifier should run before fusion, on the concrete shape, dtype, and traffic accounting, and should return a regime rather than a boolean "these ops are adjacent."
3. The scheduler should be allowed to emit two plans for one site and pick at runtime from the regime, or from a measured cache keyed by a shape bucket.

ComFuse showed that joint MatMul and epilogue fusion is profitable, and it also reported a self-attention case where the joint schedule loses to a FlashAttention-style kernel. That counterexample is a missing switch, not a reason to abandon fusion. Stage-Stream, the streaming mechanism used to carry the joint schedule, is prior work. Nothing in this paper re-derives it or implements it.

**Contributions**

- An analytical SCIC for GEMM, attention (prefill and decode, MHA and GQA), norms, reductions, softmax, elementwise chains, ragged MoE grouped GEMM, and mixed-dtype GEMM epilogues, scored against datasheet ridges for H100 FP16, H100 FP8, A100 FP16, and A40 FP16.
- A dual-plan selector stub: MemMode, ComputeMode, or both inside the ridge band, with an explicit plan cache that stores no measured winner until one is provided.
- A correction to a byte-fraction intuition: under a cold-cache model, fusing RMSNorm into GEMM saves a *larger* fraction of traffic as M grows, while the strict Roofline speedup of that fusion stays near 1×. The shape dependence that actually flips a regime is elsewhere (score traffic, expert token counts, the GEMM's own M).

**Non-contributions**

- Stage-Stream, and any claim to have built it.
- A new attention algorithm. IO-aware attention is treated as an existing plan, not as a result of this paper.
- CUDA kernels for either mode, and any end-to-end speedup. Those measurements are **TBD**.

## 2. Background

### 2.1 Roofline

Arithmetic intensity and the machine ridge are

\[
\mathrm{AI} = \frac{\mathrm{FLOPs}}{\mathrm{bytes}}, \qquad R = \frac{\mathrm{peak\ FLOP/s}}{\mathrm{peak\ byte/s}}.
\]

A kernel whose AI is below R is bandwidth-limited on the classical Roofline; above R it is compute-limited. Time is \(\max(\mathrm{FLOPs}/P,\ \mathrm{bytes}/B)\) with no cache, no launch, and no overlap. We use that model as a classifier feature, not as a latency prediction. Datasheet anchors, dense Tensor Core, sparsity off:

| machine | peak | memory | R (FLOP/byte) |
| --- | --- | --- | --- |
| H100 SXM, FP16 | 989 TFLOP/s | HBM3 3.35 TB/s | 295.22 |
| H100 SXM, FP8 | 1979 TFLOP/s | HBM3 3.35 TB/s | 590.75 |
| A100 80GB SXM, FP16 | 312 TFLOP/s | HBM2e 2.039 TB/s | 153.02 |
| A40, FP16 | 149.7 TFLOP/s | GDDR6 696 GB/s | 215.09 |

A40's 149.7 TFLOP/s is the dense FP16 Tensor Core peak with FP16 accumulate. The datasheet's structural-sparsity figure (299.4) is not used. These four rows are datasheet anchors. The A40 sweep in §7.2 does not replace this row with a measured peak.

> **TBD:** replace P and B with measured peaks on the evaluation GPU. The paper must not quote the table above as a measured ridge.

SCIC's three-way label is a *policy band* around that knee, carried over from the project note and exposed as flags:

- memory-bound if AI < 0.5 R,
- compute-bound if AI > 2 R,
- ridge-region otherwise.

The strict knee (AI ≷ R) is reported beside the band as `roofline_limit`. Inside the band the selector emits both plans because a small error in bytes or in R flips the knee.

### 2.2 What "fusion" means here

We mean a choice about HBM traffic and about how much epilogue code shares a mainloop with a tensor-core producer.

- **MemMode.** Fuse the epilogue. Do not write the intermediate. Prefer a smaller tile. Optimize for bytes and for the number of launches.
- **ComputeMode.** Do not pull a heavy epilogue into the MMA mainloop. Use a larger tile. On GEMM-like producers, materialize the output. Protect the pipeline.
- **Attention, both modes.** Do not materialize the score matrix. That decision is forced by the intensity model in §3.2. The modes still differ in tile size and in whether a further epilogue (output projection, residual, dropout) joins the mainloop.

Neither mode is a kernel in this repository. `kernel_emitted` is false on every row.

### 2.3 Traffic accounting

Bytes are a cold-HBM lower bound: each live tensor moves once. There is no L2 reuse, no causal mask, and no software pipelining. GEMM counts a multiply-add as two FLOPs. Softmax is charged 5 FLOPs per score and 3 memory passes (the project-note convention). RMSNorm is charged 4 FLOPs per element. These constants fix the order of magnitude of O(1)-intensity ops; they are not instruction traces.

> **TBD:** calibrate pass counts and FLOP constants against SASS or against Nsight on one kernel per kind.

## 3. Shape-dependent intensity

### 3.1 GEMM

For \(C = AB\) with shapes \([M,K]\), \([K,N]\), \([M,N]\) and element sizes \(e_A, e_B, e_C\),

\[
\mathrm{FLOPs} = 2MNK, \qquad \mathrm{bytes} = MK e_A + KN e_B + MN e_C.
\]

Uniform FP16 (\(e = 2\)) gives \(\mathrm{AI} = MNK / (MK + KN + MN)\). The square projection \(N = K = d\) collapses to

\[
\mathrm{AI}(M) = \frac{2Md}{(d + 2M)\,e}.
\]

Setting AI equal to a target \(t\) and solving for M,

\[
M = \frac{t\, e\, d}{2\,(d - t\, e)} \quad (d > t\, e),
\]

which for FP16 is the note's formula \(M^\star = Rd/(d - 2R)\). For \(d = 4096\):

| anchor | M at 0.5 R | M at R | M at 2 R |
| --- | --- | --- | --- |
| H100 FP16 | 159.1 | 344.9 | 829.6 |
| A100 FP16 | 79.5 | 165.4 | 359.8 |
| A40 FP16 | 113.5 | 240.3 | 544.6 |
| H100 FP8 | 159.2 | 345.2 | 830.2 |

The A40 row is the same closed form on the A40 datasheet preset (`--machine a40`). The third package's `summary.json`, and the archived `crossings.json` that matches it, were produced with `--machine A40_FP16` and store that row: ridge 215.08620689655172, M at 0.5 R = 113.50330752177987, M\* = 240.32584259093773, M at 2 R = 544.5531470469126. Those are datasheet crossings, not a measured A40 ridge.

FP8 doubles both the Tensor-Core peak and the AI of an all-FP8 GEMM (half the bytes), so \(M^\star\) barely moves. Quantizing only the weights, while leaving the FP16 peak in place, does move AI; it does not by itself lift decode. At M = 1, N = K = 4096, FP16 AI is 1.00 and W8A8 with an FP16 store is still ≈ 2.0, both memory-bound against R ≈ 295. The mixed-dtype rows are a byte what-if: an INT8 Tensor-Core peak is not the FP16 ridge.

| label | shape M, N, K | AI FP16 | vs H100 band |
| --- | --- | --- | --- |
| decode, M=1 | 1, 4096, 4096 | 1.00 | memory-bound |
| micro-batch M=4 | 4, 4096, 4096 | 3.99 | memory-bound |
| decode M=32 | 32, 4096, 4096 | 31.5 | memory-bound |
| M=256 | 256, 4096, 4096 | 228 | ridge-region |
| M=345 (near M*) | 345, 4096, 4096 | 295 | ridge-region |
| prefill M=2048 | 2048, 4096, 4096 | 1024 | compute-bound |
| square 4096 | 4096, 4096, 4096 | 1365 | compute-bound |
| K=64 | 4096, 4096, 64 | 62.1 | memory-bound |
| expert, 16 tokens | 16, 4096, 14336 | 15.9 | memory-bound |
| expert, 256 tokens | 256, 4096, 14336 | 237 | ridge-region |
| expert, 2048 tokens | 2048, 4096, 14336 | 1247 | compute-bound |

For a uniform dtype, AI is unchanged if N and K are swapped, so the expert rows match both \([t, H]\times[H, I]\) and the \((M, N, K)\) order stored in `gemm_shapes.csv`. A linear layer is not "a compute kernel." Decode and prefill of that layer are different regimes on the same GPU.

On the seven shapes timed in §7.2, `regime_vs_h100_datasheet` and `regime_vs_requested_datasheet` agree: memory-bound at M ∈ {1, 4, 32}, ridge-region at M ∈ {344, 512}, compute-bound at M ∈ {2048, 4096}. The strict knee does not. Analytical AI at M = 344 is 294.5284281, just below the H100 ridge (295.22) and above the A40 datasheet ridge (215.08620689655172), so the three-way band is ridge-region while the strict Roofline limit is compute. Neither column is a measured ridge. The third upload did not include `decision_log.csv`. `agreement.json` summarizes a torch-profiler byte AI and is not SCIC accuracy (§7.2).

### 3.2 Attention

Let \(Q, K, V\) have shape \([B, H, S, D]\).

**Scores written out.** QK\(^\top\) has AI \(= 2SD / ((2D + S)\, e)\). Softmax over the score matrix has AI \(= 5 / (3e)\), which is \(5/6 \approx 0.833\) in FP16 and does not depend on S. A softmax kernel is memory-bound on every GPU in the table.

**IO-aware prefill.** QK\(^\top\) and PV, with scores kept on chip and HBM traffic equal to Q, K, V, and O once, have

\[
\mathrm{AI} = S / e.
\]

Head dimension cancels. In FP16, AI = S/2. Against the H100 FP16 ridge the FA-style plan crosses 0.5 R near S ≈ 295, R near S ≈ 590, and 2 R near S ≈ 1181.

**Materialized prefill, same FLOPs.** Charging three memory passes over the score tensor,

\[
\mathrm{bytes} = (4\,BHSD + 3\,BHS^2)\, e,
\]

AI saturates at \(4D / (3e)\). For D = 128 and FP16 that ceiling is ≈ 85.3, which is under 0.5 R on H100 (147.6). **Unfused attention, in this accounting, is memory-bound at every sequence length.** Fusion is not a constant-factor saving on top of a compute-bound kernel. It is what makes the compute-bound regime reachable.

Roofline time ratio of the IO-aware accounting over the materialized one (H100 datasheet, B = 1, H = 32, D = 128, no launch term). This is a model, not a kernel measurement:

| S | AI, scores in HBM | AI, FA-style | model speedup |
| --- | --- | --- | --- |
| 128 | 36.6 | 64 | 1.75× |
| 512 | 64.0 | 256 | 4.00× |
| 1024 | 73.1 | 512 | 4.04× |
| 2048 | 78.8 | 1024 | 3.75× |
| 8192 | 83.6 | 4096 | 3.53× |

At S = 128 both sides are bandwidth-limited. By S = 1024 the IO-aware side has crossed into compute while the materialized side has not, which is why the ratio stops climbing.

**Decode.** Query length \(S_q = 1\), cache length \(S_{kv}\), \(H_{kv}\) KV heads. As \(S_{kv}\) grows,

\[
\mathrm{AI} \to \frac{2 H_q}{H_{kv}\, e}.
\]

MHA in FP16 tends to 1. GQA with \(H_q / H_{kv} = 4\) tends to 4. GQA with ratio 8 tends to 8. All of these are memory-bound on H100 for every cache length in the sweep (128 through 32768). "Attention is compute-bound" is a prefill statement about an IO-aware plan. It is the wrong label for decode.

The selector classifies the IO-aware variant, and it refuses to emit a plan that writes scores. MemMode versus ComputeMode then only chooses tile size and whether an *additional* epilogue joins that mainloop. Long sequences (FA AI > 2 R, S = 8192 in the sweep) take ComputeMode: large tiles, epilogue fission, scores still on chip. Short sequences (S = 128) take MemMode. S = 512 (AI = 256, inside the band) emits both. That is the self-attention counterexample turned into a switch instead of a single joint schedule.

### 3.3 Norms, reductions, elementwise chains

RMSNorm on \([M, K]\) has AI \(\approx 4 / (2e) = 1\) in FP16 once the weight vector is amortized, and slightly less at M = 1. LayerNorm, a row reduction, SiLU, GELU, residual add, and the SwiGLU elementwise product are the same kind of object: a handful of FLOPs per byte. SCIC labels them memory-bound. The selector emits only MemMode and asks that they be fused into the producer or into one chain. There is no compute-mode fission of a residual add.

### 3.4 GEMM+RMSNorm, and a corrected byte story

Unfused traffic is A, B, write C, reread C, read the norm weight, write Y. Fused traffic drops the C round-trip. For the square projection in FP16 the saved fraction of unfused bytes is

\[
\frac{2M}{d + 4M}
\]

to a very close approximation (the norm weight adds a \(+d\) in the denominator). It **increases** with M:

| M | byte-save fraction | strict Roofline speedup | speedup if each launch costs 5 µs |
| --- | --- | --- | --- |
| 1 | 0.05% | 1.000× | 1.33× |
| 32 | 1.5% | 1.02× | 1.34× |
| 256 | 10% | 1.11× | 1.38× |
| 2048 | 33% | 1.00× | 1.07× |
| 4096 | 40% | 1.00× | 1.03× |

At M = 2048 and M = 4096 both the fused and the unfused accounting are compute-limited, so deleting bytes does not move Roofline time. The 5 µs column is a sensitivity study (`ASSUMED_NOT_MEASURED` in the CSV). It is the only version of this microbenchmark in which small M prefers fusion, and it prefers fusion because of launch count, not because the epilogue is a large fraction of weight traffic.

An earlier project note claimed the opposite fraction. The cold-cache algebra does not support that claim. Small-M GEMMs are memory-bound because KN bytes dominate. Epilogue fusion is still part of MemMode, for launches and for chains of elementwise ops, but it is the wrong place to look for an order-of-magnitude regime change. Attention scores and ragged expert shapes are the right place.

> **TBD:** measure launch latency and GEMM+RMSNorm kernel time on H100. Keep the 5 µs column out of the camera-ready unless it is replaced by that measurement.

### 3.5 Ragged MoE

Expert \(e\) computes \([t_e, H] \times [H, I]\). With H = 4096, I = 14336, FP16, and inactive experts not loaded:

| scenario | tokens | AI aggregate | AI min | AI max | regimes |
| --- | --- | --- | --- | --- | --- |
| decode, 8×2 | 16 | 2.00 | 2.00 | 2.00 | memory / memory |
| balanced, 8×256 | 2048 | 237 | 237 | 237 | ridge / ridge |
| skewed, 1024…8 | 2040 | 236 | 7.98 | 775 | aggregate ridge; min memory; max compute |
| prefill, 8×2048 | 16384 | 1247 | 1247 | 1247 | compute / compute |
| one-hot, 2048 on one expert | 2048 | 1247 | 1247 | 1247 | compute |
| one-hot, but all 8 weights move | 2048 | 237 | 1247 | 1247 | aggregate falls to ridge |

The skewed row is the policy point. One grouped kernel with one epilogue template is the wrong granularity: the 8-token expert is memory-bound and the 1024-token expert is compute-bound (AI 775 > 2 R ≈ 590). The one-hot comparison is the dispatch point. Skipping inactive weights leaves a compute-bound expert; touching them moves the same FLOPs back to the ridge, and the AI matches the balanced layer because the FLOPs and the weight traffic match. Both rows are in `moe_grouped.csv`. Which one a runtime actually does is an implementation fact, **TBD** per framework.

### 3.6 Quantized epilogues

Narrower weights raise AI by shrinking KN. They do not change the selector's logic: the site is still classified, then MemMode or ComputeMode is chosen from the band. Two cautions are encoded in `quant_epilogue.csv`.

- Pairing INT8 bytes with an FP16 peak is a what-if. `peak_dtype_matches_operands` is false on those rows.
- A dequant-and-store epilogue is itself O(1) intensity. MemMode folds it into the producer. ComputeMode materializes the producer output and runs the epilogue separately, which matters if the accumulator is wide (INT32) and the mainloop is already compute-bound.

> **TBD:** W8A8 and FP8 peaks measured on the evaluation GPU, then a rerun of this table against that ridge.

## 4. SCIC

A site is a kind, a shape, and per-tensor dtypes. SCIC returns FLOPs, bytes, AI, AI/R, the three-way regime, and the strict Roofline limit. The source flag is `analytical` until a measured AI is supplied. When it is supplied, the regime is recomputed from the measurement and the analytical AI is kept so the absolute error can be scored.

The feature is queryable at compile time for concrete shapes and at runtime for bucketed shapes. It is not "are these two ops neighbors in the graph." Neighbor fusion is still necessary; it is not sufficient, because neighbors change regime when M, S, or \(t_e\) change.

**Planned calibration, not done.** For each kind, collect Nsight DRAM bytes and achieved FLOPs, set measured AI = FLOPs / DRAM bytes, and report:

> **TBD:** classification accuracy of analytical AI versus measured AI.
> **TBD:** fraction of fusion decisions that flip when the regime is taken from the measurement instead of the formula.
> **TBD:** the sites where the cold-cache bound is most wrong (L2-resident weights, causal masks, paged KV).

The hook is `classify_site(..., measured_ai=...)`. The third A40 bundle leaves `achieved_ai` empty (`achieved_ai_status=TBD_NSIGHT`). `proxy_ai` is filled from `torch.profiler` and stuck at 4096 on every bench row, with status `TORCH_PROFILER_MEMORY_NOT_DRAM`. Archived `agreement.json` has `scic_accuracy_status=PROXY_TORCH_MEMORY_NOT_VALID_SCIC_ACCURACY`, `quote_as_scic_accuracy=false`, `n_with_byte_ai=7`, `n_torch_profiler_memory=7`, `n_nsight_dram_isolated=0`, and both inspection rates equal to 0.2857142857142857 (`n_decision_flips_any_byte_source=5`). Do not quote 0.2857142857142857 as SCIC accuracy. `quote_as_scic_accuracy` is true only when byte AI comes from isolated Nsight DRAM (`NSIGHT_DRAM`). A process-sum Nsight capture, a `torch.profiler` memory estimate, and a caching-allocator delta produce rates for inspection and are marked not valid as that accuracy number.

## 5. Dual-plan selection

Given a report, the stub applies four rules.

1. Elementwise chains, reductions, softmax, RMSNorm, and LayerNorm emit MemMode only.
2. If the regime is memory-bound, emit MemMode only. If compute-bound, emit ComputeMode only.
3. If the regime is ridge-region, emit both. The primary plan is MemMode when AI < R and ComputeMode otherwise. `measured_winner` is empty.
4. Attention never sets `materialize_intermediate`. ComputeMode on attention means "FA-style IO, large tiles, do not also fuse the output epilogue," which is the reading we take of the self-attention counterexample. It does not mean "go back to writing scores."

Shape buckets for the plan cache are 1, 2–8, 9–32, 33–128, 129–512, 513–2048, 2049–8192, and 8193+. The key is `(kind, dtype, machine, bucket)`. `PlanCache.record_measured_winner` exists so a serving run can fill the ridge band later. Nothing in this repository calls it.

Per-expert MoE rows are separate sites. A skewed layer therefore carries a heterogeneous set of primary modes (MemMode on the tail, ComputeMode on the head, both on the experts whose AI sits in the band). The aggregate row is not a substitute for those rows.

**What a runtime would do, and does not do here.** On a captured shape, look up the bucket. If one plan was emitted, run it. If both were emitted, use the cached measurement or run a short calibration. Guard the choice the way a dynamic-shape compiler already guards specialized kernels.

> **TBD:** plan-switch overhead on a length trace and on an MoE routing histogram.
> **TBD:** gap to an oracle that picks the faster plan with knowledge of the timed kernels.
> **TBD:** the kernel bodies themselves. MemMode and ComputeMode are schedules to be written in Triton, CUTLASS, or the compiler IR of choice. This paper's artifact stops at the decision.

Inductor with dynamic shapes still tends to freeze fusion decisions even when the shape symbol is free. The claim is that the missing input is the regime, not another symbolic dimension. We have not modified Inductor. A prototype on top of it is **TBD**.

## 6. Where this sits relative to joint fusion

Joint MatMul–epilogue fusion answers "can these ops share a kernel." SCIC plus the dual plan answers "which of the legal kernels should exist for this shape."

| question | joint fusion | this draft |
| --- | --- | --- |
| Fuse MatMul with a cheap epilogue? | yes, as one schedule | MemMode yes; ComputeMode no, when AI > 2 R |
| Write attention scores? | a joint schedule can lose to an IO-aware kernel | neither plan writes them; the mode selects tiles and extra epilogue |
| One template for every expert? | typical grouped-GEMM epilogue | per-expert regime; heterogeneous plans in one layer |
| Stage-Stream | their mechanism for the joint schedule | not used, not claimed, not implemented |
| Evidence in this draft | — | closed-form AI, a decision table, the third A40 `torch.matmul` curve, a copy-kernel bandwidth estimate (`ESTIMATE_COPY_NOT_NSIGHT`), and a fusion proxy whose ratios are not fusion speedups (§7.2). Fused-kernel time **TBD** |

We expect the head-to-head to be interesting only in the ridge band and on the attention epilogue, where the two policies disagree. Far into the memory side both policies fuse. Far into the compute side the dual plan refuses a fusion that joint fusion might still attempt. That disagreement is the experiment. It has not been run.

## 7. Evaluation status

### 7.1 Analytical reproduction

```bash
pip install -r requirements.txt && pip install -e .
python -m shapefuse selfcheck
bash scripts/run_cpu.sh
```

`python -m shapefuse selfcheck` locks the closed forms used in §3 (ridges, including the A40 datasheet preset, M*, GEMM AI at M = 1 and M = 2048, softmax, FA AI = S/2, MoE aggregates, the byte-fraction direction, and the attention plan switch).

CPU and NumPy timers from `scripts/run_cpu.sh` are `PLACEHOLDER_NO_GPU`. They are not in the table below.

### 7.2 NVIDIA A40, third package (`PARTIAL_VENDOR_MATMUL`)

Archived into `paper/data/a40/` (`bench_gemm.csv`, `summary.json`, `gpu_measurements.json`, `crossings.json`, `agreement.json`, `fusion_proxy.csv`, `bandwidth_estimate.json`, `nsight_skipped.txt`, `nsight_error.log`). This upload did not include `decision_log.csv`. The matmul command recorded in `summary.json` is:

```text
/root/test/shapefuse/__main__.py all --out /root/test/results/a40 --device cuda --require-cuda --bench-level a40 --machine A40_FP16 --profile torch --fusion-proxy
```

`nsight_skipped.txt` records that `TRY_NSIGHT=1` was set and `ncu` was on `PATH`, and that `scripts/nsight_dram.sh` failed. `nsight_error.log` is:

```text
==PROF== Connected to process 30939 (/usr/bin/python3.10)
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0. For instructions on enabling permissions and to get more information see https://developer.nvidia.com/ERR_NVGPUCTRPERM
==PROF== Disconnected from process 30939
```

Environment, from `summary.json`: device NVIDIA A40, PyTorch 2.5.1+cu121, CUDA 12.1, Python 3.10.12. `--machine A40_FP16` selects the A40 datasheet anchor (ridge 215.08620689655172 FLOP/byte, dense FP16 Tensor Core 149.7 TFLOP/s, GDDR6 696 GB/s). `summary.json` and `crossings.json` include `A40_FP16` with M(0.5 R) = 113.50330752177987, M\* = 240.32584259093773, M(2 R) = 544.5531470469126. `--bench-level a40` is the seven shapes with N = K = 4096.

`gpu_measurement_status` is `PARTIAL_VENDOR_MATMUL`. `bench.nsight_status` is still `TBD_RUN_PROFILER` because no `nsight_dram.json` was ingested. `nsight_rows_filled` is 0. `nsight_dram_bytes` and `achieved_ai` are empty on every row (`achieved_ai_status=TBD_NSIGHT`). `kernel_emitted` is 0. No MemMode or ComputeMode kernel ran.

#### Vendor matmul curve

What was timed: stock `torch.matmul`, requested dtype FP16. Every row has `kernel=pytorch_matmul`, `fused_plan_timed=false`, `placeholder=false`. Medians and TFLOP/s below are the CSV literals. Analytical AI is the FP16 cold-cache value in that file. Both regime columns are datasheet bands.

| label | M | median_s | achieved TFLOP/s | analytical AI | vs A40 datasheet | vs H100 datasheet |
| --- | --- | --- | --- | --- | --- | --- |
| decode_gemv | 1 | 7.072000206e-05 | 0.4744687645 | 0.9995 | memory-bound | memory-bound |
| micro_batch | 4 | 7.864000276e-05 | 1.706736054 | 3.992 | memory-bound | memory-bound |
| decode_batch32 | 32 | 7.822399959e-05 | 13.7265012 | 31.51 | memory-bound | memory-bound |
| near_mstar | 344 | 0.000141967997 | 81.30511704 | 294.5 | ridge-region | ridge-region |
| ridge_m512 | 512 | 0.0001788479984 | 96.05849291 | 409.6 | ridge-region | ridge-region |
| prefill_m2048 | 2048 | 0.0006588959992 | 104.2948763 | 1024 | compute-bound | compute-bound |
| square_large | 4096 | 0.00119116801 | 115.3816693 | 1365 | compute-bound | compute-bound |

`summary.json` records `achieved_tflops_max` as 115.38166932883347 and `fraction_of_requested_datasheet_peak` as 0.7707526341271441. That is this maximum divided by the A40 datasheet peak 149.7, about 77.1% of that datasheet number. `fraction_of_h100_fp16_datasheet_peak` is 0.11666498415453333, the same rate over the H100 datasheet 989. `measured_hbm_bandwidth_bytes_per_s` and `measured_ridge_flops_per_byte` are null.

What the curve supports: achieved TFLOP/s rises from 0.4744687645 at M = 1 through 1.706736054 (M = 4), 13.7265012 (M = 32), 81.30511704 (M = 344), 96.05849291 (M = 512), and 104.2948763 (M = 2048), to 115.3816693 at M = 4096. For M ≤ 32 the median stays near 7.07×10⁻⁵ to 7.86×10⁻⁵ s while FLOPs grow by 32×. That is the launch-dominated end of a vendor matmul. It does not locate a measured ridge, and it does not score SCIC against DRAM bytes.

What it does not support: a fused-kernel speedup, MemMode versus ComputeMode, Nsight achieved AI, a measured GDDR bandwidth, or an H100 latency.

![NVIDIA A40 torch.matmul FP16 throughput](fig_a40_torch_matmul_tflops.png)

*Figure. Seven `torch.matmul` rates from the third A40 `bench_gemm.csv`. The dashed line is the A40 datasheet dense FP16 peak (149.7 TFLOP/s), not a measurement. Log scale. Not a fused-plan comparison. Nsight bytes are absent (`ERR_NVGPUCTRPERM`).*

No H100 matmul has been run. Those cells stay empty:

| shape | H100 median s | achieved TFLOP/s | Nsight DRAM bytes | achieved AI |
| --- | --- | --- | --- | --- |
| 1×4096×4096 | TBD | TBD | TBD | TBD |
| 32×4096×4096 | TBD | TBD | TBD | TBD |
| 344×4096×4096 | TBD | TBD | TBD | TBD |
| 2048×4096×4096 | TBD | TBD | TBD | TBD |
| 4096×4096×4096 | TBD | TBD | TBD | TBD |

#### Profiler bytes are not achieved AI

Every bench row has `proxy_byte_source=key_averages.self_device_memory_usage`, `proxy_ai_status=TORCH_PROFILER_MEMORY_NOT_DRAM`, and `proxy_ai=4096`. The byte counts are 8192, 32768, 262144, 2818048, 4194304, 16777216, and 33554432. Those equal the FP16 output matrix alone (`M × 4096 × 2`). Dividing `2MNK` by that count cancels M and leaves K, so `proxy_ai` is stuck at 4096. An AI of 4096 sits above 2 R on both the A40 and the H100 datasheet bands, so the profiler label is compute-bound on all seven shapes. The analytical labels are compute-bound only at M ∈ {2048, 4096}.

`agreement.json` therefore records `regime_agreement_rate_any_byte_source=0.2857142857142857`, `decision_agreement_rate_any_byte_source=0.2857142857142857`, and `n_decision_flips_any_byte_source=5`. `n_nsight_dram_isolated` and `n_nsight_process_sum` are 0. `scic_accuracy_status` is `PROXY_TORCH_MEMORY_NOT_VALID_SCIC_ACCURACY`. `quote_as_scic_accuracy` is false. Do not quote 0.2857142857142857 as SCIC accuracy.

#### Copy-kernel bandwidth (`ESTIMATE_COPY_NOT_NSIGHT`)

`bandwidth_estimate.json` is present. `status` is `ESTIMATE_COPY_NOT_NSIGHT`. `not_nsight`, `not_a_measured_ridge`, and `do_not_write_measured_ridge_flops_per_byte` are true. The probe timed a device-to-device copy (10 repeats, 268435456 FP16 elements, read plus write, 10737418240 bytes in 0.01883420753479004 s). `estimated_bandwidth_gb_per_s` is 570.1019392595165 (about 570.1 GB/s). `estimated_ridge_flops_per_byte` is 262.5846181025793. The numerator is the datasheet FLOP peak, 149.7 TFLOP/s, divided by that copy bandwidth. `measured_hbm_bandwidth_bytes_per_s` and `measured_ridge_flops_per_byte` are null. This file does not replace the datasheet ridge of 215.08620689655172, and it is not Nsight DRAM.

#### Fusion proxy (`PROXY_NOT_FUSION_SPEEDUP`)

`summary.json` records `fusion_proxy.ran=true`, `status=PROXY_NOT_FUSED_GEMM`, `triton_used=true`, `torch_compile_used=true`, and `claims_memmode_kernel`, `claims_computemode_kernel`, and `claims_stage_stream` all false. The note in that object says not to report the time ratio as a fusion speedup.

Every data row in `fusion_proxy.csv` has `time_ratio_status=PROXY_TIMING_NOT_A_FUSION_SPEEDUP`, `memmode_kernel=false`, `computemode_kernel=false`, `kernel_emitted=false`, and `ok=true`. The ratio is unfused-region time divided by this variant's time. It is a proxy timing.

matmul + RMSNorm, three variants that did run:

- `unfused_two_regions` (`TWO_KERNEL_BASELINE_NOT_A_FUSED_GEMM`): matmul and RMSNorm timed as two regions and added. Ratio 1 by construction.
- `eager_chain` (`EAGER_CHAIN_STILL_SEPARATE_OPS`): the same two ops inside one Python region. Eager mode still launches both. CSV ratios: 1.398085063 (M = 1), 1.328853438 (M = 4), 1.393103491 (M = 32), 1.332332706 (M = 344), 1.143115217 (M = 512), 1.016681221 (M = 2048), 1.012774456 (M = 4096).
- `torch_compile` (`TORCH_COMPILE_PROXY_FUSION_NOT_VERIFIED`): `torch.compile` on that chain. This repo does not inspect the graph, so a ratio is not evidence of fusion. CSV ratios: 1.378291104 (M = 1), 1.509902163 (M = 4), 1.188235255 (M = 32), 1.201624475 (M = 344), 1.175736637 (M = 512), 1.161318498 (M = 2048), 1.211543482 (M = 4096).

bias + GELU, on an already materialized matrix, not a GEMM fusion:

- `unfused_two_regions` versus `hand_elementwise_stub` (`ELEMENTWISE_HAND_FUSED_STUB_NOT_GEMM_FUSION`). Hand-stub ratios from the CSV: 1.132994609 (M = 1), 1.117102881 (M = 4), 1.138145736 (M = 32), 1.101210125 (M = 344), 1.107857388 (M = 512), 1.022433835 (M = 2048), 1.013019741 (M = 4096).
- `triton` ran. All seven rows have `ok=true` and honesty `TRITON_ELEMENTWISE_STUB_NOT_GEMM_FUSION`. CSV ratios: 1.993904752 (M = 1), 1.51406735 (M = 4), 1.505031954 (M = 32), 1.557734173 (M = 344), 1.879358184 (M = 512), 4.484666033 (M = 2048), 5.375718689 (M = 4096). `triton_used` is true. The stub is one elementwise kernel on an already materialized matrix. It is not a fused GEMM, and the ratio stays `PROXY_TIMING_NOT_A_FUSION_SPEEDUP`. The previous package's `NameError('tl is not defined')` is not this package's result.

### 7.3 Next server run

Nsight is still the missing measurement. The third package already passed `TRY_NSIGHT=1` with `ncu` on `PATH`. The capture failed with `ERR_NVGPUCTRPERM`. The next run has to fix that permission, reboot, and pass `TRY_NSIGHT=1` again. Steps for an administrator are in [`docs/NSIGHT_PERMISSIONS.md`](../docs/NSIGHT_PERMISSIONS.md): `modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0`, or the same option in `/etc/modprobe.d/`, then reboot, then verify `ncu`.

```bash
# After the permission change and a reboot. DRAM bytes are still TBD.
TRY_NSIGHT=1 bash scripts/run_a40.sh
# If PyTorch is already installed:
TRY_NSIGHT=1 SKIP_INSTALL=1 bash scripts/run_a40.sh
```

`ncu` is Nsight Compute. It is not the NVIDIA driver and not `nvidia-smi`. If `ncu` is not on `PATH`, `scripts/run_a40.sh` says so, writes `results/a40/nsight_skipped.txt` when `TRY_NSIGHT=1`, and still runs the matmul sweep. If `ncu` fails with `ERR_NVGPUCTRPERM`, the script says so, keeps `nsight_error.log`, writes `nsight_skipped.txt`, and points at `docs/NSIGHT_PERMISSIONS.md`. `achieved_ai` stays empty in both cases. Install Nsight Compute when `ncu` is missing, check `ncu --version`, and list metrics with `ncu --query-metrics | grep dram__bytes` before expecting `nsight_dram.json`. A permission error is not fixed by changing `NCU_METRICS`.

The script checks `nvidia-smi` and `torch.cuda.is_available()`, prints the device name, and warns if that name does not contain `A40`. It then runs `--machine A40_FP16`, `--bench-level a40`, `--profile torch`, and `--fusion-proxy`. After the sweep it runs a copy-kernel bandwidth probe (`scripts/bandwidth_probe.sh`, or the same module). That file is `bandwidth_estimate.json` with status `ESTIMATE_COPY_NOT_NSIGHT`. The ridge numerator is the datasheet FLOP peak. It does not fill `measured_ridge_flops_per_byte`. The third package already wrote that estimate (§7.2); another copy does not turn it into a measured ridge.

What this package already showed about the hooks (these are results, with the limits above):

- The Triton bias+GELU stub launched. Honesty on those rows is `TRITON_ELEMENTWISE_STUB_NOT_GEMM_FUSION`. The stub is still one elementwise kernel, not a fused GEMM.
- `torch.profiler` memory was read from `self_device_memory_usage`. Every row is `TORCH_PROFILER_MEMORY_NOT_DRAM`, and `proxy_ai` is stuck at 4096 because the counts match the output matrix. That status does not set `quote_as_scic_accuracy`.

Honesty of the outputs:

| file | what it is |
| --- | --- |
| `bench_gemm.csv` | `torch.matmul` only. `achieved_ai` stays empty unless `nsight_dram.json` was ingested |
| `proxy_ai` | FLOPs / allocator bytes. In this package: `TORCH_PROFILER_MEMORY_NOT_DRAM`, value 4096, output-sized counts. `TORCH_ALLOCATOR_DELTA_NOT_DRAM` if only the caching allocator moved. `TORCH_PROFILER_EMPTY` if neither did. Not DRAM |
| `fusion_proxy.csv` | matmul then RMSNorm, versus an eager chain and `torch.compile`; bias+GELU as two ops versus one Python expression; Triton elementwise stub (`TRITON_ELEMENTWISE_STUB_NOT_GEMM_FUSION` in this package). `time_ratio_status` stays `PROXY_TIMING_NOT_A_FUSION_SPEEDUP`. `memmode_kernel` and `computemode_kernel` are false |
| `agreement.json` | inspection rates when a byte AI exists. This package: `PROXY_TORCH_MEMORY_NOT_VALID_SCIC_ACCURACY`, rate 0.2857142857142857, `quote_as_scic_accuracy` false. Do not quote that rate as SCIC accuracy |
| `nsight_dram.json` | only if `TRY_NSIGHT=1` and `ncu` succeeds. Default metrics `dram__bytes_read.sum` + `dram__bytes_write.sum`, summed over the short process. Status `NSIGHT_DRAM_PROCESS_SUM` |
| `nsight_error.log` | `ncu` stderr when the capture fails. This package: `ERR_NVGPUCTRPERM` |
| `nsight_skipped.txt` | written when `TRY_NSIGHT=1` but `ncu` is missing or the capture fails |
| `bandwidth_estimate.json` | copy-kernel estimate. Status `ESTIMATE_COPY_NOT_NSIGHT`. This package: 570.1019392595165 GB/s, estimated ridge 262.5846181025793. Not a measured ridge |

`scripts/run_h100.sh` is unchanged in default behavior (matmul sweep, profile off, no fusion proxy). `PROFILE=torch FUSION_PROXY=1` opts into the same hooks.

### 7.4 What a complete evaluation still needs

| experiment | status |
| --- | --- |
| Analytical AI, regimes, crossings, plan table, figures | done, this repo |
| A40 `torch.matmul` latency and TFLOP/s across M, A40 datasheet regimes | done, §7.2, third package, `PARTIAL_VENDOR_MATMUL`, `--machine A40_FP16` |
| A40 fusion proxy (not a fused GEMM) | ran, §7.2. Triton bias+GELU rows are `TRITON_ELEMENTWISE_STUB_NOT_GEMM_FUSION` with `ok=true`. Ratios are `PROXY_TIMING_NOT_A_FUSION_SPEEDUP` |
| A40 copy-kernel bandwidth estimate | written, §7.2, `ESTIMATE_COPY_NOT_NSIGHT`, 570.1019392595165 GB/s, estimated ridge 262.5846181025793. `measured_ridge_flops_per_byte` is null |
| H100 `torch.matmul` latency and TFLOP/s across M | **TBD**, script is `scripts/run_h100.sh` |
| Nsight DRAM bytes and achieved AI | **TBD**. `nsight_status` is `TBD_RUN_PROFILER`. Third run: `TRY_NSIGHT=1`, `ncu` on `PATH`, failed with `ERR_NVGPUCTRPERM`. Next: admin steps in `docs/NSIGHT_PERMISSIONS.md`, reboot, then `TRY_NSIGHT=1` again. The failure is not a measured zero |
| Measured ridge (peak FLOP/s and memory byte/s on the box) | **TBD**. The copy-kernel file is `ESTIMATE_COPY_NOT_NSIGHT` and is not this row |
| SCIC accuracy vs isolated Nsight AI | **TBD**. Archived `agreement.json` is `PROXY_TORCH_MEMORY_NOT_VALID_SCIC_ACCURACY`, rate 0.2857142857142857, `quote_as_scic_accuracy` false. Do not quote that rate as SCIC accuracy |
| MemMode vs ComputeMode vs always-fuse vs never-fuse | **TBD** (no such kernels; the fusion proxy is not this row) |
| Attention: IO-aware mainloop with and without an extra fused epilogue, short S vs long S | **TBD** |
| MoE skew: per-expert plan vs one template; weights of idle experts loaded or not | **TBD** |
| Quantized epilogue under a measured W8A8/FP8 ridge | **TBD** |
| Plan-cache hit rate and switch overhead on a serving trace | **TBD** |
| End-to-end tokens/s | **TBD** |
| CUDA launch latency to replace the 5 µs sensitivity | **TBD** |

## 8. Related work

**Roofline.** Williams, Waterman, and Patterson (CACM 2009) define the intensity–ridge picture used here. We use it as a per-site compiler feature.

**IO-aware attention.** FlashAttention and its successors show that not writing the score matrix changes both traffic and the practical kernel. We treat that algorithm as a plan that SCIC can select, not as a contribution. The analytical consequence in §3.2 is that, for D = 128 FP16, the alternative of writing scores does not become compute-bound on an H100-class ridge.

**Joint fusion.** ComFuse (bibliographic entry **TBD**) fuses MatMul with its epilogue and uses Stage-Stream to carry that schedule. The self-attention result reported there, joint fusion losing to an IO-aware kernel, is the motivation for ComputeMode on long sequences. Stage-Stream itself is not part of this design.

**Compiler stacks.** TorchInductor, Triton, CUTLASS, and cuBLAS already fuse or specialize many of the sites above. Dynamic shapes do not by themselves retile the fusion decision when the regime crosses R. A prototype that connects SCIC to one of these stacks is **TBD**.

**Grouped GEMM in serving stacks.** TensorRT-LLM and related MoE kernels fuse expert GEMMs with epilogues. Public descriptions do not, as far as this draft verifies, key that fusion on a per-expert intensity bucket. Verification against those implementations is **TBD**.

**Named in the project notes, citation not yet checked.** FuseFlow, Neptune, and RedFuser appear in the internal note as algebraic or schedule-level fusion work. Confirm authors, year, and venue before camera-ready. **TBD.**

## 9. Limitations

- Cold HBM, no L2, no causal mask, no paged KV, no overlap. Decode AI in particular will move if a page table or a batch of short queries changes bytes.
- The 0.5 R / 2 R band is a knob. A measured error distribution should set it.
- Softmax's 5 FLOPs and 3 passes, and RMSNorm's 4 FLOPs, are conventions.
- Datasheet peaks are not the chip in the rack. FP8's near-invariant M* depends on the peak doubling in step with the byte halving.
- The dual plan does not model register spills. ComputeMode's refusal to fuse is a policy motivated by the self-attention counterexample, not a spill cost. A joint cost model (AI together with spill and shared-memory pressure) is future work; this draft does not claim one.
- No MemMode or ComputeMode kernel, and no tokens/s. The third A40 package in §7.2 is `torch.matmul` latency and TFLOP/s, fusion-proxy timings with status `PROXY_TIMING_NOT_A_FUSION_SPEEDUP`, and a copy-kernel bandwidth estimate (`ESTIMATE_COPY_NOT_NSIGHT`, 570.1019392595165 GB/s, estimated ridge 262.5846181025793 from the datasheet FLOP peak divided by that copy bandwidth). `measured_ridge_flops_per_byte` is null. `quote_as_scic_accuracy` is false. The torch-profiler agreement rate 0.2857142857142857 has status `PROXY_TORCH_MEMORY_NOT_VALID_SCIC_ACCURACY` and is not SCIC accuracy.
- Nsight was blocked by `ERR_NVGPUCTRPERM`. `TRY_NSIGHT=1` found `ncu` on `PATH`, and `ncu` could not read GPU performance counters (`nsight_error.log`). `achieved_ai` is still empty. `nsight_status` remains `TBD_RUN_PROFILER`. The admin steps are in `docs/NSIGHT_PERMISSIONS.md`; after a reboot the same capture is `TRY_NSIGHT=1` again.
- The Triton bias+GELU rows in this package did launch (`ok=true`, `TRITON_ELEMENTWISE_STUB_NOT_GEMM_FUSION`). They are an elementwise stub on an already materialized matrix, not a fused GEMM.

## 10. Conclusion

On an H100-class Roofline the FP16 square projection of width 4096 changes regime near M\* ≈ 345. The same closed form on the A40 datasheet (149.7 TFLOP/s, 696 GB/s, R ≈ 215.09) puts M\* at 240.32584259093773, and that crossing is in the third A40 `summary.json` and `crossings.json`. IO-aware attention changes regime with S while materialized attention does not, and a skewed MoE layer contains both regimes in one forward. Fusion decided by operator kind misses those crossings. SCIC publishes the regime from shape and traffic. The dual plan turns the regime into MemMode, ComputeMode, or both, and it does so without taking ownership of Stage-Stream. The device evidence so far is the third A40 `torch.matmul` curve under `--machine A40_FP16` and status `PARTIAL_VENDOR_MATMUL`: throughput rises from 0.4744687645 TFLOP/s at M = 1 to 115.3816693 TFLOP/s at M = 4096, which is 0.7707526341271441 of the A40 datasheet peak and not a measured ridge. The three-way A40 and H100 datasheet bands agree on those seven shapes. `TRY_NSIGHT=1` found `ncu` and then failed with `ERR_NVGPUCTRPERM`, so `achieved_ai` is still empty (`TBD_RUN_PROFILER`). `proxy_ai` is `TORCH_PROFILER_MEMORY_NOT_DRAM` and stuck at 4096. `quote_as_scic_accuracy` is false; the inspection rate 0.2857142857142857 is `PROXY_TORCH_MEMORY_NOT_VALID_SCIC_ACCURACY` and is not SCIC accuracy. A copy-kernel estimate is `ESTIMATE_COPY_NOT_NSIGHT` (570.1019392595165 GB/s, estimated ridge 262.5846181025793); `measured_ridge_flops_per_byte` is null. A fusion proxy ran; its time ratios are `PROXY_TIMING_NOT_A_FUSION_SPEEDUP`. The Triton bias+GELU rows are `TRITON_ELEMENTWISE_STUB_NOT_GEMM_FUSION` with `ok=true`, which is an elementwise stub, not a fused GEMM. Fused-kernel time, a measured ridge, and tokens/s are still **TBD**.

## A. Formulas implemented in `shapefuse.intensity`

| site | AI |
| --- | --- |
| GEMM | \(2MNK / (MK e_A + KN e_B + MN e_C)\) |
| square projection | \(2Md / ((d+2M)\,e)\) |
| QK\(^\top\) to HBM | \(2SD / ((2D+S)\,e)\) |
| softmax | \(5/(3e)\) |
| FA-style prefill | \(S/e\) |
| unfused prefill | \(4 B H S^2 D / ((4 BHSD + 3 BHS^2)\,e)\), saturating at \(4D/(3e)\) |
| FA-style decode | \(4 H_q S_q S_{kv} D / ((2 H_q S_q D + 2 H_{kv} S_{kv} D)\,e)\) per batch item, → \(2 H_q/(H_{kv} e)\) |
| RMSNorm | \(4MK / ((2MK+K)\,e)\) |
| grouped expert | \(2 t H I / ((tH)\,e_A + (HI)\,e_B + (tI)\,e_C)\) |

Regime: memory if AI < 0.5 R, compute if AI > 2 R, else ridge. Roofline seconds: \(\max(\mathrm{FLOPs}/P,\ \mathrm{bytes}/B)\).

## B. Artifact map

| claim in the text | file produced by the CLI |
| --- | --- |
| GEMM table, M* | `gemm_shapes.csv`, `crossings.json`, `fig_gemm_ai_vs_M.png` |
| Attention | `attention_intensity.csv`, `attention_decode.csv`, `fig_attn_ai_vs_S.png` |
| Epilogue fraction | `rmsnorm_epilogue.csv`, `fig_epilogue_tradeoff.png` |
| MoE | `moe_grouped.csv`, `fig_moe_skew_ai.png` |
| Quantization what-if | `quant_epilogue.csv` |
| Elementwise | `ew_chains.csv` |
| Plan decisions | `dual_plans.csv`, `fig_plan_regions.png` |
| Device matmul | `bench_gemm.csv`, `gpu_measurements.json`; third A40 archive in `paper/data/a40/`, figure `paper/fig_a40_torch_matmul_tflops.png` |
| A40 crossings | `crossings.json` key `A40_FP16` in that archive, matching `summary.json` (datasheet, not measured) |
| Agreement | `agreement.json` in the third archive. `quote_as_scic_accuracy` is false (`PROXY_TORCH_MEMORY_NOT_VALID_SCIC_ACCURACY`). This upload has no `decision_log.csv` |
| Fusion proxy | `fusion_proxy.csv` (`PROXY_TIMING_NOT_A_FUSION_SPEEDUP`; Triton rows `TRITON_ELEMENTWISE_STUB_NOT_GEMM_FUSION`; not MemMode / ComputeMode) |
| Nsight DRAM | `nsight_dram.json` when `TRY_NSIGHT=1` and `ncu` succeed. This package: `nsight_error.log` (`ERR_NVGPUCTRPERM`) and `nsight_skipped.txt`. Admin steps: `docs/NSIGHT_PERMISSIONS.md` |
| Copy-kernel estimate | `bandwidth_estimate.json`, status `ESTIMATE_COPY_NOT_NSIGHT`, estimated ridge 262.5846181025793, `measured_ridge_flops_per_byte` null |

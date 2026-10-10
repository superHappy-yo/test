# Shape-Conditioned Intensity Classification for Plan-Polymorphic Operator Fusion

**Target venues:** MLSys or ASPLOS.
**Status:** draft. Analytical results are produced by this repository. The second NVIDIA A40 package is archived under `paper/data/a40/` (`PARTIAL_VENDOR_MATMUL`, `--machine A40_FP16`, datasheet ridge ≈ 215.09, M\* ≈ 240.3). It times `torch.matmul` and a fusion proxy. Nsight DRAM bytes were not captured (`nsight_status` `TBD_RUN_PROFILER`). `quote_as_scic_accuracy` is false. Fused MemMode/ComputeMode kernels and a measured ridge are not in that run.
**Authors:** TBD.

> Remaining **TBD** markers are empty on purpose. Do not fill them with datasheet arithmetic, with the CPU proxy, with `proxy_ai`, or by relabeling the A40 `torch.matmul` sweep or `fusion_proxy.csv` as a fused-kernel result. This package's three-way regime column uses the A40 FP16 datasheet because the run passed `--machine A40_FP16`. That ridge was not measured on the card. Nsight is still missing.

## Abstract

Whether a tensor operator is compute-bound or memory-bound is a function of shape, dtype, and which intermediates touch HBM, not a label of the operator's name. On an H100-class Roofline (datasheet dense FP16 Tensor Core 989 TFLOP/s, HBM3 3.35 TB/s, ridge R ≈ 295 FLOP/byte), a single FP16 GEMM with N = K = 4096 has arithmetic intensity AI ≈ 1 at M = 1 and AI = 1024 at M = 2048. The same attention block is memory-bound for every sequence length we sweep if it materializes scores, and compute-bound for long sequences if it keeps scores off HBM. A skewed mixture-of-experts layer holds memory-bound and compute-bound experts at once.

Fusion that is decided once per operator kind cannot track that crossing. We describe a shape-conditioned intensity classifier (SCIC) that places each producer–consumer site on the Roofline before scheduling, and a dual-plan policy that emits a memory-oriented plan (MemMode), a compute-oriented plan (ComputeMode), or both when the site sits in a band around the ridge. Relative to joint MatMul–epilogue fusion, the added mechanism is the classification and the switch. We do not implement Stage-Stream, and we do not claim it.

The classifier, the plan table, and the closed-form figures are analytical. One NVIDIA A40 run (`--machine A40_FP16`) times stock `torch.matmul` at seven shapes (`PARTIAL_VENDOR_MATMUL` in §7.2). The largest rate in that CSV is 115.0155266 TFLOP/s, and `fraction_of_requested_datasheet_peak` is 0.7683067910328436 (about 76.8% of the A40 datasheet dense FP16 peak, 149.7). That fraction is not a measured ridge. A fusion proxy ran; its time ratios are `PROXY_TIMING_NOT_A_FUSION_SPEEDUP`, not a fused-kernel speedup. Nsight byte counts, a measured ridge, MemMode versus ComputeMode, and end-to-end tokens/s are **TBD**.

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

The A40 row is the same closed form on the A40 datasheet preset (`--machine a40`). The second uploaded `crossings.json` was produced with `--machine A40_FP16` and stores that row: ridge 215.08620689655172, M at 0.5 R = 113.50330752177987, M\* = 240.32584259093773, M at 2 R = 544.5531470469126. Those are datasheet crossings, not a measured A40 ridge.

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

On the seven shapes timed in §7.2, `regime_vs_h100_datasheet` and `regime_vs_requested_datasheet` agree: memory-bound at M ∈ {1, 4, 32}, ridge-region at M ∈ {344, 512}, compute-bound at M ∈ {2048, 4096}. The strict knee does not. Analytical AI at M = 344 is 294.5284281, just below the H100 ridge (295.22) and above the A40 datasheet ridge (215.08620689655172). `decision_log.csv` records `analytical_roofline_limit=compute` for `near_mstar` under `A40_FP16`, with `analytical_emit_both=true` because the three-way band is still ridge-region. Neither column is a measured ridge.

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

The hook is `classify_site(..., measured_ai=...)`. The second A40 bundle leaves `achieved_ai` empty and `proxy_ai` empty (`TORCH_PROFILER_EMPTY`), so this draft still has no measured-AI accuracy to report. Archived `agreement.json` has `scic_accuracy_status=TBD_NO_MEASURED_AI`, `quote_as_scic_accuracy=false`, `n_with_byte_ai=0`, and `n_torch_profiler_memory=0`. `quote_as_scic_accuracy` is true only when byte AI comes from isolated Nsight DRAM (`NSIGHT_DRAM`). A process-sum Nsight capture, a `torch.profiler` memory estimate, and a caching-allocator delta produce rates for inspection and are marked not valid as that accuracy number.

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
| Evidence in this draft | — | closed-form AI, a decision table, one A40 `torch.matmul` curve, and a fusion proxy whose ratios are not fusion speedups (§7.2). Fused-kernel time **TBD** |

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

### 7.2 NVIDIA A40, second package (`PARTIAL_VENDOR_MATMUL`)

Archived into `paper/data/a40/` (`bench_gemm.csv`, `summary.json`, `gpu_measurements.json`, `crossings.json`, `decision_log.csv`, `agreement.json`, `fusion_proxy.csv`). The recorded command is:

```text
/root/test/shapefuse/__main__.py all --out /root/test/results/a40 --device cuda --require-cuda --bench-level a40 --machine A40_FP16 --profile torch --fusion-proxy
```

Environment, from `summary.json`: device NVIDIA A40, PyTorch 2.5.1+cu121, CUDA 12.1, Python 3.10.12. `--machine A40_FP16` selects the A40 datasheet anchor (ridge 215.08620689655172 FLOP/byte, dense FP16 Tensor Core 149.7 TFLOP/s, GDDR6 696 GB/s). `crossings.json` includes `A40_FP16` with M(0.5 R) = 113.50330752177987, M\* = 240.32584259093773, M(2 R) = 544.5531470469126. `--bench-level a40` is the seven shapes with N = K = 4096.

`gpu_measurement_status` is `PARTIAL_VENDOR_MATMUL`. `bench.nsight_status` is `TBD_RUN_PROFILER`. `nsight_rows_filled` is 0. `nsight_dram_bytes` and `achieved_ai` are empty on every row (`achieved_ai_status=TBD_NSIGHT`). Every row has `proxy_ai_status=TORCH_PROFILER_EMPTY`, so `proxy_ai` is empty too. `agreement.json` reports `n_shapes=7`, `n_with_byte_ai=0`, `n_torch_profiler_memory=0`, `scic_accuracy_status=TBD_NO_MEASURED_AI`, `quote_as_scic_accuracy=false`. `kernel_emitted` is 0. No MemMode or ComputeMode kernel ran.

#### Vendor matmul curve

What was timed: stock `torch.matmul`, requested dtype FP16. Every row has `kernel=pytorch_matmul`, `fused_plan_timed=false`, `placeholder=false`. Medians and TFLOP/s below are the CSV literals. Analytical AI is the FP16 cold-cache value in that file. Both regime columns are datasheet bands.

| label | M | median_s | achieved TFLOP/s | analytical AI | vs A40 datasheet | vs H100 datasheet |
| --- | --- | --- | --- | --- | --- | --- |
| decode_gemv | 1 | 7.160000131e-05 | 0.4686373099 | 0.9995 | memory-bound | memory-bound |
| micro_batch | 4 | 7.948799804e-05 | 1.688528222 | 3.992 | memory-bound | memory-bound |
| decode_batch32 | 32 | 7.948800176e-05 | 13.50822514 | 31.51 | memory-bound | memory-bound |
| near_mstar | 344 | 0.000143648006 | 80.35422786 | 294.5 | ridge-region | ridge-region |
| ridge_m512 | 512 | 0.0001795520037 | 95.68185721 | 409.6 | ridge-region | ridge-region |
| prefill_m2048 | 2048 | 0.0006596159935 | 104.1810347 | 1024 | compute-bound | compute-bound |
| square_large | 4096 | 0.001194959998 | 115.0155266 | 1365 | compute-bound | compute-bound |

`summary.json` records `achieved_tflops_max` as 115.01552661761667 and `fraction_of_requested_datasheet_peak` as 0.7683067910328436. That is this maximum divided by the A40 datasheet peak 149.7, about 76.8% of that datasheet number. `fraction_of_h100_fp16_datasheet_peak` is 0.11629476907746883, the same rate over the H100 datasheet 989. `measured_hbm_bandwidth_bytes_per_s` and `measured_ridge_flops_per_byte` are null.

`decision_log.csv` on the A40 datasheet ridge, with no byte AI: M ∈ {1, 4, 32} is memory-bound, primary MemMode, `emit_both` false. M ∈ {344, 512} is ridge-region, strict limit compute, primary ComputeMode, `emit_both` true. M ∈ {2048, 4096} is compute-bound, primary ComputeMode, `emit_both` false. `byte_ai_status` is `TBD_NO_MEASURED_AI` on every row.

What the curve supports: achieved TFLOP/s rises from 0.4686373099 at M = 1 through 1.688528222 (M = 4), 13.50822514 (M = 32), 80.35422786 (M = 344), 95.68185721 (M = 512), and 104.1810347 (M = 2048), to 115.0155266 at M = 4096. For M ≤ 32 the median stays near 7.2×10⁻⁵ to 7.9×10⁻⁵ s while FLOPs grow by 32×. That is the launch-dominated end of a vendor matmul. It does not locate a measured ridge, and it does not score SCIC against DRAM bytes.

What it does not support: a fused-kernel speedup, MemMode versus ComputeMode, Nsight achieved AI, a measured GDDR bandwidth, or an H100 latency.

![NVIDIA A40 torch.matmul FP16 throughput](fig_a40_torch_matmul_tflops.png)

*Figure. Seven `torch.matmul` rates from the second A40 `bench_gemm.csv`. The dashed line is the A40 datasheet dense FP16 peak (149.7 TFLOP/s), not a measurement. Log scale. Not a fused-plan comparison. Nsight bytes are absent.*

No H100 matmul has been run. Those cells stay empty:

| shape | H100 median s | achieved TFLOP/s | Nsight DRAM bytes | achieved AI |
| --- | --- | --- | --- | --- |
| 1×4096×4096 | TBD | TBD | TBD | TBD |
| 32×4096×4096 | TBD | TBD | TBD | TBD |
| 344×4096×4096 | TBD | TBD | TBD | TBD |
| 2048×4096×4096 | TBD | TBD | TBD | TBD |
| 4096×4096×4096 | TBD | TBD | TBD | TBD |

#### Fusion proxy (`PROXY_NOT_FUSION_SPEEDUP`)

`summary.json` records `fusion_proxy.ran=true`, `status=PROXY_NOT_FUSED_GEMM`, `triton_used=false`, `torch_compile_used=true`, and `claims_memmode_kernel`, `claims_computemode_kernel`, and `claims_stage_stream` all false. The note in that object says not to report the time ratio as a fusion speedup.

Every data row in `fusion_proxy.csv` has `time_ratio_status=PROXY_TIMING_NOT_A_FUSION_SPEEDUP`, `memmode_kernel=false`, `computemode_kernel=false`, and `kernel_emitted=false`. The ratio is unfused-region time divided by this variant's time. It is a proxy timing. It is not a fusion speedup.

matmul + RMSNorm, three variants that did run:

- `unfused_two_regions` (`TWO_KERNEL_BASELINE_NOT_A_FUSED_GEMM`): matmul and RMSNorm timed as two regions and added. Ratio 1 by construction.
- `eager_chain` (`EAGER_CHAIN_STILL_SEPARATE_OPS`): the same two ops inside one Python region. Eager mode still launches both. CSV ratios: 1.388005185 (M = 1), 1.46125371 (M = 4), 1.42554845 (M = 32), 1.343799918 (M = 344), 1.113447022 (M = 512), 1.013099176 (M = 2048), 1.006475964 (M = 4096).
- `torch_compile` (`TORCH_COMPILE_PROXY_FUSION_NOT_VERIFIED`): `torch.compile` on that chain. This repo does not inspect the graph, so a ratio is not evidence of fusion. CSV ratios: 0.8089665914 (M = 1, slower than the two-region sum), 1.315476276 (M = 4), 1.284950052 (M = 32), 1.184333985 (M = 344), 1.156591259 (M = 512), 1.162509833 (M = 2048), 1.201995369 (M = 4096).

bias + GELU, on an already materialized matrix, not a GEMM fusion:

- `unfused_two_regions` versus `hand_elementwise_stub` (`ELEMENTWISE_HAND_FUSED_STUB_NOT_GEMM_FUSION`). Hand-stub ratios from the CSV: 1.162706454 (M = 1), 1.127369426 (M = 4), 1.149630012 (M = 32), 1.113151956 (M = 344), 1.106452186 (M = 512), 1.018106944 (M = 2048), 1.011971539 (M = 4096).
- `triton` did not run. All seven rows have `ok=false`, honesty `TRITON_KERNEL_FAILED`, and empty `median_s`. The error text is `NameError('tl is not defined')` at `BLOCK: tl.constexpr`. Triton was importable enough to reach the kernel annotation and then failed. There is no Triton timing in this package.

### 7.3 Next server run

Nsight is still the missing measurement. The next run should pass `TRY_NSIGHT=1`.

```bash
# Strongly recommended. DRAM bytes are still TBD.
TRY_NSIGHT=1 bash scripts/run_a40.sh
# If PyTorch is already installed:
TRY_NSIGHT=1 SKIP_INSTALL=1 bash scripts/run_a40.sh
```

`ncu` is Nsight Compute. It is not the NVIDIA driver and not `nvidia-smi`. If `ncu` is not on `PATH`, `scripts/run_a40.sh` says so, writes `results/a40/nsight_skipped.txt` when `TRY_NSIGHT=1`, and still runs the matmul sweep. `achieved_ai` stays empty. Install Nsight Compute, check `ncu --version`, and list metrics with `ncu --query-metrics | grep dram__bytes` before expecting `nsight_dram.json`.

The script checks `nvidia-smi` and `torch.cuda.is_available()`, prints the device name, and warns if that name does not contain `A40`. It then runs `--machine A40_FP16`, `--bench-level a40`, `--profile torch`, and `--fusion-proxy`. After the sweep it runs a copy-kernel bandwidth probe (`scripts/bandwidth_probe.sh`, or the same module). That file is `bandwidth_estimate.json` with status `ESTIMATE_COPY_NOT_NSIGHT`. The ridge numerator is the datasheet FLOP peak. It does not fill `measured_ridge_flops_per_byte`.

Code changes relative to the package that produced §7.2, for that next run only (they are not results):

- The Triton bias+GELU stub imports `triton.language as tl` in module scope, which is what `@triton.jit` uses when it evaluates `tl.constexpr`. The archived `NameError` came from a function-local import. The stub is still one elementwise kernel, not a fused GEMM.
- `torch.profiler` memory is read from `self_device_memory_usage` (PyTorch 2.5's `key_averages()` field). If those events are still empty, a positive caching-allocator delta can fill `proxy_ai` with status `TORCH_ALLOCATOR_DELTA_NOT_DRAM`. `TORCH_PROFILER_EMPTY` remains the status when neither source has bytes. Neither status is DRAM, and neither sets `quote_as_scic_accuracy`.

Honesty of the outputs:

| file | what it is |
| --- | --- |
| `bench_gemm.csv` | `torch.matmul` only. `achieved_ai` stays empty unless `nsight_dram.json` was ingested |
| `proxy_ai` | FLOPs / allocator bytes. `TORCH_PROFILER_MEMORY_NOT_DRAM` if profiler events fired. `TORCH_ALLOCATOR_DELTA_NOT_DRAM` if only the caching allocator moved. `TORCH_PROFILER_EMPTY` if neither did. Not DRAM |
| `fusion_proxy.csv` | matmul then RMSNorm, versus an eager chain and `torch.compile` if it runs; bias+GELU as two ops versus one Python expression; Triton elementwise stub if the import and the launch succeed. `time_ratio_status` stays `PROXY_TIMING_NOT_A_FUSION_SPEEDUP`. `memmode_kernel` and `computemode_kernel` are false |
| `decision_log.csv`, `agreement.json` | analytical SCIC plan versus the regime implied by whatever byte AI exists. `quote_as_scic_accuracy` is false for profiler memory, for the allocator delta, and for `NSIGHT_DRAM_PROCESS_SUM` |
| `nsight_dram.json` | only if `TRY_NSIGHT=1` and `ncu` succeeds. Default metrics `dram__bytes_read.sum` + `dram__bytes_write.sum`, summed over the short process. Status `NSIGHT_DRAM_PROCESS_SUM` |
| `nsight_skipped.txt` | written when `TRY_NSIGHT=1` but `ncu` is missing or the capture fails |
| `bandwidth_estimate.json` | copy-kernel estimate. Status `ESTIMATE_COPY_NOT_NSIGHT`. Not Nsight |

`scripts/run_h100.sh` is unchanged in default behavior (matmul sweep, profile off, no fusion proxy). `PROFILE=torch FUSION_PROXY=1` opts into the same hooks.

### 7.4 What a complete evaluation still needs

| experiment | status |
| --- | --- |
| Analytical AI, regimes, crossings, plan table, figures | done, this repo |
| A40 `torch.matmul` latency and TFLOP/s across M, A40 datasheet regimes | done, §7.2, `PARTIAL_VENDOR_MATMUL`, `--machine A40_FP16` |
| A40 fusion proxy (not a fused GEMM) | ran, §7.2. Triton rows failed with `NameError('tl is not defined')`. Ratios are `PROXY_TIMING_NOT_A_FUSION_SPEEDUP` |
| H100 `torch.matmul` latency and TFLOP/s across M | **TBD**, script is `scripts/run_h100.sh` |
| Nsight DRAM bytes and achieved AI | **TBD**. `nsight_status` is `TBD_RUN_PROFILER`. Next run: `TRY_NSIGHT=1`. If `ncu` is missing, that is an install gap, not a zero |
| Measured ridge (peak FLOP/s and memory byte/s on the box) | **TBD**. A copy-kernel file, if the next run writes one, is `ESTIMATE_COPY_NOT_NSIGHT` and is not this row |
| SCIC accuracy vs isolated Nsight AI | **TBD**. Archived `agreement.json` is `TBD_NO_MEASURED_AI`, `quote_as_scic_accuracy` false |
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
- No MemMode or ComputeMode kernel, no Nsight trace, no tokens/s. The A40 numbers in §7.2 are `torch.matmul` latencies and TFLOP/s, plus fusion-proxy timings that are not fusion speedups. `quote_as_scic_accuracy` is false. The Triton bias+GELU rows in that package failed before producing a time.

## 10. Conclusion

On an H100-class Roofline the FP16 square projection of width 4096 changes regime near M\* ≈ 345. The same closed form on the A40 datasheet (149.7 TFLOP/s, 696 GB/s, R ≈ 215.09) puts M\* at 240.32584259093773, and that crossing is in the second A40 `crossings.json`. IO-aware attention changes regime with S while materialized attention does not, and a skewed MoE layer contains both regimes in one forward. Fusion decided by operator kind misses those crossings. SCIC publishes the regime from shape and traffic. The dual plan turns the regime into MemMode, ComputeMode, or both, and it does so without taking ownership of Stage-Stream. The device evidence so far is one A40 `torch.matmul` curve under `--machine A40_FP16` and status `PARTIAL_VENDOR_MATMUL`: throughput rises from 0.4686373099 TFLOP/s at M = 1 to 115.0155266 TFLOP/s at M = 4096, which is 0.7683067910328436 of the A40 datasheet peak and not a measured ridge. The three-way A40 and H100 datasheet bands agree on those seven shapes. Nsight was not captured (`TBD_RUN_PROFILER`), `proxy_ai` is `TORCH_PROFILER_EMPTY`, and `quote_as_scic_accuracy` is false. A fusion proxy ran; its time ratios are `PROXY_TIMING_NOT_A_FUSION_SPEEDUP`, and the Triton bias+GELU rows failed with `NameError('tl is not defined')`. Fused-kernel time, a measured ridge, and tokens/s are still **TBD**.

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
| Device matmul | `bench_gemm.csv`, `gpu_measurements.json`; second A40 archive in `paper/data/a40/`, figure `paper/fig_a40_torch_matmul_tflops.png` |
| A40 crossings | `crossings.json` key `A40_FP16` in that archive (datasheet, not measured) |
| Decision log | `decision_log.csv`, `agreement.json` (SCIC accuracy only if `quote_as_scic_accuracy`; the archive is false) |
| Fusion proxy | `fusion_proxy.csv` (`PROXY_TIMING_NOT_A_FUSION_SPEEDUP`; not MemMode / ComputeMode) |
| Nsight DRAM | `nsight_dram.json` when `TRY_NSIGHT=1` and `ncu` succeed. Missing `ncu` is `nsight_skipped.txt` |
| Copy-kernel estimate | `bandwidth_estimate.json`, status `ESTIMATE_COPY_NOT_NSIGHT`, not a measured ridge |

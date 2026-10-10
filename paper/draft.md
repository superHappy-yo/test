# Shape-Conditioned Intensity Classification for Plan-Polymorphic Operator Fusion

**Target venues:** MLSys or ASPLOS.
**Status:** draft. Analytical results are produced by this repository. Device results are not.
**Authors:** TBD.

> **TBD** markers below are empty on purpose. Do not replace them with datasheet arithmetic or with the CPU proxy timer.

## Abstract

Whether a tensor operator is compute-bound or memory-bound is a function of shape, dtype, and which intermediates touch HBM, not a label of the operator's name. On an H100-class Roofline (datasheet dense FP16 Tensor Core 989 TFLOP/s, HBM3 3.35 TB/s, ridge R ≈ 295 FLOP/byte), a single FP16 GEMM with N = K = 4096 has arithmetic intensity AI ≈ 1 at M = 1 and AI = 1024 at M = 2048. The same attention block is memory-bound for every sequence length we sweep if it materializes scores, and compute-bound for long sequences if it keeps scores off HBM. A skewed mixture-of-experts layer holds memory-bound and compute-bound experts at once.

Fusion that is decided once per operator kind cannot track that crossing. We describe a shape-conditioned intensity classifier (SCIC) that places each producer–consumer site on the Roofline before scheduling, and a dual-plan policy that emits a memory-oriented plan (MemMode), a compute-oriented plan (ComputeMode), or both when the site sits in a band around the ridge. Relative to joint MatMul–epilogue fusion, the added mechanism is the classification and the switch. We do not implement Stage-Stream, and we do not claim it.

The classifier, the plan table, and the figures in this draft are closed-form. Vendor-matmul latencies, Nsight byte counts, fused-kernel comparisons, and end-to-end tokens/s are **TBD**.

## 1. Introduction

Compilers for GPU kernels already fuse a matrix multiply with the elementwise and normalization code that consumes it. The usual justification is traffic: the product matrix is expensive to write and to read back. That justification is real, and it is incomplete. The same multiply is a different machine regime at decode (a handful of rows) and at prefill (thousands of rows). A fusion that pays for itself on one side of the ridge can crowd registers and the MMA pipeline on the other. Attention makes the gap sharper. FlashAttention-style IO and a score-materializing implementation are not two tunings of one intensity; they are different intensities, and for a 128-wide head in FP16 the materialized version never reaches the H100 ridge no matter how long the sequence is.

This draft argues for plan-polymorphic fusion:

1. Intensity is shape-dependent. GEMM, attention, and grouped GEMM each cross or fail to cross the ridge inside the shape range of a serving stack.
2. A classifier should run before fusion, on the concrete shape, dtype, and traffic accounting, and should return a regime rather than a boolean "these ops are adjacent."
3. The scheduler should be allowed to emit two plans for one site and pick at runtime from the regime, or from a measured cache keyed by a shape bucket.

ComFuse showed that joint MatMul and epilogue fusion is profitable, and it also reported a self-attention case where the joint schedule loses to a FlashAttention-style kernel. That counterexample is a missing switch, not a reason to abandon fusion. Stage-Stream, the streaming mechanism used to carry the joint schedule, is prior work. Nothing in this paper re-derives it or implements it.

**Contributions**

- An analytical SCIC for GEMM, attention (prefill and decode, MHA and GQA), norms, reductions, softmax, elementwise chains, ragged MoE grouped GEMM, and mixed-dtype GEMM epilogues, scored against datasheet ridges for H100 FP16, H100 FP8, and A100 FP16.
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

| machine | peak | HBM | R (FLOP/byte) |
| --- | --- | --- | --- |
| H100 SXM, FP16 | 989 TFLOP/s | 3.35 TB/s | 295.22 |
| H100 SXM, FP8 | 1979 TFLOP/s | 3.35 TB/s | 590.75 |
| A100 80GB SXM, FP16 | 312 TFLOP/s | 2.039 TB/s | 153.02 |

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
| H100 FP8 | 159.2 | 345.2 | 830.2 |

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

A linear layer is not "a compute kernel." Decode and prefill of that layer are different regimes on the same GPU.

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

The hook is `classify_site(..., measured_ai=...)`. This draft leaves `measured_ai` null.

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
| Evidence in this draft | — | closed-form AI and a decision table; kernel time **TBD** |

We expect the head-to-head to be interesting only in the ridge band and on the attention epilogue, where the two policies disagree. Far into the memory side both policies fuse. Far into the compute side the dual plan refuses a fusion that joint fusion might still attempt. That disagreement is the experiment. It has not been run.

## 7. Evaluation status

Reproducing the analytical tables:

```bash
pip install -r requirements.txt && pip install -e .
python -m shapefuse selfcheck
bash scripts/run_cpu.sh
```

`python -m shapefuse selfcheck` locks the closed forms used in §3 (ridge, M*, GEMM AI at M = 1 and M = 2048, softmax, FA AI = S/2, MoE aggregates, the byte-fraction direction, and the attention plan switch).

On the GPU server:

```bash
bash scripts/run_h100.sh
```

That command times `torch.matmul` for seven shapes at N = K = 4096. It is a check that achieved TFLOP/s collapses at small M on the real device. It is not a fusion study. The output marks `fused_plan_timed=false` and leaves `achieved_ai` empty.

### 7.1 What a complete evaluation contains

| experiment | status |
| --- | --- |
| Analytical AI, regimes, crossings, plan table, figures | done, this repo |
| H100 `torch.matmul` latency and TFLOP/s across M | **TBD**, script is `scripts/run_h100.sh` |
| Nsight DRAM bytes and achieved AI | **TBD** |
| Measured ridge (peak FLOP/s and HBM byte/s on the box) | **TBD** |
| SCIC accuracy vs measured AI | **TBD** |
| MemMode vs ComputeMode vs always-fuse vs never-fuse | **TBD** (no kernels yet) |
| Attention: IO-aware mainloop with and without an extra fused epilogue, short S vs long S | **TBD** |
| MoE skew: per-expert plan vs one template; weights of idle experts loaded or not | **TBD** |
| Quantized epilogue under a measured W8A8/FP8 ridge | **TBD** |
| Plan-cache hit rate and switch overhead on a serving trace | **TBD** |
| End-to-end tokens/s | **TBD** |
| CUDA launch latency to replace the 5 µs sensitivity | **TBD** |

### 7.2 Placeholder numbers

No latency in this draft is an H100 measurement. CPU and NumPy timers, when the CPU script is used, are labeled `PLACEHOLDER_NO_GPU`. A partial CUDA matmul sweep, once it exists, is `PARTIAL_VENDOR_MATMUL` and still does not fill the fusion rows.

| shape | H100 median ms | achieved TFLOP/s | Nsight DRAM bytes | achieved AI |
| --- | --- | --- | --- | --- |
| 1×4096×4096 | TBD | TBD | TBD | TBD |
| 32×4096×4096 | TBD | TBD | TBD | TBD |
| 344×4096×4096 | TBD | TBD | TBD | TBD |
| 2048×4096×4096 | TBD | TBD | TBD | TBD |
| 4096×4096×4096 | TBD | TBD | TBD | TBD |

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
- No kernel, no trace, no tokens/s.

## 10. Conclusion

On an H100-class Roofline the FP16 square projection of width 4096 changes regime near M\* ≈ 345, IO-aware attention changes regime with S while materialized attention does not, and a skewed MoE layer contains both regimes in one forward. Fusion decided by operator kind misses those crossings. SCIC publishes the regime from shape and traffic. The dual plan turns the regime into MemMode, ComputeMode, or both, and it does so without taking ownership of Stage-Stream. The numerical argument in this draft is analytical. The device argument is still **TBD**.

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
| Device matmul | `bench_gemm.csv`, `gpu_measurements.json` (TBD on a GPU) |

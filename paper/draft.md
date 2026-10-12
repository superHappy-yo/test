# Shape-conditioned intensity classification

**Submission source:** [`paper/jsa/main.tex`](jsa/main.tex), an Elsevier `elsarticle` manuscript for the *Journal of Systems Architecture*. This note is the short English companion. Full-precision cells live in [`paper/data/bakeoff_v4/`](data/bakeoff_v4/) and [`paper/data/a40/`](data/a40/). Do not invent latencies, and do not quote an earlier bakeoff archive as if it were the idle-GPU run.

**Authors:** Happy Super (placeholder; affiliation TBD).

## Claim

Fusion decided by operator kind misses shape-dependent Roofline crossings. A width-4096 FP16 GEMM moves from about 1 FLOP/byte at one decode row to 1024 FLOP/byte at a 2048-row prefill on a cold-HBM model. IO-aware attention moves with sequence length. A score-materializing plan does not.

SCIC labels a site against a datasheet ridge (memory if AI < 0.5R, compute if AI > 2R, ridge otherwise). The dual-plan rule emits MemMode, ComputeMode, or both inside the band. The timed prototype does not emit a custom kernel. It binds those names to stock PyTorch paths:

- GEMM: `always_fuse` is `torch.compile` of MatMul+RMSNorm; `always_unfuse` is the eager chain. Scope `--calibrate all-gemm` picks the faster median (1 warmup, 3 iterations). `scic_selected` copies that median.
- Attention: both paths call library SDPA. The executable policy `compile_sdpa_chain` compiles SDPA with RMSNorm in every regime and is not wall-clock calibrated.

`quote_as_scic_accuracy` is false. Nsight DRAM counters failed with `ERR_NVGPUCTRPERM`. There is no measured ridge, no FlashAttention-3 kernel, no Stage-Stream schedule, and no end-to-end tokens/s.

## What the idle A40 run shows

Archive: `paper/data/bakeoff_v4/` (package 0.3.1, PyTorch 2.5.1+cu121, CUDA 12.1, NVIDIA A40). The run waited for an idle GPU. Ten timed iterations after three warmups. 77/77 rows ok. Seven GEMM decisions are `measured_wallclock`. Five attention decisions are `library_sdpa`.

Datasheet anchor used by the classifier: A40 FP16, R = 215.09 FLOP/byte, M* = 240.3 for N = K = 4096. That ridge was not measured on the card.

**GEMM (N = K = 4096).** The probe matches the faster of fuse and unfuse on all seven shapes. It agrees with the datasheet primary only at M = 256. Geometric-mean speedup versus eager: datasheet primary 0.921×, always-fuse 0.959×, always-unfuse 1.001×, calibrated selector 1.055×. The measured switch from unfuse to fuse lies between M = 256 and M = 512, inside the ridge band. At M ≤ 128 the compiled chain is slower than eager (0.763× to 0.864×). At M ≥ 512 it is faster (1.066× to 1.187×). The cold Roofline model predicts a fusion speedup of 1.00× for M ≥ 512, so that measured gain is not the C-matrix round trip.

**Attention (B = 1, H = 8, D = 64).** Every selected row copies the compiled SDPA-plus-norm chain. Versus a score-materializing eager baseline the geometric mean is 1.432×. Eager SDPA is faster than that chain on all five shapes (latency ratio 1.23× to 1.67×, geometric mean 1.45×). Splitting RMSNorm off SDPA is the faster of the two timed epilogue placements on four shapes; at S = 2048 the two medians differ by 0.07%. The data do not support “fuse prefill, unfuse decode.”

**Stock matmul, separate package** (`paper/data/a40/`, `PARTIAL_VENDOR_MATMUL`). Achieved TFLOP/s of `torch.matmul` rises from 0.47 at M = 1 to 115.4 at M = 4096 (77.1% of the 149.7 TFLOP/s datasheet peak). `achieved_ai` is empty. This is shape dependence of a vendor GEMM, not a fused-plan result.

## Outside the claim

Custom MemMode and ComputeMode CUDA kernels, Stage-Stream, FlashAttention-3, counter-based SCIC accuracy, a measured ridge, plan-switch overhead, and tokens/s are not in this archive. A torch-profiler byte estimate on the matmul sweep agrees with the analytical regime on 2 of 7 shapes and is marked not valid as accuracy. Earlier archives `paper/data/bakeoff/` and `paper/data/bakeoff_v2/` ran on busy GPUs. Their milliseconds are not pooled with `bakeoff_v4`.

## Figures

Regenerate from the CSVs:

```bash
python paper/jsa/scripts/plot_jsa_figures.py
```

Outputs land in `paper/jsa/figures/`.

# Paper draft

- [`draft.md`](draft.md) is the English MLSys/ASPLOS-style draft. Analytical numbers come from `shapefuse`. The A40 table in §7.2 is the third upload in [`data/a40/`](data/a40/) (`PARTIAL_VENDOR_MATMUL`, `--machine A40_FP16`) and is motivation, not the fusion result. `ncu` failed with `ERR_NVGPUCTRPERM` ([`docs/NSIGHT_PERMISSIONS.md`](../docs/NSIGHT_PERMISSIONS.md)). Fusion-proxy ratios are not fusion speedups. The copy-kernel file is `ESTIMATE_COPY_NOT_NSIGHT`. §7.3 quotes two Docker bakeoffs (`BAKEOFF_WALLCLOCK`, no Nsight, no FlashAttention-3 claim). [`data/bakeoff/`](data/bakeoff/) is the selector **before** the library-SDPA epilogue fix. [`data/bakeoff_v2/`](data/bakeoff_v2/) is the wall-clock after `compile_sdpa_chain`. That second run is not an idle GPU (physical GPU 1 at 92% utilization, 13490 MiB already allocated). GEMM still maps ComputeMode to `always_unfuse`; calibrating that choice is TODO.
- [`zh_outline.md`](zh_outline.md) is the Chinese outline of the same argument.
- [`fig_a40_torch_matmul_tflops.png`](fig_a40_torch_matmul_tflops.png) plots that matmul sweep. The dashed line is the A40 datasheet peak, not a measured ridge.

Stage-Stream is discussed only as prior work. It is not claimed here.

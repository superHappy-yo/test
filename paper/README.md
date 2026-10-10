# Paper draft

- [`draft.md`](draft.md) is the English MLSys/ASPLOS-style draft. Analytical numbers come from `shapefuse`. The A40 table in §7.2 is the second upload in [`data/a40/`](data/a40/) (`PARTIAL_VENDOR_MATMUL`, `--machine A40_FP16`). Nsight is still missing. Fusion-proxy ratios are not fusion speedups. Anything that still needs a measurement is marked **TBD**.
- [`zh_outline.md`](zh_outline.md) is the Chinese outline of the same argument.
- [`fig_a40_torch_matmul_tflops.png`](fig_a40_torch_matmul_tflops.png) plots that matmul sweep. The dashed line is the A40 datasheet peak, not a measured ridge.

Stage-Stream is discussed only as prior work. It is not claimed here.

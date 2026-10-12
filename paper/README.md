# Paper

The submission manuscript is [`jsa/main.tex`](jsa/main.tex) (*Journal of Systems Architecture*, Elsevier `elsarticle`). [`draft.md`](draft.md) is a short English companion. Full-precision cells are in [`data/bakeoff_v4/`](data/bakeoff_v4/) (idle GPU, calibrated GEMM) and [`data/a40/`](data/a40/) (stock `torch.matmul` only).

- [`jsa/README-OVERLEAF.md`](jsa/README-OVERLEAF.md) — how to compile, and how to regenerate figures with [`jsa/scripts/plot_jsa_figures.py`](jsa/scripts/plot_jsa_figures.py).
- [`zh_outline.md`](zh_outline.md) — Chinese outline of the same argument. The submission is English.
- `ncu` failed with `ERR_NVGPUCTRPERM` ([`docs/NSIGHT_PERMISSIONS.md`](../docs/NSIGHT_PERMISSIONS.md)). `quote_as_scic_accuracy` is false. No custom MemMode or ComputeMode kernel, no Stage-Stream, no FlashAttention-3, no tokens/s.

[`data/bakeoff/`](data/bakeoff/) and [`data/bakeoff_v2/`](data/bakeoff_v2/) are earlier busy-GPU archives. Quoted latencies in the manuscript come from [`data/bakeoff_v4/`](data/bakeoff_v4/).

Stage-Stream is discussed only as prior work. It is not claimed here.

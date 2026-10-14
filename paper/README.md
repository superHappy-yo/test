# Paper

The submission manuscript is [`jsa/main.tex`](jsa/main.tex) (*Journal of Systems Architecture*, Elsevier `cas-dc`), with synchronized Chinese text in [`jsa/jsa_main_zh.md`](jsa/jsa_main_zh.md). The main executed supplement and reproduction instructions are in [`data/jsa_supplement/`](data/jsa_supplement/README.md), including optimized compiler/library baselines, recent artifacts, complete small-model inference, and actual vLLM generation. Corrected pilot measurements are in [`data/jsa_replication/`](data/jsa_replication/README.md), complete-output checks in [`data/jsa_numerics/`](data/jsa_numerics/README.md), and scope replays in [`data/jsa_reanalysis/`](data/jsa_reanalysis/README.md). The separate stock `torch.matmul` archive is [`data/a40/`](data/a40/).

The original FP16 eager RMSNorm can overflow, so the legacy GEMM ratios in `data/bakeoff_v4/`, `draft.md`, and `zh_outline.md` are retained for provenance and are not equivalent-computation speedup evidence. Current performance claims use the numerically validated FP32-normalization runs.

- [`jsa/README-OVERLEAF.md`](jsa/README-OVERLEAF.md) — how to compile, and how to regenerate figures with [`jsa/scripts/plot_jsa_figures.py`](jsa/scripts/plot_jsa_figures.py).
- [`zh_outline.md`](zh_outline.md) — Chinese outline of the same argument. The submission is English.
- `ncu` failed with `ERR_NVGPUCTRPERM` ([`docs/NSIGHT_PERMISSIONS.md`](../docs/NSIGHT_PERMISSIONS.md)). `quote_as_scic_accuracy` is false. No custom MemMode or ComputeMode kernel, no Stage-Stream, no FlashAttention-3, no tokens/s.

[`data/bakeoff/`](data/bakeoff/) and [`data/bakeoff_v2/`](data/bakeoff_v2/) are earlier busy-GPU archives. Quoted latencies in the manuscript come from [`data/bakeoff_v4/`](data/bakeoff_v4/).

Stage-Stream is discussed only as prior work. It is not claimed here.

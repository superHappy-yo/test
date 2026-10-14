# Journal of Systems Architecture manuscript

This directory contains the English submission source for *Journal of Systems Architecture* and its synchronized Chinese reading version. The current manuscript uses the Elsevier `cas-dc` double-column class.

| File | Role |
| --- | --- |
| `main.tex` | English manuscript: abstract, keywords, sections, tables and appendices |
| `jsa_main_zh.md` | Synchronized Chinese reading version |
| `prototype_execution.tex`, `supplement_evaluation.tex`, `supplement_results.tex`, `vllm_results.tex` | Main executable evaluation and measured supplementary results |
| `references.bib` | Bibliography using `cas-model2-names` with numbered `natbib` citations |
| `figures/` | PDF and PNG figures from the primary corrected run and the separate stock-matmul sweep |
| `scripts/plot_jsa_figures.py` | Regenerates those figures. Not required to compile the paper |
| `scripts/run_jsa_replications.py` | Runs supplemental independent-process A40 measurements into a new directory |
| `scripts/analyze_jsa_evidence.py` | Audits selections and replays calibration scopes from archived results |
| `scripts/check_jsa_numerics.py` | Validates complete outputs and reproduces the legacy FP16 overflow |

The compile environment must supply the `cas-dc` class, its dependencies, and the `cas-model2-names` bibliography style declared by `main.tex`; these template files are not vendored here. Use a CAS template environment when uploading to Overleaf.

## Figures

- `fig_roofline_gemm` — A40 datasheet Roofline and the closed-form GEMM intensity curve, with bakeoff decisions marked
- `fig_a40_torch_matmul_tflops` — stock `torch.matmul` TFLOP/s versus \(M\)
- `fig_gemm_plan_speedup` — corrected fuse, unfuse, compile, and selected GEMM paths
- `fig_attn_library_paths` — library attention latency and speedup versus materialized eager
- `fig_oracle_gap` — selected latency divided by the faster of fuse and unfuse

## Regenerate the figures

From the repository root, with the package importable (`pip install -e .` if needed) and `matplotlib` installed:

```bash
python paper/jsa/scripts/plot_jsa_figures.py
```

The script reads only archived CSVs and JSON under `paper/data/`. It does not run a GPU benchmark.

## Compile

Engine: pdfLaTeX.

```bash
cd paper/jsa
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or `latexmk -pdf main.tex`.

Document class: `\documentclass[a4paper,fleqn]{cas-dc}`.
Bibliography style: `\bibliographystyle{cas-model2-names}`.

The author information and template preamble already present in the source are retained by the revision. The bibliography follows the appendices.

## Supplemental evidence

The main evaluation uses `paper/data/jsa_supplement/` and its paired summaries. The corrected pilot is retained in Appendix C and uses `paper/data/jsa_replication/run_01/`; all three corrected GPU processes are in `paper/data/jsa_replication/`. Complete-output validation is in `paper/data/jsa_numerics/`, and the derived audit and scope replay are in `paper/data/jsa_reanalysis/`. The original `bakeoff_v4` files remain unchanged, but their overflow-prone GEMM normalization makes those timing ratios unsuitable as equivalent-computation speedups.

Regenerate the supplemental summaries from the repository root:

```bash
python paper/jsa/scripts/analyze_jsa_evidence.py --replications paper/data/jsa_replication
```

For commands to validate outputs and repeat GPU measurements into a new directory, see `paper/data/jsa_replication/README.md`. The runner requires a passing numerical record for the benchmark-source hash. Replications intentionally skip max-autotune, use the stated timing protocol, and do not modify the original archive. The scope ablation reuses corrected primary-run probes and medians; it is not a new GPU timing experiment.

## Edit before submission

1. Check the existing author, affiliation, contact and funding information in `main.tex`.
2. Highlights are currently commented out in the template; the synchronized reading version does not add a separate highlights section.
3. Do not invent Nsight DRAM bytes, achieved arithmetic intensity, or custom MemMode/ComputeMode kernels. Complete random-weight small-model throughput is measured; it does not measure pretrained task accuracy. `quote_as_scic_accuracy` stays false.

## Optional zip

A flat archive for an Overleaf upload can be rebuilt from this directory:

```bash
cd paper/jsa
zip -r ../jsa-overleaf.zip main.tex prototype_execution.tex supplement_evaluation.tex supplement_results.tex vllm_results.tex adapter_ablation.tex v2_method.tex v2_evaluation.tex jsa_main_zh.md references.bib README-OVERLEAF.md figures thumbnails -x '*.DS_Store'
cp ../jsa-overleaf.zip ../../artifacts/jsa-overleaf.zip
```

The Git tree is the source of truth. The zip is only a convenience copy of the same files.

## Executed supplement

See [supplement evidence and reproduction](../data/jsa_supplement/README.md) for the 21+9 shape matrix, independent real-dispatch timing, optimized baselines, recent fusion artifacts, and the complete small Qwen3 model. Run `scripts/analyze_jsa_supplement.py` then `scripts/plot_jsa_supplement.py` from the repository root to regenerate derived summaries and figures. The main model result is a slowdown, reported explicitly in both manuscripts. Temporary environments and public source checkouts are isolated under ignored `.experiments/`.

## Concrete-candidate revision

`v2_method.tex` and `v2_evaluation.tex` describe SCIC-v2 and its separately
archived results in `paper/data/jsa_v2/`. Regenerate completed-worker summaries
with `scripts/analyze_jsa_v2.py`, figures with `scripts/plot_jsa_v2.py`, and both
language evaluation sections with `scripts/update_jsa_v2_text.py`. The artifact
README states the strict numerical gates, resource fallback, corrected controls,
and distinction between exhaustive-scan and actual lazy preparation accounting.

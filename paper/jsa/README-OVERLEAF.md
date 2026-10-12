# Journal of Systems Architecture manuscript

This directory is the submission source. It is an Elsevier `elsarticle` preprint for *Journal of Systems Architecture*.

| File | Role |
| --- | --- |
| `main.tex` | Manuscript: abstract, highlights, keywords, sections, tables |
| `references.bib` | Numbered bibliography (`elsarticle-num`). Real venues and arXiv ids; no fabricated DOIs |
| `figures/` | PDF and PNG figures plotted from `paper/data/bakeoff_v4/` and `paper/data/a40/` |
| `scripts/plot_jsa_figures.py` | Regenerates those figures. Not required to compile the paper |

`elsarticle` is provided by the TeX installation (Overleaf includes it). It is not vendored here.

## Figures

- `fig_roofline_gemm` — A40 datasheet Roofline and the closed-form GEMM intensity curve, with bakeoff decisions marked
- `fig_a40_torch_matmul_tflops` — stock `torch.matmul` TFLOP/s versus \(M\)
- `fig_gemm_plan_speedup` — fuse, unfuse, compile, and the selected GEMM path
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

Document class: `\documentclass[preprint,12pt]{elsarticle}`.
Journal line: `\journal{Journal of Systems Architecture}`.

For a review or camera-ready upload, switch the class option as the JSA guide requires (`review` or `final`).

## Edit before submission

1. Author and affiliation in `main.tex`. The placeholder is Happy Super, affiliation TBD, `happy.super@example.com`.
2. Highlights, if the camera-ready contribution list changes. Each bullet is kept under the Elsevier 85-character guidance.
3. Do not fill Nsight DRAM bytes, achieved arithmetic intensity, custom MemMode or ComputeMode kernels, or end-to-end tokens/s. Those measurements are not in the archive. `quote_as_scic_accuracy` stays false.

## Optional zip

A flat archive for an Overleaf upload can be rebuilt from this directory:

```bash
cd paper/jsa
zip -r ../jsa-overleaf.zip main.tex references.bib README-OVERLEAF.md figures -x '*.DS_Store'
cp ../jsa-overleaf.zip ../../artifacts/jsa-overleaf.zip
```

The Git tree is the source of truth. The zip is only a convenience copy of the same files.

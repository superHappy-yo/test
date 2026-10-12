# Overleaf upload: Journal of Systems Architecture (Elsevier)

This folder is an **Overleaf-ready** LaTeX project for *Journal of Systems Architecture* (JSA), converted from `paper/draft.md` on the shapefuse repository branch.

## What is included

| File | Purpose |
|------|---------|
| `main.tex` | `elsarticle` manuscript (`preprint`, 12pt), abstract, Elsevier highlights, keywords, sections |
| `references.bib` | Citations referenced in the draft (no fabricated DOIs) |
| `figures/fig_a40_torch_matmul_tflops.png` | A40 matmul motivation figure from the draft |
| `README-OVERLEAF.md` | This file |

`elsarticle` is **not** vendored; Overleaf provides it via the Elsevier template.

## Upload to Overleaf

1. Download **`paper/jsa-overleaf.zip`** from the repository (or use `artifacts/jsa-overleaf.zip`).
2. In [Overleaf](https://www.overleaf.com), **New Project → Upload Project**.
3. Select the zip file. The project root should contain `main.tex`, `references.bib`, `figures/`, and this README **flat** at the top level (no extra nesting required).

## Compile

- **Engine:** pdfLaTeX (default) or **latexmk** (`latexmk -pdf main.tex`).
- **Bibliography:** `elsarticle-num` (numbered citations).
- Run pdfLaTeX → BibTeX → pdfLaTeX ×2 if references show as `?`.

## Journal metadata

- Document class: `\documentclass[preprint,12pt]{elsarticle}`
- Journal line: `\journal{Journal of Systems Architecture}`

For final submission, switch from `preprint` to the option required by Elsevier (often `review` or `final`; check the JSA author guide).

## What you should edit before submission

1. **Authors and affiliations** — placeholder `Happy Super` / `Affiliation TBD` in `main.tex`; replace email and `\affiliation{...}`.
2. **Highlights** — five bullets in `\begin{highlights}...\end{highlights}`; tune for the camera-ready contribution list if needed.
3. **Abstract and keywords** — align with your final JSA systems-architecture framing.
4. **Figures** — additional analytical figures from the repo (`fig_gemm_ai_vs_M.png`, etc.) are **not** copied here; add them under `figures/` and uncomment/add `\includegraphics` as needed.
5. **TBD claims** — the draft intentionally leaves Nsight DRAM, tokens/s, and custom kernels as future work; do not fill those from datasheet math or profiler proxies.

## Source of truth

English content and numbers follow `paper/draft.md` and archived CSVs under `paper/data/` (especially `bakeoff_v4/` for calibrated latency). `paper/zh_outline.md` is reference only.

## Packaging script (maintainers)

From repo root:

```bash
(cd paper/jsa && zip -r ../jsa-overleaf.zip . -x '*.DS_Store')
cp paper/jsa-overleaf.zip artifacts/jsa-overleaf.zip
```

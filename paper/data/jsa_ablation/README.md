# Adapter overhead diagnosis

This follow-up separates invocation overhead from candidate-selection headroom.
It does not implement new MemMode/ComputeMode kernels. Earlier operator, HF,
and vLLM measurements remain in `../jsa_supplement/` and are not pooled with
this protocol.

## Results

All 588 final rows pass the stated numerical gates. Relative to the original
compiled model under this follow-up protocol:

| Path | Prefill host-time ratio | 64-step decode host-time ratio |
| --- | ---: | ---: |
| Legacy SCIC | 0.871 | 0.647 |
| Prebound SCIC | 0.889 | 0.703 |
| Fixed compiled, prebound | 0.904 | 0.701 |
| Static strict inlining | 0.973 | 0.897 |

Values below one are slowdowns. Inlining partly recovers integration loss but
does not exceed the original model. One guarded lookup replaces the redundant
second lookup in the general adapter; the fixed-compiled one-lookup control
has a 1.032 decode ratio relative to legacy fixed-compiled. Its prefill interval
includes one, and an unusually slow host batch is retained in the primary
geometric mean rather than discarded.

The strict projection candidate minimum is 1.155 host / 1.130 event relative
to fixed compiled. Leave-one-seed-out known-shape selection gives 1.150 host
(cluster interval [1.122, 1.188]) / 1.127 event, selecting native at
M=1/8/128/1024 and compiled at M=2048/16384. This offline check excludes
runtime dispatch/probe costs and does not demonstrate complete-model speedup
or the incremental value of the analytical feature.

## Protocol

- PyTorch 2.5.1+cu121, Transformers 4.51.3, two NVIDIA A40 GPUs on the same
  host. Device streams run concurrently, with serial benchmarks within each
  device. Three independent processes per device/workload use seeds 0, 1, 2.
- Complete random-weight six-layer Qwen3: 83,900,160 parameters, hidden 1024,
  FFN 3072, 16 query heads, 8 KV heads, head dimension 64, vocabulary 8192.
  Batches 1/8, prompts 128/2048, 64 teacher-forced decode steps, static KV cache.
- Seven arms: original compiled model; fixed compiled with legacy two-lookups,
  one lookup, or prebinding; static strict inlining; legacy SCIC; prebound SCIC.
  SCIC prebinding reuses the corresponding legacy **pre-compilation** choices
  within the process. The two variants are adjacent in otherwise seeded
  shuffled policy order; manifests record the order.
- All arms use `dynamic=False`, internal CUDA Graphs disabled, and outer
  `emulate_precision_casts=True`. Position tensors are preallocated. These
  common settings differ from the earlier HF experiment.
- Full eager and compiled logits are checked against the unmodified eager
  model at prefill and every one of 64 decode steps, with unchanged absolute
  and relative tolerances of 1e-2. Candidate preparation and validation occur
  before timing. Prefill: 5 warmups, 20 event samples, then a separate 20-call
  synchronized host batch. Decode: 2 warmup loops, 20 event samples, then a
  separate 20-loop host batch. A loop comprises all 64 steps.
- Prebinding freezes one callable for each layer/phase after validation. The
  caller checks batch and phase sequence length; the compiled caller uses
  static guards. These are benchmark specializations, not an API for arbitrary
  dynamic layouts. The general exact-signature executor remains separate.
- Static strict inlining removes the outer projection/add/norm custom op and
  nested compiled wrapper; the FP16 linear boundary remains. It has no selector
  and cannot establish an analytical-selection benefit.
- A separate projection matrix covers actual row counts 1, 8, 128, 1024, 2048,
  16384 at width 1024. Weight standard deviation is 0.02 and norm weights are
  ones, matching the initialized model. Seven invocation variants are audited
  at 1e-3 tolerances, with 50 warmups, 200 events and a separate 200-call host
  batch. The compiled callable is shared by direct/two-lookup/one-lookup/
  prebound/opaque-prebound variants, avoiding candidate-choice confounding.
- The runner requires two idle samples (GPU utilization <=5%, memory <=256 MiB)
  before each process. Commands, failures, timeouts, and elapsed times are kept.
  Final timing processes never start a profiler. Initial seed-0 trials that
  interleaved profiling and timing are retained as `diagnostic_initial_*` and
  excluded from latency/candidate summaries. Their traces are used only to
  describe execution structure. Seed 0 is remeasured without profiling.

The complete matrix comprises 24 model processes with 336 latency rows and
6 projection processes with 252 rows. `smoke/` and interrupted directories are
excluded from timing analysis, as are the initial in-process-profiler trials.
The initial arbitrary-norm-weight projection smoke
test was rejected at the unchanged 1e-3 gate; it supplies no performance claim
for the rejected signature. This diagnosis does not establish applicability
to pretrained norm weights or extend the optimized path to vLLM.

## Reproduction

From the repository root, use an environment matching
`provenance/environment.txt`. Each runner owns one GPU; the two commands can
run concurrently as in the recorded study.

```bash
python paper/jsa/scripts/run_jsa_ablation.py --python /path/to/python --out paper/data/jsa_ablation/device0 --gpu 0 --runs 3
python paper/jsa/scripts/run_jsa_ablation.py --python /path/to/python --out paper/data/jsa_ablation/device1 --gpu 1 --runs 3
python paper/jsa/scripts/analyze_jsa_ablation.py
python paper/jsa/scripts/plot_jsa_ablation.py
python paper/jsa/scripts/update_jsa_ablation_text.py
```

Use `--resume` to skip completed processes; failed/interrupted directories are
retained with an incomplete suffix when rerun. Never overwrite measured runs.
The text/figure scripts require the complete matrix before publishing.

## Reading the evidence

- `analysis/paired_ratios.csv`: same-process, same-workload, same-phase
  comparisons, including absolute host milliseconds saved.
- `analysis/summary.csv`: geometric means and percentile 95% bootstrap
  intervals. Resampling keeps six GPU/run clusters over the fixed workload
  matrix; individual timing iterations are not independent experiments. This
  interval does not model all shared-host effects or production variability.
- `analysis/candidate_minimum.csv` and `candidate_summary.csv`: optimistic
  native/compiled per-shape minima, with host/event winners. These use regular
  samples, so they are limited diagnostic ceilings, not measured selectors,
  held-out tuning results, or global schedule optima.
- `analysis/candidate_cross_validation.csv` and `candidate_cv_summary.csv`:
  leave out both devices of each seed and choose each known shape's candidate
  only from the other two seeds' event medians. This checks ranking stability
  while keeping test timings out of selection. It excludes actual dispatch and
  probe costs and does not test unseen shapes. The best fixed candidate is also
  chosen only on training data.
- `analysis/profile_counts.csv`: CUDA event and CPU operation counts from
  separately archived diagnostic trials. CUDA event names/counts do not measure DRAM bytes or
  identify an isolated causal fusion benefit. CPU self times are retained in
  raw details but are not used to quantify steady-state cost shares.
- `diagnostic_profiler_effect/`: the same warmed compiled original-model decode
  callable measured before and after a CUDA-profiler session, in separate
  diagnostic processes. These within-process batches are not independent
  experiments and supply no timing rows to the main matrix.
  The measured after/before host-time speed ratios are 0.853 on GPU 0 and
  0.834 on GPU 1, after the profiler context exits. Initial inline-relative
  apparent gains are excluded; the final matrix never initializes the profiler.
- `provenance/measured_sources/`: exact sources used for the matrix, before
  promoting a verified one-lookup optimization to the general adapter. The
  initial runner version also generated the retained profiler trials; final
  runner and optimized adapter snapshots are in `provenance/final_sources/`.
  For byte-identical historical execution, use the measured module snapshots
  in an isolated copy of the project; the current ablation module explicitly
  preserves the legacy two-lookup hot path.
- `NEXT_EXPERIMENT.md`: candidate-design and validation criteria for deciding
  whether a further method experiment is justified.

Raw event samples, output errors, selection records, prepare times and profiler
data are in each process's `details.json`. Source fingerprints, options and
software/hardware metadata are in `manifest.json`.

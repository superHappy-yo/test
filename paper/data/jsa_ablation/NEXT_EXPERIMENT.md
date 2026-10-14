# Criteria for the next method experiment

The current diagnostic compares invocation variants while holding the available
native/compiled algorithms fixed. A static integration improvement should be
reported separately from an analytical-selection benefit.

## Decision gate

1. Check the measured native/compiled candidate minimum against fixed compiled,
   separately for host and event timing, and inspect wins across all six runs.
2. If the candidate minimum remains approximately fixed compiled, stop adjusting
   SCIC thresholds for this candidate set. A different threshold cannot create
   a faster execution path.
3. If complementary winning regions appear with a meaningful repeatable margin,
   compare best static, independent empirical calibration, analytical-only,
   calibrated SCIC and the measured candidate minimum. Use independent probe
   and regular samples, and account for startup amortization.

## Next executable comparison

The unprofiled strict-projection data show complementary winners. A
leave-one-seed-out check (holding out both devices of the seed) chooses native
for M=1/8/128/1024 and compiled for M=2048/16384. This is offline ranking
evidence, not a measured selector. It motivates testing empirical calibration
before replacing the candidate set.

Use a common invocation path for fixed-native, fixed-compiled, default
ridge-plus-compute SCIC and all-GEMM empirical calibration, with the original
compiled model as an additional system baseline. Default SCIC skips memory
shapes, although those include the observed native-winning rows. Keep the
existing thresholds unchanged so that the comparison isolates calibration
coverage, and record every decision and preparation cost. Test whether the
local headroom survives the adapter and full model before describing any
system improvement. A full-calibration win alone does not validate the
analytical feature.

## Concrete candidate design, subject to that gate

Implement and validate actual schedules before attributing results to MemMode
or ComputeMode. A bounded initial target is the model's real FP16 width-1024
projection/residual/RMSNorm function, rather than a synthetic single matmul.

- A small-row candidate should target decode rows M=1/8, with a concrete
  GEMV/small-GEMM schedule, row/tile mapping, reduction strategy and launch
  count. A kernel name is not a resource argument: record the actual generated
  launch configuration and relevant compiler resource metadata.
- A throughput candidate should retain an optimized GEMM and efficient
  residual/norm epilogue for prefill rows M=128/1024/2048/16384. Whole-row
  normalization depends on every output column; a tiled implementation must
  include any partial-statistic buffers, extra reductions and final writes in
  its complete-function timing and traffic description.
- Preserve the FP16 product rounding, FP16 residual sum, FP32 statistics, cast
  before norm affine, and both returned outputs. Validate complete outputs at
  the existing gates, across independent seeds and stress amplitudes. Keep
  rejected candidates in the evidence, without timing claims.
- Compare direct and controlled whole-function CUDA Graph execution separately.
  Check generated code and launches; do not claim a full GEMM/reduction fusion
  just because a chain is compiled.
- Use all existing native/default/max-autotune baselines in matching software
  environments. Avoid adding a weak eager epilogue only to the new candidate.
- A warmed standalone width-1024 weight matrix can have different cache reuse
  from six model layers. Verify promising candidates in the complete model and
  only then in actual vLLM generation with output-token validation. Do not
  extrapolate a standalone speed ratio to system throughput.

The analytical feature must show incremental value beyond empirical tuning:
for example, fewer probes at comparable held-out selection quality, after
including dispatch and amortized preparation costs. A winning static kernel
alone does not establish that value.

# Full-output numerical validation

`output_validation.json` checks all seven GEMM and five attention shapes,
using seed 0, FP16 interfaces and explicit FP32 normalization intermediates.
The record includes the benchmark-source hash. The timing runner requires
a passing validation record for its exact source.

For GEMM, the reference is FP32 normalization of the FP16 matrix product;
relative and absolute tolerances are both 1e-3. For attention, the reference
uses FP32 attention and normalization before casting to FP16, with both
tolerances 1e-2. Every complete corrected output passes. Maximum compiled
GEMM error is 0.001953125; maximum attention error across checked paths is
0.00390625. These checks cover the fixed benchmark inputs, not all inputs.

The script also evaluates the explicit legacy eager and compiled formulas.
Legacy eager FP16 squaring overflows on six GEMM shapes. At M=4096 it
produces 971 all-zero rows, with maximum eager--compiled difference about
5.49. The old single-element finite check misses this error. Those timing
ratios are consequently excluded from equivalent-computation claims.

```bash
CUDA_VISIBLE_DEVICES=GPU-b356d1b0-c3da-551d-07d4-ee0a3c97bbc4 \
  python paper/jsa/scripts/check_jsa_numerics.py \
  --out results/jsa-numerics-new.json
```

This validates numerical outputs, not SCIC classification against DRAM
performance counters; `quote_as_scic_accuracy` remains false.

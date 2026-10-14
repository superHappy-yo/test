# Numerically validated A40 measurements

Three independent processes run the corrected benchmark on A40 GPU
`GPU-b356d1b0-c3da-551d-07d4-ee0a3c97bbc4`. The first run is the primary
run for the manuscript tables and figures; the other two assess repeatability.

Matrix inputs and outputs are FP16. RMSNorm statistics and weighting use
FP32, with the final result cast to the input dtype. Complete outputs at
all twelve workload shapes pass the checks in `../jsa_numerics/` before
timing. The replication runner verifies the validation record against
the current benchmark-source hash.

Each process uses three warmups and ten regular iterations. `all-gemm`
calibration uses one warmup and three iterations. Attention retains the
fixed compiled-SDPA-chain policy. Max-autotune is intentionally skipped:
each run contains 65 successful rows and 12 skipped rows. PyTorch is
2.5.1+cu121, CUDA 12.1, Python 3.10.12.

Startup requires two consecutive samples with utilization at most 5% and
used device memory at most 256 MiB. The manifest preserves commands,
samples, return codes, and source hash; each run retains CSVs, summary JSON,
logs, and during-run samples. Monitoring includes the benchmark itself.

GEMM probe agreement is 7/7, 6/7, and 7/7, with selected geometric-mean
speedups of 1.296, 1.292, and 1.303 over eager. All probes select compile,
so always-fuse gives the same selected speedups. Attention rankings vary
by shape and process. Timings use CUDA events, and selected medians copy
baseline medians. Setup, compilation, calibration, dispatch and switching
costs are excluded. No DRAM classifier accuracy or custom kernel is measured.

To reproduce into new directories from the repository root:

```bash
CUDA_VISIBLE_DEVICES=GPU-b356d1b0-c3da-551d-07d4-ee0a3c97bbc4 \
  python paper/jsa/scripts/check_jsa_numerics.py \
  --out results/jsa-numerics-new.json
python paper/jsa/scripts/run_jsa_replications.py \
  --gpu-uuid GPU-b356d1b0-c3da-551d-07d4-ee0a3c97bbc4 --runs 3 \
  --numerics results/jsa-numerics-new.json --out results/jsa-replication-new
python paper/jsa/scripts/analyze_jsa_evidence.py \
  --primary results/jsa-replication-new/run_01 \
  --replications results/jsa-replication-new --out results/jsa-analysis-new
```

Original `bakeoff_v4` files are unchanged. The first supplemental runs of
the old FP16 normalization are in `../jsa_replication_legacy/`; their GEMM
latency ratios are not equivalent-computation speedups.

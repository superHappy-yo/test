# Corrected selection audit and scope replay

`evidence_audit.json` summarizes the primary corrected run and all three
processes separately. It also retains an explicitly marked legacy archive
summary for provenance; legacy GEMM ratios are not mathematically equivalent
computation speedups.

`calibration_scope.csv` replays primary-run probe winners and regular path
medians under five calibration scopes. Unprobed shapes follow their
analytical binding. Speedups are 1.018 (none), 1.109 (default ridge only),
and 1.296 for ridge-and-compute, wide ridge, and all GEMM. This replay
does not measure new latency, probe cost, or dispatch overhead, and does
not establish an optimal ridge band.

```bash
python paper/jsa/scripts/analyze_jsa_evidence.py \
  --replications paper/data/jsa_replication
```

The script validates required path measurements and copied selections and
derives geometric means from unrounded recorded latency ratios. Numerical
validation is stored separately in `../jsa_numerics/`.

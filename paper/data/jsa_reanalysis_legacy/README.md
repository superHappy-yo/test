> Legacy evidence: these measurements used FP16 eager squaring that overflows on six GEMM shapes. They are retained for provenance and are excluded from equivalent-computation speedup claims. Current validated results are in `../jsa_replication/` and output checks in `../jsa_numerics/`.

# Selection audit and calibration-scope replay

`evidence_audit.json` checks the original `bakeoff_v4` choices and summarizes
the three new GPU processes in `../jsa_replication/`. Each process is kept
separate; measurements are not pooled across archives or runs.

`calibration_scope.csv` is an offline ablation. It reuses the original
three-iteration GEMM probe winners and the original regular path medians.
Shapes outside each calibration scope follow their analytical primary.
The five replay policies use zero, three, five, six or seven recorded
probe choices. The resulting geometric-mean speedups are 0.921, 0.949,
0.991, 1.016 and 1.055. This replay does not measure new GPU latencies,
probe cost, dispatcher overhead or DRAM arithmetic intensity.

Regenerate these files from the repository root:

```bash
python paper/jsa/scripts/analyze_jsa_evidence.py \
  --replications paper/data/jsa_replication
```

The script uses only the Python standard library, validates required
timings and copied selections, and derives geometric means from the
unrounded recorded latencies.

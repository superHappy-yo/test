#!/usr/bin/env python3
"""Same compiled decode callable before/after one CUDA-profiler session.

Three batches per side are within-process checks, not independent experiments.
This intervention does not supply any performance result for the main matrix.
"""
import argparse
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from shapefuse import ablation
from shapefuse.executor import source_fingerprint
from shapefuse.io_utils import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    original_measure = ablation.measure
    invocations = 0
    evidence = {}

    def intervention(fn, inputs, **options):
        nonlocal invocations
        invocations += 1
        if invocations == 1:  # Prefill, no profiler initialized here.
            return original_measure(fn, inputs, **options)
        before = [original_measure(fn, inputs, **options) for _ in range(3)]
        trace = ablation.profiler_summary(lambda: fn(*inputs))
        after = [original_measure(fn, inputs, **options) for _ in range(3)]
        evidence.update(before=before, after=after, profile=trace,
            geomean_host_speed_ratio_after_vs_before=math.exp(sum(math.log(
                a["host_batch_ms_per_call"]/b["host_batch_ms_per_call"])
                for a,b in zip(before,after))/3),
            geomean_event_speed_ratio_after_vs_before=math.exp(sum(math.log(
                a["gpu_median_ms"]/b["gpu_median_ms"])
                for a,b in zip(before,after))/3))
        return before[0]

    ablation.measure = intervention
    args.stage="system"
    args.case="b1_s128"
    args.methods="original"
    args.seed=0
    args.warmup=5
    args.repeats=20
    args.profile=False
    status = ablation.run_system(args)
    evidence.update(source_sha256=source_fingerprint(Path(__file__)),
                    model_status=status,
                    limitation="Same process and callable; batches are not independent runs. Main matrix initializes no profiler.")
    write_json(Path(args.out)/"profiler_effect.json", evidence)
    return status


if __name__ == "__main__":
    raise SystemExit(main())

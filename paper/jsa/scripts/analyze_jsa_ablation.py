#!/usr/bin/env python3
"""Paired adapter diagnostics; timing iterations are not independent runs."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from shapefuse.io_utils import write_csv, write_json


def gm(values):
    return math.exp(sum(math.log(float(v)) for v in values)/len(values)) if len(values) else None


def interval_by_cluster(rows, metric):
    clusters = {}
    for row in rows:
        clusters.setdefault(row["cluster"], []).append(row[metric])
    values = [gm(v) for v in clusters.values()]
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(values), size=(4000, len(values)))
    boots = np.exp(np.log(values)[draws].mean(axis=1))
    return dict(clusters=len(clusters), ci95_low=float(np.percentile(boots, 2.5)),
                ci95_high=float(np.percentile(boots, 97.5)))


def cross_validate_candidates(oracle):
    """Hold out both devices of a seed; choose only from the other two seeds.

    This tests ranking stability on known shapes, not unseen-shape accuracy or
    measured runtime dispatch. Training minimizes geometric-mean event time.
    """
    result = []
    for row in oracle:
        fold = row["cluster"].split("_run")[-1]
        training = [r for r in oracle if r["cluster"].split("_run")[-1] != fold]
        same_case = [r for r in training if r["case"] == row["case"]]
        if not same_case or not training:
            continue
        selected = min(("native", "compiled"), key=lambda name: gm([r[f"{name}_gpu_ms"] for r in same_case]))
        static = min(("native", "compiled"), key=lambda name: gm([r[f"{name}_gpu_ms"] for r in training]))
        result.append(dict(process=row["process"], cluster=row["cluster"], case=row["case"],
            held_out_seed=int(fold), training_pairs=len(same_case), selected=selected, best_static_training=static,
            host_speedup=row["compiled_host_ms"]/row[f"{selected}_host_ms"],
            gpu_speedup=row["compiled_gpu_ms"]/row[f"{selected}_gpu_ms"],
            host_speedup_vs_best_static=row[f"{static}_host_ms"]/row[f"{selected}_host_ms"],
            gpu_speedup_vs_best_static=row[f"{static}_gpu_ms"]/row[f"{selected}_gpu_ms"]))
    return result


def collect(root):
    rows, profiles, statuses = [], [], []
    def add_profiles(details, stage, case, cluster, process):
        for detail in details:
            for phase in ("prefill", "decode64") if stage == "system" else ("projection",):
                profile = detail.get(f"profile_{phase}" if stage == "system" else "profile")
                if not profile:
                    continue
                cpu = {r["name"]: r["count"] for r in profile["cpu"]}
                profiles.append(dict(process=process, cluster=cluster, stage=stage,
                    case=detail.get("case", case), phase=phase,
                    policy=detail.get("policy", detail.get("method")),
                    cuda_events=profile["cuda_event_count"],
                    compiled_regions=cpu.get("Torch-Compiled Region", 0),
                    dynamo_lookups=cpu.get("TorchDynamo Cache Lookup", 0),
                    project_add_norm_calls=cpu.get("shapefuse::project_add_norm", 0),
                    linear_half_calls=cpu.get("shapefuse::linear_half", 0)))
    for path in sorted(root.glob("device*/gpu*_run*/*/*/latency.csv")):
        if ".incomplete_" in str(path) or not (path.parent/"completed.json").exists():
            continue
        rel = path.relative_to(root)
        stage, case = rel.parts[2:4]
        cluster = rel.parts[1]
        process = str(rel.parent)
        with path.open() as handle:
            raw_rows = list(csv.DictReader(handle))
        for raw in raw_rows:
            row = dict(raw, cluster=cluster, process=process, stage=stage, case=case if stage == "system" else raw["case"])
            for name in ("gpu_median_ms", "host_batch_ms_per_call"):
                if row.get(name):
                    row[name] = float(row[name])
            rows.append(row)
        details = json.loads((path.parent/"details.json").read_text())
        add_profiles(details, stage, case, cluster, process)
        for detail in details:
            statuses.append(dict(process=process, stage=stage, case=detail.get("case", case),
                method=detail.get("method", detail.get("policy")),
                status=detail["execution_status"], error=detail.get("error", "")))
    # Initial in-process-profiler trials are excluded from every latency and
    # candidate summary. Their separately archived traces describe structure.
    for path in sorted(root.glob("device*/diagnostic_initial_gpu*_run00/*/*/details.json")):
        rel = path.relative_to(root)
        add_profiles(json.loads(path.read_text()), rel.parts[2], rel.parts[3],
                     rel.parts[1], str(rel.parent))
    return rows, profiles, statuses


def analyze(root, out):
    rows, profiles, statuses = collect(root)
    write_csv(out/"all_latency.csv", rows)
    write_csv(out/"profile_counts.csv", profiles)
    write_csv(out/"status.csv", statuses)
    groups = {}
    for row in rows:
        if row["execution_status"] == "ok":
            key = (row["stage"], row["process"], row["case"], row.get("phase", "projection"))
            groups.setdefault(key, {})[row.get("policy") or row["method"]] = row
    pairs, oracle = [], []
    for key, methods in groups.items():
        stage, process, case, phase = key
        baselines = ("original", "fixed_compile", "scic") if stage == "system" else ("compiled_direct", "legacy_double")
        for baseline in baselines:
            if baseline not in methods:
                continue
            for policy, row in methods.items():
                base = methods[baseline]
                pairs.append(dict(stage=stage, process=process, cluster=row["cluster"], case=case,
                    phase=phase, baseline=baseline, policy=policy,
                    host_speedup=base["host_batch_ms_per_call"]/row["host_batch_ms_per_call"],
                    gpu_speedup=base["gpu_median_ms"]/row["gpu_median_ms"],
                    host_ms_saved=base["host_batch_ms_per_call"]-row["host_batch_ms_per_call"]))
        if stage == "projection" and all(name in methods for name in ("native", "compiled_direct")):
            native, comp = methods["native"], methods["compiled_direct"]
            oracle.append(dict(process=process, cluster=comp["cluster"], case=case,
                native_host_ms=native["host_batch_ms_per_call"], compiled_host_ms=comp["host_batch_ms_per_call"],
                native_gpu_ms=native["gpu_median_ms"], compiled_gpu_ms=comp["gpu_median_ms"],
                host_winner="native" if native["host_batch_ms_per_call"] < comp["host_batch_ms_per_call"] else "compiled",
                gpu_winner="native" if native["gpu_median_ms"] < comp["gpu_median_ms"] else "compiled",
                host_speedup=comp["host_batch_ms_per_call"]/min(native["host_batch_ms_per_call"],comp["host_batch_ms_per_call"]),
                gpu_speedup=comp["gpu_median_ms"]/min(native["gpu_median_ms"],comp["gpu_median_ms"])))
    write_csv(out/"paired_ratios.csv", pairs)
    write_csv(out/"candidate_minimum.csv", oracle)
    cv = cross_validate_candidates(oracle)
    write_csv(out/"candidate_cross_validation.csv", cv)
    summaries = []
    for stage, baseline, phase, policy in sorted({(r["stage"],r["baseline"],r["phase"],r["policy"]) for r in pairs}):
        selected = [r for r in pairs if (r["stage"],r["baseline"],r["phase"],r["policy"]) == (stage,baseline,phase,policy)]
        for case in ["all", *sorted({r["case"] for r in selected})]:
            subset = selected if case == "all" else [r for r in selected if r["case"] == case]
            summaries.append(dict(stage=stage, baseline=baseline, phase=phase, policy=policy, case=case,
                pairs=len(subset), cases=len({r["case"] for r in subset}),
                geomean_host_speedup=gm([r["host_speedup"] for r in subset]),
                geomean_gpu_speedup=gm([r["gpu_speedup"] for r in subset]),
                median_host_ms_saved=float(np.median([r["host_ms_saved"] for r in subset])),
                **interval_by_cluster(subset, "host_speedup")))
    write_csv(out/"summary.csv", summaries)
    candidate_summary = []
    for case in ["all", *sorted({r["case"] for r in oracle})] if oracle else []:
        subset = oracle if case == "all" else [r for r in oracle if r["case"] == case]
        candidate_summary.append(dict(case=case, pairs=len(subset),
            native_host_wins=sum(r["host_winner"] == "native" for r in subset),
            native_gpu_wins=sum(r["gpu_winner"] == "native" for r in subset),
            geomean_host_speedup=gm([r["host_speedup"] for r in subset]),
            geomean_gpu_speedup=gm([r["gpu_speedup"] for r in subset]),
            **interval_by_cluster(subset, "host_speedup")))
    write_csv(out/"candidate_summary.csv", candidate_summary)
    cv_summary = []
    for case in ["all", *sorted({r["case"] for r in cv})] if cv else []:
        subset = cv if case == "all" else [r for r in cv if r["case"] == case]
        cv_summary.append(dict(case=case, pairs=len(subset),
            selected_native=sum(r["selected"]=="native" for r in subset),
            geomean_host_speedup=gm([r["host_speedup"] for r in subset]),
            geomean_gpu_speedup=gm([r["gpu_speedup"] for r in subset]),
            geomean_host_speedup_vs_best_static=gm([r["host_speedup_vs_best_static"] for r in subset]),
            geomean_gpu_speedup_vs_best_static=gm([r["gpu_speedup_vs_best_static"] for r in subset]),
            **interval_by_cluster(subset,"host_speedup")))
    write_csv(out/"candidate_cv_summary.csv", cv_summary)
    write_json(out/"summary.json", dict(rows=len(rows), failed=sum(r["status"] != "ok" for r in statuses),
        summaries=summaries, candidate_summary=candidate_summary, candidate_cv_summary=cv_summary,
        limitations=["Ratios pair the same process, workload and phase; device/run clusters are resampled over the fixed case matrix.",
                     "The candidate minimum is optimistic and limited to the measured native/compiled pair, not a new selector or global kernel optimum.",
                     "Cross-validation holds out both devices of each seed, tests known-shape ranking stability, and excludes runtime dispatch/probe costs. It is not unseen-shape generalization or measured selector latency.",
                     "Prebound calls specialize fixed phase shapes; they do not implement a general dynamic-shape runtime.",
                     "Profiler counts are collected after timing and do not measure DRAM traffic or isolate device-resource causality.",
                     "This protocol emulates precision casts in all outer models and preallocates decode positions; old model measurements are not pooled."]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT/"paper/data/jsa_ablation")
    parser.add_argument("--out", type=Path, default=ROOT/"paper/data/jsa_ablation/analysis")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    analyze(args.root, args.out)

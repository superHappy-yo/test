#!/usr/bin/env python3
"""Audit archived selections and replay calibration scopes from recorded data.

Scope ablations reuse the archived probe winners and regular path medians.
They are offline replays, not new GPU timings or estimates of setup overhead.
Replication summaries retain independent process results without pooling them
with the original archive. Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
ARCHIVE = REPO / "paper/data/bakeoff_v4"
FUSE = "always_fuse"
UNFUSE = "always_unfuse"


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def geomean(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Geometric means require finite positive values.")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def read_run(directory: Path) -> tuple[list[dict], dict]:
    decisions = load_csv(directory / "bakeoff_decisions.csv")
    rows = load_csv(directory / "bakeoff_latency.csv")
    indexed = {(row["op"], row["label"], row["baseline"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate latency keys in {directory}")
    if len(decisions) != 12:
        raise ValueError(f"Expected twelve decisions in {directory}")
    if len({(row["op"], row["label"]) for row in decisions}) != len(decisions):
        raise ValueError(f"Duplicate decision keys in {directory}")
    for decision in decisions:
        for baseline in ("eager_unfused", FUSE, UNFUSE, "scic_selected"):
            row = indexed[(decision["op"], decision["label"], baseline)]
            if row["ok"] != "true" or float(row["median_ms"]) <= 0:
                raise ValueError(f"Invalid required latency in {directory}: {baseline}")
        selected = indexed[(decision["op"], decision["label"], "scic_selected")]
        chosen = indexed[(decision["op"], decision["label"], decision["selected_baseline"])]
        if float(selected["median_ms"]) != float(chosen["median_ms"]):
            raise ValueError("Selected median does not match its recorded source.")
    return decisions, indexed


def summarize(directory: Path) -> dict:
    directory = directory.resolve()
    decisions, indexed = read_run(directory)
    source = directory.relative_to(REPO) if directory.is_relative_to(REPO) else directory
    output = {"source": str(source), "groups": {}}
    for op in ("matmul_rmsnorm", "attention"):
        items = [row for row in decisions if row["op"] == op]
        if len(items) != (7 if op == "matmul_rmsnorm" else 5):
            raise ValueError(f"Unexpected shape count for {op} in {directory}")
        speedups, gaps, details, library_ratios = [], [], [], []
        for decision in items:
            key = (op, decision["label"])
            times = {name: float(indexed[(*key, name)]["median_ms"])
                     for name in ("eager_unfused", FUSE, UNFUSE, "scic_selected")}
            winner = FUSE if times[FUSE] < times[UNFUSE] else UNFUSE
            speedups.append(times["eager_unfused"] / times["scic_selected"])
            gaps.append(times["scic_selected"] / min(times[FUSE], times[UNFUSE]))
            details.append({
                "label": decision["label"], "regular_winner": winner,
                "probe_winner": decision["measured_winner"],
                "selected": decision["selected_baseline"],
                "probe_matches_regular": decision["measured_winner"] == winner,
                "analytical_matches_regular": (decision["analytical_baseline"] == winner
                                               if op == "matmul_rmsnorm" else None),
                "latency_ms": times,
            })
            if op == "attention":
                sdpa = indexed[(*key, "sdpa")]
                if sdpa["ok"] != "true":
                    raise ValueError("Missing eager SDPA comparison")
                library_ratios.append(times[FUSE] / float(sdpa["median_ms"]))
        output["groups"][op] = {
            "shapes": len(items), "selected_speedup_geomean": geomean(speedups),
            "selected_gap_geomean": geomean(gaps),
            "probe_matches_regular": (sum(row["probe_matches_regular"] for row in details)
                                      if op == "matmul_rmsnorm" else None),
            "analytical_matches_regular": (sum(row["analytical_matches_regular"] for row in details)
                                           if op == "matmul_rmsnorm" else None),
            "details": details,
        }
        if library_ratios:
            output["groups"][op]["compiled_over_eager_sdpa_geomean"] = geomean(library_ratios)
            output["groups"][op]["compiled_slower_than_eager_sdpa_shapes"] = sum(
                ratio > 1 for ratio in library_ratios)
    return output


def scope_ablation(directory: Path) -> list[dict]:
    decisions, indexed = read_run(directory)
    gemm = [row for row in decisions if row["op"] == "matmul_rmsnorm"]
    settings = [
        ("datasheet_only", "none", 0.5, 2.0),
        ("ridge_only", "ridge", 0.5, 2.0),
        ("ridge_compute", "ridge_compute", 0.5, 2.0),
        ("wide_ridge_only", "ridge", 0.1, 8.0),
        ("all_gemm", "all", 0.5, 2.0),
    ]
    output = []
    for policy, scope, lower, upper in settings:
        speedups, gaps, calibrated = [], [], 0
        for decision in gemm:
            ratio = float(decision["analytical_ai"]) / float(decision["ridge_flops_per_byte"])
            probe = (scope == "all" or (scope == "ridge" and lower <= ratio <= upper)
                     or (scope == "ridge_compute" and ratio >= lower))
            calibrated += probe
            choice = decision["measured_winner"] if probe else decision["analytical_baseline"]
            key = ("matmul_rmsnorm", decision["label"])
            latency = float(indexed[(*key, choice)]["median_ms"])
            eager = float(indexed[(*key, "eager_unfused")]["median_ms"])
            fastest = min(float(indexed[(*key, name)]["median_ms"]) for name in (FUSE, UNFUSE))
            speedups.append(eager / latency)
            gaps.append(latency / fastest)
        output.append({"policy": policy, "band_lower_R": lower, "band_upper_R": upper,
                       "probed_shapes": calibrated,
                       "speedup_vs_eager_geomean": geomean(speedups),
                       "gap_to_two_path_min_geomean": geomean(gaps)})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=Path)
    parser.add_argument("--primary", type=Path, default=REPO / "paper/data/jsa_replication/run_01")
    parser.add_argument("--out", type=Path, default=REPO / "paper/data/jsa_reanalysis")
    args = parser.parse_args()
    runs = []
    if args.replications:
        runs = sorted(args.replications.glob("run_*/bakeoff_summary.json"))
        if not runs:
            parser.error("No completed replication runs found")
    output = {
        "scope": "GPU latency selection, not DRAM intensity validation",
        "legacy_archive": summarize(ARCHIVE), "primary": summarize(args.primary),
        "scope_ablation": scope_ablation(args.primary),
        "replications": [summarize(path.parent) for path in runs],
        "ablation_note": "Offline replay of primary-run probes; no new latency or overhead measurements.",
        "legacy_note": "The legacy FP16 normalization has numerical overflow; its latency ratios are not equivalent-computation speedups.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "evidence_audit.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8")
    with (args.out / "calibration_scope.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output["scope_ablation"][0]))
        writer.writeheader()
        writer.writerows(output["scope_ablation"])
    for row in output["scope_ablation"]:
        print(row)
    for run in output["replications"]:
        print(run["source"], {op: {k: v for k, v in group.items() if k != "details"}
                              for op, group in run["groups"].items()})


if __name__ == "__main__":
    main()

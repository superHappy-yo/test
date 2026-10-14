#!/usr/bin/env python3
"""Repeat the existing CUDA bakeoff without modifying the archived results.

The selected GPU must have two consecutive idle samples before each process.
Compilation and calibration costs are excluded by the existing harness.
Max-autotune is intentionally skipped; all other paths use the archived
three-warmup, ten-iteration protocol and the one-warmup, three-iteration probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone


REPO = Path(__file__).resolve().parents[3]


def gpu_sample(uuid: str) -> dict:
    result = subprocess.run(
        ["nvidia-smi", f"--id={uuid}",
         "--query-gpu=uuid,name,utilization.gpu,memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    fields = [field.strip() for field in result.stdout.strip().split(",")]
    if len(fields) != 5 or fields[0] != uuid:
        raise RuntimeError(f"Unexpected GPU query: {result.stdout!r}")
    return {
        "utc": datetime.now(timezone.utc).isoformat(), "uuid": fields[0],
        "name": fields[1], "utilization_pct": int(fields[2]),
        "memory_used_mib": int(fields[3]), "memory_total_mib": int(fields[4]),
    }


def wait_for_idle(uuid: str, timeout: float) -> list[dict]:
    samples = []
    consecutive = 0
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        sample = gpu_sample(uuid)
        samples.append(sample)
        if sample["name"] != "NVIDIA A40":
            raise RuntimeError("This replication uses the A40 FP16 anchor.")
        idle = sample["utilization_pct"] <= 5 and sample["memory_used_mib"] <= 256
        consecutive = consecutive + 1 if idle else 0
        if consecutive == 2:
            return samples
        time.sleep(2)
    raise RuntimeError("GPU did not meet the idle criterion before the timeout.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", type=Path, default=REPO / "paper/data/jsa_replication")
    parser.add_argument("--idle-timeout", type=float, default=120)
    parser.add_argument("--numerics", type=Path,
                        default=REPO / "paper/data/jsa_numerics/output_validation.json")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    validation = json.loads(args.numerics.read_text(encoding="utf-8"))
    source_hash = hashlib.sha256((REPO / "shapefuse/bakeoff.py").read_bytes()).hexdigest()
    if not validation.get("passed") or validation.get("source_sha256") != source_hash:
        parser.error("Run check_jsa_numerics.py for the current benchmark source first")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    environment["DYNAMO_CACHE_SIZE_LIMIT"] = "64"
    environment["PYTHONUNBUFFERED"] = "1"
    manifest = {
        "gpu_uuid": args.gpu_uuid, "runs_requested": args.runs,
        "idle_criterion": {"consecutive_samples": 2, "max_utilization_pct": 5,
                           "max_memory_used_mib": 256},
        "timing": "CUDA events; existing shapefuse bakeoff",
        "max_autotune": "intentionally skipped",
        "monitor_note": "During-run memory and utilization include the benchmark.",
        "normalization": "FP32 statistics and weighting; cast final output to input dtype",
        "numerical_validation": str(args.numerics.resolve()),
        "source_sha256": source_hash,
        "runs": [],
    }
    for number in range(1, args.runs + 1):
        run_dir = args.out / f"run_{number:02d}"
        run_dir.mkdir()
        idle_samples = wait_for_idle(args.gpu_uuid, args.idle_timeout)
        command = [
            sys.executable, "-m", "shapefuse", "bakeoff", "--out", str(run_dir),
            "--device", "cuda", "--require-cuda", "--machine", "A40_FP16",
            "--level", "decode-prefill", "--attention", "--skip-max-autotune",
            "--skip-plots", "--calibrate", "all-gemm", "--repeats", "10",
            "--warmup", "3",
        ]
        record = {"run": number, "command": command, "idle_samples": idle_samples}
        print(f"Starting replication {number}/{args.runs} on {args.gpu_uuid}", flush=True)
        start = time.monotonic()
        with (run_dir / "run.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=REPO, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT)
            samples = []
            try:
                while process.poll() is None:
                    samples.append(gpu_sample(args.gpu_uuid))
                    time.sleep(2)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
            record["returncode"] = process.returncode
        record["elapsed_seconds_including_setup"] = time.monotonic() - start
        (run_dir / "gpu_monitor.json").write_text(
            json.dumps(samples, indent=2) + "\n", encoding="utf-8")
        manifest["runs"].append(record)
        (args.out / "replication_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if process.returncode != 0:
            raise RuntimeError(f"Replication {number} failed; see {run_dir / 'run.log'}")
        print(f"Finished replication {number}: {record['elapsed_seconds_including_setup']:.1f}s",
              flush=True)


if __name__ == "__main__":
    main()

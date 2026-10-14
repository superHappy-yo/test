#!/usr/bin/env python3
"""Run isolated adapter diagnostics on one GPU, with resumable evidence."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from shapefuse.io_utils import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--stages", default="projection,system")
    parser.add_argument("--cases", default="b1_s128,b1_s2048,b8_s128,b8_s2048")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=args.resume)
    records = json.loads((out / "runs.json").read_text()) if (out / "runs.json").exists() else []
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
               TORCHINDUCTOR_CACHE_DIR=str(ROOT / ".experiments" / f"ablation_cache_gpu{args.gpu}"))
    for run in range(args.runs):
        for stage in args.stages.split(","):
            cases = args.cases.split(",") if stage == "system" else ["projection"]
            for case in cases:
                dest = out / f"gpu{args.gpu}_run{run:02d}" / stage / case
                if (dest / "completed.json").exists() and args.resume:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    dest.rename(dest.with_name(dest.name + f".incomplete_{int(time.time())}"))
                idle = []
                deadline = time.monotonic()+120
                while len(idle) < 2:
                    sample = subprocess.check_output(["nvidia-smi", "-i", args.gpu,
                        "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], text=True).strip()
                    util, mem = map(int, sample.split(","))
                    if util <= 5 and mem <= 256:
                        idle.append(sample)
                    else:
                        idle.clear()
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"GPU {args.gpu} is occupied: {sample}")
                    time.sleep(1)
                cmd = [args.python, "-m", "shapefuse.ablation", "--stage", stage,
                       "--out", str(dest), "--case", case if stage == "system" else "b1_s128",
                       "--seed", str(run), "--warmup", "5" if stage == "system" else "50",
                       "--repeats", "20" if stage == "system" else "200"]
                record = dict(gpu=args.gpu, run=run, stage=stage, case=case, command=cmd, idle=idle)
                print("START", args.gpu, run, stage, case, flush=True)
                start = time.monotonic()
                with dest.with_suffix(".log").open("w") as log:
                    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        proc.wait(timeout=args.timeout)
                        record.update(exit_code=proc.returncode,
                                      status="completed" if proc.returncode == 0 else "failed")
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                        record["status"] = "timeout"
                record["seconds"] = time.monotonic()-start
                records.append(record)
                write_json(out / "runs.json", records)
                print("END", record["status"], round(record["seconds"], 1), flush=True)
    write_json(out / "completed.json", dict(processes=len(records), included=sum(not r.get("excluded", False) for r in records),
               failed=sum(r["status"] != "completed" for r in records)))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Replicate supplementary measurements, isolating failures with timeouts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from shapefuse.io_utils import write_json
from shapefuse.supplement import operator_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stage", default="operator", choices=("operator","research","system"))
    parser.add_argument("--graphs", default="none,manual")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--cases", default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-cases", action="store_true",
                        help="Operator/research stage: run the shape matrix in one process per device/run/graph")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=args.resume)
    cases = ([f"b{b}_s{s}" for b in (1,8) for s in (128,2048)] if args.stage=="system"
             else [c["label"] for c in operator_cases()
                   if args.stage!="research" or c["op"]=="attention"])
    if args.cases:
        cases = args.cases.split(",")
    if args.batch_cases:
        if args.stage == "system" or args.cases:
            raise ValueError("--batch-cases requires the full operator or research matrix")
        cases=["all"]
    records=json.loads((out/"runs.json").read_text()) if args.resume and (out/"runs.json").exists() else []
    for gpu in args.gpus.split(","):
        for run in range(args.runs):
            for graph in args.graphs.split(","):
                for case in cases:
                    env = os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1",
                               TOKENIZERS_PARALLELISM="false")
                    dest=out/f"gpu{gpu}_run{run:02d}"/graph/case
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if (dest/"completed.json").exists() and args.resume:
                        print("RESUME_SKIP",gpu,run,graph,case,flush=True)
                        continue
                    if dest.exists():
                        dest.rename(dest.with_name(dest.name+f".incomplete_{int(time.time())}"))
                    cmd=[args.python,"-m","shapefuse.cli","supplement","--stage",args.stage,
                         "--out",str(dest),"--seed",str(run),"--case",case,"--graph-mode",graph,
                         "--warmup","50" if args.stage!="system" else "5",
                         "--repeats","200" if args.stage!="system" else "20"]
                    if args.methods:
                        cmd += ["--methods", args.methods]
                    if args.batch_cases:
                        i=cmd.index("--case");del cmd[i:i+2]
                    # Require two idle samples before creating a GPU context.
                    idle=[]
                    deadline=time.monotonic()+120
                    while len(idle)<2:
                        sample=subprocess.check_output(["nvidia-smi","-i",gpu,
                            "--query-gpu=utilization.gpu,memory.used","--format=csv,noheader,nounits"],
                            text=True).strip()
                        util,mem=map(int,sample.split(","))
                        if util<=5 and mem<=256:
                            idle.append(sample)
                        else:
                            idle.clear()
                        if time.monotonic()>deadline:
                            raise RuntimeError(f"GPU {gpu} is occupied: {sample}; resume into a new output root")
                        time.sleep(1)
                    record=dict(gpu=gpu,run=run,graph=graph,case=case,command=cmd,idle=idle)
                    print(gpu,run,graph,case,flush=True)
                    start=time.time()
                    with dest.with_suffix(".log").open("w") as log:
                        try:
                            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                                                  start_new_session=True)
                            try:
                                proc.wait(timeout=max(args.timeout,900) if args.batch_cases else args.timeout)
                                record["exit_code"]=proc.returncode
                                record["status"]="completed" if proc.returncode==0 else "failed"
                            except subprocess.TimeoutExpired:
                                import signal
                                os.killpg(proc.pid,signal.SIGKILL)
                                proc.wait()
                                record["status"]="timeout"
                        except Exception as exc:
                            record.update(status="failed",error=str(exc))
                    record["seconds"]=time.time()-start
                    records.append(record)
                    write_json(out/"runs.json",records)
    write_json(out/"completed.json",dict(processes=len(records),
               completed=sum(r["status"]=="completed" for r in records)))

if __name__=="__main__":
    main()

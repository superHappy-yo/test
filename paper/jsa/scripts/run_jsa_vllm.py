#!/usr/bin/env python3
"""Isolate each vLLM policy/seed in its own engine worker process."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from shapefuse.io_utils import write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--python",default=sys.executable)
    p.add_argument("--out",required=True)
    p.add_argument("--gpus",default="0")
    p.add_argument("--runs",type=int,default=3)
    p.add_argument("--policies",default="original,fixed_native,fixed_compile,scic")
    p.add_argument("--requests",type=int,default=128)
    p.add_argument("--timeout",type=int,default=900)
    p.add_argument("--case")
    args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    records=[]
    for gpu in args.gpus.split(","):
        for seed in range(args.runs):
            policies=args.policies.split(",")
            __import__("random").Random(seed).shuffle(policies)
            for policy in policies:
                idle=[];deadline=time.monotonic()+120
                while len(idle)<2:
                    sample=subprocess.check_output(["nvidia-smi","-i",gpu,
                        "--query-gpu=utilization.gpu,memory.used","--format=csv,noheader,nounits"],text=True).strip()
                    util,mem=map(int,sample.split(","))
                    if util<=5 and mem<=256: idle.append(sample)
                    else: idle.clear()
                    if time.monotonic()>deadline: raise RuntimeError(f"GPU {gpu} occupied: {sample}")
                    time.sleep(1)
                dest=out/f"gpu{gpu}_run{seed:02d}"/policy;dest.parent.mkdir(parents=True,exist_ok=True)
                cmd=[args.python,"-m","shapefuse.cli","supplement","--stage","system",
                     "--framework","vllm","--out",str(dest),"--methods",policy,
                     "--seed",str(seed),"--requests",str(args.requests)]
                if args.case: cmd += ["--case",args.case]
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONUNBUFFERED="1",
                    PYTHONPATH=str(ROOT),VLLM_NO_USAGE_STATS="1",DO_NOT_TRACK="1",HF_HUB_DISABLE_TELEMETRY="1")
                print(gpu,seed,policy,flush=True);start=time.monotonic()
                record=dict(gpu=gpu,seed=seed,policy=policy,command=cmd,idle=idle)
                with dest.with_suffix(".log").open("w") as log:
                    proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    try:
                        proc.wait(timeout=args.timeout)
                        record.update(exit_code=proc.returncode,status="completed" if proc.returncode==0 else "failed")
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL);proc.wait();record["status"]="timeout"
                record["seconds"]=time.monotonic()-start;records.append(record)
                write_json(out/"runs.json",records)
    write_json(out/"completed.json",dict(processes=len(records),completed=sum(r["status"]=="completed" for r in records)))


if __name__=="__main__": main()

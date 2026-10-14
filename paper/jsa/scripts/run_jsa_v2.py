#!/usr/bin/env python3
"""Serial isolated workers with idle gates, timeouts, and resumable artifacts."""
import argparse,json,os,subprocess,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
p=argparse.ArgumentParser();p.add_argument('--python',default=str(ROOT/'.experiments/envs/vllm/bin/python'))
p.add_argument('--out',default=str(ROOT/'paper/data/jsa_v2'));p.add_argument('--stage',default='operator')
p.add_argument('--gpus',default='0,1');p.add_argument('--runs',type=int,default=3);p.add_argument('--run-offset',type=int,default=0)
p.add_argument('--timeout',type=int,default=3600);p.add_argument('--prior');p.add_argument('--requests',type=int,default=128)
a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
records_path=out/('runs.json' if a.stage=='operator' and a.run_offset==0 else f'runs_{a.stage}_{a.run_offset}.json')
records=json.loads(records_path.read_text()) if records_path.exists() else []
for gpu in a.gpus.split(','):
 for seed in range(a.run_offset,a.run_offset+a.runs):
  dest=out/f'gpu{gpu}_run{seed:02d}'/a.stage
  if (dest/'completed.json').exists():continue
  dest.parent.mkdir(parents=True,exist_ok=True)
  policies=('original','best-fixed','legacy','full-calibration','scic-v2','scic-v2-no-ai') if a.stage.startswith('serving') else ('all',)
  failures=0
  for policy in policies:
   leaf=dest/policy if policy!='all' else dest
   if (leaf/'completed.json').exists():continue
   leaf.parent.mkdir(parents=True,exist_ok=True)
   samples=[];deadline=time.monotonic()+120
   while len(samples)<2:
    v=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip()
    util,mem=map(int,v.split(','));samples=samples+[v] if util<=5 and mem<=256 else []
    if time.monotonic()>deadline:raise RuntimeError(f'GPU occupied {gpu}: {v}')
    time.sleep(1)
   env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',TORCHINDUCTOR_CACHE_DIR=str(ROOT/'.experiments'/f'v2_cache_gpu{gpu}'))
   if a.stage=='operator-legacy':
    cmd=[a.python,'-m','shapefuse.v2_legacy_experiment','--out',str(leaf),'--seed',str(seed)]
   elif a.stage.startswith('operator'):
    cmd=[a.python,'-m','shapefuse.v2_experiment','--out',str(leaf),'--seed',str(seed)]
    if a.stage=='operator-vllm':cmd+=['--semantic','vllm']
   elif a.stage.startswith('hf'):
    cmd=[a.python,'-m','shapefuse.v2_system','--out',str(leaf),'--seed',str(seed)]
    if a.stage=='hf-pretrained':cmd+=['--pretrained']
    if a.prior:cmd+=['--prior',a.prior]
   elif a.stage.startswith('serving'):
    cmd=[a.python,'-m','shapefuse.v2_serving_experiment','--out',str(leaf),'--seed',str(seed),'--policy',policy,'--requests',str(a.requests)]
    if a.stage=='serving-pretrained':cmd+=['--pretrained']
    if a.prior:cmd+=['--prior',a.prior]
   else:raise ValueError(a.stage)
   start=time.monotonic();print('START',gpu,seed,a.stage,policy,flush=True)
   with leaf.with_suffix('.log').open('w') as log:
    proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:proc.wait(timeout=a.timeout);status='completed' if proc.returncode==0 else 'failed'
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();status='timeout'
   failures+=status!='completed'
   records.append(dict(gpu=gpu,seed=seed,stage=a.stage,policy=policy,status=status,seconds=time.monotonic()-start,command=cmd,idle=samples))
   records_path.write_text(json.dumps(records,indent=2)+'\n');print('END',status,round(records[-1]['seconds'],1),flush=True)
  if a.stage.startswith('serving'):
   dest.mkdir(exist_ok=True);(dest/'completed.json').write_text(json.dumps(dict(failures=failures))+'\n')

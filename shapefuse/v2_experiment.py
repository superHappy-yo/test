"""Isolated SCIC-v2 operator experiment. No profiler imported or initialized."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback

from shapefuse.v2_policy import LatencyPrior,choose,training_fixed
from shapefuse.io_utils import write_json


def geometries():
    from shapefuse.supplement import operator_cases
    cases=[c for c in operator_cases() if c['op']=='gemm']
    for m,n,k in [(m,1024,1024) for m in (8,1024,16384)]+[(m,2048,2048) for m in (1,8,128,2048)]+[(8,n,n) for n in (4096,8192)]+[(8,n,k) for k,n in ((1024,3072),(3072,1024))]:
        cases.append(dict(label=f'v2_m{m}_n{n}_k{k}',M=m,N=n,K=k,op='gemm'))
    assert len(cases)==32 and len({(c['M'],c['N'],c['K']) for c in cases})==32
    return sorted(cases,key=lambda c:(c['N'] in (2048,8192),c['N'],c['K'],c['M']))


def check(actual,expected):
    import torch
    outputs=[]
    for a,b in zip(actual,expected):
        diff=a.float()-b.float();den=float(torch.linalg.vector_norm(b.float()))
        rel=float(torch.linalg.vector_norm(diff))/max(den,1e-20)
        finite=bool(torch.isfinite(a).all())
        outputs.append(dict(finite=finite,allclose=bool(torch.allclose(a,b,atol=1e-3,rtol=1e-3)),
                            relative_l2=rel,max_abs=float(diff.abs().max())))
    return dict(passed=all(r['finite'] and r['allclose'] and r['relative_l2']<=1e-3 for r in outputs),outputs=outputs)


def timing(call,*,warm=50,samples=200,batches=5,batch_size=40):
    import torch
    for _ in range(warm):call()
    torch.cuda.synchronize()
    events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(samples)]
    for a,b in events:a.record();call();b.record()
    torch.cuda.synchronize()
    gpu=[a.elapsed_time(b) for a,b in events]
    host=[]
    for _ in range(batches):
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(batch_size):call()
        torch.cuda.synchronize();host.append((time.perf_counter()-start)*1000/batch_size)
    return dict(host_ms=statistics.median(host),host_samples_ms=host,gpu_ms=statistics.median(gpu) if gpu else None,gpu_samples_ms=gpu,
                warm=warm,batches=batches,batch_size=batch_size)


def inputs_for(c,seed,*,stress=False,unit=False):
    import torch
    gen=torch.Generator().manual_seed(seed)
    m,n,k=c['M'],c['N'],c['K']
    def draw(shape,scale=1):return (torch.randn(shape,generator=gen)*scale).to('cuda',torch.float16)
    return draw((m,k),4 if stress else 1),draw((n,k),.2 if stress else .02),draw((m,n),.1 if stress else 1),torch.ones(n,device='cuda',dtype=torch.float16) if unit else draw((n,),.2)+1


def run(args):
    import torch
    from shapefuse.v2_kernels import NAMES,candidate,reference
    from shapefuse.executor import graph_callable
    from shapefuse.supplement import environment
    torch.set_num_threads(1)
    torch._dynamo.config.recompile_limit=512
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    write_json(out/'manifest.json',dict(environment=environment(),protocol='v2-1',seed=args.seed,
        primary='median synchronized host batches',train_widths=[1024,4096,3072],test_square_widths=[2048,8192],profiler=False,
        semantic=args.semantic,expected_calls=1024,v2_sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('shapefuse').glob('v2_*.py')}))
    rows=[];records=[]
    cases=geometries()
    if args.smoke:cases=[cases[1],next(c for c in cases if c['N']==2048)]
    with torch.inference_mode():
        for ci,c in enumerate(cases):
            train=not (c['K']==c['N'] and c['N'] in (2048,8192))
            inputs=inputs_for(c,args.seed+100)
            semantic=args.semantic=='vllm'
            raw={'native':candidate('native',vllm=semantic),**{n:candidate(n,vllm=semantic) for n in NAMES}}
            funcs={n:torch.compile(fn,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False,'emulate_precision_casts':True}) for n,fn in raw.items() if n!='native'}
            funcs['eager']=raw['native']
            funcs['compile-default']=torch.compile(raw['native'],fullgraph=True,dynamic=False,options={'triton.cudagraphs':False,'emulate_precision_casts':True})
            funcs['max-autotune']=torch.compile(raw['native'],fullgraph=True,dynamic=False,mode='max-autotune-no-cudagraphs',options=None)
            valid={};costs={};audit={}
            for name,fn in funcs.items():
                t=time.perf_counter()
                try:
                    fn(*inputs);torch.cuda.synchronize();compilation_ms=(time.perf_counter()-t)*1000
                    t=time.perf_counter();checks=[]
                    # Three independent input seeds, unit/nonunit gamma and stress.
                    for seed in (11,23,37):
                        for stress,unit in ((False,True),(False,False),(True,False)):
                            test=inputs_for(c,seed,stress=stress,unit=unit)
                            checks.append(check(fn(*test),reference(*test,vllm=semantic)))
                    torch.cuda.synchronize();validation_ms=(time.perf_counter()-t)*1000
                    passed=all(v['passed'] for v in checks)
                    audit[name]=dict(passed=passed,checks=checks)
                    costs[name]=dict(compilation_ms=compilation_ms,validation_ms=validation_ms)
                    if passed:valid[name]=fn
                except Exception as exc:
                    audit[name]=dict(passed=False,error=repr(exc),traceback=traceback.format_exc())
                print('GATE',ci,c['label'],name,audit[name]['passed'],flush=True)
            # Don't allow the last numerical test to change the timed data.
            for reuse in ('same','rotated'):
                ins=[inputs] if reuse=='same' else [inputs_for(c,args.seed+100+i) for i in range(6)]
                for mode in ('direct','graph'):
                    calls={};counter=[0]
                    names=list(valid)
                    __import__('random').Random(args.seed*1000+ci*10+(mode=='graph')+2*(reuse=='rotated')).shuffle(names)
                    for name in names:
                        fn=valid[name]
                        try:
                            runners=[graph_callable(fn,v) for v in ins] if mode=='graph' else [fn]*len(ins)
                            def call(runners=runners,ins=ins):
                                i=counter[0]%len(ins);counter[0]+=1
                                return runners[i](*ins[i])
                            calls[name]=call
                            measured=timing(call,warm=3 if args.smoke else 50,samples=5 if args.smoke else 200,
                                            batch_size=4 if args.smoke else 40)
                            row=dict(**c,candidate=name,mode=mode,reuse=reuse,train=train,passed=True,**measured)
                            rows.append(row)
                        except Exception as exc:
                            records.append(dict(**c,candidate=name,mode=mode,reuse=reuse,status='timing-failed',error=repr(exc)))
                    records.append(dict(**c,mode=mode,reuse=reuse,train=train,audit=audit,costs=costs))
            write_json(out/'latencies.json',rows);write_json(out/'records.json',records)
            del funcs,valid,raw,inputs,ins,calls
            torch.cuda.empty_cache()
    # Fit using training geometries only; never select features/hyperparameters using held-out results.
    trainrows=[r for r in rows if r['train'] and r['candidate'] in NAMES]
    priors={p:LatencyPrior(analytical=p!='scic-v2-no-ai').fit(trainrows) for p in ('scic-v2','scic-v2-no-ai')}
    write_json(out/'priors.json',{p:v.dump() for p,v in priors.items()})
    # Independent live probes on held-out shapes, separate from oracle steady samples.
    selections=[]
    with torch.inference_mode():
        for c in cases:
            if not (c['N']==c['K'] and c['N'] in (2048,8192)):continue
            for reuse in ('same','rotated'):
                ins=[inputs_for(c,args.seed+100+i) for i in range(1 if reuse=='same' else 6)]
                for mode in ('direct','graph'):
                    subset=[r for r in rows if r['label']==c['label'] and r['mode']==mode and r['reuse']==reuse and r['candidate'] in NAMES]
                    available=[r['candidate'] for r in subset]
                    if not available:
                        fallback=next(r for r in rows if r['label']==c['label'] and r['mode']==mode and r['reuse']==reuse and r['candidate']=='eager')
                        for policy in ('legacy','full-calibration','scic-v2','scic-v2-no-ai'):
                            selections.append(dict(**c,mode=mode,reuse=reuse,policy=policy,status='native-fallback',selected='eager',static='eager',probe={},predicted={},probe_ms=0,fallback='no-valid-fused-candidate',oracle='eager',selected_host_ms=fallback['host_ms'],oracle_host_ms=fallback['host_ms'],expected_calls=1024))
                        write_json(out/'selections.json',selections);continue
                    # Best fixed candidate is training-only and context-specific.
                    training=[r for r in trainrows if r['mode']==mode and r['reuse']==reuse]
                    static=training_fixed(rows,mode,reuse)
                    callables={};counter=[0]
                    for name in available:
                        fn=torch.compile(candidate(name,vllm=args.semantic=='vllm'),fullgraph=True,dynamic=False,options={'triton.cudagraphs':False,'emulate_precision_casts':True})
                        fn(*ins[0]);torch.cuda.synchronize()
                        runners=[graph_callable(fn,v) for v in ins] if mode=='graph' else [fn]*len(ins)
                        def call(runners=runners,ins=ins):
                            i=counter[0]%len(ins);counter[0]+=1;return runners[i](*ins[i])
                        callables[name]=call
                    for policy in ('legacy','full-calibration','scic-v2','scic-v2-no-ai'):
                        def probe(name):
                            t=time.perf_counter();v=timing(callables[name],warm=3,samples=0,batch_size=8)
                            return v['host_ms'],(time.perf_counter()-t)*1000
                        chosen,probes,pred,cost,fallback=choose(policy,available,priors.get(policy),
                            (c['M'],c['N'],c['K']),mode,reuse,static,probe)
                        selections.append(dict(**c,mode=mode,reuse=reuse,policy=policy,selected=chosen,static=static,
                            probe=probes,predicted=pred,probe_ms=cost,fallback=fallback,
                            oracle=min(subset,key=lambda r:r['host_ms'])['candidate'],
                            selected_host_ms=next(r['host_ms'] for r in subset if r['candidate']==chosen),
                            oracle_host_ms=min(r['host_ms'] for r in subset),expected_calls=1024))
                        if policy.startswith('scic-v2'):
                            for expected_calls in (1,16,128,8192):
                                best=min(probes,key=probes.get);inc=static if static in probes else chosen
                                selected=best if expected_calls*(probes[inc]-probes[best])>cost else inc
                                selections[-1].setdefault('amortization',[]).append(dict(expected_calls=expected_calls,selected=selected))
                    write_json(out/'selections.json',selections)
                    del callables
                    torch.cuda.empty_cache()
    write_json(out/'completed.json',dict(rows=len(rows),selections=len(selections),seed=args.seed))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--semantic',choices=['hf','vllm'],default='hf');p.add_argument('--smoke',action='store_true')
    run(p.parse_args())
if __name__=='__main__':main()

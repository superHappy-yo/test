"""Original SCIC dispatcher and semantic-preserving max-autotune controls."""
import argparse,time,traceback,random
from pathlib import Path
import torch
from shapefuse.v2_experiment import geometries,inputs_for,check,timing
from shapefuse.executor import graph_callable
from shapefuse.model_system import ProjectionSite,project_add_norm,register_operator
from shapefuse.io_utils import write_json
from shapefuse.supplement import environment


def precision_reference(x,w,r,g):
 c=torch.nn.functional.linear(x,w)
 summed=(c.float()+r.float()).half()
 v=summed.float()
 normalized=v*torch.rsqrt(v.square().mean(-1,keepdim=True)+1e-6)
 return normalized.half()*g,summed

def run(a):
 register_operator()
 torch.set_num_threads(1);torch._dynamo.config.recompile_limit=512
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rows=[];records=[]
 write_json(out/'manifest.json',dict(environment=environment(),seed=a.seed,profiler=False,
   policies=['unmodified ProjectionSite(scic)','max_autotune=True, emulate_precision_casts=True, cudagraphs=False']))
 with torch.inference_mode():
  for ci,c in enumerate(geometries()):
   ins=inputs_for(c,a.seed+100);site=ProjectionSite('scic')
   funcs={'legacy-scic':site.invoke,'max-autotune-precision':torch.compile(precision_reference,fullgraph=True,dynamic=False,
      options={'max_autotune':True,'triton.cudagraphs':False,'emulate_precision_casts':True})}
   valid={}
   for name,fn in funcs.items():
    record=dict(**c,candidate=name)
    try:
     start=time.perf_counter();fn(*ins);torch.cuda.synchronize();record['compilation_ms']=(time.perf_counter()-start)*1000
     audits=[]
     for seed in (11,23,37):
      for stress,unit in ((False,True),(False,False),(True,False)):
       v=inputs_for(c,seed,stress=stress,unit=unit)
       audits.append(check(fn(*v),project_add_norm(*v)))
     record.update(audit=audits,prepared=site.prepared if name=='legacy-scic' else [])
     if not all(v['passed'] for v in audits):raise ArithmeticError('Component gate failed')
     valid[name]=fn;record['status']='passed'
    except Exception as exc:record.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    records.append(record);print('CONTROL',c['label'],name,record['status'],flush=True)
   for reuse in ('same','rotated'):
    pool=[ins] if reuse=='same' else [inputs_for(c,a.seed+100+i) for i in range(6)]
    for mode in ('direct','graph'):
     names=list(valid);random.Random(a.seed*100+ci).shuffle(names)
     for name in names:
      fn=valid[name];runners=[graph_callable(fn,v) for v in pool] if mode=='graph' else [fn]*len(pool);counter=[0]
      def call():
       i=counter[0]%len(pool);counter[0]+=1;return runners[i](*pool[i])
      rows.append(dict(**c,reuse=reuse,mode=mode,candidate=name,**timing(call)))
   write_json(out/'latencies.json',rows);write_json(out/'records.json',records)
   torch.cuda.empty_cache()
 write_json(out/'completed.json',dict(rows=len(rows),failed=sum(r['status']!='passed' for r in records)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=0);run(p.parse_args())

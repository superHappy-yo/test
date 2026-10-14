"""Modern Qwen3 full-model validation and timing without profiler initialization."""
import argparse,copy,hashlib,json,time,traceback
from pathlib import Path
import torch
from shapefuse.io_utils import write_json
from shapefuse.v2_experiment import timing
from shapefuse.v2_runtime import adapt_hf
from shapefuse.v2_policy import LatencyPrior

POLICIES=('original','best-fixed','legacy','full-calibration','scic-v2','scic-v2-no-ai')
CHECKPOINT=Path('.experiments/models/Qwen3-0.6B-c1899de')


def load_prior(path,policy):
    import numpy as np
    data=json.loads(Path(path).read_text())[policy]
    from shapefuse.v2_policy import VERSION
    if data.get('feature_version')!=VERSION:raise ValueError('Latency prior feature version mismatch')
    prior=LatencyPrior(analytical=data['analytical'])
    prior.models={n:(np.array(v['mean']),np.array(v['scale']),np.array(v['coef']),v['error_log']) for n,v in data['models'].items()}
    return prior



def fixed_from_artifact(path):
    from shapefuse.v2_policy import training_fixed
    rows=Path(path).with_name('latencies.json')
    return training_fixed(json.loads(rows.read_text())) if rows.exists() else 'blas4'


def compare(actual,expected):
    checks=[]
    for a,b in zip(actual,expected):
        diff=a.float()-b.float();rel=float(torch.linalg.vector_norm(diff))/max(float(torch.linalg.vector_norm(b.float())),1e-20)
        checks.append(dict(max_abs=float(diff.abs().max()),relative_l2=rel,
            passed=bool(torch.isfinite(a).all() and torch.allclose(a,b,atol=1e-2,rtol=1e-2)) and rel<=1e-3))
    return dict(passed=all(v['passed'] for v in checks),outputs=checks)


def run(args):
    from transformers import Qwen3Config,Qwen3ForCausalLM,StaticCache
    from shapefuse.model_system import MODEL_CONFIG
    from shapefuse.supplement import environment
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    torch._dynamo.config.recompile_limit=512
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        if args.pretrained:
            base=Qwen3ForCausalLM.from_pretrained(str(CHECKPOINT),dtype=torch.float16,attn_implementation='sdpa',local_files_only=True).eval().cuda()
        else:
            cfg=Qwen3Config(**MODEL_CONFIG);cfg._attn_implementation='sdpa'
            base=Qwen3ForCausalLM(cfg).eval().half().cuda()
    config=base.config
    write_json(out/'manifest.json',dict(environment=environment(),config=config.to_dict(),parameters=sum(p.numel() for p in base.parameters()),
       pretrained=args.pretrained,revision='c1899de289a04d12100db370d81485cdf75e47ca' if args.pretrained else None,seed=args.seed,
       profiler=False,teacher_forced=True,decode_steps=64,prior_source=args.prior,outer_compile='default, fullgraph, no graphs, preserve casts'))
    rows=[];records=[]
    cases=[(1,128),(8,2048)] if args.pretrained else [(b,s) for b in (1,8) for s in (128,2048)]
    if args.case:cases=[(int(args.case.split('_')[0][1:]),int(args.case.split('_')[1][1:]))]
    policies=list(POLICIES)
    # Original is reference; other order rotates independently across processes.
    import random
    tail=policies[1:];random.Random(args.seed).shuffle(tail);policies=[policies[0]]+tail
    for b,s in cases:
        gen=torch.Generator().manual_seed(args.seed+b+s)
        tokens=torch.randint(3,config.vocab_size,(b,s+64),generator=gen).cuda()
        mask=torch.ones((b,s+64),device='cuda',dtype=torch.long)
        prepos=torch.arange(s,device='cuda').unsqueeze(0).expand(b,-1)
        positions=[torch.full((b,1),s+i,device='cuda',dtype=torch.long) for i in range(64)]
        reference=None
        for policy in policies:
            record=dict(case=f'b{b}_s{s}',policy=policy,seed=args.seed)
            print('SYSTEM',args.pretrained,b,s,policy,flush=True)
            try:
                with torch.inference_mode():
                    model=copy.deepcopy(base);preparers=[];setup=time.perf_counter()
                    if policy!='original':
                        prior=load_prior(args.prior,policy) if policy.startswith('scic-v2') else None
                        model,preparers=adapt_hf(model,policy,batch=b,prompt=s,prior=prior,static=fixed_from_artifact(args.prior))
                    cache=StaticCache(config=config,max_cache_len=s+64)
                    def prefill():
                        for layer in cache.layers:layer.cumulative_length.zero_()
                        return model(tokens[:,:s],attention_mask=mask,position_ids=prepos,past_key_values=cache,use_cache=True,logits_to_keep=1).logits
                    def decode(collect=False):
                        for layer in cache.layers:layer.cumulative_length.fill_(s)
                        results=[]
                        for i in range(64):
                            y=model(tokens[:,s+i:s+i+1],attention_mask=mask,position_ids=positions[i],past_key_values=cache,use_cache=True,logits_to_keep=1).logits
                            if collect:results.append(y.clone())
                        return results
                    model=torch.compile(model,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False,'emulate_precision_casts':True})
                    actual=[prefill().clone(),*decode(True)];torch.cuda.synchronize()
                    if reference is None:reference=actual
                    audit=compare(actual,reference)
                    record.update(setup_ms=(time.perf_counter()-setup)*1000,audit=audit,
                       preparation=[dict(layer=i,**r) for i,p in enumerate(preparers) for r in p.records],
                       token_sha256=hashlib.sha256(torch.stack([v.argmax(-1) for v in actual]).cpu().numpy().tobytes()).hexdigest())
                    if not audit['passed']:
                        # Preserve strict failure. Optional exploratory timing cannot support acceptance.
                        record['strict_numerics_failed']=True
                        if not all(torch.isfinite(a).all() and torch.allclose(a,r,atol=1e-2,rtol=1e-2) for a,r in zip(actual,reference)):
                            raise ArithmeticError('Full logits elementwise gate failed')
                        print('STRICT_GATE_FAILED: exploratory timing only',flush=True)
                    pre=timing(prefill,warm=5,samples=20,batches=5,batch_size=4)
                    prefill();dec=timing(decode,warm=2,samples=10,batches=5,batch_size=2)
                    for phase,v in (('prefill',pre),('decode64',dec)):
                        rows.append(dict(case=record['case'],policy=policy,seed=args.seed,phase=phase,pretrained=args.pretrained,numerics_passed=audit['passed'],**v))
                    record['status']='passed' if audit['passed'] else 'strict-gate-failed'
                    print('PASS',pre['host_ms'],dec['host_ms'],flush=True)
            except Exception as exc:
                record.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
                print('FAIL',repr(exc),flush=True)
            records.append(record);write_json(out/'latencies.json',rows);write_json(out/'records.json',records)
            if 'model' in locals():del model
            torch._dynamo.reset();torch.cuda.empty_cache()
    write_json(out/'completed.json',dict(rows=len(rows),failures=sum(r['status']!='passed' for r in records)))


def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=0)
 p.add_argument('--prior',default='paper/data/jsa_v2/gpu0_run00/operator/priors.json');p.add_argument('--pretrained',action='store_true');p.add_argument('--case')
 run(p.parse_args())
if __name__=='__main__':main()

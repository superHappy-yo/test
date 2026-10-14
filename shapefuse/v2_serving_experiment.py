"""Actual vLLM request benchmark, primary process free from TorchProfiler."""
import argparse,asyncio,hashlib,json,os,random,time
from pathlib import Path
from shapefuse.io_utils import write_json


async def run(args):
    import numpy as np
    from vllm import SamplingParams,ModelRegistry
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from shapefuse.model_system import MODEL_CONFIG
    from shapefuse.supplement import environment
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    os.environ['SHAPEFUSE_V2_POLICY']=args.policy
    if args.prior:os.environ['SHAPEFUSE_V2_SERVING_PRIOR']=str(Path(args.prior).resolve())
    cls='ScicQwen3ForCausalLM' if args.policy=='legacy' else 'V2Qwen3ForCausalLM'
    location='shapefuse.vllm_system:ScicQwen3ForCausalLM' if args.policy=='legacy' else 'shapefuse.v2_vllm:V2Qwen3ForCausalLM'
    if args.policy=='legacy':os.environ['SHAPEFUSE_VLLM_POLICY']='scic'
    ModelRegistry.register_model(cls,location)
    if args.pretrained:
        source=Path('.experiments/models/Qwen3-0.6B-c1899de')
        model=out/'model';model.mkdir(exist_ok=True)
        config=json.loads((source/'config.json').read_text());config['architectures']=[cls]
        write_json(model/'config.json',config)
        weights=model/'model.safetensors'
        if not weights.exists():weights.symlink_to((source/'model.safetensors').resolve())
    else:
        model=out/'model';model.mkdir(exist_ok=True);config=dict(MODEL_CONFIG,architectures=[cls]);write_json(model/'config.json',config)
    setup=time.perf_counter()
    engine=AsyncLLM.from_engine_args(AsyncEngineArgs(model=str(model.resolve()),load_format='auto' if args.pretrained else 'dummy',skip_tokenizer_init=True,dtype='half',seed=args.seed,
       tensor_parallel_size=1,max_model_len=4096,max_num_seqs=16,max_num_batched_tokens=4096,gpu_memory_utilization=.5,
       worker_cls='shapefuse.v2_vllm.V2Worker' if args.policy!='legacy' else 'auto',
       worker_extension_cls='shapefuse.vllm_system.ProjectionStatsWorkerExtension' if args.policy=='legacy' else 'shapefuse.v2_vllm.V2Stats',
       enable_prefix_caching=True,enable_chunked_prefill=True))
    write_json(out/'manifest.json',dict(environment=environment(),config=config,policy=args.policy,pretrained=args.pretrained,
       seed=args.seed,setup_ms=(time.perf_counter()-setup)*1000,profiler=False,requests=args.requests,output_tokens=64,prior=args.prior))
    params=SamplingParams(temperature=0,max_tokens=64,ignore_eos=True,detokenize=False)
    async def request(i,length):
        rng=random.Random(args.seed*1000000+i)
        tokens=[rng.randrange(3,config['vocab_size']) for _ in range(length)]
        begin=time.perf_counter();arrivals=[];generated=[]
        async for result in engine.generate({'prompt_token_ids':tokens},params,request_id=str(i)):
            if result.outputs:
                ids=list(result.outputs[0].token_ids)
                if len(ids)>len(generated):arrivals.extend([time.perf_counter()-begin]*(len(ids)-len(generated)));generated=ids
        if len(generated)!=64:raise RuntimeError('Wrong output length')
        return dict(request=i,latency_s=time.perf_counter()-begin,ttft_s=arrivals[0],itl_s=[b-a for a,b in zip(arrivals,arrivals[1:])],tokens=generated,
                    output_sha256=hashlib.sha256(json.dumps(generated).encode()).hexdigest())
    rows=[];records=[]
    try:
        serial=0
        for length in (128,2048):
            for concurrency in ((1,8) if args.pretrained else (1,8,16)):
                if args.case and args.case!=f's{length}_c{concurrency}':continue
                for _ in range(2):
                    await asyncio.gather(*(request(serial+j,length) for j in range(concurrency)));serial+=concurrency
                start=time.perf_counter();samples=[]
                for offset in range(0,args.requests,concurrency):
                    batch=min(concurrency,args.requests-offset)
                    samples.extend(await asyncio.gather(*(request(serial+j,length) for j in range(batch))));serial+=batch
                duration=time.perf_counter()-start;itl=[v for sample in samples for v in sample['itl_s']]
                rows.append(dict(case=f's{length}_c{concurrency}',policy=args.policy,seed=args.seed,pretrained=args.pretrained,requests=len(samples),throughput=len(samples)*64/duration,
                    ttft_p50_ms=float(np.percentile([r['ttft_s'] for r in samples],50))*1000,ttft_p95_ms=float(np.percentile([r['ttft_s'] for r in samples],95))*1000,
                    itl_p50_ms=float(np.percentile(itl,50))*1000,itl_p95_ms=float(np.percentile(itl,95))*1000))
                records.append(dict(case=rows[-1]['case'],duration_s=duration,requests=samples))
                write_json(out/'latencies.json',rows);write_json(out/'requests.json',records)
        stats=await engine.collective_rpc('projection_stats' if args.policy=='legacy' else 'v2_stats');write_json(out/'preparation.json',stats)
    finally:engine.shutdown()
    write_json(out/'completed.json',dict(rows=len(rows)))


def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=0)
 p.add_argument('--policy',default='original');p.add_argument('--pretrained',action='store_true');p.add_argument('--case');p.add_argument('--requests',type=int,default=128);p.add_argument('--prior')
 args=p.parse_args();asyncio.run(run(args))
if __name__=='__main__':main()

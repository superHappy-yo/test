"""vLLM V1 engine benchmark with a small random-weight Qwen3 architecture."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import types

from shapefuse.io_utils import write_csv, write_json
from shapefuse.model_system import MODEL_CONFIG, ProjectionSite, register_operator


def project_vllm_strict(x, weight, residual, norm_weight):
    import torch
    product=torch.ops.shapefuse.linear_half(x,weight)
    values=product.float()+residual.float()
    y=values*torch.rsqrt(values.square().mean(-1,keepdim=True)+1e-6)*norm_weight.float()
    return y.to(product.dtype),values.to(product.dtype)


# Lazy import preserves the optional nature of the connector in the core CLI.
try:
    from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM as _Base
except ImportError:
    _Base=object


class ScicQwen3ForCausalLM(_Base):
    def __init__(self, *, vllm_config, prefix=""):
        import torch
        super().__init__(vllm_config=vllm_config,prefix=prefix)
        policy=os.environ.get("SHAPEFUSE_VLLM_POLICY","original")
        if policy=="original":
            return
        if vllm_config.parallel_config.tensor_parallel_size!=1 or vllm_config.quant_config is not None:
            raise ValueError("SCIC adapter requires TP=1, unquantized Qwen3")
        register_operator()
        for layer in self.model.layers:
            original_project=layer.self_attn.o_proj.forward
            norm=layer.post_attention_layernorm
            def native(x,w,r,nw,project=original_project,norm=norm):
                product,_=project(x)
                # Native fused norm mutates its buffers; scratch residual avoids
                # input aliasing and probe contamination. Cost remains timed.
                return norm(product,r.clone())
            site=ProjectionSite(policy,native=native,compile_fn=project_vllm_strict,identity="vllm")
            layer._sf_site=site
            layer.self_attn.o_proj.forward=types.MethodType(lambda self,x:(x,None),layer.self_attn.o_proj)
            def forward(self,positions,hidden_states,residual):
                if residual is None:
                    residual=hidden_states
                    hidden_states=self.input_layernorm(hidden_states)
                else:
                    hidden_states,residual=self.input_layernorm(hidden_states,residual)
                raw=self.self_attn(positions=positions,hidden_states=hidden_states)
                normalized,residual=torch.ops.shapefuse.project_add_norm(
                    raw,self.self_attn.o_proj.weight,residual,self.post_attention_layernorm.weight,
                    self._sf_site.site_id)
                return self.mlp(normalized),residual
            layer.forward=types.MethodType(forward,layer)


def _projection_report(model):
    model=getattr(model,"_orig_mod",model)
    sites=[(i,layer._sf_site) for i,layer in enumerate(model.model.layers)
           if hasattr(layer,"_sf_site")]
    return dict(parameters=sum(p.numel() for p in model.parameters()),
                prepared=[dict(layer=i,**record) for i,site in sites for record in site.prepared],
                calibration_ms=sum(record["calibration_ms"] for _,site in sites for record in site.prepared),
                cache_hits=sum(site.executor.hits for _,site in sites),
                cache_misses=sum(site.executor.misses for _,site in sites))


class ProjectionStatsWorkerExtension:
    """Named, read-only RPC; does not require serializing executable functions."""
    def projection_stats(self):
        return _projection_report(self.get_model())


async def _one_policy(args,policy,out):
    from vllm import SamplingParams,ModelRegistry
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from shapefuse.supplement import environment
    os.environ["SHAPEFUSE_VLLM_POLICY"]=policy
    ModelRegistry.register_model("ScicQwen3ForCausalLM","shapefuse.vllm_system:ScicQwen3ForCausalLM")
    model_dir=out/"model"
    model_dir.mkdir()
    config=dict(MODEL_CONFIG,architectures=["ScicQwen3ForCausalLM"])
    write_json(model_dir/"config.json",config)
    start=time.perf_counter()
    engine=AsyncLLM.from_engine_args(AsyncEngineArgs(
        model=str(model_dir),load_format="dummy",skip_tokenizer_init=True,dtype="half",
        seed=args.seed,tensor_parallel_size=1,max_model_len=4096,max_num_seqs=16,
        max_num_batched_tokens=4096,gpu_memory_utilization=0.5,
        worker_extension_cls="shapefuse.vllm_system.ProjectionStatsWorkerExtension",
        enable_prefix_caching=True,enable_chunked_prefill=True))
    setup_ms=(time.perf_counter()-start)*1000
    write_json(out/"manifest.json",environment()|{"model":config,"policy":policy,
               "pretrained":False,"runtime":"vLLM V1 AsyncLLM, token-ID inputs",
               "setup_ms":setup_ms,"prefix_caching":True,"chunked_prefill":True,
               "compile_and_cudagraphs":"framework defaults; identical configurations",
               "weights":"vLLM seeded dummy initialization"})
    rows,raw=[],[]
    if args.requests < 1:
        raise ValueError("--requests must be positive")
    params=SamplingParams(temperature=0,max_tokens=64,ignore_eos=True,detokenize=False)
    async def request(i,length):
        # Unique, predetermined token prefixes avoid cross-request cache reuse.
        import random
        rng=random.Random(args.seed*1000000+i)
        tokens=[rng.randrange(3,8192) for _ in range(length)]
        begin=time.perf_counter()
        arrivals=[]
        generated=[]
        previous=0
        async for result in engine.generate({"prompt_token_ids":tokens},params,request_id=str(i)):
            if result.outputs:
                ids=list(result.outputs[0].token_ids)
                if len(ids)>previous:
                    arrivals.extend([time.perf_counter()-begin]*(len(ids)-previous))
                    generated=ids
                    previous=len(ids)
        if len(generated)!=64:
            raise RuntimeError(f"Expected 64 tokens, got {len(generated)}")
        return dict(request=i,length=length,latency_s=time.perf_counter()-begin,
                    ttft_s=arrivals[0],itl_s=[b-a for a,b in zip(arrivals,arrivals[1:])],
                    output_sha256=hashlib.sha256(json.dumps(generated).encode()).hexdigest(),
                    output_tokens=len(generated))
    try:
        serial=0
        for length in (128,2048):
            for concurrency in (1,8,16):
                label=f"s{length}_c{concurrency}"
                if args.case and args.case!=label:
                    continue
                for _ in range(2):
                    await asyncio.gather(*(request(serial+j,length) for j in range(concurrency)))
                    serial+=concurrency
                samples=[]
                begin=time.perf_counter()
                # Fixed batches; a partial last batch is explicitly recorded.
                for offset in range(0,args.requests,concurrency):
                    batch=min(concurrency,args.requests-offset)
                    samples.extend(await asyncio.gather(*(
                        request(serial+j,length) for j in range(batch))))
                    serial+=batch
                duration=time.perf_counter()-begin
                import numpy as np
                itl=[v for s in samples for v in s["itl_s"]]
                row=dict(case=label,policy=policy,seed=args.seed,concurrency=concurrency,
                         input_length=length,output_length=64,requests=len(samples),
                         output_tokens_per_second=sum(s["output_tokens"] for s in samples)/duration,
                         ttft_p50_ms=float(np.percentile([s["ttft_s"] for s in samples],50))*1000,
                         ttft_p95_ms=float(np.percentile([s["ttft_s"] for s in samples],95))*1000,
                         itl_p50_ms=float(np.percentile(itl,50))*1000,
                         itl_p95_ms=float(np.percentile(itl,95))*1000,
                         latency_p50_ms=float(np.percentile([s["latency_s"] for s in samples],50))*1000,
                         latency_p95_ms=float(np.percentile([s["latency_s"] for s in samples],95))*1000,
                         execution_status="ok")
                rows.append(row)
                raw.append(dict(case=label,requests=samples,duration_s=duration))
                write_csv(out/"latency.csv",rows)
                write_json(out/"details.json",raw)
        try:
            reports=await engine.collective_rpc("projection_stats")
            write_json(out/"projection_report.json",reports)
        except Exception as exc:
            write_json(out/"projection_report.json",dict(status="unavailable",error=str(exc)))
    finally:
        engine.shutdown()
    write_json(out/"completed.json",{"rows":len(rows)})


def run_vllm(args):
    if args.requests < 1:
        raise ValueError("--requests must be positive")
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    policies=args.methods.split(",") if args.methods else ["original"]
    if len(policies)!=1:
        raise ValueError("Run vLLM policies in separate worker processes")
    if policies[0] not in ("original","fixed_native","fixed_compile","scic"):
        raise ValueError(f"Unknown vLLM policy: {policies[0]}")
    asyncio.run(_one_policy(args,policies[0],out))
    return 0

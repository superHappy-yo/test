"""Explicit preparation and compiler-visible, frozen deployment endpoints."""
import time
import torch
from shapefuse.v2_kernels import NAMES,candidate,reference
from shapefuse.v2_policy import PreparedPlan,signature,choose
from shapefuse.v2_experiment import check,timing


class Preparer:
    def __init__(self,policy,prior=None,static='blas4',*,semantic='hf',epsilon=1e-6,expected_calls=1024):
        self.policy=policy;self.prior=prior;self.static=static;self.semantic=semantic
        self.epsilon=epsilon;self.expected_calls=expected_calls;self.cache={};self.records=[]
    def prepare(self,inputs,*,mode='direct',reuse='same',weight_version=0):
        key=signature(inputs,semantic=self.semantic,epsilon=self.epsilon,weight_version=weight_version)
        key+=(mode,reuse,self.policy,self.expected_calls)
        if key in self.cache:return self.cache[key]
        if torch.cuda.is_current_stream_capturing():raise RuntimeError('Prepare before graph capture')
        x,w,r,g=inputs
        contiguous=all(t.is_contiguous() and t.dtype==torch.float16 for t in inputs)
        valid={};audits={};compile_ms=0.;validate_ms=0.;probe={};cost=0.
        shape=(x.numel()//x.shape[-1],w.shape[0],w.shape[1])
        pred=self.prior.predict(shape,mode,reuse) if self.prior else {}
        static=self.static
        prep_costs={}
        def validate(name):
            nonlocal compile_ms,validate_ms
            if name in audits:return
            raw=candidate(name,vllm=self.semantic=='vllm',eps=self.epsilon)
            compiled=torch.compile(raw,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False,'emulate_precision_casts':True})
            start=time.perf_counter()
            try:
                compiled(*inputs);torch.cuda.synchronize()
                compilation=(time.perf_counter()-start)*1000;compile_ms+=compilation
                start=time.perf_counter()
                checks=[check(compiled(*inputs),reference(*inputs,vllm=self.semantic=='vllm',eps=self.epsilon))]
                for seed in (11,23,37):
                    generator=torch.Generator(device=x.device).manual_seed(seed)
                    for stress,unit in ((False,True),(False,False),(True,False)):
                        xx=torch.randn(x.shape,generator=generator,device=x.device,dtype=x.dtype)*(4 if stress else 1)
                        rr=torch.randn(r.shape,generator=generator,device=r.device,dtype=r.dtype)*(.1 if stress else 1)
                        gg=torch.ones_like(g) if unit else g
                        test=(xx,w,rr,gg)
                        checks.append(check(compiled(*test),reference(*test,vllm=self.semantic=='vllm',eps=self.epsilon)))
                result={'passed':all(v['passed'] for v in checks),'checks':checks}
                torch.cuda.synchronize();validation=(time.perf_counter()-start)*1000;validate_ms+=validation
                prep_costs[name]=compilation+validation;audits[name]=result
                if result['passed']:valid[name]=(raw,compiled)
            except Exception as exc:
                elapsed=(time.perf_counter()-start)*1000;compile_ms+=elapsed;prep_costs[name]=elapsed
                audits[name]=dict(passed=False,error=repr(exc))
        def sample(name):
            nonlocal cost
            validate(name)
            if name in valid and name not in probe:
                start=time.perf_counter()
                measured=timing(lambda:valid[name][1](*inputs),warm=3,samples=0,batch_size=8)
                elapsed=(time.perf_counter()-start)*1000
                probe[name]=measured['host_ms'];cost+=elapsed
        fallback=''
        if contiguous:
            if self.policy=='best-fixed':validate(static)
            elif self.policy=='full-calibration':
                for name in NAMES:sample(name)
            elif self.policy.startswith('scic-v2'):
                sample(static)
                challenger=min(NAMES,key=lambda n:pred.get(n,{'ms':float('inf')})['ms'])
                sample(challenger)
                if not probe or any(n not in audits and pred.get(n,{'lower':0})['lower']<.98*min(probe.values()) for n in NAMES):
                    for name in NAMES:sample(name)
            else:
                for name in NAMES:sample(name)
        if not valid:
            selected='native';raw=candidate(selected,vllm=self.semantic=='vllm',eps=self.epsilon)
            fallback='no-valid-candidate' if contiguous else 'unsupported-layout'
        else:
            if static not in valid:static=next(iter(valid))
            selected=min(probe,key=probe.get) if probe else static
            # Include incremental compilation and validation, not just timers.
            incremental=cost+sum(v for n,v in prep_costs.items() if n!=static)
            if self.policy.startswith('scic-v2') and self.expected_calls*(probe.get(static,float('inf'))-probe.get(selected,float('inf')))<=incremental:
                selected=static;fallback='not-amortized'
            raw=valid[selected][0]
        plan=PreparedPlan(selected,raw,key,probe,pred,cost,validate_ms,compile_ms,fallback)
        self.cache[key]=plan
        self.records.append(dict(M=x.numel()//x.shape[-1],N=w.shape[0],K=w.shape[1],selected=selected,
            probe=probe,predicted=pred,probe_ms=cost,validation_ms=validate_ms,compilation_ms=compile_ms,
            preparation_ms=cost+validate_ms+compile_ms,audit=audits,fallback=fallback))
        return plan


def adapt_hf(model,policy,*,batch,prompt,prior=None,static='blas4'):
    """Modern Transformers Qwen3: decisions frozen before outer compilation.

    Traced shape branches choose two prepared endpoints; generated code contains
    neither an exact-key dispatcher nor nested compiled projection functions.
    """
    preparers=[]
    class Layer(torch.nn.Module):
        def __init__(self,layer):
            super().__init__()
            self.self_attn=layer.self_attn;self.mlp=layer.mlp
            self.input_layernorm=layer.input_layernorm;self.post_attention_layernorm=layer.post_attention_layernorm
            self.projection=self.self_attn.o_proj;self.self_attn.o_proj=torch.nn.Identity()
            self.preparer=Preparer(policy,prior,static,epsilon=self.post_attention_layernorm.variance_epsilon)
            self.legacy_site=None
            if policy=='legacy':
                from shapefuse.model_system import ProjectionSite,register_operator
                register_operator();self.legacy_site=ProjectionSite('scic')
            preparers.append(self.preparer)
            for phase,m in (('prefill',batch*prompt),('decode',batch)):
                w=self.projection.weight;g=self.post_attention_layernorm.weight
                # Calibrate loaded weights, representative unit-scale activations.
                inp=(torch.randn(m,w.shape[1],device=w.device,dtype=w.dtype),w,
                     torch.randn(m,w.shape[0],device=w.device,dtype=w.dtype),g)
                if self.legacy_site is None:
                    setattr(self,phase+'_fn',self.preparer.prepare(inp).fn)
                else:
                    self.legacy_site.invoke(*inp)
                    self.preparer.records=self.legacy_site.prepared
            self.decode_m=batch
        def forward(self,hidden_states,attention_mask=None,position_ids=None,past_key_values=None,
                    use_cache=False,position_embeddings=None,**kwargs):
            residual=hidden_states
            hidden_states=self.input_layernorm(hidden_states)
            raw,_=self.self_attn(hidden_states=hidden_states,attention_mask=attention_mask,
                position_ids=position_ids,past_key_values=past_key_values,use_cache=use_cache,
                position_embeddings=position_embeddings,**kwargs)
            shape=raw.shape
            flat=raw.reshape(-1,shape[-1]);rr=residual.reshape(-1,residual.shape[-1])
            if self.legacy_site is not None:
                normalized,residual=torch.ops.shapefuse.project_add_norm(flat,self.projection.weight,rr,self.post_attention_layernorm.weight,self.legacy_site.site_id)
            else:
                fn=self.decode_fn if raw.numel()//shape[-1]==self.decode_m else self.prefill_fn
                normalized,residual=fn(flat,self.projection.weight,rr,self.post_attention_layernorm.weight)
            normalized=normalized.reshape(residual.shape[0],-1) if len(shape)==2 else normalized.reshape(*shape[:-1],self.projection.weight.shape[0])
            residual=residual.reshape_as(normalized)
            return residual+self.mlp(normalized)
    model.model.layers=torch.nn.ModuleList([Layer(layer) for layer in model.model.layers])
    return model,preparers

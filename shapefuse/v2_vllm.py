"""vLLM opt-in compiler-visible integration, prepared after checkpoint loading."""
import os,types
import torch
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM as Base
from vllm.v1.worker.gpu_worker import Worker
from shapefuse.v2_kernels import serving_norm
from shapefuse.v2_runtime import Preparer


class V2Qwen3ForCausalLM(Base):
    def __init__(self,*,vllm_config,prefix=''):
        super().__init__(vllm_config=vllm_config,prefix=prefix)
        self._v2_policy=os.environ.get('SHAPEFUSE_V2_POLICY','original')
        self._v2_prepared=[]
        if self._v2_policy=='original':return
        if vllm_config.parallel_config.tensor_parallel_size!=1 or vllm_config.quant_config is not None:
            raise ValueError('SCIC-v2 supports FP16 TP1 only')
        for layer in self.model.layers:
            layer._v2_original_o_proj_forward=layer.self_attn.o_proj.forward
            layer._v2_original_forward=layer.forward
            layer.self_attn.o_proj.forward=types.MethodType(lambda self,x:(x,None),layer.self_attn.o_proj)
            layer._v2_config=(0,4)
            def forward(self,positions,hidden_states,residual):
                if residual is None:
                    residual=hidden_states;hidden_states=self.input_layernorm(hidden_states)
                else:hidden_states,residual=self.input_layernorm(hidden_states,residual)
                raw=self.self_attn(positions=positions,hidden_states=hidden_states)
                config=self._v2_config
                y=serving_norm(raw,self.self_attn.o_proj.weight,residual,self.post_attention_layernorm.weight,
                               config[0],config[1],self.post_attention_layernorm.variance_epsilon)
                return self.mlp(y),residual
            layer.forward=types.MethodType(forward,layer)
    def prepare_v2(self):
        if self._v2_policy=='original':return
        from shapefuse.v2_system import load_prior,fixed_from_artifact
        policy=self._v2_policy
        # Serving semantic training artifact must be explicitly supplied.
        priorpath=os.environ.get('SHAPEFUSE_V2_SERVING_PRIOR')
        prior=load_prior(priorpath,policy) if priorpath and policy.startswith('scic-v2') else None
        def native_fallback(layer,index,records,reason):
            # Restore the framework layer before tracing; no hot-path branch.
            layer.self_attn.o_proj.forward=layer._v2_original_o_proj_forward
            layer.forward=layer._v2_original_forward
            self._v2_prepared.extend(dict(layer=index,deployed='native',deployment_fallback=reason,
                mixture_decode_prefill=[64,1],**r) for r in records)
        for i,layer in enumerate(self.model.layers):
            static=fixed_from_artifact(priorpath) if priorpath else 'blas4'
            prep=Preparer(policy,prior,static,semantic='vllm',epsilon=layer.post_attention_layernorm.variance_epsilon)
            w=layer.self_attn.o_proj.weight;g=layer.post_attention_layernorm.weight
            plans=[]
            for phase,m in (('decode',8),('prefill',2048)):
                ins=(torch.randn(m,w.shape[1],device=w.device,dtype=w.dtype),w,
                     torch.randn(m,w.shape[0],device=w.device,dtype=w.dtype),g)
                plan=prep.prepare(ins)
                if plan.candidate=='native':
                    native_fallback(layer,i,prep.records,f'no-valid-candidate-{phase}')
                    break
                split=int(plan.candidate[5:]) if plan.candidate.startswith('split') else 0
                warps=int(plan.candidate[4:]) if plan.candidate.startswith('blas') else 4
                plans.append(plan)
            if len(plans)!=2:continue
            # A marked-unbounded dynamic token dimension cannot be guarded by M<=16.
            # Freeze one candidate per layer using a declared 64-decode/one-prefill mixture.
            common=set(plans[0].probe)&set(plans[1].probe)
            selected=min(common,key=lambda n:64*plans[0].probe[n]+plans[1].probe[n]) if common else plans[0].candidate
            if policy.startswith('scic-v2') and static in common:
                saving=((64*plans[0].probe[static]+plans[1].probe[static])-(64*plans[0].probe[selected]+plans[1].probe[selected]))/65
                total=sum(p.probe_ms+p.validation_ms+p.compilation_ms for p in plans)
                if 1024*saving<=total:selected=static
            from shapefuse.v2_kernels import candidate,reference
            from shapefuse.v2_experiment import check
            joint_valid=True
            for m in (8,2048):
                ins=(torch.randn(m,w.shape[1],device=w.device,dtype=w.dtype),w,torch.randn(m,w.shape[0],device=w.device,dtype=w.dtype),g)
                if not check(candidate(selected,vllm=True)(*ins),reference(*ins,vllm=True))['passed']:
                    joint_valid=False
                    break
            if not joint_valid:
                native_fallback(layer,i,prep.records,'joint-candidate-validation-failed')
                continue
            split=int(selected[5:]) if selected.startswith('split') else 0
            warps=int(selected[4:]) if selected.startswith('blas') else 4
            layer._v2_config=(split,warps)
            self._v2_prepared.extend(dict(layer=i,deployed=selected,mixture_decode_prefill=[64,1],**r) for r in prep.records)


class V2Worker(Worker):
    def load_model(self,*,load_dummy_weights=False):
        super().load_model(load_dummy_weights=load_dummy_weights)
        model=self.model_runner.get_model()
        if hasattr(model,'prepare_v2'):
            with torch.inference_mode():model.prepare_v2()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


class V2Stats:
    def v2_stats(self):
        model=self.get_model()
        return dict(prepared=model._v2_prepared,policy=model._v2_policy,
                    coverage='one frozen candidate per layer, unconstrained dynamic token count; 64:1 decode/prefill calibration mixture')

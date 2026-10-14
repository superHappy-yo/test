"""A rejected candidate restores framework execution before compilation."""
import importlib.util
import types
import unittest
from unittest.mock import patch

@unittest.skipUnless(importlib.util.find_spec('vllm'),'vLLM environment required')
class ServingFallbackTests(unittest.TestCase):
 def fixture(self):
  import torch
  from shapefuse.v2_vllm import V2Qwen3ForCausalLM
  model=V2Qwen3ForCausalLM.__new__(V2Qwen3ForCausalLM)
  torch.nn.Module.__init__(model)
  layer=torch.nn.Module();layer.self_attn=torch.nn.Module()
  layer.self_attn.o_proj=torch.nn.Linear(4,4,bias=False,dtype=torch.float16)
  layer.post_attention_layernorm=torch.nn.Module()
  layer.post_attention_layernorm.weight=torch.nn.Parameter(torch.ones(4,dtype=torch.float16))
  layer.post_attention_layernorm.variance_epsilon=1e-6
  layer._v2_original_o_proj_forward=layer.self_attn.o_proj.forward
  layer._v2_original_forward=layer.forward
  layer.self_attn.o_proj.forward=types.MethodType(lambda self,x:(x,None),layer.self_attn.o_proj)
  layer.forward=types.MethodType(lambda self,*args:None,layer)
  model.model=torch.nn.Module();model.model.layers=torch.nn.ModuleList([layer])
  model._v2_policy='scic-v2';model._v2_prepared=[]
  return model,layer
 def test_no_admitted_candidate_restores_native_layer(self):
  model,layer=self.fixture()
  prep=types.SimpleNamespace(records=[{'selected':'native','fallback':'no-valid-candidate'}],prepare=lambda _:types.SimpleNamespace(candidate='native'))
  with patch.dict('os.environ',{},clear=True),patch('shapefuse.v2_vllm.Preparer',return_value=prep):model.prepare_v2()
  self.assertEqual(layer.forward,layer._v2_original_forward)
  self.assertEqual(layer.self_attn.o_proj.forward,layer._v2_original_o_proj_forward)
  self.assertEqual(model._v2_prepared[0]['deployment_fallback'],'no-valid-candidate-decode')
  self.assertEqual(model._v2_prepared[0]['deployed'],'native')
 def test_joint_validation_failure_restores_native_layer(self):
  model,layer=self.fixture()
  plan=types.SimpleNamespace(candidate='blas4',probe={'blas4':.1},probe_ms=0,validation_ms=0,compilation_ms=0)
  prep=types.SimpleNamespace(records=[{'selected':'blas4'}],prepare=lambda _:plan)
  with patch.dict('os.environ',{},clear=True),patch('shapefuse.v2_vllm.Preparer',return_value=prep),patch('shapefuse.v2_kernels.candidate',return_value=lambda *args:None),patch('shapefuse.v2_kernels.reference',return_value=None),patch('shapefuse.v2_experiment.check',return_value={'passed':False}):model.prepare_v2()
  self.assertEqual(layer.forward,layer._v2_original_forward)
  self.assertEqual(layer.self_attn.o_proj.forward,layer._v2_original_o_proj_forward)
  self.assertEqual(model._v2_prepared[0]['deployment_fallback'],'joint-candidate-validation-failed')

if __name__=='__main__':unittest.main()

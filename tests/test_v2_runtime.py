import unittest
from unittest.mock import patch
try:
 import torch
 from shapefuse.v2_kernels import blas_norm,split_norm,serving_norm,reference
 from shapefuse.v2_runtime import Preparer
 from shapefuse.v2_policy import signature
except ImportError:
 torch=None

@unittest.skipUnless(torch is not None and torch.cuda.is_available(),'CUDA/Triton required')
class RuntimeTests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(1)
  self.inputs=(torch.randn(2,128,device='cuda',dtype=torch.float16),torch.randn(128,128,device='cuda',dtype=torch.float16)*.02,
               torch.randn(2,128,device='cuda',dtype=torch.float16),torch.ones(128,device='cuda',dtype=torch.float16))
 def test_schema_fake_aot(self):
  for op,args in [(blas_norm,(*self.inputs,4,False,1e-6)),(split_norm,(*self.inputs,4,False,1e-6)),(serving_norm,(*self.inputs,0,4,1e-6))]:
   self.assertTrue(all(v=='SUCCESS' for v in torch.library.opcheck(op,args).values()))
 def test_mutating_serving_semantics(self):
  x,w,r,g=self.inputs
  expected=reference(x,w,r,g,vllm=True)
  actual=serving_norm(x,w,r,g,0,4,1e-6)
  self.assertTrue(torch.allclose(actual,expected[0],atol=1e-3,rtol=1e-3))
  self.assertTrue(torch.equal(r,expected[1]))
  self.assertNotEqual(actual.data_ptr(),r.data_ptr())
 def test_dynamic_token_dimension(self):
  x,w,r,g=self.inputs
  fn=torch.compile(lambda x,r:serving_norm(x,w,r,g,0,4,1e-6),fullgraph=True,dynamic=True,options={'triton.cudagraphs':False})
  for m in (8,128):
   xx=torch.randn(m,128,device='cuda',dtype=torch.float16);rr=xx.clone()
   torch._dynamo.mark_dynamic(xx,0);torch._dynamo.mark_dynamic(rr,0)
   with torch.inference_mode():yy=fn(xx,rr)
   self.assertEqual(tuple(yy.shape),(m,128))
 def test_layout_and_weight_version_invalidate(self):
  key=signature(self.inputs)
  self.inputs[1].add_(.01)
  self.assertNotEqual(key,signature(self.inputs))
  self.assertNotEqual(signature(self.inputs),signature(self.inputs,epsilon=1e-5))
  self.assertNotEqual(signature(self.inputs),signature(self.inputs,semantic='vllm'))
 def test_unprepared_capture_is_rejected(self):
  with patch('torch.cuda.is_current_stream_capturing',return_value=True):
   with self.assertRaisesRegex(RuntimeError,'Prepare before graph capture'):
    Preparer('best-fixed').prepare(self.inputs)
 def test_layout_fallback_and_graph_capture(self):
  x,w,r,g=self.inputs
  bad=(x[:,::2],w[:,::2],r,g)
  with torch.inference_mode():
   plan=Preparer('best-fixed').prepare(bad)
   self.assertEqual(plan.fallback,'unsupported-layout')
   plan=Preparer('best-fixed').prepare(self.inputs)
   plan.fn(*self.inputs)
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):actual=plan.fn(*self.inputs)
   graph.replay()
   self.assertTrue(torch.allclose(actual[0],reference(*self.inputs)[0],atol=1e-3,rtol=1e-3))

if __name__=='__main__':unittest.main()

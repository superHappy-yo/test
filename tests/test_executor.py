import unittest
from unittest.mock import patch

from shapefuse.executor import PlanExecutor, calibration_applies, exact_key
from shapefuse.machines import get_machine
from shapefuse.scic import Site, classify_site


class ExecutorTests(unittest.TestCase):
    def test_explicit_custom_operator_schema_registers_and_executes_on_cpu(self):
        import torch
        from shapefuse.model_system import register_operator
        register_operator()
        x=torch.arange(6,dtype=torch.float32).reshape(2,3)
        weight=torch.ones((4,3))
        self.assertTrue(torch.equal(torch.ops.shapefuse.linear_half(x,weight),
                                    torch.nn.functional.linear(x,weight)))

    def test_scope_boundary_uses_actual_regime_name(self):
        self.assertTrue(calibration_applies("ridge-region", "ridge-only"))
        self.assertFalse(calibration_applies("compute-bound", "ridge-only"))
        self.assertTrue(calibration_applies("compute-bound", "ridge-compute"))
        self.assertFalse(calibration_applies("memory-bound", "ridge-compute"))
        with self.assertRaises(ValueError):
            calibration_applies("ridge-region", "typo")

    def test_cached_policy_really_invokes_selected_path_and_rejects_unprepared_shape(self):
        import torch
        calls = []
        def native(x):
            calls.append("native")
            return x + 1
        def compile_path(x):
            calls.append("compile")
            return x + 1
        ex = PlanExecutor({"native":native, "compile_default":compile_path}, scope="off")
        x = torch.zeros(4, 64)
        report = classify_site(Site("gemm", "test", {"M":4,"N":64,"K":64}), get_machine("A40"))
        ex.prepare(report, (x,))
        self.assertEqual(calls, [])
        self.assertTrue(torch.equal(ex(x), x+1))
        self.assertEqual(calls, ["compile"])
        with self.assertRaises(RuntimeError):
            ex(torch.zeros(8,64))  # Both row counts share the old coarse bucket.
        self.assertNotEqual(exact_key((x,), "a", "gpu"), exact_key((x.T,), "a", "gpu"))
        self.assertNotEqual(exact_key((x,), "a", "gpu"), exact_key((x.double(),), "a", "gpu"))
        self.assertNotEqual(exact_key((x,), "a", "gpu"), exact_key((x,), "b", "gpu"))

    def test_independent_probe_is_cached_without_copying_latency(self):
        import torch
        fn = lambda x: x+1
        ex = PlanExecutor({"native":fn, "compile_default":fn}, scope="all-gemm")
        x = torch.zeros(4,64)
        report = classify_site(Site("gemm","test",dict(M=4,N=64,K=64)),get_machine("A40"))
        with patch("shapefuse.executor.measure",side_effect=[{"gpu_median_ms":1},{"gpu_median_ms":2}]) as m:
            entry = ex.prepare(report,(x,))
            self.assertEqual(entry.candidate,"compile_default")
            self.assertIs(ex.prepare(report,(x,)),entry)
            self.assertEqual(m.call_count,2)
        self.assertTrue(torch.equal(ex(x),x+1))

    def test_shape_matrix_and_complete_residual_chain(self):
        import torch
        from shapefuse.supplement import operator_cases
        from shapefuse.model_system import project_add_norm
        self.assertEqual(sum(c["op"]=="gemm" for c in operator_cases()),21)
        self.assertEqual(sum(c["op"]=="attention" for c in operator_cases()),9)
        x=torch.full((2,16),20,dtype=torch.float16)
        w=torch.ones((8,16),dtype=torch.float16)
        r=torch.ones((2,8),dtype=torch.float16)
        before=r.clone()
        y,summed=project_add_norm(x,w,r,torch.ones(8,dtype=torch.float16))
        self.assertTrue(torch.equal(before,r))
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(torch.equal(summed,torch.full_like(summed,321)))
        self.assertTrue(torch.allclose(y,torch.ones_like(y)))

if __name__=="__main__":
    unittest.main()

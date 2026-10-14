import unittest
from types import SimpleNamespace
from unittest.mock import patch

from shapefuse.ablation import BoundEndpoint, LegacyLookupSite, SingleLookupSite, output_audit
from shapefuse.executor import PlanExecutor, PreparedPath, exact_key


class AblationTests(unittest.TestCase):
    def test_general_adapter_checks_signature_once_and_invokes_cached_path_once(self):
        import torch
        from shapefuse import model_system as ms
        x = torch.zeros(2,4)
        site = ms.ProjectionSite.__new__(ms.ProjectionSite)
        site.executor = PlanExecutor({})
        key = exact_key((x,),"","")
        calls = []
        site.executor.cache[key] = PreparedPath("native",lambda a: calls.append(a) or a+1,
                                               0,{},"native","memory-bound","")
        with patch("shapefuse.model_system.exact_key",wraps=exact_key) as signature:
            self.assertTrue(torch.equal(site.invoke(x),x+1))
            self.assertEqual(signature.call_count,1)
        self.assertEqual(len(calls),1)
        self.assertEqual(site.executor.hits,1)
        # A changed layout still reaches the unprepared-capture rejection;
        # removing the second lookup must not reuse an incompatible entry.
        with patch("torch.cuda.is_current_stream_capturing",return_value=True):
            with self.assertRaises(RuntimeError):
                site.invoke(x.T)
            with self.assertRaises(RuntimeError):
                site.invoke(x.double())

    def test_legacy_control_still_constructs_two_signatures(self):
        import torch
        x = torch.zeros(2,4)
        site = LegacyLookupSite.__new__(LegacyLookupSite)
        site.executor = PlanExecutor({})
        key = exact_key((x,),"","")
        site.executor.cache[key] = PreparedPath("native",lambda a: a+1,0,{},"native","memory-bound","")
        with patch("shapefuse.ablation.exact_key",wraps=exact_key) as first, \
             patch("shapefuse.executor.exact_key",wraps=exact_key) as second:
            self.assertTrue(torch.equal(site.invoke(x),x+1))
            self.assertEqual(first.call_count,1)
            self.assertEqual(second.call_count,1)

    def test_single_lookup_invokes_cached_function_once(self):
        import torch
        x = torch.zeros(2, 4)
        calls = []
        site = SingleLookupSite.__new__(SingleLookupSite)
        site.executor = PlanExecutor({})
        key = exact_key((x,), "", "")
        site.executor.cache[key] = PreparedPath("native", lambda a: calls.append(a) or a+1,
                                               0, {}, "native", "memory-bound", "")
        with patch("shapefuse.ablation.exact_key", wraps=exact_key) as signature:
            self.assertTrue(torch.equal(site.invoke(x), x+1))
            self.assertEqual(signature.call_count, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(site.executor.hits, 1)

    def test_bound_phase_rejects_layout_change_before_freezing(self):
        import torch
        x = torch.zeros(2, 4)
        executor = PlanExecutor({})
        key = exact_key((x,), "", "")
        executor.cache[key] = PreparedPath("native", lambda a: a+1, 0, {}, "native", "memory-bound", "")
        endpoint = BoundEndpoint.__new__(BoundEndpoint)
        endpoint.site = SimpleNamespace(executor=executor, invoke=lambda a: executor(a))
        endpoint.entry = endpoint.signature = None
        endpoint.frozen = False
        endpoint.calls = 0
        endpoint.forced_candidate = None
        with self.assertRaises(RuntimeError):
            endpoint.freeze()
        endpoint.invoke(x)
        with self.assertRaises(ValueError):
            endpoint.invoke(x.T)
        endpoint.freeze()
        with patch("shapefuse.ablation.exact_key", side_effect=AssertionError("hot path inspected signature")):
            self.assertTrue(torch.equal(endpoint.invoke(x), x+1))
        self.assertEqual(endpoint.calls, 1)

    def test_full_history_gate_includes_last_decode_output_and_finite_values(self):
        import torch
        ref = [torch.ones(2, 3), torch.ones(2, 3)]
        self.assertEqual(output_audit(ref, ref), 0)
        bad = [ref[0], ref[1]+1]
        with self.assertRaises(ArithmeticError):
            output_audit(bad, ref)
        with self.assertRaises(ArithmeticError):
            output_audit([ref[0]], ref)
        with self.assertRaises(ArithmeticError):
            output_audit([ref[0], torch.full((2,3), float("nan"))], ref)


if __name__ == "__main__":
    unittest.main()

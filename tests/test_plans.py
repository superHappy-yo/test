"""Selector policy: regime switch, attention scores, ragged MoE."""

from __future__ import annotations

import unittest

from shapefuse.machines import H100_SXM_FP16
from shapefuse.plans import (
    POLICY_COMPILE_SDPA_CHAIN,
    PlanCache,
    moe_is_heterogeneous,
    select_plan,
    shape_bucket,
)
from shapefuse.scic import Site, classify_site
from shapefuse.sweeps import build_sweeps


class PlanTests(unittest.TestCase):
    def test_buckets(self) -> None:
        self.assertEqual(shape_bucket(1), "1")
        self.assertEqual(shape_bucket(8), "2-8")
        self.assertEqual(shape_bucket(9), "9-32")
        self.assertEqual(shape_bucket(9000), "8193+")

    def test_cache_records_measured_winner(self) -> None:
        cache = PlanCache()
        self.assertIsNone(cache.lookup("k"))
        cache.record_measured_winner("k", "MemMode", "latency_s", 1.2e-5)
        self.assertEqual(cache.lookup("k")["mode"], "MemMode")
        self.assertEqual(cache.lookup("k")["source"], "measured")
        cache.record_measured_winner("gemm|m2048", "always_fuse", "wallclock_median_ms", 0.7)
        stored = cache.lookup("gemm|m2048")
        self.assertEqual(stored["mode"], "always_fuse")
        self.assertEqual(stored["source"], "measured_wallclock")
        self.assertEqual(stored["metric"], "wallclock_median_ms")

    def test_no_kernel_and_no_measured_winner_by_default(self) -> None:
        bundle = build_sweeps(H100_SXM_FP16)
        self.assertGreater(len(bundle.plans), 10)
        for row in bundle.plans:
            self.assertFalse(row["kernel_emitted"])
            self.assertEqual(row["measured_winner"], "")
            self.assertIn("Stage-Stream is not", row["rationale"] + row["plan_note"])

    def test_gemm_switch(self) -> None:
        bundle = build_sweeps(H100_SXM_FP16)
        by_label = {row["label"]: row for row in bundle.plans if row["kind"] == "gemm"}
        self.assertEqual(by_label["decode_gemv"]["primary_mode"], "MemMode")
        self.assertFalse(by_label["decode_gemv"]["emit_both"])
        self.assertEqual(by_label["prefill_long"]["primary_mode"], "ComputeMode")
        self.assertTrue(by_label["prefill_long"]["materialize_intermediate"])
        self.assertTrue(by_label["moe_expert_med"]["emit_both"])

    def test_attention_never_materializes_scores(self) -> None:
        bundle = build_sweeps(H100_SXM_FP16)
        attn = [row for row in bundle.plans if row["kind"] == "attention_prefill"]
        self.assertTrue(attn)
        for row in attn:
            self.assertFalse(row["materialize_intermediate"])
        by = {row["label"]: row for row in attn}
        self.assertEqual(by["prefill_S128"]["primary_mode"], "MemMode")
        self.assertEqual(by["prefill_S8192"]["primary_mode"], "ComputeMode")
        self.assertEqual(by["prefill_S8192"]["epilogue_policy"], POLICY_COMPILE_SDPA_CHAIN)
        self.assertEqual(by["prefill_S8192"]["fuse_epilogue"], False)
        self.assertTrue(by["prefill_S512"]["emit_both"])
        self.assertEqual(by["prefill_S512"]["epilogue_policy"], POLICY_COMPILE_SDPA_CHAIN)

    def test_intrinsic_ops_stay_mem_mode(self) -> None:
        bundle = build_sweeps(H100_SXM_FP16)
        for row in bundle.plans:
            if row["kind"] in {"ew_chain", "softmax", "reduction", "rmsnorm"}:
                self.assertEqual(row["primary_mode"], "MemMode")
                self.assertFalse(row["emit_both"])

    def test_skewed_moe_spans_regimes(self) -> None:
        bundle = build_sweeps(H100_SXM_FP16)
        skewed = next(r for r in bundle.moe if r["scenario"] == "skewed_8e")
        self.assertEqual(skewed["regime_min_expert"], "memory-bound")
        self.assertEqual(skewed["regime_max_expert"], "compute-bound")
        self.assertEqual(skewed["regime_aggregate"], "ridge-region")
        self.assertTrue(skewed["heterogeneous"])
        self.assertTrue(moe_is_heterogeneous(skewed["expert_regimes"]))
        experts = [r for r in bundle.plans if r["kind"] == "moe_expert"]
        modes = {r["primary_mode"] for r in experts}
        self.assertEqual(modes, {"MemMode", "ComputeMode"})

    def test_measured_ai_overrides_regime(self) -> None:
        site = Site("gemm", "override", {"M": 1, "N": 4096, "K": 4096})
        report = classify_site(site, H100_SXM_FP16, measured_ai=5000.0)
        self.assertTrue(report.calibrated)
        self.assertEqual(report.regime, "compute-bound")
        self.assertEqual(select_plan(report, length=1).primary.mode, "ComputeMode")


if __name__ == "__main__":
    unittest.main()

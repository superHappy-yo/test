"""Closed forms that the paper quotes."""

from __future__ import annotations

import math
import unittest

from shapefuse.intensity import (
    attn_fa_decode,
    attn_fa_prefill,
    attn_unfused_prefill,
    classify,
    gemm_ai,
    gemm_m_for_target_ai,
    gemm_plus_rmsnorm_traffic,
    moe_grouped,
    roofline_fusion_speedup,
    softmax_scores,
)
from shapefuse.machines import H100_SXM_FP16
from shapefuse.selfcheck import assert_invariants


class IntensityTests(unittest.TestCase):
    def test_invariants(self) -> None:
        assert_invariants()

    def test_unfused_attention_stays_memory_bound_on_h100(self) -> None:
        ridge = H100_SXM_FP16.ridge
        for s in (128, 512, 2048, 8192):
            ai = attn_unfused_prefill(1, 32, s, 128, 2).ai
            self.assertLess(ai, 0.5 * ridge, msg=f"S={s} AI={ai}")
            self.assertEqual(classify(ai, ridge), "memory-bound")
        # Ceiling is 4*D/(3*elem); S=8192 is close but not fully saturated.
        self.assertAlmostEqual(attn_unfused_prefill(1, 32, 8192, 128, 2).ai, 4 * 128 / 6, delta=2.0)

    def test_fa_ai_is_s_over_elem(self) -> None:
        for s in (128, 512, 2048):
            self.assertAlmostEqual(attn_fa_prefill(1, 32, s, 128, 2).ai, s / 2)

    def test_decode_asymptote(self) -> None:
        mha = attn_fa_decode(1, 32, 32, 32768, 128, 2, 1).ai
        gqa = attn_fa_decode(1, 32, 8, 32768, 128, 2, 1).ai
        self.assertAlmostEqual(mha, 1.0, delta=0.01)
        self.assertAlmostEqual(gqa, 4.0, delta=0.01)
        self.assertEqual(classify(mha, H100_SXM_FP16.ridge), "memory-bound")

    def test_softmax_constant(self) -> None:
        a = softmax_scores(1, 32, 128, 2).ai
        b = softmax_scores(2, 16, 8192, 2).ai
        self.assertAlmostEqual(a, b)
        self.assertAlmostEqual(a, 5.0 / 6.0)

    def test_epilogue_fraction_grows_with_m_but_roofline_speedup_dies(self) -> None:
        machine = H100_SXM_FP16
        small = gemm_plus_rmsnorm_traffic(1, 4096, 4096, 2)
        large = gemm_plus_rmsnorm_traffic(2048, 4096, 4096, 2)
        self.assertLess(small["byte_save_fraction"], 0.001)
        self.assertGreater(large["byte_save_fraction"], 0.3)
        sp_small = roofline_fusion_speedup(
            small["flops"], small["bytes_unfused"], small["bytes_fused"], machine, 0.0
        )
        sp_large = roofline_fusion_speedup(
            large["flops"], large["bytes_unfused"], large["bytes_fused"], machine, 0.0
        )
        self.assertAlmostEqual(sp_small["speedup"], 1.0, delta=0.01)
        self.assertAlmostEqual(sp_large["speedup"], 1.0, delta=1e-6)
        with_launch = roofline_fusion_speedup(
            small["flops"], small["bytes_unfused"], small["bytes_fused"], machine, 5e-6
        )
        self.assertGreater(with_launch["speedup_with_launch"], 1.2)

    def test_m_star_infinite_when_unreachable(self) -> None:
        # A tiny N cannot host an AI as large as the H100 ridge.
        self.assertTrue(math.isinf(gemm_m_for_target_ai(64, H100_SXM_FP16.ridge, 2)))

    def test_moe_one_hot_weight_load_matches_balanced_traffic(self) -> None:
        balanced, _ = moe_grouped([256] * 8, 4096, 14336, 2, 2, 2)
        touched, _ = moe_grouped([2048], 4096, 14336, 2, 2, 2, load_empty_experts=7)
        self.assertAlmostEqual(balanced.ai, touched.ai, places=6)

    def test_w8a8_does_not_flip_decode_gemm(self) -> None:
        ai = gemm_ai(1, 4096, 4096, 1, 1, 2)
        self.assertLess(ai, 3)
        self.assertEqual(classify(ai, H100_SXM_FP16.ridge), "memory-bound")


if __name__ == "__main__":
    unittest.main()

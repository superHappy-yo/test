"""Regression coverage for FP16 overflow in benchmark normalization."""

import importlib.util
import unittest

from shapefuse.bakeoff import _torch_rmsnorm


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class RmsNormNumericsTests(unittest.TestCase):
    def test_large_fp16_values_do_not_collapse_to_zero(self):
        import torch

        values = torch.tensor([[300.0, 400.0]], dtype=torch.float16)
        result = _torch_rmsnorm(values, torch.ones(2, dtype=torch.float16))
        # The RMS of [300, 400] is sqrt(125000), despite both squares
        # exceeding the maximum finite FP16 value.
        expected = torch.tensor([[0.848528137, 1.13137085]], dtype=torch.float16)
        self.assertEqual(result.dtype, values.dtype)
        self.assertTrue(torch.isfinite(result).all().item())
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()

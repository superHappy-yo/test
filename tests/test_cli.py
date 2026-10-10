"""The CLI must fill an output directory without a GPU."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from shapefuse.cli import main


class CliTests(unittest.TestCase):
    def test_all_cpu_writes_results(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "results"
            code = main(
                [
                    "all",
                    "--out",
                    str(dest),
                    "--device",
                    "cpu",
                    "--bench-level",
                    "smoke",
                    "--cpu-proxy",
                ]
            )
            self.assertEqual(code, 0)
            summary = json.loads((dest / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["gpu_measurement_status"], "PLACEHOLDER_NO_GPU")
            self.assertIn("Stage-Stream", summary["non_contributions"][0])
            self.assertEqual(summary["plan_counts"]["kernels_emitted"], 0)
            gpu = json.loads((dest / "gpu_measurements.json").read_text(encoding="utf-8"))
            self.assertIsNone(gpu["bench"]["nsight_dram_bytes"])
            self.assertIsNone(gpu["bench"]["measured_ridge_flops_per_byte"])
            for name in (
                "gemm_shapes.csv",
                "attention_intensity.csv",
                "attention_decode.csv",
                "rmsnorm_epilogue.csv",
                "ew_chains.csv",
                "moe_grouped.csv",
                "quant_epilogue.csv",
                "dual_plans.csv",
                "bench_gemm.csv",
                "gemm_M_sweep_cpu.csv",
                "fig_gemm_ai_vs_M.png",
                "fig_attn_ai_vs_S.png",
                "fig_moe_skew_ai.png",
                "fig_epilogue_tradeoff.png",
                "fig_plan_regions.png",
                "fig_cpu_gflops_vs_M.png",
                "SEND_BACK.txt",
            ):
                self.assertTrue((dest / name).is_file(), name)
            bench = (dest / "bench_gemm.csv").read_text(encoding="utf-8")
            self.assertIn("TBD_NSIGHT", bench)
            self.assertIn("false", bench)  # fused_plan_timed

    def test_unknown_machine(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            code = main(["analytical", "--out", str(Path(tmp) / "o"), "--machine", "nope", "--skip-plots"])
            self.assertEqual(code, 2)

    def test_cuda_required_without_gpu(self) -> None:
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and torch.cuda.is_available():
            self.skipTest("CUDA is available in this environment")
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            code = main(
                [
                    "bench",
                    "--out",
                    str(Path(tmp) / "o"),
                    "--device",
                    "cuda",
                    "--require-cuda",
                    "--bench-level",
                    "smoke",
                ]
            )
            self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()

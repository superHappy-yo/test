"""Datasheet aliases, Nsight JSON, agreement rates, and the fusion proxy."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from shapefuse.agreement import build_decision_log, summarize_agreement
from shapefuse.cli import main
from shapefuse.fusion_proxy import HONESTY_BIAS_HAND, HONESTY_TWO_KERNEL
from shapefuse.intensity import gemm_bytes, gemm_flops
from shapefuse.machines import A40_FP16, get_machine
from shapefuse.profiler_hooks import NSIGHT_DRAM, NSIGHT_PROCESS_SUM, parse_ncu_csv


class NextRunTests(unittest.TestCase):
    def test_aliases_and_a40_datasheet(self) -> None:
        machine = get_machine("a40")
        self.assertEqual(machine.name, "A40_FP16")
        self.assertEqual(machine.peak_flops_per_s, 149.7e12)
        self.assertEqual(machine.peak_bytes_per_s, 696e9)
        self.assertEqual(get_machine("A40").name, A40_FP16.name)
        self.assertEqual(get_machine("h100").name, "H100_SXM_FP16")
        self.assertEqual(get_machine("a100").name, "A100_SXM80_FP16")
        self.assertIn("sparsity", machine.source)
        self.assertNotIn("measured", machine.source.lower())

    def test_ncu_csv_sums_dram_metrics(self) -> None:
        text = "\n".join(
            [
                "==PROF== some banner",
                '"Metric Name","Metric Value"',
                '"dram__bytes_read.sum","1000"',
                '"dram__bytes_write.sum","250"',
                '"other.sum","9"',
            ]
        )
        found = parse_ncu_csv(text, ["dram__bytes_read.sum", "dram__bytes_write.sum"])
        self.assertEqual(found["dram__bytes_read.sum"], 1000.0)
        self.assertEqual(found["dram__bytes_write.sum"], 250.0)
        self.assertNotIn("other.sum", found)

    def test_agreement_flips_only_when_byte_ai_crosses(self) -> None:
        machine = get_machine("h100")
        flops = gemm_flops(1, 4096, 4096)
        cold = gemm_bytes(1, 4096, 4096, 2, 2, 2)
        rows = [
            _bench_row("decode_gemv", 1, flops / cold, NSIGHT_DRAM),
            _bench_row("forced_compute", 1, 5000.0, NSIGHT_DRAM),
        ]
        # Second row reuses M=1 but a huge AI, so the plan flips.
        rows[1]["label"] = "forced_compute"
        log = build_decision_log(rows, machine)
        summary = summarize_agreement(log)
        self.assertEqual(summary["n_with_byte_ai"], 2)
        self.assertEqual(summary["n_decision_flips_any_byte_source"], 1)
        self.assertAlmostEqual(summary["regime_agreement_rate_nsight_isolated"], 0.5)
        self.assertTrue(summary["quote_as_scic_accuracy"])
        self.assertEqual(summary["scic_accuracy_status"], "NSIGHT_DRAM")
        self.assertFalse(any(row["kernel_emitted"] for row in log))

    def test_process_sum_is_not_quoted_as_accuracy(self) -> None:
        machine = get_machine("a40")
        rows = [_bench_row("decode_gemv", 1, 1.0, NSIGHT_PROCESS_SUM)]
        summary = summarize_agreement(build_decision_log(rows, machine))
        self.assertFalse(summary["quote_as_scic_accuracy"])
        self.assertIn("PROCESS_SUM", summary["scic_accuracy_status"])
        self.assertIsNone(summary["regime_agreement_rate_nsight_isolated"])
        self.assertEqual(summary["regime_agreement_rate_nsight_process_sum"], 1.0)

    def test_torch_proxy_does_not_count_as_nsight(self) -> None:
        machine = get_machine("a40")
        row = _bench_row("decode_gemv", 1, None, "TBD_NSIGHT")
        row["proxy_ai"] = 1.0
        row["proxy_ai_status"] = "TORCH_PROFILER_MEMORY_NOT_DRAM"
        summary = summarize_agreement(build_decision_log([row], machine))
        self.assertFalse(summary["quote_as_scic_accuracy"])
        self.assertEqual(summary["n_torch_profiler_memory"], 1)
        self.assertEqual(summary["n_nsight_dram_isolated"], 0)
        self.assertIn("NOT_VALID_SCIC_ACCURACY", summary["scic_accuracy_status"])

    def test_cli_a40_machine_and_fusion_proxy(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out"
            code = main(
                [
                    "all",
                    "--out",
                    str(dest),
                    "--device",
                    "cpu",
                    "--bench-level",
                    "smoke",
                    "--machine",
                    "a40",
                    "--skip-plots",
                    "--fusion-proxy",
                ]
            )
            self.assertEqual(code, 0)
            summary = json.loads((dest / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["machine"], "A40_FP16")
            self.assertIn("A40_FP16", summary["crossings"])
            self.assertEqual(summary["gpu_measurement_status"], "PLACEHOLDER_NO_GPU")
            self.assertFalse(summary["fusion_proxy"]["claims_memmode_kernel"])
            self.assertFalse(summary["fusion_proxy"]["claims_computemode_kernel"])
            self.assertFalse(summary["fusion_proxy"]["claims_stage_stream"])
            proxy = (dest / "fusion_proxy.csv").read_text(encoding="utf-8")
            self.assertIn(HONESTY_TWO_KERNEL, proxy)
            self.assertIn(HONESTY_BIAS_HAND, proxy)
            self.assertIn("PROXY_TIMING_NOT_A_FUSION_SPEEDUP", proxy)
            self.assertNotIn("MemMode kernel", proxy)
            bench = (dest / "bench_gemm.csv").read_text(encoding="utf-8")
            self.assertIn("regime_vs_requested_datasheet", bench)
            self.assertIn("A40_FP16", bench)

    def test_cli_ingests_nsight_json_on_cpu_timing(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out"
            nsight = Path(tmp) / "nsight.json"
            nsight.write_text(
                json.dumps(
                    {
                        "status": "NSIGHT_DRAM_PROCESS_SUM",
                        "rows": [
                            {"label": "smoke_m64", "dram_bytes": 4096.0},
                            {"label": "smoke_m256", "dram_bytes": 8192.0},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            code = main(
                [
                    "bench",
                    "--out",
                    str(dest),
                    "--device",
                    "cpu",
                    "--bench-level",
                    "smoke",
                    "--machine",
                    "a40",
                    "--skip-plots",
                    "--nsight-json",
                    str(nsight),
                ]
            )
            self.assertEqual(code, 0)
            agreement = json.loads((dest / "agreement.json").read_text(encoding="utf-8"))
            self.assertFalse(agreement["quote_as_scic_accuracy"])
            self.assertEqual(agreement["n_with_byte_ai"], 2)
            bench = (dest / "bench_gemm.csv").read_text(encoding="utf-8")
            self.assertIn("NSIGHT_DRAM_PROCESS_SUM", bench)
            gpu = json.loads((dest / "gpu_measurements.json").read_text(encoding="utf-8"))
            self.assertEqual(gpu["status"], "PLACEHOLDER_NO_GPU")
            self.assertTrue(gpu["do_not_treat_as_fused_kernel_study"])


def _bench_row(label: str, m: int, achieved_ai: float | None, status: str) -> dict:
    return {
        "label": label,
        "M": m,
        "N": 4096,
        "K": 4096,
        "achieved_ai": achieved_ai,
        "achieved_ai_status": status,
        "proxy_ai": None,
        "proxy_ai_status": "PROFILER_OFF",
    }


if __name__ == "__main__":
    unittest.main()

"""Datasheet aliases, Nsight JSON, agreement rates, and the fusion proxy."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from shapefuse.agreement import build_decision_log, summarize_agreement
from shapefuse.bandwidth_probe import ESTIMATE_STATUS, estimate_from_timing
from shapefuse.cli import main
from shapefuse.fusion_proxy import HONESTY_BIAS_HAND, HONESTY_TWO_KERNEL
from shapefuse.intensity import gemm_bytes, gemm_flops
from shapefuse.machines import A40_FP16, get_machine
from shapefuse.profiler_hooks import (
    NSIGHT_DRAM,
    NSIGHT_PROCESS_SUM,
    TORCH_ALLOCATOR_DELTA,
    TORCH_EMPTY,
    TORCH_MEMORY,
    bytes_from_profiler_events,
    chrome_trace_memory_net,
    collect_proxy_bytes,
    parse_ncu_csv,
)


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
            self.assertIn("proxy_byte_source", bench)
            send = (dest / "SEND_BACK.txt").read_text(encoding="utf-8")
            self.assertIn("TRY_NSIGHT=1", send)
            self.assertIn("PROXY_TIMING_NOT_A_FUSION_SPEEDUP", send)
            self.assertIn("ESTIMATE_COPY_NOT_NSIGHT", send)
            self.assertIn("ncu", send)

    def test_profiler_reads_device_memory_field(self) -> None:
        # PyTorch 2.5 key_averages() has self_device_memory_usage and does
        # not define self_cuda_memory_usage. getattr(..., default 0) was empty.
        events = [_Evt(self_device_memory_usage=4096, device_memory_usage=8192)]
        nbytes, source = bytes_from_profiler_events(events, cuda=True)
        self.assertEqual(nbytes, 4096)
        self.assertIn("self_device_memory_usage", source)

    def test_legacy_cuda_memory_name_still_counts(self) -> None:
        events = [_Evt(self_cuda_memory_usage=2048)]
        nbytes, source = bytes_from_profiler_events(events, cuda=True)
        self.assertEqual(nbytes, 2048)
        self.assertIn("self_device_memory_usage", source)

    def test_chrome_trace_net_is_profiler_not_allocator_delta(self) -> None:
        text = json.dumps(
            {
                "traceEvents": [
                    {"name": "[memory]", "args": {"Bytes": 100, "Device Type": 1}},
                    {"name": "[memory]", "args": {"Bytes": -40, "Device Type": 1}},
                    {"name": "[memory]", "args": {"Bytes": 999, "Device Type": 0}},
                ]
            }
        )
        self.assertEqual(chrome_trace_memory_net(text, cuda=True), 60)
        chosen = collect_proxy_bytes(
            events=[_Evt(self_device_memory_usage=0, device_memory_usage=0)],
            chrome_trace=text,
            allocator_delta=5,
            cuda=True,
        )
        self.assertEqual(chosen["status"], TORCH_MEMORY)
        self.assertEqual(chosen["bytes"], 60.0)
        self.assertEqual(chosen["source"], "chrome_trace.memory")
        self.assertNotIn("NSIGHT", chosen["status"])

    def test_allocator_delta_keeps_a_distinct_status(self) -> None:
        chosen = collect_proxy_bytes(
            events=[_Evt(self_device_memory_usage=0, device_memory_usage=0)],
            chrome_trace="{}",
            allocator_delta=8192,
            cuda=True,
        )
        self.assertEqual(chosen["status"], TORCH_ALLOCATOR_DELTA)
        self.assertEqual(chosen["bytes"], 8192.0)
        self.assertNotEqual(chosen["status"], TORCH_MEMORY)
        empty = collect_proxy_bytes(
            events=[_Evt(self_device_memory_usage=0)],
            chrome_trace="{}",
            allocator_delta=0,
            cuda=True,
        )
        self.assertEqual(empty["status"], TORCH_EMPTY)
        self.assertIsNone(empty["bytes"])

    def test_allocator_delta_is_not_scic_accuracy(self) -> None:
        machine = get_machine("a40")
        row = _bench_row("decode_gemv", 1, None, "TBD_NSIGHT")
        row["proxy_ai"] = 12.0
        row["proxy_ai_status"] = TORCH_ALLOCATOR_DELTA
        summary = summarize_agreement(build_decision_log([row], machine))
        self.assertFalse(summary["quote_as_scic_accuracy"])
        self.assertEqual(summary["n_torch_profiler_memory"], 0)
        self.assertEqual(summary["n_torch_allocator_delta"], 1)
        self.assertIn("NOT_VALID_SCIC_ACCURACY", summary["scic_accuracy_status"])
        self.assertNotIn("NSIGHT_DRAM", summary["scic_accuracy_status"])

    def test_triton_tl_is_visible_as_a_module_global(self) -> None:
        import ast
        import importlib
        import inspect
        import sys
        import textwrap
        import types

        import shapefuse.fusion_proxy as fusion_proxy

        language = types.ModuleType("triton.language")

        class _Constexpr:
            pass

        language.constexpr = _Constexpr
        triton_mod = types.ModuleType("triton")
        seen: dict[str, object] = {}

        def jit(fn):
            src = textwrap.dedent(inspect.getsource(fn))
            tree = ast.parse(src)
            fndef = tree.body[0]
            self.assertIsInstance(fndef, ast.FunctionDef)
            evaluated = None
            for arg in fndef.args.args:
                if arg.annotation is None:
                    continue
                segment = ast.get_source_segment(src, arg.annotation)
                # Triton uses the kernel globals, not the enclosing function locals.
                evaluated = eval(segment, fn.__globals__)  # noqa: S307
            seen["annotation"] = evaluated
            return fn

        triton_mod.jit = jit
        triton_mod.language = language
        triton_mod.cdiv = lambda n, block: (n + block - 1) // block
        saved = {name: sys.modules.get(name) for name in ("triton", "triton.language")}
        sys.modules["triton"] = triton_mod
        sys.modules["triton.language"] = language
        try:
            reloaded = importlib.reload(fusion_proxy)
            reloaded._BIAS_GELU_KERNEL = None
            reloaded._bias_gelu_kernel()
            self.assertIs(seen["annotation"], _Constexpr)
            self.assertIn("tl", reloaded._bias_gelu_kernel.__globals__)
        finally:
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
            importlib.reload(fusion_proxy)

    def test_copy_bandwidth_estimate_is_not_a_measured_ridge(self) -> None:
        payload = estimate_from_timing(
            seconds=0.5,
            repeats=4,
            numel=1000,
            element_bytes=2,
            datasheet_peak_flops_per_s=149.7e12,
        )
        self.assertEqual(payload["status"], ESTIMATE_STATUS)
        self.assertEqual(payload["bytes_moved"], 16000)
        self.assertEqual(payload["estimated_bandwidth_bytes_per_s"], 32000.0)
        self.assertAlmostEqual(payload["estimated_ridge_flops_per_byte"], 149.7e12 / 32000.0)
        self.assertIsNone(payload["measured_hbm_bandwidth_bytes_per_s"])
        self.assertIsNone(payload["measured_ridge_flops_per_byte"])
        self.assertTrue(payload["not_nsight"])
        self.assertTrue(payload["not_a_measured_ridge"])
        self.assertTrue(payload["do_not_write_measured_ridge_flops_per_byte"])

    def test_run_a40_script_recommends_nsight_and_documents_missing_ncu(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = (root / "scripts" / "run_a40.sh").read_text(encoding="utf-8")
        self.assertIn("TRY_NSIGHT=1", script)
        self.assertIn("ncu: NOT on PATH", script)
        self.assertIn("nsight_skipped.txt", script)
        self.assertIn("Nsight Compute", script)
        self.assertIn("bandwidth_estimate.json", script)
        self.assertIn("ERR_NVGPUCTRPERM", script)
        self.assertIn("docs/NSIGHT_PERMISSIONS.md", script)
        probe = (root / "scripts" / "bandwidth_probe.sh").read_text(encoding="utf-8")
        self.assertIn("ESTIMATE_COPY_NOT_NSIGHT", probe)
        nsight = (root / "scripts" / "nsight_dram.sh").read_text(encoding="utf-8")
        self.assertIn("ERR_NVGPUCTRPERM", nsight)
        self.assertIn("docs/NSIGHT_PERMISSIONS.md", nsight)
        readme = (root / "README.md").read_text(encoding="utf-8")
        self.assertIn("docs/NSIGHT_PERMISSIONS.md", readme)
        perms = (root / "docs" / "NSIGHT_PERMISSIONS.md").read_text(encoding="utf-8")
        self.assertIn("NVreg_RestrictProfilingToAdminUsers=0", perms)
        self.assertIn("modprobe nvidia", perms)
        self.assertIn("/etc/modprobe.d/", perms)
        self.assertIn("ERR_NVGPUCTRPERM", perms)

    def test_paper_quotes_third_a40_package(self) -> None:
        import csv

        root = Path(__file__).resolve().parents[1]
        draft = (root / "paper" / "jsa" / "main.tex").read_text(encoding="utf-8").replace("\\_", "_")
        zh = (root / "paper" / "jsa" / "jsa_main_zh.md").read_text(encoding="utf-8")
        with (root / "paper" / "data" / "a40" / "bench_gemm.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        table = draft.split("\\label{tab:matmul}", 1)[1].split("\\midrule", 1)[1].split("\\bottomrule", 1)[0]
        displayed = {}
        for line in table.splitlines():
            if "&" in line:
                cells = [cell.strip() for cell in line.split("&")]
                displayed[int(cells[1])] = float(cells[3])
        # Supplementary tables can renumber the pilot. Locate its data content.
        stock_section = zh.split("### C.1", 1)[1].split("### C.2", 1)[0]
        chinese_table = stock_section.split("| Label", 1)[1].split("![", 1)[0]
        displayed_zh = {}
        for line in chinese_table.splitlines():
            if line.startswith("|"):
                cells = [cell.strip() for cell in line.split("|")[1:-1]]
                if len(cells) >= 4 and cells[1].isdigit():
                    displayed_zh[int(cells[1])] = float(cells[3])
        for row in rows:
            expected = float(row["achieved_tflops"])
            self.assertAlmostEqual(displayed[int(row["M"])], expected, delta=expected * 0.01)
            self.assertAlmostEqual(displayed_zh[int(row["M"])], expected, delta=expected * 0.01)
            self.assertEqual(row["proxy_ai_status"], "TORCH_PROFILER_MEMORY_NOT_DRAM")
            self.assertEqual(row["proxy_ai"], "4096")
            self.assertEqual(row["achieved_ai_status"], "TBD_NSIGHT")
            self.assertEqual(row["requested_machine"], "A40_FP16")
        summary = json.loads((root / "paper" / "data" / "a40" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["machine"], "A40_FP16")
        self.assertEqual(summary["gpu_measurement_status"], "PARTIAL_VENDOR_MATMUL")
        self.assertEqual(summary["bench"]["device_name"], "NVIDIA A40")
        self.assertEqual(summary["bench"]["nsight_status"], "TBD_RUN_PROFILER")
        self.assertIsNone(summary["bench"]["measured_ridge_flops_per_byte"])
        self.assertFalse(summary["decision_agreement"]["quote_as_scic_accuracy"])
        self.assertEqual(
            summary["decision_agreement"]["scic_accuracy_status"],
            "PROXY_TORCH_MEMORY_NOT_VALID_SCIC_ACCURACY",
        )
        self.assertTrue(summary["fusion_proxy"]["triton_used"])
        self.assertFalse(summary["fusion_proxy"]["claims_memmode_kernel"])
        bandwidth = json.loads((root / "paper" / "data" / "a40" / "bandwidth_estimate.json").read_text(encoding="utf-8"))
        self.assertEqual(bandwidth["status"], "ESTIMATE_COPY_NOT_NSIGHT")
        self.assertIsNone(bandwidth["measured_ridge_flops_per_byte"])
        for document in (draft, zh):
            document = document.replace(r"\allowbreak ", "")
            self.assertIn("ERR_NVGPUCTRPERM", document)
            self.assertIn("ESTIMATE_COPY_NOT_NSIGHT", document)
            self.assertIn("quote_as_scic_accuracy", document)
        self.assertNotIn("SCIC accuracy is 0.2857142857142857", draft)
        self.assertNotIn("SCIC 准确率是 0.2857142857142857", zh)
        crossings = json.loads((root / "paper" / "data" / "a40" / "crossings.json").read_text(encoding="utf-8"))
        star = "240.32584259093773"
        self.assertEqual(crossings["A40_FP16"]["M_star_AI_equals_ridge"], float(star))
        self.assertEqual(summary["crossings"]["A40_FP16"]["M_star_AI_equals_ridge"], float(star))
        self.assertIn(f"{float(star):.1f}", draft)
        self.assertIn(f"{float(star):.1f}", zh)
        skipped = (root / "paper" / "data" / "a40" / "nsight_skipped.txt").read_text(encoding="utf-8")
        error = (root / "paper" / "data" / "a40" / "nsight_error.log").read_text(encoding="utf-8")
        self.assertIn("TRY_NSIGHT=1", skipped)
        self.assertIn("ncu was on PATH", skipped)
        self.assertIn("ERR_NVGPUCTRPERM", error)

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


class _Evt:
    def __init__(self, **kwargs: object) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


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

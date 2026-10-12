"""SCIC path selection and the bakeoff CSV schema. No invented GPU numbers."""

from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

from shapefuse.bakeoff import (
    ATTN_LEVELS,
    GEMM_LEVELS,
    Measurement,
    assemble_latency_rows,
    calibration_from_medians,
    configure_dynamo_cache_limit,
    decision_row,
    decide_attention,
    decide_gemm,
    scic_selection_honesty,
    skipped_calibration,
)
from shapefuse.cli import main
from shapefuse.intensity import Thresholds
from shapefuse.machines import A40_FP16, H100_SXM_FP16
from shapefuse.plan_paths import (
    BASELINE_ALWAYS_FUSE,
    BASELINE_ALWAYS_UNFUSE,
    BASELINE_EAGER,
    BASELINE_SCIC,
    HONESTY_ANALYTICAL_FALLBACK,
    HONESTY_GPU_CONTENTION,
    HONESTY_MEASURED_OVERRIDE,
    HONESTY_WALLCLOCK_NOT_DRAM,
    SELECTION_ANALYTICAL,
    SELECTION_ANALYTICAL_FALLBACK,
    SELECTION_MEASURED,
    analytical_executable_path,
    attach_measured_winner,
    gemm_calibration_applies,
    select_executable_path,
)
from shapefuse.scic import Site, classify_site
from shapefuse.plans import POLICY_COMPILE_SDPA_CHAIN, select_plan


class PathSelectionTests(unittest.TestCase):
    def test_gemm_switches_across_the_a40_datasheet_ridge(self) -> None:
        thresholds = Thresholds()
        chosen = []
        emit_both = []
        for shape in GEMM_LEVELS["decode-prefill"]:
            report, decision, path = decide_gemm(shape, A40_FP16, thresholds)
            chosen.append(path)
            emit_both.append(decision.emit_both)
            self.assertFalse(decision.kernel_emitted)
            self.assertIn(path, {BASELINE_ALWAYS_FUSE, BASELINE_ALWAYS_UNFUSE})
            self.assertEqual(path == BASELINE_ALWAYS_FUSE, decision.primary.fuse_epilogue)
            self.assertEqual(report.machine, "A40_FP16")
        self.assertIn(BASELINE_ALWAYS_FUSE, chosen)
        self.assertIn(BASELINE_ALWAYS_UNFUSE, chosen)
        self.assertTrue(any(emit_both))
        by_label = {
            shape.label: decide_gemm(shape, A40_FP16, thresholds)[2]
            for shape in GEMM_LEVELS["decode-prefill"]
        }
        self.assertEqual(by_label["decode_m1"], BASELINE_ALWAYS_FUSE)
        self.assertEqual(by_label["prefill_m2048"], BASELINE_ALWAYS_UNFUSE)
        self.assertEqual(by_label["prefill_m4096"], BASELINE_ALWAYS_UNFUSE)

    def test_attention_keeps_scores_off_hbm_and_fuses_library_epilogue(self) -> None:
        thresholds = Thresholds()
        for shape in ATTN_LEVELS["decode-prefill"]:
            _report, decision, path = decide_attention(shape, A40_FP16, thresholds)
            self.assertFalse(decision.primary.materialize_intermediate)
            self.assertFalse(decision.kernel_emitted)
            self.assertEqual(decision.epilogue_policy, POLICY_COMPILE_SDPA_CHAIN)
            self.assertEqual(path, BASELINE_ALWAYS_FUSE)
        by_label = {
            shape.label: decide_attention(shape, A40_FP16, thresholds)
            for shape in ATTN_LEVELS["decode-prefill"]
        }
        self.assertEqual(by_label["prefill_s128"][1].primary.mode, "MemMode")
        self.assertTrue(by_label["prefill_s128"][1].primary.fuse_epilogue)
        # Analytical primary stays ComputeMode. The timed path does not.
        self.assertEqual(by_label["prefill_s2048"][1].primary.mode, "ComputeMode")
        self.assertFalse(by_label["prefill_s2048"][1].primary.fuse_epilogue)
        self.assertEqual(by_label["prefill_s2048"][2], BASELINE_ALWAYS_FUSE)
        self.assertTrue(by_label["prefill_s512"][1].emit_both)
        self.assertEqual(by_label["prefill_s512"][1].primary.mode, "ComputeMode")
        self.assertEqual(by_label["prefill_s512"][2], BASELINE_ALWAYS_FUSE)
        self.assertEqual(by_label["decode_sk2048"][1].primary.mode, "MemMode")
        self.assertEqual(by_label["decode_sk2048"][2], BASELINE_ALWAYS_FUSE)

    def test_h100_memory_and_compute_match_the_selector(self) -> None:
        mem = classify_site(Site("gemm", "m1", {"M": 1, "N": 4096, "K": 4096}), H100_SXM_FP16)
        comp = classify_site(Site("gemm", "m2048", {"M": 2048, "N": 4096, "K": 4096}), H100_SXM_FP16)
        self.assertEqual(select_plan(mem, length=1).primary.mode, "MemMode")
        self.assertEqual(decide_gemm(GEMM_LEVELS["decode-prefill"][0], H100_SXM_FP16, Thresholds())[2], BASELINE_ALWAYS_FUSE)
        self.assertEqual(select_plan(comp, length=2048).primary.mode, "ComputeMode")


class CalibrationOverrideTests(unittest.TestCase):
    def _compute_gemm(self):
        shape = next(s for s in GEMM_LEVELS["decode-prefill"] if s.label == "prefill_m2048")
        report, decision, chosen = decide_gemm(shape, A40_FP16, Thresholds())
        self.assertEqual(chosen, BASELINE_ALWAYS_UNFUSE)
        self.assertTrue(gemm_calibration_applies(decision, "ridge-compute"))
        self.assertEqual(analytical_executable_path(decision), BASELINE_ALWAYS_UNFUSE)
        return shape, report, decision

    def test_faster_fuse_probe_overrides_compute_gemm(self) -> None:
        shape, report, decision = self._compute_gemm()
        calibration = calibration_from_medians(0.70, 0.88, repeats=3, warmup=1)
        self.assertEqual(calibration.winner, BASELINE_ALWAYS_FUSE)
        self.assertEqual(calibration.source, SELECTION_MEASURED)
        self.assertIn(HONESTY_WALLCLOCK_NOT_DRAM, calibration.honesty)
        self.assertIn(HONESTY_GPU_CONTENTION, calibration.honesty)
        updated = attach_measured_winner(decision, calibration.winner)
        chosen = select_executable_path(updated)
        self.assertEqual(chosen, BASELINE_ALWAYS_FUSE)
        self.assertEqual(updated.measured_winner, BASELINE_ALWAYS_FUSE)
        self.assertEqual(analytical_executable_path(updated), BASELINE_ALWAYS_UNFUSE)
        row = decision_row(
            op="matmul_rmsnorm",
            label=shape.label,
            shape=shape.shape_key(),
            report=report,
            decision=updated,
            chosen=chosen,
            analytical=BASELINE_ALWAYS_UNFUSE,
            calibration=calibration,
        )
        self.assertEqual(row["analytical_baseline"], BASELINE_ALWAYS_UNFUSE)
        self.assertEqual(row["measured_winner"], BASELINE_ALWAYS_FUSE)
        self.assertEqual(row["selected_baseline"], BASELINE_ALWAYS_FUSE)
        self.assertEqual(row["selection_source"], SELECTION_MEASURED)
        self.assertEqual(row["calibration_fuse_ms"], 0.70)
        self.assertEqual(row["calibration_unfuse_ms"], 0.88)
        self.assertFalse(row["quote_as_scic_accuracy"])
        self.assertFalse(row["nsight_used"])
        flags = scic_selection_honesty(calibration, BASELINE_ALWAYS_UNFUSE, chosen)
        self.assertIn(HONESTY_MEASURED_OVERRIDE, flags)
        self.assertIn(HONESTY_WALLCLOCK_NOT_DRAM, flags)

    def test_missing_or_tied_probe_keeps_the_analytical_baseline(self) -> None:
        _shape, report, decision = self._compute_gemm()
        for fuse_ms, unfuse_ms in ((None, 1.0), (1.0, None), (1.0, 1.0), (float("nan"), 1.0)):
            calibration = calibration_from_medians(fuse_ms, unfuse_ms, repeats=1, warmup=1)
            self.assertEqual(calibration.winner, "")
            self.assertEqual(calibration.source, SELECTION_ANALYTICAL_FALLBACK)
            self.assertIn(HONESTY_ANALYTICAL_FALLBACK, calibration.honesty)
            chosen = select_executable_path(attach_measured_winner(decision, calibration.winner or None))
            self.assertEqual(chosen, BASELINE_ALWAYS_UNFUSE)
        skipped = skipped_calibration(SELECTION_ANALYTICAL, "outside scope")
        self.assertFalse(skipped.attempted)
        self.assertEqual(skipped.winner, "")
        row = decision_row(
            op="matmul_rmsnorm",
            label="decode_m1",
            shape="M=1,N=4096,K=4096",
            report=report,
            decision=decision,
            chosen=BASELINE_ALWAYS_UNFUSE,
            analytical=BASELINE_ALWAYS_UNFUSE,
            calibration=skipped,
        )
        self.assertEqual(row["selection_source"], SELECTION_ANALYTICAL)
        self.assertEqual(row["measured_winner"], "")
        self.assertEqual(row["analytical_baseline"], BASELINE_ALWAYS_UNFUSE)

    def test_ridge_compute_scope_skips_memory_gemm_and_attention(self) -> None:
        thresholds = Thresholds()
        by_label = {
            shape.label: decide_gemm(shape, A40_FP16, thresholds)[1]
            for shape in GEMM_LEVELS["decode-prefill"]
        }
        for label in ("decode_m1", "decode_m32"):
            self.assertFalse(gemm_calibration_applies(by_label[label], "ridge-compute"))
            self.assertTrue(gemm_calibration_applies(by_label[label], "all-gemm"))
        for label in ("ridge_m128", "ridge_m256", "ridge_m512", "prefill_m2048", "prefill_m4096"):
            self.assertTrue(gemm_calibration_applies(by_label[label], "ridge-compute"))
        self.assertFalse(gemm_calibration_applies(by_label["prefill_m2048"], "off"))
        attn = decide_attention(ATTN_LEVELS["decode-prefill"][2], A40_FP16, thresholds)[1]
        self.assertEqual(attn.primary.mode, "ComputeMode")
        self.assertFalse(gemm_calibration_applies(attn, "all-gemm"))
        self.assertEqual(select_executable_path(attn), BASELINE_ALWAYS_FUSE)

    def test_override_honesty_lands_on_the_scic_latency_row(self) -> None:
        shape, report, decision = self._compute_gemm()
        calibration = calibration_from_medians(0.4, 0.5, repeats=3, warmup=1)
        updated = attach_measured_winner(decision, calibration.winner)
        chosen = select_executable_path(updated)
        rows = assemble_latency_rows(
            op="matmul_rmsnorm",
            label=shape.label,
            shape=shape.shape_key(),
            baselines=("eager_unfused", "always_fuse", "always_unfuse", "scic_selected"),
            measurements={
                "eager_unfused": Measurement(1.0, True, "", "EAGER"),
                "always_fuse": Measurement(0.4, True, "", "FUSE"),
                "always_unfuse": Measurement(0.9, True, "", "UNFUSE"),
            },
            chosen=chosen,
            report=report,
            emit_both=decision.emit_both,
            primary_mode=decision.primary.mode,
            fuse_epilogue=decision.primary.fuse_epilogue,
            placeholder=False,
            backend="cuda-torch",
            device_name="NVIDIA A40",
            dtype_ran="fp16",
            selection_honesty=scic_selection_honesty(calibration, BASELINE_ALWAYS_UNFUSE, chosen),
        )
        scic = next(row for row in rows if row["baseline"] == BASELINE_SCIC)
        self.assertEqual(scic["median_ms"], 0.4)
        self.assertEqual(scic["timing_copied_from"], BASELINE_ALWAYS_FUSE)
        self.assertEqual(scic["selected_baseline"], BASELINE_ALWAYS_FUSE)
        self.assertIn(HONESTY_MEASURED_OVERRIDE, scic["honesty"])
        self.assertIn(HONESTY_WALLCLOCK_NOT_DRAM, scic["honesty"])
        self.assertIn(HONESTY_GPU_CONTENTION, scic["honesty"])
        self.assertFalse(scic["quote_as_scic_accuracy"])


class AssembleTests(unittest.TestCase):
    def test_scic_copies_the_selected_median_and_sets_honesty(self) -> None:
        report, decision, chosen = decide_gemm(
            GEMM_LEVELS["decode-prefill"][0], A40_FP16, Thresholds()
        )
        self.assertEqual(chosen, BASELINE_ALWAYS_FUSE)
        measurements = {
            "eager_unfused": Measurement(2.0, True, "", "EAGER"),
            "torch_compile_default": Measurement(0.8, True, "", "COMPILE"),
            "torch_compile_max_autotune": Measurement(None, False, "skipped", "SKIPPED"),
            "always_fuse": Measurement(0.5, True, "", "FUSE"),
            "always_unfuse": Measurement(1.5, True, "", "UNFUSE"),
        }
        rows = assemble_latency_rows(
            op="matmul_rmsnorm",
            label="decode_m1",
            shape="M=1,N=4096,K=4096",
            baselines=(
                "eager_unfused",
                "torch_compile_default",
                "torch_compile_max_autotune",
                "always_fuse",
                "always_unfuse",
                "scic_selected",
            ),
            measurements=measurements,
            chosen=chosen,
            report=report,
            emit_both=decision.emit_both,
            primary_mode=decision.primary.mode,
            fuse_epilogue=decision.primary.fuse_epilogue,
            placeholder=False,
            backend="cuda-torch",
            device_name="NVIDIA A40",
            dtype_ran="fp16",
        )
        by = {row["baseline"]: row for row in rows}
        self.assertEqual(by[BASELINE_EAGER]["speedup_vs_eager"], 1.0)
        self.assertEqual(by[BASELINE_SCIC]["median_ms"], 0.5)
        self.assertEqual(by[BASELINE_SCIC]["timing_copied_from"], BASELINE_ALWAYS_FUSE)
        self.assertAlmostEqual(by[BASELINE_SCIC]["speedup_vs_eager"], 4.0)
        self.assertIn("TIMING_COPIED_FROM_always_fuse", by[BASELINE_SCIC]["honesty"])
        self.assertIn("NO_NSIGHT", by[BASELINE_SCIC]["honesty"])
        self.assertIn("NOT_AN_FA3_KERNEL", by[BASELINE_SCIC]["honesty"])
        self.assertIn("NOT_SCIC_ACCURACY", by[BASELINE_SCIC]["honesty"])
        self.assertFalse(by[BASELINE_SCIC]["quote_as_scic_accuracy"])
        self.assertFalse(by[BASELINE_SCIC]["claims_fa3"])
        self.assertFalse(by[BASELINE_SCIC]["claims_stage_stream"])
        self.assertFalse(by[BASELINE_SCIC]["claims_custom_kernel"])
        self.assertFalse(by[BASELINE_SCIC]["nsight_used"])
        self.assertFalse(by[BASELINE_SCIC]["graph_inspected"])
        self.assertEqual(by["torch_compile_max_autotune"]["median_ms"], None)
        self.assertIsNone(by["torch_compile_max_autotune"]["speedup_vs_eager"])

    def test_missing_selected_path_does_not_invent_a_median(self) -> None:
        report, decision, chosen = decide_gemm(
            GEMM_LEVELS["decode-prefill"][0], A40_FP16, Thresholds()
        )
        measurements = {
            "eager_unfused": Measurement(2.0, True, "", "EAGER"),
            "always_fuse": Measurement(None, False, "compile failed", "FAIL"),
            "always_unfuse": Measurement(1.5, True, "", "UNFUSE"),
        }
        rows = assemble_latency_rows(
            op="matmul_rmsnorm",
            label="decode_m1",
            shape="M=1,N=4096,K=4096",
            baselines=("eager_unfused", "always_fuse", "always_unfuse", "scic_selected"),
            measurements=measurements,
            chosen=chosen,
            report=report,
            emit_both=False,
            primary_mode=decision.primary.mode,
            fuse_epilogue=True,
            placeholder=True,
            backend="cpu-numpy",
            device_name="cpu",
            dtype_ran="fp32",
        )
        scic = next(row for row in rows if row["baseline"] == BASELINE_SCIC)
        self.assertFalse(scic["ok"])
        self.assertIsNone(scic["median_ms"])
        self.assertIsNone(scic["speedup_vs_eager"])
        self.assertIn("SCIC_SELECTED_PATH_DID_NOT_RUN", scic["honesty"])
        self.assertIn("PLACEHOLDER_NO_GPU", scic["honesty"])

    def test_prefill_compute_copies_compile_sdpa_chain(self) -> None:
        shape = next(s for s in ATTN_LEVELS["decode-prefill"] if s.label == "prefill_s2048")
        report, decision, chosen = decide_attention(shape, A40_FP16, Thresholds())
        self.assertEqual(decision.primary.mode, "ComputeMode")
        self.assertEqual(chosen, BASELINE_ALWAYS_FUSE)
        measurements = {
            "eager_unfused": Measurement(2.0, True, "", "EAGER"),
            "always_fuse": Measurement(0.4, True, "", "FUSE"),
            "always_unfuse": Measurement(1.5, True, "", "UNFUSE"),
        }
        rows = assemble_latency_rows(
            op="attention",
            label="prefill_s2048",
            shape="B=1,H=8,S=2048,D=64",
            baselines=("eager_unfused", "always_fuse", "always_unfuse", "scic_selected"),
            measurements=measurements,
            chosen=chosen,
            report=report,
            emit_both=decision.emit_both,
            primary_mode=decision.primary.mode,
            fuse_epilogue=decision.primary.fuse_epilogue,
            epilogue_policy=decision.epilogue_policy,
            placeholder=False,
            backend="cuda-torch",
            device_name="NVIDIA A40",
            dtype_ran="fp16",
        )
        scic = next(row for row in rows if row["baseline"] == BASELINE_SCIC)
        self.assertEqual(scic["median_ms"], 0.4)
        self.assertEqual(scic["timing_copied_from"], BASELINE_ALWAYS_FUSE)
        self.assertEqual(scic["epilogue_policy"], POLICY_COMPILE_SDPA_CHAIN)
        self.assertIn("LIBRARY_SDPA_EPILOGUE_COMPILE_SDPA_CHAIN", scic["honesty"])
        self.assertIn("LIBRARY_SDPA_CHAIN_NOT_ANALYTICAL_PRIMARY", scic["honesty"])
        self.assertNotIn("RIDGE_BAND_EMIT_BOTH_TIMED_PRIMARY_ONLY", scic["honesty"])
        self.assertFalse(scic["quote_as_scic_accuracy"])

    def test_ridge_prefill_does_not_claim_the_analytical_primary(self) -> None:
        shape = next(s for s in ATTN_LEVELS["decode-prefill"] if s.label == "prefill_s512")
        report, decision, chosen = decide_attention(shape, A40_FP16, Thresholds())
        self.assertTrue(decision.emit_both)
        self.assertEqual(decision.primary.mode, "ComputeMode")
        self.assertEqual(chosen, BASELINE_ALWAYS_FUSE)
        measurements = {
            "eager_unfused": Measurement(2.0, True, "", "EAGER"),
            "always_fuse": Measurement(0.5, True, "", "FUSE"),
            "always_unfuse": Measurement(1.2, True, "", "UNFUSE"),
        }
        rows = assemble_latency_rows(
            op="attention",
            label="prefill_s512",
            shape="B=1,H=8,S=512,D=64",
            baselines=("scic_selected",),
            measurements=measurements,
            chosen=chosen,
            report=report,
            emit_both=decision.emit_both,
            primary_mode=decision.primary.mode,
            fuse_epilogue=decision.primary.fuse_epilogue,
            epilogue_policy=decision.epilogue_policy,
            placeholder=False,
            backend="cuda-torch",
            device_name="NVIDIA A40",
            dtype_ran="fp16",
        )
        scic = rows[0]
        self.assertEqual(scic["timing_copied_from"], BASELINE_ALWAYS_FUSE)
        self.assertNotIn("RIDGE_BAND_EMIT_BOTH_TIMED_PRIMARY_ONLY", scic["honesty"])
        self.assertIn("LIBRARY_SDPA_CHAIN_NOT_ANALYTICAL_PRIMARY", scic["honesty"])

    def test_dynamo_limit_is_a_noop_without_torch(self) -> None:
        self.assertIsNone(configure_dynamo_cache_limit())


class BakeoffCliTests(unittest.TestCase):
    def test_cpu_smoke_writes_the_schema(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "bakeoff"
            code = main(
                [
                    "bakeoff",
                    "--out",
                    str(dest),
                    "--device",
                    "cpu",
                    "--level",
                    "smoke",
                    "--attention",
                    "--skip-max-autotune",
                    "--machine",
                    "A40_FP16",
                ]
            )
            self.assertEqual(code, 0)
            summary = json.loads((dest / "bakeoff_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["gpu_measurement_status"], "PLACEHOLDER_NO_GPU")
            self.assertFalse(summary["nsight_used"])
            self.assertFalse(summary["quote_as_scic_accuracy"])
            self.assertFalse(summary["claims_fa3"])
            self.assertFalse(summary["claims_stage_stream"])
            self.assertTrue(summary["do_not_invent_numbers"])
            self.assertTrue((dest / "SEND_BACK.txt").is_file())
            with (dest / "bakeoff_latency.csv").open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertTrue(rows)
            for name in ("baseline", "shape", "median_ms", "speedup_vs_eager", "honesty"):
                self.assertIn(name, rows[0])
            baselines = {row["baseline"] for row in rows}
            self.assertIn("eager_unfused", baselines)
            self.assertIn("torch_compile_default", baselines)
            self.assertIn("torch_compile_max_autotune", baselines)
            self.assertIn("sdpa", baselines)
            self.assertIn("always_fuse", baselines)
            self.assertIn("always_unfuse", baselines)
            self.assertIn("scic_selected", baselines)
            for row in rows:
                self.assertEqual(row["quote_as_scic_accuracy"], "false")
                self.assertEqual(row["claims_fa3"], "false")
                self.assertEqual(row["nsight_used"], "false")
                self.assertEqual(row["claims_stage_stream"], "false")
                self.assertEqual(row["claims_custom_kernel"], "false")
                self.assertIn("NO_NSIGHT", row["honesty"])
                self.assertIn("NOT_AN_FA3_KERNEL", row["honesty"])
                if row["baseline"] == "eager_unfused" and row["ok"] == "true":
                    self.assertEqual(row["speedup_vs_eager"], "1")
            with (dest / "bakeoff_decisions.csv").open(encoding="utf-8") as fh:
                decisions = list(csv.DictReader(fh))
            gemm = next(row for row in decisions if row["label"] == "decode_m1" and row["op"] == "matmul_rmsnorm")
            self.assertEqual(gemm["selected_baseline"], "always_fuse")
            self.assertEqual(gemm["analytical_baseline"], "always_fuse")
            self.assertEqual(gemm["measured_winner"], "")
            self.assertEqual(gemm["selection_source"], "analytical")
            self.assertEqual(gemm["calibration_attempted"], "false")
            self.assertEqual(gemm["primary_mode"], "MemMode")
            self.assertEqual(gemm["quote_as_scic_accuracy"], "false")
            attn = [row for row in decisions if row["op"] == "attention"]
            self.assertTrue(attn)
            for row in attn:
                self.assertEqual(row["materialize_intermediate"], "false")
                self.assertEqual(row["claims_fa3"], "false")
                self.assertEqual(row["epilogue_policy"], "compile_sdpa_chain")
                self.assertEqual(row["selected_baseline"], "always_fuse")
                self.assertEqual(row["selection_source"], "library_sdpa")
                self.assertEqual(row["measured_winner"], "")
                self.assertIn("LIBRARY_SDPA_NOT_WALLCLOCK_CALIBRATED", row["calibration_honesty"])
            summary_cal = summary["gemm_calibration"]
            self.assertEqual(summary_cal["scope"], "ridge-compute")
            self.assertTrue(summary_cal["not_dram_ai"])
            self.assertTrue(summary_cal["gpu_contention_can_flip_winner"])
            self.assertFalse(summary_cal["quote_as_scic_accuracy"])

    def test_cpu_all_gemm_probe_falls_back_without_cuda_compile(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "bakeoff"
            code = main(
                [
                    "bakeoff",
                    "--out",
                    str(dest),
                    "--device",
                    "cpu",
                    "--level",
                    "smoke",
                    "--skip-max-autotune",
                    "--machine",
                    "A40_FP16",
                    "--calibrate",
                    "all-gemm",
                ]
            )
            self.assertEqual(code, 0)
            with (dest / "bakeoff_decisions.csv").open(encoding="utf-8") as fh:
                decisions = list(csv.DictReader(fh))
            gemm_rows = [row for row in decisions if row["op"] == "matmul_rmsnorm"]
            self.assertTrue(gemm_rows)
            for row in gemm_rows:
                self.assertEqual(row["calibration_attempted"], "true")
                self.assertEqual(row["selection_source"], "analytical_fallback")
                self.assertEqual(row["measured_winner"], "")
                self.assertEqual(row["selected_baseline"], row["analytical_baseline"])
                self.assertIn("WALLCLOCK_CALIBRATION_NOT_DRAM_AI", row["calibration_honesty"])
                self.assertIn("ANALYTICAL_FALLBACK_NO_MEASURED_WINNER", row["calibration_honesty"])
                self.assertIn("GPU_CONTENTION_CAN_FLIP_WINNER", row["calibration_honesty"])
                self.assertEqual(row["quote_as_scic_accuracy"], "false")
            decode = next(row for row in gemm_rows if row["label"] == "decode_m1")
            self.assertEqual(decode["analytical_baseline"], "always_fuse")
            self.assertEqual(decode["selected_baseline"], "always_fuse")

    def test_script_does_not_call_nsight(self) -> None:
        text = Path("scripts/run_baseline_bakeoff.sh").read_text(encoding="utf-8")
        self.assertIn("bakeoff", text)
        self.assertIn("SEND_BACK.txt", text)
        self.assertNotIn("nsight_dram.sh", text)
        self.assertNotIn("ncu ", text)
        self.assertIn("--require-cuda", text)
        self.assertIn("--attention", text)
        self.assertIn("cache_size_limit", text)
        self.assertIn("Idle GPU", text)
        self.assertIn("--calibrate", text)
        self.assertIn("not DRAM", text)
        readme = Path("README.md").read_text(encoding="utf-8")
        self.assertIn("bash scripts/run_baseline_bakeoff.sh", readme)
        self.assertIn("not DRAM", readme)
        self.assertIn("busy gpu", readme.lower())


if __name__ == "__main__":
    unittest.main()

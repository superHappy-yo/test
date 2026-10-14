import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ablation_analysis", ROOT/"paper/jsa/scripts/analyze_jsa_ablation.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
from shapefuse.io_utils import write_csv, write_json


class AblationAnalysisTests(unittest.TestCase):
    def test_candidate_validation_excludes_both_devices_of_the_held_out_seed(self):
        rows = []
        for gpu in (0,1):
            for run in (0,1,2):
                rows.append(dict(cluster=f"gpu{gpu}_run{run:02d}", process=f"gpu{gpu}_run{run:02d}/projection",
                    case="m1", native_host_ms=100 if run==0 else 1, native_gpu_ms=100 if run==0 else 1,
                    compiled_host_ms=1 if run==0 else 2, compiled_gpu_ms=1 if run==0 else 2))
        held = [r for r in analysis.cross_validate_candidates(rows) if r["held_out_seed"]==0]
        self.assertEqual(len(held),2)
        self.assertTrue(all(r["selected"]=="native" for r in held))
        self.assertTrue(all(r["training_pairs"]==4 for r in held))
        self.assertTrue(all(r["gpu_speedup"]==.01 for r in held))

    def test_archived_profiler_trial_supplies_only_structure_not_latency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = root/"device0/diagnostic_initial_gpu0_run00/system/b1_s128"
            dest.mkdir(parents=True)
            write_csv(dest/"latency.csv", [dict(policy="original",phase="prefill",execution_status="ok",
                                                gpu_median_ms=999,host_batch_ms_per_call=999)])
            write_json(dest/"completed.json",{})
            write_json(dest/"details.json",[dict(policy="original",execution_status="ok",
                profile_prefill=dict(cpu=[dict(name="Torch-Compiled Region",count=1)],
                                     cuda_event_count=109))])
            rows, profiles, statuses = analysis.collect(root)
            self.assertEqual(rows, [])
            self.assertEqual(statuses, [])
            self.assertEqual(len(profiles), 1)
            self.assertEqual(profiles[0]["cuda_events"],109)

    def test_model_pairs_do_not_mix_case_or_process_and_failed_rows_are_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for run, case, base_time, modified_time in ((0,"b1_s128",10,5),
                                                        (0,"b8_s128",100,100),
                                                        (1,"b1_s128",20,10)):
                dest = root/f"device0/gpu0_run{run:02d}/system/{case}"
                dest.mkdir(parents=True)
                rows = [dict(policy=policy,phase="prefill",execution_status=status,
                             gpu_median_ms=t,host_batch_ms_per_call=t)
                        for policy,t,status in (("original",base_time,"ok"),
                                                 ("fixed_compile",modified_time,"ok"),
                                                 ("bad",1,"failed"))]
                write_csv(dest/"latency.csv",rows)
                write_json(dest/"completed.json",{})
                write_json(dest/"details.json",[{"policy":r["policy"],"execution_status":r["execution_status"]} for r in rows])
            out = root/"analysis"
            out.mkdir()
            analysis.analyze(root,out)
            import csv
            with (out/"paired_ratios.csv").open() as handle:
                pairs = list(csv.DictReader(handle))
            selected = [r for r in pairs if r["baseline"]=="original" and r["policy"]=="fixed_compile"]
            self.assertEqual(sorted(float(r["host_speedup"]) for r in selected),[1,2,2])
            self.assertFalse(any(r["policy"]=="bad" for r in pairs))
            with (out/"summary.csv").open() as handle:
                summaries = list(csv.DictReader(handle))
            overall = next(r for r in summaries if r["baseline"]=="original" and
                           r["policy"]=="fixed_compile" and r["case"]=="all")
            self.assertEqual(int(overall["clusters"]),2)
            self.assertEqual(int(overall["pairs"]),3)


if __name__ == "__main__":
    unittest.main()

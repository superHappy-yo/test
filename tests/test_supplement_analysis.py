import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


class SupplementAnalysisTests(unittest.TestCase):
    def test_serving_rejects_a_mismatch_in_the_last_request(self):
        script=Path(__file__).resolve().parents[1]/"paper/jsa/scripts/analyze_jsa_vllm.py"
        spec=importlib.util.spec_from_file_location("serving_analysis",script)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"raw";out=Path(tmp)/"analysis";out.mkdir()
            for policy in ("original","scic"):
                process=root/"vllm_gpu1"/"gpu1_run00"/policy;process.mkdir(parents=True)
                (process/"completed.json").write_text('{}')
                (process/"manifest.json").write_text(json.dumps({'visible_devices':'1'}))
                requests=[dict(request=i,output_sha256=f'tokens_{i}',output_tokens=64) for i in range(128)]
                if policy=='scic':requests[-1]['output_sha256']='mismatched_final_request'
                (process/"details.json").write_text(json.dumps([dict(case='s128_c1',requests=requests)]))
                row=dict(case='s128_c1',policy=policy,seed=0,input_length=128,concurrency=1,
                         requests=128,execution_status='ok',output_tokens_per_second=1000,
                         ttft_p50_ms=1,ttft_p95_ms=2,itl_p50_ms=1,itl_p95_ms=2,
                         latency_p50_ms=64,latency_p95_ms=80)
                with (process/"latency.csv").open('w') as f:
                    writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerow(row)
            module.analyze(root,out)
            result=json.loads((out/'serving.json').read_text())
            self.assertEqual(result['rejected_pairs'],1)
            self.assertNotIn('scic',{r['policy'] for r in result['global_summary']})

    def test_pairs_shapes_separately_in_a_batched_process(self):
        script=Path(__file__).resolve().parents[1]/"paper/jsa/scripts/analyze_jsa_supplement.py"
        spec=importlib.util.spec_from_file_location("supplement_analysis",script)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"raw";out=Path(tmp)/"analysis";out.mkdir()
            process=root/"torch214_operators"/"gpu0_run00"/"none"/"all"
            process.mkdir(parents=True)
            (process/"completed.json").write_text(json.dumps({"cases":2}))
            rows=[]
            for case,native,selected in (("small",4,1),("large",8,4)):
                for method,latency in (("native",native),("compile_default",selected),("scic_ridge-compute",selected)):
                    rows.append(dict(case=case,op="gemm",method=method,graph_mode="none",
                                     execution_status="ok",gpu_median_ms=latency,
                                     host_batch_ms_per_call=latency,calibration_ms=""))
            with (process/"latency.csv").open("w") as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            module.summarize(root,out)
            with (out/"paired_ratios.csv").open() as f:
                pairs=list(csv.DictReader(f))
            selected={r["case"]:float(r["speedup_vs_native"]) for r in pairs if r["method"]=="scic_ridge-compute"}
            self.assertEqual(selected,{"small":4,"large":2})
            self.assertEqual(len(pairs),6)


if __name__=="__main__": unittest.main()

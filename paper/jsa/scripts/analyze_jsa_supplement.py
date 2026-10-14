#!/usr/bin/env python3
"""Paired, process-level summaries; bootstrap processes, not timing iterations."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from shapefuse.io_utils import write_csv,write_json


def geomean(values):
    return math.exp(sum(math.log(x) for x in values)/len(values)) if values else None


def collect(root):
    rows=[]
    for path in root.rglob("latency.csv"):
        if "smoke" in str(path) or ".incomplete_" in str(path):
            continue
        if not (path.parent/"completed.json").exists():
            continue
        with path.open() as f:
            case_rows=list(csv.DictReader(f))
        for row in case_rows:
            row["path"]=str(path.relative_to(root))
            row["suite"]=path.relative_to(root).parts[0]
            row["family"]=("torch251" if row["suite"].startswith("torch251") else
                           "torch214" if row["suite"].startswith("torch214") else row["suite"])
            row["process"]=str(path.parent.relative_to(root))
            for k in ("gpu_median_ms","host_batch_ms_per_call","calibration_ms","setup_ms",
                      "tokens_per_second","output_tokens_per_second"):
                if row.get(k):
                    row[k]=float(row[k])
            rows.append(row)
    return rows


def summarize(root,out):
    rows=collect(root)
    write_csv(out/"all_latency.csv",rows)
    status=[]
    for suite in sorted({r["suite"] for r in rows}):
        selected=[r for r in rows if r["suite"]==suite]
        for method in sorted({r.get("method") or r.get("policy") for r in selected}):
            candidates=[r for r in selected if (r.get("method") or r.get("policy"))==method]
            status.append(dict(suite=suite,method=method,attempts=len(candidates),
                               admitted=sum(r["execution_status"]=="ok" for r in candidates),
                               rejected=sum(r["execution_status"]!="ok" for r in candidates)))
    write_csv(out/"candidate_status.csv",status)
    calibration=[]
    for key in sorted({(r["family"],r.get("graph_mode"),r.get("method")) for r in rows
                       if isinstance(r.get("calibration_ms"),(int,float))}):
        selected=[r for r in rows if (r["family"],r.get("graph_mode"),r.get("method"))==key
                  and isinstance(r.get("calibration_ms"),(int,float)) and r["execution_status"]=="ok"]
        values=[r["calibration_ms"] for r in selected if r["calibration_ms"]>0]
        calibration.append(dict(suite=key[0],graph=key[1],method=key[2],rows=len(selected),
                                probed=len(values),median_ms=float(np.median(values)) if values else 0,
                                p95_ms=float(np.percentile(values,95)) if values else 0,
                                selected_native=sum(r.get("selected_candidate")=="native" for r in selected),
                                selected_compile=sum(r.get("selected_candidate")=="compile_default" for r in selected)))
    write_csv(out/"calibration_summary.csv",calibration)
    grouped={}
    for row in rows:
        if row["execution_status"]!="ok":
            continue
        if row.get("method"):
            key=(row["family"],row["process"],row["graph_mode"],row["op"],row["case"])
            grouped.setdefault(key,{})[row["method"]]=row
    pairs=[]
    for key,methods in grouped.items():
        # Pair within both process and shape, including batched operator runs.
        native=methods.get("native")
        if not native:
            continue
        pool=[r["gpu_median_ms"] for name,r in methods.items() if not name.startswith("scic_")]
        for name,row in methods.items():
            pair=dict(suite=key[0],process=key[1],graph=key[2],op=key[3],
                              case=row["case"],method=name,
                              speedup_vs_native=native["gpu_median_ms"]/row["gpu_median_ms"],
                              host_speedup_vs_native=native["host_batch_ms_per_call"]/row["host_batch_ms_per_call"],
                              gap_to_measured_pool=row["gpu_median_ms"]/min(pool),
                              calibration_ms=row.get("calibration_ms"),
                              selected=row.get("selected_candidate"))
            for candidate in ("compile_default","compile_max_no_graph"):
                if candidate in methods:
                    pair[f"speedup_vs_{candidate}"]=methods[candidate]["gpu_median_ms"]/row["gpu_median_ms"]
            pairs.append(pair)
    write_csv(out/"paired_ratios.csv",pairs)
    summaries=[]
    keys=sorted({(p["suite"],p["graph"],p["op"],p["method"]) for p in pairs})
    for key in keys:
        selected=[p for p in pairs if (p["suite"],p["graph"],p["op"],p["method"])==key]
        # Resample independent GPU/process-run clusters across a fixed shape set.
        clusters={}
        for p in selected:
            parts=Path(p["process"]).parts
            cluster=next((v for v in parts if v.startswith("gpu")),p["process"])
            clusters.setdefault(cluster,[]).append(p["speedup_vs_native"])
        per_run=[geomean(v) for v in clusters.values()]
        rng=np.random.default_rng(0)
        boots=[geomean(rng.choice(per_run,size=len(per_run)).tolist()) for _ in range(2000)]
        summary=dict(suite=key[0],graph=key[1],op=key[2],method=key[3],
                         rows=len(selected),cases=len({p["case"] for p in selected}),clusters=len(clusters),
                         geomean_speedup_vs_native=geomean([p["speedup_vs_native"] for p in selected]),
                         geomean_host_speedup_vs_native=geomean([p["host_speedup_vs_native"] for p in selected]),
                         geomean_gap_to_measured_pool=geomean([p["gap_to_measured_pool"] for p in selected]),
                         ci95_low=float(np.percentile(boots,2.5)),ci95_high=float(np.percentile(boots,97.5)))
        for candidate in ("compile_default","compile_max_no_graph"):
            values=[p[f"speedup_vs_{candidate}"] for p in selected if f"speedup_vs_{candidate}" in p]
            summary[f"paired_rows_vs_{candidate}"]=len(values)
            summary[f"geomean_speedup_vs_{candidate}"]=geomean(values)
            paired_clusters={}
            for p in selected:
                metric=f"speedup_vs_{candidate}"
                if metric not in p:
                    continue
                cluster=next((v for v in Path(p["process"]).parts if v.startswith("gpu")),p["process"])
                paired_clusters.setdefault(cluster,[]).append(p[metric])
            cluster_ratios=[geomean(v) for v in paired_clusters.values()]
            if cluster_ratios:
                rng=np.random.default_rng(0)
                interval=[geomean(rng.choice(cluster_ratios,len(cluster_ratios)).tolist()) for _ in range(2000)]
                summary[f"ci95_low_vs_{candidate}"]=float(np.percentile(interval,2.5))
                summary[f"ci95_high_vs_{candidate}"]=float(np.percentile(interval,97.5))
        summaries.append(summary)
    write_csv(out/"operator_summary.csv",summaries)
    models=[]
    for row in rows:
        if not row.get("policy") or not row.get("phase") or row["execution_status"]!="ok":
            continue
        native=next((r for r in rows if r["process"]==row["process"] and r.get("policy")=="original"
                     and r.get("phase")==row["phase"]),None)
        if native:
            models.append(dict(suite=row["suite"],process=row["process"],batch=row["batch"],
                          prompt_length=row["prompt_length"],phase=row["phase"],policy=row["policy"],
                          host_speedup=native["host_batch_ms_per_call"]/row["host_batch_ms_per_call"],
                          gpu_speedup=native["gpu_median_ms"]/row["gpu_median_ms"]))
    write_csv(out/"system_paired_ratios.csv",models)
    system_summary=[]
    for key in sorted({(r["batch"],r["prompt_length"],r["phase"],r["policy"]) for r in models}):
        selected=[r for r in models if (r["batch"],r["prompt_length"],r["phase"],r["policy"])==key]
        values=[r["host_speedup"] for r in selected]
        rng=np.random.default_rng(0)
        bootstrap=[geomean(rng.choice(values,len(values)).tolist()) for _ in range(2000)]
        native=[r["host_batch_ms_per_call"] for r in rows if r.get("policy")=="original"
                and r.get("batch")==key[0] and r.get("prompt_length")==key[1]
                and r.get("phase")==key[2] and r["execution_status"]=="ok"]
        system_summary.append(dict(batch=key[0],prompt_length=key[1],phase=key[2],policy=key[3],
                            pairs=len(values),geomean_host_speedup=geomean(values),
                            ci95_low=float(np.percentile(bootstrap,2.5)),
                            ci95_high=float(np.percentile(bootstrap,97.5)),
                            native_median_ms=float(np.median(native))))
    write_csv(out/"system_summary.csv",system_summary)
    write_json(out/"summary.json",dict(rows=len(rows),successful=sum(r["execution_status"]=="ok" for r in rows),
               operators=summaries,system=models,system_summary=system_summary,
               limitations=["Kernel fusion requires inspected generated code; compiled chain is not proof.",
                            "Pool oracle is a measured candidate minimum, not a global optimal schedule.",
                            "Random-weight model profiling does not measure language task accuracy."]))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",default="paper/data/jsa_supplement")
    p.add_argument("--out",default="paper/data/jsa_supplement/analysis")
    args=p.parse_args()
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    summarize(Path(args.root),out)

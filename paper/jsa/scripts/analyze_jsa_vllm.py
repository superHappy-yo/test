#!/usr/bin/env python3
"""Pair serving measurements only after validating every generated token hash."""
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


def gm(values):
    return math.exp(sum(math.log(v) for v in values)/len(values))


def interval(values):
    rng=np.random.default_rng(0)
    boot=[gm(rng.choice(values,len(values)).tolist()) for _ in range(2000)]
    return float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))


def analyze(root,out):
    records=[]
    for suite in sorted(root.glob("vllm_gpu*")):
        for path in sorted(suite.rglob("latency.csv")):
            if not (path.parent/"completed.json").exists():continue
            manifest=json.loads((path.parent/"manifest.json").read_text())
            details={d['case']:d for d in json.loads((path.parent/"details.json").read_text())}
            with path.open() as f:
                case_rows=list(csv.DictReader(f))
            for row in case_rows:
                if row['execution_status']!='ok':continue
                row.update(gpu=manifest['visible_devices'],path=str(path.relative_to(root)))
                for key in ('output_tokens_per_second','ttft_p50_ms','ttft_p95_ms','itl_p50_ms',
                            'itl_p95_ms','latency_p50_ms','latency_p95_ms'):
                    row[key]=float(row[key])
                row['tokens']={s['request']:s['output_sha256'] for s in details[row['case']]['requests']}
                if int(row['requests'])!=128 or len(row['tokens'])!=128:
                    raise ValueError(f"Formal serving measurement requires 128 requests: {path}")
                records.append(row)
    baselines={(r['gpu'],r['seed'],r['case']):r for r in records if r['policy']=='original'}
    pairs=[]
    for row in records:
        native=baselines.get((row['gpu'],row['seed'],row['case']))
        if native is None:continue
        equal=row['tokens']==native['tokens']
        pair={k:row[k] for k in ('gpu','seed','case','policy','input_length','concurrency','path')}
        pair.update(tokens_identical=equal,requests=int(row['requests']),
                    execution_status='ok' if equal else 'numerical_rejected',
                    throughput_ratio=row['output_tokens_per_second']/native['output_tokens_per_second'],
                    ttft_p95_speedup=native['ttft_p95_ms']/row['ttft_p95_ms'],
                    itl_p95_speedup=native['itl_p95_ms']/row['itl_p95_ms'],
                    native_output_tokens_per_second=native['output_tokens_per_second'],
                    output_tokens_per_second=row['output_tokens_per_second'],
                    native_ttft_p95_ms=native['ttft_p95_ms'],ttft_p95_ms=row['ttft_p95_ms'],
                    native_itl_p95_ms=native['itl_p95_ms'],itl_p95_ms=row['itl_p95_ms'])
        pairs.append(pair)
    write_csv(out/'paired_serving.csv',pairs)
    summaries=[]
    for key in sorted({(p['case'],p['policy']) for p in pairs if p['execution_status']=='ok'}):
        chosen=[p for p in pairs if (p['case'],p['policy'])==key and p['execution_status']=='ok']
        values=[p['throughput_ratio'] for p in chosen];lo,hi=interval(values)
        result=dict(case=key[0],policy=key[1],pairs=len(chosen),
                    gpus=','.join(sorted({p['gpu'] for p in chosen})),
                    input_length=chosen[0]['input_length'],concurrency=chosen[0]['concurrency'],
                    geomean_throughput_ratio=gm(values),ci95_low=lo,ci95_high=hi,
                    geomean_ttft_p95_speedup=gm([p['ttft_p95_speedup'] for p in chosen]),
                    geomean_itl_p95_speedup=gm([p['itl_p95_speedup'] for p in chosen]))
        for field in ('native_output_tokens_per_second','output_tokens_per_second',
                      'native_ttft_p95_ms','ttft_p95_ms','native_itl_p95_ms','itl_p95_ms'):
            result[field]=float(np.median([p[field] for p in chosen]))
        summaries.append(result)
    write_csv(out/'serving_summary.csv',summaries)
    global_summary=[]
    for policy in sorted({p['policy'] for p in pairs if p['execution_status']=='ok'}):
        selected=[p for p in pairs if p['policy']==policy and p['execution_status']=='ok']
        clusters={}
        for p in selected:clusters.setdefault((p['gpu'],p['seed']),[]).append(p['throughput_ratio'])
        values=[gm(v) for v in clusters.values()];lo,hi=interval(values)
        global_summary.append(dict(policy=policy,pairs=len(selected),clusters=len(clusters),
                                   cases=len({p['case'] for p in selected}),
                                   geomean_throughput_ratio=gm([p['throughput_ratio'] for p in selected]),
                                   ci95_low=lo,ci95_high=hi))
    write_json(out/'serving.json',dict(measurement_rows=len(records),paired_rows=len(pairs),
               rejected_pairs=sum(p['execution_status']!='ok' for p in pairs),
               summaries=summaries,global_summary=global_summary,
               limitations=['Random dummy weights; no task-accuracy claim.',
                            'Fixed batches of token-ID requests; no production arrival distribution.',
                            'Batch throughput includes deterministic prompt construction; TTFT starts after construction.',
                            'AsyncLLM includes client/IPC/scheduling/sampling; HTTP and tokenizer are excluded.',
                            'ITL uses client-observed streamed arrivals; coalesced tokens share timestamps.',
                            'Displayed p95 is the median of per-process p95 values, not a pooled percentile.']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='paper/data/jsa_supplement')
    p.add_argument('--out',default='paper/data/jsa_supplement/analysis');args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);analyze(Path(args.root),out)

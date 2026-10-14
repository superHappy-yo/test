#!/usr/bin/env python3
"""Acceptance uses independent GPU/run clusters and strict numerical validity."""
import argparse,collections,json,math
from pathlib import Path
import numpy as np


def summarize(items):
    # Input pairs (cluster, positive ratio); collapse within process first.
    groups=collections.defaultdict(list)
    for cluster,value in items:
        if value>0 and math.isfinite(value):groups[cluster].append(math.log(value))
    if not groups:return dict(n=0,clusters=0,geomean=None,ci95=None)
    vals=np.array([np.mean(v) for _,v in sorted(groups.items())]);rng=np.random.default_rng(20261013)
    res=np.exp(np.mean(rng.choice(vals,(5000,len(vals)),replace=True),axis=1))
    return dict(n=len(items),clusters=len(vals),geomean=float(np.exp(vals.mean())),ci95=np.quantile(res,[.025,.975]).tolist())


def analyze(root):
    exclusions=json.loads((root/'exclusions.json').read_text()) if (root/'exclusions.json').exists() else {}
    operator=[];selections=[];systems=[];serving=[];preparations=[];deployments=[];failures=[];counts=collections.Counter();cost_records={}
    for run in sorted(root.glob('gpu*_run*')):
        cluster=run.name
        for stage in ('operator','operator-vllm','operator-legacy'):
            dest=run/stage
            if not (dest/'completed.json').exists() or f'{cluster}/{stage}' in exclusions:continue
            rows=json.loads((dest/'latencies.json').read_text())
            operator.extend(dict(cluster=cluster,domain=stage,**r) for r in rows)
            counts[stage]+=1
            if (dest/'selections.json').exists():
                selections.extend(dict(cluster=cluster,domain=stage,**r) for r in json.loads((dest/'selections.json').read_text()))
            for r in json.loads((dest/'records.json').read_text()):
                if 'costs' in r:cost_records[(cluster,stage,r['label'],r['mode'],r['reuse'])]=r
                if r.get('status') in ('failed','timing-failed'):failures.append(dict(cluster=cluster,domain=stage,**r))
                for name,audit in r.get('audit',{}).items() if isinstance(r.get('audit'),dict) else []:
                    if not audit['passed']:counts[f'{stage}_rejected_{name}']+=.25
        for stage in ('hf','hf-pretrained'):
            dest=run/stage
            if not (dest/'completed.json').exists():continue
            counts[stage]+=1
            systems.extend(dict(cluster=cluster,domain=stage,**r) for r in json.loads((dest/'latencies.json').read_text()))
            records=json.loads((dest/'records.json').read_text())
            failures.extend(dict(cluster=cluster,domain=stage,**r) for r in records if r['status']!='passed')
            for r in records:
                if 'setup_ms' not in r:continue
                costs=r.get('preparation',[])
                endpoints=[v['selected'] for v in costs if 'selected' in v]
                if endpoints:deployments.append(dict(domain=stage,cluster=cluster,case=r['case'],policy=r['policy'],
                    strict=r['status']=='passed',native=endpoints.count('native'),endpoints=len(endpoints)))
                preparations.append(dict(cluster=cluster,domain=stage,case=r['case'],policy=r['policy'],
                    numerics_passed=r['status']=='passed',setup_ms=r['setup_ms'],
                    preparation_ms=sum(v.get('preparation_ms',0) for v in costs),
                    compilation_ms=sum(v.get('compilation_ms',0) for v in costs),
                    validation_ms=sum(v.get('validation_ms',0) for v in costs),
                    probe_ms=sum(v.get('probe_ms',0) for v in costs)))
        for stage in ('serving','serving-pretrained'):
            originals={}
            reference=run/stage/'original'/'requests.json'
            if reference.exists():
                for r in json.loads(reference.read_text()):
                    originals[r['case']]=[v['output_sha256'] for v in r['requests']]
            for dest in sorted((run/stage).glob('*')) if (run/stage).exists() else []:
                if not dest.is_dir() or not (dest/'completed.json').exists():continue
                counts[stage+'_workers']+=1
                details={r['case']:[v['output_sha256'] for v in r['requests']] for r in json.loads((dest/'requests.json').read_text())}
                if (dest/'preparation.json').exists():
                    reports=json.loads((dest/'preparation.json').read_text())
                    loaded=[v for report in reports for v in report.get('prepared',[])]
                    layers={v['layer']:v.get('deployed') for v in loaded if 'deployed' in v}
                    if layers:deployments.append(dict(domain=stage,cluster=cluster,case='all generation workloads',policy=dest.name,
                        strict=None,native=sum(v=='native' for v in layers.values()),endpoints=len(layers)))
                for r in json.loads((dest/'latencies.json').read_text()):
                    matched=details[r['case']]==originals.get(r['case'])
                    serving.append(dict(cluster=cluster,domain=stage,hashes_match=matched,**r))
                    if not matched:failures.append(dict(cluster=cluster,domain=stage,case=r['case'],policy=r['policy'],status='token-hash-mismatch'))
    # Failed engine starts do not produce request files and must remain visible.
    for path in sorted(root.glob('runs_serving*.json')):
        for r in json.loads(path.read_text()):
            if r['status']!='completed':failures.append(dict(domain=r['stage'],cluster=f"gpu{r['gpu']}_run{r['seed']:02d}",
                policy=r['policy'],status=r['status'],seconds=r['seconds'],source=str(path)))
    result=dict(protocol='5000 GPU/run cluster bootstraps; no cross-domain pooling',counts=dict(counts),operator=[],selection=[],system=[],serving=[],preparation=[],deployment=deployments,failures=len(failures))
    for domain in ('operator','operator-vllm'):
        data=[r for r in operator if r['domain']==domain]
        index={(r['cluster'],r['label'],r['mode'],r['reuse'],r['candidate']):r for r in data}
        for mode in ('direct','graph'):
            for reuse in ('same','rotated'):
                subset=[r for r in data if r['mode']==mode and r['reuse']==reuse]
                for name in ('eager','max-autotune','split1','split4','split8','blas4','blas8','legacy-scic','max-autotune-precision'):
                    pairs=[]
                    comparison=subset if name not in ('legacy-scic','max-autotune-precision') else [r for r in operator if r['domain']=='operator-legacy' and r['mode']==mode and r['reuse']==reuse and domain=='operator']
                    for r in comparison:
                        if r['candidate']!=name:continue
                        baseline=index.get((r['cluster'],r['label'],mode,reuse,'compile-default'))
                        if baseline:pairs.append((r['cluster'],baseline['host_ms']/r['host_ms']))
                    result['operator'].append(dict(domain=domain,mode=mode,reuse=reuse,candidate=name,baseline='compile-default',**summarize(pairs)))
                sel=[r for r in selections if r['domain']==domain and r['mode']==mode and r['reuse']==reuse and 'policy' in r]
                for policy in ('scic-v2','scic-v2-no-ai'):
                    pairs=[];total=[];probe=[];ai=[];fixed=[];coverage=0
                    for r in sel:
                        if r['policy']!=policy:continue
                        full=next((v for v in sel if v['cluster']==r['cluster'] and v['label']==r['label'] and v['policy']=='full-calibration'),None)
                        if not full:continue
                        coverage+=1;pairs.append((r['cluster'],full['selected_host_ms']/r['selected_host_ms']))
                        static=index.get((r['cluster'],r['label'],mode,reuse,r['static'])) or index.get((r['cluster'],r['label'],mode,reuse,'eager'))
                        if static:fixed.append((r['cluster'],static['host_ms']/r['selected_host_ms']))
                        rec=cost_records[(r['cluster'],domain,r['label'],mode,reuse)]
                        common=sum(v['compilation_ms']+v['validation_ms'] for n,v in rec['costs'].items() if n.startswith(('blas','split')))
                        total.append((r['cluster'],(common+full['probe_ms'])/(common+r['probe_ms'])))
                        if r['probe_ms']>0:probe.append((r['cluster'],full['probe_ms']/r['probe_ms']))
                        if policy=='scic-v2':
                            noai=next((v for v in sel if v['cluster']==r['cluster'] and v['label']==r['label'] and v['policy']=='scic-v2-no-ai'),None)
                            if noai:ai.append((r['cluster'],noai['selected_host_ms']/r['selected_host_ms']))
                    speed=summarize(pairs);prep=summarize(total)
                    result['selection'].append(dict(domain=domain,mode=mode,reuse=reuse,policy=policy,coverage=coverage,
                        steady_vs_full=speed,steady_vs_fixed=summarize(fixed),total_preparation_speed=prep,probe_speed=summarize(probe),ai_vs_no_ai=summarize(ai),
                        acceptance=bool(speed['geomean'] and speed['geomean']>=.98 and prep['geomean']>=1/.7)))
    for domain in ('hf','hf-pretrained'):
        prep=[r for r in preparations if r['domain']==domain]
        for policy in ('best-fixed','legacy','scic-v2','scic-v2-no-ai'):
            pairs=[];setup=[];valid=0
            for r in prep:
                if r['policy']!=policy:continue
                full=next((b for b in prep if b['cluster']==r['cluster'] and b['case']==r['case'] and b['policy']=='full-calibration'),None)
                if not full:continue
                if r['preparation_ms']>0:pairs.append((r['cluster'],full['preparation_ms']/r['preparation_ms']))
                setup.append((r['cluster'],full['setup_ms']/r['setup_ms']))
                valid+=r['numerics_passed'] and full['numerics_passed']
            result['preparation'].append(dict(domain=domain,policy=policy,component_preparation_speed=summarize(pairs),
                setup_speed=summarize(setup),strict_valid_pairs=valid,
                interpretation='Actual lazy preparation under the observed compiler cache state; descriptive even when logits fail; not a cold-cache startup or acceleration claim.'))
        data=[r for r in systems if r['domain']==domain]
        for phase in ('prefill','decode64'):
            for policy in ('best-fixed','legacy','full-calibration','scic-v2','scic-v2-no-ai'):
                pairs=[];exploratory=[]
                for r in data:
                    if r['phase']!=phase or r['policy']!=policy:continue
                    base=next((b for b in data if b['cluster']==r['cluster'] and b['case']==r['case'] and b['phase']==phase and b['policy']=='original'),None)
                    if not base:continue
                    pair=(r['cluster'],base['host_ms']/r['host_ms'])
                    exploratory.append(pair)
                    if r['numerics_passed']:pairs.append(pair)
                strict=summarize(pairs)
                complete=len(pairs)==len([b for b in data if b['policy']=='original' and b['phase']==phase])
                result['system'].append(dict(domain=domain,phase=phase,policy=policy,strict=strict,exploratory=summarize(exploratory),
                    complete_numerical_coverage=complete,acceleration_claim=bool(complete and strict['clusters']>=3 and strict['geomean'] and strict['geomean']>=1.02 and strict['ci95'][0]>1)))
    for domain in ('serving','serving-pretrained'):
        data=[r for r in serving if r['domain']==domain]
        for policy in ('best-fixed','legacy','full-calibration','scic-v2','scic-v2-no-ai'):
            pairs=[];ttft=[];itl=[]
            for r in data:
                if r['policy']!=policy or not r['hashes_match']:continue
                base=next((b for b in data if b['cluster']==r['cluster'] and b['case']==r['case'] and b['policy']=='original'),None)
                if base:
                    pairs.append((r['cluster'],r['throughput']/base['throughput']))
                    ttft.append((r['cluster'],base['ttft_p50_ms']/r['ttft_p50_ms']))
                    if r['itl_p50_ms']>0:itl.append((r['cluster'],base['itl_p50_ms']/r['itl_p50_ms']))
            throughput=summarize(pairs)
            result['serving'].append(dict(domain=domain,policy=policy,throughput=throughput,ttft=summarize(ttft),itl=summarize(itl),
                complete_generation_coverage=len(pairs)==len([b for b in data if b['policy']=='original']),generation_speed_claim=bool(len(pairs)==len([b for b in data if b['policy']=='original']) and throughput['clusters']>=3 and throughput['geomean'] and throughput['geomean']>=1.02 and throughput['ci95'][0]>1),
                component_validation_only=True,task_accuracy_measured=False))
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (root/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='paper/data/jsa_v2');a=p.parse_args()
    result=analyze(Path(a.root));print(json.dumps(result['counts'],indent=2));print('Failures',result['failures'])

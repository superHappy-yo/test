#!/usr/bin/env python3
"""Describe training support and uncertainty without fitting to held-out results."""
import json, math, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from shapefuse.v2_policy import features
root=ROOT/'paper/data/jsa_v2'
excluded=json.loads((root/'exclusions.json').read_text())
results=[]
for path in sorted(root.glob('gpu*_run*/operator*/priors.json')):
 relative=str(path.parent.relative_to(root))
 if relative in excluded or path.parent.name not in ('operator','operator-vllm'):continue
 if not (path.parent/'completed.json').exists():continue
 rows=json.loads((path.parent/'latencies.json').read_text())
 priors=json.loads(path.read_text())
 for policy,prior in priors.items():
  for candidate,model in prior['models'].items():
   training=[r for r in rows if r['train'] and r['candidate']==candidate]
   scaled=np.asarray(model['scale']);coef=np.asarray(model['coef'])
   design=np.asarray([features(r['M'],r['N'],r['K'],candidate,r['mode'],r['reuse'],analytical=prior['analytical']) for r in training])
   centered=design-design.mean(axis=0)
   results.append(dict(worker=relative,policy=policy,candidate=candidate,
     training_geometries=len({(r['M'],r['N'],r['K']) for r in training}),
     training_rows=len(training),feature_columns=len(scaled),constant_columns=int((design.std(axis=0)<1e-12).sum()),
     centered_design_rank=int(np.linalg.matrix_rank(centered)),
     loog_error_log=model['error_log'],multiplicative_error=math.exp(min(model['error_log'],100)),
     coefficient_l2=float(np.linalg.norm(coef[1:])),
     note='LOOG residual margin is empirical, not a calibrated confidence interval.'))
(root/'prior_audit.json').write_text(json.dumps({'uses_held_out_results':False,'rows':results},indent=2)+'\n')
for policy in ('scic-v2','scic-v2-no-ai'):
 for name in ('split1','split4','split8','blas4','blas8'):
  rows=[r for r in results if r['policy']==policy and r['candidate']==name]
  if rows:print(policy,name,'train geometries',sorted({r['training_geometries'] for r in rows}),'LOOG multiplicative median',round(float(np.median([r['multiplicative_error'] for r in rows])),2))

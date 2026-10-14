"""Training-only latency prior and explicit preparation; no hot-path lookup."""
from dataclasses import dataclass
import hashlib
import math
import time
import numpy as np

VERSION = 'scic-v2-policy-2'
POLICIES = ('legacy','full-calibration','scic-v2','scic-v2-no-ai')


def features(m,n,k,name,mode,reuse, *, analytical=True):
    split=int(name[5:]) if name.startswith('split') else 1
    warps=int(name[4:]) if name.startswith('blas') else 4
    base=[math.log2(v) for v in (m,n,k)]+[float(mode=='graph'),float(reuse=='rotated'),
          math.log2(split),warps/4,float(name.startswith('split')), math.log2(m*n),math.log2(n*k)]
    if analytical:
        flops=2*m*n*k+5*m*n+2*m+((split-1)*m*n if name.startswith('split') else 0)
        # Explicit cold-traffic assumption, including FP32 split partials.
        traffic=2*(m*k+n*k+3*m*n+n)+(8*split*m*n if name.startswith('split') else 4*m*n)
        roof=max(flops/149.7e12,traffic/696e9)*1e3
        base += [math.log(flops),math.log(traffic),math.log(roof),math.log(flops/traffic/(149.7e12/696e9))]
    return base


class LatencyPrior:
    def __init__(self, *, analytical=True):
        self.analytical=analytical
        self.models={}
    def fit(self, rows):
        rows=[r for r in rows if r.get('train',True)]
        for name in sorted({r['candidate'] for r in rows}):
            subset=[r for r in rows if r['candidate']==name and r.get('passed',True)]
            x=np.asarray([features(r['M'],r['N'],r['K'],name,r['mode'],r['reuse'],analytical=self.analytical) for r in subset])
            y=np.log([r['host_ms'] for r in subset])
            if len(y)<3: continue
            def fit_arrays(xx,yy):
                mean=xx.mean(0);scale=xx.std(0);scale[scale<1e-12]=1
                z=np.column_stack([np.ones(len(xx)),(xx-mean)/scale])
                penalty=np.eye(z.shape[1])*1e-3;penalty[0,0]=0
                coef=np.linalg.solve(z.T@z+penalty,z.T@yy)
                return mean,scale,coef
            residuals=[]
            groups=[(r['M'],r['N'],r['K']) for r in subset]
            for group in sorted(set(groups)):
                train=np.array([v!=group for v in groups]);test=~train
                if train.sum()<3:continue
                mu,sd,coef=fit_arrays(x[train],y[train])
                residuals.extend(abs(y[test]-np.column_stack([np.ones(test.sum()),(x[test]-mu)/sd])@coef))
            mu,sd,coef=fit_arrays(x,y)
            self.models[name]=(mu,sd,coef,float(np.quantile(residuals,.95)) if residuals else float('inf'))
        return self
    def predict(self, shape,mode,reuse):
        result={}
        for name,(mu,sd,coef,err) in self.models.items():
            x=np.array(features(*shape,name,mode,reuse,analytical=self.analytical))
            log=float(np.r_[1,(x-mu)/sd]@coef)
            result[name]={'ms':math.exp(np.clip(log,-30,30)), 'lower':math.exp(np.clip(log-err,-30,30)), 'error_log':err}
        return result
    def dump(self):
        return {'feature_version':VERSION,'analytical':self.analytical,'ridge':1e-3,'models':{n:{'mean':mu.tolist(),'scale':sd.tolist(),'coef':coef.tolist(),'error_log':err} for n,(mu,sd,coef,err) in self.models.items()}}


@dataclass
class PreparedPlan:
    candidate: str
    fn: object
    key: tuple
    probe: dict
    predicted: dict
    probe_ms: float
    validation_ms: float
    compilation_ms: float
    fallback: str=''


def signature(inputs, *, semantic='hf',epsilon=1e-6,weight_version=0):
    import torch
    from shapefuse.v2_kernels import VERSION as kv
    import triton
    from pathlib import Path
    source_hash=hashlib.sha256(Path(__file__).with_name('v2_kernels.py').read_bytes()).hexdigest()
    return (VERSION,kv,source_hash,triton.__version__,torch.version.cuda,torch.__version__,str(torch.cuda.get_device_properties(inputs[0].device)),
            semantic,epsilon,weight_version,torch.get_float32_matmul_precision(),
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            getattr(torch.backends.cuda.matmul,'allow_fp16_accumulation',False),
            tuple((tuple(t.shape),tuple(t.stride()),str(t.dtype),str(t.device),t._version if not t.is_inference() else 'inference',t.data_ptr()) for t in inputs))


def choose(policy,available,prior,shape,mode,reuse,static,probe,*,expected_calls=1024):
    """probe(name) returns host latency and elapsed probe time. Validation precedes this."""
    if not available: raise ValueError('No validated candidate')
    static=static if static in available else available[0]
    pred=prior.predict(shape,mode,reuse) if prior else {}
    samples={};cost=0.
    def sample(name):
        nonlocal cost
        if name not in samples:
            latency,elapsed=probe(name);samples[name]=latency;cost+=elapsed
    if policy=='legacy':
        # Preserve old memory skip; old algorithm probes GPU events elsewhere.
        intensity=2*math.prod(shape)/(2*(shape[0]*shape[2]+shape[1]*shape[2]+shape[0]*shape[1]))
        if intensity<.5*(149.7e12/696e9):return static,samples,pred,cost,'legacy-memory-skip'
        for name in available:sample(name)
    elif policy=='full-calibration':
        for name in available:sample(name)
    elif policy in ('scic-v2','scic-v2-no-ai'):
        sample(static)
        challenger=min(available,key=lambda n:pred.get(n,{'ms':float('inf')})['ms'])
        sample(challenger)
        winner=min(samples,key=samples.get)
        if any(n not in samples and pred.get(n,{'lower':0})['lower']<.98*samples[winner] for n in available):
            for name in available:sample(name)
    else:raise ValueError(policy)
    winner=min(samples,key=samples.get)
    if policy.startswith('scic-v2') and expected_calls*(samples[static]-samples[winner])<=cost:
        return static,samples,pred,cost,'not-amortized'
    return winner,samples,pred,cost,''


def training_fixed(rows,mode='direct',reuse='rotated'):
    """Rank fixed candidates on training only, charging native fallback for rejection."""
    from shapefuse.v2_kernels import NAMES
    subset=[r for r in rows if r.get('train') and r['mode']==mode and r['reuse']==reuse]
    geometries=sorted({r['label'] for r in subset})
    scores={}
    for name in NAMES:
        costs=[]
        for geometry in geometries:
            candidates=[r for r in subset if r['label']==geometry]
            selected=next((r for r in candidates if r['candidate']==name),None)
            fallback=next((r for r in candidates if r['candidate']=='eager'),None)
            if selected is None:selected=fallback
            if selected is None:break
            costs.append(math.log(selected['host_ms']))
        if len(costs)==len(geometries) and costs:scores[name]=sum(costs)/len(costs)
    return min(scores,key=scores.get) if scores else 'blas4'

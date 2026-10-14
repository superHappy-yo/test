import unittest
from shapefuse.v2_policy import LatencyPrior,choose,features

class PolicyTests(unittest.TestCase):
    def test_memory_is_probed_and_budget_expands(self):
        class Prior:
            def predict(self,*args):return {'blas4':{'ms':2,'lower':1},'blas8':{'ms':1,'lower':.9},'split1':{'ms':3,'lower':.5}}
        called=[]
        def probe(n):called.append(n);return {'blas4':2,'blas8':1,'split1':.8}[n],.01
        chosen,samples,_,_,_=choose('scic-v2',['blas4','blas8','split1'],Prior(),(1,1024,1024),'direct','same','blas4',probe)
        self.assertEqual(chosen,'split1');self.assertEqual(called,['blas4','blas8','split1'])
    def test_low_reuse_retains_incumbent(self):
        class Prior:
            def predict(self,*args):return {'blas4':{'ms':2,'lower':1},'blas8':{'ms':1,'lower':.9}}
        selected,*_=choose('scic-v2',['blas4','blas8'],Prior(),(1,1024,1024),'direct','same','blas4',lambda n:(2 if n=='blas4' else 1,10),expected_calls=1)
        self.assertEqual(selected,'blas4')
    def test_heldout_rows_cannot_change_fit(self):
        rows=[dict(M=m,N=1024,K=1024,candidate='blas4',mode='direct',reuse='same',host_ms=.01+m/1e6,train=True) for m in (1,8,128,512)]
        reference=LatencyPrior().fit(rows).dump()
        poison=dict(rows[0],train=False,host_ms=1e-20,N=8192,K=8192)
        self.assertEqual(reference,LatencyPrior().fit(rows+[poison]).dump())
    def test_no_ai_has_fewer_features_and_fit_is_finite(self):
        self.assertEqual(len(features(1,1024,1024,'blas4','direct','same',analytical=False)),10)
        rows=[dict(M=m,N=1024,K=1024,candidate='blas4',mode='direct',reuse='same',host_ms=.01+m/1e6) for m in (1,8,128,512)]
        prior=LatencyPrior().fit(rows)
        self.assertGreater(prior.predict((64,1024,1024),'direct','same')['blas4']['ms'],0)

if __name__=='__main__':unittest.main()

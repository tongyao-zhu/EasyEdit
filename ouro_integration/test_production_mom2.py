import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from .production_mom2 import bind_production_moment,sha,forbid_statistic_collection

class BindingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        r=Path(self.temp.name);cache=r/'source.npz';cache.write_bytes(b'CPU binding fixture, not scientific statistics')
        ident=r/'identity.json';ident.write_text(json.dumps(dict(samples=100000,tokens=78683546,K=4,occurrence=3,
            layer=17,precision='float32',cache=False,attention='native_sdpa_unchanged',TF32=False,model_manifest_sha256='a'*64)))
        self.binding=dict(cache_file=str(cache),cache_sha256=sha(cache),identity_file=str(ident),identity_sha256=sha(ident),model_snapshot='snapshot')
        self.runtime=NS(stats_path=r/'runtime',model=NS(config=NS(_name_or_path='/model/snapshot')),
            identity=dict(model_manifest_sha256='a'*64,corpus_manifest_sha256=sha(ident),K=4,occurrence=3),
            hparams=NS(mom2_adjustment=True,layers=[17],mom2_dataset='wikipedia',mom2_dtype='float32',mom2_n_samples=100000))
    def test_exact_copy_and_reuse(self):
        p=bind_production_moment(self.runtime,self.binding)
        self.assertEqual(sha(p),self.binding['cache_sha256'])
        self.assertEqual(p,bind_production_moment(self.runtime,self.binding))
    def test_reject_wrong_depth_and_model(self):
        self.runtime.identity['K']=3
        with self.assertRaises(AssertionError):bind_production_moment(self.runtime,self.binding)
        self.runtime.identity['K']=4;self.runtime.identity['model_manifest_sha256']='b'*64
        with self.assertRaises(AssertionError):bind_production_moment(self.runtime,self.binding)
    def test_reject_source_or_destination_corruption(self):
        p=bind_production_moment(self.runtime,self.binding);p.write_bytes(b'wrong')
        with self.assertRaises(AssertionError):bind_production_moment(self.runtime,self.binding)
        self.binding['cache_sha256']='0'*64
        with self.assertRaises(AssertionError):bind_production_moment(self.runtime,self.binding)
    def test_collection_guard_restores(self):
        import importlib
        from .upstream import load_rome
        ls=importlib.import_module(load_rome().__package__+'.layer_stats');original=ls.load_dataset
        with forbid_statistic_collection():
            with self.assertRaises(RuntimeError):ls.load_dataset('must-not-download')
        self.assertIs(ls.load_dataset,original)

    def test_k2_requires_its_own_bound_identity(self):
        self.runtime.identity.update(K=2, occurrence=1)
        with self.assertRaises(AssertionError):
            bind_production_moment(self.runtime,self.binding)
        identity_path=Path(self.binding['identity_file'])
        identity=json.loads(identity_path.read_text())
        identity.update(K=2, occurrence=1)
        identity_path.write_text(json.dumps(identity))
        self.binding['identity_sha256']=sha(identity_path)
        self.runtime.identity['corpus_manifest_sha256']=sha(identity_path)
        result=bind_production_moment(self.runtime,self.binding)
        self.assertEqual(sha(result),self.binding['cache_sha256'])

    def test_reject_nonterminal_or_noninteger_fit_coordinate(self):
        for depth,occ in [(2,0),(4,1),(True,0),(2,True)]:
            self.runtime.identity.update(K=depth,occurrence=occ)
            with self.assertRaises(AssertionError):
                bind_production_moment(self.runtime,self.binding)

if __name__=='__main__':unittest.main()

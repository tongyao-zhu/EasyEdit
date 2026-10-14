import hashlib
import importlib
import random
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from ouro_integration.adapter import FixedDepth
from ouro_integration.trace_scope import OuroTraceScope
from ouro_integration.upstream import load_rome
from ouro_integration.moment_data import PinnedTextRows, pinned_wikipedia_access


def rows_for(texts, total=23):
    order=list(range(total));random.Random(1).shuffle(order);order=order[:len(texts)]
    records=[dict(sample_index=i,source_row_index=j,text=t,text_sha256=hashlib.sha256(t.encode()).hexdigest()) for i,(j,t) in enumerate(zip(order,texts))]
    return order,records,PinnedTextRows(total,order,records)

class Tokenizer:
    def encode(self,text,truncation=True,max_length=None):
        values=[ord(x)-96 for x in text]
        return values[:max_length] if truncation else values

class Decoder(nn.Module):
    def __init__(self):
        super().__init__();self.mlp=nn.Linear(2,2,bias=False)
        with torch.no_grad():self.mlp.weight.copy_(torch.eye(2)*.5)
    def forward(self,x,current_ut):return x+self.mlp(x)

class Model(nn.Module):
    def __init__(self):
        super().__init__();self.config=SimpleNamespace(model_type='ouro',total_ut_steps=4,use_cache=False,max_position_embeddings=8,_name_or_path='toy')
        self.model=nn.Module();self.model.layers=nn.ModuleList([Decoder(),Decoder()]);self.model.total_ut_steps=4;self.model.gradient_checkpointing=False;self.early_exit_step=3
    def forward(self,input_ids,attention_mask=None,position_ids=None):
        x=torch.stack([input_ids.float(),input_ids.float()**2],-1)
        for ut in range(self.model.total_ut_steps):
            for layer in self.model.layers:x=layer(x,current_ut=ut)
        return x

class MomentDataTests(unittest.TestCase):
    def setUp(self):
        self.rome=load_rome();self.ls=importlib.import_module(self.rome.__package__+'.layer_stats')
        self.rs=importlib.import_module('_ouro_easyedit_upstream.util.runningstats')
    def test_full_population_index_space_exact_sampler(self):
        order,records,rows=rows_for(['a','bc','def'])
        self.assertEqual(list(self.rs.FixedRandomSubsetSampler(rows,seed=1,end=3)),order)
        self.assertEqual(len(rows),23)
        self.assertEqual([rows[i]['text'] for i in order],['a','bc','def'])
        with self.assertRaises(KeyError):rows[next(i for i in range(23) if i not in order)]
    def test_corrupt_order_text_duplicate_fail_closed(self):
        order,recs,_=rows_for(['a','bc'])
        with self.assertRaises(ValueError):PinnedTextRows(23,list(reversed(order)),recs)
        with self.assertRaises(ValueError):PinnedTextRows(23,[order[0]]*2,recs)
        recs[0]['text']='altered'
        with self.assertRaises(ValueError):PinnedTextRows(23,order,recs)
    def test_dataset_request_namespace_and_exception_cleanup(self):
        rows=rows_for(['a','bc'])[2];original=self.ls.load_dataset
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            with pinned_wikipedia_access(self.ls,rows):
                self.assertIs(self.ls.load_dataset('wikimedia/wikipedia','20231101.en')['train'],rows)
                with self.assertRaises(ValueError):self.ls.load_dataset('Salesforce/wikitext','wikitext-2-raw-v1')
                with self.assertRaises(ValueError):self.ls.load_dataset('wikimedia/wikipedia','20231101.en',revision='changed')
                with self.assertRaises(RuntimeError),pinned_wikipedia_access(self.ls,rows):pass
                raise RuntimeError('intentional')
        self.assertIs(self.ls.load_dataset,original)
    def test_original_collector_matches_independent_masked_moment(self):
        texts=['ab','c','defghijkl','bc'];rows=rows_for(texts)[2];tok=Tokenizer();model=Model();name='model.layers.0.mlp'
        original_tally=self.ls.tally
        # Test only: avoid multiprocessing and CUDA; keep production collector untouched.
        def cpu_tally(*a,**kw):kw.update(num_workers=0,pin_memory=False);return original_tally(*a,**kw)
        with tempfile.TemporaryDirectory() as d:
            for k,occ in [(2,0),(2,1),(3,2),(4,3)]:
                with pinned_wikipedia_access(self.ls,rows),patch.object(self.ls,'normalize_device',lambda _:torch.device('cpu')),patch.object(self.ls,'tally',cpu_tally),FixedDepth(model,k),OuroTraceScope(model,self.rome.nethook,occ,k):
                    s=self.ls.layer_stats(model,tok,name,Path(d)/str(k)/str(occ),'wikipedia',['mom2'],sample_size=4,precision='float32',hparams=SimpleNamespace(device='cpu'),progress=lambda iterable,**kwargs:iterable)
                vectors=[]
                for t in texts:
                    for z in tok.encode(t,max_length=8):vectors.append([z*1.5**(2*occ),z*z*1.5**(2*occ)])
                x=torch.tensor(vectors,dtype=torch.float64);expected=x.T@x/len(x)
                self.assertEqual(s.mom2.count,len(x));torch.testing.assert_close(s.mom2.moment().double(),expected,rtol=2e-6,atol=1e-6)
                # A cache hit must never request network/data or execute a forward.
                with patch.object(self.ls,'load_dataset',side_effect=AssertionError('cache used data')),patch.object(model,'forward',side_effect=AssertionError('cache used model')):
                    cached=self.ls.layer_stats(model,tok,name,Path(d)/str(k)/str(occ),'wikipedia',['mom2'],sample_size=4,precision='float32',hparams=SimpleNamespace(device='cpu'),progress=lambda iterable,**kwargs:iterable)
                self.assertTrue(torch.equal(cached.mom2.moment(),s.mom2.moment()))
                for mod in model.modules():self.assertFalse(mod._forward_hooks);self.assertFalse(mod._forward_pre_hooks)
    def test_original_inverse_and_cache_formula(self):
        u=importlib.import_module(self.rome.__package__+'.compute_u');cov=torch.tensor([[5.,1.],[1.,2.]])
        model=Model();hp=SimpleNamespace(device='cpu',stats_dir='unused')
        with patch.object(u,'inv_mom2_cache',{}),patch.object(u,'normalize_device',lambda _:torch.device('cpu')),patch.object(u,'layer_stats',return_value=SimpleNamespace(mom2=SimpleNamespace(moment=lambda:cov))) as collect:
            inv=u.get_inv_cov(model,Tokenizer(),'model.layers.0.mlp','wikipedia',4,'float32',hparams=hp)
            torch.testing.assert_close(inv@cov,torch.eye(2),atol=1e-7,rtol=1e-6)
            self.assertIs(inv,u.get_inv_cov(model,Tokenizer(),'model.layers.0.mlp','wikipedia',4,'float32',hparams=hp));self.assertEqual(collect.call_count,1)

if __name__=='__main__':unittest.main()

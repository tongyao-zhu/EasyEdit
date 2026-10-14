import importlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
from torch import nn

from ouro_integration.context_generation import generate_full_prefix
from ouro_integration.rome_runtime import ROMERuntime, cache_digest
from ouro_integration.rome_tokens import prepare_atomic_request
from ouro_integration.upstream import load_rome


class Batch(dict):
    def to(self,device): return Batch({k:v.to(device) for k,v in self.items()})


class Tokenizer:
    padding_side = 'right'
    pad_token_id = 0
    def __call__(self,texts,padding=True,return_tensors='pt'):
        ids=[[ord(c)%29+1 for c in t] for t in texts]
        width=max(map(len,ids))
        return Batch(input_ids=torch.tensor([r+[0]*(width-len(r)) for r in ids]),
                     attention_mask=torch.tensor([[1]*len(r)+[0]*(width-len(r)) for r in ids]))
    def decode(self,ids,skip_special_tokens=True):
        return ' '.join(str(x) for x in ids if x!=0)


class CacheReferenceModel(nn.Module):
    """Exact integer prefix state makes cached/full-prefix logits comparable."""
    def __init__(self):
        super().__init__(); self.anchor=nn.Parameter(torch.zeros(1)); self.calls=[]
        self.name_or_path='toy'; self.config=SimpleNamespace(model_type='ouro',total_ut_steps=4,use_cache=False)
        self.model=nn.Module(); self.model.total_ut_steps=4; self.model.layers=nn.ModuleList([])
        self.early_exit_step=3
    def forward(self,input_ids,attention_mask=None,past_key_values=None,use_cache=False):
        self.calls.append({'length':input_ids.shape[1],'cache':use_cache})
        state=input_ids.cumsum(1)
        if past_key_values is not None: state=state+past_key_values[:,None]
        center=state.remainder(31)
        vocab=torch.arange(32)[None,None,:]
        logits=-(vocab-center[:,:,None]).abs().float()*.19+vocab.float()*.0001
        return SimpleNamespace(logits=logits,past_key_values=state[:,-1])


def identity(**updates):
    d={k:'a'*64 for k in ('model_manifest_sha256','tokenizer_manifest_sha256','source_sha256','corpus_manifest_sha256')}
    d.update(K=4,occurrence=3,precision='float32',attention='sdpa',cache=False,
             mom2_dataset='wikipedia',mom2_n_samples=100000,mom2_dtype='float32',
             context_seed=17,context_template_length_params=[[5,10],[10,10]])
    d.update(updates);return d


class ROMECPUTests(unittest.TestCase):
    def test_real_rome_import_without_eager_editors(self):
        rome=load_rome()
        self.assertTrue(callable(rome.apply_rome_to_model))
        self.assertTrue(Path(rome.__file__).as_posix().endswith('easyeditor/models/rome/rome_main.py'))

    def test_full_prefix_matches_upstream_samples_and_rng(self):
        upstream=load_rome().generate_fast
        for prompts,rep,length in [(['A','BB','CCCC'],2,13),(['A','B'],1,8),(['long','A'],1,4),(['longer','A'],1,3)]:
            with self.subTest(prompts=prompts,length=length):
                a,b=CacheReferenceModel(),CacheReferenceModel();tok=Tokenizer()
                torch.manual_seed(17); expected=upstream(a,tok,prompts,n_gen_per_prompt=rep,max_out_len=length)
                state=torch.random.get_rng_state().clone()
                torch.manual_seed(17); actual=generate_full_prefix(b,tok,prompts,n_gen_per_prompt=rep,max_out_len=length)
                self.assertEqual(actual,expected)
                self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
                self.assertTrue(all(not c['cache'] for c in b.calls))
                self.assertTrue(all(c['cache'] for c in a.calls))
                self.assertEqual(len(a.calls),len(b.calls))

    def test_generator_rejects_unsupported_paths(self):
        tok=Tokenizer();tok.padding_side='left'
        with self.assertRaises(ValueError):generate_full_prefix(CacheReferenceModel(),tok,['A'])
        tok.padding_side='right'
        with self.assertRaises(ValueError):generate_full_prefix(CacheReferenceModel(),tok,['A'],vanilla_generation=True)
        with self.assertRaises(ValueError):generate_full_prefix(CacheReferenceModel(),tok,[''])

    def test_atomic_boundary_rejects_eval_and_duplicate_subject(self):
        r=dict(case_id=1,family_id='x',prompt='A wrote',subject='A',target_new='B')
        self.assertEqual(prepare_atomic_request(r)['target_new'],' B')
        self.assertEqual(r['target_new'],'B')
        with self.assertRaises(ValueError):prepare_atomic_request(dict(r,multi_hop='answer'))
        with self.assertRaises(ValueError):prepare_atomic_request(dict(r,prompt='A and A wrote'))

    def test_cache_digest_separates_all_provenance(self):
        original=cache_digest(identity())
        for field,value in [('K',3),('occurrence',1),('model_manifest_sha256','b'*64),
            ('tokenizer_manifest_sha256','b'*64),('source_sha256','b'*64),('corpus_manifest_sha256','b'*64),
            ('mom2_n_samples',999),('context_seed',18)]:
            kw={field:value}
            if field=='K': kw['occurrence']=2
            self.assertNotEqual(original,cache_digest(identity(**kw)))
        d=identity();del d['corpus_manifest_sha256']
        with self.assertRaises(ValueError):cache_digest(d)

    def test_runtime_caches_isolated_and_exception_restores(self):
        rome=load_rome();u=importlib.import_module(rome.__package__+'.compute_u')
        old=(rome.generate_fast,rome.CONTEXT_TEMPLATES_CACHE,u.inv_mom2_cache,rome.get_context_templates)
        model=CacheReferenceModel();hp=SimpleNamespace(**{k:identity()[k] for k in
            ('mom2_dataset','mom2_n_samples','mom2_dtype','context_template_length_params')},stats_dir='unused')
        with tempfile.TemporaryDirectory() as folder:
            runtime=ROMERuntime(model,hp,identity(),folder)
            torch.manual_seed(23);state=torch.random.get_rng_state().clone()
            with self.assertRaisesRegex(RuntimeError,'intentional'),runtime:
                first=rome.get_context_templates(model,Tokenizer(),hp.context_template_length_params)
                self.assertEqual(len(first),21)
                self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
                self.assertIs(first,rome.get_context_templates(model,Tokenizer(),hp.context_template_length_params))
                u.inv_mom2_cache[('model','layer')]='sentinel'
                self.assertEqual(len(old[2]),0)
                with self.assertRaises(RuntimeError),ROMERuntime(model,hp,identity(),folder):pass
                raise RuntimeError('intentional')
            self.assertEqual((rome.generate_fast,rome.CONTEXT_TEMPLATES_CACHE,u.inv_mom2_cache,rome.get_context_templates),old)
            self.assertEqual(len(model._forward_hooks),0)
            self.assertEqual(len(model._forward_pre_hooks),0)
            with runtime:
                self.assertEqual(u.inv_mom2_cache[('model','layer')],'sentinel')
                self.assertIs(rome.CONTEXT_TEMPLATES_CACHE,first)
            with ROMERuntime(model,hp,identity(occurrence=2),folder):
                self.assertEqual(u.inv_mom2_cache,{})
                self.assertIsNone(rome.CONTEXT_TEMPLATES_CACHE)
            manifest=runtime.stats_path/'IDENTITY.json';manifest.write_text('{}')
            with self.assertRaises(ValueError),runtime:pass
            self.assertEqual((rome.generate_fast,rome.CONTEXT_TEMPLATES_CACHE,u.inv_mom2_cache,rome.get_context_templates),old)


if __name__=='__main__': unittest.main()

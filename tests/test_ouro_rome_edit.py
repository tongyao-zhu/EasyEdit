"""Transactional wiring tests; mocked upstream is NOT real-model acceptance."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from ouro_integration.rome_edit import edited_rome


class Runtime:
    def __init__(self, behavior):
        self.model=nn.Module(); self.model.model=nn.Module()
        self.model.model.layers=nn.ModuleList([nn.Module()])
        self.model.model.layers[0].mlp=nn.Module()
        self.model.model.layers[0].mlp.down_proj=nn.Linear(2,2,bias=False)
        self.hparams=SimpleNamespace(use_chat_template=False,fact_token='subject_last',layers=[0],
            v_loss_layer=0,rewrite_module_tmp='model.layers.{}.mlp.down_proj',
            mlp_module_tmp='model.layers.{}.mlp',layer_module_tmp='model.layers.{}',
            context_template_length_params=[[5,10],[10,10]])
        self.digest='test_identity'; self.active=False
        self.rome=SimpleNamespace(__package__='mocked',get_context_templates=lambda *args:['{}'],
                                  apply_rome_to_model=behavior)
    def __enter__(self):self.active=True;return self
    def __exit__(self,*args):self.active=False


class TransactionTests(unittest.TestCase):
    request=dict(case_id=1,family_id='x',prompt='A wrote',subject='A',target_new='B')

    def test_eval_fields_rejected_before_runtime_access(self):
        with self.assertRaises(ValueError),edited_rome(None,None,dict(self.request,answer='evaluation')):pass

    def run_transaction(self, fail_fit=False, fail_eval=False, zero=False):
        calls=[]
        def upstream(model,tok,requests,hp,**kwargs):
            self.assertEqual(requests,[self.request]);self.assertEqual(requests[0]['target_new'],'B')
            name='model.layers.0.mlp.down_proj.weight';weight=dict(model.named_parameters())[name]
            old=weight.detach().clone(); calls.append(1)
            with torch.no_grad():
                if not zero:weight.add_(.1)
            weight.requires_grad_(False)
            if fail_fit:raise RuntimeError('injected fit failure')
            return model,{name:old}
        runtime=Runtime(upstream);tok=SimpleNamespace(padding_side='left')
        weight=next(runtime.model.parameters());old=weight.detach().clone()
        with patch('ouro_integration.rome_edit.importlib.import_module',return_value=object()), \
             patch('ouro_integration.rome_edit.audit_tokens',return_value={'scope':'mock only'}) as audit:
            def execute():
                with edited_rome(runtime,tok,self.request) as result:
                    self.assertFalse(runtime.active)
                    self.assertEqual(tok.padding_side,'left')
                    self.assertTrue(weight.requires_grad)
                    self.assertEqual(result['zero_update'],zero)
                    self.assertEqual(torch.equal(weight,old),zero)
                    if fail_eval:raise RuntimeError('injected evaluation failure')
            if fail_fit or fail_eval:
                with self.assertRaisesRegex(RuntimeError,'injected'):execute()
            else:execute()
            audit.assert_called_once()
        self.assertEqual(calls,[1]);self.assertFalse(runtime.active)
        self.assertTrue(torch.equal(weight,old));self.assertTrue(weight.requires_grad)
        self.assertEqual(tok.padding_side,'left')

    def test_success_deploys_outside_scope_and_restores(self):self.run_transaction()
    def test_partial_fit_failure_restores(self):self.run_transaction(fail_fit=True)
    def test_evaluation_failure_restores(self):self.run_transaction(fail_eval=True)
    def test_zero_update_is_retained(self):self.run_transaction(zero=True)

    def test_token_failure_prevents_fit(self):
        called=[];runtime=Runtime(lambda *args,**kwargs:called.append(1));tok=SimpleNamespace(padding_side='left')
        old=next(runtime.model.parameters()).detach().clone()
        with patch('ouro_integration.rome_edit.importlib.import_module',return_value=object()), \
             patch('ouro_integration.rome_edit.audit_tokens',side_effect=ValueError('token mismatch')):
            with self.assertRaisesRegex(ValueError,'token mismatch'),edited_rome(runtime,tok,self.request):pass
        self.assertEqual(called,[]);self.assertFalse(runtime.active)
        self.assertEqual(tok.padding_side,'left')
        self.assertTrue(torch.equal(next(runtime.model.parameters()),old))


if __name__=='__main__':unittest.main()

"""Fit-depth adapter contracts; upstream optimizer is mocked, not efficacy-tested."""
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest
from torch import nn
from ouro_integration.fit_baseline import fit


class FitDepthTests(unittest.TestCase):
    def test_depth_is_used_and_state_restored_on_success_and_failure(self):
        request=dict(case_id=1,family_id=1,prompt='{} lives in',subject='X',target_new='Y')
        for depth in (2,4):
            for failure in (False,True):
                model=nn.Linear(2,2,bias=False)
                model.config=NS(model_type='ouro',total_ut_steps=4,use_cache=True)
                model.model=NS(total_ut_steps=4,gradient_checkpointing=False)
                model.early_exit_step=3
                tokenizer=NS(padding_side='left')
                observed=[]
                def apply(m,t,requests,hparams,return_orig_weights):
                    observed.append((m.config.total_ut_steps,m.model.total_ut_steps,
                                     m.early_exit_step,m.config.use_cache,t.padding_side))
                    self.assertEqual(requests,[request])
                    self.assertTrue(return_orig_weights)
                    m.weight.requires_grad_(False)
                    if failure:raise RuntimeError('synthetic optimizer failure')
                    return m,{'weight':m.weight.detach().clone()}
                upstream=NS(FTHyperParams=lambda **kw:NS(**kw),apply_ft_to_model=apply)
                with patch('ouro_integration.fit_baseline.load_ft',return_value=upstream):
                    if failure:
                        with self.assertRaisesRegex(RuntimeError,'synthetic optimizer'):
                            fit(model,tokenizer,request,{},depth=depth)
                    else:fit(model,tokenizer,request,{},depth=depth)
                self.assertEqual(observed,[(depth,depth,depth-1,False,'right')])
                self.assertEqual((model.config.total_ut_steps,model.model.total_ut_steps,
                                  model.early_exit_step,model.config.use_cache),(4,4,3,True))
                self.assertEqual(tokenizer.padding_side,'left')
                self.assertTrue(model.weight.requires_grad)
                self.assertFalse(model._forward_pre_hooks)

    def test_reject_unplanned_depth_before_model_access(self):
        for depth in (1,3,5,True,2.0):
            with self.assertRaises(ValueError):fit(None,None,{}, {},depth=depth)


if __name__=='__main__':unittest.main()

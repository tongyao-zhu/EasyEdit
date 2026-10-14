import unittest
from types import SimpleNamespace
import torch
from torch import nn
from ouro_integration.evaluation_bridge import WeightState
from ouro_integration.fit_baseline import fit
from ouro_integration.analyze import recompute
from ouro_integration import frozen_evaluation as E


class BenchmarkTests(unittest.TestCase):
    def test_teacher_swap_restores_exact_deployed_weights(self):
        m=nn.Linear(3,2,bias=False)
        original=m.weight.detach().clone()
        with torch.no_grad():m.weight.add_(0.123)
        edited=m.weight.detach().clone();pid=id(m.weight)
        state=WeightState(m,{'weight':original})
        state.enabled=False
        self.assertTrue(torch.equal(m.weight,original))
        state.enabled=False
        state.enabled=True
        self.assertTrue(torch.equal(m.weight,edited))
        self.assertEqual(id(m.weight),pid)

    def test_fit_rejects_evaluation_fields_before_model_access(self):
        with self.assertRaises(ValueError):
            fit(None,None,{'prompt':'x','target_new':'y','multihop_questions':['leak']},{})

    def test_raw_metric_recomputation_keeps_zero_and_missing_distinct(self):
        rows=[{'kind':'direct','correct':False},{'kind':'multihop','correct':True},
              {'kind':'unchanged_chain','correct':True},
              {'kind':'locality','correct':False,'base_correct':False,'output_agreement':True}]
        got=recompute(rows)
        self.assertEqual(got['Joint'],0)
        self.assertEqual(got['A_MH'],1)
        self.assertIsNone(got['locality_base_correct_accuracy_change'])

    def test_existing_parser_contract(self):
        self.assertEqual(E.parse_answer('\nAnswer: NEW-York!\nExtra'),'new york')
        self.assertEqual(E.parse_answer(''), '')


if __name__=='__main__':unittest.main()

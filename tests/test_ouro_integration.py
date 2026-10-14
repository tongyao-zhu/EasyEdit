import unittest
from types import SimpleNamespace
import torch
from torch import nn
from ouro_integration import FixedDepth, SharedCallTrace
from ouro_integration.upstream import load_ft


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(3, 3, bias=False)

    def forward(self, hidden, current_ut=0):
        return hidden + self.mlp.down_proj(hidden).tanh()


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="ouro", total_ut_steps=4, use_cache=True)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Decoder()])
        self.model.total_ut_steps = 4
        self.early_exit_step = None

    def forward(self, hidden, **kwargs):
        for k in range(self.model.total_ut_steps):
            hidden = self.model.layers[0](hidden, current_ut=k)
        return hidden


class IntegrationTests(unittest.TestCase):
    def test_depth_sharing_gradient_and_restore(self):
        torch.manual_seed(17)
        model = Toy()
        original = model.model.layers[0].mlp.down_proj.weight
        for k in (2, 3, 4):
            with FixedDepth(model, k), SharedCallTrace(model, 0, gradients=True) as trace:
                model(torch.ones(1, 2, 3)).sum().backward()
            self.assertEqual([r['loop'] for r in trace.records], list(range(1, k+1)))
            self.assertTrue(all(r['parameter_id'] == id(original) for r in trace.records))
            self.assertTrue(all(r['gradient_norm'] > 0 for r in trace.records))
            self.assertIs(model.model.layers[0].mlp.down_proj.weight, original)
            self.assertEqual(model.model.total_ut_steps, 4)
            self.assertIsNone(model.early_exit_step)
            self.assertTrue(model.config.use_cache)

    def test_exception_restores_and_removes_hooks(self):
        model = Toy()
        for kwargs in ({'labels': torch.zeros(1)}, {'use_cache': True},
                       {'use_weighted_exit': True}, {'exit_at_step': 0}):
            with self.assertRaises(ValueError), FixedDepth(model, 2):
                model(torch.ones(1, 2, 3), **kwargs)
            self.assertEqual(model.model.total_ut_steps, 4)
            self.assertEqual(len(model._forward_pre_hooks), 0)

    def test_real_upstream_entrypoint(self):
        ft = load_ft()
        self.assertTrue(callable(ft.apply_ft_to_model))
        self.assertTrue(ft.__file__.endswith('easyeditor/models/ft/ft_main.py'))


if __name__ == '__main__':
    unittest.main()

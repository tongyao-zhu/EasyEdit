import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from torch import nn
from ouro_integration.adapter import FixedDepth
from ouro_integration.trace_scope import OuroTraceScope

spec=importlib.util.spec_from_file_location('standalone_nethook',Path(__file__).resolve().parents[1]/'easyeditor/util/nethook.py')
N=importlib.util.module_from_spec(spec);spec.loader.exec_module(N)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__();self.mlp=nn.Linear(2,2,bias=False)
        with torch.no_grad():self.mlp.weight.copy_(torch.eye(2)*.5)
    def forward(self, x, current_ut):return x+self.mlp(x)

class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.config=SimpleNamespace(model_type='ouro',total_ut_steps=4,use_cache=False)
        self.model=nn.Module();self.model.layers=nn.ModuleList([Decoder(),Decoder()]);self.model.total_ut_steps=4;self.model.gradient_checkpointing=False;self.early_exit_step=3
    def forward(self,x):
        for ut in range(self.model.total_ut_steps):
            for d in self.model.layers:x=d(x,current_ut=ut)
        return x

class ScopeTests(unittest.TestCase):
    def setUp(self):self.m=Toy();self.x=torch.tensor([[1.,2.]]);self.name='model.layers.0.mlp'
    def test_default_last_and_first_stop_preserved(self):
        with N.Trace(self.m,self.name,retain_input=True) as tr:self.m(self.x)
        torch.testing.assert_close(tr.input,self.x*1.5**6)
        with N.Trace(self.m,self.name,retain_input=True,stop=True) as tr:self.m(self.x)
        torch.testing.assert_close(tr.input,self.x)
        with N.Trace(self.m,self.name):pass
    def test_every_supported_occurrence_matches_native(self):
        baseline_ids={n:id(p) for n,p in self.m.named_parameters()}
        for k in [2,3,4]:
            for occurrence in range(k):
                with FixedDepth(self.m,k),OuroTraceScope(self.m,N,occurrence,k):
                    with N.Trace(self.m,self.name,retain_input=True,clone=True) as tr:y=self.m(self.x)
                    torch.testing.assert_close(tr.input,self.x*1.5**(2*occurrence))
                    self.assertEqual(tr.invocations_seen,k);self.assertEqual(tr.invocations_selected,1)
                    torch.testing.assert_close(y,self.x*1.5**(2*k))
        self.assertEqual(baseline_ids,{n:id(p) for n,p in self.m.named_parameters()})
    def test_latent_intervention_once_and_gradient(self):
        delta=torch.tensor([[.2,.3]],requires_grad=True)
        with FixedDepth(self.m,4),OuroTraceScope(self.m,N,1,4):
            with N.Trace(self.m,self.name,edit_output=lambda output:output+delta) as tr:y=self.m(self.x)
            expected=self.x*1.5**8+delta*1.5**5
            torch.testing.assert_close(y,expected);y.sum().backward()
            torch.testing.assert_close(delta.grad,torch.ones_like(delta)*1.5**5)
            self.assertEqual(tr.invocations_selected,1)
    def test_statistics_stop_is_selected_call_not_first(self):
        with FixedDepth(self.m,4),OuroTraceScope(self.m,N,2,4) as scope:
            for _ in range(2):
                with N.Trace(self.m,self.name,retain_input=True,stop=True) as tr:self.m(self.x)
                torch.testing.assert_close(tr.input,self.x*1.5**4)
                self.assertEqual(tr.invocations_seen,3);self.assertEqual(tr.invocations_selected,1)
                self.assertFalse(scope.active)
    def test_tracedict_and_repeated_forward_reset(self):
        names=[self.name,'model.layers.1']
        with FixedDepth(self.m,3),OuroTraceScope(self.m,N,1,3):
            for scale in [1,2]:
                with N.TraceDict(self.m,names) as tr:self.m(self.x*scale)
                self.assertEqual(tr[self.name].invocations_selected,1)
                torch.testing.assert_close(tr['model.layers.1'].output,self.x*scale*1.5**4)
    def test_physical_weight_update_still_applies_every_loop(self):
        with torch.no_grad():self.m.model.layers[0].mlp.weight.add_(torch.eye(2)*.1)
        with FixedDepth(self.m,4),OuroTraceScope(self.m,N,3,4):
            with N.Trace(self.m,self.name) as tr:y=self.m(self.x)
        torch.testing.assert_close(y,self.x*(1.6*1.5)**4)
        self.assertEqual(tr.invocations_selected,1)
    def test_exceptions_remove_all_hooks_and_policy(self):
        original=self.m(self.x)
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            with FixedDepth(self.m,4),OuroTraceScope(self.m,N,2,4):
                with N.Trace(self.m,self.name,edit_output=lambda output:(_ for _ in ()).throw(RuntimeError('intentional'))):self.m(self.x)
        for mod in self.m.modules():self.assertFalse(mod._forward_hooks);self.assertFalse(mod._forward_pre_hooks)
        with N.Trace(self.m,self.name) as tr:y=self.m(self.x)
        self.assertEqual(tr.invocations_selected,4);torch.testing.assert_close(y,original)
    def test_invalid_and_unobserved_traces_fail_closed(self):
        with self.assertRaises(ValueError):OuroTraceScope(self.m,N,4,4)
        with self.assertRaisesRegex(RuntimeError,'no invocation'):
            with FixedDepth(self.m,4),OuroTraceScope(self.m,N,1,4):
                with N.Trace(self.m,self.name):pass
        with self.assertRaisesRegex(RuntimeError,'outside supported'):
            with FixedDepth(self.m,4),OuroTraceScope(self.m,N,1,4):
                with N.Trace(self.m):self.m(self.x)

if __name__=='__main__':unittest.main()

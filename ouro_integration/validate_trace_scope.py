"""Real-model instrumentation acceptance, not an editing-effect experiment."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adapter import FixedDepth, SharedCallTrace
from .trace_scope import OuroTraceScope
from .validate import model_hashes, tensor_hash, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--model-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    report = {'status': 'running', 'scope': 'instrumentation only; no fit or scientific evaluation',
              'checks': {}, 'precision': 'FP32', 'cache': False, 'records': []}
    started = time.monotonic()
    try:
        import hashlib
        for name, expected in json.loads(a.model_manifest.read_text()).items():
            h = hashlib.sha256()
            with (a.model / name).open('rb') as f:
                for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
            assert h.hexdigest() == expected, name
        report['checks']['model_files_bound'] = True
        request = json.loads(a.request.read_text())
        assert set(request) == {'case_id', 'family_id', 'prompt', 'subject', 'target_new'}
        report['case_id'] = request['case_id']
        torch.manual_seed(17)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        tok = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(str(a.model), trust_remote_code=True,
            local_files_only=True, torch_dtype=torch.float32, attn_implementation='sdpa').to('cuda:0').eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
        original_hashes = model_hashes(model)
        report['original_parameter_hashes'] = original_hashes
        inputs = tok(request['prompt'], return_tensors='pt').to('cuda:0')
        report['input_ids'] = inputs.input_ids.tolist()
        path = Path(__file__).resolve().parents[1] / 'easyeditor/util/nethook.py'
        spec = importlib.util.spec_from_file_location('_scope_acceptance_nethook', path)
        nh = importlib.util.module_from_spec(spec); spec.loader.exec_module(nh)
        names = ['model.layers.17.mlp.down_proj', 'model.layers.17.mlp', 'model.layers.17']
        for k in (2, 3, 4):
            native = {name: [] for name in names}
            handles = []
            for name in names:
                def observe(module, args, kwargs, output, name=name):
                    value = output[0] if isinstance(output, tuple) else output
                    inp = args[0] if args else kwargs['hidden_states']
                    native[name].append((inp.detach().clone(), value.detach().clone()))
                handles.append(nh.get_module(model, name).register_forward_hook(observe, with_kwargs=True))
            try:
                with FixedDepth(model, k), torch.no_grad():
                    reference = model(**inputs).logits.detach().clone()
            finally:
                for handle in handles: handle.remove()
            assert all(len(values) == k for values in native.values())
            for occurrence in range(k):
                with FixedDepth(model, k), OuroTraceScope(model, nh, occurrence, k) as scope:
                    with torch.no_grad(), nh.TraceDict(model, names, retain_input=True, clone=True, detach=True) as traces:
                        actual = model(**inputs).logits
                    assert torch.equal(reference, actual)
                    for name, trace in traces.items():
                        out = trace.output[0] if isinstance(trace.output, tuple) else trace.output
                        assert trace.invocations_seen == k and trace.invocations_selected == 1
                        assert torch.equal(trace.input, native[name][occurrence][0])
                        assert torch.equal(out, native[name][occurrence][1])
                    # Statistics-style early stop must stop at the selected native visit.
                    with torch.no_grad(), nh.Trace(model, names[0], retain_input=True, clone=True, stop=True) as stopped:
                        model(**inputs)
                    assert stopped.invocations_seen == occurrence+1 and stopped.invocations_selected == 1
                    assert torch.equal(stopped.input, native[names[0]][occurrence][0])
                    assert not scope.active and not scope.current
                    # A zero latent intervention must leave the full forward exact.
                    calls = []
                    def zero(output, layer):
                        calls.append(layer)
                        return output + torch.zeros_like(output)
                    with torch.no_grad(), nh.Trace(model, names[1], edit_output=zero):
                        zero_logits = model(**inputs).logits
                    assert calls == [names[1]] and torch.equal(reference, zero_logits)
                    report['records'].append({'K': k, 'occurrence_zero_based': occurrence,
                        'input_hash': tensor_hash(native[names[0]][occurrence][0]),
                        'output_hash': tensor_hash(native[names[0]][occurrence][1]),
                        'native_capture_equal': True, 'stop_correct': True, 'zero_intervention_equal': True,
                        'events': scope.events})
            report['checks'][f'K{k}_every_invocation_and_stop'] = True
        # ROME-like latent optimization must receive a gradient at the chosen call only.
        delta = torch.zeros(model.config.hidden_size, device='cuda:0', requires_grad=True)
        calls = []
        def intervene(output, layer):
            calls.append(layer)
            result = output.clone()
            result[:, -1, :] = result[:, -1, :] + delta
            return result
        with FixedDepth(model, 4), OuroTraceScope(model, nh, 3, 4):
            with nh.Trace(model, names[1], edit_output=intervene):
                logits = model(**inputs).logits
            (-logits[:, -1].log_softmax(-1)[:, int(reference[0, -1].argmax())].mean()).backward()
        assert calls == [names[1]] and delta.grad is not None
        assert torch.isfinite(delta.grad).all() and float(delta.grad.norm()) > 0
        report['latent_gradient_norm'] = float(delta.grad.norm())
        report['checks']['one_latent_call_with_gradient'] = True
        # A physical update is NEVER gated by the instrumentation convention.
        weight = model.model.layers[17].mlp.down_proj.weight
        backup = weight.detach().clone()
        observed_hashes = []
        with torch.no_grad(): weight[0, 0].add_(0.001)
        changed_hash = tensor_hash(weight)
        assert changed_hash != tensor_hash(backup)
        def observe_weight(module, args): observed_hashes.append(tensor_hash(module.weight))
        handle = model.model.layers[17].mlp.down_proj.register_forward_pre_hook(observe_weight)
        try:
            with FixedDepth(model, 4), OuroTraceScope(model, nh, 3, 4), SharedCallTrace(model, 17) as trace:
                with torch.no_grad(), nh.Trace(model, names[0]): model(**inputs)
            assert observed_hashes == [changed_hash]*4
            assert [r['loop'] for r in trace.records] == [1,2,3,4]
            assert len({r['parameter_id'] for r in trace.records}) == 1
        finally:
            handle.remove()
            with torch.no_grad(): weight.copy_(backup)
        report['checks']['physical_update_shared_all_four_calls'] = True
        report['restored_parameter_hashes'] = model_hashes(model)
        assert report['restored_parameter_hashes'] == original_hashes
        report['checks']['all_parameter_hashes_restored'] = True
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
        assert nh._trace_invocation_filter.get() is None
        report['checks']['all_hooks_and_policy_removed'] = True
        report['peak_allocated_GiB'] = torch.cuda.max_memory_allocated()/2**30
        report['status'] = 'passed_instrumentation_only'
    except BaseException as exc:
        report['status'], report['error'] = 'failed', repr(exc)
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        write(a.output/'report.json', report)


if __name__ == '__main__':
    main()

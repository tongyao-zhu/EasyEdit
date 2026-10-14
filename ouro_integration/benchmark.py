"""Frozen native-deployment M0/FT-L/FT-M experiment, independent cases."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import frozen_evaluation as E
from .evaluation_bridge import EvaluationBridge
from .fit_baseline import fit
from .validate import model_hashes, tensor_hash


def dump(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def verify(root):
    freeze = json.loads((root/'FREEZE.json').read_text())
    for name, expected in freeze['files'].items():
        if sha(root/name) != expected: raise ValueError('Frozen file mismatch: '+name)
    return freeze


def load(root):
    spec = verify(root)['spec']
    for name, expected in json.loads((root/'model_manifest.json').read_text()).items():
        if sha(Path(spec['model'])/name) != expected: raise ValueError('Model mismatch: '+name)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(spec['model'], local_files_only=True)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(spec['model'], trust_remote_code=True,
        local_files_only=True, torch_dtype=torch.float32, attn_implementation='sdpa').to('cuda:0').eval()
    for p in model.parameters(): p.requires_grad_(False)
    E._ROOT = root
    return spec, model, tok


def configuration(root, method):
    return json.loads((root/'configs'/f'{method}.json').read_text())


def evaluate(adapter, record, method, depth, reference):
    calls, tokens = adapter.total_recurrent_calls, adapter.token_positions_processed
    out = E.evaluate_case(adapter, record, depth, method, base_reference=reference)
    out['metrics']['loop_token_positions_including_padding'] = adapter.token_positions_processed-tokens
    assert out['metrics']['evaluation_actual_decoder_calls'] == adapter.total_recurrent_calls-calls
    assert all(r['cache'] is False for r in out['generations'])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--shard', type=int, required=True, choices=[0,1])
    a = p.parse_args(); root = a.root
    out = root/f'shard{a.shard}'; out.mkdir(exist_ok=False)
    started = time.monotonic()
    status = {'status':'running', 'completed_cases':[], 'failed_case':None}
    current = None
    try:
        spec, model, tok = load(root)
        original_hashes = model_hashes(model)
        parameter_name = spec['parameter_name']
        weight = dict(model.named_parameters())[parameter_name]
        original = weight.detach().clone()
        requests = {r['case_id']:r for r in map(json.loads,(root/'data/train.jsonl').read_text().splitlines())}
        records = {r['case_id']:r for r in map(json.loads,(root/'data/eval.jsonl').read_text().splitlines())}
        selected = spec['case_ids'][a.shard::2]
        for current in selected:
            case = out/str(current); case.mkdir()
            case_start = time.monotonic()
            assert tensor_hash(weight) == original_hashes[parameter_name]
            bridge = EvaluationBridge(model, tok, {parameter_name:original})
            base = {}
            for k in (2,3,4):
                base[k] = evaluate(bridge, records[current], 'M0', k, None)
                dump(case/f'M0_K{k}.json',base[k])
            for method in spec['methods'][1:]:
                fit_start = time.monotonic()
                try:
                    with (case/f'{method}_fit.log').open('x') as log, contextlib.redirect_stdout(log):
                        _, backup = fit(model, tok, requests[current], configuration(root,method))
                    torch.cuda.synchronize()
                    fit_seconds = time.monotonic()-fit_start
                    assert set(backup)=={parameter_name}
                    assert torch.equal(backup[parameter_name],original)
                    hashes = model_hashes(model)
                    changed = [n for n in hashes if hashes[n]!=original_hashes[n]]
                    assert set(changed).issubset({parameter_name})
                    assert torch.isfinite(weight).all(), 'Nonfinite update'
                    # Save the actual deployed matrix, avoiding reconstruction roundoff.
                    checkpoint = case/f'{method}_weight.pt'
                    torch.save({parameter_name:weight.detach().cpu()},checkpoint)
                    dump(case/f'{method}_edit.json',{'method':method,'case_id':current,
                        'family_id':requests[current]['family_id'],'fit_seconds':fit_seconds,
                        'fit_K':4,'changed_parameters':changed,'update_frobenius':float((weight-original).norm()),
                        'parameter_count':weight.numel(),'checkpoint_sha256':sha(checkpoint),
                        'weight_sha256':hashes[parameter_name],'hyperparameters':configuration(root,method),
                        'status':'completed','zero_update':not bool(changed)})
                    for k in (2,3,4):
                        result = evaluate(bridge,records[current],method,k,base[k])
                        dump(case/f'{method}_K{k}.json',result)
                finally:
                    with torch.no_grad(): weight.copy_(original)
                    for param in model.parameters(): param.requires_grad_(False); param.grad=None
                    bridge.edit.edited=None;bridge.edit._enabled=True
                assert model_hashes(model)==original_hashes, 'Independent-case rollback failed'
            dump(case/'DONE.json',{'status':'completed','seconds':time.monotonic()-case_start})
            status['completed_cases'].append(current)
        status['status']='completed'
        assert model_hashes(model)==original_hashes
    except BaseException as exc:
        status.update(status='failed',failed_case=current,error=repr(exc))
        raise
    finally:
        status['elapsed_seconds']=time.monotonic()-started
        status['peak_allocated_GiB']=torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_initialized() else 0
        dump(out/'STATUS.json',status)


if __name__=='__main__': main()

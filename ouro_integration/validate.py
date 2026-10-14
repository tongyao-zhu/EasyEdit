"""One-case integration acceptance. Not a scientific baseline or method selection.

Uses a prespecified atomic request from the original screening population.
No multi-hop/locality answers enter this process. FP32, no KV cache, K=2/3/4.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adapter import FixedDepth, SharedCallTrace
from .upstream import load_ft


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def model_hashes(model):
    return {n: tensor_hash(p) for n, p in model.named_parameters()}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--model-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'status': 'running', 'scope': 'one atomic request, integration only',
              'torch': torch.__version__, 'transformers': transformers.__version__,
              'precision': 'float32', 'attention': 'sdpa', 'cache': False,
              'checks': {}, 'GPU_count': 1}
    try:
        model_manifest = json.loads(args.model_manifest.read_text())
        for name, expected in model_manifest.items():
            h = hashlib.sha256()
            with (args.model / name).open('rb') as f:
                for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                    h.update(chunk)
            if h.hexdigest() != expected:
                raise ValueError('Model snapshot hash mismatch: ' + name)
        report['checks']['model_snapshot_hashes'] = True
        request = json.loads(args.request.read_text())
        if set(request) != {'case_id', 'family_id', 'prompt', 'target_new', 'subject'}:
            raise ValueError('Atomic request schema violation')
        torch.manual_seed(17)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        load_start = time.monotonic()
        tok = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        tok.padding_side = 'right'
        model = AutoModelForCausalLM.from_pretrained(
            str(args.model), trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.float32, attn_implementation='sdpa').to('cuda:0').eval()
        report['load_seconds'] = time.monotonic() - load_start
        report['parameter_count'] = sum(p.numel() for p in model.parameters())
        original_hashes = model_hashes(model)
        for p in model.parameters():
            p.requires_grad_(False)
        inputs = tok(request['prompt'], return_tensors='pt').to('cuda:0')
        raw = []
        for k in (2, 3, 4):
            # Independent reference: configure native model explicitly, no adapter.
            old = (model.model.total_ut_steps, model.config.total_ut_steps, model.early_exit_step)
            model.model.total_ut_steps = model.config.total_ut_steps = k
            model.early_exit_step = k - 1
            with torch.no_grad():
                reference = model(**inputs, use_cache=False).logits.detach()
            model.model.total_ut_steps, model.config.total_ut_steps, model.early_exit_step = old
            with FixedDepth(model, k), SharedCallTrace(model, 17) as trace, torch.no_grad():
                observed = model(**inputs).logits
            assert torch.equal(reference, observed), 'Adapter/native logits differ'
            assert [r['loop'] for r in trace.records] == list(range(1, k + 1))
            assert len({r['parameter_id'] for r in trace.records}) == 1
            report['checks'][f'K{k}_native_equal_shared_calls'] = True
            # Gradient observation at every use of the same physical parameter.
            weight = model.model.layers[17].mlp.down_proj.weight
            weight.requires_grad_(True)
            with FixedDepth(model, k), SharedCallTrace(model, 17, gradients=True) as grad_trace:
                logits = model(**inputs).logits[:, -1]
                loss = -logits.log_softmax(-1)[:, int(reference[0, -1].argmax())].mean()
                loss.backward()
            assert len(grad_trace.records) == k
            assert all(r.get('gradient_norm', 0) > 0 for r in grad_trace.records)
            assert torch.isfinite(weight.grad).all() and weight.grad.norm() > 0
            raw.append({'k': k, 'input_ids': inputs.input_ids.tolist(),
                        'native_last_argmax': int(reference[0, -1].argmax()),
                        'gradient_trace': grad_trace.records})
            weight.grad = None
            weight.requires_grad_(False)
            report['checks'][f'K{k}_all_calls_receive_gradient'] = True
        ft = load_ft()
        hp = ft.FTHyperParams(
            layers=[16, 17], num_steps=2, lr=5e-4, weight_decay=0, kl_factor=0,
            norm_constraint=5e-4, rewrite_module_tmp='model.layers.{}.mlp.down_proj',
            layer_module_tmp='model.layers.{}', mlp_module_tmp='model.layers.{}.mlp',
            attn_module_tmp='model.layers.{}.self_attn', ln_f_module='model.norm',
            lm_head_module='lm_head', device=0, alg_name='FT', model_name=str(args.model),
            objective_optimization='target_new', batch_size=1, max_length=128)
        report['diagnostic_hparams'] = vars(hp)
        report['diagnostic_hparams_note'] = 'Two steps and two matrices test integration, not final benchmark hyperparameters'
        weights = {f'model.layers.{i}.mlp.down_proj.weight':
                   model.model.layers[i].mlp.down_proj.weight.detach().clone() for i in (16, 17)}
        flags = {n: p.requires_grad for n, p in model.named_parameters()}
        with FixedDepth(model, 4):
            fit_start = time.monotonic()
            edited, backup = ft.apply_ft_to_model(model, tok, [request], hp, return_orig_weights=True)
            torch.cuda.synchronize()
            report['ft_seconds'] = time.monotonic() - fit_start
        assert edited is model and set(backup) == set(weights)
        after_hashes = model_hashes(model)
        changed = [n for n in original_hashes if original_hashes[n] != after_hashes[n]]
        assert set(changed) == set(weights), 'Unexpected parameter change or zero update'
        assert all(torch.equal(backup[n], weights[n]) for n in weights)
        report['changed_parameters'] = changed
        report['checks']['upstream_FT_two_matrices_only'] = True
        for k in (2, 3, 4):
            with FixedDepth(model, k), SharedCallTrace(model, 17) as trace, torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=8, do_sample=False, use_cache=False,
                                     pad_token_id=tok.pad_token_id)
            assert all(r['parameter_id'] == id(model.model.layers[17].mlp.down_proj.weight) for r in trace.records)
            assert len(trace.records) % k == 0
            raw.append({'k': k, 'kind': 'atomic_prompt_diagnostic_after_FT',
                        'generated_ids': ids[0, inputs.input_ids.shape[1]:].tolist(),
                        'text': tok.decode(ids[0, inputs.input_ids.shape[1]:]),
                        'projection_calls': len(trace.records)})
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in backup:
                    p.copy_(backup[name])
                p.requires_grad_(flags[name])
        assert model_hashes(model) == original_hashes, 'Rollback failed'
        report['checks']['all_parameter_hashes_restored'] = True
        report['checks']['no_persistent_hooks'] = all(
            not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
        assert report['checks']['no_persistent_hooks']
        write(args.output / 'raw_diagnostics.json', raw)
        report['peak_allocated_GiB'] = torch.cuda.max_memory_allocated() / 2**30
        report['peak_reserved_GiB'] = torch.cuda.max_memory_reserved() / 2**30
        report['status'] = 'passed_integration_only'
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = repr(exc)
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        write(args.output / 'report.json', report)


if __name__ == '__main__':
    main()

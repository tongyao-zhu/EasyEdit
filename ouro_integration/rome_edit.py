"""Atomic-only, reversible invocation of the unchanged upstream ROME algorithm.

The yielded model has edited physical weights and no editor instrumentation.
Evaluation belongs outside this module; every native loop uses those weights.
Exceptions restore declared matrices, flags and tokenizer state. Unexpected
parameter mutation fails the whole case rather than silently contaminating it.
"""
from contextlib import contextmanager
import importlib

import torch

from .rome_tokens import prepare_atomic_request, audit_tokens
from .validate import model_hashes


@contextmanager
def edited_rome(runtime, tok, request):
    prepare_atomic_request(request)  # Reject evaluation fields before model access.
    hp, model = runtime.hparams, runtime.model
    if getattr(hp, 'use_chat_template', False) or hp.fact_token != 'subject_last':
        raise ValueError('Only the audited plain-text subject_last interface is supported')
    if not hp.layers or len(set(hp.layers)) != len(hp.layers):
        raise ValueError('Nonempty, distinct physical layers required')
    if any(type(i) is not int or not 0 <= i < len(model.model.layers) for i in hp.layers):
        raise ValueError('Physical layer outside this model')
    if type(hp.v_loss_layer) is not int or not 0 <= hp.v_loss_layer < len(model.model.layers):
        raise ValueError('Loss trace layer outside this model')
    for name, expected in [('rewrite_module_tmp','model.layers.{}.mlp.down_proj'),
                           ('mlp_module_tmp','model.layers.{}.mlp'),
                           ('layer_module_tmp','model.layers.{}')]:
        if getattr(hp,name) != expected:
            raise ValueError('Unvalidated Ouro module mapping: '+name)
    params = dict(model.named_parameters())
    ids = {n:id(p) for n,p in params.items()}
    flags = {n:p.requires_grad for n,p in params.items()}
    names = [hp.rewrite_module_tmp.format(i)+'.weight' for i in hp.layers]
    if any(n not in params or params[n].ndim != 2 for n in names):
        raise ValueError('Rewrite matrix not found')
    before = model_hashes(model)
    if getattr(runtime,'baseline_hashes',before) != before:
        raise ValueError('Runtime caches cannot be reused on a different parameter state')
    runtime.baseline_hashes = before
    backups = {n:params[n].detach().clone() for n in names}
    side = tok.padding_side
    try:
        tok.padding_side = 'right'
        with runtime:
            rome = runtime.rome
            contexts = rome.get_context_templates(model,tok,hp.context_template_length_params)
            repr_tools = importlib.import_module(rome.__package__+'.repr_tools')
            token_evidence = audit_tokens(tok,request,contexts,repr_tools)
            edited, returned = rome.apply_rome_to_model(model,tok,[request],hp,
                copy=False,return_orig_weights=True)
            if edited is not model or set(returned) != set(backups):
                raise RuntimeError('Unexpected upstream model/backup contract')
            if any(not torch.equal(returned[n],backups[n]) for n in names):
                raise RuntimeError('Upstream backup does not match the base matrix')
        after = model_hashes(model)
        changed = [n for n in before if before[n] != after[n]]
        if not set(changed).issubset(names):
            raise RuntimeError('ROME modified an undeclared parameter')
        if any(not torch.isfinite(params[n]).all() for n in names):
            raise RuntimeError('Non-finite deployed matrix')
        for n,p in params.items(): p.requires_grad_(flags[n])
        tok.padding_side = side
        yield {'model':model, 'token_audit':token_evidence,
               'contexts':list(contexts), 'changed_parameters':changed,
               'zero_update':not changed, 'before_hashes':before, 'edited_hashes':after,
               'rewrite_parameters':names, 'cache_identity':runtime.digest}
    finally:
        with torch.no_grad():
            for n in names: params[n].copy_(backups[n])
        for n,p in params.items(): p.requires_grad_(flags[n])
        tok.padding_side = side
        if {n:id(p) for n,p in model.named_parameters()} != ids:
            raise RuntimeError('Parameter identities changed during the edit')
        if model_hashes(model) != before:
            raise RuntimeError('Whole-model rollback failed; stop this worker')

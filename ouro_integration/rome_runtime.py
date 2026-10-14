"""Temporary execution wiring for upstream ROME; no change to u/v/solve formulas.

One runtime per worker process. Statistics and generated contexts cannot leak
across explicitly bound identities. This is integration, not an accepted recipe.
"""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
from threading import Lock

import torch

from .adapter import FixedDepth
from .context_generation import generate_full_prefix
from .trace_scope import OuroTraceScope
from .upstream import load_rome


_runtime_lock = Lock()
IDENTITY_FIELDS = {'model_manifest_sha256', 'tokenizer_manifest_sha256', 'source_sha256',
                   'corpus_manifest_sha256', 'K', 'occurrence', 'precision', 'attention',
                   'cache', 'mom2_dataset', 'mom2_n_samples', 'mom2_dtype', 'context_seed',
                   'context_template_length_params'}


def cache_digest(identity):
    if set(identity) != IDENTITY_FIELDS:
        raise ValueError('Cache identity must specify the full provenance contract')
    for key in ('model_manifest_sha256', 'tokenizer_manifest_sha256', 'source_sha256', 'corpus_manifest_sha256'):
        value = identity[key]
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Invalid provenance digest: '+key)
    if type(identity['K']) is not int or identity['K'] not in (2,3,4):
        raise ValueError('Unsupported K')
    if type(identity['occurrence']) is not int or not 0 <= identity['occurrence'] < identity['K']:
        raise ValueError('Invalid occurrence')
    if (identity['precision'],identity['attention'],identity['cache']) != ('float32','sdpa',False):
        raise ValueError('Only the declared FP32 full-prefix SDPA path is supported')
    if type(identity['context_seed']) is not int or identity['context_seed'] < 0:
        raise ValueError('Explicit nonnegative context seed required')
    return hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class ROMERuntime:
    def __init__(self, model, hparams, identity, stats_root):
        self.model, self.hparams = model, deepcopy(hparams)
        self.identity = deepcopy(identity)
        self.digest = cache_digest(identity)
        self.stats_path = Path(stats_root)/self.digest
        for attr in ('mom2_dataset','mom2_n_samples','mom2_dtype','context_template_length_params'):
            if getattr(hparams,attr) != identity[attr]:
                raise ValueError('Hyperparameter/cache identity mismatch: '+attr)
        self.stack = None
        self.inverse_cache = {}
        self.context_templates = None

    def __enter__(self):
        if cache_digest(self.identity) != self.digest:
            raise ValueError('Cache identity was mutated')
        for attr in ('mom2_dataset','mom2_n_samples','mom2_dtype','context_template_length_params'):
            if getattr(self.hparams,attr) != self.identity[attr]:
                raise ValueError('Runtime hyperparameter identity changed')
        if self.stack is not None or not _runtime_lock.acquire(blocking=False):
            raise RuntimeError('ROME runtime is non-reentrant and process-exclusive')
        self.stack = ExitStack()
        self.stack.callback(_runtime_lock.release)
        try:
            self.rome = load_rome()
            u = importlib.import_module(self.rome.__package__+'.compute_u')
            self.stats_path.mkdir(parents=True,exist_ok=True)
            manifest = self.stats_path/'IDENTITY.json'
            encoded = json.dumps(self.identity,sort_keys=True,indent=2)+'\n'
            if manifest.exists():
                if manifest.read_text()!=encoded: raise ValueError('Statistics identity mismatch on disk')
            else:
                with manifest.open('x') as f: f.write(encoded)
            self.hparams.stats_dir = str(self.stats_path)
            old = (self.rome.generate_fast,self.rome.CONTEXT_TEMPLATES_CACHE,u.inv_mom2_cache,
                   self.rome.get_context_templates)
            def restore():
                self.context_templates = self.rome.CONTEXT_TEMPLATES_CACHE
                (self.rome.generate_fast,self.rome.CONTEXT_TEMPLATES_CACHE,u.inv_mom2_cache,
                 self.rome.get_context_templates) = old
            self.stack.callback(restore)
            self.rome.generate_fast = generate_full_prefix
            self.rome.CONTEXT_TEMPLATES_CACHE = self.context_templates
            u.inv_mom2_cache = self.inverse_cache
            def contexts(model, tok, length_params):
                if model is not self.model or length_params != self.identity['context_template_length_params']:
                    raise ValueError('Context generator identity mismatch')
                if self.rome.CONTEXT_TEMPLATES_CACHE is None:
                    device = next(model.parameters()).device
                    devices = [device.index] if device.type == 'cuda' else []
                    with torch.random.fork_rng(devices=devices):
                        torch.manual_seed(self.identity['context_seed'])
                        return old[3](model,tok,length_params)
                return old[3](model,tok,length_params)
            self.rome.get_context_templates = contexts
            self.stack.enter_context(FixedDepth(self.model,self.identity['K']))
            self.scope = self.stack.enter_context(OuroTraceScope(self.model,self.rome.nethook,
                self.identity['occurrence'],self.identity['K']))
            return self
        except BaseException:
            self.stack.close(); self.stack = None
            raise

    def __exit__(self, *args):
        try: return self.stack.__exit__(*args)
        finally: self.stack = None

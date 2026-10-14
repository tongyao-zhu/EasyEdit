"""Explicit physical batching override; never changes tokenization or sample order.

Upstream layer_stats ties batch_tokens to max sequence length. This context only
changes its collation factory after TokenizedDataset keeps the native maxlen.
One oversized document remains a single complete document, never split/truncated.
"""
from contextlib import contextmanager
from threading import Lock
_lock=Lock()

@contextmanager
def physical_moment_batches(layer_stats_module, padded_token_budget):
    if type(padded_token_budget) is not int or padded_token_budget < 1:
        raise ValueError('Positive physical padded-token budget required')
    if not _lock.acquire(blocking=False):
        raise RuntimeError('Physical moment batching is process-exclusive')
    original=layer_stats_module.length_collation
    layer_stats_module.length_collation=lambda upstream_budget: original(min(upstream_budget,padded_token_budget))
    try:
        yield
    finally:
        layer_stats_module.length_collation=original
        _lock.release()

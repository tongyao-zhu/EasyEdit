"""Declared contiguous partition of an already frozen upstream sample.

Only partitions data-loader iteration across workers. Collector, tokenization,
feature masking and per-worker SecondMoment.add remain upstream. Merge adds
FP32 unnormalized sums in fixed shard order and divides once by total tokens.
"""
from contextlib import contextmanager
from threading import Lock
import importlib
import torch
_lock=Lock()

@contextmanager
def contiguous_moment_shard(layer_stats_module, row_order, start, end):
    if not (type(start) is int and type(end) is int and 0<=start<end<=len(row_order)):
        raise ValueError('Invalid shard interval')
    if start%100 or end%100:
        raise ValueError('Shard boundaries must preserve original 100-document groups')
    if len(set(row_order))!=len(row_order):raise ValueError('Duplicate source rows')
    if not _lock.acquire(blocking=False):raise RuntimeError('Shard wiring is process-exclusive')
    original=layer_stats_module.tally
    rs=importlib.import_module('_ouro_easyedit_upstream.util.runningstats')
    def tally(stat,dataset,cache=None,**kwargs):
        if kwargs.get('sample_size')!=len(row_order) or kwargs.get('random_sample')!=1 or kwargs.get('batch_size')!=100:
            raise ValueError('Unexpected original collector sample/batch configuration')
        if dataset is None:raise ValueError('No shard cache reuse; each allocation is unique')
        kwargs.pop('sample_size');kwargs.pop('random_sample')
        return original(stat,dataset,cache=cache,sampler=rs.FixedSubsetSampler(row_order[start:end]),**kwargs)
    layer_stats_module.tally=tally
    try:yield
    finally:
        layer_stats_module.tally=original
        _lock.release()

def merge_second_moments(states):
    """Fixed-order FP32 sum merge; states are upstream SecondMoment instances."""
    if not states:raise ValueError('No shard statistics')
    rs=importlib.import_module('_ouro_easyedit_upstream.util.runningstats')
    result=rs.SecondMoment()
    shape=None
    for stat in states:
        if stat.count<=0 or stat.mom2.dtype!=torch.float32 or not torch.isfinite(stat.mom2).all():
            raise ValueError('Invalid shard sum')
        if shape is None:shape=stat.mom2.shape;result.mom2=torch.zeros_like(stat.mom2,device='cpu')
        if stat.mom2.shape!=shape:raise ValueError('Incompatible shard shape')
        result.mom2.add_(stat.mom2.cpu());result.count+=int(stat.count)
    return result

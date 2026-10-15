"""Bind an audited full-corpus cache; never alter the upstream inverse formula."""
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import shutil

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()

def bind_production_moment(runtime, binding):
    """All expected digests are frozen by the caller before its GPU allocation."""
    identity_path=Path(binding['identity_file'])
    cache=Path(binding['cache_file'])
    assert sha(identity_path)==binding['identity_sha256']
    assert sha(cache)==binding['cache_sha256']
    identity=json.loads(identity_path.read_text())
    depth = runtime.identity['K']
    occurrence = runtime.identity['occurrence']
    assert type(depth) is int and depth in (2, 4)
    assert type(occurrence) is int and occurrence == depth - 1
    expected={'samples':100000,'tokens':78683546,'K':depth,'occurrence':occurrence,'layer':17,
              'precision':'float32','cache':False,'attention':'native_sdpa_unchanged','TF32':False}
    for k,v in expected.items():assert identity[k]==v,(k,identity[k])
    assert identity['model_manifest_sha256']==runtime.identity['model_manifest_sha256']
    assert runtime.identity['corpus_manifest_sha256']==binding['identity_sha256']
    hp=runtime.hparams
    assert hp.mom2_adjustment and hp.layers==[17]
    assert (hp.mom2_dataset,hp.mom2_dtype,hp.mom2_n_samples)==('wikipedia','float32',100000)
    model_name=runtime.model.config._name_or_path.rsplit('/',1)[-1]
    assert model_name==binding['model_snapshot']
    relative=Path(model_name)/'wikipedia_stats/model.layers.17.mlp.down_proj_float32_mom2_100000.npz'
    dest=runtime.stats_path/relative
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():assert sha(dest)==binding['cache_sha256']
    else:
        with cache.open('rb') as src,dest.open('xb') as dst:shutil.copyfileobj(src,dst)
        assert sha(dest)==binding['cache_sha256']
    receipt=runtime.stats_path/'PRODUCTION_BINDING.json'
    encoded=json.dumps(binding,sort_keys=True,indent=2)+'\n'
    if receipt.exists():assert receipt.read_text()==encoded
    else:receipt.write_text(encoded)
    return dest

@contextmanager
def forbid_statistic_collection():
    """A missing production cache must fail, not silently launch a new collection."""
    from .upstream import load_rome
    rome=load_rome();ls=importlib.import_module(rome.__package__+'.layer_stats')
    old=ls.load_dataset
    def forbidden(*args,**kwargs):raise RuntimeError('Production moment cache miss: recollection forbidden')
    ls.load_dataset=forbidden
    try:yield
    finally:ls.load_dataset=old

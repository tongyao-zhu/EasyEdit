"""Prespecified 128-case EasyEdit ROME production-mom2 baseline; reuse frozen FT/M0/no-mom2."""
import argparse
import contextlib
import json
from pathlib import Path
import time

import torch

from .benchmark import load, dump, sha, evaluate
from .evaluation_bridge import EvaluationBridge
from .rome_runtime import ROMERuntime
from .production_mom2 import bind_production_moment, forbid_statistic_collection
from .rome_edit import edited_rome
from .upstream import load_rome
from .validate import model_hashes, tensor_hash


def assert_replay(actual, expected):
    if len(actual['generations']) != len(expected['generations']):
        raise ValueError('M0 replay query count mismatch')
    for got, old in zip(actual['generations'],expected['generations']):
        for key in ('kind','prompt','input_ids','generated_ids','text','correct'):
            if got[key] != old[key]:raise ValueError('M0 replay mismatch: '+key)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--shard',type=int,required=True,choices=[0,1]);a=p.parse_args();r=a.root
    out=r/f'shard{a.shard}';out.mkdir(exist_ok=True)
    if (out/'STATUS.json').exists():raise RuntimeError('No rerun of an existing shard')
    status={'status':'running','completed_cases':[],'failed_case':None};started=time.monotonic();current=None
    try:
        spec,model,tok=load(r);method=spec['new_method'];original_hashes=model_hashes(model)
        name=spec['parameter_name'];weight=dict(model.named_parameters())[name];original=weight.detach().clone()
        requests={x['case_id']:x for x in map(json.loads,(r/'data/train.jsonl').read_text().splitlines())}
        records={x['case_id']:x for x in map(json.loads,(r/'data/eval.jsonl').read_text().splitlines())}
        hp=load_rome().ROMEHyperParams(**json.loads((r/'configs'/f'{method}.json').read_text()))
        assert hp.mom2_adjustment is True
        identity=json.loads((r/'runtime_identity.json').read_text())
        runtime=ROMERuntime(model,hp,identity,out/'statistics_identity')
        runtime.context_templates=json.loads((r/'CONTEXTS.json').read_text())
        bound=bind_production_moment(runtime,json.loads((r/'PRODUCTION_BINDING.json').read_text()))
        dump(out/'PRODUCTION_CACHE.json',{'path':str(bound),'sha256':sha(bound)})
        selected=spec['case_ids'][a.shard::2]
        for current in selected:
            case=out/str(current);case.mkdir(exist_ok=True);case_start=time.monotonic()
            if any(case.glob(method+'*')):raise RuntimeError('Existing scientific output; no retry')
            assert tensor_hash(weight)==original_hashes[name]
            bridge=EvaluationBridge(model,tok,{name:original})
            base={k:json.loads((case/f'M0_K{k}.json').read_text()) for k in (2,3,4)}
            if current==selected[0]:
                replay=out/'base_replay';replay.mkdir(exist_ok=False)
                for k in (2,3,4):
                    actual=evaluate(bridge,records[current],'M0',k,None)
                    dump(replay/f'M0_K{k}.json',actual);assert_replay(actual,base[k])
                dump(replay/'STATUS.json',{'status':'passed_exact_tokens_and_correctness','case_id':current})
            fit_started=time.monotonic()
            try:
                with (case/f'{method}_fit.log').open('x') as log,contextlib.redirect_stdout(log):
                    with forbid_statistic_collection(),edited_rome(runtime,tok,requests[current]) as edit:
                        torch.cuda.synchronize();fit_seconds=time.monotonic()-fit_started
                        checkpoint=case/f'{method}_weight.pt'
                        torch.save({name:weight.detach().cpu()},checkpoint)
                        dump(case/f'{method}_edit.json',{'method':method,'case_id':current,
                            'family_id':requests[current]['family_id'],'status':'completed',
                            'fit_K':4,'fit_seconds_including_token_and_state_audits':fit_seconds,
                            'changed_parameters':edit['changed_parameters'],'zero_update':edit['zero_update'],
                            'update_frobenius':float((weight-original).norm()),'parameter_count':weight.numel(),
                            'checkpoint_sha256':sha(checkpoint),'weight_sha256':tensor_hash(weight),
                            'hyperparameters':vars(runtime.hparams),'runtime_identity':identity,
                            'token_audit':edit['token_audit'],'before_hashes':edit['before_hashes'],
                            'edited_hashes':edit['edited_hashes']})
                        for k in (2,3,4):
                            result=evaluate(bridge,records[current],method,k,base[k])
                            dump(case/f'{method}_K{k}.json',result)
            finally:
                with torch.no_grad():weight.copy_(original)
                for parameter in model.parameters():parameter.requires_grad_(False);parameter.grad=None
                bridge.edit.edited=None;bridge.edit._enabled=True
            restored=model_hashes(model);assert restored==original_hashes
            dump(case/'ROME_MOM2_DONE.json',{'status':'completed','seconds':time.monotonic()-case_start,
                                      'restored_parameter_hashes':restored})
            status['completed_cases'].append(current)
        status['status']='completed'
    except BaseException as exc:
        status.update(status='failed',failed_case=current,error=repr(exc));raise
    finally:
        status['elapsed_seconds']=time.monotonic()-started
        status['peak_allocated_GiB']=torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_initialized() else 0
        dump(out/'STATUS.json',status)


if __name__=='__main__':main()

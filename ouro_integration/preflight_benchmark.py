"""Outcome-blind acceptance for frozen generation and FT entry points."""
import argparse
import contextlib
import json
from pathlib import Path
import torch
from .benchmark import load, configuration, dump
from .evaluation_bridge import EvaluationBridge
from .adapter import FixedDepth
from .fit_baseline import fit
from .validate import model_hashes
from . import frozen_evaluation as E


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    out=r/'preflight';out.mkdir(exist_ok=False)
    report={'status':'running','checks':{}}
    try:
        spec,model,tok=load(r);start=model_hashes(model);name=spec['parameter_name']
        w=dict(model.named_parameters())[name];original=w.detach().clone()
        requests=list(map(json.loads,(r/'data/train.jsonl').read_text().splitlines()))
        records=list(map(json.loads,(r/'data/eval.jsonl').read_text().splitlines()))
        assert [x['case_id'] for x in requests]==spec['case_ids']
        for req in requests:
            combined=req['prompt']+' '+req['target_new']
            pt=tok(req['prompt'])['input_ids'];ct=tok(combined)['input_ids']
            assert ct[:len(pt)]==pt and len(ct)>len(pt),'FT-M token boundary mismatch'
            assert len(ct)<=256
        manifest=E.frozen_prompt_manifest()
        for rec in records:
            for item in E._items(rec,'full',manifest):
                assert 0<len(tok(item['prompt'],add_special_tokens=False)['input_ids'])<=2048
        report['checks']['all128_input_token_boundaries_and_lengths']=True
        bridge=EvaluationBridge(model,tok,{name:original})
        # First eight prompts of the predeclared first case, no score inspection.
        prompts=[x['prompt'] for x in E._items(records[0],'full',manifest)][:8]
        for k in (2,3,4):
            batch=bridge.batched_generate(prompts,k=k,max_new_tokens=4)
            single=bridge.batched_generate(prompts,k=k,max_new_tokens=4,batch_size=1)
            assert [x['generated_ids'] for x in batch]==[x['generated_ids'] for x in single],'Batch token mismatch'
            # Independent native HF generation must match before the stop boundary.
            for row in single[:2]:
                tokens=tok(row['prompt'],return_tensors='pt',add_special_tokens=False).to('cuda:0')
                with FixedDepth(model,k),torch.no_grad():
                    native=model.generate(**tokens,max_new_tokens=4,use_cache=False,do_sample=False,pad_token_id=tok.pad_token_id)
                got=native[0,tokens.input_ids.shape[1]:].tolist()
                assert got[:len(row['generated_ids'])]==row['generated_ids']
            report['checks'][f'K{k}_batch_single_and_native_tokens']=True
        for method in spec['methods'][1:]:
            hp=configuration(r,method);hp['num_steps']=2
            with (out/f'{method}.log').open('x') as log,contextlib.redirect_stdout(log):
                _,backup=fit(model,tok,requests[0],hp)
            assert set(backup)=={name} and torch.isfinite(w).all()
            edited=w.detach().clone()
            bridge.edit.enabled=False;assert torch.equal(w,original)
            bridge.edit.enabled=True;assert torch.equal(w,edited)
            with torch.no_grad():w.copy_(original)
            assert model_hashes(model)==start
            report['checks'][f'{method}_fit_and_teacher_swap_restore']=True
        report['status']='passed_correctness_only'
    except BaseException as exc:
        report.update(status='failed',error=repr(exc));raise
    finally:
        dump(out/'REPORT.json',report)


if __name__=='__main__':main()

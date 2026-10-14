"""One prespecified ROME integration case, not a scientific method comparison.

Use the pinned EasyEdit llama-2 example recipe, explicitly WITHOUT mom2, with
declared Ouro location mapping. No covariance-adjusted ROME claim is made.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adapter import FixedDepth, SharedCallTrace
from .rome_runtime import ROMERuntime
from .rome_tokens import audit_tokens
from .rome_edit import edited_rome
from .upstream import load_rome
from .validate import model_hashes, tensor_hash, write


def main():
    p=argparse.ArgumentParser()
    for name in ('model','model-manifest','request','requests','configuration','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    started=time.monotonic();report={'status':'running','scope':'single-case upstream ROME example without mom2; integration only','checks':{}}
    try:
        manifest=json.loads(a.model_manifest.read_text())
        for name,h in manifest.items():
            digest=hashlib.sha256()
            with (a.model/name).open('rb') as f:
                for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
            assert digest.hexdigest()==h,name
        configuration=json.loads(a.configuration.read_text())
        request=json.loads(a.request.read_text())
        requests=[json.loads(line) for line in a.requests.read_text().splitlines()]
        assert len(requests)==128 and len({r['family_id'] for r in requests})==128 and requests[0]==request
        assert configuration['mom2_adjustment'] is False
        report['configuration']=configuration
        torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        tok=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True);tok.padding_side='right'
        if tok.pad_token_id is None:tok.pad_token_id=tok.eos_token_id
        model=AutoModelForCausalLM.from_pretrained(str(a.model),trust_remote_code=True,local_files_only=True,
            torch_dtype=torch.float32,attn_implementation='sdpa').to('cuda:0').eval()
        assert len(model.model.layers)==24
        original=model_hashes(model);report['original_parameter_hashes']=original
        rome=load_rome();hp=rome.ROMEHyperParams(**configuration)
        hp.model_name=str(a.model)
        token_keys=['tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt']
        token_identity={n:manifest[n] for n in token_keys}
        identity=dict(model_manifest_sha256=hashlib.sha256(a.model_manifest.read_bytes()).hexdigest(),
            tokenizer_manifest_sha256=hashlib.sha256(json.dumps(token_identity,sort_keys=True).encode()).hexdigest(),
            source_sha256=hashlib.sha256((Path(__file__).resolve().parents[2]/'FREEZE.json').read_bytes()).hexdigest(),
            corpus_manifest_sha256=hashlib.sha256(b'{"status":"not_used","mom2_adjustment":false}').hexdigest(),
            K=4,occurrence=3,precision='float32',attention='sdpa',cache=False,mom2_dataset=hp.mom2_dataset,
            mom2_n_samples=hp.mom2_n_samples,mom2_dtype=hp.mom2_dtype,context_seed=17,
            context_template_length_params=hp.context_template_length_params)
        report['identity']=identity
        runtime=ROMERuntime(model,hp,identity,a.output/'statistics_identity')
        with runtime:
            contexts=rome.get_context_templates(model,tok,hp.context_template_length_params)
            repr_tools=importlib.import_module(rome.__package__+'.repr_tools')
            token_checks=[audit_tokens(tok,r,contexts,repr_tools) for r in requests]
        write(a.output/'CONTEXTS.json',contexts)
        write(a.output/'TOKEN_AUDIT_128.json',token_checks)
        report['checks']['all128_actual_context_token_contract']=len(token_checks)==128
        assert len(contexts)==21
        inputs=tok(request['prompt'],return_tensors='pt').to('cuda:0')
        base={}
        for k in (2,3,4):
            with FixedDepth(model,k),torch.no_grad():base[k]=model(**inputs).logits.detach().clone()
        names=[hp.rewrite_module_tmp.format(i)+'.weight' for i in hp.layers]
        weights={n:dict(model.named_parameters())[n].detach().clone() for n in names}
        captured={};original_execute=rome.execute_rome
        def capture(*args,**kwargs):
            result=original_execute(*args,**kwargs)
            captured.update({n:(u.detach().clone(),v.detach().clone()) for n,(u,v) in result.items()})
            return result
        rome.execute_rome=capture
        raw=[]
        try:
            fit_start=time.monotonic()
            with edited_rome(runtime,tok,request) as edited:
                torch.cuda.synchronize();report['fit_and_audit_seconds']=time.monotonic()-fit_start
                assert set(captured)==set(names)
                for n,(u,v) in captured.items():
                    update=rome.upd_matrix_match_shape(u.unsqueeze(1)@v.unsqueeze(0),weights[n].shape)
                    assert torch.equal(dict(model.named_parameters())[n],weights[n]+update)
                    assert torch.isfinite(u).all() and torch.isfinite(v).all()
                torch.save({n:(u.cpu(),v.cpu()) for n,(u,v) in captured.items()},a.output/'FACTORS.pt')
                report['factor_file_sha256']=hashlib.sha256((a.output/'FACTORS.pt').read_bytes()).hexdigest()
                report['checks']['deployed_weights_match_rank_one_factors']=True
                report['zero_update']=edited['zero_update']
                report['changed_parameters']=edited['changed_parameters']
                report['edited_parameter_hashes']=edited['edited_hashes']
                report['token_contract_case']=edited['token_audit']
                assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
                report['checks']['no_editor_hooks_during_deployment']=True
                for k in (2,3,4):
                    with FixedDepth(model,k),SharedCallTrace(model,17) as trace,torch.no_grad():
                        logits=model(**inputs).logits
                        ids=model.generate(**inputs,max_new_tokens=8,do_sample=False,use_cache=False,pad_token_id=tok.pad_token_id)
                    assert len(trace.records)%k==0
                    assert all(r['parameter_id']==id(model.model.layers[17].mlp.down_proj.weight) for r in trace.records)
                    raw.append({'K':k,'input_ids':inputs.input_ids.tolist(),
                        'generated_ids':ids[0,inputs.input_ids.shape[1]:].tolist(),
                        'text':tok.decode(ids[0,inputs.input_ids.shape[1]:]),
                        'base_last_logits_sha256':tensor_hash(base[k][:,-1]),
                        'edited_last_logits_sha256':tensor_hash(logits[:,-1]),'calls':trace.records})
                report['checks']['shared_deployment_K234']=True
        finally:rome.execute_rome=original_execute
        restored=model_hashes(model);report['restored_parameter_hashes']=restored
        assert restored==original
        for k in (2,3,4):
            with FixedDepth(model,k),torch.no_grad():assert torch.equal(model(**inputs).logits,base[k])
        report['checks']['whole_model_and_logits_restored']=True
        report['checks']['no_remaining_hooks']=all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
        assert report['checks']['no_remaining_hooks']
        write(a.output/'raw_atomic_diagnostics.json',raw)
        report['peak_allocated_GiB']=torch.cuda.max_memory_allocated()/2**30
        report['status']='passed_single_case_no_mom2_integration_only'
    except BaseException as exc:
        report['status']='failed';report['error']=repr(exc);raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started;write(a.output/'report.json',report)


if __name__=='__main__':main()

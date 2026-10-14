"""CPU tokenizer precheck, not acceptance of a generated scientific context bank."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

from .rome_tokens import audit_tokens
from .upstream import load_rome


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--model-manifest',type=Path,required=True)
    p.add_argument('--requests',type=Path,required=True)
    p.add_argument('--requests-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('This precheck must run with empty CUDA')
    a.output.mkdir(exist_ok=False)
    manifest=json.loads(a.model_manifest.read_text());verified={}
    for name in ('config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt'):
        value=hashlib.sha256((a.model/name).read_bytes()).hexdigest()
        if value != manifest[name]:raise ValueError('Tokenizer binding mismatch: '+name)
        verified[name]=value
    if hashlib.sha256(a.requests.read_bytes()).hexdigest()!=a.requests_sha256:
        raise ValueError('Atomic request file hash mismatch')
    requests=[json.loads(line) for line in a.requests.read_text().splitlines()]
    if len(requests)!=128 or len({r['family_id'] for r in requests})!=128:
        raise ValueError('Expected original128 screening families')
    tok=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True)
    tok.padding_side='right'
    if tok.pad_token_id is None:tok.pad_token_id=tok.eos_token_id
    rome=load_rome();repr_tools=importlib.import_module(rome.__package__+'.repr_tools')
    fixtures=['{}','The. {}','Therefore. {}','Because. {}','I. {}','You. {}']
    rows=[]
    for request in requests:
        try:rows.append(audit_tokens(tok,request,fixtures,repr_tools))
        except Exception as exc:
            rows.append({'case_id':request['case_id'],'family_id':request['family_id'],
                         'status':'failed','error':repr(exc)})
    failures=[r for r in rows if r['status']=='failed']
    report={'status':'failed' if failures else 'passed_CPU_fixtures_only','GPU_hours':0,
            'cases':len(rows),'failures':len(failures),'tokenizer_binding':verified,
            'atomic_requests_sha256':a.requests_sha256,'contexts':fixtures,
            'scope':'clean prompt + five fixed CPU stress fixtures only; no model generation or fit',
            'limitation':'Actual generated context bank must pass audit_tokens again before any fit',
            'rows':rows}
    (a.output/'TOKEN_REPORT.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()

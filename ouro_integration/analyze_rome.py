"""CPU-only frozen family-paired analysis; missing cells remain explicit."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np


def mean(values):
    return sum(values)/len(values) if values else None


def recompute(rows):
    direct=[x for x in rows if x['kind']=='direct'];mh=[x for x in rows if x['kind']=='multihop']
    chain=[x for x in rows if x['kind']=='unchanged_chain'];loc=[x for x in rows if x['kind']=='locality']
    strict=int(all(x['correct'] for x in direct));uc=int(all(x['correct'] for x in chain))
    good=[x for x in loc if x['base_correct']]
    return {'E_QA':mean([x['correct'] for x in direct]),'E_QA_strict':strict,
            'A_MH':mean([x['correct'] for x in mh]),'U_chain_all':uc,
            'Joint':strict*uc*mean([x['correct'] for x in mh]),
            'locality_raw_accuracy':mean([x['correct'] for x in loc]),
            'locality_output_agreement':mean([x['output_agreement'] for x in loc]),
            'locality_base_correct_accuracy_change':mean([int(x['correct'])-1 for x in good])}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    out=r/'analysis';out.mkdir(exist_ok=False)
    spec=json.loads((r/'FREEZE.json').read_text())['spec']
    records={};status=[];audit=[]
    for idx,cid in enumerate(spec['case_ids']):
        for method in spec['methods']:
            for k in [2,3,4]:
                path=r/f'shard{idx%2}'/str(cid)/f'{method}_K{k}.json'
                row={'case_id':cid,'method':method,'k':k,'status':'not_run_or_incomplete'}
                if path.exists():
                    data=json.loads(path.read_text());check=recompute(data['generations']);m=data['metrics']
                    for key,value in check.items():
                        observed=m[key]
                        assert value==observed or value is not None and observed is not None and abs(value-observed)<1e-12,(cid,method,k,key)
                    records[(cid,method,k)]=m;row['status']='completed'
                    audit.append({'case_id':cid,'method':method,'k':k,'independent_raw_metrics_matched':list(check)})
                status.append(row)
    (out/'matrix_status.json').write_text(json.dumps(status,indent=2)+'\n')
    (out/'per_family_metrics.json').write_text(json.dumps(list(records.values()),indent=2)+'\n')
    (out/'raw_metric_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    metrics=['E_QA','E_QA_strict','A_MH','Joint','U_chain_all','locality_raw_accuracy','locality_output_agreement','locality_base_correct_accuracy_change','locality_kl']
    rng=np.random.default_rng(17)
    bootstrap=rng.integers(0,128,size=(10000,128))
    comparisons=[];summaries=[];transitions=[]
    for k in [2,3,4]:
        for method in spec['methods']:
            cells=[records.get((cid,method,k)) for cid in spec['case_ids']]
            for metric in metrics:
                valid=[c[metric] for c in cells if c is not None and c.get(metric) is not None]
                summaries.append({'method':method,'k':k,'metric':metric,'available_family_count':len(valid),
                                  'expected_family_count':128,'available_mean':mean(valid),'full_matrix':all(c is not None for c in cells)})
        for right,left in itertools.combinations(spec['methods'],2):
            for metric in metrics:
                pairs=[(records.get((cid,left,k)),records.get((cid,right,k))) for cid in spec['case_ids']]
                eligible=[(x,y) for x,y in pairs if x is not None and y is not None and x.get(metric) is not None and y.get(metric) is not None]
                full=len(eligible)==128
                delta=np.array([x[metric]-y[metric] for x,y in eligible])
                ci=np.quantile(delta[bootstrap].mean(1),[.025,.975]).tolist() if full else None
                comparisons.append({'left':left,'right':right,'k':k,'metric':metric,'paired_family_count':len(eligible),
                    'full_population':full,'available_mean_difference':float(delta.mean()) if len(delta) else None,
                    'paired95CI':ci,'primary':k==4 and metric=='A_MH' and left==spec['new_method'],'CI_status':'computed' if full else 'withheld_incomplete_or_structurally_missing'})
            for kind in ['E_QA_strict','U_chain_all']:
                counts={str((x,y)):0 for x in [0,1] for y in [0,1]}
                for cid in spec['case_ids']:
                    l,rr=records.get((cid,left,k)),records.get((cid,right,k))
                    if l is not None and rr is not None:counts[str((int(rr[kind]),int(l[kind])))]+=1
                transitions.append({'left':left,'right':right,'k':k,'metric':kind,'baseline_to_edited':counts})
    # Within-method paired depth comparisons, using the exact same stored edit.
    for method in spec['methods']:
        for high,low in [(3,2),(4,2),(4,3)]:
            for metric in metrics:
                pairs=[(records.get((cid,method,high)),records.get((cid,method,low))) for cid in spec['case_ids']]
                eligible=[(x,y) for x,y in pairs if x is not None and y is not None and x.get(metric) is not None and y.get(metric) is not None]
                d=np.array([x[metric]-y[metric] for x,y in eligible]);full=len(d)==128
                comparisons.append({'method':method,'high_K':high,'low_K':low,'metric':metric,'paired_family_count':len(d),'full_population':full,
                    'available_mean_difference':float(d.mean()) if len(d) else None,'paired95CI':np.quantile(d[bootstrap].mean(1),[.025,.975]).tolist() if full else None,'primary':False})
    for name,value in [('summary',summaries),('paired_comparisons',comparisons),('repair_break',transitions)]:
        (out/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
    # Query-level repair/break is descriptive; retain every query, no successful-edit subset.
    query_rows=[]
    for k in (2,3,4):
        for right,left in itertools.combinations(spec['methods'],2):
            for idx,cid in enumerate(spec['case_ids']):
                paths=[r/f'shard{idx%2}'/str(cid)/f'{method}_K{k}.json' for method in (right,left)]
                if not all(p.exists() for p in paths):continue
                old,new=[json.loads(p.read_text())['generations'] for p in paths]
                assert len(old)==len(new)
                for q,(before,after) in enumerate(zip(old,new)):
                    assert (before['kind'],before['prompt'])==(after['kind'],after['prompt'])
                    b,a=int(before['correct']),int(after['correct'])
                    query_rows.append({'case_id':cid,'K':k,'left':left,'right':right,'query_index':q,
                        'kind':before['kind'],'before':b,'after':a,'repair':int(not b and a),'break':int(b and not a)})
    for method in spec['methods']:
        for high,low in [(3,2),(4,2),(4,3)]:
            for idx,cid in enumerate(spec['case_ids']):
                paths=[r/f'shard{idx%2}'/str(cid)/f'{method}_K{k}.json' for k in (low,high)]
                if not all(p.exists() for p in paths):continue
                old,new=[json.loads(p.read_text())['generations'] for p in paths]
                assert len(old)==len(new)
                for q,(before,after) in enumerate(zip(old,new)):
                    assert (before['kind'],before['prompt'])==(after['kind'],after['prompt'])
                    b,a=int(before['correct']),int(after['correct'])
                    query_rows.append({'case_id':cid,'left_K':high,'right_K':low,'left':method,'right':method,
                        'query_index':q,'kind':before['kind'],'before':b,'after':a,
                        'repair':int(not b and a),'break':int(b and not a)})
    (out/'query_repair_break.json').write_text(json.dumps(query_rows,indent=2)+'\n')
    components=[]
    for method in spec['methods']:
        for k in (2,3,4):
            cells=[records.get((cid,method,k)) for cid in spec['case_ids']]
            for d,c in itertools.product((0,1),repeat=2):
                group=[x for x in cells if x is not None and x['E_QA_strict']==d and x['U_chain_all']==c]
                components.append({'method':method,'K':k,'direct_strict':d,'chain_all':c,
                    'families':len(group),'full_population_denominator':128,
                    'MH_contribution_over128':sum(x['A_MH'] for x in group)/128,
                    'joint_contribution_over128':sum(x['Joint'] for x in group)/128,
                    'complete':all(x is not None for x in cells),
                    'interpretation':'post-edit descriptive decomposition; not subgroup causal evidence'})
    (out/'joint_decomposition.json').write_text(json.dumps(components,indent=2)+'\n')
    done=json.loads((r/'DONE.json').read_text())
    text=['# EasyEdit ROME example without moment adjustment','',f"Execution: {done['status']}; completed cells {len(records)}/1536 (1152 frozen FT/M0 references;384 new ROME cells).",
          f"Actual allocation: {done['GPU_hours']:.6f} GPU-hours.",'',
          'FP32 native full-prefix. Same physical edit active every loop. Original128 only. Atomic-only training.',
          'ROME-EE-no-mom2 uses the unchanged pinned EasyEdit ROME algorithm with its llama-2 example configuration, explicitly without second-moment adjustment. K4 fitting, internal occurrence3 zero-based; final weight active all loops.',
          'Physical layer17 is inherited from prior screening, not newly tuned; full down-projection matrix, no rank constraint.',
          'This is an exploratory reused-screening comparison, not held-out confirmation. K3 is not used to tune anything.',
          'Primary metric: K4 A_MH, new ROME minus M0/FT-L/FT-M. Paired family bootstrap10000 seed17; other pair/depth contrasts descriptive, no uncorrected confirmatory significance claims.',
          'Missing cells stay explicit; full-population CIs are withheld on incomplete pairs. No failed case is removed.',
          'ROME/MEMIT/AlphaEdit remain not run pending faithful repeated-module adaptation. P1g/P2/P3 not run.',
          'Historical BF16 failures, qualified-pass limits, metadata deviations, early source snapshot gaps and human semantic review pending remain applicable to historical evidence.',
          'These results alone cannot establish a loop-specific editing failure, universal loop benefit or a first study.','',
          '| Method | K | Direct | Multi-hop | Joint | Locality raw |','|---|---:|---:|---:|---:|---:|']
    for method in spec['methods']:
        for k in [2,3,4]:
            vals=[]
            for metric in ['E_QA','A_MH','Joint','locality_raw_accuracy']:
                x=next(s for s in summaries if s['method']==method and s['k']==k and s['metric']==metric)
                vals.append('not complete' if x['available_family_count']!=128 else f"{100*x['available_mean']:.3f}%")
            text.append(f"| {method} | {k} | "+' | '.join(vals)+' |')
    (r/'AUTO_RESULTS_EASYEDIT_ROME.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'completed_cells':len(records),'expected_cells':1536,'raw_metric_audit_cells':len(audit),'paired_comparisons':len(comparisons)}))


if __name__=='__main__':main()

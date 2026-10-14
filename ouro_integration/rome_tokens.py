"""Fail-closed, CPU-only checks of the unchanged upstream ROME token objective."""
from copy import deepcopy


ATOMIC_FIELDS = {'case_id', 'family_id', 'prompt', 'subject', 'target_new'}


def prepare_atomic_request(request):
    if set(request) != ATOMIC_FIELDS:
        raise ValueError('Only atomic edit request fields are allowed')
    for field in ('prompt', 'subject', 'target_new'):
        if not isinstance(request[field], str) or not request[field]:
            raise ValueError('Empty or non-text atomic request field')
    if any(c in request['prompt']+request['subject']+request['target_new'] for c in '{}'):
        raise ValueError('Literal format braces are not supported')
    if request['prompt'].count(request['subject']) != 1:
        raise ValueError('Exactly one subject occurrence is required')
    result = deepcopy(request)
    result['prompt'] = result['prompt'].replace(result['subject'], '{}')
    # Match execute_rome exactly, including its single-space special case.
    if result['target_new'] != ' ':
        result['target_new'] = ' ' + result['target_new']
    return result


def audit_tokens(tok, request, context_templates, repr_tools):
    """Return evidence, not corrected inputs; any mismatch blocks the editor.

    Caller passes the original request to execute_rome, not the prepared request
    returned internally here (which already has upstream's leading space).
    Contexts must be the actual fixed context bank before a scientific fit.
    """
    req = prepare_atomic_request(request)
    if tok.padding_side != 'right' or not tok.is_fast:
        raise ValueError('Right padding and fast-tokenizer offset evidence required')
    if not context_templates or context_templates[0] != '{}':
        raise ValueError('The first upstream context must be the clean prompt')
    if any(t.count('{}') != 1 or '{' in t.replace('{}','') or '}' in t.replace('{}','')
           for t in context_templates):
        raise ValueError('Malformed context bank')
    target = tok.encode(req['target_new'], add_special_tokens=False)
    if not target:
        raise ValueError('Empty token target')
    prefix_text = tok.decode(target[:-1])
    templates = [t.format(req['prompt'])+prefix_text for t in context_templates] + ['{} is a']
    texts = [t.format(req['subject']) for t in templates]
    batch = tok(texts, padding=True)
    indices = repr_tools.get_words_idxs_in_templates(tok, templates, [req['subject']]*len(templates), 'last')
    rows = []
    for i, (template, text, upstream_idx) in enumerate(zip(templates, texts, indices)):
        encoded = tok(text, return_offsets_mapping=True)
        offsets, ids = encoded['offset_mapping'], encoded['input_ids']
        begin = len(template.split('{}')[0]); end = begin+len(req['subject'])
        overlap = [j for j,(a,b) in enumerate(offsets) if b > begin and a < end and b > a]
        if not overlap or upstream_idx != [overlap[-1]]:
            raise ValueError(f'Subject index mismatch for context {i}')
        mask = batch['attention_mask'][i]
        if mask != [1]*len(ids)+[0]*(len(mask)-len(ids)) or batch['input_ids'][i][:len(ids)] != ids:
            raise ValueError(f'Batch right-padding mismatch for context {i}')
        row = {'context_index': i, 'input_ids': ids, 'subject_last_index': overlap[-1],
               'subject_token_offsets': list(offsets[overlap[-1]]), 'kind': 'KL' if i==len(texts)-1 else 'rewrite'}
        if i < len(context_templates):
            clean = context_templates[i].format(req['prompt']).format(req['subject'])
            complete = tok.encode(clean+req['target_new'])
            if complete[-len(target):] != target or ids != complete[:-1]:
                raise ValueError(f'Teacher-forcing target boundary mismatch for context {i}')
            first = len(ids)-len(target)
            if first < 0:
                raise ValueError('Target has no preceding prediction position')
            row['prediction_positions'] = list(range(first, len(ids)))
            row['labels'] = target
        rows.append(row)
    return {'case_id': request['case_id'], 'family_id': request['family_id'],
            'target_ids': target, 'status': 'passed_token_contract_only', 'contexts': rows}

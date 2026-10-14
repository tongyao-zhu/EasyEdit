"""Frozen evaluation interface; this module is never imported by optimization.

Evaluation is answer-only greedy QA. The direct_locality path constructs only
held-out direct QA and locality items, and does not access multi-hop fields.
"""
from __future__ import annotations

import json
import pathlib
import time
import unicodedata

PARSER_VERSION = 'first_nonempty_literal_prefix_nfkc_casefold_unicode_punct_space_v1'
_ROOT = pathlib.Path(__file__).resolve().parents[1]


def normalize_answer(text):
    text = unicodedata.normalize('NFKC', str(text)).casefold()
    return ' '.join(''.join(' ' if unicodedata.category(c).startswith('P') else c for c in text).split())


def parse_answer(text):
    line = next((x.strip() for x in str(text).splitlines() if x.strip()), '')
    for prefix in ('Answer:', 'A:'):
        if line.startswith(prefix):
            line = line[len(prefix):].strip()
            break
    return normalize_answer(line)


def frozen_prompt_manifest():
    return json.loads((_ROOT / 'data' / 'prompt_manifest.json').read_text())


def qa_prompt(question, manifest=None):
    manifest = manifest or frozen_prompt_manifest()
    return manifest['qa_template'].format(question=question)


def _mean(values):
    return sum(values) / len(values) if values else None


def _reference_rows(base_reference):
    if base_reference is None:
        return {}
    if isinstance(base_reference, dict) and 'generations' in base_reference:
        return {r['query_id']:r for r in base_reference['generations'] if 'query_id' in r}
    if isinstance(base_reference, list):
        return {r['query_id']:r for r in base_reference if 'query_id' in r}
    if isinstance(base_reference, dict):
        return base_reference
    raise TypeError('base_reference must be a prior evaluate_case result, rows, or query_id mapping')


def _items(record, scope, manifest):
    if scope not in ('full', 'direct_locality'):
        raise ValueError('scope must be full or direct_locality')
    items = []
    def add(query_id, kind, question, expected, old=None, prompt=None, **extra):
        items.append({'query_id':query_id, 'kind':kind, 'question':question,
                      'prompt':prompt if prompt is not None else qa_prompt(question, manifest),
                      'aliases':list(expected), 'old_aliases':list(old or []), **extra})
    for i, q in enumerate(record['direct_questions']):
        add(f'direct:{i}', 'direct', q, record['direct_aliases'], record['old_direct_aliases'])
    # This branch is deliberately the only reader of downstream/MH fields.
    if scope == 'full':
        add('train_cloze', 'train_cloze', None, record['direct_aliases'],
            record['old_direct_aliases'], prompt=record['train_cloze'])
        for i, h in enumerate(record['unchanged_chain']):
            add(f'unchanged:{i}', 'unchanged_chain', h['question'], h['aliases'])
        for i, q in enumerate(record['multihop_questions']):
            add(f'multihop:{i}', 'multihop', q, record['new_answer_aliases'], record['old_answer_aliases'])
    for category, hops in record['locality'].items():
        for i, h in enumerate(hops):
            add(f'locality:{category}:{i}', 'locality', h['question'], h['aliases'],
                locality_category=category, triple_ids=h['triple_ids'])
    return items


def _generate(adapter, prompts, k, gates, manifest):
    kwargs = dict(k=k, max_new_tokens=manifest['generation']['max_new_tokens'],
                  use_cache=True, gates=gates, max_input_tokens=manifest['evaluation_max_input_tokens'],
                  stop_at_newline=True)
    if hasattr(adapter, 'batched_generate'):
        return adapter.batched_generate(prompts, batch_size=8, **kwargs)
    return adapter.greedy_generate(prompts, **kwargs)


def target_loglikelihoods(adapter, prompt, old_target, new_target, k, gates=None):
    """Score the full old/new objects at the atomic cloze, never first tokens only."""
    import torch
    try:
        from .ouro_adapter import encode_answers
    except ImportError:
        from ouro_adapter import encode_answers
    adapter.set_gates(gates)
    batch = encode_answers(adapter.tokenizer, [prompt, prompt], [old_target, new_target],
                           device=adapter.device, max_length=256)
    calls_before = adapter.total_recurrent_calls
    with torch.no_grad():
        out = adapter.forward(batch['input_ids'], batch['attention_mask'], k=k, use_cache=False)
        active = batch['answer_mask'][:, 1:]
        # Only answer positions are converted to full-vocabulary FP32 probabilities.
        selected = out.logits[:, :-1][active].float()
        selected_labels = batch['input_ids'][:, 1:][active]
        token_logprobs = selected.log_softmax(-1).gather(-1, selected_labels[:, None]).squeeze(-1)
        counts = active.sum(-1).tolist()
        offset, result = 0, {}
        for row, (name, target, count) in enumerate(zip(['old','new'], [old_target,new_target], counts)):
            values = token_logprobs[offset:offset + count].tolist()
            positions = active[row].nonzero(as_tuple=False).flatten() + 1
            result[name] = {'target':target, 'logprob_sum':sum(values), 'logprob_mean':sum(values)/count,
                           'target_token_count':count, 'target_token_logprobs':values,
                           'target_token_ids':batch['input_ids'][row,positions].tolist(),
                           'input_ids':batch['input_ids'][row].tolist(),
                           'answer_mask':batch['answer_mask'][row].tolist()}
            offset += count
    result['recurrent_block_calls_total'] = adapter.total_recurrent_calls - calls_before
    result['basis'] = 'atomic training cloze with complete concatenated old/new objects; no-cache fresh forwards'
    return result


def locality_kl(adapter, prompts, k, gates=None):
    """KL(base_K || edited_K) on all valid locality-prompt positions.

    Base teacher is recomputed at the same depth with edit application disabled;
    the fitted factors are never reset, overwritten or reoptimized here.
    """
    if not prompts:
        return []
    import torch
    adapter.set_gates(gates)
    tokenizer = adapter.tokenizer
    old_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    try:
        tokens = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors='pt')
    finally:
        tokenizer.padding_side = old_side
    ids, mask = tokens['input_ids'].to(adapter.device), tokens['attention_mask'].to(adapter.device)
    if ids.shape[1] > 2048:
        raise ValueError('locality KL input overflow; no truncation allowed')
    enabled = adapter.edit.enabled
    calls_before = adapter.total_recurrent_calls
    try:
        with torch.no_grad():
            # Hold logits in native precision, and upcast one row at a time to
            # bound memory while still computing the full-vocabulary KL.
            adapter.edit.enabled = False
            teacher = adapter.forward(ids, mask, k=k, use_cache=False).logits.detach()
            adapter.edit.enabled = enabled
            student = adapter.forward(ids, mask, k=k, use_cache=False).logits.detach()
            values = []
            for row in range(ids.shape[0]):
                valid = mask[row].bool()
                tlog = teacher[row,valid].float().log_softmax(-1)
                slog = student[row,valid].float().log_softmax(-1)
                kl = (tlog.exp() * (tlog - slog)).sum(-1)
                values.append({'mean_kl':float(kl.mean()), 'valid_positions':int(valid.sum()),
                               'teacher_depth':k, 'teacher':'unedited same-depth',
                               'full_vocabulary':True})
    finally:
        adapter.edit.enabled = enabled
    calls = adapter.total_recurrent_calls - calls_before
    for row in values:
        row['batch_recurrent_block_calls'] = calls
        row['recurrent_call_accounting'] = 'shared batch total, count once per KL batch'
    return values


def evaluate_case(adapter, eval_record, k, method, gates=None, scope='full', base_reference=None):
    """Return raw generations and per-case metrics; model failures propagate.

    Base-reference locality denominators are explicit. Missing references or
    categories yield None, never zero. This function performs no optimization.
    """
    started = time.perf_counter()
    calls_before = getattr(adapter, 'total_recurrent_calls', None)
    manifest = frozen_prompt_manifest()
    record = eval_record
    requested = _items(record, scope, manifest)
    raw = _generate(adapter, [x['prompt'] for x in requested], k, gates, manifest)
    if len(raw) != len(requested):
        raise RuntimeError('generation returned the wrong number of examples')
    reference = _reference_rows(base_reference)
    rows = []
    for generated, item in zip(raw, requested):
        if generated['prompt'] != item['prompt']:
            raise RuntimeError('generation/input ordering mismatch')
        parsed = parse_answer(generated['text'])
        row = {**generated, **item, 'case_id':record['case_id'], 'family_id':record['family_id'],
               'split':record['split'], 'method':method, 'depth':k, 'scope':scope,
               'parser_version':PARSER_VERSION, 'parsed_answer':parsed, 'parse_failure':not bool(parsed),
               'correct':parsed in {normalize_answer(x) for x in item['aliases']},
               'old_correct':parsed in {normalize_answer(x) for x in item['old_aliases']} if item['old_aliases'] else None,
               'status':'completed'}
        base = reference.get(item['query_id'])
        if base is None and method == 'M0':
            base = row
        if row['kind'] == 'locality':
            row['base_correct'] = bool(base['correct']) if base is not None else None
            row['base_parsed_answer'] = base['parsed_answer'] if base is not None else None
            row['output_agreement'] = parsed == base['parsed_answer'] if base is not None else None
        rows.append(row)
    local_rows = [r for r in rows if r['kind'] == 'locality']
    kl_rows = locality_kl(adapter, [r['prompt'] for r in local_rows], k, gates)
    for row, kl in zip(local_rows, kl_rows):
        row['locality_kl'] = kl
    direct = [r for r in rows if r['kind'] == 'direct']
    metrics = {'case_id':record['case_id'], 'family_id':record['family_id'], 'split':record['split'],
               'method':method, 'depth':k, 'scope':scope, 'status':'completed',
               'E_QA':_mean([r['correct'] for r in direct]),
               'E_QA_strict':int(all(r['correct'] for r in direct)),
               'direct_question_count':len(direct),
               'old_direct_accuracy':_mean([r['old_correct'] for r in direct]),
               'old_direct_official_correct':bool(direct[0]['old_correct']),
               'old_direct_strict':bool(all(r['old_correct'] for r in direct)),
               'old_answer_persistence_direct':_mean([r['old_correct'] for r in direct]),
               'parse_failure_rate':_mean([r['parse_failure'] for r in rows]),
               'truncation_rate':_mean([r['truncated'] for r in rows]),
               'mean_generated_tokens':_mean([len(r['generated_ids']) for r in rows]),
               'generation_count':len(rows),
               'locality_raw_accuracy':_mean([r['correct'] for r in local_rows]),
               'locality_qa_count':len(local_rows),
               'locality_output_agreement':_mean([r['output_agreement'] for r in local_rows if r['output_agreement'] is not None]),
               'locality_kl':_mean([x['mean_kl'] for x in kl_rows]),
               'locality_base_correct_count':None, 'locality_base_correct_accuracy':None,
               'locality_base_correct_accuracy_change':None,
               'locality_categories':{}}
    if local_rows and all(r['base_correct'] is not None for r in local_rows):
        base_correct = [r for r in local_rows if r['base_correct']]
        metrics['locality_base_correct_count'] = len(base_correct)
        metrics['locality_base_correct_accuracy'] = _mean([r['correct'] for r in base_correct])
        if base_correct:
            metrics['locality_base_correct_accuracy_change'] = metrics['locality_base_correct_accuracy'] - 1.0
    for category in record['locality']:
        selected = [r for r in local_rows if r['locality_category'] == category]
        metrics['locality_categories'][category] = {'count':len(selected), 'accuracy':_mean([r['correct'] for r in selected]),
            'output_agreement':_mean([r['output_agreement'] for r in selected if r['output_agreement'] is not None]),
            'availability':'available' if selected else 'unavailable_in_pinned_source'}
    if scope == 'full':
        train = next(r for r in rows if r['kind'] == 'train_cloze')
        chain = [r for r in rows if r['kind'] == 'unchanged_chain']
        mh = [r for r in rows if r['kind'] == 'multihop']
        if len(mh) != 3 or not chain:
            raise ValueError('full eligible evaluation requires three multi-hop questions and at least one unchanged fact')
        ll = target_loglikelihoods(adapter, record['train_cloze'], record['target_old'], record['target_new'], k, gates)
        train['target_loglikelihoods'] = ll
        metrics.update({'E_train':int(train['correct']),
            'U_chain':_mean([r['correct'] for r in chain]), 'U_chain_all':int(all(r['correct'] for r in chain)),
            'A_MH':_mean([r['correct'] for r in mh]), 'A_any3':int(any(r['correct'] for r in mh)),
            'old_multihop_mean_EM':_mean([r['old_correct'] for r in mh]),
            'old_multihop_any3':int(any(r['old_correct'] for r in mh)),
            'old_multihop_all3':int(all(r['old_correct'] for r in mh)),
            'old_answer_persistence_multihop':_mean([r['old_correct'] for r in mh]),
            'old_target_logprob_sum':ll['old']['logprob_sum'], 'old_target_logprob_mean':ll['old']['logprob_mean'],
            'new_target_logprob_sum':ll['new']['logprob_sum'], 'new_target_logprob_mean':ll['new']['logprob_mean'],
            'old_target_token_count':ll['old']['target_token_count'], 'new_target_token_count':ll['new']['target_token_count'],
            'target_loglikelihood_basis':ll['basis']})
        metrics['Joint'] = metrics['E_QA_strict'] * metrics['U_chain_all'] * metrics['A_MH']
        # These are valid strata flags only for the unedited model, frozen prior
        # to comparisons; edited methods do not receive a reselected stratum.
        metrics['preedit_known_at_depth'] = (metrics['old_direct_official_correct'] and bool(metrics['U_chain_all'])) if method == 'M0' else None
        metrics['preedit_old_correct_at_depth'] = (metrics['preedit_known_at_depth'] and bool(metrics['old_multihop_any3'])) if method == 'M0' else None
    else:
        metrics.update({'E_train':None,'U_chain':None,'U_chain_all':None,'A_MH':None,'A_any3':None,'Joint':None,
                        'downstream_evaluation_status':'not_run_selection_scope'})
    metrics['evaluation_wall_seconds'] = time.perf_counter() - started
    metrics['evaluation_actual_decoder_calls'] = (adapter.total_recurrent_calls - calls_before) if calls_before is not None else None
    metrics['evaluation_input_tokens_presented_for_generation'] = sum(len(r['input_ids']) for r in rows)
    metrics['evaluation_generated_tokens'] = sum(len(r['generated_ids']) for r in rows)
    metrics['evaluation_cost_includes'] = 'generation, same-depth locality KL teacher/student, and full-target old/new likelihoods when scope=full'
    return {'generations':rows, 'metrics':metrics}


def evaluate_context_controls(adapter, eval_record, k=4):
    """P0 capability diagnostics. Caller must provide the original unedited model.

    The controls live only here and must not enter any editing/selection loss.
    """
    if float(adapter.delta_norm().detach().cpu()) != 0:
        raise ValueError('context controls require an unedited zero-delta adapter')
    manifest = frozen_prompt_manifest()
    atomic_fact = eval_record['train_cloze'] + ' ' + eval_record['target_new'] + '.'
    requests = []
    for i, question in enumerate(eval_record['multihop_questions']):
        requests.append({'control':'exact_atomic_fact_in_prompt', 'query_id':f'context_atomic:{i}',
            'prompt':'Fact: ' + atomic_fact + '\n' + qa_prompt(question, manifest),
            'aliases':eval_record['new_answer_aliases']})
    for i, hop in enumerate(eval_record['oracle_bridge_questions']):
        requests.append({'control':'oracle_bridge_constituent', 'query_id':f'oracle_bridge:{i}',
            'prompt':qa_prompt(hop['question'], manifest), 'aliases':hop['aliases']})
    raw = _generate(adapter, [x['prompt'] for x in requests], k, None, manifest)
    rows = []
    for generated, request in zip(raw, requests):
        parsed = parse_answer(generated['text'])
        rows.append({**generated, **request, 'case_id':eval_record['case_id'], 'family_id':eval_record['family_id'],
                     'depth':k, 'method':'M0_context_control', 'parsed_answer':parsed,
                     'correct':parsed in {normalize_answer(x) for x in request['aliases']},
                     'parser_version':PARSER_VERSION, 'parse_failure':not bool(parsed), 'status':'completed'})
    return {'generations':rows, 'metrics':{name:_mean([r['correct'] for r in rows if r['control']==name]) for name in ['exact_atomic_fact_in_prompt','oracle_bridge_constituent']}}


def knowledge_strata_from_baselines(at_depth_2, at_depth_4):
    """Pure pre-edit flags; pass baseline metrics (or None for an unrun depth)."""
    for metrics in (at_depth_2, at_depth_4):
        if metrics is not None and (metrics['method'] != 'M0' or metrics['scope'] != 'full'):
            raise ValueError('knowledge strata require unedited full-scope evaluation')
    known4 = at_depth_4['preedit_known_at_depth'] if at_depth_4 is not None else None
    known2 = at_depth_2['preedit_known_at_depth'] if at_depth_2 is not None else None
    return {'S_known4':known4,
            'S_old_correct4':at_depth_4['preedit_old_correct_at_depth'] if at_depth_4 is not None else None,
            'S_known24':bool(known2 and known4) if known2 is not None and known4 is not None else None}

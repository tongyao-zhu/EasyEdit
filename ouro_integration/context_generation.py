"""Upstream ROME context sampling with native full-prefix evaluation.

Preserve generate_fast's top-k sampling order, total-length stopping, prompt
replication and text cleanup. Only KV execution is replaced; numerical equality
between real cached/uncached Ouro is NOT claimed. No early EOS stopping is added.
"""
import unicodedata

import torch


def generate_full_prefix(model, tok, prompts, n_gen_per_prompt=1, top_k=5,
                         max_out_len=200, vanilla_generation=False):
    if vanilla_generation:
        raise ValueError('Only the upstream ROME top-k context path is supported')
    if tok.padding_side != 'right' or tok.pad_token_id is None:
        raise ValueError('Explicit right padding and pad token are required')
    if not prompts or type(n_gen_per_prompt) is not int or n_gen_per_prompt < 1:
        raise ValueError('Nonempty prompts and positive replication required')
    if type(top_k) is not int or top_k < 1 or type(max_out_len) is not int or max_out_len < 1:
        raise ValueError('Positive integer top-k and total length required')
    inp = [prompt for prompt in prompts for _ in range(n_gen_per_prompt)]
    batch = tok(inp, padding=True, return_tensors='pt').to(next(model.parameters()).device)
    input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
    lengths = attention_mask.sum(1)
    if (lengths < 1).any():
        raise ValueError('Empty token prefix')
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :]
    if not torch.equal(attention_mask.bool(), positions < lengths[:, None]):
        raise ValueError('Attention mask is not contiguous right padding')
    batch_size, cursor = input_ids.size(0), int(lengths.min())
    with torch.no_grad():
        # Intentionally preserve upstream width-based stopping, including the
        # case where an initial long prompt already meets max_out_len.
        while input_ids.size(1) < max_out_len:
            if not attention_mask[:, :cursor].all():
                raise RuntimeError('Generation exposed an unfilled prefix')
            out = model(input_ids=input_ids[:, :cursor],
                        attention_mask=attention_mask[:, :cursor], use_cache=False)
            logits = out if type(out) is torch.Tensor else out.logits
            probs = torch.nn.functional.softmax(logits[:, -1, :], dim=1)
            tk = torch.topk(probs, top_k, dim=1).indices
            weights = torch.gather(probs, 1, tk)
            weights = weights / weights.sum(1)[:, None]
            new_toks = torch.gather(tk, 1, torch.multinomial(weights, 1))
            if cursor == input_ids.size(1):
                attention_mask = torch.cat([attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1)
                input_ids = torch.cat([input_ids, input_ids.new_full((batch_size, 1), tok.pad_token_id)], dim=1)
            last_non_masked = attention_mask.sum(1) - 1
            for i in range(batch_size):
                new_idx = last_non_masked[i] + 1
                if last_non_masked[i].item() + 1 != cursor:
                    continue
                if new_idx < max_out_len:
                    input_ids[i][new_idx] = new_toks[i]
                    attention_mask[i][new_idx] = 1
            cursor += 1
    texts = [tok.decode(x, skip_special_tokens=True) for x in input_ids.detach().cpu().numpy().tolist()]
    return [unicodedata.normalize('NFKD', x).replace('\n\n', ' ').replace('<|endoftext|>', '') for x in texts]

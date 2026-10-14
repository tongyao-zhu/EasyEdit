"""Exact historical answer tokenization helper; no editing adapter imported."""
import torch

def encode_answers(tokenizer, prompts, answers, device=None, max_length=256):
    """Left-pad concatenated strings and label every token overlapping answer.

    Whitespace preceding the answer belongs to the answer span. Offset mapping
    handles BPE boundary merges; a merged boundary token is included in full.
    No separately-tokenized answer is appended to a differently tokenized prompt.
    """
    if len(prompts) != len(answers) or not prompts:
        raise ValueError("prompts and answers must be nonempty and aligned")
    examples, offsets_all, texts, boundaries = [], [], [], []
    for prompt, answer in zip(prompts, answers):
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("empty target")
        separator = "" if prompt.endswith(tuple(" \n\t")) or answer.startswith(tuple(" \n\t")) else " "
        text = prompt + separator + answer
        boundary = len(prompt)
        tokenized = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = tokenized["input_ids"], tokenized["offset_mapping"]
        if len(ids) > max_length:
            raise ValueError(f"editing input overflow: {len(ids)} > {max_length}")
        mask = [int(end > boundary and end > start) for start, end in offsets]
        if not any(mask) or mask[0]:
            raise ValueError("target must follow at least one context token")
        examples.append((ids, mask))
        offsets_all.append(offsets)
        texts.append(text)
        boundaries.append(boundary)
    width = max(len(ids) for ids, _ in examples)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    ids_out, attn_out, answer_out = [], [], []
    for ids, mask in examples:
        n = width - len(ids)
        ids_out.append([pad] * n + ids)
        attn_out.append([0] * n + [1] * len(ids))
        answer_out.append([0] * n + mask)
    return {
        "input_ids": torch.tensor(ids_out, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attn_out, dtype=torch.long, device=device),
        "answer_mask": torch.tensor(answer_out, dtype=torch.bool, device=device),
        "texts": texts, "answer_boundaries": boundaries, "offset_mapping": offsets_all,
    }

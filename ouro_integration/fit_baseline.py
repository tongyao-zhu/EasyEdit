"""Atomic-only interface to unmodified EasyEdit FT; no evaluation import."""
import torch
from .adapter import FixedDepth
from .upstream import load_ft


def fit(model, tok, request, configuration, *, depth=4):
    if type(depth) is not int or depth not in (2, 4):
        raise ValueError('Only prespecified fit depths K2 and K4 are supported')
    allowed = {'case_id', 'family_id', 'prompt', 'subject', 'target_new'}
    if set(request) != allowed:
        raise ValueError('Unexpected request fields: evaluation is not supervision')
    ft = load_ft()
    hp = ft.FTHyperParams(**configuration)
    flags = {n:p.requires_grad for n,p in model.named_parameters()}
    old_side = tok.padding_side
    tok.padding_side = 'right'
    try:
        torch.manual_seed(17)
        with FixedDepth(model, depth):
            edited, backup = ft.apply_ft_to_model(model, tok, [request], hp, return_orig_weights=True)
    finally:
        tok.padding_side = old_side
        for n,p in model.named_parameters(): p.requires_grad_(flags[n])
    return edited, backup

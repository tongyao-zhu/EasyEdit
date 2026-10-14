"""Native full-prefix generation behind the unchanged historical evaluator.

Evaluation-only weight swapping supplies same-depth M0 locality teachers. It
never gates individual recurrent invocations and is never used in fitting.
"""
import time
import torch
from .adapter import FixedDepth


class WeightState:
    def __init__(self, model, originals):
        self.model, self.originals = model, originals
        self.edited = None
        self._enabled = True

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        value = bool(value)
        if value == self._enabled:
            return
        params = dict(self.model.named_parameters())
        if not value:
            self.edited = {n: params[n].detach().clone() for n in self.originals}
            source = self.originals
        else:
            source = self.edited
        with torch.no_grad():
            for n, w in source.items():
                params[n].copy_(w)
        self._enabled = value
        if value:
            self.edited = None


class EvaluationBridge:
    def __init__(self, model, tokenizer, originals):
        self.model, self.tokenizer = model, tokenizer
        self.device = next(model.parameters()).device
        self.edit = WeightState(model, originals)
        self.total_recurrent_calls = 0
        self.gates = None
        self.token_positions_processed = 0

    def set_gates(self, gates):
        if gates is not None:
            raise ValueError('No per-loop edit gates in native deployment')

    def forward(self, input_ids, attention_mask, k=4, use_cache=False, logits_to_keep=0):
        if use_cache:
            raise ValueError('Full-prefix backend cannot accept KV cache')
        positions = attention_mask.long().cumsum(-1) - 1
        positions.masked_fill_(attention_mask == 0, 0)
        with FixedDepth(self.model, k):
            out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             position_ids=positions, use_cache=False,
                             logits_to_keep=logits_to_keep)
        self.total_recurrent_calls += k * len(self.model.model.layers)
        self.token_positions_processed += input_ids.numel() * k
        return out

    def batched_generate(self, prompts, k=4, max_new_tokens=24, use_cache=True,
                         gates=None, max_input_tokens=2048, stop_at_newline=True,
                         batch_size=8):
        # Legacy evaluator asks for caching. Backend choice is explicitly frozen
        # to no-cache; raw rows record actual execution, not the legacy hint.
        self.set_gates(gates)
        tok, rows = self.tokenizer, []
        pad = tok.pad_token_id
        for start in range(0, len(prompts), batch_size):
            part = prompts[start:start+batch_size]
            initial = [tok(p, add_special_tokens=False)['input_ids'] for p in part]
            if any(not x or len(x) > max_input_tokens for x in initial):
                raise ValueError('Input empty or overflow; never truncate')
            width = max(map(len, initial))
            ids = torch.tensor([[pad]*(width-len(x))+x for x in initial], device=self.device)
            mask = torch.tensor([[0]*(width-len(x))+[1]*len(x) for x in initial], device=self.device)
            generated = [[] for _ in part]; done = [False]*len(part)
            reasons = ['max_new_tokens']*len(part)
            calls = self.total_recurrent_calls
            token_work = self.token_positions_processed
            if self.device.type == 'cuda': torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    pred = self.forward(ids, mask, k=k, logits_to_keep=1).logits[:, -1].argmax(-1).tolist()
                    append_ids, append_mask = [], []
                    for j, token in enumerate(pred):
                        was_done = done[j]
                        if not was_done:
                            generated[j].append(token)
                            if token == tok.eos_token_id:
                                reasons[j], done[j] = 'eos', True
                            elif stop_at_newline and '\n' in tok.decode(generated[j], skip_special_tokens=True):
                                reasons[j], done[j] = 'answer_boundary', True
                        append_ids.append(pad if was_done else token)
                        append_mask.append(0 if was_done else 1)
                    ids = torch.cat((ids, torch.tensor(append_ids, device=self.device)[:, None]), dim=1)
                    mask = torch.cat((mask, torch.tensor(append_mask, device=self.device)[:, None]), dim=1)
                    if all(done): break
            if self.device.type == 'cuda': torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            calls = self.total_recurrent_calls-calls
            token_work = self.token_positions_processed-token_work
            for j, prompt in enumerate(part):
                rows.append({'prompt':prompt, 'input_ids':initial[j], 'generated_ids':generated[j],
                    'text':tok.decode(generated[j], skip_special_tokens=True),
                    'latency_seconds':elapsed/len(part), 'batch_wall_seconds':elapsed,
                    'latency_measurement':'batch_wall_amortized_per_row',
                    'batch_size':len(part), 'batch_index':start//batch_size,
                    'truncated':reasons[j]=='max_new_tokens', 'stop_reason':reasons[j], 'k':k,
                    'gates':[1.]*k, 'recurrent_block_calls':calls/len(part),
                    'batch_recurrent_block_calls':calls, 'batch_loop_token_positions':token_work,
                    'cache':False, 'backend':'native_fp32_full_prefix'})
        return rows

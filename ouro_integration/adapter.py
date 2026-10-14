"""Keep native Ouro computation and parameter sharing; never gate an edit.

Cache acceleration is deliberately unvalidated here. FixedDepth disables it.
Ouro's labels path mixes exit logits, so callers must compute their objective
from returned final-depth logits rather than silently use that different loss.
"""
from contextlib import AbstractContextManager


class FixedDepth(AbstractContextManager):
    def __init__(self, model, depth):
        if type(depth) is not int or depth not in (2, 3, 4):
            raise ValueError("Validated depth candidates are 2, 3, 4")
        if getattr(model.config, "model_type", None) != "ouro":
            raise ValueError("Expected Ouro, not a name-based architecture guess")
        self.model, self.depth = model, depth
        self.saved = []
        self.handle = None

    def __enter__(self):
        if self.saved:
            raise RuntimeError("Context already entered")
        if getattr(self.model.model, "gradient_checkpointing", False):
            raise ValueError("Checkpoint recomputation semantics not yet validated")
        self.parameter_ids = {n: id(p) for n, p in self.model.named_parameters()}
        for obj, attr, value in [
            (self.model.config, "total_ut_steps", self.depth),
            (self.model.model, "total_ut_steps", self.depth),
            (self.model, "early_exit_step", self.depth - 1),
            (self.model.config, "use_cache", False),
        ]:
            self.saved.append((obj, attr, getattr(obj, attr)))
            setattr(obj, attr, value)

        def check_inputs(module, args, kwargs):
            # Native positional slots: ids, mask, positions, cache, embeds, labels.
            if len(args) > 5 and args[5] is not None or kwargs.get("labels") is not None:
                raise ValueError("Ouro labels use a weighted-exit loss; compute loss from logits")
            if len(args) > 3 and args[3] is not None or kwargs.get("past_key_values") is not None:
                raise ValueError("KV caching is not validated in this integration")
            if len(args) > 6 and args[6] or kwargs.get("use_cache", False):
                raise ValueError("KV caching is not validated in this integration")
            if kwargs.get("use_weighted_exit", False):
                raise ValueError("Weighted exit differs from fixed-depth inference")
            if kwargs.get("exit_at_step", self.depth - 1) not in (None, self.depth - 1):
                raise ValueError("Conflicting exit depth")

        self.handle = self.model.register_forward_pre_hook(check_inputs, with_kwargs=True)
        return self.model

    def __exit__(self, exc_type, exc, tb):
        self.handle.remove()
        for obj, attr, value in reversed(self.saved):
            setattr(obj, attr, value)
        self.saved.clear()
        if {n: id(p) for n, p in self.model.named_parameters()} != self.parameter_ids:
            raise RuntimeError("Integration changed parameter identities or sharing")
        return False


class SharedCallTrace(AbstractContextManager):
    """Observe every invocation of one physical MLP without changing outputs.

    Records actual current_ut supplied by native Ouro, not guessed hook counts.
    Optional output-gradient norms establish that all calls reach one parameter.
    This is a diagnostic observer, not an editing rule for ROME/MEMIT.
    """
    def __init__(self, model, layer, gradients=False):
        self.model, self.layer, self.gradients = model, layer, gradients
        self.records, self.handles = [], []
        self.current = None

    def __enter__(self):
        decoder = self.model.model.layers[self.layer]
        projection = decoder.mlp.down_proj

        def before(module, args, kwargs):
            if "current_ut" not in kwargs:
                raise RuntimeError("Missing native current_ut")
            self.current = int(kwargs["current_ut"])

        def observe(module, args, output):
            if self.current is None:
                raise RuntimeError("Projection outside an observed decoder call")
            record = {"loop": self.current + 1, "parameter_id": id(module.weight),
                      "shape": list(output.shape)}
            self.records.append(record)
            if self.gradients and output.requires_grad:
                output.register_hook(lambda grad, r=record: r.update(
                    gradient_norm=float(grad.detach().float().norm())) or None)

        self.handles = [decoder.register_forward_pre_hook(before, with_kwargs=True),
                        projection.register_forward_hook(observe)]
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.current = None
        return False

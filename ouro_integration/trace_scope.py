"""Select a native Ouro invocation for upstream editor instrumentation.

This is a location convention during key/value/statistic estimation, NOT a
deployment rule. Parameters and the native forward are never replaced. A stored
edit remains in the same physical weight and is used on every recurrent visit.
The selected zero-based occurrence must be fixed before examining edit outcomes.
"""
from contextlib import AbstractContextManager


class OuroTraceScope(AbstractContextManager):
    def __init__(self, model, nethook, occurrence, depth):
        if type(depth) is not int or depth not in (2, 3, 4):
            raise ValueError('Depth must be an explicitly supported fixed depth')
        if type(occurrence) is not int or not 0 <= occurrence < depth:
            raise ValueError('Occurrence is zero based and must be below depth')
        if getattr(model.config, 'model_type', None) != 'ouro':
            raise ValueError('Native Ouro current_ut is required')
        self.model, self.nethook = model, nethook
        self.occurrence, self.depth = occurrence, depth
        self.handles, self.current, self.events = [], {}, []
        self.active = False
        self.policy = None

    def __enter__(self):
        if self.handles or self.policy is not None:
            raise RuntimeError('Scope already active')
        if getattr(self.model.model, 'gradient_checkpointing', False):
            raise ValueError('Checkpoint recomputation is not validated')
        if self.model.model.total_ut_steps != self.depth or self.model.early_exit_step != self.depth-1:
            raise ValueError('Enter FixedDepth before OuroTraceScope')
        self.parameter_ids = {n:id(p) for n,p in self.model.named_parameters()}
        self.owner = {}
        for index, decoder in enumerate(self.model.model.layers):
            for module in decoder.modules():
                if id(module) in self.owner and self.owner[id(module)] != index:
                    raise ValueError('Cross-physical-layer module aliasing is unsupported')
                self.owner[id(module)] = index

        def start(module, args, kwargs):
            if self.active:
                raise RuntimeError('Nested model forwards are unsupported')
            self.current.clear()
            self.active = True

        def finish(module, args, kwargs, output):
            self.active = False
            self.current.clear()

        try:
            self.handles.append(self.model.register_forward_pre_hook(start, with_kwargs=True))
            self.handles.append(self.model.register_forward_hook(finish, with_kwargs=True, always_call=True))
            for index, decoder in enumerate(self.model.model.layers):
                def before(module, args, kwargs, index=index):
                    if not self.active or 'current_ut' not in kwargs:
                        raise RuntimeError('Missing native current_ut / root forward scope')
                    ut = kwargs['current_ut']
                    if type(ut) is not int or not 0 <= ut < self.depth:
                        raise RuntimeError('Invalid native current_ut')
                    self.current[index] = ut
                self.handles.append(decoder.register_forward_pre_hook(before, with_kwargs=True))
            self.policy = self.nethook.use_trace_invocation_filter(self.select)
            self.policy.__enter__()
        except BaseException:
            for h in self.handles: h.remove()
            self.handles.clear()
            self.policy = None
            raise
        return self

    def select(self, module, layer_name):
        index = self.owner.get(id(module))
        if not self.active or index is None or index not in self.current:
            raise RuntimeError('Trace outside supported native decoder invocation: '+str(layer_name))
        ut = self.current[index]
        selected = ut == self.occurrence
        self.events.append({'layer':layer_name,'physical_layer':index,'current_ut':ut,'selected':selected})
        return selected

    @property
    def cache_namespace(self):
        # Callers must also bind model, corpus, precision, tokenizer and source.
        return f'ouro_native_fullprefix_K{self.depth}_occ{self.occurrence}'

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.policy is not None:
                self.policy.__exit__(exc_type, exc, tb)
        finally:
            self.policy = None
            for h in self.handles: h.remove()
            self.handles.clear()
            self.active = False
            self.current.clear()
        if {n:id(p) for n,p in self.model.named_parameters()} != self.parameter_ids:
            raise RuntimeError('Trace scope changed parameter identities')
        return False

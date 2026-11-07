"""Thread/task-local stage metadata without modifying model messages."""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar


def canonical_stage(name):
    if name in ('layout', 'identify', 'describe', 'recognition'):
        return 'recognition'
    if name.startswith('select') or name == 'selection':
        return 'selection'
    if name.startswith('portion_') or name == 'portion':
        return 'portion'
    if name in ('queries', 'multiquery', 'query_rewrite'):
        return 'query_rewrite'
    raise ValueError('Unknown pipeline stage: ' + name)


class StageChat:
    def __init__(self, provider, dataset, max_completion_tokens=4096):
        if dataset not in ('nutrition5k', 'asa24'):
            raise ValueError('Unknown stage dataset')
        self.provider, self.dataset = provider, dataset
        self.max_completion_tokens = max_completion_tokens
        self._current = ContextVar('diet_stage', default='query_rewrite')

    @contextmanager
    def stage(self, name):
        token = self._current.set(canonical_stage(name))
        try:
            yield
        finally:
            self._current.reset(token)

    def __call__(self, messages):
        return self.provider.chat(messages, max_completion_tokens=self.max_completion_tokens,
            dataset=self.dataset, stage=self._current.get())


class StageRoutingMixin:
    def _stage(self, name, *args, **kwargs):
        chat = self.chat
        context = chat.stage(name) if isinstance(chat, StageChat) else nullcontext()
        with context:
            return super()._stage(name, *args, **kwargs)

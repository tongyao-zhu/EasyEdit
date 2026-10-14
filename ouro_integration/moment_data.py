"""Pinned public text access for the unchanged upstream second-moment collector.

Preserves full-dataset indexing so upstream FixedRandomSubsetSampler picks its
original rows. No replacement of the estimator, tokenizer, collation or inverse.
"""
from contextlib import contextmanager
import gzip
import hashlib
import json
from pathlib import Path
from threading import Lock

_lock = Lock()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class PinnedTextRows:
    """Sparse, fail-closed view of the original full corpus index space."""
    def __init__(self, total_rows, row_order, records):
        if type(total_rows) is not int or total_rows <= 0:
            raise ValueError('Invalid full corpus size')
        if len(row_order) != len(records) or len(set(row_order)) != len(row_order):
            raise ValueError('Duplicate or incomplete selected rows')
        self.total_rows = total_rows
        self.rows = {}
        for sample_index, (row_index, record) in enumerate(zip(row_order, records)):
            if type(row_index) is not int or not 0 <= row_index < total_rows:
                raise ValueError('Invalid source row index')
            if record['sample_index'] != sample_index or record['source_row_index'] != row_index:
                raise ValueError('Selection order mismatch')
            if hashlib.sha256(record['text'].encode()).hexdigest() != record['text_sha256']:
                raise ValueError('Text hash mismatch')
            self.rows[row_index] = {'text': record['text']}

    def __len__(self):
        return self.total_rows

    def __getitem__(self, index):
        # Unknown rows fail rather than silently changing the sampling population.
        if index not in self.rows:
            raise KeyError('Row outside the frozen corpus sample: ' + str(index))
        return dict(self.rows[index])


def load_verified_corpus(root, tokenization, binding_sha256):
    root, tokenization = Path(root), Path(tokenization)
    binding_path = tokenization / 'TOKEN_BINDING.json'
    if file_sha256(binding_path) != binding_sha256:
        raise ValueError('Tokenization binding mismatch')
    binding = json.loads(binding_path.read_text())
    status = json.loads((tokenization / 'STATUS.json').read_text())
    if status['status'] != 'passed_CPU_corpus_and_token_contract_only' or status['binding_sha256'] != binding_sha256:
        raise ValueError('No successful corpus/token verification')
    sample_path, source_path = root / 'SAMPLE_MANIFEST.json', root / 'SOURCE_MANIFEST.json'
    for path, expected in [(sample_path, binding['sample_manifest_sha256']),
                           (source_path, binding['source_manifest_sha256']),
                           (tokenization/'TOKEN_ROWS.json', binding['token_rows_sha256']),
                           (tokenization/'tokens.int64.bin', binding['tokens_sha256'])]:
        if file_sha256(path) != expected:
            raise ValueError('Bound corpus/token file changed: ' + str(path))
    sample = json.loads(sample_path.read_text())
    selected = root / 'SELECTED_TEXTS.jsonl.gz'
    if file_sha256(selected) != sample['selected_file_sha256']:
        raise ValueError('Selected text file changed')
    with gzip.open(selected, 'rt') as stream:
        records = [json.loads(line) for line in stream]
    if len(records) != binding['samples']:
        raise ValueError('Sample count mismatch')
    return PinnedTextRows(sample['total_corpus_rows'], sample['row_indices'], records), binding


@contextmanager
def pinned_wikipedia_access(layer_stats_module, rows):
    """Only bind data access. Caller separately binds K/occurrence/runtime/cache."""
    if not isinstance(rows, PinnedTextRows):
        raise TypeError('Expected validated sparse corpus rows')
    if not _lock.acquire(blocking=False):
        raise RuntimeError('Pinned dataset access is process-exclusive')
    original = layer_stats_module.load_dataset
    def pinned(name, config, *args, **kwargs):
        if (name, config) != ('wikimedia/wikipedia', '20231101.en') or args or kwargs:
            raise ValueError('Unbound dataset request')
        return {'train': rows}
    layer_stats_module.load_dataset = pinned
    try:
        yield
    finally:
        layer_stats_module.load_dataset = original
        _lock.release()

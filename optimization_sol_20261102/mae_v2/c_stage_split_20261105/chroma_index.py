"""Local-only original Chroma index copy; never constructs an embedding client."""
from pathlib import Path
import hashlib
import json
import math
import re
import shutil
import threading

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
REPO = V2.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hashes(directory):
    return {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}


def prepare_copy():
    target = HERE / 'prepared/chroma'
    marker = HERE / 'prepared/chroma_source.json'
    if marker.exists():
        value = json.loads(marker.read_text())
        if tree_hashes(REPO / 'index/chroma') != value['source_files']:
            raise ValueError('original_Chroma_index_changed')
        if not target.is_dir():
            raise ValueError('prepared_Chroma_missing')
        return value
    source = REPO / 'index/chroma'
    before = tree_hashes(source)
    if not before or target.exists():
        raise ValueError('index_copy_requires_complete_source_and_new_target')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    if tree_hashes(source) != before or tree_hashes(target) != before:
        raise ValueError('index_copy_not_identical')
    value = {'source': 'index/chroma', 'source_files': before,
             'method': 'original_Chromadb_0.4.24_L2_k4', 'api_calls': 0}
    marker.write_text(json.dumps(value, indent=2) + '\n')
    return value


class SearchIndex:
    def __init__(self, verify_vectors=True):
        import chromadb
        import numpy as np
        self.lock = threading.Lock()
        self.client = chromadb.PersistentClient(path=str(HERE / 'prepared/chroma'),
            settings=chromadb.Settings(anonymized_telemetry=False))
        self.collection = self.client.get_collection('openai_embed', embedding_function=None)
        descriptor = json.loads((V2 / 'data/vectors/collection.json').read_text())
        vectors = np.load(V2 / 'data/vectors/embeddings.npy', allow_pickle=False, mmap_mode='r')
        codes = json.loads((V2 / 'data/vectors/codes.json').read_text())
        if self.collection.count() != 5624 or descriptor['count'] != 5624 or vectors.shape != (5624, 3072):
            raise ValueError('full_FNDDS_index_required')
        if (self.collection.metadata or {}).get('hnsw:space', 'l2') != 'l2':
            raise ValueError('original_L2_index_required')
        self.by_id = {}
        for start in range(0, 5624, 256):
            ids = descriptor['ids'][start:start+256]
            include = ['documents', 'metadatas'] + (['embeddings'] if verify_vectors else [])
            got = self.collection.get(ids=ids, include=include)
            positions = {identifier: n for n, identifier in enumerate(got['ids'])}
            if set(positions) != set(ids):
                raise ValueError('index_ID_set_changed')
            for offset, identifier in enumerate(ids):
                source_i, actual_i = start + offset, positions[identifier]
                if (got['documents'][actual_i] != descriptor['documents'][source_i]
                        or got['metadatas'][actual_i] != descriptor['metadatas'][source_i]):
                    raise ValueError('index_document_or_metadata_changed')
                match = re.match(r'Food code: (\d{8})(?!\d)', got['documents'][actual_i])
                if match is None or match.group(1) != codes[source_i]:
                    raise ValueError('index_code_order_changed')
                if verify_vectors and not np.array_equal(np.asarray(got['embeddings'][actual_i], dtype=np.float32), vectors[source_i]):
                    raise ValueError('index_embedding_changed')
                self.by_id[identifier] = match.group(1)
        self.dimension = 3072

    def search4(self, vector):
        if len(vector) != self.dimension or any(not math.isfinite(v) for v in vector):
            raise ValueError('invalid_query_embedding')
        # Serializes only local HNSW lookups, never network requests.
        with self.lock:
            result = self.collection.query(query_embeddings=[list(vector)], n_results=4,
                                           include=['distances'])
        codes = [self.by_id[identifier] for identifier in result['ids'][0]]
        if len(codes) != 4 or len(set(codes)) != 4:
            raise ValueError('invalid_original_top4')
        return codes

"""No-key, no-model verification of a fresh mixed-C checkout."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import socket
import sys

import run_mixed_c as run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', action='store_true', help='Also validate original runtime and load its copied Chroma index')
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 10):
        raise ValueError('Use the project Python 3.10 environment')
    def blocked(*a, **kw):
        raise AssertionError('Offline check attempted a network connection')
    socket.create_connection = blocked
    socket.socket.connect = blocked
    sources = run.source_hashes()
    m = run.modules()
    assert m['config'].STAGE_MODELS == {
        'recognition': 'gemini-3.8-flash', 'selection': 'gemini-3.8-flash',
        'portion': {'nutrition5k': 'gemini-3.8-flash', 'asa24': 'gpt-6.1-sol'}}
    versions = {name: importlib.metadata.version(name) for name in ('chromadb', 'numpy', 'Pillow', 'httpx')}
    assert versions['chromadb'] == '0.4.24' and versions['numpy'] == '1.26.4'
    result = {'status': 'passed', 'source_files': len(sources), 'models': m['config'].MODELS,
              'versions': versions, 'runtime_checked': False, 'new_api_calls': 0}
    if args.runtime:
        import runtime
        result['runtime'] = runtime.verify()
        from chromadb.telemetry.product.posthog import Posthog
        Posthog.capture = lambda *a, **k: None
        with run.original_index(m) as index:
            result.update(runtime_checked=True, vectors=index.collection.count(), dimensions=index.dimension)
            catalog = run.read(run.V2 / 'round11/prepared/catalog.json')
            assert len(catalog['foods']) == 5624 and len(catalog['asa_options']) == 555
            # Exercise actual production wiring without sending a request.
            n5 = m['pipeline'].make_component_pipeline(blocked, blocked, catalog, index.search4)
            asa = m['pipeline'].make_asa_pipeline(blocked, blocked, catalog, index.search4)
            assert n5.foods == asa.foods == catalog['foods']
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()

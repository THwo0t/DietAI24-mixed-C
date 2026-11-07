"""Portable wrapper around the unchanged original mixed-C inference factories.

Images only by default. ASA reference codes are opened only after ALL identities
are sealed; weight/nutrient labels are accepted by tools/score.py, never here.
Historical label-median fallbacks and calibration are deliberately not invoked.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / 'optimization_sol_20261102/mae_v2'
METHOD = V2 / 'c_stage_split_20261105'
N5_CONFIG = dict(crop_review=False, portion_mode='total_ratio', profile='C', identity_profile='A')
ASA_CONFIG = dict(variant='optimized', image_mode='square256', candidate_cap=32,
                  portion_mode='continuous_option', examples=False)
TARGETS = ('weight', 'energy', 'fat', 'carbs', 'protein')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode(raw):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError('Duplicate JSON key')
            result[k] = v
        return result
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON value')))


def read(path):
    return decode(Path(path).read_bytes())


def source_hashes(allow_modified=False):
    manifest = read(ROOT / 'provenance/source_manifest.json')
    expected = {**manifest['frozen_python_files'], **manifest['extra_original_files']}
    actual = {name: file_sha(ROOT / name) for name in expected}
    if not allow_modified and actual != expected:
        raise ValueError('Original source changed; use a new branch and explicit --allow-modified-source')
    actual['tools/run_mixed_c.py'] = file_sha(__file__)
    return actual


def modules(run_dir=None):
    sys.path[:0] = [str(METHOD), str(V2 / 'src'), str(V2)]
    result = {name: importlib.import_module(name) for name in
              ('config', 'stage_routes', 'provider', 'newcommon', 'pipeline', 'chroma_index')}
    # Storage relocation only. No prompt, parser, model, search or nutrition changes.
    result['provider'].ROOT = ROOT
    result['newcommon'].V2 = ROOT
    if run_dir is not None:
        result['newcommon'].HERE = run_dir
    return result


def input_rows(images=None, manifest=None, dataset=None, round_number=None):
    if round_number is not None and manifest is None:
        raise ValueError('--round requires the fixed development manifest')
    if manifest:
        raw = read(manifest)
        if round_number is not None and (not isinstance(raw, dict)
                or raw.get('cohort') != 'development' or 'images' not in raw or 'samples' not in raw):
            raise ValueError('--round requires the fixed development manifest')
        if isinstance(raw, dict) and 'images' in raw:
            selected = None
            if round_number is not None:
                selected = {r['image_id'] for r in raw.get('samples', []) if r.get('round') == round_number}
                if not selected:
                    raise ValueError('This manifest has no requested round')
            raw = [{'sample_id': x['image_id'], 'image_path': x['path'],
                    'image_sha256': x['sha256']} for x in raw['images']
                    if (selected is None or x['image_id'] in selected)
                    and (dataset is None or (x['dataset'] == 'asa24') == (dataset == 'asa24'))]
        if not isinstance(raw, list) or not raw:
            raise ValueError('Manifest requires a nonempty image list')
    else:
        raw = [{'sample_id': 'image_' + str(i + 1), 'image_path': str(p)} for i, p in enumerate(images or [])]
    rows = []
    for row in raw:
        if not isinstance(row, dict) or not {'sample_id', 'image_path'} <= set(row) <= {
                'sample_id', 'image_path', 'image_sha256'}:
            raise ValueError('Blind manifest must contain only sample_id, image_path and optional image_sha256')
        sid = row['sample_id']
        if not isinstance(sid, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,180}', sid):
            raise ValueError('Invalid sample ID')
        path = Path(row['image_path']).expanduser()
        path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
        actual = file_sha(path)
        if row.get('image_sha256', actual) != actual:
            raise ValueError('Image checksum mismatch: ' + sid)
        rows.append({'sample_id': sid, 'image_path': str(path), 'image_sha256': actual})
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate image IDs')
    return rows


def credentials():
    from dotenv import dotenv_values
    path = ROOT / '.env'
    if not path.is_file() or path.stat().st_mode & 0o077:
        raise ValueError('Create local .env and chmod 600 .env; its contents are never printed')
    values = dotenv_values(path, interpolate=False)
    keys = tuple(values.get(k) for k in ('OPENAI_API_KEY', 'EMBEDDING_API_KEY', 'VISION_API_KEY'))
    if not all(isinstance(k, str) and k and '$' not in k and 'replace' not in k.lower() for k in keys):
        raise ValueError('Three local role keys are required')
    return keys


def reference_codes(path, rows, raw_bytes=None):
    raw = decode(Path(path).read_bytes() if raw_bytes is None else raw_bytes)
    if not isinstance(raw, list):
        raise ValueError('ASA reference codes require a list of association_id, sample_id, code')
    ids = {r['sample_id'] for r in rows}
    out = []
    for item in raw:
        if (not isinstance(item, dict) or set(item) != {'association_id', 'sample_id', 'code'}
                or item['sample_id'] not in ids or not isinstance(item['association_id'], str)
                or not re.fullmatch(r'[A-Za-z0-9_.-]{1,180}', item['association_id'])
                or not isinstance(item['code'], str) or not re.fullmatch(r'[0-9]{8}', item['code'])):
            raise ValueError('Invalid ASA reference-code association; no weight or nutrition fields allowed')
        out.append(item)
    if len({a['association_id'] for a in out}) != len(out):
        raise ValueError('Duplicate ASA association ID')
    return out


def sealed_reference_codes(path, rows, store):
    identity_binding = {'ids': sorted(r['sample_id'] for r in rows)}
    identity_seal = store.get('identity_seal', 'all', identity_binding)
    if identity_seal is None or set(identity_seal) != set(identity_binding['ids']):
        raise ValueError('All blind identities must be sealed before reading reference codes')
    # One read: decode and hash exactly the same bytes, after the identity seal.
    raw = Path(path).read_bytes()
    references = reference_codes(path, rows, raw)
    value = {'sha256': hashlib.sha256(raw).hexdigest(), 'references': references}
    binding = {'source': str(Path(path).resolve()), 'identity_seal_sha256': digest(identity_seal)}
    store.put('reference_seal', 'all', binding, value)
    if store.get('reference_seal', 'all', binding) != value:
        raise ValueError('Reference code seal mismatch')
    return references


def output_directory(value):
    path = Path(os.path.abspath(Path(value).expanduser()))
    if path != path.resolve() or ROOT / 'runs' not in path.parents:
        raise ValueError('Output must be a non-symlink directory below this clone runs/')
    return path


def verify_prepared_graph(directory, files):
    if directory.resolve() != directory or any(p.is_symlink() for p in directory.rglob('*')):
        raise ValueError('Prepared Chroma graph contains a symlink')
    expected = {k.removeprefix('index/chroma/'): v for k, v in files.items()
                if k.startswith('index/chroma/') and k != 'index/chroma/chroma.sqlite3'}
    observed = {str(p.relative_to(directory)): file_sha(p) for p in directory.rglob('*')
                if p.is_file() and p.name != 'chroma.sqlite3'}
    if observed != expected:
        raise ValueError('Prepared Chroma graph changed; preserve it and restore the original copy')


@contextmanager
def original_index(m):
    import runtime
    parent = METHOD / 'prepared'
    parent.mkdir(parents=True, exist_ok=True)
    with (parent / 'index.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        m['chroma_index'].prepare_copy()
        verify_prepared_graph(parent / 'chroma', runtime.specification()['files'])
        index = m['chroma_index'].SearchIndex()
        try:
            yield index
        finally:
            index.client._system.stop()


def finite_prediction(pred):
    return isinstance(pred, dict) and set(pred) == set(TARGETS) and all(
        type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in pred.values())


def portable_run(rows, dataset, pipe, store, fingerprint, refs_loader=None):
    """Injectable orchestration. Cache all stages; retain failed outputs as None."""
    def cached(stage, sid, binding, fn):
        value = store.get(stage, sid, binding)
        if value is None:
            value = fn()
            store.put(stage, sid, binding, value)
        return value

    def identify(row):
        sid, image = row['sample_id'], Path(row['image_path'])
        if file_sha(image) != row['image_sha256']:
            raise ValueError('Image changed during inference')
        if dataset == 'nutrition5k':
            layout = cached('layout', sid, row, lambda: pipe.layout(image))
            identity = cached('identity', sid, {'row': row, 'layout': fingerprint(layout)},
                lambda: pipe.identify(image, N5_CONFIG, layout=layout))
        else:
            identity = cached('identity', sid, row,
                lambda: pipe.identify_select(image, 'asa24', ASA_CONFIG))
        return identity

    def predict(row, identity, refs):
        sid, image = row['sample_id'], Path(row['image_path'])
        binding = {'row': row, 'identity': fingerprint(identity)}
        portion = cached('portion', sid, binding, lambda:
            pipe.portion(image, identity, N5_CONFIG) if dataset == 'nutrition5k' else
            pipe.portion(image, identity, 'asa24', ASA_CONFIG))
        pred = portion.get('prediction')
        if portion['status'] == 'ok' and not finite_prediction(pred):
            raise ValueError('Invalid successful prediction')
        if portion['status'] != 'ok' and pred is not None:
            raise ValueError('A failed prediction must be null')
        associations = []
        for ref in refs:
            code = ref['code']
            item = dict(ref, status='not_matched', prediction=None, failure=None)
            if identity.get('status') != 'ok':
                item.update(status='failed', failure=identity.get('failure') or 'identity_failed')
            elif code in identity.get('final_codes', []):
                options = pipe.catalog.get('asa_options', {}).get(code, [])
                if not options:
                    item.update(status='failed', failure='missing_asa_options')
                else:
                    ctx = {'code': code, 'options': options}
                    # Same continuous-weight prompt as E2E for a matching primary
                    # code: exact-request caching reuses its reply, then the original
                    # parser projects onto ASA's legal options, not E2E's options.
                    result = cached('conditional', sid + '.' + code,
                        dict(binding, context=ctx), lambda: pipe.portion(image, identity,
                            'asa24', ASA_CONFIG, conditional_context=ctx))
                    if result['status'] == 'ok' and not finite_prediction(result['prediction']):
                        raise ValueError('Invalid ASA conditional prediction')
                    item.update(status=result['status'], prediction=result['prediction'], failure=result.get('failure'))
            associations.append(item)
        record = {'sample_id': sid, 'image_sha256': row['image_sha256'], 'dataset': dataset,
            'status': portion['status'], 'prediction': pred,
            'failure': None if pred is not None else identity.get('failure') or portion.get('failure'),
            'final_codes': identity.get('final_codes', []),
            'raw_leaves': portion.get('raw_leaves', portion.get('portions', [])),
            'conditional_associations': associations}
        return cached('record', sid, dict(binding, references=refs), lambda: record)

    return identify, predict


def population(rows, dataset, pipe, store, workers=32, refs_loader=None):
    identify, predict = portable_run(rows, dataset, pipe, store, digest)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        identities = list(executor.map(identify, rows))
    sealed = {r['sample_id']: digest(i) for r, i in zip(rows, identities)}
    binding = {'ids': sorted(sealed)}
    store.put('identity_seal', 'all', binding, sealed)
    if store.get('identity_seal', 'all', binding) != sealed:
        raise ValueError('Blind identity seal readback mismatch')
    refs = refs_loader() if refs_loader is not None else []
    lookup = {r['sample_id']: [a for a in refs if a['sample_id'] == r['sample_id']] for r in rows}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(predict, r, i, lookup[r['sample_id']]) for r, i in zip(rows, identities)]
        return [f.result() for f in futures]


def execute(args, rows, source, output):
    import runtime
    runtime.verify()
    m = modules(output)
    with original_index(m) as index:
        return execute_loaded(args, rows, source, output, m, index)


def execute_loaded(args, rows, source, output, m, index):
    import runtime
    catalog = read(V2 / 'round11/prepared/catalog.json')
    spec = {'schema': 1, 'dataset': args.dataset, 'images': rows, 'sources': source,
            'runtime_sha256': runtime.specification()['sha256'],
            'models': m['config'].MODELS, 'stage_models': m['config'].STAGE_MODELS,
            'reference_codes_path': str(args.reference_codes.resolve()) if args.reference_codes else None,
            'workers': args.workers, 'no_calibration': True, 'failed_prediction': None}
    frozen = digest(spec)
    freeze_path = output / 'freeze.json'
    if freeze_path.exists() and read(freeze_path) != spec:
        raise ValueError('Run inputs or code changed; use a different output directory')
    m['provider'].atomic_json(freeze_path, spec)
    store = m['newcommon'].Store(output / 'stages', frozen, len(rows))
    if (output / 'predictions.json').exists():
        saved = read(output / 'predictions.json')
        if saved['freeze_sha256'] != frozen or saved['records_sha256'] != digest(saved['records']):
            raise ValueError('Completed prediction seal changed')
        if args.reference_codes:
            sealed_reference_codes(args.reference_codes, rows, store)
        print(json.dumps({'status': 'already_completed', 'new_api_calls': 0, 'output': str(output)}))
        return
    provider = m['newcommon'].CachedProvider('N5' if args.dataset == 'nutrition5k' else 'ASA',
        allow_paid=True, keys=credentials())
    chat = m['stage_routes'].StageChat(provider, args.dataset, 8192 if args.dataset == 'nutrition5k' else 4096)
    factory = m['pipeline'].make_component_pipeline if args.dataset == 'nutrition5k' else m['pipeline'].make_asa_pipeline
    pipe = factory(chat, provider.embed, catalog, index.search4)
    try:
        records = population(rows, args.dataset, pipe, store, args.workers,
            (lambda: sealed_reference_codes(args.reference_codes, rows, store)) if args.reference_codes else None)
        m['provider'].atomic_json(output / 'predictions.json',
            {'freeze_sha256': frozen, 'records_sha256': digest(records), 'records': records})
        summary = {'status': 'completed', 'images': len(rows),
            'successful_e2e': sum(r['status'] == 'ok' for r in records),
            'failed_e2e': sum(r['status'] == 'failed' for r in records),
            'output': str(output), 'calibration': False, 'label_imputation': False}
        m['provider'].atomic_json(output / 'completion.json', summary)
        print(json.dumps(summary, ensure_ascii=False))
    except BaseException:
        provider.stopped = True
        raise
    finally:
        provider.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['nutrition5k', 'asa24'], default='nutrition5k')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--image', action='append', type=Path)
    source.add_argument('--manifest', type=Path)
    parser.add_argument('--reference-codes', type=Path, help='ASA associations only; never weight/nutrient labels')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/new-inputs')
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--round', type=int, dest='round_number', help='Select one round from the development image manifest')
    parser.add_argument('--allow-paid', action='store_true')
    parser.add_argument('--allow-modified-source', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error('workers must be between 1 and 32')
    if args.reference_codes and args.dataset != 'asa24':
        parser.error('reference codes are forbidden for Nutrition5k and unlabelled external images')
    rows = input_rows(args.image, args.manifest, args.dataset, args.round_number)
    sources = source_hashes(args.allow_modified_source)
    if not args.allow_paid:
        print(json.dumps({'status': 'dry_run_no_API', 'images': len(rows), 'dataset': args.dataset,
            'workers': args.workers, 'reference_codes_not_read': True,
            'destinations': ['https://api.zicc.cc/v1', 'https://api.openai.com/v1'],
            'message': 'Real inference requires local .env and explicit --allow-paid'}, ensure_ascii=False))
        return
    output = output_directory(args.output)
    os.umask(0o077)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args, rows, sources, output)


if __name__ == '__main__':
    main()

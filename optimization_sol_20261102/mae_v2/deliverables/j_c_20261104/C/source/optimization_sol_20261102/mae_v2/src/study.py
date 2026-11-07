#!/usr/bin/env python3
"""MAE v2: isolated, resumable finite development and frozen evaluation.

Development functions load only explicitly named development files. Sealed
validation access is delegated to validation_adapter after a passing freeze.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import threading

sys.dont_write_bytecode = True

from opt_provider import ROOT, Provider, ProviderStop, atomic_json, fingerprint, scoped, sha, stamp, usage
import legacy_adapter as legacy
from opt_metrics import KEYS, fold_medians
import metrics_v2 as metrics

ORDER = ('baseline', 'asa-retrieval-portion', 'examples-2', 'A', 'B', 'C', 'D', 'E')
BASE = {'variant': 'baseline', 'image_mode': 'square256', 'candidate_cap': 32,
        'portion_modes': {'asa24': 'direct', 'nutrition5k': 'direct'}, 'examples': False}
ASA = {**BASE, 'variant': 'optimized',
       'portion_modes': {'asa24': 'continuous_option', 'nutrition5k': 'total_ratio'}}
EXAMPLES = {**ASA, 'image_mode': 'native', 'examples': True}
FIRST = {'A': {'crop_review': False, 'portion_mode': 'direct'},
         'B': {'crop_review': True, 'portion_mode': 'direct'}}
INFERENCE_FILES = ('cohorts_v2.py', 'legacy_adapter.py', 'opt_provider.py',
                   'opt_pipeline.py', 'pipeline_v2.py', 'opt_metrics.py',
                   'metrics_v2.py', 'study.py', 'validation_adapter.py')


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    atomic_json(path, value)


def event(**value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def source_hashes():
    return {name: sha(ROOT / 'src' / name) for name in INFERENCE_FILES}


def dev_inputs():
    from cohorts_v2 import verify_dev_only, load_dev
    verify_dev_only()
    rows, labels = load_dev()
    if any(r['split'] != 'dev' for r in rows):
        raise ValueError('Development loader returned a sealed record')
    return rows, labels, load(ROOT / 'data/catalog.json')


def artifact(row):
    return fingerprint(row['sample_id'])[:24] + '.json'


def stage_dir(name, phase='development'):
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('Unsafe run name')
    return scoped(ROOT / 'runs' / phase / name)


def bind_run(directory, name, config, rows, source=None):
    directory.mkdir(parents=True, exist_ok=True)
    for part in ('identities', 'records'):
        (directory / part).mkdir(exist_ok=True)
    expected = {'name': name, 'config': config, 'manifest_sha256': fingerprint(rows),
                'code_sha256': source_hashes(), 'catalog_sha256': sha(ROOT / 'data/catalog.json'),
                'protocol_sha256': sha(ROOT / 'PROTOCOL.md'),
                'experiment_sha256': sha(ROOT / 'config/experiment.json'),
                'candidate_sha256': sha(ROOT / 'config/candidates.json'),
                'vector_sha256': sha(ROOT / 'data/vectors/manifest.json'),
                'source': source}
    path = directory / 'run_config.json'
    if path.exists() and load(path) != expected:
        raise ValueError('Run drift: preserve existing artifacts and resolve provenance before resuming')
    save(path, expected)
    return expected


def seal_identities(directory, rows):
    seal = {r['sample_id']: sha(directory / 'identities' / artifact(r)) for r in rows}
    path = directory / 'identities_sealed.json'
    if path.exists() and load(path) != seal:
        raise ValueError('Identity changed after sealing')
    save(path, seal)
    return seal


def read_identity(directory, row):
    seal = load(directory / 'identities_sealed.json')
    path = directory / 'identities' / artifact(row)
    if seal.get(row['sample_id']) != sha(path):
        raise ValueError('Source identity seal mismatch')
    return load(path)


def completed_records(directory, rows, allow_subset=False):
    path = directory / 'records_sealed.json'
    if not path.exists():
        raise ValueError('Source records have not been sealed')
    seal = load(path)
    ids = {r['sample_id'] for r in rows}
    if not ids.issubset(seal) or (not allow_subset and set(seal) != ids):
        raise ValueError('Source record population differs')
    records = []
    for row in rows:
        file = directory / 'records' / artifact(row)
        if sha(file) != seal[row['sample_id']]:
            raise ValueError('Source record changed after completion')
        record = load(file)
        if any(record.get(key) != row[key] for key in ('sample_id', 'dataset')):
            raise ValueError('Sealed record identity differs from its manifest row')
        records.append(record)
    return records


def finish_records(directory, rows, expected):
    if expected['code_sha256'] != source_hashes():
        raise ValueError('Inference source changed during the run; results retained but not scored')
    seal = {r['sample_id']: sha(directory / 'records' / artifact(r)) for r in rows}
    path = directory / 'records_sealed.json'
    if path.exists() and load(path) != seal:
        raise ValueError('Completed raw record changed')
    save(path, seal)
    predictions = [load(directory / 'records' / artifact(r)) for r in rows]
    save(directory / 'predictions.json', predictions)
    return predictions


def rebind_prediction(pred, row):
    pred = deepcopy(pred)
    pred.update({k: row[k] for k in ('sample_id', 'dataset', 'group_id', 'fold')})
    pred['method_fallback'] = bool(pred.get('method_fallback', False))
    pred['native_complete'] = bool(not pred.get('fallback') and not pred.get('repair_used'))
    return pred


def import_historical(name, source_name, rows, labels, catalog):
    """Import only raw label-blind development inference; recompute all scoring."""
    source = ROOT.parent / 'runs/development' / source_name
    historical_rows = {r['sample_id']: r for r in (
        json.loads(line) for line in (ROOT.parent / 'data/cohorts/dev.jsonl').read_text().splitlines())}
    historical_labels = load(ROOT.parent / 'data/cohorts/dev_labels.json')
    source_config = load(source / 'run_config.json')
    source_seal = load(source / 'identities_sealed.json')
    source_status = load(source / 'status.json')
    if source_status.get('status') != 'completed' or source_config.get('config', {}).get('examples'):
        raise ValueError('Historical source is unfinished or depends on supervised examples')
    directory = stage_dir(name)
    binding = {'historical_run': source_name, 'run_config_sha256': sha(source / 'run_config.json'),
               'original_predictions_sha256': sha(source / 'predictions.json')}
    expected = bind_run(directory, name, source_config['config'], rows, binding)
    imported = []
    for row in rows:
        sid = row['sample_id']
        old = historical_rows[sid]
        if old['split'] != 'dev' or old['image_sha256'] != row['image_sha256']:
            raise ValueError('Historical response image/split mismatch')
        if historical_labels[sid]['truth'] != labels[sid]['truth']:
            raise ValueError('Historical development reference mismatch')
        original_identity = source / 'identities' / artifact(row)
        if source_seal.get(sid) != sha(original_identity):
            raise ValueError('Historical identity seal mismatch')
        original_record = source / 'records' / artifact(row)
        identity = load(original_identity)
        pred = rebind_prediction(load(original_record), row)
        pred['historical_import'] = {'record_sha256': sha(original_record),
                                      'identity_sha256': sha(original_identity), **binding}
        # Fold medians and fitting are rebuilt by crossvalidate; no old OOF value
        # or calibration is imported as a v2 estimate.
        save(directory / 'identities' / artifact(row), identity)
        save(directory / 'records' / artifact(row), pred)
        imported.append(pred)
    seal_identities(directory, rows)
    finish_records(directory, rows, expected)
    save(directory / 'status.json', {'status': 'completed', 'historical_raw_import': True,
                                    'new_api_calls': 0, 'completed': len(rows)})
    return directory


def score_run(directory, rows, labels, catalog, estimated_calls, registration_index,
              examples=False):
    raw = completed_records(directory, rows)
    result = metrics.crossvalidate(rows, labels, raw, catalog['asa_options'],
        {c: f['density'] for c, f in catalog['foods'].items()}, calibrate=False, examples=examples)
    save(directory / 'oof_predictions.json', result['predictions'])
    save(directory / 'metrics.json', result['metrics'])
    save(directory / 'fold_parameters.json', result['fold_parameters'])
    return {'name': directory.name, 'directory': str(directory.relative_to(ROOT)),
            'metrics': result['metrics'], 'estimated_chat_calls': estimated_calls,
            'registration_index': registration_index, 'examples': examples,
            'config': load(directory / 'run_config.json')['config'], 'calibrated': False}


def component_pipeline(provider, catalog):
    from pipeline_v2 import ComponentPipeline
    legacy_pipeline = legacy.make_pipeline(provider, catalog)
    return ComponentPipeline(lambda messages: provider.chat(messages, max_completion_tokens=8192),
                             provider.embed, catalog, legacy_pipeline.retriever)


def to_prediction(row, identity, result, fallback, catalog):
    valid = result.get('status') == 'ok' and result.get('prediction') is not None
    repaired = bool(identity.get('repaired') or result.get('repaired'))
    method_fallback = bool(result.get('method_fallback'))
    items = identity.get('items', identity.get('leaves', []))
    codes = list(dict.fromkeys(i['primary_code'] for i in items if i.get('primary_code')))
    retrieved = list(dict.fromkeys(identity.get('retrieved_codes', [])))
    if not set(codes).issubset(retrieved):
        raise ValueError('Component code is absent from retrieved candidates')
    leaves = []
    by_id = {i.get('component_id', i.get('item_id')): i for i in items}
    for p in result.get('portions', []) if valid else []:
        cid = p.get('component_id', p.get('item_id'))
        item = by_id.get(cid)
        if item is None:
            raise ValueError('Portion identity has no matching component')
        code = item['primary_code']
        leaves.append({'item_id': cid, 'component_id': cid, 'parent_id': item['parent_id'],
                       'code': code, 'grams': p['grams'], 'density': catalog['foods'][code]['density']})
    return {'sample_id': row['sample_id'], 'dataset': row['dataset'], 'group_id': row['group_id'],
            'fold': row['fold'], 'final_codes': codes, 'primary_code': codes[0] if codes else None,
            'retrieved_codes': retrieved, 'items': items, 'parents': identity.get('parents', []),
            'description': identity.get('description', ''), 'identity_status': identity.get('status'),
            'raw_prediction': result.get('prediction') if valid else None,
            'prediction': result['prediction'] if valid else dict(fallback),
            'native_complete': valid and not repaired and not method_fallback,
            'repair_used': repaired, 'fallback': not valid,
            'fallback_reason': None if valid else result.get('failure', 'incomplete_dish'),
            'method_fallback': method_fallback, 'method_fallback_reason': result.get('method_fallback_reason'),
            'raw_leaves': leaves, 'portion_result': result, 'conditional': None, 'e2e': None}


def run_component(name, config, rows, labels, catalog, allow_paid, workers=4,
                  source=None, phase='development', frozen_fallback=None):
    directory = stage_dir(name, phase)
    source_binding = ({'directory': str(source.relative_to(ROOT)),
                       'identities_sha256': sha(source / 'identities_sealed.json'),
                       'records_sha256': sha(source / 'records_sealed.json')} if source else None)
    expected = bind_run(directory, name, config, rows, source_binding)
    if source:
        completed_records(source, rows)  # Validate contents, not only the seal file hash.
    status = {'status': 'running', 'started': stamp(), 'phase': 'identity', 'completed': 0, 'total': len(rows)}
    provider = Provider(directory, allow_paid, namespace='mae-v2-' + phase)
    pipeline = component_pipeline(provider, catalog)
    lock = threading.Lock()
    profile = config['profile']

    def progress(count, total):
        with lock:
            status.update(completed=count, total=total, updated=stamp())
            save(directory / 'status.json', status)
        if count % 10 == 0 or count == total:
            event(run=name, phase=status['phase'], completed=count, total=total)

    def identity_step(row):
        path = directory / 'identities' / artifact(row)
        if path.exists():
            return
        picture = legacy.image_path(row)
        if source and profile in ('C', 'D', 'E'):
            value = read_identity(source, row)
        else:
            layout = read_identity(source, row).get('layout_result') if source and profile == 'B' else None
            value = pipeline.identify(picture, config, layout=layout)
        save(path, value)

    def portion_step(row):
        path = directory / 'records' / artifact(row)
        if path.exists():
            return
        picture = legacy.image_path(row)
        identity = read_identity(directory, row)
        fallback = frozen_fallback or fold_medians(rows, labels, heldout_fold=row['fold'], dataset='nutrition5k')
        if source and profile in ('D', 'E'):
            base = load(source / 'records' / artifact(row))
            if base.get('fallback'):
                pred = rebind_prediction(base, row)
                pred.update(method_fallback=True, method_fallback_reason='base_incomplete')
                save(path, pred)
                return
            if profile == 'D':
                final_identity, result = pipeline.review_codes(picture, identity, base['portion_result'], config)
                # Preserve the sealed pre-review input; output identity is separately bound.
                save(directory / 'reviewed_identities' / artifact(row), final_identity)
                identity = final_identity
            else:
                result = pipeline.standard_portions(picture, identity, base['portion_result'], config)
        else:
            result = pipeline.portion(picture, identity, config)
        pred = to_prediction(row, identity, result, fallback, catalog)
        save(path, pred)

    try:
        legacy.run_parallel(rows, identity_step, workers, progress)
        seal_identities(directory, rows)
        status.update(phase='portion', completed=0)
        legacy.run_parallel(rows, portion_step, workers, progress)
        finish_records(directory, rows, expected)
        status.update(status='completed')
    except BaseException as exc:
        status.update(status='interrupted', error=str(exc) if isinstance(exc, ProviderStop) else type(exc).__name__)
        raise
    finally:
        save(directory / 'usage.json', usage(directory))
        save(directory / 'status.json', status)
        provider.close()
    return directory


def run_legacy(name, config, rows, labels, catalog, allow_paid, workers=4, source=None,
               phase='development', training_rows=None, training_labels=None, frozen_fallback=None):
    directory = stage_dir(name, phase)
    source_binding = ({'directory': str(source.relative_to(ROOT)),
                       'identities_sha256': sha(source / 'identities_sealed.json')} if source else None)
    expected = bind_run(directory, name, config, rows, source_binding)
    provider = Provider(directory, allow_paid, namespace='mae-v2-' + phase)
    pipeline = legacy.make_pipeline(provider, catalog)
    status = {'status': 'running', 'phase': 'identity', 'started': stamp(), 'completed': 0, 'total': len(rows)}
    lock = threading.Lock()
    train_rows, train_labels = training_rows or rows, training_labels or labels

    def progress(count, total):
        with lock:
            status.update(completed=count, total=total, updated=stamp())
            save(directory / 'status.json', status)
        if count % 10 == 0 or count == total:
            event(run=name, phase=status['phase'], completed=count, total=total)

    def identify(row):
        path = directory / 'identities' / artifact(row)
        if path.exists():
            return
        value = (read_identity(source, row) if source else pipeline.identify_select(
            legacy.image_path(row), row['dataset'], legacy.per_dataset(config, row['dataset'])))
        save(path, value)

    def portion(row):
        path = directory / 'records' / artifact(row)
        if path.exists():
            return
        identity = read_identity(directory, row)
        fallback = (frozen_fallback[row['dataset']] if frozen_fallback else fold_medians(
            train_rows, train_labels, heldout_fold=row['fold'], dataset=row['dataset']))
        examples, sources = ([], [])
        if config.get('examples'):
            bank = stage_dir('examples-2') / 'identities' if phase != 'development' else directory / 'identities'
            examples, sources = legacy.choose_examples(row, identity, train_rows, train_labels, bank, provider)
        pred = legacy.infer_portions(pipeline, row, identity, config, catalog, labels.get(row['sample_id']), fallback, examples)
        pred = rebind_prediction(pred, row)
        pred['example_sources'] = sources
        save(path, pred)

    try:
        legacy.run_parallel(rows, identify, workers, progress)
        seal_identities(directory, rows)
        status.update(phase='portion', completed=0)
        legacy.run_parallel(rows, portion, workers, progress)
        finish_records(directory, rows, expected)
        status['status'] = 'completed'
    except BaseException as exc:
        status.update(status='interrupted', error=str(exc) if isinstance(exc, ProviderStop) else type(exc).__name__)
        raise
    finally:
        save(directory / 'usage.json', usage(directory))
        save(directory / 'status.json', status)
        provider.close()
    return directory


def calibrate_winner(candidate, rows, labels, catalog, suffix):
    if candidate.get('examples'):
        return candidate
    directory = ROOT / candidate['directory']
    raw = completed_records(directory, rows)
    result = metrics.compare_calibration(rows, labels, raw, catalog['asa_options'],
            {c: f['density'] for c, f in catalog['foods'].items()}, dataset=rows[0]['dataset'])
    save(directory / ('calibration-' + suffix + '.json'), result)
    save(directory / 'calibration-summary.json', {
        'raw_metrics': result['raw_metrics'], 'calibrated_metrics': result['calibrated_metrics'],
        'calibration_enabled': result['calibration_enabled'],
        'calibration_reason': result['calibration_reason'],
        'mass_factor': result['selected_parameters']['mass_factor'],
        'raw_factor': result['selected_parameters']['raw_factor'], 'round': suffix})
    # The metrics API explicitly names the chosen output; no round-dependent
    # fitting of unsuccessful candidates is permitted here.
    selected = result.get('selected', result)
    out = deepcopy(candidate)
    out.update(metrics=selected['metrics'], calibrated=bool(selected['calibration_enabled']),
               calibration_enabled=bool(selected['calibration_enabled']),
               calibration_artifact=str((directory / ('calibration-' + suffix + '.json')).relative_to(ROOT)))
    return out


def choose(candidates, dataset):
    selected = metrics.select_candidate(candidates, dataset)
    if selected is None:
        return None
    return selected


def retain(candidates, candidate):
    """Replace one raw winner with its single retained calibrated/raw profile."""
    for index, previous in enumerate(candidates):
        if previous['name'] == candidate['name']:
            candidates[index] = candidate
            return
    raise ValueError('Cannot calibrate an unregistered candidate')


def nutrient_failed(candidate):
    values = candidate['metrics']['datasets']['nutrition5k']['primary']['metrics']
    return any(not values[k]['pass_mae'] or not values[k]['pass_per_unit'] for k in KEYS[1:])


def weight_failed(candidate):
    value = candidate['metrics']['datasets']['nutrition5k']['primary']['metrics']['weight']
    return not value['pass_mae'] or not value['pass_per_unit']


def selection(candidates, rows, labels, catalog, bootstrap=False):
    profiles, merged = {}, []
    for dataset in ('asa24', 'nutrition5k'):
        selected = metrics.select_profile(candidates, dataset)
        if selected is None:
            return {'status': 'recognition_floor_not_met', 'joint_pass': False, 'profiles': {},
                    'validation_opened_for_inference_or_scoring': False}
        profiles[dataset] = selected
        if selected.get('calibration_artifact'):
            result = load(ROOT / selected['calibration_artifact'])
            predictions = result.get('selected', result)['predictions']
        else:
            predictions = load(ROOT / selected['directory'] / 'oof_predictions.json')
        merged.extend(p for p in predictions if p['dataset'] == dataset)
    final = metrics.evaluate(rows, labels, merged, bootstrap=bootstrap)
    result = {'status': 'development_selection', 'scope': 'development_not_independent_validation',
              'profiles': profiles, 'metrics': final, 'joint_pass': final['joint_pass'],
              'candidate_names': [c['name'] for c in candidates],
              'validation_opened_for_inference_or_scoring': False}
    save(ROOT / 'runs/development/selected_oof_predictions.json', merged)
    save(ROOT / 'runs/development/selection.json', result)
    return result


def develop(args):
    rows, labels, catalog = dev_inputs()
    asa = [r for r in rows if r['dataset'] == 'asa24']
    n5 = [r for r in rows if r['dataset'] == 'nutrition5k']
    candidates = []
    for name, source, subset, calls, index in (
            ('baseline', 'baseline', rows, 4, 0),
            ('asa-retrieval-portion', 'retrieval-portion', asa, 4, 1)):
        directory = import_historical(name, source, subset, labels, catalog)
        candidates.append(score_run(directory, subset, labels, catalog, calls, index))
    asa_raw = candidates[-1]
    retain(candidates, calibrate_winner(asa_raw, asa, labels, catalog, 'asa'))
    native_source = import_historical('reference-identities', 'native-portion', n5, labels, catalog)
    directory = run_legacy('examples-2', EXAMPLES, n5, labels, catalog, args.allow_paid, args.workers,
                           source=native_source)
    candidates.append(score_run(directory, n5, labels, catalog, 4, 2, examples=True))
    first = []
    for name in ('A', 'B'):
        cfg = {**FIRST[name], 'profile': name}
        directory = run_component(name, cfg, n5, labels, catalog, args.allow_paid, args.workers,
                                  source=stage_dir('A') if name == 'B' else None)
        candidate = score_run(directory, n5, labels, catalog, 3 if name == 'A' else 4, ORDER.index(name))
        first.append(candidate)
        candidates.append(candidate)
        selection(candidates, rows, labels, catalog)
    upstream = choose(first, 'nutrition5k')
    cconfig = {**upstream['config'], 'profile': 'C', 'portion_mode': 'total_ratio',
               'identity_profile': upstream['name']}
    directory = run_component('C', cconfig, n5, labels, catalog, args.allow_paid, args.workers,
                              source=ROOT / upstream['directory'])
    c = score_run(directory, n5, labels, catalog, upstream['estimated_chat_calls'], ORDER.index('C'))
    first.append(c)
    candidates.append(c)
    raw_winner = choose(first, 'nutrition5k')
    retain(candidates, calibrate_winner(raw_winner, n5, labels, catalog, 'round1'))
    result = selection(candidates, rows, labels, catalog)
    decision = {'first_round_raw_winner': raw_winner['name'], 'D': False, 'E': False,
                'reason': 'Only N5 failures can trigger N5 expansion'}
    current = result['profiles'].get('nutrition5k') if result.get('profiles') else metrics.select_profile(candidates, 'nutrition5k')
    if current and not all(v['pass_mae'] and v['pass_per_unit'] for v in
                          current['metrics']['datasets']['nutrition5k']['primary']['metrics'].values()):
        from pipeline_v2 import ComponentPipeline
        offline_pipeline = ComponentPipeline(None, None, catalog, None)
        decision['D'] = nutrient_failed(current)
        decision['E'] = weight_failed(current) and any(
            offline_pipeline.standard_portion_coverage(read_identity(ROOT / raw_winner['directory'], row))['eligible_leaf_count']
            for row in n5)
        save(ROOT / 'runs/development/round2_decision.json', decision)
        second = []
        for name in ('D', 'E'):
            if not decision[name]:
                continue
            cfg = {**raw_winner['config'], 'profile': name, 'base_profile': raw_winner['name']}
            directory = run_component(name, cfg, n5, labels, catalog, args.allow_paid, args.workers,
                                      source=ROOT / raw_winner['directory'])
            value = score_run(directory, n5, labels, catalog,
                              raw_winner['estimated_chat_calls'] + (2 if name == 'D' else 1), ORDER.index(name))
            candidates.append(value)
            second.append(value)
        if second:
            winner = choose(first + second, 'nutrition5k')
            if winner['name'] != raw_winner['name']:
                retain(candidates, calibrate_winner(winner, n5, labels, catalog, 'round2'))
    save(ROOT / 'runs/development/round2_decision.json', decision)
    result = selection(candidates, rows, labels, catalog, bootstrap=10000)
    save(ROOT / 'runs/development/candidates.json', candidates)
    completion = {'status': 'ready_for_review' if result['joint_pass'] else 'development_targets_not_met',
                  'completed': stamp(), 'registered_candidates_exhausted': not result['joint_pass'],
                  'validation_opened_for_inference_or_scoring': False,
                  'joint_pass': result['joint_pass'], 'active_development_rows': len(rows)}
    save(ROOT / 'runs/development/completion.json', completion)
    event(**completion)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('audit', 'verify-dev', 'dev', 'evaluate', 'freeze', 'validate', 'demo'))
    parser.add_argument('--allow-paid', action='store_true')
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=4)
    parser.add_argument('--review', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'audit':
        from cohorts_v2 import prepare
        event(**prepare())
    elif args.command == 'verify-dev':
        from cohorts_v2 import verify_dev_only
        event(**verify_dev_only())
    elif args.command == 'dev':
        if not args.allow_paid:
            raise ProviderStop('Explicit paid development flag is required')
        root = ROOT / 'runs/development'
        root.mkdir(parents=True, exist_ok=True)
        with (root / '.job.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ProviderStop('Another development job holds the run lock') from None
            save(root / 'job.json', {'pid': os.getpid(), 'started': stamp(), 'workers': args.workers})
            develop(args)
    elif args.command == 'evaluate':
        rows, labels, catalog = dev_inputs()
        candidates = load(ROOT / 'runs/development/candidates.json')
        result = selection(candidates, rows, labels, catalog, bootstrap=10000)
        event(status=result['status'], joint_pass=result['joint_pass'])
    elif args.command in ('freeze', 'validate'):
        from validation_adapter import freeze, validate
        (freeze if args.command == 'freeze' else validate)(args)
    else:
        path = ROOT / 'runs/development/selection.json'
        if not path.is_file():
            event(status='development_not_complete', api_calls=0)
        else:
            saved = load(path)
            event(status=saved['status'], metrics=saved.get('metrics'), api_calls=0,
                  independent_validation=False)


if __name__ == '__main__':
    try:
        main()
    except ProviderStop as exc:
        event(status='stopped', reason=str(exc))
        raise SystemExit(2)

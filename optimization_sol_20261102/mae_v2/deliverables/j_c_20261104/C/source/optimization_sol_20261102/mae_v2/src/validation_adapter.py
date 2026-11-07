"""Sealed-data boundary and one frozen baseline/optimized validation batch.

Only this module can read validation labels. Before all inference completes it
returns ASA24 food codes alone; no reference quantities or N5 targets leave the
adapter. Final scoring starts after both complete output populations exist.
"""
from copy import deepcopy
import fcntl
import json
import math
from pathlib import Path
import re

from opt_provider import ROOT, ProviderStop, atomic_json, fingerprint, scoped, sha, stamp
import metrics_v2 as metrics


def load(path):
    return json.loads(Path(path).read_text())


def _require_development_gate(selection):
    if not selection.get('joint_pass') or not selection.get('metrics', {}).get('joint_pass'):
        raise ValueError('All twenty development metrics and the 15% recognition gate must pass')
    score = selection['metrics']
    checks = score.get('checks', [])
    wanted = {(d, k, s) for d in ('asa24', 'nutrition5k') for k in metrics.KEYS for s in ('mae', 'per_unit')}
    if len(checks) != 20 or {(c.get('dataset'), c.get('target'), c.get('statistic')) for c in checks} != wanted:
        raise ValueError('Twenty distinct fixed metric checks are required')
    for check in checks:
        limit = metrics.THRESHOLDS[check['dataset']][check['target']][check['statistic'] == 'per_unit']
        value = check.get('value')
        if (type(value) not in (int, float) or not math.isfinite(value)
                or not 0 <= value <= limit or check.get('threshold') != limit):
            raise ValueError('Development metrics do not meet the fixed full-precision limits')
    asa = score.get('datasets', {}).get('asa24', {})
    n, hits = asa.get('n'), asa.get('exact_n')
    if type(n) is not int or type(hits) is not int or n <= 0 or not metrics.minimum_exact_n(n) <= hits <= n:
        raise ValueError('Development recognition does not meet the all-row 15% floor')


def _review_gate(path, hashes):
    expected = ROOT / 'evidence/review-one-completion.json'
    if path is None or Path(path).resolve() != expected.resolve() or not expected.is_file():
        raise ValueError('The exact independent checkpoint-one completion evidence is required')
    review = load(expected)
    if (review.get('status') != 'approved_or_fixed' or review.get('unresolved_p1_p2') != 0
            or review.get('code_sha256') != hashes
            or review.get('protocol_sha256') != sha(ROOT / 'PROTOCOL.md')):
        raise ValueError('Independent review is incomplete or belongs to another source snapshot')
    for key, relative in (
        ('selection_sha256', 'runs/development/selection.json'),
        ('dev_freeze_sha256', 'data/cohorts/dev_freeze.json'),
        ('validation_freeze_sha256', 'data/freeze/validation_freeze.json'),
        ('catalog_sha256', 'data/catalog.json')):
        if review.get(key) != sha(ROOT / relative):
            raise ValueError('Review artifact binding differs from the completed study')
    return review


def _sealed_manifest():
    seal_path = ROOT / 'data/freeze/validation_freeze.json'
    if not seal_path.is_file():
        raise ValueError('The 200-image replacement cohort is not ready')
    seal = load(seal_path)
    if seal.get('images') != 200:
        raise ValueError('Frozen Nutrition5k population must have exactly 200 images')
    required = {'data/cohorts/sealed/validation.jsonl', 'data/cohorts/sealed/validation_labels.json'}
    if not required <= seal['artifacts'].keys():
        raise ValueError('Validation seal does not cover both manifest and reference adapter input')
    for relative, expected in seal['artifacts'].items():
        if relative not in required and not re.fullmatch(
                r'data/(asa24|nutrition5k)/validation/images/[0-9a-f]{64}\.png', relative):
            raise ValueError('Forbidden path in validation seal')
        path = scoped(ROOT / relative)
        if sha(path) != expected:
            raise ValueError('Sealed validation artifact mismatch')
    path = ROOT / 'data/cohorts/sealed/validation.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if any(r['split'] != 'validation' or r.get('fold') is not None for r in rows):
        raise ValueError('Invalid frozen validation manifest')
    counts = {d: sum(r['dataset'] == d for r in rows) for d in ('asa24', 'nutrition5k')}
    if (len(rows) != 278 or any(r['dataset'] not in ('asa24', 'nutrition5k') for r in rows)
            or counts != {'asa24': 78, 'nutrition5k': 200} or len({r['sample_id'] for r in rows}) != len(rows)):
        raise ValueError('Validation sample count/identity mismatch')
    for row in rows:
        if (row['image_path'] not in seal['artifacts']
                or row['image_sha256'] != seal['artifacts'][row['image_path']]):
            raise ValueError('Validation image is not bound to its manifest declaration')
    return rows


def _conditional_codes(rows):
    """No size, portion, weight, nutrients or N5 information crosses this API."""
    labels = load(ROOT / 'data/cohorts/sealed/validation_labels.json')
    if set(labels) != {r['sample_id'] for r in rows}:
        raise ValueError('Sealed reference population differs')
    result = {}
    for row in rows:
        if row['dataset'] != 'asa24':
            continue
        code = labels[row['sample_id']].get('food_code')
        if not isinstance(code, str) or re.fullmatch(r'[0-9]{8}', code) is None:
            raise ValueError('Conditional adapter requires an eight-digit code string')
        result[row['sample_id']] = {'food_code': code}
    return result


def freeze(args):
    import study
    rows, labels, catalog = study.dev_inputs()
    selected = load(ROOT / 'runs/development/selection.json')
    _require_development_gate(selected)
    hashes = study.source_hashes()
    _review_gate(args.review, hashes)
    # Independent scoring is not opened here. File integrity and neutral
    # manifest verification are part of the reviewed preparer boundary.
    val_rows = _sealed_manifest()
    density = {c: f['density'] for c, f in catalog['foods'].items()}
    profiles = deepcopy(selected['profiles'])
    for dataset, profile in profiles.items():
        subset = [r for r in rows if r['dataset'] == dataset]
        raw = study.completed_records(ROOT / profile['directory'], subset, allow_subset=True)
        if profile.get('calibration_artifact'):
            parameters = load(ROOT / profile['calibration_artifact'])['selected_parameters']
        else:
            parameters = metrics.fit_final(subset, labels, raw, catalog['asa_options'], density,
                                             dataset=dataset, enabled=False, examples=profile.get('examples', False))
        profile['parameters'] = parameters
    baseline_raw = study.completed_records(study.stage_dir('baseline'), rows)
    baseline = {ds: metrics.fit_final(rows, labels, baseline_raw, catalog['asa_options'], density,
                                     dataset=ds, enabled=False) for ds in ('asa24', 'nutrition5k')}
    binding = {'schema_version': 2, 'code_sha256': hashes,
               'protocol_sha256': sha(ROOT / 'PROTOCOL.md'),
               'config_sha256': {p.name: sha(p) for p in sorted((ROOT / 'config').glob('*.json'))},
               'dev_freeze_sha256': sha(ROOT / 'data/cohorts/dev_freeze.json'),
               'validation_freeze_sha256': sha(ROOT / 'data/freeze/validation_freeze.json'),
               'catalog_sha256': sha(ROOT / 'data/catalog.json'),
               'vectors_sha256': sha(ROOT / 'data/vectors/manifest.json'),
               'review_sha256': sha(args.review),
               'selection_sha256': sha(ROOT / 'runs/development/selection.json'),
               'validation_manifest_fingerprint': fingerprint(val_rows),
               'profiles': profiles, 'baseline_config': study.BASE, 'baseline_parameters': baseline,
               'conditional_protocol': 'true_code_and_complete_legal_catalog_only',
               'retries': {'http_attempts_per_payload': 3, 'format_repairs_per_stage': 1},
               'bootstrap_resamples': 10000, 'validation_policy': 'one_frozen_batch_no_tuning'}
    path = ROOT / 'runs/frozen/method.json'
    if path.exists() and load(path) != binding:
        raise ValueError('A different method has already been frozen; no overwrite permitted')
    atomic_json(path, binding)
    method_seal = ROOT / 'runs/frozen/method_seal.json'
    expected_seal = {'method_sha256': sha(path)}
    if method_seal.exists() and load(method_seal) != expected_seal:
        raise ValueError('Frozen method seal mismatch')
    atomic_json(method_seal, expected_seal)
    study.event(status='frozen', method_sha256=sha(path), validation_model_calls=0)


def _verify_method():
    import study
    path = ROOT / 'runs/frozen/method.json'
    if not path.is_file():
        raise ValueError('No independently reviewed passing method is frozen')
    frozen = load(path)
    seal = ROOT / 'runs/frozen/method_seal.json'
    if not seal.is_file() or load(seal).get('method_sha256') != sha(path):
        raise ValueError('Frozen method content differs from its independent seal')
    if frozen['code_sha256'] != study.source_hashes():
        raise ValueError('Frozen inference source drift')
    checks = {'protocol_sha256': ROOT / 'PROTOCOL.md',
              'dev_freeze_sha256': ROOT / 'data/cohorts/dev_freeze.json',
              'validation_freeze_sha256': ROOT / 'data/freeze/validation_freeze.json',
              'catalog_sha256': ROOT / 'data/catalog.json',
              'vectors_sha256': ROOT / 'data/vectors/manifest.json',
              'review_sha256': ROOT / 'evidence/review-one-completion.json',
              'selection_sha256': ROOT / 'runs/development/selection.json'}
    if any(sha(p) != frozen[key] for key, p in checks.items()):
        raise ValueError('Frozen artifact drift')
    if {p.name: sha(p) for p in sorted((ROOT / 'config').glob('*.json'))} != frozen['config_sha256']:
        raise ValueError('Frozen model/candidate configuration drift')
    return frozen


def validate(args):
    if not args.allow_paid:
        raise ProviderStop('Explicit paid validation flag is required')
    import study
    frozen = _verify_method()
    rows = _sealed_manifest()
    if fingerprint(rows) != frozen['validation_manifest_fingerprint']:
        raise ValueError('Frozen manifest drift')
    root = ROOT / 'runs/validation'
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.job.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ProviderStop('The one validation batch is already running') from None
        identity = {'method_sha256': sha(ROOT / 'runs/frozen/method.json')}
        path = root / 'batch.json'
        if path.exists() and load(path) != identity:
            raise ValueError('A validation batch exists for another method')
        atomic_json(path, identity)
        if (root / 'completion.json').is_file():
            study.event(status='already_completed', new_api_calls=0)
            return
        train, train_labels, catalog = study.dev_inputs()
        contexts = _conditional_codes(rows)
        base = study.run_legacy('baseline', frozen['baseline_config'], rows, contexts, catalog,
                args.allow_paid, args.workers, phase='validation', training_rows=train,
                training_labels=train_labels,
                frozen_fallback={d: p['fallback'] for d, p in frozen['baseline_parameters'].items()})
        optimized = []
        for dataset in ('asa24', 'nutrition5k'):
            profile = frozen['profiles'][dataset]
            cfg = deepcopy(profile['config'])
            subset = [r for r in rows if r['dataset'] == dataset]
            name = 'optimized-' + dataset
            if dataset == 'asa24' or 'profile' not in cfg:
                directory = study.run_legacy(name, cfg, subset, contexts, catalog,
                    args.allow_paid, args.workers, phase='validation', training_rows=train,
                    training_labels=train_labels, frozen_fallback={dataset: profile['parameters']['fallback']})
            else:
                source = None
                if cfg['profile'] in ('D', 'E'):
                    base_cfg = {**cfg, 'profile': cfg['base_profile']}
                    source = study.run_component(name + '-base', base_cfg, subset, {}, catalog,
                        args.allow_paid, args.workers, phase='validation',
                        frozen_fallback=profile['parameters']['fallback'])
                directory = study.run_component(name, cfg, subset, {}, catalog,
                    args.allow_paid, args.workers, source=source, phase='validation',
                    frozen_fallback=profile['parameters']['fallback'])
            optimized.extend(study.completed_records(directory, subset))
        baseline = study.completed_records(base, rows)
        if len(optimized) != len(rows) or len(baseline) != len(rows):
            raise ValueError('Both complete populations must exist before scoring')
        expected = {r['sample_id']: r['dataset'] for r in rows}
        for population in (optimized, baseline):
            actual = {p.get('sample_id'): p.get('dataset') for p in population}
            if actual != expected or len(actual) != len(population):
                raise ValueError('Complete unique sample IDs and datasets are required before scoring')
        _verify_method()
        # First evaluation access: all registered inference is now complete.
        labels = load(ROOT / 'data/cohorts/sealed/validation_labels.json')
        density = {c: f['density'] for c, f in catalog['foods'].items()}
        results = {}
        for arm, predictions in (('baseline', baseline), ('optimized', optimized)):
            applied = [metrics.apply_parameters(p,
                frozen['baseline_parameters'][p['dataset']] if arm == 'baseline'
                else frozen['profiles'][p['dataset']]['parameters'], catalog['asa_options'], density)
                for p in predictions]
            atomic_json(root / (arm + '-scored-predictions.json'), applied)
            results[arm] = metrics.evaluate(rows, labels, applied, bootstrap=10000)
        atomic_json(root / 'metrics.json', results)
        completion = {'status': 'completed', 'finished': stamp(), 'joint_pass': results['optimized']['joint_pass'],
                      'independent_acceptance_scope': '200_images_with_preregistered_linkage_groups',
                      'method_sha256': identity['method_sha256'], 'no_post_validation_tuning': True}
        atomic_json(root / 'completion.json', completion)
        study.event(**completion)

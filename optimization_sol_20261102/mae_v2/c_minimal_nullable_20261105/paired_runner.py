"""Offline-first contract and injected paired execution; no provider or label-file loader.

CLI never constructs clients. The adapter deliberately has no paid backend yet.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

TARGETS = ('weight', 'energy', 'fat', 'carbs', 'protein')
VARIANTS = ('stage_split_raw', 'nullable_raw')
MODELS = {'recognition': 'gemini-3.8-flash', 'selection': 'gemini-3.8-flash',
          'portion': {'nutrition5k': 'gemini-3.8-flash', 'asa24': 'gpt-6.1-sol'}}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def require(value, reason):
    if not value:
        raise ValueError(reason)


def request_key(*, messages, images_sha256, model, parameters, source_binding):
    """Only exact payload reuse; no identity/record cache imported across variants."""
    return digest(dict(messages=messages, images_sha256=images_sha256, model=model,
                       parameters=parameters, source_binding=source_binding))


def valid_vector(value):
    return isinstance(value, dict) and all(type(value.get(k)) in (int, float)
        and math.isfinite(value[k]) and value[k] >= 0 for k in TARGETS)


def preflight(manifest, *, verify_images=True):
    """Manifest contains blind IDs/hash/proof metadata, never nutritional labels.

    Exclusion proof is an externally reviewed artifact binding the complete pool.
    Its hash alone is provenance, not independent proof of its scientific claims.
    """
    rows = manifest['pool']
    require(len(rows) >= 3000, 'pool_requires_at_least_3000_unique_images')
    ids, images, rgbs, groups = set(), set(), set(), {}
    for row in rows:
        sid, image, group = row['sample_id'], row['image_sha256'], (row['dataset'], row['group_id'])
        require(row['dataset'] in ('asa24', 'nutrition5k'), 'unknown_dataset')
        require(sid not in ids and image not in images, 'duplicate_pool_id_or_image')
        require(len(image) == 64 and all(c in '0123456789abcdef' for c in image), 'invalid_image_hash')
        rgb = row['unified_rgb_sha256']
        require(len(rgb) == 64 and all(c in '0123456789abcdef' for c in rgb) and rgb not in rgbs, 'invalid_or_duplicate_rgb_hash')
        require(type(row.get('reference_count')) is int and row['reference_count'] > 0, 'reference_association_count_required')
        require(row.get('truth_complete') is True, 'five_target_answers_not_verified')
        require(row.get('acceptance_excluded') is True, 'acceptance_complete_group_not_excluded')
        if verify_images:
            from PIL import Image
            path = Path(row['image_path'])
            require(path.is_absolute(), 'absolute_image_path_required')
            require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == image,
                    'missing_or_changed_pool_image')
            with Image.open(path) as image_file:
                actual_rgb = hashlib.sha256(str(image_file.size).encode() + b'\0' + image_file.convert('RGB').tobytes()).hexdigest()
            require(actual_rgb == rgb, 'changed_rgb_or_unsupported_rgb_hash_algorithm')
        ids.add(sid); images.add(image); rgbs.add(rgb); groups[sid] = group
    proof = manifest['exclusion_proof']
    require(proof.get('review_status') == 'approved' and proof.get('pool_sha256') == digest(rows),
            'reviewed_exclusion_proof_not_bound_to_pool')
    proof_path = Path(proof['artifact_path'])
    require(proof_path.is_file() and hashlib.sha256(proof_path.read_bytes()).hexdigest() == proof['sha256'],
            'exclusion_proof_artifact_missing_or_changed')
    seen, owners = set(), {}
    byid = {r['sample_id']: r for r in rows}
    rounds = manifest['rounds']
    require(bool(rounds), 'no_round_prepared')
    for number, batch in enumerate(rounds):
        require(len(batch) == 300 and len(set(batch)) == 300 and set(batch) <= ids, 'round_requires_300_pool_images')
        require(not seen.intersection(batch), 'image_crosses_rounds')
        require(all(sum(byid[s]['dataset'] == d for s in batch) == 150 for d in ('asa24', 'nutrition5k')),
                'round_requires_150_ASA_150_N5')
        for sid in batch:
            group = groups[sid]
            require(owners.setdefault(group, number) == number, 'complete_group_crosses_rounds')
        seen.update(batch)
    require(manifest.get('models') == MODELS and manifest.get('correction_applied') is False,
            'frozen_stage_models_and_no_correction_required')
    return {'status': 'passed', 'pool_unique_images': len(ids), 'rounds': len(rounds),
            'manifest_sha256': digest(manifest), 'new_api_calls': 0, 'rgb_hashes_verified': verify_images}


def seal_identities(rows, identities):
    expected = {(variant, row['sample_id']) for variant in VARIANTS for row in rows}
    require(set(identities) == expected, 'all_paired_blind_identities_required')
    return digest([{'variant': v, 'sample_id': s, 'identity_sha256': digest(identities[v, s])}
                   for v, s in sorted(expected)])


def execute_injected(rows, pipes, configs, catalog, label_loader, seal_writer, seal_reader, workers=1):
    """Offline/mocked adapter. Caller supplies frozen pipeline objects and durable seal I/O.

    No cache lookup by filename. Pipelines may share only exact-request caches.
    Full paired identities are persisted and read back BEFORE label_loader runs.
    No median/default fallback is created by this adapter.
    """
    require(len({row['sample_id'] for row in rows}) == len(rows), 'duplicate_input_image_id')
    require(type(workers) is int and 1 <= workers <= 32, 'workers_must_be_1_to32')
    tasks = [(variant, row) for variant in VARIANTS for row in rows]
    def identify(task):
        variant, row = task
        pipe = pipes[variant][row['dataset']]
        if row['dataset'] == 'asa24':
            value = pipe.identify_select(row['image_path'], 'asa24', configs['asa24'])
        else:
            value = pipe.identify(row['image_path'], configs['nutrition5k'])
        return (variant, row['sample_id']), value
    with ThreadPoolExecutor(max_workers=workers) as executor:
        identities = dict(executor.map(identify, tasks))
    sealed = seal_identities(rows, identities)
    envelope = [{'variant': v, 'sample_id': sid, 'identity': deepcopy(value)}
                for (v, sid), value in sorted(identities.items())]
    seal_writer({'sha256': sealed, 'identities': envelope})
    saved = seal_reader()
    restored = {(r['variant'], r['sample_id']): r['identity'] for r in saved['identities']}
    require(saved['sha256'] == sealed == seal_identities(rows, restored), 'blind_identity_seal_changed')
    labels = label_loader()  # only permitted after real read-back validation
    records = {v: [] for v in VARIANTS}
    def portion(task):
        variant, row = task
        sid, dataset = row['sample_id'], row['dataset']
        identity = restored[variant, sid]
        codes = identity.get('final_codes', [])
        record = dict(sample_id=sid, dataset=dataset, variant=variant,
                      identity_status=identity.get('status'), final_codes=deepcopy(codes),
                      pending_leaf_ids=deepcopy(identity.get('pending_leaf_ids', [])),
                      prediction=None, conditional=None, fallback=False)
        pipe = pipes[variant][dataset]
        if dataset == 'asa24':
            references = asa_references(labels[sid])
            require(row.get('reference_count', len(references)) == len(references), 'frozen_reference_count_mismatch')
            record['references'] = []
            for ref in references:
                code = ref['food_code']
                branch = dict(ref_id=ref['ref_id'], code=code, hit=code in codes,
                              prediction=None, fallback=False)
                if identity.get('status') == 'ok' and code in codes:
                    context = dict(code=code, options=deepcopy(catalog['asa_options'][code]))
                    result = pipe.portion(row['image_path'], identity, 'asa24', configs[dataset],
                                          conditional_context=context)
                    grams = result.get('continuous_grams') if result.get('status') == 'ok' else None
                    if type(grams) in (float, int) and math.isfinite(grams) and grams >= 0:
                        option = min(context['options'], key=lambda o: (abs(o['grams']-grams), o['grams'], o['option_id']))
                        branch['prediction'] = {'weight': option['grams'], **{k: option['grams'] * catalog['foods'][code]['density'][k]/100 for k in TARGETS[1:]}}
                branch['status'] = ('complete' if valid_vector(branch['prediction']) else
                    'needs_review' if identity.get('status') == 'needs_review' else
                    'failed' if identity.get('status') != 'ok' else 'not_hit' if not branch['hit'] else 'failed')
                record['references'].append(branch)
            record['hit'] = any(b['hit'] for b in record['references'])
        elif identity.get('status') == 'ok':
            result = pipe.portion(row['image_path'], identity, configs[dataset])
            if result.get('status') == 'ok' and valid_vector(result.get('prediction')):
                record['prediction'] = deepcopy(result['prediction'])
        if dataset == 'asa24':
            hit_branches = [b for b in record['references'] if b['hit']]
            record['status'] = ('needs_review' if identity.get('status') == 'needs_review' else
                'failed' if identity.get('status') != 'ok' else 'not_hit' if not hit_branches else
                'complete' if all(b['status'] == 'complete' for b in hit_branches) else 'failed')
        else:
            record['status'] = ('complete' if valid_vector(record['prediction']) else
                'needs_review' if identity.get('status') == 'needs_review' else 'failed')
        return variant, record
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for variant, record in executor.map(portion, tasks):
            records[variant].append(record)
    return records


def asa_references(label):
    refs = label.get('references')
    if refs is None:
        refs = [dict(ref_id='single', food_code=label['food_code'], truth=label['truth'])]
    require(isinstance(refs, list) and bool(refs), 'all_legal_reference_associations_required')
    require(all(isinstance(x.get('ref_id'), str) and x['ref_id'] for x in refs), 'stable_reference_id_required')
    require(len({x['ref_id'] for x in refs}) == len(refs), 'duplicate_reference_id')
    require(all(valid_vector(x.get('truth', x.get('targets'))) for x in refs), 'five_reference_targets_required')
    return [{**x, 'truth': x.get('truth', x.get('targets'))} for x in refs]


def error_summary(entries):
    """Primary historical association-row MAE plus additional equal-image MAE.

    entries = (sample_id, prediction, truth); never minimize over references.
    """
    errors = {}
    for sid, prediction, truth in entries:
        require(valid_vector(prediction) and valid_vector(truth), 'five_finite_targets_required_for_scoring')
        errors.setdefault(sid, []).append({k: abs(prediction[k]-truth[k]) for k in TARGETS})
    n = sum(map(len, errors.values()))
    return dict(scored_unique_images=len(errors), scored_reference_associations=n,
        association_row_mae={k: sum(e[k] for rows in errors.values() for e in rows)/n if n else None for k in TARGETS},
        equal_image_mae={k: sum(sum(e[k] for e in rows)/len(rows) for rows in errors.values())/len(errors)
                        if errors else None for k in TARGETS})


def report(records, labels, expected_rows):
    require(len({row['sample_id'] for row in expected_rows}) == len(expected_rows), 'duplicate_expected_input')
    result = {'correction_applied': False, 'datasets': {},
              'primary_metric': 'association_row_mae',
              'additional_metric': 'equal_image_mae_not_directly_comparable_to_historical_row_MAE'}
    for dataset in ('asa24', 'nutrition5k'):
        indexed = {v: {r['sample_id']: r for r in records[v] if r['dataset'] == dataset} for v in VARIANTS}
        require(all(len(indexed[v]) == sum(row['dataset'] == dataset for row in records[v]) for v in VARIANTS), 'duplicate_reporting_record')
        require(all(set(indexed[v]) == {r['sample_id'] for r in expected_rows if r['dataset'] == dataset} for v in VARIANTS), 'fixed_input_population_mismatch')
        require(set(indexed[VARIANTS[0]]) == set(indexed[VARIANTS[1]]), 'paired_population_mismatch')
        profiles, entries, hitkeys = {}, {}, {}
        for v in VARIANTS:
            rows = list(indexed[v].values())
            entries[v], hitkeys[v] = {}, set()
            branch_count = 0
            pending_hit = 0
            for row in rows:
                sid = row['sample_id']
                require(row.get('fallback') is False, 'historical_fallback_import_not_supported')
                if dataset == 'asa24':
                    refs = {ref['ref_id']: ref for ref in asa_references(labels[sid])}
                    branches = row['references']
                    require({b['ref_id'] for b in branches} == set(refs) and len(branches) == len(refs),
                            'frozen_reference_population_mismatch')
                    for branch in branches:
                        key = (sid, branch['ref_id'])
                        require(branch['code'] == refs[branch['ref_id']]['food_code'], 'conditional_prediction_code_mismatch')
                        require(type(branch.get('hit')) is bool and branch['hit'] == (branch['code'] in row['final_codes']), 'selected_code_hit_binding_mismatch')
                        require(branch.get('fallback') is False, 'historical_fallback_import_not_supported')
                        eligible = row['identity_status'] == 'ok' and branch['status'] == 'complete' and branch['hit']
                        require(branch.get('prediction') is None or (eligible and valid_vector(branch['prediction'])), 'noncomplete_branch_residual_prediction')
                        require(branch['status'] != 'complete' or (eligible and valid_vector(branch.get('prediction'))), 'complete_branch_prediction_required')
                        if branch['hit']:
                            hitkeys[v].add(key)
                            if valid_vector(branch['prediction']):
                                entries[v][key] = (sid, branch['prediction'], refs[branch['ref_id']]['truth'])
                            else:
                                pending_hit += 1
                    hit_branches = [b for b in branches if b['hit']]
                    expected_status = ('needs_review' if row['identity_status'] == 'needs_review' else
                        'failed' if row['identity_status'] != 'ok' else 'not_hit' if not hit_branches else
                        'complete' if all(b['status'] == 'complete' for b in hit_branches) else 'failed')
                    require(row['status'] == expected_status, 'ASA_image_completion_status_mismatch')
                    branch_count += len(branches)
                else:
                    key = (sid, 'single')
                    branch_count += 1
                    eligible = row['identity_status'] == 'ok' and row['status'] == 'complete'
                    require(row.get('prediction') is None or (eligible and valid_vector(row['prediction'])), 'noncomplete_N5_residual_prediction')
                    require(row['status'] != 'complete' or (eligible and valid_vector(row.get('prediction'))), 'complete_N5_prediction_required')
                    if eligible:
                        entries[v][key] = (sid, row['prediction'], labels[sid]['truth'])
            profiles[v] = dict(input_unique_images=len(rows), reference_association_count=branch_count,
                complete_image_count=sum(row['status'] == 'complete' for row in rows),
                empty_code_image_count=sum(bool(row.get('pending_leaf_ids')) for row in rows),
                failed_image_count=sum(row['status'] == 'failed' for row in rows),
                needs_review_image_count=sum(row['status'] == 'needs_review' for row in rows),
                explicit_fallback_count=sum(row['fallback'] for row in rows),
                selected_code_hit_images=len({s for s, ref in hitkeys[v]}),
                selected_code_hit_associations=len(hitkeys[v]),
                not_hit_reference_associations=branch_count-len(hitkeys[v]) if dataset == 'asa24' else None,
                pending_hit_associations=pending_hit,
                **error_summary(list(entries[v].values())))
        common_hit = set.intersection(*(hitkeys[v] for v in VARIANTS))
        common_complete = set.intersection(*(set(entries[v]) for v in VARIANTS))
        common_hit_complete = common_hit & common_complete
        result['datasets'][dataset] = dict(profiles=profiles,
            common_hit_reference_ids=sorted(common_hit), common_complete_reference_ids=sorted(common_complete),
            common_hit_complete_reference_ids=sorted(common_hit_complete),
            paired_common_complete={v: error_summary([entries[v][k] for k in sorted(common_complete)]) for v in VARIANTS},
            paired_common_hit_complete={v: error_summary([entries[v][k] for k in sorted(common_hit_complete)]) for v in VARIANTS})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--allow-paid', action='store_true')
    parser.add_argument('--authorization', type=Path)
    args = parser.parse_args()
    try:
        value = preflight(json.loads(args.manifest.read_text()))
        if args.allow_paid:
            raise ValueError('paid_backend_not_connected: concrete authorization and reviewed pool adapter still required')
        print(json.dumps(value, ensure_ascii=False, indent=2))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({'status': 'blocked', 'reason': str(exc), 'new_api_calls': 0}, ensure_ascii=False))
        raise SystemExit(2)


if __name__ == '__main__':
    main()

"""One blind identity and one N5 prediction per image; ASA portion-only pairing."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import importlib.util
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('portion_first_old_pair_contract', HERE.parent / 'c_minimal_nullable_20261105/paired_runner.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
digest, require, valid_vector = old.digest, old.require, old.valid_vector
TARGETS, MODELS = old.TARGETS, old.MODELS
asa_references = old.asa_references

def error_summary(entries):
    result = old.error_summary(entries)
    n = len(entries)
    means = {k:sum(t[k] for _,_,t in entries)/n if n else None for k in TARGETS}
    by_image = {}
    for sid,_,truth in entries:
        by_image.setdefault(sid,[]).append(truth)
    equal_means = {k:sum(sum(t[k] for t in ts)/len(ts) for ts in by_image.values())/len(by_image) if by_image else None for k in TARGETS}
    result.update(ground_truth_mean=means, equal_image_ground_truth_mean=equal_means,
        association_row_per_unit={k:result['association_row_mae'][k]/means[k] if means[k] else None for k in TARGETS},
        equal_image_per_unit={k:result['equal_image_mae'][k]/equal_means[k] if equal_means[k] else None for k in TARGETS})
    return result
VARIANTS = ('continuous_nullable', 'portion_first_nullable')
PORTION_MODES = dict(zip(VARIANTS, ('continuous_option', 'portion_first')))
MANIFEST_SHA256 = '4244eac5e12e855e41eee842454352f981fd5deca7326a006508281059a9252d'


def preflight(manifest, *, verify_images=True):
    require(digest(manifest) == MANIFEST_SHA256, 'original_3000_manifest_required')
    result = old.preflight(manifest, verify_images=verify_images)
    rows = {r['sample_id']: r for r in manifest['pool']}
    first, second = ([rows[s] for s in manifest['rounds'][i]] for i in (0, 1))
    require(not {r['sample_id'] for r in first} & {r['sample_id'] for r in second}, 'round01_image_overlap')
    require(not {(r['dataset'], r['group_id']) for r in first} & {(r['dataset'], r['group_id']) for r in second}, 'round01_complete_group_overlap')
    result.update(round_number=2, unique_images=300, asa_reference_associations=sum(r['reference_count'] for r in second if r['dataset']=='asa24'))
    return result


def seal_identities(rows, identities):
    require(set(identities) == {r['sample_id'] for r in rows}, 'all_shared_blind_identities_required')
    return digest([dict(sample_id=s, identity_sha256=digest(identities[s])) for s in sorted(identities)])


def n5_prediction(row, identity, result, catalog):
    """Reconstruct known-only sum from sealed leaves; never turn a failure into zero."""
    require(identity.get('input_image_sha256') == row['image_sha256'], 'N5_identity_image_binding')
    state = result.get('status')
    if state not in ('ok', 'partial', 'empty'):
        require(result.get('prediction') is None, 'failed_N5_has_prediction')
        return 'failed', None
    leaves = identity.get('leaves', [])
    require(bool(leaves) and len({x['component_id'] for x in leaves}) == len(leaves), 'N5_identity_leaves')
    known = [x for x in leaves if x['primary_code'] is not None]
    pending = [x['component_id'] for x in leaves if x['primary_code'] is None]
    require(pending == identity.get('pending_leaf_ids', []), 'N5_pending_identity_binding')
    require(identity.get('status') == ('needs_review' if pending else 'ok'), 'N5_identity_state')
    require(state == ('empty' if not known else 'partial' if pending else 'ok'), 'N5_portion_state')
    if pending:
        require(result.get('identity') == identity and result.get('pending_leaf_ids') == pending, 'N5_partial_identity_binding')
        require(result.get('prediction_scope') == 'known_items_only' and result.get('complete') is False, 'N5_partial_scope')
    if state == 'empty':
        require(result.get('all_codes_empty') is True and result.get('physical_total_grams') is None, 'N5_empty_contract')
        masses = {}
    elif state == 'partial':
        require(result.get('all_codes_empty') is False, 'N5_partial_all_empty')
        masses = result.get('leaf_grams', {})
        require(set(masses) == {x['component_id'] for x in leaves}, 'N5_partial_all_physical_masses')
    else:
        portions = result.get('portions', [])
        require(len(portions) == len(leaves) and len({x['component_id'] for x in portions}) == len(leaves), 'N5_complete_portions')
        masses = {x['component_id']: x['grams'] for x in portions}
        require(set(masses) == {x['component_id'] for x in leaves}, 'N5_complete_mass_binding')
    require(all(type(g) in (int,float) and math.isfinite(g) and g > 0 for g in masses.values()), 'N5_positive_physical_masses')
    totals = dict.fromkeys(TARGETS, 0.)
    for leaf in known:
        codes = [leaf['primary_code']] + leaf.get('alternative_codes', [])
        require(1 <= len(codes) <= 3 and len(set(codes)) == len(codes) and all(c in catalog['foods'] for c in codes), 'N5_mean_codes')
        grams = masses[leaf['component_id']]
        totals['weight'] += grams
        for k in TARGETS[1:]:
            totals[k] += grams * sum(catalog['foods'][c]['density'][k] for c in codes) / len(codes) / 100
    require(valid_vector(result.get('prediction')) and all(math.isclose(totals[k], result['prediction'][k], rel_tol=1e-12, abs_tol=1e-10) for k in TARGETS), 'N5_reconstructed_prediction_mismatch')
    return ('complete' if state == 'ok' else state), totals


def asa_option(result, options, mode):
    if result.get('status') != 'ok':
        require(result.get('prediction') is None, 'failed_ASA_has_prediction')
        return None
    require(bool(options) and len({o['option_id'] for o in options}) == len(options), 'ASA_legal_options')
    if mode == 'continuous_option':
        grams = result.get('continuous_grams')
        require(type(grams) in (int,float) and math.isfinite(grams) and grams >= 0, 'ASA_continuous_grams')
        return min(options, key=lambda o: (abs(o['grams']-grams), o['grams'], o['option_id']))
    require(mode == 'portion_first', 'ASA_unknown_mode')
    matches = [o for o in options if o['option_id'] == result.get('option_id')]
    require(len(matches) == 1, 'ASA_illegal_option_id')
    return matches[0]


def execute_injected(rows, pipes, configs, catalog, label_loader, seal_writer, seal_reader, workers=1):
    require(len({r['sample_id'] for r in rows}) == len(rows), 'duplicate_input')
    require(type(workers) is int and 1 <= workers <= 32, 'workers_must_be_1_to32')
    def identify(row):
        pipe = pipes[row['dataset']]
        value = (pipe.identify_select(row['image_path'], 'asa24', configs['asa24']) if row['dataset']=='asa24'
                 else pipe.identify(row['image_path'], configs['nutrition5k']))
        if row['dataset']=='asa24':
            require(value.get('input_image_sha256', row['image_sha256']) == row['image_sha256'], 'ASA_existing_identity_hash_mismatch')
            value = dict(deepcopy(value), input_image_sha256=row['image_sha256'])
        return row['sample_id'], value
    with ThreadPoolExecutor(max_workers=workers) as pool:
        identities = dict(pool.map(identify, rows))
    sealed = seal_identities(rows, identities)
    seal_writer(dict(sha256=sealed, identities=[dict(sample_id=s, identity=deepcopy(i)) for s,i in sorted(identities.items())]))
    saved = seal_reader()
    require(len(saved['identities']) == len(rows), 'shared_identity_envelope_count')
    restored = {x['sample_id']: x['identity'] for x in saved['identities']}
    require(saved['sha256'] == sealed == seal_identities(rows, restored), 'blind_identity_seal_changed')
    labels = label_loader()
    def portions(row):
        sid, dataset = row['sample_id'], row['dataset']
        identity, pipe = restored[sid], pipes[dataset]
        base = dict(sample_id=sid, dataset=dataset, identity_status=identity.get('status'),
                    identity_sha256=digest(identity), final_codes=deepcopy(identity.get('final_codes', [])),
                    pending_leaf_ids=deepcopy(identity.get('pending_leaf_ids', [])), fallback=False, prediction=None)
        if dataset == 'nutrition5k':
            if identity.get('status') in ('ok','needs_review'):
                value = pipe.portion(row['image_path'], identity, configs[dataset])
                state, prediction = n5_prediction(row, identity, value, catalog)
            else:
                value, state, prediction = dict(status='failed', prediction=None), 'failed', None
            base.update(status=state, prediction=prediction, shared_n5_result_sha256=digest(value),
                        shared_n5_prediction=True)
            return {v: dict(deepcopy(base), variant=v) for v in VARIANTS}
        if identity.get('status') in ('ok', 'needs_review'):
            require(identity.get('input_image_sha256') == row['image_sha256'], 'ASA_identity_image_binding')
        records = {}
        refs = asa_references(labels[sid])
        require(row['reference_count'] == len(refs), 'ASA_reference_count')
        for variant in VARIANTS:
            record = dict(deepcopy(base), variant=variant, references=[])
            for ref in refs:
                code = ref['food_code']
                branch = dict(ref_id=ref['ref_id'], code=code, hit=code in base['final_codes'], prediction=None, fallback=False)
                if identity.get('status') == 'ok' and branch['hit']:
                    context = dict(code=code, options=deepcopy(catalog['asa_options'][code]))
                    config = dict(configs['asa24'], portion_mode=PORTION_MODES[variant])
                    value = pipe.portion(row['image_path'], identity, 'asa24', config, conditional_context=context)
                    option = asa_option(value, context['options'], PORTION_MODES[variant])
                    if option is not None:
                        grams = option['grams']
                        branch.update(option_id=option['option_id'], legal_grams=grams,
                                      prediction={'weight':grams, **{k:grams*catalog['foods'][code]['density'][k]/100 for k in TARGETS[1:]}})
                branch['status'] = ('complete' if valid_vector(branch['prediction']) else 'needs_review' if identity.get('status')=='needs_review' else 'failed' if identity.get('status')!='ok' else 'not_hit' if not branch['hit'] else 'failed')
                record['references'].append(branch)
            hit = [b for b in record['references'] if b['hit']]
            record['status'] = ('needs_review' if identity.get('status')=='needs_review' else 'failed' if identity.get('status')!='ok' else 'not_hit' if not hit else 'complete' if all(b['status']=='complete' for b in hit) else 'failed')
            records[variant] = record
        return records
    output = {v:[] for v in VARIANTS}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for records in pool.map(portions, rows):
            for v in VARIANTS:
                output[v].append(records[v])
    return output


def report(records, labels, expected_rows):
    """Fixed input denominator, explicit failures, association and equal-image MAE."""
    result = dict(correction_applied=False, datasets={}, primary_metric='association_row_mae')
    expected_by_id = {r['sample_id']:r for r in expected_rows}
    expected = set(expected_by_id)
    require(len(expected)==len(expected_rows), 'duplicate_expected_rows')
    indexed = {}
    for v in VARIANTS:
        indexed[v] = {r['sample_id']:r for r in records[v]}
        require(len(indexed[v])==len(records[v]) and set(indexed[v])==expected, 'fixed_input_population_mismatch')
        require(all(r['dataset']==expected_by_id[r['sample_id']]['dataset'] and r['variant']==v for r in records[v]), 'record_dataset_or_variant_binding')
    for sid in expected:
        a,b = [indexed[v][sid] for v in VARIANTS]
        require(all(a[k]==b[k] for k in ('identity_sha256','identity_status','final_codes','pending_leaf_ids')), 'paired_identity_mismatch')
        if a['dataset']=='nutrition5k':
            require(all(a[k]==b[k] for k in ('status','prediction','shared_n5_result_sha256')), 'N5_shared_result_mismatch')
    for dataset in ('asa24','nutrition5k'):
        profiles, entries = {}, {}
        for v in VARIANTS:
            rows = [r for r in records[v] if r['dataset']==dataset]
            entries[v] = {}
            hit_keys, hit_images, pending_hit = set(), set(), 0
            branch_count = 0
            for row in rows:
                require(row['fallback'] is False, 'fallback_forbidden')
                sid = row['sample_id']
                if dataset=='nutrition5k':
                    branch_count += 1
                    eligible = row['status'] in ('complete','partial','empty')
                    require((valid_vector(row.get('prediction')) if eligible else row.get('prediction') is None), 'N5_prediction_state')
                    require(row['identity_status']==('ok' if row['status']=='complete' else 'needs_review') if eligible else True, 'N5_scoring_identity_state')
                    if row['status']=='empty':
                        require(all(x==0 for x in row['prediction'].values()), 'empty_must_be_zero')
                    if eligible:
                        entries[v][(sid,'single')] = (sid,row['prediction'],labels[sid]['truth'])
                else:
                    refs = {r['ref_id']:r for r in asa_references(labels[sid])}
                    branch_count += len(refs)
                    require(expected_by_id[sid]['reference_count']==len(refs), 'frozen_reference_count_mismatch')
                    require(len(row['references'])==len(refs) and {b['ref_id'] for b in row['references']}==set(refs), 'ASA_frozen_reference_population')
                    for branch in row['references']:
                        ref = refs[branch['ref_id']]
                        require(branch['code']==ref['food_code'] and branch['hit']==(branch['code'] in row['final_codes']), 'ASA_selected_code_binding')
                        eligible = row['identity_status']=='ok' and branch['hit'] and branch['status']=='complete'
                        require((valid_vector(branch.get('prediction')) if eligible else branch.get('prediction') is None), 'ASA_prediction_state')
                        if branch['hit']:
                            hit_keys.add((sid,branch['ref_id'])); hit_images.add(sid)
                            pending_hit += int(not eligible)
                        if eligible:
                            entries[v][(sid,branch['ref_id'])] = (sid,branch['prediction'],ref['truth'])
                    hits = [b for b in row['references'] if b['hit']]
                    expected_state = ('needs_review' if row['identity_status']=='needs_review' else 'failed' if row['identity_status']!='ok' else 'not_hit' if not hits else 'complete' if all(b['status']=='complete' for b in hits) else 'failed')
                    require(row['status']==expected_state, 'ASA_completion_state_binding')
            summary = error_summary(list(entries[v].values()))
            profiles[v] = dict(input_unique_images=len(rows), reference_association_count=branch_count,
                selected_code_hit_images=len(hit_images), selected_code_hit_associations=len(hit_keys),
                pending_hit_associations=pending_hit, not_hit_reference_associations=branch_count-len(hit_keys) if dataset=='asa24' else None,
                selected_code_hit_image_rate=len(hit_images)/len(rows) if dataset=='asa24' and rows else None,
                selected_code_hit_association_rate=len(hit_keys)/branch_count if dataset=='asa24' and branch_count else None,
                explicit_fallback_count=0, calculated_image_count=summary['scored_unique_images'],
                calculation_rate=summary['scored_unique_images']/len(rows) if rows else None,
                complete_image_count=sum(r['status']=='complete' for r in rows),
                partial_image_count=sum(r['status']=='partial' for r in rows), empty_image_count=sum(r['status']=='empty' for r in rows),
                failed_image_count=sum(r['status']=='failed' for r in rows), needs_review_image_count=sum(r['status']=='needs_review' for r in rows),
                **summary)
            if dataset=='nutrition5k':
                profiles[v]['fixed_population_mae'] = summary['association_row_mae'] if len(entries[v])==len(rows) else None
        common = set.intersection(*(set(entries[v]) for v in VARIANTS))
        result['datasets'][dataset] = dict(profiles=profiles, common_complete_reference_ids=sorted(common),
            paired_common_calculated={v:error_summary([entries[v][k] for k in sorted(common)]) for v in VARIANTS})
    return result

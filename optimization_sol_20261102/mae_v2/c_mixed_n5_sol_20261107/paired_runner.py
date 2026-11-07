"""One blind identity; paired N5 portion models, one shared ASA option prediction."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import importlib.util
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('mixed_c_original_pair_contract', HERE.parent / 'c_minimal_nullable_20261105/paired_runner.py')
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
VARIANTS = ('gemini_portion', 'sol_portion')
N5_PORTION_MODELS = dict(zip(VARIANTS, ('gemini-3.8-flash', 'gpt-6.1-sol')))
ASA_PORTION_MODE = 'continuous_option'
MANIFEST_SHA256 = '4244eac5e12e855e41eee842454352f981fd5deca7326a006508281059a9252d'


def preflight(manifest, *, verify_images=True):
    require(digest(manifest) == MANIFEST_SHA256, 'original_3000_manifest_required')
    result = old.preflight(manifest, verify_images=verify_images)
    rows = {r['sample_id']: r for r in manifest['pool']}
    prior = [rows[s] for i in range(6) for s in manifest['rounds'][i]]
    current = [rows[s] for s in manifest['rounds'][6]]
    require(len(current)==300 and sum(r['dataset']=='asa24' for r in current)==150 and sum(r['dataset']=='nutrition5k' for r in current)==150, 'round07_fixed_300')
    require(not {r['sample_id'] for r in prior} & {r['sample_id'] for r in current}, 'prior_round_image_overlap')
    require(not {r['image_sha256'] for r in prior} & {r['image_sha256'] for r in current}, 'prior_round_image_hash_overlap')
    require(not {(r['dataset'], r['group_id']) for r in prior} & {(r['dataset'], r['group_id']) for r in current}, 'prior_round_complete_group_overlap')
    result.update(round_number=7, unique_images=300, asa_reference_associations=sum(r['reference_count'] for r in current if r['dataset']=='asa24'))
    return result


def seal_identities(rows, identities):
    require(set(identities) == {r['sample_id'] for r in rows}, 'all_shared_blind_identities_required')
    return digest([dict(sample_id=s, identity_sha256=digest(identities[s])) for s in sorted(identities)])


def n5_prediction(row, identity, result, catalog):
    """Check original complete C mass and mean code densities against sealed identity."""
    require(identity.get('input_image_sha256') == row['image_sha256'], 'N5_identity_image_binding')
    if result.get('status') != 'ok':
        require(result.get('prediction') is None, 'failed_N5_has_prediction')
        return None
    leaves, portions = identity['leaves'], result['portions']
    masses = {item['component_id']: item['grams'] for item in portions}
    require(len(leaves) == len(portions) == len(masses) and masses and
            set(masses) == {leaf['component_id'] for leaf in leaves}, 'N5_complete_mass_binding')
    require(all(type(g) in (int, float) and math.isfinite(g) and g > 0 for g in masses.values()),
            'N5_positive_physical_masses')
    totals = dict.fromkeys(TARGETS, 0.)
    for leaf in leaves:
        codes = [leaf['primary_code']] + leaf['alternative_codes']
        require(1 <= len(codes) <= 3 and len(set(codes)) == len(codes) and
                all(code in catalog['foods'] for code in codes), 'N5_mean_codes')
        grams = masses[leaf['component_id']]
        totals['weight'] += grams
        for k in TARGETS[1:]:
            totals[k] += grams * (sum(catalog['foods'][code]['density'][k] for code in codes) / len(codes)) / 100
    require(valid_vector(result.get('prediction')) and
            all(math.isclose(totals[k], result['prediction'][k], rel_tol=1e-12, abs_tol=1e-10)
                for k in TARGETS), 'N5_reconstructed_prediction_mismatch')
    return result['prediction']


def asa_option(result, options):
    if result.get('status') != 'ok':
        require(result.get('prediction') is None, 'failed_ASA_has_prediction')
        return None
    grams = result.get('continuous_grams')
    require(type(grams) in (int, float) and math.isfinite(grams) and grams >= 0, 'ASA_continuous_grams')
    option = min(options, key=lambda o: (abs(o['grams']-grams), o['grams'], o['option_id']))
    require(result.get('option_id') == option['option_id'], 'ASA_original_legal_projection_changed')
    return option


def execute_injected(rows, pipes, configs, catalog, label_loader, seal_writer, seal_reader, workers=1):
    require(len({r['sample_id'] for r in rows}) == len(rows), 'duplicate_input')
    require(type(workers) is int and 1 <= workers <= 32, 'workers_must_be_1_to32')
    def identify(row):
        pipe = pipes[row['dataset']]
        value = (pipe.identify_select(row['image_path'], 'asa24', configs['asa24']) if row['dataset']=='asa24'
                 else pipe.identify(row['image_path'], configs['nutrition5k']))
        if row['dataset']=='asa24':
            require(value.get('input_image_sha256', row['image_sha256']) == row['image_sha256'],
                    'ASA_existing_identity_hash_mismatch')
            value = dict(deepcopy(value), input_image_sha256=row['image_sha256'])
        require(value.get('status') in ('ok', 'failed'), 'non_original_identity_status')
        if value.get('status') == 'ok':
            require(value.get('input_image_sha256') == row['image_sha256'], 'identity_image_binding')
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
                    fallback=False, prediction=None)
        if dataset == 'nutrition5k':
            results = {}
            for variant in VARIANTS:
                model = N5_PORTION_MODELS[variant]
                if identity.get('status') == 'ok':
                    value = pipe.portion(row['image_path'], identity, configs[dataset], portion_model=model)
                    prediction = n5_prediction(row, identity, value, catalog)
                else:
                    value = dict(status='failed', failure=identity.get('failure'), prediction=None)
                    prediction = None
                results[variant] = dict(deepcopy(base), variant=variant,
                    status='complete' if prediction is not None else 'failed',
                    failure=value.get('failure'), prediction=prediction,
                    n5_result_sha256=digest(value), n5_portion_model=model)
            return results
        if identity.get('status') == 'ok':
            require(identity.get('input_image_sha256') == row['image_sha256'], 'ASA_identity_image_binding')
        records = {}
        refs = asa_references(labels[sid])
        require(row['reference_count'] == len(refs), 'ASA_reference_count')
        shared_asa = {}
        for variant in VARIANTS:
            record = dict(deepcopy(base), variant=variant, references=[])
            for ref in refs:
                code = ref['food_code']
                branch = dict(ref_id=ref['ref_id'], code=code, hit=code in base['final_codes'], prediction=None, fallback=False)
                if identity.get('status') == 'ok' and branch['hit']:
                    context = dict(code=code, options=deepcopy(catalog['asa_options'][code]))
                    config = configs['asa24']
                    if code not in shared_asa:
                        shared_asa[code] = pipe.portion(row['image_path'], identity, 'asa24', config, conditional_context=context)
                    value = shared_asa[code]
                    option = asa_option(value, context['options'])
                    if option is not None:
                        grams = option['grams']
                        branch.update(option_id=option['option_id'], legal_grams=grams,
                                      prediction={'weight':grams, **{k:grams*catalog['foods'][code]['density'][k]/100 for k in TARGETS[1:]}})
                branch['status'] = ('complete' if valid_vector(branch['prediction']) else 'failed' if identity.get('status')!='ok' else 'not_hit' if not branch['hit'] else 'failed')
                record['references'].append(branch)
            hit = [b for b in record['references'] if b['hit']]
            record['status'] = ('failed' if identity.get('status')!='ok' else 'not_hit' if not hit else 'complete' if all(b['status']=='complete' for b in hit) else 'failed')
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
        require(all(a[k]==b[k] for k in ('identity_sha256','identity_status','final_codes')), 'paired_identity_mismatch')
        if a['dataset']=='asa24':
            require(a['references']==b['references'] and a['status']==b['status'], 'ASA_shared_result_mismatch')
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
                    require(row.get('n5_portion_model')==N5_PORTION_MODELS[v], 'N5_variant_portion_model_binding')
                    branch_count += 1
                    require(row['status'] in ('complete', 'failed'), 'N5_prediction_status')
                    eligible = row['status'] == 'complete'
                    require((valid_vector(row.get('prediction')) if eligible else row.get('prediction') is None), 'N5_prediction_state')
                    require(not eligible or row['identity_status'] == 'ok', 'N5_scoring_identity_state')
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
                    expected_state = ('failed' if row['identity_status']!='ok' else 'not_hit' if not hits else 'complete' if all(b['status']=='complete' for b in hits) else 'failed')
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
                failed_image_count=sum(r['status']=='failed' for r in rows),
                **summary)
            if dataset=='nutrition5k':
                profiles[v]['fixed_population_mae'] = summary['association_row_mae'] if len(entries[v])==len(rows) else None
        common = set.intersection(*(set(entries[v]) for v in VARIANTS))
        result['datasets'][dataset] = dict(profiles=profiles, calculated_reference_ids={v:sorted(entries[v]) for v in VARIANTS}, common_complete_reference_ids=sorted(common),
            paired_common_calculated={v:error_summary([entries[v][k] for k in sorted(common)]) for v in VARIANTS})
    return result

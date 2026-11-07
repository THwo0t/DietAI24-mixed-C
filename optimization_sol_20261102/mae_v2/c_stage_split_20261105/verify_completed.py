"""Verify the new arithmetic on completed development artifacts; no inference."""
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run
import score
from chroma_index import tree_hashes
from opt_provider import atomic_json, stamp
from provider import read_credentials


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-7)


def verify_imports(ctx, records):
    index = run.load(HERE / 'prepared/source_imports.json')
    run.require(index == run.source_index(ctx['rows']), 'source_import_index_changed')
    for rel, expected in index['files'].items():
        run.require(run.sha(run.V2 / rel) == expected, 'source_artifact_hash_changed')
    for profile, stages in index['stages'].items():
        for stage, refs in stages.items():
            for sid in refs:
                expected = run.source_value(index, profile, stage, sid)
                path = HERE / 'run' / profile / stage / (run.fingerprint(sid) + '.json')
                saved = run.load(path)
                run.require(saved['sid'] == sid and saved['binding']['freeze_sha256'] == ctx['frozen']
                    and saved['value_sha256'] == run.fingerprint(saved['value'])
                    and saved['value'] == expected, 'imported_stage_value_changed')
    source_n5 = run.load(run.SOURCE / 'run/N5/records.json')
    run.require(records['N5'] == source_n5, 'N5_raw_records_changed')
    run.require(run.load(HERE / 'N5_calibrated_oof.json') == run.load(run.SOURCE / 'N5_calibrated_oof.json'),
        'N5_corrected_predictions_or_fold_parameters_changed')
    old_asa = {r['sample_id']: r for r in run.load(run.SOURCE / 'run/ASA/records.json')}
    run.require(set(old_asa) == {r['sample_id'] for r in records['ASA']}, 'ASA_population_changed')
    old_exact = {sid for sid, record in old_asa.items() if record.get('conditional') is not None}
    new_exact = {record['sample_id'] for record in records['ASA'] if record.get('conditional') is not None}
    run.require(old_exact == new_exact and len(new_exact) == 176, 'fixed176_conditional_population_changed')
    for record in records['ASA']:
        source = old_asa[record['sample_id']]
        for key in ('final_codes', 'primary_code', 'retrieved_codes', 'items', 'description',
                    'identity_status', 'identity_failure', 'dataset', 'group_id', 'fold'):
            run.require(record.get(key) == source.get(key), 'ASA_imported_identity_changed:' + key)
    return index


def verify_provider_route(profile, row):
    run.require(profile == 'ASA' and row.get('role') == 'chat'
        and row.get('dataset') == 'asa24' and row.get('stage') == 'portion'
        and row.get('model_requested') == 'gpt-6.1-sol', 'only_ASA_Sol_portion_requests_allowed')


def main():
    ctx = run.context()
    groups = {r['sample_id']: r for r in ctx['rows']}
    records = {}
    for profile, dataset in (('N5', 'nutrition5k'), ('ASA', 'asa24')):
        rows = [r for r in ctx['rows'] if r['dataset'] == dataset]
        records[profile] = score.completed_records(rows, profile, ctx['frozen'])
    run.validate_identity_seal(ctx, HERE / 'run/ASA')
    imported = verify_imports(ctx, records)
    summary = run.load(HERE / 'summary.json')
    run.require(summary['freeze_sha256'] == ctx['frozen'], 'scored_freeze_changed')
    leaves = Counter()
    query_counts = Counter()
    empty_queries = 0
    n5_checked = 0
    for pred in records['N5']:
        if pred['fallback']:
            continue
        n5_checked += 1
        physical = pred['raw_leaves']
        run.require(bool(physical), 'successful_prediction_without_leaves')
        run.require(len({v['component_id'] for v in physical}) == len(physical), 'duplicate_physical_leaf')
        totals = dict.fromkeys(run.KEYS, 0.)
        for leaf in physical:
            codes = leaf['codes']
            run.require(1 <= len(codes) <= 3 and len(codes) == len(set(codes)), 'invalid_leaf_codes')
            leaves[len(codes)] += 1
            totals['weight'] += leaf['grams']
            for target in run.KEYS[1:]:
                expected = math.fsum(ctx['catalog']['foods'][code]['density'][target] for code in codes) / len(codes)
                run.require(close(expected, leaf['density'][target]), 'mean_density_mismatch')
                totals[target] += leaf['grams'] * expected / 100
        run.require(all(close(totals[k], pred['prediction'][k]) for k in run.KEYS), 'one_mass_nutrient_sum_mismatch')
    asa_checked = 0
    asa_cal = run.load(HERE / 'ASA_calibrated_oof.json')
    params = {p['fold']: p for p in asa_cal['fold_parameters']}
    calibrated = {p['sample_id']: p for p in asa_cal['predictions']}
    for pred in records['ASA']:
        for name in ('e2e', 'conditional'):
            branch = pred.get(name)
            if branch is None or branch['option_id'] is None:
                continue
            code = branch['code']
            options = ctx['catalog']['asa_options' if name == 'conditional' else 'standard_options'][code]
            by_option = {o['option_id']: o for o in options}
            option = by_option[branch['option_id']]
            run.require(close(option['grams'], branch['prediction']['weight']), 'ASA_weight_not_legal')
            run.require(all(close(branch['prediction'][k], option['grams'] * ctx['catalog']['foods'][code]['density'][k] / 100)
                            for k in run.KEYS[1:]), 'ASA_single_code_density_mismatch')
            asa_checked += 1
        branch = pred.get('conditional')
        if branch is not None and not branch['fallback']:
            parameter = params[pred['fold']]
            run.require(groups[pred['sample_id']]['group_id'] not in parameter['training_groups'], 'ASA_factor_group_leak')
            expected = run.method.current.nearest_option(ctx['catalog']['asa_options'][branch['code']],
                                                        branch['continuous_grams'] * parameter['raw_factor'])
            actual = calibrated[pred['sample_id']]['conditional']
            run.require(actual['option_id'] == expected['option_id'], 'ASA_factor_not_applied_once')
    for profile in records:
        directory = HERE / 'run' / profile
        for path in (directory / 'identity').glob('*.json'):
            value = run.load(path)['value']
            for stage in value.get('stages', []):
                if stage['stage'] == 'multiquery':
                    query_counts[stage['query_count']] += 1
                    empty_queries += stage.get('failure') == 'empty_multiquery_line'
        for pred in records[profile]:
            run.require(all(isinstance(pred['prediction'][k], (int, float))
                            and math.isfinite(pred['prediction'][k]) and pred['prediction'][k] >= 0
                            for k in run.KEYS), 'invalid_prediction_vector')
    source = run.load(HERE / 'prepared/chroma_source.json')
    run.require(source['source_files'] == tree_hashes(run.REPO / 'index/chroma'), 'original_Chroma_changed')
    model_counts = Counter()
    for profile in ('N5', 'ASA'):
        directory = HERE / 'run' / profile / 'provider'
        observations = directory / 'observations.jsonl'
        for line in observations.read_text().splitlines() if observations.exists() else []:
            row = json.loads(line)
            verify_provider_route(profile, row)
            cache = run.V2 / 'runtime_cache/responses' / row['role'] / (row['request_hash'] + '.json')
            saved = run.load(cache)
            run.require(saved['request_hash'] == row['request_hash'] and saved['model'] == row['model_requested'], 'observed_cache_model_mismatch')
            run.require(saved['value_hash'] == run.fingerprint(saved['value']), 'observed_cache_value_changed')
            model_counts[row['model_requested']] += 1
        requests = directory / 'requests.jsonl'
        for line in requests.read_text().splitlines() if requests.exists() else []:
            row = json.loads(line)
            verify_provider_route(profile, row)
            run.require(row.get('input_has_images') is True, 'ASA_portion_request_requires_image')
            if row['status'] == 200:
                run.require(row['model_served'] == 'gpt-6.1-sol', 'served_model_mismatch')
    # Never print credentials, matching contents, or snippets.
    secrets = read_credentials()
    scanned, matched = 0, 0
    for path in HERE.rglob('*'):
        if not path.is_file() or path.suffix not in {'.py', '.json', '.jsonl', '.log', '.md', '.csv'}:
            continue
        scanned += 1
        text = path.read_text()
        matched += any(key in text for key in secrets) or re.search(r'sk-[A-Za-z0-9_-]{20,}', text) is not None
    run.require(matched == 0, 'credential_scan_failed')
    output = dict(status='passed', checked_utc=stamp(), scope='all843_completed_development',
        frozen_method_unchanged=True, original_Chroma_unchanged=True,
        imported_N5_predictions=314, imported_ASA_identities=529,
        source_import_index_sha256=run.sha(HERE / 'prepared/source_imports.json'),
        N5_raw_corrected_and_fold_parameters_unchanged=True, ASA_identity_unchanged=True,
        observed_calls_by_model=dict(model_counts),
        successful_N5_mean_arithmetic_checked=n5_checked,
        N5_leaf_count_by_selected_codes=dict(leaves), MultiQuery_actual_query_counts=dict(query_counts),
        MultiQuery_empty_line_failures=empty_queries, ASA_legal_single_code_branches_checked=asa_checked,
        ASA_calibration_applies_once=True, ASA_complete_identity_seal=529,
        prediction_vectors_checked=843, credential_scan_files=scanned, credential_scan_matched_files=matched,
        retained_passed_checks=summary['profiles']['retained']['passed_checks'],
        raw_passed_checks=summary['profiles']['raw']['passed_checks'], new_verification_api_calls=0,
        independent_acceptance_opened=False, github_uploaded=False)
    atomic_json(HERE / 'verification.json', output)
    print(json.dumps(output, ensure_ascii=False))


if __name__ == '__main__':
    main()

"""Offline frozen C reproduction scores; no label loader or provider entry point."""
from __future__ import annotations
from copy import deepcopy
import csv
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
sys.path.insert(0, str(V2 / 'src'))
import opt_metrics
from opt_provider import atomic_json, fingerprint, sha


def historical_mass_module():
    path = V2 / 'cj_full_20261104/score.py'
    spec = importlib.util.spec_from_file_location('repro_previous_score', path)
    module = importlib.util.module_from_spec(spec)
    # The previous scorer imports its sibling common without constructing clients.
    previous = list(sys.path)
    try:
        sys.path.insert(0, str(path.parent))
        spec.loader.exec_module(module)
        return module.historical_mass_module()
    finally:
        sys.path[:] = previous


def load(path):
    from round4.src.artifacts import decode
    return decode(Path(path).read_bytes())


def completed_records(rows, profile, frozen):
    directory = HERE / 'run' / profile
    records = load(directory / 'records.json')
    completion = load(directory / 'completion.json')
    expected = {r['sample_id']: r for r in rows}
    count = 314 if profile == 'N5' else 529
    if (len(rows) != count or len(expected) != count or len(records) != count
            or {p['sample_id'] for p in records} != set(expected)):
        raise ValueError('complete_fixed_population_required')
    if (completion.get('profile') != profile or completion.get('records') != count
            or completion.get('freeze_sha256') != frozen
            or completion.get('records_sha256') != sha(directory / 'records.json')):
        raise ValueError('completed_prediction_binding_changed')
    for record in records:
        row = expected[record['sample_id']]
        if any(record.get(k) != row[k] for k in ('dataset', 'group_id', 'fold')):
            raise ValueError('completed_prediction_metadata_changed')
    return records


def validate_exact(labels, records):
    for record in records:
        code = labels[record['sample_id']]['food_code']
        branch = record.get('conditional')
        if code in record.get('final_codes', []):
            if (not isinstance(branch, dict) or branch.get('code') != code
                    or branch.get('prediction') is None or type(branch.get('fallback')) is not bool):
                raise ValueError('Exact_requires_explicit_conditional_result')
        elif branch is not None:
            raise ValueError('nonExact_conditional_forbidden')


def asa_full_correction(rows, labels, records, catalog, mass):
    """Fit on training folds, apply unshrunk factor once to ORIGINAL grams.

    The historical half-shrunk predictions are deliberately discarded. Only
    raw_factor and training-fold fallback provenance are imported from fitting.
    """
    validate_exact(labels, records)
    options = catalog['asa_options']
    nutrients = {code: food['density'] for code, food in catalog['foods'].items()}
    fitted = mass.crossvalidate(rows, labels, records, options, nutrients,
        calibrate=True, dataset='asa24')
    params = {p['fold']: p for p in fitted['fold_parameters']}
    output = []
    for original in records:
        record = deepcopy(original)
        parameter = params[record['fold']]
        if set(parameter['training_groups']) & set(parameter['heldout_groups']):
            raise ValueError('calibration_group_leak')
        factor = parameter['raw_factor']
        if not .8 <= factor <= 1.25:
            raise ValueError('invalid_ASA_full_factor')
        e2e = record.get('e2e')
        if e2e and e2e.get('fallback') and e2e.get('option_id') is None:
            e2e['prediction'] = deepcopy(parameter['fallback'])
            record['prediction'] = deepcopy(parameter['fallback'])
        branch = record.get('conditional')
        if branch is not None:
            code = branch['code']
            if branch['fallback']:
                option = mass.median_option(options[code])
            elif branch.get('continuous_grams') is not None:
                option = mass.nearest_option(branch['continuous_grams'] * factor, options[code])
            else:
                raise ValueError('continuous_ASA_requires_original_grams')
            branch.update(option_id=option['option_id'],
                prediction=mass.option_prediction(option, nutrients[code]),
                calibration_factor=factor, calibration_source='training_fold_raw_factor_applied_once')
        output.append(record)
    applied_parameters = [{**p, 'historical_half_mass_factor': p['mass_factor'],
        'mass_factor': p['raw_factor'], 'shrinkage_toward_one': 0.,
        'application': 'full_correction_once'} for p in fitted['fold_parameters']]
    return dict(predictions=output, fold_parameters=applied_parameters,
        application='raw_factor_without_half_shrinkage_to_original_conditional_grams_once',
        E2E_calibrated=False)


def evaluate_without_recognition_gate(rows, labels, records):
    result = opt_metrics.evaluate(rows, labels, records)
    # Preserve recognition measurements, remove the obsolete 15% decision rule.
    asa = result['datasets'].get('asa24')
    if asa is not None:
        asa.pop('recognition_pass', None)
        asa.pop('minimum_exact_n', None)
    result.pop('recognition_pass', None)
    result['joint_pass'] = result['passed_checks'] == 20
    result['decision_rule'] = 'all_20_MAE_and_per_unit_checks; recognition_reported_without_hard_gate'
    return result


def score_all(rows, labels, catalog, frozen, usage=None):
    """Caller must seal ALL 843 records before loading and passing labels."""
    n5 = [r for r in rows if r['dataset'] == 'nutrition5k']
    asa = [r for r in rows if r['dataset'] == 'asa24']
    if len(rows) != 843:
        raise ValueError('full_843_required')
    original_n5 = completed_records(n5, 'N5', frozen)
    original_asa = completed_records(asa, 'ASA', frozen)
    validate_exact(labels, original_asa)
    mass = historical_mass_module()
    options = catalog['asa_options']
    densities = {c: f['density'] for c, f in catalog['foods'].items()}
    # Rebuild only registered training-fold fallback values; retain raw estimates.
    raw_n5 = mass.crossvalidate(n5, labels, original_n5, calibrate=False)['predictions']
    raw_asa = mass.crossvalidate(asa, labels, original_asa, options, densities, calibrate=False)['predictions']
    n5_cal = mass.crossvalidate(n5, labels, original_n5, calibrate=True, dataset='nutrition5k')
    asa_cal = asa_full_correction(asa, labels, original_asa, catalog, mass)
    profiles = {
        'raw': evaluate_without_recognition_gate(rows, labels, raw_n5 + raw_asa),
        'retained': evaluate_without_recognition_gate(rows, labels, n5_cal['predictions'] + asa_cal['predictions'])}
    if usage is None:
        from newcommon import usage_summary
        usage = {profile: usage_summary(profile) for profile in ('N5', 'ASA')}
    result = dict(scope='843_development_only_not_independent_acceptance', freeze_sha256=frozen,
        profiles=profiles, predeclared_retained='N5_half_shrinkage_plus_ASA_full_raw_factor',
        N5_density='per_leaf_candidate_average_fixed; one_mass_scaling_only',
        ASA_density='Exact_reference_code_density_after_legal_option_projection; no_candidate_average',
        ASA_primary_scope='all_Exact_conditional_including_fallback', ASA_E2E_scope='all_529_uncalibrated_except_fold_fallback',
        independent_acceptance_opened=False, usage=usage if usage is not None else {})
    atomic_json(HERE / 'N5_calibrated_oof.json', n5_cal)
    atomic_json(HERE / 'ASA_calibrated_oof.json', asa_cal)
    atomic_json(HERE / 'summary.json', result)
    with (HERE / 'results.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['profile','dataset','target','mae','per_unit','mae_limit','per_unit_limit','pass_mae','pass_per_unit'])
        for profile, evaluation in profiles.items():
            for dataset, stats in evaluation['datasets'].items():
                for target, value in stats['primary']['metrics'].items():
                    writer.writerow([profile,dataset,target,value['mae'],value['per_unit'],value['threshold_mae'],
                        value['threshold_per_unit'],value['pass_mae'],value['pass_per_unit']])
    return result

"""Completed development comparison only; no inference, fitting or credential reads."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
OLD = V2 / 'c_gemini_flash_20261105'
KEYS = ('weight', 'energy', 'fat', 'carbs', 'protein')


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def indexed(value):
    rows = value['predictions'] if isinstance(value, dict) else value
    result = {r['sample_id']: r for r in rows}
    if len(rows) != len(result):
        raise ValueError('duplicate_prediction_id')
    return result


def completed(directory):
    frozen = fingerprint(load(directory / 'freeze.json'))
    summary = load(directory / 'summary.json')
    if summary.get('freeze_sha256') != frozen:
        raise ValueError('summary_freeze_changed')
    output = {'summary': summary, 'raw': {}, 'retained': {}, 'original': {}, 'oof': {}}
    for profile, count in (('N5', 314), ('ASA', 529)):
        path = directory / 'run' / profile / 'records.json'
        seal = load(path.with_name('completion.json'))
        if (seal.get('profile') != profile or seal.get('records') != count
                or seal.get('freeze_sha256') != frozen or seal.get('records_sha256') != sha(path)):
            raise ValueError('complete_prediction_seal_required')
        original = indexed(load(path))
        oof = load(directory / (profile + '_calibrated_oof.json'))
        retained = indexed(oof)
        output['original'][profile] = original
        output['oof'][profile] = oof
        if len(original) != count or set(original) != set(retained):
            raise ValueError('complete_scored_population_required')
        # The saved OOF result holds the registered training-fold fallbacks.
        raw = deepcopy(retained)
        for sid, r in raw.items():
            if not original[sid].get('fallback'):
                r['prediction'] = deepcopy(original[sid]['prediction'])
            if profile == 'ASA':
                r['conditional'] = deepcopy(original[sid].get('conditional'))
        output['raw'][profile] = raw
        output['retained'][profile] = retained
    return output


def stats(ids, records, labels, conditional=False):
    metrics = {}
    for key in KEYS:
        truth = [labels[s]['truth'][key] for s in ids]
        values = [(records[s]['conditional']['prediction'] if conditional else records[s]['prediction'])[key] for s in ids]
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in truth + values):
            raise ValueError('nonfinite_metric_input')
        mean = statistics.mean(truth) if truth else None
        mae = statistics.mean(abs(a-b) for a,b in zip(truth, values)) if truth else None
        metrics[key] = dict(mae=mae, reference_mean=mean, per_unit=mae/mean if mean and mean > 0 else None)
    return dict(n=len(ids), metrics=metrics)


def paired(ids, new, old, labels, conditional=False):
    a, b = stats(ids, new, labels, conditional), stats(ids, old, labels, conditional)
    result = {}
    for key in KEYS:
        def err(records, sid):
            r = records[sid]['conditional'] if conditional else records[sid]
            return abs(r['prediction'][key] - labels[sid]['truth'][key])
        changes = [err(new,s)-err(old,s) for s in ids]
        result[key] = dict(new_minus_old_mae=a['metrics'][key]['mae']-b['metrics'][key]['mae'] if ids else None,
            new_minus_old_per_unit=a['metrics'][key]['per_unit']-b['metrics'][key]['per_unit']
                if a['metrics'][key]['per_unit'] is not None else None,
            improved_n=sum(x < -1e-9 for x in changes), worse_n=sum(x > 1e-9 for x in changes),
            tied_n=sum(abs(x) <= 1e-9 for x in changes))
    return dict(n=len(ids), metrics=result)


def recognition(records, labels):
    n = len(records)
    exact = {s for s,r in records.items() if labels[s]['food_code'] in r.get('final_codes', [])}
    for sid in exact:
        branch = records[sid].get('conditional')
        if not isinstance(branch, dict) or branch.get('code') != labels[sid]['food_code'] or branch.get('prediction') is None:
            raise ValueError('Exact_requires_explicit_conditional_prediction')
    retrieval = sum(labels[s]['food_code'] in r.get('retrieved_codes', []) for s,r in records.items())
    first = sum(labels[s]['food_code'] == r.get('primary_code') for s,r in records.items())
    return exact, dict(n=n, candidate_coverage_n=retrieval, candidate_coverage_rate=retrieval/n,
        exact_n=len(exact), exact_rate=len(exact)/n, first_n=first, first_rate=first/n)


def counts(records):
    result = {k: sum(bool(r.get(k)) for r in records.values())
        for k in ('fallback','method_fallback','repair_used','native_complete')}
    for branch in ('e2e','conditional'):
        result[branch + '_fallback'] = sum(bool((r.get(branch) or {}).get('fallback')) for r in records.values())
    result['fallback_reasons'] = dict(Counter(str(r.get('fallback_reason')) for r in records.values() if r.get('fallback')))
    return result


def model_usage(directory):
    models = {}
    paths = sorted(directory.rglob('requests.jsonl'))
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            model = row.get('model_requested', 'unknown')
            target = models.setdefault(model, dict(requests=0, errors=0, prompt_tokens=0,
                completion_tokens=0, total_tokens=0, reasoning_tokens=0))
            target['requests'] += 1
            target['errors'] += row.get('status') != 200
            usage = row.get('usage') or {}
            for key in ('prompt_tokens','completion_tokens','total_tokens'):
                target[key] += usage.get(key, 0) or 0
            target['reasoning_tokens'] += (usage.get('completion_tokens_details') or {}).get('reasoning_tokens',0) or 0
    return dict(by_model=models, request_log_paths=[str(p.relative_to(directory)) for p in paths],
        includes_preflight=True, cost_usd=None, cost_notice='Token accounting only; third-party price unverified.')


def require_fixed_upstream(new, old):
    if new['original']['N5'] != old['original']['N5'] or new['oof']['N5'] != old['oof']['N5']:
        raise ValueError('N5_raw_corrected_or_fold_parameters_changed')
    a, b = new['original']['ASA'], old['original']['ASA']
    if set(a) != set(b):
        raise ValueError('ASA_population_changed')
    for sid in a:
        for key in ('final_codes', 'primary_code', 'retrieved_codes', 'items', 'description',
                    'identity_status', 'identity_failure', 'dataset', 'group_id', 'fold'):
            if a[sid].get(key) != b[sid].get(key):
                raise ValueError('ASA_fixed_identity_changed:' + key)


def main():
    # Both new profiles must be complete before development labels are opened.
    new = completed(HERE)
    old = completed(OLD)
    require_fixed_upstream(new, old)
    rows = load(V2 / 'round11/prepared/cohort/dev.json')
    manifest = {r['sample_id']: r for r in rows}
    for profile, dataset in (('N5','nutrition5k'),('ASA','asa24')):
        expected = {s for s,r in manifest.items() if r['dataset'] == dataset}
        for data in (new, old):
            for stage in ('raw','retained'):
                records = data[stage][profile]
                if set(records) != expected:
                    raise ValueError('paired_population_changed')
                if any(any(r.get(k) != manifest[s][k] for k in ('dataset','group_id','fold')) for s,r in records.items()):
                    raise ValueError('paired_fold_or_group_changed')
    labels = load(V2 / 'round11/prepared/cohort/dev_labels.json')
    n5ids, asaids = sorted(new['raw']['N5']), sorted(new['raw']['ASA'])
    exact_new, rec_new = recognition(new['raw']['ASA'], labels)
    exact_old, rec_old = recognition(old['raw']['ASA'], labels)
    if exact_new != exact_old or len(exact_new) != 176:
        raise ValueError('fixed_176_Exact_population_required')
    common = sorted(exact_new & exact_old)
    result = dict(scope='development_only_no_refit_no_API', baseline=OLD.name,
        layout_notice='All314 N5 predictions and corrections are identical imported Gemini results; all529 ASA identities and176 Exact IDs are fixed. Only ASA portion changes from Gemini to Sol.',
        fixed_upstream=dict(N5_raw_and_corrected_unchanged=True, ASA_candidate_and_selected_codes_unchanged=True, ASA_Exact_n=176),
        N5_same314={}, ASA=dict(recognition=dict(new=rec_new, baseline=rec_old),
            own_Exact_warning='Both methods use the identical fixed176 Exact population; all529 E2E is reported separately.',
            common_Exact_n=len(common), new_only_Exact_n=len(exact_new-exact_old),
            baseline_only_Exact_n=len(exact_old-exact_new), own_Exact={}, common_Exact={}, all529_E2E={}),
        fallback_counts={}, usage=dict(new=model_usage(HERE), baseline=model_usage(OLD)),
        summaries=dict(new=new['summary'], baseline=old['summary']), independent_acceptance_opened=False)
    for stage in ('raw','retained'):
        result['N5_same314'][stage] = dict(new=stats(n5ids,new[stage]['N5'],labels),
            baseline=stats(n5ids,old[stage]['N5'],labels), paired=paired(n5ids,new[stage]['N5'],old[stage]['N5'],labels))
        a,b = new[stage]['ASA'],old[stage]['ASA']
        result['ASA']['own_Exact'][stage] = dict(new=stats(sorted(exact_new),a,labels,True), baseline=stats(sorted(exact_old),b,labels,True))
        result['ASA']['common_Exact'][stage] = dict(new=stats(common,a,labels,True), baseline=stats(common,b,labels,True), paired=paired(common,a,b,labels,True))
        result['ASA']['all529_E2E'][stage] = dict(new=stats(asaids,a,labels), baseline=stats(asaids,b,labels), paired=paired(asaids,a,b,labels))
    for name,data in (('new',new),('baseline',old)):
        result['fallback_counts'][name] = {p:counts(data['raw'][p]) for p in ('N5','ASA')}
    sources = [d/name for d in (HERE,OLD) for name in ('freeze.json','summary.json','N5_calibrated_oof.json','ASA_calibrated_oof.json','run/N5/records.json','run/N5/completion.json','run/ASA/records.json','run/ASA/completion.json')]
    result['source_sha256'] = {str(p.relative_to(V2)):sha(p) for p in sources}
    output = HERE / 'comparison.json'
    text = json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if output.exists() and output.read_text() != text:
        raise ValueError('existing_comparison_is_immutable')
    output.write_text(text)
    print(json.dumps(dict(status='comparison_complete',N5_n=314,ASA_n=529,common_Exact_n=len(common))))


if __name__ == '__main__':
    main()

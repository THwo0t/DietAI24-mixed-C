"""Frozen v1 inference helpers, without v1 orchestration or validation entrypoints.
Source: optimization_sol_20261102/src/optimize.py at 4707add.
Only v2-scoped inputs/outputs are used. Historical responses are imported by
explicit provenance checks in study.py, never by this module.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
from opt_provider import ROOT, fingerprint, sha, scoped
from opt_metrics import fold_medians, median_option, option_prediction

def load(path):
    return json.loads(Path(path).read_text())

def artifact_id(record):
    return fingerprint(record['sample_id'])[:24]

def make_pipeline(provider, catalog):
    import numpy as np
    from opt_pipeline import Pipeline, Retriever
    saved = load(ROOT / 'data/vectors/manifest.json')
    for name, expected in saved['files'].items():
        if sha(ROOT / 'data/vectors' / name) != expected:
            raise ValueError('Saved vector provenance mismatch')
    codes = load(ROOT / 'data/vectors/codes.json')
    if set(codes) != set(catalog['foods']):
        raise ValueError('Vector catalog food-code mismatch')
    return Pipeline(provider.chat, provider.embed, catalog,
                    Retriever(codes, np.load(ROOT / 'data/vectors/embeddings.npy', allow_pickle=False), catalog))

def image_path(row):
    path = scoped(ROOT / row['image_path'])
    if sha(path) != row['image_sha256']:
        raise ValueError('Input image checksum mismatch')
    return path

def per_dataset(config, dataset):
    value = deepcopy(config)
    value['portion_mode'] = value.pop('portion_modes')[dataset]
    return value

def row_median(row, rows, labels):
    return fold_medians(rows, labels, heldout_fold=row['fold'], dataset=row['dataset'])

def run_parallel(rows, function, workers, progress):
    # Cancel pending work on a systemic failure; all completed artifacts remain resumable.
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {executor.submit(function, row): row for row in rows}
    count = 0
    try:
        for future in as_completed(futures):
            future.result()
            count += 1
            progress(count, len(rows))
    except BaseException:
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True)

def fallback_branch(code, options, catalog, failure, median_vector):
    if options and code in catalog['foods']:
        option = median_option(options)
        return {'prediction': option_prediction(option, catalog['foods'][code]['density']),
                'option_id': option['option_id'], 'continuous_grams': None,
                'fallback': True, 'fallback_reason': failure, 'code': code}
    return {'prediction': dict(median_vector), 'option_id': None, 'continuous_grams': None,
            'fallback': True, 'fallback_reason': failure, 'code': code}

def branch_result(result, code, options, catalog, fallback_vector):
    if result.get('status') != 'ok' or result.get('prediction') is None:
        out = fallback_branch(code, options, catalog, result.get('failure', 'portion_failed'), fallback_vector)
    else:
        out = {k: result.get(k) for k in ('prediction', 'option_id', 'continuous_grams', 'code')}
        out.update(fallback=False, fallback_reason=None)
    out['raw_result'] = result
    return out

def infer_portions(pipeline, row, identity, config, catalog, label, fallback_vector, examples=()):
    from opt_pipeline import ContractError
    dataset = row['dataset']
    cfg = per_dataset(config, dataset)
    picture = image_path(row)
    codes = identity.get('final_codes', []) if identity.get('status') == 'ok' else []
    pred = {'sample_id': row['sample_id'], 'dataset': dataset, 'group_id': row['group_id'],
            'fold': row['fold'], 'final_codes': codes, 'primary_code': codes[0] if codes else None,
            'retrieved_codes': identity.get('retrieved_codes', []), 'items': identity.get('items', []),
            'description': identity.get('description', ''), 'identity_status': identity.get('status'),
            'identity_failure': identity.get('failure'), 'conditional': None, 'e2e': None}

    def portion(context=None):
        try:
            return pipeline.portion(picture, identity, dataset, cfg, conditional_context=context,
                                    examples=list(examples))
        except ContractError as exc:
            return {'status': 'failed', 'failure': str(exc), 'prediction': None, 'stages': []}

    result = portion()
    pred['repair_used'] = bool(identity.get('repaired') or result.get('repaired'))
    if dataset == 'nutrition5k':
        valid = result.get('status') == 'ok' and result.get('prediction') is not None
        pred.update(raw_prediction=result.get('prediction'),
                    prediction=result['prediction'] if valid else dict(fallback_vector),
                    native_complete=valid, fallback=not valid,
                    fallback_reason=None if valid else result.get('failure', 'incomplete_dish'),
                    portion_result=result)
    else:
        code = codes[0] if codes else None
        pred['e2e'] = branch_result(result, code, catalog['standard_options'].get(code, []), catalog, fallback_vector)
        pred.update(raw_prediction=result.get('prediction'), prediction=pred['e2e']['prediction'],
                    native_complete=not pred['e2e']['fallback'], fallback=pred['e2e']['fallback'])
        if label is not None and label['food_code'] in codes:
            true_code = label['food_code']
            options = catalog['asa_options'][true_code]
            # Only this explicit protocol adapter can supply the true code/catalogue.
            conditional = portion({'code': true_code, 'options': options})
            pred['conditional'] = branch_result(conditional, true_code, options, catalog, fallback_vector)
            pred['repair_used'] |= bool(conditional.get('repaired'))
    return pred

def choose_examples(row, identity, train_rows, labels, identity_dir, provider):
    """Two description-nearest examples; no same group or held-out development fold."""
    import numpy as np
    if any(r.get('split') != 'dev' for r in train_rows):
        raise ValueError('Example retrieval only accepts development records')
    description = identity.get('description', '').strip()
    if not description:
        return [], []
    eligible = [r for r in train_rows if r['dataset'] == row['dataset']
                and r['group_id'] != row['group_id']
                and (row['split'] != 'dev' or r['fold'] != row['fold'])]
    choices = []
    query = np.asarray(provider.embed([description])[0])
    for other in eligible:
        path = identity_dir / (artifact_id(other) + '.json')
        saved = load(path)
        text = saved.get('description', '').strip()
        if saved.get('status') != 'ok' or not text:
            continue
        vector = np.asarray(provider.embed([text])[0])
        similarity = float(query @ vector / (np.linalg.norm(query) * np.linalg.norm(vector)))
        choices.append((-similarity, other['sample_id'], other, text))
    examples, sources, used_groups = [], [], set()
    for _, _, other, text in sorted(choices):
        if other['group_id'] in used_groups:
            continue
        used_groups.add(other['group_id'])
        examples.append({'description': text, 'reference_weight': labels[other['sample_id']]['truth']['weight']})
        sources.append({'sample_id': other['sample_id'], 'group_id': other['group_id'], 'fold': other['fold']})
        if len(examples) == 2:
            break
    return examples, sources

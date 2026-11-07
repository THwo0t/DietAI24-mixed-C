"""Score sealed predictions offline. Labels never enter inference."""
import argparse
import json
import statistics
from pathlib import Path

from run_mixed_c import read, digest, finite_prediction, TARGETS


def score(seal, labels, protocol='e2e'):
    if not isinstance(seal, dict) or seal.get('records_sha256') != digest(seal.get('records')):
        raise ValueError('Prediction seal mismatch')
    records = seal['records']
    if protocol == 'conditional':
        records = [dict(x, sample_id=x['association_id']) for r in records for x in r['conditional_associations']]
    if not records or len({r['sample_id'] for r in records}) != len(records):
        raise ValueError('No records or duplicate scoring IDs')
    if not isinstance(labels, dict) or set(labels) != {r['sample_id'] for r in records}:
        raise ValueError('Labels must exactly match the sealed evaluation population')
    if not all(finite_prediction(x) and x['weight'] > 0 for x in labels.values()):
        raise ValueError('Each label needs five finite nonnegative fields and positive weight')
    pairs = []
    for r in records:
        if r['status'] == 'ok':
            if not finite_prediction(r['prediction']):
                raise ValueError('Invalid successful prediction')
            pairs.append((r['prediction'], labels[r['sample_id']]))
        elif r['status'] not in {'failed', 'not_matched'} or r['prediction'] is not None:
            raise ValueError('Failures and unmatched references must retain null predictions')
    metrics = {}
    for k in TARGETS:
        mae = statistics.fmean(abs(p[k] - t[k]) for p, t in pairs) if pairs else None
        mean = statistics.fmean(t[k] for _, t in pairs) if pairs else None
        metrics[k] = {'mae': mae, 'reference_mean': mean, 'perunit': mae / mean if mean else None}
    return {'protocol': protocol, 'attempted': len(records), 'scored': len(pairs),
            'failed': sum(r['status'] == 'failed' for r in records),
            'not_matched': sum(r['status'] == 'not_matched' for r in records),
            'successful_subset': metrics,
            'fixed_population_mae': {k: v['mae'] for k, v in metrics.items()} if len(pairs) == len(records) else None,
            'perunit_definition': 'sum absolute error / sum reference quantity, on the same scored rows',
            'new_api_calls': 0, 'label_imputation': False, 'calibration': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--protocol', choices=['e2e', 'conditional'], default='e2e')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = score(read(args.predictions), read(args.labels), args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(value, ensure_ascii=False))


if __name__ == '__main__':
    main()

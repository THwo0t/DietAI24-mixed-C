"""Check paired counts, recompute score summaries and audit paid request routes."""
from collections import Counter, defaultdict
import json
import math

import run


def main():
    d = run.DIRECTORY
    frozen = run.load(d / 'freeze.json')
    complete = run.load(d / 'completion.json')
    assert run.load(d / 'worker_state.json') == {'status': 'completed', 'worker_exited': True}
    assert complete['freeze_sha256'] == run.digest(frozen)
    assert complete['results_sha256'] == run.sha(d / 'results.json')
    seal = run.load(d / 'blind_identity_seal.json')
    identities = {row['sample_id']: row['identity'] for row in seal['identities']}
    assert len(seal['identities']) == len(identities) == 300
    assert seal['sha256'] == run.paired.seal_identities(frozen['rows'], identities)
    rows = {row['sample_id']: row for row in frozen['rows']}
    labels = run.original.allowed_labels(run.load(run.MANIFEST), frozen['rows'])
    entries, records = {}, {}
    for variant in run.VARIANTS:
        path = d / 'run' / variant / 'records.json'
        assert complete['records_sha256'][variant] == run.sha(path)
        array = run.load(path)
        indexed = records[variant] = {r['sample_id']: r for r in array}
        assert len(array) == len(indexed) == len(rows) and set(indexed) == set(rows)
        entries[variant] = {'asa24': {}, 'nutrition5k': {}}
        for sid, record in indexed.items():
            identity = identities[sid]
            dataset = rows[sid]['dataset']
            assert record['variant'] == variant and record['dataset'] == dataset and not record['fallback']
            assert record['identity_sha256'] == run.digest(identity) and record['final_codes'] == identity.get('final_codes', [])
            if dataset == 'nutrition5k':
                assert record['shared_n5_prediction'] and record['status'] in ('complete', 'failed')
                if record['status'] == 'complete':
                    entries[variant][dataset][sid] = record['prediction'], labels[sid]['truth']
                else:
                    assert record['prediction'] is None
                continue
            refs = {r['ref_id']: r for r in labels[sid]['references']}
            assert len(record['references']) == len(refs) and {r['ref_id'] for r in record['references']} == set(refs)
            for branch in record['references']:
                ref = refs[branch['ref_id']]
                assert branch['code'] == ref['food_code'] and branch['hit'] == (branch['code'] in record['final_codes'])
                if branch['status'] == 'complete':
                    assert branch['hit'] and branch['prediction']['weight'] == branch['legal_grams']
                    entries[variant][dataset][(sid, branch['ref_id'])] = branch['prediction'], ref['truth']
                else:
                    assert branch['prediction'] is None
    for sid in rows:
        if rows[sid]['dataset'] == 'nutrition5k':
            a, b = (records[v][sid] for v in run.VARIANTS)
            assert {k: v for k, v in a.items() if k != 'variant'} == {k: v for k, v in b.items() if k != 'variant'}

    result = run.load(d / 'results.json')['datasets']
    common_counts = {}
    for ds in ('asa24', 'nutrition5k'):
        common = set.intersection(*(set(entries[v][ds]) for v in run.VARIANTS))
        common_counts[ds] = len(common)
        for variant in run.VARIANTS:
            values = entries[variant][ds]
            for keys, expected in ((set(values), result[ds]['profiles'][variant]),
                                   (common, result[ds]['paired_common_calculated'][variant])):
                assert len(keys) == expected['scored_reference_associations']
                for k in run.paired.TARGETS:
                    ae = math.fsum(abs(values[s][0][k] - values[s][1][k]) for s in keys)
                    true = math.fsum(values[s][1][k] for s in keys)
                    for field, want in (('association_row_mae', ae / len(keys) if keys else None),
                                        ('association_row_per_unit', ae / true if true else None)):
                        actual = expected[field][k]
                        assert actual is None if want is None else math.isclose(actual, want, abs_tol=1e-10)

    counts, successes, sequences = Counter(), set(), defaultdict(set)
    for p in d.glob('run/*/provider/requests.jsonl'):
        for line in p.read_text().splitlines():
            e = json.loads(line)
            role, key, attempt = e['role'], e['request_hash'], e['attempt']
            assert 1 <= attempt <= 3 and attempt not in sequences[role, key]
            sequences[role, key].add(attempt)
            model = run.argon.ARGON if role == 'chat' else frozen['embedding_model']
            assert e['model_requested'] == model and e['endpoint'] == frozen['destinations'][role]
            if e['status'] == 200:
                assert e['model_served'] == model and (role, key) not in successes
                successes.add((role, key))
            else:
                counts['errors'] += 1
            counts['transmissions'] += 1
            counts[role] += 1
            counts['reported_tokens'] += e['usage'].get('total_tokens', 0)
    receipts = { (p.parent.name, p.stem): run.load(p)
                 for p in d.glob('run/attempt_receipts/*/*.json') }
    assert set(receipts) == set(sequences)
    for (role, key), seq in sequences.items():
        assert seq == set(range(min(seq), max(seq) + 1))
        ledger = run.load(run.V2 / 'runtime_cache/attempts' / role / (key + '.json'))
        assert receipts[role, key]['request_hash'] == ledger['request_hash'] == key
        assert receipts[role, key]['cumulative_attempts'] == ledger['attempts'] == max(seq)
    verification = dict(status='passed', blind_identities=300, records=600, common_counts=common_counts,
                        usage=counts, successful_transmissions=len(successes))
    run.immutable(d / 'verification.json', verification)
    print(json.dumps(verification, ensure_ascii=False))


if __name__ == '__main__':
    main()

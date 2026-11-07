"""Offline replay of the 57 round-09 ASA portion choices against the new factory."""
from pathlib import Path
import socket
from unittest.mock import patch

import asa_portion as pipeline

run = pipeline.round09


def offline(*args, **kwargs):
    raise RuntimeError('network disabled in cached replay')


def replay(factory, modules, catalog, identity, bound, attempts):
    remaining = iter(attempts)
    prompts = []

    def chat(messages):
        prompts.append(messages)
        return next(remaining)['raw']

    pipe = factory(modules, chat, offline, catalog, offline)
    value = pipe.portion(bound['row']['image_path'], identity, *bound['args'], **bound['kwargs'])
    return value, prompts


def main():
    d = run.DIRECTORY
    freeze = run.load(d / 'freeze.json')
    done = run.load(d / 'completion.json')
    assert done['status'] == 'completed' and done['freeze_sha256'] == run.digest(freeze)
    assert done['results_sha256'] == run.sha(d / 'results.json')
    assert all(run.sha(path) == value for path, value in freeze['files'].items())
    assert all(run.sha(d / 'run' / variant / 'records.json') == sha
               for variant, sha in done['records_sha256'].items())
    seal = run.load(d / 'blind_identity_seal.json')
    identities = {r['sample_id']: r['identity'] for r in seal['identities']}
    assert len(seal['identities']) == len(identities) == 300
    assert seal['sha256'] == run.paired.seal_identities(freeze['rows'], identities)
    manifest = run.load(run.MANIFEST)
    assert run.digest(manifest) == freeze['manifest_sha256']
    labels = run.original.allowed_labels(manifest, freeze['rows'])
    modules = run.original.modules(d)
    argon_modules = run.argon.argon_modules(modules, d)
    catalog = run.load(run.V2 / 'round11/prepared/catalog.json')
    fixed, recovered = {}, []
    skipped_repairs = 0
    old_factory = lambda m, *args: run.make_asa(m['pipeline'], *args)
    with patch.object(socket.socket, 'connect', offline), patch.object(socket, 'create_connection', offline), \
         patch.object(modules['pipeline'].current, 'encode_image', return_value='local replay image'):
        for path in sorted((d / 'run/shared/ASA/portion').glob('*.json')):
            blob = run.load(path)
            bound = blob['binding']['input']
            if bound['args'][-1]['portion_mode'] != 'portion_first':
                continue
            sid = bound['row']['sample_id']
            identity = identities[sid]
            raw = blob['value']
            assert blob['binding']['freeze_sha256'] == run.digest(freeze)
            assert bound['identity_sha256'] == run.digest(identity)
            assert blob['value_sha256'] == modules['provider'].fingerprint(raw)
            attempts = raw['stages'][-1]['attempts']
            before, old_messages = replay(old_factory, modules, catalog, identity, bound, attempts)
            after, new_messages = replay(pipeline.make_asa, argon_modules, catalog, identity, bound, attempts)
            assert before == raw and new_messages[0] == old_messages[0] and after['status'] == 'ok'
            key = sid, raw['code']
            assert key not in fixed
            fixed[key] = after['prediction']
            skipped_repairs += len(old_messages) - len(new_messages)
            if before['status'] == 'ok':
                assert all(before[k] == after[k] for k in
                           ('prediction', 'option_id', 'options', 'continuous_grams', 'protocol'))
                assert new_messages == old_messages[:len(new_messages)]
            else:
                assert len(new_messages) == 1 and len(old_messages) == 2
                recovered.append(dict(sample_id=sid, code=raw['code'], option_id=after['option_id'],
                                      weight=after['prediction']['weight']))

    old_rows, new_rows = [], []
    for record in run.load(d / 'run/continuous_option/records.json'):
        if record['dataset'] != 'asa24':
            continue
        sid = record['sample_id']
        refs = {r['ref_id']: r for r in labels[sid]['references']}
        for branch in record['references']:
            if branch['status'] != 'complete':
                continue
            code = branch['code']
            truth = refs[branch['ref_id']]['truth']
            old_rows.append((sid, branch['prediction'], truth))
            new_rows.append((sid, fixed[sid, code], truth))
    assert len(fixed) == len(new_rows) == len(old_rows) == 57 and len(recovered) == 5
    result = dict(status='cache_replay_passed', original_successes_unchanged=52, recovered_on_first_reply=5,
                  redundant_format_retries_in_cache=skipped_repairs,
                  new_api_calls=0, round09_freeze_sha256=run.digest(freeze),
                  paired_original=run.paired.error_summary(old_rows),
                  paired_compatible=run.paired.error_summary(new_rows), recovered=recovered)
    run.immutable(Path(__file__).with_name('cache_replay.json'), result)
    print(result['status'], '52 unchanged, 5 recovered, 57/57, 0 model calls,', skipped_repairs, 'unneeded repairs')


if __name__ == '__main__':
    main()

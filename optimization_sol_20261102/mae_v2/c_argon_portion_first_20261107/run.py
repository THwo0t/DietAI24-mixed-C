"""Round 09: same Argon identities and N5 results, paired ASA portion methods."""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
from threading import Event

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V2 = HERE.parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ARGON_SOURCE = V2 / 'c_mixed_argon_20261107/run.py'
PAIR_SOURCE = V2 / 'c_portion_first_20261105/paired_runner.py'
argon = load_module('round09_argon_transport', ARGON_SOURCE)
paired = load_module('round09_shared_pair', PAIR_SOURCE)
original = argon.original
require, digest = paired.require, paired.digest
load, sha, immutable = original.load, original.sha, original.immutable

MANIFEST = argon.MANIFEST
DIRECTORY = HERE / 'runs/round_09'
EXPERIMENT = 'round09_all_argon_asa_portion_first'
CONFIGS = {'asa24': original.ASA_CONFIG, 'nutrition5k': original.C_CONFIG}
VARIANTS = ('continuous_option', 'portion_first')
MODELS = argon.MODELS['all_argon']
paired.VARIANTS = VARIANTS
paired.PORTION_MODES = {variant: variant for variant in VARIANTS}


def make_asa(method, chat, embed, catalog, search4):
    class PortionFirstASA(method.ASAPipeline):
        def _stage(self, name, prompt, images, parser, stages):
            if name == 'portion_conditional_option_direct':
                # Same presentation as round 02. The original parser still
                # has the full local options and performs the conversion.
                marker = 'options for this food, not labels for the image: '
                prefix, separator, payload = prompt.partition(marker)
                require(bool(separator), 'portion_first_prompt_boundary')
                options, end = json.JSONDecoder().raw_decode(payload)
                checked = method.current.validate_options(options)
                visible = [{key: row[key] for key in ('option_id', 'description')} for row in checked]
                prompt = prefix + separator + json.dumps(visible) + payload[end:]
            return super()._stage(name, prompt, images, parser, stages)

        def portion(self, image_path, identity, dataset, config, conditional_context=None, examples=None):
            if dataset == 'asa24' and config.get('portion_mode') == 'portion_first':
                require(conditional_context is not None, 'portion_first_requires_conditional_ASA')
                config = {**config, 'portion_mode': 'direct'}
            return super().portion(image_path, identity, dataset, config,
                                   conditional_context=conditional_context, examples=examples)

    return PortionFirstASA(chat, embed, catalog, method.MultiQuery(chat, embed, search4, catalog['foods']))


def preflight(manifest):
    require(digest(manifest) == original.paired.MANIFEST_SHA256, 'original_3000_manifest_required')
    paired.old.preflight(manifest)
    by_id = {row['sample_id']: row for row in manifest['pool']}
    return [by_id[sid] for sid in manifest['rounds'][8]]


def freeze(manifest, rows, directory, modules):
    versions = {name: importlib.metadata.version(name) for name in ('chromadb', 'numpy', 'Pillow', 'httpx')}
    require(versions['chromadb'] == '0.4.24', 'original_chroma_version_required')
    sources = original.source_files(manifest)
    for path in (Path(__file__).resolve(), ARGON_SOURCE, PAIR_SOURCE):
        sources[str(path)] = sha(path)
    descriptor = load(directory / 'prepared/chroma_source.json')
    index = modules['chroma_index']
    original_files = descriptor['source_files']
    require(index.tree_hashes(original.REPO / 'index/chroma') == original_files, 'source_index_changed')
    actual = index.tree_hashes(directory / 'prepared/chroma')
    require(original.index_scientific(actual) == original.index_scientific(original_files), 'index_graph_changed')
    if not (directory / 'freeze.json').exists():
        require(actual == original_files, 'new_index_copy_full_attestation_required')
    return dict(schema=1, experiment=EXPERIMENT, round_number=9, rows=rows, manifest_sha256=digest(manifest),
                variants=list(VARIANTS), models=MODELS, configs=CONFIGS, files=sources,
                destinations=original.DESTINATIONS, chroma_source=descriptor, python=sys.version, versions=versions,
                embedding_model='text-embedding-3-large', max_attempts=3, max_concurrency=32,
                retry_waits_seconds=[30, 60], request_parameters=dict(temperature=1, asa24_max_completion_tokens=4096,
                                                                  nutrition5k_max_completion_tokens=8192),
                identity_policy='one_shared_blind_identity_per_image', n5_policy='one_shared_original_C_prediction',
                option_fields_sent=['option_id', 'description'], correction_applied=False,
                independent_acceptance='sealed', downloads_authorized=False)


def approval_scope(frozen):
    fields = ('experiment', 'round_number', 'manifest_sha256', 'variants', 'models', 'configs', 'destinations',
              'embedding_model', 'max_attempts', 'max_concurrency', 'option_fields_sent',
              'independent_acceptance', 'downloads_authorized', 'correction_applied')
    return dict(**{k: frozen[k] for k in fields}, freeze_sha256=digest(frozen),
                sample_ids=[r['sample_id'] for r in frozen['rows']])


def authorize(auth, frozen):
    require(all(auth.get(k) == v for k, v in approval_scope(frozen).items()), 'authorization_scope_mismatch')
    text = auth.get('explicit_user_authorization')
    require(isinstance(text, str) and text.strip() and
            not any(x in text.casefold() for x in ('pending', 'placeholder', '待批准', '待确认')),
            'explicit_user_authorization_required')


def execute(manifest, frozen, directory, base):
    rows = frozen['rows']
    index = base['chroma_index'].SearchIndex()
    catalog = load(V2 / 'round11/prepared/catalog.json')
    modules = argon.argon_modules(base, directory)
    stopped, providers, pipes, labels = Event(), {}, {}, {}
    status = 'failed'
    write = base['provider'].atomic_json
    write(directory / 'worker_state.json', dict(status='running', worker_exited=False))
    try:
        provider_class = original.round_provider_class(modules, directory, stopped)
        method = modules['pipeline']
        for dataset, profile, tokens in (('asa24', 'ASA', 4096), ('nutrition5k', 'N5', 8192)):
            provider = providers[profile] = provider_class(profile, allow_paid=True, sleep=original.retry_sleep)
            chat = modules['stage_routes'].StageChat(provider, dataset, tokens)
            if dataset == 'asa24':
                pipe = make_asa(method, chat, provider.embed, catalog, index.search4)
            else:
                pipe = method.make_component_pipeline(chat, provider.embed, catalog, index.search4)
            store = modules['newcommon'].Store(directory / 'run/shared' / profile, digest(frozen), 150)
            pipes[dataset] = argon.ImageBoundPipe(pipe, store, rows, stopped)

        # A real sample from this batch is cached before opening concurrency.
        first = next(row for row in rows if row['dataset'] == 'asa24')
        pipes['asa24'].identify_select(first['image_path'], 'asa24', CONFIGS['asa24'])

        def read_labels():
            labels.update(original.allowed_labels(manifest, rows))
            return labels

        seal = directory / 'blind_identity_seal.json'
        records = paired.execute_injected(rows, pipes, CONFIGS, catalog, read_labels,
            lambda value: immutable(seal, value), lambda: load(seal), workers=32)
        require(all(sha(row['image_path']) == row['image_sha256'] for row in rows), 'image_changed_before_scoring')
        require(freeze(manifest, rows, directory, base) == frozen, 'freeze_changed_during_execution')
        for variant in VARIANTS:
            require(all(r['status'] not in ('partial', 'empty', 'needs_review') for r in records[variant]),
                    'original_C_must_not_emit_nullable_states')
            immutable(directory / 'run' / variant / 'records.json', records[variant])
        result = paired.report(records, labels, rows)
        for profile in result['datasets']['nutrition5k']['profiles'].values():
            profile['fixed_population_mae'] = (profile['association_row_mae']
                if profile['scored_unique_images'] == 150 else None)
        immutable(directory / 'results.json', result)
        immutable(directory / 'completion.json', dict(status='completed', freeze_sha256=digest(frozen),
            results_sha256=sha(directory / 'results.json'), records_sha256={v:
                sha(directory / 'run' / v / 'records.json') for v in VARIANTS}))
        status = 'completed'
    finally:
        stopped.set()
        for profile, provider in providers.items():
            provider.stopped = True
            provider.close()
            write(directory / 'run' / profile / 'usage.json', modules['newcommon'].usage_summary(profile))
        write(directory / 'worker_state.json', dict(status=status, worker_exited=True))
    return dict(status=status, images=len(rows), variants=list(VARIANTS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--authorization', type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    manifest = load(MANIFEST)
    rows = preflight(manifest)
    with original.locked(DIRECTORY):
        base = original.modules(DIRECTORY)
        if args.prepare:
            base['chroma_index'].prepare_copy()
        frozen = freeze(manifest, rows, DIRECTORY, base)
        if args.prepare:
            immutable(DIRECTORY / 'freeze.json', frozen)
            immutable(DIRECTORY / 'approval_scope.json', approval_scope(frozen))
            result = dict(status='prepared_no_API', freeze_sha256=digest(frozen), images=len(rows),
                          asa_reference_associations=sum(r['reference_count'] for r in rows if r['dataset'] == 'asa24'))
        else:
            require(load(DIRECTORY / 'freeze.json') == frozen, 'scientific_freeze_changed')
            require(not (DIRECTORY / 'completion.json').exists(), 'round_already_completed_do_not_rerun')
            authorize(load(args.authorization), frozen)
            immutable(DIRECTORY / 'authorization_receipt.json', dict(sha256=sha(args.authorization)))
            result = execute(manifest, frozen, DIRECTORY, base)
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

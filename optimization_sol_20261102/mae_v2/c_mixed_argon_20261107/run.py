"""Fixed round 08: original mixed C vs Argon at every chat stage."""
import argparse
from contextvars import ContextVar
from copy import deepcopy
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from threading import Event

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V2 = HERE.parent
sys.path.insert(0, str(V2 / 'c_mixed_n5_sol_20261107'))
import production_runner as original

MANIFEST = V2 / 'c_minimal_nullable_20261105/pool_preparation/paired_manifest.json'
DIRECTORY = HERE / 'runs/round_08'
VARIANTS = ('mixed_c', 'all_argon')
ARGON = 'gemini-4-argon'
MODELS = {
    'mixed_c': dict(query_rewrite='gpt-6.1-sol', **original.paired.MODELS),
    'all_argon': dict(query_rewrite=ARGON, recognition=ARGON, selection=ARGON,
                      portion=dict(asa24=ARGON, nutrition5k=ARGON)),
}
EXPERIMENT = 'mixed_c_round08_all_chat_argon'
CONFIGS = {'asa24': original.ASA_CONFIG, 'nutrition5k': original.C_CONFIG}
require, digest = original.paired.require, original.paired.digest
load, sha, immutable = original.load, original.sha, original.immutable


def isolated_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def comparison_engine():
    engine = isolated_module('argon_paired_evaluation',
        V2 / 'c_minimal_nullable_20261105/paired_runner.py')
    engine.VARIANTS = VARIANTS
    base_summary = engine.error_summary

    def summary(entries):
        result = base_summary(entries)
        truth_sum = {k: sum(truth[k] for _, _, truth in entries) for k in engine.TARGETS}
        result['association_row_per_unit'] = {
            k: result['association_row_mae'][k] * len(entries) / truth_sum[k] if truth_sum[k] else None
            for k in engine.TARGETS}
        return result

    engine.error_summary = summary
    return engine


def argon_modules(base, arm_directory):
    """Load separate transport globals before starting any worker threads."""
    provider = isolated_module('argon_provider', original.OLD / 'provider.py')
    provider.CHAT_MODEL = provider.VISION_MODEL = ARGON
    provider.MODELS = {**provider.MODELS, 'llm': ARGON, 'llm_vision': ARGON}
    provider.STAGE_MODELS = {k: deepcopy(v) for k, v in MODELS['all_argon'].items() if k != 'query_rewrite'}
    # CachedProvider imports its base class and model whitelist at module load.
    old = sys.modules['provider']
    try:
        sys.modules['provider'] = provider
        common = isolated_module('argon_cached_provider', original.OLD / 'newcommon.py')
    finally:
        sys.modules['provider'] = old
    common.HERE = arm_directory

    # Keep the existing JSON handling at each stage, including raw query text
    # and ASA continuous grams. The setting is local to the worker context.
    json_mode = ContextVar('argon_original_stage_json_mode', default=False)
    normalize = provider.extract_unique_json
    provider.extract_unique_json = lambda raw: normalize(raw) if json_mode.get() else raw

    class ArgonProvider(common.CachedProvider):
        def chat(self, messages, max_completion_tokens=4096, *, dataset=None, stage=None):
            token = json_mode.set(stage in ('recognition', 'selection') or
                                  (dataset == 'nutrition5k' and stage == 'portion'))
            try:
                return super().chat(messages, max_completion_tokens, dataset=dataset, stage=stage)
            finally:
                json_mode.reset(token)

    common.CachedProvider = ArgonProvider
    return {**base, 'provider': provider, 'newcommon': common}


def preflight(manifest):
    require(digest(manifest) == original.paired.MANIFEST_SHA256, 'original_3000_manifest_required')
    comparison_engine().preflight(manifest)
    rows = {r['sample_id']: r for r in manifest['pool']}
    return [rows[s] for s in manifest['rounds'][7]]


def freeze(manifest, rows, directory, base):
    versions = {name: importlib.metadata.version(name) for name in ('chromadb', 'numpy', 'Pillow', 'httpx')}
    require(versions['chromadb'] == '0.4.24', 'original_chroma_version_required')
    files = original.source_files(manifest)
    files[str(Path(__file__).resolve())] = sha(__file__)
    descriptor = load(directory / 'prepared/chroma_source.json')
    original_files = descriptor['source_files']
    index = base['chroma_index']
    require(index.tree_hashes(original.REPO / 'index/chroma') == original_files, 'source_index_changed')
    actual = index.tree_hashes(directory / 'prepared/chroma')
    require(original.index_scientific(actual) == original.index_scientific(original_files), 'index_graph_changed')
    if not (directory / 'freeze.json').exists():
        require(actual == original_files, 'new_index_copy_full_attestation_required')
    return dict(schema=1, experiment=EXPERIMENT, manifest_sha256=digest(manifest),
        round_number=8, rows=rows, variants=list(VARIANTS), models=MODELS,
        destinations=original.DESTINATIONS, configs=CONFIGS, files=files, python=sys.version, versions=versions,
        chroma_source=descriptor, embedding_model='text-embedding-3-large',
        max_attempts=3, max_concurrency=32, retry_waits_seconds=[30, 60],
        request_parameters=dict(temperature=1, asa24_max_completion_tokens=4096,
                                nutrition5k_max_completion_tokens=8192),
        downloads_authorized=False, independent_acceptance='sealed', correction_applied=False)


def approval_scope(frozen):
    return {**{k: frozen[k] for k in ('experiment', 'manifest_sha256', 'round_number', 'variants', 'models',
        'destinations', 'configs', 'max_attempts', 'max_concurrency', 'downloads_authorized',
        'independent_acceptance', 'correction_applied')},
        'sample_ids': [r['sample_id'] for r in frozen['rows']], 'freeze_sha256': digest(frozen)}


def authorize(auth, frozen):
    require(all(auth.get(k) == v for k, v in approval_scope(frozen).items()), 'authorization_scope_mismatch')
    authorization = auth.get('explicit_user_authorization')
    require(isinstance(authorization, str) and authorization.strip() and
            not any(x in authorization.casefold() for x in ('pending', 'placeholder', '待批准', '待确认')),
            'explicit_user_authorization_required')


class ImageBoundPipe(original.DurablePipe):
    def cached(self, stage, image, binding, call):
        expected = self.rows[str(image)]['image_sha256']
        try:
            require(sha(image) == expected, 'image_changed_before_stage')

            def checked_call():
                value = call()
                require(sha(image) == expected, 'image_changed_during_stage')
                return value

            return super().cached(stage, image, binding, checked_call)
        except BaseException:
            self.stopped.set()
            raise


def execute(manifest, frozen, directory, base):
    rows, engine = frozen['rows'], comparison_engine()
    index = base['chroma_index'].SearchIndex()
    catalog = load(V2 / 'round11/prepared/catalog.json')
    stopped, providers, pipes = Event(), {}, {}
    labels = {}
    status = 'failed'
    write = base['provider'].atomic_json
    base['newcommon'].HERE = directory / 'arms/mixed_c'
    modules = {'mixed_c': base, 'all_argon': argon_modules(base, directory / 'arms/all_argon')}
    write(directory / 'worker_state.json', dict(status='running', worker_exited=False))
    try:
        for variant, m in modules.items():
            arm = directory / 'arms' / variant
            provider_class = original.round_provider_class(m, arm, stopped)
            pipes[variant] = {}
            for dataset, profile, tokens, factory in (
                ('asa24', 'ASA', 4096, m['pipeline'].make_asa_pipeline),
                ('nutrition5k', 'N5', 8192, m['pipeline'].make_component_pipeline),
            ):
                provider = providers[variant, profile] = provider_class(profile, allow_paid=True, sleep=original.retry_sleep)
                pipe = factory(m['stage_routes'].StageChat(provider, dataset, tokens), provider.embed, catalog, index.search4)
                store = m['newcommon'].Store(arm / 'stages' / profile, digest(frozen), 150)
                pipes[variant][dataset] = ImageBoundPipe(pipe, store, rows, stopped)

        # First actual Argon image is a fixed member of the paid population.
        # Its durable stage is reused below, so this adds no test image or reroll.
        first = next(row for row in rows if row['dataset'] == 'asa24')
        pipes['all_argon']['asa24'].identify_select(first['image_path'], 'asa24', CONFIGS['asa24'])

        def load_labels():
            labels.update(original.allowed_labels(manifest, rows))
            return labels

        seal = directory / 'blind_identity_seal.json'
        records = engine.execute_injected(rows, pipes, CONFIGS, catalog, load_labels,
            lambda value: immutable(seal, value), lambda: load(seal), workers=32)
        require(all(sha(row['image_path']) == row['image_sha256'] for row in rows), 'image_changed_before_scoring')
        require(freeze(manifest, rows, directory, base) == frozen, 'freeze_changed_during_execution')
        for variant in VARIANTS:
            immutable(directory / 'arms' / variant / 'records.json', records[variant])
        result = engine.report(records, labels, rows)
        for profile in result['datasets']['nutrition5k']['profiles'].values():
            profile['fixed_population_mae'] = (profile['association_row_mae']
                if profile['scored_unique_images'] == 150 else None)
        immutable(directory / 'results.json', result)
        immutable(directory / 'completion.json', dict(status='completed', freeze_sha256=digest(frozen),
            results_sha256=sha(directory / 'results.json'), records_sha256={v:
                sha(directory / 'arms' / v / 'records.json') for v in VARIANTS}))
        status = 'completed'
    finally:
        stopped.set()
        for (variant, profile), provider in providers.items():
            provider.stopped = True
            provider.close()
            write(directory / 'arms' / variant / 'run' / profile / 'usage.json',
                  modules[variant]['newcommon'].usage_summary(profile))
        write(directory / 'worker_state.json', dict(status=status, worker_exited=True))
    return {'status': status, 'images': len(rows), 'variants': list(VARIANTS)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--authorization', type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    manifest = load(MANIFEST)
    rows = preflight(manifest)
    if not (args.prepare or args.run):
        print(json.dumps(dict(status='preflight_no_API', images=len(rows), round_number=8)))
        return
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
            require(args.authorization is not None, 'explicit_authorization_file_required')
            require(load(DIRECTORY / 'freeze.json') == frozen, 'scientific_freeze_changed')
            require(not (DIRECTORY / 'completion.json').exists(), 'round_already_completed_do_not_rerun')
            authorize(load(args.authorization), frozen)
            immutable(DIRECTORY / 'authorization_receipt.json', dict(sha256=sha(args.authorization)))
            result = execute(manifest, frozen, DIRECTORY, base)
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

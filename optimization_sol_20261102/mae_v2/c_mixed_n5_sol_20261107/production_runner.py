"""One immutable paired round; defaults to read-only preflight, never paid by default."""
import argparse
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
from threading import Event
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V2 = HERE.parent
REPO = V2.parents[1]
OLD = V2 / 'c_stage_split_20261105'
sys.path.insert(0, str(HERE))
import paired_runner as paired
sys.path.insert(0, str(V2 / 'c_minimal_nullable_20261105'))
from label_adapter import adapt_asa_labels, load_allowed_n5_labels
C_CONFIG = dict(crop_review=False, portion_mode='total_ratio', profile='C', identity_profile='A')
ASA_CONFIG = dict(variant='optimized', image_mode='square256', candidate_cap=32,
                  portion_mode='continuous_option', examples=False)
N5_PORTION_MODEL = ContextVar('mixed_n5_portion_model', default=None)
EXPERIMENT = 'mixed_c_n5_total_ratio_only_model_change'

DESTINATIONS = {'chat': 'https://api.zicc.cc/v1/chat/completions',
                'embedding': 'https://api.openai.com/v1/embeddings'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def immutable(path, value):
    path = Path(path)
    if path.exists():
        paired.require(load(path) == value, 'immutable_artifact_changed:' + path.name)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + '\n'
    # An interrupted older write may leave a .tmp file. Never reopen or delete
    # it: use a fresh exclusive same-directory name on every attempt.
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
            prefix='.' + path.name + '.tmp-', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def locked(directory):
    directory.mkdir(parents=True, exist_ok=True)
    paired.require(not directory.is_symlink(), 'run_directory_symlink_rejected')
    with (directory / 'process.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def modules(directory):
    """No clients or credentials at import. Redirect only in-memory runtime roots."""
    sys.path[:0] = [str(OLD), str(V2 / 'src'), str(V2)]
    out = {}
    for name in ('config', 'stage_routes', 'provider', 'newcommon', 'pipeline', 'chroma_index'):
        existing = sys.modules.get(name)
        paired.require(existing is None or Path(existing.__file__).resolve() == OLD / (name + '.py'),
                       'foreign_module_collision:' + name)
        out[name] = importlib.import_module(name)
    paired.require(out['config'].STAGE_MODELS == paired.MODELS, 'frozen_model_route_changed')
    paired.require(out['provider'].CHAT_URL == DESTINATIONS['chat'] and
                   out['provider'].EMBED_URL == DESTINATIONS['embedding'], 'provider_destination_changed')
    out['newcommon'].HERE = directory
    out['chroma_index'].HERE = directory
    install_thread_local_route(out['provider'])
    return out


def install_thread_local_route(provider_module):
    if getattr(provider_module.stage_model, '_mixed_n5_context', None) is N5_PORTION_MODEL:
        return
    original = provider_module.stage_model

    def stage_model(messages, dataset=None, stage=None):
        model, images = original(messages, dataset, stage)
        override = N5_PORTION_MODEL.get()
        if images and dataset == 'nutrition5k' and stage == 'portion' and override is not None:
            paired.require(override in paired.N5_PORTION_MODELS.values(), 'unregistered_N5_model')
            return override, images
        return model, images

    stage_model._mixed_n5_context = N5_PORTION_MODEL
    provider_module.stage_model = stage_model


def retry_sleep(seconds):
    time.sleep({2: 30, 8: 60}[seconds])


def source_files(manifest):
    historical_freeze = OLD / 'freeze.json'
    attestation = {REPO / rel: expected for rel, expected in load(historical_freeze)['files'].items()}
    vectors = V2 / 'data/vectors'
    required_inputs = {V2 / 'data/catalog.json', V2 / 'round11/prepared/catalog.json'}
    required_inputs.update(vectors / name for name in
                           ('manifest.json', 'collection.json', 'codes.json', 'embeddings.npy'))
    paired.require(required_inputs <= attestation.keys(), 'historical_scientific_input_attestation_missing')
    # Only attested vector files belong to the frozen input. A new adjacent
    # file cannot be substituted for an original or silently become an input.
    sources = {path: expected for path, expected in attestation.items()
               if path.suffix == '.py' or path in required_inputs or vectors in path.parents}
    for path, expected in sources.items():
        paired.require(path.is_file() and sha(path) == expected,
                       'historical_scientific_source_drift:' + str(path.relative_to(REPO)))
    dependencies = [V2 / 'c_minimal_nullable_20261105' / name
                    for name in ('paired_runner.py', 'label_adapter.py')]
    additions = {path: sha(path) for path in [*HERE.glob('*.py'), *dependencies, historical_freeze]}
    exclusion = manifest['exclusion_proof']
    additions[Path(exclusion['artifact_path'])] = exclusion['sha256']
    labels = manifest['label_sources']
    additions[Path(labels['asa']['path'])] = labels['asa']['sha256']
    for table, path in labels['n5']['source_paths'].items():
        additions[Path(path)] = labels['n5']['source_sha256'][table]
    paired.require(not sources.keys() & additions.keys(), 'historical_source_binding_collision')
    for path, expected in additions.items():
        paired.require(path.is_absolute() and sha(path) == expected, 'source_or_opaque_label_hash_changed')
    return {str(p): v for p, v in sorted((sources | additions).items())}


def round_rows(manifest, number):
    paired.require(number == 7, 'only_round07_authorized_for_preparation')
    byid = {row['sample_id']: row for row in manifest['pool']}
    return [byid[sid] for sid in manifest['rounds'][number-1]]


def index_scientific(files):
    return {k: v for k, v in files.items() if not Path(k).name.startswith('chroma.sqlite3')}


def freeze_value(manifest, number, directory, m):
    versions = {key: importlib.metadata.version(key) for key in ('chromadb', 'numpy', 'Pillow', 'httpx')}
    paired.require(versions['chromadb'] == '0.4.24', 'original_chroma_version_required')
    return dict(schema=1, experiment=EXPERIMENT,
        python=sys.version, versions=versions,
        manifest_sha256=paired.digest(manifest), round_number=number,
        rows=round_rows(manifest, number), variants=list(paired.VARIANTS),
        models=paired.MODELS, n5_portion_models=paired.N5_PORTION_MODELS, destinations=DESTINATIONS,
        request_policy=dict(temperature=1, max_completion_tokens={'asa24':4096, 'nutrition5k':8192},
                            embedding_model='text-embedding-3-large', cache_namespaces={'ASA':'development', 'N5':'mae-v2-development'},
                            exact_request_cache_only=True, preserve_global_attempt_ledger=True,
                            retry_waits_seconds=[30, 60]),
        configs={'asa24': ASA_CONFIG, 'nutrition5k': C_CONFIG},
        identity_policy='one_shared_blind_identity_per_image',
        n5_policy='shared_identity_and_original_total_ratio_only_portion_model_changes',
        n5_response_policy='shared_original_Gemini_unique_JSON_normalizer_and_original_C_parser',
        asa_policy='any_selected_reference_code_one_shared_continuous_option_per_code',
        failure_policy='contract_failure_None_original_format_repair_no_fallback',
        independent_acceptance='sealed', downloads_authorized=False,
        correction_applied=False, workers_max=32, max_attempts=3,
        files=source_files(manifest), chroma_source=load(directory / 'prepared/chroma_source.json'))


def prepare(manifest, number, directory):
    paired.preflight(manifest)
    round_rows(manifest, number)
    with locked(directory):
        m = modules(directory)
        m['chroma_index'].prepare_copy()
        original = load(directory / 'prepared/chroma_source.json')['source_files']
        actual = m['chroma_index'].tree_hashes(directory / 'prepared/chroma')
        paired.require(index_scientific(original) == index_scientific(actual), 'prepared_index_scientific_drift')
        if not (directory / 'freeze.json').exists():
            paired.require(actual == original, 'new_index_copy_full_attestation_required')
        value = freeze_value(manifest, number, directory, m)
        immutable(directory / 'freeze.json', value)
        immutable(directory / 'manifest.json', manifest)
        return {'status': 'prepared_no_API', 'freeze_sha256': paired.digest(value),
                'round_number': number, 'unique_images': len(value['rows']), 'new_api_calls': 0}


def authorize(auth, frozen, allow_paid):
    paired.require(allow_paid and auth is not None, 'paid_requires_flag_and_concrete_authorization')
    paired.require(auth.get('freeze_sha256') == paired.digest(frozen), 'authorization_freeze_mismatch')
    expected = dict(manifest_sha256=frozen['manifest_sha256'], round_number=frozen['round_number'],
        sample_ids=[r['sample_id'] for r in frozen['rows']], variants=list(paired.VARIANTS),
        models=paired.MODELS, n5_portion_models=paired.N5_PORTION_MODELS, destinations=DESTINATIONS, max_attempts=3, max_concurrency=32,
        downloads_authorized=False, independent_acceptance='sealed', correction_applied=False)
    paired.require(all(auth.get(k) == v for k, v in expected.items()), 'concrete_authorization_scope_mismatch')
    paired.require(auth.get('experiment') == frozen['experiment'] and auth.get('configs') == frozen['configs'], 'new_experiment_authorization_required')
    paired.require(isinstance(auth.get('explicit_user_authorization'), str) and
                   auth['explicit_user_authorization'].strip() and
                   not any(term in auth['explicit_user_authorization'].casefold()
                           for term in ('pending', 'placeholder', '待批准', '待确认')),  'explicit_user_authorization_required')


def verify(manifest, number, directory):
    paired.preflight(manifest)
    m = modules(directory)
    frozen = load(directory / 'freeze.json')
    paired.require(load(directory / 'manifest.json') == manifest, 'frozen_manifest_changed')
    paired.require(frozen == freeze_value(manifest, number, directory, m), 'scientific_freeze_changed')
    descriptor = frozen['chroma_source']
    paired.require(m['chroma_index'].tree_hashes(REPO / 'index/chroma') == descriptor['source_files'],
                   'original_index_changed')
    paired.require(index_scientific(m['chroma_index'].tree_hashes(directory / 'prepared/chroma')) ==
                   index_scientific(descriptor['source_files']), 'prepared_index_scientific_drift')
    return frozen, m


class DurablePipe:
    """Save new variant-specific stages; model responses reuse exact provider payloads."""
    def __init__(self, pipe, store, rows, stopped):
        self.pipe, self.store, self.stopped = pipe, store, stopped
        self.rows = {r['image_path']: r for r in rows}

    def cached(self, stage, image, binding, call):
        if self.stopped.is_set():
            raise RuntimeError('provider_stopped_before_next_stage')
        try:
            row = self.rows[str(image)]
            bound = dict(row=row, **binding)
            sid = row['sample_id']
            if stage == 'portion':
                sid += '-' + paired.digest(binding)
            value = self.store.get(stage, sid, bound)
            if value is None:
                value = call()
                self.store.put(stage, sid, bound, value)
            return value
        except BaseException:
            self.stopped.set()
            raise

    def identify_select(self, image, dataset, config):
        return self.cached('identity', image, dict(config=config),
                           lambda: self.pipe.identify_select(image, dataset, config))

    def identify(self, image, config):
        layout = self.cached('layout', image, dict(config=config), lambda: self.pipe.layout(image))
        return self.cached('identity', image, dict(config=config, layout_sha256=paired.digest(layout)),
                           lambda: self.pipe.identify(image, config, layout=layout))

    def portion(self, image, identity, *args, portion_model=None, **kwargs):
        def evaluate():
            token = N5_PORTION_MODEL.set(portion_model)
            try:
                return self.pipe.portion(image, identity, *args, **kwargs)
            finally:
                N5_PORTION_MODEL.reset(token)

        binding = dict(identity_sha256=paired.digest(identity), args=list(args),
                       kwargs=kwargs, portion_model=portion_model)
        return self.cached('portion', image, binding, evaluate)


def allowed_labels(manifest, rows):
    """Called ONLY by execute_injected after durable full blind seal read-back."""
    sources = manifest['label_sources']
    asa = sources['asa']
    paired.require(sha(asa['path']) == asa['sha256'], 'ASA_label_source_changed')
    ids = [r['sample_id'] for r in rows if r['dataset'] == 'asa24']
    labels = adapt_asa_labels(load(asa['path']), ids)
    n5 = sources['n5']
    labels.update(load_allowed_n5_labels(n5['source_paths'],
        [r for r in rows if r['dataset'] == 'nutrition5k'], n5['source_sha256']))
    paired.require(set(labels) == {r['sample_id'] for r in rows}, 'fixed_label_population_mismatch')
    return labels


def round_provider_class(m, directory, stopped):
    class RoundProvider(m['newcommon'].CachedProvider):
        def _request(self, role, payload, *, route=None):
            if stopped.is_set():
                raise m['provider'].ProviderStop('Peer stage stopped')
            return super()._request(role, payload, route=route)

        def chat(self, messages, max_completion_tokens=4096, *, dataset=None, stage=None):
            result = super().chat(messages, max_completion_tokens, dataset=dataset, stage=stage)
            # Gemini already uses this normalizer in the original provider.
            if (dataset, stage, N5_PORTION_MODEL.get()) == ('nutrition5k', 'portion', m['provider'].CHAT_MODEL):
                return m['provider'].extract_unique_json(result)
            return result

        def _reserve_attempt(self, cache_id, role):
            if stopped.is_set():
                raise m['provider'].ProviderStop('Peer stage stopped')
            count = super()._reserve_attempt(cache_id, role)
            m['provider'].atomic_json(directory / 'run/attempt_receipts' / role / (cache_id + '.json'),
                dict(request_hash=cache_id, cumulative_attempts=count, global_ledger_preserved=True))
            return count

    return RoundProvider


def run(manifest, number, directory, allow_paid, authorization, workers):
    paired.require(type(workers) is int and workers == 32, 'authorized_run_requires_32_workers')
    with locked(directory):
        frozen, m = verify(manifest, number, directory)
        authorize(load(authorization) if authorization else None, frozen, allow_paid)
        immutable(directory / 'authorization_receipt.json', dict(path=str(authorization), sha256=sha(authorization)))
        paired.require(not (directory / 'completion.json').exists(), 'round_already_completed_do_not_rerun')
        rows = frozen['rows']
        index = m['chroma_index'].SearchIndex()  # validates 5624 unchanged documents, metadata and vectors
        catalog = load(V2 / 'round11/prepared/catalog.json')
        providers = {}
        stopped = Event()
        status = 'failed'
        m['provider'].atomic_json(directory / 'worker_state.json', dict(status='running', worker_exited=False))
        try:
            RoundProvider = round_provider_class(m, directory, stopped)
            for profile in ('N5', 'ASA'):
                # First credential access/client construction, only after all gates.
                providers[profile] = RoundProvider(profile, allow_paid=True, sleep=retry_sleep)
            pipes = {}
            method = m['pipeline']
            for dataset, profile, tokens, factory in (
                ('asa24', 'ASA', 4096, method.make_asa_pipeline),
                ('nutrition5k', 'N5', 8192, method.make_component_pipeline)):
                provider = providers[profile]
                pipe = factory(m['stage_routes'].StageChat(provider, dataset, tokens), provider.embed,
                               catalog, index.search4)
                store = m['newcommon'].Store(directory / 'run' / 'shared_mixed_c' / profile, paired.digest(frozen), 150)
                pipes[dataset] = DurablePipe(pipe, store, rows, stopped)
            label_state = {}
            def loader():
                label_state.update(allowed_labels(manifest, rows))
                return label_state
            records = paired.execute_injected(rows, pipes, frozen['configs'], catalog, loader,
                lambda value: immutable(directory / 'run/blind_identity_seal.json', value),
                lambda: load(directory / 'run/blind_identity_seal.json'), workers=workers)
            verify(manifest, number, directory)
            for variant in paired.VARIANTS:
                immutable(directory / 'run' / variant / 'records.json', records[variant])
            immutable(directory / 'results.json', paired.report(records, label_state, rows))
            immutable(directory / 'completion.json', dict(status='completed', freeze_sha256=paired.digest(frozen),
                records_sha256={v: sha(directory / 'run' / v / 'records.json') for v in paired.VARIANTS},
                results_sha256=sha(directory / 'results.json')))
            status = 'completed'
        finally:
            stopped.set()
            for provider in providers.values():
                provider.stopped = True
                provider.close()
            for profile in providers:
                m['provider'].atomic_json(directory / 'run' / profile / 'usage.json', m['newcommon'].usage_summary(profile))
            m['provider'].atomic_json(directory / 'worker_state.json', dict(status=status, worker_exited=True))
        return dict(status=status, unique_images=len(rows), variants=list(paired.VARIANTS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=V2 / 'c_minimal_nullable_20261105/pool_preparation/paired_manifest.json')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--round', type=int, default=7)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--allow-paid', action='store_true')
    parser.add_argument('--authorization', type=Path)
    args = parser.parse_args()
    directory = HERE / 'runs' / ('round_%02d' % args.round)
    try:
        manifest = load(args.manifest)
        if args.run:
            result = run(manifest, args.round, directory, args.allow_paid, args.authorization, args.workers)
        elif args.prepare:
            paired.require(not args.allow_paid, 'prepare_is_offline_remove_paid_flag')
            result = prepare(manifest, args.round, directory)
        else:
            paired.require(not args.allow_paid, 'preflight_is_offline_remove_paid_flag')
            round_rows(manifest, args.round)
            result = paired.preflight(manifest)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({'status': 'blocked', 'reason': str(exc)}, ensure_ascii=False))
        raise SystemExit(2)


if __name__ == '__main__':
    main()

"""Offline-only tests. Transport, credentials, cache and ledgers are isolated."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, Event
import unittest
from unittest.mock import patch

import httpx
import production_runner as runner

paired = runner.paired
GEMINI, SOL = paired.N5_PORTION_MODELS.values()
A, B = '12345678', '23456789'


class RouteTests(unittest.TestCase):
    def test_32_threads_exact_payload_cache_and_original_routes(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            m = runner.modules(directory)
            p, common = m['provider'], m['newcommon']
            stopped = Event()
            sent = []
            barrier = Barrier(32)
            keys = ('fake-sol-key', 'fake-embed-key', 'fake-gemini-key')

            def handler(req):
                body = json.loads(req.content)
                sent.append((body, req.headers['Authorization']))
                return httpx.Response(200, json={'model': body['model'], 'choices': [
                    {'message': {'content': 'Explanation {"grams":60}'}, 'finish_reason': 'stop'}]})

            def work(index):
                model = (GEMINI, SOL)[index % 2]
                token = runner.N5_PORTION_MODEL.set(model)
                try:
                    messages = [{'role': 'user', 'content': [
                        {'type': 'text', 'text': 'plate ' + str(index // 2)},
                        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,YQ=='}}]}]
                    barrier.wait(timeout=10)
                    result = client.chat(messages, max_completion_tokens=8192, dataset='nutrition5k', stage='portion')
                    self.assertEqual(result, '{"grams":60}')
                    self.assertEqual(client.chat(messages, max_completion_tokens=8192,
                                                 dataset='nutrition5k', stage='portion'), result)
                    self.assertEqual(p.stage_model(messages, 'nutrition5k', 'recognition'), (GEMINI, True))
                    self.assertEqual(p.stage_model(messages, 'nutrition5k', 'selection'), (GEMINI, True))
                    self.assertEqual(p.stage_model(messages, 'asa24', 'portion'), (SOL, True))
                    self.assertEqual(p.stage_model([{'role': 'user', 'content': 'rewrite'}],
                                                   'nutrition5k', 'query_rewrite'), (SOL, False))
                finally:
                    runner.N5_PORTION_MODEL.reset(token)
                self.assertEqual(p.stage_model(messages, 'nutrition5k', 'portion'), (GEMINI, True))

            with patch.object(p, 'ROOT', directory), patch.object(common, 'V2', directory),                     patch.object(p, 'read_credentials', side_effect=AssertionError('credentials touched')):
                client = runner.round_provider_class(m, directory, stopped)(
                    'N5', allow_paid=True, keys=keys, transport=httpx.MockTransport(handler))
                try:
                    with ThreadPoolExecutor(max_workers=32) as pool:
                        list(pool.map(work, range(32)))
                finally:
                    client.close()
            self.assertEqual(Counter(body['model'] for body, _ in sent), {GEMINI: 16, SOL: 16})
            by_input = {}
            for body, key in sent:
                self.assertEqual(key, 'Bearer ' + keys[2 if body['model'] == GEMINI else 0])
                model = body.pop('model')
                self.assertEqual(set(body), {'messages', 'temperature', 'max_completion_tokens'})
                by_input.setdefault(json.dumps(body, sort_keys=True), set()).add(model)
            self.assertEqual(len(by_input), 16)
            self.assertTrue(all(models == {GEMINI, SOL} for models in by_input.values()))
            receipts = list((directory / 'run/attempt_receipts/chat').glob('*.json'))
            self.assertEqual(len(receipts), 32)
            self.assertTrue(all(runner.load(p)['cumulative_attempts'] == 1 for p in receipts))

    def test_original_N5_prompt_and_mass_parser_through_mock_http(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            m = runner.modules(directory)
            p, common, original = m['provider'], m['newcommon'], m['pipeline']
            stopped, calls = Event(), []
            catalog = {'foods': {A: {'description': 'cooked plain rice',
                                    'density': dict.fromkeys(paired.TARGETS[1:], 200.)}}}
            child = dict(component_id='p1c1', description='rice')
            parent = dict(parent_id='p1', description='rice', accounting='whole', components=[child])
            leaf = dict(component_id='p1c1', parent_id='p1', primary_code=A, alternative_codes=[])
            identity = dict(status='ok', parents=[parent], leaves=[leaf], input_image_sha256='x')
            row = dict(sample_id='s', image_path='unused', image_sha256='x')

            def handler(req):
                data = json.loads(req.content)
                calls.append(data)
                model = data['model']
                grams = 75 if model == GEMINI else 90
                raw = 'Explanation ' + json.dumps(dict(total_grams=grams, parents=[
                    dict(parent_id='p1', ratio=1, components=[dict(component_id='p1c1', ratio=1)])]))
                return httpx.Response(200, json={'model': model, 'choices': [
                    {'message': {'content': raw}, 'finish_reason': 'stop'}]})

            with patch.object(p, 'ROOT', directory), patch.object(common, 'V2', directory), \
                    patch.object(original.component, 'encode_image', return_value='data:image/png;base64,YQ=='):
                client = runner.round_provider_class(m, directory, stopped)(
                    'N5', allow_paid=True, keys=('fake-s','fake-e','fake-g'), transport=httpx.MockTransport(handler))
                try:
                    chat = m['stage_routes'].StageChat(client, 'nutrition5k', 8192)
                    pipe = original.make_component_pipeline(chat, None, catalog, None)
                    store = common.Store(directory / 'stages', 'synthetic-freeze')
                    durable = runner.DurablePipe(pipe, store, [row], stopped)
                    with patch.object(pipe, '_checked_identity', return_value=([parent], [leaf])):
                        results = [durable.portion('unused', identity, runner.C_CONFIG, portion_model=model)
                                   for model in (GEMINI, SOL)]
                    weights = [paired.n5_prediction(row, identity, result, catalog)['weight'] for result in results]
                    self.assertEqual(weights, [75., 90.])
                    self.assertEqual([call['model'] for call in calls], [GEMINI, SOL])
                    self.assertEqual(dict(calls[0], model=SOL), calls[1])
                    prompt = calls[0]['messages'][1]['content'][0]['text']
                    self.assertIn('Estimate total plate grams and relative parent masses', prompt)
                    self.assertIn('cooked plain rice', prompt)
                    self.assertNotIn('portion_model', prompt)
                finally:
                    client.close()

    def test_retry_bound_survives_new_provider_and_peer_stop(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            m = runner.modules(directory)
            p, common = m['provider'], m['newcommon']
            stopped, sent, waits = Event(), [], []
            messages = [{'role': 'user', 'content': 'query retry'}]
            def handler(req):
                sent.append(req)
                return httpx.Response(503, json={'error': {'type': 'server_error'}})

            with patch.object(p, 'ROOT', directory), patch.object(common, 'V2', directory),                     patch.object(runner.time, 'sleep', side_effect=waits.append):
                provider = runner.round_provider_class(m, directory, stopped)
                for _ in range(2):
                    client = provider('N5', allow_paid=True, keys=('s', 'e', 'g'),
                                      transport=httpx.MockTransport(handler), sleep=runner.retry_sleep)
                    try:
                        with self.assertRaises(p.ProviderStop):
                            client.chat(messages, dataset='nutrition5k', stage='query_rewrite')
                    finally:
                        client.close()
                stopped.set()
                client = provider('ASA', allow_paid=True, keys=('s', 'e', 'g'),
                                  transport=httpx.MockTransport(handler))
                try:
                    with self.assertRaisesRegex(p.ProviderStop, 'Peer stage stopped'):
                        client.chat(messages, dataset='asa24', stage='query_rewrite')
                finally:
                    client.close()
            self.assertEqual(len(sent), 3)
            self.assertEqual(waits, [30, 60])


class PairedTests(unittest.TestCase):
    def test_historical_catalog_and_vector_drift_rejected(self):
        manifest = runner.load(runner.V2 / 'c_minimal_nullable_20261105/pool_preparation/paired_manifest.json')
        attestation = runner.load(runner.OLD / 'freeze.json')['files']
        sources = runner.source_files(manifest)
        scientific_inputs = (
            runner.V2 / 'round11/prepared/catalog.json',
            runner.V2 / 'data/catalog.json',
            *(runner.V2 / 'data/vectors' / name
              for name in ('manifest.json', 'collection.json', 'codes.json', 'embeddings.npy')),
        )
        original_sha = runner.sha
        for path in scientific_inputs:
            with self.subTest(file=path.name):
                relative = str(path.relative_to(runner.REPO))
                self.assertEqual(sources[str(path)], attestation[relative])
                with patch.object(runner, 'sha', side_effect=lambda file: '0'*64 if file == path else original_sha(file)):
                    with self.assertRaisesRegex(ValueError, 'historical_scientific_source_drift:'):
                        runner.source_files(manifest)

    def test_sealed_once_any_code_ASA_and_common_N5_with_explicit_failure(self):
        rows = [
            dict(sample_id='asa', image_path='a', image_sha256='a'*64, dataset='asa24', reference_count=2),
            dict(sample_id='n5-good', image_path='n1', image_sha256='b'*64, dataset='nutrition5k', reference_count=1),
            dict(sample_id='n5-fail', image_path='n2', image_sha256='c'*64, dataset='nutrition5k', reference_count=1),
        ]
        counts, saved = Counter(), {}
        labels_ready = Event()
        configs = {'asa24': runner.ASA_CONFIG, 'nutrition5k': runner.C_CONFIG}
        cat = {'foods': {A: {'density':dict.fromkeys(paired.TARGETS[1:], 100.)},
                         B: {'density':dict.fromkeys(paired.TARGETS[1:], 300.)}},
               'asa_options': {B: [{'option_id':'low','grams':45.}, {'option_id':'high','grams':65.}]}}
        truth = dict.fromkeys(paired.TARGETS, 50.)
        refs = [{'ref_id':s, 'food_code':B, 'truth':truth} for s in ('ref1','ref2')]

        class Pipe:
            def identify_select(self, image, dataset, config):
                self_outer.assertFalse(labels_ready.is_set())
                counts[image] += 1
                return dict(status='ok', final_codes=[A, B])

            def identify(self, image, config):
                self_outer.assertFalse(labels_ready.is_set())
                counts[image] += 1
                return dict(status='ok', input_image_sha256={'n1':'b'*64, 'n2':'c'*64}[image],
                    leaves=[dict(component_id='leaf', primary_code=A, alternative_codes=[B])],
                    final_codes=[A,B])

            def portion(self, image, identity, *args, **kwargs):
                self_outer.assertTrue(labels_ready.is_set())
                if image == 'a':
                    counts['asa-portion'] += 1
                    self_outer.assertEqual(args[1]['portion_mode'], 'continuous_option')
                    self_outer.assertEqual(kwargs['conditional_context']['code'], B)
                    return dict(status='ok', continuous_grams=55., option_id='low')
                model = kwargs['portion_model']
                counts[image,model] += 1
                self_outer.assertEqual(args, (runner.C_CONFIG,))
                if image == 'n2' and model == SOL:
                    return dict(status='failed', failure='portion_contract', prediction=None)
                grams = 40. if model == GEMINI else 50.
                return dict(status='ok', portions=[dict(component_id='leaf', grams=grams)],
                            prediction={'weight':grams, **dict.fromkeys(paired.TARGETS[1:], grams*2)})

        def writer(value):
            saved.update(deepcopy(value))
        def reader():
            self.assertEqual(len(saved['identities']), 3)
            counts['reader'] += 1
            return deepcopy(saved)
        def loader():
            self.assertEqual(counts['reader'], 1)
            labels_ready.set()
            return {'asa': {'references':refs}, 'n5-good':{'truth':truth}, 'n5-fail':{'truth':truth}}
        self_outer = self
        pipe = Pipe()
        records = paired.execute_injected(rows, dict.fromkeys(configs, pipe), configs, cat,
                                          loader, writer, reader, workers=3)
        self.assertEqual([counts[k] for k in ('a','n1','n2','asa-portion')], [1,1,1,1])
        self.assertTrue(all(counts[(i,m)] == 1 for i in ('n1','n2') for m in (GEMINI,SOL)))
        result = paired.report(records, loader(), rows)['datasets']
        n5 = result['nutrition5k']
        self.assertEqual(n5['common_complete_reference_ids'], [('n5-good','single')])
        self.assertIsNone(n5['profiles']['sol_portion']['fixed_population_mae'])
        self.assertEqual(n5['profiles']['sol_portion']['calculation_rate'], 0.5)
        self.assertEqual(n5['paired_common_calculated']['sol_portion']['association_row_mae']['weight'], 0.)
        self.assertEqual(result['asa24']['profiles']['sol_portion']['selected_code_hit_associations'], 2)

    def test_portion_contract_failure_is_immutable_per_model_on_resume(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            m = runner.modules(path)
            store = m['newcommon'].Store(path / 'stages', 'synthetic-freeze')
            calls = []
            class Pipe:
                def portion(self, image, identity, config):
                    calls.append(runner.N5_PORTION_MODEL.get())
                    return dict(status='failed', failure='portion_contract', prediction=None)
            row = dict(sample_id='s', image_path='a')
            with patch.object(m['provider'], 'ROOT', path):
                for _ in range(2):
                    pipe = runner.DurablePipe(Pipe(), store, [row], Event())
                    for model in (GEMINI, SOL):
                        self.assertIsNone(pipe.portion('a', {}, runner.C_CONFIG, portion_model=model)['prediction'])
            self.assertEqual(calls, [GEMINI, SOL])
            self.assertIsNone(runner.N5_PORTION_MODEL.get())

    def test_authorization_binds_model_and_scope(self):
        frozen = dict(manifest_sha256='x', round_number=7, rows=[{'sample_id':'s'}],
                      experiment=runner.EXPERIMENT, configs={'asa24':runner.ASA_CONFIG,'nutrition5k':runner.C_CONFIG})
        auth = dict(freeze_sha256=paired.digest(frozen), manifest_sha256='x', round_number=7,
            sample_ids=['s'], variants=list(paired.VARIANTS), models=paired.MODELS,
            n5_portion_models=paired.N5_PORTION_MODELS, destinations=runner.DESTINATIONS,
            max_attempts=3, max_concurrency=32, downloads_authorized=False,
            independent_acceptance='sealed', correction_applied=False,
            experiment=frozen['experiment'], configs=frozen['configs'],
            explicit_user_authorization='synthetic unit test authorization')
        runner.authorize(auth, frozen, True)
        for bad in (None, dict(auth, n5_portion_models={}), dict(auth, explicit_user_authorization='待确认')):
            with self.assertRaises(ValueError):
                runner.authorize(bad, frozen, True)
        with self.assertRaises(ValueError):
            runner.authorize(auth, frozen, False)

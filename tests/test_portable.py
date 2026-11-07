import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import run_mixed_c as run
import score


class Store:
    def __init__(self):
        self.rows = {}

    def get(self, stage, sid, binding):
        saved = self.rows.get((stage, sid))
        if saved is None:
            return None
        assert saved[0] == binding
        return saved[1]

    def put(self, stage, sid, binding, value):
        if (stage, sid) in self.rows:
            if self.rows[(stage, sid)] != (binding, value):
                raise ValueError('Immutable stage changed')
        self.rows[(stage, sid)] = (binding, value)


class PortableTests(unittest.TestCase):
    def test_output_must_stay_in_ignored_runs_without_symlinks(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'runs').mkdir()
            (root / 'examples').mkdir()
            (root / 'runs/link').symlink_to(root / 'examples', target_is_directory=True)
            with patch.object(run, 'ROOT', root):
                self.assertEqual(run.output_directory(root / 'runs/valid'), root / 'runs/valid')
                for p in [root / 'examples/private', root / 'runs/link/private']:
                    with self.assertRaises(ValueError):
                        run.output_directory(p)

    def test_round_rejected_for_single_images_and_custom_blind_list(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'manifest.json'
            p.write_text('[{"sample_id":"a","image_path":"missing.png"}]')
            with self.assertRaisesRegex(ValueError, '--round'):
                run.input_rows(images=['missing.png'], round_number=8)
            with self.assertRaisesRegex(ValueError, '--round'):
                run.input_rows(manifest=p, round_number=8)

    def test_reference_read_once_after_seal_and_changed_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'refs.json'
            p.write_text('[{"association_id":"a","sample_id":"s0","code":"12345678"}]')
            rows = [{'sample_id': 's0'}]
            store = Store()
            with patch.object(Path, 'read_bytes', side_effect=AssertionError('read before seal')):
                with self.assertRaisesRegex(ValueError, 'sealed'):
                    run.sealed_reference_codes(p, rows, store)
            store.put('identity_seal', 'all', {'ids': ['s0']}, {'s0': 'identity_digest'})
            raw = p.read_bytes()
            with patch.object(Path, 'read_bytes', return_value=raw) as reader:
                run.sealed_reference_codes(p, rows, store)
                self.assertEqual(reader.call_count, 1)
            p.write_text(p.read_text().replace('12345678', '23456789'))
            with self.assertRaises(ValueError):
                run.sealed_reference_codes(p, rows, store)

    def test_corrupted_prepared_graph_rejected_even_if_vectors_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'header.bin'
            p.write_bytes(b'original graph')
            expected = {'index/chroma/header.bin': run.file_sha(p)}
            run.verify_prepared_graph(Path(d), expected)
            p.write_bytes(b'changed graph')
            with self.assertRaises(ValueError):
                run.verify_prepared_graph(Path(d), expected)

    def test_failed_asa_identity_is_failed_association_not_normal_miss(self):
        class Pipe:
            def identify_select(self, image, dataset, config):
                return {'status': 'failed', 'failure': 'identity_contract', 'final_codes': []}
            def portion(self, *a, **kw):
                return {'status': 'failed', 'failure': 'identity_failed', 'prediction': None}
        with tempfile.TemporaryDirectory() as d:
            rows = self.image_rows(Path(d), 1)
            result = run.population(rows, 'asa24', Pipe(), Store(), 1,
                lambda: [{'association_id': 'a', 'sample_id': 's0', 'code': '12345678'}])
            a = result[0]['conditional_associations'][0]
            self.assertEqual(a['status'], 'failed')
            self.assertEqual(a['failure'], 'identity_contract')
            self.assertIsNone(a['prediction'])

    def image_rows(self, directory, count=2):
        rows = []
        for i in range(count):
            path = directory / ('image' + str(i) + '.bin')
            path.write_bytes(bytes([i]))
            rows.append(dict(sample_id='s' + str(i), image_path=str(path), image_sha256=run.file_sha(path)))
        return rows

    def test_asa_references_loaded_only_after_all_identities_sealed(self):
        events = []
        store = Store()
        class Pipe:
            catalog = {'asa_options': {'12345678': [{'option_id': 'a', 'grams': 60.}]}}
            def identify_select(self, image, dataset, config):
                events.append('identify')
                return {'status': 'ok', 'final_codes': ['12345678']}
            def portion(self, image, identity, dataset, config, conditional_context=None):
                events.append('conditional' if conditional_context else 'e2e')
                if conditional_context:
                    self.assert_context = conditional_context
                grams = 60. if conditional_context else 100.
                return {'status': 'ok', 'prediction': {k: grams for k in run.TARGETS}, 'portions': []}
        pipe = Pipe()
        with tempfile.TemporaryDirectory() as d:
            rows = self.image_rows(Path(d))
            def references():
                self.assertEqual(events.count('identify'), 2)
                self.assertIn(('identity_seal', 'all'), store.rows)
                events.append('read_references')
                return [{'association_id': 'a', 'sample_id': 's0', 'code': '12345678'},
                        {'association_id': 'b', 'sample_id': 's1', 'code': '99999999'}]
            result = run.population(rows, 'asa24', pipe, store, 2, references)
            self.assertEqual(result[0]['prediction']['weight'], 100.)
            self.assertEqual(result[0]['conditional_associations'][0]['prediction']['weight'], 60.)
            self.assertEqual(result[1]['conditional_associations'][0]['status'], 'not_matched')
            self.assertIsNone(result[1]['conditional_associations'][0]['prediction'])
            self.assertEqual(set(pipe.assert_context), {'code', 'options'})
            self.assertLess(events.index('read_references'), events.index('conditional'))
            # Resumption reuses immutable stages, including original failures.
            before = list(events)
            run.population(rows, 'asa24', pipe, store, 2, references)
            self.assertEqual(events, before + ['read_references'])

    def test_n5_failure_is_null_not_truth_or_zero_and_not_resampled(self):
        events = []
        class Pipe:
            def layout(self, image):
                events.append('layout')
                return {'status': 'failed', 'failure': 'layout_contract'}
            def identify(self, image, config, layout):
                events.append('identity')
                return {'status': 'failed', 'failure': 'layout_contract', 'final_codes': []}
            def portion(self, image, identity, config):
                events.append('portion')
                return {'status': 'failed', 'prediction': None, 'failure': 'identity_failed'}
        with tempfile.TemporaryDirectory() as d:
            rows = self.image_rows(Path(d), 1)
            store = Store()
            result = run.population(rows, 'nutrition5k', Pipe(), store, 1)
            self.assertIsNone(result[0]['prediction'])
            self.assertEqual(result[0]['failure'], 'layout_contract')
            run.population(rows, 'nutrition5k', Pipe(), store, 1)
            self.assertEqual(events, ['layout', 'identity', 'portion'])

    def test_manifest_rejects_truth_and_image_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            row = self.image_rows(root, 1)[0]
            p = root / 'inputs.json'
            p.write_text(json.dumps([dict(row, weight=123)]))
            with self.assertRaises(ValueError):
                run.input_rows(manifest=p)
            p.write_text(json.dumps([dict(row, image_sha256='0' * 64)]))
            with self.assertRaises(ValueError):
                run.input_rows(manifest=p)

    def test_reference_code_file_rejects_current_image_weight(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'refs.json'
            p.write_text(json.dumps([dict(association_id='a', sample_id='s0', code='12345678', grams=40)]))
            with self.assertRaises(ValueError):
                run.reference_codes(p, [{'sample_id': 's0'}])

    def test_score_reports_failed_denominator_and_perunit(self):
        rows = [{'sample_id': 'a', 'status': 'ok', 'prediction': {k: 50. for k in run.TARGETS}},
                {'sample_id': 'b', 'status': 'failed', 'prediction': None}]
        seal = {'records': rows, 'records_sha256': run.digest(rows)}
        labels = {'a': {k: 100. for k in run.TARGETS}, 'b': {k: 1000. for k in run.TARGETS}}
        result = score.score(seal, labels)
        self.assertEqual((result['attempted'], result['scored'], result['failed']), (2, 1, 1))
        self.assertIsNone(result['fixed_population_mae'])
        self.assertEqual(result['successful_subset']['protein']['perunit'], .5)
        rows[1]['prediction'] = {k: 0 for k in run.TARGETS}
        seal['records_sha256'] = run.digest(rows)
        with self.assertRaises(ValueError):
            score.score(seal, labels)

    def test_raw_production_asa_uses_same_grams_but_different_legal_options(self):
        m = run.modules()
        food = {'description': 'plain food', 'density': {k: 10. for k in run.TARGETS[1:]}}
        opts = lambda grams: [{'option_id': str(x), 'description': 'portion', 'grams': x} for x in grams]
        catalog = {'foods': {'12345678': food}, 'standard_options': {'12345678': opts([50., 100.])}}
        calls = []
        pipe = m['pipeline'].make_asa_pipeline(lambda messages: calls.append(messages) or '{"grams":78}', None, catalog, None)
        identity = {'status': 'ok', 'items': [{'item_id': 'i1', 'description': 'food', 'primary_code': '12345678'}], 'final_codes': ['12345678']}
        with patch.object(m['pipeline'].current, 'encode_image', return_value='data:image/png;base64,AA=='):
            first = pipe.portion('unused', identity, 'asa24', run.ASA_CONFIG)
            second = pipe.portion('unused', identity, 'asa24', run.ASA_CONFIG,
                conditional_context={'code': '12345678', 'options': opts([60., 90.])})
        self.assertEqual(first['continuous_grams'], second['continuous_grams'])
        self.assertEqual(first['prediction']['weight'], 100.)
        self.assertEqual(second['prediction']['weight'], 90.)
        self.assertEqual(calls[0], calls[1])


if __name__ == '__main__':
    unittest.main()

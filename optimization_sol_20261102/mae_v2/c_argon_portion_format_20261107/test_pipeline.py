from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import asa_portion as pipeline


class FormatTest(TestCase):
    def run_portion(self, raw):
        run = pipeline.round09
        base = run.original.modules(Path(__file__).parent)
        m = run.argon.argon_modules(base, Path(__file__).parent)
        code = '12345678'
        catalog = {'foods': {code: {'description': 'rice', 'density':
                    dict(energy=100., fat=2., carbs=20., protein=3.)}}}
        identity = dict(status='ok', items=[dict(item_id='i1', description='rice', primary_code=code)],
                        final_codes=[code])
        options = [dict(option_id='a', description='1 cup', grams=123.)]
        pipe = pipeline.make_asa(m, lambda messages: raw, None, catalog, None)
        with patch.object(m['pipeline'].current, 'encode_image', return_value='local test'):
            # The new default chooses a legal portion even if the old config
            # still says continuous_option.
            return pipe.portion('unused', identity, 'asa24', run.CONFIGS['asa24'],
                                conditional_context=dict(code=code, options=options))

    def test_wrapped_valid_option_accepted_in_first_reply(self):
        for raw in ('{"option_id":"a"}', '```json\n{"option_id":"a"}\n```',
                    'The matching portion is {"option_id":"a"}.'):
            with self.subTest(raw=raw):
                result = self.run_portion(raw)
                self.assertEqual(result['prediction']['weight'], 123.)
                self.assertEqual(len(result['stages'][0]['attempts']), 1)

    def test_only_one_legal_object_is_accepted(self):
        for raw in ('```json\n{"option_id":"unknown"}\n```',
                    '```json\n[{"option_id":"a"}]\n```',
                    'Either {"option_id":"a"} or {"option_id":"b"}',
                    '```json\n{"option_id":"a","option_id":"a"}\n```',
                    '```json\n{"option_id":"a","grams":123}\n```',
                    '```json\n{"option_id":"a"', '{"grams":123}'):
            with self.subTest(raw=raw):
                result = self.run_portion(raw)
                self.assertEqual(result['status'], 'failed')
                self.assertIsNone(result['prediction'])
                self.assertEqual(len(result['stages'][0]['attempts']), 2)

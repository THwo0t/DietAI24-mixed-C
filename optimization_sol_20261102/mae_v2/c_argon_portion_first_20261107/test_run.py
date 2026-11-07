"""Check round 09's actual portion branch and shared Argon transport offline."""
from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import httpx
import run


class PortionTest(TestCase):
    def test_two_portion_prompts_use_argon_and_legal_local_grams(self):
        code = '12345678'
        options = [dict(option_id='a', description='2.5 cups', grams=617.25),
                   dict(option_id='b', description='50 g', grams=50.)]
        catalog = {'foods': {code: {'description': 'cooked rice', 'density':
            dict(energy=100., fat=2., carbs=20., protein=3.)}}}
        identity = dict(status='ok', items=[dict(item_id='i1', description='rice', primary_code=code)],
                        final_codes=[code])
        with TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp).resolve()
            base = run.original.modules(root)
            m = run.argon.argon_modules(base, root)
            stack.enter_context(patch.object(m['provider'], 'ROOT', root))
            stack.enter_context(patch.object(m['newcommon'], 'V2', root))
            method = m['pipeline']
            stack.enter_context(patch.object(method.current, 'encode_image',
                                             return_value='data:image/png;base64,dGVzdA=='))
            payloads = []

            def mock(request):
                self.assertEqual(str(request.url), run.original.DESTINATIONS['chat'])
                self.assertEqual(request.headers['Authorization'], 'Bearer mock-vision')
                data = json.loads(request.content)
                self.assertEqual(data['model'], run.argon.ARGON)
                payloads.append(data)
                answer = '{"grams":600}' if len(payloads) == 1 else '{"option_id":"b"}'
                return httpx.Response(200, json={'model': run.argon.ARGON, 'choices': [
                    {'message': {'content': answer}, 'finish_reason': 'stop'}]})

            provider = m['newcommon'].CachedProvider('ASA', allow_paid=True,
                keys=('mock-sol', 'mock-embedding', 'mock-vision'), transport=httpx.MockTransport(mock))
            stack.callback(provider.close)
            pipe = run.make_asa(method, m['stage_routes'].StageChat(provider, 'asa24'),
                                provider.embed, catalog, None)
            for mode, weight in (('continuous_option', 617.25), ('portion_first', 50.)):
                result = pipe.portion('unused.png', identity, 'asa24', {'portion_mode': mode},
                                      conditional_context={'code': code, 'options': options})
                self.assertEqual(result['status'], 'ok')
                self.assertEqual(result['prediction']['weight'], weight)
            self.assertEqual(len(payloads), 2)
            first, second = payloads
            self.assertEqual({k: v for k, v in first.items() if k != 'messages'},
                             {k: v for k, v in second.items() if k != 'messages'})
            continuous = first['messages'][1]['content'][0]['text']
            select = second['messages'][1]['content'][0]['text']
            self.assertNotIn('2.5 cups', continuous)
            self.assertIn('2.5 cups', select)
            self.assertIn('50 g', select)
            for forbidden in ('617.25', '"grams"', 'reference_weight', 'target_option', 'density'):
                self.assertNotIn(forbidden, select)
            self.assertEqual(first['messages'][1]['content'][1:], second['messages'][1]['content'][1:])
            self.assertIs(type(pipe).identify_select, method.ASAPipeline.identify_select)

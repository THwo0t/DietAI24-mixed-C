"""Offline stage routing, exact-cache and restart contract tests."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for name in ('config','provider','newcommon','stage_routes'):
    spec = importlib.util.spec_from_file_location(name, HERE / (name+'.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
import provider as p
import newcommon as common
from stage_routes import StageChat, StageRoutingMixin

IMAGE = [{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,AA=='}}]}]
TEXT = [{'role':'user','content':'rewrite rice'}]
KEYS = ('dummy-sol-secret-10001','dummy-embed-secret-20002','dummy-gemini-secret-30003')


class StageProviderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)/'root'; self.root.mkdir()
        self.patches = [patch.object(p,'ROOT',self.root),patch.object(p,'HERE',self.root),
            patch.object(common,'HERE',self.root/'experiment'),patch.object(common,'V2',self.root),
            patch.object(p,'read_credentials',side_effect=AssertionError('Credential reads forbidden'))]
        for value in self.patches:value.start()
        self.clients=[];self.sent=[]

    def tearDown(self):
        for c in self.clients:c.close()
        for value in reversed(self.patches):value.stop()
        self.tmp.cleanup()

    def client(self, handler=None, cached=False):
        def send(request):
            self.sent.append(request)
            body=json.loads(request.content)
            if handler:return handler(request)
            if request.url.path.endswith('/embeddings'):
                return httpx.Response(200,json={'model':body['model'],'data':[{'embedding':[1.]*3072}]})
            return httpx.Response(200,json={'model':body['model'],'choices':[{'message':{'content':'Explanation {"grams":"001.20"}'},'finish_reason':'stop'}]})
        args=dict(allow_paid=True,keys=KEYS,transport=httpx.MockTransport(send),sleep=lambda _:None)
        c=common.CachedProvider('ASA',**args) if cached else p.Provider(self.root/'run',**args)
        self.clients.append(c);return c

    def test_routes_keys_and_gemini_only_format(self):
        c=self.client()
        cases=[('nutrition5k','recognition',p.VISION_MODEL,2),('asa24','selection',p.VISION_MODEL,2),
            ('nutrition5k','portion',p.VISION_MODEL,2),('asa24','portion',p.CHAT_MODEL,0)]
        for dataset,stage,model,index in cases:
            value=c.chat(IMAGE,dataset=dataset,stage=stage)
            self.assertEqual(value,'{"grams":"001.20"}' if model==p.VISION_MODEL else 'Explanation {"grams":"001.20"}')
        c.chat(TEXT,dataset='asa24',stage='query_rewrite');c.embed(['rice'])
        # Recognition and selection have identical exact payloads and share cache.
        self.assertEqual(len(self.sent),4)
        for request,(model,index) in zip(self.sent,[(p.VISION_MODEL,2),(p.CHAT_MODEL,0),(p.CHAT_MODEL,0),(p.EMBED_MODEL,1)]):
            self.assertEqual(json.loads(request.content)['model'],model)
            self.assertEqual(request.headers['Authorization'],'Bearer '+KEYS[index])
            self.assertNotIn('stage',json.loads(request.content))
            self.assertNotIn('dataset',json.loads(request.content))

    def test_image_missing_unknown_stage_and_forged_route_rejected(self):
        c=self.client()
        for route in ({},{'dataset':'asa24'},{'dataset':'asa24','stage':'unknown'},
                {'dataset':'unknown','stage':'portion'},{'dataset':'asa24','stage':'query_rewrite'}):
            with self.assertRaises(p.ProviderStop):c.chat(IMAGE,**route)
        payload={'model':p.VISION_MODEL,'messages':IMAGE}
        with self.assertRaises(p.ProviderStop):
            c._request('chat',payload,route={'dataset':'asa24','stage':'portion'})
        with self.assertRaises(p.ProviderStop):c._request('chat',payload)
        self.assertFalse(self.sent)

    def test_repair_history_keeps_explicit_stage_and_sol_image_route(self):
        c=self.client()
        repair=IMAGE+[{'role':'assistant','content':'bad'},{'role':'user','content':'repair'}]
        c.chat(repair,dataset='asa24',stage='portion')
        self.assertEqual(json.loads(self.sent[0].content)['model'],p.CHAT_MODEL)

    def test_cache_hash_old_exact_contract_and_stage_observations(self):
        c=self.client(cached=True)
        c.chat(IMAGE,dataset='asa24',stage='recognition')
        request=json.loads(self.sent[0].content)
        cache_id=p.fingerprint({'url':p.CHAT_URL,'payload':request,'namespace':'development'})
        self.assertTrue((self.root/'runtime_cache/responses/chat'/f'{cache_id}.json').is_file())
        c.chat(IMAGE,dataset='nutrition5k',stage='selection')
        self.assertEqual(len(self.sent),1)
        rows=[json.loads(v) for v in (c.run_dir/'observations.jsonl').read_text().splitlines()]
        self.assertEqual([(r['dataset'],r['stage']) for r in rows],[('asa24','recognition'),('nutrition5k','selection')])
        log=json.loads((c.run_dir/'requests.jsonl').read_text().splitlines()[0])
        self.assertEqual(log['stage'],'recognition')
        self.assertEqual(log['dataset'],'asa24')

    def test_context_threads_and_whole_repair_stage(self):
        barrier=threading.Barrier(2);seen=[];lock=threading.Lock()
        class Fake:
            def chat(self,messages,**kwargs):
                with lock:seen.append(kwargs['stage'])
                return kwargs['stage']
        chat=StageChat(Fake(),'asa24')
        def work(name):
            with chat.stage(name):
                barrier.wait()
                a=chat(IMAGE);b=chat(IMAGE)
            return a,b,chat(TEXT)
        with ThreadPoolExecutor(2) as pool:
            a=pool.submit(work,'identify');b=pool.submit(work,'portion_conditional_continuous_option')
            self.assertEqual(a.result(),('recognition','recognition','query_rewrite'))
            self.assertEqual(b.result(),('portion','portion','query_rewrite'))
        class Base:
            def __init__(self,fn):self.chat=fn
            def _stage(self,name,*args,**kwargs):return self.chat(IMAGE),self.chat(IMAGE)
        class Pipeline(StageRoutingMixin,Base):pass
        self.assertEqual(Pipeline(chat)._stage('select'),('selection','selection'))
        self.assertEqual(Pipeline(lambda _: 'offline')._stage('identify'),('offline','offline'))
        self.assertEqual(chat(TEXT),'query_rewrite')

    def test_wrong_served_model_and_persistent_three_attempts(self):
        c=self.client(lambda _:httpx.Response(200,json={'model':'wrong'}))
        with self.assertRaises(p.ProviderStop):c.chat(IMAGE,dataset='asa24',stage='portion')
        for _ in range(2):
            with self.assertRaises(p.ProviderStop):
                self.client(lambda _:httpx.Response(401,json={'error':{'type':'denied'}})).chat(IMAGE,dataset='asa24',stage='portion')
        with self.assertRaisesRegex(p.ProviderStop,'exhausted'):
            self.client().chat(IMAGE,dataset='asa24',stage='portion')
        self.assertEqual(len(self.sent),3)

    def test_unique_json_negative_cases_and_secret_redaction(self):
        for raw in ('[{"x":1}]','[{"x":1}','{"x":1} {"x":2}','{"x":'):
            self.assertEqual(p.extract_unique_json(raw),raw)
        c=self.client(lambda req:httpx.Response(200,json={'model':json.loads(req.content)['model'],
            'usage':{'unexpected':' '.join(KEYS)},'choices':[{'message':{'content':' '.join(KEYS)},'finish_reason':'stop'}]}))
        c.chat(IMAGE,dataset='asa24',stage='recognition')
        for path in self.root.rglob('*.json*'):
            for key in KEYS:self.assertNotIn(key,path.read_text())


if __name__=='__main__':unittest.main(verbosity=2)

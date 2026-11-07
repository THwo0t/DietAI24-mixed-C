"""Offline injected-call tests; no Provider or historical files are modified."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('c_repro_new_pipeline',Path(__file__).with_name('pipeline.py'))
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)

A='12345678';B='23456789';C='34567890'
def catalog():
    return {'foods':{code:{'description':'food '+code,'density':dict(energy=n,fat=n/10,carbs=n/5,protein=n/20)} for code,n in [(A,100.),(B,200.),(C,300.)]}}

class RetrievalTests(unittest.TestCase):
    def test_release_literal_prompt_and_ordered_union_no_seventh_query(self):
        chats=[];embeds=[];search=[]
        def chat(messages):chats.append(messages);return 'one\ntwo\nthree\nfour\nfive\noriginal'
        def embed(q):embeds.extend(q);return list(range(len(q)))
        def find(v):search.append(v);return [A,B] if v%2==0 else [B,C]
        stages=[];mq=p.MultiQuery(chat,embed,find,catalog()['foods'])
        self.assertEqual(mq.retrieve('original',stages,'leaf'),[A,B,C])
        self.assertEqual(len(embeds),6);self.assertEqual(search,list(range(6)))
        self.assertEqual(chats,[[{'role':'user','content':p.QUERY_TEMPLATE.format(question='original')}]])
        self.assertIn('decriptions should docus',chats[0][0]['content'])
        self.assertFalse(stages[0]['include_original'])
    def test_unexpected_count_and_numbering_preserved(self):
        q=[];mq=p.MultiQuery(lambda m:'1. alpha\n2. beta',lambda x:q.extend(x) or [[1]]*len(x),lambda v:[A],catalog()['foods'])
        stages=[];self.assertEqual(mq.retrieve('x',stages,'i'),[A]);self.assertEqual(q,['1. alpha','2. beta'])
        self.assertEqual(stages[0]['query_count'],2)
    def test_eleven_lines_preserve_six_queries_and_remove_only_separators(self):
        calls=[];stages=[];embedded=[]
        lines=['1. alpha','  2. beta  ','3. gamma','4. delta','5. epsilon','6. original']
        raw='\n\n'.join(lines)
        mq=p.MultiQuery(lambda m:calls.append(m) or raw,lambda x:embedded.extend(x) or [[1]]*len(x),lambda x:[A],catalog()['foods'])
        self.assertEqual(mq.retrieve('x',stages,'i'),[A])
        self.assertEqual(len(calls),1);self.assertEqual(embedded,lines)
        self.assertEqual(stages[0]['queries'],lines)
        self.assertEqual(stages[0]['raw_query_lines'],raw.strip().split('\n'))
        self.assertEqual(stages[0]['blank_separator_lines'],[1,3,5,7,9])
        self.assertEqual(stages[0]['attempts'][0]['raw'],raw)
        self.assertEqual(stages[0]['query_count'],6)
    def test_all_empty_lines_remain_recorded_failure_no_embedding(self):
        stages=[]
        mq=p.MultiQuery(lambda m:' \n \t\n ',lambda q:self.fail('empty query sent'),None,catalog()['foods'])
        with self.assertRaises(p.EmptyMultiQueryLine):mq.retrieve('x',stages,'i')
        self.assertEqual(stages[0]['queries'],[])
        self.assertEqual(stages[0]['attempts'][0]['raw'],' \n \t\n ')
        self.assertFalse(stages[0]['attempts'][0]['valid'])
    def test_transport_exception_not_format_repaired(self):
        class Stop(BaseException):pass
        def chat(m):raise Stop()
        with self.assertRaises(Stop):p.MultiQuery(chat,None,None,catalog()['foods']).retrieve('x',[],'i')
    def test_actual_nine_queries_union_is_not_capped_at_32(self):
        foods={str(10000000+i):{} for i in range(36)}
        mq=p.MultiQuery(lambda m:'\n'.join('q'+str(i) for i in range(9)),
            lambda q:list(range(len(q))),lambda v:[str(10000000+v*4+i) for i in range(4)],foods)
        stages=[];codes=mq.retrieve('x',stages,'i')
        self.assertEqual(len(codes),36);self.assertEqual(stages[0]['query_count'],9)

class AccountingTests(unittest.TestCase):
    def leaf(self):return dict(component_id='p1c1',item_id='p1c1',parent_id='p1',parent_description='food',accounting='whole',description='food',broad_query='food',primary_code=A,alternative_codes=[B,C])
    def test_code_mean_counts_one_physical_mass(self):
        pipe=p.ComponentPipeline(None,None,catalog(),None)
        identity={'parents':[{'parent_id':'p1'}],'leaves':[self.leaf()]}
        out=pipe._aggregate(identity,{'p1c1':50.},[],method='total_ratio')
        self.assertEqual(out['prediction']['weight'],50.)
        self.assertEqual(out['prediction']['energy'],100.)
        self.assertEqual(out['raw_leaves'][0]['density']['energy'],200.)
        self.assertEqual(out['raw_leaves'][0]['codes'],[A,B,C])
    def test_four_alternatives_duplicate_or_foreign_code_rejected(self):
        leaf=self.leaf();candidates={'p1c1':[A,B,C]}
        for alts in [[B,C,A],[A],['99999999']]:
            raw=json.dumps({'items':[{'component_id':'p1c1','primary_code':A,'alternative_codes':alts}]})
            with self.assertRaises(p.ContractError):p.ComponentPipeline._parse_selection(raw,[leaf],candidates)
    def test_asa_original_identity_prompt_preserved_and_mq_only_embeddings(self):
        identity={'items':[{'item_id':'i1','description':'food','broad_query':'broad','region_ids':['r1']}]}
        selection={'items':[{'item_id':'i1','primary_code':A,'alternative_codes':[B]}]}
        answers=iter([json.dumps(identity),'one\ntwo',json.dumps(selection)])
        calls=[];embedded=[]
        def chat(messages):calls.append(messages);return next(answers)
        def embed(queries):embedded.extend(queries);return [[1]]*len(queries)
        pipe=p.make_asa_pipeline(chat,embed,catalog(),lambda v:[A,B])
        with patch.object(p.current,'encode_image',return_value='data:image/png;base64,YQ=='):
            result=pipe.identify_select('unused','asa24',{'variant':'optimized','image_mode':'native'})
        self.assertEqual(embedded,['one','two'])
        self.assertEqual(result['final_codes'],[A,B])
        self.assertIn('This is a single-food reference photo',calls[0][1]['content'][0]['text'])
        self.assertEqual([s['stage'] for s in result['stages']],['identify','multiquery','select'])
    def test_asa_empty_query_returns_failed_identity_with_raw(self):
        identity=json.dumps({'items':[{'item_id':'i1','description':'food','broad_query':'food','region_ids':['r1']}]})
        replies=iter([identity,' \n\t\n '])
        pipe=p.make_asa_pipeline(lambda m:next(replies),lambda q:self.fail('empty query embedded'),catalog(),lambda v:[])
        with patch.object(p.current,'encode_image',return_value='data:image/png;base64,YQ=='):
            out=pipe.identify_select('unused','asa24',{'variant':'optimized'})
        self.assertEqual(out['status'],'failed');self.assertEqual(out['failure'],'empty_multiquery_line')
        self.assertEqual(out['stages'][0]['attempts'][0]['raw'],identity)
        self.assertEqual(out['stages'][1]['attempts'][0]['raw'],' \n\t\n ')
        self.assertEqual(out['stages'][1]['queries'],[])
    def test_ASA_preserves_36_candidates_despite_parent_cap_argument(self):
        cat=catalog();cat['foods']={str(10000000+i):deep for i in range(36) for deep in [cat['foods'][A].copy()]}
        identity={'items':[{'item_id':'i1','description':'food','broad_query':'broad','region_ids':['r1']}]}
        selected={'items':[{'item_id':'i1','primary_code':'10000035','alternative_codes':[]}]}
        answers=iter([json.dumps(identity),'\n'.join('q'+str(i) for i in range(9)),json.dumps(selected)])
        pipe=p.make_asa_pipeline(lambda m:next(answers),lambda q:list(range(len(q))),cat,
            lambda v:[str(10000000+v*4+i) for i in range(4)])
        with patch.object(p.current,'encode_image',return_value='data:image/png;base64,YQ=='):
            out=pipe.identify_select('unused','asa24',{'variant':'optimized','candidate_cap':32})
        self.assertEqual(out['status'],'ok');self.assertEqual(len(out['candidates']['i1']),36)
        self.assertEqual(out['final_codes'],['10000035'])
    def test_C_only_empty_query_is_wrapped_other_retrieval_errors_propagate(self):
        pipe=p.make_component_pipeline(None,None,catalog(),None)
        stages=[{'stage':'multiquery','attempts':[{'raw':'a\n\nb','valid':False}],'queries':['a','','b']}]
        with patch.object(p.component.ComponentPipeline,'identify',side_effect=p.EmptyMultiQueryLine(stages)):
            out=pipe.identify('unused',{})
        self.assertEqual(out['failure'],'empty_multiquery_line');self.assertEqual(out['stages'],stages)
        for error in [p.ContractError('invalid_multiquery_candidates'),ValueError('integrity')]:
            with patch.object(p.component.ComponentPipeline,'identify',side_effect=error):
                with self.assertRaises(type(error)):pipe.identify('unused',{})

if __name__=='__main__':unittest.main()

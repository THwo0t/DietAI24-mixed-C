"""C layout/quantity with release MultiQuery retrieval and same-food code means.

No provider construction, credentials, downloads, labels, or index mutation.
search4(vector) is injected and must return the original Chroma L2 top four codes.
"""
from copy import deepcopy
import ast
import importlib.util
import json
from pathlib import Path
import sys

V2=Path(__file__).resolve().parents[1]
REPO=V2.parents[1]
sys.path[:0]=[str(V2/'src'),str(V2)]
import opt_pipeline as current
from stage_routes import StageRoutingMixin

class RoutedCurrentPipeline(StageRoutingMixin, current.Pipeline):
    pass

def _snapshot():
    directory=V2/'deliverables/j_c_20261104/C/source/optimization_sol_20261102/mae_v2/src'
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    opt=load('c_repro_snapshot_opt',directory/'opt_pipeline.py')
    old=sys.modules.get('opt_pipeline')
    try:
        sys.modules['opt_pipeline']=opt
        component=load('c_repro_snapshot_components',directory/'pipeline_v2.py')
    finally:
        if old is None:sys.modules.pop('opt_pipeline',None)
        else:sys.modules['opt_pipeline']=old
    return opt,component

historical,component=_snapshot()
ContractError=historical.ContractError

class EmptyMultiQueryLine(ContractError):
    """Only this model-output error is eligible for whole-identity fallback."""
    def __init__(self,stages):
        super().__init__('empty_multiquery_line')
        self.stages=deepcopy(stages)

def release_query_template():
    """Read the literal from release source, preserving whitespace and typos."""
    tree=ast.parse((REPO/'code/rag_food_code.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='setup_retrieval_prompt')
    call=function.body[0].value
    return ast.literal_eval(next(k.value for k in call.keywords if k.arg=='template'))

QUERY_TEMPLATE=release_query_template()

class MultiQuery:
    """One original PromptTemplate user message; LangChain line parser/union order.

    include_original=False: prompt itself asks for original as sixth entry.
    Sol adapter removes separator-only lines after the original line parser;
    every nonempty line is preserved verbatim, without six-line enforcement.
    """
    def __init__(self,chat,embed,search4,foods):
        self.chat,self.embed,self.search4,self.foods=chat,embed,search4,foods
    def retrieve(self,description,stages,item_id):
        prompt=QUERY_TEMPLATE.format(question=description)
        raw=self.chat([{'role':'user','content':prompt}])
        if not isinstance(raw,str):raise ContractError('invalid_multiquery_output_type')
        raw_query_lines=raw.strip().split('\n')
        queries=[line for line in raw_query_lines if line.strip()]
        blank_separator_lines=[i for i,line in enumerate(raw_query_lines) if not line.strip()]
        stage={'stage':'multiquery','item_id':item_id,'attempts':[{'raw':raw,'valid':True}],
            'raw_query_lines':raw_query_lines,'blank_separator_lines':blank_separator_lines,
            'queries':queries,'query_count':len(queries),'include_original':False,'k':4,'metric':'squared_l2'}
        stages.append(stage)
        if not queries:
            stage.update(failure='empty_multiquery_line',protocol_deviation='All query lines empty; no query generated or redrawn')
            stage['attempts'][0]['valid']=False
            raise EmptyMultiQueryLine(stages)
        vectors=[]
        for start in range(0,len(queries),12):
            batch=self.embed(queries[start:start+12])
            if len(batch)!=len(queries[start:start+12]):raise ContractError('embedding_count_mismatch')
            vectors.extend(batch)
        codes=[];per_query=[]
        for vector in vectors:
            found=list(self.search4(vector))
            if not 1<=len(found)<=4 or len(set(found))!=len(found) or any(c not in self.foods for c in found):
                raise ContractError('invalid_multiquery_candidates')
            per_query.append(found)
            codes.extend(c for c in found if c not in codes)
        stage.update(codes_by_query=per_query,codes=codes,query_count=len(queries))
        return codes

class ComponentPipeline(StageRoutingMixin, component.ComponentPipeline):
    def __init__(self,chat,embed,catalog,retriever):
        """retriever is MultiQuery, not the historical hybrid Retriever."""
        super().__init__(chat,embed,catalog,retriever)
    def _retrieve(self,leaves,stages,stage_name='retrieval'):
        return {leaf['component_id']:self.retriever.retrieve(leaf['description'],stages,leaf['component_id']) for leaf in leaves}
    @staticmethod
    def _parse_selection(raw,leaves,candidates):
        data=historical._object(historical._json(raw),{'items'})
        indexed={l['component_id']:l for l in leaves};selected={}
        if not isinstance(data['items'],list) or len(data['items'])!=len(leaves):raise ContractError('missing_or_extra_selection')
        for row in data['items']:
            historical._object(row,{'component_id','primary_code','alternative_codes'})
            cid=row['component_id'];alts=row['alternative_codes']
            if not isinstance(cid,str) or cid not in indexed or cid in selected:raise ContractError('unknown_or_duplicate_selection')
            if not isinstance(alts,list) or len(alts)>2:raise ContractError('too_many_alternative_codes')
            codes=[historical._code(row['primary_code'])]+[historical._code(c) for c in alts]
            if len(set(codes))!=len(codes) or not set(codes)<=set(candidates[cid]):raise ContractError('selected_code_outside_candidates')
            selected[cid]={**deepcopy(indexed[cid]),'primary_code':codes[0],'alternative_codes':codes[1:]}
        return [selected[l['component_id']] for l in leaves]
    def _select(self,leaves,candidates,images,stages,name='select'):
        context=[{'component_id':l['component_id'],'description':l['description'],
            'parent_description':l['parent_description'],'accounting':l['accounting'],
            'candidates':self._candidates(candidates[l['component_id']])} for l in leaves]
        prompt=('Select one to three candidate codes for every fixed accounting leaf, best match first. '
            'Alternatives describe uncertain identities of the SAME physical food, never extra portions. '
            'Preserve leaf IDs and interpretation; do not add, merge, split or change quantities. '
            'An ingredient must not become an entire mixed dish. Use visually supported preparation. '
            'Do not choose codes to optimize nutrient values. Return ONLY JSON '
            '{"items":[{"component_id":"p1c1","primary_code":"12345678","alternative_codes":[]}]}. Context: '+json.dumps(context))
        return self._stage(name,prompt,images,lambda raw:self._parse_selection(raw,leaves,candidates),stages)
    def identify(self,*args,**kwargs):
        try:
            result=super().identify(*args,**kwargs)
        except EmptyMultiQueryLine as exc:
            return self._result(exc.stages,status='failed',failure='empty_multiquery_line',
                parents=[],items=[],leaves=[],candidates={},retrieved_codes=[],
                final_codes=[],selected_codes=[])
        if result['status']=='ok':
            result['final_codes']=result['selected_codes']=list(dict.fromkeys(c for leaf in result['leaves'] for c in [leaf['primary_code']]+leaf['alternative_codes']))
        return result
    def _checked_identity(self,image_path,identity):
        parents,leaves=super()._checked_identity(image_path,identity)
        saved={l['component_id']:l for l in identity['leaves']}
        for leaf in leaves:
            row=saved[leaf['component_id']]
            parsed=self._parse_selection(json.dumps({'items':[{'component_id':leaf['component_id'],
                'primary_code':row['primary_code'],'alternative_codes':row.get('alternative_codes',[])}]}),[leaf],identity['candidates'])[0]
            leaf.update(primary_code=parsed['primary_code'],alternative_codes=parsed['alternative_codes'])
        return parents,leaves
    def _aggregate(self,identity,masses,stages,**extra):
        # Historical aggregate validates masses/ownership. Revalue each ONE mass.
        result=super()._aggregate(identity,masses,stages,**extra)
        byid={l['component_id']:l for l in identity['leaves']}
        totals=dict.fromkeys(historical.TARGETS,0.)
        for p in result['portions']:
            leaf=byid[p['component_id']];codes=[leaf['primary_code']]+leaf.get('alternative_codes',[])
            if not 1<=len(codes)<=3 or len(set(codes))!=len(codes):raise ContractError('invalid_mean_codes')
            density={k:sum(historical._number(self.foods[c]['density'][k]) for c in codes)/len(codes) for k in historical.NUTRIENTS}
            p.update(codes=codes,density=density)
            totals['weight']+=p['grams']
            for k in historical.NUTRIENTS:totals[k]+=p['grams']*density[k]/100
        result.update(prediction=totals,raw_prediction=deepcopy(totals),leaves=deepcopy(result['portions']),raw_leaves=deepcopy(result['portions']))
        return result

class ASAPipeline(RoutedCurrentPipeline):
    """Retain optimized identity/selection and downstream portion interfaces.

    _stage intercepts the original retrieval boundary after optimized identity;
    the old hybrid embed/search is replaced without changing identity prompt.
    """
    def identify_select(self,image_path,dataset,config):
        if dataset!='asa24' or config.get('variant','optimized')!='optimized':raise current.ContractError('ASA_optimized_only')
        # A per-call proxy receives parent hybrid requests but never pays the old
        # broad+specific embeddings. Parent metadata is corrected below.
        local_stages=[]
        mq=self.retriever
        class Proxy:
            def hybrid(self,queries,vectors,cap=32):return mq.retrieve(queries[0],local_stages,'i1')
        delegate=RoutedCurrentPipeline(self.chat,lambda queries:[None]*len(queries),self.catalog,Proxy())
        # Keep the original identity's actual attempt/raw audit if MQ fails.
        original_stage=delegate._stage
        identity_stages=[]
        def observed_stage(name,prompt,image,parser,stages):
            value=original_stage(name,prompt,image,parser,stages)
            if name=='identify':identity_stages[:]=deepcopy(stages)
            return value
        delegate._stage=observed_stage
        try:
            result=delegate.identify_select(image_path,dataset,config)
        except EmptyMultiQueryLine as exc:
            return delegate._result(identity_stages+exc.stages,status='failed',failure='empty_multiquery_line',
                description=None,items=[],candidates={},retrieved_codes=[],selected_codes=[],final_codes=[])
        result['stages']=[s for s in result['stages'] if s['stage']!='retrieval']
        position=next((i for i,s in enumerate(result['stages']) if s['stage']=='select'),len(result['stages']))
        result['stages'][position:position]=local_stages
        return result

def make_component_pipeline(chat,embed,catalog,search4):
    return ComponentPipeline(chat,embed,catalog,MultiQuery(chat,embed,search4,catalog['foods']))

def make_asa_pipeline(chat,embed,catalog,search4):
    return ASAPipeline(chat,embed,catalog,MultiQuery(chat,embed,search4,catalog['foods']))

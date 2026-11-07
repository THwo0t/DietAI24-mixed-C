"""Offline negative paths and one-application calibration checks."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('repro_score_test_target', Path(__file__).with_name('score.py'))
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def fixture():
    code = '12345678'
    options = [dict(option_id=str(g), description=str(g), grams=float(g)) for g in (100,110,125)]
    density = dict(energy=100., fat=2., carbs=10., protein=5.)
    catalog = dict(foods={code:dict(density=density)}, asa_options={code:options})
    rows=[];labels={};records=[]
    for n in range(12):
        sid = 'synthetic-'+str(n)
        row=dict(sample_id=sid,dataset='asa24',split='dev',fold=n//6,group_id='g'+str(n))
        rows.append(row)
        truth=dict(weight=125.,**{k:1.25*v for k,v in density.items()})
        labels[sid]=dict(food_code=code,truth=truth)
        pred=dict(weight=100.,**density)
        branch=dict(code=code,continuous_grams=100.,option_id='100',prediction=pred,fallback=False)
        records.append({**row,'final_codes':[code],'primary_code':code,'retrieved_codes':[code],
            'prediction':copy.deepcopy(pred),'raw_prediction':copy.deepcopy(pred),
            'conditional':copy.deepcopy(branch),'e2e':copy.deepcopy(branch),
            'fallback':False,'native_complete':True,'repair_used':False})
    return rows,labels,records,catalog


class ScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mass=s.historical_mass_module()

    def test_ASA_unshrunk_training_factor_applies_to_original_grams_once(self):
        rows,labels,records,catalog=fixture()
        original=copy.deepcopy(records)
        out=s.asa_full_correction(rows,labels,records,catalog,self.mass)
        params={p['fold']:p for p in out['fold_parameters']}
        for r in out['predictions']:
            p=params[r['fold']]
            expected=self.mass.nearest_option(100*p['raw_factor'],catalog['asa_options']['12345678'])
            self.assertEqual(r['conditional']['option_id'],expected['option_id'])
            self.assertEqual(r['conditional']['prediction']['weight'],125.)
            self.assertEqual(r['conditional']['prediction']['protein'],6.25)
            self.assertEqual(r['conditional']['continuous_grams'],100.)
            self.assertEqual(r['e2e']['prediction']['weight'],100.)
            self.assertFalse(set(p['training_groups']) & set(p['heldout_groups']))
        self.assertEqual(records,original)

    def test_missing_Exact_branch_is_rejected_before_legacy_autofill(self):
        rows,labels,records,catalog=fixture()
        records[0]['conditional']=None
        with self.assertRaisesRegex(ValueError,'Exact_requires'):
            s.asa_full_correction(rows,labels,records,catalog,self.mass)

    def test_nonExact_branch_cannot_be_used(self):
        rows,labels,records,catalog=fixture()
        records[0]['final_codes']=[]
        with self.assertRaisesRegex(ValueError,'nonExact'):
            s.asa_full_correction(rows,labels,records,catalog,self.mass)

    def test_conditional_fallback_stays_fallback_and_gets_legal_median(self):
        rows,labels,records,catalog=fixture()
        records[0]['conditional'].update(fallback=True,continuous_grams=None,fallback_reason='format')
        out=s.asa_full_correction(rows,labels,records,catalog,self.mass)
        self.assertTrue(out['predictions'][0]['conditional']['fallback'])
        self.assertEqual(out['predictions'][0]['conditional']['option_id'],'110')

    def test_N5_average_density_and_mass_scaled_once_with_historical_half_factor(self):
        rows,labels,records,_=fixture()
        for row,record in zip(rows,records):
            row['dataset']=record['dataset']='nutrition5k'
            labels[row['sample_id']].pop('food_code')
            record['conditional']=record['e2e']=None
            record['raw_leaves']=[dict(item_id='i1',grams=100.,codes=['12345678','87654321'],
                density=dict(energy=100.,fat=2.,carbs=10.,protein=5.))]
        result=self.mass.crossvalidate(rows,labels,records,calibrate=True,dataset='nutrition5k')
        params={p['fold']:p for p in result['fold_parameters']}
        for record in result['predictions']:
            factor=params[record['fold']]['mass_factor']
            self.assertAlmostEqual(factor,1+.5*(params[record['fold']]['raw_factor']-1))
            self.assertAlmostEqual(record['prediction']['weight'],100*factor)
            self.assertAlmostEqual(record['prediction']['protein'],5*factor)
            self.assertEqual(record['raw_leaves'][0]['grams'],100.)

    def test_recognition_is_reported_without_15percent_gate(self):
        legacy=dict(passed_checks=20,recognition_pass=False,joint_pass=False,
            datasets=dict(asa24=dict(exact_n=1,exact_rate=.01,minimum_exact_n=80,recognition_pass=False)))
        with patch.object(s.opt_metrics,'evaluate',return_value=legacy):
            result=s.evaluate_without_recognition_gate([],{},[])
        self.assertTrue(result['joint_pass'])
        self.assertEqual(result['datasets']['asa24']['exact_n'],1)
        self.assertNotIn('recognition_pass',result['datasets']['asa24'])

    def test_incomplete_records_rejected_before_scoring(self):
        with tempfile.TemporaryDirectory() as root,patch.object(s,'HERE',Path(root)), \
             patch.object(s,'load',side_effect=[[],{}]):
            with self.assertRaisesRegex(ValueError,'complete_fixed_population'):
                s.completed_records([], 'N5', 'frozen')


if __name__=='__main__':unittest.main()

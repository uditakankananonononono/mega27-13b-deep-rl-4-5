import csv,json,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from net import QNet
from parkinson import transition,trajectory,ACTIONS
from sepsis import load,value,policy_value
from geo_analysis import run as geo_run
ROOT=Path(__file__).resolve().parents[1]
class ResearchTests(unittest.TestCase):
 def test_data_counts_and_uniqueness(self):
  for disease in ('sepsis','parkinson'):
   with (ROOT/'data'/disease/'accession_features.csv').open() as f: rows=list(csv.DictReader(f))
   self.assertEqual(len(rows),150);self.assertEqual(len({r['accession'] for r in rows}),150)
   self.assertEqual(len([k for k in rows[0] if k not in ('accession','label','cohort')]),48)
 def test_accession_prefix(self):
  for disease,prefix in [('sepsis','GSM'),('parkinson','GSM')]:
   with (ROOT/'data'/disease/'accession_features.csv').open() as f:
    self.assertTrue(all(r['accession'].startswith(prefix) for r in csv.DictReader(f)))
 def test_hash_derived(self):
  import hashlib
  for disease in ('sepsis','parkinson'):
   d=ROOT/'data'/disease; meta=json.loads((d/'provenance.json').read_text());self.assertEqual(hashlib.sha256((d/'accession_features.csv').read_bytes()).hexdigest(),meta['derived_sha256'])
 def test_negative_disclosure(self):
  for name in ('sepsis','parkinson'):
   result=json.loads((ROOT/'results'/f'{name}.json').read_text());self.assertIn('model',result)
  p=json.loads((ROOT/'results'/'parkinson.json').read_text());self.assertIn('NOT COMPARABLE',p['published_benchmark_comparison'])
 def test_net_shapes_and_fit(self):
  net=QNet(3,3,width=8,seed=4);x=np.random.default_rng(3).normal(size=(50,3));y=x@np.array([[1.,0.,-.5],[0.,1.,0.],[.2,-.5,1.]])
  before=np.mean((net.predict(x)-y)**2);net.fit(x,y,epochs=500,lr=.004,seed=5);self.assertLess(np.mean((net.predict(x)-y)**2),before*.3)
 def test_transition_constraints(self):
  for action in range(3):
   s,r=transition(np.array([.7,.4,.5]),action,.0)
   self.assertTrue(np.all(np.isfinite(s)));self.assertTrue(np.all(s>=0));self.assertLessEqual(s[0],1.5);self.assertLess(r,0)
 def test_stimulation_effect(self):
  state=np.array([.7,.4,.5]);low,_=transition(state,0,0);high,_=transition(state,2,0);self.assertLess(high[0],low[0])
 def test_seed_reproducibility(self):
  self.assertEqual(trajectory(lambda s: 0,60100),trajectory(lambda s: 0,60100))
 def test_sepsis_baselines(self):
  result=json.loads((ROOT/'results'/'sepsis.json').read_text())
  for key,expected in [('random',.78),('expert',.78),('optimal',.88)]:self.assertLess(abs(result['our_exact_values'][key]-expected),.01)
  self.assertLess(result['neural_greedy']['expected_return'],result['our_exact_values']['optimal'])
 def test_native_simulator_audit(self):
  d=json.loads((ROOT/'results'/'sepsis_native.json').read_text())
  self.assertTrue(d['exact_table_rows_equal_official_gym_env'])
  self.assertEqual(set(d['rollouts']),{'random','optimal','expert'})
  for key,item in d['rollouts'].items():
   self.assertEqual(item['n'],200)
   self.assertLess(abs(item['mean_return']-d['exact_table_expected_returns'][key]),3*item['binomial_se_return_approx'])
 def test_sepsis_pivot_negative(self):
  base=json.loads((ROOT/'results'/'sepsis.json').read_text())
  pivot=json.loads((ROOT/'results'/'sepsis_pivot.json').read_text())
  self.assertEqual(pivot['feature_dim'],47)
  self.assertLess(pivot['neural_expected_return'],base['our_exact_values']['optimal'])
  self.assertLess(pivot['neural_expected_return'],base['neural_greedy']['expected_return'])
  self.assertAlmostEqual(sum(x['initial_mass'] for x in pivot['sofa_initial_state_strata'].values()),1.0,places=10)
 def test_parkinson_pivot_tradeoff(self):
  d=json.loads((ROOT/'results'/'parkinson_pivot.json').read_text())
  self.assertEqual(d['sample_size'],100)
  self.assertLess(d['evaluation']['0.12']['paired_heavy_minus_original_mean_reward'],0)
  self.assertGreater(d['evaluation']['0.4']['paired_heavy_minus_original_mean_reward'],0)
  self.assertLess(d['evaluation']['0.12']['heavier_weight_0_40']['mean_energy'],d['evaluation']['0.12']['original_weight_0_12']['mean_energy'])
 def test_tool_count_no_padding(self):
  tools=json.loads((ROOT/'evidence'/'scientific_tools.json').read_text());self.assertEqual(len({t['name'] for t in tools['used']}),tools['verified_distinct_count']);self.assertEqual(tools['verified_distinct_count'],6);self.assertEqual(tools['project_counts'],{'sepsis':6,'parkinson':4})
 def test_descriptive_probes_not_treatment(self):
  for disease in ('sepsis','parkinson'):
   result=json.loads((ROOT/'results'/f'{disease}_geo.json').read_text());self.assertEqual(result['accessions_used'],150);self.assertEqual(result['features_used'],48);self.assertEqual(result['test_n'],45);self.assertTrue(0<=result['heldout_auc']<=1)
if __name__=='__main__':unittest.main()

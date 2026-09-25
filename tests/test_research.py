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
   self.assertEqual(len(rows),{'sepsis':479,'parkinson':438}[disease]);self.assertEqual(len({r['accession'] for r in rows}),{'sepsis':479,'parkinson':438}[disease])
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
 def test_sepsis_model_sensitivity_scoped(self):
  for c in ('10','100'):
   name='sepsis_model_sensitivity_10.json' if c=='10' else 'sepsis_model_sensitivity.json'
   d=json.loads((ROOT/'results'/name).read_text());self.assertEqual(d['draws'],12)
   self.assertEqual(d['pseudo_concentration'],int(c))
   self.assertEqual(len(d['perturbed_fixed_policy_values']),12)
   self.assertTrue(all(0<=v<=1 for v in d['perturbed_fixed_policy_values']))
   self.assertIn('NOT a clinical confidence interval',d['limitation'])
 def test_mimic_demo_scope_and_counts(self):
  d=json.loads((ROOT/'results'/'sepsis_mimic_demo.json').read_text())
  self.assertEqual(d['sample_subjects'],100)
  self.assertEqual(d['suspected_sepsis_icd9_admissions'],38)
  self.assertEqual(d['selected_input_rows'],1728)
  self.assertFalse(json.loads((ROOT/'evidence'/'gates.json').read_text())['sepsis']['public_mimic_demo_policy_validated'])
 def test_geo_cohort_split_audits(self):
  s=json.loads((ROOT/'results'/'sepsis_geo_cohort.json').read_text())
  p=json.loads((ROOT/'results'/'parkinson_geo_cohort.json').read_text())
  self.assertEqual((s['train_count'],s['test_count']),(263,216))
  self.assertEqual((p['train_count'],p['test_count']),(293,75))
  self.assertGreater(p['cohort_heldout_auc'],.5)
  for d,expected in ((s,479),(p,438)):self.assertEqual(d['accessions_used'],expected);self.assertEqual(len(d['test_accessions']),d['test_count'])
 def test_independent_geo_series_scope(self):
  d=json.loads((ROOT/'results'/'geo_independent.json').read_text());p=d['parkinson'];s=d['sepsis']
  self.assertEqual((p['source_gsm_count'],p['independent_eligible_pd_healthy'],p['shared_probe_count']),(105,72,13))
  self.assertEqual((s['source_gsm_count'],s['measured_probe_rows'],s['unique_subject_group_ids'],s['sepsis_day1_count'],s['day1_unique_group_ids']),(163,24840,54,35,35))
  self.assertEqual(s['day1_labels']['disease status: sepsis nonsurvivor'],9)
  self.assertEqual(s['first_finite_numeric_probe_count'],48)
  self.assertTrue(0<=p['external_auc']<=1)
  self.assertIn('Neither source validates',d['limitation'])
 def test_public_eicu_static_cohort_not_policy_data(self):
  d=json.loads((ROOT/'results'/'sepsis_dryad_eicu.json').read_text())
  self.assertEqual((d['rows'],d['columns']),(13717,169))
  self.assertEqual(d['year_counts'],{'2014':6397,'2015':7320})
  self.assertEqual(d['vasopressor_24h_value_counts']['1.0'],98)
  self.assertEqual(d['descriptive_24h_vasopressor_by_28day_outcome']['1.0']['deaths'],18)
  self.assertIn('confounded by indication',d['limitations'])
  self.assertEqual(d['policy_validation'],'no')
  self.assertEqual(json.loads((ROOT/'evidence'/'gates.json').read_text())['sepsis']['policy_trajectory_datasets'],0)
 def test_sepsis_paired_perturbations_not_patient_intervals(self):
  d=json.loads((ROOT/'results'/'sepsis_policy_perturb_compare.json').read_text())
  self.assertEqual(d['draws'],12)
  self.assertEqual(len(d['paired_centroid_minus_onehot']),12)
  self.assertTrue(all(v<0 for v in d['paired_centroid_minus_onehot']))
  for a,b,c in zip(d['perturbed_model_values']['centroid'],d['perturbed_model_values']['onehot'],d['paired_centroid_minus_onehot']):
   self.assertAlmostEqual(a-b,c)
  self.assertIn('NOT independent hospitals',d['limitation'])
 def test_sepsis_equal_epoch_feature_tradeoff(self):
  d=json.loads((ROOT/'results'/'sepsis_budget_match.json').read_text())
  self.assertEqual(len(d['split_results']),3)
  for row in d['split_results']:
   a=row['controllers']['onehot'];b=row['controllers']['centroid']
   self.assertEqual(a['train_match'],1.0)
   self.assertGreater(b['heldout_match'],a['heldout_match'])
   self.assertLess(b['model_return'],a['model_return'])
  self.assertIn('not equal parameter budget',d['limitation'])
 def test_sepsis_exact_occupancy_identity(self):
  d=json.loads((ROOT/'results'/'sepsis_occupancy_audit.json').read_text())
  self.assertEqual(len(d['splits']),3)
  for split in d['splits']:
   for name in ('onehot','centroid'):
    p=split['policies'][name]
    self.assertAlmostEqual(p['visit_weighted_exact_q_shortfall'],d['optimal_model_value']-p['model_value'],places=8)
    self.assertAlmostEqual(p['train_visit_weighted_shortfall']+p['heldout_visit_weighted_shortfall'],p['visit_weighted_exact_q_shortfall'],places=8)
    self.assertGreater(p['visits_per_initial_case'],10)
   self.assertGreater(split['policies']['centroid']['visit_weighted_exact_q_shortfall'],split['policies']['onehot']['visit_weighted_exact_q_shortfall'])
  self.assertIn('cannot estimate patient',d['limitations'])
 def test_sepsis_centroid_holdout_tradeoff(self):
  c=json.loads((ROOT/'results'/'sepsis_centroid_holdout.json').read_text())
  o=json.loads((ROOT/'results'/'sepsis_state_holdout.json').read_text())
  for a,b in zip(c['splits'],o['splits']):
   self.assertEqual(a['test_states'],b['test_states'])
   self.assertGreater(a['heldout_action_match'],b['heldout_action_match'])
   self.assertLess(a['full_model_expected_return'],b['full_model_expected_return'])
  self.assertIn('Neither independent transition model nor patient',c['limitation'])
 def test_sepsis_state_holdout_exposes_memorization(self):
  d=json.loads((ROOT/'results'/'sepsis_state_holdout.json').read_text())
  self.assertEqual(len(d['splits']),3)
  for split in d['splits']:
   self.assertEqual(len(split['train_states']),570);self.assertEqual(len(split['test_states']),143)
   self.assertFalse(set(split['train_states'])&set(split['test_states']))
   self.assertEqual(split['train_action_match'],1.0)
   self.assertLess(split['heldout_action_match'],.25)
   self.assertLess(split['full_model_expected_return'],d['exact_model_optimal_value'])
  self.assertIn('not external model generalization',d['limitation'])
 def test_sepsis_rank_pivot_is_model_memorization(self):
  d=json.loads((ROOT/'results'/'sepsis_rank_pivot.json').read_text())
  self.assertEqual(set(d['models']),{'32','64','128'})
  for m in d['models'].values():
   self.assertEqual(m['exact_optimal_action_match'],1.0)
   self.assertAlmostEqual(m['model_expected_return'],d['initial_state_optimal_value'],places=9)
  self.assertIn('training-state memorization',d['selection_note'])
 def test_sepsis_neural_action_rank_gap(self):
  d=json.loads((ROOT/'results'/'sepsis_rank_audit.json').read_text())
  self.assertAlmostEqual(d['model_policy_value'],json.loads((ROOT/'results'/'sepsis.json').read_text())['neural_greedy']['expected_return'],places=9)
  self.assertGreater(d['state_regret_under_exact_q']['positive_regret_fraction'],.6)
  self.assertLess(d['initial_mass_regret'],d['optimal_model_value']-d['model_policy_value'])
  self.assertIn('not clinical regret',d['limitations'])
 def test_second_sepsis_series_perfect_auc_flagged(self):
  d=json.loads((ROOT/'results'/'sepsis_second_series.json').read_text())
  self.assertEqual((d['day1_subject_count'],d['day1_deaths'],d['first_finite_probe_count']),(35,9,48))
  self.assertEqual(d['out_of_fold_auc'],1.0)
  self.assertIn('suspicious',d['limitations'])
  self.assertIn('precede all 26 eligible survivor',d['source_metadata_ordering'])
 def test_dryad_adaptive_action_trace_not_policy_validation(self):
  d=json.loads((ROOT/'results'/'parkinson_dryad_adaptive.json').read_text())
  self.assertEqual(d['zip_sha256'],'0f5f17fc319bc94b1146bad570e27a8d71807a5dc6d4721493c25595de6880e4')
  self.assertEqual([a['action_rows'] for a in d['hands']],[4196,4193])
  self.assertEqual(d['dataset_people_as_readme'],1)
  self.assertEqual([a['neural_time_overlap_with_action_example_seconds'] for a in d['hands']],[0,0])
  self.assertEqual(d['policy_validation'],'no')
  self.assertEqual(json.loads((ROOT/'evidence'/'gates.json').read_text())['parkinson']['policy_trajectory_datasets'],0)
 def test_tremordb_fixed_conditions_not_adaptive(self):
  d=json.loads((ROOT/'results'/'parkinson_tremordb.json').read_text())
  self.assertEqual((d['record_count'],d['subject_count'],d['pair_count']),(55,15,25))
  self.assertEqual(d['paired_by_medication']['True']['pair_count'],13)
  self.assertEqual(d['paired_by_medication']['False']['pair_count'],12)
  self.assertIn('No LFP beta',d['limitation'])
  self.assertEqual(json.loads((ROOT/'evidence'/'gates.json').read_text())['parkinson']['policy_trajectory_datasets'],0)
 def test_native_adam_pivot_all_off_negative(self):
  d=json.loads((ROOT/'results'/'parkinson_native_reinforce_adam.json').read_text())
  self.assertEqual(len(d['training']),3);self.assertEqual(d['eval_steps'],1000)
  self.assertEqual(d['eval_action_on_fraction'],0)
  self.assertAlmostEqual(d['eval_mean_reward'],json.loads((ROOT/'results'/'parkinson_env0_off_1000.json').read_text())['mean_step_reward'])
  self.assertGreater(d['parameter_drift_l2_from_initial']['w1'],.2)
  self.assertIn('All-OFF deterministic collapse',d['limitations'])
 def test_native_deep_rl_short_pilot_scope(self):
  d=json.loads((ROOT/'results'/'parkinson_native_reinforce.json').read_text())
  self.assertEqual(len(d['training']),4);self.assertEqual(d['eval_steps'],1000)
  self.assertEqual(d['eval_seed'],222);self.assertEqual(len(d['trained_weight_sha256']),64)
  self.assertIn('No evidence of meaningful learning',d['limitations'])
  self.assertLess(max(d['parameter_drift_l2_from_initial'].values()),.0001)
  self.assertFalse(json.loads((ROOT/'evidence'/'gates.json').read_text())['parkinson']['published_benchmark_comparable'])
 def test_native_upstream_psd_distinct_estimands(self):
  d=json.loads((ROOT/'results'/'parkinson_published_psd_seed222.json').read_text())
  self.assertAlmostEqual(100*d['high_beta_psd']/d['off_beta_psd'],d['high_pct_of_off'])
  self.assertGreater(d['high_pct_of_off'],25)
  self.assertIn('no learned RL controller',d['limitation'])
  self.assertEqual(d['n_steps'],5555)
 def test_adaptive_native_threshold_honest(self):
  d=json.loads((ROOT/'results'/'parkinson_env0_threshold_1200_seed223.json').read_text())
  self.assertEqual(d['steps'],1200);self.assertTrue(0<d['action_on_fraction']<1)
  self.assertIn('NOT deep RL',d['policy'])
  self.assertTrue(d['energy']<6000)
 def test_full_env0_horizon_single_seed_not_benchmark(self):
  off=json.loads((ROOT/'results'/'parkinson_env0_full_off_seed222.json').read_text())
  high=json.loads((ROOT/'results'/'parkinson_env0_full_high_seed222.json').read_text())
  self.assertEqual(off['steps_completed'],5555);self.assertEqual(high['steps_completed'],5555)
  self.assertEqual(off['seed'],high['seed']);self.assertEqual(off['physical_volts'],0);self.assertEqual(high['physical_volts'],5)
  self.assertLess(high['mean_step_beta_component'],off['mean_step_beta_component'])
  gate=json.loads((ROOT/'evidence'/'gates.json').read_text())['parkinson']
  self.assertFalse(gate['published_benchmark_comparable'])
 def test_pilot_results_name_training_config_not_published_steps(self):
  for name in ('off','high','negative5'):
   d=json.loads((ROOT/'results'/f'parkinson_env0_{name}_1000.json').read_text())
   self.assertEqual(d['training_config_steps'],5555)
   self.assertNotIn('published_steps',d)
   self.assertIn('10 x 1500',d['paper_eval_protocol'])
  gate=json.loads((ROOT/'evidence'/'gates.json').read_text())['parkinson']
  self.assertFalse(gate['published_eval_protocol_10x1500_x6_reproduced'])
 def test_full_neuron_pilot_and_correct_action_mapping(self):
  d={name:json.loads((ROOT/'results'/f'parkinson_env0_{name}_1000.json').read_text()) for name in ('off','high','negative5')}
  self.assertEqual([d[k]['n_neurons'] for k in d],[512]*3)
  self.assertEqual([d[k]['steps'] for k in d],[1000]*3)
  self.assertEqual(d['off']['action_normalized'],0)
  self.assertEqual(d['off']['absolute_stimulation_energy'],0)
  self.assertEqual(d['negative5']['action_normalized'],-1)
  self.assertIn('NOT DBS OFF',d['negative5']['policy'])
  self.assertLess(d['high']['mean_step_beta_component'],d['off']['mean_step_beta_component'])
 def test_dbs_native_smoke_is_not_benchmark(self):
  d=json.loads((ROOT/'results'/'parkinson_native_smoke.json').read_text())
  self.assertEqual(d['policies']['off']['actions'],4)
  self.assertEqual(d['policies']['high']['actions'],4)
  self.assertIn('not published',d['configuration'])
 def test_parkinson_pivot_tradeoff(self):
  d=json.loads((ROOT/'results'/'parkinson_pivot.json').read_text())
  self.assertEqual(d['sample_size'],100)
  self.assertLess(d['evaluation']['0.12']['paired_heavy_minus_original_mean_reward'],0)
  self.assertGreater(d['evaluation']['0.4']['paired_heavy_minus_original_mean_reward'],0)
  self.assertLess(d['evaluation']['0.12']['heavier_weight_0_40']['mean_energy'],d['evaluation']['0.12']['original_weight_0_12']['mean_energy'])
 def test_gate_ledger_conservative(self):
  d=json.loads((ROOT/'evidence'/'gates.json').read_text())
  for disease,tools in [('sepsis',6),('parkinson',8)]:
   row=d[disease]
   self.assertEqual(row['used_science_tools'],{'sepsis':12,'parkinson':10}[disease])
   self.assertEqual(row['verified_gsm_samples_used'],{'sepsis':514,'parkinson':510}[disease])
   self.assertEqual(row['distinct_source_studies'],2)
   self.assertEqual(row['policy_trajectory_datasets'],0)
   self.assertFalse(row['gate_complete'])
 def test_tool_count_no_padding(self):
  tools=json.loads((ROOT/'evidence'/'scientific_tools.json').read_text());self.assertEqual(len({t['name'] for t in tools['used']}),tools['verified_distinct_count']);self.assertEqual(tools['verified_distinct_count'],17);self.assertEqual(tools['project_counts'],{'sepsis':12,'parkinson':10})
 def test_descriptive_probes_not_treatment(self):
  for disease in ('sepsis','parkinson'):
   result=json.loads((ROOT/'results'/f'{disease}_geo.json').read_text());self.assertEqual(result['accessions_used'],{'sepsis':479,'parkinson':438}[disease]);self.assertEqual(result['features_used'],48);self.assertEqual(result['test_n'],{'sepsis':144,'parkinson':132}[disease]);self.assertTrue(0<=result['heldout_auc']<=1)
if __name__=='__main__':unittest.main()

class FigshareImmuneAuditTests(unittest.TestCase):
 def test_derived_sepsis_immune_cohorts_not_policy_data(self):
  d=json.loads((ROOT/'results'/'sepsis_figshare_immune.json').read_text())
  e,m=d['cohorts']['eicu'],d['cohorts']['mimic_iv']
  self.assertEqual((e['rows'],e['unique_ids'],m['rows'],m['unique_ids']),(6860,6860,2423,2423))
  self.assertEqual(e['nlr_measurements_available_day1_to_day7'][0],6860)
  self.assertEqual(m['nlr_measurements_available_day1_to_day7'][-1],976)
  self.assertLess(e['baseline_nlr_rank_auc_descriptive'],.52)
  self.assertLess(m['baseline_nlr_rank_auc_descriptive'],.57)
  self.assertIn('No treatment-policy training',d['limitation'])
  gate=json.loads((ROOT/'evidence'/'gates.json').read_text())['sepsis']
  self.assertEqual(gate['policy_trajectory_datasets'],0)
  self.assertFalse(gate['figshare_policy_validated'])

class ZenodoSepsisWorkflowTests(unittest.TestCase):
 def test_authentic_events_not_icu_policy(self):
  d=json.loads((ROOT/'results'/'sepsis_zenodo_workflow.json').read_text())
  self.assertEqual((d['real_events'],d['real_case_ids']),(2654,1050))
  self.assertEqual(d['activities']['IV Antibiotics'],823)
  self.assertEqual(d['excluded_synthetic_log_names'],['x10.log','x100.log','x1000.log'])
  self.assertIn('not an ICU time-series',d['limitation'])
  self.assertEqual(json.loads((ROOT/'evidence'/'gates.json').read_text())['sepsis']['policy_trajectory_datasets'],0)

class SepsisFluidTrialAuditTests(unittest.TestCase):
 def test_randomized_guidance_summary_not_rl_trajectory(self):
  d=json.loads((ROOT/'results'/'sepsis_zenodo_fluid_trial.json').read_text())
  a=d['arms']['static_cvp'];b=d['arms']['dynamic_ivc']
  self.assertEqual((d['records'],d['columns']),(123,58))
  self.assertEqual((a['records'],b['records']),(62,61))
  self.assertEqual((a['mortality_observed'],a['mortality_missing'],a['30day_deaths']),(61,1,28))
  self.assertEqual((b['mortality_observed'],b['mortality_missing'],b['30day_deaths']),(61,0,21))
  self.assertIn('p=0.196',d['interpretation'])
  self.assertIn('No dose timing',d['limitation'])
  self.assertEqual(json.loads((ROOT/'evidence'/'gates.json').read_text())['sepsis']['policy_trajectory_datasets'],0)

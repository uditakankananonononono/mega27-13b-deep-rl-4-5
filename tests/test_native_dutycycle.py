import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def test_frozen_reward_and_matching_seed_integrity():
 j=json.loads((ROOT/'results/parkinson_native_dutycycle.json').read_text());assert len(j['rows'])==6
 for r in j['rows']:
  a=np.asarray(r['actions']);v=np.asarray(r['rewards']);assert len(a)==len(v)==150
  assert a.sum()==76;assert np.isclose(v.mean(),r['mean_reward'])
  assert np.isclose(np.mean(-v-.05*a),r['mean_beta_component'])
  assert r['beta_excess_vs_on']>0
 assert j['mean_reward_half_minus_on']<0
 assert 'half' not in j['grid_winner_windows']
 assert [x['policy'] for x in j['exact_envelope_nonnegative_lambda']]==['on','threshold','off']
 assert j['exact_envelope_nonnegative_lambda'][-1]['upper'] is None
def test_cost_envelope_matches_expected_boundaries():
 j=json.loads((ROOT/'results/parkinson_native_dutycycle.json').read_text());m=j['frozen_policy_mean_beta_abs_voltage']
 def winner(x):return min(m,key=lambda k:m[k]['mean_beta_component']+x*m[k]['mean_abs_voltage'])
 assert winner(.01)=='on';assert winner(1)=='threshold';assert winner(5)=='off'
 assert np.isclose((m['half']['mean_beta_component']-m['on']['mean_beta_component'])/(m['on']['mean_abs_voltage']-m['half']['mean_abs_voltage']),j['pooled_mean_lambda_crossover'])

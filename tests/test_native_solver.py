import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from merge_native_solver import compute
def test_all_solver_outcomes_retained():
 j=json.loads((ROOT/'results/native_solver_audit.json').read_text());assert compute()==j and len(j['rows'])==3
 for r in j['rows']:assert r['beta']['phase0']<r['beta']['on']<r['beta']['off']
 old=json.loads((ROOT/'results/native_broad_1931_off.json').read_text());assert j['rows'][0]['beta']['off']==old['mean_beta_component']
def test_solver_action_accounting():
 p=json.loads((ROOT/'src/native_solver_plan.json').read_text())
 for variant in p['variants']:
  for policy in p['policies']:
   j=json.loads((ROOT/f'results/native_solver_1931_{policy}_{variant}.json').read_text());assert j['steps']==60
   assert j['voltage_step_magnitude']=={'off':0,'on':300,'phase0':150}[policy]
   assert np.isclose(j['mean_beta_component'],-np.mean(j['rewards'])-.05*np.mean(j['actions']))

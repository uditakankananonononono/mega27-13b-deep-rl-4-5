import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from merge_native_broad import compute
def test_complete_broad_grid():
 j=json.loads((ROOT/'results/native_broad_audit.json').read_text());assert compute()==j
 assert [r['seed'] for r in j['rows']]==[1561,1687,1801,1931]
 assert j['rows'][0]['half_duty_phases_worse_than_off']==['phase0','phase3']
 assert j['rows'][-1]['half_duty_phases_better_than_on']==['phase0','phase1','phase2','phase3']
def test_all_actions_and_costs():
 for seed in [1561,1687,1801,1931]:
  for policy in ['off','on','phase0','phase1','phase2','phase3']:
   j=json.loads((ROOT/f'results/native_broad_{seed}_{policy}.json').read_text());a=[int(policy=='on') if policy in ['off','on'] else int((t+int(policy[-1]))%4<2) for t in range(60)]
   assert j['actions']==a and j['voltage_step_magnitude']==5*sum(a)
   assert np.isclose(j['mean_beta_component'],-np.mean(j['rewards'])-.05*np.mean(a))

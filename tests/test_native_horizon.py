import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from merge_native_horizon import compute
def test_complete_selected_seed_horizon():
 j=json.loads((ROOT/'results/native_horizon_audit.json').read_text());assert compute()==j
 assert j['rows'][0]['beta']['on']<j['rows'][0]['beta']['phase0']<j['rows'][0]['beta']['off']
 assert j['rows'][1]['beta']['phase0']<j['rows'][1]['beta']['on'] and j['rows'][1]['beta']['phase1']<j['rows'][1]['beta']['on']
 for seed in [1561,1931]:
  for policy in ['off','on','phase0','phase1']:
   j=json.loads((ROOT/f'results/native_horizon_{seed}_{policy}.json').read_text());a=[int(policy=='on') if policy in ['off','on'] else int((t+int(policy[-1]))%4<2) for t in range(120)]
   assert j['actions']==a and j['voltage_step_magnitude']==5*sum(a)
   assert np.isclose(j['mean_beta_component'],-np.mean(j['rewards'])-.05*np.mean(a))

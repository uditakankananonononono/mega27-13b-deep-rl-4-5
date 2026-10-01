import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from merge_native_matched import compute
def test_saved_summary():
 j=json.loads((ROOT/'results/native_matched_audit.json').read_text());assert compute()==j
 assert len(j['rows'])==2 and all(r['all_phases_between_controls'] for r in j['rows'])
def test_exact_control_accounting():
 for seed in [1331,1447]:
  for policy,a in [('off',0),('on',1)]:
   j=json.loads((ROOT/f'results/native_matched_{seed}_{policy}.json').read_text())
   assert j['steps']==60 and j['actions']==[a]*60 and j['voltage_step_magnitude']==300*a
   assert np.isclose(j['mean_beta_component'],-np.mean(j['rewards'])-.05*a)

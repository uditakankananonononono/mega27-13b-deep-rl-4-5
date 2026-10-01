import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from merge_native_phase import merge

def test_frozen_native_phase_grid_accounting():
 j=json.loads((ROOT/'results/native_phase_audit.json').read_text());assert merge()==j
 assert len(j['rows'])==8 and j['source_commit']=='aa0b10b9502e4f62dea755ce6468da024235c319'
 assert len(set(r['lfp_sha256'] for r in j['rows']))==8
 for r in j['rows']:
  a=np.array(r['actions']);reward=np.array(r['rewards']);assert len(a)==len(reward)==r['steps']==60
  assert a.tolist()==[int((step+r['phase'])%4<2) for step in range(60)]
  assert a.sum()==30 and r['physical_abs_voltage_total']==150 and r['on_fraction']==.5
  assert np.isclose(-reward.mean()-.05*a.mean(),r['mean_beta_component'])
  assert r['switch_count'] in [29,30]
 for s in j['summary']:assert s['phase_range']>0

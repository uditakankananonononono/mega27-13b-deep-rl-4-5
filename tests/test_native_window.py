import copy,hashlib,json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from native_window_audit import compute,validate

def test_window_ledger_and_partition_accounting():
 j=json.loads((ROOT/'results/native_window_audit.json').read_text());assert compute()==j
 assert len(j['rows'])==14 and len(j['source_records'])==8
 for seed in [1561,1931]:
  rows=[r for r in j['rows'] if r['seed']==seed];parts=rows[:4];whole=rows[-1]
  for policy in ['off','on','phase0','phase1']:
   for metric in ['mean_reward','mean_beta_component','on_fraction']:
    assert sum(r['policies'][policy][metric] for r in parts)/4==pytest.approx(whole['policies'][policy][metric])
   assert sum(r['policies'][policy]['voltage_step_magnitude'] for r in parts)==whole['policies'][policy]['voltage_step_magnitude']
 assert j['rows'][0]['policies']['phase0']['above_both']
 assert not j['rows'][6]['policies']['phase0']['above_both']
 assert all(r['policies']['phase0']['below_both'] and r['policies']['phase1']['below_both'] for r in j['rows'][7:])

@pytest.mark.parametrize('field,value',[('seed',1),('policy','wrong'),('source_commit','wrong'),('steps',60),('actions',[0]*120),('rewards',[float('nan')]*120),('mean_beta_component',True),('mean_reward',0),('on_fraction',0),('voltage_step_magnitude',0)])
def test_window_source_integrity_rejects_tampering(field,value):
 raw=(ROOT/'src/native_horizon_plan.json').read_bytes();p=json.loads(raw);r=json.loads((ROOT/'results/native_horizon_1561_phase1.json').read_text());h=hashlib.sha256(raw).hexdigest();validate(r,p,1561,'phase1',h);r[field]=value
 with pytest.raises(ValueError):validate(r,p,1561,'phase1',h)

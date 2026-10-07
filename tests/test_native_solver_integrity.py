import copy,hashlib,json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from merge_native_full_solver import validate_record
@pytest.mark.parametrize('field,value',[('seed',1),('policy','off'),('variant','wrong'),('source_commit','wrong'),('steps',59),('mean_reward',0),('mean_beta_component',0),('on_fraction',0),('voltage_step_magnitude',0),('rewards',[float('nan')]*60),('actions',[0]*60)])
def test_tampered_native_ledger_rejected(field,value):
 raw=(ROOT/'src/native_full_solver_plan.json').read_bytes();p=json.loads(raw);r=json.loads((ROOT/'results/native_full_solver_1931_on_Dopri5_1e-5.json').read_text());validate_record(r,p,'Dopri5_1e-5','on',hashlib.sha256(raw).hexdigest());r[field]=value
 with pytest.raises(ValueError):validate_record(r,p,'Dopri5_1e-5','on',hashlib.sha256(raw).hexdigest())

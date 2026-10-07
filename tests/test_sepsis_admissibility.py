import hashlib,json,sys
from pathlib import Path
import numpy as np
import pytest
from scipy.sparse import csr_matrix
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sepsis_admissibility_audit import mask_from_file,restricted_value
from sepsis import value

def test_new_source_result_contract():
 j=json.loads((ROOT/'results/sepsis_admissibility_audit.json').read_text())
 assert j['plan_sha256']==hashlib.sha256((ROOT/'src/sepsis_admissibility_plan.json').read_bytes()).hexdigest()
 assert len(j['transient_support_counts'])==713
 assert sum(25-n for n in j['transient_support_counts'])==j['inadmissible_transient_rows']==15587
 assert j['mean_fill_rows_above_1e12']==0 and j['mean_fill_max_abs_residual']<1e-12
 assert j['full_residual']<1e-10 and j['restricted_residual']<1e-10
 assert j['full_optimal']==pytest.approx(j['restricted_optimal'],abs=1e-12)
 assert j['full_uniform_value']==pytest.approx(j['admissible_uniform_value'],abs=1e-12)
 assert j['inadmissible_argmax_max_gap_to_supported_best']<1e-12

def test_mask_parser_checks_all_rows(tmp_path):
 p=tmp_path/'support.txt';p.write_text(' '.join(['1']*716)+'\n'+('0\n'*716));mask=mask_from_file(p);assert mask.shape==(716,25) and mask.sum()==716
 p.write_text(' '.join(['1']*716)+'\n'+('25\n'*716))
 with pytest.raises(ValueError):mask_from_file(p)

def test_synthetic_mean_fill_bellman_equivalence():
 # Every transient action is an arrival at death/discharge. Unsupported means
 # are mixtures, so cannot beat the two supported rows.
 rows=np.zeros((716,25,716));mask=np.zeros((716,25),bool);mask[:,:2]=True
 rows[:,0,713]=1;rows[:,1,714]=1;rows[:,2:,713]=.5;rows[:,2:,714]=.5
 p=csr_matrix(rows.reshape(-1,716));r=np.zeros(716);r[714]=1
 full,_,_,_=value(p,r);restricted,_,_,_=restricted_value(p,r,mask)
 assert np.allclose(full,restricted) and np.allclose(full[:713],1)

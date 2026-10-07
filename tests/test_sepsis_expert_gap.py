import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def test_model_gap_partition_identity():
 j=json.loads((ROOT/'results/sepsis_expert_gap_audit.json').read_text())
 assert j['plan_sha256']==hashlib.sha256((ROOT/'src/sepsis_expert_gap_plan.json').read_bytes()).hexdigest()
 assert len(j['states'])==713
 assert abs(j['performance_difference_identity_residual'])<1e-8
 assert j['occupancy_replay_max_abs_error']==0
 assert np.isclose(sum(s['supported_gap'] for s in j['states']),j['supported_gap_contribution'])
 assert np.isclose(sum(s['unsupported_gap'] for s in j['states']),j['unsupported_gap_contribution'])
 assert np.isclose(j['supported_gap_contribution']+j['unsupported_gap_contribution'],j['optimal_minus_expert_model_gap'])
 assert np.isclose(j['unsupported_gap_share'],j['unsupported_gap_contribution']/j['optimal_minus_expert_model_gap'])
 assert j['minimum_Q_shortfall']>=-1e-8

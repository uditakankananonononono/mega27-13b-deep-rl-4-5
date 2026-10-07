import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def test_expert_support_occupancy_snapshot():
 j=json.loads((ROOT/'results/sepsis_expert_support_audit.json').read_text());ss=j['states']
 assert j['plan_sha256']==hashlib.sha256((ROOT/'src/sepsis_expert_support_plan.json').read_bytes()).hexdigest()
 assert len(ss)==j['states_with_positive_unsupported_mass']==713
 assert j['positive_expert_unsupported_entries']==7014
 assert j['positive_expert_action_entries']==9252
 assert j['expert_row_sum_max_residual']<1e-12 and j['occupancy_residual']<1e-8
 d=np.array([s['expert_model_expected_visits'] for s in ss]);u=np.array([s['unsupported_probability'] for s in ss])
 assert np.isclose(d.sum(),j['expert_model_expected_transient_decisions'])
 assert np.isclose(d@u,j['expected_unsupported_decisions_per_initial_model_case'])
 assert np.isclose(d@u/d.sum(),j['occupancy_weighted_unsupported_fraction'])

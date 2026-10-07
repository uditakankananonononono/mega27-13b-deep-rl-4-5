import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def test_absorption_snapshot():
 j=json.loads((ROOT/'results/sepsis_absorption_audit.json').read_text())
 assert j['plan_sha256']==hashlib.sha256((ROOT/'src/sepsis_absorption_plan.json').read_bytes()).hexdigest()
 for r in j['policies']:
  assert len(r['states'])==713 and r['transient_spectral_radius']<1
  assert r['max_abs_absorption_residual']<1e-8
  assert r['max_abs_absorption_sum_minus_one']<1e-8
  assert abs(r['survival_absorption_minus_bellman'])<1e-8
  assert np.isclose(sum(r['initial_terminal_absorption']),1)
  assert r['initial_terminal_absorption'][2]==0

import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sepsis_capacity_scope_audit import compute
def test_actual_parameter_counts():
 j=compute();assert j==json.loads((ROOT/'results/sepsis_capacity_scope_audit.json').read_text())
 assert [r['parameters'] for r in j['architectures']]==[51673,8857]

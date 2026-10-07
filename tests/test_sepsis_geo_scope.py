import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sepsis_geo_scope_audit import compute
def test_archived_split_identity():
 j=compute();assert j==json.loads((ROOT/'results/sepsis_geo_scope_audit.json').read_text())
 assert j['random_test_and_cohort_test_intersection']==67
 assert j['random_train_and_cohort_test_intersection']==149
 assert j['unique_accessions']==479

"""Reconcile exact archived split identities, units, and method differences."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def compute():
 files=json.loads((ROOT/'research/sepsis_geo_scope_plan.json').read_text())['inputs'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
 prov=json.loads((ROOT/files[1]).read_text());assert hashes[files[0]]==prov['derived_sha256']
 rows=list(csv.DictReader((ROOT/files[0]).open()));ids=[r['accession'] for r in rows];assert len(set(ids))==len(ids)
 perm=np.random.default_rng(73).permutation(len(rows));cut=int(.7*len(rows));random_test=[ids[i] for i in perm[cut:]]
 cohort_test=[r['accession'] for r in rows if r['cohort']=='validation']
 assert random_test==json.loads((ROOT/files[2]).read_text())['heldout_accessions']
 assert cohort_test==json.loads((ROOT/files[3]).read_text())['test_accessions']
 return {'plan_sha256':hashlib.sha256((ROOT/'research/sepsis_geo_scope_plan.json').read_bytes()).hexdigest(),'input_sha256':hashes,'csv_rows':len(rows),'unique_accessions':len(set(ids)),'feature_count':len(rows[0])-3,'random_train_n':cut,'random_test_n':len(random_test),'cohort_train_n':sum(r['cohort']=='discovery' for r in rows),'cohort_test_n':len(cohort_test),'random_test_and_cohort_test_intersection':len(set(random_test)&set(cohort_test)),'random_train_and_cohort_test_intersection':len(set(ids[i] for i in perm[:cut])&set(cohort_test)),'preprocessing':{'random':'training median centering; training standard deviation; clipping [-5,5]','cohort':'training mean centering; training standard deviation; clipping [-5,5]'},'limits':json.loads((ROOT/'research/sepsis_geo_scope_plan.json').read_text())['limits']}
if __name__=='__main__':
 j=compute();(ROOT/'results/sepsis_geo_scope_audit.json').write_text(json.dumps(j,indent=2)+'\n');print(j)

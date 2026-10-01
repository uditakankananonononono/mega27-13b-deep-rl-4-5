"""Merge complete frozen phase grid, no winning-phase selection for execution."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def merge():
 raw=(ROOT/'src/native_phase_plan.json').read_bytes();p=json.loads(raw);rows=[json.loads((ROOT/f'results/native_phase_{s}_{a}.json').read_text()) for s in p['seeds'] for a in p['phases']];assert all(r['plan_sha256']==hashlib.sha256(raw).hexdigest() for r in rows)
 summary=[]
 for seed in p['seeds']:
  r=[x for x in rows if x['seed']==seed];values=[x['mean_beta_component'] for x in r];summary.append({'seed':seed,'min_beta':min(values),'max_beta':max(values),'phase_range':max(values)-min(values),'phase_of_minimum':r[int(np.argmin(values))]['phase'],'phase_of_maximum':r[int(np.argmax(values))]['phase']})
 out={'source':'https://github.com/NevVerVer/DBS-Gym','source_commit':p['source_commit'],'plan_sha256':hashlib.sha256(raw).hexdigest(),'rows':rows,'summary':summary,'limits':p['limits']};(ROOT/'results/native_phase_audit.json').write_text(json.dumps(out,indent=2)+'\n');return out
if __name__=='__main__':print(merge()['summary'])

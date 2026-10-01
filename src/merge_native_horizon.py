import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def compute():
 raw=(ROOT/'src/native_horizon_plan.json').read_bytes();p=json.loads(raw);rows=[]
 for seed in p['seeds']:
  rr={policy:json.loads((ROOT/f'results/native_horizon_{seed}_{policy}.json').read_text()) for policy in p['policies']}
  assert all(r['steps']==120 and r['plan_sha256']==hashlib.sha256(raw).hexdigest() for r in rr.values())
  rows.append({'seed':seed,'beta':{k:r['mean_beta_component'] for k,r in rr.items()},'reward':{k:r['mean_reward'] for k,r in rr.items()}})
 return {'plan_sha256':hashlib.sha256(raw).hexdigest(),'rows':rows,'selection':p['selection'],'limits':p['limits']}
if __name__=='__main__':
 j=compute();(ROOT/'results/native_horizon_audit.json').write_text(json.dumps(j,indent=2)+'\n');print(j)

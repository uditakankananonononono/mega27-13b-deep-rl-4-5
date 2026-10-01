import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def compute():
 raw=(ROOT/'src/native_broad_plan.json').read_bytes();p=json.loads(raw);rows=[]
 for seed in p['seeds']:
  rr={policy:json.loads((ROOT/f'results/native_broad_{seed}_{policy}.json').read_text()) for policy in p['policies']}
  assert all(r['plan_sha256']==hashlib.sha256(raw).hexdigest() and r['steps']==60 for r in rr.values())
  beta={k:r['mean_beta_component'] for k,r in rr.items()};reward={k:r['mean_reward'] for k,r in rr.items()}
  rows.append({'seed':seed,'beta':beta,'reward':reward,'half_duty_phases_worse_than_off':[k for k in beta if k.startswith('phase') and beta[k]>beta['off']],'half_duty_phases_better_than_on':[k for k in beta if k.startswith('phase') and beta[k]<beta['on']]})
 return {'plan_sha256':hashlib.sha256(raw).hexdigest(),'rows':rows,'limits':p['limits']}
if __name__=='__main__':
 j=compute();(ROOT/'results/native_broad_audit.json').write_text(json.dumps(j,indent=2)+'\n');print(j)

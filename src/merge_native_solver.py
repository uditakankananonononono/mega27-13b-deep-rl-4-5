import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def compute():
 raw=(ROOT/'src/native_solver_plan.json').read_bytes();p=json.loads(raw);rows=[]
 for variant in p['variants']:
  rr={policy:json.loads((ROOT/f"results/native_solver_{p['seed']}_{policy}_{variant}.json").read_text()) for policy in p['policies']};assert all(r['steps']==60 and r['plan_sha256']==hashlib.sha256(raw).hexdigest() for r in rr.values())
  rows.append({'variant':variant,'beta':{k:r['mean_beta_component'] for k,r in rr.items()},'reward':{k:r['mean_reward'] for k,r in rr.items()}})
 return {'plan_sha256':hashlib.sha256(raw).hexdigest(),'rows':rows,'limits':p['limits']}
if __name__=='__main__':
 j=compute();(ROOT/'results/native_solver_audit.json').write_text(json.dumps(j,indent=2)+'\n');print(j)

import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def validate_record(r,p,variant,policy,plan_hash):
 import math
 n=p['horizon_steps'];actions=[int(policy=='on') if policy in ['off','on'] else int(step%4<2) for step in range(n)]
 if any(r.get(k)!=v for k,v in {'seed':p['seed'],'policy':policy,'variant':variant,'source_commit':p['source_commit'],'plan_sha256':plan_hash,'steps':n,'actions':actions}.items()):raise ValueError('record identity or schedule')
 rewards=r.get('rewards',[])
 if len(rewards)!=n or any(not isinstance(x,(int,float)) or not math.isfinite(x) for x in rewards):raise ValueError('reward geometry')
 expected={'mean_reward':sum(rewards)/n,'mean_beta_component':sum(-x-.05*a for x,a in zip(rewards,actions))/n,'on_fraction':sum(actions)/n,'voltage_step_magnitude':5*sum(actions)}
 if any(not isinstance(r.get(k),(int,float)) or not math.isclose(r[k],v,rel_tol=1e-10,abs_tol=1e-10) for k,v in expected.items()):raise ValueError('reward/action accounting')

def compute():
 raw=(ROOT/'src/native_full_solver_plan.json').read_bytes();p=json.loads(raw);rows=[]
 for variant in p['variants']:
  rr={policy:json.loads((ROOT/f"results/native_full_solver_{p['seed']}_{policy}_{variant}.json").read_text()) for policy in p['policies']}
  for policy,r in rr.items():validate_record(r,p,variant,policy,hashlib.sha256(raw).hexdigest())
  rows.append({'variant':variant,'beta':{k:r['mean_beta_component'] for k,r in rr.items()},'reward':{k:r['mean_reward'] for k,r in rr.items()}})
 return {'plan_sha256':hashlib.sha256(raw).hexdigest(),'rows':rows,'limits':p['limits']}
if __name__=='__main__':
 j=compute();(ROOT/'results/native_full_solver_audit.json').write_text(json.dumps(j,indent=2)+'\n');print(j)

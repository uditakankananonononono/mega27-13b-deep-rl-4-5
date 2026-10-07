"""Window-conditioned accounting of saved native trajectories; no new simulation."""
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def validate(r,p,seed,policy,h):
 n=p['horizon_steps'];phase=int(policy[-1]) if policy.startswith('phase') else None
 actions=[int(policy=='on') if phase is None else int((i+phase)%4<2) for i in range(n)]
 if any(r.get(k)!=v for k,v in {'seed':seed,'policy':policy,'source_commit':p['source_commit'],'plan_sha256':h,'steps':n,'actions':actions}.items()):raise ValueError('identity or schedule')
 rr=r.get('rewards',[])
 if len(rr)!=n or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in rr):raise ValueError('reward geometry')
 expected={'mean_reward':sum(rr)/n,'mean_beta_component':sum(-x-.05*a for x,a in zip(rr,actions))/n,'on_fraction':sum(actions)/n,'voltage_step_magnitude':5*sum(actions)}
 if any(isinstance(r.get(k),bool) or not isinstance(r.get(k),(int,float)) or not math.isclose(r[k],v,rel_tol=1e-10,abs_tol=1e-10) for k,v in expected.items()):raise ValueError('summary accounting')
def compute():
 raw=(ROOT/'src/native_window_plan.json').read_bytes();p=json.loads(raw);oldraw=(ROOT/'src/native_horizon_plan.json').read_bytes();old=json.loads(oldraw);rows=[];source=[]
 for seed in old['seeds']:
  records={}
  for policy in old['policies']:
   path=ROOT/f'results/native_horizon_{seed}_{policy}.json';b=path.read_bytes();r=json.loads(b);validate(r,old,seed,policy,hashlib.sha256(oldraw).hexdigest());records[policy]=r;source.append({'seed':seed,'policy':policy,'record_sha256':hashlib.sha256(b).hexdigest()})
  for start,end in p['windows']:
   stats={}
   for policy,r in records.items():
    rewards=r['rewards'][start:end];actions=r['actions'][start:end];n=end-start
    stats[policy]={'mean_reward':sum(rewards)/n,'mean_beta_component':sum(-x-.05*a for x,a in zip(rewards,actions))/n,'on_fraction':sum(actions)/n,'voltage_step_magnitude':5*sum(actions)}
   for phase in ['phase0','phase1']:
    beta=stats[phase]['mean_beta_component'];stats[phase]['beta_minus_off']=beta-stats['off']['mean_beta_component'];stats[phase]['beta_minus_on']=beta-stats['on']['mean_beta_component'];stats[phase]['below_both']=beta<min(stats['off']['mean_beta_component'],stats['on']['mean_beta_component']);stats[phase]['above_both']=beta>max(stats['off']['mean_beta_component'],stats['on']['mean_beta_component'])
   rows.append({'seed':seed,'window':[start,end],'policies':stats})
 return {'plan_sha256':hashlib.sha256(raw).hexdigest(),'source_records':source,'rows':rows,'limits':p['limits']}
if __name__=='__main__':
 j=compute();(ROOT/'results/native_window_audit.json').write_text(json.dumps(j,indent=2)+'\n')
 for r in j['rows']:print(r['seed'],r['window'],{k:round(v['mean_beta_component'],6) for k,v in r['policies'].items()})

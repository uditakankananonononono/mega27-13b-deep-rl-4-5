"""Affine in-model duty-penalty envelope; not treatment utility."""
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
from native_window_audit import compute as windows

def interval(policy,stats):
 lo=0.;hi=math.inf;b=stats[policy]['mean_beta_component'];f=stats[policy]['on_fraction']
 for other,r in stats.items():
  if other==policy:continue
  slope=f-r['on_fraction'];bound=r['mean_beta_component']-b
  if slope==0:
   if bound<0:return None
  elif slope>0:hi=min(hi,bound/slope)
  else:lo=max(lo,bound/slope)
 if hi<lo:return None
 return {'lower':lo,'upper':None if math.isinf(hi) else hi,'endpoints':'inclusive ties'}

def compute():
 raw=(ROOT/'src/native_penalty_plan.json').read_bytes();priorpath=ROOT/'results/native_window_audit.json';old=json.loads(priorpath.read_text())
 if windows()!=old:raise ValueError('window source mismatch')
 rows=[]
 for r in old['rows']:
  stats=r['policies'];bounds={k:interval(k,stats) for k in stats};cross=[];names=list(stats)
  for i,a in enumerate(names):
   for b in names[i+1:]:
    slope=stats[a]['on_fraction']-stats[b]['on_fraction']
    cross.append({'a':a,'b':b,'lambda':(stats[b]['mean_beta_component']-stats[a]['mean_beta_component'])/slope if slope else None,'parallel':not bool(slope)})
  scores={k:v['mean_beta_component']+.05*v['on_fraction'] for k,v in stats.items()};best=min(scores.values());rows.append({'seed':r['seed'],'window':r['window'],'optimal_lambda_intervals':bounds,'pair_crossovers':cross,'original_lambda_scores':scores,'original_lambda_winners':[k for k,v in scores.items() if v==best]})
 return {'plan_sha256':hashlib.sha256(raw).hexdigest(),'source_sha256':hashlib.sha256(priorpath.read_bytes()).hexdigest(),'rows':rows,'limits':json.loads(raw)['limits']}
if __name__=='__main__':
 j=compute();(ROOT/'results/native_penalty_audit.json').write_text(json.dumps(j,indent=2)+'\n')
 for r in j['rows']:print(r['seed'],r['window'],r['optimal_lambda_intervals'])

"""Native 512-neuron short-horizon multiseed audit, NOT published SAC.
Each invocation computes one policy replicate and saves an atomic checkpoint.
Six independent training seeds, matched environment seeds and simple baselines.
"""
import sys,json,hashlib,argparse,time
from pathlib import Path
import numpy as np
import parkinson_native_reinforce_adam as P
ROOT=Path(__file__).resolve().parents[1]
def train(upstream,seed):
 state=ROOT/'data/parkinson'/f'trainstate_{seed}.npz';log=state.with_suffix('.json')
 par=P.policy_params(seed);rng=np.random.default_rng(seed+1000);m={k:np.zeros_like(v) for k,v in par.items()};v={k:np.zeros_like(v) for k,v in par.items()};traces=[]
 if state.exists():
  with np.load(state,allow_pickle=False) as z:
   par={k:z['par_'+k].copy() for k in par};m={k:z['m_'+k].copy() for k in par};v={k:z['v_'+k].copy() for k in par}
  meta=json.loads(log.read_text());traces=meta['traces'];rng.bit_generator.state=meta['rng']
 for ep in range(len(traces),3):
  e,o=P.make_env(upstream,seed+100+ep);gr=[];rr=[];aa=[]
  for _ in range(100):
   x=P.features(o);p,h1,h2=P.forward(par,x);a=int(rng.random()<p);gr.append(P.gradient(par,x,h1,h2,a,p));o,r,d,t,_=e.step(np.array([a],np.float32));rr.append(float(r));aa.append(a)
   if d or t:break
  rt=np.zeros(len(rr));acc=0
  for j in range(len(rr)-1,-1,-1):acc=rr[j]+.99*acc;rt[j]=acc
  adv=(rt-rt.mean())/(rt.std()+1e-8);upd={k:sum(adv[j]*g[k] for j,g in enumerate(gr))/len(gr) for k in par};norm=np.sqrt(sum(np.sum(g*g) for g in upd.values()));scale=min(1,1/(norm+1e-8))
  for k in par:
   g=upd[k]*scale;m[k]=.9*m[k]+.1*g;v[k]=.999*v[k]+.001*g*g;par[k]+=.03*(m[k]/(1-.9**(ep+1)))/(np.sqrt(v[k]/(1-.999**(ep+1)))+1e-7)
  traces.append({'episode':ep,'seed':seed+100+ep,'steps':len(rr),'mean_reward':float(np.mean(rr)),'on_fraction':float(np.mean(aa))});print('train',seed,ep,flush=True)
  np.savez_compressed(state,**{'par_'+k:x for k,x in par.items()},**{'m_'+k:x for k,x in m.items()},**{'v_'+k:x for k,x in v.items()})
  log.write_text(json.dumps({'traces':traces,'rng':rng.bit_generator.state}))
  if len(traces)<3:return None,traces
 return par,traces
def evaluate(upstream,seed,kind,par):
 sys.path.insert(0,upstream)
 from environment.utils import calc_beta_band_power
 e,o=P.make_env(upstream,seed);rr=[];aa=[];wave=[]
 for _ in range(150):
  a=0 if kind=='off' else 1 if kind=='on' else int(calc_beta_band_power(o.ravel(),.0005,12.5,21)>.001) if kind=='threshold' else int(P.forward(par,P.features(o))[0]>=.5)
  o,r,d,t,_=e.step(np.array([a],np.float32));rr.append(float(r));aa.append(a);wave.extend(np.asarray(e.theta_mean).tolist())
  if d or t:break
 return {'steps':len(rr),'mean_reward':float(np.mean(rr)),'on_fraction':float(np.mean(aa)),'mean_beta_component':float(np.mean(-np.array(rr)-.05*np.array(aa))),'switches':int(np.sum(np.diff(aa)!=0)),'lfp_sha256':hashlib.sha256(np.array(wave).tobytes()).hexdigest()}
def run(upstream,i):
 target=ROOT/'results'/'native_multiseed_parts';target.mkdir(exist_ok=True);file=target/f'replicate_{i}.json'
 if file.exists():print('already computed',i);return
 seed=910+i;weight=ROOT/'data'/'parkinson'/f'multiseed_{i}_weights.npz';training=weight.with_suffix('.json')
 if weight.exists() and training.exists():
  with np.load(weight,allow_pickle=False) as z:par={k:z[k].copy() for k in z.files}
  traces=json.loads(training.read_text())
 else:
  par,traces=train(upstream,seed)
  if par is None:return
  np.savez_compressed(weight,**par);training.write_text(json.dumps(traces,indent=2))
 policies={}
 for name in ['rl','off','on','threshold']:
  checkpoint=target/f'eval_{i}_{name}.json'
  if checkpoint.exists():policies[name]=json.loads(checkpoint.read_text());continue
  policies[name]=evaluate(upstream,1220+i,name,par);checkpoint.write_text(json.dumps(policies[name],indent=2));print('eval',i,name,policies[name],flush=True)
  return # bounded one-policy stage; later invocation resumes immediately
 row={'replicate':i,'training_init_seed':seed,'eval_seed':1220+i,'training':traces,'weight_sha256':hashlib.sha256(weight.read_bytes()).hexdigest(),'policies':policies}
 file.with_suffix('.tmp').write_text(json.dumps(row,indent=2)+'\n');file.with_suffix('.tmp').replace(file)
 aggregate(upstream)
def aggregate(upstream):
 parts=[json.loads(p.read_text()) for p in sorted((ROOT/'results/native_multiseed_parts').glob('replicate_*.json'))];out={'source':'https://github.com/NevVerVer/DBS-Gym','source_commit':'aa0b10b9502e4f62dea755ce6468da024235c319','planned_replicates':6,'completed_replicates':len(parts),'eval_steps':150,'training_per_replicate':'3 episodes x 100 steps, Adam REINFORCE, binary 0/+5 V','replicates':parts,'limits':['Short horizon and little training; not six-evaluation published SAC benchmark','Simulator only, no clinical evidence','Evaluation seeds differ, but adjacent training seed sets overlap; paired Student-t intervals are descriptive and do not establish independent-training uncertainty','Threshold .001 is a fixed pilot heuristic, no tuning']}
 if len(parts)>=2:
  from scipy.stats import t
  comparisons={}
  for name in ['off','on','threshold']:
   d=np.array([p['policies']['rl']['mean_reward']-p['policies'][name]['mean_reward'] for p in parts]);se=d.std(ddof=1)/np.sqrt(len(d));half=t.ppf(.975,len(d)-1)*se
   comparisons[name]={'paired_mean_rl_minus_baseline':float(d.mean()),'t95': [float(d.mean()-half),float(d.mean()+half)],'n':len(d)}
  out['comparisons']=comparisons
 (ROOT/'results/parkinson_native_multiseed.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--upstream',required=True);a.add_argument('--replicate',type=int,required=True);x=a.parse_args();run(x.upstream,x.replicate)

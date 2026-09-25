"""Pilot deep REINFORCE controller in actual 512-neuron DBS-Gym Env0.
Small training horizon and one evaluation seed are NOT the published SAC benchmark.
"""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np

def policy_params(seed=810):
 rng=np.random.default_rng(seed)
 return {'w1':rng.normal(0,.10,(4,8)),'b1':np.zeros(8),'w2':rng.normal(0,.10,(8,8)),
         'b2':np.zeros(8),'w3':rng.normal(0,.10,(8,)),'b3':np.zeros(1)}

def features(obs,last=0):
 from environment.utils import calc_beta_band_power
 x=np.asarray(obs,dtype=float).ravel();return np.array([np.clip(calc_beta_band_power(x,.0005,12.5,21)*1000,0,10)/10,
       np.clip(np.std(x)*5,0,10)/10,np.clip(np.mean(x)*5,-10,10)/10,last],dtype=float)

def forward(par,x):
 h1=np.tanh(x@par['w1']+par['b1']);h2=np.tanh(h1@par['w2']+par['b2']);logit=h2@par['w3']+par['b3'][0]
 p=float(1/(1+np.exp(-np.clip(logit,-20,20))));return p,h1,h2

def gradient(par,x,h1,h2,a,p):
 dg=a-p;g={};g['w3']=h2*dg;g['b3']=np.array([dg]);h2d=(1-h2*h2)*par['w3']*dg
 g['w2']=np.outer(h1,h2d);g['b2']=h2d;h1d=(1-h1*h1)*(par['w2']@h2d);g['w1']=np.outer(x,h1d);g['b1']=h1d
 return g

def make_env(upstream,seed):
 sys.path.insert(0,upstream)
 from environment.env import SpatialKuramoto
 from environment.env_configs.env0 import params_dict_train
 from environment.utils import generate_w0_with_locus
 p=copy.deepcopy(params_dict_train);p['reward_func']='bbpow_action';p['verbose']=0;p['rand_seed']=seed
 np.random.seed(seed)
 w,nc,ng,wt,wl,lmask=generate_w0_with_locus(512,p['grid_size'],p['coord_modif'],p['locus_center'],p['locus_size'],p['wmuL'],p['wsdL'],show=False,vertical_layer=4)
 p.update({'w0':w,'w0_without_locus':wt,'locus_without_w0':wl,'locus_mask':lmask,'neur_coords':nc,'neur_grid':ng})
 env=SpatialKuramoto(p);obs,_=env.reset(seed=seed);return env,obs

def run(upstream,out,train_episodes=4,train_steps=180,eval_steps=1200,weights=None):
 par=policy_params();rng=np.random.default_rng(824);traces=[];start=time.monotonic()
 if weights:
  with np.load(weights,allow_pickle=False) as z:par={key:z[key].copy() for key in par}
  traces=json.loads(Path(weights).with_name('native_reinforce_training.json').read_text())
 for episode in range(0 if weights else train_episodes):
  seed=820+episode;env,obs=make_env(upstream,seed);last=0;grads=[];rewards=[];actions=[]
  for i in range(train_steps):
   x=features(obs,last);p,h1,h2=forward(par,x);a=int(rng.random()<p);grads.append(gradient(par,x,h1,h2,a,p));obs,r,done,_,_=env.step(np.array([float(a)],dtype=np.float32));last=a;rewards.append(float(r));actions.append(a)
   if done:break
  # REINFORCE: reward-to-go, centered within the episode and clipped gradient norm.
  ret=np.zeros(len(rewards));acc=0
  for t in range(len(rewards)-1,-1,-1):acc=rewards[t]+.99*acc;ret[t]=acc
  adv=(ret-np.mean(ret))/(np.std(ret)+1e-8)
  upd={key:sum(adv[t]*g[key] for t,g in enumerate(grads))/len(grads) for key in par}
  norm=np.sqrt(sum(np.sum(g*g) for g in upd.values()));scale=min(1,1/(norm+1e-8))
  for key in par:par[key]+=0.01*upd[key]*scale
  traces.append({'seed':seed,'steps':len(rewards),'mean_reward':float(np.mean(rewards)),'on_fraction':float(np.mean(actions)),'gradient_norm':float(norm)})
  print('train',episode+1,'steps',len(rewards),'reward',round(np.mean(rewards),4),'on_fraction',round(np.mean(actions),3),'elapsed',round(time.monotonic()-start,1),flush=True)
 weights_out=Path(__file__).resolve().parents[1]/'data'/'parkinson'/'native_reinforce_weights.npz'
 training_out=Path(__file__).resolve().parents[1]/'data'/'parkinson'/'native_reinforce_training.json'
 np.savez_compressed(weights_out,**par)
 training_out.write_text(json.dumps(traces,indent=2))
 if eval_steps<=0:return {'training':traces,'pilot_evaluation':'pending'}
 env,obs=make_env(upstream,222);actions=[];rewards=[];wave=[];last=0
 for i in range(eval_steps):
  p,_,_=forward(par,features(obs,last));a=int(p>=.5);obs,r,done,_,_=env.step(np.array([float(a)],dtype=np.float32));last=a;actions.append(a);rewards.append(float(r));wave.extend(np.asarray(env.theta_mean,dtype=float).tolist())
  if (i+1)%400==0:print('eval',i+1,'elapsed',round(time.monotonic()-start,1),flush=True)
  if done:break
 result={'source':'https://github.com/NevVerVer/DBS-Gym','source_commit':'aa0b10b9502e4f62dea755ce6468da024235c319',
  'algorithm':'two hidden tanh layers, Bernoulli REINFORCE, within-episode centered discounted returns gamma .99, learning rate .01, norm clipping 1; binary physical 0 or +5 V actions',
  'feature_source':'prior actual upstream LFP window, beta power, standard deviation, mean, prior action only',
  'training':traces,'eval_seed':222,'eval_steps':len(rewards),'eval_mean_reward':float(np.mean(rewards)),
  'eval_beta_component':float(np.mean(-np.asarray(rewards)-.05*np.asarray(actions))),
  'eval_action_on_fraction':float(np.mean(actions)),'eval_stimulation_energy':float(np.sum(actions)*5),
  'eval_lfp_sha256':hashlib.sha256(np.asarray(wave).tobytes()).hexdigest(),
  'trained_weight_sha256':hashlib.sha256((Path(weights) if weights else weights_out).read_bytes()).hexdigest(),
  'limitations':f'Pilot deep RL in native full-neuron Env0, but only {len(traces)} 180-step training episodes and one {len(rewards)}-step evaluation. NOT published six-run 5555-step SAC benchmark, no patient data or clinical validation.'}
 Path(out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True);return result
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--upstream',required=True);a.add_argument('--out',default='results/parkinson_native_reinforce.json');a.add_argument('--train-episodes',type=int,default=4);a.add_argument('--train-steps',type=int,default=180);a.add_argument('--eval-steps',type=int,default=1200);a.add_argument('--weights');x=a.parse_args();run(x.upstream,x.out,x.train_episodes,x.train_steps,x.eval_steps,x.weights)

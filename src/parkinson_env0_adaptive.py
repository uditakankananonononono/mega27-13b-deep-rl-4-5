"""Native DBS-Gym full-neuron adaptive controller pilot.
Threshold uses observed upstream LFP beta power, not prior 3-state surrogate.
No training or published SAC/PPO model is implied.
"""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np

def run(upstream,out,seed=223,steps=1200,threshold=0.001):
 sys.path.insert(0,upstream)
 from environment.env import SpatialKuramoto
 from environment.env_configs.env0 import params_dict_train
 from environment.utils import generate_w0_with_locus,calc_beta_band_power,units2sec
 p=copy.deepcopy(params_dict_train);p['reward_func']='bbpow_action';p['verbose']=0;p['rand_seed']=seed
 np.random.seed(seed);w,nc,ng,wt,wl,lmask=generate_w0_with_locus(512,p['grid_size'],p['coord_modif'],p['locus_center'],p['locus_size'],p['wmuL'],p['wsdL'],show=False,vertical_layer=4)
 p.update({'w0':w,'w0_without_locus':wt,'locus_without_w0':wl,'locus_mask':lmask,'neur_coords':nc,'neur_grid':ng})
 e=SpatialKuramoto(p);obs,_=e.reset(seed=seed);dt=units2sec(p['verbose_dt']);r=[];beta=[];actions=[];lfp=[];t=time.monotonic()
 for i in range(steps):
  b=float(calc_beta_band_power(obs.ravel(),dt,12.5,21));a=1.0 if b>threshold else 0.0
  obs,reward,done,trunc,_=e.step(np.array([a],dtype=np.float32));r.append(float(reward));beta.append(b);actions.append(a);lfp.extend(np.asarray(e.theta_mean,dtype=float).tolist())
  if (i+1)%400==0:print('step',i+1,'elapsed',round(time.monotonic()-t,1),flush=True)
  if done or trunc:break
 d={'source':'https://github.com/NevVerVer/DBS-Gym','source_commit':'aa0b10b9502e4f62dea755ce6468da024235c319','model':'native 512-neuron Env0',
    'policy':'fixed, arbitrary threshold on upstream observable LFP beta spectral power; NOT deep RL','threshold':threshold,'threshold_selected':'pilot heuristic, no independent tuning or clinical grounding',
    'seed':seed,'steps':len(r),'training_config_steps':5555,'paper_eval_protocol':'10 x 1500 steps per evaluation, six Env0 evaluations; checked repo eval0 config differs','mean_reward':float(np.mean(r)),
    'mean_beta_reward_component':float(np.mean(-np.asarray(r)-.01*5*np.asarray(actions))),
    'action_on_fraction':float(np.mean(actions)),'energy':float(np.sum(actions)*5),
    'global_low_beta_power':float(calc_beta_band_power(np.asarray(lfp),dt,12.5,21)),
    'mean_observed_beta_before_action':float(np.mean(beta)),'lfp_sha256':hashlib.sha256(np.asarray(lfp).tobytes()).hexdigest(),
    'limitations':'Single seed, shortened episode, arbitrary threshold; not published benchmark or clinical recommendation.'}
 Path(out).write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2));return d
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--upstream',required=True);a.add_argument('--out',required=True);a.add_argument('--steps',type=int,default=1200);a.add_argument('--seed',type=int,default=223);a.add_argument('--threshold',type=float,default=.001);x=a.parse_args();run(x.upstream,x.out,x.seed,x.steps,x.threshold)

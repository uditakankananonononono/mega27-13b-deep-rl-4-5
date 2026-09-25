"""Full 512-neuron Env0 pilot at a short, explicit horizon.
NOT a reproduction of the published 5,555-step DBS-Gym benchmark.
"""
import argparse,copy,hashlib,json,sys,time,os
from pathlib import Path
import numpy as np

def run(upstream,out,action,steps,seed):
 sys.path.insert(0,upstream)
 from environment.env import SpatialKuramoto
 from environment.env_configs.env0 import params_dict_train
 from environment.utils import generate_w0_with_locus,calc_beta_band_power,units2sec
 p=copy.deepcopy(params_dict_train)
 assert p['num_oscillators']==512 and p['grid_size']==[8,8,8]
 p['reward_func']='bbpow_action';p['total_episode_len']=max(float(p['total_episode_len']),float(steps*p['electrode_width']+steps*p['electrode_pause']+1));p['verbose']=0;p['rand_seed']=seed
 np.random.seed(seed)
 w,nc,ng,wt,wl,lmask=generate_w0_with_locus(512,p['grid_size'],p['coord_modif'],p['locus_center'],p['locus_size'],p['wmuL'],p['wsdL'],show=False,vertical_layer=4)
 p.update({'w0':w,'w0_without_locus':wt,'locus_without_w0':wl,'locus_mask':lmask,'neur_coords':nc,'neur_grid':ng})
 env=SpatialKuramoto(p);obs,_=env.reset(seed=seed);rewards=[];wave=[];t=time.monotonic()
 for i in range(steps):
  obs,r,done,trunc,_=env.step(np.array([action],dtype=np.float32));rewards.append(float(r));wave.extend(np.asarray(env.theta_mean,dtype=float).tolist())
  if i and (i+1)%500==0:print(f'{i+1}/{steps} elapsed {time.monotonic()-t:.1f}s',flush=True)
  if done or trunc:break
 wave=np.asarray(wave);dt=units2sec(p['verbose_dt']);power=calc_beta_band_power(wave,dt,12.5,21)
 result={'source':'https://github.com/NevVerVer/DBS-Gym','source_commit':'aa0b10b9502e4f62dea755ce6468da024235c319',
  'policy':'DBS OFF' if action==0. else ('HF-DBS' if action==1. else f'fixed {action}'),
  'action_normalized':action,'n_neurons':512,'grid':[8,8,8],'env':'env0','seed':seed,'steps':len(rewards),'published_steps':5555,'configuration_total_episode_len_units':float(p['total_episode_len']),
  'reward_function':'bbpow_action','mean_step_reward':float(np.mean(rewards)),'total_reward':float(sum(rewards)),
  'mean_step_beta_component':float(np.mean([-r-.01*abs(float(env.rescale_action(action))) for r in rewards])),
  'global_low_beta_power':float(power),'episode_truncated_to_steps':len(rewards)<int(p['total_episode_len']/(p['electrode_width']+p['electrode_pause'])),'absolute_stimulation_energy':float(len(rewards)*abs(float(env.rescale_action(action)))),
  'lfp_points':len(wave),'lfp_sha256':hashlib.sha256(wave.tobytes()).hexdigest(),'runtime_seconds':float(time.monotonic()-t),
  'comparison_caveat':'512 neurons and actual upstream env0, but horizon is shorter than published 5555 steps, one seed, reward metrics and normalization may differ. Do not compare directly to published percent power benchmark.'}
 Path(out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True);return result
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--upstream',required=True);a.add_argument('--out',required=True);a.add_argument('--action',type=float,required=True);a.add_argument('--steps',type=int,default=1000);a.add_argument('--seed',type=int,default=222);x=a.parse_args();run(x.upstream,x.out,x.action,x.steps,x.seed)

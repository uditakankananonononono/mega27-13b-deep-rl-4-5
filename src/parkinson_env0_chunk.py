"""Chunked exact upstream 512-neuron Env0 execution; checkpoints avoid process limits.
Config must remain fixed across chunks. No patient data or published-score claim.
"""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np

def chunk(upstream,checkpoint,action,total_steps,chunk_steps=1600,seed=222):
 sys.path.insert(0,upstream)
 from environment.env import SpatialKuramoto
 from environment.env_configs.env0 import params_dict_train
 from environment.utils import generate_w0_with_locus,calc_beta_band_power,units2sec
 p=copy.deepcopy(params_dict_train);assert p['num_oscillators']==512 and p['grid_size']==[8,8,8]
 p['reward_func']='bbpow_action';p['verbose']=0;p['rand_seed']=seed
 assert int(p['total_episode_len']/(p['electrode_width']+p['electrode_pause']))>=total_steps, 'Requested steps exceed upstream episode config'
 np.random.seed(seed)
 w,nc,ng,wt,wl,lmask=generate_w0_with_locus(512,p['grid_size'],p['coord_modif'],p['locus_center'],p['locus_size'],p['wmuL'],p['wsdL'],show=False,vertical_layer=4)
 p.update({'w0':w,'w0_without_locus':wt,'locus_without_w0':wl,'locus_mask':lmask,'neur_coords':nc,'neur_grid':ng})
 env=SpatialKuramoto(p);obs,_=env.reset(seed=seed);c=Path(checkpoint);rewards=[];wave=[];start=0
 if c.exists():
  with np.load(c,allow_pickle=False) as state:
   assert state['action'].item()==action and state['seed'].item()==seed and state['target'].item()==total_steps
   start=int(state['current_step']);env.current_step=start;env.current_time=float(state['current_time']);env.sol_state=state['sol_state'];env.theta_state=state['theta_state'];
   rewards=state['rewards'].tolist();wave=state['wave'].tolist()
  print('RESUME',start,'theta',env.theta_state.shape,'time',env.current_time,flush=True)
 limit=min(total_steps,start+chunk_steps);t=time.monotonic()
 for i in range(start,limit):
  obs,r,done,trunc,_=env.step(np.array([action],dtype=np.float32));rewards.append(float(r));wave.extend(np.asarray(env.theta_mean,dtype=float).tolist())
  if (i+1)%500==0:print('step',i+1,'elapsed',round(time.monotonic()-t,2),'sec',flush=True)
  if (done or trunc) and i+1<total_steps:raise RuntimeError(f'Environment ended before requested {total_steps} at step {i+1}')
 array=np.asarray(wave,dtype=float)
 np.savez_compressed(c,action=action,seed=seed,target=total_steps,current_step=limit,current_time=env.current_time,
  sol_state=np.asarray(env.sol_state),theta_state=np.asarray(env.theta_state),rewards=np.asarray(rewards),wave=array)
 result={'source':'https://github.com/NevVerVer/DBS-Gym','source_commit':'aa0b10b9502e4f62dea755ce6468da024235c319',
  'policy': 'DBS OFF' if action==0 else 'HF-DBS +5V' if action==1 else f'fixed action {action}',
  'action_normalized':action,'physical_volts':float(env.rescale_action(action)),'n_neurons':512,'grid':[8,8,8],
  'env':'env0','seed':seed,'steps_completed':limit,'target_steps':total_steps,'upstream_config_episode_units':p['total_episode_len'],
  'mean_step_reward':float(np.mean(rewards)),'mean_step_beta_component':float(np.mean(-np.asarray(rewards)-.01*abs(float(env.rescale_action(action))))),
  'global_low_beta_power':float(calc_beta_band_power(array,units2sec(p['verbose_dt']),12.5,21)),
  'absolute_stimulation_energy':float(limit*abs(float(env.rescale_action(action)))),'lfp_points':len(array),
  'lfp_sha256':hashlib.sha256(array.tobytes()).hexdigest(),
  'comparison_caveat':'Full neuron count and upstream Env0 episode; one seed and custom full-LFP aggregation differ from published six-run Table 1. Not yet a reproduced published statistic.'}
 out=c.with_suffix('.json');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
 return result
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--upstream',required=True);a.add_argument('--checkpoint',required=True);a.add_argument('--action',type=float,required=True);a.add_argument('--target',type=int,default=5555);a.add_argument('--chunk-size',type=int,default=1600);a.add_argument('--seed',type=int,default=222);x=a.parse_args();chunk(x.upstream,x.checkpoint,x.action,x.target,x.chunk_size,x.seed)

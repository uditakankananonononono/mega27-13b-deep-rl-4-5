"""Scaled DBS-Gym smoke benchmark: one episode each for fixed-off and fixed-high.
Upstream code unmodified; reduced 8-neuron configuration is NOT published KDD benchmark.
"""
import argparse,copy,json,sys
from pathlib import Path
import numpy as np

def config(upstream,seed=222):
 sys.path.insert(0,str(Path(upstream)))
 from environment.env_configs.env0 import params_dict_train
 from environment.utils import generate_w0_with_locus
 p=copy.deepcopy(params_dict_train);p.update({'num_oscillators':8,'grid_size':[2,2,2],'elec_coords':[[1,1,1]],'rec_coords':[[0,0,0]],'locus_center':[1,1,1],
  'transient_state_len':10.,'observe_wind_counts':3,'total_episode_len':4.,'verbose_dt':.05,'reward_func':'bbpow_action','log_path':None,'verbose':0,'rand_seed':seed})
 np.random.seed(seed)
 w,nc,ng,wt,wl,lmask=generate_w0_with_locus(8,p['grid_size'],p['coord_modif'],p['locus_center'],p['locus_size'],p['wmuL'],p['wsdL'],show=False,vertical_layer=1)
 p.update({'w0':w,'w0_without_locus':wt,'locus_without_w0':wl,'locus_mask':lmask,'neur_coords':nc,'neur_grid':ng});return p

def run(upstream,out):
 from environment.env import SpatialKuramoto
 p=config(upstream);results={}
 for name,action in [('off',0.0),('high',1.0)]:
  env=SpatialKuramoto(copy.deepcopy(p));obs,info=env.reset(seed=222);returns=[];lows=[]
  for i in range(4):
   obs,r,done,trunc,_=env.step(np.array([action],dtype=np.float32));returns.append(float(r));lows.append(float(np.mean(obs)))
   if done or trunc:break
  results[name]={'actions':len(returns),'mean_reward':float(np.mean(returns)),'sum_reward':float(np.sum(returns)),'mean_observed_lfp':float(np.mean(lows)),'individual_rewards':returns}
  env.close()
 d={'upstream':'https://github.com/NevVerVer/DBS-Gym','upstream_snapshot':'aa0b10b9502e4f62dea755ce6468da024235c319',
  'configuration':'actual upstream SpatialKuramoto env0, reduced to 8 neurons on 2x2x2 grid, 4 action steps, reward bbpow_action; not published 512-neuron 5000-unit benchmark',
  'dependencies_used':['Gymnasium','JAX','Diffrax','DBS-Gym','SciPy','NumPy'], 'policies':results,
  'caveat':'Smoke test only. Reward and LFP at tiny size/episode cannot be compared with published KDD baselines or interpreted clinically. Prior synthetic neural policy has different observation/action semantics and was not evaluated here.'}
 Path(out).write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2));return d
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--upstream',required=True);a.add_argument('--out',default='results/parkinson_native_smoke.json');args=a.parse_args();sys.path.insert(0,args.upstream);run(args.upstream,args.out)

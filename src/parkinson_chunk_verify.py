"""Compare resumed 50+50 Env0 steps with uninterrupted 100 steps, same seed."""
import copy,sys,json
import numpy as np
from pathlib import Path

def verify(upstream,check):
 sys.path.insert(0,upstream)
 from environment.env import SpatialKuramoto
 from environment.env_configs.env0 import params_dict_train
 from environment.utils import generate_w0_with_locus,calc_beta_band_power,units2sec
 p=copy.deepcopy(params_dict_train);p['reward_func']='bbpow_action';p['verbose']=0;p['rand_seed']=222
 np.random.seed(222)
 w,nc,ng,wt,wl,lmask=generate_w0_with_locus(512,p['grid_size'],p['coord_modif'],p['locus_center'],p['locus_size'],p['wmuL'],p['wsdL'],show=False,vertical_layer=4)
 p.update({'w0':w,'w0_without_locus':wt,'locus_without_w0':wl,'locus_mask':lmask,'neur_coords':nc,'neur_grid':ng})
 e=SpatialKuramoto(p);x,_=e.reset(seed=222);r=[];wave=[]
 for i in range(100):
  x,z,d,tr,info=e.step(np.array([0.],dtype=np.float32));r.append(float(z));wave.extend(np.asarray(e.theta_mean,dtype=float).tolist())
 with np.load(check,allow_pickle=False) as state:
  a=state['rewards'];b=state['wave'];assert len(a)==100;assert len(b)==len(wave)
  diffr=float(np.max(np.abs(np.asarray(r)-a)));diffw=float(np.max(np.abs(np.asarray(wave)-b)))
  print(json.dumps({'max_reward_abs_diff':diffr,'max_lfp_abs_diff':diffw,'resume_equivalent':diffr<1e-4 and diffw<1e-4}))
  assert diffr<1e-4 and diffw<1e-4
if __name__=='__main__':verify(sys.argv[1],sys.argv[2])

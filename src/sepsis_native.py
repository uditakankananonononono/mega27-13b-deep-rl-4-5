"""Verify our sparse Bellman model against the official Gymnasium ICU-Sepsis environment.
Requires the separate upstream checkout; no upstream code vendored into this repository.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
from sepsis import load,value,policy_value

def run(upstream,table_dir,out,n=200,seed=913):
 sys.path.insert(0,str(Path(upstream)/'packages'/'icu_sepsis'))
 import icu_sepsis,gymnasium as gym
 p,r,mu,expert=load(table_dir);v,q,it,res=value(p,r)
 # Gymnasium protocol and official environment are both used, not merely imported.
 env=gym.make('Sepsis/ICU-Sepsis-v2')
 assert env.action_space.n==25
 official=env.unwrapped
 assert official.observation_space.n==716
 x=np.array([official._tx_mat[s,a] for s,a in [(0,0),(80,12),(456,24),(712,7)]])
 assert np.allclose(x,p.toarray().reshape(716,25,716)[[0,80,456,712],[0,12,24,7],:])
 # Independent trajectories, not individual patient outcomes.
 policies={'random':lambda st,g:int(g.integers(25)),'optimal':lambda st,g:int(q[st].argmax()),'expert':lambda st,g:int(g.choice(25,p=expert[st]))}
 stats={};g=np.random.default_rng(seed)
 for name,choose in policies.items():
  rewards=[];lengths=[]
  for i in range(n):
   obs,info=env.reset(seed=int(g.integers(2**31)));total=0
   for t in range(500):
    a=choose(obs,g);obs,reward,terminated,truncated,_=env.step(a);total+=reward
    if terminated or truncated:break
   rewards.append(total);lengths.append(t+1)
  stats[name]={'n':n,'mean_return':float(np.mean(rewards)),'mean_length':float(np.mean(lengths)),
    'binomial_se_return_approx':float(np.std(rewards,ddof=1)/np.sqrt(n))}
 result={'source':'https://github.com/icu-sepsis/icu-sepsis','upstream_snapshot':'c9a6d19b85d34883304afaf7cbe1ff02ede9e12b',
  'gymnasium_api':'gym.make(Sepsis/ICU-Sepsis-v2), reset, step for every policy trajectory',
  'exact_table_rows_equal_official_gym_env':True,'seed':seed,'episodes_per_policy':n,'rollouts':stats,
  'exact_table_expected_returns':{'random':.7800709044652784,'expert':.7818448901578219,'optimal':float(mu@v)},
  'scope':'same source environment, simulated trajectories; no independent patient data'}
 Path(out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--upstream',required=True);a.add_argument('--table-dir',required=True);a.add_argument('--out',default='results/sepsis_native.json');a.add_argument('--episodes',type=int,default=200);x=a.parse_args();run(x.upstream,x.table_dir,x.out,n=x.episodes)

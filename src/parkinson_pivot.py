"""Safety-cost sensitivity inside the SAME synthetic Parkinson surrogate.
Train several reward weights and audit true evaluation under alternative weights.
No assertion about actual DBS physiology.
"""
import json,argparse
import numpy as np
from net import QNet
from parkinson import ACTIONS,transition,HORIZON

def train(weight,seed=333,samples=2400,rounds=10,epochs=15):
 rng=np.random.default_rng(seed);s=np.column_stack([rng.uniform(0,1.4,samples),rng.uniform(.05,1,samples),rng.uniform(0,1,samples)]);a=rng.integers(0,3,samples);noise=rng.normal(0,.05,samples)
 nxt=np.empty_like(s);base=np.empty(samples)
 for i in range(samples):nxt[i],base[i]=transition(s[i],a[i],noise[i])
 # Base energy coefficient is .12 in source; requested preference weight substitutes rather than adds.
 r=base+(.12-weight)*ACTIONS[a]**2
 net=QNet(3,3,width=32,seed=seed)
 for k in range(rounds):
  target=net.predict(s);boot=0 if k==0 else .95*np.max(net.predict(nxt),axis=1)
  target[np.arange(samples),a]=r+boot;net.fit(s,target,epochs=epochs,seed=seed+k,lr=.001)
 return net

def evaluate(policy,seed,weight):
 rng=np.random.default_rng(seed);s=np.array([rng.uniform(.35,.9),rng.uniform(.15,.65),rng.uniform(.1,.7)]);total=0;energy=0;bursts=0
 for t in range(HORIZON):
  a=int(policy(s));s,r=transition(s,a,rng.normal(0,.05));total+=r+(.12-weight)*ACTIONS[a]**2;energy+=ACTIONS[a]**2;bursts+=int(s[0]>.5)
 return [total,energy,bursts]

def run(out):
 # Evaluate both learned policies under the SAME standard energy coefficient, then sensitivity.
 policies={'original_weight_0_12':train(.12),'heavier_weight_0_40':train(.40)}
 result={'model':'custom synthetic Parkinson beta-burst surrogate only; not DBS-Gym','train_weights':[.12,.40],
   'seeds':[70200,70299],'sample_size':100,'evaluation':{},'caveat':'The synthetic transition model is assumed, not patient-calibrated. Reward weights represent fictional preference penalties. No clinical safety or published benchmark comparison.'}
 for weight in (.12,.40):
  rows={name:np.array([evaluate(lambda s,m=model:np.argmax(m.predict(s[None,:])[0]),seed,weight) for seed in range(70200,70300)]) for name,model in policies.items()}
  result['evaluation'][str(weight)]={name:{'mean_reward':float(v[:,0].mean()),'mean_energy':float(v[:,1].mean()),'mean_bursts':float(v[:,2].mean())} for name,v in rows.items()}
  result['evaluation'][str(weight)]['paired_heavy_minus_original_mean_reward']=float((rows['heavier_weight_0_40'][:,0]-rows['original_weight_0_12'][:,0]).mean())
 with open(out,'w') as f:json.dump(result,f,indent=2)
 print(json.dumps(result,indent=2));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',default='results/parkinson_pivot.json');a=p.parse_args();run(a.out)

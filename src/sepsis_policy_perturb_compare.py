"""Compare fixed model policies on identical synthetic transition perturbations, not patients."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from sepsis import load,value,policy_value
from net import QNet

def run(tables,out,draws=12,concentration=10,seed=781):
 p,r,mu,_=load(tables);opt,q,_,_=value(p,r);best=np.argmax(q,axis=1);n=716*25
 # Fixed split 31 from prior state-ID holdout. Both trained on original model labels.
 train=np.sort(np.random.default_rng(31).permutation(713)[:570]);y=np.zeros((570,25));y[np.arange(570),best[train]] = 1
 centers=np.loadtxt(Path(tables)/'extras/stateClusterCenters.csv',delimiter=',');assert centers.shape==(716,47)
 inputs={'onehot':np.eye(716),'centroid':np.clip(np.nan_to_num(centers,nan=0,posinf=0,neginf=0),-15,15)/15};actions={'exact_original_optimal':best}
 for name,x in inputs.items():
  net=QNet(x.shape[1],25,width=64,seed=31);net.fit(x[train],y,epochs=150,seed=131,batch=96)
  actions[name]=net.predict(x).argmax(1)
 base={k:policy_value(p,r,mu,a)['expected_return'] for k,a in actions.items()}
 rng=np.random.default_rng(seed);m=p.tocsr();supports=[m.indices[m.indptr[i]:m.indptr[i+1]] for i in range(n)];weights=[m.data[m.indptr[i]:m.indptr[i+1]] for i in range(n)];assert all(len(k) for k in supports)
 histories={k:[] for k in actions}
 for j in range(draws):
  data=[]
  for idx,prob in zip(supports,weights):
   data.extend([1.] if len(idx)==1 else rng.dirichlet(prob*concentration).tolist())
  perturbed=csr_matrix((np.array(data),m.indices,m.indptr),shape=m.shape)
  for name,a in actions.items():histories[name].append(policy_value(perturbed,r,mu,a)['expected_return'])
  print('draw',j+1,flush=True)
 result={'source':'https://github.com/icu-sepsis/icu-sepsis','seed':seed,'pseudo_concentration':concentration,'draws':draws,
 'protocol':'Same stochastic transition draw across three fixed policies; one-hot and centroid each trained 150 epochs with original-model best labels on split seed 31 train IDs; exact original best map included as reference. Nonzero transition support held fixed.',
 'original_model_values':base,'perturbed_model_values':histories,
 'paired_centroid_minus_onehot':list(map(float,np.array(histories['centroid'])-histories['onehot'])),
 'paired_original_optimal_minus_onehot':list(map(float,np.array(histories['exact_original_optimal'])-histories['onehot'])),
 'limitation':'Pseudo concentration is invented, not patient transition counts. Perturbed models are NOT independent hospitals, holdout patients, uncertainty intervals, or clinically valid policy evaluations. Policies were trained with exact labels from original same MDP, and split 31 was previously inspected; no selection inference.'}
 Path(out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'original':base,'mean_each':{k:float(np.mean(v)) for k,v in histories.items()},'centroid_better_draws':sum(v>0 for v in result['paired_centroid_minus_onehot'])},indent=2));return result
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--tables',required=True);a.add_argument('--out',default='results/sepsis_policy_perturb_compare.json');a.add_argument('--draws',type=int,default=12);a.add_argument('--concentration',type=float,default=10);x=a.parse_args();run(x.tables,x.out,x.draws,x.concentration)

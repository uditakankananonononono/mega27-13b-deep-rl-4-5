"""Equal-epoch one-hot versus centroid exact-label experiment on fixed state splits."""
import argparse,json
from pathlib import Path
import numpy as np
from net import QNet
from sepsis import load,value,policy_value

def run(table,out,epochs=150):
 p,r,mu,_=load(table);v,q,_,_=value(p,r);best=q[:713].argmax(1)
 raw=np.loadtxt(Path(table)/'extras/stateClusterCenters.csv',delimiter=',')
 assert raw.shape==(716,47)
 inputs={'onehot':np.eye(716),'centroid':np.clip(np.nan_to_num(raw,nan=0,posinf=0,neginf=0),-15,15)/15}
 result=[]
 for seed in (31,47,83):
  perm=np.random.default_rng(seed).permutation(713);train=np.sort(perm[:570]);test=np.sort(perm[570:]);y=np.zeros((570,25));y[np.arange(570),best[train]]=1
  row={'seed':seed,'train_states':train.tolist(),'test_states':test.tolist(),'controllers':{}}
  for name,x in inputs.items():
   net=QNet(x.shape[1],25,width=64,seed=seed);loss=net.fit(x[train],y,epochs=epochs,seed=seed+100,batch=96)
   action=net.predict(x).argmax(1)
   row['controllers'][name]={'train_match':float(np.mean(action[train]==best[train])),'heldout_match':float(np.mean(action[test]==best[test])),
     'heldout_mean_q_shortfall':float(np.mean(q[test,best[test]]-q[test,action[test]])),
     'model_return':policy_value(p,r,mu,action)['expected_return'],'end_train_huber_loss':loss[-1]}
  result.append(row)
 d={'source':'https://github.com/icu-sepsis/icu-sepsis','method':'Same three 570/143 state splits, network width 64, exact best-action targets, 150 epochs, seed, learning rate, batch size for one-hot and centroid. Only input representation/dimension differs; one-hot 716 features versus centroid 47, so parameter count and effective capacity also differ.',
 'split_results':result,'exact_model_optimal_value':float(mu@v),
 'limitation':'Equal-epoch same-MDP comparison, not equal parameter budget, independent transition-model validation or a patient treatment-policy test. Hyperparameters selected after earlier pilots and the split seeds have been viewed.'}
 Path(out).write_text(json.dumps(d,indent=2)+'\n');print(json.dumps([{'seed':t['seed'],'controllers':t['controllers']} for t in result],indent=2));return d
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--table',required=True);a.add_argument('--out',default='results/sepsis_budget_match.json');a.add_argument('--epochs',type=int,default=150);x=a.parse_args();run(x.table,x.out,x.epochs)

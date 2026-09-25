"""Same-MDP state holdout using ICU-Sepsis 47 cluster centroids; no patient validation."""
import argparse,json
from pathlib import Path
import numpy as np
from net import QNet
from sepsis import load,value,policy_value

def run(table,out,epochs=150):
 p,r,mu,_=load(table);v,q,_,_=value(p,r);best=q[:713].argmax(1)
 centers=np.loadtxt(Path(table)/'extras/stateClusterCenters.csv',delimiter=',');assert centers.shape==(716,47)
 x=np.clip(np.nan_to_num(centers,nan=0,posinf=0,neginf=0),-15,15)/15
 records=[]
 for seed in (31,47,83):
  rng=np.random.default_rng(seed);perm=rng.permutation(713);train=np.sort(perm[:570]);test=np.sort(perm[570:]);y=np.zeros((len(train),25));y[np.arange(len(train)),best[train]]=1
  net=QNet(47,25,width=64,seed=seed);net.fit(x[train],y,epochs=epochs,seed=seed+100,batch=96)
  predicted=net.predict(x).argmax(1);ret=policy_value(p,r,mu,predicted)['expected_return']
  records.append({'seed':seed,'train_states':train.tolist(),'test_states':test.tolist(),'train_action_match':float(np.mean(predicted[train]==best[train])),'heldout_action_match':float(np.mean(predicted[test]==best[test])),'heldout_exact_q_gap_mean':float(np.mean(q[test,best[test]]-q[test,predicted[test]])),'full_model_expected_return':ret})
 d={'source':'https://github.com/icu-sepsis/icu-sepsis','method':'47 published state centroid dimensions, clipping and /15 scaling fixed before split; exact best-action one-hot training labels on 570 states, 143 held out, width 64 x 150 epochs; same three seeds as one-hot audit',
  'exact_model_optimal_value':float(mu@v),'splits':records,'limitation':'Held-out state IDs all belong to the same estimated MDP used to compute exact labels. Neither independent transition model nor patient treatment outcomes; centroid features may or may not transfer to another hospital and contain no transition-count uncertainty.'}
 Path(out).write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({'splits':[{k:v for k,v in x.items() if k not in ('train_states','test_states')} for x in records]},indent=2));return d
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--table',required=True);a.add_argument('--out',default='results/sepsis_centroid_holdout.json');a.add_argument('--epochs',type=int,default=150);x=a.parse_args();run(x.table,x.out,x.epochs)

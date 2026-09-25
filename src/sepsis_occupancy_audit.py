"""Exact in-model occupancy audit for state-holdout policies, never patient occupancy."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve
from net import QNet
from sepsis import load,value,policy_value

def occupancy(p,mu,action):
 n=713;rows=np.arange(n)*25+action[:n];trans=p[rows,:n].tocsr()
 visits=spsolve(eye(n,format='csc')-trans.T.tocsc(),mu[:n])
 assert np.all(visits>=-1e-8), 'Expected transient visits must be nonnegative'
 assert np.linalg.norm(visits-(mu[:n]+trans.T@visits),ord=np.inf)<1e-8
 return np.maximum(visits,0)

def run(table,out,epochs_onehot=100,epochs_centroid=150):
 p,r,mu,_=load(table);v,q,_,_=value(p,r);best=np.argmax(q[:713],axis=1)
 centers=np.loadtxt(Path(table)/'extras/stateClusterCenters.csv',delimiter=',');c=np.clip(np.nan_to_num(centers,nan=0,posinf=0,neginf=0),-15,15)/15
 x=np.eye(716);results=[]
 for seed in (31,47,83):
  perm=np.random.default_rng(seed).permutation(713);train=np.sort(perm[:570]);test=np.sort(perm[570:]);y=np.zeros((570,25));y[np.arange(570),best[train]]=1
  split={'seed':seed,'test_count':len(test),'policies':{}}
  for name,inputs,epochs in (('onehot',x,epochs_onehot),('centroid',c,epochs_centroid)):
   net=QNet(inputs.shape[1],25,width=64,seed=seed);net.fit(inputs[train],y,epochs=epochs,seed=seed+100,batch=96)
   action=np.argmax(net.predict(inputs),axis=1);visit=occupancy(p,mu,action)
   # First-decision regret at the policy's actual visited states under optimal continuation.
   gap=q[:713].max(1)-q[np.arange(713),action[:713]]
   polvalue=policy_value(p,r,mu,action)['expected_return']
   split['policies'][name]={'model_value':polvalue,'visits_per_initial_case':float(np.sum(visit)),
    'visit_mass_test_fraction':float(np.sum(visit[test])/np.sum(visit)),
    'visit_weighted_action_match':float(np.dot(visit,action[:713]==best)/np.sum(visit)),
    'visit_weighted_exact_q_shortfall':float(np.dot(visit,gap)),
    'train_visit_fraction':float(np.sum(visit[train])/np.sum(visit)),
    'initial_distribution_exact_q_shortfall':float(np.dot(mu[:713],gap)),
    'train_visit_weighted_shortfall':float(np.dot(visit[train],gap[train])),
    'heldout_visit_weighted_shortfall':float(np.dot(visit[test],gap[test]))}
  results.append(split)
 d={'source':'https://github.com/icu-sepsis/icu-sepsis','model':'released ICU-Sepsis MDP; first 713 nonterminal states; row-selected p for deterministic learned actions',
 'definition':'Expected undiscounted visits d solves (I-P_pi,transposed)d=mu for transient states. Exact Q gap at each visited state assumes optimal continuation after that action. Sum d(s)*[V*(s)-Q*(s,pi(s))] equals J(optimal)-J(pi) under proper absorbing MDP and converged Bellman values.',
 'optimal_model_value':float(mu@v),'splits':results,
 'limitations':'This is model-internal occupancy under two policies trained on exact same-model labels. It cannot estimate patient state visitation, causal clinical regret, independent transport or treatment safety.'}
 Path(out).write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(results,indent=2));return d
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--table',required=True);a.add_argument('--out',default='results/sepsis_occupancy_audit.json');z=a.parse_args();run(z.table,z.out)

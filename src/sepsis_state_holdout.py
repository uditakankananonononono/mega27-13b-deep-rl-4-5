"""Internal state-label holdout stress test, not independent MDP/patient validation."""
import argparse,json
from pathlib import Path
import numpy as np
from net import QNet
from sepsis import load,value,policy_value

def run(table,out,epochs=100):
 p,r,mu,_=load(table);optimal,q,_,_=value(p,r)
 best=np.argmax(q[:713],axis=1);records=[]
 for seed in (31,47,83):
  rng=np.random.default_rng(seed);permutation=rng.permutation(713);train=np.sort(permutation[:570]);test=np.sort(permutation[570:]);assert not set(train)&set(test)
  x=np.eye(716);label=np.zeros((len(train),25));label[np.arange(len(train)),best[train]]=1
  net=QNet(716,25,width=64,seed=seed);net.fit(x[train],label,epochs=epochs,seed=seed+100,batch=96)
  predicted=net.predict(x).argmax(1)
  result=policy_value(p,r,mu,predicted)
  majority=np.bincount(best[train],minlength=25).argmax()
  records.append({'seed':seed,'train_states':train.tolist(),'test_states':test.tolist(),
    'train_action_match':float(np.mean(predicted[train]==best[train])),
    'heldout_action_match':float(np.mean(predicted[test]==best[test])),
    'heldout_majority_baseline':float(np.mean(best[test]==majority)),
    'heldout_exact_q_gap_mean':float(np.mean(q[test,best[test]]-q[test,predicted[test]])),
    'full_model_expected_return':result['expected_return']})
 d={'source':'https://github.com/icu-sepsis/icu-sepsis','epochs':epochs,'method':'Three predetermined 570/143 state-label holdouts, width-64 one-hot QNet trained on exact best-action labels only for 570 states per split; same model and one-hot state IDs, no patient independence.',
 'exact_model_optimal_value':float(mu@optimal),'splits':records,
 'limitation':'One-hot unseen states have no learned state representation; failure is an expected representation diagnostic, not external model generalization or a clinical treatment-policy estimate. Holdout labels arise from the same published transition model and no independent patients.'}
 Path(out).write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({'splits':[{k:v for k,v in x.items() if k not in ('train_states','test_states')} for x in records]},indent=2));return d
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--table',required=True);a.add_argument('--out',default='results/sepsis_state_holdout.json');a.add_argument('--epochs',type=int,default=100);x=a.parse_args();run(x.table,x.out,x.epochs)

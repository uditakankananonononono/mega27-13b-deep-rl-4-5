"""Model-only action-ranking pivot with exact Q-derived labels; no patient policy validation."""
import argparse,json
from pathlib import Path
import numpy as np
from net import QNet
from sepsis import load,value,policy_value

def run(table,out,epochs=150):
 p,r,mu,expert=load(table);v,q,_,_=value(p,r);x=np.eye(716);best=q.argmax(1)
 # Huber regression to a large action-margin target, not clinical outcomes.
 # Rank target differs from Q regression: every suboptimal action label 0, exact best 1.
 target=np.zeros((716,25));target[np.arange(713),best[:713]]=1
 outputs={}
 for width in (32,64,128):
  net=QNet(716,25,width=width,seed=271+width);loss=net.fit(x,target,epochs=epochs,seed=517+width,lr=.002,batch=96)
  a=net.predict(x).argmax(1);d=policy_value(p,r,mu,a)
  outputs[str(width)]={'epochs':epochs,'terminal_huber_loss':loss[-1],'exact_optimal_action_match':float(np.mean(a[:713]==best[:713])),'model_expected_return':d['expected_return'],'gap_from_exact_optimal':float(mu@v-d['expected_return'])}
 result={'source':'https://github.com/icu-sepsis/icu-sepsis','method':'QNet width sweep on exact-model best-action one-hot target, 150 epochs; no independent patients, no held-out model',
   'initial_state_optimal_value':float(mu@v),'same_model_baseline_neural_q_regression':0.7864485428365543,'models':outputs,
   'selection_note':'All 713 nonterminal exact-best actions were used as training labels. 100% agreement is training-state memorization, NOT out-of-sample generalization. All three models evaluated on the same simulator that generated their labels; no scientific or clinical improvement over exact value iteration.',
   'clinical_limitation':'Exact-best labels are produced by the published model, not observed treatment benefit. This model-only matching is a tautological mimic of an exact dynamic-programming optimum, not deep RL discovery, independent validation or clinical safety.'}
 Path(out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--table',required=True);a.add_argument('--out',default='results/sepsis_rank_pivot.json');a.add_argument('--epochs',type=int,default=150);x=a.parse_args();run(x.table,x.out,x.epochs)

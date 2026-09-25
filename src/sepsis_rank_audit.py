"""Diagnose why low neural Q regression error gives weak ICU-Sepsis control.
Reproduce fixed seed and exact table Q labels; model-only, not patient outcome.
"""
import argparse,json
from pathlib import Path
import numpy as np
from net import QNet
from sepsis import load,value,policy_value

def run(table,out,epochs=50):
 p,r,mu,expert=load(table);opt,q,_,_=value(p,r);x=np.eye(716);target=q.copy();target[713:715]=0
 net=QNet(716,25,width=64,seed=27);loss=net.fit(x,target,epochs=epochs,seed=27,batch=128)
 pred=net.predict(x);a=pred.argmax(1);best=q.argmax(1);mu=np.ravel(mu);qgap=q[np.arange(713),best[:713]]-q[np.arange(713),a[:713]]
 advantage=q[:713].max(1)-np.partition(q[:713],-2,axis=1)[:,-2]
 randomv,_,_,_=value(p,r,np.full_like(expert,1/25));policy=policy_value(p,r,mu,a)
 d={'source':'https://github.com/icu-sepsis/icu-sepsis','epochs':epochs,'training_loss_last':loss[-1],
 'model_mse_nonterminal':float(np.mean((pred[:713]-q[:713])**2)), 'action_agreement_nonterminal':float(np.mean(a[:713]==best[:713])),
 'state_regret_under_exact_q':{'mean':float(qgap.mean()),'median':float(np.median(qgap)),'p95':float(np.quantile(qgap,.95)),'max':float(qgap.max()),'positive_regret_fraction':float(np.mean(qgap>1e-10))},
 'optimal_action_margin':{'median':float(np.median(advantage)),'p95':float(np.quantile(advantage,.95))},
 'initial_mass_mismatched_action':float(mu[(a!=best)].sum()),'initial_mass_regret':float(np.sum(mu[:713]*qgap)),
 'model_policy_value':policy['expected_return'],'optimal_model_value':float(mu@opt),'random_model_value':float(mu@randomv),
 'top_10_initial_mass_regret_states':[{'state':int(i),'initial_prob':float(mu[i]),'exact_best_action':int(best[i]),'neural_action':int(a[i]),'exact_q_gap':float(qgap[i]),'weighted_gap':float(mu[i]*qgap[i])} for i in np.argsort(mu[:713]*qgap)[-10:][::-1]],
 'limitations':'Q-gap is under the same fitted simulator and one step assuming optimal continuation, not clinical regret. Neural network trains on exact Q labels from all model states; no independent patient outcomes or off-policy evaluation. Initial-mass gap is not total policy-value gap because later mismatches compound.'}
 Path(out).write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({k:v for k,v in d.items() if k!='top_10_initial_mass_regret_states'},indent=2));return d
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--table',required=True);a.add_argument('--out',default='results/sepsis_rank_audit.json');x=a.parse_args();run(x.table,x.out)

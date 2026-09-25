"""Exact MDP baselines and neural fitted-Q distillation in published ICU-Sepsis.
This is a simulation, never evidence of clinical safety or patient survival.
"""
import argparse,json,hashlib
import numpy as np
from scipy.sparse import csr_matrix
from net import QNet

def load(directory):
    d={k:np.loadtxt(f'{directory}/{k}.csv',delimiter=',') for k in ('transitionFunction','rewardFunction','initialStateDistribution','expertPolicy')}
    p=d['transitionFunction'].reshape((716,25,716));r=d['rewardFunction'];mu=d['initialStateDistribution'];expert=d['expertPolicy']
    assert np.allclose(p.sum(axis=2),1) and np.isclose(mu.sum(),1) and np.allclose(expert[:713].sum(axis=1),1)
    assert np.array_equal(np.flatnonzero(r),[714]);return csr_matrix(p.reshape(-1,716)),r,mu,expert

def value(p,r,policy=None,tol=1e-10,maxit=2000):
    """Terminal value zero AFTER arrival reward; gamma=1, no repeated terminal reward."""
    n=716;v=np.zeros(n); active=np.arange(n)!=713;active&=np.arange(n)!=714;active&=np.arange(n)!=715
    for it in range(maxit):
        q=np.asarray(p@(r+v)).reshape((716,25))
        new=q.max(axis=1) if policy is None else np.einsum('sa,sa->s',policy,q)
        new[~active]=0
        delta=np.max(np.abs(new-v));v=new
        if delta<tol:break
    return v,q,it+1,float(delta)

def policy_value(p,r,mu,action):
    policy=np.zeros((len(action),25));policy[np.arange(len(action)),action]=1
    v,_,it,res=value(p,r,policy)
    return {'expected_return':float(mu@v),'iterations':it,'residual':res}

def run(directory,out,epochs=100):
    p,r,mu,expert=load(directory); opt,q,it,res=value(p,r)
    states=np.arange(716);x=np.eye(716);q_target=q.copy();q_target[713:715]=0
    net=QNet(716,25,width=64,seed=27);loss=net.fit(x,q_target,epochs=epochs,seed=27,batch=128)
    action=np.argmax(net.predict(x),axis=1);randompolicy=np.full_like(expert,1/25)
    rv,_,_,_=value(p,r,randompolicy);ev,_,_,_=value(p,r,expert)
    outputs={'model':'ICU-Sepsis published tabular MDP v2; exact transition tables, not original MIMIC patient records','transition_source':'https://github.com/icu-sepsis/icu-sepsis/blob/main/icu-sepsis-csv-tables.tar.gz',
        'published_reference':'https://github.com/icu-sepsis/icu-sepsis#baselines','published_reference_values':{'random':.78,'expert':.78,'optimal':.88},
        'our_exact_values':{'random':float(mu@rv),'expert':float(mu@ev),'optimal':float(mu@opt)},'neural_greedy':policy_value(p,r,mu,action),
        'neural_agreement_with_optimal_states':float(np.mean(action[:713]==np.argmax(q[:713],axis=1))),
        'training':{'epochs':epochs,'start_loss':loss[0],'end_loss':loss[-1],'seed':27},
        'value_iteration':{'iterations':it,'residual':res},'negative_or_caveat':'Deep network is trained on exact model-derived Q labels and same published MDP, not held-out patient data. Does not demonstrate treatment benefit or a benchmark break.'}
    with open(out,'w') as f:json.dump(outputs,f,indent=2)
    print(json.dumps(outputs,indent=2));return outputs
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--table-dir',required=True);a.add_argument('--out',default='results/sepsis.json');a.add_argument('--epochs',type=int,default=100);args=a.parse_args();run(args.table_dir,args.out,args.epochs)

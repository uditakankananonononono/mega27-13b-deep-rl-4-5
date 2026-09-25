"""Within-project pivot: state-cluster-feature network and SOFA-stratified evaluation.
Avoid representing improved architecture as a clinical improvement.
"""
import argparse,json
from pathlib import Path
import numpy as np
from net import QNet
from sepsis import load,value,policy_value

def run(table_dir,out,epochs=100):
 p,r,mu,expert=load(table_dir)
 centers=np.loadtxt(Path(table_dir)/'extras/stateClusterCenters.csv',delimiter=',')
 sofa=np.loadtxt(Path(table_dir)/'extras/sofaScores.csv',delimiter=',')
 assert centers.shape==(716,47) and sofa.shape==(716,)
 v,q,it,delta=value(p,r);q[713:]=0
 # To evaluate state abstraction and whether it reduces overfitting, standardize only the nonterminal states.
 x=np.nan_to_num(centers,nan=0,posinf=0,neginf=0);x=np.clip(x,-15,15)/15
 net=QNet(47,25,width=128,seed=31);loss=net.fit(x,q,epochs=epochs,seed=31,batch=96,lr=.0015)
 action=net.predict(x).argmax(1);neural=policy_value(p,r,mu,action)
 # Stratification is over initial-state mass, not risk-adjusted clinical cohort.
 strata={}
 for label,select in [('lower_sofa',sofa<7),('higher_sofa',sofa>=7)]:
  # Terminal initial mass is not assumed; normalize within stratum.
  local=mu*select;mass=local.sum();local/=mass
  pol=np.zeros((716,25));pol[np.arange(716),action]=1
  vn,_,_,_=value(p,r,pol)
  local_expert=expert.copy();local_expert[713:]=1/25
  ve,_,_,_=value(p,r,local_expert)
  strata[label]={'initial_mass':float(mass),'n_states':int(select.sum()),'neural_expected_reward':float(local@vn),'expert_expected_reward':float(local@ve)}
 result={'model':'ICU-Sepsis v2 same exact transition model; 47 published state cluster center features','source':'https://github.com/icu-sepsis/icu-sepsis/blob/main/icu-sepsis-csv-tables.tar.gz',
   'training_epochs':epochs,'training_loss_start_end':[loss[0],loss[-1]],'feature_dim':47,'neural_expected_return':neural['expected_return'],
   'optimal_expected_return':float(mu@v),'action_match_optimal_states':float(np.mean(action[:713]==np.argmax(q[:713],axis=1))),
   'sofa_initial_state_strata':strata,
   'caveat':'SOFA is a model table annotation, not separate clinical validation. Same-source model-derived Q targets. State-cluster features may create unequal approximation error across strata. No causal safety claim.'}
 Path(out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--table-dir',required=True);a.add_argument('--out',default='results/sepsis_pivot.json');a.add_argument('--epochs',type=int,default=100);args=a.parse_args();run(args.table_dir,args.out,args.epochs)

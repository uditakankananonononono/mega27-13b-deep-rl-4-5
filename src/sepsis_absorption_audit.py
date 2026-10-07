"""Released model properness, not patient termination or safety."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from sepsis import load,value
ROOT=Path(__file__).resolve().parents[1]
def compute(upstream,table):
 path=ROOT/'src/sepsis_absorption_plan.json';plan=json.loads(path.read_bytes());table=Path(table)
 if subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD']).decode().strip()!=plan['source_commit']:raise ValueError('source pin')
 p,r,mu,expert=load(table);opt,q,_,_=value(p,r);greedy=np.zeros_like(expert);greedy[np.arange(716),q.argmax(1)]=1;policies={'expert':expert,'uniform':np.full_like(expert,1/25),'optimal_argmax':greedy};rows=[]
 for name,pi in policies.items():
  selector=csr_matrix((pi[:713].ravel(),(np.repeat(np.arange(713),25),np.arange(713*25))),shape=(713,713*25));full=(selector@p[:713*25]).toarray();T=full[:,:713];C=full[:,713:];eig=np.linalg.eigvals(T);rho=float(np.max(np.abs(eig)));A=np.eye(713)-T;B=np.linalg.solve(A,C);duration=np.linalg.solve(A,np.ones(713));res=float(np.max(np.abs(A@B-C)));sumres=float(np.max(np.abs(B.sum(1)-1)));v,_,_,_=value(p,r,pi);arrival=float(mu[:713]@B[:,1]);bellman=float(mu@v)
  if rho>=1 or res>1e-8 or sumres>1e-8 or B.min()<-1e-8 or abs(arrival-bellman)>1e-8:raise ValueError('properness/absorption check')
  rows.append({'policy':name,'transient_spectral_radius':rho,'max_abs_absorption_residual':res,'max_abs_absorption_sum_minus_one':sumres,'minimum_absorption_probability':float(B.min()),'initial_terminal_absorption':[float(mu[:713]@B[:,i]) for i in range(3)],'survival_absorption_minus_bellman':arrival-bellman,'initial_expected_decisions':float(mu[:713]@duration),'min_state_expected_decisions':float(duration.min()),'max_state_expected_decisions':float(duration.max()),'states':[{'state':i,'absorption':B[i].tolist(),'expected_decisions':float(duration[i])} for i in range(713)]})
 j={'plan_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_commit':plan['source_commit'],'source_url':'https://github.com/icu-sepsis/icu-sepsis','file_sha256':{f:hashlib.sha256((table/f).read_bytes()).hexdigest() for f in ['transitionFunction.csv','rewardFunction.csv','initialStateDistribution.csv','expertPolicy.csv']},'terminal_indices':[713,714,715],'terminal_labels':['death','survival','additional terminal,interpretation not inferred'],'policies':rows,'limits':plan['limits']};(ROOT/'results/sepsis_absorption_audit.json').write_text(json.dumps(j,indent=2)+'\n');print([{k:v for k,v in r.items() if k!='states'} for r in rows]);return j
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--upstream',required=True);a.add_argument('--table',required=True);x=a.parse_args();compute(x.upstream,x.table)

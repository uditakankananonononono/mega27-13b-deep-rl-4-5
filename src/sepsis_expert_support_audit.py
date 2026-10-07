"""Aggregate expert probability versus estimator support, not patient safety."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix,eye
from scipy.sparse.linalg import spsolve
from sepsis import load
from sepsis_admissibility_audit import mask_from_file
ROOT=Path(__file__).resolve().parents[1]
def compute(upstream,table):
 path=ROOT/'src/sepsis_expert_support_plan.json';plan=json.loads(path.read_bytes());table=Path(table)
 if subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD']).decode().strip()!=plan['source_commit']:raise ValueError('source pin')
 p,r,mu,expert=load(table);mask=mask_from_file(table/'extras/admissibleActions.txt');n=713
 if np.min(expert)<0 or np.max(np.abs(expert[:n].sum(axis=1)-1))>1e-12:raise ValueError('expert stochasticity')
 unsupported=np.sum(expert[:n]*~mask[:n],axis=1);ii=np.repeat(np.arange(n),25);selector=csr_matrix((expert[:n].ravel(),(ii,np.arange(n*25))),shape=(n,n*25));trans=selector@p[:n*25,:n];d=spsolve(eye(n,format='csc')-trans.T.tocsc(),mu[:n]);res=float(np.linalg.norm(d-(mu[:n]+trans.T@d),ord=np.inf))
 if np.min(d)<-1e-8 or res>1e-8 or not np.all(np.isfinite(d)):raise ValueError('occupancy')
 files=['transitionFunction.csv','expertPolicy.csv','initialStateDistribution.csv','extras/admissibleActions.txt'];j={'plan_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_commit':plan['source_commit'],'source_url':'https://github.com/icu-sepsis/icu-sepsis','file_sha256':{f:hashlib.sha256((table/f).read_bytes()).hexdigest() for f in files},'expert_row_sum_max_residual':float(np.max(np.abs(expert[:n].sum(axis=1)-1))),'positive_expert_action_entries':int(np.sum(expert[:n]>0)),'positive_expert_unsupported_entries':int(np.sum((expert[:n]>0)&~mask[:n])),'states_with_positive_unsupported_mass':int(np.sum(unsupported>0)),'unweighted_state_mean_unsupported_mass':float(unsupported.mean()),'initial_weighted_unsupported_mass':float(mu[:n]@unsupported),'expert_model_expected_transient_decisions':float(d.sum()),'occupancy_weighted_unsupported_fraction':float(d@unsupported/d.sum()),'expected_unsupported_decisions_per_initial_model_case':float(d@unsupported),'occupancy_residual':res,'states':[{'state':i,'unsupported_probability':float(unsupported[i]),'expert_model_expected_visits':float(d[i]),'source_admissible_actions':int(mask[i].sum())} for i in range(n)],'limits':plan['limits']}
 (ROOT/'results/sepsis_expert_support_audit.json').write_text(json.dumps(j,indent=2)+'\n');print({k:v for k,v in j.items() if k not in ['states','file_sha256']});return j
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--upstream',required=True);a.add_argument('--table',required=True);x=a.parse_args();compute(x.upstream,x.table)

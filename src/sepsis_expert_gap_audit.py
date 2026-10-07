"""Exact model performance-difference partition, never causal patient regret."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix,eye
from scipy.sparse.linalg import spsolve
from sepsis import load,value
from sepsis_admissibility_audit import mask_from_file
ROOT=Path(__file__).resolve().parents[1]
def compute(upstream,table):
 path=ROOT/'src/sepsis_expert_gap_plan.json';plan=json.loads(path.read_bytes());table=Path(table);prior=json.loads((ROOT/'results/sepsis_expert_support_audit.json').read_text())
 if subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD']).decode().strip()!=plan['source_commit']:raise ValueError('source pin')
 for f,d in prior['file_sha256'].items():
  if hashlib.sha256((table/f).read_bytes()).hexdigest()!=d:raise ValueError('table digest')
 p,r,mu,expert=load(table);mask=mask_from_file(table/'extras/admissibleActions.txt');n=713
 opt,q,oi,ores=value(p,r);ev,eq,ei,eres=value(p,r,expert)
 if max(ores,eres)>1e-8:raise ValueError('solver residual')
 selector=csr_matrix((expert[:n].ravel(),(np.repeat(np.arange(n),25),np.arange(n*25))),shape=(n,n*25));trans=selector@p[:n*25,:n];d=spsolve(eye(n,format='csc')-trans.T.tocsc(),mu[:n]);prior_d=np.array([s['expert_model_expected_visits'] for s in prior['states']])
 if np.max(np.abs(d-prior_d))>1e-8:raise ValueError('occupancy replay')
 gaps=opt[:n,None]-q[:n];weighted=d[:,None]*expert[:n]*gaps;sup=weighted*mask[:n];unsup=weighted*~mask[:n];gap=float(mu@(opt-ev));identity=float(weighted.sum()-gap)
 if gaps.min()<-1e-8 or abs(identity)>1e-8:raise ValueError('performance difference identity')
 j={'plan_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_commit':plan['source_commit'],'prior_sha256':hashlib.sha256((ROOT/'results/sepsis_expert_support_audit.json').read_bytes()).hexdigest(),'optimal_model_value':float(mu@opt),'expert_model_value':float(mu@ev),'optimal_minus_expert_model_gap':gap,'occupancy_weighted_Q_shortfall':float(weighted.sum()),'performance_difference_identity_residual':identity,'supported_gap_contribution':float(sup.sum()),'unsupported_gap_contribution':float(unsup.sum()),'unsupported_gap_share':float(unsup.sum()/gap),'minimum_Q_shortfall':float(gaps.min()),'occupancy_replay_max_abs_error':float(np.max(np.abs(d-prior_d))),'optimal_bellman_residual':ores,'expert_bellman_residual':eres,'states':[{'state':s,'occupancy':float(d[s]),'supported_gap':float(sup[s].sum()),'unsupported_gap':float(unsup[s].sum())} for s in range(n)],'limits':plan['limits']};(ROOT/'results/sepsis_expert_gap_audit.json').write_text(json.dumps(j,indent=2)+'\n');print({k:v for k,v in j.items() if k!='states'});return j
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--upstream',required=True);a.add_argument('--table',required=True);x=a.parse_args();compute(x.upstream,x.table)

"""New public admissibility metadata as a finite-table semantic oracle."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
from sepsis import load,value
ROOT=Path(__file__).resolve().parents[1]

def mask_from_file(path):
 lines=Path(path).read_text().splitlines();counts=[int(x) for x in lines[0].split()];rows=[[int(x) for x in line.split()] for line in lines[1:]]
 if len(counts)!=716 or len(rows)!=716:raise ValueError('support state count')
 mask=np.zeros((716,25),bool)
 for i,(n,row) in enumerate(zip(counts,rows)):
  if n!=len(row) or n<1 or len(set(row))!=len(row) or any(a<0 or a>=25 for a in row):raise ValueError('support row')
  mask[i,row]=True
 return mask

def restricted_value(p,r,mask,tol=1e-10,maxit=2000):
 v=np.zeros(716)
 for it in range(maxit):
  q=np.asarray(p@(r+v)).reshape(716,25);new=np.where(mask,q,-np.inf).max(axis=1);new[713:]=0;delta=float(np.max(np.abs(new-v)));v=new
  if delta<tol:break
 return v,q,it+1,delta

def run(upstream,table):
 raw=(ROOT/'src/sepsis_admissibility_plan.json').read_bytes();plan=json.loads(raw)
 if subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD']).decode().strip()!=plan['source_commit']:raise ValueError('source commit')
 table=Path(table);archive=Path(upstream)/plan['archive'];p,r,mu,expert=load(table);mask=mask_from_file(table/'extras/admissibleActions.txt');residuals=[]
 for s in range(713):
  rows=p[s*25:(s+1)*25,:].toarray();mean=rows[mask[s]].mean(axis=0)
  residuals.extend(float(np.max(np.abs(row-mean))) for row in rows[~mask[s]])
 full,q,it,res=value(p,r);restricted,rq,rit,rres=restricted_value(p,r,mask);uniform=np.full((716,25),1/25);admissible_uniform=mask/mask.sum(axis=1,keepdims=True);uv,_,_,_=value(p,r,uniform);av,_,_,_=value(p,r,admissible_uniform);actions=q.argmax(axis=1)
 files=['transitionFunction.csv','rewardFunction.csv','initialStateDistribution.csv','expertPolicy.csv','extras/admissibleActions.txt']
 out={'plan_sha256':hashlib.sha256(raw).hexdigest(),'source_url':plan['source_url'],'source_commit':plan['source_commit'],'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'file_sha256':{f:hashlib.sha256((table/f).read_bytes()).hexdigest() for f in files},'transient_support_counts':mask[:713].sum(axis=1).tolist(),'inadmissible_transient_rows':int((~mask[:713]).sum()),'total_transient_rows':713*25,'mean_fill_max_abs_residual':max(residuals,default=0),'mean_fill_rows_above_1e12':sum(v>1e-12 for v in residuals),'full_optimal':float(mu@full),'restricted_optimal':float(mu@restricted),'optimal_max_state_difference':float(np.max(np.abs(full-restricted))),'full_iterations':it,'restricted_iterations':rit,'full_residual':res,'restricted_residual':rres,'full_uniform_value':float(mu@uv),'admissible_uniform_value':float(mu@av),'uniform_max_state_difference':float(np.max(np.abs(uv-av))),'full_argmax_inadmissible_states':np.flatnonzero(~mask[np.arange(713),actions[:713]]).tolist(),'full_argmax_inadmissible_count':int(np.sum(~mask[np.arange(713),actions[:713]])),'inadmissible_argmax_max_gap_to_supported_best':float(np.max(np.abs(q[:713].max(axis=1)-np.where(mask[:713],q[:713],-np.inf).max(axis=1)))),'states_with_single_admissible_action':int(np.sum(mask[:713].sum(axis=1)==1)),'limits':plan['limits']}
 (ROOT/'results/sepsis_admissibility_audit.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k not in ['transient_support_counts','file_sha256']});return out
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--upstream',required=True);a.add_argument('--table',required=True);x=a.parse_args();run(x.upstream,x.table)

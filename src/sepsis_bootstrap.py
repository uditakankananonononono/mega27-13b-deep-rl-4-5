"""Finite-model transition uncertainty audit via Dirichlet resampling.
Pseudo-count strength is hypothetical: not a posterior over actual MIMIC patients.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from sepsis import load,value,policy_value

def run(tables,out,draws=24,concentration=100,seed=781):
 p,r,mu,expert=load(tables);opt,q,_,_=value(p,r);a=np.argmax(q,axis=1)
 rng=np.random.default_rng(seed);matrix=p.toarray();nstate=716;nact=25
 # Preserve structural support: Dirichlet on each nonzero support, pseudocount alpha=p*C.
 support=[np.flatnonzero(matrix[row]>0) for row in range(nstate*nact)]
 vals=[]
 for j in range(draws):
  ind=[];col=[];prob=[]
  for row,idx in enumerate(support):
   if len(idx)==1:draw=np.ones(1)
   else:draw=rng.dirichlet(matrix[row,idx]*concentration)
   ind.extend([row]*len(idx));col.extend(idx.tolist());prob.extend(draw.tolist())
  res=csr_matrix((prob,(ind,col)),shape=(nstate*nact,nstate))
  val=policy_value(res,r,mu,a)['expected_return'];vals.append(val)
 result={'description':'Dirichlet perturbation on nonzero model transition supports, fixed original exact-optimal action map; synthetic model sensitivity only',
    'source':'https://github.com/icu-sepsis/icu-sepsis','seed':seed,'draws':draws,'pseudo_concentration':concentration,
    'original_exact_optimal_value':float(mu@opt),'perturbed_fixed_policy_values':vals,'mean':float(np.mean(vals)),
    'p05_p95':list(map(float,np.quantile(vals,[.05,.95]))),'limitation':'Pseudocount concentration is arbitrary; transition supports held fixed. This is NOT a clinical confidence interval, bootstrap over patients, robust policy certificate or published benchmark improvement.'}
 Path(out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='perturbed_fixed_policy_values'},indent=2));return result
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--tables',required=True);a.add_argument('--out',default='results/sepsis_model_sensitivity.json');a.add_argument('--draws',type=int,default=24);a.add_argument('--concentration',type=float,default=100);x=a.parse_args();run(x.tables,x.out,x.draws,x.concentration)

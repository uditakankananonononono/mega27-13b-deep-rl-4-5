"""Exact affine frozen-policy cost envelope over lambda >=0.
Uses archived actual native rewards/actions; no retraining or simulator mutation.
"""
import json,itertools
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'results/parkinson_native_dutycycle.json';j=json.loads(p.read_text());m=j['frozen_policy_mean_beta_abs_voltage'];crossings=[0.]
 for a,b in itertools.combinations(m,2):
  A,B=m[a],m[b];den=A['mean_abs_voltage']-B['mean_abs_voltage']
  if abs(den)>1e-12:
   x=(B['mean_beta_component']-A['mean_beta_component'])/den
   if x>0:crossings.append(x)
 crossings=sorted(set(crossings));windows=[]
 for a,b in zip(crossings,crossings[1:]+[float('inf')]):
  probe=(a+b)/2 if np.isfinite(b) else a+max(1,a)
  winner=min(m,key=lambda k:m[k]['mean_beta_component']+probe*m[k]['mean_abs_voltage'])
  if windows and windows[-1]['policy']==winner:windows[-1]['upper']=None if not np.isfinite(b) else b
  else:windows.append({'policy':winner,'lower':a,'upper':None if not np.isfinite(b) else b})
 j['exact_envelope_nonnegative_lambda']=windows;j['pairwise_nonnegative_crossings']=crossings
 assert not any(x['policy']=='half' for x in windows)
 p.write_text(json.dumps(j,indent=2)+'\n');print(windows)
if __name__=='__main__':main()

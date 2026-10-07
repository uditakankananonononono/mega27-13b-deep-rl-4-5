import json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from native_penalty_audit import compute,interval

def test_envelopes_reproduce_and_match_direct_scoring():
 j=json.loads((ROOT/'results/native_penalty_audit.json').read_text());assert compute()==j
 old=json.loads((ROOT/'results/native_window_audit.json').read_text())
 for r,p in zip(j['rows'],old['rows']):
  checks=[0,.05,.2,1,5,20]
  for b in r['optimal_lambda_intervals'].values():
   if b:checks.extend([b['lower']+1e-7] if b['upper'] is None else [(b['lower']+b['upper'])/2])
  for lam in checks:
   scores={k:v['mean_beta_component']+lam*v['on_fraction'] for k,v in p['policies'].items()};best=min(scores.values())
   for k,b in r['optimal_lambda_intervals'].items():
    expected=b is not None and b['lower']<=lam and (b['upper'] is None or lam<=b['upper'])
    assert expected==(scores[k]==pytest.approx(best,abs=1e-9))
 assert all(r['optimal_lambda_intervals']['phase0'] is None and r['optimal_lambda_intervals']['phase1'] is None for r in [j['rows'][4],j['rows'][5],j['rows'][6]])

def test_parallel_and_tie_intervals():
 s={'a':{'mean_beta_component':1.,'on_fraction':0.},'b':{'mean_beta_component':1.,'on_fraction':0.}}
 assert interval('a',s)=={'lower':0.,'upper':None,'endpoints':'inclusive ties'}
 s['b']['mean_beta_component']=0.
 assert interval('a',s) is None

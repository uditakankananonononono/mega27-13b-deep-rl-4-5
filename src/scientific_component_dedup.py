"""Audit a historical mixed-component ledger without enlarging counts."""
import json,re,hashlib
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
LIBS={'NumPy','SciPy','Matplotlib','Gymnasium','JAX','Diffrax','pandas'}
ENGINES={'ICU-Sepsis','DBS-Gym'}
def compute():
 p=ROOT/'evidence/scientific_tools.json';j=json.loads(p.read_text());rows=[];names=set();urls=set()
 for x in j['used']:
  name=x['name'].casefold();url=x['url'].rstrip('/').casefold();assert name not in names and url not in urls;names.add(name);urls.add(url)
  files=re.findall(r'(?:src|results|data)/[A-Za-z0-9_./-]+\.(?:py|json|csv)',x['evidence']);evidence={}
  for f in files:
   path=ROOT/f;assert path.exists(),f;evidence[f]=hashlib.sha256(path.read_bytes()).hexdigest()
  assert evidence,x['name']
  scope=x.get('scope','Both projects').lower();projects=['sepsis'] if scope=='sepsis project only' else ['parkinson'] if scope=='parkinson project only' else ['sepsis','parkinson']
  rows.append({'name':x['name'],'url':x['url'],'category':'library' if x['name'] in LIBS else 'simulation_engine' if x['name'] in ENGINES else 'data_archive_or_product','projects':projects,'local_evidence_sha256':evidence})
 counts={project:sum(project in r['projects'] for r in rows) for project in ['sepsis','parkinson']};assert counts==j['project_counts'];assert len(rows)==j['verified_distinct_count']
 return {'plan_sha256':hashlib.sha256((ROOT/'research/scientific_component_dedup_plan.json').read_bytes()).hexdigest(),'historical_ledger_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'distinct_mixed_components':len(rows),'category_counts':dict(Counter(r['category'] for r in rows)),'project_counts':counts,'components':rows,'limits':json.loads((ROOT/'research/scientific_component_dedup_plan.json').read_text())['limits']}
if __name__=='__main__':
 j=compute();(ROOT/'evidence/scientific_component_dedup.json').write_text(json.dumps(j,indent=2)+'\n');print({k:v for k,v in j.items() if k!='components'})

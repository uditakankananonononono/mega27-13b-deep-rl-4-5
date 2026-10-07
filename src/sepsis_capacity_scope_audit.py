"""Actual parameter accounting only, never effective-capacity or performance proof."""
import json,hashlib
from pathlib import Path
from net import QNet
ROOT=Path(__file__).resolve().parents[1]
def compute():
 rows=[]
 for name,d in [('onehot',716),('centroid',47)]:
  n=QNet(d,25,width=64);count=sum(p.size for p in n.w+n.b);formula=(d+1)*64+65*64+65*25
  if count!=formula:raise ValueError('parameter formula')
  rows.append({'representation':name,'input_dimensions':d,'parameters':int(count),'first_layer_parameters':int(n.w[0].size+n.b[0].size),'optimizer_moment_scalar_slots':int(sum(p.size for p in n.m+n.v))})
 return {'plan_sha256':hashlib.sha256((ROOT/'research/sepsis_capacity_scope_plan.json').read_bytes()).hexdigest(),'source_net_sha256':hashlib.sha256((ROOT/'src/net.py').read_bytes()).hexdigest(),'architectures':rows,'parameter_ratio_onehot_to_centroid':rows[0]['parameters']/rows[1]['parameters'],'limits':json.loads((ROOT/'research/sepsis_capacity_scope_plan.json').read_text())['limits']}
if __name__=='__main__':
 j=compute();(ROOT/'results/sepsis_capacity_scope_audit.json').write_text(json.dumps(j,indent=2)+'\n');print(j)

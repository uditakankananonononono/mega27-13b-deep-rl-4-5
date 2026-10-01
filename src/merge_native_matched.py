"""Keep all matched control and phase outcomes, without selecting a policy."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def compute():
 raw=(ROOT/'src/native_matched_plan.json').read_bytes();p=json.loads(raw);rows=[]
 for seed in p['seeds']:
  off=json.loads((ROOT/f'results/native_matched_{seed}_off.json').read_text());on=json.loads((ROOT/f'results/native_matched_{seed}_on.json').read_text())
  assert all(r['plan_sha256']==hashlib.sha256(raw).hexdigest() and r['steps']==60 for r in [off,on])
  phases=[json.loads((ROOT/f'results/native_phase_{seed}_{phase}.json').read_text()) for phase in range(4)]
  rows.append({'seed':seed,'off_beta':off['mean_beta_component'],'on_beta':on['mean_beta_component'],'off_reward':off['mean_reward'],'on_reward':on['mean_reward'],'phase_beta':[r['mean_beta_component'] for r in phases],'all_phases_between_controls':all(on['mean_beta_component']<r['mean_beta_component']<off['mean_beta_component'] for r in phases),'half_duty_fraction_of_off_on_suppression':[(off['mean_beta_component']-r['mean_beta_component'])/(off['mean_beta_component']-on['mean_beta_component']) for r in phases]})
 return {'plan_sha256':hashlib.sha256(raw).hexdigest(),'rows':rows,'limits':p['limits']}
if __name__=='__main__':
 j=compute();(ROOT/'results/native_matched_audit.json').write_text(json.dumps(j,indent=2)+'\n');print(json.dumps(j,indent=2))

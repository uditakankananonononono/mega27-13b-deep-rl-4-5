"""Bounded actual native phase-offset pilot; preserve every phase, no controller tuning."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
import parkinson_native_reinforce_adam as P
ROOT=Path(__file__).resolve().parents[1]
def run(upstream,seed,phase):
 raw=(ROOT/'src/native_phase_plan.json').read_bytes();p=json.loads(raw)
 assert seed in p['seeds'] and phase in p['phases']
 assert subprocess.check_output(['git','-C',upstream,'rev-parse','HEAD']).decode().strip()==p['source_commit']
 target=ROOT/f'results/native_phase_{seed}_{phase}.json'
 if target.exists():print('ALREADY COMPLETE',target);return
 start=time.monotonic();e,o=P.make_env(upstream,seed);actions=[];rewards=[];wave=[]
 for step in range(p['horizon_steps']):
  a=int((step+phase)%4<2);o,r,done,truncated,_=e.step(np.array([a],np.float32));actions.append(a);rewards.append(float(r));wave.extend(np.asarray(e.theta_mean).tolist())
  if done or truncated:break
 out={'plan_sha256':hashlib.sha256(raw).hexdigest(),'seed':seed,'phase':phase,'steps':len(actions),'mean_reward':float(np.mean(rewards)),'mean_beta_component':float(np.mean(-np.array(rewards)-.05*np.array(actions))),'on_fraction':float(np.mean(actions)),'physical_abs_voltage_total':float(5*np.sum(actions)),'switch_count':int(np.sum(np.diff(actions)!=0)),'lfp_sha256':hashlib.sha256(np.array(wave).tobytes()).hexdigest(),'actions':actions,'rewards':rewards,'elapsed_seconds':time.monotonic()-start,'source_commit':p['source_commit'],'limits':p['limits']}
 target.write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k not in ['actions','rewards']},flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--upstream',required=True);a.add_argument('--seed',type=int,required=True);a.add_argument('--phase',type=int,required=True);x=a.parse_args();run(x.upstream,x.seed,x.phase)

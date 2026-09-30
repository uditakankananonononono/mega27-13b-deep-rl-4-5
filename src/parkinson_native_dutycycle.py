"""Native open-loop duty-cycle controls. Not learned or feedback controllers.
One policy/seed per bounded invocation; exact matched native pilot initialization.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import parkinson_native_reinforce_adam as P
ROOT=Path(__file__).resolve().parents[1];PIN='aa0b10b9502e4f62dea755ce6468da024235c319'
def run(upstream,replicate,duty):
 import subprocess
 assert subprocess.check_output(['git','-C',upstream,'rev-parse','HEAD']).decode().strip()==PIN
 output=ROOT/'results/native_dutycycle_parts';output.mkdir(exist_ok=True);target=output/f'replicate_{replicate}_duty_{duty}.json'
 if target.exists():print('already computed',target);return
 seed=1220+replicate;e,o=P.make_env(upstream,seed);actions=[];rewards=[];wave=[]
 for step in range(150):
  a=int(step%4<duty);o,r,d,t,_=e.step(np.array([a],np.float32));actions.append(a);rewards.append(float(r));wave.extend(np.asarray(e.theta_mean).tolist())
  if d or t:break
 result={'source':'https://github.com/NevVerVer/DBS-Gym','source_commit':PIN,'replicate':replicate,'eval_seed':seed,'on_steps_per_four':duty,'action_rule':'fixed periodic four-step cycle, ON first duty steps, no observation dependence','steps':len(actions),'on_fraction':float(np.mean(actions)),'switches':int(np.sum(np.diff(actions)!=0)),'mean_reward':float(np.mean(rewards)),'mean_beta_component':float(np.mean(-np.array(rewards)-.05*np.array(actions))),'physical_abs_action_total':float(5*np.sum(actions)),'rewards':rewards,'actions':actions,'lfp_sha256':hashlib.sha256(np.array(wave).tobytes()).hexdigest(),'limits':['Open-loop periodic pilot, not adaptive or learned','Short150step horizon and selected duty grid','One stimulation phase offset, no phase robustness','No clinical or published-SAC normalization equivalence']}
 target.write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ['actions','rewards']},flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--upstream',required=True);a.add_argument('--replicate',type=int,required=True);a.add_argument('--duty',type=int,required=True,choices=[1,2,3]);x=a.parse_args();run(x.upstream,x.replicate,x.duty)

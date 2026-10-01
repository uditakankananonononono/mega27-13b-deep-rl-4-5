"""Broader native seed/phase controls; independent reset at every seed/policy."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
import parkinson_native_reinforce_adam as P
ROOT=Path(__file__).resolve().parents[1]
def run(upstream,seed,policy,variant):
 raw=(ROOT/'src/native_full_solver_plan.json').read_bytes();p=json.loads(raw)
 assert seed==p['seed'] and policy in p['policies'] and variant in p['variants']
 assert subprocess.check_output(['git','-C',upstream,'rev-parse','HEAD']).decode().strip()==p['source_commit']
 target=ROOT/f'results/native_full_solver_{seed}_{policy}_{variant}.json'
 if target.exists():return json.loads(target.read_text())
 start=time.monotonic();import sys;sys.path.insert(0,upstream)
 from environment.env import KuramotoJAX
 from diffrax import Dopri5,Tsit5,PIDController
 original=KuramotoJAX.__init__
 def configured(self,*args,**kwargs):
  original(self,*args,**kwargs);self.solver=Tsit5() if variant.startswith('Tsit5') else Dopri5();tol=1e-6 if variant.endswith('1e-6') else 1e-5;self.stepsize_controller=PIDController(rtol=tol,atol=tol)
 KuramotoJAX.__init__=configured
 try:env,obs=P.make_env(upstream,seed)
 finally:KuramotoJAX.__init__=original
 actions=[];rewards=[];wave=[]
 for step in range(p['horizon_steps']):
  a=int(policy=='on') if policy in ['off','on'] else int((step+int(policy[-1]))%4<2);obs,r,done,truncated,_=env.step(np.array([a],np.float32));actions.append(a);rewards.append(float(r));wave.extend(np.asarray(env.theta_mean).tolist())
  if done or truncated:break
 out={'plan_sha256':hashlib.sha256(raw).hexdigest(),'seed':seed,'policy':policy,'variant':variant,'steps':len(actions),'mean_reward':float(np.mean(rewards)),'mean_beta_component':float(np.mean(-np.array(rewards)-.05*np.array(actions))),'on_fraction':float(np.mean(actions)),'voltage_step_magnitude':float(5*np.sum(actions)),'lfp_sha256':hashlib.sha256(np.array(wave).tobytes()).hexdigest(),'actions':actions,'rewards':rewards,'elapsed_seconds':time.monotonic()-start,'source_commit':p['source_commit'],'limits':p['limits']}
 target.write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k not in ['actions','rewards']},flush=True);return out
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--upstream',required=True);a.add_argument('--seed',type=int,required=True);a.add_argument('--policy',choices=['off','on','phase0','phase1','phase2','phase3'],required=True);a.add_argument('--variant',required=True);x=a.parse_args();run(x.upstream,x.seed,x.policy,x.variant)

"""Exploratory nonlinear beta-burst surrogate; NOT DBS-Gym or a clinical simulator.
Train a neural fitted Q controller on independent simulated transitions.
"""
import argparse,json
import numpy as np
from net import QNet
ACTIONS=np.array([0.0,.5,1.0]);HORIZON=70

def transition(state,action,noise):
    beta,drive,drift=state
    amplitude=ACTIONS[action]
    next_beta=np.clip(.85*beta+.12*drive+.04*drift-.19*amplitude*(1-.25*drift)+noise,0,1.5)
    next_drive=np.clip(.94*drive+.045+.035*np.sin(8*next_beta),.05,1.0)
    next_drift=np.clip(.99*drift+.01+.001*np.sin(3*next_beta),0,1)
    reward=-next_beta**2-.12*amplitude**2-.035*(amplitude>0)
    return np.array([next_beta,next_drive,next_drift]),float(reward)

def trajectory(policy,seed,horizon=HORIZON):
    g=np.random.default_rng(seed);s=np.array([g.uniform(.35,.9),g.uniform(.15,.65),g.uniform(.1,.7)]);total=0;bursts=0;energy=0
    for t in range(horizon):
        a=int(policy(s));s,r=transition(s,a,g.normal(0,.05));total+=r;bursts+=int(s[0]>.5);energy+=ACTIONS[a]**2
    return total,bursts,energy

def train(seed=101,rounds=12,samples=3000,epochs=20):
    rng=np.random.default_rng(seed)
    states=np.column_stack([rng.uniform(0,1.4,samples),rng.uniform(.05,1,samples),rng.uniform(.0,1,samples)])
    actions=rng.integers(0,3,size=samples)
    noise=rng.normal(0,.05,size=samples)
    ns=np.empty_like(states);rewards=np.empty(samples)
    for i in range(samples):ns[i],rewards[i]=transition(states[i],actions[i],noise[i])
    net=QNet(3,3,width=32,seed=seed);loss=[]
    for k in range(rounds):
        pred=net.predict(states);targets=pred.copy()
        boot=0 if k==0 else .95*np.max(net.predict(ns),axis=1)
        targets[np.arange(samples),actions]=rewards+boot
        loss.append(net.fit(states,targets,epochs=epochs,seed=seed+k,lr=.001)[-1])
    return net,loss

def run(out,seed=101,rounds=12,samples=3000,epochs=20,n=120):
    net,loss=train(seed,rounds,samples,epochs)
    controllers={'neural_fqi':lambda s:np.argmax(net.predict(s[None,:])[0]),'always_off':lambda s:0,'fixed_high':lambda s:2,'threshold_beta_0_5':lambda s:2 if s[0]>.5 else 0}
    seeds=range(60100,60100+n);outcomes={k:np.asarray([trajectory(p,s) for s in seeds]) for k,p in controllers.items()}
    results={'model':'custom synthetic beta-burst control surrogate inspired by DBS-Gym, NOT execution of DBS-Gym',
      'reference':'https://github.com/NevVerVer/DBS-Gym','published_benchmark_comparison':'NOT COMPARABLE: different simulator, observation/action/reward/horizon; no DBS-Gym score claimed',
      'heldout_seed_range':[60100,60100+n-1],'n_trajectories':n,'training_samples':samples,'training_rounds':rounds,'epochs_each_round':epochs,'loss_first_last':[loss[0],loss[-1]],
      'controllers':{k:{'mean_return':float(v[:,0].mean()),'sd_return':float(v[:,0].std(ddof=1)), 'mean_beta_burst_steps':float(v[:,1].mean()),'mean_energy':float(v[:,2].mean())} for k,v in outcomes.items()},
      'paired_diff_neural_minus_threshold':float(np.mean(outcomes['neural_fqi'][:,0]-outcomes['threshold_beta_0_5'][:,0])),
      'limitations':['Synthetic dynamics were chosen by us; not calibrated to patient stimulation responses','GEO blood expression is descriptive and does not establish neural state or treatment effect','No clinical or DBS-Gym published-benchmark result']}
    rng=np.random.default_rng(755);d=outcomes['neural_fqi'][:,0]-outcomes['threshold_beta_0_5'][:,0];bootstrap=np.mean(d[rng.integers(0,n,size=(3000,n))],axis=1)
    results['paired_bootstrap_95']=list(map(float,np.quantile(bootstrap,[.025,.975])))
    with open(out,'w') as f:json.dump(results,f,indent=2)
    print(json.dumps(results,indent=2));return results
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',default='results/parkinson.json');p.add_argument('--rounds',type=int,default=12);p.add_argument('--samples',type=int,default=3000);p.add_argument('--epochs',type=int,default=20);a=p.parse_args();run(a.out,rounds=a.rounds,samples=a.samples,epochs=a.epochs)

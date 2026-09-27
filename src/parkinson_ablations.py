"""Ablation study (verdict #17): energy penalty, observation noise, drift
contribution on the custom surrogate. Trains fitted-Q on each ablated variant,
evaluates on 60 held-out seeds, reports mean return + burst/energy metrics.
"""
import json
import numpy as np
import parkinson as P
from net import QNet
from parkinson_model_uncertainty import trajectory, make_pid

def make_transition(beta_decay=0.85, stim_gain=0.19, drive_freq=8.0,
                    noise_sd=0.05, energy_w=0.12, switch_w=0.035,
                    drift_couple=True):
    def transition(state, action, noise):
        beta, drive, drift = state
        amplitude = P.ACTIONS[action]
        d = drift if drift_couple else 0.0
        next_beta = np.clip(beta_decay*beta + .12*drive + .04*d - stim_gain*amplitude*(1-.25*d) + noise, 0, 1.5)
        next_drive = np.clip(.94*drive + .045 + .035*np.sin(drive_freq*next_beta), .05, 1.0)
        next_drift = np.clip(.99*drift + .01 + .001*np.sin(3*next_beta), 0, 1)
        reward = -next_beta**2 - energy_w*amplitude**2 - switch_w*(amplitude > 0)
        return np.array([next_beta, next_drive, next_drift]), float(reward)
    return transition, noise_sd

def train_on(trans, noise_sd, seed=101, rounds=12, samples=3000, epochs=20):
    rng = np.random.default_rng(seed)
    states = np.column_stack([rng.uniform(0,1.4,samples), rng.uniform(.05,1,samples), rng.uniform(.0,1,samples)])
    actions = rng.integers(0,3,size=samples)
    noise = rng.normal(0, noise_sd, size=samples)
    ns = np.empty_like(states); rewards = np.empty(samples)
    for i in range(samples): ns[i], rewards[i] = trans(states[i], actions[i], noise[i])
    net = QNet(3,3,width=32,seed=seed)
    for k in range(rounds):
        pred = net.predict(states); targets = pred.copy()
        boot = 0 if k==0 else .95*np.max(net.predict(ns),axis=1)
        targets[np.arange(samples),actions] = rewards + boot
        net.fit(states,targets,epochs=epochs,seed=seed+k,lr=.001)
    return net

ABL = {
    "full_model": {},
    "no_energy_penalty": {"energy_w": 0.0, "switch_w": 0.0},
    "double_energy_penalty": {"energy_w": 0.24, "switch_w": 0.07},
    "no_obs_noise": {"noise_sd": 0.001},
    "double_obs_noise": {"noise_sd": 0.10},
    "drift_decoupled": {"drift_couple": False},
}

def main():
    n = 60; seeds = range(80100, 80100+n)
    rows = {}
    for name, kw in ABL.items():
        trans, sd = make_transition(**kw)
        net = train_on(trans, sd)
        pol = lambda s: int(np.argmax(net.predict(s[None,:])[0]))
        rets, acts = [], []
        for s_ in seeds:
            g = np.random.default_rng(s_)
            st = np.array([g.uniform(.35,.9), g.uniform(.15,.65), g.uniform(.1,.7)])
            total, on = 0, 0
            for t in range(70):
                a = int(pol(st)); st, r = trans(st, a, g.normal(0, sd)); total += r; on += int(a > 0)
            rets.append(total); acts.append(on / 70)
        pid_r = [trajectory(make_pid(2.0,0.1,0.1), s_, trans, sd) for s_ in seeds]
        rows[name] = {"ablation": kw, "neural_mean_return": float(np.mean(rets)),
                      "neural_sd": float(np.std(rets, ddof=1)),
                      "stim_on_fraction": float(np.mean(acts)),
                      "pid_mean_return": float(np.mean(pid_r)),
                      "neural_minus_pid": float(np.mean(rets) - np.mean(pid_r))}
        print(name, rows[name]["neural_mean_return"], rows[name]["stim_on_fraction"], rows[name]["neural_minus_pid"], flush=True)
    out = {"design": "fitted-Q retrained per ablated variant, 60 held-out seeds 80100-80159, horizon 70",
           "n_trajectories": n, "results": rows,
           "limitations": ["Ablations of a synthetic surrogate show which reward/dynamics terms drive OUR controller, not clinical behavior"]}
    json.dump(out, open("results/parkinson_ablations.json","w"), indent=1)
    print("saved results/parkinson_ablations.json")

if __name__ == "__main__":
    main()

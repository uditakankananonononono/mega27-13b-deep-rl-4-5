"""Model-uncertainty ensemble (verdict #2): does the controller generalize across
plausible simulators? Train fitted-Q on the nominal surrogate; evaluate on
parameter-perturbed replicas WITHOUT retraining, vs a controller retrained on
each replica (transfer gap). Also evaluates PID transfer for reference.
"""
import json
import numpy as np
import parkinson as P

PERTS = {
    "nominal": {},
    "beta_decay_0.80": {"beta_decay": 0.80},
    "beta_decay_0.90": {"beta_decay": 0.90},
    "stim_gain_0.15": {"stim_gain": 0.15},
    "stim_gain_0.23": {"stim_gain": 0.23},
    "drive_sin_12": {"drive_freq": 12.0},
    "noise_0.10": {"noise_sd": 0.10},
}

def make_transition(beta_decay=0.85, stim_gain=0.19, drive_freq=8.0, noise_sd=0.05):
    def transition(state, action, noise):
        beta, drive, drift = state
        amplitude = P.ACTIONS[action]
        next_beta = np.clip(beta_decay*beta + .12*drive + .04*drift - stim_gain*amplitude*(1-.25*drift) + noise, 0, 1.5)
        next_drive = np.clip(.94*drive + .045 + .035*np.sin(drive_freq*next_beta), .05, 1.0)
        next_drift = np.clip(.99*drift + .01 + .001*np.sin(3*next_beta), 0, 1)
        reward = -next_beta**2 - .12*amplitude**2 - .035*(amplitude > 0)
        return np.array([next_beta, next_drive, next_drift]), float(reward)
    return transition, noise_sd

def trajectory(policy, seed, trans, noise_sd, horizon=70):
    g = np.random.default_rng(seed)
    s = np.array([g.uniform(.35,.9), g.uniform(.15,.65), g.uniform(.1,.7)])
    total = 0
    for t in range(horizon):
        a = int(policy(s)); s, r = trans(s, a, g.normal(0, noise_sd)); total += r
    return total

def train_on(trans, noise_sd, seed=101, rounds=12, samples=3000, epochs=20):
    rng = np.random.default_rng(seed)
    states = np.column_stack([rng.uniform(0,1.4,samples), rng.uniform(.05,1,samples), rng.uniform(.0,1,samples)])
    actions = rng.integers(0,3,size=samples)
    noise = rng.normal(0, noise_sd, size=samples)
    ns = np.empty_like(states); rewards = np.empty(samples)
    for i in range(samples): ns[i], rewards[i] = trans(states[i], actions[i], noise[i])
    from net import QNet
    net = QNet(3,3,width=32,seed=seed)
    for k in range(rounds):
        pred = net.predict(states); targets = pred.copy()
        boot = 0 if k==0 else .95*np.max(net.predict(ns),axis=1)
        targets[np.arange(samples),actions] = rewards + boot
        net.fit(states,targets,epochs=epochs,seed=seed+k,lr=.001)
    return net

def make_pid(kp, ki, kd):
    st = {"int":0.0,"prev":0.0}
    def pid(s):
        e = s[0]
        st["int"] = min(5.0, max(-5.0, st["int"]+e))
        u = kp*e + ki*st["int"] + kd*(e-st["prev"]); st["prev"]=e
        return int(np.argmin(np.abs(P.ACTIONS - np.clip(u,0,1))))
    return pid

def main():
    n = 60; seeds = range(70100, 70100+n)
    nominal_trans, nominal_sd = make_transition()
    net0 = train_on(nominal_trans, nominal_sd)
    rows = {}
    for name, kw in PERTS.items():
        trans, sd = make_transition(**kw)
        # transferred nominal controller
        pol0 = lambda s: int(np.argmax(net0.predict(s[None,:])[0]))
        r_transfer = [trajectory(pol0, s, trans, sd) for s in seeds]
        # retrained controller on this replica
        net_r = train_on(trans, sd)
        polr = lambda s: int(np.argmax(net_r.predict(s[None,:])[0]))
        r_retrain = [trajectory(polr, s, trans, sd) for s in seeds]
        # PID reference (fixed nominal gains)
        r_pid = [trajectory(make_pid(2.0,0.1,0.1), s, trans, sd) for s in seeds]
        rows[name] = {"perturbation": kw,
            "transfer_mean_return": float(np.mean(r_transfer)), "transfer_sd": float(np.std(r_transfer, ddof=1)),
            "retrained_mean_return": float(np.mean(r_retrain)), "retrained_sd": float(np.std(r_retrain, ddof=1)),
            "pid_mean_return": float(np.mean(r_pid)), "pid_sd": float(np.std(r_pid, ddof=1)),
            "transfer_gap_retrained_minus_transfer": float(np.mean(r_retrain)-np.mean(r_transfer))}
        print(name, rows[name]["transfer_mean_return"], rows[name]["retrained_mean_return"], rows[name]["pid_mean_return"], flush=True)
    out = {"design": "fitted-Q trained on nominal surrogate, evaluated unmodified on 6 perturbed replicas vs retrained-on-replica and fixed-gain PID; 60 held-out seeds 70100-70159",
           "n_trajectories": n, "results": rows,
           "limitations": ["All replicas are our own synthetic surrogate family; this measures sensitivity to OUR parameter choices, not to real neural dynamics"]}
    json.dump(out, open("results/parkinson_model_uncertainty.json","w"), indent=1)
    print("saved results/parkinson_model_uncertainty.json")

if __name__ == "__main__":
    main()

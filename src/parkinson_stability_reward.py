"""Multi-objective reward with stimulation-stability term (verdict #10).
Adds a switching-rate penalty to the surrogate reward; retrain fitted-Q;
compare return, burst steps, energy, and switch count vs the nominal reward.
"""
import json
import numpy as np
import parkinson as P
from net import QNet

SWITCH_W = 0.05  # penalty per action change

def transition_stab(state, action, noise, prev_action):
    ns, r = P.transition(state, action, noise)
    return ns, r - SWITCH_W * float(action != prev_action)

def trajectory(policy, seed, stab, horizon=70):
    g = np.random.default_rng(seed)
    s = np.array([g.uniform(.35,.9), g.uniform(.15,.65), g.uniform(.1,.7)])
    total, bursts, energy, switches = 0, 0, 0.0, 0
    prev = 0
    for t in range(horizon):
        a = int(policy(s))
        if stab:
            s2, r = transition_stab(s, a, g.normal(0,.05), prev)
        else:
            s2, r = P.transition(s, a, g.normal(0,.05))
        total += r; bursts += int(s2[0] > .5); energy += P.ACTIONS[a]**2
        switches += int(a != prev); prev = a; s = s2
    return total, bursts, energy, switches

def train(stab, seed=101, rounds=12, samples=3000, epochs=20):
    rng = np.random.default_rng(seed)
    states = np.column_stack([rng.uniform(0,1.4,samples), rng.uniform(.05,1,samples), rng.uniform(0,1,samples)])
    actions = rng.integers(0,3,size=samples)
    prev_actions = rng.integers(0,3,size=samples)
    noise = rng.normal(0,.05,size=samples)
    ns = np.empty_like(states); rewards = np.empty(samples)
    for i in range(samples):
        if stab:
            ns[i], rewards[i] = transition_stab(states[i], actions[i], noise[i], prev_actions[i])
        else:
            ns[i], rewards[i] = P.transition(states[i], actions[i], noise[i])
    net = QNet(3,3,width=32,seed=seed)
    for k in range(rounds):
        pred = net.predict(states); targets = pred.copy()
        boot = 0 if k==0 else .95*np.max(net.predict(ns),axis=1)
        targets[np.arange(samples),actions] = rewards + boot
        net.fit(states,targets,epochs=epochs,seed=seed+k,lr=.001)
    return net

def main():
    n = 120; seeds = range(90100, 90100+n)
    net_plain = train(False)
    net_stab = train(True)
    rows = {}
    for name, net, stab in (("nominal_reward", net_plain, False),
                            ("stability_reward", net_stab, True)):
        out = np.asarray([trajectory(lambda s: int(np.argmax(net.predict(s[None,:])[0])), s_, stab) for s_ in seeds])
        rows[name] = {"mean_return": float(out[:,0].mean()), "sd_return": float(out[:,0].std(ddof=1)),
                      "mean_bursts": float(out[:,1].mean()), "mean_energy": float(out[:,2].mean()),
                      "mean_switches": float(out[:,3].mean())}
        print(name, rows[name], flush=True)
    res = {"design": "fitted-Q trained/evaluated under nominal reward vs reward + 0.05/action-change; 120 held-out seeds 90100-90219",
           "switch_penalty": SWITCH_W, "results": rows,
           "reading": "stability term should cut switching without losing symptom control - check bursts row"}
    res["delta_stab_minus_nominal"] = {k: round(rows["stability_reward"][k] - rows["nominal_reward"][k], 4)
                                       for k in ("mean_return","mean_bursts","mean_energy","mean_switches")}
    json.dump(res, open("results/parkinson_stability_reward.json","w"), indent=1)
    print("saved results/parkinson_stability_reward.json")

if __name__ == "__main__":
    main()

"""PID + MPC baselines and policy-trigger interpretability on the custom beta-burst surrogate (verdict #8/#15/#16).
Identical held-out seeds and metrics as src/parkinson.py. NOT DBS-Gym; no clinical claim.
"""
import argparse, json
import numpy as np
from parkinson import transition, trajectory, train, ACTIONS, HORIZON

def make_pid(kp, ki, kd):
    state = {"int": 0.0, "prev": 0.0}
    def pid(s):
        e = s[0]
        state["int"] = min(5.0, max(-5.0, state["int"] + e))
        u = kp * e + ki * state["int"] + kd * (e - state["prev"])
        state["prev"] = e
        return int(np.argmin(np.abs(ACTIONS - np.clip(u, 0, 1))))
    return pid

def mpc_factory(horizon=3, gamma=0.95):
    def mpc(s):
        best_a, best_v = 0, -1e18
        for a0 in range(3):
            def rec(state, t):
                if t == horizon: return 0.0
                vals = []
                for a in range(3):
                    ns, r = transition(state, a, 0.0)
                    vals.append(r + gamma * rec(ns, t + 1))
                return max(vals)
            ns, r = transition(np.asarray(s, dtype=float), a0, 0.0)
            v = r + gamma * rec(ns, 1)
            if v > best_v: best_v, best_a = v, a0
        return best_a
    return mpc

def main(out):
    net, _ = train(101, 12, 3000, 20)
    # PID gain selection on tuning seeds only (disjoint from held-out 60100..)
    tune = range(50100, 50140)
    best, best_score = None, -1e18
    for kp in (0.5, 1.0, 2.0):
        for ki in (0.0, 0.1):
            for kd in (0.0, 0.1):
                sc = np.mean([trajectory(make_pid(kp, ki, kd), s)[0] for s in tune])
                if sc > best_score: best_score, best, = sc, (kp, ki, kd)
    controllers = {
        'neural_fqi': lambda s: int(np.argmax(net.predict(s[None, :])[0])),
        'always_off': lambda s: 0,
        'fixed_high': lambda s: 2,
        'threshold_beta_0_5': lambda s: 2 if s[0] > .5 else 0,
        'pid': make_pid(*best),
        'mpc_h3': mpc_factory(3),
    }
    n = 120; seeds = range(60100, 60100 + n)
    outcomes = {k: np.asarray([trajectory(p, s) for s in seeds]) for k, p in controllers.items()}
    rng = np.random.default_rng(755)
    boot = {}
    for k in ('always_off','fixed_high','threshold_beta_0_5','pid','mpc_h3'):
        d = outcomes['neural_fqi'][:, 0] - outcomes[k][:, 0]
        b = np.mean(d[rng.integers(0, n, size=(3000, n))], axis=1)
        boot[k] = {'paired_diff_mean': float(d.mean()), 'bootstrap_95': list(map(float, np.quantile(b, [.025, .975])))}
    # policy-trigger interpretability: stimulation probability by state bin
    g = np.random.default_rng(9)
    grid = np.column_stack([g.uniform(0, 1.4, 20000), g.uniform(.05, 1, 20000), g.uniform(0, 1, 20000)])
    acts = np.argmax(net.predict(grid), axis=1)
    bins = np.linspace(0, 1.4, 8)
    trig = {f"beta {bins[i]:.2f}-{bins[i+1]:.2f}": float(np.mean(acts[(grid[:,0]>=bins[i])&(grid[:,0]<bins[i+1])] > 0))
            for i in range(7)}
    # state-dimension ablation: shuffle each dim, measure action-change rate
    abl = {}
    base = acts.copy()
    for j, name in enumerate(('beta','drive','drift')):
        gs = grid.copy(); gs[:, j] = g.permutation(gs[:, j])
        abl[name] = float(np.mean(np.argmax(net.predict(gs), axis=1) != base))
    res = {'model': 'custom synthetic beta-burst control surrogate, NOT DBS-Gym; identical protocol to results/parkinson.json',
           'pid_gains_tuned_on_disjoint_seeds_50100_50139': best,
           'mpc': 'model-based, horizon 3, gamma 0.95, exhaustive 3^h search, uses known transition mean',
           'heldout_seed_range': [60100, 60100 + n - 1], 'n_trajectories': n,
           'controllers': {k: {'mean_return': float(v[:,0].mean()), 'sd_return': float(v[:,0].std(ddof=1)),
                               'mean_beta_burst_steps': float(v[:,1].mean()), 'mean_energy': float(v[:,2].mean())}
                           for k, v in outcomes.items()},
           'paired_neural_minus_baseline': boot,
           'policy_trigger_stim_fraction_by_beta_bin': trig,
           'policy_ablation_action_change_rate': abl,
           'limitations': ['Synthetic dynamics chosen by us; MPC exploits known model (upper bound); not calibrated to patients']}
    with open(out, 'w') as f: json.dump(res, f, indent=2)
    print(json.dumps(res['controllers'], indent=1))
    print(json.dumps(res['paired_neural_minus_baseline'], indent=1))
    print(json.dumps(trig, indent=1)); print(json.dumps(abl, indent=1))

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', default='results/parkinson_pid_mpc.json'); a = p.parse_args()
    main(a.out)

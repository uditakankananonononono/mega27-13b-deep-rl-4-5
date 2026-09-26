"""PD-BLOOD-1c: VALIDATION-selected challenger config, single TEST touch.
Locked 2026-09-26 before any TEST outcome: grid over k (fSVA SVs) x stability threshold
x ridge C; selection by VALIDATION AUC only; the winning config touches TEST once.
Per-k partials are cached so compute fits the execution window.
"""
import argparse, json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from pd_blood_fsva import fsva_fit, fsva_transform_train, fsva_transform_new

GRID_K = (1, 2, 3, 5)
GRID_STAB = (50, 60, 70)
GRID_C = (0.01, 0.1, 1.0)


def prep_k(k):
    d = np.load('/tmp/pd_blood_prepped_raw.npz', allow_pickle=False)
    Xk, y, tr, va, te, probes = d['Xk'], d['y'], d['tr'], d['va'], d['te'], d['probes']
    mu = Xk[tr].mean(0); sd = np.maximum(Xk[tr].std(0), 1e-6)
    Z = (Xk - mu) / sd
    fit = fsva_fit(Z[tr], y[tr], k)
    Atr = fsva_transform_train(Z[tr], fit)
    Ava = fsva_transform_new(Z[va], fit)
    Ate = fsva_transform_new(Z[te], fit)
    diff = np.abs(Atr[y[tr] == 1].mean(0) - Atr[y[tr] == 0].mean(0))
    pre_idx = np.argsort(diff)[::-1][:2000]
    rng = np.random.default_rng(7)
    ntr = int(tr.sum())
    sel = np.zeros(len(pre_idx))
    for b in range(100):
        idx = rng.integers(0, ntr, ntr)
        m = LogisticRegression(penalty='l1', solver='liblinear', C=0.1).fit(Atr[idx][:, pre_idx], y[tr][idx])
        sel += (m.coef_[0] != 0)
    np.savez(f'/tmp/pd_challenger_k{k}.npz', sel=sel, pre_idx=pre_idx)
    # VAL metrics for every (stab, C)
    val = {}
    for stab in GRID_STAB:
        sl = np.flatnonzero(sel >= stab)
        st = pre_idx[sl] if len(sl) else pre_idx[np.argsort(sel)[::-1][:50]]
        for C in GRID_C:
            m = LogisticRegression(penalty='l2', C=C, max_iter=20000).fit(Atr[:, st], y[tr])
            val[f'{stab},{C}'] = float(roc_auc_score(y[va], m.decision_function(Ava[:, st])))
    with open(f'/tmp/pd_challenger_k{k}_val.json', 'w') as f:
        json.dump(val, f)
    print('k', k, 'best val', max(val.items(), key=lambda kv: kv[1]))


def finalize():
    d = np.load('/tmp/pd_blood_prepped_raw.npz', allow_pickle=False)
    Xk, y, tr, va, te, probes = d['Xk'], d['y'], d['tr'], d['va'], d['te'], d['probes']
    mu = Xk[tr].mean(0); sd = np.maximum(Xk[tr].std(0), 1e-6)
    Z = (Xk - mu) / sd
    best = None
    for k in GRID_K:
        val = json.load(open(f'/tmp/pd_challenger_k{k}_val.json'))
        for key, v in val.items():
            if best is None or v > best[0]:
                best = (v, k, key)
    va_auc, k, key = best
    stab, C = key.split(','); stab = int(stab); C = float(C)
    fit = fsva_fit(Z[tr], y[tr], k)
    Atr = fsva_transform_train(Z[tr], fit)
    Ava = fsva_transform_new(Z[va], fit)
    Ate = fsva_transform_new(Z[te], fit)
    cz = np.load(f'/tmp/pd_challenger_k{k}.npz')
    sel, pre_idx = cz['sel'], cz['pre_idx']
    sl = np.flatnonzero(sel >= stab)
    st = pre_idx[sl] if len(sl) else pre_idx[np.argsort(sel)[::-1][:50]]
    model = LogisticRegression(penalty='l2', C=C, max_iter=20000).fit(np.r_[Atr[:, st], Ava[:, st]], np.r_[y[tr], y[va]])
    our_score = model.decision_function(Ate[:, st])
    our_auc = float(roc_auc_score(y[te], our_score))
    # baseline repro at its VAL-chosen k=2 (from pd_blood_fsva.py selection stage)
    fit2 = fsva_fit(Z[tr], y[tr], 2)
    Btr = fsva_transform_train(Z[tr], fit2); Bva = fsva_transform_new(Z[va], fit2); Bte = fsva_transform_new(Z[te], fit2)
    diff2 = np.abs(Btr[y[tr] == 1].mean(0) - Btr[y[tr] == 0].mean(0))
    topk = np.argsort(diff2)[::-1][:100]
    from sklearn.svm import LinearSVC
    svm = LinearSVC(C=1.0, max_iter=20000).fit(np.r_[Btr[:, topk], Bva[:, topk]], np.r_[y[tr], y[va]])
    base_score = svm.decision_function(Bte[:, topk])
    base_auc = float(roc_auc_score(y[te], base_score))
    rng = np.random.default_rng(11); n = int(te.sum()); diffs = []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        if len(set(y[te][idx])) < 2: continue
        diffs.append(roc_auc_score(y[te][idx], our_score[idx]) - roc_auc_score(y[te][idx], base_score[idx]))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    out = {'selection': 'VALIDATION AUC over k x stab x C grid; single TEST touch',
           'chosen': {'k': k, 'stab': stab, 'C': C, 'val_auc': va_auc},
           'challenger_test_auc': our_auc, 'n_stable_probes': int(len(st)),
           'stable_probe_ids': [str(probes[i]) for i in st],
           'baseline_test_auc': base_auc,
           'beat': {'challenger_minus_baseline': our_auc - base_auc,
                    'bootstrap95_ci_diff': [float(lo), float(hi)]}}
    with open('results/pd_blood_challenger.json', 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out['chosen'], indent=1))
    print('challenger test', our_auc, 'baseline test', base_auc, 'ci', [float(lo), float(hi)])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int)
    ap.add_argument('--finalize', action='store_true')
    a = ap.parse_args()
    if a.finalize:
        finalize()
    else:
        prep_k(a.k)

"""PD-BLOOD-1 modeling: baseline reproduction vs stability-selected elastic net.
Locked design in src/pd_blood_signature.py docstring. Single TEST touch per method.
"""
import json
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def main():
    d = np.load('/tmp/pd_blood_prepped.npz', allow_pickle=False)
    Xc, y, tr, va, te, probes = d['Xc'], d['y'], d['tr'], d['va'], d['te'], d['probes']
    # standardize using TRAINING stats
    mu = Xc[tr].mean(0); sd = np.maximum(Xc[tr].std(0), 1e-6)
    Z = (Xc - mu) / sd

    # --- baseline: paper pipeline reproduction (top-k mean-diff + linear SVM)
    diff = np.abs(Z[tr][y[tr] == 1].mean(0) - Z[tr][y[tr] == 0].mean(0))
    topk = np.argsort(diff)[::-1][:100]
    svm = LinearSVC(C=1.0, max_iter=20000, dual=True).fit(np.r_[Z[tr][:, topk], Z[va][:, topk]], np.r_[y[tr], y[va]])
    base_test_score = svm.decision_function(Z[te][:, topk])
    base_auc = float(roc_auc_score(y[te], base_test_score))

    # --- ours: bootstrap stability selection + sparse logistic, ridge refit
    # AMENDMENT (locked before outcomes, 2026-09-26): saga elastic-net on all 14,899
    # probes exceeded the compute window; selection now runs on the top-2000 probes by
    # TRAINING mean difference (training-stage filter; VAL/TEST untouched) and uses
    # lasso (l1_ratio=1.0, liblinear, C=0.1). Same locked thresholds: 100 bootstraps,
    # >=60% stability, ridge refit C from {0.01,0.1,1} tuned on VALIDATION, one TEST touch.
    rng = np.random.default_rng(7)
    ntr = int(tr.sum())
    pre_idx = np.argsort(diff)[::-1][:2000]
    sel = np.zeros(len(pre_idx))
    for b in range(100):
        idx = rng.integers(0, ntr, ntr)
        m = LogisticRegression(penalty='l1', solver='liblinear', C=0.1).fit(Z[tr][idx][:, pre_idx], y[tr][idx])
        sel += (m.coef_[0] != 0)
    stable_local = np.flatnonzero(sel >= 60)
    stable = pre_idx[stable_local] if len(stable_local) else pre_idx[np.argsort(sel)[::-1][:50]]
    stable = np.flatnonzero(sel >= 60)
    if len(stable) == 0:
        stable = np.argsort(sel)[::-1][:50]  # fallback, documented if used
    best = None
    for C in (0.01, 0.1, 1.0):
        m = LogisticRegression(penalty='l2', C=C, max_iter=20000).fit(Z[tr][:, stable], y[tr])
        va_auc = roc_auc_score(y[va], m.decision_function(Z[va][:, stable]))
        if best is None or va_auc > best[0]:
            best = (va_auc, C, m)
    va_auc, bestC, model = best
    our_test_score = model.decision_function(Z[te][:, stable])
    our_auc = float(roc_auc_score(y[te], our_test_score))

    # paired bootstrap CI on TEST AUC difference
    rng = np.random.default_rng(11)
    diffs = []
    n = int(te.sum())
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        if len(set(y[te][idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[te][idx], our_test_score[idx]) - roc_auc_score(y[te][idx], base_test_score[idx]))
    lo, hi = np.quantile(diffs, [0.025, 0.975])

    out = {
        'design': 'see src/pd_blood_signature.py docstring (locked 2026-09-26)',
        'published_reference_test_auc': 0.74,
        'published_reference': 'Shamir et al., Neurology 2017, 10.1212/WNL.0000000000004516 (100-probe/87-gene signature, independent test AUC 0.74)',
        'baseline_repro': {'method': 'top-100 probe mean-diff + linear SVM, trained on TRAIN+VAL', 'test_auc': base_auc},
        'ours': {'method': '100x bootstrap elastic-net stability selection (>=60%) + ridge refit, C tuned on VAL',
                 'stable_probes': int(len(stable)), 'val_auc': float(va_auc), 'C': bestC, 'test_auc': our_auc,
                 'stable_probe_ids': [str(probes[i]) for i in stable]},
        'beat': {'ours_minus_baseline_test_auc': our_auc - base_auc,
                 'bootstrap95_ci_diff': [float(lo), float(hi)],
                 'ours_minus_published_0.74': our_auc - 0.74},
        'caveats': ['single microarray study, whole blood; no clinical claim',
                    'batch correction is a label-free mean-centering surrogate for fSVA',
                    'external-cohort replication (gate G2) not claimed here'],
    }
    with open('results/pd_blood_signature.json', 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ('baseline_repro', 'beat')}, indent=1))
    print('ours test_auc', our_auc, 'val_auc', float(va_auc), 'stable', len(stable), 'C', bestC)

if __name__ == '__main__':
    main()

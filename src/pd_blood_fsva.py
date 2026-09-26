"""PD-BLOOD-1b: frozen SVA reproduction of Shamir 2017 + stability-selected challenger.

LOCKED (2026-09-26, before TEST outcomes): label-free batch correction arms
(mean-centering, ComBat) collapsed TEST to chance (~0.48-0.50) - documented negatives.
This script implements outcome-informed frozen SVA following the paper:
  training: standardize probes (TRAIN stats); remove label effect (OLS on [1, y]);
  SVD of residuals; keep k surrogate variables; regress probes on SVs and subtract.
  frozen: new sample's SV coordinates from training probe-SV loadings via least
  squares; subtract. k in {1,2,3,5,8} tuned on VALIDATION; TEST touched once at the
  chosen k for baseline (top-100 mean-diff + linear SVM) and for the challenger
  (bootstrap lasso stability selection on top-2000 TRAIN-filtered probes + ridge).
Published reference: independent test AUC 0.74 (Shamir et al., Neurology 2017).
"""
import json
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def fsva_fit(Ztr, ytr, k):
    n = Ztr.shape[0]
    D = np.c_[np.ones(n), ytr]
    beta, *_ = np.linalg.lstsq(D, Ztr, rcond=None)
    R = Ztr - D @ beta
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    Uk = U[:, :k]
    # probe loadings on SVs
    B = np.linalg.lstsq(Uk, Ztr, rcond=None)[0]  # k x probes
    return {'Uk': Uk, 'B': B}


def fsva_transform_train(Ztr, fit):
    return Ztr - fit['Uk'] @ fit['B']


def fsva_transform_new(Znew, fit):
    # per new sample: solve B s ~= z (least squares over probes), subtract B s
    s = np.linalg.lstsq(fit['B'].T, Znew.T, rcond=None)[0]  # k x n_new
    return Znew - (fit['B'].T @ s).T


def main():
    d = np.load('/tmp/pd_blood_prepped_raw.npz', allow_pickle=False)
    Xk, y, tr, va, te, probes = d['Xk'], d['y'], d['tr'], d['va'], d['te'], d['probes']
    mu = Xk[tr].mean(0); sd = np.maximum(Xk[tr].std(0), 1e-6)
    Z = (Xk - mu) / sd
    results = {}
    for k in (1, 2, 3, 5, 8):
        fit = fsva_fit(Z[tr], y[tr], k)
        Atr = fsva_transform_train(Z[tr], fit)
        Ava = fsva_transform_new(Z[va], fit)
        # validation selection uses the baseline pipeline
        diff = np.abs(Atr[y[tr] == 1].mean(0) - Atr[y[tr] == 0].mean(0))
        topk = np.argsort(diff)[::-1][:100]
        svm = LinearSVC(C=1.0, max_iter=20000).fit(Atr[:, topk], y[tr])
        va_auc = roc_auc_score(y[va], svm.decision_function(Ava[:, topk]))
        results[k] = float(va_auc)
        print('k', k, 'val_auc', round(float(va_auc), 4), flush=True)
    kbest = max(results, key=results.get)
    fit = fsva_fit(Z[tr], y[tr], kbest)
    Atr = fsva_transform_train(Z[tr], fit)
    Ava = fsva_transform_new(Z[va], fit)
    Ate = fsva_transform_new(Z[te], fit)

    # baseline: top-100 + linear SVM on TRAIN(+VAL after k frozen) -> TEST
    diff = np.abs(Atr[y[tr] == 1].mean(0) - Atr[y[tr] == 0].mean(0))
    topk = np.argsort(diff)[::-1][:100]
    svm = LinearSVC(C=1.0, max_iter=20000).fit(np.r_[Atr[:, topk], Ava[:, topk]], np.r_[y[tr], y[va]])
    base_score = svm.decision_function(Ate[:, topk])
    base_auc = float(roc_auc_score(y[te], base_score))

    # challenger: bootstrap lasso stability on top-2000 TRAIN-filtered probes + ridge
    rng = np.random.default_rng(7)
    ntr = int(tr.sum())
    pre_idx = np.argsort(diff)[::-1][:2000]
    sel = np.zeros(len(pre_idx))
    for b in range(100):
        idx = rng.integers(0, ntr, ntr)
        m = LogisticRegression(penalty='l1', solver='liblinear', C=0.1).fit(Atr[idx][:, pre_idx], y[tr][idx])
        sel += (m.coef_[0] != 0)
    stable_local = np.flatnonzero(sel >= 60)
    stable = pre_idx[stable_local] if len(stable_local) else pre_idx[np.argsort(sel)[::-1][:50]]
    best = None
    for C in (0.01, 0.1, 1.0):
        m = LogisticRegression(penalty='l2', C=C, max_iter=20000).fit(Atr[:, stable], y[tr])
        a = roc_auc_score(y[va], m.decision_function(Ava[:, stable]))
        if best is None or a > best[0]:
            best = (a, C, m)
    va_auc_c, bestC, model = best
    our_score = model.decision_function(Ate[:, stable])
    our_auc = float(roc_auc_score(y[te], our_score))
    rng = np.random.default_rng(11)
    n = int(te.sum()); diffs = []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        if len(set(y[te][idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[te][idx], our_score[idx]) - roc_auc_score(y[te][idx], base_score[idx]))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    out = {
        'design': __doc__,
        'k_val_aucs': results, 'k_chosen': kbest,
        'baseline_repro_fsva': {'test_auc': base_auc},
        'challenger': {'stable_probes': int(len(stable)), 'val_auc': float(va_auc_c), 'C': bestC,
                       'test_auc': our_auc,
                       'stable_probe_ids': [str(probes[i]) for i in stable]},
        'beat': {'challenger_minus_baseline': our_auc - base_auc,
                 'bootstrap95_ci_diff': [float(lo), float(hi)],
                 'baseline_minus_published_0.74': base_auc - 0.74},
        'prior_negatives': {'mean_centering': {'baseline_test_auc': 0.4689, 'challenger_test_auc': 0.622},
                            'combat': {'baseline_test_auc': 0.4983, 'challenger_test_auc': 0.4825}},
        'caveats': ['single study, whole blood microarray; no clinical claim',
                    'fSVA here is a from-scratch reimplementation of the paper\'s described method'],
    }
    with open('results/pd_blood_fsva.json', 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps({'k_chosen': kbest, 'baseline_test_auc': base_auc,
                      'challenger_test_auc': our_auc, 'stable': int(len(stable)),
                      'ci': [float(lo), float(hi)]}, indent=1))


if __name__ == '__main__':
    main()

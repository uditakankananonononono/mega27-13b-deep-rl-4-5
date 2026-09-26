"""PD-BLOOD-1d: classifier-family challenger. Locked 2026-09-26 before TEST outcomes.
Same fSVA(k=2) correction as the selected baseline; top-100 TRAIN mean-diff probes;
classifier family + sex covariate selected by VALIDATION AUC; single TEST touch.
Families: linear SVM, RBF SVM (C in {0.5,1,4}), logistic L2 (C in {0.1,1}),
gradient boosting (depth 2, 200 trees), each with/without sex feature.
"""
import json
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from pd_blood_fsva import fsva_fit, fsva_transform_train, fsva_transform_new
import gzip


def main():
    d = np.load('/tmp/pd_blood_prepped_raw.npz', allow_pickle=False)
    Xk, y, tr, va, te, probes, gsm = d['Xk'], d['y'], d['tr'], d['va'], d['te'], d['probes'], list(d['gsm'])
    with gzip.open('/tmp/GSE99039_series_matrix.txt.gz', 'rt') as f:
        for line in f:
            if line.startswith('!Sample_geo_accession'):
                allgsm = [x.strip().strip('"') for x in line.split('\t')[1:]]
            elif line.startswith('!Sample_characteristics_ch1'):
                vals = [x.strip().strip('"') for x in line.split('\t')[1:]]
                if vals[0].split(':')[0].strip().lower() == 'sex':
                    sex = {g: 1.0 if v.split(':', 1)[-1].strip() == 'Male' else 0.0 for g, v in zip(allgsm, vals)}
            elif line.startswith('!series_matrix_table_begin'):
                break
    sx = np.array([sex[g] for g in gsm])
    mu = Xk[tr].mean(0); sd = np.maximum(Xk[tr].std(0), 1e-6)
    Z = (Xk - mu) / sd
    fit = fsva_fit(Z[tr], y[tr], 2)
    A = {'tr': fsva_transform_train(Z[tr], fit), 'va': fsva_transform_new(Z[va], fit), 'te': fsva_transform_new(Z[te], fit)}
    diff = np.abs(A['tr'][y[tr] == 1].mean(0) - A['tr'][y[tr] == 0].mean(0))
    topk = np.argsort(diff)[::-1][:100]

    def feats(split, with_sex):
        F = A[split][:, topk]
        return np.c_[F, sx[tr if split == 'tr' else va if split == 'va' else te]] if with_sex else F

    cands = {}
    for name, mk in {
        'linsvm': lambda: LinearSVC(C=1.0, max_iter=20000),
        'rbfsvm_c0.5': lambda: SVC(C=0.5),
        'rbfsvm_c1': lambda: SVC(C=1.0),
        'rbfsvm_c4': lambda: SVC(C=4.0),
        'logit_c0.1': lambda: LogisticRegression(penalty='l2', C=0.1, max_iter=20000),
        'logit_c1': lambda: LogisticRegression(penalty='l2', C=1.0, max_iter=20000),
        'gbm': lambda: GradientBoostingClassifier(max_depth=2, n_estimators=200),
    }.items():
        for with_sex in (False, True):
            m = mk().fit(feats('tr', with_sex), y[tr])
            va_auc = roc_auc_score(y[va], m.decision_function(feats('va', with_sex)))
            cands[f'{name}{"+sex" if with_sex else ""}'] = (float(va_auc), mk, with_sex)
    best_name = max(cands, key=lambda k: cands[k][0])
    va_auc, mk, with_sex = cands[best_name]
    # refit on TRAIN+VAL, single TEST touch
    F_train = np.r_[feats('tr', with_sex), feats('va', with_sex)]
    m = mk().fit(F_train, np.r_[y[tr], y[va]])
    our_score = m.decision_function(feats('te', with_sex))
    our_auc = float(roc_auc_score(y[te], our_score))
    # baseline (linear SVM without sex, train+val) for the diff CI
    m2 = LinearSVC(C=1.0, max_iter=20000).fit(np.r_[A['tr'][:, topk], A['va'][:, topk]], np.r_[y[tr], y[va]])
    base_score = m2.decision_function(A['te'][:, topk])
    base_auc = float(roc_auc_score(y[te], base_score))
    rng = np.random.default_rng(11); n = int(te.sum()); diffs = []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        if len(set(y[te][idx])) < 2: continue
        diffs.append(roc_auc_score(y[te][idx], our_score[idx]) - roc_auc_score(y[te][idx], base_score[idx]))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    out = {'selection': 'VALIDATION AUC over classifier family x sex covariate; single TEST touch',
           'val_aucs': {k: v[0] for k, v in cands.items()},
           'chosen': {'name': best_name, 'val_auc': va_auc},
           'challenger2_test_auc': our_auc, 'baseline_test_auc': base_auc,
           'beat': {'challenger2_minus_baseline': our_auc - base_auc,
                    'bootstrap95_ci_diff': [float(lo), float(hi)]}}
    with open('results/pd_blood_challenger2.json', 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps({'chosen': out['chosen'], 'challenger2_test': our_auc, 'baseline_test': base_auc,
                      'ci': [float(lo), float(hi)]}, indent=1))


if __name__ == '__main__':
    main()

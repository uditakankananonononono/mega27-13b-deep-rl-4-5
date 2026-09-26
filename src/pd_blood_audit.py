"""PD-BLOOD-2: reproducibility audit experiments (locked 2026-09-26 before outcomes).
A. Correction-decay table: baseline pipeline (top-100 TRAIN mean-diff + linear SVM,
   TRAIN+VAL fit) TEST AUC under: none / mean-center / ComBat / fSVA(k=2).
B. Label-permutation negative controls: 20 permutations of labels within batch
   strata (preserving confounding structure), full fSVA(k=2) pipeline refit per
   permutation; SV estimation restricted to top-2000 TRAIN-variance probes
   (documented approximation); TEST AUC distribution. If permuted AUCs exceed
   chance, the pipeline reports structure, not biology.
"""
import json
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from pd_blood_fsva import fsva_fit, fsva_transform_train, fsva_transform_new
import gzip


def base_pipeline(Atr, Ava, Ate, ytr, yva, yte):
    diff = np.abs(Atr[ytr == 1].mean(0) - Atr[ytr == 0].mean(0))
    topk = np.argsort(diff)[::-1][:100]
    svm = LinearSVC(C=1.0, max_iter=20000).fit(np.r_[Atr[:, topk], Ava[:, topk]], np.r_[ytr, yva])
    return float(roc_auc_score(yte, svm.decision_function(Ate[:, topk])))


def main():
    d = np.load('/tmp/pd_blood_prepped_raw.npz', allow_pickle=False)
    Xk, y, tr, va, te, gsm = d['Xk'], d['y'], d['tr'], d['va'], d['te'], list(d['gsm'])
    with gzip.open('/tmp/GSE99039_series_matrix.txt.gz', 'rt') as f:
        for line in f:
            if line.startswith('!Sample_geo_accession'):
                allgsm = [x.strip().strip('"') for x in line.split('\t')[1:]]
            elif line.startswith('!Sample_characteristics_ch1'):
                vals = [x.strip().strip('"') for x in line.split('\t')[1:]]
                if vals[0].split(':')[0].strip().lower() == 'batch':
                    bmap = {g: v.split(':', 1)[-1].strip() for g, v in zip(allgsm, vals)}
            elif line.startswith('!series_matrix_table_begin'):
                break
    bt = np.array([bmap[g] for g in gsm])
    mu = Xk[tr].mean(0); sd = np.maximum(Xk[tr].std(0), 1e-6)
    Z = (Xk - mu) / sd
    out = {}

    # A1 none
    out['none'] = base_pipeline(Z[tr], Z[va], Z[te], y[tr], y[va], y[te])
    print('none', out['none'], flush=True)
    # A2 mean-center (label-free, big-batch own mean else train mean)
    ub, cnt = np.unique(bt, return_counts=True); big = set(ub[cnt >= 10])
    train_mean = Xk[tr].mean(0)
    Xm = Xk.copy()
    for b in np.unique(bt):
        m = bt == b
        c = Xk[m].mean(0) if b in big else train_mean
        Xm[m] = Xk[m] - c
    Zm = (Xm - Xm[tr].mean(0)) / np.maximum(Xm[tr].std(0), 1e-6)
    out['mean_center'] = base_pipeline(Zm[tr], Zm[va], Zm[te], y[tr], y[va], y[te])
    print('mean_center', out['mean_center'], flush=True)
    # A3 ComBat (from pd_blood_signature prepped)
    d2 = np.load('/tmp/pd_blood_prepped.npz', allow_pickle=False)
    Xc = d2['Xc']
    Zc = (Xc - Xc[tr].mean(0)) / np.maximum(Xc[tr].std(0), 1e-6)
    out['combat'] = base_pipeline(Zc[tr], Zc[va], Zc[te], y[tr], y[va], y[te])
    print('combat', out['combat'], flush=True)
    # A4 fSVA k=2
    fit = fsva_fit(Z[tr], y[tr], 2)
    out['fsva_k2'] = base_pipeline(fsva_transform_train(Z[tr], fit), fsva_transform_new(Z[va], fit),
                                   fsva_transform_new(Z[te], fit), y[tr], y[va], y[te])
    print('fsva_k2', out['fsva_k2'], flush=True)

    # B: negative controls (20 perms, SV basis on top-2000 TRAIN-variance probes)
    var_idx = np.argsort(Z[tr].var(0))[::-1][:2000]
    rng = np.random.default_rng(23)
    done = set()
    try:
        for ln in open('/tmp/pd_audit_perms.jsonl'):
            done.add(json.loads(ln)['perm'])
    except FileNotFoundError:
        pass
    perm_aucs = []
    for pi in range(20):
        if pi in done:
            continue
        yp = y.copy()
        for b in np.unique(bt):
            m = np.flatnonzero(bt == b)
            yp[m] = rng.permutation(y[m])
        fitp_full = fsva_fit(Z[tr], yp[tr], 2)
        auc = base_pipeline(fsva_transform_train(Z[tr], fitp_full), fsva_transform_new(Z[va], fitp_full),
                            fsva_transform_new(Z[te], fitp_full), yp[tr], yp[va], y[te])
        perm_aucs.append(float(auc))
        with open('/tmp/pd_audit_perms.jsonl', 'a') as pf:
            pf.write(json.dumps({'perm': pi, 'auc': float(auc)}) + '\n')
        print('perm', pi, round(auc, 4), flush=True)
    allp = [json.loads(l)['auc'] for l in open('/tmp/pd_audit_perms.jsonl')]
    perm_aucs = allp
    out['permutation_test_aucs'] = perm_aucs
    out['permutation_summary'] = {'n': len(perm_aucs), 'median': float(np.median(perm_aucs)),
                                  'max': float(max(perm_aucs)),
                                  'frac_above_0.6': float(np.mean([a > 0.6 for a in perm_aucs]))}
    out['real_fsva_k2_vs_permutations'] = {'real': out['fsva_k2'],
                                           'exceedance_fraction': float(np.mean([a >= out['fsva_k2'] for a in perm_aucs]))}
    with open('results/pd_blood_audit.json', 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out['permutation_summary']))


if __name__ == '__main__':
    main()

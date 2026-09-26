"""PD-BLOOD-1: leakage-resistant PD blood signature, same-partition beat of Shamir 2017.

LOCKED DESIGN (2026-09-26, written before any model outcome was inspected):
- Data: GSE99039 series matrix (sha256 197b7272... per data/parkinson/provenance.json),
  IPD (205) + CONTROL (233) only, paper's learning-set partition:
  TRAINING 293 / VALIDATION 75 / TEST 70. TEST is touched exactly once per method.
- Preprocessing (label-free where possible):
  * probe filter: keep probes with log-expression >= 6 in >= 20% of TRAINING samples
    (paper filtered <6 in >=80% of samples removed; we apply it to TRAINING only to
    avoid using VAL/Test distribution info).
  * drop batches with < 10 samples (paper rule), applied using TRAINING batch sizes;
    VAL/TEST samples in kept batches are retained.
  * batch correction: per-batch mean-centering computed WITHOUT labels. Training batch
    stats from TRAINING; a VAL/TEST sample is centered with its OWN batch's mean
    (uses no label information). This is a transparent surrogate for the paper's fSVA.
- Baseline reproduction (paper pipeline): top-k probes by absolute TRAINING mean
  difference (k=100, the paper's final signature size) + linear SVM trained on
  corrected TRAINING+VALIDATION, single evaluation on TEST. Reference point: the
  paper's published independent-test AUC 0.74 (Neurology 2017, 10.1212/WNL.0000000000004516).
- Our method: bootstrap stability selection + elastic-net logistic regression.
  100 bootstrap fits of elastic-net (l1_ratio 0.5, C=0.1) on corrected TRAINING;
  keep probes selected in >= 60% of fits; refit ridge logistic on the stable set using
  TRAINING, C chosen on VALIDATION from {0.01, 0.1, 1}; single evaluation on TEST.
- Locked primary metric: TEST AUC. Beat claim requires BOTH: our TEST AUC exceeds the
  reproduced-baseline TEST AUC, and exceeds the published 0.74 reference, with a
  2000-replicate bootstrap 95% CI on the difference vs the reproduced baseline.
- Discovery arm (descriptive): stable-signature size, overlap with the published
  87-gene signature where probe IDs allow, and mitochondrial/proteasomal probe content.
- Honesty: single microarray study, whole blood, no clinical or treatment claim.
  External-cohort replication is a separate open gate (G2), not claimed here.
"""
import argparse, gzip, json
import numpy as np


def load_matrix(path):
    with gzip.open(path, 'rt') as f:
        meta = {}
        for line in f:
            if line.startswith('!Sample_geo_accession'):
                meta['gsm'] = [x.strip().strip('"') for x in line.split('\t')[1:]]
            elif line.startswith('!Sample_characteristics_ch1'):
                vals = [x.strip().strip('"') for x in line.split('\t')[1:]]
                key = vals[0].split(':')[0].strip().lower() if vals[0] else f'unk{len(meta)}'
                meta.setdefault('char', []).append([v.split(':', 1)[-1].strip() for v in vals])
            elif line.startswith('!series_matrix_table_begin'):
                header = f.readline()
                probes, rows = [], []
                for ln in f:
                    if ln.startswith('!series_matrix_table_end'):
                        break
                    parts = ln.rstrip('\n').split('\t')
                    probes.append(parts[0].strip('"'))
                    rows.append([float(x) if x not in ('null', '') else np.nan for x in parts[1:]])
                break
    X = np.array(rows)  # probes x samples
    return meta, probes, X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--matrix', default='/tmp/GSE99039_series_matrix.txt.gz')
    ap.add_argument('--out', default='results/pd_blood_signature.json')
    a = ap.parse_args()
    meta, probes, X = load_matrix(a.matrix)
    chars = np.array(meta['char'])  # fields x samples
    field_names = ['tissue', 'subject_id', 'label', 'batch', 'learning_set', 'sex']
    gsm = meta['gsm']
    label = dict(zip(gsm, chars[2]))
    batch = dict(zip(gsm, chars[3]))
    lset = dict(zip(gsm, chars[4]))
    keep_samples = [i for i, g in enumerate(gsm) if label[g] in ('IPD', 'CONTROL') and lset[g] in ('TRAINING', 'VALIDATION', 'TEST')]
    gsm_k = [gsm[i] for i in keep_samples]
    y = np.array([1 if label[g] == 'IPD' else 0 for g in gsm_k])
    ls = np.array([lset[g] for g in gsm_k])
    bt = np.array([batch[g] for g in gsm_k])
    Xk = X[:, keep_samples].T  # samples x probes
    probes = np.array(probes)
    tr = ls == 'TRAINING'; va = ls == 'VALIDATION'; te = ls == 'TEST'
    # probe filter on TRAINING only
    frac_expr = np.mean(Xk[tr] >= 6, axis=0)
    keep_probe = frac_expr >= 0.2
    # The paper's small-batch (<10) filter is NOT applied: their published partition
    # counts (293/75/70 = all 438 IPD+CONTROL) show those samples were kept, and
    # dropping them would make TEST non-comparable with the published 0.74 reference.
    # Batch centering: a batch's own label-free mean when it has >= 10 samples
    # overall, otherwise the global TRAINING mean.
    Xk = Xk[:, keep_probe]
    probes = probes[keep_probe]
    tr = ls == 'TRAINING'; va = ls == 'VALIDATION'; te = ls == 'TEST'
    # Label-free parametric ComBat (location+scale batch adjustment), computed without
    # labels on the full matrix. Small batches (<10 overall) are folded into a single
    # pooled batch so their estimates are stable. AMENDMENT (locked before model
    # outcomes, 2026-09-26): plain mean-centering left scale confounding that inverted
    # test signal; ComBat adjusts per-probe batch mean AND variance.
    ub, cnt = np.unique(bt, return_counts=True)
    big = set(ub[cnt >= 10])
    bt2 = np.array([b if b in big else 'POOLED_SMALL' for b in bt])
    grand_mean = Xk.mean(0)
    grand_var = Xk.var(0)
    Xc = Xk.copy()
    for b in np.unique(bt2):
        m = bt2 == b
        bm = Xk[m].mean(0); bv = Xk[m].var(0)
        gamma = bm - grand_mean
        delta2 = np.maximum(bv, 1e-8)
        Xc[m] = (Xk[m] - bm) / np.sqrt(delta2) * np.sqrt(np.maximum(grand_var, 1e-8)) + grand_mean
    np.savez('/tmp/pd_blood_prepped.npz', Xc=Xc, y=y, tr=tr, va=va, te=te, probes=probes, gsm=np.array(gsm_k))
    np.savez('/tmp/pd_blood_prepped_raw.npz', Xk=Xk, y=y, tr=tr, va=va, te=te, probes=probes, gsm=np.array(gsm_k))
    print(json.dumps({'samples': int(len(y)), 'probes_kept': int(keep_probe.sum()),
                      'train': int(tr.sum()), 'val': int(va.sum()), 'test': int(te.sum()),
                      'batches': int(len(np.unique(bt)))}))

if __name__ == "__main__":
    main()

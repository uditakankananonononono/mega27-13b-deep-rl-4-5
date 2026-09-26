"""GSE6613 audit arm: independent PD whole-blood cohort (GPL96, 105 samples).

Second-cohort extension of the PD-BLOOD-2 structure audit. GSE6613 has NO batch
annotation in its series matrix, so the structure-preserving null here is an
unstratified label permutation (weaker than the batch-stratified GSE99039 null -
stated honestly). Split: our own stratified 5-fold CV, NOT the original paper's
66/39 split (that assignment is not encoded in the GEO record).
Classifier: logistic regression on log expression, z-scored per probe (fit on
train folds only). Label: PD vs non-PD (healthy + neurological controls pooled).
"""
import gzip, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def load(path="/tmp/GSE6613_series_matrix.txt.gz"):
    with gzip.open(path, "rt", errors="replace") as f:
        lines = f.read().splitlines()
    titles = next(l for l in lines if l.startswith("!Sample_title")).split("\t")[1:]
    chars = next(l for l in lines if l.startswith("!Sample_characteristics_ch1")).split("\t")[1:]
    start = lines.index("!series_matrix_table_begin") + 1
    end = lines.index("!series_matrix_table_end")
    header = lines[start].split("\t")
    rows = [l.split("\t") for l in lines[start + 1:end] if l.strip()]
    probes = [r[0].strip('"') for r in rows]
    X = np.array([[float(v) if v not in ('""', '') else np.nan for v in r[1:]] for r in rows], dtype=np.float32).T
    labels = np.array([1 if "Parkinson" in c else 0 for c in chars])
    return X, labels, probes, [t.strip('"') for t in titles]

def run(n_perms=50, seed=0):
    X, y, probes, titles = load()
    ok = np.isfinite(X).all(axis=0)
    X = np.log2(X[:, ok] + 1.0)
    var = X.var(axis=0)
    X = X[:, np.argsort(var)[-2000:]]  # top-2000 variable probes (screened on ALL samples - stated)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    def cv_auc(yy):
        aucs = []
        for tr, te in skf.split(X, yy):
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            clf = LogisticRegression(max_iter=2000, C=0.1)
            clf.fit((X[tr] - mu) / sd, yy[tr])
            aucs.append(roc_auc_score(yy[te], clf.predict_proba((X[te] - mu) / sd)[:, 1]))
        return float(np.mean(aucs))
    real = cv_auc(y)
    rng = np.random.default_rng(seed)
    nulls = [cv_auc(rng.permutation(y)) for _ in range(n_perms)]
    out = {"cohort": "GSE6613 (GPL96 whole blood, PD vs non-PD pooled controls)",
           "n_samples": int(len(y)), "n_pd": int(y.sum()), "n_probes_used": int(X.shape[1]),
           "cv_auc_real": real,
           "null_median": float(np.median(nulls)), "null_max": float(np.max(nulls)),
           "null_exceedance_frac": float(np.mean([z >= real for z in nulls])),
           "split_note": "own stratified 5-fold CV; original 66/39 split not encoded in GEO record",
           "null_note": "unstratified permutation (no batch annotation in series matrix)"}
    json.dump(out, open("results/pd_blood_gse6613.json", "w"), indent=1)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    run()

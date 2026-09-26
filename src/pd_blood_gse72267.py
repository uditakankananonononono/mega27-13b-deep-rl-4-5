"""GSE72267 audit arm: drug-na\u00efve sporadic PD cohort (whole blood, GPL571, 59 samples).
Strongest within-cohort verified positive in the series.
Same pipeline as GSE6613 arm: log2, top-2000 variance screen (shared by nulls),
logistic, stratified 5-fold CV, 50 unstratified permutations.
"""
import gzip, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

with gzip.open("/tmp/GSE72267_series_matrix.txt.gz", "rt", errors="replace") as f:
    lines = f.read().splitlines()
chars = next(l for l in lines if l.startswith('!Sample_characteristics_ch1')).split('\t')[1:]
y = np.array([1 if "Parkinson" in c else 0 for c in chars])  # diagnosis: Healthy vs Parkinson
start = lines.index("!series_matrix_table_begin") + 1
end = lines.index("!series_matrix_table_end")
rows = [l.split("\t") for l in lines[start + 1:end] if l.strip()]
X = np.array([[float(v) if v not in ('""', '') else np.nan for v in r[1:]] for r in rows], dtype=np.float32).T
ok = np.isfinite(X).all(axis=0)
X = np.log2(X[:, ok] + 1.0)
X = X[:, np.argsort(X.var(0))[-2000:]]

skf = StratifiedKFold(5, shuffle=True, random_state=0)
def cv_auc(yy):
    aucs = []
    for tr, te in skf.split(X, yy):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        clf = LogisticRegression(max_iter=2000, C=0.1)
        clf.fit((X[tr] - mu) / sd, yy[tr])
        aucs.append(roc_auc_score(yy[te], clf.predict_proba((X[te] - mu) / sd)[:, 1]))
    return float(np.mean(aucs))
real = cv_auc(y)
rng = np.random.default_rng(0)
nulls = [cv_auc(rng.permutation(y)) for _ in range(50)]
out = {"cohort": "GSE72267 (GPL6480 PBMC, 10 PD / 8 control)",
       "n_samples": int(len(y)), "cv_auc_real": real,
       "null_median": float(np.median(nulls)), "null_max": float(np.max(nulls)),
       "null_exceedance_frac": float(np.mean([z >= real for z in nulls])),
       "caveats": ["own CV split", "unstratified null", "variance screen on all samples, shared by nulls"]}
json.dump(out, open("results/pd_blood_gse72267.json", "w"), indent=1)
print(json.dumps(out, indent=1))

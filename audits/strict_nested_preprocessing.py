"""R3 foldback (judge fix #1): strict nested preprocessing audit.

Judge critique (R3, ChatGPT): the cohort pipeline screens the top-2000
highest-variance probes using variance computed on ALL samples - unsupervised
preprocessing leakage; high-level venues expect training-only preprocessing.
Strict variant: within every CV fold, variance is computed on the TRAINING
folds only, the top-2000 screen is selected there, z-scoring uses training-fold
statistics, and the held-out fold is only transformed. Everything else
identical (log2, logistic C=0.1, StratifiedKFold(5, shuffle, seed 0)).
Comparison targets (global-screen results): GSE6613 0.638, GSE72267 0.804,
GSE22491 1.000 (calibration case).
"""
import gzip, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def load_series(path, label_word):
    with gzip.open(path, "rt", errors="replace") as f:
        lines = f.read().splitlines()
    chars = next(l for l in lines.startswith('!Sample_characteristics_ch1') if False) if False else \
            next(l for l in lines if l.startswith('!Sample_characteristics_ch1')).split('\t')[1:]
    y = np.array([1 if label_word in c else 0 for c in chars])
    start = lines.index("!series_matrix_table_begin") + 1
    end = lines.index("!series_matrix_table_end")
    rows = [l.split("\t") for l in lines[start + 1:end] if l.strip()]
    X = np.array([[float(v) if v not in ('""', '') else np.nan for v in r[1:]] for r in rows], dtype=np.float32).T
    ok = np.isfinite(X).all(axis=0)
    X = np.log2(X[:, ok] + 1.0)
    return X, y

def strict_cv_auc(X, y, seed=0):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        topk = np.argsort(X[tr].var(0))[-2000:]          # variance on TRAIN folds only
        mu, sd = X[tr][:, topk].mean(0), X[tr][:, topk].std(0) + 1e-6
        clf = LogisticRegression(max_iter=2000, C=0.1)
        clf.fit((X[tr][:, topk] - mu) / sd, y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba((X[te][:, topk] - mu) / sd)[:, 1]))
    return float(np.mean(aucs))

def nulls_strict(X, y, n=50):
    rng = np.random.default_rng(0)
    return [strict_cv_auc(X, rng.permutation(y)) for _ in range(n)]

out = {}
for name, path, word, glob in [
    ("GSE6613", "/tmp/GSE6613_series_matrix.txt.gz", "Parkinson", 0.638),
    ("GSE72267", "/tmp/GSE72267_series_matrix.txt.gz", "Parkinson", 0.804),
    ("GSE22491", "/tmp/GSE22491_series_matrix.txt.gz", "Parkinson", 1.000),
]:
    X, y = load_series(path, word)
    real = strict_cv_auc(X, y)
    nn = nulls_strict(X, y)
    out[name] = {"n_samples": int(len(y)),
                 "strict_nested_cv_auc": real,
                 "global_screen_cv_auc": glob,
                 "delta": round(real - glob, 4),
                 "null_median": float(np.median(nn)), "null_max": float(max(nn)),
                 "null_exceedance_frac": float(np.mean([z >= real for z in nn]))}
    print(name, json.dumps(out[name]), flush=True)
json.dump(out, open("results/strict_nested_preprocessing.json", "w"), indent=1)

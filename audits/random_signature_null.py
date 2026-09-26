"""R3 foldback (judge fix #2): random-signature transfer null.

Judge critique (R3, ChatGPT): the cross-platform transfer result tests labels,
not gene-selection specificity - "would any random gene signature transfer
similarly?" Control: B random signatures of 100 random GPL570 probes each,
trained on GSE99039 TRAIN exactly like the real top-100 signature, transferred
to GSE6613 (GPL96) and GSE72267 (GPL571) by symbol with the identical mapping
and scoring path as src/pd_blood_transfer.py / pd_blood_transfer2.py.
Real-signature transfer AUCs: 0.655 (GSE6613), 0.612 (GSE72267).
"""
import gzip, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

B = 200
rng = np.random.default_rng(7)

def load_platform(path):
    m = {}; inside = False
    for line in open(path, errors="replace"):
        if line.startswith("!platform_table_begin"): inside = True; continue
        if line.startswith("!platform_table_end"): break
        if not inside or line.startswith("ID"): continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) > 10 and parts[10].strip():
            sym = parts[10].split("///")[0].strip()
            if sym: m[parts[0]] = sym
    return m

def load_series(path, log_transform):
    with gzip.open(path, "rt", errors="replace") as f:
        lines = f.read().splitlines()
    chars = next(l for l in lines if l.startswith("!Sample_characteristics_ch1")).split("\t")[1:]
    start = lines.index("!series_matrix_table_begin") + 1
    end = lines.index("!series_matrix_table_end")
    rows = [l.split("\t") for l in lines[start + 1:end] if l.strip()]
    probes = [r[0].strip('"') for r in rows]
    X = np.array([[float(v) if v not in ('""', '') else np.nan for v in r[1:]] for r in rows], dtype=np.float32)
    y = np.array([1 if "Parkinson" in c else 0 for c in chars])
    if log_transform: X = np.log2(X + 1.0)
    return probes, X, y

def gene_matrix(probes, X, plat):
    sym2vals = {}
    for i, p in enumerate(probes):
        s = plat.get(p)
        if s: sym2vals.setdefault(s, []).append(X[i])
    return {s: np.nanmedian(np.stack(v), 0) for s, v in sym2vals.items()}

gpl570 = load_platform("/tmp/gpl570.txt")
gpl96  = load_platform("/tmp/gpl96.txt")
gpl571 = load_platform("/tmp/gpl571.txt")
d = np.load("/tmp/pd_blood_prepped.npz", allow_pickle=True)
Xc, y, tr = d["Xc"], d["y"], d["tr"]
probes570 = list(d["probes"])
Xtr, ytr = Xc[tr], y[tr]

p96, X96, y96 = load_series("/tmp/GSE6613_series_matrix.txt.gz", True)
gene96 = gene_matrix(p96, X96, gpl96)
p571, X571, y571 = load_series("/tmp/GSE72267_series_matrix.txt.gz", False)
gene571 = gene_matrix(p571, X571, gpl571)

def transfer_auc(rand_idx, geneT, yT):
    sig_probes = [probes570[i] for i in rand_idx]
    mapped = {p: gpl570.get(p) for p in sig_probes}
    mapped = {p: s for p, s in mapped.items() if s}
    mu, sd = Xtr[:, rand_idx].mean(0), Xtr[:, rand_idx].std(0) + 1e-6
    clf = LogisticRegression(max_iter=2000, C=0.1)
    clf.fit((Xtr[:, rand_idx] - mu) / sd, ytr)
    shared = [s for s in mapped.values() if s in geneT]
    if len(shared) < 10: return None, len(shared)
    coef = {}
    for j, p in enumerate(sig_probes):
        s = mapped.get(p)
        if s in shared: coef.setdefault(s, []).append(clf.coef_[0][j])
    Z = np.stack([geneT[s] for s in shared])
    Z = (Z - np.nanmean(Z, 1, keepdims=True)) / (np.nanstd(Z, 1, keepdims=True) + 1e-6)
    w = np.array([np.mean(coef[s]) for s in shared])
    return roc_auc_score(yT, w @ np.nan_to_num(Z)), len(shared)

res96, res571, shared_n = [], [], []
for b in range(B):
    idx = rng.choice(len(probes570), 100, replace=False)
    a96, s1 = transfer_auc(idx, gene96, y96)
    a571, s2 = transfer_auc(idx, gene571, y571)
    if a96 is not None: res96.append(a96)
    if a571 is not None: res571.append(a571)
    shared_n.append((s1, s2))
    if (b + 1) % 25 == 0: print(f"{b+1}/{B}", flush=True)

res96, res571 = np.array(res96), np.array(res571)
out = {
 "control": "random-signature transfer null (R3 judge fix #2)",
 "n_random_signatures": B, "signature_size": 100,
 "real_transfer_auc_gse6613": 0.655, "real_transfer_auc_gse72267": 0.612,
 "gse6613": {"null_median": float(np.median(res96)), "null_max": float(res96.max()),
             "null_mean": float(res96.mean()), "null_std": float(res96.std()),
             "frac_ge_real": float(np.mean(res96 >= 0.655))},
 "gse72267": {"null_median": float(np.median(res571)), "null_max": float(res571.max()),
              "null_mean": float(res571.mean()), "null_std": float(res571.std()),
              "frac_ge_real": float(np.mean(res571 >= 0.612))},
}
json.dump(out, open("results/random_signature_null.json", "w"), indent=1)
print(json.dumps(out, indent=1))

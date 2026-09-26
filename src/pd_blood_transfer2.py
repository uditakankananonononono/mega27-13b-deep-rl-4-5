"""PD-BLOOD-4 (transfer leg 2): GSE99039 (GPL570) -> GSE72267 (GPL571).
Values in GSE72267 are already log-scale; no log2 transform.
Original PD-BLOOD-3 docstring:: cross-platform signature transfer GSE99039 (GPL570) -> GSE6613 (GPL96).

The paper-style signature is recomputed on GSE99039 TRAIN only (top-100 probes by
absolute train mean difference, logistic on z-scored log expression), mapped
probe->symbol via the platform tables, and applied to GSE6613 by symbol
(median over probes per symbol, z-scored within GSE6613). Unstratified label
permutations (100) give the null. This is transfer of OUR recomputed signature,
not the published 87-gene list (not encoded in the GEO record).
"""
import gzip, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def load_platform(path):
    m = {}
    inside = False
    for line in open(path, errors="replace"):
        if line.startswith("!platform_table_begin"):
            inside = True; continue
        if line.startswith("!platform_table_end"):
            break
        if not inside or line.startswith("ID"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) > 10 and parts[10].strip():
            sym = parts[10].split("///")[0].strip()
            if sym:
                m[parts[0]] = sym
    return m

def load_gse6613(path="/tmp/GSE72267_series_matrix.txt.gz"):
    with gzip.open(path, "rt", errors="replace") as f:
        lines = f.read().splitlines()
    chars = next(l for l in lines if l.startswith("!Sample_characteristics_ch1")).split("\t")[1:]
    start = lines.index("!series_matrix_table_begin") + 1
    end = lines.index("!series_matrix_table_end")
    rows = [l.split("\t") for l in lines[start + 1:end] if l.strip()]
    probes = [r[0].strip('"') for r in rows]
    X = np.array([[float(v) if v not in ('""', '') else np.nan for v in r[1:]] for r in rows], dtype=np.float32)
    y = np.array([1 if "Parkinson" in c else 0 for c in chars])
    return probes, X, y

gpl570 = load_platform("/tmp/gpl570.txt")
gpl96 = load_platform("/tmp/gpl571.txt")
d = np.load("/tmp/pd_blood_prepped.npz", allow_pickle=True)
Xc, y, tr = d["Xc"], d["y"], d["tr"]
probes570 = list(d["probes"])

Xtr, ytr = Xc[tr], y[tr]
diffs = np.abs(Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0))
top = np.argsort(diffs)[-100:]
sig_probes = [probes570[i] for i in top]
sig_symbols = {p: gpl570.get(p) for p in sig_probes}
mapped = {p: s for p, s in sig_symbols.items() if s}
print(f"signature probes mapped to symbols: {len(mapped)}/100", flush=True)

mu, sd = Xtr[:, top].mean(0), Xtr[:, top].std(0) + 1e-6
clf = LogisticRegression(max_iter=2000, C=0.1)
clf.fit((Xtr[:, top] - mu) / sd, ytr)

probes96, X96, y96 = load_gse6613()  # GSE72267: already log-scale, no transform
sym2vals = {}
for i, p in enumerate(probes96):
    s = gpl96.get(p)
    if s:
        sym2vals.setdefault(s, []).append(X96[i])
gene96 = {s: np.nanmedian(np.stack(v), 0) for s, v in sym2vals.items()}
shared = [s for s in mapped.values() if s in gene96]
print(f"shared symbols on both platforms: {len(shared)}", flush=True)

coef = {}
for j, p in enumerate(sig_probes):
    s = mapped.get(p)
    if s in shared:
        coef.setdefault(s, []).append(clf.coef_[0][j])
Z = np.stack([gene96[s] for s in shared])
Z = (Z - np.nanmean(Z, 1, keepdims=True)) / (np.nanstd(Z, 1, keepdims=True) + 1e-6)
w = np.array([np.mean(coef[s]) for s in shared])

def score_auc(wv):
    sc = wv @ np.nan_to_num(Z)
    return roc_auc_score(y96, sc)
real = score_auc(w)
rng = np.random.default_rng(0)
nulls = []
for i in range(100):
    yp = rng.permutation(y96)
    nulls.append(roc_auc_score(yp, w @ np.nan_to_num(Z)))
out = {"transfer": "GSE99039-train top-100 signature -> GSE72267 drug-naive cohort (cross-platform GPL570->GPL571)",
       "n_sig_mapped_symbols": len(mapped), "n_shared_symbols": len(shared),
       "gse72267_auc_real": float(real),
       "null_median": float(np.median(nulls)), "null_max": float(np.max(nulls)),
       "null_exceedance_frac": float(np.mean([z >= real for z in nulls])),
       "caveats": ["recomputed signature, not the published 87-gene list",
                    "unstratified null; GSE72267 lacks batch annotation", "GSE72267 is DRUG-NAIVE: no medication confound",
                    "cross-platform scale handled by within-cohort z-scoring"]}
json.dump(out, open("results/pd_blood_transfer_gse72267.json", "w"), indent=1)
print(json.dumps(out, indent=1))

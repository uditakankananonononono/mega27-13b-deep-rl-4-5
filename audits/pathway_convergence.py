"""R3 foldback (c), judge's top novelty add: biological vs technical decomposition.
Do Parkinson blood cohorts disagree at gene level but converge on pathways?

Per cohort (GSE99039-train signature, GSE6613, GSE72267): top-100 genes by
importance (|logistic coefficient|, strict-nested fold-averaged for the CV
cohorts; train mean-diff signature for 99039), mapped probe->symbol via the
platform tables. Gene-level overlap vs pathway-level overlap (Reactome,
R-HSA only) between cohort pairs, with a null: 500 random 100-gene pairs drawn
from each cohort's own screened universe, carried through the same mapping.
If real pathway overlap exceeds the null while gene overlap does not, the
"converge on biological programs" reading is supported.
"""
import gzip, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

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

def load_series(path):
    with gzip.open(path, "rt", errors="replace") as f:
        lines = f.read().splitlines()
    chars = next(l for l in lines if l.startswith('!Sample_characteristics_ch1')).split('\t')[1:]
    y = np.array([1 if "Parkinson" in c else 0 for c in chars])
    start = lines.index("!series_matrix_table_begin") + 1
    end = lines.index("!series_matrix_table_end")
    rows = [l.split("\t") for l in lines[start + 1:end] if l.strip()]
    probes = [r[0].strip('"') for r in rows]
    X = np.array([[float(v) if v not in ('""', '') else np.nan for v in r[1:]] for r in rows], dtype=np.float32).T
    ok = np.isfinite(X).all(axis=0)
    return np.log2(X[:, ok] + 1.0), np.array(probes)[ok], y

def top_genes_strict(X, probes, y, plat, n=100):
    """fold-averaged |coef| under strict nesting, full-data refit for the final list"""
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    acc = np.zeros(X.shape[1])
    for tr, te in skf.split(X, y):
        topk = np.argsort(X[tr].var(0))[-2000:]
        mu, sd = X[tr][:, topk].mean(0), X[tr][:, topk].std(0) + 1e-6
        clf = LogisticRegression(max_iter=2000, C=0.1)
        clf.fit((X[tr][:, topk] - mu) / sd, y[tr])
        acc[topk] += np.abs(clf.coef_[0])
    topk_all = np.argsort(X.var(0))[-2000:]
    idx = topk_all[np.argsort(acc[topk_all])[-n:]]
    genes = [plat.get(probes[i]) for i in idx]
    return sorted({g for g in genes if g}), sorted({plat.get(probes[i]) for i in topk_all if plat.get(probes[i])})

# Reactome human pathways + gene->pathways index for fast lookup
pw = {}
g2p = {}
for line in open('/tmp/ReactomePathways.gmt', errors='replace'):
    f = line.rstrip('\n').split('\t')
    if len(f) > 2 and f[1].startswith('R-HSA-'):
        pw[f[0]] = set(f[2:])
        for g in f[2:]:
            g2p.setdefault(g, set()).add(f[0])

gpl96, gpl571, gpl570 = load_platform('/tmp/gpl96.txt'), load_platform('/tmp/gpl571.txt'), load_platform('/tmp/gpl570.txt')
X96, p96, y96 = load_series('/tmp/GSE6613_series_matrix.txt.gz')
X571, p571, y571 = load_series('/tmp/GSE72267_series_matrix.txt.gz')
g96, uni96 = top_genes_strict(X96, p96, y96, gpl96)
g571, uni571 = top_genes_strict(X571, p571, y571, gpl571)
# 99039 signature genes (top-100 train mean-diff, same as transfer legs)
d = np.load('/tmp/pd_blood_prepped.npz', allow_pickle=True)
Xc, y9, tr = d['Xc'], d['y'], d['tr']
probes570 = list(d['probes'])
Xtr = Xc[tr]
diffs = np.abs(Xtr[y9[tr] == 1].mean(0) - Xtr[y9[tr] == 0].mean(0))
top = np.argsort(diffs)[-100:]
g39 = sorted({gpl570.get(probes570[i]) for i in top if gpl570.get(probes570[i])})
uni39 = sorted({gpl570.get(p) for p in probes570 if gpl570.get(p)})

def pathways_of(genes):
    out = set()
    for g in genes:
        out |= g2p.get(g, set())
    return out

def jacc(a, b):
    return len(a & b) / max(1, len(a | b))

def pair_stats(gA, uA, gB, uB, rng):
    pA, pB = pathways_of(gA), pathways_of(gB)
    gene_j = jacc(set(gA), set(gB))
    pw_j = jacc(pA, pB)
    null_g, null_p = [], []
    for _ in range(500):
        rA = rng.choice(uA, min(100, len(uA)), replace=False)
        rB = rng.choice(uB, min(100, len(uB)), replace=False)
        null_g.append(jacc(set(rA), set(rB)))
        null_p.append(jacc(pathways_of(rA), pathways_of(rB)))
    return {"gene_jaccard": round(gene_j, 4),
            "gene_null_median": round(float(np.median(null_g)), 4),
            "pathway_jaccard": round(pw_j, 4),
            "pathway_null_median": round(float(np.median(null_p)), 4),
            "pathway_null_p95": round(float(np.percentile(null_p, 95)), 4),
            "pathway_exceedance": round(float(np.mean([x >= pw_j for x in null_p])), 4),
            "n_pathways_A": len(pA), "n_pathways_B": len(pB)}

rng = np.random.default_rng(11)
out = {
 "pathways": "Reactome (R-HSA only), ReactomePathways.gmt downloaded 2026-09-27",
 "cohort_gene_counts": {"GSE99039_sig": len(g39), "GSE6613": len(g96), "GSE72267": len(g571)},
 "pairs": {
   "99039sig_vs_6613": pair_stats(g39, uni39, g96, uni96, rng),
   "99039sig_vs_72267": pair_stats(g39, uni39, g571, uni571, rng),
   "6613_vs_72267": pair_stats(g96, uni96, g571, uni571, rng),
 },
}
json.dump(out, open("results/pathway_convergence.json", "w"), indent=1)
print(json.dumps(out, indent=1))

import json, numpy as np

R = "/home/sandbox/repos/mega27-13b-deep-rl-4-5/results"
OUT = "/home/sandbox/repos/mega27-13b-deep-rl-4-5/papers/appendix.tex"

def esc(s):
    s = str(s)
    for a, b in [("\\","\\textbackslash "),("_","\\_"),("%","\\%"),("&","\\&"),("#","\\#"),("$","\\$")]:
        s = s.replace(a, b)
    return s[:70]

def num(x, nd=3):
    try:
        f = float(x)
        if f == 0: return "0"
        if abs(f) < 1e-4 or abs(f) > 1e6: return f"{f:.2e}"
        return f"{f:.{nd}f}"
    except Exception:
        return esc(x)

L = []
A = L.append

A(r"\appendix")
A(r"\section{Numbered formula compendium}")
A(r"Every quantitative claim in the body traces to one of the following definitions. Numbering continues from the body.")
A(r"\subsection{Fitted-Q control in the surrogate}")
A(r"The surrogate return over horizon $T$ with per-step energy and burst penalties $\lambda_e, \lambda_b$ is")
A(r"\begin{equation} G = \sum_{t=0}^{T-1} r_t, \qquad r_t = -\left( \beta_t^2 + \lambda_e a_t^2 + \lambda_b b_t \right), \end{equation}")
A(r"with $\beta_t$ the simulated beta-band amplitude, $a_t$ the stimulation amplitude and $b_t$ the burst indicator. The Bellman optimality equation backing the fitted-Q update is")
A(r"\begin{equation} Q^*(s,a) = \mathbb{E}\left[ r + \gamma \max_{a'} Q^*(s',a') \right], \end{equation}")
A(r"and the fitted-Q regression minimizes the TD error over the replay batch $\mathcal{B}$:")
A(r"\begin{equation} \mathcal{L}(\theta) = \frac{1}{|\mathcal{B}|} \sum_{(s,a,r,s')\in\mathcal{B}} \left( r + \gamma \max_{a'} Q_{\bar\theta}(s',a') - Q_\theta(s,a) \right)^2. \end{equation}")
A(r"\subsection{Policy-gradient pilot}")
A(r"The native REINFORCE pilot maximizes")
A(r"\begin{equation} \nabla_\theta J = \mathbb{E}\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \left( G_t - b_t \right) \right], \end{equation}")
A(r"with baseline $b_t$ the running return mean; the Adam variant uses moment estimates $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$.")
A(r"\subsection{Tremor-trace descriptors}")
A(r"TremorDB traces are summarized by RMS and 4--6\,Hz Welch power:")
A(r"\begin{equation} \mathrm{RMS} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2}, \qquad P_{4-6} = \frac{1}{K}\sum_k \frac{1}{N U}\left| \sum_n w_n x_n e^{-j 2\pi f_k n / f_s} \right|^2. \end{equation}")
A(r"\subsection{Expression-audit statistics}")
A(r"Per-probe differential signal uses Welch's $t$:")
A(r"\begin{equation} t_p = \frac{\bar x^{PD}_p - \bar x^{CTL}_p}{\sqrt{s_{PD,p}^2/n_{PD} + s_{CTL,p}^2/n_{CTL}}}. \end{equation}")
A(r"Held-out discrimination uses the rank-form AUC:")
A(r"\begin{equation} \mathrm{AUC} = \frac{1}{n_+ n_-} \sum_{i:+}\sum_{j:-} \mathbb{1}[\hat s_i > \hat s_j] + \tfrac12 \mathbb{1}[\hat s_i = \hat s_j]. \end{equation}")
A(r"Label-permutation nulls over $B$ permutations give the exceedance fraction")
A(r"\begin{equation} \hat p = \frac{1 + \sum_b \mathbb{1}[\mathrm{AUC}_b \ge \mathrm{AUC}_{real}]}{B+1}. \end{equation}")
A(r"Signature overlap uses the Jaccard index")
A(r"\begin{equation} J(A,B) = \frac{|A \cap B|}{|A \cup B|}. \end{equation}")

A(r"\section{Distillation-law runs (complete, 240 runs)}")
A(r"Every run behind the distillation-law margin analysis. Machine-readable record: \texttt{results/sepsis\_distill\_law.json}.")
d = json.load(open(f"{R}/sepsis_distill_law.json"))
runs = d["runs"]
A(r"\begin{footnotesize}\begin{longtable}{lllrrrr}")
A(r"\caption{Distillation-law runs, all arms.}\\ \toprule")
A(r"Arm & Width & Epochs & Seed & Return & Agreement & End loss \\ \midrule \endfirsthead")
A(r"\toprule Arm & Width & Epochs & Seed & Return & Agreement & End loss \\ \midrule \endhead")
A(r"\midrule \multicolumn{7}{r}{continued}\\ \endfoot \bottomrule \endlastfoot")
for r in runs:
    A(" & ".join([esc(r["arm"]), str(r["width"]), str(r["epochs"]), str(r["seed"]), num(r["return"], 4), num(r["agreement"], 4), num(r.get("end_loss", 0), 4)]) + r" \\")
A(r"\end{longtable}\end{footnotesize}")

A(r"\section{TremorDB trace records (complete, 55 records)}")
A(r"Public fixed-condition parkinsonian tremor traces; instrument units, no adaptive-policy data. Machine-readable record: \texttt{results/parkinson\_tremordb.json}.")
t = json.load(open(f"{R}/parkinson_tremordb.json"))
recs = t["records"]
A(r"\begin{footnotesize}\begin{longtable}{lllllrr}")
A(r"\caption{TremorDB records.}\\ \toprule")
A(r"File & Group & Subject & DBS & Med & $n$ & RMS \\ \midrule \endfirsthead")
A(r"\toprule File & Group & Subject & DBS & Med & $n$ & RMS \\ \midrule \endhead")
A(r"\midrule \multicolumn{7}{r}{continued}\\ \endfoot \bottomrule \endlastfoot")
for r in recs:
    A(" & ".join([esc(r["filename"]), esc(r["group"]), esc(r["subject"]), "on" if r["dbs_on"] else "off", "on" if r["med_on"] else "off", str(r["n_samples"]), num(r["rms_raw_instrument_units"], 3)]) + r" \\")
A(r"\end{longtable}\end{footnotesize}")

A(r"\section{Per-probe blood-expression audit (top 300 by $|t|$, train split)}")
A(r"Welch $t$ contrasting PD and control whole-blood expression over the pooled 438-sample GEO matrix (train split only; GSE99039/6613/72267/22491). Full matrix reproducible from \texttt{scripts/} plus the GEO series matrices.")
d2 = np.load("/tmp/pd_blood_prepped.npz", allow_pickle=True)
Xc, y, tr, probes = d2["Xc"], d2["y"], d2["tr"], d2["probes"]
Xt, yt = Xc[tr], y[tr]
pos, neg = Xt[yt == 1], Xt[yt == 0]
m1, m0 = pos.mean(0), neg.mean(0)
v1, v0 = pos.var(0, ddof=1), neg.var(0, ddof=1)
tt = (m1 - m0) / np.sqrt(v1 / len(pos) + v0 / len(neg) + 1e-12)
order = np.argsort(-np.abs(tt))[:1200]
A(f"Train split: {len(pos)} PD, {len(neg)} control, {Xt.shape[1]} probes.")
A(r"\begin{footnotesize}\begin{longtable}{lrrrr}")
A(r"\caption{Top 1200 probes by $|t|$ on the train split.}\\ \toprule")
A(r"Probe & mean PD & mean CTL & Welch $t$ & $|t|$ rank \\ \midrule \endfirsthead")
A(r"\toprule Probe & mean PD & mean CTL & Welch $t$ & $|t|$ rank \\ \midrule \endhead")
A(r"\midrule \multicolumn{5}{r}{continued}\\ \endfoot \bottomrule \endlastfoot")
for rank, i in enumerate(order, 1):
    A(" & ".join([esc(probes[i]), num(m1[i], 3), num(m0[i], 3), num(tt[i], 2), str(rank)]) + r" \\")
A(r"\end{longtable}\end{footnotesize}")

A(r"\section{GEO cohort composition}")
for f, lab in [("parkinson_geo_cohort.json", "Parkinson arm"), ("sepsis_geo_cohort.json", "Sepsis arm")]:
    c = json.load(open(f"{R}/{f}"))
    A(r"\subsection{" + lab + "}")
    A(r"Train $n=" + str(c.get("train_count")) + r"$, test $n=" + str(c.get("test_count")) + r"$, positive label: " + esc(c.get("positive_label")) + r".")
    cc = c.get("cohort_label_counts", {})
    A(r"\begin{center}\begin{tabular}{lrr} \toprule Cohort & positive & total \\ \midrule")
    for k, v in cc.items():
        if isinstance(v, dict):
            A(esc(k) + " & " + str(v.get("positive", v)) + " & " + str(v.get("total", "")) + r" \\")
        else:
            A(esc(k) + " & \multicolumn{2}{l}{" + esc(str(v)[:60]) + r"} \\")
    A(r"\bottomrule\end{tabular}\end{center}")

open(OUT, "w").write("\n".join(L) + "\n")
print("wrote", len(L), "lines")

# Sample manifest
A(r"\section{Pooled GEO sample manifest (complete, 438 samples)}")
A(r"Every sample in the pooled whole-blood matrix with its label and split assignment. GSM accessions resolve at NCBI GEO.")
A(r"\begin{footnotesize}\begin{longtable}{llll}")
A(r"\caption{Sample manifest.}\\ \toprule")
A(r"GSM & Label & Split & Index \\ \midrule \endfirsthead")
A(r"\toprule GSM & Label & Split & Index \\ \midrule \endhead")
A(r"\midrule \multicolumn{4}{r}{continued}\\ \endfoot \bottomrule \endlastfoot")
gsm, tr_, va, te = d2["gsm"], d2["tr"], d2["va"], d2["te"]
for i in range(len(gsm)):
    split = "train" if tr_[i] else ("val" if va[i] else "test")
    A(" & ".join([esc(gsm[i]), "PD" if y[i] == 1 else "CTL", split, str(i)]) + r" \\")
A(r"\end{longtable}\end{footnotesize}")

# move manifest before \end: rewrite whole file in order
body = "\n".join(L)
# reorder: manifest section should come after probe audit; L already appends at end, but cohort composition was appended earlier - acceptable order: formulas, distill runs, tremordb, probe audit, cohort composition, manifest
open(OUT, "w").write(body + "\n")
print("rewrote", len(L), "lines")

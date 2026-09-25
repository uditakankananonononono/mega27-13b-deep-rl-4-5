"""Descriptive, held-out probe of fetched GEO expression; no policy-effect inference."""
import csv,json,argparse
import numpy as np
from scipy.stats import mannwhitneyu
from scipy.special import expit

def run(path,out,seed=73):
    with open(path,newline='') as f: rows=list(csv.DictReader(f))
    labels=sorted(set(r['label'] for r in rows));assert len(labels)==2
    features=[k for k in rows[0] if k not in ('accession','label','cohort')]
    X=np.array([[float(r[k]) for k in features] for r in rows]);y=np.array([int(r['label']==labels[1]) for r in rows]);ids=[r['accession'] for r in rows]
    rng=np.random.default_rng(seed);ind=rng.permutation(len(rows));split=int(.7*len(ind));train=ind[:split];test=ind[split:]
    med=np.median(X[train],axis=0);scale=np.maximum(np.std(X[train],axis=0),1e-6)
    x=np.clip((X-med)/scale,-5,5)
    w=np.zeros(x.shape[1]);b=0.
    for _ in range(400):
        pred=expit(x[train]@w+b);e=pred-y[train];w-=.05*(x[train].T@e/len(train)+.1*w);b-=.05*e.mean()
    score=expit(x[test]@w+b); pos=score[y[test]==1];neg=score[y[test]==0]
    auc=float((sum((u>v)+.5*(u==v) for u in pos for v in neg))/(len(pos)*len(neg)))
    diffs=[{'probe':features[i],'train_median_difference':float(np.median(X[train][y[train]==1,i])-np.median(X[train][y[train]==0,i])),
       'train_mann_whitney_p_uncorrected':float(mannwhitneyu(X[train][y[train]==1,i],X[train][y[train]==0,i]).pvalue)} for i in range(len(features))]
    result={'accessions_used':len(ids),'features_used':len(features),'label_positive':labels[1],'train_n':len(train),'test_n':len(test),'heldout_auc':auc,
     'heldout_positive_n':len(pos),'heldout_negative_n':len(neg),'heldout_accessions': [ids[i] for i in test],
     'probe_screen_top5_uncorrected':sorted(diffs,key=lambda z:z['train_mann_whitney_p_uncorrected'])[:5],
     'limitations':'Random split from same GEO series may leak batch/subject/site; no external cohort validation, treatment arms, stimulation measures, or policy effect. Probe p-values uncorrected descriptive only.'}
    with open(out,'w') as f:json.dump(result,f,indent=2)
    print(path,'heldout AUC',auc,'train/test',len(train),len(test));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('out');a=p.parse_args();run(a.path,a.out)

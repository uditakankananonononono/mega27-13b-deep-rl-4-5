"""Day-one GSE54514 survival-label cross-validation, no treatment actions.
Repeated daily blood samples are excluded by day-one selection and subject ID check.
"""
import argparse,json,gzip,csv
from pathlib import Path
import numpy as np
from scipy.special import expit
from geo_independent import matrix,auc

def run(source,out):
 m,f,rows=matrix(source,[]);labels=m['!Sample_characteristics_ch1'][1];days=m['!Sample_characteristics_ch1'][2];groups=m['!Sample_characteristics_ch1'][3]
 ids=m['!Sample_geo_accession'][0]
 ii=[j for j,(v,d) in enumerate(zip(labels,days)) if v in ('disease status: sepsis survivor','disease status: sepsis nonsurvivor') and d.endswith('_D1')]
 subjects=[groups[j].split(': ')[1] for j in ii];assert len(ii)==len(set(subjects))==35
 chosen=[k for k,v in f.items() if np.all(np.isfinite(v[ii]))][:48];assert len(chosen)==48
 X=np.stack([f[k][ii] for k in chosen],axis=1);y=np.array([int(labels[j].endswith('nonsurvivor')) for j in ii]);assert y.sum()==9
 # Fixed 5 folds stratified by observed class, no tuning on held-out fold.
 rng=np.random.default_rng(54514);folds=np.empty(len(y),dtype=int)
 for cls in (0,1):
  inds=rng.permutation(np.flatnonzero(y==cls));folds[inds]=np.arange(len(inds))%5
 score=np.empty(len(y));fold_info=[]
 for fold in range(5):
  tr=folds!=fold;te=folds==fold
  med=np.median(X[tr],axis=0);scale=np.maximum(np.std(X[tr],axis=0),1e-6)
  xx=np.clip((X[tr]-med)/scale,-5,5);tx=np.clip((X[te]-med)/scale,-5,5);w=np.zeros(X.shape[1]);b=0.
  for _ in range(400):
   p=expit(xx@w+b);e=p-y[tr];w-=.05*(xx.T@e/tr.sum()+.1*w);b-=.05*e.mean()
  score[te]=expit(tx@w+b)
  fold_info.append({'fold':fold,'train_n':int(tr.sum()),'test_n':int(te.sum()),'test_deaths':int(y[te].sum())})
 r={'source':'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE54514','first_finite_probe_count':48,
  'day1_subject_count':35,'day1_deaths':9,'day1_survivors':26,'folds':fold_info,'out_of_fold_auc':auc(y,score),
  'evaluated_accessions':[ids[j] for j in ii],'evaluated_subject_group_ids':subjects,
  'test_scores':list(map(float,score)),'labels_death':list(map(int,y)),
  'source_matrix_index_for_day1_samples':ii,'source_metadata_ordering':'All 9 eligible nonsurvivor day-one columns precede all 26 eligible survivor day-one columns in original matrix. Cross-validation does not block batch effects aligned to that ordering.',
  'limitations':'Perfect within-series five-fold AUC on 35 people is suspicious given source matrix status-block ordering; potential batch/confounding leakage cannot be resolved here. Only 9 deaths, arbitrary first 48 probes, no external series compatible at probe-ID level, no treatment actions or ICU-Sepsis state mapping. NOT a verified mortality biomarker, clinical result, or policy metric.'}
 Path(out).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'n':35,'deaths':9,'auc':r['out_of_fold_auc'],'folds':fold_info},indent=2));return r
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--source',required=True);a.add_argument('--out',default='results/sepsis_second_series.json');x=a.parse_args();run(x.source,x.out)

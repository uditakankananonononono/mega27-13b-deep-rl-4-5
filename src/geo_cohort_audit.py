"""Cohort-aware exploratory split of GEO matrices; no therapeutic validation."""
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.special import expit

def audit(path,out):
 with open(path,newline='') as f:rows=list(csv.DictReader(f))
 features=[x for x in rows[0] if x not in ('accession','label','cohort')]
 x=np.array([[float(r[k]) for k in features] for r in rows]);cohort=np.array([r['cohort'] for r in rows]);labels=sorted({r['label'] for r in rows});y=np.array([r['label']==labels[1] for r in rows],int)
 counts={key:{lab:int(sum((cohort==key)&(np.array([r['label'] for r in rows])==lab))) for lab in labels} for key in sorted(set(cohort))}
 if 'discovery' in counts:train=cohort=='discovery';test=cohort=='validation'
 else:train=cohort=='TRAINING';test=cohort=='VALIDATION'
 if not (sum(train)>=30 and sum(test)>=20):raise ValueError(counts)
 mu=x[train].mean(0);sd=np.maximum(x[train].std(0),1e-6);z=np.clip((x-mu)/sd,-5,5)
 w=np.zeros(len(features));b=0
 for i in range(400):
  p=expit(z[train]@w+b);e=p-y[train];w-=.05*(z[train].T@e/sum(train)+.1*w);b-=.05*e.mean()
 score=expit(z[test]@w+b);p=score[y[test]==1];n=score[y[test]==0]
 auc=float(np.mean([(u>v)+.5*(u==v) for u in p for v in n]))
 result={'features_used':len(features),'accessions_used':len(rows),'train_count':int(sum(train)),'test_count':int(sum(test)),'cohort_label_counts':counts,'positive_label':labels[1],
  'cohort_heldout_auc':auc,'test_accessions':[r['accession'] for r,t in zip(rows,test) if t],
  'limitations':'Use original GEO cohort labels, not independent studies; only first 150 eligible GEO matrix samples and first 48 probes, no batch balancing, no policy effect, no clinical biomarker claim.'}
 Path(out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='test_accessions'},indent=2));return result
if __name__=='__main__':audit(sys.argv[1],sys.argv[2])

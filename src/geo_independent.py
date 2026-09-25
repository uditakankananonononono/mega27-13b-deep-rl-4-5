"""Independent GEO-series parsing and Parkinson blood-label transfer probe.
These cohorts have no stimulation treatment trajectories. Never claim policy validation.
"""
import csv,gzip,hashlib,json,argparse
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.special import expit

def matrix(path,labels):
    meta={};features={};rows=0
    with gzip.open(path,'rt') as f:
      for line in f:
        if line.startswith('!Sample_'):
          r=next(csv.reader([line],delimiter='\t'));meta.setdefault(r[0],[]).append([s.strip('"') for s in r[1:]])
        if line.startswith('!series_matrix_table_begin'):
          header=next(csv.reader([next(f)],delimiter='\t'))
          assert len(header)-1==len(meta['!Sample_geo_accession'][0])
          for line in f:
            if line.startswith('!series_matrix_table_end'):break
            r=next(csv.reader([line],delimiter='\t'));rows+=1
            if r[0].strip('"') in labels or (not labels and len(features)<48):features[r[0].strip('"')]=np.array([float(v) if v not in ('null','NA','') else np.nan for v in r[1:]],dtype=float)
          break
    return meta,features,rows

def auc(y,z):
 p=z[y==1];n=z[y==0];return float(sum((v>u)+.5*(v==u) for v in p for u in n)/(len(p)*len(n)))

def run(reference,source6613,source54514,out):
 with open(reference,newline='') as f:base=list(csv.DictReader(f))
 features=[k for k in base[0] if k not in ('accession','label','cohort')]
 m,f,n=matrix(source6613,features)
 ids=m['!Sample_geo_accession'][0];status=m['!Sample_characteristics_ch1'][0];keep=[i for i,x in enumerate(status) if x in ("Parkinson's disease",'healthy control')]
 common=[k for k in features if k in f and np.all(np.isfinite(f[k][keep]))]
 assert len(common)==13 and len(set(ids))==len(ids)
 X=np.array([[float(r[k]) for k in common] for r in base]);y=np.array([int(r['label']=='IPD') for r in base]);ext=np.array([[f[k][i] for k in common] for i in keep]);ey=np.array([int(status[i]=="Parkinson's disease") for i in keep]);assert len(ey)==72
 # Affymetrix reference log2-scale; external matrix has unlogged intensities.
 # Transform all external intensities without using labels, then scale by training cohort only.
 assert np.min(ext)>0
 ext=np.log2(ext);med=np.median(X,axis=0);scale=np.maximum(np.std(X,axis=0),1e-6)
 x=np.clip((X-med)/scale,-5,5);xe=np.clip((ext-med)/scale,-5,5)
 w=np.zeros(len(common));b=0.
 for _ in range(400):
    p=expit(x@w+b);e=p-y;w-=.05*(x.T@e/len(y)+.1*w);b-=.05*e.mean()
 score=expit(xe@w+b)
 seps,sepf,ns=matrix(source54514,[]);sids=seps['!Sample_geo_accession'][0];groups=seps['!Sample_characteristics_ch1'][3];labels54514=seps['!Sample_characteristics_ch1'][1]
 day=seps['!Sample_characteristics_ch1'][2]
 groupids=[v.partition(': ')[2] for v in groups]
 day1=[i for i,(c,d) in enumerate(zip(labels54514,day)) if c in ('disease status: sepsis survivor','disease status: sepsis nonsurvivor') and d.endswith('_D1')]
 selected_sepsis=np.array([v[day1] for v in sepf.values()],dtype=float).T
 finite_sepsis=np.all(np.isfinite(selected_sepsis),axis=0)
 selected_sepsis=selected_sepsis[:,finite_sepsis]
 assert selected_sepsis.shape[0]==35 and selected_sepsis.shape[1]>=40
 survivor=np.array([labels54514[i]=='disease status: sepsis survivor' for i in day1])
 sepsis_probe_deltas=np.median(selected_sepsis[survivor],axis=0)-np.median(selected_sepsis[~survivor],axis=0)
 result={'source':{'parkinson':'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE6613','sepsis':'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE54514'},
    'parkinson':{'source_sha256':hashlib.sha256(Path(source6613).read_bytes()).hexdigest(),'matrix_probe_rows':n,'source_gsm_count':len(ids),'independent_eligible_pd_healthy':len(ey),'external_accessions':[ids[i] for i in keep], 'label_counts':dict(Counter(status[i] for i in keep)),'shared_probe_count':len(common),'shared_probe_ids':common,'reference_gse99039_train_count':len(y), 'external_auc':auc(ey,score),'external_mean_score_pd':float(np.mean(score[ey==1])),'external_mean_score_healthy':float(np.mean(score[ey==0])),'processing':'External positive intensity log2, all probes standardized using GSE99039 training moments only; no external labels used in scaling or training. Convenience first 48 reference probes, ridge-like logistic fit, no model selection.'},
    'sepsis':{'source_sha256':hashlib.sha256(Path(source54514).read_bytes()).hexdigest(),'matrix_probe_rows':ns,'source_gsm_count':len(sids),'measured_probe_rows':ns,'first_finite_numeric_probe_count':int(selected_sepsis.shape[1]),'first_probe_survivor_minus_nonsurvivor_median':float(sepsis_probe_deltas[0]),'first_probe_id':list(sepf)[0],'all_status_counts':dict(Counter(labels54514)),'unique_subject_group_ids':len(set(groupids)),'sepsis_day1_count':len(day1),'day1_unique_group_ids':len(set(groupids[i] for i in day1)),'day1_labels':dict(Counter(labels54514[i] for i in day1)),'day1_accessions':[sids[i] for i in day1], 'repeat_sample_caveat':'163 GSM columns contain serial repeated blood samples, not 163 unique people; day-1 sepsis patient selection isolates unique group IDs but has only 35 people.'},
    'limitation':'Independent GSE6613 external blood diagnosis labels test only cross-study marker transfer, not DBS feedback or treatment response. Platform, era, preprocessing and case mix differ. GSE54514 serial blood is no sepsis action trajectory. Neither source validates an ICU or DBS policy.'}
 Path(out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'parkinson_auc':result['parkinson']['external_auc'],'shared_probes':len(common),'pd_test_counts':result['parkinson']['label_counts'],'sepsis_day1':len(day1),'sepsis_day1_labels':result['sepsis']['day1_labels']},indent=2));return result
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--parkinson-reference',required=True);a.add_argument('--gse6613',required=True);a.add_argument('--gse54514',required=True);a.add_argument('--out',default='results/geo_independent.json');x=a.parse_args();run(x.parkinson_reference,x.gse6613,x.gse54514,x.out)

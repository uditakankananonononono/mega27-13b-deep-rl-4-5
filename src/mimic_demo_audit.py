"""Audit publicly released MIMIC-III *demo* action rows for sepsis context.
Not a full MIMIC dataset or usable off-policy evaluation cohort.
"""
import argparse,hashlib,json
from pathlib import Path
import pandas as pd

def run(raw,out):
 raw=Path(raw); names=['INPUTEVENTS_MV.csv','INPUTEVENTS_CV.csv','D_ITEMS.csv','ICUSTAYS.csv','ADMISSIONS.csv','PATIENTS.csv','DIAGNOSES_ICD.csv']
 for n in names:
  if not (raw/n).exists():raise FileNotFoundError(raw/n)
 d={n:pd.read_csv(raw/n,dtype={'icd9_code':str} if n=='DIAGNOSES_ICD.csv' else None) for n in names}
 codes=d['DIAGNOSES_ICD.csv']['icd9_code'].fillna('');selector=codes.str.startswith(('038','99591','99592','78552'))
 suspected=set(d['DIAGNOSES_ICD.csv'].loc[selector,'hadm_id'].astype(int));items=d['D_ITEMS.csv'][['itemid','label']]
 events=pd.concat([d['INPUTEVENTS_MV.csv'],d['INPUTEVENTS_CV.csv']],ignore_index=True).merge(items,on='itemid',how='left')
 iv=events['label'].fillna('').str.contains(r'Norepinephrine|Vasopressin|\.9% Normal Saline|Lactated Ringers',case=False,regex=True)
 candidates=events[events['hadm_id'].isin(suspected)&iv].copy()
 categories={'norepinephrine':r'Norepinephrine','vasopressin':r'Vasopressin','isotonic_saline':r'\.9% Normal Saline','lactated_ringers':r'Lactated Ringers'}
 counts={k:{'input_rows':int(candidates['label'].fillna('').str.contains(v,case=False,regex=True).sum()),'distinct_stays':int(candidates[candidates['label'].fillna('').str.contains(v,case=False,regex=True)]['icustay_id'].nunique())} for k,v in categories.items()}
 chosen=candidates[['subject_id','hadm_id','icustay_id','itemid','label','amount','amountuom','rate','rateuom']].drop_duplicates();dest=Path(out).with_suffix('.csv');chosen.to_csv(dest,index=False)
 r={'source':'https://physionet.org/content/mimiciii-demo/1.4/','license_scope':'Public 100-patient MIMIC-III demo; source license should be checked before redistribution; only aggregate JSON committed, row-level derived CSV retained locally',
    'source_sha256':{n:hashlib.sha256((raw/n).read_bytes()).hexdigest() for n in names},
    'source_rows':{n:len(v) for n,v in d.items()},'sample_subjects':d['PATIENTS.csv']['subject_id'].nunique(),
    'suspected_sepsis_icd9_admissions':len(suspected),'icd9_codes':'038*, 99591, 99592, 78552; rough administrative proxy NOT adjudicated sepsis',
    'selected_input_rows':len(candidates),'selected_distinct_stays':int(candidates['icustay_id'].nunique()),'category_counts':counts,
    'derived_rows':len(chosen),'derived_table_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
    'caveat':'Demo deliberately contains only 100 patients and may not be representative. These event rows do not form a cleaned sepsis policy benchmark, cannot be matched to ICU-Sepsis model state IDs, and are NOT added to GEO accession count or an efficacy claim.'}
 Path(out).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));return r
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--raw',required=True);a.add_argument('--out',default='results/sepsis_mimic_demo.json');x=a.parse_args();run(x.raw,x.out)

"""Audit publicly released static sepsis cohort; not a sequential RL policy dataset."""
import argparse,hashlib,json
from pathlib import Path
import pandas as pd
import numpy as np

def audit(path,out):
 b=Path(path).read_bytes();d=pd.read_csv(path,sep='\t',low_memory=False)
 assert d.shape[1]==169 and len(d)==13717
 assert set(d['HOSPITALDISCHARGEYEAR'].dropna())=={2014,2015}
 use=['AGE','SOFA','WBC','TEMPERATURE','HEARTRATE','MEANBP','GCS_SCORE','CREATININE','URINE']
 result={'source':'https://datadryad.org/dataset/doi:10.5061/dryad.hmgqnk9wb','source_tsv_sha256':hashlib.sha256(b).hexdigest(),'rows':len(d),'columns':len(d.columns),
 'distinct_hospital_ids':int(d.HOSPITALID.nunique()),'year_counts':{str(k):int(v) for k,v in d.HOSPITALDISCHARGEYEAR.value_counts().sort_index().items()},
 'unit_28day_mortality_n':int(d.DEATHD28_UNIT.sum()),'unit_28day_mortality_rate':float(d.DEATHD28_UNIT.mean()),
 'vasopressor_24h_field':'MEDS','vasopressor_24h_value_counts':{str(k):int(v) for k,v in d.MEDS.value_counts(dropna=False).items()},
 'source_sepsis_type_code_counts':{str(k):int(v) for k,v in d.SEPSIS_ADMISSION.value_counts(dropna=False).sort_index().items()},
 'missing_fraction_early_measures':{k:float(d[k].isna().mean()) for k in use},
 'example_age_mean':float(d.AGE.mean()),'example_sofa_median':float(d.SOFA.median()),
 'source_rows_are_independent_episode_trajectories':False,'policy_validation':'no',
 'limitations':'13,717 rows from a static first-day risk cohort, not 13,717 separate accession IDs or time-indexed ICU action-response trajectories. MEDS summarizes first-24h vasopressor use, not a dose/time policy; no per-step fluids, timed state/action/outcome, behavior propensities, or known mapping into ICU-Sepsis 25 interventions. Mortality and discharge variables must not be used as baseline model features. This source cannot validate treatment-policy effects.'}
 Path(out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--tsv',required=True);a.add_argument('--out',default='results/sepsis_dryad_eicu.json');x=a.parse_args();audit(x.tsv,x.out)

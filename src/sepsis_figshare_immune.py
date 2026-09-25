"""Audit public derived eICU/MIMIC-IV sepsis immune trajectories; no action-policy claim."""
import argparse,hashlib,json
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

def audit(eicu,mimic,out):
 info={}
 for cohort,path,idcol,outcome in [('eicu',eicu,'patientunitstayid','deathd28_unit'),('mimic_iv',mimic,'stay_id','death_within_28_days')]:
  df=pd.read_csv(path,low_memory=False)
  assert len(df)==df[idcol].nunique()
  complete=[int(df[f'nlr{i}'].notna().sum()) for i in range(1,8)]
  valid=df[['nlr1',outcome]].dropna();case=valid.loc[valid[outcome]==1,'nlr1'];control=valid.loc[valid[outcome]==0,'nlr1']
  # Descriptive probability of rank ordering, not a prospective predictor; one baseline feature.
  stat=mannwhitneyu(case,control,alternative='two-sided');auc=float(stat.statistic/(len(case)*len(control)))
  info[cohort]={'source_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'rows':len(df),'columns':len(df.columns),'id_field':idcol,'unique_ids':int(df[idcol].nunique()),
   'outcome_field':outcome,'28day_death_count':int(df[outcome].sum()),'nlr_measurements_available_day1_to_day7':complete,
   'baseline_nlr_case_median':float(case.median()),'baseline_nlr_control_median':float(control.median()),
   'baseline_nlr_rank_auc_descriptive':auc,'baseline_nlr_mann_whitney_p_exploratory':float(stat.pvalue),
   'vasopressor_summary_fields':[x for x in df if any(z in x.lower() for z in ['vasop','norepi','fluid'])]}
 d={'source':'https://figshare.com/articles/dataset/_b_Derived_ICU_Sepsis_Dataset_from_eICU_and_MIMIC-IV_Immune_Cell_Trajectories_SOFA_and_Outcomes_b_/31146940',
 'source_api':'https://api.figshare.com/v2/articles/31146940','cohorts':info,
 'purpose':'Verify measured seven-day immune-trajectory availability and a single descriptive baseline rank statistic within each derived sepsis cohort; not cross-source policy validation.',
 'limitation':'eICU deathd28_unit and MIMIC-IV death_within_28_days are not identical endpoints; rank AUCs are same-cohort retrospective descriptions, with no training/test split and no adjustment for case mix or missingness. Later-day NLR availability depends on survival/discharge/measurement and must not be treated as random. Cohort sizes are stays, not new GEO accessions. Source contains no timestamped vasopressor plus IV-fluid action sequence or matching ICU-Sepsis 25-action states. No treatment-policy training, off-policy evaluation, effect or clinical recommendation.'}
 Path(out).write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2));return d
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--eicu',required=True);a.add_argument('--mimic',required=True);a.add_argument('--out',default='results/sepsis_figshare_immune.json');x=a.parse_args();audit(x.eicu,x.mimic,x.out)

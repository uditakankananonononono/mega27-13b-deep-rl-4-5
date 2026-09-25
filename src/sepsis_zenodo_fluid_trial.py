"""Aggregate audit of published sepsis randomized fluid-guidance trial data; no RL policy inference."""
import argparse,hashlib,json
from pathlib import Path
import pandas as pd

def run(source,out):
 reader=pd.io.stata.StataReader(source);labels=reader.variable_labels();d=reader.read(convert_categoricals=False)
 assert labels['intervention']=='0=static CVP, 1=dynamic IVC'
 assert labels['mortality']=='0=no 30-day mortality, 1=30-day mortality'
 assert len(d)==123 and d.shape[1]==58 and d['intervention'].value_counts().to_dict()=={0:62,1:61}
 outgroups={}
 for arm,code in [('static_cvp',0),('dynamic_ivc',1)]:
  x=d.loc[d['intervention']==code];mort=x['mortality'].dropna()
  outgroups[arm]={'records':len(x),'mortality_observed':len(mort),'mortality_missing':int(x['mortality'].isna().sum()),
   '30day_deaths':int(mort.sum()),'30day_death_fraction_observed':float(mort.mean()),
   'fluid_72h_nonmissing':int(x['total_ivfluid_mL_in_72_h'].notna().sum()),'fluid_72h_median_ml':float(x['total_ivfluid_mL_in_72_h'].median()),
   'norepinephrine_use_nonmissing':int(x['receive_NE'].notna().sum()),'norepinephrine_use_yes':int(x['receive_NE'].sum()),
   'accumulated_ne_nonmissing':int(x['accumulated_NE_mg'].notna().sum()),'accumulated_ne_median_mg_observed':float(x['accumulated_NE_mg'].median()),
   'initial_sofa_nonmissing':int(x['initial_SOFA'].notna().sum()),'initial_sofa_median':float(x['initial_SOFA'].median())}
 r={'source':'https://zenodo.org/records/10579408','source_doi':'10.5281/zenodo.10579408','source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
 'associated_paper':'https://pmc.ncbi.nlm.nih.gov/articles/PMC11342037/','associated_paper_doi':'10.12688/f1000research.147875.2',
 'license':'CC BY 4.0 per Zenodo record','records':len(d),'columns':len(d.columns),'arms':outgroups,
 'trial_design_from_paper':'Single-blind, stratified by APACHE II <25 versus >=25, block randomization 2 or 4 at Thammasat University Hospital, August 2016 to April 2020. Paper enrolled 124 (62/62); one dynamic withdrawal yields 123 released records; one static record lacks 30-day survival.',
 'interpretation':'Aggregate descriptive readback of a randomized comparison of guidance strategies, not a newly estimated causal effect or reproduced adjusted trial analysis. Published primary 30-day mortality comparison p=0.196 and relative risk 0.8 (95% CI 0.5-1.2), p=0.201. These arm labels are strategy assignments, not sequential fluid/vasopressor decisions.',
 'limitation':'No dose timing, repeated decision state, action propensity or 4-hour fluid/vasopressor action history in this 123-row patient summary. The published mortality difference is statistically uncertain; no efficacy or individualized RL-policy inference. Per-patient rows are not committed.'}
 Path(out).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));return r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',default='results/sepsis_zenodo_fluid_trial.json');a=p.parse_args();run(a.source,a.out)

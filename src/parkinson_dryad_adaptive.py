"""Audit public figure-level human adaptive-DBS traces, never claim RL evaluation."""
import argparse,hashlib,json,zipfile,io
from pathlib import Path
import numpy as np
import pandas as pd

def summarize(zip_path,out):
 digest=hashlib.sha256(Path(zip_path).read_bytes()).hexdigest()
 with zipfile.ZipFile(zip_path) as z:
  names=z.namelist();csvs=[n for n in names if n.lower().endswith('.csv')]
  def read(figure,file):
   path=f'move_adbs_min_data/Fig{figure}/{file}.csv';return pd.read_csv(io.BytesIO(z.read(path))),path
  rows=[]
  for hand in ('left','right'):
   p,pp=read(5,f'fig5b_{hand}_accel_pred');t,tp=read(5,f'fig5b_{hand}_accel_true');n,np_=read(2,f'fig2b_neural_{hand}')
   assert all(p.time.diff().dropna()>0) and all(t.time.diff().dropna()>0)
   assert {'time','acceleration','state','stim'}<=set(p) and {'time','acceleration','state'}<=set(t)
   assert {'time','beta','gamma'}<=set(n)
   assert set(p['stim'].round(3).unique()) <= {1.6,1.7,1.8,1.9,2.0,2.1,2.2}
   merged=pd.merge_asof(p.sort_values('time'),t[['time','state']].sort_values('time'),on='time',direction='nearest',tolerance=.075,suffixes=('_pred','_true'))
   valid=merged.dropna(subset=['state_true'])
   rows.append({'hand':hand,'files':[pp,tp,np_],'action_rows':len(p),'true_state_rows':len(t),'neural_band_rows_separate_figure':len(n),
    'action_start_s':float(p.time.min()),'action_end_s':float(p.time.max()),'neural_start_s':float(n.time.min()),'neural_end_s':float(n.time.max()),
    'predicted_state_changes':int(np.sum(np.diff(p.state.to_numpy())!=0)),'action_changes_including_ramps':int(np.sum(np.diff(p.stim.to_numpy())!=0)),
    'fraction_at_low_1_6mA':float(np.mean(p.stim==1.6)),'fraction_at_high_2_2mA':float(np.mean(p.stim==2.2)),
    'intermediate_ramp_rows':int(np.sum(~p.stim.isin([1.6,2.2]))),'mean_amplitude_mA_example':float(np.mean(p.stim)),
    'near_match_true_state_rows':len(valid),'near_match_predicted_vs_true_state_agreement':float(np.mean(valid.state_pred==valid.state_true)),
    'neural_time_overlap_with_action_example_seconds':float(max(0,min(p.time.max(),n.time.max())-max(p.time.min(),n.time.min()))),
    'source_columns':{'prediction':p.columns.tolist(),'true':t.columns.tolist(),'neural':n.columns.tolist()}})
 result={'source':'https://datadryad.org/dataset/doi:10.5061/dryad.4xgxd25hw','api_version':'https://datadryad.org/api/v2/versions/359256',
  'zip_sha256':digest,'csv_file_count':len(csvs),'dataset_people_as_readme':1,'hands':rows,
  'interpretation':'Fig5b example time series shows actual adaptive stimulation amplitude and movement prediction, paired with separate true movement trace. Fig2b example beta/gamma traces are from a different time interval and figure; cannot join to Fig5 action histories. Rows are time points, not independent patients or datasets.',
  'policy_validation':'no','limitation':'One participant and figure-level excerpts; Fig5 actions respond to predicted movement, but no synchronized implant LFP state or rewards in these rows, no logged behavior propensities, no counterfactual adaptive action outcomes, and no independent held-out patient. Do not train/evaluate an RL treatment policy or infer clinical benefit from this descriptive trace.'}
 Path(out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--zip',required=True);a.add_argument('--out',default='results/parkinson_dryad_adaptive.json');x=a.parse_args();summarize(x.zip,x.out)

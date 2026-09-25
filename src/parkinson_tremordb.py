"""Read public Parkinson tremor time series under fixed DBS/medication conditions.
Do not treat condition-level recordings as adaptive actions or causal estimates.
"""
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np
from scipy.signal import welch

def run(root,out):
 import urllib.request
 root=Path(root);root.mkdir(parents=True,exist_ok=True)
 dirs=('renh','renl','refh','refl','ronh','ronl','rofh','rofl');records=[]
 for d in dirs:
  url=f'https://physionet.org/files/tremordb/1.0.0/{d}/{d}.hea';head=root/f'{d}.hea'
  if not head.exists():urllib.request.urlretrieve(url,head)
  lines=head.read_text().splitlines();assert re.match(r'^'+d+r' \d+ 100$',lines[0]);files=[]
  for line in lines[1:]:
   if not line.strip() or line.startswith('#'):break
   files.append(line.split()[0])
  if d=='rofh':
   assert 'v5rof.rit' in files # upstream header typo, directory lists v5rof.let
   files[files.index('v5rof.rit')]='v5rof.let'
  assert int(lines[0].split()[1])==len(files)
  for filename in files:
   url=f'https://physionet.org/files/tremordb/1.0.0/{d}/{filename}';path=root/filename
   if not path.exists():urllib.request.urlretrieve(url,path)
   a=np.loadtxt(path);assert a.ndim==1 and len(a)>1000 and np.all(np.isfinite(a))
   f,p=welch(a,fs=100,nperseg=min(2048,len(a)))
   power=float(np.trapezoid(p[(f>=4)&(f<=6)],f[(f>=4)&(f<=6)]))
   records.append({'filename':filename,'group':d,'subject':filename.split('r')[0], 'dbs_on':d.startswith('re'),
    'med_on':d.endswith('n'+'h') or d.endswith('n'+'l'),'n_samples':len(a),
    'rms_raw_instrument_units':float(np.sqrt(np.mean(a*a))),'tremor_4_6hz_welch_power_raw_instrument_units':power,
    'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 # Group suffix distinguishes high/low tremor cohort; first 3 letters encode DBS and medication.
 for r in records:
  r['med_on']=r['group'][2]=='n'
 pairs=[]
 for med in (True,False):
  for subject in sorted({r['subject'] for r in records}):
   on=[r for r in records if r['subject']==subject and r['med_on']==med and r['dbs_on']]
   off=[r for r in records if r['subject']==subject and r['med_on']==med and not r['dbs_on']]
   if len(on)==1 and len(off)==1:
    a,b=on[0],off[0]
    pairs.append({'subject':subject,'med_on':med,'on_file':a['filename'],'off_file':b['filename'],
       'on_off_rms_ratio':a['rms_raw_instrument_units']/b['rms_raw_instrument_units'] if b['rms_raw_instrument_units'] else None,
       'on_off_4_6hz_power_ratio':a['tremor_4_6hz_welch_power_raw_instrument_units']/b['tremor_4_6hz_welch_power_raw_instrument_units'] if b['tremor_4_6hz_welch_power_raw_instrument_units'] else None})
 d={'source':'https://physionet.org/content/tremordb/1.0.0/','records':records,'paired_fixed_condition_records':pairs,
    'record_count':len(records),'subject_count':len({r['subject'] for r in records}),'pair_count':len(pairs),
    'on_off_power_ratio_median':float(np.median([r['on_off_4_6hz_power_ratio'] for r in pairs if r['on_off_4_6hz_power_ratio'] is not None])),
    'paired_by_medication':{str(med):{'pair_count':sum(r['med_on']==med for r in pairs),'median_on_off_power_ratio':float(np.median([r['on_off_4_6hz_power_ratio'] for r in pairs if r['med_on']==med]))} for med in (True,False)},
    'limitation':'Fixed-condition finger tremor velocity, heterogeneous instrument ranges, medication conditions and within-subject repeated recordings. No LFP beta, step-level adaptive stimulation actions, randomized counterfactual, or clinical policy-value estimate. RMS/power in raw instrument units; paired ratios descriptive only.'}
 Path(out).write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({'record_count':len(records),'subject_count':d['subject_count'],'pair_count':len(pairs),'median_on_off_4_6_power':d['on_off_power_ratio_median']},indent=2));return d
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--root',default='/tmp/13b-tremor');a.add_argument('--out',default='results/parkinson_tremordb.json');x=a.parse_args();run(x.root,x.out)

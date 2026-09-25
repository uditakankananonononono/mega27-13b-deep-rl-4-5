"""Audit authentic hospital sepsis-process event log, not ICU action-policy trajectories."""
import argparse,collections,csv,hashlib,io,json,statistics,zipfile
from pathlib import Path

def run(source,out):
 with zipfile.ZipFile(source) as z:
  assert sorted(n for n in z.namelist() if not n.startswith('__MACOSX'))==['real.log','x10.log','x100.log','x1000.log']
  rows=list(csv.reader(io.StringIO(z.read('real.log').decode())))
 assert all(len(r)==3 and r[2].isdigit() for r in rows)
 by_case=collections.defaultdict(list)
 for case,activity,ts in rows:by_case[case].append((int(ts),activity))
 activities=collections.Counter(r[1] for r in rows)
 durations=[];triage_to_abx=[];patterns=collections.Counter();releases=collections.Counter()
 for records in by_case.values():
  records=sorted(records);patterns[' > '.join(a for _,a in records)]+=1
  starts=[t for t,a in records if a=='ER Sepsis Triage'];abx=[t for t,a in records if a=='IV Antibiotics'];end=[(t,a) for t,a in records if a.startswith('Release ')]
  if starts and abx and min(abx)>=min(starts):triage_to_abx.append((min(abx)-min(starts))/3600)
  if starts and end and end[-1][0]>=min(starts):durations.append((end[-1][0]-min(starts))/3600);releases[end[-1][1]]+=1
 d={'source':'https://zenodo.org/records/3989590','source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),'authentic_log_sha256':hashlib.sha256(zipfile.ZipFile(source).read('real.log')).hexdigest(),
    'real_events':len(rows),'real_case_ids':len(by_case),'activities':dict(activities),'event_fields':['case_id','activity_label','unix_timestamp'],
    'top_patterns':patterns.most_common(5),'released_cases':sum(releases.values()),'release_labels':dict(releases),
    'triage_to_antibiotic_pairs':len(triage_to_abx),'triage_to_antibiotic_median_hours':statistics.median(triage_to_abx),
    'triage_to_release_pairs':len(durations),'triage_to_release_median_hours':statistics.median(durations),
    'excluded_synthetic_log_names':['x10.log','x100.log','x1000.log'],
    'limitation':'Original event log is a simplified hospital workflow excerpt, not an ICU time-series; its Release A-E labels are not mortality or health outcomes. IV Antibiotics is an activity event without dose or fluid/vasopressor actions, physiologic state, transition or intervention propensity. The three amplified logs are synthetic and excluded. Case IDs are neither independent GEO GSM accessions nor validated treatment trajectories; durations are descriptive and may include nonclinical delays. No off-policy RL or clinical efficacy estimate.'}
 Path(out).write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--source',required=True);a.add_argument('--out',default='results/sepsis_zenodo_workflow.json');x=a.parse_args();run(x.source,x.out)

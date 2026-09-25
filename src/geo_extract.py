"""Stream GEO series matrix; retain identifiers, labels, and measured expression features.
Only rows with actual numeric measurements and specified cohort metadata count.
"""
import argparse, csv, gzip, hashlib, json, math
from pathlib import Path
import numpy as np

def extract(path, accession, dest, limit=150, features=48):
    meta = {}; rows=[]; header=None
    with gzip.open(path,'rt',errors='replace') as handle:
        for line in handle:
            if line.startswith('!Sample_geo_accession'):
                meta['accession']=next(csv.reader([line],delimiter='\t'))[1:]
            elif line.startswith('!Sample_title'):
                meta['title']=next(csv.reader([line],delimiter='\t'))[1:]
            elif line.startswith('!Sample_characteristics_ch1'):
                vals=next(csv.reader([line],delimiter='\t'))[1:]
                for i,v in enumerate(vals):
                    if ': ' in v:
                        key, value=v.split(': ',1)
                        meta.setdefault(key,['']*len(vals))[i]=value
            elif line.startswith('!series_matrix_table_begin') or line.startswith('!Series_matrix_table_begin'):
                header=next(csv.reader([next(handle)],delimiter='\t'))
                break
        if header is None: raise ValueError('missing matrix')
        ids=meta['accession']; assert header[1:]==ids, 'metadata/matrix sample order mismatch'
        for line in handle:
            if line.startswith('!series_matrix_table_end') or line.startswith('!Series_matrix_table_end'): break
            vals=next(csv.reader([line],delimiter='\t'))
            if len(vals)!=len(header): continue
            try: row=[float(v) for v in vals[1:]]
            except ValueError: continue
            if all(math.isfinite(x) for x in row): rows.append((vals[0],row))
            if len(rows)==features: break
    if accession=='GSE65682':
        candidate=[i for i,a in enumerate(ids) if meta.get('mortality_event_28days',['']*len(ids))[i] in ('0','1')]
        label_key='mortality_event_28days'; meaning='observed 28-day mortality; not treatment assignment/outcome under policy'
    elif accession=='GSE99039':
        candidate=[i for i,a in enumerate(ids) if meta.get('disease label',['']*len(ids))[i] in ('IPD','CONTROL')]
        label_key='disease label'; meaning='idiopathic Parkinson disease versus control; not stimulation outcome'
    else: raise ValueError(accession)
    chosen=candidate[:limit]
    if len(chosen)<120 or len(rows)<features: raise ValueError(f'underfilled: samples={len(chosen)}, features={len(rows)}')
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    with (dest/'accession_features.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['accession','label','cohort']+[x[0] for x in rows])
        for i in chosen:
            cohort=(meta.get('learning set',meta.get('endotype_cohort',['']*len(ids))))[i]
            writer.writerow([ids[i],meta[label_key][i],cohort]+[f'{r[1][i]:.8g}' for r in rows])
    digest=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    csvdigest=hashlib.sha256((dest/'accession_features.csv').read_bytes()).hexdigest()
    stats={'series':accession,'url':f'https://ftp.ncbi.nlm.nih.gov/geo/series/{accession[:5]}nnn/{accession}/matrix/{accession}_series_matrix.txt.gz',
           'source_sha256':digest,'source_bytes':Path(path).stat().st_size,'derived_sha256':csvdigest,
           'used_accessions':len(chosen),'used_numeric_features':len(rows),'label':label_key,'interpretation':meaning,
           'labels':{str(k):sum(meta[label_key][i]==k for i in chosen) for k in set(meta[label_key][i] for i in chosen)},
           'feature_accession_alignment_verified':True}
    (dest/'provenance.json').write_text(json.dumps(stats,indent=2)+'\n')
    return stats
if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('path');a.add_argument('accession');a.add_argument('dest');a.add_argument('--limit',type=int,default=150);args=a.parse_args()
    print(json.dumps(extract(args.path,args.accession,args.dest,args.limit),indent=2))

import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from scientific_component_dedup import compute
def test_dedup_replay():
 j=compute();assert j==json.loads((ROOT/'evidence/scientific_component_dedup.json').read_text());assert j['distinct_mixed_components']==17;assert j['category_counts']=={'library':7,'data_archive_or_product':8,'simulation_engine':2}

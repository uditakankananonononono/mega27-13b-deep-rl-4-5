import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sepsis_current_page_audit import audit
def test_current_pdf_identity():
 j=audit(ROOT/'papers/sepsis.pdf');assert j==json.loads((ROOT/'results/sepsis_current_page_audit.json').read_text());assert j['physical_pages']==23;assert j['certified'] is False

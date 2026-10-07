"""Layout candidates, not scientific-content certification. Uses actual PDF text."""
from pathlib import Path
import subprocess,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def audit(pdf):
 text=subprocess.check_output(['pdftotext','-layout',str(pdf),'-'],text=True);pages=text.split('\f')
 if not pages[-1].strip():pages.pop()
 return {'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'physical_pages':len(pages),'full_body_candidates':len(pages)-1,'mixed_frontmatter_body_pages':1,'conservative_full_body_shortfall_to50':max(0,51-len(pages)),'certified':False,'classification_basis':'Existing manuscript body-only structure checked in TeX;page1mixedtitleabstractbody. Ordinary headings are not standalone headingpages. This is not certification of novelty,substance or journal readiness.','pages':[{'physical_page':i+1,'classification':'mixed_frontmatter_body' if i==0 else 'body_candidate','words':len(p.split()),'first_lines':[x.strip() for x in p.splitlines() if x.strip()][:2]} for i,p in enumerate(pages)]}
if __name__=='__main__':
 (ROOT/'results/sepsis_current_page_audit.json').write_text(json.dumps(audit(ROOT/'papers/sepsis.pdf'),indent=2)+'\n')

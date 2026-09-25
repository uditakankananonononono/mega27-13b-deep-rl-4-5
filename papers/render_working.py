"""Render work-in-progress TeX manuscript as readable TNR PDF when fontspec is absent.
Math equations are plotted through Matplotlib mathtext; PDF is clearly labeled WORKING.
"""
import re,sys
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT,TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import letter
from xml.sax.saxutils import escape
from io import BytesIO
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
for name,file in [('TNR','Times'),('TNR-Bold','Timesbd'),('TNR-Italic','Timesi')]:pdfmetrics.registerFont(TTFont(name,'/tmp/13b-fonts/'+file+'.TTF'))
pdfmetrics.registerFontFamily('TNR',normal='TNR',bold='TNR-Bold',italic='TNR-Italic')
base=ParagraphStyle('body',fontName='TNR',fontSize=11,leading=13.1,spaceAfter=4)
heading=ParagraphStyle('heading',parent=base,fontName='TNR-Bold',fontSize=15,leading=19,spaceBefore=8,keepWithNext=True)
title=ParagraphStyle('title',parent=heading,fontSize=18,leading=22,alignment=TA_CENTER,spaceAfter=17)
small=ParagraphStyle('small',parent=base,fontSize=9,leading=12)
formula=ParagraphStyle('formula',parent=base,fontSize=9,leading=12,leftIndent=12,rightIndent=8,spaceAfter=8)
def clean(t):
 t=t.replace('\\caution','Nonclinical study. No result in this manuscript supports patient treatment decisions.')
 t=re.sub(r'\\url\{([^}]*)\}',r'\1',t)
 t=re.sub(r'\\emph\{([^}]*)\}',r'\1',t)
 t=re.sub(r'\\textbf\{([^}]*)\}',r'\1',t)
 t=re.sub(r'\\texttt\{([^}]*)\}',r'\1',t)
 t=re.sub(r'\\section\{([^}]*)\}',r'\n\n## \1\n\n',t)
 t=re.sub(r'\\begin\{(?:center|enumerate|tabular)\}(?:\{[^}]*\})?','',t)
 t=re.sub(r'\\end\{(?:center|enumerate|tabular)\}','',t)
 t=t.replace('\\toprule','').replace('\\midrule','').replace('\\bottomrule','').replace('\\\\','; ').replace('&',' | ')
 t=t.replace('\\_','_').replace('\\%','%').replace('\\item','\n-')
 t=t.replace('\\ ', ' ')
 t=t.replace('\\text{','').replace('\\mathrm{','').replace('\\mathbf{','').replace('\\{','{').replace('\\}','}')
 t=t.replace('\\in',' in ').replace('\\dots','...').replace('\\cdot',' times ').replace('\\leq',' <= ').replace('\\geq',' >= ')
 t=t.replace('\\pi','pi').replace('\\mu','mu').replace('\\gamma','gamma').replace('\\theta','theta').replace('\\sigma','sigma')
 t=t.replace('\\emph','').replace('\\rm ','').replace('\\^','^').replace('\\_','_').replace('\\{','{')
 t=t.replace('\\','').replace('$','').replace('{','').replace('}','')
 return t

def render(name):
 src=(ROOT/(name+'.tex')).read_text();src=src.replace('\\input{shared.tex}','').replace('\\end{document}','')
 title_match=re.search(r'\\title\{([^}]*)\}',src)
 report_title=title_match.group(1) if title_match else name
 src=re.sub(r'\\(title|author|date)\{[^}]*\}|\\maketitle|\\begin\{abstract\}|\\end\{abstract\}','',src)
 pieces=re.split(r'(\\begin\{equation\}.*?\\end\{equation\})',src,flags=re.S)
 doc=SimpleDocTemplate(str(ROOT/(name+'.pdf')),pagesize=letter,leftMargin=65,rightMargin=65,topMargin=48,bottomMargin=48,title=report_title,author='MEGA27 13b')
 story=[Paragraph(escape(report_title),title),Paragraph('WORKING RESEARCH REPORT - September 25, 2026',small),Spacer(1,8)]
 number=0
 for piece in pieces:
  if not piece.strip():continue
  if piece.startswith('\\begin{equation}'):
   number+=1;eq=piece.replace('\\begin{equation}','').replace('\\end{equation}','').strip().replace('\\\\','; ')
   # Render formula as math, preserving numbering. This is display math, not raw TeX prose.
   eq=eq.replace('\\frac1{','\\frac{1}{').replace('\\tfrac12','\\frac{1}{2}')
   eq=eq.replace('\\mathbf 1','\\mathrm{1}').replace('\\mathbb E','\\mathrm{E}').replace('\\mathcal N','N')
   eq=eq.replace('\\begin{cases}e^2/2&|e|\\leq 1\\\\|e|-1/2&|e|>1.\\end{cases}',r'\\mathrm{Huber}(e)')
   eq=eq.replace('\\simN','\\sim N')
   if number==11 and name=='sepsis': eq=r'\ell_\delta(e)=e^2/2\ (|e|\leq 1);\quad |e|-1/2\ (|e|>1)'  
   if '\\begin{cases}' in eq: eq=r'\ell_\delta(e)=\mathrm{Huber}(e)'  
   eq=eq.replace('\\text{','\\mathrm{').replace('\\mathbf{','\\mathrm{')
   eq=eq.replace('\\arg\\min','\\min').replace('\\arg\\max','\\max')
   try:
    fig,ax=plt.subplots(figsize=(8.5,.52),dpi=170);fig.patch.set_alpha(0);ax.axis('off')
    ax.text(.01,.50,f'$({number})\\quad {eq}$',va='center',fontsize=11)
    out=BytesIO();fig.savefig(out,format='png',transparent=True,bbox_inches='tight',pad_inches=.04);plt.close(fig);out.seek(0)
    from PIL import Image as PILImage
    im=PILImage.open(out);w,h=im.size;out.seek(0)
    ratio=min(450/w,40/h,1);story.append(Image(out,width=w*ratio,height=h*ratio));story.append(Spacer(1,7))
   except Exception as error:
    plt.close('all');print('formula render fallback',name,number,error)
    story.append(Paragraph(escape(f'({number}) {eq}'),formula))
   continue
  piece=clean(piece)
  for block in re.split(r'\n\s*\n',piece):
   text=block.strip()
   if not text:continue
   if text.startswith('## '):
    story.append(Paragraph(escape(text[3:]),heading))
    if text.startswith('## Observed model results') or text.startswith('## Simulated outcomes'):
     story.append(Image(str(ROOT.parent/'results'/(name+'_comparison.png')),width=455,height=228))
    continue
   text=re.sub(r'\s+',' ',text)
   story.append(Paragraph(escape(text),base))
 def footer(canvas,doc):
  canvas.saveState();canvas.setFont('TNR',8);canvas.drawString(65,35,'WORKING - not a clinical study or finished 20-page paper');canvas.drawRightString(550,35,str(doc.page));canvas.restoreState()
 doc.build(story,onFirstPage=footer,onLaterPages=footer)
 print(name,number,ROOT/(name+'.pdf'))
if __name__=='__main__':
 for name in sys.argv[1:]:render(name)

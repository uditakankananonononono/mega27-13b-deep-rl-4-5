"""Correct interpretation and integrate native results, preserving evidence."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ledger=[]
def edit(path,old,new):
 p=ROOT/path;s=p.read_text();assert old in s,(path,old[:70]);p.write_text(s.replace(old,new));ledger.append({'file':path,'before':old,'after':new})
edit('papers/parkinson.tex','the one nonlinearity PID cannot track','a tested model coupling associated with a larger neural-PID gap')
edit('papers/parkinson.tex','Policy-trigger analysis shows the learned controller is physiologically sensible - stimulation probability rises monotonically with beta state - yet still loses on magnitude selection.','In the synthetic surrogate, stimulation probability rises with its assigned beta state. This is consistency with an assumed feedback variable, not physiological validation. The learned controller still loses to tuned classical controls.')
edit('papers/parkinson.tex','A native DBS-Gym pilot at 512 neurons is reported with its protocol mismatches preserved rather than patched.','Native 512-neuron REINFORCE pilots collapse to constant actions (five ON and one OFF across six seed pairs), and a periodic half-duty open-loop control is never on the pooled four-policy reward envelope. These negative results retain their short-horizon and overlapping-training-seed limitations; none reproduces the published SAC protocol.')
edit('papers/parkinson.tex','None tests a learned RL controller at published scale.','Later native REINFORCE pilots use the 512-neuron spatial scale, but not the published SAC algorithm, episode protocol or evaluation replication. They show collapse rather than an effective learned controller.')
edit('papers/parkinson.tex','Forty genuinely used science/data tools are also far away: eight are documented after a reduced native-simulator smoke test, not 40. This is not a finished 20-substantive-page paper.','The earlier eight-tool tally was a smoke-test checkpoint, not a current final count; later data products are described individually. A fresh deduplicated used-tool audit and the substantive body-page gate remain open. PDF total pages do not establish either gate.')
edit('papers/parkinson.tex',"the policy is overwhelmingly beta-triggered, which is the physiologically intended behavior, yet it still loses to PID because its magnitude selection among the 3 levels is suboptimal.","the policy is beta-triggered within this assigned synthetic model. It still loses to PID in the tested objective. Trigger probability alone does not identify magnitude selection as the complete causal explanation or validate physiological appropriateness.")
edit('papers/parkinson.tex','be impossible with this data, not merely unattempted: \\textbf{molecular biomarkers of\ndisease status do not identify stimulation-relevant states}', 'be unsupported by this data: \\textbf{the available molecular disease-status labels do not establish stimulation-relevant states}')
edit('papers/sepsis.tex','The strict sepsis science-tool ledger counts 8 distinct used tools, far short of 40. The paper is a working report, not the requested 20 substantive pages. The missing gate remains conspicuous.','The earlier eight-tool ledger was a checkpoint; later public data products are discussed below. A fresh deduplicated used-tool audit and substantive body-page audit are still required. This working report does not claim completion from PDF page totals.')
# Repair bare underscores in prose, not math/code formulas. This is typesetting.
p=ROOT/'papers/sepsis.tex';s=p.read_text();out=[];math=False;env=False;i=0;count=0
while i<len(s):
 if s.startswith('\\begin{equation}',i):env=True
 if s.startswith('\\end{equation}',i):env=False
 ch=s[i]
 if ch=='$' and (i==0 or s[i-1]!='\\'):math=not math
 if ch=='_' and not(math or env) and (i==0 or s[i-1]!='\\'):out.append('\\_');count+=1
 else:out.append(ch)
 i+=1
p.write_text(''.join(out));ledger.append({'file':'papers/sepsis.tex','typesetting_bare_prose_underscores_escaped':count})
(ROOT/'results/manuscript_claim_consistency.json').write_text(json.dumps(ledger,indent=2)+'\n')

"""Scientific visualization of simulator returns, contextual cohort AUC; different endpoints."""
import json,matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
plt.rcParams.update({'font.family':'Times New Roman','font.size':11,'figure.dpi':160})
for name in ('sepsis','parkinson'):
 d=json.loads(Path(f'results/{name}.json').read_text());fig,ax=plt.subplots(figsize=(7,3.5))
 if name=='sepsis':
  vals=d['our_exact_values'];labels=['Random','Expert','Optimal','Neural'];values=[vals['random'],vals['expert'],vals['optimal'],d['neural_greedy']['expected_return']]
  ax.set_ylabel('Expected survival reward in MDP (not clinical)');ax.set_ylim(0,1)
 else:
  labels=list(d['controllers']);values=[d['controllers'][x]['mean_return'] for x in labels];labels=['Neural','Off','High','Threshold'];ax.set_ylabel('Surrogate-only mean episode reward');ax.set_ylim(-22,0)
 ax.bar(labels,values,color=['#385f82','#758fa7','#455264','#a96c46'])
 for i,v in enumerate(values):ax.text(i,v+(.015 if name=='sepsis' else -.7),f'{v:.3f}',ha='center',va='bottom' if name=='sepsis' else 'top')
 ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
 ax.set_title(('ICU-Sepsis published transition model' if name=='sepsis' else 'Custom Parkinson surrogate: NOT DBS-Gym'))
 fig.tight_layout();fig.savefig(f'results/{name}_comparison.png');plt.close(fig)

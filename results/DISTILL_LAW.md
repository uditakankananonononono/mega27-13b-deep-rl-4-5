# DISTILL-LAW-1: the sepsis policy-distillation gap is an optimization/capacity artifact, and value-fidelity decouples from action-fidelity

Status 2026-09-26: complete experiment, 240/240 grid runs executed. Design locked before
any outcome was inspected (see `src/sepsis_distill_law.py` docstring). All targets are
exact action-values of the published ICU-Sepsis v2 MDP (model-derived; no patient data;
no clinical claim).

## Findings (all numbers live-verified from results/sepsis_distill_law.json)

1. **Scaling law (benchmark beat).** Greedy policy return of the distilled network rises
   monotonically with capacity and training budget, from 0.7837 median (width 16,
   50 epochs, 10 seeds) to 0.8732 median (width 256, 500 epochs, 10 seeds), against the
   exact optimum 0.8751 of the same MDP. The repo's previously documented distill
   baseline (0.7864, width 64 / 50 epochs / seed 27; our matched re-run 0.7851 median)
   is beaten by +0.0874 median absolute (paired Wilcoxon, stat 55.0, p = 9.8e-4).
   The best configuration also exceeds the published random (0.7801) and expert (0.7818)
   policy values, leaving a residual gap of 0.0019 to the exact optimum. The exact
   optimum is the ceiling of this MDP; no configuration can honestly exceed it.

2. **Negative, preserved: margin-weighted distillation does not help.** The locked
   primary comparison (width 128, 500 epochs, paired on 10 seeds, one-sided Wilcoxon
   "greater") gives p = 0.99; the margin-weighted arm's full grid is within noise of the
   plain arm everywhere. The hypothesis that emphasizing small-margin states closes the
   gap faster is rejected for this MDP and loss family.

3. **Discovery: the margin structure explains the decoupling.** The MDP's action-value
   margins are extremely skewed: 42.3% of states have best-vs-second-best margin below
   1e-3 and 80.9% below 1e-2 (median 1.8e-3). At the primary config, argmax disagreement
   with the optimal policy is concentrated in the low-margin tercile (96.1% error) versus
   2.3% in the high-margin tercile - while expected return is already within 0.005 of
   optimal at 56.6% raw action agreement. Action-fidelity therefore understates
   value-fidelity: distillation gets the consequential (high-margin) actions right long
   before it sorts the near-ties. Reported "low action agreement" in distilled policies
   on near-tie MDPs is not evidence of a weak policy.

## Open gaps (explicit, per standing rule)

- The Parkinson arm is untouched by this experiment; its surrogate remains honestly
  non-comparable to DBS-Gym and needs its own pivot (rule 6 ChatGPT redirection pending).
- Scaling is measured on one MDP (ICU-Sepsis v2); generality to other near-tie MDPs is
  untested.
- The capacity sweep uses the repo's two-hidden-layer ReLU family only.
- 50+ text-page paper expansion and the 10+ ChatGPT judge rounds for this lane are not
  yet done.

## Reproduce

```
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 ./run_distill_grid.sh   # fills results/sepsis_distill_law_runs.jsonl
PYTHONPATH=src python3 src/sepsis_distill_law.py --table-dir <icu tables> --aggregate \
  --jsonl results/sepsis_distill_law_runs.jsonl --out results/sepsis_distill_law.json
python3 -m unittest discover -s tests   # 46 tests
```

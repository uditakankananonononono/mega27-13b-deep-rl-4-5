# MEGA27 item 13b: sepsis and Parkinson deep-RL treatment simulations

**Research prototypes only, not clinical decision systems.** Two disease tasks distinct from 13a HIV, T1D, and chemotherapy. Neither project satisfies all requested gates. The results are simulations; GEO expression is descriptive contextual data, never a causal validation of treatment policy. No unmeasured clinical benefit or benchmark record is claimed.

## Sepsis: ICU treatment actions

`src/sepsis.py` reads the unmodified ICU-Sepsis v2 model tables from [Choudhary et al.](https://github.com/icu-sepsis/icu-sepsis). It solves optimal, expert and random policies exactly by sparse Bellman updates, then trains a two-hidden-layer neural Q network on the model's Q values (distillation). The published rounded expected survival returns random/expert/optimal are 0.78/0.78/0.88; reproduced exact values are in `results/sepsis.json`. The neural policy is weaker, not a benchmark break. These are virtual simulator end states, not observed patient outcomes. GSE65682 blood-expression context was fetched and extracted to 479 accession-level records with measured features and a separate descriptive mortality-label probe. Critically, those records do **not** contain the action trajectories needed to test ICU policies.

## Parkinson disease: adaptive brain stimulation

`src/parkinson.py` trains a neural fitted-Q controller in a deliberately simple **custom surrogate** of beta oscillations, action-energy penalty and drift, compared on the same 120 held-out simulated seeds with fixed-high, off and threshold policies. It is not DBS-Gym and its scores cannot be compared to [Kuzmina et al.'s DBS-Gym](https://github.com/NevVerVer/DBS-Gym). GSE99039 blood-expression features (438 accessions) have PD/control labels but no DBS intervention outcomes or neural beta waveform; they are analyzed separately and cannot validate the surrogate.

## Reproduce

Python 3.10+ with NumPy and SciPy. Fetch the original GEO gzip files using URLs and verify SHA-256 in `data/*/provenance.json`; run `python src/geo_extract.py SOURCE.gz GSE65682 data/sepsis` (similarly GSE99039 and `data/parkinson`). Download the published [ICU-Sepsis CSV tables](https://github.com/icu-sepsis/icu-sepsis/blob/main/icu-sepsis-csv-tables.tar.gz), unpack outside the repository, and run:

```
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python src/sepsis.py --table-dir /path/to/icu-sepsis-csv-tables
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python src/parkinson.py
python src/geo_analysis.py data/sepsis/accession_features.csv results/sepsis_geo.json
python src/geo_analysis.py data/parkinson/accession_features.csv results/parkinson_geo.json
python -m unittest discover -s tests -v
```

The raw matrices are not in Git because they are 107 MB and 64 MB. `data/*/accession_features.csv` includes the measured numeric subset and sample IDs; full origin checksum is recorded. Neural runs are deterministic on this local CPU configuration, not guaranteed bit-identical on all BLAS systems.

## Gate ledger

| Project | Genuinely used distinct science/data tools | Accession records fetched, numeric data used | Numbered formulas | Times New Roman paper | Benchmark |
|---|---:|---:|---:|---|---|
| Sepsis | 9/40 | 514/120, across two GEO series; contextual only | 14/10 in working paper | Work in progress | Baselines reproduced; neural worse than optimal |
| Parkinson | 10/40 | 510/120, across two GEO series; contextual only | 14/10 in working paper | Work in progress | Not comparable to DBS-Gym |

Counting 479+35 day-one (sepsis) or 438+72 (Parkinson) distinct GSM identifiers with measured numeric features and labels, across two series per project. These are accession-level blood sample records, not independent treatment datasets; GSE54514 repeated follow-up samples are not added. Strict project-specific tool evidence in `evidence/scientific_tools.json`; no tool-count padding. Next work: run Parkinson against the real DBS-Gym simulator, obtain policy-relevant datasets where permitted, robust independent validation, expand genuinely used tools, and write/check substantial manuscripts. This is a progress checkpoint, not completion.

## Working manuscripts

`papers/sepsis.tex` and `papers/parkinson.tex` each include 14 numbered formulas. Current working PDFs are 10 sepsis and 9 Parkinson rendered pages, not the ~20 substantive pages requested. The renderer `papers/render_working.py` uses embedded Times New Roman regular/bold and Matplotlib equation/plot images. Because the installed LuaLaTeX lacks a compatible fontspec, these PDFs are an interim readable edition, not verified final LaTeX typesetting. Source TeX remains the primary manuscript source.

### Sepsis representation pivot

`src/sepsis_pivot.py` trains a second neural Q model with the published 47-dimensional state cluster centers and stratifies simulated model return using published SOFA state annotations. Result 0.785245 versus original one-hot neural 0.786449 and optimal 0.875142: another honest negative. Exact JSON in `results/sepsis_pivot.json`. This does not add a new external tool or patient dataset.

### Parkinson reward-cost pivot

`src/parkinson_pivot.py` trains two policies at synthetic energy coefficients 0.12 and 0.40, each evaluated under both coefficients on 100 matched new seeds. The return ordering flips (-3.785 versus +2.600 for heavy-minus-original), revealing sensitivity to the invented utility function. The heavy-weight controller cuts simulator energy from 37.635 to 14.830 but increases beta-burst steps from 0.5 to 10.01. This is a within-project negative/tradeoff, not DBS-Gym or patient validation.

### Native ICU-Sepsis cross-check

`src/sepsis_native.py` actually instantiates the official v2 environment via Gymnasium, verifies selected transition rows match the CSV tables, then samples 200 episodes per random/optimal/expert policy. Observed means .740/.865/.795, within ordinary Monte Carlo fluctuation around exact .780/.875/.782 at n=200 (per-policy SE .031/.024/.029). These additional native simulator/Gymnasium tools count for sepsis only, moving sepsis to 6/40 at that checkpoint; public MIMIC-III demo and pandas audit later raised it to 8/40; Parkinson rose from 8/40 after a separate tiny native smoke test to 9/40 after actual PhysioNet tremor data. Runs are still simulated and not new patients. See `results/sepsis_native.json`.

### Native DBS-Gym smoke, not a benchmark

`src/parkinson_native.py` executed the unmodified upstream SpatialKuramoto class in a deliberately reduced 8-neuron env0 configuration for four steps each with off and high fixed actions. Off reward 0.0, high reward -0.2 over four steps. This reveals that the tiny configuration has no detectable beta penalty and high stimulation only pays energy cost. It is a configuration-induced negative, **not** the 512-neuron, long-horizon published DBS-Gym benchmark. JAX/Diffrax/Gymnasium/DBS-Gym were genuinely exercised and are added to Parkinson tool ledger (8/40); full-scale published baseline and neural controller in the native environment remain missing.

### Full-neuron Env0 pilot

At 512 neurons and 1000 action steps (one seed, shorter than the repository training configuration's ~5555 steps; the published Table 1 protocol instead uses 10 x 1500-step evaluation episodes), upstream DBS-Gym OFF 0 V, high +5 V, and -5 V controls were executed. Mean per-step beta reward component OFF 19.9901, +5 V 6.7811, -5 V 8.0725; +5 V/OFF ratio 33.9%. Published Env0 Table 1 HF-DBS/OFF 19.8% over six evaluations, but this pilot differs in horizon/aggregation, so benchmark gate remains missing. An initial normalized -1 action was accidentally called OFF, but -1 actually rescales to -5 V; mislabeled result was corrected and kept as `results/parkinson_env0_negative5_1000.json`, then OFF 0 V was rerun. See `evidence/dbs_published_table1.json`.

The paper states an Env0 evaluation of 10 x 1500-step episodes and six repeats, but upstream `environment/env_configs/env0.py` has `eval0.total_episode_len=1000` model units (around 1111 actions), while its training configuration is 5000 units (~5555 actions). The repository's `aDBS_RL/evaluate_HF_DBS.py` main block imports `eval_envs_list` from env2, not env0. These version/protocol mismatches mean neither our training-config rollout nor a naive replay of that script can be represented as a Table 1 Env0 replication without further reconciliation.

### Full Env0 horizon checkpoint

`src/parkinson_env0_chunk.py` resumed the 512-neuron upstream environment for the complete 5555 steps of its 5000-unit Env0 episode on matched seed 222. The 50+50 resume test exactly matched an uninterrupted 100-step run in reward and LFP. Full OFF versus +5 V upstream mean beta-component ratio is 22.48%; the published Table 1 Env0 high/zero mean is 19.8% (SD 1.9, six evaluations), source `evidence/dbs_published_table1.json`. The paper actually evaluates Env0 with 10 x 1500-step episodes under varying initialization, repeated six times, not this 5555-step training-config trajectory. One seed and no upstream-trained deep-RL policy mean the benchmark gate STILL OPEN. Outputs: `results/parkinson_env0_full_off_seed222.json`, `results/parkinson_env0_full_high_seed222.json`.

### Native observable-threshold negative

An arbitrary threshold policy on true DBS-Gym Env0 LFP features (512 neurons, 1200 steps, seed 223) yields 65.8% high-amplitude actions, energy 3950 (versus fixed-high 6000 for 1200 steps), and mean beta component 11.178. This is **not deep RL** and is not compared with matched-seed controls or the published multi-episode evaluation. It is a next pilot, not a result gate. Details: `src/parkinson_env0_adaptive.py`, `results/parkinson_env0_threshold_1200_seed223.json`.

### Cohort-split molecular stress tests

Using original GEO cohort labels rather than random splitting, full eligible GSE65682 discovery-to-validation AUC is .595 (263 train, 216 test) versus random .553; full eligible GSE99039 TRAINING-to-VALIDATION AUC is .566 (293 train, 75 validation) versus random .596. Same GSE per disease and first 48 arbitrary probes; neither demonstrates clinical biomarkers or policy effects. Earlier first-150 Parkinson AUC .421 was subset-specific and is retained in Git history as a negative, not presented as the full-cohort result. Scripts/results: `src/geo_cohort_audit.py`, `results/*_geo_cohort.json`.

### Expanded GEO accession accounting, same studies

Re-extraction now retains all 479 GSE65682 mortality-labelled samples and all 438 GSE99039 IPD/control samples with 48 finite measured probes, rather than arbitrary first-150 cutoffs. Source gzip SHA256 hashes are unchanged. The larger sample counts do not add a second study or any action-policy trajectory. Updated random split AUC sepsis .553 and Parkinson .596; cohort split sepsis .595 (263/216) and Parkinson .566 (293/75). The earlier Parkinson first-150 cohort inversion (.421) remains documented in Git history as a subset-specific negative, not a robust all-cohort finding. Current data/results supersede that limited subset; benchmark and clinical gates are unchanged.

### Public MIMIC-III demo action-row audit (NOT off-policy validation)

Fetched seven public files from [PhysioNet MIMIC-III 100-patient demo](https://physionet.org/content/mimiciii-demo/1.4/). After a rough ICD-9 `038*`, `99591`, `99592`, `78552` admission filter (38 admissions, *not adjudicated sepsis*), joined InputEvents MV/CV with D_ITEMS for norepinephrine, vasopressin, 0.9% saline and lactated ringers: 1,728 event rows across 23 ICU stays. These are repeated event rows, **not 1,728 independent dataset accessions**. It does not map to ICU-Sepsis state/action abstraction or permit a causal policy evaluation; no efficacy claim. Source hashes, label counts and script in `results/sepsis_mimic_demo.json`, `src/mimic_demo_audit.py`. Row-level derived CSV and raw demo are withheld from repository pending license review. Actual PhysioNet data and pandas joins raise sepsis science-tool count to 8/40, not 40.

### Synthetic transition sensitivity, not a patient bootstrap

`src/sepsis_bootstrap.py` reruns fixed-policy value evaluation under 12 seeded Dirichlet perturbations of each nonzero ICU-Sepsis transition row. Concentration 100: mean 0.875879, empirical 5th-95th percentiles 0.870695-0.880670; concentration 10: mean 0.875441, range 0.854448-0.889248. This holds the original optimal action map fixed and transition support unchanged. The concentration is a hypothetical pseudo-count, **not** observed patient sample size. This is neither a patient bootstrap nor a clinical confidence interval or a new published benchmark. See `results/sepsis_model_sensitivity*.json`; initial exact value was 0.875142.

### Parkinson filtered-PSD reuse and denominator check

To reduce full-neuron rerun time, `src/parkinson_published_psd_audit.py` reuses cached one-seed OFF/+5V full-horizon LFP arrays and exactly applies upstream `evaluate_HF_DBS.py` filtered-PSD protocol. This reuses the 5555-step training-config trajectory, whereas the published protocol has 10 x 1500-step evaluation episodes. Ratio +5V/OFF = 27.536%, notably different from our windowed-reward ratio 22.48% and raw global-LFP ratio 20.97%; published six-evaluation fixed-high Env0 ratio 19.8% SD 1.9. Upstream filter utility matched the reproduced filter exactly in a local cross-check. The mismatch is retained, not advertised as benchmark replication, and no native deep-RL controller was evaluated. Checkpoint wave arrays are not in Git; their SHA256 digests and recalculation code are recorded.

### Second GEO series per disease: independent blood context, still no actions

Fetched and parsed GSE6613 (105 GSM, 50 PD and 22 healthy eligible) and GSE54514 (163 serial whole-blood GSM, only 54 people / 35 distinct day-one sepsis patients). Parkinson 13 shared probes across the GSE99039/GSE6613 platforms gave independent blood-label AUC **0.534** with all 438 original samples as training source; this negative does not validate stimulation actions. Source hashes, accession IDs, cohort counts and numeric calculations: `results/geo_independent.json`, `src/geo_independent.py`. Sepsis repeated blood draws are GSM accessions but not independent people or policy trajectories.

### Native full-neuron short deep-RL pilot, not published SAC

A two-layer tanh Bernoulli REINFORCE policy trained on 4 x 180 action steps in real 512-neuron DBS-Gym Env0, then evaluated for 1000 steps on seed 222. Native evaluation mean reward -13.554 and 50% on-time. Parameter audit found weight updates under 1e-6 L2 and output-bias update 2.9e-5, probabilities near 0.5 with prior-action threshold artifacts. Thus there is no evidence of effective learning despite running the update and native policy. It is not the published 10 x 1500-step, six-evaluation SAC comparison or clinical evidence. Code, seeds, weight checksum and negative cautions in `src/parkinson_native_reinforce.py`, `results/parkinson_native_reinforce.json`.

### Parkinson native optimizer pivot: all-OFF collapse

After negligible REINFORCE parameter drift, Adam episode updates, broader initial weights and no prior-action shortcut made weights change measurably (0.293/0.390 L2 in hidden layers). Yet deterministic native 1000-step seed-222 evaluation chose OFF every step: mean reward -19.990, the archived OFF control, worse than matched fixed-high -6.831 on this single seed/horizon. This is a preserved negative, not an RL benchmark victory. Script, metrics and trained weights: `src/parkinson_native_reinforce_adam.py`, `results/parkinson_native_reinforce_adam.json`, `data/parkinson/native_reinforce_adam_weights.npz`.

### Human Parkinson tremor recordings: fixed conditions only

Fetched and numerically analyzed 55 PhysioNet Parkinsonian tremor finger-velocity recordings across 15 subjects; 25 within-subject DBS-on/off pairs at fixed medication state gave descriptive median 4-6 Hz power ratio .873 (med-on .805 across 13 pairs, med-off 1.332 across 12). Mixed results, heterogeneous calibration and nonrandomized condition order. These are *not* LFP or step-level adaptive DBS action trajectories and cannot validate a treatment policy. Script and all source hashes: `src/parkinson_tremordb.py`, `results/parkinson_tremordb.json`; source https://physionet.org/content/tremordb/1.0.0/ . Parkinson science/data tool count 9/40.

### Suspicious perfect within-series sepsis AUC

On only 35 distinct day-one GSE54514 sepsis subjects (9 deaths), a first-48-probe, five-fold descriptive classifier produced out-of-fold AUC 1.0. All nine eligible nonsurvivors precede all survivors in the source matrix; subject-split folds cannot rule out source status block/batch confounding. The outside GSE65682 probe IDs do not match the 48 convenience probes, so this is **not** verified external validation or a mortality biomarker, and contains no treatment actions. Exact folds, indices and scores in `results/sepsis_second_series.json`, code `src/sepsis_second_series.py`.

### Treatment-trajectory availability audit

[MIMIC-Sepsis](https://github.com/yongh7/MIMIC-sepsis) publishes preprocessing code, not an openly downloadable full MIMIC-IV patient trajectory table. Its README requires credentialed MIMIC-IV access, research training, DUA and a local PostgreSQL import. Current 13b sepsis data therefore do not support external off-policy treatment validation; simulator MDP, 100-patient public demo and GEO blood labels are distinct sources. The public demo source specifies [ODbL 1.0](https://physionet.org/content/mimiciii-demo/view-license/1.4/); row-level clinical exports remain withheld.

### Neural Q-ranking failure analysis

Reconstructed one-hot sepsis network at the original seed. Nonterminal Q MSE .000558 and loss .000348 still yielded exact-best action match 5.75%, positive exact-model one-step Q shortfall in 61.9% of nonterminal states, and full policy value .786449 vs exact optimal .875142. One-step initial-weighted regret .00486 is not the full .08869 policy-value gap because later mistakes compound. These are model-only diagnostics, not patient outcomes. Code/results: `src/sepsis_rank_audit.py`, `results/sepsis_rank_audit.json`.

### Sepsis rank-target pivot: memorize exact MDP policy, not patient benefit

Three neural widths trained on all 713 exact-model best-action labels reached 100% **training-state** agreement and identical exact-model value .875142. This fixes the prior Q-regression network's ranking error (.786449) by supervised memorization of value iteration, not independent validation or deep-RL treatment discovery. Full sweep `src/sepsis_rank_pivot.py`, `results/sepsis_rank_pivot.json`. Patient outcomes and clinical claim unchanged.

### Internal sepsis state holdout: memorization does not transfer

Three fixed 570/143 state-ID splits of exact best-action labels left held-out optimal-action agreement at .126/.182/.196, despite perfect agreement on their training IDs. Same-split majority-action reference: .224/.196/.154. Full same-model returns .856390/.854850/.856915, below exact .875142. A one-hot input for a never-trained state ID lacks a meaningful shared representation. This is an internal negative, not independent patient or model validation: `src/sepsis_state_holdout.py`, `results/sepsis_state_holdout.json`.

### Sepsis centroid state-transfer pivot

Replacing one-hot state IDs with the published 47-dimensional centroids on the same three state holdouts improves held-out exact-action agreement to .266/.371/.343 from .126/.182/.196. But whole same-model returns decline on every split to .845072/.852847/.850264 (one-hot .856390/.854850/.856915). Both are supervised on exact model labels, and the centroid run uses 150 rather than 100 epochs. This is neither a controlled single-factor causal comparison nor patient validation: `src/sepsis_centroid_holdout.py`, `results/sepsis_centroid_holdout.json`.

### Exact in-model occupancy audit

For the three holdout splits, sparse policy-induced visitation solves account for the centroid policies' lower full-model values despite their better unweighted held-out action agreement. One-hot weighted Q shortfalls .018752/.020291/.018226 versus centroid .030070/.022295/.024877 equal each policy's exact-model optimal value gap up to numerical residual. Centroid has nonzero training-state weighted shortfall in all splits; one-hot training-state shortfall is zero. These are expected state visits under the *estimated MDP*, not patient counts or clinical regret: `src/sepsis_occupancy_audit.py`, `results/sepsis_occupancy_audit.json`.

### Equal-epoch sepsis feature comparison

At 150 epochs for both representations on the three fixed state splits, centroid held-out optimal-action agreement remains higher (.266/.371/.343 versus one-hot .133/.196/.189), yet full same-model returns remain lower (.845072/.852847/.850264 versus .856932/.855019/.856675). Input dimensions and parameter counts are not matched, and these splits had already been viewed; this is an engineering follow-up, not fresh external validation: `src/sepsis_budget_match.py`, `results/sepsis_budget_match.json`.

### One-participant human adaptive stimulation figure traces

The public [Dryad movement-responsive aDBS dataset](https://datadryad.org/dataset/doi:10.5061/dryad.4xgxd25hw) was downloaded and SHA256-checked. Its Fig5b excerpt has 4,196 left- and 4,193 right-hand time-indexed records of movement prediction and stimulation amplitude from **one** person, plus separate true movement-state traces. It records high/low stimulation around 2.2/1.6 mA with ramps. The Fig2 beta/gamma neural traces are from a different time interval (615-795 seconds versus Fig5 965-1175 seconds) and cannot be joined to these actions. This is a genuine additional Parkinson-specific data product (10/40), but 8,389 samples are not patients or independent accessions; without synchronized implant LFP, reward and logged propensities, it does **not** provide off-policy RL treatment validation. `src/parkinson_dryad_adaptive.py`, `results/parkinson_dryad_adaptive.json`.

### Static public eICU sepsis cohort is not sequential treatment evidence

Fetched and SHA256-checked the public [Dryad eICU sepsis risk cohort](https://datadryad.org/dataset/doi:10.5061/dryad.hmgqnk9wb). It contains 13,717 summary rows, 169 fields, 2014/2015 split 6,397/7,320, and first-24h vasopressor-use field `MEDS` (98 yes, 13,368 no, 251 missing); 28-day ICU deaths 1,276. We counted this as one genuinely analyzed Sepsis-specific data product, raising 9/40 tools. These are **not** 13,717 GSM accessions and not timed vasopressor/fluid dose actions. No patient policy-validation dataset was gained. `src/sepsis_dryad_eicu_audit.py`, `results/sepsis_dryad_eicu.json`; row-level data are not copied into Git.

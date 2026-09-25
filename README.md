# MEGA27 item 13b: sepsis and Parkinson deep-RL treatment simulations

**Research prototypes only, not clinical decision systems.** Two disease tasks distinct from 13a HIV, T1D, and chemotherapy. Neither project satisfies all requested gates. The results are simulations; GEO expression is descriptive contextual data, never a causal validation of treatment policy. No unmeasured clinical benefit or benchmark record is claimed.

## Sepsis: ICU treatment actions

`src/sepsis.py` reads the unmodified ICU-Sepsis v2 model tables from [Choudhary et al.](https://github.com/icu-sepsis/icu-sepsis). It solves optimal, expert and random policies exactly by sparse Bellman updates, then trains a two-hidden-layer neural Q network on the model's Q values (distillation). The published rounded expected survival returns random/expert/optimal are 0.78/0.78/0.88; reproduced exact values are in `results/sepsis.json`. The neural policy is weaker, not a benchmark break. These are virtual simulator end states, not observed patient outcomes. GSE65682 blood-expression context was fetched and sampled to 150 accession-level records with measured features and a separate descriptive mortality-label probe. Critically, those records do **not** contain the action trajectories needed to test ICU policies.

## Parkinson disease: adaptive brain stimulation

`src/parkinson.py` trains a neural fitted-Q controller in a deliberately simple **custom surrogate** of beta oscillations, action-energy penalty and drift, compared on the same 120 held-out simulated seeds with fixed-high, off and threshold policies. It is not DBS-Gym and its scores cannot be compared to [Kuzmina et al.'s DBS-Gym](https://github.com/NevVerVer/DBS-Gym). GSE99039 blood-expression features (150 accessions) have PD/control labels but no DBS intervention outcomes or neural beta waveform; they are analyzed separately and cannot validate the surrogate.

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
| Sepsis | 6/40 | 150/120, from one GEO series; contextual only | 14/10 in working paper | Work in progress | Baselines reproduced; neural worse than optimal |
| Parkinson | 8/40 | 150/120, from one GEO series; contextual only | 14/10 in working paper | Work in progress | Not comparable to DBS-Gym |

Counting one GEO matrix with 150 GSM identifiers as 150 distinct **accession-level sample records**, not 150 independent studies or treatment datasets. Strict project-specific tool evidence in `evidence/scientific_tools.json`; no tool-count padding. Next work: run Parkinson against the real DBS-Gym simulator, obtain policy-relevant datasets where permitted, robust independent validation, expand genuinely used tools, and write/check substantial manuscripts. This is a progress checkpoint, not completion.

## Working manuscripts

`papers/sepsis.tex` and `papers/parkinson.tex` each include 14 numbered formulas. Current working PDFs are 4 rendered pages each, not the ~20 substantive pages requested. The renderer `papers/render_working.py` uses embedded Times New Roman regular/bold and Matplotlib equation/plot images. Because the installed LuaLaTeX lacks a compatible fontspec, these PDFs are an interim readable edition, not verified final LaTeX typesetting. Source TeX remains the primary manuscript source.

### Sepsis representation pivot

`src/sepsis_pivot.py` trains a second neural Q model with the published 47-dimensional state cluster centers and stratifies simulated model return using published SOFA state annotations. Result 0.785245 versus original one-hot neural 0.786449 and optimal 0.875142: another honest negative. Exact JSON in `results/sepsis_pivot.json`. This does not add a new external tool or patient dataset.

### Parkinson reward-cost pivot

`src/parkinson_pivot.py` trains two policies at synthetic energy coefficients 0.12 and 0.40, each evaluated under both coefficients on 100 matched new seeds. The return ordering flips (-3.785 versus +2.600 for heavy-minus-original), revealing sensitivity to the invented utility function. The heavy-weight controller cuts simulator energy from 37.635 to 14.830 but increases beta-burst steps from 0.5 to 10.01. This is a within-project negative/tradeoff, not DBS-Gym or patient validation.

### Native ICU-Sepsis cross-check

`src/sepsis_native.py` actually instantiates the official v2 environment via Gymnasium, verifies selected transition rows match the CSV tables, then samples 200 episodes per random/optimal/expert policy. Observed means .740/.865/.795, within ordinary Monte Carlo fluctuation around exact .780/.875/.782 at n=200 (per-policy SE .031/.024/.029). These additional native simulator/Gymnasium tools count for sepsis only, moving sepsis to 6/40; Parkinson now 8/40 after a separate tiny native smoke test. Runs are still simulated and not new patients. See `results/sepsis_native.json`.

### Native DBS-Gym smoke, not a benchmark

`src/parkinson_native.py` executed the unmodified upstream SpatialKuramoto class in a deliberately reduced 8-neuron env0 configuration for four steps each with off and high fixed actions. Off reward 0.0, high reward -0.2 over four steps. This reveals that the tiny configuration has no detectable beta penalty and high stimulation only pays energy cost. It is a configuration-induced negative, **not** the 512-neuron, long-horizon published DBS-Gym benchmark. JAX/Diffrax/Gymnasium/DBS-Gym were genuinely exercised and are added to Parkinson tool ledger (8/40); full-scale published baseline and neural controller in the native environment remain missing.

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
| Sepsis | 3/40 | 150/120, from one GEO series; contextual only | 12/10 in working paper | Work in progress | Baselines reproduced; neural worse than optimal |
| Parkinson | 3/40 | 150/120, from one GEO series; contextual only | 12/10 in working paper | Work in progress | Not comparable to DBS-Gym |

Counting one GEO matrix with 150 GSM identifiers as 150 distinct **accession-level sample records**, not 150 independent studies or treatment datasets. Strict tool evidence in `evidence/scientific_tools.json`; no tool-count padding. Next work: run Parkinson against the real DBS-Gym simulator, obtain policy-relevant datasets where permitted, robust independent validation, expand genuinely used tools, and write/check substantial manuscripts. This is a progress checkpoint, not completion.

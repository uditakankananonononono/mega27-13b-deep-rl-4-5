# Distinct validation candidate ranking, 7 October 2026

1. 13b: use freshly pinned public ICU-Sepsis admissible-action metadata to test whether mean-filled full-action transitions and their exact optimal values agree with the restricted admissible-action MDP, and quantify source support. Winner: complete finite-table equivalence claim, not another parameter grid; can decide implementation semantics without restricted patient data.
2. 14: independently import the public BioModels vonDassow2000 spatial ODE/SBML and test a documented published segmentation trajectory against the local reduced-model scope. Deferred: genuinely different biology/model, but import/simulator and species/time mapping need more preparation than the finite-table oracle.
3. 07: qualify and freeze GSE63789 WT/perturbed paired RNA/footprint fields for cross-source yeast translation transfer of fixed existing sequence features. Deferred: new experimental endpoint, but normalization, transcript joins, replicate independence and condition overlap must be settled before scoring.
4. 23b: reproduce original DNA-Fountain sequence-read filtering using public PRJEB19305/PRJEB19307 sequencing records against a pinned receiver. Deferred: physical read evidence is valuable, but large read downloads, original Python2 receiver and molecular/read-count budgets need a separate ingestion unit, not another four-message codec grid.

Source leads, read live before ranking:
- https://github.com/icu-sepsis/icu-sepsis
- https://arxiv.org/html/2406.05646v2
- https://pypi.org/project/icu-sepsis/
- https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63789
- https://www.frontiersin.org/journals/cell-and-developmental-biology/articles/10.3389/fcell.2023.1201673/full
- https://www.ebi.ac.uk/biomodels/BIOMD0000001065
- https://github.com/TeamErlich/dna-fountain

No candidate implies clinical, biological or physical validation before its source and outcome are actually checked.

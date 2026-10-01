# Sepsis action-trajectory source qualification, 1 October 2026

This is source qualification, not newly downloaded patient trajectories, access approval, policy evaluation or treatment efficacy. Three live sources were fetched during this run. Existing cached trial results are orientation only and are not presented as a rerun.

## ICU-Sepsis public benchmark
https://github.com/icu-sepsis/icu-sepsis

The current upstream README describes a simulated tabular MDP with716states/25actions, derived fromMIMIC-III. Its publicly described transitionFunction.csv, rewardFunction.csv, initialStateDistribution.csv and expertPolicy.csv are estimated model tables, not identified longitudinal patient events. The README explicitly fills inadmissible action transitions with a mean over admissible actions. This is a model extrapolation choice; an optimal value in that estimated model is not an observed treatment-policy survival difference. Public simulated rollouts can test implementation, not independent clinical validation.

## Randomized guidance-strategy release
https://zenodo.org/records/10579408

The live record concerns the CVP versus IVC fluid-guidance trial and its released dataset. Prior local audit describes123patient summaryrecords/58variables, aggregate72-hourfluid and norepinephrine totals, with missing30daymortality. This run does not re-download or read its Stata file, so those prior counts are explicitly cached, not current rerun results. Strategy assignment, accumulated dose and arm mortality do not establish timestamped state/action/reward sequences or action propensities. Before qualifying any file, inspect actual per-patient timing and intervention fields. Do not turn a randomized strategy arm into a25-action sequential policy label.

## MIMIC-IV as a potential trajectory source
https://physionet.org/content/mimiciv/

The live official page lists access only for credentialed users who sign theDUA, required CITI Data or Specimens Only Research training, and PhysioNetCredentialedHealthDataLicense1.5.0/DUA1.5.0. It describes patient information, provider orders and hospital/ICU tables, but these are not a ready-made external sepsis policy cohort. No access/account/credential/training status was checked in this run and no data request or user account mutation was made. Do not claim the user lacks access. Current run has no patient files. Proper extraction would still require population/sepsis definitions, time alignment, fluid/vasopressor dose definitions, missingness, censoring, mortality linkage, logging-policy support and separation from training patients. Credentialed access is a route, not evidence of granted access or an approved new study.

## Qualification gate before scoring
1. Source permission and patient-data use restrictions verified independently.
2. Actual file readback with durablepatient/stayidentifiers,decisiontimestamps,statevariables,interventions/doses,terminaloutcomes andmissingness.
3. Frozen cohort/action/timebin/preprocessing definitions before policy scoring, including train/external patient overlap and action support.
4. Off-policy identification assumptions and estimator checks; never treat model rollouts as causal effects.
5. Aggregate/report without publishing patient records or disclosing restricted data.

No qualified new independent action-trajectory dataset is landed here. The remaining gap is a source and extraction protocol that actually supplies eligible longitudinal state/action/outcome records with authorized access, not another positive simulator return.

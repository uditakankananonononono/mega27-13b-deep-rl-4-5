# Lane-13b AMENDMENT QUEUE - LOCKED BEFORE EXECUTION
Verdict: PROVIDED round 1 (WhatsApp 10:54:47 IST, wamid...RTlERDhGOQA=, verbatim in judge/round_provided1_verdict_whatsapp.txt).
Locked: 2026-09-27 10:58 IST, against 56pp build d381c5a. No execution began before this lock.

## Spine (hers): pivot "AI adaptive DBS" -> "Benchmarking requirements for trustworthy adaptive DBS algorithms" (computational methods paper). Convert limitations into the contribution (#12). One story (#11): AI control methods. Cautious clinical language (#19): "computational model of adaptive stimulation", never "treat Parkinson disease".

## Weakness items (#1-#19) mapped vs current paper
- #1 RL not clinically connected -> SPINE pivot (title/abstract/intro restructure)
- #2 model uncertainty study (multiple plausible simulators; does the controller generalize across environments): NEW - cheap, we own the simulator
- #3 data-readiness framework ("what information is required before RL validation"): NEW - editorial+formal; paper already documents tremor-data insufficiency, reframe as framework
- #4 novelty = safety-aware validation (failure detection, uncertainty estimation, policy instability scoring): NEW - policy instability measurable across our 240 runs + env0 seeds
- #5 PD transcriptomics disconnected -> INTEGRATE as "can molecular biomarkers identify states relevant to stimulation" or demote to dataset-limitations demonstration (with #6)
- #6 AUC 0.596 weak -> reframe as dataset-limitations demonstration (already the paper's framing - strengthen)
- #7 within-subject variability analysis on tremor records (15 subjects, fixed conditions): NEW - cheap on existing tremordb records (55 traces, paired conditions)
- #8 benchmark vs fixed/threshold/PID controllers identical conditions: PARTIAL (always-off/fixed-high/beta-threshold already benchmarked; ADD PID)
- #9 many seeds + CIs + statistical testing: PARTIAL (120 held-out seeds + bootstrap CI already; extend native runs multi-seed)
- #10 safety multi-objective reward (symptom/energy/stability): PARTIAL (energy+burst penalties exist; add stimulation-stability term)
- #11 choose one story: AI control methods (with PD as motivating application)
- #12 limitations -> contribution: EDITORIAL spine (methods-paper framing)
- #13 predictive biomarker beyond PD/control (tremor severity/DBS response): NEW - check public labels; if none exist, that IS the data-readiness answer
- #14 external benchmark vs published adaptive DBS sims / open benchmarks: PARTIAL (DBS-Gym noncomparability already documented; formalize as benchmark-requirements table)
- #15 justify fitted-Q vs DQN/actor-critic/MPC: PARTIAL (REINFORCE + Adam pivot exists; add MPC baseline on our simulator - cheap)
- #16 interpretability: which states trigger stimulation: NEW - policy analysis on trained Q-net (cheap)
- #17 ablations (energy penalty/noise/drift): NEW - cheap on our simulator
- #18 strongest result hidden: "many attractive AI-health claims fail without proper data alignment" -> EDITORIAL headline
- #19 cautious clinical language: EDITORIAL sweep (grep 'treat|therapy|patient benefit')

## NOT-to-do: no "treat Parkinson" language; no pretending trajectories exist; PD transcriptomics not presented as controlling DBS.


## LANDED 2026-09-27 (second wave)
- #13: predictive-biomarker label audit LANDED as data-readiness proof: enumerated every public record - tremordb (condition flags only), GEO blood (disease/control only), 1-participant adaptive excerpt (amplitude+LFP, no outcome scale). The public record for adaptive DBS has signals and actions but NO outcomes -> R5 (severity/response labels exist only in clinical/trial records). Paper section; 59pp.
- #7: within-subject tremor variability LANDED. 55 traces/15 subjects: median per-subject CV 1.14 (max/min power ratio median 46.8, up to 4723); 25 med-matched DBS on/off pairs: DBS-on reduces 4-6Hz power in only 52% (median ratio 0.87). Framed as R4 (validation data must document condition structure finely enough to recover the expected physiological effect). results/parkinson_tremor_within_subject.json, paper section.
- SPINE RESTRUCTURE LANDED (#1/#11/#12/#18): title now "Benchmarking Requirements for Trustworthy Adaptive Neurostimulation Algorithms: A Computational Methods Study in Which Tuned Classical Control Defeats Deep Fitted-Q". Abstract rewritten around R1/R2/R3 + data-readiness; strongest result (PID/MPC defeat neural) is the headline, not hidden. #19 cautious-language grep clean (no unnegated treat/therapy/patient-benefit). 58pp.
- #17: ablations LANDED (energy penalty off/2x, obs noise ~0/2x, drift decoupled; retrained per variant, 60 seeds 80100-80159). Findings: energy terms carry most objective difficulty (energy-free: neural == PID exactly, +0.004); noise monotonically degrades and controller prudently stimulates less (0.98->0.90); drift-coupled efficacy is the one nonlinearity PID cannot track (gap -1.07 -> -0.16 decoupled). R3 added (ablation identity of every reward term). results/parkinson_ablations.json, paper section, 58pp.
- #2: model-uncertainty ensemble LANDED. Fitted-Q transferred unmodified to 6 perturbed simulator replicas vs retrained and fixed-gain PID (60 seeds 70100-70159). Transfer gap small/sign-variable (-0.28..+0.88); PID gap UNIVERSAL: PID beats transferred neural on 6/6 replicas and retrained on 5/6. R1 survives misspecification; R2 added (state the model family the uncertainty study spans). results/parkinson_model_uncertainty.json, src/parkinson_model_uncertainty.py, paper section.
- #8/#15: PID (gains tuned on 40 disjoint seeds; Kp2.0 Ki0.1 Kd0.1) + MPC (h3, model-known upper bound) baselines on identical 120 held-out seeds. RESULT IS A POSITIVE-FRAMED NEGATIVE: neural fitted-Q (-10.44+/-1.27) LOSES to PID (-9.04+/-1.12) and MPC (-9.61+/-1.14), paired bootstrap CIs exclude zero; beats always-off/fixed-high/threshold. Framed as benchmarking requirement R1 (must clear tuned classical control). results/parkinson_pid_mpc.json, src/parkinson_pid_mpc.py, new paper section.
- #16: policy-trigger interpretability LANDED: stimulation fraction by beta bin (0.755@low-beta -> 1.0@beta>0.6); permutation ablation beta 39.6% vs drive 12.7% vs drift 10.2% action-change - policy is beta-triggered as intended but magnitude selection suboptimal.

## First deliverables (cheap, existing data)

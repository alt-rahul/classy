# CASMI 2026 - roadmap to beat 0.552 (written 2026-11-20, 24 days to the final deadline)

Supersedes the work queue of PLAN.md section 6 and the ranked list of RESEARCH.md section 8. PLAN.md stays the
reference for module conventions; AGENT_RULES.md still applies with one change (section 3, "rule change").
Sources reconciled here: KAGGLE_INTEL.md (board, host rulings, community measurements), RESEARCH.md (literature),
DATA_AUDIT.md, EXPERIMENTS.md. Package ids are R1, R2, ...; old WP ids are quoted where a package continues one.

---

## 1. Strategic call

**The final submission is our own self-trained pipeline. The fork is a benchmark and a fallback, never the base.**

Reasons:
1. Reproducibility rule. The fork carries weights without training code (the v80 scorer kernel output, parts of
   the v4b engine bundle). The host may remove such submissions "regardless of ranking". Everything in our
   pipeline is either trained by our code on permitted data or an official checkpoint with its original training
   code (ms-pred ICEBERG/GLACIER on MassSpecGym, FRIGID official).
2. No edge in the fork. 518 teams sit at exactly 0.495. Its last +0.030 came in steps of 0.001 to 0.006 selected
   on 125 molecules (standard error 0.016, winner's curse about +0.02), so its private score is more likely 0.47
   to 0.49 than 0.495. Copying it buys the bronze line at best.
3. No room in the fork. Its scoring run uses up to 8.6 h of the 9 h limit and several GPU hours per submission.
   Nothing substantial can be added, and every public attempt to swap one of its models lost (-0.007 to -0.024)
   because features, encoder and ranker were fitted to each other. We must own the three jointly.
4. The gap to the leader is structural. The public stack is frozen channels fused by hand-set reciprocal-rank
   weights; it keeps the base leader at rank 1 almost always and has no working class-3 route. A ranker trained
   jointly on simulated class mixes, with structure-aware isomer evidence and a database-style class-3 pool, is
   the only kind of change that can be worth +0.05.

What the arithmetic of the board says (used to place the bets). With class 1 at 16 percent scoring 0.93 and
class 3 scoring about 0.05, the public 0.495 implies a class-2 MRR of 0.72 if class 3 is 39 percent, and an
impossible 0.99 if class 3 is 52 percent. So on the public split class 3 is at most about 40 percent, the public
stack already solves class 2 at about 0.6 to 0.7, and the leader (0.552) either has class 2 near 0.8 or scores
on class 3. Remaining head-room: class-2 rank 1 among isomers (each +0.10 class-2 MRR is +0.045) and class 3
(each +0.10 class-3 MRR is +0.03 to +0.04). Class 1 is saturated except for coverage by external libraries.

Role of the fork:
- Benchmark: its public score calibrates the board once; optionally one commit run on molecules it has never
  seen (R16) gives its class-2 number on our panel.
- Source of ideas and open data, not weights: we reuse the public data assets (PubChem prior table, external
  library dumps) and reimplement mechanisms.
- Fallback: it is our second final selection only if our own pipeline is not within 0.02 of it on the public
  board by 2026-12-12 (section 4). Removal risk then touches that one selection only.
- Never a feature source inside a final submission (inherits the eligibility risk and the 8.6 h runtime).

Dates: parity core by 2026-11-27 (M2), parity by 2026-12-02 (M3), bets land by 2026-12-09, freeze 2026-12-11,
final selection 2026-12-13. GPU budget: about 30 h per week, about 100 h in total (section 3.4).

---

## 2. Validation protocol (binding for every decision from now on)

### 2.1 What was wrong with the v1 protocol

- NP-TIMS C2 is far too easy: the prior alone scores 0.822 there (0.503 on its 14 pubmed-zero molecules) and
  0.154 on the board. All 250 structures are famous and in 6 other libraries (median).
- "Worst-case blend on NP-TIMS" with mixes down to 20 percent class 3 does not match the community estimate
  (class 1 about 16 percent, class 3 up to 40 percent).
- C3 by deleting the truth overstates generator reach about 3x (close analogs and duplicates stay).
- SYN-TIMS was only a sanity check; it is in fact the only honest proxy for class-2 molecules that sit in
  PubChem but in no NP database and have no public spectra (prior gain there matches the board: +0.075).

### 2.2 Cells

| cell | molecules | mask | pool | role |
|---|---|---|---|---|
| C1 | NP-TIMS 250 (primary), NP-OTHER eligible 1,871 (secondary) | query library's own spectra hidden; other train libraries and external libraries visible | all tiers | class 1 as on the test: timsTOF query, references from other instruments |
| C2-NP | NP-TIMS + NP-OTHER molecules that are in the NP tier | all spectra of the molecule hidden, in train and in external libraries | all tiers | class 2, natural-product route. Weighted by obscurity `w_obs` (2.3) |
| C2-PC | SYN-TIMS without public spectra (about 2,900), NP-OTHER in PubChem but not in the NP tier, C2-ZEN (new) | as C2-NP | all tiers | class 2, PubChem-only route: truth among about 10,000 decoys with weak priors |
| C3h | NP-TIMS + NP-OTHER | C2 mask, plus the truth and every structure with Morgan-2 Tanimoto >= 0.75 to it removed from every tier, from the library and from generator seeds | all tiers + GEN | class 3, realistic ("hard"). Used for decisions |
| C3e | same | C2 mask, plus only the truth removed (the old C3) | all tiers + GEN | reported only; an upper bound |

In every cell the encoder and all trained models have never seen the molecule (v_holdout models; 2.5).

### 2.3 Changes to the splits: `splits/holdout_v2.csv` (owner R1)

1. Keep every row of `holdout_v1.csv` unchanged (ik14, mkey, set, fold, eligible_c1, query_lib, primary,
   mono_mass, formula). bundle_v1 and all models already exclude them; this must stay true.
2. Add columns: `in_np`, `in_pubchem`, `pmid_count`, `sid_count` (NaN until R3), `n_ext_lib` (0 until R2),
   `has_public_spectra` = eligible_c1 or n_ext_lib > 0, `c2_cell` in {NP, PC, none}, `w_obs`, `fast`.
3. Add set `C2-ZEN`: structures of the Enveda-180 Zenodo release that are absent from train (816 to 3,719 by
   public counts; take all, at most 1,500, folds 0-4), queries built with `validation.build_queries`. They are
   never trained on by anything. They enter the external library only behind the per-call mask.
4. Drop the SYN-TIMS C1 cell (n = 44). SYN-TIMS molecules with `has_public_spectra` leave C2-PC.
5. `w_obs` = (1 + pmid_count)^(-alpha) on C2-NP, with alpha in [0, 1.5] chosen so that the weighted MRR of the
   b0 prior-only pipeline over C1 + C2-NP equals 0.154 / (pi_C1 + pi_C2NP) (0.33 for the default mix). This
   ties the fame of the validation set to the one board number that measures it. Recompute when the mix changes.
6. `fast` panel (every experiment): all NP-TIMS, 750 NP-OTHER, 500 SYN-TIMS, 300 C2-ZEN, stratified by fold and
   pmid bin. Full panel only for milestones. Target: fast panel under 15 min from the feature store (R5).

### 2.4 The decision number

LB-proxy P(mix) = pi_1 C1 + pi_2np C2-NP + pi_2pc C2-PC + pi_3 C3h, with C1 from NP-TIMS, out-of-fold.

| mix | C1 | C2-NP | C2-PC | C3 | meaning |
|---|---|---|---|---|---|
| A (default) | 0.17 | 0.30 | 0.15 | 0.38 | community central estimate |
| B | 0.16 | 0.24 | 0.12 | 0.48 | class 3 concentrated in the private split |
| C | 0.22 | 0.36 | 0.17 | 0.25 | class 3 over-estimated |

Rules:
- A change is adopted when P(A) rises by at least 0.005 with a paired bootstrap 90 percent interval above 0 on
  the full panel, min(P(A), P(B), P(C)) does not fall, and no cell falls by more than its own paired interval.
- Rank-1 changes and tail changes are reported separately (top-1 rate and MRR on ranks 2-25).
- Seed noise: LightGBM runs are 5-seed bags; a single-seed difference under 0.006 is nothing.
- Spectrum-blind audit at every milestone: rerun with all spectra replaced by one constant spectrum. The
  surviving MRR is prior. If a change raises the blind score on C2-NP by more than it raises the real score,
  it learned fame, not chemistry: reject.
- Never tune on the 250 NP-TIMS alone, never on the public board, never on the visible test file.

### 2.5 Models and cross-fitting

- Encoder and every neural model: trained on train minus holdout_v2 and never refitted on the holdouts (they
  are 2 percent of structures). This removes the "refit shifts the ranker's features" trap. For class 1 test
  molecules the encoder is then sharper than in validation, which only helps the right candidate.
- Ranker: 5-fold grouped by `fold`; groups are (molecule, cell); final ranker fitted on all groups, 5 seeds.
- Ranker training groups: NP-like sets carry at least 85 percent of the group weight. SYN-TIMS groups (C2-PC
  only) carry at most 15 percent; keep them only if an ablation shows NP cells do not fall (RESEARCH 7.2
  measured a loss from synthetic rows, KAGGLE_INTEL needs them for prior calibration: the ablation decides).

### 2.6 Estimating the class mix (owner R1, 6 to 8 CPU submissions over the first week)

Each probe is a submission restricted to one source; its public score is a linear equation in the shares.

| probe | list contains | measures |
|---|---|---|
| P1 | library hits only (cosine >= threshold), nothing else | pi_1 x local C1 MRR of the same rule (public community value 0.151) |
| P2 | NP tier by prior (done: 0.154) | fame of class 1 + class 2-NP; sets `w_obs` |
| P3 | best pipeline, NP-tier candidates only | pi_1 + pi_2np part |
| P4 | best pipeline, PubChem-only candidates (not in NP tier, no library spectra) | pi_2pc x MRR |
| P5 | only structures absent from PubChem and from the NP tier (generator output) | pi_3 x class-3 MRR, exactly: class 1 and 2 cannot score |
| P6 | full pipeline with and without external libraries | class-1 coverage gain |

Solve for the shares by non-negative least squares against the local cell MRRs of the same probe pipelines,
with a bootstrap using a board standard error of 0.03 per probe. Expect shares to +/- 0.08: enough to drop a
mix from the table, not to tune on. P5 is the only direct reading of class-3 progress and is repeated whenever
the class-3 pool changes materially. The public/private difference stays unknown; mix B covers it.

---

## 3. Work queue

Legend: MUST = needed for parity with the public stack; BET = aimed at the gap. Effort in agent-hours.
Compute: L = local 4 CPU (scarce), KC = Kaggle CPU kernel (4 vCPU, 31 GB, 12 h, no weekly quota: every job over
30 CPU-minutes goes here), KG = Kaggle GPU. Expected gains are public-board MRR under mix A.

Rule change for all packages: the inference path may import torch (preinstalled on Kaggle, CPU and GPU). A
CPU-only fallback must exist for every GPU stage, guarded by a time budget.

### 3.1 Overview

| id | package | type | class | expected gain | effort | compute | depends on |
|---|---|---|---|---|---|---|---|
| R1 | validation v2, LB-proxy, class-mix probes | MUST | all | decision quality | 6 | L (1 core) | WP7 (done) |
| R2 | external spectral libraries | MUST | 1 | +0.005 to +0.015 | 8 | L (2 cores) or KC | WP5 library index |
| R3 | PubChem documentation priors + Kaggle job runner | MUST | 2 | +0.03 to +0.07 with R5 | 5 | KC | WP3 tier |
| R4 | contrastive encoder E1 (in progress) | MUST | 2, 3 | carrier of most class-2 score | running | KG 2 x 8 h | bundle_v1 |
| R5 | candidate table, PubChem shortlist, feature store | MUST | all | enables R8; PubChem route | 12 | L + KC | R1; stubs until R3, R4 |
| R6 | analog evidence features (mass-shift propagation) | MUST | 2, 3 | +0.02 to +0.04 | 8 | L (2 cores) | library open index |
| R7 | MetFrag-lite fragment features + fragment mass calibration | MUST | 2 | +0.01 to +0.04 | 10 | L (1 core dev), KC | none |
| R8 | joint ranker v1, class-mix weights, rank-1 gate | MUST + BET | all | +0.03 to +0.06 over fusion | 12 | L | R1, R5; R3, R4, R6, R7 as they land |
| R9 | Kaggle inference kernel, time guards, extremes | MUST | all | no zero scores | 6 + ongoing | KC, KG | each milestone |
| R10 | neighbour network over test molecules | MUST | 2, 3 | +0.003 to +0.006 | 6 | L | R8 |
| R11 | forward models on top-K (GLACIER, ICEBERG) | MUST/BET | 2 | 0 to +0.015 | 10 | KG | R5, R8 |
| R12 | ICEBERG PubChem atlas: probe, then cache | BET | 2 | unknown, up to +0.03 | 4 probe, 10 build | network, KC | none (probe) |
| R13 | class-3 virtual-analog tier + GEN in the ranker | BET | 3 | +0.01 to +0.04 | 16 | KC build, L | generate.py, R5, R8 |
| R14 | encoder E2 trained with simulated spectra | BET | 2 | +0.02 to +0.05 | 14 | KG 10-25 h | R4, R11 or R12 |
| R15 | fingerprint-conditioned decoder (FRIGID official) | BET | 3 | +0.005 to +0.02 | 14 | KG | R4, R13 |
| R16 | fork benchmark on C2-ZEN (optional) | info | 2 | calibration | 3 | KG 4-8 h | R1, R2 |
| R17 | final fit, freeze, selection | MUST | all | - | 6 | L, KG | all |

### 3.2 Packages

**R1 - Validation v2, LB-proxy and probes.**
- Mechanism: section 2. Outputs: `splits/holdout_v2.csv`, `scripts/make_splits.py --v2`, cells and `w_obs` in
  `casmi/validation.py`, `scripts/evaluate.py --panel fast|full --mix A,B,C --blind`, `scripts/class_mix.py`
  (probe table to shares), a "Protocol v2" block in EXPERIMENTS.md with b0 and b1 rescored.
- Acceptance: b0 gives P(A) = 0.154 +/- 0.01 by construction; b1 (library + prior, WP8) predicted by P(A)
  within 0.03 of its public score (first out-of-sample check of the proxy); C3h removes a median of at least 3
  structures per molecule; fast panel from stored pools in under 15 min on 2 cores.
- Parallel with everything. Blocks R8 and all adoption decisions.

**R2 - External spectral libraries (class 1 coverage).**
- Mechanism: class 1 means "has public spectra", not "is in train". Add reference spectra of structures absent
  from train as LIB candidates with source and instrument flags; similarity thresholds calibrated per source.
- Inputs, fast path first: Kaggle datasets `taeyangg4/casmi26-extlib-gnps-full` (198,581 spectra, 147,414
  structures), `takumuhata/casmi26-extlib` (CC0), `samartalwar/...massbankharmonized` (drop `is_casmi_subset`),
  the `external_reference_spectra/` of the NP knowledge table; then primary sources where a dump is newer
  (GNPS, MassBank EU, MoNA, MSnLib, MassSpecGym, Enveda-180 Zenodo). Drop any NIST-derived record except the
  host-approved GNPS-NIST14-MATCHES.
- Outputs: `scripts/build_extlib.py`, `data/processed/extlib/` (same layout as the library, mkey per record,
  licence and source id per record), dataset `casmi-extlib`, `n_ext_lib` per holdout molecule for R1, the
  C2-ZEN query spectra, ARTEFACTS rows.
- Acceptance: at least 100,000 structures not in train; per source, the cosine and matched-peak threshold giving
  95 percent rank-1 precision for timsTOF queries of train structures present in both; C1 (NP-TIMS) not lower;
  on NP-OTHER molecules with external spectra and all train spectra hidden, MRR at least 0.80.
- Gain: +0.005 to +0.015 (public overlay: +0.006 to +0.009). Each molecule moved from class 2 handling to a
  clean library hit gains about 0.5.

**R3 - PubChem documentation priors, and the Kaggle job runner.**
- Mechanism: substance count is the strongest single prior in public (documentation prior 0.15 x [ln(1+SID) +
  ln(1+PMID) + ...]; removing PubChem candidates with it cost 0.055). We have only pmid_count and n_cids.
- Do: `scripts/kaggle_job.py` (push a CPU kernel with the code dataset, attached datasets and a command; fetch
  or publish the output; generalises `kaggle_encoder_kernel.py`). First job: attach
  `dmitriigluzdov/casmi26-pubchem-popularity-prior`, join `pc_lsid, pc_lpmid, pc_lpatent, pc_lmesh,
  pc_lsynonyms, pc_n_vendors, pc_n_depositors, pc_np, pc_lotus_refs` to our 120 shards by ik14, write
  `casmi-db-pubchem` v2 and the same columns for the NP tier. Do not take `pc_src_mask` (may encode NIST).
  Verify `sid_count` and `pmid_count` on 10,000 random CIDs against PubChem FTP (CID-SID) and keep the public
  build script next to ours in ARTEFACTS (data provenance).
- Acceptance: join coverage at least 97 percent of our tier rows; prior-only MRR on C2-PC and on weighted C2-NP
  above the pmid-only prior, including on the pmid_count = 0 subset; a prior-only probe submission (NP +
  PubChem in one ranking) above 0.154.
- Gain: with PubChem candidates in the pool (R5), +0.03 to +0.07. The prior weight is learned by R8 on
  `w_obs`-weighted groups, never hand-set from NP-TIMS.

**R4 - Encoder E1 (in progress; listed for its gates).**
- Peak-token transformer, focal fingerprint head + candidate-contrastive head with a molecule tower, queries of
  1-4 merged acquisitions, negatives from the 10 ppm window (bundle_v1).
- Acceptance: encoder score alone, NP-TIMS C2 over the NP tier, MRR at least 0.50 (debug under 0.45; the public
  FPNet reaches 0.47 to 0.53 on a 712k pool); C2-PC over full PubChem windows: report top-1, top-25 and MRR
  (target MRR at least 0.30, top-200 recall at least 0.85); focal + contrastive beats BCE on both.
- Outputs: `models/encoder/v1/`, dataset `casmi-models`, CPU inference under 2 min for 1,500 spectra, both
  scores (cosine, Bernoulli likelihood) exposed to R5. No refit on holdouts (2.5).

**R5 - Candidate table, PubChem shortlist and feature store (continues WP10).**
- Mechanism: one table per (molecule, cell): LIB + NP + PubChem (+ GEN later) candidates with all features,
  cached to parquet so that ranker experiments never recompute features. PubChem windows (median 10,470
  candidates) get fingerprints on the fly (`chem.fp_mol`, about 1 ms each, 4 processes) and a two-pass
  shortlist: every NP-tier and LIB candidate is kept; PubChem-only candidates are kept if in the top 300 by
  encoder score or the top 100 by prior within their formula's top group. Never prune by library similarity.
- Features: library match (4.2), encoder cosine and likelihood with in-pool z-scores and rank, priors (R3),
  mass error with the fitted timsTOF error shape, formula evidence (explained intensity per candidate formula
  pooled over spectra and adducts, NP-tier formula frequency; `casmi/formula.py` exists), source flags, pool
  context (pool sizes per source, best and second-best library similarity, number of LIB candidates, gap
  between the top two encoder scores, number of same-formula isomers), slots for R6, R7, R11, R12, R13.
- Outputs: `casmi/features.py`, `scripts/build_features.py`, `data/processed/features/<panel>/<cell>.parquet`,
  stage `features` in the pipeline registry, a KC job for the full panel.
- Acceptance: shortlist keeps the truth for at least 95 percent of molecules whose truth is in the pool (C2-PC
  and C2-NP); 400 test molecules featurised in under 45 min on a Kaggle CPU kernel; fast panel rebuilt in
  under 60 min; byte-identical tables across two runs.
- Milestone M1 comes out of R5 with the fallback fusion (PLAN 4.6 with R3 priors and encoder z): target P(A)
  at least 0.36, public at least 0.34.

**R6 - Analog evidence features.**
- Mechanism: the core of the public 0.34 pipeline and of Engine 2. For each query, the top 80 spectral
  neighbours by entropy similarity and modified cosine within 200 Da (open index exists); for each candidate,
  sum over neighbours of similarity^3 x Tanimoto(candidate, neighbour structure), the best such term, the
  number of neighbours whose mass difference to the query is explained by one of the 74 edit rules applied to
  the candidate, and the MS2Query-style mean similarity to spectra of the 10 library molecules most similar to
  the candidate. Masks apply per cell (C3h also removes near-duplicate neighbours).
- Outputs: `casmi/analog.py`, feature columns in R5, neighbour lists cached for R10 and R13.
- Acceptance: analog score alone, weighted C2-NP MRR at least 0.25; added to encoder + prior in the fallback
  fusion, P(A) +0.02 or more; under 1 s per molecule on one core.

**R7 - MetFrag-lite fragment features and fragment mass calibration.**
- Mechanism: isomers are about 98 percent of database-route errors and only structure-level fragment evidence
  separates them (public AUC 0.757 against 0.500 for formula-level). Numba bond-breaking of 1 and 2 bonds with
  ring handling, H shifts -2..+3, tolerance 0.005 Da, linear intensity, 2-bond cuts weighted 0.6 (the settings
  that held out of sample in public); features per candidate over all spectra and adducts: explained intensity,
  explained peak count, weighted score, mean cuts per explained peak, in-formula-group z-score and rank.
  No 3-bond cuts, no hand-written bond propensities (both lost in public).
- Calibration sub-task: the host reported a +0.49 mDa error on positive-mode fragments of enveda-180 that was
  corrected in the rerun test. Measure fragment mass residuals against sub-formula masses on enveda-180,
  NP-TIMS and the visible test, per polarity; if a systematic offset exists in train but not in test-like
  data, add a correction at ingestion (`spectra.py`, config flag) and report its effect on library cosine.
- Outputs: `casmi/fragment.py`, feature columns in R5 for the top 100 candidates per molecule.
- Acceptance: AUC truth versus same-formula decoys at least 0.72 on C2-PC and C2-NP; as one extra term in the
  fallback fusion, net rank-1 wins minus losses at least +3 percent of molecules on both cells; at most 3 s per
  molecule for 100 candidates on one core.

**R8 - Joint ranker v1 (continues WP11). The main edge.**
- Mechanism: one LightGBM lambdarank over the whole pool (LIB + NP + PubChem, GEN from R13 on), groups =
  (molecule, cell), group weights = mix A shares x `w_obs`, truncation level 25, monotone constraints on count
  priors, 5-seed bag. Pool-context features let it learn what the public stack hard-codes: when a library hit
  is decisive, when the prior should fade, that for a class-2 or class-3 molecule any candidate with reference
  spectra that do not match is wrong, and where generated candidates belong.
- Rank-1 gate (our version of "leader-safe"): a base ranker on robust features (library, prior, encoder,
  analog) names the leader; the full ranker may replace it only when its margin exceeds a threshold fitted
  out-of-fold so that replacements are right at least 60 percent of the time (break-even is 50 percent).
  Report flips won and lost per cell. Ranks 2-25 always follow the full ranker.
- Also fit: a molecule-level novelty score (probability that the truth is in no tier, from pool-context
  features) as an input feature and for R13's slot policy.
- Outputs: `casmi/ranker.py`, `scripts/train_ranker.py`, `models/ranker/v1/`, stage `rank`, EXPERIMENTS rows
  with the feature-group ablation (drop one group at a time) and the SYN-groups ablation (2.5).
- Acceptance: P(A) at least +0.03 over the fallback fusion on the same features; min over mixes not lower; C1
  (NP-TIMS) at least 0.93; blind audit passes; seed standard deviation under 0.006.
- Milestone M2: target P(A) at least 0.45, public at least 0.44.

**R9 - Inference kernel and robustness (continues WP14).**
- One private notebook `casmi-inference` for all configs; stage time guards with a total budget of 3 h CPU
  (GPU stages optional, each with a CPU fallback); per-molecule try/except down to the b0 answer.
- Tests before every milestone submission: synthetic extremes (1,159 Da, 16 spectra, each of the 10 adducts,
  a molecule in no tier, a single 3-peak spectrum, a list-valued collision energy), a commit run on the visible
  test equal to the local file byte for byte, peak RSS under 20 GB.
- Read the GPU quota meter before and after the fork's scoring run to learn whether scoring reruns consume the
  weekly quota; record it in KAGGLE.md. If they do, the default inference path stays CPU.

**R10 - Neighbour network.**
- Mechanism: test molecules come from the same samples. Link test molecules (modified cosine >= 0.6, >= 6
  matched peaks, same polarity) and 5 to 10 train anchors per query; re-sort ranks 2-25 by z(ranker score) +
  lambda x z(support), support = sum over neighbours of cos^4 x best Tanimoto to the neighbour's top 5
  (lambda 0.35 in public). Extension: exact edit consistency (when two linked molecules differ by a rule mass,
  reward candidate pairs related by that rule). The leader moves only through the R8 gate.
- Validation: batches of 400 holdout molecules drawn with mix A and mixed cells; kill the edit extension if
  fewer than 5 percent of molecules have an edit partner. Output: `casmi/network.py`, stage `network`.
- Acceptance: batch P(A) not lower, at least +0.003, tail MRR up; then adopt (public evidence: +0.005 twice).

**R11 - Forward-model features on the top-K.**
- Official ms-pred GLACIER and ICEBERG checkpoints trained on MassSpecGym, with their training code (eligible).
  Features for the top 30 same-formula candidates per molecule at each query's adduct and energy: entropy
  similarity, matched intensity, matched peaks; collision-energy offset fitted on enveda-180.
- Warning from public: +0.02 on class 2 but about 0 on structures outside MassSpecGym. So acceptance is
  measured only on C2-ZEN and SYN-TIMS structures absent from MassSpecGym: net rank-1 wins at least +3 percent
  of molecules and P(A) +0.005 through R8; otherwise drop the features and keep the package only as the
  simulator for R14. Inference at most 30 min on a T4, skipped under the CPU fallback.

**R12 - ICEBERG PubChem atlas.**
- Facts (probed today): 91,424,208 PubChem structures, [M+H]+ and [M-H]- predictions, per-formula MGF files on
  a read-only WebDAV (`https://iceberg-ms.mit.edu/dav/public/h_plus|h_minus/C9/O4/H8/C9H8O4.mgf`), login by a
  site account, NIST'23 structures excluded, Orbitrap conditions; licence and total size not stated.
- Probe (4 h, network only): account and terms; bytes and structures per formula for 50 formulas across the
  mass range; throughput; which energies are stored. On NP-TIMS, C2-ZEN and 300 SYN-TIMS molecules, download
  their formulas and measure (a) entropy similarity of timsTOF queries to the prediction of the true structure
  against same-formula decoys (AUC, top-1 within formula), (b) the gain as one feature through R8.
- Proceed to the cache if AUC is at least 0.75 and top-1 within formula at least 0.30 on molecules outside
  MassSpecGym: stream formulas in priority order (formulas of the NP tier, then PubChem formulas by NP-tier
  mass density), compact to the top 24 peaks (uint32 m/z, uint8 intensity, about 120 bytes per spectrum),
  mass-sharded like the PubChem tier, dataset `casmi-atlas` (target at most 40 GB). At inference this gives a
  forward-model score for every same-formula candidate with no GPU, usable inside the shortlist, which the
  top-30 rescoring of the public stack cannot do. It is also the simulated-spectra source for R14.
- Kill: AUC under 0.65, or under 2 MB/s sustained, or terms that forbid redistribution inside a private Kaggle
  dataset. Fallback: only the 854k NP-tier structures (for R14).

**R13 - Class 3: virtual-analog tier, GEN candidates in the ranker, slot policy.**
- Reasoning: class 3 may be 25 to 48 percent of the test; public class-3 MRR is 0.03 to 0.08 and no public
  generator works. The task text says the novel molecules are "hypothesised natural products" and "natural
  product analogs". The public generators edit only the 6 to 20 nearest spectral parents; reach, not ranking,
  is their first limit (46 percent with close analogs, 16.5 percent without).
- Mechanism: turn generation into database lookup. Offline, apply the 74 rules of `casmi/generate.py` once at
  every site to every NP-tier structure and every NP-library train structure (about 0.9M seeds), keep products
  that are in neither PubChem nor the NP tier, store ik14, SMILES, mass, seed row and rule id in mass-sorted
  shards (`data/processed/db/va/`, dataset `casmi-db-va`; expect 20M to 60M rows, 1 to 3 GB). Second source to
  evaluate in the same format: the 67M natural-product-like database (Tay et al. 2023, Scientific Data, CC BY;
  verify the deposit and licence first). At inference a second edit step is applied on demand to the best
  20 seeds, as today.
- Ranking: VA and GEN candidates enter the R8 pool with encoder scores, fragment features, atlas-free
  features only, plus parent evidence (the seed's spectral similarity if it has library spectra, the seed's
  rank in the pool at mass minus delta, rule frequency among known NP pairs, analog-neighbour support from
  R6). The ranker is trained on C3h groups that contain the truth among VA candidates and on C1/C2 groups that
  contain VA distractors, so the interleaving of database and generated candidates is learned, using the
  novelty score of R8.
- Outputs: `scripts/build_va.py` (KC jobs, seeds sharded over 4 kernels), `casmi/va.py`, masks by seed row for
  C3h, probe submissions P5.
- Acceptance: truth in the VA + GEN pool for at least 25 percent of C3h molecules and 45 percent of C3e; C3h
  MRR at least 0.08; C1 and C2 cells not lower by more than 0.005; P5 on the board at least 0.016 (two class-3
  molecules at rank 1, or equivalent). If two differently built P5 probes score 0.000, class 3 becomes tail
  insurance (ranks 15-25 only) and R15 is cancelled.

**R14 - Encoder E2 with simulated spectra.**
- Mechanism: class 2 is inside PubChem or COCONUT, so its candidate space can be simulated before the test;
  the only large leak-free gain in the audited literature (MIST 9.6 to 40.3 percent top-1). Train E1's recipe
  on real + simulated pairs with an "is simulated" input flag (set to real at inference), starting at 1:1.
- Source of simulations, in order of preference: the atlas cache (R12; no GPU), else GLACIER from R11 over the
  NP tier (854k structures x 2 adducts x 3 energies, 6 to 10 h T4).
- First experiment (one session): only the mass-window neighbours of the fast-panel molecules (about 60k
  structures); held-out structures keep their simulated spectra and lose their real ones, as at test time.
  Proceed to the full tier at +0.03 MRR on C2-NP and C2-PC encoder-alone; kill under +0.01.
- Adoption needs the ranker refitted on E2 features (R8 rerun) and P(A) +0.01.

**R15 - Fingerprint-conditioned decoder.**
- FRIGID-base official checkpoints (host-approved), driven by E1's fingerprint posterior and the top 2 formulas;
  outputs filtered by formula, de-duplicated, added as GEN candidates with generation frequency as a feature,
  the 10 most frequent expanded by the zero-delta positional rules.
- Stage A, oracle fingerprints and formulas on 300 NP-like holdouts: at least 25 percent in the top 10, else
  stop. Stage B, E1 posterior on C3h: truth among 25 outputs for at least 8 percent, else stop. Inference at
  most 60 min on a T4. Starts only after R13's first P5 probe is positive or R13 reach is under 15 percent.

**R16 - Fork benchmark (optional, one GPU session).** Run the fork's commit path on 300 C2-ZEN molecules in
test format (they are in none of its training data; check its external library for them first). Gives the
public stack's class-2 PubChem-route MRR to compare with ours on identical molecules. Skip if GPU is short.

**R17 - Final fit and freeze.** Ranker on all groups, 5 seeds; thresholds and gates frozen from out-of-fold
values; two configs (section 4.4); three identical end-to-end runs; ARTEFACTS, SUBMISSIONS and a reproduction
script (`scripts/run_all.sh`) complete, so the pipeline can be published if required for eligibility.

### 3.3 Order and parallelism

At most 4 agents at once; local CPU is the constraint, so each wave has at most 2 local-CPU-heavy packages.

| wave | dates | packages (compute) | exit |
|---|---|---|---|
| 0 (running) | now | WP8 finish and submit b1 (L), R4 E1 run 1 (KG) | b1 on the board; E1 gate |
| 1 | 11-21 to 11-23 | R1 (L light), R2 (L 2 cores), R3 (KC), R7 (L 1 core); as R1 and R3 finish: R6 (L 2 cores), R12 probe (network) | protocol v2 live; priors, extlib, fragment and analog features exist |
| 2 | 11-23 to 11-27 | R5 (L + KC, starts when R1 lands, stubs for missing features), R8 (starts on R5's first tables), R9, R13 build (KC) | M1 (R5 + fusion), then M2 (ranker v1) |
| 3 | 11-27 to 12-02 | R10 (L), R11 (KG), R13 ranking (L), R12 cache (KC, if the probe passed), E1 run 2 or R14 experiment (KG) | M3 = parity; first P5 probe |
| 4 | 12-02 to 12-09 | R14 (KG), R15 (KG, conditional), R16 (optional), R8 refits with each new feature block | M4, M5 |
| 5 | 12-09 to 12-14 | R17, robustness, final selection; no new features after 12-11 | two finals selected by 12-13 |

Critical path: R1 -> R5 -> R8 -> M2. R3, R4, R6, R7 feed R5 as columns and can land late without blocking it.

### 3.4 GPU budget (about 100 h in total, about 25 h left this week after the fork run)

| use | hours |
|---|---|
| E1: two training runs | 16 |
| R11 packaging and evaluation | 6 |
| R14: simulation (0 if the atlas works) + two training runs | 18 to 28 |
| R15 stages A, B and inference tests | 10 |
| R16 | 0 to 8 |
| milestone submissions with a GPU stage (if reruns consume quota), 1 h each | 10 |
| reserve | 12 |

---

## 4. Submission plan

### 4.1 Milestones that produce a submission

| milestone | content | local gate | public target | date |
|---|---|---|---|---|
| M0 | b1: library match + prior (WP8) | proxy check of R1 | about 0.20 to 0.28 | 11-21 |
| M1 | + PubChem priors, encoder E1, PubChem shortlist, fallback fusion | P(A) >= 0.36 | >= 0.34 | 11-24 |
| M2 | + analog, fragment, external libraries, ranker v1 with gate | P(A) >= 0.45 | >= 0.44 | 11-27 |
| M3 | + network, forward models or atlas feature, VA tier in the tail | P(A) >= 0.49 | within 0.015 of the fork | 12-02 |
| M4 | + class-3 pool promoted by the ranker, atlas shortlist | P(A) >= M3 + 0.02 | >= 0.51 | 12-06 |
| M5 | + encoder E2, decoder if alive | P(A) >= M4 + 0.02 | >= 0.53 | 12-10 |

Probes P1 to P6 (section 2.6) are extra CPU submissions; they never count as milestones. With 5 submissions
per day and 55 minutes between them there is no shortage; GPU-stage submissions are the only costly ones.

### 4.2 What "better" means

1. Local first: the adoption rule of 2.4. Only locally adopted changes are submitted (probes excepted).
2. Board second, as a veto, not as a selector. Two submissions on the same 125 molecules differ by chance with
   a standard error of about 0.016. A locally adopted change is kept unless the board is worse by more than
   0.02; in that case submit one variant that isolates the change, and if that is again worse by more than
   0.02, revert and record a protocol failure in EXPERIMENTS.md (the proxy needs fixing before the next step).
3. A board gain without a local gain is never adopted.
4. After each milestone compare the predicted P(A) with the public score. A persistent offset is expected
   (the mix is uncertain); what must hold is that the ordering of our own milestones agrees. If two
   consecutive milestones disagree in sign with differences above 0.02 on both sides, stop feature work and
   repair the protocol (re-estimate the mix with fresh probes).

### 4.3 The fork as a benchmark

- Its public score (pending) is recorded once as F. Expected 0.495; its private expectation is F - 0.01 to
  F - 0.02 because its constants were selected on this board.
- "Parity" for us means public >= F - 0.015 with P(A) and the fast-panel class cells on record. Because our
  constants are not tuned on the board, parity on public is expected to be an advantage on private.
- No fork variants are submitted; no further GPU hours go to the fork except R16.

### 4.4 The two final selections

- Final 1: the config with the best P(A) among those whose min over mixes A, B, C is also the best or within
  0.003 of the best, 5-seed bagged, with every gate at its out-of-fold value.
- Final 2, by this decision list:
  1. If our best public score is below F - 0.02 on 12-12: Final 2 is the unchanged fork (accepting its
     removal risk for that slot only).
  2. Else if the P5 probe showed class-3 hits: the same pipeline fitted with mix B group weights (generated
     candidates placed more aggressively). The two finals then bracket the class-3 share.
  3. Else: the same pipeline with count priors capped (ranker trained with `w_obs` at alpha + 0.5) and the
     transductive network off, as the low-variance variant.
- Both finals must have passed the R9 extremes and a full Kaggle run within 50 percent of the time limit.
- Private-board robustness is the criterion: never pick between two of our own configs by a public difference
  under 0.02.

---

## 5. Risks and kill criteria

| # | risk | signal | action |
|---|---|---|---|
| 1 | Parity not reached in time | M2 public under 0.40 on 11-28, or M3 under F - 0.04 on 12-04 | stop all BET packages except R13's tail insurance; all agents on R5/R8 feature gaps found by the ablation; fork becomes Final 2 |
| 2 | Local proxy does not track the board | 4.2 rule 4 fires | freeze features, run fresh probes, refit `w_obs` and the mix |
| 3 | Encoder E1 weak | NP-TIMS C2 encoder-alone under 0.45 after run 2 | train the sanpier recipe (Apache 2.0, full training code) with our data loader as a second arm; keep the CPU MLP as a third view |
| 4 | Prior overfits fame | blind audit C2-NP rises faster than the real score; C2-PC falls | raise alpha, add monotone caps, drop membership flags that fail on pmid_count = 0 (public saw curated flags fail on the board) |
| 5 | Learned rank-1 changes lose (the public experience, down to -0.11) | flips right less than 60 percent out-of-fold, or a board drop above 0.02 at M2 | tighten the gate until flips are at least 65 percent right; the base leader always remains available |
| 6 | Class-3 work yields nothing | two P5 probes at 0.000, or VA reach under 15 percent on C3h | class 3 becomes tail insurance (ranks 15-25); cancel R15; move the effort to R12/R14 |
| 7 | Generated candidates displace right database candidates | C1 or C2 cells fall by more than 0.005 with VA on | cap VA/GEN candidates at rank >= 3 unless the novelty score is above its 90 percent precision point |
| 8 | Forward models only memorise MassSpecGym | no gain on C2-ZEN and SYN structures outside MassSpecGym | drop R11 features; keep the simulator for R14 only |
| 9 | Atlas unusable (terms, size, throughput, domain gap) | R12 kill criteria | NP-tier subset only, or GLACIER simulation on GPU |
| 10 | Simulated-spectra training does not transfer to timsTOF | under +0.01 after one attempt with the flag and shared augmentation | stop R14; keep E1 |
| 11 | Runtime or memory failure on the hidden test | any stage above its guard in a commit run; "Notebook Threw Exception" | per-stage budgets, CPU fallbacks, per-molecule fallback to b0; rerun R9 extremes |
| 12 | Eligibility | any artefact whose training code or data licence is unclear | not in a final: no NIST-derived data, no `pc_src_mask`, no third-party weights except official ms-pred and FRIGID checkpoints; provenance of every dataset in ARTEFACTS.md |
| 13 | Local CPU contention slows everything | fast panel above 30 min, agents waiting | move feature builds to Kaggle CPU kernels through `scripts/kaggle_job.py`; at most 2 CPU-heavy local jobs |
| 14 | Train and test fragment masses differ (electron-mass correction) | R7 calibration finds a polarity-dependent offset | correct at ingestion, rebuild the library, retrain E1 only if encoder inputs shift by more than its augmentation jitter (3 ppm) |
| 15 | Public and private splits differ in class mix | not observable | finals bracket the class-3 share (4.4); decisions use the min over mixes |

---

## 6. Decisions needed from the coordinator or the user

1. Confirm the strategic call: own pipeline for the finals, fork as Final 2 only under rule 4.4.1.
2. Publication: the reproducibility rule speaks of "our own published code". Standing orders keep everything
   private. Is publishing the code after the deadline acceptable if a prize or ranking requires it?
3. ICEBERG atlas: it needs a site account (an email address) and its terms are not stated; the model behind it
   is not documented on the page (NIST'23 structures are excluded from the files). The host approved the atlas
   by name; keep the forum post link in ARTEFACTS.md. Who registers the account?
4. Allow torch in the inference path (AGENT_RULES change) and GPU stages in milestone submissions.
5. Approve 6 to 8 probe submissions (section 2.6) during the first week.
6. Approve use of public Kaggle datasets as data sources (prior table, external libraries; CC BY-NC-SA and
   "Other" licences), with provenance recorded.
7. Report the GPU quota before and after the fork's scoring run (does a scoring rerun consume quota?).

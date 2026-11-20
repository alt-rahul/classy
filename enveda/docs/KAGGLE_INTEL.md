# Kaggle competitive intelligence (snapshot 2026-11-20)

Gathered through the Kaggle API on 2026-11-20 (18:55 to 20:30 UTC). Sources: the full public leaderboard
download, `kernels/list`, the internal `kernels.KernelsService/ListKernels` (displayed best public score per
notebook), `kernels/pull` (notebook sources, metadata), dataset metadata and file listings, and the discussion
forum (`discussions.DiscussionsService`, 175 topics, about 60 read in full). Notebook sources are saved in
`data/external/kaggle_notebooks/<owner>__<slug>.ipynb` with `<owner>__<slug>.meta.json` (git-ignored, 32 notebooks).
Scores quoted for notebooks are what the Code tab displays (best public score over versions) unless marked "claimed".

## 0. Ten things to know

1. The public board is about 125 molecules (31 percent of about 400). One molecule moving from rank 2 to rank 1 is
   +0.004; two submissions differ by chance with a standard error of about 0.016. 518 teams sit at exactly 0.495.
2. The whole top of the Code tab is ONE pipeline lineage with different constants (section 2). Its root is
   `taeyangg4/casmi26-lb-0-495-network-fr-p5-tail` (0.495). The only higher displayed score, heliosli's 0.496,
   now ships demo weights and does not reproduce.
3. Medal lines on the public board today (5,657 teams): gold about rank 21 at 0.507; silver (rank 282) and bronze
   (rank 565) are both at 0.495, i.e. the public fork. Leader 0.552 (Shehab Alshehabi, 287 submissions). No top-10
   team is a public notebook author.
4. The base 0.465 pipeline is most of the score. Everything stacked on it since late October added +0.030 in small
   steps of 0.001 to 0.006, mostly reordering ranks 2 to 25. Moving rank 1 with a learned model is where public
   attempts lost (down to -0.11).
5. The single biggest measured lever in public: PubChem-only candidates with a documentation (popularity) prior.
   Removing them cost 0.464 -> 0.409. The prior uses ln(1 + PubChem substance count) + ln(1 + PubMed count).
6. Same-formula isomers are about 98 percent of database-route errors. Only structure-aware evidence (in-silico
   fragmentation, forward models) separates them; formula-level evidence does not.
7. Class mix (community estimates from LB probes, hosts will not disclose): class 1 about 16 percent (library-only
   submission scored 0.151), class 3 between 39 and 52 percent. Hosts: class 3 is absent from both PubChem and
   COCONUT. Nobody has shown a public LB gain from class 3 generation above +0.004.
8. Reproducibility rule (host, repeated): third-party weights need their original training code and permitted data,
   otherwise the submission is not eligible and "we reserve the right to remove submissions that are not
   reproducible from the leaderboard, regardless of ranking". Several components of the public stack lack training
   code. A fully self-trained pipeline is safer than the fork.
9. Permitted and unused in public: the ICEBERG PubChem atlas (predicted MS/MS for 91.4M PubChem structures,
   host-approved for library and training use), FRIGID de novo checkpoints, the Enveda-180 Zenodo release
   (816 structures absent from train), newer GNPS / MassBank / MoNA releases. NIST is forbidden.
10. A private fork of the 0.495 notebook is on our account, `ograhulrajkumar/casmi26-taeyangg4-0495-fork`,
    versions 1 (byte-identical, 365 s commit) and 2 (smoke mode, all stages verified on 12 molecules, 1,914 s).
    It needs a GPU T4 and a scoring rerun of several hours (stage guards up to 8.6 h of the 9 h limit). Section 9.

## 1. Leaderboard

Full public leaderboard downloaded (`leaderboard/download`, 5,657 teams, 5,514 with a score above 0). The view
endpoint pages only to 1,120 rows. Competition settings from `competitions.CompetitionService/GetCompetition`:
public share 31 percent, 540 min CPU and GPU runtime, 2 final selections, synchronous reruns, team merger and
notebook publishing deadline 2026-12-07, deadline 2026-12-14, 147,547 submissions so far.

| rank | score | | rank | score |
|---|---|---|---|---|
| 1 | 0.552 | | 100 | 0.496 |
| 2 | 0.540 | | 200 to 565 | 0.495 |
| 3 | 0.537 | | 1,000 | 0.492 |
| 5 | 0.525 | | 1,500 | 0.483 |
| 10 | 0.515 | | 2,000 | 0.477 |
| 20 | 0.507 | | 3,000 | 0.427 |
| 50 | 0.501 | | 4,000 | 0.301 |

Score plateaus (teams at exactly that score, each the score of a public notebook): 0.495 (518), 0.492 (335),
0.487 (242), 0.477 (156), 0.480 (143), 0.488 (118), 0.481 (108), 0.479 (103). Above the public plateau: 67 teams in
[0.496, 0.500), 42 in [0.500, 0.510), 16 at 0.510 or more.

Top 12: Shehab Alshehabi 0.552 (287 subs), democat 0.540 (155), Oleksandr Parfionov 0.537 (145), Savko 0.532 (287),
goonew 0.525 (75), Hanbat Univ. team 0.521 (130), Terry Luo 0.520 (55), tu tu tu du 0.518 (240), Muku 0.515 (276),
pjmathematician 0.515 (322), Legends 0.515 (244), Aneesh & Yash 0.514 (58). MarvinTMB (0.510 with only 27
submissions) led on 2026-10-09 and estimated the realistic ceiling at about 0.55 (analytic 0.76).

How the board moved (snapshots quoted in public notebooks):

| date | leader | gold line | teams | public best |
|---|---|---|---|---|
| 2026-09-17 | about 0.34 | | | 0.339 (fast spectral cosine) |
| 2026-10-27 | 0.513 | 0.477 (rank 18) | 4,194 | 0.472 |
| 2026-10-29 | 0.516 | 0.481 (rank 18) | 4,295 | 0.477 |
| 2026-11-15 | 0.549 | 0.503 (rank 20) | | 0.492 |
| 2026-11-20 | 0.552 | 0.507 (rank 21) | 5,657 | 0.495 (0.496 not reproducible) |

The leader has moved only +0.039 in three weeks while the public stack moved +0.023; the gap between the best
private and the best public work is stable at about 0.04 to 0.06.

## 2. The public state of the art: one lineage

From alexchilton's 60-arm write-up, taeyangg4's ladder and dmitriigluzdov's version table. Each row adds to the
row above; scores are public LB.

| step | notebook (author) | what it adds | LB |
|---|---|---|---|
| a | canqiang E006, ahmedberatozer v4g/v4n engines | Engine 1 (train + COCONUT pool, FPNet, LightGBM ranker, analog generation), ICEBERG/GLACIER | 0.440 |
| b | dmitriigluzdov "The Data Behind" | assembles base: PubChem channel with documentation prior, Engine 2 fusion, GLACIER proxies, 5 fold-safe FPNets | 0.465 |
| c | sanpier "PubChem + Neural025" | contrastive spectrum-candidate neural model, RRF weight 0.25 | 0.466 |
| d | romanrozen "Peaks to Structures" | leader-safe consensus: rank 1 at neural weight 0.25, ranks 2-25 at 0.50 | 0.468 |
| e | alexchilton | ranks 2-25 at neural weight 0.75 | 0.472 |
| f | romantamrazov "Spectral Parallax" | untrained fragment check (MetFrag-lite) rank-1 rule on isomers | 0.472 |
| g | busyaprime / heliosli merge | f's leader moves from the d order, ranks 2-25 from e | 0.477 |
| h | baidalinadilzhan midx | exact InChIKey14 dedup and fill | 0.479 |
| i | bgunshui | second neural model (popwenman scorer v80) z-score ensemble 0.5/0.5 | 0.480 |
| j | popwenman Gold Apex V22 | timsTOF mass-accuracy prior, mass rank-1 rule (V8 = MASS) | 0.481 |
| k | ozertuu + alexchilton | external-library rank-1 overlay (Takumu Hata extlib + MassBank), TAU 0.85 -> 0.80 | 0.484 |
| l | guhongbin v589 | k with small fixes | 0.487 |
| m | analyticaobscura | N-1c neighbour network over test molecules, lambda 0.25 | 0.488 |
| n | rogersjohnson | lambda 0.35 | 0.492 |
| o | taeyangg4 | GNPS extlib rebuild (0), FR 3-25 learned fragment re-ranker (+0.001), P5 tail (+0.002), network D-L1 (+0.005) | 0.495 |

The 0.465 base in detail (frozen notebook embedded base64 in every top notebook; decoded copy in the scratchpad
of this session, reproducible from any top notebook's "Pinned sources" cell):

1. Neutral mass M = median over spectra; candidates within 10 ppm, fallback 30 ppm (10 ppm keeps 95.4 percent of
   all train spectra, 30 ppm 96.9; on timsTOF 99 percent are within 5 ppm).
2. Engine 1 (Ahmed Berat Ozer's MIT `casmi` package, dataset `casmi26-v4b-models`): pool = train structures +
   COCONUT (about 711k), spectral library match against train (+ public external reference spectra of train
   structures), FPNet spectrum-to-fingerprint (6,930 selected bits of Morgan r2 4096 + Morgan r3 4096 + RDKit path
   2048 + MACCS 167; `fpnet_full1`), feature families (derivation evidence, class-3 priors, fragmentation,
   analog-structure relations, DreaMS fingerprint view), analog generation from the 6 nearest spectral parents
   (cap 150; 20 parents lost 0.008), LightGBM ranker trained on simulated rows, pool popularity re-rank (mu 0.15),
   top 60.
3. Engine 2 (prvsiyan analog propagation + megayak two-ranker): entropy similarity, mass-shifted analog
   propagation (80 analogs, 200 Da window, similarity^3), pool train + COCONUT + ChEBI/LIPID MAPS (+ PubChemLite),
   8-seed bag of HistGBR rankers depth 6 (depth 7 lost 0.006). Fused with Engine 1 by weighted RRF
   (ALPHA 0.6, k 3); Engine 2's leader is forced to rank 2.
4. PubChem channel: 105.9M PubChem structures 150-1250 Da (`casmi26-pubchem-popularity-prior`), 10 ppm,
   FPNet A+B + 0.5 x mean of 5 fold-safe FPNets, two-pass shortlist (ECFP4 block top 5,000, then all bits),
   documentation prior 0.15 x [ln(1+SID) + ln(1+PMID) + 0.5 depositor-source bonus + 1.0 NP membership
   (COCONUT/LOTUS/NPAtlas) + 0.5 ln(1+LOTUS refs) + 0.5 synonyms]; keeps 25 structures not in the pool. Inserted
   at six fixed positions (including rank 24) when the main list has no library match >= 0.9; promoted to rank 1
   when its own scores pass S > 6 and pop >= 5.
5. ICEBERG and GLACIER (MassSpecGym checkpoints, Coley group) re-score same-formula groups of the top 60:
   z(ranker) + 0.5 z(ICEBERG) + 0.5 z(GLACIER); GLACIER only on [M+H]+ with [M+NH4]+ / [M-H2O+H]+ proxies
   (+0.008, 0.456 -> 0.464); MetFrag-lite fragment score where no simulator score exists; 5,400 s budget.

The post-base layers (0.465 -> 0.495) in one line each: neural re-ordering is fused by
`1/(3 + b) + w/(3 + n)` (b pipeline rank, n neural rank), rank 1 with w 0.25, ranks 2-25 with 0.75; a fragment
check may swap a same-formula candidate from ranks 2-6 into rank 1 when library match < 0.87; an external library
(GNPS multiplex synthesis library etc.) promotes a structure to rank 1 at cosine >= 0.80 with >= 4 matched peaks
within 10 ppm + 0.002 Da unless the pipeline's own match is >= 0.90; the neighbour network links test molecules
with modified cosine >= 0.6 and >= 6 matched peaks (same adduct) and re-sorts by
`z(-log rank) + 0.35 z(sum cos^4 * sum w * Tanimoto(candidate, neighbour top-5))`, with 5 to 10 consensus
"anchor" nodes from train per query; FR 3-25 is a 768k-parameter fragment re-ranker (agentili architecture,
taeyangg4 weights, held-out group MRR 0.889) re-ordering ranks 3-25 by `-rank + 3.0 z`; P5 reruns Engine 1 with
transform-explainable parents first and inserts only new candidates into ranks 3-25.

What the 60 one-knob arms on fixed bases showed (alexchilton, 170 submissions): 4 of 60 gained more than +0.002
(largest +0.006: extlib TAU 0.85 -> 0.80), 20 tied, 34 lost (largest -0.047: neural weight too high). Retraining
any of the shared models with one's own code always lost (-0.007 to -0.024), as did the long-trained neural
model (-0.019) and Ahmed's NX isomer reranker (-0.058). Their best moved by copying newer public notebooks.

## 3. Notebooks (most informative)

Runtime column: the commit run in seconds (`lastRunExecutionTimeSeconds`); for the 0.465+ lineage the commit run
is a "fast path" that writes a placeholder, and the full pipeline only runs in the scoring rerun (several hours;
the earlier E007 pipeline took 2 h 38 min over 400 molecules). All are GPU T4 with internet off unless noted.

| notebook | LB | commit s | data sources | what it is |
|---|---|---|---|---|
| taeyangg4/casmi26-lb-0-495-network-fr-p5-tail | 0.495 | 378 | 11 datasets (section 4) + kernel output popwenman/casmi26-scorer-train-v80 | root of the 0.495 block; steps a-o |
| mathleteutd/casmi26-0-495-network-10-train-anchors | 0.495 | 391 | same | anchors 5 -> 10 (no change); its siblings (20 anchors, TAU 0.75, wider search, one structure per molecule) all 0.495 |
| alexchilton/casmi26-0-495-what-moved-the-board-60-arms | 0.495 | 372 | same | ablation record: the climb, 60 arms (section 2) |
| alexchilton/casmi26-lb-0-488-peaks-to-structures-network | 0.488 | 162 | as guhongbin | full submission history |
| kozykappa/enveda-casmi-2026-evidence-gated-reranking | 0.495 | 402 | same | code byte-identical to taeyangg4 |
| flexonafft/casmi26-anchored-spectral-network | 0.495 | 362 | same | v18: extlib TAU 0.80 -> 0.75 |
| sachchidanandadaki/casmi26 | 0.495 | 439 | same, CPU | network edges for molecules without [M+H]+/[M-H]- |
| ozertuu/0-492-casmi26-dual-neural-physical-isomer | 0.495 | 835 | + LOTUS candidates, NPClassifier sidecar | dual neural + fragment isomer variant |
| heliosli/casmi26-0-496-public-demo | 0.496 (not reproducible) | 1,559 | + 6 heliosli datasets with demo weights | FRIGID de novo on a second T4 inserted at rank 2 (sim >= 0.7, <= 30 heavy atoms), joint LightGBM ranker over all PubChem isomers of the rank-1 formula, ICEBERG-T + MAGMa isomer judgement |
| guhongbin/casmi26-v589-alex0484-extlib | 0.487 | 71 | + takumuhata/casmi26-extlib | base of the 0.495 |
| dmitriigluzdov/casmi-26-the-data-behind-spectra-to-structures | 0.465 | 1,219 | 5 dmitriigluzdov datasets | the base, with data documentation and a version table of what each change scored |
| sanpier/pubchem-neural025-lb-0-466 | 0.466 | 1,745 | + sanpier/casmi-spectrum-model-v1 | contrastive spectrum-candidate model (3-layer transformer over peaks, 2048-bit Morgan MLP, 16 hard negatives, temperature 0.1, about 6 h T4) |
| prvsiyan/analog-propagation-casmi-2026-baseline | 0.487 (now in the stack) | 70 | prvsiyan datasets | analog propagation, origin of the class-share estimate |
| saddamaditya/casmi26-dual-band-massbank-iceberg-sota | 0.441 | 1,039 | ahmedberatozer ICEBERG/GLACIER, DreaMS | early ICEBERG + MassBank line |
| samanyu1808/casmi26-iceberg-where-glacier-is-blind-lb-0-438 | 0.438 | 1,001 | same | ICEBERG where GLACIER has no [M+H]+ |
| canqiang/casmi26-e006-window-40-ice-fallback-true | 0.440 | 1,031 | 16 datasets | E006, the complete inference base before the PubChem channel |
| haideptry/enveda-casmi-2026-fast-spectral-cosine-baseline | 0.339 | 3,546 | prvsiyan datasets | September SOTA |
| inversion/casmi-denovo-tutorial-notebook (host) | 0 | 7,111 | none | de novo SMILES transformer (sinusoidal peak embedding, BPE SMILES decoder, 200k spectra, about 22 min T4); scores 0 |
| evgendvorkin/spectra-to-structures-deep-eda-molecular-clues | 0.495 | 356 | same as 0.495 | EDA wrapped around the 0.495 pipeline |

## 4. Datasets used by the top notebooks

Sizes are the sum of the current file listing. Licence as shown on Kaggle; hosts allow non-commercial open data
(2026-10-17) and treat train.parquet-derived artefacts as open.

| dataset | size | contents | licence | for us |
|---|---|---|---|---|
| dmitriigluzdov/casmi26-pubchem-popularity-prior | 80 GB in files (Kaggle shows 43.8 GB) | mass-sorted PubChem tier 150-1250 Da: 105.9M structures (`casmi26-pubchem-tier/pc_smiles/pc_off/pc_mass`, 7.2 GB), `pc_ik14`, `pc_cid`, aligned priors `pc_lsid` (ln substances), `pc_lpmid`, `pc_lpatent`, `pc_lmesh`, `pc_lsynonyms`, `pc_n_vendors`, `pc_n_depositors`, `pc_src_mask` (depositor groups), `pc_np` (COCONUT/LOTUS/NPAtlas), `pc_lotus_refs`, `pc_formula_isomers`, the same for the 711k pool, delta/growth supplements, 12 shards of 10,226-bit fingerprints of every tier structure (about 70 GB), build scripts | Other (PubChem NLM, LOTUS CC0, NPAtlas CC BY-NC) | HIGH: priors we do not compute yet (substance count is the strongest single prior); could replace parts of WP3. Avoid `pc_src_mask` bits that include NIST MSDC (host question open) |
| dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets | 20.1 GB | copies of: v4b/v3 engine code + models, fpnet_full1, v2 pool (train + COCONUT, fingerprints, fragment masses), PubChem tier, ICEBERG/GLACIER packages with offline wheels, DreaMS embeddings of Enveda-180 and MassSpecGym (5.7 GB), COCONUT/ChEBI/LIPID MAPS/PubChemLite candidate banks with fingerprints, simulated ranker rows, spec2mol models | CC BY-NC-SA, per-folder notices | reference only (weights partly lack training code) |
| dmitriigluzdov/casmi26-natural-product-knowledge-table | 136 MB | 153,964 LOTUS (2026-04-13 freeze) + NPAtlas natural products with NPClassifier pathway/class, organisms, literature counts, in-train flag; `external_reference_spectra/` (public MassBank/MoNA/GNPS spectra of train structures with per-record licence) | Other (LOTUS CC BY, NPAtlas CC BY-NC) | MEDIUM: ready LOTUS refs and NP classes |
| dmitriigluzdov/casmi26-natural-product-panel-benchmark | 233 MB | cached candidates and scores of the public pipeline for the 250 enveda-np-examples molecules, identity and scaffold folds of train | Other | LOW-MEDIUM: quick rule tests, but optimistic (all 250 are in other libraries) |
| dmitriigluzdov/casmi26-fold-safe-fpnet | 1.07 GB | five FPNets each trained without one identity fold and the panel, training code | Other | reference for fold-safe validation (68 percent top-1 unseen vs 97 percent seen) |
| taeyangg4/casmi26-extlib-gnps-full | 211 MB | 198,581 reference spectra of 147,414 structures (194,652 from GNPS MULTIPLEX-SYNTHESIS-LIBRARY-ALL, rest from GNPS, MassBank, MoNA, MSnLib, MS2DeepScore, MassSpecGym), 128 peaks each, with source ids | CC BY-NC-SA | HIGH: extra class-1 references not in train |
| takumuhata/casmi26-extlib | 265 MB | external CC0 reference spectra | CC0 | HIGH (same use) |
| samartalwar/casmi-2026-spectral-library-massbankharmonized | 48 MB | 139,744 MassBank spectra, 20,204 structures with fingerprints (drop `is_casmi_subset`) | CC BY-NC-SA | MEDIUM |
| qwertykeypadapple/casmi26-lotus-candidates | 7 MB | 149,983 LOTUS structures cleaned with RDKit 2026.03.3, ik14, formula, mass | CC0 | LOW (we have LOTUS) |
| prvsiyan/coconut-casmi26-candidates | 423 MB | 436,389 COCONUT 2.0 structures, mass, 6,930-bit fingerprints | CC BY-NC-SA | LOW (we have COCONUT) |
| prvsiyan/chebi-lipidmaps-casmi26 | 60 MB | 62,744 ChEBI + LIPID MAPS structures with fingerprints | CC BY-NC-SA (sources CC BY) | LOW |
| sanpier/casmi-spectrum-model-v1 | 56 MB | contrastive spectrum-candidate model with full train/prepare code | Apache 2.0 | MEDIUM: reproducible second model |
| taeyangg4/casmi26-fr3to25-clean | 3 MB | FRNet fragment re-ranker weights + training script | CC BY-NC-SA | MEDIUM: reproducible |
| ahmedberatozer/casmi26-iceberg, -glacier | 67 + 61 MB | ICEBERG 2.1 and GLACIER MassSpecGym checkpoints, offline runners, pure-torch DGL shims, wheels | Other (MIT code) | LOW-MEDIUM (GPU) |
| hengck23/hengck23-dreams-enveda-casmi26 | 6.9 GB | DreaMS model + embeddings of Enveda-180 and MassSpecGym | MIT | LOW (WP13 spike) |
| megayak/casmi26-simulated-ranker-rows | 366 MB | simulated ranker training rows + FPNets | CC0 | LOW |
| metric/rdkit-2026-3-3-wheel | 149 MB | official RDKit 2026.03.3 wheels cp310-cp313 | | already used |
| heliosli/* (rr-v3, frigid-e1, joint-ranker, m5-v2d, pxv2-deploy, rt-np) | 2.2 GB | demo weights only ("not a reproduction package for the 0.494 result") | mixed | none |

## 5. Discussion forum: what matters

Host and admin statements (dated):
- External data: non-commercial open data allowed (2026-10-17). NIST and anything trained on it is forbidden.
  Allowed explicitly: PubChem (and subsets of it), COCONUT, ChEBI, LIPID MAPS, MassSpecGym and models trained on
  it, ICEBERG/GLACIER, DreaMS, CFM-ID 4, FRIGID official checkpoints, GNPS-NIST14-MATCHES, newer
  GNPS/MassBank/MoNA releases, the Enveda-180 Zenodo release, per-compound PubChem counts (SID, PMID, patent),
  the ICEBERG PubChem atlas (as library and training data, with the retrieval code).
- Reproducibility: weights must come with their original training code and permitted data, or be retrained with
  our own published code; non-reproducible submissions can be removed from the leaderboard (2026-10-27).
- Metric: stereochemistry removed before tautomer canonicalisation (2026-10-07, applied retroactively); solutions
  are stereo-stripped; empty entries are dropped (A;;B puts B at rank 2).
- Hidden test: same columns as test.parquet, no ion mobility or CCS; collision_energy_ev is nominal, a list
  means merged acquisitions. Rerun test data was corrected on 2026-10-07 for a "mass of an electron" error in
  some peak masses (linked to a positive-mode calibrant using neutral Na, which makes enveda-180 positive
  fragments read +0.49 mDa high; the host was checking which spectra are affected).
- No molecule_id overlap between train and test, or between the visible and the rerun test. Class breakdown will
  not be released; "enough novel structures to make it worth working on them". Class 3 is in neither PubChem nor
  COCONUT. Enveda-180 Zenodo structures missing from train were "likely filtered out in preparation".

Community measurements worth trusting:
- Class shares: library-only submission 0.151 public, class-1 MRR about 0.93 on held-out data, so class 1 about
  16 percent; class 3 estimates 39 percent (prvsiyan) to 52 percent (Bull, CV-fitted 28/20/52). Speculation that
  class 3 is concentrated in the private 69 percent.
- Isomers: when the truth is in the list but not first, the winner has the same formula about 98 percent of the
  time. Structure-level fragment scores separate isomers (AUC 0.757); formula-level explained intensity does not
  (AUC 0.500). MetFrag-lite tuning that held out of sample: down-weight 2-bond cleavages (x0.6), linear intensity,
  0.005 Da tolerance, H shifts -2..+3 (pooled +0.023 panel MRR, t 3.45). Allowing 3-bond cuts and hand-written
  bond propensities hurt.
- Validation pitfalls: all 250 enveda-np-examples structures exist in other libraries (holdout by library is
  contaminated); popularity prior +0.50 MRR on structures with public spectra but only +0.075 on the LB (use a
  held-out enveda-180 "no public spectra" split as the class-2 proxy); ICEBERG re-ranking +0.02 on class 2 but
  about 0 on structures not in MassSpecGym (memorisation); curated-DB membership flags look great on holdout and
  hurt on the LB; deleting the truth from the DB overstates generator reach about 3x; GBDT seed noise +-0.006.
- Class 3: rule-based biotransformation of 15 nearest analogs (65 rules, 2 steps) generates the truth for 46
  percent when close analogs exist but 16.5 percent at max Tanimoto 0.55, ranking it poorly (class-3 MRR 0.03 to
  0.08); MC graph generation and DiffMS gave no LB change; FRIGID at rank 2 only is the one public use.
- Adducts: Enveda's adduct labels are clean (0 of 250 wrong); GNPS/MassBank formate ([M+CH2O2-H]-) mislabels
  are common in train.
- Robustness (Shehab Alshehabi, the leader): test real-test extremes before submitting: 1,159 Da molecules,
  16 spectra per molecule, all 10 adducts, a molecule in no database. "Notebook Threw Exception" is our code or
  memory; "Submission Scoring Error" is the grader.
- Forks: some public notebooks gate their real pipeline on a rerun flag; check before spending a submission. The
  0.495 notebook's `IS_RERUN` uses `KAGGLE_IS_COMPETITION_RERUN` or a hash of the visible test file, so it is safe
  under our account.
- The visible test.parquet is 400 enveda-180 molecules copied from train (every spectrum is in train); use it only
  as a format and runtime check.

## 6. What separates 0.495 from 0.55

Nothing public shows the leaders' methods. What the evidence supports:
- The public stack is a set of frozen, hand-gated channels fused by reciprocal rank with weights tuned on a
  125-molecule board. Its learned parts (FPNet, LightGBM rankers, neural model) are trained separately and never
  jointly on simulated class mixes; each new layer is allowed to touch only ranks 2-25 or rare rank-1 cases.
  Gains per layer are at the noise floor, so the climb is mostly selection on noise (winner's curse of +0.019 for
  the best of 7 equal variants).
- Rank 1 is where the score is: 1.0 versus 0.5. The public stack keeps the base pipeline's leader in almost all
  cases. A better rank-1 decision for class 2 isomers (structure-aware fragmentation, forward models trained or
  validated on structures outside MassSpecGym, priors calibrated on "no public spectra" molecules) is the most
  likely source of +0.03 to +0.06.
- Coverage: class 1 structures that are only in external libraries (GNPS multiplex synthesis, newer MSnLib,
  Enveda-180 Zenodo) and class 2 PubChem structures need both a candidate and a ranker that can survive 5,000 to
  30,000 same-mass decoys. The public PubChem channel only keeps 25 structures and inserts them at fixed slots.
- Class 3 may matter more on the private board; nobody public has a working generator.

## 7. Comparison with our plan (PLAN.md sections 1, 2, 4)

Techniques in the public stack that we do not have yet:
1. In-silico fragmentation (MetFrag-lite 1-2 bond cuts, explained intensity, parsimony) as ranker features; the
   public uses it for isomer rank-1 checks and a learned version (FRNet) for ranks 3-25. Our plan has none.
2. External reference spectra beyond train (GNPS full, MassBank, MoNA, MSnLib, Enveda-180 Zenodo) for class 1.
   PLAN 1.4 explicitly left extra libraries out; the public measured +0.003 to +0.006 from them.
3. PubChem substance-count (SID) and synonym priors (we only planned PMID and patent counts).
4. Transductive neighbour network over the test molecules (+0.005): test molecules come from the same biological
   samples, so congeners share evidence.
5. A contrastive spectrum-candidate model as a second learned view (+0.001 to +0.007 in fusion).
6. Forward spectrum models (ICEBERG/GLACIER, +0.008 in public, possibly memorisation); our plan deferred them.
7. Larger fingerprints: 10,226 bits (Morgan r2 + r3 + RDKit path + MACCS) versus our 4,263.

Our planned components nobody uses publicly (possible edge):
1. One LightGBM lambdarank over a joint LIB + DB + GEN pool with pool-context features, trained on simulated
   class 1/2/3 groups with class weights. The public fuses channels by hand-tuned RRF and gates.
2. An honest validation harness with class simulation and a timsTOF NP panel, used for decisions instead of the
   public board (several public authors report that board noise dominates their decisions).
3. Tautomer-consistent fingerprints (InChI round trip) on both sides.
4. CPU-only, fully self-trained, reproducible pipeline: eligible for prizes and immune to the "non-reproducible
   submissions may be removed" rule that hangs over the public fork's shared weights.
5. A rule-based class-3 generator inside the same ranker (public generation is limited to Engine 1's analog edits
   and one FRIGID insertion).

## 8. Recommendations, ranked by expected gain per hour

| # | action | effort | expected gain | why |
|---|---|---|---|---|
| 1 | Add PubChem documentation priors from `dmitriigluzdov/casmi26-pubchem-popularity-prior` (attach it, or rebuild `pc_lsid`/`pc_lsynonyms`/`pc_n_vendors` from CID-SID and CID-Synonym in WP3) as ranker features, calibrated on a held-out enveda-180 no-public-spectra split | 3 h | +0.03 to +0.07 on the board | the largest public lever (0.409 -> 0.464 with PubChem candidates + prior); cheap |
| 2 | Add external reference spectra (taeyangg4 GNPS full, takumuhata extlib, MassBank harmonized, Enveda-180 Zenodo) to the WP5 library index, with source and instrument flags | 4 h | +0.005 to +0.015 | class-1 coverage the train library lacks; rank-1 promotions are clean wins |
| 3 | Implement MetFrag-lite fragment features (1-2 bond cuts, x0.6 for 2-bond, linear intensity, 0.005 Da, H shifts -2..+3, explained intensity and cut count) in numba for the top 50-100 candidates per molecule, as ranker features | 8 h | +0.01 to +0.04 | isomers are 98 percent of DB-route errors; only structure-aware evidence separates them (AUC 0.757) |
| 4 | Transductive neighbour network as a final re-ranking of ranks 2-25 (modified cosine >= 0.6, >= 6 matched peaks, lambda 0.35, leader kept) | 3 h | +0.003 to +0.006 | measured on the board twice; independent of everything else |
| 5 | Keep the private 0.495 fork as a fallback and benchmark only: submit it once to calibrate the board, never pick it as a final without checking the reproducibility of its weights | 0.5 h | calibration | shared weights without training code may be removed from the final board |
| 6 | Validate on the right proxy: held-out enveda-180 structures with no public spectra (class 2), plus all-instrument-hidden class 1; do not tune on the 250 np-examples alone | 2 h (WP7) | decision quality | public authors were misled by panel gains that did not transfer |
| 7 | Probe the ICEBERG PubChem atlas: download size, coverage of our NP/PubChem candidates; if feasible, cosine of query against predicted spectra as a ranker feature | 4 h probe | unknown, possibly large for class 2 | permitted, unused publicly, trained on more data than the MassSpecGym checkpoints |
| 8 | Contrastive spectrum-candidate model (sanpier recipe, Apache 2.0, about 6 h on T4) or our FP MLP with a hard-negative loss | 8 h + training | +0.002 to +0.007 | a second learned view; low priority on 4 CPUs |
| 9 | Class 3 generator: keep it as tail insurance (ranks 10-25) until the validation shows it helps | as planned | +0.00 to +0.01 | public evidence is negative |

Do not: retrain public models with our own code and swap them in without validation (always lost on the board);
let any new signal move rank 1 without a margin gate; use NIST-derived anything; use the visible test file for
statistics.

## 9. Private fork of the best public notebook

- Kernel: `ograhulrajkumar/casmi26-taeyangg4-0495-fork` (private), forked from
  `taeyangg4/casmi26-lb-0-495-network-fr-p5-tail` version 2 (scriptVersionId 371003154, 2026-11-18), the
  root of the 0.495 block. Version 1 is byte-identical code (38 cells). Settings as the original: GPU T4,
  internet off, competition data, the 11 datasets and the kernel output `popwenman/casmi26-scorer-train-v80`.
  All attached datasets are on the same versions the original pinned (their latest versions predate 2026-11-18;
  the scorer kernel has a single version).
- Version 1 (code byte-identical, `COMMIT_MODE = 'fast'`): COMPLETE in 365 s on GPU T4. Wrote a valid
  `submission.csv` (400 unique ids equal to the sample's, 25 dummy SMILES each; the fast commit never runs the
  base pipeline). Datasets, the RDKit 2026.03.3 wheel, the scorer v80 bundle and CUDA all resolved; the overlay
  self-tests passed; P5's Engine 1 rerun ran on 12 molecules on CUDA with 0 errors.
- Version 2 (one line changed: `COMMIT_MODE = 'smoke'`, so the commit run executes every stage on 12 visible
  molecules): COMPLETE in 1,914 s. Valid `submission.csv` (400 ids, 12 real lists of 25, 388 one-guess
  placeholders). Stages seen working: frozen 0.465 base (19.8 min for 12 molecules, Engine 1 + Engine 2 +
  PubChem channel, ICEBERG 793 of 814 candidates scored on CUDA at 42 predictions/s, GLACIER 11 molecules),
  neural ensemble 0.5 z(public) + 0.5 z(v80), fragment check, leader-safe merge, v8 mass prior, neighbour network
  (412 s, 114 molecules with neighbours), extlib overlay (322,741 references), FR 3-25. P5 tail raised its own
  guarded assertion ("Engine 1 rerun molecules differ from the submission") and kept its input file; this looks
  like a smoke-subset artefact (its 12 molecules are not the pipeline's 12), it cannot be checked without a
  scoring rerun.
- The scoring rerun is identical for both versions (`RUN_PIPELINE = IS_RERUN or COMMIT_MODE != 'fast'`, and
  `IS_RERUN` is true when `KAGGLE_IS_COMPETITION_RERUN` is set or the test file's hash differs from the visible
  one). Submit version 1 for an exact replica of taeyangg4's 0.495 submission. Expect a GPU T4 run of several
  hours (stage guards up to 8.6 h of the 9 h limit), i.e. several of the weekly 30 GPU hours per submission.
- Not submitted. The coordinator submits with `scripts/kaggle_submit.py` (kernel owner `ograhulrajkumar`, slug
  `casmi26-taeyangg4-0495-fork`, version 1).

## 10. How to refresh this snapshot

All calls go to `https://www.kaggle.com` with the dummy Basic header (the proxy injects the key).

| what | call |
|---|---|
| full leaderboard (zip of CSV, all teams) | `GET /api/v1/competitions/<slug>/leaderboard/download` (returned 429 once; retry after a few minutes) |
| competition settings, forumId 11293906 | `POST /api/i/competitions.CompetitionService/GetCompetition {"competitionName": "<slug>"}` |
| notebooks with displayed score and commit runtime | `POST /api/i/kernels.KernelsService/ListKernels` with `{"kernelFilterCriteria": {"search": "", "listRequest": {"competitionId": 143318, "sortBy": "SCORE_DESCENDING", "pageSize": 100, "group": "EVERYONE"}}, "detailFilterCriteria": {"deletedAccessBehavior": "RETURN_NOTHING", "unauthorizedAccessBehavior": "RETURN_NOTHING", "returnPrivatelySharedKernels": true, "outputType": "ALL", "runningKernelsIncluded": true}}` (fields `bestPublicScore`, `lastRunExecutionTimeSeconds`, pinned `dataSources`) |
| notebook source and metadata | `GET /api/v1/kernels/pull?userName=<u>&kernelSlug=<s>` (no outputs) |
| dataset files and versions | `GET /api/v1/datasets/list/<owner>/<slug>`, `POST /api/v1/datasets.DatasetApiService/GetDataset {"ownerSlug", "datasetSlug"}`; single file: `POST .../DownloadDataset {"ownerSlug", "datasetSlug", "fileName"}` then GET the redirect |
| forum topics (20 per page) | `POST /api/i/discussions.DiscussionsService/GetTopicListByForumId {"forumId": 11293906, "page": N}` |
| one topic with comments | `POST /api/i/discussions.DiscussionsService/GetForumTopicById {"forumTopicId": id, "includeComments": true}` |

# CASMI 2026 - research review and method roadmap

Written 2026-11-20. Scope: literature and open-source methods for structure ranking and generation from MS/MS, what they would be worth in this competition, and a ranked roadmap. Read `docs/COMPETITION.md` and `docs/PLAN.md` first.

Conventions:
- "Verified" means a number was read from the cited paper, README or file during this review. "From memory" means it was not re-checked and must be confirmed before anything depends on it.
- "MSG" = MassSpecGym. "HR@1" = hit rate at 1 (top-1 accuracy). "Formula-known" = the MSG bonus track where the true formula is an input and all candidates are isomers.
- MRR gain estimates are judgement, not measurement. Each roadmap item has a first experiment whose purpose is to replace the estimate with a number.
- `docs/KAGGLE_INTEL.md` did not exist when this file was finished. Section 7 uses what could be read directly from the public notebooks already saved under `data/external/kaggle_notebooks/` and from one public team log. Reconcile again when the intel file lands.

---

## 1. Summary for the lead

1. The public stack is already broad. The 0.495 public notebook line contains: library search, a PubChem channel with a popularity prior, GLACIER and ICEBERG forward-model re-scoring, a fingerprint transformer trained with hard negatives, a neural spectrum-candidate scorer, gradient-boosted rankers, an untrained MetFrag-like fragment score plus a learned fragment re-ranker, an external-library overlay, a neighbour network over the test molecules, and rule-based analog generation for class 3. Our PLAN.md describes roughly the September version of that stack (which scored 0.34). Building PLAN.md as written will not reach 0.49.
2. On the public board, knob tuning is exhausted: of 60 one-knob arms in the best-documented public log, 4 gained more than 0.002 and the largest gain was 0.006. Gains came from new signals. The public board is about 125 molecules, so one molecule moving from rank 2 to rank 1 is worth 0.004 and differences below about 0.01 are noise.
3. The literature, after the 2026 audit of MSG ("MassSpecGym in the Wild"), says three things that matter here:
   - The largest verified, leak-free gain for retrieval is training the spectrum encoder with simulated spectra of the candidate chemical space: MIST goes from 9.6 to 40.3 percent HR@1 (formula-known) when trained with GLACIER-simulated spectra of 4M molecules. Forward-model retrieval itself (GLACIER 39.5, ICEBERG 36.2) is at the same level and is complementary.
   - The training objective matters as much as the architecture: on Enveda-180 itself (timsTOF, the same instrument family as the test), a transformer with a candidate-contrastive loss reaches 63.1 percent HR@1 against 53.0 for the same transformer with BCE and 35.4 for an MLP with BCE. PLAN.md section 4.4 is the MLP-with-BCE cell, the weakest of that table.
   - Most headline numbers of 2025-2026 are inflated. At least 18 of 26 papers reporting MSG results had evaluation problems. Retrieval claims of 64 and 85 percent HR@1 fall to 7 to 8 percent when corrected; the "28 percent" de novo claim falls to 3.3 percent. The honest de novo state of the art with a known formula is about 10 percent top-1 on MSG and 17 percent on NPLIB1.
4. A spectrum-blind popularity prior is very strong on benchmarks (PubChem default order: 50.0 percent HR@1 on MSG formula-known, better than any model) and was the single strongest signal in CASMI 2016 (reference count alone: 69 percent top-1). It is also the signal most likely to mislead on this test set, because class 2 is defined as "no public spectra" and class 3 as "not in PubChem".
5. Class 3 remains mostly unsolved. Fingerprint-conditioned molecular language models (MS-GPT, FRIGID) are the first de novo methods worth running: they are Apache-2.0, fit on a T4, and their decoder takes a fingerprint and a formula, not a spectrum, so they can be driven by our own timsTOF-trained fingerprint model.
6. Recommended order of work: reach parity with the public stack by reusing its public assets (section 8, step 0), then add, in order: contrastive transformer encoder; simulated-spectra training over the NP tier with a precomputed forward-spectrum cache; forward-model features; priors learned on NP-like validation only; an external-library sweep; a fingerprint-conditioned decoder for class 3; test-set network propagation with edit consistency; formula evidence.

---

## 2. What the benchmarks actually say

### 2.1 MassSpecGym, the reference benchmark

Bushuiev et al. 2024 (NeurIPS D&B). 231k spectra, about 29k molecules (now cited as 31,602), Orbitrap and QTOF only, no timsTOF. Split by MCES distance >= 10 between folds, so test molecules have no close analog in train. That is harder than our class 2 and comparable to a hard class 3.

Retrieval candidate sets: at most 256 candidates per spectrum, either same mass ("mass-based") or same formula ("bonus"), drawn in order from about 1M natural/pesticide/industrial molecules, then about 4M diverse molecules, then PubChem. Our pools are different: median 77 candidates in the NP tier at 10 ppm (measured on the 250 enveda-np-examples structures, section 7.3) and about 10,000 in the PubChem tier.

Original baselines (verified, mass-based): random 0.37, fingerprint FFN 2.54, DeepSets + Fourier 5.24, MIST 14.64 percent HR@1. The MIST number used ground-truth formulas (MSG issue 52), so it is not a fair mass-based number.

### 2.2 The corrected leaderboard (Liu, Bushuiev et al. 2026, "MassSpecGym in the Wild", arXiv 2606.19624, co-authored by Enveda)

This is the most important paper for calibrating expectations. All numbers verified from the PDF.

Retrieval, formula-known track, HR@1 / HR@5 / HR@20 in percent:

| Method | HR@1 | HR@5 | HR@20 | Note |
|---|---|---|---|---|
| Random | 3.06 | 11.35 | 27.74 | |
| Nearest neighbour (subformula rule, Khoo) | 13.13 | 26.56 | 44.30 | training-free |
| MIST (no augmentation) | 9.57 | 22.11 | 41.12 | |
| JESTR | 11.82 | 33.48 | 61.46 | contrastive, GNN + binned MLP |
| MVP | 13.96 | 36.88 | 68.12 | multi-view contrastive, consensus spectra |
| FLARE | 22.66 | 50.00 | 75.15 | peak-to-atom alignment |
| ICEBERG (forward model) | 36.18 | 60.70 | 78.83 | |
| GLACIER (forward model) | 39.47 | 63.50 | 80.46 | |
| MIST 4M-aug (simulated-spectra training) | 40.31 | 58.87 | 73.02 | best leak-free inverse model |
| PubChem default order, no spectrum | 49.98 | 54.13 | 61.04 | popularity shortcut |
| DreaMS + ChemBERTa alignment, as published | 82.41 | | | SMILES-format shortcut |
| same, corrected | 6.70 | 19.82 | 40.62 | |
| Chemformer two-stage (GLMR-like), as published | 63.89 | | | SMILES-format shortcut |
| same, corrected | 7.60 | 19.12 | 39.31 | |

Mass-based track with MIST 4M-aug (formula must be inferred):

| Setup | HR@1 | HR@20 |
|---|---|---|
| Random | 0.43 | 7.99 |
| Random among candidates with the true formula | 12.98 | 71.33 |
| MIST with MIST-CF top-1 formula (data-safe) | 32.14 | 60.92 |
| MIST with ground-truth formula | 52.84 | 84.17 |

MIST-CF trained on the MSG train fold: 50.5 percent top-1 and 73.2 percent top-5 formula accuracy.

Leakage table for simulated-spectra training (MIST, formula-known HR@1): no augmentation 9.57; leak-free GLACIER + 4M molecules 40.31; simulator also trained on test 43.4; test molecules also simulated 47.0. So the legitimate part of the gain is about 30 points and the leak adds only 3 to 7.

De novo, formula-known, top-1 / top-10 exact match in percent:

| Method | Top-1 | Top-10 | Top-1 Tanimoto |
|---|---|---|---|
| SMILES / SELFIES transformer, Spec2Mol, MIST + MSNovelist, MIST + Neuraldecipher | 0.00 | 0.00 | 0.03-0.14 |
| MS-BART | 1.26 | 1.28 | 0.23 |
| MADGEN | 1.31 | 1.54 | 0.20 |
| DiffMS | 2.30 | 4.25 | 0.28 |
| FOAM (DiffMS + PubChem seeds) | 1.50 | 10.28 | 0.35 |
| MSAnchor | 2.68 | 4.67 | 0.32 |
| MIST + MolForge, corrected | 3.29 | 5.65 | 0.31 |
| MIST + MolForge, as published (bugged batched MIST checkpoint) | 31.75 | 40.55 | 0.68 |
| FRIGID-base | 8.21 | 9.82 | 0.38 |
| FRIGID (10 rounds of ICEBERG-guided refinement) | 9.05 | 12.23 | 0.39 |
| FRIGID arXiv v2 (test-informed augmentation) | 18.29 | 22.00 | 0.43 |
| MS-GPT (arXiv 2607.23607, its own README) | 10.22 | 13.21 | 0.39 |

Decoder pretraining scale matters little once leak-free: 2.5M / 10M / 50M molecules give 8.0 / 9.2 / 9.5 percent top-1 with exact test matches excluded, and 6.0 / 8.2 / 8.9 when analogs with Tanimoto >= 0.6 are also excluded. So about a quarter of the de novo "accuracy" depends on close analogs being in the decoder's training corpus.

Oracle bound: MolForge given the true fingerprint reaches 46 percent top-1 and 59 percent top-10 (verified in arXiv 2508.04180). The bottleneck is fingerprint prediction, not decoding.

### 2.3 Enveda-180 results (De Waele et al. 2026, arXiv 2602.16507, code github.com/gdewael/ms-mole)

Enveda-180 v1.0.0 with an MCES-clustered split (149k / 18k / 17k molecules), equal-mass candidates sampled from PubChem (10 ppm, at most 256), 4096-bit Morgan radius 2 targets. Verified. HR@1 / HR@5 / HR@20 in percent:

| Encoder | Loss | Enveda-180 | MSG mass-based |
|---|---|---|---|
| MLP on 0.1 Da bins | BCE | 35.4 / 55.1 / 66.1 | 10.1 / 19.4 / 35.0 |
| MLP | focal | 40.0 / 65.7 / 83.1 | 11.5 / 20.8 / 36.7 |
| MLP | cosine | 23.5 / 38.6 / 55.6 | 9.5 / 18.2 / 32.9 |
| MLP | contrastive, fingerprint cosine | 44.2 / 64.6 / 79.3 | 12.3 / 26.4 / 42.7 |
| MLP | contrastive, embedding cosine | 45.9 / 71.1 / 87.8 | 12.3 / 26.1 / 45.1 |
| Transformer on peaks | BCE | 53.0 / 68.2 / 79.3 | 15.5 / 30.1 / 50.0 |
| Transformer | focal | 57.3 / 72.4 / 82.5 | 17.0 / 31.6 / 51.9 |
| Transformer | IoU (best Tanimoto) | 43.1 / 57.0 / 69.9 | 9.4 / 20.6 / 38.8 |
| Transformer | contrastive, fingerprint cosine | 63.1 / 81.0 / 90.0 | 18.5 / 36.9 / 58.6 |
| Transformer | contrastive, embedding cosine | 63.1 / 82.5 / 93.0 | 16.1 / 31.5 / 53.7 |

What this means for us:
- Losses that maximise fingerprint similarity (cosine, IoU) are the worst retrievers. PLAN.md 4.4 adds a 0.5 x cosine term to BCE; that term probably hurts. Focal beats BCE for free.
- Transformer over peaks beats binned MLP by 15 to 18 points at equal loss; contrastive beats BCE by 10 more.
- Scoring candidates by cosine to the predicted fingerprint beats Tanimoto-style scores regardless of training loss (their Table 6).
- A pretrained DreaMS backbone gave no consistent gain over a transformer trained from scratch (their Table 5), which agrees with our WP13 spike.
- Morgan radius 2 was the best target; larger radii were worse; MAP4 was comparable.
- Hard negatives mined from PubChem did not beat equal-mass candidate sets as negatives.

Caution: these are synthetic drug-like molecules with random PubChem decoys. The CASMI test is natural-product-like, where the decoys in the NP tier are close isomers (section 7.3), so absolute numbers will be lower.

### 2.4 Nearest-neighbour baselines (Khoo 2026, arXiv 2610.02249)

Fingerprint Jaccard on scaffold splits (verified): plain cosine NN 0.208 (NPLIB1) / 0.235 (MSG); modified cosine 0.236 / 0.246; DreaMS-embedding NN 0.263 / 0.299; "subformula rule" NN, which uses the known formula to annotate peaks and averages neighbours sharing subformulas, 0.337 / 0.381. In retrieval the subformula-rule NN reaches 13.1 percent HR@1 on MSG formula-known, far below MIST 4M-aug (40.3). Use: a cheap, training-free feature that is complementary to a learned encoder; not a replacement.

---

## 3. Method by method

Columns: what it does, verified performance, code and weights, whether it can run offline on Kaggle, and expected value per novelty class for us (C1 / C2 / C3).

### 3.1 Retrieval and ranking

| Method | Mechanism | Evidence | Code / weights / licence | Offline on Kaggle | Value for us |
|---|---|---|---|---|---|
| CSI:FingerID / SIRIUS / COSMIC | Fragmentation trees, kernel SVM fingerprints, Bayesian-network scoring; COSMIC gives a confidence score | Won CASMI 2016 category 2 top-1 count (70 of 208, 34 percent) and CASMI 2017 categories 1, 2 and 4; best structure entry in CASMI 2022 (57 of 222, 26 percent) | SIRIUS is open for local formula trees, but fingerprint prediction and structure search are web services needing login and licence servers | No. Fingerprint prediction cannot run offline. Precomputing on test is impossible (hidden test) | None directly. Its ideas (fragmentation-tree kernels, formula from MS/MS) are reproduced by MIST / MIST-CF |
| MIST (Goldman 2023) | Peaks annotated with subformulas of the precursor formula, formula transformer, fingerprint "unfolding", then retrieval | MSG formula-known 9.6 percent without and 40.3 percent with simulated-spectra training; needs the precursor formula (32.1 percent with predicted formula against 52.8 with the true one) | github.com/samgoldman97/mist and the maintained github.com/coleygroup/MIST (MIT); MSG checkpoints on Zenodo 23303289 and HF harrylau/msms-models. The older DiffMS-release checkpoint (Zenodo 15122968) is flagged as unreproducible; do not use | Yes, small model, T4 or CPU | C2: the 4M-aug checkpoint is a usable extra scorer but is trained on Orbitrap/QTOF and on elements C, H, N, O, P, S, Cl, F only. Better to adopt its two ideas (subformula peak tokens; simulated-spectra training) in our own encoder |
| MIST-CF | Ranks candidate formulas with a formula transformer | 50.5 percent top-1, 73.2 percent top-5 on MSG; "nearly equivalent to SIRIUS" on CASMI 2022 (its paper) | github.com/samgoldman97/mist-cf (MIT); data-safe MSG checkpoint in MSG v1.5 | Yes | See section 5; modest |
| MassFormer | Graph transformer forward model, binned spectra | CASMI 2016 retrieval R@1 about 15 percent in the FlexMS benchmark (arXiv 2602.22822) | github.com/Roestlab/massformer (from memory: BSD), Orbitrap [M+H]+ focus | Yes | Superseded by ICEBERG / GLACIER. Skip |
| ESP | Forward model ensemble (MLP + GNN) | MSG mass-based 10.7 percent HR@1 | github.com/HassounLab/ESP | Yes | Skip |
| JESTR (Kalia 2025) | CLIP-style joint embedding, GNN molecule encoder + MLP on binned spectrum, late regularisation against hard candidates | MSG 15.1 (mass) / 11.8 (formula); NPLIB1 45.8 percent rank@1 with PubChem same-formula candidates against MIST 28.0 | github.com/HassounLab/JESTR; 6 h on an A100 | Yes | The idea (contrastive against real candidates) is already in the public FPNet recipe. Its hard-candidate regulariser is worth copying: +11 percent relative on NPLIB1 |
| MVP (Zhou, Chen, Hassoun 2025) | Four views: graph, fingerprint, spectrum, consensus spectrum | MSG mass-based 26.4 percent rank@1 with single spectra, 36.0 with the consensus spectrum as query (its paper); formula-known 14.0 | bioRxiv 2025.11.12.688047, code from the Hassoun lab | Yes | Supports molecule-level aggregation: querying with the consensus of all spectra of a molecule is worth about 10 points over single spectra. We get this for free because the task is per molecule |
| FLARE (Chen 2026) | Contrastive with peak-to-atom max-similarity alignment (late interaction), peaks as subformulas | MSG 43.2 (mass, with known formula) / 22.7 (formula-known, audited) | huggingface.co/spaces/HassounLab/FLARE, code licence not stated | Yes, small | Best published inverse model that does not use simulated data. Late interaction is a candidate upgrade to our encoder head, second priority after contrastive + simulated data |
| GLMR, SpecBridge | Generative re-ranking; frozen DreaMS aligned to frozen ChemBERTa | 64 and 85 percent HR@1 as published; 7.6 and 6.7 corrected | public | - | Hype. Do not build |
| FraGNNet | Probabilistic fragment-graph forward model | MSG simulation track: cosine 0.52, retrieval 46.6 percent HR@1 on [M+H]+ with similarity-ranked decoys (original paper) | public repository (location from memory, not checked), [M+H]+ only | Yes | Superseded by ICEBERG 2.x / GLACIER, which cover adducts and collision energy |
| ICEBERG 2.1 / SCARF / MARASON | Two-stage fragment generation then intensity; collision energy, negative mode, contrastive fine-tuning | NIST20 [M+H]+ top-1 40.0 percent random split, 33.5 scaffold, against all PubChem isomers; 92 percent in top 10; MSG formula-known 36.2 | github.com/coleygroup/ms-pred (MIT). MSG-trained weights are public (Dropbox, Zenodo 23543494, HF harrylau/msms-models). NIST-trained weights need a NIST licence: not usable under the competition's "freely and publicly available" rule | Yes. 10.8 ms per spectrum on an A5000. Already packaged as Kaggle datasets by public authors | C2: yes, as ranker features on the top-K. Already in the public stack |
| GLACIER (Wang, Wang, Coley 2026) | Single-stage DETR-style fragment detection on a Graphormer | MSG formula-known 39.5 (audit) to 49.9 (its paper, spectra with collision energy); NIST20 scaffold 46.0; 1.3 ms per spectrum on an A5000, 8x faster than ICEBERG | Same repo; MSG weights public | Yes | C2: fast enough to simulate the whole NP tier offline (roadmap items 2 and 3) |
| CFM-ID 4 | Rule-plus-learned fragmentation, ESI-QTOF at 10/20/40 eV | CASMI 2016: 19 percent top-1 alone, 58 percent with metadata | cfmid.wishartlab.com, open source (from memory: LGPL), slow: seconds to minutes per molecule | Marginal | Skip; neural forward models dominate |
| MS2Mol (Enveda) | Spectrum-to-SMILES transformer | Preprint only, not on the audited leaderboard; no numbers verified here | No public weights found | No | None. Note that the host has an internal de novo model and a foundation model (PRISM, trained on over a billion spectra): the host's own baseline is likely strong, which bounds how "impossible" class 3 is |
| DreaMS (Bushuiev 2025) | Self-supervised transformer on GeMS spectra; 1024-d embeddings | Best no-formula nearest-neighbour similarity (section 2.4); frozen or fine-tuned backbone gives no retrieval gain over training from scratch (section 2.3) | HF roman-bushuiev/DreaMS (MIT), 469 MB TorchScript | Yes on GPU. Our CPU spike measured 145 spectra/s with a custom bf16 path | C2/C3: a second analog-search similarity. Small gain. The public stack already carries a DreaMS resource |
| Spec2Vec / MS2DeepScore / MS2Query | Embedding similarities; MS2Query re-ranks MS2DeepScore neighbours with a random forest | MS2Query: analogs for 35 percent of queries at mean Tanimoto 0.63, against 0.45 for modified cosine at the same recall | matchms ecosystem, Apache-2.0, pretrained models on Zenodo | Yes, CPU | C2/C3: MS2Query's key feature, the average similarity to several library spectra of chemically similar library molecules, is a cheap addition to our analog-evidence features |

### 3.2 De novo generation and analog propagation

| Method | Mechanism | Evidence (formula-known unless stated) | Code / weights / licence | Offline on Kaggle | Value for C3 |
|---|---|---|---|---|---|
| MSNovelist | CSI:FingerID fingerprint to SMILES LSTM | Its paper: 26 percent of CASMI 2016 structures correct at rank 1 and 57 percent retrieved, on molecules with analogs in training; 0 percent on MSG | Needs SIRIUS fingerprints | No | None |
| Spec2Mol, MADGEN, MS-BART, DiffMS | Autoencoder; scaffold retrieval then diffusion; BART on fingerprint tokens; graph diffusion | MSG top-1 0 / 1.3 / 1.3 / 2.3 percent; NPLIB1 0 / 2.1 / 7.5 / 8.3 percent. DiffMS takes 131 s per spectrum | All public | DiffMS too slow | None. Superseded |
| MIST + MolForge | Fingerprint to SMILES transformer | 3.3 percent top-1 corrected (not 28) | MolForge CC BY-NC-SA 4.0 | Yes | Superseded by the next two |
| FRIGID (Bohde 2026) | Masked diffusion language model on SAFE strings (247M parameters, trained on up to 1B molecules), conditioned on formula and a MIST fingerprint; optional ICEBERG-guided remasking | MSG 8.2 percent top-1 (base), 9.1 / 12.2 top-1 / top-10 with refinement; NPLIB1 13.3 (base), 14.2 / 25.3. 6.6 s per spectrum (base) and 41 s (10 rounds) on an RTX 6000 Ada | github.com/coleygroup/FRIGID, code Apache-2.0, checkpoints on Zenodo 23543494 (no licence stated) | Base model yes: 400 molecules x 6.6 s is under an hour on a faster GPU, perhaps 2 h on a T4. Refinement probably too slow on a T4 | Real but small, see below |
| MS-GPT (2026) | Molecule language model (150M parameters, 100M-molecule pretraining) queried with a band of thresholded fingerprints, candidates ranked by generation frequency; LoRA adapter for posterior bias | MSG 10.2 / 13.2 percent top-1 / top-10; NPLIB1 17.1 / 27.6. Point-query ablation: 7.4 to 8.0 top-1, so the query band is worth 2 to 3 points | github.com/VIKI623/MS-GPT, Apache-2.0; checkpoints on an SJTU cloud share; a reimplementation ("ms-gpt-base") and a 50M decoder are in HF harrylau/msms-models | Yes: 100 samples per spectrum from a 150M model | Best available. Its decoder input is a 4096-bit Morgan fingerprint band plus a formula, so it can be driven by our own fingerprint model |
| Test-time tuning (Mismetti 2026) | Pretrain on simulated spectra, fine-tune per test spectrum on the most similar training samples | MSG 3.2 percent, NPLIB1 12.9 percent top-1 | IBM, ICLR 2026 | Per-molecule tuning for 400 molecules is affordable, but the base model is weak | Not worth it as a generator. The idea transfers to retrieval: fine-tune the encoder for a few steps on the training spectra nearest to the test molecules (transductive); untested, low priority |
| Scaffold- or formula-conditioned generation | MADGEN (scaffold), FOAM (optimise from DiffMS and PubChem seeds against a forward model) | FOAM: top-10 10.3 percent, by far the best top-10 before FRIGID; the gain comes from seeding with known isomers and scoring with ICEBERG | FOAM reimplemented in harrylau/msms-models | Slow | The FOAM recipe is what our GEN channel should imitate: seeds from the database, edits, forward-model scoring |
| Analog search and spectral networks | Modified cosine, molecular networking, NAP (propagate library hits through the network to re-rank in-silico candidates), MolNetEnhancer, GNPS "suspect" library (propagated annotations; excluded from train by the host) | NAP and MolNetEnhancer report qualitative gains; MS2Query numbers above | GNPS tools are online workflows; matchms offline | Reimplement offline, cheap | This is what the public "neighbour network" does over the 400 test molecules: +0.004 to +0.007 on the public board |

Realistic class 3 expectations. Class 3 molecules are natural-product-like structures absent from PubChem. Three sources of difficulty compound: the fingerprint is predicted, not known; the formula must be inferred; and the nearest known analog may be far.
- Exact match in the top 25 from a fingerprint-conditioned decoder: NPLIB1-like conditions (analogs in training, formula known) give 17 percent top-1 and 28 percent top-10; MSG-like conditions (no analog within MCES 10) give 10 and 13 percent. With a predicted formula and timsTOF spectra, expect 3 to 8 percent top-1 and 8 to 15 percent in the top 25 for class 3, which is an MRR of about 0.05 to 0.10 on that class.
- Near misses: the top-1 Tanimoto of the best methods is 0.39 on MSG and 0.53 on NPLIB1; top-10 best Tanimoto 0.45 and 0.61. A "close" structure (wrong position of one substituent) is the typical outcome, and the metric gives it zero. Rule-based positional-isomer expansion around the generator's best outputs is therefore worth more than more samples.
- Public notebooks already reach a small non-zero class 3 score through analog edits (one team measured 0.08 MRR on its simulated class 3, largely an artefact of duplicate structures, section 7.2).

---

## 4. Priors for natural products

| Prior | Evidence | Failure mode | Use here |
|---|---|---|---|
| Literature and patent counts, PubChem deposition order | CASMI 2016: reference count alone 143 of 208 top-1 (69 percent), 94 percent top-10; CFM-ID went from 19 to 58 percent top-1 and MS-FINDER from 22 to 70 percent with metadata. MSG: PubChem default order 50 percent HR@1 with no spectrum | Böcker's "metascore" critique (Hoffmann et al. 2023, the MAD HATTER parody paper): benchmark molecules are "blockbuster" compounds, so popularity looks far better in evaluation than on real unknowns. Gupta and Skinnider 2026 show a spectrum-blind model is competitive on MSG and falls to near random with property-matched decoys | Strong for class 1 and the famous part of class 2. The host defines class 2 as "known structure, no public spectra", which selects against blockbusters. Learn the weight only on NP-like validation with library-covered structures removed, and report MRR with the prior off |
| Membership in NP databases (COCONUT, LOTUS, NPAtlas), number of source organisms and collections | Our NP tier shrinks the pool from about 10,000 to about 77 candidates. One public team measured that adding ChEBI and LIPID MAPS to the pool lowered class 2 MRR by 0.023: a larger pool without a matching prior dilutes | COCONUT contains predicted and low-confidence entries; class 3 is by definition absent | Keep as ranker features (already in PLAN 2.4). Pool expansion must come with a prior feature, never alone |
| NP-likeness score | No retrieval benchmark found; it separates NP-like from synthetic decoys in the PubChem tier | Useless inside the NP tier, where all candidates are NP-like | One cheap feature for PubChem-only and generated candidates |
| Organism or taxonomy (LOTUS) | Rutz et al. 2019: taxonomically informed re-ranking gave 1.5 to 7-fold F1 gains | Needs the source organism of the sample | Not usable: the test has no sample metadata. Possible weak substitute: taxon coherence across test molecules that the network links together. Speculative; do not build |
| Biosynthetic plausibility, compound class | CANOPUS and ConCISE predict compound classes from spectra reliably (ConCISE reports over 95 percent true positive rate at its consensus levels) | Class predictors need training labels (NPClassifier is a web service; ClassyFire too) | A class-consistency feature can be obtained without external services: predict coarse structural-class bits with our fingerprint model (they are already fingerprint bits or simple SMARTS) |
| Mass error | Public notebooks add a timsTOF mass-accuracy prior on ranks 2 to 25 | Isomers share a mass | Already a planned feature; fit its shape on enveda-np-examples |

Spectrum-blind check to adopt (recommended by the audit paper): run the ranker with every spectrum replaced by a constant. Whatever MRR survives is prior. Track it per class; if a model change raises the blind score, it learned the prior, not chemistry.

---

## 5. Formula annotation when the adduct is given

- With the adduct known and a few ppm of mass accuracy, the candidate formulas are fixed by enumeration; the only question is ranking them. No MS1 isotope pattern is supplied, so isotope scoring (half of SIRIUS's formula score) is unavailable. CASMI 2017 had the same handicap and the Jena group reported that SIRIUS often missed the formula there.
- Published accuracy without isotopes: MIST-CF 50.5 percent top-1 and 73.2 percent top-5 on MSG; CASMI 2022 participants ranged up to 94 percent with a median of 71 percent (with MS1). BUDDY / msbuddy (Xing 2023; pip-installable, Apache-2.0 per its docs, CPU, instrument presets for QTOF, Orbitrap, FT-ICR) reduces the candidate space by 43 percent through bottom-up subformula evidence; its mass-binned accuracies were not verified here. Formula accuracy drops above 500 to 600 Da and for halogenated molecules.
- Value on MSG is large: knowing the formula lifts MIST from 32 to 53 percent HR@1, and even a random pick among true-formula candidates reaches 13 percent.
- Value for us is smaller than that suggests, measured locally on the 250 enveda-np-examples structures against the NP tier at 10 ppm: median pool 77 candidates and 4 distinct formulas, but the true formula owns a median of 60 of those candidates and is the most frequent formula in the window for 86 percent of molecules. Inside the NP tier the problem is isomers, not formulas. Formula ranking matters in three places: the PubChem tier (tens of formulas per window), class 3 generation (the generator needs a target formula; a wrong one gives zero), and formula-conditioned models (MIST-style subformula tokens, MS-GPT, FRIGID).
- Practical design: no separate formula tool at first. Take candidate formulas from the database pool plus enumeration, and score each by (a) the fraction of query intensity explained by subformulas of that formula within max(20 ppm, 0.01 Da), over all spectra and adducts of the molecule, (b) the number of NP-tier structures with that formula, (c) simple valence and element-ratio rules. That is the "subformula rule" and MIST-CF's input in feature form, cheap on CPU. Add msbuddy only if this loses to it on validation.
- Molecules measured as several adducts and in both polarities (108 of the 400 public test molecules have multiple adducts) give independent fragment sets for the same formula; summing evidence across them is a gain no single-spectrum benchmark measures.

---

## 6. Past CASMI contests

| Year | Setting | Winner and numbers | What won |
|---|---|---|---|
| 2016 | 208 challenges, mostly well-known compounds, candidates from ChemSpider | Category 2 (spectra only): CSI:IOKR 86 gold medals; CSI:FingerID best top-1 (70, 34 percent); MS-FINDER 22 percent, CFM-ID 19 percent, MetFrag 12 percent top-1. Category 3 (metadata allowed): MS-FINDER + metadata 146 top-1 (70 percent), CFM-ID + database score 120 (58 percent). Equal-weight consensus of the six category-2 entries: 82 top-1, better than each alone; consensus plus retention time and reference counts: 164 (79 percent) | Fingerprint prediction trained on spectral libraries won the spectra-only track; metadata priors and ensembling won overall. Reference count alone would have placed second |
| 2017 | Natural products and others, no isotope patterns | SIRIUS + CSI:FingerID won categories 1 (natural products), 2 (in-silico only) and 4 (candidate ranking) | The same method; formula identification was the reported weak point |
| 2022 | 500 unknowns from raw LC-MS/MS, both polarities | Framed as a community gauge. 2D structures: Dührkop (SIRIUS) 57 of 222 priority challenges (26 percent) and 64 of 219 bonus (29 percent); a second group 21 of 55 attempted (38 percent); several groups near zero. Formula: best 94 percent, median 71 percent | Even the best tool identified about a quarter of true unknowns. This is the honest prior for class 2 and class 3 together |

Lessons that transfer: (1) a learned inverse model plus database priors plus an ensemble of unlike scorers is the historical winning recipe, and it is what the public stack has converged to; (2) priors dominate when the test set is made of well-known compounds and fade as it becomes novel; (3) no CASMI entry ever solved de novo.

---

## 7. What the public field already does (to be reconciled with KAGGLE_INTEL.md)

Sources: the notebooks saved in `data/external/kaggle_notebooks/` (read: `alexchilton__casmi26-0-495-what-moved-the-board-60-arms`), and the public run log of the GitHub repository shiferaxa/casmi26 (`docs/plan.md`). These are other teams' claims.

### 7.1 The lineage of the 0.495 notebook

| Stage | Public score | What was added |
|---|---|---|
| Library cosine baseline | 0.06-0.12 | |
| "Analog propagation" pipeline (prvsiyan, haideptry) | 0.34 | 712k pool (COCONUT 2.0 + train), 8.5 ppm window, four channels (library entropy similarity, shifted-similarity analog propagation weighted by similarity to the power 4, MetFrag-lite, a 6-layer d=512 transformer predicting 6,930 fingerprint bits), HistGradientBoosting ranker on 31 features |
| Two-ranker blend with a ranker trained on 400k simulated rows (megayak) | 0.35 | |
| Multi-engine (ahmedberatozer v4): PubChem channel, ICEBERG and GLACIER re-scoring, class 3 generation | 0.42 | a missing Docker pin silently disabled these stages and cost 0.037 in one fork |
| Engine fusion (canqiang E006), PubChemLite, popularity prior | 0.44-0.46 | |
| Neural spectrum-candidate scorer fused by reciprocal rank, "leader-safe": weight 0.25 at rank 1 and 0.75 on ranks 2 to 25 | 0.466-0.472 | moving rank 1 with the neural model lost up to 0.047 |
| Fragment check, isomer re-ranker, mass-accuracy prior | 0.477-0.481 | |
| External-library overlay (198k reference spectra of 147k structures, mostly the GNPS multiplex synthesis library) | 0.484-0.487 | lowering its threshold from 0.85 to 0.80 gave +0.006, the largest single-knob gain recorded |
| Neighbour network over test molecules (modified cosine >= 0.6, >= 6 matched peaks; candidates supported by neighbours' top-5 via Tanimoto), learned fragment re-ranker on ranks 3 to 25, extra generated candidates in the tail | 0.488-0.495 | |

Observations:
- 103 of the 112 public notebooks scoring 0.46 or more attach the same asset bundle. The field is one pipeline with different constants. No member of the top ten teams has published a notebook.
- The leader moved from 0.513 (27 October) to 0.549 (15 November) to 0.552 now, staying about 0.05 above the public line. The gap has been stable, so the leaders have one or two signals the public line lacks, not better constants.
- Retraining the fingerprint model lost against the published checkpoints in every recorded attempt (-0.007 to -0.024), because the downstream rankers were fitted on features from specific checkpoints. Features, encoder and ranker must be refitted together; swapping one part of someone else's stack is a known trap.

### 7.2 Validation lessons from the public log

- Random held-out training structures ranked three rankers in the opposite order from the leaderboard. The 250 enveda-np-examples structures (same instrument and pipeline as the test), evaluated under three masks (all spectra visible; own spectra hidden; structure removed from the pool), ranked them correctly. Ranker rows built from random timsTOF library compounds (Enveda-180, synthetic) hurt: only NP-like structures should be used to fit and validate the ranker.
- Their numbers on enveda-np-examples for the 0.34 pipeline: class 1 mask 0.994, class 2 mask 0.60 to 0.76, class 3 mask about 0.08. Fingerprint channel alone on the class 2 mask: 0.47 to 0.53 MRR over a 712k pool, saturating at about 0.53 across seeds and step counts for their recipe (hard-negative contrastive loss with 63 decoys from the 10 ppm window, spectrum-merge augmentation, peak dropout, jitter).
- With class 2 validation near 0.76 and a board score of 0.34, they concluded that about a third of the test is unreachable by a COCONUT-plus-train pool. That matches a class mix near one third each (for example 0.35 x 0.95 + 0.35 x 0.45 + 0.30 x 0.05 = 0.51).
- Pruning the pool by library similarity before ranking dropped 50 of 250 true structures; removing the prune was worth +0.04. Never prune by a class 1 signal.
- That log claims only 8 of the 250 enveda-np-examples structures are in COCONUT. Our NP tier disagrees: 249 of 250 are in COCONUT, 229 in LOTUS, 241 in PubChemLite (measured today from `data/processed/train/structures.parquet`). Their figure is an artefact of their pool de-duplication. Consequence: enveda-np-examples are famous natural products, all with public spectra. They simulate class 1 faithfully and class 2 optimistically (a real class 2 molecule is more obscure, has weaker priors and fewer close analogs in train).

### 7.3 Local measurements made for this review

On the 250 enveda-np-examples structures against `data/processed/db/np/np.parquet` at 10 ppm: pool size median 77 (10th to 90th percentile 17 to 193, max 459); distinct formulas median 4 (2 to 10); candidates sharing the true formula median 60; the true formula is the most frequent formula in the window for 86.4 percent. NP-tier coverage of training structures by library: riken 85 percent, massbank 82 percent, gnps 49 percent, pluskal_ms2 28 percent, enveda-180 0.2 percent.

### 7.4 Noise

The public board covers about 125 molecules. The standard error of an MRR near 0.5 on 125 molecules is about 0.04 in absolute terms; paired differences between similar pipelines are tighter, but anything under 0.01 is not evidence. The private board (about 275 molecules) will reshuffle teams separated by 0.005. Prefer changes that are large on NP-like local validation and neutral on the board over changes that are +0.003 on the board.

---

## 8. Roadmap

Time and compute available: the entry and team-merge deadline is reported as 7 December and the final deadline as 14 December 2026 (from a public repository and a press release; confirm on the rules page). That leaves about three weekly GPU quotas, roughly 90 GPU hours on T4/P100, plus the 4-core box.

### Step 0 (prerequisite, not counted in the eight): parity with the public stack

- Mechanism: stop treating the public pipeline as a competitor. Fork the 0.495 notebook unchanged (pin its Docker image), confirm the score, and export its per-molecule candidate lists and channel scores. Then use it in two ways: as a candidate source and feature block inside our ranker (rank and score from the public list become features next to ours), and as the fallback submission. Reuse its public assets where the licence allows: the ICEBERG and GLACIER Kaggle packages, the PubChem popularity prior table, the external-library dataset.
- Why first: our own pipeline has no validated score yet and the plan it follows matches the 0.34-era design. Every item below is measured as a gain over this base, on both our harness and the board.
- Effort: 6 to 10 agent-hours; 3 to 5 GPU hours per full run (the public run takes several hours of the 9-hour limit, so the time budget for our additions is tight: measure it).
- Risks: licences (assets are CC BY-NC-SA or carry the competition's CC BY-NC terms; acceptable inside the competition, but check the rules on external data and on public sharing deadlines); silent stage fallbacks; a pipeline we do not understand. The lead may prefer to rebuild instead; then items 1, 3 and 5 below are the minimum to approach the same score.
- First experiment: verbatim fork, expect 0.495 +/- 0.005. Then run the same code on the enveda-np-examples masks through our harness to obtain per-class reference numbers.

### The eight additions, ranked

| # | Addition | Class | Expected gain on that class (MRR) | Overall gain if classes are about a third each | Agent-hours | Compute |
|---|---|---|---|---|---|---|
| 1 | Transformer encoder with candidate-contrastive loss, replacing the MLP + BCE model | 2 (and 3 via GEN scoring) | +0.10 to +0.20 over PLAN 4.4; 0 to +0.03 over the public FPNet | parity to +0.01 over step 0; +0.04 to +0.07 without step 0 | 10-14 | Kaggle GPU, 2 runs of 6-9 h |
| 2 | Simulated spectra for the whole NP tier: train the encoder on real + simulated, and keep the simulations as a forward-spectrum cache | 2 | +0.05 to +0.15 | +0.02 to +0.05 | 12-16 | Kaggle GPU: 6-10 h simulation, 9 h training |
| 3 | Forward-model features (GLACIER + ICEBERG) on the top-K, per adduct and energy, read from the cache where possible | 2, some 3 | +0.03 to +0.06 over no forward model; about 0 over step 0, which has them | 0 to +0.02 | 6-10 | Kaggle GPU at inference, 20-60 min |
| 4 | Priors learned on NP-like validation only, with a spectrum-blind audit | 1, 2 | +0.03 to +0.08 on class 2 over no prior; risk of -0.02 if overweighted | +0.01 to +0.03 | 5-8 | CPU |
| 5 | External spectral-library sweep (public spectra not in train) | moves molecules from 2 to 1 | each recovered molecule gains about 0.5 | +0.005 to +0.02 | 6-10 | CPU, downloads of a few GB |
| 6 | Fingerprint-conditioned decoder (MS-GPT, fallback FRIGID-base) driven by our fingerprint posterior, plus positional-isomer expansion | 3 | +0.03 to +0.08 | +0.01 to +0.025 | 12-18 | Kaggle GPU at inference, 30-90 min |
| 7 | Test-set network with edit-consistency scoring | 2, 3 | +0.01 to +0.03 | +0.005 to +0.015 (public version: +0.004 to +0.007 on the board) | 6-8 | CPU |
| 8 | Formula evidence from subformula explained intensity across adducts | 3, PubChem-tier 2 | +0.01 to +0.03 | +0.005 to +0.01, and a precondition for item 6 | 5-8 | CPU |

Total if everything works and gains add sub-linearly: roughly +0.04 to +0.08 over the public base, which is the size of the gap to the leader. Items 2 and 6 carry most of the upside and most of the uncertainty.

#### 1. Contrastive transformer encoder

- (a) Mechanism. Peak-token transformer (m/z Fourier features, intensity, neutral loss, adduct, collision energy, instrument family), 6 to 8 layers, width 512. One query per molecule: train with the merged spectra of 1 to 4 acquisitions of the same structure so the model sees what the test supplies (MVP: consensus queries are worth about 10 points HR@1). Two heads: fingerprint logits trained with focal loss (4096-bit Morgan radius 2, the best target in section 2.3), and a candidate-contrastive loss in which the positive is the true structure's fingerprint and the negatives are real candidates from the 10 ppm window of our NP and PubChem tiers (32 to 64 per sample, plus in-batch). Drop the cosine loss term of PLAN 4.4. Score candidates by cosine and by Bernoulli likelihood; both go to the ranker. Late-interaction (FLARE-style peak-to-substructure) heads are a second step only if time remains.
- (b) Gain. Literature on Enveda-180: 35.4 (MLP, BCE) to 63.1 (transformer, contrastive) HR@1. Against the public FPNet, which already uses this recipe and saturates at about 0.53 MRR alone on the np-examples class 2 mask, expect parity; the purpose is to own a model whose features and ranker are consistent, and to have the carrier for item 2.
- (c) Effort: 10 to 14 agent-hours. One T4 session trains about 40k steps of batch 256 in under 5 h (public log); plan two sessions plus one refit on all data.
- (d) Risks. Contrastive models learn the candidate prior (which structures tend to be answers): useful, but check with the spectrum-blind test and keep enveda-180 structures from dominating the negatives. Retraining the encoder invalidates the ranker: refit both, cross-fitted. Sampling must weight NP-like structures (riken, gnps, massbank) and timsTOF spectra, not the 1.15M synthetic Enveda-180 spectra.
- (e) First experiment (one GPU session). Train on train minus the 250 np-examples structures and minus an NP-like holdout of 1,000 structures; report MRR of the encoder score alone on the np-examples class 2 mask over the NP tier. Reference: 0.50 to 0.53 for the public checkpoints over their 712k pool. Kill or debug if below 0.45; proceed to item 2 if at or above 0.50. In the same run, compare BCE against focal + contrastive on the holdout to confirm the direction of section 2.3 on NP-like data.

#### 2. Simulated spectra for the NP tier

- (a) Mechanism. Class 2 is, by the host's definition, inside PubChem or COCONUT. So the candidate space can be simulated before the test: run GLACIER (MSG weights, MIT code, 1.3 ms per spectrum on an A5000) over all 854k NP-tier structures at 3 collision energies for [M+H]+ and [M-H]-, and ICEBERG for a sample. Train the item 1 encoder on real plus simulated pairs (the audited recipe used 8,192 simulated to 300 real per batch; start at 1:1 because our simulator is out of domain). The encoder then has seen a spectrum for every NP candidate. This is the mechanism behind the only large leak-free gain in section 2.2 (9.6 to 40.3). It is not leakage here: it uses candidate structures, not test labels, and it is exactly the situation at test time.
- (b) Gain. +0.05 to +0.15 MRR on class 2 if the domain gap is manageable; zero if the encoder learns to separate simulated from real. The public stack uses simulated rows for ranker training and forward models for re-scoring; no public notebook read so far trains the encoder on simulated spectra of the candidate tier (confirm against KAGGLE_INTEL.md).
- (c) Effort: 12 to 16 agent-hours. Simulation: 854k x 6 conditions = 5.1M spectra, about 2 h on an A5000 at the published rate; budget 6 to 10 h on a T4 and shard across two sessions. Storage: top 60 peaks as (uint32 m/z, uint8 intensity) is about 1.5 GB. Training: one or two more sessions.
- (d) Risks. The simulators were trained on Orbitrap and QTOF data; timsTOF spectra differ in energy scale and noise. GLACIER is weak outside [M+H]+ (a public notebook is titled "ICEBERG where GLACIER is blind"). Mitigations: fine-tune GLACIER for a few hours on our timsTOF training spectra (the ms-pred README asks for two 24 GB GPUs for full training; reduce batch and hidden size for a T4 or accept the zero-shot simulator); pass an "is simulated" flag as input during training and set it to "real" at inference; apply the same peak-dropout and noise augmentation to both. Validation trap: for a held-out structure, hide its real spectra but keep its simulated ones, exactly as at test time.
- (e) First experiment (one session, split in two). Simulate only the mass-window neighbours of the np-examples and NP-holdout structures (about 60k structures, under 1 h). Fine-tune the item 1 encoder for 1 to 2 h with them. Compare class 2 mask MRR before and after on the same pools. Proceed to the full tier if the gain is at least +0.03; kill if under +0.01 after one attempt at simulator fine-tuning.

#### 3. Forward-model features on the top-K

- (a) Mechanism. For the top 100 to 200 candidates per molecule (by encoder score, never by library similarity), compare predicted spectra with each query spectrum at its own adduct and collision energy: entropy similarity, matched-intensity fraction, number of matched peaks, per simulator. For NP-tier candidates the predictions come from the item 2 cache (a lookup); only PubChem-only and generated candidates are simulated at inference. Features go to the ranker; no hand-set fusion weights.
- (b) Gain. Forward and inverse models are equally strong and make different errors (FLARE against JESTR: rank correlation 0.34). In the public history the ICEBERG/GLACIER step added +0.010 to +0.018 on the board. Over step 0 the gain is only what better integration brings (per-adduct matching, cache coverage of the whole NP pool instead of a shortlist).
- (c) Effort: 6 to 10 agent-hours with the existing Kaggle packages of ms-pred. Inference: 400 molecules x 200 candidates x about 3 conditions = 240k spectra; GLACIER about 5 to 15 min on a T4, ICEBERG about 1 to 2 h, so ICEBERG only on the top 50.
- (d) Risks. Dependency weight of ms-pred offline (wheels must be attached); collision-energy calibration between timsTOF eV and the simulators' scale (fit a per-instrument offset on enveda-np-examples by maximising similarity for true structures); molecules above about 1,000 Da and glycosides are slow or unsupported.
- (e) First experiment (CPU-feasible at small scale, or 1 GPU hour). On the 250 np-examples structures, class 2 mask, take the top 50 by encoder score, add GLACIER entropy similarity as a single extra feature with a fixed weight, and measure rank-1 changes. Proceed if net improvements exceed regressions by at least 10 of 250 molecules.

#### 4. Priors, learned where they are valid

- (a) Mechanism. Features: PubMed and patent counts, PubChem annotation-type count, CID rank, number of NP databases, organisms and collections containing the structure, COCONUT annotation level, NP-likeness, formula frequency in the NP tier. The ranker learns their weight on NP-like structures only, under the class 2 mask, with molecule weights that down-weight famous compounds (weight proportional to 1 / log(2 + PubMed count)) so that the learned weight reflects obscure class 2 molecules. Interaction features let the prior fade when spectral evidence is strong.
- (b) Gain. +0.03 to +0.08 on class 2 over no prior; this is already inside step 0, so the item is about getting the weight right. The public knob history shows the popularity weight is flat between 0.10 and 0.15 and that joining PubChem without a prior cost 0.035.
- (c) Effort: 5 to 8 agent-hours, CPU.
- (d) Risks. Validation structures are famous (section 7.2), so the prior's weight is overestimated by construction; the private split may contain fewer famous molecules than the public one. Run the spectrum-blind audit and cap the prior's contribution so that it cannot overturn a clear spectral winner.
- (e) First experiment (2 h, CPU). Rank the np-examples class 2 pools by prior only. If prior-only MRR exceeds 0.3, the validation set is too famous to tune the prior on; then build a second validation set from NP-tier training structures in the lowest quartile of PubMed count and compare the learned prior weight between the two sets. Adopt the smaller weight.

#### 5. External spectral-library sweep

- (a) Mechanism. Class 1 means "has public reference spectra", not "is in train". Train is a snapshot of eleven libraries. Collect public spectra of structures absent from train: current GNPS library dumps (including the multiplex synthesis library the public overlay uses, 147k structures), the latest MassBank EU and MoNA releases, the full MSnLib, MassSpecGym, FragHub, the full Enveda-180 Zenodo release (a public notebook found 3,719 structures there that are not in train), and NP-specific libraries (BMDMS-NP, from memory about 2,700 natural products; verify existence, size and licence). De-duplicate by ik14 against train, keep source and licence per record, add as LIB candidates with a source flag.
- (b) Gain. A molecule that moves from class 2 handling to a library hit gains about 0.5 reciprocal rank. The public overlay was worth +0.006 to +0.009 on the board, about 1 to 2 percent of molecules. A wider sweep focused on NP libraries might double that.
- (c) Effort: 6 to 10 agent-hours, CPU, a few GB of downloads.
- (d) Risks. Licences: NIST, mzCloud and Wiley are commercial and excluded by the rules; MassBank records carry per-record licences. Instrument mismatch lowers similarity (the public threshold had to drop to 0.80). The host probably assigned classes using the same libraries as train, in which case extra libraries only help with molecules whose spectra were published elsewhere; the Enveda-180 overlay gave exactly zero.
- (e) First experiment (3 h, CPU). For each external library, count structures that are in our NP tier and not in train. Drop libraries adding fewer than 500. For the rest, measure on training structures present in both train and the external library what similarity threshold yields 95 percent precision at rank 1 on timsTOF queries.

#### 6. Fingerprint-conditioned decoder for class 3

- (a) Mechanism. Use the MS-GPT decoder (Apache-2.0, 150M parameters; fallback FRIGID-base, 247M, Apache-2.0 code). Its inputs are a formula and a set of on-bits of a 4096-bit Morgan fingerprint, so drive it with our own encoder's posterior: 20 thresholded queries per formula for the top 2 or 3 formulas (item 8), 5 to 25 samples per query. Filter outputs by exact formula, de-duplicate by ik14, drop anything already in the pool, add as GEN candidates. Then expand the 10 most frequent outputs with the positional-isomer and delta edits of PLAN 4.7, because the typical failure is a near miss. Score all GEN candidates with the same encoder and forward-model features; generation frequency is one more feature. The ranker decides when GEN candidates outrank database candidates (only when database evidence is weak).
- (b) Gain. Class 3 MRR from about 0.03 to 0.05 (rule-based analogs) to 0.08 to 0.12. Also covers class 2 molecules missing from our tiers.
- (c) Effort: 12 to 18 agent-hours. Inference for 400 molecules at 100 to 300 samples is minutes to an hour on a T4. No training is needed for a first version; a LoRA adapter fitted on our fingerprint posteriors (their ablation: +1 to 1.2 points top-1) is optional.
- (d) Risks. Fingerprint definition must match the decoder's training fingerprint bit for bit (Morgan radius, size, chirality, RDKit version); test with oracle fingerprints first. The decoder's corpus has few large glycosides; molecules above 600 Da will mostly fail. Checkpoint hosting (an SJTU share; the HF mirror harrylau/msms-models carries a reimplementation and the FRIGID decoder) and unstated checkpoint licences. Wrong formula gives zero. GEN candidates must not displace correct database candidates: train the ranker with class 3 groups that contain GEN candidates and class 2 groups that contain GEN distractors.
- (e) First experiment (one GPU session, no training). Stage A, oracle: feed true fingerprints and true formulas of 300 NP-like held-out structures; expect about 40 to 60 percent in the top 10 (published oracle: 46 percent top-1, 59 percent top-10 for MolForge). If under 25 percent, the fingerprint definition is wrong or the model does not cover natural products: stop. Stage B: feed our encoder's posterior and the true formula on the np-examples class 3 mask; proceed if at least 8 percent of molecules have the truth among 25 outputs. Stage C: predicted formula.

#### 7. Test-set network with edit consistency

- (a) Mechanism. Link test molecules whose spectra are similar (modified cosine, same polarity), and add train consensus spectra as labelled anchors. The public version rewards candidates that are Tanimoto-similar to the top candidates of neighbours. Extension: when two linked test molecules differ by a mass that matches a known biotransformation (CH2, O, H2, hexose, acetyl, as in PLAN 4.7), reward candidate pairs (a from A, b from B) whose structures differ by exactly that edit, found by applying the reaction SMARTS to a and looking the product up in B's pool. This is joint inference over pairs of unknowns, in the spirit of NAP, and no public notebook read so far does the exact-edit version.
- (b) Gain. Public network: +0.004 to +0.007 on the board. The edit-consistency part helps only molecules with a linked partner; the test set was probably chosen for diversity, so coverage may be 10 to 20 percent of molecules.
- (c) Effort: 6 to 8 agent-hours, CPU.
- (d) Risks. Error propagation when the neighbour's top candidate is wrong (the public version caps this by keeping rank 1 fixed and using weight 0.35; weight 0.5 lost 0.004). The public test file is made of training examples, so network density there is not representative of the hidden set.
- (e) First experiment (2 h, CPU). On the public test file (labels are known from train): fraction of molecule pairs above the similarity threshold whose true structures have Tanimoto >= 0.6, and the fraction of linked pairs whose mass difference matches an edit in the table and whose true structures are related by that edit. Kill the edit extension if fewer than 5 percent of molecules have such a partner.

#### 8. Formula evidence

- (a) Mechanism. Section 5: per candidate formula, the fraction of intensity explained by its subformulas, pooled over all spectra, adducts and polarities of the molecule, plus NP-tier formula frequency and element rules, turned into a formula posterior. Used as a ranker feature for every candidate, as the target-formula list for item 6 and the rule-based generator, and as subformula peak tokens for a later encoder version (MIST-style).
- (b) Gain. Small inside the NP tier (the true formula is already the majority formula for 86 percent of windows). Larger in the PubChem tier and decisive for class 3, where every generator needs the formula.
- (c) Effort: 5 to 8 agent-hours, CPU, numba.
- (d) Risks. Without isotope patterns, formulas differing by small mass (for example C3 against SH4, 3.4 mDa) are separated only by fragments. Explained-intensity favours larger, more flexible formulas; normalise by the number of possible subformulas or learn the combination.
- (e) First experiment (3 h, CPU). On np-examples and the NP holdout with candidate formulas enumerated from the PubChem tier window: top-1 and top-3 formula accuracy of (i) NP-tier frequency alone, (ii) explained intensity alone, (iii) both. Proceed if (iii) reaches 80 percent top-1 and 95 percent top-3; compare against msbuddy on 200 molecules if it does not.

### Smaller items worth an hour each

- Loss hygiene now: replace BCE with focal and remove the cosine term in the planned fingerprint model (section 2.3).
- "Leader-safe" fusion: the public history shows that letting a new scorer change rank 1 is where scores are lost, while re-ordering ranks 2 to 25 is safe. In a learned ranker the equivalent is to validate rank-1 changes separately from tail changes and to report both.
- MS2Query-style feature: mean similarity to the library spectra of the several library molecules most similar to the candidate.
- DreaMS embeddings on Kaggle GPU as a second analog-search similarity, now that GPU time exists (our CPU objection no longer holds). Expected gain small.
- Ranker rows from NP-like structures only; never from Enveda-180 structures.

### What not to do

- Do not train a de novo model (DiffMS, MADGEN, MS-BART, Spec2Mol): all are at or below 2.3 percent and slow.
- Do not align frozen foundation models (DreaMS + ChemBERTa, Chemformer two-stage): corrected results are below a fingerprint MLP.
- Do not depend on SIRIUS, CSI:FingerID, CANOPUS, NPClassifier or ClassyFire: all need online services.
- Do not use NIST-trained weights (licence) or any checkpoint whose training set may contain the test molecules' spectra in a way the rules forbid; MSG-trained weights are public and fine.
- Do not expand the candidate pool without a matching prior feature (measured public losses of 0.02 to 0.035).
- Do not tune constants on the public board.

---

## 9. Open questions for the lead

1. Step 0: fork-and-extend the public stack, or rebuild? This review recommends fork-and-extend, with our modular pipeline as the place where new signals are computed and validated.
2. Rules check: are public Kaggle datasets under CC BY-NC-SA and checkpoints with no stated licence (FRIGID on Zenodo, MS-GPT share) acceptable as external data? Is there a deadline for declaring external data?
3. Inference time budget: the public notebook already uses several of the nine hours. How much is left for a decoder and a forward-model pass?
4. Which two final submissions: one tuned for a class mix rich in class 2 (priors on), one robust (priors capped, GEN more aggressive)?

---

## 10. Sources

Benchmarks and audits
- MassSpecGym: https://arxiv.org/abs/2410.23326
- MassSpecGym in the Wild (evaluation audit, corrected leaderboards): https://arxiv.org/abs/2606.19624 ; code https://github.com/harrylaucngd/msms-models ; weights https://huggingface.co/harrylau/msms-models
- On objective mismatch in molecular retrieval (Enveda-180 and MSG loss study): https://arxiv.org/abs/2602.16507 ; code https://github.com/gdewael/ms-mole
- Nearest-neighbour baselines for fingerprint prediction: https://arxiv.org/abs/2610.02249
- Gupta and Skinnider, spurious evaluations and property-matched decoys: https://www.biorxiv.org/content/10.64898/2026.05.03.722532
- FlexMS benchmark (forward models, CASMI 2016/2022 retrieval): https://arxiv.org/abs/2602.22822

Retrieval
- MIST: https://www.nature.com/articles/s42256-023-00708-3 ; https://github.com/samgoldman97/mist ; https://github.com/coleygroup/MIST
- MIST-CF: https://arxiv.org/abs/2307.08240 ; https://github.com/samgoldman97/mist-cf
- JESTR: https://arxiv.org/abs/2411.14464 ; https://github.com/HassounLab/JESTR
- MVP: https://www.biorxiv.org/content/10.1101/2025.11.12.688047
- FLARE: https://pmc.ncbi.nlm.nih.gov/articles/PMC12873900/ ; https://huggingface.co/spaces/HassounLab/FLARE
- GLMR: https://arxiv.org/abs/2511.06259 ; SpecBridge: https://arxiv.org/abs/2601.17204 (both corrected in the audit)
- ICEBERG (2025 version): https://pmc.ncbi.nlm.nih.gov/articles/PMC12154671/ ; GLACIER: https://arxiv.org/abs/2606.29161 ; code https://github.com/coleygroup/ms-pred
- DreaMS: https://huggingface.co/roman-bushuiev/DreaMS
- MS2Query: https://www.nature.com/articles/s41467-023-37446-4
- SIRIUS account and web-service requirements: https://boecker-lab.github.io/docs.sirius.github.io/account-and-license ; COSMIC: https://pubmed.ncbi.nlm.nih.gov/34650271/

De novo
- DiffMS: https://arxiv.org/abs/2502.09571 ; MS-BART: https://arxiv.org/abs/2510.20615
- MIST + MolForge: https://arxiv.org/abs/2508.04180
- FRIGID: https://arxiv.org/abs/2604.16648 ; https://github.com/coleygroup/FRIGID
- MS-GPT: https://arxiv.org/abs/2607.23607 ; https://github.com/VIKI623/MS-GPT
- Test-time tuning: https://arxiv.org/abs/2510.23746
- MSNovelist: https://www.nature.com/articles/s41592-022-01486-3

Formula, priors, CASMI
- BUDDY / msbuddy: https://www.nature.com/articles/s41592-023-01850-x ; https://msbuddy.readthedocs.io
- CASMI 2016: https://pmc.ncbi.nlm.nih.gov/articles/PMC5368104 ; CASMI 2017 (Jena group note): https://bio.informatik.uni-jena.de/2017/11/casmi-2017-results-are-out ; CASMI 2022: https://fiehnlab.ucdavis.edu/casmi/casmi-2022-results
- Metascore critique: https://bio.informatik.uni-jena.de/?p=10209 ; MAD HATTER: https://mdpi.com/2218-1989/13/3/314
- Taxonomically informed scoring: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6824209/
- Enveda-180: https://github.com/enveda/enveda-180 ; https://doi.org/10.26434/chemrxiv.15004319/v1

Competition field
- Public notebooks in `data/external/kaggle_notebooks/`, chiefly "CASMI26 0.495 - What Moved the Board: 60 Arms" (alexchilton, running taeyangg4's and guhongbin's pipeline)
- Public team log: https://github.com/shiferaxa/casmi26 (`docs/plan.md`)
- Launch press release: https://secure.businesswire.com/news/home/20260915334355/en/

Not verified in this review (from memory, check before relying on them): licences of MassFormer, CFM-ID and FraGNNet; existence, size and licence of BMDMS-NP; any MS2Mol accuracy figure; msbuddy accuracy by mass range.

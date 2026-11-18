# CASMI 2026 - solution plan

Status: design, written before `train.parquet` arrived. Implementers work from this file alone.
Read `docs/COMPETITION.md` first. All paths below are relative to `enveda/` unless they start with `/`.

Conventions used in this document:
- "ik14" = first block of the InChIKey as shipped by a source (train column `inchikey14`, PubChem, COCONUT).
- "mkey" = the metric key: RDKit 2026.03.3 tautomer canonicalisation, then InChIKey first block. Only mkey decides whether a guess is correct. ik14 and mkey agree for most molecules but not all.
- "M" = neutral monoisotopic mass of a molecule.
- Estimates of MRR are informed guesses to rank work, not measurements. Replace them with numbers in `docs/EXPERIMENTS.md` as soon as they exist.

---

## 1. Approach

### 1.1 The shape of the problem

Three facts drive the design.

1. The adduct is given for every test spectrum and precursor m/z is accurate to a few ppm (on the public test file the spread of implied M across a molecule's spectra is 2.1 ppm at the 90th percentile and 5.5 ppm at the maximum, including molecules measured in both polarities). So M is known almost exactly before looking at a single fragment. Every candidate source can be filtered to a window of about 10 ppm around M. This turns "structure elucidation" into "rank the few hundred to few tens of thousands of structures that have this mass".
2. The score is MRR@25 with one correct answer per molecule. Rank 1 is worth 1.0, rank 2 is 0.5, rank 10 is 0.1. Almost all value is in the top 3. Wrong guesses only cost by occupying ranks. So the list should be ordered by estimated probability of being correct, whatever source a candidate came from, and it should always be filled to 25.
3. The class mix is hidden. A design with a hard switch ("this is class 1, answer from the library only") loses badly when the switch is wrong. A design that scores all candidates on one scale degrades gracefully.

### 1.2 One pipeline, one candidate pool, one ranker

For every `molecule_id`:

1. Estimate M from all spectra (section 3).
2. Build one candidate pool, the union of three sources, each candidate tagged with where it came from:
   - LIB - structures in the training library whose mass is inside the window (class 1). They carry spectral-match features against their own reference spectra.
   - DB - structures from the NP tier (COCONUT, NPAtlas, LOTUS, PubChemLite) and from the PubChem tier inside the window (class 2).
   - GEN - structures generated for this molecule by editing spectrally or structurally similar known compounds so that they hit the inferred formula (class 3).
3. Compute features per candidate: spectral match to the candidate's own reference spectra (LIB only), agreement between the candidate's fingerprint and a fingerprint predicted from the query spectra, similarity of the candidate to the structures of the nearest library spectra (analog evidence), database priors, mass error, source flags, and pool-level context (pool size, best library similarity in the pool, and so on).
4. Score every candidate with a single gradient-boosted ranker (LightGBM lambdarank) trained on held-out molecules under simulated class 1, class 2 and class 3 conditions.
5. Sort, deduplicate by mkey, emit the top 25 SMILES.

Why this handles the hidden mix: the ranker sees pool-level context. If a LIB candidate matches the query at 0.9 similarity it goes to rank 1 (class 1 behaviour). If the best LIB similarity is 0.2, the ranker has learned from simulated class 2 and 3 groups that library candidates are then usually wrong and DB candidates take over. If the DB pool is empty or scores uniformly badly, GEN candidates fill the list. No explicit class decision is taken.

A deterministic fallback fusion exists for the period before the ranker is trained and as a safety net (section 4.6).

### 1.3 Per class: what we build and what to expect

Class 1 (structure has public spectra, most of which are in train).
- Mass-window filter on the library, then spectral similarity between each query spectrum and the candidate's reference spectra, same polarity, both per-spectrum and against a merged consensus spectrum (the test contains merged 20/40/60 eV spectra).
- With mass known, the candidate pool from the library is typically 1 to 30 structures, so this is close to solved when the structure really is in train. Expected MRR on true class 1: 0.8 to 0.95. Main loss: isomers with near-identical spectra, and instrument mismatch (Orbitrap or QTOF references against timsTOF queries).

Class 2 (in PubChem or COCONUT, no public spectra).
- Candidates from the DB by mass. Rank by predicted-fingerprint agreement plus analog evidence plus priors.
- This is the classic CSI:FingerID-style retrieval setting. Against all of PubChem, fingerprint-only ranking typically gives top-1 around 10 to 25 percent. Priors change that a lot here because the test is natural-product-like: membership in COCONUT/NPAtlas/LOTUS shrinks the effective pool from thousands to tens. Expected MRR on true class 2: 0.25 to 0.45, the widest uncertainty of all, and the place where most of the engineering effort should go.

Class 3 (not in PubChem).
- Honest position: no method, including GPU diffusion or transformer decoders, reaches meaningful exact-match accuracy on truly novel structures (published exact-match rates on MassSpecGym de novo are around 0 to a few percent). On 4 CPUs we do not train a generative model.
- What is realistic: novel natural products are usually close analogs of known ones (one methyl, hydroxyl, sugar, acyl group, double bond, or a positional isomer away). We infer the formula, find seed structures (nearest library analogs and top DB candidates at nearby formulas), enumerate rule-based edits that land exactly on the target formula, and rank them with the same fingerprint model. Expected MRR on true class 3: 0.01 to 0.05. It is cheap insurance, not a main lever.
- The generator costs classes 1 and 2 almost nothing: GEN candidates carry no database prior and sit at the tail of the list unless library and DB evidence is weak or absent.

### 1.4 What is realistic on 4 CPUs, ranked by expected gain per hour

| Rank | Component | Effort (h) | Helps class | Expected effect | Verdict |
|---|---|---|---|---|---|
| 1 | Exact metric + neutral mass + library lookup by mass with spectral similarity | 4 | 1 | Takes class 1 from 0 to about 0.85 | Build first |
| 2 | NP-tier DB (COCONUT etc.) mass lookup with priors | 3 | 2 | Takes class 2 from 0 to about 0.10-0.20 with no model at all | Build first, gives a valid submission before train arrives |
| 3 | Honest validation harness (class 1/2/3 simulation) | 5 | all | No direct gain, but every later decision depends on it | Build before any modelling |
| 4 | Fingerprint-prediction MLP + candidate scoring | 6 + 3-5 h training | 2, 3 | Class 2 to about 0.25-0.40 | Core model |
| 5 | Analog evidence (kNN over library spectra, Tanimoto to candidates) | 3 | 2, 3 | Training-free, complements the MLP, +0.02-0.05 on class 2 | Build |
| 6 | Learned ranker fusing everything | 4 | all | +0.03-0.08 overall versus hand fusion, and robustness to class mix | Build |
| 7 | Full PubChem tier | 4 (mostly waiting) | 2 | Recovers class 2 molecules absent from NP tier; needed for coverage, risk of diluting ranks if priors are weak | Build, gate by validation |
| 8 | Formula inference + rule-based analog generator | 8 | 3 | +0.01-0.05 on class 3 only | Build after 1-6 work |
| 9 | DreaMS pretrained embeddings (TorchScript, CPU) for analog search | 4 + embedding time | 1, 2 | Possibly better analog retrieval across instruments; unknown CPU throughput | Timeboxed spike only |
| 10 | Forward spectrum prediction (ICEBERG) to rerank top candidates | 8+ | 2 | Possible +0.02; heavy dependencies, slow on CPU | Do not start unless everything else is done |
| 11 | Training a de novo generative model | days, GPU | 3 | Near zero exact-match | Do not build |

Things deliberately left out: SIRIUS (needs a login and online services for structure search, Java, not usable offline on Kaggle), MIST-CF (heavy, GPU-oriented), extra spectral libraries (train already aggregates GNPS, MassBank, MoNA, RIKEN, MSnLib, MS-DIAL; FragHub is a merge of the same sources and is a possible later addition, see section 8).

---

## 2. Candidate databases

### 2.1 Sources

Verified on 2026-11-18 by directory listing or HEAD request unless marked otherwise. Nothing large was downloaded.

| Source | URL | Size | Verified | Use |
|---|---|---|---|---|
| PubChem CID-SMILES | `https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz` | 1.49 GB | HEAD 200, dated 2026-11-17 | SMILES per CID |
| PubChem CID-Mass | `.../Compound/Extras/CID-Mass.gz` | 1.3 GB | listing | formula, monoisotopic mass, exact mass per CID |
| PubChem CID-InChI-Key | `.../Compound/Extras/CID-InChI-Key.gz` | 6.9 GB | listing | InChIKey per CID (for ik14 dedup); stream it, do not keep it |
| PubChem CID-PMID | `.../Compound/Extras/CID-PMID.gz` | 311 MB | listing | literature count per CID (popularity prior) |
| PubChem CID-Synonym-filtered | `.../Compound/Extras/CID-Synonym-filtered.gz` | 1.0 GB | listing | optional: synonym count prior. Skip unless priors prove valuable |
| PubChem CID-Patent | `.../Compound/Extras/CID-Patent.gz` | 6.1 GB | listing | not used (too big for the value; PubChemLite carries patent counts for the relevant subset) |
| COCONUT full CSV | `https://coconut.s3.uni-jena.de/prod/downloads/2026-11/coconut_csv-11-2026.zip` | 248 MB | HEAD 200 | NP tier, about 700k structures. Column names NOT verified (zip not opened); expected: identifier, canonical_smiles, standard_inchi_key, molecular_formula, exact mass, np_likeness, annotation_level, citation/organism counts |
| COCONUT lite CSV | `.../2026-11/coconut_csv_lite-11-2026.zip` | 199 MB | HEAD 200 | fallback if the full file is awkward |
| NPAtlas | `https://www.npatlas.org/static/downloads/NPAtlas_download.tsv` | 34 MB | HEAD 200, header read | microbial NPs. Columns include `compound_inchikey`, `compound_smiles`, `compound_accurate_mass`, `compound_molecular_formula`, `origin_type`, `gnps_ids` |
| LOTUS metadata | Zenodo record 7534077, `structure_metadata.tsv.gz` | 96 MB | Zenodo API | NP structures with organism links (2022 freeze). Columns not verified |
| PubChemLite + CCS | Zenodo record 22953182, `PubChemLite_CCSbase_20260925.csv` | 225 MB | Zenodo API, header read | about 400k annotated PubChem compounds. Columns: `Identifier` (CID), `FirstBlock` (ik14), `PubMed_Count`, `Patent_Count`, `MolecularFormula`, `SMILES`, `InChIKey`, `MonoisotopicMass`, `AnnoTypeCount`, `BioPathway`, `FoodRelated`, `DrugMedicInfo`, ... |
| MassSpecGym molecule files | HF dataset `roman-bushuiev/MassSpecGym`, `data/molecules/...` | 0.3-11 GB each | HF API listing | not needed; PubChem FTP is the primary source. Listed as a fallback (`MassSpecGym_retrieval_molecules_pubchem_118M.tsv`, 11.4 GB) |

Class 2 is defined by the organisers as "in PubChem or COCONUT", so PubChem plus COCONUT is by construction a complete candidate source for class 2. NPAtlas, LOTUS and PubChemLite add no coverage beyond that; they add priors.

### 2.2 Filters (applied to every source)

- Single component: no `.` in SMILES.
- Net formal charge zero. Reason: none of the ten test adducts is `[M]+` or `[M]-`, and the organisers dropped spectra whose precursor is inconsistent with structure plus adduct, so permanently charged species cannot appear (zwitterions with net charge 0 are kept). In PubChem `CID-Mass` charged formulas carry a `+`/`-` suffix; drop those.
- Monoisotopic mass 100 to 1300 Da (test range is 157 to 1159; margin for safety).
- Elements subset of {C, H, N, O, P, S, F, Cl, Br, I}. WP6 must check this against train labels of the NP-heavy libraries and widen it (B, Si, Se) only if more than 0.5 percent of those labels need it.
- At least 3 carbon atoms. No isotope labels (`[13C]`, `[2H]`, ...): drop SMILES containing an isotope specification.
- Must parse in RDKit 2026.03.3. For PubChem this is checked lazily at inference for candidates actually retrieved, not for all 100M rows.
- Deduplicate on ik14 within each source (stereoisomers collapse; keep the row with the lowest CID or the highest annotation level).

### 2.3 What is stored

Two tiers with the same schema, so lookup code is shared.

`data/processed/db/np/` (NP tier, one file `np.parquet`, sorted by `mono_mass`), expected about 0.9-1.1M rows, about 80 MB:

| Column | Type | Note |
|---|---|---|
| ik14 | str | source InChIKey first block |
| smiles | str | as shipped by the best source (COCONUT preferred), stereo kept |
| formula | str | Hill notation, recomputed with RDKit for the NP tier |
| mono_mass | f64 | recomputed with RDKit for the NP tier |
| in_coconut, in_npatlas, in_lotus, in_pubchemlite | bool | membership flags |
| coconut_annotation_level | i8 | null if absent or column not available |
| coconut_np_likeness | f32 | null if absent |
| n_organisms, n_citations | i32 | from COCONUT/LOTUS if available, else null |
| pubmed_count, patent_count, anno_type_count | i32 | from PubChemLite |
| cid | i64 | PubChem CID if known |
| fp | fixed binary(512) | 4096-bit Morgan radius 2, precomputed for the NP tier only (about 0.5 GB) so inference skips fingerprinting here. Store in a sidecar `np_fp.npy` (uint8, packed) aligned to row order |

`data/processed/db/pubchem/mass_XXXX.parquet` (PubChem tier, one shard per 10 Da of mono_mass, 120 shards, rows sorted by mono_mass inside a shard), expected 75-95M rows, 3-4.5 GB total with zstd:

| Column | Type | Note |
|---|---|---|
| ik14 | str | from CID-InChI-Key |
| smiles | str | from CID-SMILES, lowest CID per ik14 |
| formula | str | from CID-Mass |
| mono_mass | f64 | from CID-Mass |
| cid | i32 | lowest CID in the ik14 group |
| pmid_count | i32 | max over the group, 0 if none |
| n_cids | i16 | number of CIDs collapsed into this ik14 (stereo-variant count, a weak popularity signal) |

No fingerprints and no NP-likeness are precomputed for the PubChem tier (4096 bits for 90M rows would be 46 GB). They are computed at inference for retrieved candidates only. Measured here: SMILES parse plus Morgan fingerprint costs 0.1-0.3 ms per molecule on one core, so 10M candidates take about 10 minutes on 4 cores.

The NP tier is a subset of the union used for class 2 retrieval. At lookup time the two tiers are concatenated and deduplicated on ik14, with NP-tier rows winning (they carry richer priors).

### 2.4 Priors available to the ranker

- Tier flags: in LIB, in_coconut, in_npatlas, in_lotus, in_pubchemlite, PubChem only.
- Counts (log1p): pubmed_count or pmid_count, patent_count, anno_type_count, n_organisms, n_citations, n_cids.
- NP-likeness: RDKit Contrib `NP_Score` (`publicnp.model.gz` is present in the installed RDKit, verified), computed at inference for candidates. Use COCONUT's own value only as a cross-check.
- Formula frequency: how many DB rows share the candidate's formula at this mass (formulas with many NP-tier entries are more plausible).
- Training-set chemical-space prior: max Tanimoto of the candidate to any library structure found by analog search (section 4.3).

Warning that implementers must respect: popularity priors look better in validation than they are. See section 5.6.

### 2.5 Disk and RAM budget (this machine: 31 GB free, 15 GB RAM)

| Item | Disk | Lifetime |
|---|---|---|
| PubChem raw downloads (SMILES, Mass, PMID) | 3.1 GB | delete after WP3 |
| CID-InChI-Key | 0 GB if streamed (`curl | zcat | cut`), else 6.9 GB | never keep |
| PubChem intermediate parquet | about 6 GB | delete after WP3 |
| PubChem tier final | 3-4.5 GB | keep, upload to Kaggle |
| NP tier raw + final | 0.6 GB + 0.6 GB | keep final |
| train.parquet | 3 GB | keep |
| processed spectra, library index, features | 4-6 GB | keep |
| models | under 0.5 GB | keep |
| Peak | about 20 GB | leaves about 10 GB headroom |

RAM: the PubChem build must be streaming (DuckDB with `memory_limit='8GB'` and a temp directory under `data/external/tmp`, or polars lazy with `sink_parquet`). Never load a full PubChem column into pandas.

---

## 3. Neutral mass and formula handling

### 3.1 Adduct table

All test adducts are singly charged. M = precursor_mz - shift. Values computed with RDKit isotope masses and electron mass 0.000548580 Da.

| Adduct | shift (Da), precursor_mz = M + shift | Polarity |
|---|---|---|
| [M+H]+ | +1.007276 | positive |
| [M+NH4]+ | +18.033826 | positive |
| [M-H2O+H]+ | -17.003288 | positive |
| [M-2H2O+H]+ | -35.013853 | positive |
| [M+Na]+ | +22.989221 | positive |
| [M+K]+ | +38.963158 | positive |
| [M-H]- | -1.007276 | negative |
| [M-H2O-H]- | -19.017841 | negative |
| [M+CH2O2-H]- | +44.998203 | negative |
| [M+Cl]- | +34.969401 | negative |

Implement as a dict in `casmi/adducts.py` with a parser that also understands the general pattern `[xM+A-B]n+` for train (multimers, doubly charged, other adducts). Train spectra with adducts the parser cannot handle are kept for the library but flagged `adduct_known=False`. An unknown adduct in the hidden test (should not happen) falls back to `[M+H]+` or `[M-H]-` by `ionization_mode` and is logged.

### 3.2 Aggregating a molecule's spectra into M

1. M_i = precursor_mz_i - shift(adduct_i) for every spectrum.
2. M = median of M_i. If all |M_i - M| / M are under 15 ppm, done.
3. Otherwise (a mislabelled adduct or a bad precursor): cluster M_i with a 15 ppm linkage, keep the largest cluster, break ties in favour of the cluster containing `[M+H]+` or `[M-H]-`, set M to that cluster's median, and mark the outlier spectra `mass_outlier=True`. Outlier spectra are still used for fragment evidence but with their own precursor for neutral-loss features.
4. Record `m_spread_ppm` (max minus min over the kept cluster) and `n_spectra`, `n_adducts`, `has_pos`, `has_neg` as molecule-level features.

Mixed adducts and polarities are an asset: 108 of 400 public test molecules have more than one adduct and 97 have both polarities. Their agreement confirms M; nothing else is needed from them at this step.

### 3.3 Tolerances

- Candidate retrieval window: |mass - M| <= max(10 ppm x M, 0.002 Da). Default until train arrives.
- WP6 refines it: take `precursor_error_ppm` for `enveda-180` and `enveda-np-examples` restricted to the ten test adducts, and the same quantity for the 1,213 public test spectra joined back to their train labels. Set the window to the 99.5th percentile of |error| of the per-molecule median, rounded up, but not below 5 ppm. The organisers' own cut-off for "inconsistent precursor" is unknown; the public test file shows what survived it.
- The absolute ppm error of each candidate is a ranker feature, so the window can be generous.
- Fragment matching tolerance: 0.01 Da for timsTOF against timsTOF, 0.02 Da across instruments. Tune on validation (section 5).

### 3.4 Formula

For LIB and DB candidates the formula is a property of the candidate; we never need to commit to one formula. Formula inference is only needed for GEN (which formula should generated structures have) and as a ranker feature:

1. Enumerate formulas over C, H, N, O, P, S, F, Cl, Br, I within the mass window, with integer and non-negative ring-plus-double-bond equivalents, Senior/valence check, and loose element-ratio rules (H/C 0.2-3.1, N/C <= 1.3, O/C <= 1.2, P <= 2, S <= 3, halogens <= 4 in total). At 350 Da and 10 ppm this is typically 5 to 40 formulas.
2. Score each formula by:
   - `expl_frac`: fraction of query intensity (after cleaning) whose fragment m/z can be written as a sub-formula of (formula + adduct), within the fragment tolerance, pooled over the molecule's spectra. Implement by precomputing sub-formula masses per candidate formula (bounded enumeration, cached).
   - `db_count_np`, `db_count_pubchem`: number of DB structures with that formula.
   - `lib_count`: number of library structures with that formula.
   - mass error in ppm.
3. Combine with a small logistic model fitted on held-out molecules (target: formula equals true formula). Output a probability per formula; GEN uses the top 3 formulas with probability above 0.05.

`msbuddy` 1.4.1 installs on Python 3.13 (dry-run verified) and can be tried as an alternative scorer, but it must not become a Kaggle dependency unless it clearly wins; the scheme above needs only numpy.

---

## 4. Scoring and ranking models

### 4.1 Spectrum preprocessing (shared by everything)

`casmi/spectra.py`, one function `clean(mz, inten, precursor_mz) -> (mz, inten)`:
1. Drop peaks with m/z > precursor_mz + 2 Da (already done in test; apply to train so both match).
2. Record the precursor peak intensity separately as `prec_intensity` (a feature: surviving precursor signals low fragmentation), then exclude peaks in [precursor_mz - 0.5, precursor_mz + 2] from the similarity and model vectors by default (flag `keep_precursor`, tune on validation).
3. Relative intensity floor 0.5 percent of base peak (test spectra have a median of 230 peaks, mostly noise).
4. Keep the top 100 peaks by intensity.
5. Renormalise to base peak 1. Models use sqrt intensity.

Variants (floor 0.1/0.5/1/2 percent, top 60/100/150) are a validation knob; fix one default in `config.py` and do not let modules diverge.

Consensus spectra: for each (structure, polarity, adduct family, instrument family) in the library, and for each test molecule and polarity, build a merged spectrum: pool cleaned peaks from all member spectra, merge peaks within 0.01 Da (intensity-weighted m/z, summed intensity), renormalise. Test already contains merged 20/40/60 spectra, so library consensus spectra are the natural counterpart.

### 4.2 Library matching (class 1)

- Library index: all train spectra after cleaning, stored as flat arrays (`mz` float32, `inten` float32, offsets) plus a metadata table, sorted by structure mass. Plus consensus spectra.
- For a test molecule: take library structures inside the mass window. For each (query spectrum, reference spectrum) pair of the same polarity compute two similarities: cosine on sqrt intensities with greedy peak matching at the fragment tolerance, and spectral entropy similarity. Own implementation in numpy/numba (`casmi/similarity.py`); `matchms` 0.34 and `ms_entropy` 1.5.3 install here and are used only to cross-check the implementation in tests, never at inference.
- Features per LIB candidate: max and mean over query spectra of (max over the candidate's references); the same restricted to timsTOF references; the same restricted to same-adduct pairs; similarity of query consensus to reference consensus; number of matched peaks at the best pair; number of references; whether any reference is timsTOF; collision-energy distance at the best pair.
- Neutral-loss-aware (modified) cosine is not needed here because query and candidate have the same mass. It is needed in 4.3.

### 4.3 Analog evidence (classes 2 and 3, training-free)

- For each query consensus spectrum, search the whole library without mass restriction, same polarity. Two stages: (a) coarse: sparse matrix product between the query and all library consensus spectra binned at 0.05 Da, fragment channel plus neutral-loss channel (loss = precursor - fragment), sqrt intensity, L2-normalised; a scipy CSR matrix of about 600k rows x 40k columns with at most 100 non-zeros per row is about 0.5 GB and one product takes milliseconds; (b) exact modified cosine on the top 200.
- Keep the top K = 20 hits (excluding, in validation, anything the protocol removed).
- Candidate features: max and similarity-weighted mean of Tanimoto(candidate, hit structure) over the hits; Tanimoto to the single best hit; the best hit's spectral similarity; mass difference between query and best hit and whether it equals a common biotransformation delta (section 4.7).
- The hit list also seeds the class 3 generator.

### 4.4 Fingerprint prediction model (classes 2 and 3)

The main trained model. Chosen because it trains in hours on CPU, its inference is a matrix product, and one prediction per molecule scores any number of candidates.

- Target: 4096-bit Morgan radius 2 (binary, chirality off) concatenated with 167 MACCS keys, computed from `normalized_smiles` after a light standardisation shared with candidates (`casmi/chem.py: std_mol`: RDKit cleanup, uncharge, strip stereo; no tautomer canonicalisation, it is too slow for millions of candidates: measured up to 300 ms for a glycoside).
- Input per spectrum (sparse, about 100 active peaks):
  - fragment channel: m/z binned at 0.1 Da from 0 to 1300 (13,000 bins), value sqrt intensity;
  - neutral-loss channel: (precursor_mz - mz) binned at 0.1 Da from 0 to 700 (7,000 bins);
  - dense metadata (about 40 values): adduct one-hot (ten test adducts plus "other"), polarity, collision energy in eV / 100 with a missing flag, "merged energies" flag, number of energies, instrument family one-hot (timsTOF, other QTOF, Orbitrap, other/unknown), M / 1000, log10 of base_peak_intensity with a missing flag, number of peaks before cleaning (log).
- Architecture: sparse input through `torch.nn.EmbeddingBag(20000, 1024, mode='sum')` with per-sample weights (equivalent to a sparse linear layer and fast on CPU), concatenated with a 64-unit projection of the metadata, then 2 residual blocks of width 1024 (Linear, LayerNorm, GELU, dropout 0.1), then Linear to 4263 logits. About 27M parameters, about 110 MB in float32, 55 MB in float16.
- Loss: binary cross-entropy with per-bit positive weights clipped to [1, 10], plus 0.5 x (1 - cosine(sigmoid(logits), target)).
- Training data: all train spectra whose label mass is 100-1300, at least 5 peaks after cleaning, adduct parseable, |precursor_error_ppm| <= 20 (label-quality filter). Holdout structures (section 5) are excluded. Sampling: each epoch draws spectra with probability proportional to 1 / sqrt(number of spectra of that structure), so that the 1.15M enveda-180 spectra do not dominate and structures with 50 spectra do not swamp those with 2. Spectra with test adducts and timsTOF instrument get weight 2.
- Augmentation (cheap, on the fly): drop each peak with probability 0.1; multiply intensities by exp(N(0, 0.3)); add up to 10 random noise peaks at 0.5-2 percent intensity to non-timsTOF spectra (test spectra are noisier than curated libraries); with probability 0.3 replace the spectrum by the merged consensus of 2-3 spectra of the same structure and adduct at different energies (emulates the merged 20/40/60 test spectra).
- Optimiser: AdamW, lr 1e-3 with cosine decay, batch 512, `torch.set_num_threads(3)`. Budget: 3 to 5 hours wall. First measure throughput on 50k spectra and set the epoch count to fit the budget (expected 4-8k spectra per second, so an epoch of 2.3M spectra takes 5-10 minutes).
- Inference without torch: export weights to `.npz`; `casmi/fpmodel.py: predict_numpy` does the embedding-bag sum and the dense layers in numpy. Tested for equality with torch to 1e-4.
- Molecule-level prediction: average logits over the molecule's spectra, weighted by log10(base_peak_intensity) clipped to [3, 6] and divided by 3. Also keep per-polarity averages.
- Candidate scores (all become ranker features): cosine(p, c); Bernoulli log-likelihood sum over bits of c*log(p) + (1-c)*log(1-p) with p clipped to [0.01, 0.99] and calibrated per bit on held-out data; soft Tanimoto sum(p*c) / (sum(p) + sum(c) - sum(p*c)); the same three on MACCS only; rank of the candidate inside its pool under the log-likelihood.

Known weakness: candidate fingerprints come from PubChem/COCONUT tautomer forms, targets from `normalized_smiles`. WP9 must measure the effect (fingerprint of the PubChem SMILES versus the train SMILES for the same ik14, over 10k structures). If the mean Tanimoto is under 0.95, add InChI round-tripping (`MolFromInchi(MolToInchi(mol))`, which normalises mobile hydrogens cheaply) to `std_mol` for both sides and retrain.

### 4.5 The ranker

- LightGBM `lambdarank`, one group per (molecule, condition), label 1 for the candidate whose mkey equals the truth, else 0. `lambdarank_truncation_level` 25, 300-800 trees, learning rate 0.05, `num_leaves` 31, `min_data_in_leaf` 50, feature fraction 0.8. Deterministic (`seed`, `deterministic=True`, `num_threads=4`, `force_row_wise=True`).
- Features: everything from 4.2, 4.3, 4.4, 2.4, mass error ppm, source flags (is_lib, is_np, is_pubchem_only, is_gen), GEN features (4.7), formula probability (3.4), and pool context: pool size per source, best and second-best library similarity in the pool, number of LIB candidates, z-score of the candidate's fingerprint log-likelihood within the pool, molecule-level features from 3.2.
- Missing values are left as NaN (LightGBM handles them); LIB features are NaN for non-LIB candidates.
- Pool cap before ranking: at most 30,000 candidates per molecule. If the PubChem pool is larger, keep all LIB and NP-tier candidates and the PubChem candidates whose formula is among the top formulas by probability until the cap is reached. For training groups, subsample negatives to 2,000 per group (always keep the positive and the top 200 by fingerprint score).
- Training groups come from the holdout molecules, cross-validated by molecule (section 5). Class mix enters as group weights. Default 0.4 / 0.4 / 0.2 for class 1 / 2 / 3; the final choice is the one with the best worst-case over the mixes in section 5.5.
- The final model is trained on all holdout groups. The fingerprint model is refitted on all of train with the same recipe before export (the ranker was fitted on out-of-sample fingerprint scores; a refit model is slightly sharper on class 1 molecules, which only helps the correct LIB candidate).

### 4.6 Deterministic fallback fusion

Used by the baselines, and kept as a guard if the ranker misbehaves. Score = first applicable rule:
1. LIB candidate with best cosine >= 0.7 and at least 4 matched peaks: 3 + cosine.
2. NP-tier or LIB candidate: 1 + z, where z is the rank-normalised fingerprint log-likelihood within the pool (0 to 1), plus 0.1 x log1p(pubmed_count) capped at 0.3.
3. PubChem-only candidate: z.
4. GEN candidate: 0.5 x z.

Before the fingerprint model exists, z is replaced by the prior alone (annotation level, then log counts).

### 4.7 Class 3 generator

`casmi/generate.py`. Deterministic, RDKit only.

- Target formulas: top 3 from section 3.4.
- Seeds: (a) structures of the top 20 analog hits (4.3); (b) the top 50 DB/LIB candidates by fingerprint score at formulas that differ from the target by one or two "delta" units, found by querying the DB at M - delta mass.
- Delta units (formula change, edit): CH2 (O-, N-, C-methylation and demethylation), O (hydroxylation, N-oxide, epoxide; and removal), H2 (hydrogenation of C=C and C=O, dehydrogenation), CO (carbonyl insertion, formylation), C2H2O (acetylation and deacetylation), C5H8 (prenylation), C6H10O5 / C6H10O4 / C5H8O4 / C6H8O6 (hexose, deoxyhexose, pentose, glucuronide; add to OH, or remove), SO3 (sulfation), H2O (dehydration, lactone or ether ring closure and opening), CO2 (carboxylation, decarboxylation), NH vs O swaps (amide/ester, amine/hydroxyl: delta NH-O), C2H4, C3H4O2 (malonyl), C7H4O (benzoyl), C9H6O2 / C9H6O3 (coumaroyl, caffeoyl).
- Zero-delta edits (positional isomers of same-formula seeds): move a hydroxyl, methoxy, methyl, prenyl or glycosyl group to another chemically equivalent attachment atom class; shift a double bond by one position.
- Each edit is a reaction SMARTS applied at every matching site. Up to 2 edits from a seed. Products must sanitise, have exactly a target formula, net charge 0, and not already be in the pool (by ik14). Cap 3,000 products per molecule, prioritised by seed score.
- GEN features for the ranker: number of edits, edit type ids, seed spectral similarity, seed fingerprint score, target formula probability, plus the usual fingerprint scores of the product.
- Activation: always run; the ranker decides. A cheap pre-check skips generation when a LIB candidate has cosine >= 0.9 (saves time; logged).

### 4.8 Final list assembly

1. Sort the pool by ranker score descending. Ties broken by source order (LIB, NP, PubChem, GEN), then ik14 ascending, so output is deterministic.
2. Walk down the list computing mkey (cached on disk, with a per-molecule timeout fallback to ik14) and skip candidates whose mkey was already emitted, until 25 are collected. In practice this touches about 40 candidates per molecule.
3. If fewer than 25 remain, pad from the widened mass window (20 ppm), then with the globally most frequent NP-tier structures at the nearest masses. Never emit fewer than 1 SMILES; never emit an empty string.
4. Emit canonical RDKit SMILES (stereo removed), validated by re-parsing.

---

## 5. Validation protocol

The public leaderboard on the public test file is meaningless (its molecules are in train). All decisions use the local protocol below. The public file is used only to check format and runtime.

### 5.1 Metric

`casmi/metric.py`:
- `mkey(smiles) -> str | None`: `mol = Chem.MolFromSmiles(s)`; `mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)`; `Chem.MolToInchiKey(mol)[:14]`. None if parsing fails.
- `mrr_at_25(pred: dict[mol_id, list[str]], truth: dict[mol_id, str]) -> float`: for each molecule, rank of the first prediction among the first 25 whose mkey equals the truth's mkey, reciprocal, 0 if none; mean over all truth molecules (missing molecules count 0).
- The exact call sequence in the official metric notebook (`https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank`) could not be read from this machine. The user must paste that notebook's code into `docs/METRIC_NOTEBOOK.md`; WP1 then makes `mkey` identical (for example whether a cleanup or largest-fragment step precedes canonicalisation, and whether the enumerator has non-default limits) and adds its examples as tests. Until then the default above stands.
- mkey is slow for large flexible molecules (measured 0.5 ms to 300 ms). Cache in `data/processed/cache/mkey.parquet` keyed by SMILES; compute in a 4-process pool.

### 5.2 Holdout molecules

Defined once by `scripts/make_splits.py`, written to `splits/holdout_v1.csv` (small, committed to git): columns `ik14, mkey, set, fold, eligible_c1`. A structure is held out by mkey: every train row whose mkey equals a holdout mkey is treated as that molecule.

| Set | Size | Source | Query spectra | Why |
|---|---|---|---|---|
| NP-TIMS | all 250 | enveda-np-examples | its enveda-np-examples spectra | Same instrument, pipeline and chemistry as the hidden test. The primary score. |
| SYN-TIMS | 3,000 | enveda-180 structures; 1,500 with the highest RDKit NP-likeness plus 1,500 random; sampled to follow the hidden mass distribution (157-1159, median 348) as far as the library allows | its enveda-180 spectra, test adducts only | Same instrument, many molecules, wrong chemistry. Gives statistical power and ranker training groups. |
| NP-OTHER | 3,000 | structures in riken, gnps, msdial, massbank, mona that are in the NP tier and not in the two Enveda libraries, with at least 2 spectra at test adducts | spectra from one library per molecule (prefer QTOF instruments) | Right chemistry, wrong instrument. Trains the ranker on NP priors without relying on 250 molecules. |

Query emulation: restrict to the ten test adducts, apply the test cleaning (drop peaks above precursor + 2 Da; drop spectra with base_peak_intensity < 1000 where known), keep at most 16 spectra per molecule, sampled to match the public test distribution of spectra per molecule and collision energies.

`fold` is 0-4, assigned by molecule, stratified by set.

### 5.3 Conditions

Each holdout molecule is evaluated under every condition it is eligible for. The library and DB are filtered per condition, per molecule, at lookup time (a mask, not a rebuild).

- C1 (class 1): remove from the library all spectra of this molecule that come from the query's own library (so a query never matches itself or its replicate acquisitions). Spectra of the molecule from other libraries stay. Eligible only if such spectra exist (`eligible_c1`). For NP-TIMS this is exactly the hidden-test situation: timsTOF query, reference from GNPS/MassBank/RIKEN and others.
- C2 (class 2): remove all spectra of this molecule from the library. The structure stays in the DB.
- C3 (class 3): as C2, and remove the structure from every DB tier and from analog-hit lists. Removal is by mkey for candidates with the same formula as the truth (compute mkey for those; if there are more than 5,000, only for those with Tanimoto >= 0.5 to the truth).

In all conditions, the fingerprint model and any other trained model has never seen any holdout molecule's spectra. (For C1 this is slightly pessimistic, since at test time the final model has seen class 1 molecules' reference spectra. Accepted.)

Other molecules' spectra are never removed. Close analogs of holdout molecules remain in the library, as they will for the hidden test.

### 5.4 Cross-fitting

- Fingerprint model "v_holdout": trained once on train minus all holdout molecules (about 6,250 structures, under 3 percent of structures).
- Ranker and formula model: 5-fold grouped CV over holdout molecules using `fold`. Reported numbers are out-of-fold.
- Final models: fingerprint model refitted on all of train; ranker fitted on all holdout groups.

### 5.5 Reporting

`scripts/evaluate.py` prints and appends to `docs/EXPERIMENTS.md` one row per run:

- MRR@25 and top-1, top-5, top-25 hit rates for each (set, condition) cell: 3 x 3 cells.
- Candidate recall per cell: fraction of molecules whose truth is in the pool at all, before ranking (the ceiling).
- Blended MRR for NP-TIMS under four assumed mixes of class 1/2/3: 50/30/20, 30/50/20, 20/40/40, 34/33/33.
- The primary selection number is the minimum over those four blends on NP-TIMS ("worst-case blend"), with the NP-OTHER worst-case blend as tie-breaker. SYN-TIMS is a sanity check, not a selection criterion, because its chemistry is unlike the test.
- Bootstrap 90 percent interval over molecules for the primary number (250 molecules give roughly +/- 0.04; do not chase differences smaller than that).

### 5.6 Known biases of this protocol, and what to do about them

- Popularity bias in simulated class 2. Our "class 2" molecules are really class 1 molecules with their spectra hidden, so they are better known (more literature, more database entries) than true class 2 molecules. Priors such as pubmed_count will look more useful than they are. Mitigations: (a) report C2 separately for molecules with pubmed_count = 0 and for the rest; (b) train a second ranker without count priors (membership flags and NP-likeness only) and keep whichever is better on the pubmed_count = 0 subset of NP-OTHER; (c) give LightGBM monotone constraints on count features and limit their depth of interaction.
- A related signal that is real, not a bias: for a true class 2 or 3 molecule, any candidate that has library spectra is wrong by definition. The ranker learns this from C2/C3 groups through the pool-context features. Do not remove those features.
- NP-TIMS molecules are "common natural products". Hidden class 2/3 molecules will be rarer and probably larger. Watch the NP-OTHER cell, which has a wider spread.
- Class 3 simulation only tells us whether the generator can recover a known molecule from its neighbours. Enveda's actual novel structures may be further from known chemistry. Treat C3 numbers as an upper bound.

### 5.7 Use of the public test file

After train arrives, WP6 joins the 1,213 public spectra to train (by exact precursor and peak arrays, or spectrum id if shared) to learn: which libraries they come from, the precursor ppm error distribution after the organisers' cleaning, adduct and energy distributions per molecule. This calibrates 3.3 and the query emulation in 5.2. Submissions built from the full pipeline will score near 1.0 on the public leaderboard; that only confirms the plumbing.

---

## 6. Work packages

Tags: NOW = can be completed without train. NEEDS_TRAIN = needs `data/raw/train.parquet`. "code NOW" means the code and unit tests can be written now against `test.parquet` and synthetic fixtures, while the real run waits for train.

CPU budget: the machine has 4 cores and 15 GB RAM. Heavy jobs are marked with the cores they may use. Never run two "4 core" jobs together.

### Dependency overview

```
WP0 -> WP1 -> WP4 (first submission, NOW)
WP0 -> WP2 -> WP4
WP0 -> WP3
WP0 -> WP5 (code) ----\
WP0 -> WP9 (code) ----+--> after train: WP6 -> WP7 -> WP8 (second submission)
                      |                  WP6 -> WP5 (build) -> WP8
                      |                  WP7 -> WP9 (train) -> WP10 -> WP11 -> WP14
                      |                  WP10 -> WP12 -> WP11 (refit with GEN)
WP13 (optional spike, NOW)                                    WP15 runs alongside from WP8 on
```

### WP0 - Scaffolding and conventions [NOW]
- Goal: package skeleton that every other WP imports, so parallel work does not collide.
- Inputs: this plan.
- Outputs: `pyproject.toml` (package `casmi`, src layout, `pip install -e .`), `requirements.txt` (pinned: numpy, pandas, polars, pyarrow, scipy, scikit-learn, rdkit==2026.3.3, lightgbm, numba, duckdb, torch CPU as an optional extra, pytest), `src/casmi/__init__.py`, `src/casmi/config.py`, `src/casmi/paths.py`, `src/casmi/log.py`, `tests/conftest.py`, `scripts/README.md`, `docs/EXPERIMENTS.md` (header only), `docs/ARTEFACTS.md` (header only), `.gitignore` additions (`data/`, `models/`, `logs/`, `dist/` stay ignored; `splits/` and `submissions/` are tracked).
- Acceptance: `pip install -e .` works; `python -c "import casmi; print(casmi.paths.RAW)"` prints the raw data path; `pytest -q` runs (zero tests is fine).
- Estimate: 30 minutes, 1 core, negligible RAM.
- Dependencies: none. Must be merged before anything else starts.

### WP1 - Chemistry core, adducts, metric, submission IO [NOW]
- Goal: the exact metric, mass arithmetic and submission writer.
- Inputs: `data/raw/test.parquet`, `data/raw/sample_submission.csv`.
- Outputs: `src/casmi/adducts.py` (table from 3.1, parser, `neutral_mass`), `src/casmi/chem.py` (`std_mol`, `formula`, `mono_mass`, `morgan_fp`, `maccs_fp`, batch helpers with multiprocessing, `np_score`), `src/casmi/metric.py` (`mkey` with disk cache, `mrr_at_25`), `src/casmi/io.py` (`read_test`, `molecule_table` implementing 3.2, `write_submission`, `validate_submission`), tests `tests/test_adducts.py`, `tests/test_metric.py`, `tests/test_io.py`.
- Acceptance: adduct shifts equal the table in 3.1 to 1e-5; for ten known compounds the computed precursor m/z from SMILES plus adduct matches hand values; `molecule_table` on the public test gives 400 rows with M between 246.10 and 439.11 and max spread 5.5 ppm; `mkey` maps keto and enol forms of a test pair to the same key and two stereoisomers to the same key; `validate_submission` accepts `sample_submission.csv` and rejects: duplicate id, missing id, 26 guesses, empty cell, unparseable SMILES (warning only for the last); `mrr_at_25` equals hand-computed values on a 5-molecule fixture.
- Estimate: 2 hours, 1 core.
- Dependencies: WP0.
- Parallel with: WP2, WP3, WP5 code, WP9 code.

### WP2 - NP-tier database [NOW]
- Goal: `np.parquet` with priors and fingerprints.
- Inputs: COCONUT CSV zip, NPAtlas TSV, LOTUS `structure_metadata.tsv.gz`, PubChemLite CSV (URLs in 2.1; all under 250 MB each).
- Outputs: `scripts/build_npdb.py`, `src/casmi/db.py` (shared lookup API: `lookup(mass, tol_ppm, tiers) -> DataFrame`, used by both tiers), raw files in `data/external/{coconut,npatlas,lotus,pubchemlite}/`, `data/processed/db/np/np.parquet`, `data/processed/db/np/np_fp.npy`, a section in `docs/ARTEFACTS.md` with row counts, sha256 and the actual column names found in each source.
- Acceptance: row count between 0.6M and 1.5M after filters; ik14 unique; mono_mass recomputed by RDKit agrees with the source mass within 0.001 Da for 99.9 percent of rows; lookup of M = 302.0427 (quercetin) at 10 ppm returns ik14 `REFJWTPEDVJJIY` with `in_coconut` true; a lookup takes under 50 ms after load; the first step of the script prints each source's header, and the script fails loudly if an expected column is missing rather than guessing.
- Estimate: 2 hours including downloads; fingerprints for about 1M molecules take about 5 minutes on 2 cores; RAM under 4 GB.
- Dependencies: WP0 (uses `chem.py` helpers from WP1; if WP1 is not merged yet, call RDKit directly and switch afterwards).
- Parallel with: WP1, WP3 (WP3 is mostly I/O bound).

### WP3 - PubChem tier [NOW]
- Goal: the sharded, filtered, deduplicated PubChem candidate table.
- Inputs: PubChem Extras files (2.1).
- Outputs: `scripts/build_pubchem.py` (resumable, stage markers in `data/external/pubchem/.stage_*`), `data/processed/db/pubchem/mass_0100.parquet` ... `mass_1290.parquet`, `data/processed/db/pubchem/_index.json` (per shard: row count, min and max mass, sha256), entry in `docs/ARTEFACTS.md`.
- Method: stage 1 stream each gz file into parquet keyed by CID (`curl | zcat | cut` for the InChIKey file, keeping only CID and the first 14 characters of the key); stage 2 DuckDB join on CID with `memory_limit='8GB'`, apply the filters of 2.2 that can be done on strings (formula charge suffix, element whitelist via regex on the formula, `.` and isotope patterns in SMILES, mass range), write partitioned by floor(mass / 10); stage 3 per shard: group by ik14, keep the lowest CID row, attach max pmid_count and n_cids, sort by mass, write zstd level 9.
- Acceptance: total rows between 60M and 110M (log the count before and after each filter); ik14 unique within each shard; caffeine (`RYYVLZVUVIJVGH`, M 194.0804) and paclitaxel (`RCINICONZNJXQF`, M 853.3310) found by `db.lookup`; for 20 random masses from the public test, lookup of both tiers at 10 ppm returns in under 2 seconds each and the candidate counts are logged (expected hundreds to tens of thousands); total size on disk under 6 GB; raw and intermediate files deleted at the end.
- Estimate: 3 to 4 hours wall (downloads about 10 GB, about 120M rows), 2 cores, 8 GB RAM peak, 16 GB disk peak.
- Dependencies: WP0. Uses the `db.py` API from WP2; agree on the schema in 2.3 and do not change it.
- Parallel with: WP1, WP2, WP5 code, WP9 code. Do not run together with fingerprint-model training.

### WP4 - Baseline 0: first valid submission [NOW]
- Goal: a valid file in `submissions/` before train arrives, and the end-to-end skeleton (`pipeline.py`) that later baselines extend.
- Inputs: WP1, WP2 outputs; `test.parquet`.
- Method: per molecule, M from 3.2; NP-tier lookup at 10 ppm; rank by (coconut_annotation_level desc, log1p(pubmed_count + n_citations) desc, |ppm error| asc); pad per 4.8.
- Outputs: `src/casmi/pipeline.py` (function `predict(test_df, cfg) -> dict[mol_id, list[smiles]]` with pluggable stages), `src/casmi/fusion.py` (4.6), `scripts/predict.py` (CLI: `--test`, `--config`, `--out`), `submissions/YYYYMMDD_b0_npdb_prior.csv`.
- Acceptance: passes `validate_submission`; 400 rows; at most 25 guesses each; deterministic (two runs give identical bytes); runtime under 2 minutes; log the fraction of molecules with an empty NP-tier pool.
- Estimate: 1 hour, 1 core.
- Dependencies: WP1, WP2.
- Note: the public test molecules are mostly synthetic screening compounds, so expect many empty NP-tier pools and a low score if submitted. Its purpose is the plumbing. When WP3 finishes, produce `YYYYMMDD_b0b_pubchem_prior.csv` the same way with both tiers.

### WP5 - Spectrum processing, similarity, library index [code NOW, build NEEDS_TRAIN]
- Goal: cleaned spectra store, consensus spectra, similarity functions, mass-restricted and open library search.
- Inputs: `test.parquet` now; `data/processed/train/` from WP6 later.
- Outputs: `src/casmi/spectra.py` (4.1: `clean`, `merge_consensus`, `bin_sparse`), `src/casmi/similarity.py` (cosine, modified cosine, entropy similarity; numba), `src/casmi/library.py` (`Library.build`, `Library.load`, `search_by_mass`, `search_open`, per-call exclusion mask for validation conditions), `scripts/build_library.py`, tests `tests/test_spectra.py`, `tests/test_similarity.py`; after train: `data/processed/library/{peaks_mz.npy, peaks_int.npy, offsets.npy, meta.parquet, consensus_*.npy, consensus_meta.parquet, open_index_pos.npz, open_index_neg.npz}`.
- Acceptance (code): similarity functions agree with matchms `CosineGreedy`, `ModifiedCosine` and ms_entropy on 1,000 random pairs of public test spectra to 1e-6 (install those two packages in the dev environment only); identical spectra score 1.0; `clean` reduces the public test median peak count from 230 to at most 100 and keeps the base peak. Acceptance (build): self-search of 1,000 random library spectra returns themselves at rank 1; mass-restricted search for a 400-molecule query set takes under 1 minute; open search under 5 minutes; library files total under 5 GB; RAM under 8 GB when loaded.
- Estimate: code 4 hours; build 1 hour wall on 3 cores, 8 GB RAM.
- Dependencies: WP0, WP1; build needs WP6.
- Parallel with: WP1-WP3 (code phase).

### WP6 - Train ingestion, structure table, data audit [NEEDS_TRAIN]
- Goal: turn the 3 GB file into things that fit in RAM, and answer the open data questions.
- Inputs: `data/raw/train.parquet`, `test.parquet`.
- Outputs: `scripts/ingest_train.py`; `data/processed/train/spectra_<lib>.parquet` (cleaned peaks as float32 lists, all metadata, row-group streaming with pyarrow, never the whole file in memory); `data/processed/train/structures.parquet` (one row per ik14: normalized_smiles, formula, mono_mass, n_spectra per library, per instrument family and per adduct, NP-tier flags, NP-likeness, mkey); `data/processed/cache/mkey.parquet`; `docs/DATA_AUDIT.md` with: adduct counts per library; instrument family mapping from the free-text `instrument_type`; `precursor_error_ppm` quantiles per library; overlap matrix of structures between libraries; how many enveda-np-examples and enveda-180 structures also occur in other libraries (C1 eligibility); fraction of train structures present in the NP tier and in the PubChem tier (candidate recall ceiling, must be reported per library); element coverage of the filter in 2.2; public test join (5.7) with the tolerance recommendation for 3.3; fraction of structures whose mkey differs from the train ik14.
- Acceptance: row counts per library equal the table in COMPETITION.md within 1 percent after accounting for filters (log every drop reason); peak RAM under 10 GB; `structures.parquet` has about 275k rows; mkey computed for all structures (4-process pool, expected 1 to 2 hours, runs in the background at low priority and the rest of WP6 does not wait for it: write the column in a second pass).
- Estimate: 1.5 hours for ingestion and audit on 3 cores, plus the background mkey job (1 to 2 hours, 2 cores).
- Dependencies: WP0, WP1 (and WP2/WP3 for the DB overlap numbers; if WP3 is unfinished report NP tier only and rerun later).
- Parallel with: nothing heavy. This is the first thing to run when train lands and it blocks everything else.

### WP7 - Splits and validation harness [NEEDS_TRAIN, code skeleton NOW]
- Goal: section 5 as code.
- Inputs: WP6 outputs, WP2/WP3 DBs, WP5 library.
- Outputs: `src/casmi/splits.py`, `src/casmi/validation.py` (builds query sets per (set, condition), applies library and DB masks, calls `pipeline.predict`, scores), `scripts/make_splits.py`, `scripts/evaluate.py` (CLI: `--config`, `--sets`, `--conditions`, `--tag`; appends to `docs/EXPERIMENTS.md`), `splits/holdout_v1.csv` (tracked in git), `data/processed/validation/queries_<set>.parquet` (test-shaped query tables plus truth).
- Acceptance: holdout sizes as in 5.2; no holdout mkey appears in the training set handed to the fingerprint model (asserted in code, not by convention); a "cheating" predictor that returns the truth scores exactly 1.0 and a predictor that returns only CCO scores 0.0; under C2 and C3 the library mask leaves zero spectra of the query molecule (asserted); under C3 the truth is absent from the pool (asserted); query tables have the same schema as `test.parquet`, so `scripts/predict.py` runs on them unchanged; evaluation of Baseline 0 on all 9 cells completes in under 10 minutes.
- Estimate: 5 hours, 2 cores.
- Dependencies: WP1, WP6; WP5 for library masks.
- Parallel with: WP5 build, WP9 code completion.

### WP8 - Baseline 1: library match plus DB prior [NEEDS_TRAIN]
- Goal: second submission and the first honest numbers.
- Method: pool = LIB by mass plus both DB tiers; fusion 4.6 with spectral similarity for LIB candidates and priors for the rest.
- Outputs: config `configs/b1_library.yaml`, `submissions/YYYYMMDD_b1_library_prior.csv`, a row in `docs/EXPERIMENTS.md` for all 9 cells.
- Acceptance: on the public test the local score against train labels is at least 0.95 (plumbing check; investigate every miss, they reveal mass, adduct or mkey bugs); C1 MRR on NP-TIMS reported, expected above 0.7; candidate recall per cell reported.
- Estimate: 2 hours, 3 cores.
- Dependencies: WP5 build, WP7.

### WP9 - Fingerprint model [code NOW, training NEEDS_TRAIN]
- Goal: section 4.4.
- Inputs: `data/processed/train/`, `splits/holdout_v1.csv`.
- Outputs: `src/casmi/fpmodel.py` (featurisation, torch model, training loop, numpy inference, export), `scripts/train_fpmodel.py` (CLI: `--exclude-holdout/--all`, `--hours`, `--tag`), `models/fp_v1_holdout.npz`, `models/fp_v1_full.npz`, `models/fp_v1_calibration.npz` (per-bit calibration fitted on holdout), `tests/test_fpmodel.py`, a tautomer-consistency report in `docs/EXPERIMENTS.md` (see 4.4).
- Acceptance (code, NOW): featurisation runs on the public test spectra; a tiny model overfits 256 synthetic examples; numpy inference equals torch to 1e-4. Acceptance (trained): on holdout C2 queries, mean cosine between predicted and true fingerprint reported per set; ranking NP-tier-plus-PubChem candidates by fingerprint log-likelihood alone gives a top-1 rate clearly above the prior-only baseline of WP8 on the C2 cells (if it does not, stop and debug before building on it); training throughput and the loss curve are logged to `logs/`.
- Estimate: code 5 hours; training 3 to 5 hours wall for the holdout model and the same again for the full refit; 3 cores; RAM under 10 GB (load cleaned spectra as flat arrays, not Python lists).
- Dependencies: WP0, WP1, WP5 (`spectra.py`); training needs WP6 and WP7.
- Parallel with: during training only light development work on the remaining core (WP10/WP12 code against small samples). The full refit runs last, overnight, alongside nothing else heavy.

### WP10 - Candidate features and Baseline 2 [NEEDS_TRAIN]
- Goal: one feature table per (molecule, candidate), used by validation and by inference; hand-fused submission with the fingerprint model.
- Inputs: WP3, WP5, WP9 outputs.
- Outputs: `src/casmi/features.py` (pool construction with caps from 4.5, all features from 4.2-4.4 and 2.4, `FEATURE_NAMES` list as the single source of truth), `src/casmi/priors.py`, `scripts/build_features.py` (writes `data/processed/features/<set>_<condition>.parquet`), `submissions/YYYYMMDD_b2_fp_fusion.csv`, row in `docs/EXPERIMENTS.md`.
- Acceptance: feature building for 400 molecules with full PubChem pools takes under 40 minutes on 4 cores and under 12 GB RAM (log pool sizes: median, 95th percentile, max); the features of a molecule do not depend on which other molecules are in the batch (test: predict 10 molecules alone and inside the full batch, compare); Baseline 2 beats Baseline 1 on the C2 cells and does not lose more than 0.02 on C1 cells.
- Estimate: 6 hours, 4 cores for the full feature build (about 6,250 holdout molecules x up to 3 conditions; cap negatives as in 4.5 and budget 2 to 3 hours of compute; run SYN-TIMS on a 1,000-molecule subset first).
- Dependencies: WP5 build, WP7, WP9 trained (holdout model).

### WP11 - Learned ranker [NEEDS_TRAIN]
- Goal: section 4.5.
- Outputs: `src/casmi/ranker.py`, `scripts/train_ranker.py` (grouped 5-fold CV, class-mix weights, the no-count-priors variant from 5.6), `models/ranker_v1.txt` (LightGBM text format), `models/ranker_v1_features.json`, feature importance and the mix-sensitivity table in `docs/EXPERIMENTS.md`, `submissions/YYYYMMDD_r1_ranker.csv`.
- Acceptance: out-of-fold worst-case blend on NP-TIMS beats Baseline 2 by more than the bootstrap noise or the ranker is not adopted; no cell drops by more than 0.03 relative to the best baseline for that cell; training takes under 15 minutes; prediction is deterministic.
- Estimate: 4 hours, 4 cores briefly.
- Dependencies: WP10.

### WP12 - Formula inference and analog generator [code NOW for formula enumeration, rest NEEDS_TRAIN]
- Goal: sections 3.4 and 4.7.
- Outputs: `src/casmi/formula.py` (enumeration, sub-formula explanation score, logistic combiner), `src/casmi/generate.py` (edit library as reaction SMARTS with formula deltas, seed selection, enumeration with caps), `models/formula_v1.json`, `tests/test_formula.py`, `tests/test_generate.py`, C3 rows in `docs/EXPERIMENTS.md`; then refit of the ranker with GEN candidates (`models/ranker_v2.txt`).
- Acceptance: formula enumeration at 10 ppm contains the true formula for at least 99 percent of holdout molecules; top-1 formula accuracy and top-3 accuracy reported (target top-3 above 90 percent on NP-TIMS); every edit rule has a unit test with one positive example and its formula delta asserted; generation for 400 molecules takes under 20 minutes on 4 cores; under C3, the fraction of holdout molecules whose truth is generated (GEN recall) is reported per set, and the adoption rule is: include GEN in the final pipeline only if C3 MRR rises and the C1 and C2 cells do not drop by more than 0.005.
- Estimate: 8 hours, 2 cores.
- Dependencies: WP10 for seeds and scores; formula enumeration and the edit library can be written NOW.
- Parallel with: WP11.

### WP13 - Optional spike: DreaMS embeddings on CPU [NOW, timeboxed to 2 hours]
- Goal: decide whether pretrained DreaMS embeddings are affordable.
- Inputs: HF `roman-bushuiev/DreaMS`, file `DreaMS_embedding_model_torchscript.pt` (469 MB, verified to exist, with settings: 100 highest peaks, precursor intensity 1.1, embedding dim 1024). It should load with plain `torch.jit.load`, without the `dreams` package (not verified).
- Outputs: a paragraph in `docs/EXPERIMENTS.md`: does it load on CPU torch, input format, spectra per second on 4 cores, measured on the 1,213 public test spectra. No repo code unless adopted.
- Decision rule: adopt only if throughput is at least 150 spectra per second (then embedding the roughly 600k consensus spectra takes about 1 hour). If adopted it becomes a third similarity in 4.2 and 4.3 (cosine of embeddings) and a 469 MB Kaggle artefact. Otherwise drop it.
- Dependencies: none. Lowest priority of the NOW packages; run it only when a core is idle.

### WP14 - Kaggle packaging [NEEDS_TRAIN]
- Goal: a self-contained offline notebook and the artefact bundle.
- Outputs:
  - `scripts/package_kaggle.py`: builds `dist/casmi-artefacts/` and `dist/casmi-code/` and writes `dist/MANIFEST.json` (file, size, sha256).
  - `notebooks/kaggle_inference.ipynb`: installs wheels from the attached dataset with `pip install --no-index --find-links`, adds the code dataset to `sys.path`, reads `/kaggle/input/<competition>/test.parquet`, calls `casmi.pipeline.predict`, writes `/kaggle/working/submission.csv`, then runs `validate_submission`. All paths come from one config cell. No network calls. A try/except around each molecule falls back to Baseline 0 behaviour for that molecule so that one bad spectrum cannot break the run.
  - `docs/KAGGLE.md`: upload instructions for the user and the artefact list.
- Artefacts to upload as Kaggle datasets:
  1. `casmi-db` - `db/np/np.parquet`, `db/np/np_fp.npy`, `db/pubchem/mass_*.parquet`, `_index.json` (about 4 to 6 GB).
  2. `casmi-library` - library peak arrays, metadata, consensus spectra, open-search index (about 3 to 5 GB).
  3. `casmi-models` - `fp_v1_full.npz`, `fp_v1_calibration.npz`, `ranker_v*.txt`, `ranker_v*_features.json`, `formula_v1.json`, `config_final.yaml` (under 200 MB).
  4. `casmi-code` - the `casmi` package source as a zip, plus `MANIFEST.json`.
  5. `casmi-wheels` - `rdkit==2026.3.3` manylinux wheels for CPython 3.11 and 3.12 (fetch with `pip download rdkit==2026.3.3 --only-binary=:all: --python-version 3.1x --platform manylinux_2_28_x86_64 --no-deps`), and lightgbm/numba wheels in case the Kaggle image lacks them. The inference path must not need torch, matchms, duckdb or polars-specific features that Kaggle's image may lack; pandas plus pyarrow plus numpy plus numba plus rdkit plus lightgbm only.
- Acceptance: an offline dry run in a fresh virtualenv with only the wheel set, with network disabled for the process (`unshare -n` or an invalid proxy), reading artefacts only from `dist/`, reproduces the local submission for the public test byte for byte; the same run on a NP-TIMS query table (a "hidden-like" input never seen by the notebook) completes; total runtime under 90 minutes on 4 cores and peak RAM under 16 GB; the notebook contains no absolute paths outside `/kaggle/`.
- Estimate: 4 hours.
- Dependencies: WP11 (and WP12 if adopted). A first version should be built right after WP8 with Baseline 1 so that the user can test the Kaggle side early; later versions only swap artefacts.

### WP15 - Experiments, ablations and final selection [NEEDS_TRAIN, continuous]
- Goal: keep `docs/EXPERIMENTS.md` honest and choose the final submissions.
- Tasks: preprocessing knobs (4.1); fragment tolerance; with and without PubChem tier; with and without count priors; class-mix weights; GEN on or off; pool cap. One change at a time, one row each.
- Final selection: if two final submissions are allowed, choose (A) the best worst-case-blend configuration and (B) a configuration tilted to class 1 and 2 (GEN off, PubChem tier down-weighted) or the opposite tilt, whichever differs most from A in the mix it assumes.
- Estimate: ongoing; each evaluation should take under 15 minutes once features are cached.

### Parallelism summary

| Phase | Run together | Cores |
|---|---|---|
| NOW, step 1 | WP0 alone | 1 |
| NOW, step 2 | WP1, WP2, WP3, WP5 code, WP9 code | WP3 takes 2, the others share 2 (mostly editing and small tests) |
| NOW, step 3 | WP4 (after WP1 and WP2), WP7 skeleton, WP12 formula enumeration and edit rules, WP13 spike when a core is idle | light |
| Train arrives | WP6 alone (background mkey job on 2 cores afterwards) | 3 then 2 |
| After WP6 | WP5 build and WP7 together, then WP8 | 3 + 1 |
| Model phase | WP9 training on 3 cores, with WP10/WP12 development on 1 core using small samples | 4 |
| Feature phase | WP10 full feature build alone, then WP11 and WP12 together | 4, then 2 + 2 |
| End | WP9 full refit overnight alone; WP14 packaging; WP15 | 3 + 1 |

---

## 7. Code layout and conventions

```
enveda/
  pyproject.toml, requirements.txt
  configs/               YAML configs, one per pipeline variant (b0_npdb.yaml, b1_library.yaml, b2_fp.yaml, r1_ranker.yaml, final.yaml)
  splits/                holdout_v1.csv (tracked)
  src/casmi/
    __init__.py
    config.py            dataclass Config + load_config(path); every tunable lives here with its default
    paths.py             ROOT, RAW, EXTERNAL, PROCESSED, MODELS, SUBMISSIONS, LOGS; overridable by env CASMI_ROOT and CASMI_ARTEFACTS (Kaggle)
    log.py               get_logger(name): console + logs/<script>_<timestamp>.log
    adducts.py           WP1
    chem.py              WP1
    metric.py            WP1
    io.py                WP1
    db.py                WP2 (API), WP3 uses it
    priors.py            WP10
    spectra.py           WP5
    similarity.py        WP5
    library.py           WP5
    splits.py            WP7
    validation.py        WP7
    fpmodel.py           WP9
    formula.py           WP12
    generate.py          WP12
    features.py          WP10
    fusion.py            WP4
    ranker.py            WP11
    pipeline.py          WP4 creates it; WP8, WP10, WP11, WP12 add stages
  scripts/
    build_npdb.py        WP2
    build_pubchem.py     WP3
    ingest_train.py      WP6
    build_library.py     WP5
    make_splits.py       WP7
    train_fpmodel.py     WP9
    build_features.py    WP10
    train_ranker.py      WP11
    predict.py           WP4
    evaluate.py          WP7
    package_kaggle.py    WP14
    run_all.sh           rebuilds every artefact from raw inputs, in dependency order (each WP appends its line)
  tests/                 test_<module>.py, owned by the module's WP
  notebooks/kaggle_inference.ipynb
  submissions/           YYYYMMDD_<tag>.csv (tracked)
  docs/                  PLAN.md, COMPETITION.md, EXPERIMENTS.md, DATA_AUDIT.md, ARTEFACTS.md, KAGGLE.md, METRIC_NOTEBOOK.md
```

Rules:
- One owner per file (the WP listed above). A WP that needs a change in another WP's module asks for it or adds a function at the end of the file; it does not refactor it.
- `pipeline.py` is the only shared hot spot. It is a list of named stages (`mass`, `pool`, `features`, `score`, `assemble`), each a function `(state, cfg) -> state`, registered in a dict. A WP adds or replaces a stage by name in its own module and registers it; it does not edit other stages.
- Scripts are thin: argument parsing, logging set-up, a call into `casmi`. No logic that inference needs may live in `scripts/`.
- No hard-coded absolute paths. Everything goes through `casmi.paths`.
- Inference code path (`pipeline.predict` and everything it imports) may depend only on: numpy, pandas, pyarrow, scipy, numba, rdkit, lightgbm. Training and build code may also use polars, duckdb, torch, scikit-learn. Enforce with a test that imports `casmi.pipeline` in a subprocess with torch, polars and duckdb blocked.
- Determinism: fixed seeds from `Config.seed` (default 20261118); sorted iteration over molecules and candidates; stable sorts with explicit tie-breakers; LightGBM deterministic flags; no dependence on dict or set ordering of strings; multiprocessing with `imap` over sorted chunks.
- Data frames: ids and keys are strings; masses are float64; peak arrays float32.
- Submissions: `submissions/YYYYMMDD_<tag>.csv`, tag in lower snake case starting with the stage (`b0_`, `b1_`, `b2_`, `r1_`, `final_`). Every file is produced by `scripts/predict.py` with a config in `configs/`, validated before being written, and logged in `docs/EXPERIMENTS.md` with: date, tag, config, git commit, local validation numbers, and the public leaderboard score once the user reports it.
- Logging: every script logs its arguments, git commit, wall time, peak RSS, and row counts in and out of each stage.
- Tests: `pytest -q` must pass before a WP is declared done. Tests needing large data are marked `@pytest.mark.data` and skipped when the files are absent.
- Artefact persistence: `data/` and `models/` are git-ignored and the container is ephemeral. Three layers of protection: (1) every artefact is reproducible by `scripts/run_all.sh` from raw inputs and public URLs, with expected sizes and checksums in `docs/ARTEFACTS.md`; (2) `scripts/package_kaggle.py` writes the upload bundle to `dist/`, which the user should download or push to Kaggle as soon as a stage is complete, starting with the DB tiers (they do not depend on train); (3) small things that are expensive to recreate and under about 1 MB (splits, configs, feature lists, ranker text models) are committed to git. Rebuild cost if everything is lost: PubChem tier about 4 hours, library 1 hour, fingerprint model 3 to 5 hours, features 3 hours.

---

## 8. Risks and open questions

| # | Risk or question | Impact | Fallback or action |
|---|---|---|---|
| 1 | The exact metric code was not readable from this machine (kaggle.com is blocked). The tautomer call may have extra steps or non-default limits. | A mismatch silently mis-scores local validation and mis-deduplicates output. | User pastes the metric notebook into `docs/METRIC_NOTEBOOK.md`. WP1 keeps `mkey` behind one function so it is a one-line change. |
| 2 | Kaggle runtime conditions are unknown: time limit, internet off, RAM, Python version, whether rdkit is preinstalled, dataset size limits, number of final submissions, deadline. | Packaging choices. | Design assumes the strictest plausible case: offline, 4 vCPU, 16 GB peak RAM, 90 minutes, wheels for two Python versions. User confirms from the competition rules page. |
| 3 | Class mix is hidden and may be dominated by class 3. | If most molecules are class 3, expected score is low for everyone and our effort on class 2 matters less. | Worst-case-blend selection (5.5). GEN is built but gated. |
| 4 | Popularity priors overfit to simulated class 2 (5.6). | Over-ranking famous isomers on the real test. | No-count-priors ranker variant; evaluation on the pubmed_count = 0 subset; monotone constraints. |
| 5 | Chemistry shift: timsTOF training data is mostly synthetic drug-like compounds, the test is natural-product-like. Only 250 molecules are both timsTOF and natural products. | Fingerprint model may be weaker on real test chemistry than SYN-TIMS suggests. | Train on all libraries with an instrument flag, noise augmentation for non-timsTOF spectra, select on NP-TIMS and NP-OTHER, not SYN-TIMS. |
| 6 | Tautomer and standardisation mismatch between train labels and DB SMILES (4.4). | Depressed fingerprint scores for the true candidate. | Measure in WP9; InChI round-trip in `std_mol` if needed. |
| 7 | PubChem pools are very large for common formulas (tens of thousands) and dilute ranking. | Class 2 MRR falls when the PubChem tier is added. | Tier flags and priors in the ranker; pool cap by formula probability; validate "with versus without PubChem tier" and keep the tier only if the worst-case blend improves. Note that COCONUT is itself sufficient for the "confirmed natural product" part of class 2. |
| 8 | Fingerprint model trains slower than estimated on 4 CPUs. | Fewer epochs. | Throughput test first; reduce width to 512, bins to 0.2 Da, or subsample enveda-180 to at most 4 spectra per structure and adduct; the analog-evidence features (4.3) work with no training at all. |
| 9 | COCONUT and LOTUS column names were not verified (archives not opened). | WP2 may need small adjustments. | WP2 prints headers first and fails loudly; the lite CSV and the SDF are alternatives. |
| 10 | Train label noise (`precursor_error_ppm` is uncleaned). | Wrong references in the library, wrong targets. | Filter |error| > 20 ppm out of model training; keep in the library but expose the error as a feature. |
| 11 | Adduct given in the hidden test could be wrong for some spectra. | Wrong M, empty pools. | Cluster logic in 3.2; if the pool is empty at 10 ppm, retry at 20 ppm, then retry treating each spectrum's adduct as the default for its polarity. |
| 12 | Container loss before artefacts are saved. | Hours of rebuild. | Section 7, artefact persistence. Open question for the user: is there a persistent store we may push to (a private Hugging Face dataset repo or a Kaggle dataset the user uploads from `dist/`)? |
| 13 | mkey computation can be slow or hang on unusual large molecules. | Slow inference or a stuck run. | Per-call timeout in a worker process, fallback to ik14, disk cache; only computed for the top of each list. |
| 14 | Memory when loading library plus DB shards plus features on Kaggle. | Out-of-memory kill. | Process molecules in batches of 50; load PubChem shards on demand and release; memory-map the library arrays. |
| 15 | Extra public spectra outside train (FragHub March 2026 exports on Zenodo record 18888110, about 1 GB; MSnLib in MassSpecGym format, record 23494858, 135 MB, probably already in train as `pluskal_ms2`). | Possible conversion of some class 2 molecules into library matches. Organisers define class 1 by "public reference spectra", which may include sources not in train. | Low-cost check after WP6: count FragHub ik14 not present in train. Add as an extra library source only if the count is in the thousands. |
| 16 | Pretrained models that might help but are unverified: DreaMS TorchScript (WP13); MIST and ICEBERG checkpoints on Zenodo (records 23303289 and 24285358, small files, but the code has heavy dependencies); Hugging Face models `abrilrisso/massspecgym-denovo-decoder` and `nielsr/MS-GPT-MIST-MassSpecGym` (unknown quality, zero or few downloads). | Possible gains for classes 2 and 3. | Not on the critical path. Revisit only after WP11 if time remains. |
| 17 | The public test file covers M 246 to 439 only; the hidden test goes to 1,159 Da. | Large glycosides and peptides behave differently: more candidates per mass window in absolute Da, slower mkey, sparser training data. | Holdout sampling follows the hidden mass distribution (5.2); report MRR by mass tercile. |

Open questions for the user:
1. The metric notebook source (risk 1) and the rules page facts (risk 2).
2. Deadline and number of daily and final submissions, which set how far down the work-package list we go.
3. A persistent store for artefacts (risk 12).
4. Whether a GPU may be assumed at inference. The plan does not need one.

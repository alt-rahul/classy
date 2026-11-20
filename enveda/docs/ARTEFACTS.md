# Artefacts

Every artefact under `data/` and `models/` (git-ignored, ephemeral container) with what is needed
to check and rebuild it (PLAN.md section 7, "Artefact persistence"). `scripts/run_all.sh` rebuilds
everything in dependency order; `scripts/package_kaggle.py` bundles them into `dist/`.

Columns:
- artefact: short name (e.g. `np.parquet`, `pubchem shard mass_0350`).
- path: relative to the repo root (`enveda/`).
- rows: row count (or array shape).
- size: on disk, MB or GB.
- sha256: of the file (for sharded outputs, of `_index.json`, which lists per-shard hashes).
- how to rebuild: the exact command, its inputs with source URL and date, wall time and cores.

| artefact | path | rows | size | sha256 | how to rebuild |
|---|---|---|---|---|---|
| PubChem tier (120 shards) | `data/processed/db/pubchem/mass_XXXX.parquet`, `_index.json` | 90,719,171 | 2.19 GB | `f08bdde0be1c814797f3f16cbf6c4e764aba72ca84cafff9f5cbaf5ed9c96346` (`_index.json`) | `nice -n 10 python3 scripts/build_pubchem.py`; PubChem FTP Compound/Extras (CID-Mass.gz 16 Nov 2026, CID-SMILES.gz 17 Nov 2026, CID-PMID.gz 29 Aug 2025, CID-InChI-Key.gz 17 Nov 2026); about 14 min wall, 2 cores, 7 GB RAM |
| NP tier table | `data/processed/db/np/np.parquet` | 854,475 | 32.5 MB | `ba9fe19ddda93a5bc473d387f279bd1e1829fcbd430baa5859884157a477513f` | `nice -n 10 python -I scripts/build_npdb.py` (stages download, headers, table, fp, check); inputs: the 4 raw NP files below, downloaded 2026-11-18; about 10 min wall after downloads, 2 cores, 2.4 GB RAM peak |
| NP tier fingerprints | `data/processed/db/np/np_fp.npy` | (854,475, 512) uint8 | 437.5 MB | `a526ef2081ad323e26960db128e5902f21785bf6eeb882cd5197e3ad5920f9a9` | same command (`nice -n 10 python -I scripts/build_npdb.py --stage fp` alone redoes it, 11.3 min on 2 cores since the InChI round trip, rebuilt 2026-11-18); row-aligned to np.parquet, recipe in the WP2 section |
| NP raw: COCONUT | `data/external/coconut/coconut_csv-11-2026.zip` | 738,793 | 248.0 MB | `74582c17f34ce33ea8ff27ba721a7bcfb654e2521aa94d6d3caf53f19159187a` | `https://coconut.s3.uni-jena.de/prod/downloads/2026-11/coconut_csv-11-2026.zip` (2026-11 release), `build_npdb.py --stage download` |
| NP raw: NPAtlas | `data/external/npatlas/NPAtlas_download.tsv` | 36,454 | 33.7 MB | `4fa5ce93cf5e271d65649f7f5c40e031b1e3d59144cb50c12c5fc9ee041cc72b` | `https://www.npatlas.org/static/downloads/NPAtlas_download.tsv` (no version in the URL: a later download may differ) |
| NP raw: LOTUS | `data/external/lotus/structure_metadata.tsv.gz` | 555,952 | 95.8 MB | `ceab19657fa362bf72937c92b6a709e13a2377782a2ac705afa1c0f7c037b51a` | Zenodo record 7534077 (2022-09-16), md5 `a791309a82df1d7ac538624f8f7856dc` |
| NP raw: PubChemLite | `data/external/pubchemlite/PubChemLite_CCSbase_20260925.csv` | 469,602 | 225.3 MB | `f1efd923b72a87ea7aa420ae6d71ea26a5b8cc0a0eea0f10d38a6c09ec02d88e` | Zenodo record 22953182 (2026-09-25), md5 `fd1629808c2cf2896acc502e26e352f0` |

## Source notes

Per-source details (actual column names found, download dates, filter counts) go below,
one section per WP.

<!-- WP3:pubchem:start -->
## PubChem tier (WP3)

- Built 2026-11-18T02:18:04+00:00 by `scripts/build_pubchem.py` (git 763e494). Rebuild: `nice -n 10 python3 scripts/build_pubchem.py`; measured 14 min wall (stage 1 8.4 min, stage 2 2.9 min, stage 3 3.1 min; downloads ran at about 20 MB/s), 2 cores, peak RSS 4.2 GB (DuckDB memory_limit 7GB), peak disk about 8 GB of intermediates (stage 1 parquet is deleted after stage 2, stage 2 after the checks). Resumable via `data/external/pubchem/.stage_*`; `--stage 1|2|3|check|cleanup|docs` runs one stage, `--force` rebuilds from scratch.
- Location: `data/processed/db/pubchem/`, 120 shards `mass_0100.parquet` ... `mass_1290.parquet` (shard XXXX holds 10 * floor(mono_mass / 10) = XXXX), plus `_index.json` (per shard: rows, min and max mass, bytes, sha256).
- Rows: 90,719,171 (one per ik14 within a shard; ik14 is unique across the tier because one ik14 implies one formula). Size: 2.19 GB. sha256 of `_index.json`: `f08bdde0be1c814797f3f16cbf6c4e764aba72ca84cafff9f5cbaf5ed9c96346`.
- Schema (PLAN.md 2.3): `ik14` string, `smiles` string (PubChem isomeric SMILES of the lowest CID), `formula` string (PubChem Hill formula), `mono_mass` float64 (PubChem monoisotopic mass), `cid` int32 (lowest CID in the ik14 group), `pmid_count` int32 (max over the group of distinct PMIDs per CID, 0 if none), `n_cids` int16 (CIDs collapsed into the ik14, capped at 32767). Rows sorted by mono_mass then cid, row groups of 100,000 rows, zstd level 9, so a pyarrow `filters=[('mono_mass', '>=', lo), ...]` read touches one or two row groups.
- Not done here: RDKit parsing (lazy at inference, PLAN.md 2.2), fingerprints, NP-likeness.

Sources (PubChem FTP `Compound/Extras`, streamed, never stored; md5 checked against the published `.md5`):

| File | Last-Modified | Bytes | Lines parsed | Columns found (tab separated, no header) |
|---|---|---|---|---|
| CID-Mass.gz | Mon, 16 Nov 2026 23:58:38 GMT | 1,393,553,069 | 124,863,650 | CID, formula (charged species carry a suffix: `C9H18NO4+`, `C6H5O7-3`), monoisotopic mass, exact mass |
| CID-SMILES.gz | Tue, 17 Nov 2026 00:25:13 GMT | 1,488,799,841 | 124,863,650 | CID, isomeric SMILES |
| CID-PMID.gz | Fri, 29 Aug 2025 09:44:14 GMT | 326,400,588 | 51,878,601 | CID, PMID, type (1 depositor, 2 MeSH, 3 publisher, 4 BioAssay); not sorted by CID; pmid_count = distinct PMIDs per CID over all types |
| CID-InChI-Key.gz | Tue, 17 Nov 2026 00:42:25 GMT | 7,379,931,848 | 124,863,647 | CID, InChI, InChIKey (27 chars); only CID and key[:14] kept |

Filter cascade (rows remaining after each filter, applied in this order, then ik14 dedup):

| Step | Rows | Removed |
|---|---|---|
| cids_in_CID-Mass | 124,863,650 |  |
| has_smiles | 124,863,650 | 0 |
| has_inchikey | 124,863,647 | 3 |
| neutral_formula_no_charge_suffix | 121,537,007 | 3,326,640 |
| elements_CHNOPSFClBrI | 117,055,451 | 4,481,556 |
| at_least_3_C | 116,975,159 | 80,292 |
| mass_100_1300 | 114,934,024 | 2,041,135 |
| single_component_no_dot | 107,307,839 | 7,626,185 |
| no_isotope_spec | 106,915,623 | 392,216 |
| dedup on ik14 (lowest CID kept) | 90,719,171 | 16,196,452 |

Self-check: caffeine found=True (2,495 candidates at 10 ppm), paclitaxel found=True (849 candidates); 20 public-test masses at 10 ppm: candidates min 1,338, median 10,072, max 19,527, slowest query 0.032 s (pyarrow, one shard file read with row-group pruning).
<!-- WP3:pubchem:end -->

<!-- WP1:mkey-cache:start -->
## mkey cache (WP1)

`data/processed/cache/mkey.parquet` (`casmi.metric.default_cache_path()`): not a fixed artefact,
created and extended on demand by `casmi.metric.mkeys(..., cache=True)`; delete it to start over.
Columns: `smiles` (key, unique, file sorted by it), `mkey` (null if none), `status`
(`ok` / `fail` = does not parse / `error` = canonicalisation or InChI failed / `timeout` = plain
InChIKey first block used), `timeout_s` (limit in force when computed, `inf` = none), `rdkit`
(version; rows from another RDKit version are ignored). Written under an fcntl lock via temp file
and rename, so several processes may share it. Timeout rows are recomputed when a later call
allows a longer timeout. On Kaggle (read-only artefacts) it is read if shipped and never written.
<!-- WP1:mkey-cache:end -->

<!-- WP2:np:start -->
## NP tier (WP2)

Built 2026-11-18 by `scripts/build_npdb.py`; every count below is also in
`data/processed/db/np/build_report.json` (with the full filter cascade and checksums).
`mass_disagreements.csv` (same directory) lists filter-passing source rows whose RDKit mass
differs from the source mass by more than 0.001 Da (3 rows, all NPAtlas, all with 4 extra H in
the NPAtlas SMILES; none of them supplies the SMILES of an np.parquet row).

### Columns found in each source (header printed and checked by `--stage headers`)

- COCONUT (`coconut_csv-11-2026.csv` inside the zip, comma separated, quoted, 44 columns):
  `identifier, canonical_smiles, standard_inchi, standard_inchi_key, name, iupac_name,
  annotation_level, total_atom_count, heavy_atom_count, molecular_weight, exact_molecular_weight,
  molecular_formula, alogp, topological_polar_surface_area, rotatable_bond_count,
  hydrogen_bond_acceptors, hydrogen_bond_donors, hydrogen_bond_acceptors_lipinski,
  hydrogen_bond_donors_lipinski, lipinski_rule_of_five_violations, aromatic_rings_count,
  qed_drug_likeliness, formal_charge, fractioncsp3, number_of_minimal_rings, van_der_walls_volume,
  contains_sugar, contains_ring_sugars, contains_linear_sugars, murcko_framework, np_likeness,
  chemical_class, chemical_sub_class, chemical_super_class, direct_parent_classification,
  np_classifier_pathway, np_classifier_superclass, np_classifier_class,
  np_classifier_is_glycoside, organisms, collections, dois, synonyms, cas`.
  There are no organism or citation count columns: `organisms`, `dois` and `collections` are
  `|`-separated lists; counts are distinct items, unioned over all rows with the same ik14.
  `exact_molecular_weight` is the monoisotopic mass (5 decimals).
- NPAtlas (TSV, read with quoting off, 34 columns): `npaid, compound_id, compound_name,
  compound_molecular_formula, compound_molecular_weight, compound_accurate_mass,
  compound_m_plus_h, compound_m_plus_na, compound_inchi, compound_inchikey, compound_smiles,
  compound_cluster_id, compound_node_id, origin_type, genus, origin_species,
  original_reference_author_list, original_reference_year, original_reference_issue,
  original_reference_volume, original_reference_pages, original_reference_doi,
  original_reference_pmid, original_reference_title, original_reference_type,
  original_journal_title, synonyms_dois, reassignment_dois, synthesis_dois, mibig_ids, gnps_ids,
  cmmc_ids, npmrd_id, npatlas_url`.
- LOTUS `structure_metadata.tsv.gz` (TSV, 15 columns, one row per stereo-resolved structure, no
  organism links in this file): `structureCleanedSmiles, structureCleanedInchi,
  structureCleanedInchikey, structureCleaned_smiles2D, structureCleaned_inchi2D,
  structureCleaned_inchikey2D, structureCleaned_validatorLog, structureCleaned_molecularFormula,
  structureCleaned_exactMass, structureCleaned_xlogp, structureCleaned_stereocenters_unspecified,
  structureCleaned_stereocenters_total, structureCleaned_cid, structureCleaned_nameIupac,
  structureCleaned_nameTraditional`. So `n_organisms` / `n_citations` come from COCONUT only.
- PubChemLite (CSV, 36 columns): `Identifier, FirstBlock, PubMed_Count, Patent_Count,
  Related_CIDs, Synonym, MolecularFormula, SMILES, InChI, InChIKey, MonoisotopicMass, XLogP,
  CompoundName, AnnoTypeCount, AgroChemInfo, BioPathway, DrugMedicInfo, FoodRelated,
  PharmacoInfo, SafetyInfo, ToxicityInfo, KnownUse, DisorderDisease, Identification, NORMANSLE`
  and 11 `pred_CCS_A2_[adduct]` columns. Already unique on `FirstBlock` (= InChIKey[:14] on
  every row); SMILES carry no stereo.

### Filters and counts

Order: non-empty SMILES, no `.`, no isotope label in the SMILES, RDKit parses, no isotope atom,
net formal charge 0, elements in {C,H,N,O,P,S,F,Cl,Br,I}, at least 3 C, RDKit mono mass in
[100, 1300], then dedup on the source ik14 within the source (COCONUT: highest
annotation_level, then lowest identifier; LOTUS and PubChemLite: lowest CID; NPAtlas: lowest
npaid).

| Source | read | after filters | unique ik14 after filters | distinct ik14 (all rows) | SMILES used in np.parquet | in_<source> true |
|---|---|---|---|---|---|---|
| COCONUT | 738,793 | 710,141 | 460,532 | 479,689 | 460,532 | 460,605 |
| NPAtlas | 36,454 | 35,327 | 32,399 | 33,497 | 124 | 32,424 |
| LOTUS | 555,952 | 538,346 | 284,307 | 295,474 | 46,400 | 284,374 |
| PubChemLite | 469,602 | 419,879 | 419,879 | 469,602 | 347,419 | 421,472 |
| union (np.parquet) | | | 854,475 | | | |

Rows only in one source: COCONUT 201,577, NPAtlas 75, LOTUS 45,047, PubChemLite 347,410.
Membership flags and priors are taken over all rows of a source with that ik14 (before the
filters), so a structure whose COCONUT row is, for example, drawn charged still gets
`in_coconut` when its neutral SMILES came from another source (73 such rows).

### np.parquet

Sorted by (mono_mass, ik14), zstd, row groups of 100k. Columns: `ik14` (source InChIKey
first block; RDKit's InChIKey[:14] agrees on 5,000 of 5,000 sampled rows), `smiles` (as
shipped by the first source in the order COCONUT, NPAtlas, LOTUS, PubChemLite that passes the
filters; stereo kept), `formula` (RDKit `CalcMolFormula`, Hill), `mono_mass` (RDKit
`CalcExactMolWt`, both on the molecule as parsed, i.e. `casmi.chem.formula` / `mono_mass`),
`in_coconut, in_npatlas, in_lotus, in_pubchemlite` (bool), `coconut_annotation_level` (int8),
`coconut_np_likeness` (float32), `n_organisms, n_citations, n_collections` (int32, COCONUT,
null when not in COCONUT), `pubmed_count, patent_count, anno_type_count` (int32, PubChemLite,
null when not in it), `cid` (int64, PubChemLite CID, else LOTUS `structureCleaned_cid`),
`smiles_source` (which source supplied the SMILES), `coconut_id` / `npaid` (id of the row that
supplied the SMILES, null otherwise). `n_collections` (number of COCONUT source collections),
`smiles_source`, `coconut_id`, `npaid` are additions to the PLAN 2.3 schema.
RDKit mass vs the source mass of the supplying row: 854,475 of 854,475 within 0.001 Da.

### np_fp.npy recipe (for `casmi.chem.morgan_fp` users)

`np.load(..., mmap_mode="r")`, shape (854,475, 512), uint8, row i = np.parquet row i. Row i is
`casmi.chem.morgan_fp(casmi.chem.fp_mol(Chem.MolFromSmiles(smiles_i)))`, which equals
`casmi.chem.morgan_fps([smiles_i], standardize=True)[0][0]`: WP1 `std_mol` (RemoveHs, metal
disconnection, normalise, reionise, uncharge, remove stereo), then the InChI round trip
`Chem.MolFromInchi(Chem.MolToInchi(mol))` without stereo (`Config.fp_inchi_roundtrip`, default
True; the `std_mol` molecule is kept when the round trip fails or changes the heavy-atom formula,
which happened for none of the 20,000 SMILES of the tautomer proxy in docs/EXPERIMENTS.md), then
`rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096, includeChirality=False)`
(other options at RDKit defaults), `GetFingerprintAsNumPy`, `np.packbits` with bit order "big"
(bit j is `(row[j // 8] >> (7 - j % 8)) & 1`; unpack with
`np.unpackbits(row, count=4096)` or `casmi.db.unpack_fp`). `std_mol` failed on 0 rows (the
build would fingerprint the parsed molecule instead and list the rows in build_report.json).
Spot check in `--stage check`: 1,000 sampled rows recomputed with `chem.morgan_fps`, 0
mismatches (rerun after the InChI round-trip rebuild: all checks passed). If WP1 changes
`std_mol`, `fp_mol` or `morgan_fp`, or a config flips `fp_inchi_roundtrip`, rerun
`build_npdb.py --stage fp`. Before the round trip (std_mol only) the sha256 was
`c14c99ba09e191255ea43ea404f4474d978963cea2c0d532d0d56d7fa4d603a9`.

### Lookup (casmi.db) on this build

Load 0.7 s; the NP frame holds 200 MB (peak RSS during load about 630 MB above baseline). Single NP-tier lookup at 10 ppm: mean 1.6 ms,
max 4.3 ms over 1,000 queries; 400 public-test masses in one `lookup_batch`: 14 ms.
Public-test NP pool sizes (M = median over a molecule's spectra of precursor_mz minus the PLAN
3.1 shift, 400 molecules):

| window | median | p95 | max | fraction empty |
|---|---|---|---|---|
| 5 ppm | 27 | 89 | 651 | 0.005 |
| 10 ppm | 44.5 | 138 | 702 | 0.0025 |
| 20 ppm | 87.5 | 242 | 712 | 0.0025 |

With the PubChem tier as well (10 ppm, 400 molecules, one `lookup_batch`): pool median 10,467,
p95 21,182, max 33,523; 0.5 percent of rows are NP tier; 95.6 percent of NP rows in those windows
are also in the PubChem tier; 34.6 s and 5.2 GB peak RSS for the single 4.6M-row result, so call
`lookup_batch` on batches of about 50 molecules (`Config.batch_molecules`). Single both-tier
lookups (caffeine, paclitaxel, quercetin): 40-50 ms with a cold cache.
<!-- WP2:np:end -->

<!-- WP5:library:start -->
## Spectral library (WP5)

Not built yet: `data/raw/train.parquet` has not arrived. Add the table row (size, sha256 of
`library_info.json`) when `python scripts/build_library.py` has run on train.

Command: `nice -n 10 python scripts/build_library.py` (default source `data/raw/train.parquet`;
`--source data/processed/train/spectra/ --precleaned` reads WP6's already-cleaned files instead,
as `scripts/ingest_train.py --stage library` does: peaks are stored as given, not cleaned again,
and their `n_peaks_raw` / `prec_intensity` (computed from the raw spectra) are kept; `--fake-test`
builds a smoke-test library from `test.parquet` with one fake structure per molecule_id).
`library_info.json` records `precleaned`. Input `n_peaks_raw` / `prec_intensity` columns win
over recomputed values in either mode, because a second cleaning would give the cleaned count
and 0, and `fpmodel` uses both as inputs (test time computes them from raw test spectra).
Cleaning cleaned peaks again is not exactly idempotent: on the synthetic train 1 of 4,134
spectra loses a peak whose relative intensity, stored as float32(0.005) = 0.0049999999, falls
below the 0.5 percent floor.
Output `data/processed/library/` (written to `library.building/`, renamed at the end):

| file | format |
|---|---|
| `peaks_mz.npy`, `peaks_int.npy` | float32, all cleaned peaks (PLAN.md 4.1), base peak 1, memory-mappable |
| `offsets.npy` | int64, n_spectra + 1; spectrum i = `offsets[i]:offsets[i+1]` |
| `meta.parquet` | one row per spectrum in store order: spec_idx, spectrum_id, struct_idx, cons_idx, ik14, precursor_mz, adduct, adduct_family, polarity (+1/-1), instrument_family, instrument_type, ingest_lib, collision_energy (mean eV), n_energies, base_peak_intensity, precursor_error_ppm, prec_intensity, n_peaks_raw, n_peaks, molecule_id, source_row |
| `structures.parquet` | one row per ik14 sorted by (mono_mass, ik14): struct_idx, ik14, smiles, formula, mono_mass, spec_start/spec_end, cons_start/cons_end, n_spectra, n_consensus |
| `consensus_mz.npy`, `consensus_int.npy`, `consensus_offsets.npy` | consensus spectra, same layout |
| `consensus_meta.parquet` | cons_idx, struct_idx, ik14, polarity, adduct_family, instrument_family, ingest_lib, precursor_mz (reference), mono_mass, spec_start/spec_end (member rows), n_members, n_peaks, open_row |
| `open_index_pos.npz`, `open_index_neg.npz` | `scipy.sparse.save_npz` (uncompressed) CSR float32, rows = consensus of that polarity in order, 80,000 columns ([fragment | neutral loss] at 0.05 Da to 2,000 Da) |
| `library_info.json` | counts, drop reasons, cleaning settings, build time |

Scale test on synthetic train-schema data (public test spectra replicated with per-structure
m/z shifts, 150 raw peaks per spectrum, 6 libraries, 1 core under nice):

| input spectra | structures | consensus | wall | peak RSS | on disk |
|---|---|---|---|---|---|
| 500,000 | 83,115 | 368,685 | 13 s | 1.58 GB | 465 MB |
| 1,000,000 | 166,271 | 737,414 | 25 s | 2.01 GB | 926 MB |

Linear extrapolation to 2.5M train spectra: 1 to 2 minutes plus parquet decoding of the 3 GB
file, about 3.5 GB peak RSS, about 2.3 GB on disk (more if train keeps more peaks per spectrum
after cleaning: the cap is 100). Loaded 1M library: 1.1 GB RSS; 400 public-test molecules,
mass search 0.4 s (0.9 s with an exclusion mask); open search 38 ms per consensus query.
<!-- WP5:library:end -->

<!-- WP4:submissions:start -->
## Baseline 0 submissions and inference runs (WP4)

Submissions are tracked in git (`submissions/`), produced by `scripts/predict.py` from
`data/raw/test.parquet` (public test, 1,213 spectra, 400 molecules) on 2026-11-18, 2 mkey worker
processes, under `nice -n 10` with other agents busy (load average about 4):

| file | rows x guesses | bytes | sha256 | command | wall | peak RSS |
|---|---|---|---|---|---|---|
| `submissions/20261118_b0_npdb_prior.csv` | 400 x 25 | 409,015 | `5355da5cfdbeded78ad5e55892a0d03a332791e263bd3fe726510269ef3fcb8a` | `python scripts/predict.py --config configs/b0_npdb.yaml` | 46 s with an empty mkey cache, 4.6 s warm | 0.77 GB |
| `submissions/20261118_b0b_pubchem_prior.csv` | 400 x 25 | 406,945 | `b29a3a540ecb3b6d1649e51cf0e473f0cf31d13c4059c49dc27ae5c756ad515f` | `python scripts/predict.py --config configs/b0b_pubchem.yaml` | 78 s with `mkey_cache: false`, 50 s warm | 1.66 GB |

Each run writes `logs/predict_<tag>_<timestamp>.log` and the per-molecule report
`logs/<file stem>_molecules.csv` (M, M_pool, pool_attempt, pool sizes, guesses by origin, skips,
errors). Runs extend the mkey cache `data/processed/cache/mkey.parquet` (WP1 section); about
12.5k entries after both runs. Cold, warm and cache-off runs give byte-identical files.
<!-- WP4:submissions:end -->

<!-- WP9:fpmodel:start -->
## Fingerprint model (WP9)

Code phase only (2026-11-18): nothing below is built yet, because train has not arrived. Expected
outputs of `scripts/train_fpmodel.py` (sizes measured on synthetic runs of the full-size model):

| artefact | path | rows | size | sha256 | how to rebuild |
|---|---|---|---|---|---|
| holdout model | `models/fp_v1_holdout.npz` | 27.0M params (20000 x 1024 embedding, 2 x 1024 blocks, 4263 outputs) | 108 MB float32 (54 MB with `--float16`) | not built | `nice -n 10 python scripts/train_fpmodel.py --exclude-holdout --hours 4`; needs WP5 library + `splits/holdout_v1.csv`; 3 cores, under 10 GB RAM |
| full refit | `models/fp_v1_full.npz` | as above | 108 MB | not built | `... --all --hours 4` |
| per-bit calibration | `models/fp_v1_calibration.npz` | a, b (4263 each) + info_json | 70 KB | not built | written by the `--exclude-holdout` run (holdout molecules, test adducts, at most 16 spectra each) |
| checkpoints | `models/fp_v1_*_ckpt.pt` | model + AdamW state | 324 MB each | not needed for inference | written every 20 min; `--resume` continues |
| target cache | `data/processed/fp/targets_cache.npz` | one row per (ik14, SMILES hash) | about 170 MB for 275k structures | not built | built on first use (`chem.fp_mol` = std_mol + InChI round trip, Morgan 4096 + MACCS, about 1.5 ms per structure); records `n_morgan`, `radius`, `inchi_roundtrip` and is ignored and rewritten when they differ from the config (caches without `inchi_roundtrip` count as no round trip) |
| loss curves | `logs/fp_<tag>_curve.csv` | one row per 100 steps | small | - | written by every run |

The `.npz` holds float arrays plus `spec_json` (bins, widths, metadata field names, cleaning
settings, training info); load with `casmi.fpmodel.FpModel.load` (numpy only, `allow_pickle=False`).
Only the `.npz` files and the calibration go into the Kaggle bundle.
<!-- WP9:fpmodel:end -->

<!-- WP6:train:start -->
## Train ingestion (WP6) and validation tables (WP7)

Built on the real `data/raw/train.parquet` on 2026-11-20 (3,033,286,496 bytes, 2,539,608 rows,
21 row groups, 18 columns), RUNBOOK steps 1 to 6. The real file has no `spectrum_id` and no
`molecule_id` column (the synthetic file had both): ingestion assigns
`spectrum_id = "t_<row index in train.parquet>"` (= `source_row`, stable in file order), writes
`molecule_id` as null, skips the duplicate-id check, and the public test join matches by
precursor m/z plus peak arrays (all 1,213 public spectra matched, all verbatim enveda-180
rows). Validation query tables reuse these ids. A second full ingest of the same file gave
byte-identical spectra, meta, structures, lib-count and public-test files; `make_splits` run
twice gave a byte-identical holdout. Formats were rehearsed on the synthetic train file
(`scripts/make_synthetic_train.py`, 4,145 rows, 9 row groups, 543 structures, 11 libraries, 21
planted bad rows).

| artefact | rows | bytes | sha256 |
|---|---|---|---|
| `data/processed/train/spectra/` (11 files) | 2,505,117 spectra, 66.2M cleaned peaks | 600,891,035 | per file: see `sha256sum` (deterministic) |
| `data/processed/train/meta.parquet` | 2,505,117 | 111,244,880 | `315e26eb87421862d2a2487ceea3cd04f74a51094d2d6d4599302d302c84755b` |
| `data/processed/train/structures.parquet` (with mkey) | 275,775 | 24,092,836 | `2bbdad34721d23b776e4fc3d7b06eb51df69e747b0dcfaf0430731ecb060ec50` |
| `data/processed/train/structure_lib_counts.parquet` | (ik14, lib) pairs | 5,858,383 | `723083e787ba631bd6f2308b496daaf889a21c55c70367656c706eb3c93d9207` |
| `data/processed/train/public_test_truth.parquet` | 400 | 24,684 | `851b38021c9d14f2ba79a6921e5c176c02e8202b43e4dd44cb544571e2ea35cc` |
| `data/processed/train/ingest_info.json` | - | 4,376 | `4e7d16a21832cb53bb1b1f9dc76ca23d9dcf0f8aa0bc075b1a7b6104b2c10069` |
| `data/processed/library/` (WP5 index, precleaned) | 2,505,117 spectra, 275,775 structures, 625,990 consensus | 1,189,617,341 | `library_info.json` `c070c1cc3551898b99faa26ceeecb1312874024e777be17283e79fbfc0939e7e` |
| `data/processed/cache/mkey.parquet` | about 290k SMILES (all train structures + holdout DB rows + b0 guesses) | grows | not fixed |
| `splits/holdout_v1.csv` (tracked) | 6,454 ik14 rows, 6,250 molecules | 531,478 | `fa5f843a8858f1fed999845972bacf266b24aacb97b73df33de538bb18cd065c` |
| `data/processed/validation/queries_info.json` | NP-TIMS 250 / 599 spectra, SYN-TIMS 3,000 / 8,289, NP-OTHER 3,000 / 8,167 | 1,099 | `ad4886fd6ec1d4b83d86c1c9444f542d62f15f06ef41b38fc27976e247e11b3f` |
| `data/processed/validation/truth_keys.parquet` | 6,250 mkeys (np,pubchem) | 221,486 | `7842a632f8116d1a681cb20201dd5bedf9fc73799cb1fba88996a438ee15f234` |

Real-run timings (4 cores, nice 10; a second agent active from about 19:05):

| step | command | wall | peak RSS | notes |
|---|---|---|---|---|
| 1 ingest + audit | `ingest_train.py --stage all --n-jobs 3` | 271 s | 3.18 GB | pass 1 79 s (32,180 rows/s, 1.69 GB), RDKit 275,775 structures in 99 s (2,772/s on 3), PubChem membership 64 s |
| 2 mkey | `--stage mkey --mkey-jobs 4` | 1,083 s | 0.68 GB (+4 x 0.25 GB workers) | 275,775 ok, 0 timeouts; mkey != ik14 for 4,362 |
| 3 library | `--stage library` | 52 s | 3.41 GB | 1.19 GB on disk; loaded 1.8 GB RSS (2.7 GB after 400-molecule searches) |
| 4 splits | `make_splits.py --n-jobs 3` | 985 s | 3.62 GB | holdout 1 s, queries 31 s, truth keys 16 min (NP-TIMS 44 s, SYN-TIMS 9 min, NP-OTHER 6 min) |
| 5 audit | `--stage audit` | 11 s | 2.2 GB | |
| 6 evaluate b0 | `evaluate.py --config configs/b0_npdb.yaml --n-jobs 3` | 1,269 s (self check 30 s, cells 1,237 s) | 3.0 GB | cold mkey cache for the guesses: SYN-TIMS C2 376 s, NP-OTHER C1 562 s |

WP5 build acceptance on this library (1 core): self-search of 1,000 random spectra (seed
`Config.seed`): cosine 1.0 at rank 1 for 1,000, own spectrum 967 plus 18 identical-peak
duplicates plus 14 same-structure ties, 1 single-peak isomer tie (gnps vs massbank, m/z 107.05);
4.3 ms per search. Mass-restricted search for the 400 public test molecules 5.0 s (5.4 s with a
1,000-structure exclusion mask); open search 497 consensus queries in 19.2 s (39 ms each).

`data/processed/train/` (`nice -n 10 python scripts/ingest_train.py --stage all --n-jobs 3`):

| file | content |
|---|---|
| `spectra/spectra_<lib>.parquet` | one file per ingest_lib (`lib_file_name`), the 20 train columns with `ms2_mzs` / `ms2_normalized_intensities` replaced by the cleaned peaks (`spectra.clean`, list<float32>), plus `source_row, polarity, adduct_test, adduct_known, instrument_family, mono_mass` (formula mass of `molecular_formula`), `m_spec` (neutral mass from precursor and adduct), `m_err_ppm`, `collision_energy` (mean eV), `n_energies, n_peaks_raw, n_peaks, n_above_precursor, prec_intensity, query_ok`. Own subdirectory so that `Library.build(data/processed/train/spectra/)` reads only spectra |
| `meta.parquet` | one row per kept spectrum, no peaks (`META_COLUMNS` in the script), with `file_row` (row in its spectra file) and `source_row` (row in train.parquet) |
| `structures.parquet` | one row per ik14, sorted by ik14: `smiles` (most frequent normalized_smiles), `formula` / `mono_mass` (RDKit on `chem.std_mol`), `formula_given` / `mass_given`, `charge`, `n_components`, `has_isotope`, `elements`, `elements_ok`, `n_carbon`, `db_filter_ok` (PLAN 2.2), `np_score`, `n_spectra`, `n_lib_<lib>`, `n_inst_<family>`, `n_add_<test adduct>` / `n_add_other` / `n_add_unparsed`, `n_pos`, `n_neg`, `n_query_ok`, `libs`, `n_libs`, `in_np`, `in_coconut`, `in_npatlas`, `in_lotus`, `in_pubchemlite`, `coconut_annotation_level`, `pubmed_count`, `n_citations`, `in_pubchem`, `pmid_count`, `mkey`, `mkey_status` (null until `--stage mkey`) |
| `structure_lib_counts.parquet` | (ik14, ingest_lib): n_spectra, n_test_adduct, n_query_ok, n_query_ok_pos / _neg / _qtof |
| `public_test_matches.parquet`, `public_test_truth.parquet` | public test spectra found in train (by spectrum_id, else precursor within 1e-4 Da plus 90 percent of test peaks within 1e-5 Da) and the label per test molecule (PLAN 5.7; WP8's plumbing check uses the latter as truth) |
| `ingest_info.json`, `tolerance.json` | counts, drop reasons and flags per library, timings; the 3.3 tolerance numbers |

`--stage library` (`--library-out`, default `data/processed/library/`) builds the WP5 index from
`spectra/` with `Library.build(..., precleaned=True)` when `ingest_info.json` records the
config's cleaning settings (otherwise it cleans again, with a warning): the stored peaks are used
as cleaned here and the raw-spectrum `n_peaks_raw` / `prec_intensity` are kept (see WP5 section).

`data/processed/validation/` (`python scripts/make_splits.py`): `queries_<set>.parquet` (the 12
test columns and Arrow types, nothing else), `truth_<set>.parquet`, `queries_info.json` (holdout
sha256, reference distributions), `truth_keys.parquet` (per holdout mkey the NP / PubChem ik14
with that mkey, by tier set), and after each evaluation `results_<tag>.parquet` (per molecule
and condition) and `summary_<tag>.json`. `splits/holdout_v1.csv` (tracked): ik14, mkey, set,
fold, eligible_c1, query_lib, primary, mono_mass, formula.

Measured on synthetic data (1 core, nice 10, machine load 7 to 10 from other agents):

| step | input | wall | peak RSS | throughput |
|---|---|---|---|---|
| ingest pass 1, 20k-row batches, with public test join | 107,245 rows, 3 row groups, 411 MB | 13.1 s | 1.03 GB | 8,167 rows/s |
| same, one 107k-row group, no test join | 107,245 rows, 1 row group | 5.5 s | 0.96 GB | 19,402 rows/s |
| RDKit structure pass (formula, mass, NP score) | 543 structures | 0.7 s | | 808 /s per process |
| PubChem membership scan (ik14 column of 120 shards) | 543 structures | 48 s | 1.9 GB (with NP tier) | fixed cost |
| truth keys (PubChem 1 ppm window, formula, ring count, skeleton, mkey) | 40 NP-OTHER molecules | 13.2 s on 2 processes | | 0.33 s per molecule |
| evaluate b0b (NP + PubChem tiers), 9 cells | 130 molecules x 3 conditions | 48 s | 1.8 GB | 0.12 s per molecule-cell |
| evaluate b0 (NP tier) | same | 7.4 s | 1.2 GB | 0.019 s per molecule-cell |

Projection for the real file (2.5M rows, about 275k structures): pass 1 2.5M / 8-19k rows/s =
2.5 to 5 min; structures 275k / (3 x 800/s) = 2 min; PubChem scan 1 min; aggregation and
audit about 1 min: **ingest about 10 min, peak RSS about 4 GB** (cumulative state measured at
240 B per row: metadata 0.6 GB in Arrow plus 0.6 GB in pandas at the end, 0.2 GB of spectrum
ids; per-batch memory stays flat, also with a single large row group). On disk: spectra files
about 0.9 GB (373 B per row), meta.parquet about 0.1 GB. mkey pass: PLAN estimate 1 to 2 h on 4
processes. Truth keys for about 6,250 holdout molecules: 15 to 35 min on 2 processes, cached.
Evaluation of 6,250 molecules x 3 conditions: b0 about 6 min, b0b about 40 min.
<!-- WP6:train:end -->

<!-- WP14:kaggle:start -->
## Kaggle datasets (WP14)

All private, owner `ograhulrajkumar`, published 2026-11-20 with `scripts/kaggle_publish.py` from
git 9833dad (working tree). Each version holds a generated `MANIFEST.json` (per file: path,
size, sha256); the local copy is `dist/kaggle/<slug>/MANIFEST.json`. Mounted on Kaggle at
`/kaggle/input/datasets/ograhulrajkumar/<slug>/`. See docs/KAGGLE.md.

| slug | version | files | bytes (without manifest) | MANIFEST.json sha256 |
|---|---|---|---|---|
| `casmi-code` | 1 | 25 | 584,159 | `4c834891ad96b34f7a6c41516bff2bee43e697964b73a9fc580d0a1d3682ac3b` |
| `casmi-db-np` | 1 | 4 | 470,023,481 | `b4eb658b1818e41efba8e88ecc5f890cc16e97551671c4f7bdaba7b6f7b51d2f` |
| `casmi-db-pubchem` | 1 | 121 | 2,189,271,723 | `347677e9ad81a8e465cb1e06fa22a256129e14bad4dda432f33344d6bd435ef8` |

Upload time: about 10 s for casmi-db-np, 40 s for casmi-db-pubchem (3 parallel uploads), then
20 s and 183 s of Kaggle processing until READY.
<!-- WP14:kaggle:end -->

<!-- ENCODER:start -->
## Contrastive transformer encoder (RESEARCH.md roadmap item 1)

Code: `src/casmi/encoder.py`, `scripts/build_encoder_bundle.py`, `scripts/train_encoder.py`,
`scripts/kaggle_encoder_kernel.py` (generates `notebooks/kaggle_train_encoder.ipynb`).
Results: docs/EXPERIMENTS.md, section "Contrastive transformer encoder v1".

| artefact | path | rows | size | sha256 | how to rebuild |
|---|---|---|---|---|---|
| encoder training bundle v1 | `data/processed/encoder/bundle_v1/` (54 files: spectra, groups, structures, negatives, `eval_np_tims/`, `eval_np_other/`, `info.json`) | 1,963,065 spectra (58.4M peaks), 555,165 groups, 265,499 structures, 34.7M negative links, eval 250 + 3,000 molecules | 856.3 MB | `info.json` `2285fd8fa14b2111...`, `spec_mz.npy` `7bbc53bf24e71e55...`, `st_fp.npy` `216ae3c419d92dc8...`, `neg_idx.npy` `cf8e4813ccde7a7a...`; full list in the Kaggle MANIFEST below | `nice -n 10 python scripts/build_encoder_bundle.py --n-jobs 2` (inputs: library store, train meta/structures, `splits/holdout_v1.csv` sha256 `fa5f843a...cd065c`, NP tier, validation queries/truth/truth_keys); 386 s wall (targets 351 s on 2 busy cores), peak RSS 4.2 GB |

Bundle facts (`info.json`): training rows 2,505,117 library spectra -> 2,190,762 with >= 5
peaks -> 2,186,185 in 100-1300 Da -> 2,186,170 parseable adduct -> 2,105,272 |ppm| <= 20 ->
1,963,065 after `training_exclusion(holdout_v1)` (142,207 rows, 6,454 holdout ik14 / 6,250
mkeys removed). Targets `chem.morgan_fps` of the label SMILES (std_mol + InChI round trip);
for the 31,467 training structures that are also NP-tier rows the target equals the NP-tier
fingerprint for 99.73 percent. Candidate universe = 854,475 NP-tier rows (`np_fp.npy`
sha256 `a526ef20...20f9a9`, not copied) + 234,032 train-only structures. Negatives per
structure within 10 ppm (min 0.002 Da): median 119 (p10 23, p90 302), capped at 256, 18,498
structures topped up to 16 by nearest mass; 45 percent of links point into the NP tier.
Sampling shares per library: enveda-180 0.20, gnps 0.20, riken 0.135, pluskal_ms2 0.12, mona
0.10, msdial 0.07, spectraverse 0.07, massbank 0.065, drug_plus 0.0315, masaryk 0.0085;
inside a library 1 / sqrt(groups of the structure). Eval packs: NP-TIMS 250 molecules / 599
spectra, median NP pool 76.5, truth in pool 100 percent; NP-OTHER 3,000 / 8,167, median pool
49, truth in pool 96.6 percent (M estimated from the spectra by `io.mass_tables`).

Kaggle (all private, owner `ograhulrajkumar`, published with `scripts/kaggle_publish.py`):

| slug | version | content | bytes | MANIFEST.json sha256 |
|---|---|---|---|---|
| `casmi-encoder-code` | 2 | `casmi/` (src/casmi) + `scripts/train_encoder.py` | 672,033 | `51b7b4074e65370289f038a4ca04804c1d099b30867fc3bbfe1d259ceb81aaa4` |
| `casmi-encoder-bundle` | 1 | bundle_v1 | 856,343,566 | `8c6874a8d27fb0f3014a8e9ae6d5c575ec7da326e26780ef98fa724679946e9b` |

Training kernel `ograhulrajkumar/casmi-encoder-train` (private, GPU T4 x2 via machine shape
`NvidiaTeslaT4`, internet off) mounts the two datasets above plus `casmi-db-np` and
`metric/rdkit-2026-3-3-wheel`; outputs are fetched to `dist/kaggle/output/casmi-encoder-train_<stamp>/`.
<!-- ENCODER:end -->

<!-- WP8:b1:start -->
## Baseline 1: library match plus DB prior (WP8)

No new data artefact: the `lib` stage (`casmi.libmatch`, plugin in `configs/b1_library.yaml`)
reads the WP5 library `data/processed/library/` (built 2026-11-20 18:51 by WP6
`ingest_train.py --stage library`, `library_info.json` sha256 `c070c1cc...939e7e`: 2,505,117
spectra, 275,775 structures, 625,990 consensus spectra). Files b1 reads and their sizes (the
Kaggle notebook needs these as a dataset; `open_index_*.npz` are only for open/analog search,
PLAN 4.3, which b1 does not use):

| file | bytes | needed by b1 |
|---|---|---|
| `peaks_mz.npy`, `peaks_int.npy` | 264,945,508 each | yes (memory-mapped) |
| `offsets.npy` | 20,041,072 | yes |
| `meta.parquet` | 94,896,562 | yes |
| `structures.parquet` | 18,536,089 | yes |
| `consensus_mz.npy`, `consensus_int.npy` | 86,056,488 each | yes (consensus pairs) |
| `consensus_offsets.npy` | 5,008,056 | yes |
| `consensus_meta.parquet` | 22,595,475 | yes |
| `library_info.json` | 605 | yes (cleaning settings check) |
| `open_index_pos.npz`, `open_index_neg.npz` | 263,311,309 / 63,224,181 | no (WP10 analog search) |
| total | 1,189,617,341 (863,081,851 without the open index) | |

Loading costs about 1.75 GB RSS (pandas metadata frames and per-spectrum object arrays; the peak
arrays are mmap'd). Location: `cfg.lib_dir` ("" = `paths.PROCESSED/library`, so
`CASMI_ARTEFACTS/processed/library` on Kaggle). Without a library the stage logs a warning and
the run degrades to Baseline 0 behaviour.

| file | rows x guesses | bytes | sha256 | command | wall | peak RSS |
|---|---|---|---|---|---|---|
| `submissions/20261120_b1_library_prior.csv` | 400 x 25 | 405,959 | `406b36df8a93532ceb7dd04fc4b1cc9c49c3e0bdedd52bf49e5e43943239c760` | `python scripts/predict.py --config configs/b1_library.yaml` (NP + PubChem tiers, 2 mkey workers, warm mkey cache, `nice -n 10`, other agents busy) | 120 s (pipeline 118 s) | 4.85 GB |
<!-- WP8:b1:end -->

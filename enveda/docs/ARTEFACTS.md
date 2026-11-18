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
| NP tier fingerprints | `data/processed/db/np/np_fp.npy` | (854,475, 512) uint8 | 437.5 MB | `c14c99ba09e191255ea43ea404f4474d978963cea2c0d532d0d56d7fa4d603a9` | same command (`--stage fp` alone redoes it, 5.5 min on 2 cores); row-aligned to np.parquet, recipe in the WP2 section |
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
`casmi.chem.morgan_fp(casmi.chem.std_mol(Chem.MolFromSmiles(smiles_i)))`, which equals
`casmi.chem.morgan_fps([smiles_i], standardize=True)[0][0]`: WP1 `std_mol` (RemoveHs, metal
disconnection, normalise, reionise, uncharge, remove stereo), then
`rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096, includeChirality=False)`
(other options at RDKit defaults), `GetFingerprintAsNumPy`, `np.packbits` with bit order "big"
(bit j is `(row[j // 8] >> (7 - j % 8)) & 1`; unpack with
`np.unpackbits(row, count=4096)` or `casmi.db.unpack_fp`). `std_mol` failed on 0 rows (the
build would fingerprint the parsed molecule instead and list the rows in build_report.json).
Spot check in `--stage check`: 1,000 sampled rows recomputed with `chem.morgan_fps`, 0
mismatches. If WP1 changes `std_mol` or `morgan_fp`, rerun `build_npdb.py --stage fp`.

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
`--source data/processed/train/` reads WP6's parquet files instead, same columns; `--fake-test`
builds a smoke-test library from `test.parquet` with one fake structure per molecule_id).
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

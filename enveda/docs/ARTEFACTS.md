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

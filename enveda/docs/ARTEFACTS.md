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

## Source notes

Per-source details (actual column names found, download dates, filter counts) go below,
one section per WP.

# scripts/

Command-line entry points. See `docs/PLAN.md` sections 6 and 7.

## The thin-script rule

A script does only three things:

1. parse arguments (`argparse`; a `--config configs/<name>.yaml` option wherever tunables matter);
2. set up logging: `lg = casmi.log.get_logger("<script name>")`, then `casmi.log.log_run(lg, args)`
   (logs the command line, arguments, git commit and paths, and wall time plus peak RSS at exit);
3. call into the `casmi` package.

No logic that inference needs may live in `scripts/`. Anything the Kaggle notebook could need
belongs in `src/casmi/`. Other conventions:

- No hard-coded absolute paths: everything goes through `casmi.paths` (defaults can be
  redirected with `CASMI_ROOT`, `CASMI_ARTEFACTS` or `CASMI_<DIR>`).
- Tunables come from `casmi.config.load_config(args.config)`; never re-declare a default here.
- Log row counts in and out of each stage. Long builds are resumable (stage markers).
- Builds that produce artefacts add their line to `run_all.sh` and a row to `docs/ARTEFACTS.md`.
- Submissions are written only by `predict.py`, validated before writing, and logged in
  `docs/EXPERIMENTS.md`.

## Planned scripts

| Script | Owner | Purpose |
|---|---|---|
| `build_npdb.py` | WP2 | NP-tier database `data/processed/db/np/` (R3: `--stage priors` joins the PubChem documentation priors) |
| `build_pubchem.py` | WP3 | PubChem tier `data/processed/db/pubchem/mass_*.parquet` (R3: `--stage priors` appends the documentation-prior columns) |
| `eval_priors.py` | R3 | spectrum-free prior-only MRR@25 of the documentation priors vs the pmid-only prior on the class-2 proxies |
| `ingest_train.py` | WP6 | cleaned train spectra, `structures.parquet`, mkey cache, `docs/DATA_AUDIT.md` |
| `build_library.py` | WP5 | library index `data/processed/library/` |
| `make_splits.py` | WP7 | `splits/holdout_v1.csv` (tracked) |
| `train_fpmodel.py` | WP9 | fingerprint model `models/fp_v1_*.npz` |
| `build_features.py` | WP10 | `data/processed/features/<set>_<condition>.parquet` |
| `train_ranker.py` | WP11 | `models/ranker_v*.txt` and feature list |
| `predict.py` | WP4 | `--test`, `--config`, `--out`: writes `submissions/YYYYMMDD_<tag>.csv` |
| `evaluate.py` | WP7 | `--config`, `--sets`, `--conditions`, `--tag`: appends to `docs/EXPERIMENTS.md` |
| `package_kaggle.py` | WP14 | `dist/casmi-artefacts/`, `dist/casmi-code/`, `dist/MANIFEST.json` |
| `run_all.sh` | all | rebuilds every artefact from raw inputs, in dependency order |

# Runbook: when train.parquet lands

Put the file at `enveda/data/raw/train.parquet`. Run from `enveda/`. Timings are projections from the synthetic rehearsal (WP6/WP7) on this 4-core machine.

| Step | Command | Expected time | Notes |
|---|---|---|---|
| 1 | `nice -n 10 python scripts/ingest_train.py --stage all --n-jobs 3` | 10-15 min | Streams by row group, about 4 GB peak RAM. Writes `data/processed/train/spectra/*.parquet`, `structures.parquet`, `public_test_truth.parquet`. |
| 2 | `nohup nice -n 10 python scripts/ingest_train.py --stage mkey --mkey-jobs 4 > logs/mkey.out 2>&1 &` | 1-2 h, background | Metric key for every structure; steps 3-4 do not wait for it. |
| 3 | `nice -n 10 python scripts/ingest_train.py --stage library` | 3-5 min | Uses `Library.build(precleaned=True)` when cleaning settings match. |
| 4 | `nice -n 10 python scripts/make_splits.py --n-jobs 2` | 20-40 min | Writes `splits/holdout_v1.csv` (commit it). |
| 5 | `python scripts/ingest_train.py --stage audit` (after step 2) | 1 min | Writes `docs/DATA_AUDIT.md`; take `mass_tol_ppm` from its section 11. |
| 6 | `python scripts/evaluate.py --config configs/b0_npdb.yaml --tag b0_npdb_prior --n-jobs 2` | about 6 min | First honest numbers (b0b with PubChem about 40 min). |

Then, per PLAN.md section 6: WP8 (library match baseline, second submission) -> WP9 training (`scripts/train_fpmodel.py --exclude-holdout --hours 4`, 3 cores) -> WP10 features + Baseline 2 -> WP11 ranker and WP12 formula fit + GEN stage -> WP14 Kaggle packaging -> WP15 ablations.

Rules that bite:
- Model training must filter with `splits.training_exclusion(holdout, structures).filter(df)`.
- Candidate fingerprints must use `chem.morgan_fps` / `maccs_fps` / `fp_mol` (InChI round trip), never `morgan_fp(std_mol(...))`.
- WP8 must pass `queries` with `molecule_id` to `Library.search_by_mass` so validation exclusions apply per molecule.
- Numba caches need a writable `NUMBA_CACHE_DIR` on Kaggle.

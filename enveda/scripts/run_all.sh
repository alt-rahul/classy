#!/usr/bin/env bash
# Rebuild every artefact from raw inputs and public URLs, in dependency order
# (PLAN.md section 7, "Artefact persistence"; order from section 6).
#
# Rules:
# - Each WP adds its own line(s) when its script exists: replace the matching
#   commented placeholder below (keep the order), one `$PY scripts/<x>.py ...` call per
#   artefact, with a comment giving wall time and cores.
# - Steps must be resumable or idempotent; no logic here, only script calls.
# - Expected sizes and checksums of the outputs are recorded in docs/ARTEFACTS.md.
#
# Usage: scripts/run_all.sh            (PYTHON=python3.12 scripts/run_all.sh to pick one)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

# Inputs: data/raw/{train.parquet,test.parquet,sample_submission.csv} from Kaggle.

# WP2  NP-tier database: downloads 4 sources (600 MB), ~20 min wall, 2 cores, <4 GB RAM
nice -n 10 $PY -I scripts/build_npdb.py --stage all
# WP3  PubChem tier: about 15 min wall (downloads at about 20 MB/s), 2 cores, 7 GB RAM; resumable, skips finished stages
#      R3: stage `priors` (part of the default run) streams 6 more PubChem FTP files (about 21 GB compressed) and
#      appends the documentation-prior columns to every shard; see docs/ARTEFACTS.md for its wall time, 2 cores,
#      under 2 GB RSS, DuckDB memory_limit min(6 GB, available - 1 GB)
nice -n 10 $PY -I scripts/build_pubchem.py --memory-limit 6GB
# R3   NP tier priors: joins data/external/pubchem/np_priors.parquet (written by the stage above) into np.parquet,
#      under 1 min, 1 core, about 1 GB RAM
nice -n 10 $PY -I scripts/build_npdb.py --stage priors
# WP6  train ingestion + structure table + public test join + audit: estimate 10-15 min wall
#      (streaming pass 3-5 min, RDKit structure pass about 3 min on 3 jobs, PubChem scan 1 min),
#      3 cores, peak RSS about 4 GB (projected from synthetic runs, docs/ARTEFACTS.md WP6)
nice -n 10 $PY scripts/ingest_train.py --stage all --n-jobs 3
# WP6  metric keys of all structures: 1-2 h on 4 processes (normally a background job; resumable
#      through data/processed/cache/mkey.parquet), then the audit again for the mkey items
nice -n 10 $PY scripts/ingest_train.py --stage mkey --mkey-jobs 4
nice -n 10 $PY scripts/ingest_train.py --stage audit
# WP5  library index: streams train.parquet by row groups (or WP6's data/processed/train/
#      via --source); estimate 10-20 min wall, 1 core, under 4 GB RAM (see docs/ARTEFACTS.md)
$PY scripts/build_library.py
# WP7  query tables and truth keys from the tracked splits/holdout_v1.csv (drop --queries-only
#      to redraw the split): estimate 20-40 min, mostly truth keys, 2 cores, under 3 GB RAM
nice -n 10 $PY scripts/make_splits.py --queries-only --n-jobs 2
# WP9  fingerprint model: each run --hours of training (default 4) plus about 10 min of loading
#      and target fingerprints, 3 cores, <10 GB RAM; needs the WP5 library and WP7 splits
nice -n 10 $PY -I scripts/train_fpmodel.py --exclude-holdout --resume
nice -n 10 $PY -I scripts/train_fpmodel.py --all --resume
# WP10 candidate features         # $PY scripts/build_features.py
# WP11 ranker                     # $PY scripts/train_ranker.py
# WP14 Kaggle bundle              # $PY scripts/package_kaggle.py

echo "run_all.sh: done"

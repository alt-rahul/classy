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
nice -n 10 $PY -I scripts/build_pubchem.py
# WP6  train ingestion and audit  # $PY scripts/ingest_train.py
# WP5  library index              # $PY scripts/build_library.py
# WP7  holdout splits             # $PY scripts/make_splits.py   (splits/ is tracked; normally skipped)
# WP9  fingerprint model          # $PY scripts/train_fpmodel.py --exclude-holdout ; --all
# WP10 candidate features         # $PY scripts/build_features.py
# WP11 ranker                     # $PY scripts/train_ranker.py
# WP14 Kaggle bundle              # $PY scripts/package_kaggle.py

echo "run_all.sh: done"

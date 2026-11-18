"""Build the spectral library index (WP5, PLAN.md 4.1-4.3) into data/processed/library/.

Usage:
    python scripts/build_library.py                       # from data/raw/train.parquet
    python scripts/build_library.py --source data/processed/train/spectra/ --precleaned  # WP6 files
    python scripts/build_library.py --fake-test --out <dir>   # smoke test from test.parquet

Thin wrapper around ``casmi.library.Library.build``; see that module for the file formats.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from casmi import paths
from casmi.config import load_config
from casmi.log import get_logger, log_run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, default=None,
                    help="parquet file or directory of parquet files in the train schema "
                         "(default: data/raw/train.parquet)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: data/processed/library)")
    ap.add_argument("--config", type=Path, default=None, help="YAML config overrides")
    ap.add_argument("--batch-rows", type=int, default=None,
                    help="rows per streamed batch (default: Config.lib_build_batch_rows)")
    ap.add_argument("--precleaned", action="store_true",
                    help="source peaks are already cleaned with this config (WP6's "
                         "data/processed/train/spectra/): store them as given")
    ap.add_argument("--fake-test", action="store_true",
                    help="build from test.parquet with fake labels (one structure per "
                         "molecule_id); for smoke tests before train exists")
    args = ap.parse_args(argv)

    lg = get_logger("build_library")
    log_run(lg, args)
    cfg = load_config(args.config)

    import pandas as pd

    from casmi.library import Library, fake_labels_from_test

    if args.fake_test:
        source = fake_labels_from_test(pd.read_parquet(args.source or paths.TEST_PARQUET))
        lg.info("fake-label library from %s: %d rows", args.source or paths.TEST_PARQUET,
                len(source))
    else:
        source = args.source or paths.TRAIN_PARQUET
        if not Path(source).exists():
            lg.error("source not found: %s", source)
            return 2
    lib = Library.build(source, cfg, out_dir=args.out, batch_rows=args.batch_rows,
                        precleaned=args.precleaned)
    info = lib.info
    lg.info("library %s: %d spectra, %d structures, %d consensus, %d peaks; drops %s",
            lib.dir, info["n_spectra"], info["n_structures"], info["n_consensus"],
            info["n_peaks"], info["drops"])
    total = sum(f.stat().st_size for f in lib.dir.iterdir() if f.is_file())
    lg.info("library size on disk: %.1f MB", total / 1e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())

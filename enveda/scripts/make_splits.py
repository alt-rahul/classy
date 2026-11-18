"""Holdout splits and validation query tables (WP7, PLAN.md 5.2).

Reads WP6's ``structures.parquet`` and ``structure_lib_counts.parquet`` (``--train-dir``,
default data/processed/train), writes the holdout file (``--out``, default
splits/holdout_v1.csv, tracked in git) and, unless ``--no-queries``, the test-shaped query
and truth tables per set into ``--val-dir`` (default data/processed/validation).

Usage:
    python scripts/make_splits.py                                   # real run, plan sizes
    python scripts/make_splits.py --train-dir data/processed/synthetic/train \\
        --out data/processed/synthetic/holdout_synth.csv \\
        --val-dir data/processed/synthetic/validation \\
        --raw data/processed/synthetic/train_synth.parquet --n-syn 40 --n-np-other 40

Thin wrapper around ``casmi.splits.make_holdout`` and ``casmi.validation.build_queries``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from casmi import paths
from casmi.config import load_config
from casmi.log import get_logger, log_run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train-dir", type=Path, default=paths.PROCESSED / "train")
    ap.add_argument("--out", type=Path, default=paths.SPLITS / "holdout_v1.csv")
    ap.add_argument("--val-dir", type=Path, default=paths.PROCESSED / "validation")
    ap.add_argument("--raw", type=Path, default=paths.TRAIN_PARQUET,
                    help="train-schema parquet with the raw peaks for the query tables")
    ap.add_argument("--test", type=Path, default=paths.TEST_PARQUET,
                    help="public test file (reference distributions for query emulation)")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--n-np-tims", type=int, default=0, help="0 = all candidates (plan: all 250)")
    ap.add_argument("--n-syn", type=int, default=None, help="default Config.val_n_syn_tims")
    ap.add_argument("--n-np-other", type=int, default=None, help="default Config.val_n_np_other")
    ap.add_argument("--n-jobs", type=int, default=2, help="processes for missing mkeys")
    ap.add_argument("--no-queries", action="store_true", help="only write the holdout file")
    ap.add_argument("--queries-only", action="store_true",
                    help="rebuild the query tables from an existing holdout file")
    args = ap.parse_args(argv)
    lg = get_logger("make_splits")
    log_run(lg, args)
    cfg = load_config(args.config)

    from casmi.splits import SETS, holdout_molecules, load_holdout, make_holdout, save_holdout
    from casmi.validation import build_queries

    if args.queries_only:
        ho = load_holdout(args.out)
    else:
        st = pd.read_parquet(args.train_dir / "structures.parquet")
        lc = pd.read_parquet(args.train_dir / "structure_lib_counts.parquet")
        lg.info("structures %d (mkey present for %d), lib counts %d", len(st),
                int(st["mkey"].notna().sum()), len(lc))
        ho, _ = make_holdout(st, lc, cfg, n_np_tims=args.n_np_tims or None, n_syn_tims=args.n_syn,
                             n_np_other=args.n_np_other, n_jobs=args.n_jobs)
        path = save_holdout(ho, args.out)
        mols = holdout_molecules(ho)
        for s in SETS:
            m = mols[mols["set"] == s]
            if not len(m):
                lg.warning("set %s is empty", s)
                continue
            q = np.percentile(m["mono_mass"], [0, 25, 50, 75, 100])
            lg.info("%s: %d molecules (%d ik14 rows), eligible_c1 %d, folds %s, mass "
                    "min/p25/median/p75/max %s, query libs %s", s, len(m),
                    int((ho["set"] == s).sum()), int(m["eligible_c1"].sum()),
                    m["fold"].value_counts().sort_index().to_dict(),
                    "/".join(f"{x:.0f}" for x in q), m["query_lib"].value_counts().to_dict())
        lg.info("wrote %s: %d rows, %d molecules", path, len(ho), len(mols))
    if not args.no_queries:
        build_queries(ho, args.train_dir, cfg, raw=args.raw, test_path=args.test,
                      out_dir=args.val_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

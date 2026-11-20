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

Protocol v2 (ROADMAP 2.3, R1): ``--v2`` keeps holdout_v1 unchanged and writes
splits/holdout_v2.csv + holdout_v2.json (v2 columns, C2-ZEN rows when R2 has built them, w_obs
calibrated on b0's results), then, with ``--c3h``, the C3h near-neighbour sets of the NP sets
(train + NP tier; ``--c3h-pubchem N`` adds PubChem-only window neighbours for N molecules):

    python scripts/make_splits.py --v2 --c3h --n-jobs 2

Thin wrapper around ``casmi.splits.make_holdout``, ``casmi.validation.build_queries`` and
``casmi.validation.build_holdout_v2``.
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
    ap.add_argument("--no-truth-keys", action="store_true",
                    help="skip precomputing the truth keys (DB rows with the truth mkey; "
                         "evaluate.py computes them lazily and caches them)")
    ap.add_argument("--v2", action="store_true",
                    help="build splits/holdout_v2.csv from holdout_v1 (ROADMAP 2.3) and stop")
    ap.add_argument("--v2-out", type=Path, default=paths.SPLITS / "holdout_v2.csv")
    ap.add_argument("--b0-results", type=Path, default=None,
                    help="b0 per-molecule results for the w_obs calibration (default: the v2 "
                         "full-panel table if present, else results_b0_npdb_prior.parquet)")
    ap.add_argument("--c3h", action="store_true", help="--v2: also compute the C3h neighbour sets")
    ap.add_argument("--c3h-pubchem", type=int, default=0,
                    help="--v2 --c3h: PubChem-only window neighbours for this many NP-TIMS molecules")
    args = ap.parse_args(argv)
    lg = get_logger("make_splits")
    log_run(lg, args)
    cfg = load_config(args.config)
    if args.v2:
        return _v2(args, cfg, lg)

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
        res = build_queries(ho, args.train_dir, cfg, raw=args.raw, test_path=args.test,
                            out_dir=args.val_dir)
        if not args.no_truth_keys:
            from casmi.validation import truth_keys

            for s, (_, truth) in res.items():
                keys = truth_keys(truth, cfg, n_jobs=args.n_jobs,
                                  cache_path=args.val_dir / "truth_keys.parquet")
                extra = sum(len(v) for v in keys.values()) - sum(
                    len(str(g).split(";")) for g in truth["ik14_group"])
                lg.info("truth keys %s: %d molecules, %d DB ik14 beyond the train ik14 groups",
                        s, len(keys), extra)
    return 0


def _v2(args, cfg, lg) -> int:
    from casmi.splits import NP_SETS
    from casmi.validation import (build_holdout_v2, c3h_neighbours, c3h_removed_counts,
                                  load_queries, truth_keys)

    ho1_path = args.out if args.out.exists() else paths.SPLITS / "holdout_v1.csv"
    ho2, meta = build_holdout_v2(cfg, holdout_v1=ho1_path, train_dir=args.train_dir,
                                 val_dir=args.val_dir, out=args.v2_out, b0_results=args.b0_results)
    lg.info("holdout v2 counts: %s", meta["counts"])
    lg.info("w_obs alpha per mix: %s (flags %s); b0 check full %s, fast %s", meta["alpha"],
            meta["calibration"]["flags"], meta["b0_check_full"], meta["b0_check_fast"])
    if args.c3h:
        rows = []
        for s in NP_SETS:
            _, t = load_queries(s, args.val_dir)
            near = c3h_neighbours(t, cfg, train_dir=args.train_dir, n_jobs=args.n_jobs,
                                  cache_path=args.val_dir / "c3h_neighbours.parquet")
            if args.c3h_pubchem and s == "NP-TIMS":
                sub = t.sort_values("molecule_id").head(args.c3h_pubchem)
                extra = c3h_neighbours(sub, cfg, train_dir=args.train_dir, n_jobs=args.n_jobs,
                                       cache_path=args.val_dir / "c3h_neighbours.parquet",
                                       pubchem=True)
                for k, v in extra.items():
                    near.setdefault(k, {}).update(v)
            keys = truth_keys(t, cfg, n_jobs=args.n_jobs, cache_path=args.val_dir / "truth_keys.parquet")
            rows.append(c3h_removed_counts(t, keys, near))
        cnt = pd.concat(rows, ignore_index=True)
        cnt.to_parquet(args.val_dir / "c3h_removed_counts.parquet", index=False)
        for s, g in cnt.groupby("set"):
            cols = [c for c in g.columns if c.startswith("n_")]
            lg.info("C3h removed per molecule %s (n %d): median %s; mean %s", s, len(g),
                    g[cols].median().to_dict(), g[cols].mean().round(2).to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())

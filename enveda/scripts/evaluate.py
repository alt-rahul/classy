"""Local validation of the pipeline on the holdout cells (WP7, PLAN.md 5.3-5.5).

Runs ``casmi.pipeline.predict`` (with ``--config``) on every (set, condition) cell through the
``casmi.validation`` harness, after the self check (truth predictor exactly 1.0, CCO exactly
0.0), prints the cells, blends and the worst-case blend with its bootstrap interval, writes
``<val-dir>/results_<tag>.parquet`` (per molecule) and ``summary_<tag>.json``, and appends one
row to docs/EXPERIMENTS.md (``--experiments``; ``--no-append`` to skip).

Usage:
    python scripts/evaluate.py --config configs/b0_npdb.yaml --tag b0_npdb_prior
    python scripts/evaluate.py --sets NP-TIMS --conditions C1 C2 --tag quick --max-molecules 50

Synthetic dry run (all paths redirected):
    python scripts/evaluate.py --config configs/b0_npdb.yaml --tag synth_b0 \\
        --train-dir data/processed/synthetic/train --holdout data/processed/synthetic/holdout_synth.csv \\
        --val-dir data/processed/synthetic/validation --library data/processed/synthetic/library \\
        --raw data/processed/synthetic/train_synth.parquet \\
        --experiments data/processed/synthetic/EXPERIMENTS_synth.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from casmi import paths
from casmi.config import load_config
from casmi.log import get_logger, log_run


def main(argv: list[str] | None = None) -> int:
    from casmi.splits import SETS
    from casmi.validation import CONDITIONS

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--tag", default="adhoc", help="run tag (as in the submission name)")
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--train-dir", type=Path, default=paths.PROCESSED / "train")
    ap.add_argument("--holdout", type=Path, default=paths.SPLITS / "holdout_v1.csv")
    ap.add_argument("--val-dir", type=Path, default=paths.PROCESSED / "validation")
    ap.add_argument("--library", default=None,
                    help="library directory for masks and checks (default data/processed/library "
                         "if built; 'none' to skip)")
    ap.add_argument("--raw", type=Path, default=paths.TRAIN_PARQUET,
                    help="raw train parquet, used only if the query tables must be rebuilt")
    ap.add_argument("--test", type=Path, default=paths.TEST_PARQUET)
    ap.add_argument("--batch", type=int, default=None,
                    help="molecules per predictor call (default Config.batch_molecules; 1 = exact "
                         "even for union fallbacks)")
    ap.add_argument("--n-jobs", type=int, default=None,
                    help="mkey worker processes (overrides Config.n_jobs for the whole run)")
    ap.add_argument("--max-molecules", type=int, default=None, help="first N molecules per set")
    ap.add_argument("--experiments", type=Path, default=paths.DOCS / "EXPERIMENTS.md")
    ap.add_argument("--no-append", action="store_true")
    ap.add_argument("--notes", default="")
    ap.add_argument("--skip-self-check", action="store_true")
    ap.add_argument("--rebuild-queries", action="store_true")
    args = ap.parse_args(argv)
    lg = get_logger("evaluate")
    log_run(lg, args)
    cfg = load_config(args.config)
    if args.n_jobs is not None:
        cfg = cfg.replace(n_jobs=args.n_jobs)
    n_jobs = cfg.n_jobs

    import numpy as np

    from casmi.splits import load_holdout
    from casmi.validation import (append_experiments, build_queries, evaluate, format_cells,
                                  pipeline_predictor, queries_current, self_check, summarize)

    ho = load_holdout(args.holdout)
    if args.rebuild_queries or not queries_current(ho, args.val_dir, args.sets):
        lg.info("building query tables in %s", args.val_dir)
        build_queries(ho, args.train_dir, cfg, raw=args.raw, test_path=args.test,
                      out_dir=args.val_dir)
    lib = None
    lib_dir = None if args.library == "none" else (
        Path(args.library) if args.library else paths.PROCESSED / "library")
    if lib_dir is not None and (lib_dir / "meta.parquet").exists():
        from casmi.library import Library

        lib = Library.load(lib_dir, cfg)
        lg.info("library %s: %d spectra, %d structures", lib_dir, lib.n_spectra, lib.n_structures)
    elif args.library not in (None, "none"):
        lg.error("library not found: %s", lib_dir)
        return 2
    else:
        lg.warning("no library: C1/C2/C3 library checks skipped (eligible_c1 from the split)")

    common = dict(sets=args.sets, conditions=args.conditions, val_dir=args.val_dir, library=lib,
                  train_dir=args.train_dir, n_jobs=n_jobs, max_molecules=args.max_molecules)
    if not args.skip_self_check:
        self_check(cfg, **common)
    t0 = time.perf_counter()
    table, cells = evaluate(pipeline_predictor, cfg, batch=args.batch, **common)
    wall = time.perf_counter() - t0
    summ = summarize(table, cfg)
    summ["wall_s"] = round(wall, 1)
    summ["tag"] = args.tag
    summ["config"] = str(args.config) if args.config else "defaults"
    summ["cell_seconds"] = {f"{c.set}|{c.condition}": round(c.seconds, 1) for c in cells}
    summ["union_fallbacks"] = int(sum(c.n_union_fallbacks for c in cells))
    out_dir = paths.ensure_dir(args.val_dir)
    table.to_parquet(out_dir / f"results_{args.tag}.parquet", index=False)
    (out_dir / f"summary_{args.tag}.json").write_text(json.dumps(summ, indent=2, default=float))
    lg.info("%-9s %-3s %5s %7s %6s %6s %6s %7s %7s", "set", "", "n", "MRR", "top1", "top5",
            "top25", "recall", "pool")
    for key, d in summ["cells"].items():
        s, c = key.split("|")
        if not d.get("n"):
            continue
        lg.info("%-9s %-3s %5d %7.4f %6.3f %6.3f %6.3f %7.3f %7.0f", s, c, d["n"], d["mrr"],
                d["top1"], d["top5"], d["top25"], d["recall_any"], d["pool_median"])
    for s, d in summ["sets"].items():
        lo, hi = d["worst_ci"]
        lg.info("%s blends %s; worst %.4f [%.4f, %.4f]", s,
                {k: round(v, 4) for k, v in d["blends"].items()}, d["worst"], lo, hi)
    lg.info("primary (NP-TIMS worst-case blend): %.4f %s; evaluation wall %.1f s",
            summ["primary"], [round(x, 4) if np.isfinite(x) else x for x in summ["primary_ci"]], wall)
    lg.info("cells: %s", format_cells(summ))
    if not args.no_append:
        notes = args.notes
        if args.max_molecules:
            notes = (notes + f" first {args.max_molecules} molecules per set").strip()
        p = append_experiments(summ, args.tag, summ["config"], args.experiments,
                               notes=(notes + f" (eval {wall:.0f} s)").strip())
        lg.info("appended a row to %s", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())

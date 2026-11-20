"""Local validation of the pipeline on the holdout cells (WP7; protocol v2 by R1, ROADMAP 2).

Protocol v2 (default, binding for decisions): cells C1 / C2-NP / C2-PC / C3h (+ C3e and the
secondary C1o reported) on ``--panel fast`` (about 1,500 molecules, every experiment) or
``full`` (milestones), the LB-proxy P(mix) for ``--mix A,B,C`` with bootstrap intervals,
w_obs and w_fast weights from ``splits/holdout_v2.csv`` / ``holdout_v2.json``
(``scripts/make_splits.py --v2``). Writes ``<val-dir>/results_v2_<tag>.parquet`` (per
molecule and cell) and ``summary_v2_<tag>.json``, appends one row to docs/EXPERIMENTS.md.

Usage:
    python scripts/evaluate.py --config configs/b0_npdb.yaml --tag b0_npdb_prior     # fast, A,B,C
    python scripts/evaluate.py --config configs/b1_library.yaml --tag b1 --panel full
    python scripts/evaluate.py --config configs/b1_library.yaml --tag b1 --blind     # spectrum-blind audit
    python scripts/evaluate.py --config configs/b1_library.yaml --tag b1 --store-pools
    python scripts/evaluate.py --from-pools b1 --config configs/b1_variant.yaml --tag b1v   # rescore, seconds
    python scripts/evaluate.py --from-pools b1 --score-col my_rank_score --tag r8_try        # R8 scores
    python scripts/evaluate.py --compare b1 b1v                                    # paired bootstrap, adopt?
    python scripts/evaluate.py --from-v1 b0b_pubchem_prior --tag b0b_v2 --panel full  # old v1 run, v2 numbers

Protocol v1 (WP7, kept for reproducing old rows): ``--protocol v1`` (or any of ``--sets`` /
``--conditions``), e.g. ``python scripts/evaluate.py --protocol v1 --config configs/b0_npdb.yaml
--tag b0_npdb_prior``. Writes results_<tag>.parquet / summary_<tag>.json as before.

Synthetic dry run (v1, all paths redirected):
    python scripts/evaluate.py --protocol v1 --config configs/b0_npdb.yaml --tag synth_b0 \\
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
    from casmi.validation import CELLS_V2, CONDITIONS

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--tag", default="adhoc", help="run tag (as in the submission name)")
    ap.add_argument("--protocol", choices=["v1", "v2"], default=None,
                    help="default v2 (v1 when --sets or --conditions is given)")
    # v2
    ap.add_argument("--panel", choices=["fast", "full"], default="fast")
    ap.add_argument("--mix", default="A,B,C", help="mixes of the LB-proxy (Config.val_v2_mixes)")
    ap.add_argument("--cells", nargs="+", default=list(CELLS_V2), choices=list(CELLS_V2))
    ap.add_argument("--v2-sets", nargs="+", default=None, help="default: every set with query files")
    ap.add_argument("--holdout-v2", type=Path, default=paths.SPLITS / "holdout_v2.csv")
    ap.add_argument("--blind", action="store_true",
                    help="spectrum-blind audit: every query spectrum replaced by one constant spectrum")
    ap.add_argument("--store-pools", action="store_true",
                    help="store the scored pools under <val-dir>/pools/<tag>/ (for --from-pools)")
    ap.add_argument("--from-pools", default=None, metavar="TAG",
                    help="rescore the pools stored by run TAG (no pipeline run)")
    ap.add_argument("--score-col", default=None,
                    help="--from-pools: rank by this pool column (default: fusion.fuse with --config, "
                         "or the stored score without --config)")
    ap.add_argument("--inexact", action="store_true", help="--from-pools: dedupe by ik14, not mkey")
    ap.add_argument("--probe", default=None, choices=["lib_hits", "np_only", "pubchem_only", "novel"],
                    help="--from-pools: local MRRs of a probe submission (casmi.probes rule, thresholds "
                         "from --config): keep that source's rows only, no padding")
    ap.add_argument("--from-v1", default=None, metavar="TAG",
                    help="rescore the v1 per-molecule results of run TAG under protocol v2 (the v1 "
                         "C1 / C2 masks are the v2 ones; C3h is a structural 0 without a generator)")
    ap.add_argument("--compare", nargs=2, default=None, metavar=("TAG_A", "TAG_B"),
                    help="paired bootstrap of results_v2_<TAG_B> minus <TAG_A> and the adoption verdict")
    ap.add_argument("--run-c3", action="store_true",
                    help="run C3h/C3e even without a generator stage (else structural zeros)")
    ap.add_argument("--c3h-pubchem", action="store_true",
                    help="C3h also removes PubChem-only neighbours in the truth window (slow)")
    ap.add_argument("--boot", type=int, default=None, help="bootstrap resamples (Config.val_bootstrap_n)")
    # v1
    ap.add_argument("--sets", nargs="+", default=None, choices=list(SETS), help="v1 sets")
    ap.add_argument("--conditions", nargs="+", default=None, choices=list(CONDITIONS),
                    help="v1 conditions")
    ap.add_argument("--holdout", type=Path, default=paths.SPLITS / "holdout_v1.csv")
    # both
    ap.add_argument("--train-dir", type=Path, default=paths.PROCESSED / "train")
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
    protocol = args.protocol or ("v1" if (args.sets is not None or args.conditions is not None) else "v2")
    if protocol == "v1":
        return _v1(args, cfg, lg)
    return _v2(args, cfg, lg)


def _library(args, cfg, lg):
    lib_dir = None if args.library == "none" else (
        Path(args.library) if args.library else paths.PROCESSED / "library")
    if lib_dir is not None and (lib_dir / "meta.parquet").exists():
        from casmi.library import Library

        lib = Library.load(lib_dir, cfg)
        lg.info("library %s: %d spectra, %d structures", lib_dir, lib.n_spectra, lib.n_structures)
        return lib, 0
    if args.library not in (None, "none"):
        lg.error("library not found: %s", lib_dir)
        return None, 2
    lg.warning("no library: library checks skipped (eligible_c1 from the split)")
    return None, 0


# --------------------------------------------------------------------------------------
# protocol v2
# --------------------------------------------------------------------------------------

def _log_report(lg, rep: dict) -> None:
    from casmi.validation import REPORT_CELLS

    for m, d in rep["mixes"].items():
        ci = d.get("P_ci") or [float("nan")] * 2
        lg.info("P(%s) = %.4f [%.4f, %.4f]  (top-1 part %.4f, tail %.4f; alpha %.3f; pi %s)%s", m,
                d["P"], ci[0], ci[1], d["P_top1"], d["P_tail"], d["alpha"], d["pi"],
                f"; missing cells {d['missing_cells']} counted 0" if d["missing_cells"] else "")
    a = rep["mixes"].get("A") or next(iter(rep["mixes"].values()))
    lg.info("%-6s %6s %8s %8s %8s %8s %16s %8s", "cell", "n", "n_eff", "MRR_w", "top1_w", "tail_w",
            "90% CI", "MRR_raw")
    for c in REPORT_CELLS:
        d, raw = a["cells"].get(c, {}), rep["cells"].get(c, {})
        if not raw.get("n"):
            continue
        ci = d.get("ci") or [float("nan")] * 2
        lg.info("%-6s %6d %8.1f %8.4f %8.4f %8.4f [%.4f, %.4f] %8.4f", c, raw["n"], d.get("n_eff", 0),
                d["mrr"], d["top1"], d["tail"], ci[0], ci[1], raw["mrr_unweighted"])


def _v2(args, cfg, lg) -> int:
    import numpy as np
    import pandas as pd

    from casmi.splits import HOLDOUT_COLUMNS, SETS, load_holdout_v2, load_holdout_v2_meta
    from casmi.validation import (append_experiments_v2, build_queries, evaluate_v2,
                                  format_proxy, fusion_scorer, load_pools, paired_report,
                                  parse_mixes,
                                  pipeline_predictor, pool_capture_predictor, PoolStore,
                                  proxy_report, queries_current, ranks_from_pools,
                                  rescore_table, self_check_v2, v2_attributes)

    if not args.holdout_v2.exists():
        lg.error("%s not found: run scripts/make_splits.py --v2 first", args.holdout_v2)
        return 2
    ho2 = load_holdout_v2(args.holdout_v2)
    attrs = v2_attributes(ho2)
    meta = load_holdout_v2_meta(args.holdout_v2)
    mixes = parse_mixes(args.mix, cfg)
    n_jobs = cfg.n_jobs
    out_dir = paths.ensure_dir(args.val_dir)
    common = dict(panel=args.panel, mixes=mixes, meta=meta, n_boot=args.boot)

    if args.compare:
        a, b = args.compare
        ta = pd.read_parquet(out_dir / f"results_v2_{a}.parquet")
        tb = pd.read_parquet(out_dir / f"results_v2_{b}.parquet")
        rep = paired_report(ta, tb, attrs, cfg, **common)
        (out_dir / f"compare_v2_{a}__{b}.json").write_text(json.dumps(rep, indent=2, default=float))
        for m, d in rep["mixes"].items():
            lg.info("P(%s): %s %.4f -> %s %.4f, diff %+.4f [%+.4f, %+.4f] (top-1 %+.4f, tail %+.4f)",
                    m, a, d["P_a"], b, d["P_b"], d["diff"], d["ci"][0], d["ci"][1], d["top1_diff"],
                    d["tail_diff"])
            for c, v in d["cells"].items():
                ci = v["ci"] or [float("nan")] * 2
                lg.info("   %-6s n %5d  %.4f -> %.4f  diff %+.4f [%+.4f, %+.4f]", c, v["n"], v["a"],
                        v["b"], v["diff"], ci[0], ci[1])
        lg.info("adopt %s: %s (%s)", b, rep.get("adopt"), rep.get("adopt_checks"))
        return 0

    t0 = time.perf_counter()
    if args.from_v1:
        from casmi.validation import v1_to_v2

        table = v1_to_v2(pd.read_parquet(out_dir / f"results_{args.from_v1}.parquet"), attrs)
        lg.info("v1 results of %s as v2 cells: %s", args.from_v1, table["cell"].value_counts().to_dict())
    elif args.from_pools:
        src = args.from_pools
        base = pd.read_parquet(out_dir / f"results_v2_{src}.parquet")
        pools = load_pools(out_dir / "pools" / src)
        if args.score_col:
            score = args.score_col
        elif args.config is not None:
            score = fusion_scorer()
        else:
            score = None
        fb = base
        if args.probe:
            from casmi.probes import probe_mask

            pcfg = cfg.replace(probe_rule=args.probe)
            pools = pools[probe_mask(pools, pcfg)].reset_index(drop=True)
            fb = None   # probe lists carry no padding
        ranks = ranks_from_pools(pools, cfg, score, exact=not args.inexact, n_jobs=n_jobs, fallback=fb)
        table = rescore_table(base, ranks)
        if args.probe:  # molecules with no row of that source score 0
            key = set(zip(ranks["set"], ranks["cell"], ranks["molecule_id"]))
            miss = [(s_, c_, m_) not in key for s_, c_, m_ in
                    zip(table["set"], table["cell"], table["molecule_id"])]
            miss = pd.Series(miss, index=table.index) & ~table["structural"].astype(bool)
            table.loc[miss, ["rr", "rank"]] = [0.0, 0]
            table.loc[miss, ["top1", "top5", "top25"]] = False
        lg.info("rescored %d (molecule, cell) pools of run %s (%d candidate rows)", len(ranks), src, len(pools))
    else:
        v1part = ho2[ho2["set"].isin(SETS)][list(HOLDOUT_COLUMNS)]
        if args.rebuild_queries or not queries_current(v1part, args.val_dir, SETS):
            lg.info("building v1-set query tables in %s", args.val_dir)
            build_queries(v1part, args.train_dir, cfg, raw=args.raw, test_path=args.test,
                          out_dir=args.val_dir)
        lib, rc = _library(args, cfg, lg)
        if rc:
            return rc
        run = dict(panel=args.panel, cells=args.cells, sets=args.v2_sets, holdout=args.holdout_v2,
                   val_dir=args.val_dir, library=lib, train_dir=args.train_dir, n_jobs=n_jobs,
                   max_molecules=args.max_molecules)
        if not args.skip_self_check and not args.blind:
            self_check_v2(cfg, **run)
        store = PoolStore(cfg.val_v2_store_keep) if args.store_pools else None
        fn = pool_capture_predictor(store) if store is not None else pipeline_predictor
        t0 = time.perf_counter()
        table, cells = evaluate_v2(fn, cfg, batch=args.batch, blind=args.blind, store=store,
                                   run_c3=args.run_c3, c3h_pubchem=args.c3h_pubchem, **run)
        if store is not None:
            written = store.save(out_dir / "pools" / args.tag)
            lg.info("stored pools: %d files under %s", len(written), out_dir / "pools" / args.tag)
    wall = time.perf_counter() - t0
    rep = proxy_report(table, attrs, cfg, **common)
    rep.update(tag=args.tag, config=str(args.config) if args.config else "defaults",
               wall_s=round(wall, 1), blind=bool(args.blind), from_pools=args.from_pools,
               holdout_v2=str(args.holdout_v2.name))
    table.to_parquet(out_dir / f"results_v2_{args.tag}.parquet", index=False)
    (out_dir / f"summary_v2_{args.tag}.json").write_text(json.dumps(rep, indent=2, default=float))
    _log_report(lg, rep)
    lg.info("v2 %s panel, tag %s: evaluation wall %.1f s", args.panel, args.tag, wall)
    lg.info("cells: %s", format_proxy(rep))
    if not args.no_append:
        notes = args.notes
        if args.blind:
            notes = (notes + " spectrum-blind audit").strip()
        if args.from_pools:
            notes = (notes + f" rescored from the stored pools of {args.from_pools}"
                     + (f", probe {args.probe}" if args.probe else "")).strip()
        if args.from_v1:
            notes = (notes + f" v1 run {args.from_v1} rescored under protocol v2").strip()
        if args.max_molecules:
            notes = (notes + f" first {args.max_molecules} molecules per set").strip()
        p = append_experiments_v2(rep, args.tag, rep["config"], args.experiments,
                                  notes=(notes + f" (eval {wall:.0f} s)").strip())
        lg.info("appended a row to %s", p)
    return 0


# --------------------------------------------------------------------------------------
# protocol v1 (WP7)
# --------------------------------------------------------------------------------------

def _v1(args, cfg, lg) -> int:
    import numpy as np

    from casmi.splits import SETS, load_holdout
    from casmi.validation import (CONDITIONS, append_experiments, build_queries, evaluate,
                                  format_cells, pipeline_predictor, queries_current, self_check,
                                  summarize)

    n_jobs = cfg.n_jobs
    sets = args.sets or list(SETS)
    conditions = args.conditions or list(CONDITIONS)
    ho = load_holdout(args.holdout)
    if args.rebuild_queries or not queries_current(ho, args.val_dir, sets):
        lg.info("building query tables in %s", args.val_dir)
        build_queries(ho, args.train_dir, cfg, raw=args.raw, test_path=args.test,
                      out_dir=args.val_dir)
    lib, rc = _library(args, cfg, lg)
    if rc:
        return rc

    common = dict(sets=sets, conditions=conditions, val_dir=args.val_dir, library=lib,
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

#!/usr/bin/env python
"""Write a submission: test parquet -> submissions/YYYYMMDD_<tag>.csv (WP4).

Thin script (scripts/README.md): arguments, logging, then ``casmi.pipeline.run``.

    python scripts/predict.py --config configs/b0_npdb.yaml
    python scripts/predict.py --test data/raw/test.parquet --config configs/b0b_pubchem.yaml \\
        --out submissions/20261118_b0b_pubchem_prior.csv

The tag is ``--tag``, else ``submission_tag`` from the config, else the config file stem.
The file is validated before it is written (``io.write_submission``) and re-read and
validated again (``io.validate_submission``) before success is reported. Logs go to
``logs/predict_<tag>_<timestamp>.log`` (arguments, git commit, run statistics, sha256 of
the file, wall time, peak RSS); the per-molecule report goes next to it as
``logs/<out stem>_molecules.csv``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import re
import sys
import time
from pathlib import Path


def _parse(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--test", type=Path, default=None,
                    help="test-shaped parquet (default: casmi.paths.TEST_PARQUET)")
    ap.add_argument("--config", type=Path, default=None,
                    help="YAML config in configs/ (default: Config defaults)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output CSV (default: submissions/YYYYMMDD_<tag>.csv, today's date)")
    ap.add_argument("--tag", default=None,
                    help="file tag (default: submission_tag from the config, else its stem)")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    from casmi import io, log, paths, pipeline
    from casmi.config import load_config

    cfg = load_config(args.config)
    tag = args.tag or cfg.submission_tag or (args.config.stem if args.config else "default")
    lg = log.get_logger(f"predict_{tag}")
    log.log_run(lg, args)
    if not re.fullmatch(r"(b\d+[a-z]?|r\d+|final)_[a-z0-9_]+", tag):
        lg.warning("tag %r does not follow PLAN 7 (lower snake case starting with the stage)", tag)
    lg.info("config: %s | overrides: %s", args.config, cfg.overrides())
    pipeline.limit_threads(cfg.n_jobs)
    out = args.out or paths.SUBMISSIONS / f"{_dt.date.today():%Y%m%d}_{tag}.csv"

    t0 = time.perf_counter()
    test_df = io.read_test(args.test)
    res = pipeline.run(test_df, cfg)
    io.write_submission(res.preds, out, test_df=test_df, max_guesses=cfg.top_k)
    report = io.validate_submission(out, test_df=test_df, max_guesses=cfg.top_k)
    wall = time.perf_counter() - t0

    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    mol_csv = paths.ensure_dir(paths.LOGS) / f"{out.stem}_molecules.csv"
    res.molecules.to_csv(mol_csv, index=False, lineterminator="\n")
    lg.info("per-molecule report: %s", mol_csv)
    lg.info("validated %s: %d rows, %d guesses, max %d per molecule, %d unparseable, "
            "warnings %s", out, report["n_rows"], report["n_guesses"], report["max_guesses"],
            report["n_unparseable"], report["warnings"])
    lg.info("sha256 %s  %s", sha, out.name)
    lg.info("runtime: pipeline %.1f s, total %.1f s | peak RSS %.0f MB (largest child %.0f MB)",
            res.wall_s, wall, log.peak_rss_mb(), log.peak_rss_mb(children=True))
    lg.info("SUCCESS: %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

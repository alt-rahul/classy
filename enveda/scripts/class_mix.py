"""Class shares from probe submissions (ROADMAP 2.6; package R1).

Reads a probe table (``--probes``, default configs/probe_table.csv; columns probe, tag,
config, board, role, notes): one row per probe submission, ``board`` its public score (empty
until submitted), ``tag`` the local evaluation of the same config
(``<val-dir>/results_v2_<tag>.parquet`` from ``scripts/evaluate.py --tag <tag>``), ``role``
"calibration" for P2 (b0, fixes w_obs) and "share" otherwise. Solves
board_p = sum_c pi_c x MRR_c(probe p) for the shares pi (C1, C2-NP, C2-PC, C3h) by
non-negative least squares with sum pi = 1 as a soft row, re-fitting w_obs's alpha to the
calibration probe at each round (``casmi.validation.class_mix_fit``), and reports the shares
with 90 percent intervals (board standard error ``Config.val_v2_board_se`` per probe and a
molecule bootstrap of the local MRRs) and which mixes of ``Config.val_v2_mixes`` they leave.

``--design A`` instead simulates the board scores of every probe in the table that has a local
evaluation under mix A and reports how precisely the planned probes would recover the shares
(no submission needed).

Usage:
    python scripts/class_mix.py                              # fit the submitted probes
    python scripts/class_mix.py --design A --n-sim 300       # precision of the probe design
Nothing here submits anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from casmi import paths
from casmi.config import load_config
from casmi.log import get_logger, log_run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probes", type=Path, default=paths.CONFIGS / "probe_table.csv")
    ap.add_argument("--val-dir", type=Path, default=paths.PROCESSED / "validation")
    ap.add_argument("--holdout-v2", type=Path, default=paths.SPLITS / "holdout_v2.csv")
    ap.add_argument("--panel", choices=["fast", "full"], default="full",
                    help="panel of the local probe evaluations")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--design", default=None, metavar="MIX",
                    help="simulate the probes under this mix instead of fitting board scores")
    ap.add_argument("--n-sim", type=int, default=300)
    ap.add_argument("--out", type=Path, default=None, help="JSON output (default <val-dir>/class_mix.json)")
    args = ap.parse_args(argv)
    lg = get_logger("class_mix")
    log_run(lg, args)
    cfg = load_config(args.config)

    import pandas as pd

    from casmi.splits import load_holdout_v2
    from casmi.validation import MIX_CELLS, class_mix_design, class_mix_fit, v2_attributes

    pr = pd.read_csv(args.probes, dtype=str, keep_default_na=False)
    pr["board"] = pd.to_numeric(pr["board"].replace("", None), errors="coerce")
    attrs = v2_attributes(load_holdout_v2(args.holdout_v2))
    tables = {}
    for t in dict.fromkeys(pr["tag"]):
        f = args.val_dir / f"results_v2_{t}.parquet"
        if f.exists():
            tables[t] = pd.read_parquet(f)
        else:
            lg.warning("no local evaluation for %s (%s): run scripts/evaluate.py --tag %s", t, f.name, t)
    have = pr[pr["tag"].isin(tables)]
    if not len(have):
        lg.error("no probe has a local evaluation")
        return 2
    if args.design:
        res = class_mix_design(have, tables, attrs, cfg, args.design, panel=args.panel, n_sim=args.n_sim)
        lg.info("design under mix %s %s: recovered mean %s, sd %s, rmse %s", args.design,
                [round(x, 3) for x in res["pi_true"]], [round(x, 3) for x in res["mean"]],
                [round(x, 3) for x in res["sd"]], [round(x, 3) for x in res["rmse"]])
        for t, v in res["local_cells"].items():
            lg.info("  local cells %-20s %s", t, dict(zip(MIX_CELLS, [round(x, 4) for x in v])))
    else:
        res = class_mix_fit(have, tables, attrs, cfg, panel=args.panel, n_boot=args.boot)
        if "pi" not in res:
            lg.warning("%s", res.get("note"))
        else:
            ci = res.get("pi_ci") or [[float("nan")] * 2] * len(MIX_CELLS)
            for c, v, (lo, hi) in zip(MIX_CELLS, res["pi"], ci):
                lg.info("share %-6s %.3f [%.3f, %.3f]", c, v, lo, hi)
            lg.info("sum %.3f, alpha %.3f after %d rounds; mixes consistent: %s", res["sum_pi"],
                    res["alpha"], res["rounds"], res.get("mix_consistent"))
            for p in res["probes"]:
                lg.info("  %-4s board %s predicted %.4f", p["probe"],
                        "-" if p["board"] is None else f"{p['board']:.4f}", p["predicted"])
    out = args.out or args.val_dir / ("class_mix_design.json" if args.design else "class_mix.json")
    out.write_text(json.dumps(res, indent=2, default=float))
    lg.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

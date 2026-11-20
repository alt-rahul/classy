#!/usr/bin/env python3
"""R3 acceptance check: prior-only MRR@25 of the PubChem documentation priors against the
pmid-only prior on the class-2 proxy cells (docs/ROADMAP.md R3 and 2.2).

Spectrum-free: for every holdout molecule (one per mkey, ``splits/holdout_v1.csv``) the pool
is the all-tier lookup (NP + PubChem, NP rows win) at ``--tol-ppm`` around the primary ik14's
monoisotopic mass, and candidates are ordered by a prior alone. A candidate is correct when its
ik14 is one of the molecule's ik14 (the mkey group). Ties are scored by their expectation under
a uniformly random order (no tie-breaker can favour a prior).

Cells: with ``splits/holdout_v2.csv`` (R1, default when present) the binding ROADMAP 2.2 cells,
``c2_cell`` NP (weighted by its calibrated ``w_obs``, and unweighted) and PC, each also on its
``pmid_count`` = 0 subset. Always also the v1 approximations: C2-NP = NP-TIMS + NP-OTHER, with
w = (1 + pmid_count)^-alpha for alpha in {0.5, 1.0}; C2-PC = SYN-TIMS whose truth is not in the
NP tier and has no spectra in another train library. Paired bootstrap 90 percent intervals
(molecule resampling) for each prior minus the pmid-only prior. ``--reuse`` rescores the
per-molecule results of an earlier run (same molecules) against another holdout file.

Priors (declared before looking at any result, no tuning here):
    pmid      log1p(pmid_count)
    b0        Baseline 0b ordering: NP-tier rows first, then fusion.prior_score
    sid       log1p(sid_count)
    sid_pmid  log1p(sid_count) + log1p(pmid_count)
    doc       public documentation prior: log1p(sid) + log1p(pmid) + 1.0 in_np
              + 0.5 log1p(n_citations) + 0.5 log1p(synonym_count)

Usage:
    nice -n 10 python scripts/eval_priors.py --out logs/r3_eval_priors.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from casmi import paths
from casmi.config import Config
from casmi.db import CandidateDB
from casmi.fusion import prior_score
from casmi.log import get_logger, log_run

TOP = 25
PRIORS = ("pmid", "b0", "sid", "sid_pmid", "doc")


def _c(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(dtype=np.float64)


def scores(df: pd.DataFrame, cfg: Config) -> dict[str, np.ndarray]:
    pmid, sid = np.log1p(_c(df, "pmid_count")), np.log1p(_c(df, "sid_count"))
    is_np = df["in_np"].to_numpy(dtype=bool)
    return {
        "pmid": pmid,
        "b0": prior_score(df, cfg) + 1e6 * is_np,
        "sid": sid,
        "sid_pmid": sid + pmid,
        "doc": sid + pmid + 1.0 * is_np + 0.5 * np.log1p(_c(df, "n_citations"))
        + 0.5 * np.log1p(_c(df, "synonym_count")),
    }


def expected_rr(score: np.ndarray, truth: np.ndarray) -> float:
    """E[1/rank] (0 beyond TOP) of the best truth candidate under random tie order."""
    if not truth.any():
        return 0.0
    st = score[truth].max()
    other = score[~truth]
    a = int((other > st).sum())
    t = int((other == st).sum()) + 1
    ks = a + np.arange(1, t + 1)
    return float(np.where(ks <= TOP, 1.0 / ks, 0.0).mean())


def molecules(holdout: Path) -> pd.DataFrame:
    h = pd.read_csv(holdout)
    h = h[h["set"].isin(["NP-TIMS", "NP-OTHER", "SYN-TIMS"])]
    prim = h[h["primary"]].drop_duplicates("mkey").set_index("mkey")
    truth = h.groupby("mkey")["ik14"].apply(lambda s: sorted(set(s)))
    extra = [c for c in ("c2_cell", "w_obs", "pmid_count") if c in prim.columns]
    m = prim[["ik14", "set", "eligible_c1", "mono_mass", *extra]].join(truth.rename("truth"))
    m = m.rename(columns={"pmid_count": "v2_pmid"})
    return m.reset_index().sort_values(["mono_mass", "mkey"], kind="mergesort").reset_index(drop=True)


def evaluate(mols: pd.DataFrame, db: CandidateDB, cfg: Config, tol: float, batch: int, log) -> pd.DataFrame:
    rows = []
    t0 = time.time()
    for s in range(0, len(mols), batch):
        mb = mols.iloc[s:s + batch]
        res = db.lookup_batch(mb["mono_mass"].to_numpy(), tol_ppm=tol, tiers=("np", "pubchem"),
                              keys=list(range(s, s + len(mb))))
        q = res["query"].to_numpy()
        sc = scores(res, cfg)
        ik = res["ik14"].astype(object).to_numpy()
        cut = np.flatnonzero(np.diff(q)) + 1
        starts, ends = np.r_[0, cut], np.r_[cut, len(q)]
        span = {int(q[a]): (a, b) for a, b in zip(starts, ends)} if len(q) else {}
        for i, m in zip(range(s, s + len(mb)), mb.itertuples()):
            a, b = span.get(i, (0, 0))
            tr = np.isin(ik[a:b], m.truth)
            r = {"mkey": m.mkey, "set": m.set, "eligible_c1": bool(m.eligible_c1), "pool": b - a,
                 "truth_in_pool": bool(tr.any())}
            for c in ("c2_cell", "w_obs", "v2_pmid"):
                if c in mols.columns:
                    r[c] = getattr(m, c)
            if tr.any():
                sub = res.iloc[a:b][tr]
                r["truth_in_np"] = bool(sub["in_np"].any())
                r["truth_pmid"] = float(pd.to_numeric(sub["pmid_count"], errors="coerce").fillna(0).max())
                r["truth_sid"] = float(pd.to_numeric(sub["sid_count"], errors="coerce").fillna(0).max())
            else:
                r.update({"truth_in_np": False, "truth_pmid": 0.0, "truth_sid": 0.0})
            for p in PRIORS:
                r[p] = expected_rr(sc[p][a:b], tr)
            rows.append(r)
        db.release()
        if (s // batch) % 10 == 0:
            log.info("  %d / %d molecules, %.0f s", s + len(mb), len(mols), time.time() - t0)
    return pd.DataFrame(rows)


def _cell(d: pd.DataFrame, w: np.ndarray, rng, n_boot: int) -> dict:
    e = {"n": int(len(d)), "truth_in_pool": round(float(d["truth_in_pool"].mean()), 4),
         "median_pool": int(d["pool"].median())}
    for p in PRIORS:
        e[p] = round(float(np.average(d[p], weights=w)), 4)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    ww = w[idx]
    base = (d["pmid"].to_numpy()[idx] * ww).sum(1) / ww.sum(1)
    for p in PRIORS[1:]:
        diff = (d[p].to_numpy()[idx] * ww).sum(1) / ww.sum(1) - base
        e[f"{p}-pmid 90% CI"] = [round(float(np.quantile(diff, 0.05)), 4), round(float(np.quantile(diff, 0.95)), 4)]
    return e


def summarise(df: pd.DataFrame, seed: int, n_boot: int = 2000) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    if "c2_cell" in df.columns:  # holdout_v2: the binding cells and the calibrated w_obs
        for cell in ("NP", "PC"):
            for sub_name in ("all", "pmid=0"):
                mask = df["c2_cell"] == cell
                if sub_name == "pmid=0":
                    mask &= df["v2_pmid"].fillna(0) == 0
                d = df[mask]
                if not len(d):
                    continue
                for weighted in ((True, False) if cell == "NP" else (False,)):
                    w = d["w_obs"].to_numpy(dtype=np.float64) if weighted else np.ones(len(d))
                    out[f"v2 C2-{cell} | {sub_name}" + (" | w_obs" if weighted else "")] = _cell(d, w, rng, n_boot)
    cells = {
        "C2-NP": df["set"].isin(["NP-TIMS", "NP-OTHER"]),
        "C2-NP NP-TIMS": df["set"] == "NP-TIMS",
        "C2-PC": (df["set"] == "SYN-TIMS") & ~df["truth_in_np"] & ~df["eligible_c1"],
        "SYN-TIMS all": df["set"] == "SYN-TIMS",
    }
    for name, mask in cells.items():
        for sub_name, sub in (("all", mask), ("pmid=0", mask & (df["truth_pmid"] == 0))):
            d = df[sub]
            if not len(d):
                continue
            alphas = (0.0, 0.5, 1.0) if name.startswith("C2-NP") else (0.0,)
            for alpha in alphas:
                w = (1.0 + d["truth_pmid"].to_numpy()) ** (-alpha)
                key = f"{name} | {sub_name}" + (f" | w_obs alpha {alpha:g}" if alpha else "")
                out[key] = _cell(d, w, rng, n_boot)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    v2 = paths.SPLITS / "holdout_v2.csv"
    ap.add_argument("--holdout", type=Path, default=v2 if v2.exists() else paths.SPLITS / "holdout_v1.csv")
    ap.add_argument("--reuse", type=Path, default=None,
                    help="per-molecule results parquet of an earlier run: rescore it, no lookups")
    ap.add_argument("--tol-ppm", type=float, default=10.0)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--max-molecules", type=int, default=None)
    ap.add_argument("--out", type=Path, default=paths.LOGS / "r3_eval_priors.json")
    args = ap.parse_args(argv)
    log = get_logger("eval_priors")
    log_run(log, args)
    cfg = Config()
    mols = molecules(args.holdout)
    if args.max_molecules:
        mols = mols.iloc[:: max(1, len(mols) // args.max_molecules)].reset_index(drop=True)
    log.info("%d molecules: %s", len(mols), mols["set"].value_counts().to_dict())
    t0 = time.time()
    if args.reuse:
        df = pd.read_parquet(args.reuse).drop(columns=["c2_cell", "w_obs", "v2_pmid"], errors="ignore")
        keep = [c for c in ("c2_cell", "w_obs", "v2_pmid") if c in mols.columns]
        df = df.merge(mols[["mkey", *keep]], on="mkey", how="inner", validate="one_to_one")
        log.info("rescoring %d stored molecules from %s against %s", len(df), args.reuse, args.holdout)
    else:
        df = evaluate(mols, CandidateDB(cfg=cfg), cfg, args.tol_ppm, args.batch, log)
    summ = summarise(df, cfg.seed)
    for k, v in summ.items():
        log.info("%-40s %s", k, v)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out.with_suffix(".parquet"), index=False)
    args.out.write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                    "seconds": round(time.time() - t0, 1), "cells": summ}, indent=1))
    log.info("wrote %s (%.0f s)", args.out, time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())

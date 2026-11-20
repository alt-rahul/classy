"""Isomer separation by MetFrag-lite fragment features (ROADMAP R7 evaluation).

For every molecule of the chosen holdout sets under one validation condition (default C2: all
library spectra of the molecule hidden; the database pool is complete), the pipeline's
``mass``, ``pool``, ``features`` and ``score`` stages run through the validation harness
(``validation.ExcludingDB`` with the cell's exclusions), giving the current prior-only ranking
over the configured tiers (default NP + PubChem). The top ``frag_top_k`` candidates by that
score, plus every truth row, get the fragment features (``casmi.fragments.score_pool``).

Outputs (``--out``, default data/processed/fragments/):
- ``eval_<tag>_<set>_<cond>.parquet``: one row per scored candidate (molecule_id, ik14, formula,
  tier, score, in_topk, is_true, same_formula, fold, pubmed0, base_rank of the molecule,
  next_score = score of the (K+1)-th row, and every ``fragments.FEATURES`` column);
- ``eval_<tag>_summary.json`` and the printed tables:
  (1) AUC of each feature, truth versus same-formula decoys among the scored rows, mean over
      molecules with at least one such decoy (ties count one half); also versus all decoys;
  (2) MRR@25 (ik14 de-duplication) of the prior ranking and of prior + w x feature for the top
      K rows, w tuned on one half of the molecules (hash split, both sets pooled) and tested
      on the other half, both ways; rank-1 wins and losses;
  (3) formula-level comparison: sub-formula explained intensity of the true formula
      (``formula.explain``) is the same for every same-formula candidate (AUC 0.5 by
      construction); its mean is printed next to the structure-level explained intensity.

Re-runnable on other cells: ``--holdout`` / ``--val-dir`` / ``--condition`` pass through to
``casmi.validation`` (v2 cells from R1 once they exist).

Usage:
    nice -n 10 python scripts/eval_fragments.py --sets NP-TIMS NP-OTHER --tag v1
    python scripts/eval_fragments.py --sets NP-TIMS --max-molecules 30 --tag smoke
    python scripts/eval_fragments.py --analyse-only --tag v1      # re-read stored tables
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from casmi import paths
from casmi.config import load_config
from casmi.log import get_logger, log_run, peak_rss_mb

W_GRID = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5)
BONUS_FEATURES = ("frag_score_z", "frag_score_fz", "frag_expl_lin_z", "frag_expl_sqrt_z")
PRIMARY = "frag_score_z"          # fixed before looking at the results
LOWER_IS_BETTER = {"frag_mean_cuts", "frag_err_mda", "frag_log_n"}


# --------------------------------------------------------------------------------------
# building the scored tables
# --------------------------------------------------------------------------------------

def _first_truth_rank(ik: np.ndarray, is_true: np.ndarray) -> int:
    """1-based rank of the first truth row after ik14 de-duplication (0 if absent)."""
    seen: set = set()
    r = 0
    for k, t in zip(ik, is_true):
        if k in seen:
            continue
        seen.add(k)
        r += 1
        if t:
            return r
    return 0


def build_tables(args, cfg, lg) -> dict[str, pd.DataFrame]:
    from casmi import db as cdb
    from casmi import fragments as F
    from casmi import fusion, pipeline, validation
    from casmi.formula import explain, prepare_peaks, spectra_from_frame

    base_db = cdb.get_db(cfg)
    F.warmup()
    cache = F.FragmentCache(cfg)
    out: dict[str, pd.DataFrame] = {}
    for set_name in args.sets:
        t0 = time.perf_counter()
        q, t = validation.load_queries(set_name, args.val_dir)
        if args.max_molecules:
            keep = sorted(t["molecule_id"].astype(str))[:args.max_molecules]
            t = t[t["molecule_id"].astype(str).isin(keep)].reset_index(drop=True)
            q = q[q["molecule_id"].astype(str).isin(keep)].reset_index(drop=True)
        keys = validation.truth_keys(t, cfg, db=base_db,
                                     cache_path=Path(args.val_dir) / "truth_keys.parquet")
        excl = validation.condition_exclusions(t, args.condition, keys)
        spec2mol = dict(zip(q["spectrum_id"].astype(str), q["molecule_id"].astype(str)))
        hooks = validation.ValidationHooks(args.condition, excl, spec2mol, cfg.mass_tol_ppm)
        proxy = validation.ExcludingDB(base_db, hooks)
        tr = t.set_index(t["molecule_id"].astype(str))
        order = sorted(tr.index, key=lambda m: (float(tr.at[m, "mono_mass"]), m))
        qpos = q.groupby(q["molecule_id"].astype(str)).indices
        parts = []
        k = int(cfg.frag_top_k)
        for a in range(0, len(order), args.batch):
            chunk = order[a:a + args.batch]
            rows = np.sort(np.concatenate([qpos[m] for m in chunk if m in qpos]))
            sub = q.iloc[rows].reset_index(drop=True)
            hooks.active = tuple(sorted(chunk))
            st = pipeline.State(test_df=sub, db=proxy, extras={"validation": hooks})
            for fn in (pipeline.stage_mass, pipeline.stage_pool, pipeline.stage_features,
                       pipeline.stage_score):
                st = fn(st, cfg)
            pool = st.pool
            hooks.compact(chunk, keys)
            if pool is None or not len(pool):
                continue
            pool = fusion.sort_pool(pool, cfg=cfg).reset_index(drop=True)
            mids = pool["molecule_id"].astype(str).to_numpy(dtype=object)
            ik = pool["ik14"].astype(str).to_numpy(dtype=object)
            is_true = np.fromiter((x in keys.get(m, frozenset()) for m, x in zip(mids, ik)),
                                  bool, len(pool))
            pos = pool.groupby(mids, sort=False).cumcount().to_numpy()
            topk = pos < k
            sel = topk | is_true
            feats, _ = F.score_pool(pool, sub, cfg, rows=sel, cache=cache, n_jobs=1)
            df = pool.loc[sel, ["molecule_id", "ik14", "formula", "tier", "in_np", "score",
                                "smiles"]].copy()
            df["pos"] = pos[sel]
            df["in_topk"] = topk[sel]
            df["is_true"] = is_true[sel]
            for c in F.FEATURES:
                df[c] = feats.loc[sel, c].to_numpy(dtype=np.float64)
            # per-molecule context
            mol_rows = []
            for m, idx in pd.Series(np.arange(len(pool))).groupby(mids, sort=True):
                ii = idx.to_numpy()
                br = _first_truth_rank(ik[ii], is_true[ii])
                nxt = float(pool["score"].iat[ii[k]]) if len(ii) > k else -np.inf
                tf = str(tr.at[m, "formula"])
                sp = sub[sub["molecule_id"].astype(str) == m]
                try:
                    ef, _, _ = explain([tf], prepare_peaks(spectra_from_frame(sp), cfg,
                                                           tol_da=cfg.frag_tol_da))
                    fexpl = float(ef[0])
                except Exception:  # noqa: BLE001 - comparison column only
                    fexpl = np.nan
                mol_rows.append({"molecule_id": m, "base_rank": br, "next_score": nxt,
                                 "n_pool": len(ii), "true_formula": tf,
                                 "formula_expl": fexpl})
            mdf = pd.DataFrame(mol_rows)
            df = df.merge(mdf, on="molecule_id", how="left")
            parts.append(df)
            if (a // args.batch) % 10 == 0:
                lg.info("%s: %d/%d molecules, %.0f s, peak RSS %.0f MB, cache hits %d disk %d "
                        "misses %d", set_name, min(a + args.batch, len(order)), len(order),
                        time.perf_counter() - t0, peak_rss_mb(), cache.hits, cache.disk_hits,
                        cache.misses)
            try:
                base_db.release()
            except Exception:  # noqa: BLE001
                pass
        if not parts:
            continue
        tab = pd.concat(parts, ignore_index=True)
        tab["set"] = set_name
        tab["fold"] = tab["molecule_id"].map(tr["fold"]).astype(int)
        pc = tab["molecule_id"].map(tr["pubmed_count"])
        tab["pubmed0"] = pc.isna() | (pd.to_numeric(pc, errors="coerce").fillna(0) == 0)
        tab["same_formula"] = tab["formula"].astype(str) == tab["true_formula"].astype(str)
        out[set_name] = tab
        if not args.no_save:
            p = Path(args.out) / f"eval_{args.tag}_{validation.set_slug(set_name)}_{args.condition}.parquet"
            tab.to_parquet(p, index=False)
            lg.info("wrote %s (%d rows)", p, len(tab))
        lg.info("%s: %d molecules in %.0f s", set_name, tab["molecule_id"].nunique(),
                time.perf_counter() - t0)
    cache.close()
    return out


# --------------------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------------------

def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC, ties one half."""
    if not len(pos) or not len(neg):
        return np.nan
    allv = np.concatenate([pos, neg])
    r = pd.Series(allv).rank(method="average").to_numpy()
    rp = r[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auc_table(tab: pd.DataFrame, features, same_formula: bool = True) -> pd.DataFrame:
    """Mean per-molecule AUC of each feature (higher value = predicted truth)."""
    rows = []
    sub = tab[tab["same_formula"]] if same_formula else tab
    groups = [(m, g) for m, g in sub.groupby("molecule_id", sort=True)
              if g["is_true"].any() and (~g["is_true"]).any()]
    for f in features:
        vals = []
        for _, g in groups:
            x = g[f].to_numpy(dtype=np.float64)
            if f.endswith("_rank") or f.endswith("_frank"):
                x = -x
            ok = np.isfinite(x)
            y = g["is_true"].to_numpy()
            vals.append(_auc(x[ok & y], x[ok & ~y]))
        v = np.array(vals, dtype=np.float64)
        rows.append({"feature": f, "auc": float(np.nanmean(v)) if len(v) else np.nan,
                     "n_molecules": int(np.isfinite(v).sum())})
    return pd.DataFrame(rows)


def _ranks_with_bonus(tab: pd.DataFrame, feature: str, w: float, k: int) -> pd.Series:
    """Rank of the truth (0 = not found / beyond the scored rows) per molecule after adding
    w x feature (NaN = 0) to the score of the top-K rows."""
    out = {}
    for m, g in tab.groupby("molecule_id", sort=True):
        base = int(g["base_rank"].iat[0])
        if not g["is_true"].any() or base == 0:
            out[m] = 0
            continue
        top = g[g["in_topk"]]
        if not top["is_true"].any():
            out[m] = base           # truth outside the top K: untouched (beyond rank K)
            continue
        b = np.nan_to_num(top[feature].to_numpy(dtype=np.float64), nan=0.0)
        s = top["score"].to_numpy(dtype=np.float64) + w * b
        order = np.lexsort((top["pos"].to_numpy(), -s))
        ik = top["ik14"].to_numpy(dtype=object)[order]
        tr = top["is_true"].to_numpy()[order]
        r = _first_truth_rank(ik, tr)
        st = s[order][np.flatnonzero(tr)[0]]
        nxt = float(g["next_score"].iat[0])
        if st < nxt:                 # pushed below the first unscored row
            r = max(r, k + 1)
        out[m] = r
    return pd.Series(out)


def _mrr(ranks: pd.Series, top_k: int = 25) -> float:
    r = ranks.to_numpy(dtype=np.float64)
    rr = np.where((r > 0) & (r <= top_k), 1.0 / np.maximum(r, 1), 0.0)
    return float(rr.mean()) if len(rr) else np.nan


def half_of(mid: str, seed: int) -> int:
    return int(hashlib.sha1(f"{seed}|{mid}".encode()).hexdigest(), 16) % 2


def mrr_study(tab: pd.DataFrame, k: int, seed: int, features=BONUS_FEATURES) -> dict:
    mids = sorted(tab["molecule_id"].unique())
    half = {m: half_of(m, seed) for m in mids}
    tab = tab.assign(half=tab["molecule_id"].map(half))
    base = tab.groupby("molecule_id", sort=True)["base_rank"].first()
    sets = tab.groupby("molecule_id", sort=True)["set"].first()
    res: dict = {"n_molecules": len(mids), "features": {}}
    for f in features:
        grid = {w: _ranks_with_bonus(tab, f, w, k) for w in W_GRID}
        fr: dict = {"grid_mrr": {str(w): _mrr(r) for w, r in grid.items()}, "halves": []}
        test_ranks = []
        for h in (0, 1):
            train_ids = [m for m in mids if half[m] == h]
            test_ids = [m for m in mids if half[m] != h]
            best_w = max(W_GRID, key=lambda w: (round(_mrr(grid[w][train_ids]), 6), -w))
            rt = grid[best_w][test_ids]
            test_ranks.append(rt)
            b = base[test_ids]
            wins = int(((rt == 1) & (b != 1)).sum())
            losses = int(((rt != 1) & (b == 1)).sum())
            hr = {"tune_half": h, "w": best_w, "train_mrr_base": _mrr(base[train_ids]),
                  "train_mrr": _mrr(grid[best_w][train_ids]), "test_mrr_base": _mrr(b),
                  "test_mrr": _mrr(rt), "test_n": len(test_ids), "rank1_wins": wins,
                  "rank1_losses": losses}
            for s in sorted(sets.unique()):
                ids = [m for m in test_ids if sets[m] == s]
                hr[f"test_mrr_base_{s}"] = _mrr(base[ids])
                hr[f"test_mrr_{s}"] = _mrr(rt[ids])
                hr[f"wins_{s}"] = int(((rt[ids] == 1) & (base[ids] != 1)).sum())
                hr[f"losses_{s}"] = int(((rt[ids] != 1) & (base[ids] == 1)).sum())
                hr[f"n_{s}"] = len(ids)
            fr["halves"].append(hr)
        allr = pd.concat(test_ranks).sort_index()
        fr["cross_fit"] = {"mrr_base": _mrr(base), "mrr": _mrr(allr),
                           "rank1_wins": int(((allr == 1) & (base != 1)).sum()),
                           "rank1_losses": int(((allr != 1) & (base == 1)).sum())}
        for s in sorted(sets.unique()):
            ids = sets.index[sets == s]
            fr["cross_fit"][f"mrr_base_{s}"] = _mrr(base[ids])
            fr["cross_fit"][f"mrr_{s}"] = _mrr(allr[ids])
            fr["cross_fit"][f"wins_{s}"] = int(((allr[ids] == 1) & (base[ids] != 1)).sum())
            fr["cross_fit"][f"losses_{s}"] = int(((allr[ids] != 1) & (base[ids] == 1)).sum())
            fr["cross_fit"][f"n_{s}"] = int(len(ids))
        res["features"][f] = fr
    return res


def analyse(tabs: dict[str, pd.DataFrame], cfg, lg, seed: int) -> dict:
    from casmi import fragments as F

    feats = [f for f in F.FEATURES]
    allt = pd.concat(list(tabs.values()), ignore_index=True)
    summ: dict = {"auc_same_formula": {}, "auc_all": {}, "context": {}}
    groups = {s: t for s, t in tabs.items()}
    groups["all"] = allt
    for s, t in groups.items():
        a = auc_table(t, feats, True)
        b = auc_table(t, feats, False)
        p0 = auc_table(t[t["pubmed0"]], ["frag_score", "frag_score_z", "frag_expl_lin"], True)
        summ["auc_same_formula"][s] = dict(zip(a["feature"], a["auc"]))
        summ["auc_same_formula"][s]["_n"] = int(a["n_molecules"].max()) if len(a) else 0
        summ["auc_all"][s] = dict(zip(b["feature"], b["auc"]))
        summ["auc_all"][s]["_n"] = int(b["n_molecules"].max()) if len(b) else 0
        summ["auc_same_formula"][s]["_pubmed0"] = dict(zip(p0["feature"], p0["auc"]))
        summ["auc_same_formula"][s]["_pubmed0_n"] = int(p0["n_molecules"].max()) if len(p0) else 0
        mol = t.groupby("molecule_id").first()
        tru = t[t["is_true"]].groupby("molecule_id").first()
        dec = t[t["same_formula"] & ~t["is_true"]]
        summ["context"][s] = {
            "n_molecules": int(len(mol)),
            "truth_in_topk": float(t[t["in_topk"]].groupby("molecule_id")["is_true"].any()
                                   .reindex(mol.index, fill_value=False).mean()),
            "truth_in_pool": float((mol["base_rank"] > 0).mean()),
            "prior_top1": float((mol["base_rank"] == 1).mean()),
            "median_same_formula_decoys_in_topk": float(
                t[t["in_topk"] & t["same_formula"] & ~t["is_true"]].groupby("molecule_id").size()
                .reindex(mol.index, fill_value=0).median()),
            "formula_expl_mean": float(mol["formula_expl"].mean()),
            "truth_frag_expl_lin_mean": float(tru["frag_expl_lin"].mean()),
            "decoy_frag_expl_lin_mean": float(dec["frag_expl_lin"].mean()),
            "truth_frag_score_mean": float(tru["frag_score"].mean()),
            "decoy_frag_score_mean": float(dec["frag_score"].mean()),
        }
    k = int(cfg.frag_top_k)
    summ["mrr"] = mrr_study(allt, k, seed)
    # report
    lg.info("AUC truth vs same-formula decoys (mean over molecules):")
    hdr = list(groups)
    lg.info("  %-22s %s", "feature", " ".join(f"{h:>10}" for h in hdr))
    for f in feats:
        lg.info("  %-22s %s", f, " ".join(f"{summ['auc_same_formula'][h].get(f, np.nan):10.3f}"
                                          for h in hdr))
    lg.info("  %-22s %s", "n molecules", " ".join(f"{summ['auc_same_formula'][h]['_n']:10d}"
                                                  for h in hdr))
    for f, fr in summ["mrr"]["features"].items():
        cf = fr["cross_fit"]
        lg.info("MRR %s: cross-fit %.4f -> %.4f (wins %d, losses %d); halves w %s",
                f, cf["mrr_base"], cf["mrr"], cf["rank1_wins"], cf["rank1_losses"],
                [h["w"] for h in fr["halves"]])
        for s in sorted(tabs):
            lg.info("    %-9s %.4f -> %.4f (wins %d, losses %d, n %d)", s, cf[f"mrr_base_{s}"],
                    cf[f"mrr_{s}"], cf[f"wins_{s}"], cf[f"losses_{s}"], cf[f"n_{s}"])
    for s, c in summ["context"].items():
        lg.info("context %s: %s", s, {k2: round(v, 4) if isinstance(v, float) else v
                                      for k2, v in c.items()})
    return summ


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--sets", nargs="+", default=["NP-TIMS", "NP-OTHER"])
    ap.add_argument("--condition", default="C2")
    ap.add_argument("--holdout", type=Path, default=paths.SPLITS / "holdout_v1.csv",
                    help="recorded only; the query tables in --val-dir must match it")
    ap.add_argument("--val-dir", type=Path, default=paths.PROCESSED / "validation")
    ap.add_argument("--out", type=Path, default=paths.PROCESSED / "fragments")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--batch", type=int, default=20, help="molecules per pool lookup")
    ap.add_argument("--max-molecules", type=int, default=None)
    ap.add_argument("--tiers", nargs="+", default=None, help="override Config.db_tiers")
    ap.add_argument("--set-config", nargs="*", default=[],
                    help="extra Config overrides as key=value (YAML-parsed values)")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)
    lg = get_logger("eval_fragments")
    log_run(lg, args)
    cfg = load_config(args.config)
    if args.tiers:
        cfg = cfg.replace(db_tiers=tuple(args.tiers))
    if args.set_config:
        import yaml

        cfg = cfg.replace(**{kv.split("=", 1)[0]: yaml.safe_load(kv.split("=", 1)[1])
                             for kv in args.set_config})
    lg.info("config overrides: %s", cfg.overrides())
    paths.ensure_dir(args.out)
    from casmi.validation import set_slug

    if args.analyse_only:
        tabs = {s: pd.read_parquet(Path(args.out) /
                                   f"eval_{args.tag}_{set_slug(s)}_{args.condition}.parquet")
                for s in args.sets}
    else:
        tabs = build_tables(args, cfg, lg)
    summ = analyse(tabs, cfg, lg, cfg.seed)
    summ["settings"] = {"sets": args.sets, "condition": args.condition, "tag": args.tag,
                        "holdout": str(args.holdout), "overrides": cfg.overrides(),
                        "primary_feature": PRIMARY, "w_grid": list(W_GRID)}
    if not args.no_save:
        p = Path(args.out) / f"eval_{args.tag}_{args.condition}_summary.json"
        p.write_text(json.dumps(summ, indent=1, default=float), encoding="utf-8")
        lg.info("wrote %s", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())

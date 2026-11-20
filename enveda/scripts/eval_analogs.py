"""Analog evidence features (mass-shift propagation) on validation cells (ROADMAP R6 evaluation).

Stages (``--stage``, default all):

1. ``hits``: for every molecule of the chosen sets, the analog hit list of ``casmi.analogs``
   (whole library, same polarity, modified cosine on the open-index shortlist) under the
   cell's library exclusions (``ValidationHooks.library_mask``), raw and unselected (up to
   ``analog_feat_coarse_top`` structures). Leak check: under C2/C3 cells no hit may carry an
   ik14 of the molecule's masked group or the truth's mkey (AssertionError otherwise).
   Output ``hits_<tag>_<set>_<cell>.parquet``.
2. ``features``: candidate rows (``--rows``: ``fragments:<tag>`` = the R7 tables
   ``data/processed/fragments/eval_<tag>_<set>_<cell>.parquet``, i.e. the top 100 of the
   prior-only ranking plus every truth row with fragment features; ``pipeline`` = build the
   same rows here with the mass / pool / features / score stages through
   ``validation.ExcludingDB``), candidate fingerprints (NP rows from ``np_fp.npy``, others
   from SMILES; cached as ``cfp_<tag>_<set>_<cell>.npz``) and every ``analogs.FEATURES`` column
   (structure features on all rows). Output ``eval_<tag>_<set>_<cell>.parquet``.
3. ``analyse``: (a) AUC of each feature, truth versus same-formula decoys and versus all
   decoys, mean over molecules (ties one half), per set and for the NP-OTHER pubmed_count = 0
   subset; (b) MRR@25 (ik14 de-duplication) of the prior ranking and of prior + w x feature on
   the top-100 rows, w tuned on one hash half of the molecules and tested on the other (both
   ways), for NP-TIMS + NP-OTHER pooled, within NP-TIMS, and within SYN-TIMS; (c) jointly
   with R7's ``frag_score_z`` (prior + w1 x frag + w2 x analog, both tuned the same way);
   (d) feature variants (``--variants``: hit selection settings, re-scored from the stored
   hits and fingerprints, fingerprint features only); (e) when ``splits/holdout_v2.csv``
   exists, the same MRRs weighted by ``w_obs`` on the v2 cells C2-NP and C2-PC.
   Output ``eval_<tag>_<cell>_summary.json`` and the printed tables.

Cells: ``--cell`` takes a v1 condition (C1, C2, C3) or, once R1's v2 protocol is in
``casmi.validation`` (``exclusions_v2``), a v2 cell (C1, C2-NP, C2-PC, C3h, C3e); C3h then also
hides the near neighbours of the truth (they are in the masked group). The DB side of a
cell does not matter for stage 1; with ``--rows pipeline`` it goes through ExcludingDB.

Usage:
    nice -n 10 python scripts/eval_analogs.py --stage hits --sets NP-TIMS NP-OTHER SYN-TIMS
    nice -n 10 python scripts/eval_analogs.py --stage features --sets NP-OTHER --n-jobs 2
    python scripts/eval_analogs.py --stage analyse --sets NP-TIMS NP-OTHER SYN-TIMS
    python scripts/eval_analogs.py --sets NP-TIMS --max-molecules 30 --tag smoke
    python scripts/eval_analogs.py --cell C2-NP --rows pipeline --tag v2   # once R1 lands
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
PRIMARY = "an_tan_wmean_z"          # fixed before looking at the results
BONUS_FEATURES = ("an_tan_wmean_z", "an_tan_wmax_z", "an_tan_max_z", "an_tan_best_z",
                  "an_sub_w_z", "an_scaf_w_z", "an_edit_w_z")
FRAG = "frag_score_z"
LOWER_IS_BETTER: set[str] = set()
DEFAULT_VARIANTS = (
    "analog_feat_power=1", "analog_feat_power=2", "analog_feat_power=3", "analog_feat_power=6",
    "analog_feat_top_k=20", "analog_feat_top_k=100",
    "analog_feat_min_matched=1", "analog_feat_min_matched=5",
    "analog_feat_max_dm_da=200",
)
ID_COLUMNS = ("molecule_id", "ik14", "formula", "tier", "in_np", "score", "smiles", "pos",
              "in_topk", "is_true", "base_rank", "next_score", "n_pool", "true_formula", "set",
              "fold", "pubmed0", "same_formula")


def _slug(s: str) -> str:
    from casmi.validation import set_slug

    return set_slug(s)


def _path(args, kind: str, set_name: str, tag: str | None = None, ext: str = "parquet") -> Path:
    return Path(args.out) / f"{kind}_{tag or args.tag}_{_slug(set_name)}_{args.cell}.{ext}"


# --------------------------------------------------------------------------------------
# cells and exclusions
# --------------------------------------------------------------------------------------

def _load_set(args, set_name):
    from casmi import validation

    q, t = validation.load_queries(set_name, args.val_dir)
    if args.max_molecules:
        keep = sorted(t["molecule_id"].astype(str))[:args.max_molecules]
        t = t[t["molecule_id"].astype(str).isin(keep)].reset_index(drop=True)
        q = q[q["molecule_id"].astype(str).isin(keep)].reset_index(drop=True)
    return q, t


def _hooks(args, cfg, q, t):
    """ValidationHooks of the cell (v1 condition or v2 cell)."""
    from casmi import validation

    cell = args.cell
    keys: dict = {}
    if cell in getattr(validation, "CONDITIONS", ()):
        cond = cell
        excl = validation.condition_exclusions(t, cond, keys)
    else:
        if not hasattr(validation, "exclusions_v2"):
            raise SystemExit(f"cell {cell!r} needs the v2 protocol (validation.exclusions_v2)")
        cond = validation.CELL_CONDITION[cell]
        near = None
        if cell == "C3h":
            near = validation.c3h_neighbours(t, cfg, cache_path=Path(args.val_dir) /
                                             "c3h_neighbours.parquet")
        excl = validation.exclusions_v2(t, cell, keys, near)
    spec2mol = dict(zip(q["spectrum_id"].astype(str), q["molecule_id"].astype(str)))
    return validation.ValidationHooks(cond, excl, spec2mol, cfg.mass_tol_ppm), cond


def _lib_mkeys(lib) -> dict[str, str]:
    p = paths.PROCESSED / "train" / "structures.parquet"
    if not p.exists():
        return {}
    st = pd.read_parquet(p, columns=["ik14", "mkey"]).dropna()
    want = set(map(str, lib.struct_ik14))
    st = st[st["ik14"].astype(str).isin(want)]
    return dict(zip(st["ik14"].astype(str), st["mkey"].astype(str)))


# --------------------------------------------------------------------------------------
# stage 1: hits
# --------------------------------------------------------------------------------------

def _row_molecules(args, set_name) -> set[str] | None:
    if args.rows.startswith("fragments:"):
        p = _path(args, "eval", set_name, tag=args.rows.split(":", 1)[1])
        p = paths.PROCESSED / "fragments" / p.name
        if p.exists():
            return set(pd.read_parquet(p, columns=["molecule_id"])["molecule_id"].astype(str))
    return None


def stage_hits(args, cfg, lg) -> None:
    from casmi import analogs as A
    from casmi.io import mass_tables
    from casmi.library import Library, prepare_queries

    lib = Library.load(cfg=cfg)
    lib_mk = _lib_mkeys(lib)
    lg.info("library: %d spectra, %d structures; %d mkeys; peak RSS %.0f MB", lib.n_spectra,
            lib.n_structures, len(lib_mk), peak_rss_mb())
    for set_name in args.sets:
        t0 = time.perf_counter()
        q, t = _load_set(args, set_name)
        want = _row_molecules(args, set_name)
        if want is not None:
            t = t[t["molecule_id"].astype(str).isin(want)].reset_index(drop=True)
            q = q[q["molecule_id"].astype(str).isin(want)].reset_index(drop=True)
        hooks, cond = _hooks(args, cfg, q, t)
        mols = mass_tables(q, cfg)[0]
        M_of = dict(zip(mols["molecule_id"].astype(str), mols["M"].astype(float)))
        tr = t.set_index(t["molecule_id"].astype(str))
        qpos = q.groupby(q["molecule_id"].astype(str)).indices
        parts, n_leak_checked = [], 0
        ids = sorted(tr.index)
        for i, m in enumerate(ids):
            sub = q.iloc[np.sort(qpos[m])].reset_index(drop=True)
            M = float(M_of.get(m, np.nan))
            qs = prepare_queries(sub, lib.cfg, M=M if np.isfinite(M) else None)
            mask = hooks.library_mask(lib, m)
            h = A.search_hits(lib, qs, cfg, M=M, exclude_mask=mask)
            if cond in ("C2", "C3") and len(h):
                e = hooks.excl[m]
                bad = set(h["ik14"].astype(str)) & set(e.group)
                tm = str(tr.at[m, "mkey"])
                bad |= {k for k in h["ik14"].astype(str) if lib_mk.get(k) == tm}
                assert not bad, f"{m}: analog hits {sorted(bad)[:3]} of the masked molecule"
                n_leak_checked += 1
            parts.append(h.assign(molecule_id=m, M=M))
            if (i + 1) % 250 == 0:
                lg.info("%s: %d/%d molecules, %.0f s, peak RSS %.0f MB", set_name, i + 1, len(ids),
                        time.perf_counter() - t0, peak_rss_mb())
        hits = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        p = _path(args, "hits", set_name)
        hits.to_parquet(p, index=False)
        lg.info("%s (%s): %d molecules, %d hits, leak-checked %d, %.0f s (%.3f s per molecule) "
                "-> %s", set_name, args.cell, len(ids), len(hits), n_leak_checked,
                time.perf_counter() - t0, (time.perf_counter() - t0) / max(len(ids), 1), p)


# --------------------------------------------------------------------------------------
# stage 2: features
# --------------------------------------------------------------------------------------

def _first_truth_rank(ik: np.ndarray, is_true: np.ndarray) -> int:
    seen: set = set()
    r = 0
    for k, tt in zip(ik, is_true):
        if k in seen:
            continue
        seen.add(k)
        r += 1
        if tt:
            return r
    return 0


def _rows_pipeline(args, cfg, lg, set_name) -> pd.DataFrame:
    """Top-K rows of the prior-only ranking plus truth rows (as scripts/eval_fragments.py)."""
    from casmi import db as cdb
    from casmi import fusion, pipeline, validation

    base_db = cdb.get_db(cfg)
    q, t = _load_set(args, set_name)
    keys = validation.truth_keys(t, cfg, db=base_db,
                                 cache_path=Path(args.val_dir) / "truth_keys.parquet")
    hooks, _ = _hooks(args, cfg, q, t)
    hooks.excl = {m: e for m, e in hooks.excl.items()}
    proxy = validation.ExcludingDB(base_db, hooks)
    tr = t.set_index(t["molecule_id"].astype(str))
    order = sorted(tr.index, key=lambda m: (float(tr.at[m, "mono_mass"]), m))
    qpos = q.groupby(q["molecule_id"].astype(str)).indices
    k = int(args.top_k)
    parts = []
    for a in range(0, len(order), args.batch):
        chunk = order[a:a + args.batch]
        rows = np.sort(np.concatenate([qpos[m] for m in chunk if m in qpos]))
        sub = q.iloc[rows].reset_index(drop=True)
        hooks.active = tuple(sorted(chunk))
        st = pipeline.State(test_df=sub, db=proxy, extras={"validation": hooks})
        for fn in (pipeline.stage_mass, pipeline.stage_pool, pipeline.stage_features,
                   pipeline.stage_score):
            st = fn(st, cfg)
        hooks.compact(chunk, keys)
        pool = st.pool
        if pool is None or not len(pool):
            continue
        pool = fusion.sort_pool(pool, cfg=cfg).reset_index(drop=True)
        mids = pool["molecule_id"].astype(str).to_numpy(object)
        ik = pool["ik14"].astype(str).to_numpy(object)
        is_true = np.fromiter((x in keys.get(m, frozenset()) for m, x in zip(mids, ik)), bool,
                              len(pool))
        pos = pool.groupby(mids, sort=False).cumcount().to_numpy()
        sel = (pos < k) | is_true
        df = pool.loc[sel, ["molecule_id", "ik14", "formula", "tier", "in_np", "score",
                            "smiles"]].copy()
        df["pos"], df["in_topk"], df["is_true"] = pos[sel], (pos < k)[sel], is_true[sel]
        mrows = []
        for m, idx in pd.Series(np.arange(len(pool))).groupby(mids, sort=True):
            ii = idx.to_numpy()
            mrows.append({"molecule_id": m, "base_rank": _first_truth_rank(ik[ii], is_true[ii]),
                          "next_score": float(pool["score"].iat[ii[k]]) if len(ii) > k else -np.inf,
                          "n_pool": len(ii), "true_formula": str(tr.at[m, "formula"])})
        parts.append(df.merge(pd.DataFrame(mrows), on="molecule_id", how="left"))
        try:
            base_db.release()
        except Exception:  # noqa: BLE001
            pass
    tab = pd.concat(parts, ignore_index=True)
    tab["set"] = set_name
    tab["fold"] = tab["molecule_id"].map(tr["fold"]).astype(int)
    pc = tab["molecule_id"].map(tr["pubmed_count"])
    tab["pubmed0"] = pc.isna() | (pd.to_numeric(pc, errors="coerce").fillna(0) == 0)
    tab["same_formula"] = tab["formula"].astype(str) == tab["true_formula"].astype(str)
    lg.info("%s: built %d rows for %d molecules", set_name, len(tab), tab["molecule_id"].nunique())
    return tab


def _rows(args, cfg, lg, set_name) -> pd.DataFrame:
    if args.rows.startswith("fragments:"):
        tag = args.rows.split(":", 1)[1]
        p = paths.PROCESSED / "fragments" / _path(args, "eval", set_name, tag=tag).name
        tab = pd.read_parquet(p)
        if args.max_molecules:
            keep = sorted(tab["molecule_id"].astype(str).unique())[:args.max_molecules]
            tab = tab[tab["molecule_id"].astype(str).isin(keep)].reset_index(drop=True)
        return tab
    if args.rows == "pipeline":
        return _rows_pipeline(args, cfg, lg, set_name)
    raise SystemExit(f"unknown --rows {args.rows!r}")


def candidate_fps(args, cfg, lg, set_name, tab: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Packed fingerprints of the rows (cached; NP rows from np_fp.npy by ik14)."""
    from casmi import db as cdb
    from casmi.chem import morgan_fps

    p = _path(args, "cfp", set_name, ext="npz")
    key = hashlib.sha1("|".join(tab["ik14"].astype(str)).encode()).hexdigest()
    if p.exists():
        z = np.load(p, allow_pickle=False)
        if str(z["key"]) == key:
            return z["fp"], z["ok"]
    n = len(tab)
    fp = np.zeros((n, 512), np.uint8)
    ok = np.zeros(n, bool)
    t0 = time.perf_counter()
    d = cdb.np_dir()
    np_ik = pd.read_parquet(d / "np.parquet", columns=["ik14"])["ik14"].astype(str)
    row_of = pd.Series(np.arange(len(np_ik)), index=np_ik.to_numpy(object))
    row_of = row_of[~row_of.index.duplicated(keep="first")]
    r = tab["ik14"].astype(str).map(row_of).fillna(-1).astype(np.int64).to_numpy()
    has = r >= 0
    if has.any():
        arr = np.load(d / "np_fp.npy", mmap_mode="r")
        o = np.argsort(r[has], kind="mergesort")
        idx = np.flatnonzero(has)[o]
        fp[idx] = np.asarray(arr[r[has][o]])
        ok[idx] = fp[idx].any(axis=1)
    rest = np.flatnonzero(~has)
    if len(rest):
        smi = tab["smiles"].to_numpy(object)[rest]
        f, o2 = morgan_fps([s if isinstance(s, str) else "" for s in smi], cfg, n_jobs=args.n_jobs)
        f = np.asarray(f, np.uint8)
        f[~np.asarray(o2, bool)] = 0
        fp[rest], ok[rest] = f, np.asarray(o2, bool)
    np.savez(p, fp=fp, ok=ok, key=np.array(key))
    lg.info("%s: fingerprints of %d rows (%d from np_fp, %d from SMILES) in %.0f s", set_name, n,
            int(has.sum()), len(rest), time.perf_counter() - t0)
    return fp, ok


def compute_features(tab: pd.DataFrame, fp: np.ndarray, ok: np.ndarray, hits: pd.DataFrame,
                     cfg, n_jobs: int, struct: bool = True) -> pd.DataFrame:
    """analogs.FEATURES for the rows of ``tab`` from stored raw hits."""
    from casmi import analogs as A
    from casmi.chem import parallel_map

    mids = tab["molecule_id"].astype(str).to_numpy(object)
    form = tab["formula"].to_numpy(object)
    smi = tab["smiles"].to_numpy(object)
    hcols = list(A.HIT_COLUMNS)
    hb = {m: g[hcols].reset_index(drop=True)
          for m, g in hits.groupby(hits["molecule_id"].astype(str), sort=True)} if len(hits) else {}
    by = pd.Series(np.arange(len(tab))).groupby(mids, sort=True).indices
    items = []
    for m in sorted(by):
        rows = np.sort(by[m])
        items.append((m, fp[rows], ok[rows], list(form[rows]), list(smi[rows]),
                      np.full(len(rows), bool(struct)), hb.get(m)))
    nj = max(1, int(n_jobs))
    chunks = [items[i::nj] for i in range(nj)] if nj > 1 else [items]
    chunks = [c for c in chunks if c]
    res = []
    for part in parallel_map(A.features_task, [(cfg, c) for c in chunks], nj):
        res.extend(part)
    raw = np.full((len(tab), len(A.RAW_FEATURES)), np.nan)
    for m, mat in res:
        raw[np.sort(by[m])] = mat
    normed = A.normalise(pd.DataFrame(raw, columns=list(A.RAW_FEATURES)), mids)
    normed.index = tab.index
    return normed


def stage_features(args, cfg, lg) -> None:
    for set_name in args.sets:
        t0 = time.perf_counter()
        tab = _rows(args, cfg, lg, set_name)
        hp = _path(args, "hits", set_name)
        hits = pd.read_parquet(hp)
        miss = set(tab["molecule_id"].astype(str)) - set(hits["molecule_id"].astype(str))
        if miss:
            lg.warning("%s: %d molecules have no stored hit rows (no analogs or not searched)",
                       set_name, len(miss))
        fp, ok = candidate_fps(args, cfg, lg, set_name, tab)
        feats = compute_features(tab, fp, ok, hits, cfg, args.n_jobs, struct=True)
        keep = [c for c in tab.columns if c in ID_COLUMNS or c.startswith("frag_")]
        out = pd.concat([tab[keep].reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
        p = _path(args, "eval", set_name)
        out.to_parquet(p, index=False)
        lg.info("%s: %d rows, %d molecules, features in %.0f s, peak RSS %.0f MB -> %s", set_name,
                len(out), out["molecule_id"].nunique(), time.perf_counter() - t0, peak_rss_mb(), p)


# --------------------------------------------------------------------------------------
# stage 3: analysis
# --------------------------------------------------------------------------------------

def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if not len(pos) or not len(neg):
        return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank(method="average").to_numpy()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auc_table(tab: pd.DataFrame, features, same_formula: bool = True) -> dict:
    sub = tab[tab["same_formula"]] if same_formula else tab
    groups = [g for _, g in sub.groupby("molecule_id", sort=True)
              if g["is_true"].any() and (~g["is_true"]).any()]
    out = {}
    for f in features:
        if f not in sub.columns:
            continue
        vals = []
        for g in groups:
            x = g[f].to_numpy(dtype=np.float64)
            if f.endswith("_rank") or f in LOWER_IS_BETTER:
                x = -x
            okx = np.isfinite(x)
            y = g["is_true"].to_numpy()
            vals.append(_auc(x[okx & y], x[okx & ~y]))
        v = np.asarray(vals, dtype=np.float64)
        out[f] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
    out["_n"] = len(groups)
    return out


class Groups:
    """Per-molecule arrays of the top-K rows for fast re-ranking with feature bonuses."""

    def __init__(self, tab: pd.DataFrame, features):
        self.ids, self.base, self.items = [], {}, {}
        for m, g in tab.groupby("molecule_id", sort=True):
            self.ids.append(m)
            self.base[m] = int(g["base_rank"].iat[0])
            top = g[g["in_topk"]]
            if not g["is_true"].any() or self.base[m] == 0 or not top["is_true"].any():
                self.items[m] = None
                continue
            self.items[m] = {
                "score": top["score"].to_numpy(dtype=np.float64), "pos": top["pos"].to_numpy(),
                "ik": pd.factorize(top["ik14"].astype(str))[0], "true": top["is_true"].to_numpy(),
                "next": float(g["next_score"].iat[0]),
                "f": {f: np.nan_to_num(top[f].to_numpy(dtype=np.float64), nan=0.0)
                      for f in features if f in top.columns},
            }

    def ranks(self, weights: dict, k: int = 100) -> pd.Series:
        out = {}
        for m in self.ids:
            it = self.items[m]
            if it is None:
                out[m] = self.base[m]
                continue
            s = it["score"].copy()
            for f, w in weights.items():
                if w:
                    s = s + w * it["f"][f]
            order = np.lexsort((it["pos"], -s))
            r = _first_truth_rank(it["ik"][order], it["true"][order])
            st = s[order][np.flatnonzero(it["true"][order])[0]]
            if st < it["next"]:
                r = max(r, k + 1)
            out[m] = r
        return pd.Series(out)


def _mrr(ranks: pd.Series, top_k: int = 25, w: pd.Series | None = None) -> float:
    r = ranks.to_numpy(dtype=np.float64)
    rr = np.where((r > 0) & (r <= top_k), 1.0 / np.maximum(r, 1), 0.0)
    if not len(rr):
        return np.nan
    if w is None:
        return float(rr.mean())
    ww = w.reindex(ranks.index).to_numpy(dtype=np.float64)
    return float((rr * ww).sum() / ww.sum()) if ww.sum() > 0 else np.nan


def half_of(mid: str, seed: int) -> int:
    return int(hashlib.sha1(f"{seed}|{mid}".encode()).hexdigest(), 16) % 2


def cross_fit(tab: pd.DataFrame, seed: int, grids: dict[str, tuple], fixed: dict | None = None
              ) -> dict:
    """Tune the weights of ``grids`` (feature -> grid) on one hash half, test on the other,
    both ways; ``fixed`` weights are added untuned. Returns MRR / wins / losses (cross-fit),
    per set, and the chosen weights."""
    import itertools

    feats = list(grids) + list(fixed or {})
    gr = Groups(tab, feats)
    mids = list(gr.ids)
    half = {m: half_of(m, seed) for m in mids}
    base = pd.Series(gr.base)
    sets = tab.groupby("molecule_id", sort=True)["set"].first()
    combos = [dict(zip(grids, c)) for c in itertools.product(*grids.values())]
    cache = {}
    for c in combos:
        cache[tuple(c.values())] = gr.ranks({**c, **(fixed or {})})
    test_ranks, chosen = [], []
    for h in (0, 1):
        tr_ids = [m for m in mids if half[m] == h]
        te_ids = [m for m in mids if half[m] != h]
        best = max(combos, key=lambda c: (round(_mrr(cache[tuple(c.values())][tr_ids]), 6),
                                          -sum(c.values())))
        chosen.append(best)
        test_ranks.append(cache[tuple(best.values())][te_ids])
    allr = pd.concat(test_ranks).sort_index()
    res = {"weights": chosen, "n": len(mids), "mrr_base": _mrr(base), "mrr": _mrr(allr),
           "wins": int(((allr == 1) & (base != 1)).sum()),
           "losses": int(((allr != 1) & (base == 1)).sum()), "per_set": {}, "_ranks": allr}
    for s in sorted(sets.unique()):
        ids = sets.index[sets == s]
        res["per_set"][s] = {"n": int(len(ids)), "mrr_base": _mrr(base[ids]), "mrr": _mrr(allr[ids]),
                             "wins": int(((allr[ids] == 1) & (base[ids] != 1)).sum()),
                             "losses": int(((allr[ids] != 1) & (base[ids] == 1)).sum())}
    return res


def _v2_weights(tabs: dict[str, pd.DataFrame], args) -> pd.DataFrame | None:
    """Per molecule: v2 c2_cell and w_obs (None without splits/holdout_v2.csv)."""
    p = paths.SPLITS / "holdout_v2.csv"
    if not p.exists():
        return None
    try:
        from casmi import validation
        from casmi.splits import load_holdout_v2

        attrs = validation.v2_attributes(load_holdout_v2(p))
    except Exception:  # noqa: BLE001 - v2 is optional here
        return None
    rows = []
    for s in tabs:
        _, t = validation.load_queries(s, args.val_dir)
        a = attrs.reindex(t["mkey"].astype(str))
        rows.append(pd.DataFrame({"molecule_id": t["molecule_id"].astype(str).to_numpy(),
                                  "c2_cell": a["c2_cell"].fillna("none").astype(str).to_numpy(),
                                  "w_obs": pd.to_numeric(a["w_obs"], errors="coerce").fillna(1.0)
                                  .to_numpy()}))
    return pd.concat(rows, ignore_index=True).drop_duplicates("molecule_id").set_index("molecule_id")


def analyse(tabs: dict[str, pd.DataFrame], cfg, lg, args) -> dict:
    from casmi import analogs as A

    seed = int(cfg.seed)
    feats = [f for f in A.FEATURES if not f.endswith("_rank")]
    summ: dict = {"auc_same_formula": {}, "auc_all": {}, "context": {}}
    groups = dict(tabs)
    if "NP-OTHER" in tabs:
        groups["NP-OTHER pubmed0"] = tabs["NP-OTHER"][tabs["NP-OTHER"]["pubmed0"]]
    for s, t in groups.items():
        summ["auc_same_formula"][s] = auc_table(t, feats + [FRAG] if FRAG in t else feats, True)
        summ["auc_all"][s] = auc_table(t, feats + [FRAG] if FRAG in t else feats, False)
        mol = t.groupby("molecule_id").first()
        summ["context"][s] = {
            "n_molecules": int(len(mol)),
            "truth_in_topk": float(t[t["in_topk"]].groupby("molecule_id")["is_true"].any()
                                   .reindex(mol.index, fill_value=False).mean()),
            "prior_top1": float((mol["base_rank"] == 1).mean()),
            "with_hits": float((mol["an_n_hits"] > 0).mean()),
            "best_sim_median": float(mol["an_best_sim"].median()),
            "best_delta_share": float((mol["an_best_delta"] == 1).mean()),
        }
    lg.info("AUC truth vs same-formula decoys (mean over molecules):")
    hdr = list(groups)
    lg.info("  %-20s %s", "feature", " ".join(f"{h[:16]:>16}" for h in hdr))
    for f in feats + [FRAG]:
        lg.info("  %-20s %s", f, " ".join(f"{summ['auc_same_formula'][h].get(f, np.nan):16.3f}"
                                          for h in hdr))
    lg.info("  %-20s %s", "n molecules", " ".join(f"{summ['auc_same_formula'][h]['_n']:16d}"
                                                  for h in hdr))
    lg.info("AUC truth vs all decoys:")
    for f in feats + [FRAG]:
        lg.info("  %-20s %s", f, " ".join(f"{summ['auc_all'][h].get(f, np.nan):16.3f}" for h in hdr))
    # MRR studies
    np_sets = [s for s in ("NP-TIMS", "NP-OTHER") if s in tabs]
    studies = {}
    if np_sets:
        studies["NP pooled"] = pd.concat([tabs[s] for s in np_sets], ignore_index=True)
    for s in tabs:
        studies[f"within {s}"] = tabs[s]
    v2 = _v2_weights(tabs, args)
    summ["mrr"] = {}
    has_frag = all(FRAG in t.columns for t in tabs.values())
    for name, t in studies.items():
        blk: dict = {}
        flist = BONUS_FEATURES if name == "NP pooled" else (PRIMARY, "an_sub_w_z")
        for f in flist:
            blk[f] = cross_fit(t, seed, {f: W_GRID})
        if has_frag:
            blk[FRAG] = cross_fit(t, seed, {FRAG: W_GRID})
            for f in (PRIMARY, "an_sub_w_z"):
                blk[f"{FRAG}+{f}"] = cross_fit(t, seed, {FRAG: W_GRID, f: W_GRID})
        for key, r in blk.items():
            lg.info("MRR %s, %s: %.4f -> %.4f (wins %d, losses %d, n %d); weights %s", name, key,
                    r["mrr_base"], r["mrr"], r["wins"], r["losses"], r["n"], r["weights"])
            for s, ps in r["per_set"].items():
                if len(r["per_set"]) > 1:
                    lg.info("    %-9s %.4f -> %.4f (wins %d, losses %d, n %d)", s, ps["mrr_base"],
                            ps["mrr"], ps["wins"], ps["losses"], ps["n"])
            # subsets: NP-OTHER pubmed0 and v2 weighted cells
            mol = t.groupby("molecule_id", sort=True)[["set", "pubmed0", "base_rank"]].first()
            ranks = r.pop("_ranks")
            sub = {}
            ids = mol.index[(mol["set"] == "NP-OTHER") & mol["pubmed0"]]
            if len(ids):
                sub["NP-OTHER pubmed0"] = {"n": int(len(ids)), "mrr_base": _mrr(mol.loc[ids, "base_rank"]),
                                           "mrr": _mrr(ranks[ids])}
            if v2 is not None:
                vv = v2.reindex(mol.index)
                for cell, setsel in (("C2-NP", ("NP-TIMS", "NP-OTHER")), ("C2-PC", ("SYN-TIMS",))):
                    ids = mol.index[(vv["c2_cell"] == cell.split("-")[1]) & mol["set"].isin(setsel)]
                    if len(ids):
                        w = vv.loc[ids, "w_obs"]
                        sub[f"v2 {cell} (w_obs)"] = {
                            "n": int(len(ids)), "mrr_base": _mrr(mol.loc[ids, "base_rank"], w=w),
                            "mrr": _mrr(ranks[ids], w=w)}
            for sk, sv in sub.items():
                lg.info("    %-22s %.4f -> %.4f (n %d)", sk, sv["mrr_base"], sv["mrr"], sv["n"])
            r["subsets"] = sub
        summ["mrr"][name] = blk
    # analog feature alone on the scored rows (optimistic: decoys are only the prior top 100)
    summ["alone"] = {}
    for s, t in groups.items():
        rr = []
        for m, g in t.groupby("molecule_id", sort=True):
            if not g["is_true"].any():
                rr.append(0.0)
                continue
            x = g[PRIMARY].to_numpy(dtype=np.float64)
            x = np.nan_to_num(x, nan=-np.inf)
            sc = g["score"].to_numpy(dtype=np.float64)
            o = np.lexsort((g["pos"].to_numpy(), -sc, -x))
            r = _first_truth_rank(g["ik14"].astype(str).to_numpy(object)[o],
                                  g["is_true"].to_numpy()[o])
            rr.append(1.0 / r if 0 < r <= 25 else 0.0)
        summ["alone"][s] = float(np.mean(rr)) if rr else np.nan
        lg.info("%s alone (prior tie-break) on the scored rows: MRR %.4f", PRIMARY, summ["alone"][s])
    for s, c in summ["context"].items():
        lg.info("context %s: %s", s, {k: round(v, 4) if isinstance(v, float) else v
                                      for k, v in c.items()})
    return summ


def variants(tabs: dict[str, pd.DataFrame], cfg, lg, args) -> dict:
    """Fingerprint features re-computed from stored hits under other hit-selection settings."""
    import yaml

    out = {}
    seed = int(cfg.seed)
    np_sets = [s for s in ("NP-TIMS", "NP-OTHER") if s in tabs]
    for spec in ["default"] + list(args.variants):
        over = {} if spec == "default" else {kv.split("=", 1)[0]: yaml.safe_load(kv.split("=", 1)[1])
                                             for kv in spec.split(",")}
        c = cfg.replace(**over)
        vt = {}
        for s, t in tabs.items():
            hits = pd.read_parquet(_path(args, "hits", s))
            z = np.load(_path(args, "cfp", s, ext="npz"))
            f = compute_features(t, z["fp"], z["ok"], hits, c, args.n_jobs, struct=False)
            tt = t[[c2 for c2 in t.columns if not c2.startswith("an_")]].copy()
            vt[s] = pd.concat([tt.reset_index(drop=True), f.reset_index(drop=True)], axis=1)
        res = {"auc": {s: {k: v for k, v in auc_table(t, ("an_tan_wmean", "an_tan_wmax",
                                                              "an_tan_max"), True).items()}
                           for s, t in vt.items()}}
        if np_sets:
            pooled = pd.concat([vt[s] for s in np_sets], ignore_index=True)
            r = cross_fit(pooled, seed, {PRIMARY: W_GRID})
            r.pop("_ranks")
            res["mrr_np_pooled"] = {k: r[k] for k in ("mrr_base", "mrr", "wins", "losses", "weights")}
        out[spec] = res
        lg.info("variant %-32s AUC wmean %s | NP pooled MRR %.4f -> %.4f",
                spec, " ".join(f"{s[:8]}={res['auc'][s]['an_tan_wmean']:.3f}" for s in vt),
                res.get("mrr_np_pooled", {}).get("mrr_base", np.nan),
                res.get("mrr_np_pooled", {}).get("mrr", np.nan))
    return out


def stage_analyse(args, cfg, lg) -> None:
    tabs = {s: pd.read_parquet(_path(args, "eval", s)) for s in args.sets
            if _path(args, "eval", s).exists()}
    summ = analyse(tabs, cfg, lg, args)
    if args.variants:
        summ["variants"] = variants(tabs, cfg, lg, args)
    summ["settings"] = {"sets": list(tabs), "cell": args.cell, "tag": args.tag, "rows": args.rows,
                        "overrides": cfg.overrides(), "primary_feature": PRIMARY,
                        "w_grid": list(W_GRID)}
    p = Path(args.out) / f"eval_{args.tag}_{args.cell}_summary.json"
    p.write_text(json.dumps(summ, indent=1, default=float), encoding="utf-8")
    lg.info("wrote %s", p)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--stage", choices=("hits", "features", "analyse", "all"), default="all")
    ap.add_argument("--sets", nargs="+", default=["NP-TIMS", "NP-OTHER", "SYN-TIMS"])
    ap.add_argument("--cell", "--condition", dest="cell", default="C2",
                    help="v1 condition (C1, C2, C3) or v2 cell (C1, C2-NP, C2-PC, C3h, C3e)")
    ap.add_argument("--rows", default="fragments:v1",
                    help="candidate rows: fragments:<R7 tag> or pipeline")
    ap.add_argument("--val-dir", type=Path, default=paths.PROCESSED / "validation")
    ap.add_argument("--out", type=Path, default=paths.PROCESSED / "analogs")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--top-k", type=int, default=100, help="--rows pipeline: rows per molecule")
    ap.add_argument("--batch", type=int, default=20, help="--rows pipeline: molecules per lookup")
    ap.add_argument("--max-molecules", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--variants", nargs="*", default=None,
                    help="hit-selection variants for the analysis (key=value[,key=value]); "
                         "no value = the default list")
    ap.add_argument("--set-config", nargs="*", default=[],
                    help="extra Config overrides as key=value (YAML-parsed values)")
    args = ap.parse_args(argv)
    if args.variants is not None and not args.variants:
        args.variants = list(DEFAULT_VARIANTS)
    args.variants = args.variants or []
    lg = get_logger("eval_analogs")
    log_run(lg, args)
    cfg = load_config(args.config)
    if not cfg.analog_feat_fp_path and (paths.PROCESSED / "validation" / "train_fp.npy").exists():
        cfg = cfg.replace(analog_feat_fp_path="validation/train_fp")
    if args.set_config:
        import yaml

        cfg = cfg.replace(**{kv.split("=", 1)[0]: yaml.safe_load(kv.split("=", 1)[1])
                             for kv in args.set_config})
    lg.info("config overrides: %s", cfg.overrides())
    paths.ensure_dir(args.out)
    if args.stage in ("hits", "all"):
        stage_hits(args, cfg, lg)
    if args.stage in ("features", "all"):
        stage_features(args, cfg, lg)
    if args.stage in ("analyse", "all"):
        stage_analyse(args, cfg, lg)
    lg.info("done; peak RSS %.0f MB", peak_rss_mb())
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Library candidates and their spectral-match features: the "lib" pipeline stage (WP8).

PLAN.md 4.2 (library matching), 4.6 (fusion rule 1), 4.8 (assembly), 5.3 (validation masks).

Use
---
List the module as a plugin and put the stage after ``pool``::

    pipeline_plugins: [casmi.libmatch]
    pipeline_stages: [mass, pool, lib, features, score, assemble]

(``configs/b1_library.yaml``). The stage is registered under its own name, so importing
this module never changes runs that do not list it.

What the stage does, per batch
------------------------------
1. Library handle: ``state.extras["library"]`` (the validation harness passes one), else
   ``Library.load(lib_dir(cfg))``, cached per process. No library on disk: a warning, and
   the pool is left as it is (Baseline 0 behaviour), so a missing artefact cannot sink a run.
2. Per molecule (sorted ids): ``Library.search_by_mass(M, cfg.lib_mass_tol_ppm,
   queries=<the molecule's test rows>)``. Passing the rows (they carry ``molecule_id``) lets
   the validation guard apply that molecule's exclusions; when ``state.extras["validation"]``
   is present its ``library_mask`` is passed as well (the two are OR-ed, so this is
   harmless). Pairs are same-polarity only (``search_by_mass``). If the DB pool was found
   with every adduct replaced by its polarity default (``pool_attempt`` default_adduct*,
   PLAN risk 11), the library is searched at that mass too.
3. One row per LIB structure, columns ``LIB_COLUMNS``: the search output renamed with a
   ``lib_`` prefix (``LIB_FEATURE_MAP``), plus

   - ``lib_best_cos`` / ``lib_n_matched``: the similarity fusion reads (``cfg.lib_sim``:
     cos_max = best (query spectrum, reference) cosine; ent_max = same for entropy;
     cons_cos = query consensus vs reference consensus; max_cos_cons = the larger of
     cos_max and cons_cos; mean_cos_ent = (cos_max + ent_max) / 2; graded = over ALL pairs
     of the candidate, (query spectrum, reference spectrum) and (query consensus, reference
     consensus) of the same polarity (and adduct family for consensus), the pair that
     passes the stricter fusion rule (1, then 1b), then has the higher cosine, then more
     matched peaks (``graded_evidence``): the search's single best pair can be a 2-peak
     spectrum with cosine 1.0 that hides a 20-peak match at 0.98) and the matched peaks of
     the pair it came from (graded: that pair's; cons_cos: cons_n_matched; else
     n_matched_best);
   - ``lib_graded_cos`` / ``lib_graded_n`` (the best-graded pair, ``graded_evidence``) and
     ``lib_cos_n<k>`` for k in ``GRADED_KS`` (best cosine over the candidate's spectrum and
     consensus pairs with at least k matched peaks; NaN if none);
   - pool context per molecule: ``lib_pool_n`` (LIB candidates), ``lib_pool_best`` and
     ``lib_pool_second`` (best and second-best lib_best_cos, NaN if absent), ``lib_rank``
     (1 = best lib_best_cos in the molecule, ties by search order), ``lib_gap`` (own
     lib_best_cos minus the best other LIB candidate's).

4. Join with the DB pool on (molecule_id, ik14): a LIB structure that the DB pool also holds
   becomes ONE row with ``tier = "lib"``, the DB row's prior columns (in_np, counts, ...)
   and ``db_tier`` = the DB row's tier; LIB structures not in the DB pool get ``db_tier``
   NA and no priors. LIB rows keep the library SMILES (the train label). ``is_lib`` is True
   on LIB rows, False elsewhere. Rows stay grouped by molecule_id.
5. Report columns on ``state.molecules``: lib_n, lib_n_db (also in the DB pool), lib_best,
   lib_second, lib_n_rule1 (candidates passing fusion rule 1 at the configured thresholds).

Fusion (``casmi.fusion.fuse``) then applies rule 1 (lib_best_cos >= fusion_lib_cos_min and
lib_n_matched >= fusion_lib_min_matched: 3 + lib_best_cos) and, when the config turns them
on, rule 1b (band or additive) and the demotion (``fusion.lib_extra_rules``). The values in
``configs/b1_library.yaml`` and why: docs/EXPERIMENTS.md, "WP8 tuning".

Inference-path rule (PLAN 7): numpy, pandas and casmi modules only.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import paths, pipeline
from .config import Config

__all__ = [
    "LIB_FEATURE_MAP",
    "LIB_COLUMNS",
    "LIB_SIMS",
    "GRADED_KS",
    "lib_dir",
    "get_library",
    "clear_library_cache",
    "lib_similarity",
    "graded_evidence",
    "lib_candidates",
    "merge_pool",
    "stage_lib",
]

log = logging.getLogger(__name__)

# Library.search_by_mass column -> pool column.
LIB_FEATURE_MAP: dict[str, str] = {
    "struct_idx": "lib_struct_idx",
    "mass_err_ppm": "lib_mass_err_ppm",
    "n_refs": "lib_n_refs",
    "n_refs_pol": "lib_n_refs_pol",
    "n_refs_tims": "lib_n_refs_tims",
    "has_tims": "lib_has_tims",
    "n_cons": "lib_n_cons",
    "cos_max": "lib_cos_max",
    "cos_mean": "lib_cos_mean",
    "ent_max": "lib_ent_max",
    "ent_mean": "lib_ent_mean",
    "cos_max_tims": "lib_cos_max_tims",
    "cos_mean_tims": "lib_cos_mean_tims",
    "ent_max_tims": "lib_ent_max_tims",
    "ent_mean_tims": "lib_ent_mean_tims",
    "cos_max_same_adduct": "lib_cos_max_same_adduct",
    "cos_mean_same_adduct": "lib_cos_mean_same_adduct",
    "ent_max_same_adduct": "lib_ent_max_same_adduct",
    "ent_mean_same_adduct": "lib_ent_mean_same_adduct",
    "cons_cos": "lib_cons_cos",
    "cons_ent": "lib_cons_ent",
    "cons_n_matched": "lib_cons_n_matched",
    "n_matched_best": "lib_n_matched_best",
    "best_ref_lib": "lib_best_ref_lib",
    "ce_dist_best": "lib_ce_dist_best",
}
GRADED_KS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8)
_DERIVED = (("lib_graded_cos", "lib_graded_n") + tuple(f"lib_cos_n{k}" for k in GRADED_KS)
            + ("lib_best_cos", "lib_n_matched", "lib_pool_n", "lib_pool_best", "lib_pool_second",
               "lib_rank", "lib_gap"))
LIB_COLUMNS: tuple[str, ...] = tuple(LIB_FEATURE_MAP.values()) + _DERIVED
LIB_SIMS = ("cos_max", "ent_max", "cons_cos", "max_cos_cons", "mean_cos_ent", "graded")
_ROW_COLUMNS = ("molecule_id", "query_mass", "ik14", "smiles", "formula", "mono_mass", "ppm_err",
                "tier", "is_lib")

_LIB_CACHE: dict[tuple, Any] = {}
_WARNED: set[str] = set()


def _warn_once(key: str, msg: str, *args: Any) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        log.warning(msg, *args)


# --- library handle --------------------------------------------------------------------

def lib_dir(cfg: Config) -> Path:
    """Library directory: ``cfg.lib_dir`` (relative paths under ``paths.PROCESSED``), else
    ``paths.PROCESSED / "library"``."""
    d = str(getattr(cfg, "lib_dir", "") or "").strip()
    if not d:
        return paths.PROCESSED / "library"
    p = Path(d)
    return p if p.is_absolute() else paths.PROCESSED / p


def _cfg_key(cfg: Config) -> tuple:
    names = ("frag_tol_same_da", "frag_tol_cross_da", "entropy_weighted", "intensity_floor",
             "top_n_peaks", "keep_precursor", "precursor_window_low_da",
             "precursor_window_high_da", "clean_max_above_precursor_da", "consensus_merge_da",
             "intensity_transform", "mass_tol_min_da")
    return tuple(getattr(cfg, n) for n in names)


def get_library(state: pipeline.State | None, cfg: Config) -> Any:
    """The Library for this run (see module docstring, step 1), or None if none exists."""
    if state is not None and state.extras.get("library") is not None:
        lib = state.extras["library"]
        if _cfg_key(lib.cfg) != _cfg_key(cfg):
            _warn_once("cfg", "libmatch: the library handle was loaded with other cleaning or "
                       "fragment settings than the run config; its own settings are used")
        return lib
    d = lib_dir(cfg)
    key = (str(d), _cfg_key(cfg))
    if key in _LIB_CACHE:
        return _LIB_CACHE[key]
    if not (d / "meta.parquet").exists():
        _warn_once(f"missing:{d}", "libmatch: no library at %s; LIB candidates skipped", d)
        return None
    from .library import Library

    t0 = time.perf_counter()
    lib = Library.load(d, cfg)
    log.info("libmatch: library %s loaded (%d spectra, %d structures) in %.1f s", d,
             lib.n_spectra, lib.n_structures, time.perf_counter() - t0)
    _LIB_CACHE[key] = lib
    return lib


def clear_library_cache() -> None:
    _LIB_CACHE.clear()
    _WARNED.clear()


# --- features --------------------------------------------------------------------------

def lib_similarity(res: pd.DataFrame, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """(lib_best_cos, lib_n_matched) per search row, from ``cfg.lib_sim``."""
    how = str(cfg.lib_sim)
    if how not in LIB_SIMS:
        raise ValueError(f"lib_sim must be one of {LIB_SIMS}, got {how!r}")

    def num(c: str) -> np.ndarray:
        return pd.to_numeric(res[c], errors="coerce").to_numpy(dtype=np.float64)

    cos, ent, cons = num("cos_max"), num("ent_max"), num("cons_cos")
    nm, nmc = num("n_matched_best"), num("cons_n_matched")
    if how == "cos_max":
        return cos, nm
    if how == "ent_max":
        return ent, nm
    if how == "cons_cos":
        return cons, nmc
    if how == "mean_cos_ent":
        with np.errstate(invalid="ignore"):
            return (cos + ent) / 2.0, nm
    if how == "graded":
        if "_g_cos" not in res.columns:
            raise ValueError("lib_sim 'graded' needs graded_evidence columns (_g_cos, _g_n)")
        return num("_g_cos"), num("_g_n")
    # max_cos_cons: the larger of the two (cons wins only when strictly larger)
    use_cons = np.nan_to_num(cons, nan=-np.inf) > np.nan_to_num(cos, nan=-np.inf)
    return np.where(use_cons, cons, cos), np.where(use_cons, nmc, nm)


def _grade(cos: np.ndarray, n: np.ndarray, cfg: Config) -> np.ndarray:
    """Pair grade: 4 x (passes rule 1) + 2 x (passes rule 1b thresholds) + cos / 2."""
    c = np.nan_to_num(cos, nan=-1.0)
    k = np.nan_to_num(n, nan=0.0)
    r1 = (c >= cfg.fusion_lib_cos_min) & (k >= cfg.fusion_lib_min_matched)
    rb = (c >= cfg.fusion_lib_weak_cos_min) & (k >= cfg.fusion_lib_weak_min_matched)
    return r1 * 4.0 + rb * 2.0 + c / 2.0


def graded_evidence(lib: Any, qs: Any, struct_idx: np.ndarray, exclude_mask: np.ndarray | None,
                    cfg: Config) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """(cosine, matched peaks) of each candidate structure's best-graded pair, plus
    ``{k: best cosine over its pairs with >= k matched peaks}`` for k in ``GRADED_KS``.

    Pairs: every non-excluded reference spectrum of the structure against every query
    spectrum of the same polarity, and every non-excluded reference consensus against every
    query consensus of the same polarity and adduct family; fragment tolerance as in
    ``Library.search_by_mass`` (``frag_tol_same_da`` when both are timsTOF, else
    ``frag_tol_cross_da``, from the library's config). Best = highest ``_grade``, then
    cosine, then matched peaks, then lower pair index. NaN / 0 without pairs.

    The best-graded pair follows from the per-k maxima: rule 1 holds iff cos_k[K1] >=
    c1, then the chosen cosine is cos_k[K1]; else rule 1b iff cos_k[Kw] >= cw (cosine
    cos_k[Kw]); else the overall best cosine. So thresholds with K in ``GRADED_KS`` can be
    tuned offline from the ``lib_cos_n<k>`` columns.
    """
    from .library import _expand_ranges
    from .similarity import pair_scores_batch

    cand = np.asarray(struct_idx, dtype=np.int64)
    nc = cand.size
    best_cos = np.full(nc, np.nan)
    best_n = np.zeros(nc, dtype=np.float64)
    at_k = {k: np.full(nc, np.nan) for k in GRADED_KS}
    if nc == 0 or qs is None or len(qs) == 0:
        return best_cos, best_n, at_k
    lcfg = lib.cfg
    same, cross = float(lcfg.frag_tol_same_da), float(lcfg.frag_tol_cross_da)
    ex = None if exclude_mask is None else np.asarray(exclude_mask, dtype=bool)
    owners, coss, ns = [], [], []

    def run(qstore, q_pol, q_tims, q_fam, rstore, r_rows, r_own, r_pol, r_tims, r_fam):
        pq_l, pr_l, po_l, tol_l = [], [], [], []
        for q in range(len(qstore)):
            if qstore.n_peaks[q] == 0:
                continue
            sel = r_pol[r_rows] == q_pol[q]
            if q_fam is not None:
                sel &= r_fam[r_rows] == q_fam[q]
            sel = np.flatnonzero(sel)
            if not sel.size:
                continue
            pq_l.append(np.full(sel.size, q, np.int64))
            pr_l.append(r_rows[sel])
            po_l.append(r_own[sel])
            tol_l.append(np.where(r_tims[r_rows[sel]] & bool(q_tims[q]), same, cross))
        if not pq_l:
            return
        pq_, pr_ = np.concatenate(pq_l), np.concatenate(pr_l)
        cos, ncos, _, _ = pair_scores_batch(
            qstore.mz, qstore.intensity, qstore.offsets, rstore.mz, rstore.intensity,
            rstore.offsets, pq_, pr_, np.concatenate(tol_l), weighted=lcfg.entropy_weighted)
        owners.append(np.concatenate(po_l))
        coss.append(np.asarray(cos, dtype=np.float64))
        ns.append(np.asarray(ncos, dtype=np.float64))

    refs, own = _expand_ranges(lib.struct_spec[cand, 0], lib.struct_spec[cand, 1])
    if ex is not None:
        k = ~ex[refs]
        refs, own = refs[k], own[k]
    q_tims = np.asarray(qs.instrument_family == "timstof", dtype=bool)
    run(qs.store, qs.polarity, q_tims, None, lib.store, refs, own, lib.spec_pol, lib.spec_tims,
        None)
    if len(qs.cons_store):
        cr, cown = _expand_ranges(lib.struct_cons[cand, 0], lib.struct_cons[cand, 1])
        if ex is not None:
            k = ~lib.consensus_mask(ex)[cr]
            cr, cown = cr[k], cown[k]
        cq_tims = np.asarray(qs.cons_instrument_family == "timstof", dtype=bool)
        run(qs.cons_store, qs.cons_polarity, cq_tims, qs.cons_family, lib.cons_store, cr, cown,
            lib.cons_pol, lib.cons_tims, lib.cons_family)
    if not owners:
        return best_cos, best_n, at_k
    o = np.concatenate(owners)
    c = np.concatenate(coss)
    n = np.concatenate(ns)
    g = _grade(c, n, cfg)
    order = np.lexsort((np.arange(o.size), -n, -np.nan_to_num(c, nan=-1.0), -g, o))
    first = np.ones(order.size, dtype=bool)
    first[1:] = o[order][1:] != o[order][:-1]
    b = order[first]
    best_cos[o[b]] = c[b]
    best_n[o[b]] = n[b]
    for k in GRADED_KS:
        sel = n >= k
        if sel.any():
            m = np.full(nc, -np.inf)
            np.maximum.at(m, o[sel], c[sel])
            at_k[k] = np.where(np.isfinite(m), m, np.nan)
    return best_cos, best_n, at_k


def _pool_context(df: pd.DataFrame) -> pd.DataFrame:
    """lib_pool_n, lib_pool_best, lib_pool_second, lib_rank, lib_gap (rows of one or more
    molecules, in search order inside each molecule)."""
    n = len(df)
    out = {c: np.full(n, np.nan) for c in ("lib_pool_best", "lib_pool_second", "lib_gap")}
    out["lib_pool_n"] = np.zeros(n, dtype=np.int64)
    out["lib_rank"] = np.zeros(n, dtype=np.int64)
    if not n:
        return df.assign(**out)
    mids = df["molecule_id"].astype(str).to_numpy(dtype=object)
    sim = df["lib_best_cos"].to_numpy(dtype=np.float64)
    for mid in sorted(set(mids)):
        idx = np.flatnonzero(mids == mid)
        s = sim[idx]
        key = np.where(np.isnan(s), np.inf, -s)
        order = np.lexsort((np.arange(idx.size), key))
        rank = np.empty(idx.size, dtype=np.int64)
        rank[order] = np.arange(1, idx.size + 1)
        fin = np.sort(s[~np.isnan(s)])[::-1]
        best = fin[0] if fin.size else np.nan
        second = fin[1] if fin.size > 1 else np.nan
        out["lib_pool_n"][idx] = idx.size
        out["lib_rank"][idx] = rank
        out["lib_pool_best"][idx] = best
        out["lib_pool_second"][idx] = second
        other_best = np.where(rank == 1, second, best)
        out["lib_gap"][idx] = s - other_best
    return df.assign(**out)


def _search_one(lib: Any, mid: str, masses: list[float], sub: pd.DataFrame, cfg: Config,
                hooks: Any) -> pd.DataFrame | None:
    mask = None
    if hooks is not None:
        try:
            mask = hooks.library_mask(lib, mid)
        except Exception:  # noqa: BLE001 - the guard still applies its own mask
            log.exception("libmatch: validation library_mask failed for %s", mid)
    parts = []
    for M in masses:
        r = lib.search_by_mass(float(M), tol_ppm=cfg.lib_mass_tol_ppm, queries=sub,
                               exclude_mask=mask, min_da=cfg.mass_tol_min_da)
        if len(r):
            from .library import prepare_queries

            qs = prepare_queries(sub, lib.cfg, M=float(M))
            gc, gn, gk = graded_evidence(lib, qs, r["struct_idx"].to_numpy(np.int64), mask, cfg)
            r = r.assign(_M=float(M), _g_cos=gc, _g_n=gn,
                         **{f"_gk{k}": v for k, v in gk.items()})
            parts.append(r)
    if not parts:
        return None
    r = parts[0] if len(parts) == 1 else pd.concat(parts, ignore_index=True)
    return r.drop_duplicates("ik14", keep="first").reset_index(drop=True)


def lib_candidates(lib: Any, test_df: pd.DataFrame, molecules: pd.DataFrame, cfg: Config,
                   hooks: Any = None) -> pd.DataFrame:
    """LIB rows for every molecule of ``molecules`` (``_ROW_COLUMNS`` + ``LIB_COLUMNS``),
    grouped by molecule_id in sorted order, search order inside a molecule."""
    tdf = test_df.assign(molecule_id=test_df["molecule_id"].astype(str))
    pos = tdf.groupby("molecule_id", sort=True).indices
    n = len(molecules)
    mol_ids = molecules["molecule_id"].astype(str).to_numpy(dtype=object)
    m_arr = pd.to_numeric(molecules["M"], errors="coerce").to_numpy(dtype=np.float64)
    m_pool = (pd.to_numeric(molecules["M_pool"], errors="coerce").to_numpy(dtype=np.float64)
              if "M_pool" in molecules.columns else m_arr)
    attempt = (molecules["pool_attempt"].astype(object).fillna("").astype(str).to_numpy(dtype=object)
               if "pool_attempt" in molecules.columns else np.full(n, "", dtype=object))
    frames = []
    for i in np.argsort(mol_ids, kind="mergesort"):
        mid, M, mp = mol_ids[i], float(m_arr[i]), float(m_pool[i])
        if mid not in pos or not np.isfinite(M):
            continue
        masses = [M]
        if str(attempt[i]).startswith("default_adduct") and np.isfinite(mp) and abs(mp - M) > 1e-9:
            masses.append(mp)
        sub = tdf.iloc[np.sort(pos[mid])].reset_index(drop=True)
        r = _search_one(lib, mid, masses, sub, cfg, hooks)
        if r is None or not len(r):
            continue
        sim, nm = lib_similarity(r, cfg)
        f = pd.DataFrame({
            "molecule_id": mid,
            "query_mass": r["_M"].to_numpy(dtype=np.float64),
            "ik14": r["ik14"].astype(str).to_numpy(dtype=object),
            "smiles": r["smiles"].astype(object).to_numpy(dtype=object),
            "formula": r["formula"].astype(object).to_numpy(dtype=object),
            "mono_mass": r["mono_mass"].to_numpy(dtype=np.float64),
            "ppm_err": r["mass_err_ppm"].to_numpy(dtype=np.float64),
            "tier": "lib",
            "is_lib": True,
        })
        for src, dst in LIB_FEATURE_MAP.items():
            f[dst] = r[src].to_numpy()
        f["lib_graded_cos"] = r["_g_cos"].to_numpy(dtype=np.float64)
        f["lib_graded_n"] = r["_g_n"].to_numpy(dtype=np.float64)
        for k in GRADED_KS:
            f[f"lib_cos_n{k}"] = r[f"_gk{k}"].to_numpy(dtype=np.float64)
        f["lib_best_cos"] = sim
        f["lib_n_matched"] = nm
        frames.append(f)
    if not frames:
        return lib_candidates_empty()
    out = pd.concat(frames, ignore_index=True)
    out["lib_has_tims"] = out["lib_has_tims"].astype(bool)
    out["lib_best_ref_lib"] = out["lib_best_ref_lib"].astype(object)
    return _pool_context(out)


def merge_pool(pool: pd.DataFrame | None, lib_rows: pd.DataFrame | None) -> pd.DataFrame:
    """DB pool + LIB rows, deduplicated on (molecule_id, ik14) (module docstring, step 4).

    Every ``LIB_COLUMNS`` column, ``is_lib`` and ``db_tier`` exist in the result, also when
    there are no LIB rows (NaN / False / the DB tier)."""
    lib_rows = lib_candidates_empty() if lib_rows is None else lib_rows
    if pool is None:
        pool = pd.DataFrame({"molecule_id": pd.Series([], dtype=object),
                             "ik14": pd.Series([], dtype=object)})
    pool = pool.copy()
    pool["is_lib"] = False
    pool["db_tier"] = (pool["tier"].astype(object) if "tier" in pool.columns
                       else pd.Series([pd.NA] * len(pool), dtype=object, index=pool.index))
    in_lib = np.zeros(len(pool), dtype=bool)
    merged = lib_rows.copy()
    if len(lib_rows) and len(pool):
        pk = pool["molecule_id"].astype(str) + "|" + pool["ik14"].astype(str)
        lk = lib_rows["molecule_id"].astype(str) + "|" + lib_rows["ik14"].astype(str)
        in_lib = pk.isin(set(lk)).to_numpy()
        own = set(_ROW_COLUMNS) | set(LIB_COLUMNS)
        db_cols = [c for c in pool.columns if c not in own and c != "_key"]
        db_part = (pool.loc[in_lib, db_cols].assign(_key=pk[in_lib].to_numpy())
                   .drop_duplicates("_key"))
        merged = (lib_rows.assign(_key=lk.to_numpy())
                  .merge(db_part, on="_key", how="left", sort=False).drop(columns="_key"))
    elif len(lib_rows):
        merged["db_tier"] = pd.Series([pd.NA] * len(merged), dtype=object)
    parts = [f for f in (pool.loc[~in_lib], merged) if len(f)]
    out = (pd.concat(parts, ignore_index=True, sort=False) if len(parts) > 1
           else parts[0].reset_index(drop=True) if parts else pool.iloc[:0].copy())
    for c in LIB_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    out["is_lib"] = out["is_lib"].fillna(False).astype(bool)
    for c in pool.columns:
        if c not in out.columns:          # pool columns absent from LIB-only rows
            out[c] = pd.Series([pd.NA] * len(out), dtype=object, index=out.index)
    cols = list(pool.columns) + [c for c in out.columns if c not in pool.columns]
    return out[cols].sort_values("molecule_id", kind="mergesort").reset_index(drop=True)


# --- the stage -------------------------------------------------------------------------

def _rule1_count(lib_rows: pd.DataFrame, cfg: Config) -> pd.Series:
    if not len(lib_rows):
        return pd.Series(dtype=np.int64)
    cos = lib_rows["lib_best_cos"].to_numpy(dtype=np.float64)
    nm = lib_rows["lib_n_matched"].to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore"):
        ok = (cos >= cfg.fusion_lib_cos_min) & (nm >= cfg.fusion_lib_min_matched)
    return pd.Series(ok.astype(np.int64)).groupby(
        lib_rows["molecule_id"].astype(str).to_numpy()).sum()


def stage_lib(state: pipeline.State, cfg: Config) -> pipeline.State:
    """Add LIB candidates with spectral-match features to ``state.pool`` (module docstring)."""
    mols = state.molecules
    lib = get_library(state, cfg)
    ids = mols["molecule_id"].astype(str)
    if lib is None:
        state.pool = merge_pool(state.pool, None)
        state.molecules = mols.assign(lib_n=0, lib_n_db=0, lib_best=np.nan, lib_second=np.nan,
                                      lib_n_rule1=0)
        return state
    t0 = time.perf_counter()
    rows = lib_candidates(lib, state.test_df, mols, cfg, state.extras.get("validation"))
    n_db = pd.Series(dtype=np.int64)
    if state.pool is not None and len(state.pool) and len(rows):
        pk = set(state.pool["molecule_id"].astype(str) + "|" + state.pool["ik14"].astype(str))
        lk = rows["molecule_id"].astype(str) + "|" + rows["ik14"].astype(str)
        n_db = pd.Series(lk.isin(pk).astype(np.int64).to_numpy()).groupby(
            rows["molecule_id"].astype(str).to_numpy()).sum()
    state.pool = merge_pool(state.pool, rows)
    if len(rows):
        g = rows.groupby(rows["molecule_id"].astype(str))
        n = g.size()
        best = g["lib_pool_best"].first()
        second = g["lib_pool_second"].first()
    else:
        n = best = second = pd.Series(dtype=np.float64)
    r1 = _rule1_count(rows, cfg)
    state.molecules = mols.assign(
        lib_n=ids.map(n).fillna(0).astype(np.int64).to_numpy(),
        lib_n_db=ids.map(n_db).fillna(0).astype(np.int64).to_numpy(),
        lib_best=ids.map(best).astype(np.float64).to_numpy(),
        lib_second=ids.map(second).astype(np.float64).to_numpy(),
        lib_n_rule1=ids.map(r1).fillna(0).astype(np.int64).to_numpy(),
    )
    log.debug("libmatch: %d molecules, %d LIB rows, %.2f s", len(mols), len(rows),
              time.perf_counter() - t0)
    return state


def lib_candidates_empty() -> pd.DataFrame:
    """An empty LIB row frame with the right columns."""
    empty = {c: pd.Series([], dtype=object) for c in _ROW_COLUMNS}
    empty.update({c: pd.Series([], dtype=np.float64) for c in LIB_COLUMNS})
    return pd.DataFrame(empty)


pipeline.register_stage("lib", stage_lib)

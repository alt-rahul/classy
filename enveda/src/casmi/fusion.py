"""Deterministic fallback fusion (PLAN.md 4.6) and the Baseline 0 database prior (WP4).

One function, ``fuse(cands, cfg)``, scores a candidate DataFrame. Every input column is
optional: absent columns and missing values (NaN / pandas NA) are treated as "no
evidence", so the same call works for Baseline 0 (database columns only) and for later
WPs that add spectral and fingerprint features.

Columns read (all optional):

- source: ``tier`` ("lib" | "np" | "pubchem" | "gen"), ``in_np``, ``is_lib``, ``is_gen``.
  ``candidate_source`` folds them into one label, first match wins: lib (``tier == "lib"``
  or ``is_lib``), gen (``tier == "gen"`` or ``is_gen``), np (``in_np`` or
  ``tier == "np"``), else pubchem (PubChem-only).
- library match (WP8): ``lib_best_cos`` (best cosine over the candidate's references),
  ``lib_n_matched`` (matched peaks at that pair).
- fingerprint model (WP9/WP10): ``fp_ll`` (Bernoulli log-likelihood, higher is better).
- priors (db.OUT_COLUMNS): in_coconut, in_npatlas, in_lotus, in_pubchemlite,
  coconut_annotation_level, pubmed_count, pmid_count, n_citations, n_organisms,
  patent_count, anno_type_count, n_collections, n_cids.
- ``molecule_id``: the pool. Without it the whole frame is one pool.
- ``ppm_err``, ``ik14``: tie-breakers in ``sort_pool``.

Score = first applicable rule (PLAN 4.6):

1. LIB candidate with lib_best_cos >= ``cfg.fusion_lib_cos_min`` and lib_n_matched >=
   ``cfg.fusion_lib_min_matched``: 3 + lib_best_cos.
2. NP-tier or LIB candidate: 1 + z + min(``cfg.fusion_prior_coef`` x log1p(literature),
   ``cfg.fusion_prior_cap``), literature = max(pubmed_count, pmid_count).
3. PubChem-only candidate: z.
4. GEN candidate: ``cfg.fusion_gen_factor`` x z.

WP8 extension (``lib_extra_rules``, both off by default, so the rules above are unchanged
unless a config turns them on): rule 1b (``fusion_lib_weak``) scores a LIB candidate that
misses rule 1 but has lib_best_cos >= ``fusion_lib_weak_cos_min`` and lib_n_matched >=
``fusion_lib_weak_min_matched`` at 2 + cap + (1 - cap) x lib_best_cos (cap =
``fusion_prior_cap``): above every rule 2 score, below every rule 1 score; with
``fusion_lib_weak_additive`` it instead adds ``fusion_lib_weak_weight`` x clip((lib_best_cos -
weak_cos_min) / (1 - weak_cos_min), 0, 1) to the candidate's rule 2 score (prior and
similarity both count). Demotion
(``fusion_lib_demote``): a LIB candidate left for rule 2 whose lib_best_cos is measured and
below ``fusion_lib_demote_cos_max`` scores ``fusion_lib_demote_factor`` x z instead, i.e.
below every NP-tier candidate (its reference spectra contradict the query).

z is in [0, 1]. Fingerprint branch: in a pool where any candidate has ``fp_ll``, z is
the rank-normalised fp_ll within the pool (average ranks, (rank - 1) / (n - 1), a
single scored candidate gets 1; candidates without fp_ll get 0). Prior-only branch
(Baseline 0, no fingerprint model yet; PLAN 4.6 last line): z is the rank-normalised
``prior_score`` within the pool and band, the bands being {LIB, NP}, PubChem-only and
GEN. So every NP-tier candidate (score >= 1) ranks above every PubChem-only candidate
(score <= 1; an exact tie at 1 is broken by source order in ``sort_pool``), whatever
their counts.

``prior_score`` (documented monotone combination, weights in Config, all >= 0):

- NP-tier and LIB rows: w_np_sources x (number of in_coconut, in_npatlas, in_lotus,
  in_pubchemlite that are true) + w_annotation x coconut_annotation_level
  + w_literature x log1p(max(pubmed_count, pmid_count) + n_citations)
  + w_organisms x log1p(n_organisms) + w_patents x log1p(patent_count)
  + w_anno_types x log1p(anno_type_count) + w_collections x log1p(n_collections).
- PubChem-only rows: w_pmid x log1p(pmid_count) + w_n_cids x log1p(n_cids).
- GEN rows: 0.

Missing values count as 0 and negative counts are clipped to 0, so the prior never
decreases when a count or flag increases. The weights are initial guesses, to be
tuned on the WP7 validation protocol.

``sort_pool`` orders candidates by (molecule_id, score desc, source order LIB < NP <
PubChem < GEN, |ppm_err| asc, ik14 asc), stable, missing values last (PLAN 4.8 step 1).

Only numpy and pandas are used (inference-path safe).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import Config

__all__ = [
    "SOURCES",
    "SOURCE_RANK",
    "FP_COLUMN",
    "candidate_source",
    "literature_count",
    "prior_score",
    "rank_normalise",
    "add_base_features",
    "fuse",
    "sort_pool",
]

SOURCES = ("lib", "np", "pubchem", "gen")
SOURCE_RANK = {s: i for i, s in enumerate(SOURCES)}
FP_COLUMN = "fp_ll"

_NP_FLAGS = ("in_coconut", "in_npatlas", "in_lotus", "in_pubchemlite")
_BASE_FEATURES = ("source", "source_rank", "abs_ppm_err", "prior")


# --- column access with "absent or missing = no evidence" ------------------------------

def _num(df: pd.DataFrame, col: str) -> np.ndarray:
    """Column as float64 with NaN for missing values; all NaN if the column is absent."""
    if col not in df.columns:
        return np.full(len(df), np.nan, dtype=np.float64)
    s = df[col]
    try:
        return s.to_numpy(dtype=np.float64, na_value=np.nan)
    except (TypeError, ValueError):
        return pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)


def _count(df: pd.DataFrame, col: str) -> np.ndarray:
    """Non-negative count with missing -> 0."""
    v = _num(df, col)
    return np.clip(np.nan_to_num(v, nan=0.0), 0.0, None)


def _flag(df: pd.DataFrame, col: str) -> np.ndarray:
    """Boolean column with missing -> False; all False if absent."""
    if col not in df.columns:
        return np.zeros(len(df), dtype=bool)
    s = df[col]
    if s.dtype == bool:
        return s.to_numpy()
    return s.fillna(False).astype(bool).to_numpy()


def _text(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), "", dtype=object)
    return df[col].astype(object).where(df[col].notna(), "").to_numpy(dtype=object)


def _pool_codes(df: pd.DataFrame, group: str | None) -> np.ndarray:
    if group is None or group not in df.columns:
        return np.zeros(len(df), dtype=np.int64)
    codes, _ = pd.factorize(df[group].astype(object), sort=True)
    return codes.astype(np.int64)


# --- features --------------------------------------------------------------------------

def candidate_source(df: pd.DataFrame) -> np.ndarray:
    """One source label per row: "lib", "gen", "np" or "pubchem" (see module docstring)."""
    tier = _text(df, "tier")
    is_lib = (tier == "lib") | _flag(df, "is_lib")
    is_gen = ~is_lib & ((tier == "gen") | _flag(df, "is_gen"))
    is_np = ~is_lib & ~is_gen & (_flag(df, "in_np") | (tier == "np"))
    out = np.full(len(df), "pubchem", dtype=object)
    out[is_np] = "np"
    out[is_gen] = "gen"
    out[is_lib] = "lib"
    return out


def literature_count(df: pd.DataFrame) -> np.ndarray:
    """max(pubmed_count (PubChemLite), pmid_count (PubChem tier)), missing -> 0."""
    return np.maximum(_count(df, "pubmed_count"), _count(df, "pmid_count"))


def _check_weights(cfg: Config) -> None:
    names = ("fusion_w_np_sources", "fusion_w_annotation", "fusion_w_literature",
             "fusion_w_organisms", "fusion_w_patents", "fusion_w_anno_types",
             "fusion_w_collections", "fusion_w_pmid", "fusion_w_n_cids")
    bad = [n for n in names if getattr(cfg, n) < 0]
    if bad:
        raise ValueError(f"prior weights must be >= 0 (monotone prior): {bad}")


def prior_score(df: pd.DataFrame, cfg: Config | None = None,
                source: np.ndarray | None = None) -> np.ndarray:
    """Baseline 0 database prior per row (float64, >= 0). See the module docstring."""
    cfg = cfg or Config()
    _check_weights(cfg)
    src = candidate_source(df) if source is None else source
    n_src = np.zeros(len(df), dtype=np.float64)
    for f in _NP_FLAGS:
        n_src += _flag(df, f)
    ann = np.clip(_count(df, "coconut_annotation_level"), 0.0, 5.0)
    lit = literature_count(df) + _count(df, "n_citations")
    np_prior = (cfg.fusion_w_np_sources * n_src
                + cfg.fusion_w_annotation * ann
                + cfg.fusion_w_literature * np.log1p(lit)
                + cfg.fusion_w_organisms * np.log1p(_count(df, "n_organisms"))
                + cfg.fusion_w_patents * np.log1p(_count(df, "patent_count"))
                + cfg.fusion_w_anno_types * np.log1p(_count(df, "anno_type_count"))
                + cfg.fusion_w_collections * np.log1p(_count(df, "n_collections")))
    pc_prior = (cfg.fusion_w_pmid * np.log1p(_count(df, "pmid_count"))
                + cfg.fusion_w_n_cids * np.log1p(_count(df, "n_cids")))
    out = np.where((src == "np") | (src == "lib"), np_prior,
                   np.where(src == "pubchem", pc_prior, 0.0))
    return out.astype(np.float64)


def rank_normalise(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Average rank within each group mapped to [0, 1] (higher value -> closer to 1).

    (rank - 1) / (n - 1) with n the number of non-NaN values in the group; a group with
    one value gives 1.0; NaN values give NaN.
    """
    v = pd.Series(np.asarray(values, dtype=np.float64))
    g = pd.Series(np.asarray(groups))
    if len(v) == 0:
        return np.zeros(0, dtype=np.float64)
    rank = v.groupby(g, sort=True).rank(method="average").to_numpy(dtype=np.float64)
    n = v.groupby(g, sort=True).transform("count").to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(n > 1, (rank - 1.0) / (n - 1.0), 1.0)
    z[np.isnan(v.to_numpy())] = np.nan
    return z


def add_base_features(df: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Copy of ``df`` with ``source``, ``source_rank``, ``abs_ppm_err`` and ``prior``."""
    cfg = cfg or Config()
    out = df.copy()
    src = candidate_source(out)
    out["source"] = pd.Series(src, index=out.index, dtype=object)
    out["source_rank"] = np.array([SOURCE_RANK[s] for s in src], dtype=np.int8)
    out["abs_ppm_err"] = np.abs(_num(out, "ppm_err"))
    out["prior"] = prior_score(out, cfg, source=src)
    return out


# --- the fusion rules ------------------------------------------------------------------

def fuse(df: pd.DataFrame, cfg: Config | None = None, group: str | None = "molecule_id",
         return_z: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """PLAN 4.6 score per row (float64), aligned to ``df``. See the module docstring.

    ``group``: column defining the pool (None or absent: one pool). ``return_z`` also
    returns the z used (for logging and tests).
    """
    cfg = cfg or Config()
    n = len(df)
    if n == 0:
        empty = np.zeros(0, dtype=np.float64)
        return (empty, empty) if return_z else empty
    src = candidate_source(df)
    pool = _pool_codes(df, group)
    is_lib = src == "lib"
    is_np = src == "np"
    is_gen = src == "gen"

    fp = _num(df, FP_COLUMN)
    has_fp = ~np.isnan(fp)
    pool_has_fp = np.zeros(pool.max() + 1, dtype=bool)
    np.logical_or.at(pool_has_fp, pool, has_fp)
    use_fp = pool_has_fp[pool]

    z = np.zeros(n, dtype=np.float64)
    if use_fp.any():
        zf = rank_normalise(np.where(use_fp, fp, np.nan), np.where(use_fp, pool, -1))
        z[use_fp] = np.nan_to_num(zf[use_fp], nan=0.0)
    if (~use_fp).any():
        prior = _num(df, "prior") if "prior" in df.columns else prior_score(df, cfg, source=src)
        prior = np.nan_to_num(prior, nan=0.0)
        band = np.where(is_lib | is_np, 0, np.where(is_gen, 2, 1))
        key = pool * 3 + band
        zp = rank_normalise(np.where(use_fp, np.nan, prior), np.where(use_fp, -1, key))
        z[~use_fp] = zp[~use_fp]

    cos = _num(df, "lib_best_cos")
    matched = _num(df, "lib_n_matched")
    with np.errstate(invalid="ignore"):
        rule1 = (is_lib & (cos >= cfg.fusion_lib_cos_min)
                 & (matched >= cfg.fusion_lib_min_matched))
    bonus = np.minimum(cfg.fusion_prior_coef * np.log1p(literature_count(df)),
                       cfg.fusion_prior_cap)
    score = np.select(
        [rule1, is_lib | is_np, is_gen],
        [3.0 + np.nan_to_num(cos, nan=0.0), 1.0 + z + bonus, cfg.fusion_gen_factor * z],
        default=z,
    ).astype(np.float64)
    score = lib_extra_rules(score, z, cos, matched, is_lib, rule1, cfg)  # WP8; no-op by default
    return (score, z) if return_z else score


def sort_pool(df: pd.DataFrame, score_col: str = "score", group: str | None = "molecule_id",
              cfg: Config | None = None) -> pd.DataFrame:
    """Rows in emission order (PLAN 4.8 step 1), index reset.

    Keys: ``group`` asc (if present), ``score_col`` desc, source_rank asc, abs_ppm_err asc,
    ik14 asc; stable; missing values last. Missing base features are added first.
    """
    if any(c not in df.columns for c in ("source_rank", "abs_ppm_err")):
        df = add_base_features(df, cfg)
    if len(df) == 0:
        return df.reset_index(drop=True)
    keys: list[Any] = []
    asc: list[bool] = []
    if group is not None and group in df.columns:
        keys.append(group)
        asc.append(True)
    tmp = df.copy()
    tmp["_score"] = _num(df, score_col)
    tmp["_ik14"] = (df["ik14"].astype(object).where(df["ik14"].notna(), None)
                    if "ik14" in df.columns else None)
    keys += ["_score", "source_rank", "abs_ppm_err", "_ik14"]
    asc += [False, True, True, True]
    out = tmp.sort_values(keys, ascending=asc, kind="mergesort", na_position="last")
    return out.drop(columns=["_score", "_ik14"]).reset_index(drop=True)


# --- WP8: optional LIB rules (rule 1b and demotion; see the module docstring) ----------

def lib_extra_rules(score: np.ndarray, z: np.ndarray, cos: np.ndarray, matched: np.ndarray,
                    is_lib: np.ndarray, rule1: np.ndarray, cfg: Config) -> np.ndarray:
    """Apply rule 1b and the low-similarity demotion to LIB rows outside rule 1.

    Returns ``score`` unchanged (the same array) when both are off (the default).
    """
    weak_on = bool(getattr(cfg, "fusion_lib_weak", False))
    demote_on = bool(getattr(cfg, "fusion_lib_demote", False))
    if not (weak_on or demote_on) or not is_lib.any():
        return score
    out = score.copy()
    rest = is_lib & ~rule1
    measured = ~np.isnan(cos)
    weak = np.zeros(len(score), dtype=bool)
    if weak_on:
        with np.errstate(invalid="ignore"):
            weak = (rest & measured & (cos >= cfg.fusion_lib_weak_cos_min)
                    & (np.nan_to_num(matched, nan=0.0) >= cfg.fusion_lib_weak_min_matched))
        if bool(getattr(cfg, "fusion_lib_weak_additive", False)):
            span = max(1.0 - float(cfg.fusion_lib_weak_cos_min), 1e-9)
            g = np.clip((cos[weak] - cfg.fusion_lib_weak_cos_min) / span, 0.0, 1.0)
            out[weak] = score[weak] + cfg.fusion_lib_weak_weight * g
        else:
            cap = float(cfg.fusion_prior_cap)
            out[weak] = 2.0 + cap + (1.0 - cap) * np.clip(cos[weak], 0.0, 1.0)
    if demote_on:
        with np.errstate(invalid="ignore"):
            low = rest & ~weak & measured & (cos < cfg.fusion_lib_demote_cos_max)
        out[low] = cfg.fusion_lib_demote_factor * z[low]
    return out

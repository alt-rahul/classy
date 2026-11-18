"""End-to-end inference: test spectra -> up to 25 ranked SMILES per molecule (WP4).

PLAN.md section 7 ("pipeline.py is the only shared hot spot"), 4.6 and 4.8.

Stages
------
A run is a sequence of named stages, each a function ``(state, cfg) -> state``, looked
up by name in the registry ``STAGES``. The order is ``cfg.pipeline_stages`` (default
``("mass", "pool", "features", "score", "assemble")``). Baseline 0 registers:

- ``mass``: ``io.mass_tables`` -> ``state.molecules`` (one row per molecule, M, PLAN 3.2)
  and ``state.spectra`` (per spectrum).
- ``pool``: ``db.lookup_batch`` at ``cfg.mass_tol_ppm`` over ``cfg.db_tiers``. A molecule
  whose pool is empty is retried at ``cfg.mass_tol_wide_ppm``, then with every spectrum's
  adduct replaced by the default of its polarity (``[M+H]+`` / ``[M-H]-``) at both
  tolerances (PLAN risk 11). ``state.pool``: ``molecule_id``, ``query_mass`` (= M_pool),
  then ``db.OUT_COLUMNS``; rows grouped by molecule_id.
- ``features``: ``fusion.add_base_features`` (source, source_rank, abs_ppm_err, prior).
- ``score``: ``pool["score"] = fusion.fuse(pool, cfg)`` (PLAN 4.6; without fingerprint
  scores it is the prior-only branch).
- ``assemble`` (PLAN 4.8): sort with ``fusion.sort_pool`` (score desc, source order,
  |ppm| asc, ik14 asc); walk down emitting canonical non-stereo SMILES (``chem.
  canonical_smiles``, validated by re-parsing) and skipping candidates whose metric key
  (``metric.mkeys``: disk cache, per-molecule timeout with ik14 fallback) was already
  emitted, until ``cfg.top_k``. Short lists are padded from the widened window
  (``cfg.mass_tol_wide_ppm`` around M_pool, same tiers, new ik14 only), then from the
  ``cfg.pad_nearest_n`` NP-tier rows nearest in mass, both ordered by the same fusion
  score (prior-only for these rows). Never fewer than 1 guess: if nothing at all can be
  emitted the molecule gets ``LAST_RESORT_SMILES``.

Adding or replacing a stage (later WPs)
---------------------------------------
In your own module::

    from casmi import pipeline

    @pipeline.register_stage("pool")            # replace a stage by name ...
    def pool_with_library(state, cfg):
        state = pipeline.stage_pool(state, cfg)  # ... reusing the default if useful
        ...                                      # append LIB rows (tier "lib")
        return state

    pipeline.register_stage("lib_features", my_fn)   # or add a new name

and list the module and, for a new name, the order in the run's config::

    pipeline_plugins: [casmi.libmatch]
    pipeline_stages: [mass, pool, lib_features, features, score, assemble]

``run`` imports every module in ``cfg.pipeline_plugins`` (in order) before resolving
stage names, so registration happens at import. ``run(..., stages={"score": fn})``
overrides stages for one call without touching the registry (tests, ablations).
``register_stage(name, fn, replace=False)`` refuses to overwrite. Stages may keep their
own data in ``state.extras[<their name>]``; ``run(..., extras=...)`` seeds it per batch
(e.g. a library handle or a validation exclusion mask).

State conventions: ``state.pool`` has one row per (molecule_id, candidate); later stages
add columns (missing columns are "no evidence" for ``fusion``). ``state.molecules`` is
the per-molecule report: stages add columns to it and ``run`` returns the concatenation.

Batching and failure isolation
------------------------------
Molecules are processed in batches of ``cfg.batch_molecules``, ordered by M (so
consecutive batches reuse PubChem row groups). If any stage raises, the batch is rerun
one molecule at a time; a molecule that still fails gets padding only (widened window
and nearest NP rows; then without metric keys; then ``LAST_RESORT_SMILES``) and the
exception is logged. A run never crashes because of one molecule; a missing stage name
or an unimportable plugin is a configuration error and raises before any work.

Inference-path rule (PLAN 7): this module and everything it imports use only numpy,
pandas, pyarrow, scipy, numba, rdkit, lightgbm (tests/test_pipeline.py enforces it).
"""

from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import db as cdb
from . import fusion, io, metric
from .adducts import default_adduct
from .chem import canonical_smiles
from .config import Config

__all__ = [
    "State",
    "StageFn",
    "STAGES",
    "DEFAULT_STAGES",
    "LAST_RESORT_SMILES",
    "PipelineResult",
    "register_stage",
    "get_stage",
    "stage_names",
    "load_plugins",
    "resolve_stages",
    "run_stages",
    "run",
    "predict",
    "summarize",
    "limit_threads",
    "stage_mass",
    "stage_pool",
    "stage_features",
    "stage_score",
    "stage_assemble",
]

log = logging.getLogger(__name__)

DEFAULT_STAGES: tuple[str, ...] = ("mass", "pool", "features", "score", "assemble")
# Phenol: emitted only when no database row can be used for a molecule at all.
LAST_RESORT_SMILES = "Oc1ccccc1"
_ORIGINS = ("pool", "wide", "nearest", "last_resort")


@dataclass
class State:
    """Everything the stages of one batch share. Stages mutate and return it."""

    test_df: pd.DataFrame                      # spectra of this batch (io.read_test schema)
    db: Any = None                             # casmi.db.CandidateDB
    molecules: pd.DataFrame | None = None      # one row per molecule_id (+ report columns)
    spectra: pd.DataFrame | None = None        # per-spectrum mass table (io.mass_tables)
    pool: pd.DataFrame | None = None           # candidates, molecule_id + db.OUT_COLUMNS + ...
    preds: dict[str, list[str]] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def molecule_ids(self) -> list[str]:
        return sorted(set(self.test_df["molecule_id"].astype(str)))


StageFn = Callable[[State, Config], State]
STAGES: dict[str, StageFn] = {}


# --- registry --------------------------------------------------------------------------

def register_stage(name: str, fn: StageFn | None = None, *, replace: bool = True) -> Any:
    """Register ``fn`` as stage ``name``; without ``fn``, return a decorator.

    ``replace=False`` raises ValueError if ``name`` is already registered.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("stage name must be a non-empty string")

    def _add(f: StageFn) -> StageFn:
        if not callable(f):
            raise TypeError(f"stage {name!r} is not callable")
        if not replace and name in STAGES:
            raise ValueError(f"stage {name!r} already registered")
        if name in STAGES and STAGES[name] is not f:
            log.info("stage %r replaced by %s.%s", name, getattr(f, "__module__", "?"),
                     getattr(f, "__qualname__", repr(f)))
        STAGES[name] = f
        return f

    return _add if fn is None else _add(fn)


def get_stage(name: str) -> StageFn:
    try:
        return STAGES[name]
    except KeyError:
        raise KeyError(f"unknown pipeline stage {name!r}; registered: {stage_names()}") from None


def stage_names() -> tuple[str, ...]:
    return tuple(sorted(STAGES))


def load_plugins(cfg: Config) -> list[str]:
    """Import every module in ``cfg.pipeline_plugins`` (in order); returns their names."""
    names = [str(n) for n in cfg.pipeline_plugins]
    for n in names:
        importlib.import_module(n)
    return names


def resolve_stages(cfg: Config, stages: Mapping[str, StageFn] | None = None
                   ) -> list[tuple[str, StageFn]]:
    """``[(name, fn)]`` in ``cfg.pipeline_stages`` order; ``stages`` overrides by name."""
    over = dict(stages or {})
    out = []
    for name in cfg.pipeline_stages:
        out.append((name, over[name] if name in over else get_stage(name)))
    return out


def run_stages(state: State, cfg: Config, stages: Sequence[tuple[str, StageFn]] | None = None
               ) -> State:
    """Run the stages on one batch. Exceptions propagate (``run`` isolates them)."""
    for name, fn in (stages if stages is not None else resolve_stages(cfg)):
        t0 = time.perf_counter()
        n_in = 0 if state.pool is None else len(state.pool)
        out = fn(state, cfg)
        if not isinstance(out, State):
            raise TypeError(f"stage {name!r} returned {type(out).__name__}, expected State")
        state = out
        log.debug("stage %s: %d molecules, pool rows %d -> %d, %.2f s", name,
                  len(state.molecule_ids), n_in, 0 if state.pool is None else len(state.pool),
                  time.perf_counter() - t0)
    return state


# --- default stages --------------------------------------------------------------------

def _db(state: State, cfg: Config) -> Any:
    if state.db is None:
        state.db = cdb.get_db(cfg)
    return state.db


def stage_mass(state: State, cfg: Config) -> State:
    """Neutral mass per molecule (PLAN 3.2)."""
    state.molecules, state.spectra = io.mass_tables(state.test_df, cfg)
    return state


def _default_adduct_masses(test_df: pd.DataFrame, ids: Sequence[str], cfg: Config
                           ) -> dict[str, float]:
    """M per molecule with every spectrum's adduct replaced by its polarity default.
    Only molecules where at least one adduct changes are returned."""
    sub = test_df[test_df["molecule_id"].astype(str).isin(set(ids))].copy()
    if not len(sub):
        return {}
    new = []
    for a, mode in zip(sub["adduct"], sub["ionization_mode"]):
        try:
            new.append(default_adduct(mode))
        except ValueError:
            new.append(a)
    changed = {str(m) for m, a, b in zip(sub["molecule_id"], sub["adduct"], new) if a != b}
    if not changed:
        return {}
    sub["adduct"] = new
    mols = io.molecule_table(sub, cfg)
    return {str(m): float(x) for m, x in zip(mols["molecule_id"], mols["M"]) if str(m) in changed}


def stage_pool(state: State, cfg: Config) -> State:
    """Database candidates by mass with the empty-pool retries of PLAN risk 11."""
    dbh = _db(state, cfg)
    mols = state.molecules.copy()
    ids = mols["molecule_id"].astype(str).to_numpy(dtype=object)
    M = mols["M"].to_numpy(dtype=np.float64)
    n = len(ids)
    attempt = np.full(n, "empty", dtype=object)
    m_pool = M.copy()
    parts: list[pd.DataFrame] = []
    first: pd.DataFrame | None = None

    def look(sel: np.ndarray, masses: np.ndarray, tol: float, label: str) -> np.ndarray:
        nonlocal first
        if not len(sel):
            return sel
        r = dbh.lookup_batch(masses, tol_ppm=tol, tiers=cfg.db_tiers, keys=list(ids[sel]))
        if first is None:
            first = r
        found = set(r["query"].astype(str)) if len(r) else set()
        hit = np.array([ids[i] in found for i in sel], dtype=bool)
        attempt[sel[hit]] = label
        m_pool[sel[hit]] = masses[hit]
        if len(r):
            parts.append(r)
        return sel[~hit]

    rest = look(np.arange(n), M, cfg.mass_tol_ppm, "tol")
    rest = look(rest, M[rest], cfg.mass_tol_wide_ppm, "wide")
    if len(rest):
        md = _default_adduct_masses(state.test_df, list(ids[rest]), cfg)
        sel = np.array([i for i in rest if ids[i] in md], dtype=np.int64)
        if len(sel):
            mass_d = np.array([md[ids[i]] for i in sel], dtype=np.float64)
            left = look(sel, mass_d, cfg.mass_tol_ppm, "default_adduct")
            pos = {int(i): k for k, i in enumerate(sel)}
            look(left, mass_d[[pos[int(i)] for i in left]], cfg.mass_tol_wide_ppm,
                 "default_adduct_wide")

    if parts:
        pool = parts[0] if len(parts) == 1 else pd.concat(parts, ignore_index=True)
    else:
        pool = first if first is not None else dbh.lookup_batch(np.zeros(0), keys=[])
    pool = pool.rename(columns={"query": "molecule_id"})
    pool["molecule_id"] = pool["molecule_id"].astype(object)
    pool = pool.sort_values("molecule_id", kind="mergesort").reset_index(drop=True)

    def counts(df: pd.DataFrame | None, mask: np.ndarray | None = None) -> np.ndarray:
        if df is None or not len(df):
            return np.zeros(n, dtype=np.int64)
        s = df["query" if "query" in df.columns else "molecule_id"].astype(str)
        if mask is not None:
            s = s[mask]
        c = s.value_counts()
        return np.array([int(c.get(i, 0)) for i in ids], dtype=np.int64)

    mols["M_pool"] = m_pool
    mols["pool_attempt"] = attempt
    mols["pool_n_first"] = counts(first)
    mols["pool_n_np_first"] = counts(first, None if first is None or not len(first)
                                     else first["in_np"].to_numpy(dtype=bool))
    mols["pool_n"] = counts(pool)
    mols["pool_n_np"] = counts(pool, pool["in_np"].to_numpy(dtype=bool) if len(pool) else None)
    state.molecules = mols
    state.pool = pool
    return state


def stage_features(state: State, cfg: Config) -> State:
    """Baseline 0 features: source, source_rank, abs_ppm_err, prior (fusion)."""
    if state.pool is not None:
        state.pool = fusion.add_base_features(state.pool, cfg)
    return state


def stage_score(state: State, cfg: Config) -> State:
    """PLAN 4.6 fusion score (prior-only until fingerprint scores exist)."""
    if state.pool is not None:
        state.pool["score"] = fusion.fuse(state.pool, cfg)
    return state


# --- assembly (PLAN 4.8) ---------------------------------------------------------------

@dataclass
class _Acc:
    smiles: list[str] = field(default_factory=list)
    seen: set = field(default_factory=set)
    counts: dict = field(default_factory=lambda: dict.fromkeys(_ORIGINS, 0))
    n_unparseable: int = 0
    n_dup: int = 0
    n_nokey: int = 0
    error: str = ""


_Lists = dict  # molecule_id -> (smiles, ik14, origin) object arrays in emission order


def _lists_by_molecule(df: pd.DataFrame, origin: str) -> _Lists:
    """Per-molecule views of a frame grouped by molecule_id (order inside kept)."""
    if df is None or not len(df):
        return {}
    def text(col: str) -> np.ndarray:
        vals = df[col].astype(object).to_numpy(dtype=object)
        return np.array([v if isinstance(v, str) and v else None for v in vals], dtype=object)

    mids = df["molecule_id"].astype(str).to_numpy(dtype=object)
    sm = text("smiles")
    ik = text("ik14")
    org = (df["origin"].astype(object).to_numpy(dtype=object) if "origin" in df.columns
           else np.full(len(df), origin, dtype=object))
    cuts = np.flatnonzero(mids[1:] != mids[:-1]) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(mids)]
    out: _Lists = {}
    for a, b in zip(starts, ends):
        mid = mids[a]
        if mid in out:
            raise ValueError(f"candidate frame not grouped by molecule_id ({mid})")
        out[mid] = (sm[a:b], ik[a:b], org[a:b])
    return out


def _walk(lists: _Lists, acc: dict[str, _Acc], cfg: Config, canon: dict[Any, str | None],
          use_mkeys: bool = True) -> None:
    """Emit candidates in list order into ``acc`` until each molecule has ``cfg.top_k``.

    Rounds: every molecule still short sends its next (needed + margin) candidates;
    their canonical SMILES get metric keys in one ``mkeys`` call per round; a candidate
    is skipped if it does not parse or its key was already emitted. A key that cannot be
    computed falls back to the row's ik14 (else the SMILES), counted in ``n_nokey``.
    """
    k = cfg.top_k
    ptr = {mid: 0 for mid in lists}
    while True:
        chunks = {}
        for mid in sorted(lists):
            need = k - len(acc[mid].smiles)
            n = len(lists[mid][0])
            if need > 0 and ptr[mid] < n:
                chunks[mid] = (ptr[mid], min(n, ptr[mid] + need + max(0, cfg.assemble_mkey_margin)))
        if not chunks:
            return
        todo: set[str] = set()
        for mid, (a, b) in chunks.items():
            for s in lists[mid][0][a:b]:
                if s is None:
                    continue
                if s not in canon:
                    canon[s] = canonical_smiles(s)
                if canon[s] is not None:
                    todo.add(canon[s])
        uniq = sorted(todo)
        if use_mkeys and uniq:
            keys = dict(zip(uniq, metric.mkeys(uniq, cfg, n_jobs=cfg.n_jobs, cache=cfg.mkey_cache)))
        else:
            keys = {}
        for mid in sorted(chunks):
            a, b = chunks[mid]
            sm, ik, org = lists[mid]
            ac = acc[mid]
            try:
                for j in range(a, b):
                    if len(ac.smiles) >= k:
                        break
                    c = canon.get(sm[j]) if sm[j] is not None else None
                    if c is None:
                        ac.n_unparseable += 1
                        continue
                    key = keys.get(c)
                    if key is None:
                        if use_mkeys:
                            ac.n_nokey += 1
                        key = ik[j] if ik[j] is not None else c
                    if key in ac.seen:
                        ac.n_dup += 1
                        continue
                    ac.seen.add(key)
                    ac.smiles.append(c)
                    ac.counts[str(org[j]) if org[j] in ac.counts else "pool"] += 1
                ptr[mid] = b
            except Exception as exc:  # noqa: BLE001 - one molecule must not stop the rest
                log.exception("assemble: molecule %s failed while walking candidates", mid)
                ac.error = ac.error or f"walk: {type(exc).__name__}: {exc}"
                ptr[mid] = len(sm)


def _nearest_np(dbh: Any, ids: Sequence[str], masses: Sequence[float], n_rows: int
                ) -> pd.DataFrame | None:
    """The ``n_rows`` NP-tier rows nearest in mass to each molecule (ties: row order)."""
    if n_rows <= 0 or not len(ids):
        return None
    npdf = dbh.load_np()
    nm = dbh.np_mass
    if not len(nm):
        return None
    frames = []
    for mid, m in zip(ids, masses):
        if not np.isfinite(m):
            continue
        i = int(np.searchsorted(nm, m))
        lo, hi = max(0, i - n_rows), min(len(nm), i + n_rows)
        idx = np.arange(lo, hi)
        order = np.lexsort((idx, np.abs(nm[idx] - m)))[:n_rows]
        rows = npdf.iloc[np.sort(idx[order])].reset_index(drop=True)
        rows.insert(0, "molecule_id", mid)
        rows.insert(1, "query_mass", float(m))
        mm = rows["mono_mass"].to_numpy(dtype=np.float64)
        rows.insert(rows.columns.get_loc("mono_mass") + 1, "ppm_err", (mm - m) / m * 1e6)
        frames.append(rows)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _padding_lists(state: State, ids: Sequence[str], cfg: Config) -> _Lists:
    """Padding candidates per molecule (PLAN 4.8 step 3): widened window, then nearest
    NP-tier rows, each ordered by the fusion score; ik14 already in the pool skipped."""
    dbh = _db(state, cfg)
    mols = state.molecules.set_index(state.molecules["molecule_id"].astype(str))
    mcol = "M_pool" if "M_pool" in mols.columns else "M"
    masses = mols.loc[list(ids), mcol].to_numpy(dtype=np.float64)
    taken: set[tuple[str, str]] = set()
    if state.pool is not None and len(state.pool):
        taken = set(zip(state.pool["molecule_id"].astype(str), state.pool["ik14"].astype(str)))
    frames = []
    ok = np.isfinite(masses)
    try:
        if ok.any():
            wide = dbh.lookup_batch(masses[ok], tol_ppm=cfg.mass_tol_wide_ppm, tiers=cfg.db_tiers,
                                    keys=[i for i, f in zip(ids, ok) if f])
            wide = wide.rename(columns={"query": "molecule_id"})
            if len(wide):
                wide["molecule_id"] = wide["molecule_id"].astype(str).astype(object)
                keep = [(m, k) not in taken for m, k in
                        zip(wide["molecule_id"], wide["ik14"].astype(str))]
                wide = wide[np.array(keep, dtype=bool)]
                taken |= set(zip(wide["molecule_id"], wide["ik14"].astype(str)))
                wide = wide.assign(origin="wide")
                frames.append(wide)
    except Exception:  # noqa: BLE001 - padding must go on with the nearest rows
        log.exception("assemble: widened-window padding failed for %d molecules", len(ids))
    try:
        near = _nearest_np(dbh, ids, masses, cfg.pad_nearest_n)
        if near is not None and len(near):
            keep = [(m, k) not in taken for m, k in zip(near["molecule_id"], near["ik14"].astype(str))]
            near = near[np.array(keep, dtype=bool)].assign(origin="nearest")
            frames.append(near)
    except Exception:  # noqa: BLE001
        log.exception("assemble: nearest-mass padding failed for %d molecules", len(ids))
    scored = []
    for f in frames:
        if not len(f):
            continue
        f = fusion.add_base_features(f.reset_index(drop=True), cfg)
        f["score"] = fusion.fuse(f, cfg)
        scored.append(fusion.sort_pool(f, cfg=cfg))
    if not scored:
        return {}
    cols = ["molecule_id", "smiles", "ik14", "origin"]
    allp = pd.concat([f[cols] for f in scored], ignore_index=True)
    allp = allp.sort_values("molecule_id", kind="mergesort").reset_index(drop=True)
    return _lists_by_molecule(allp, "wide")


def stage_assemble(state: State, cfg: Config) -> State:
    """Final lists (PLAN 4.8): sort, mkey-dedup walk, padding, canonical SMILES."""
    use_mkeys = bool(state.extras.get("assemble_use_mkeys", True))
    mols = state.molecules
    ids = [str(m) for m in mols["molecule_id"]]
    acc = {mid: _Acc() for mid in ids}
    canon: dict[Any, str | None] = {}
    pool = state.pool
    if pool is not None and len(pool):
        if "score" not in pool.columns:
            pool = pool.assign(score=fusion.fuse(pool, cfg))
        pool = fusion.sort_pool(pool, cfg=cfg)
        state.pool = pool
        lists = {m: v for m, v in _lists_by_molecule(pool, "pool").items() if m in acc}
        _walk(lists, acc, cfg, canon, use_mkeys)
    short = [m for m in ids if len(acc[m].smiles) < cfg.top_k]
    if short:
        pads = {m: v for m, v in _padding_lists(state, short, cfg).items() if m in acc}
        _walk(pads, acc, cfg, canon, use_mkeys)
    for mid in ids:
        ac = acc[mid]
        if not ac.smiles:
            ac.smiles = [LAST_RESORT_SMILES]
            ac.counts["last_resort"] += 1
        state.preds[mid] = list(ac.smiles[:cfg.top_k])
    mols = mols.copy()
    mols["n_guesses"] = [len(state.preds[m]) for m in ids]
    mols["n_from_pool"] = [acc[m].counts["pool"] for m in ids]
    mols["n_pad_wide"] = [acc[m].counts["wide"] for m in ids]
    mols["n_pad_nearest"] = [acc[m].counts["nearest"] for m in ids]
    mols["n_last_resort"] = [acc[m].counts["last_resort"] for m in ids]
    mols["n_skip_unparseable"] = [acc[m].n_unparseable for m in ids]
    mols["n_skip_dup_mkey"] = [acc[m].n_dup for m in ids]
    mols["n_mkey_missing"] = [acc[m].n_nokey for m in ids]
    mols["error"] = [acc[m].error for m in ids]
    state.molecules = mols
    return state


for _name, _fn in (("mass", stage_mass), ("pool", stage_pool), ("features", stage_features),
                   ("score", stage_score), ("assemble", stage_assemble)):
    register_stage(_name, _fn)
del _name, _fn


# --- running ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    preds: dict[str, list[str]]     # molecule_id -> SMILES, best first, 1 to top_k each
    molecules: pd.DataFrame         # per-molecule report (sorted by molecule_id)
    wall_s: float = 0.0


def _rough_mass(sub: pd.DataFrame) -> float:
    """Fallback M when io.mass_tables fails: median precursor m/z minus a proton."""
    mz = pd.to_numeric(sub["precursor_mz"], errors="coerce").to_numpy(dtype=np.float64)
    sign = np.where(sub["ionization_mode"].astype(str).str.lower().str.startswith("neg"), -1.0, 1.0)
    m = mz - sign * 1.007276
    m = m[np.isfinite(m)]
    return float(np.median(m)) if len(m) else float("nan")


def _fallback(mid: str, sub: pd.DataFrame, cfg: Config, dbh: Any, extras: Mapping[str, Any],
              reason: str) -> tuple[list[str], dict[str, Any]]:
    """Padding-only guesses for a molecule whose stages failed."""
    row: dict[str, Any] = {"molecule_id": mid, "error": reason}
    try:
        mols = io.molecule_table(sub, cfg)
        M = float(mols["M"].iloc[0])
    except Exception:  # noqa: BLE001
        M = _rough_mass(sub)
    row.update(M=M, M_pool=M, pool_attempt="fallback")
    for use_mkeys in (True, False):
        try:
            st = State(test_df=sub, db=dbh, extras={**extras, "assemble_use_mkeys": use_mkeys})
            st.molecules = pd.DataFrame({"molecule_id": [mid], "M": [M], "M_pool": [M]})
            st = stage_assemble(st, cfg)
            rep = st.molecules.iloc[0].to_dict()
            rep.update(error=reason, pool_attempt="fallback")
            return st.preds[mid], rep
        except Exception:  # noqa: BLE001
            log.exception("fallback padding (mkeys=%s) failed for %s", use_mkeys, mid)
    row.update(n_guesses=1, n_last_resort=1)
    return [LAST_RESORT_SMILES], row


def _valid_guesses(g: Any, k: int) -> bool:
    if not isinstance(g, (list, tuple)) or not 1 <= len(g) <= k:
        return False
    return all(isinstance(s, str) and s.strip() and not any(c in s for c in ";,\n\"")
               for s in g)


def _batches(test_df: pd.DataFrame, cfg: Config) -> list[list[str]]:
    ids = sorted(set(test_df["molecule_id"].astype(str)))
    try:
        mols = io.molecule_table(test_df, cfg)
        mols = mols.assign(_id=mols["molecule_id"].astype(str))
        order = mols.sort_values(["M", "_id"], kind="mergesort")["_id"].tolist()
        have = set(order)
        order += [i for i in ids if i not in have]
    except Exception:  # noqa: BLE001 - batching by id only costs cache locality
        log.exception("pipeline: mass table for batching failed; batching by molecule_id")
        order = ids
    size = max(1, int(cfg.batch_molecules))
    return [order[i:i + size] for i in range(0, len(order), size)]


def run(test_df: pd.DataFrame, cfg: Config | None = None, *, db: Any = None,
        stages: Mapping[str, StageFn] | None = None,
        extras: Mapping[str, Any] | None = None) -> PipelineResult:
    """Run the pipeline on a test-shaped frame (``io.read_test`` output).

    ``db``: a ``casmi.db.CandidateDB`` (default ``db.get_db(cfg)``). ``stages``: per-call
    stage overrides by name. ``extras``: copied into every batch's ``state.extras``.
    Returns predictions for every molecule_id in ``test_df`` and the per-molecule report.
    """
    t0 = time.perf_counter()
    cfg = cfg or Config()
    load_plugins(cfg)
    fns = resolve_stages(cfg, stages)
    dbh = db if db is not None else cdb.get_db(cfg)
    extras = dict(extras or {})
    df = test_df.copy()
    df["molecule_id"] = df["molecule_id"].astype(str)
    pos = df.groupby("molecule_id", sort=True).indices
    batches = _batches(df, cfg)
    log.info("pipeline: %d spectra, %d molecules, %d batches of <= %d, stages %s, tiers %s",
             len(df), len(pos), len(batches), cfg.batch_molecules,
             [n for n, _ in fns], list(cfg.db_tiers))
    preds: dict[str, list[str]] = {}
    reports: list[pd.DataFrame] = []
    fallback_rows: list[dict[str, Any]] = []

    def subset(ids: Sequence[str]) -> pd.DataFrame:
        rows = np.sort(np.concatenate([pos[i] for i in ids]))
        return df.iloc[rows].reset_index(drop=True)

    def one(ids: list[str]) -> None:
        sub = subset(ids)
        try:
            st = run_stages(State(test_df=sub, db=dbh, extras=dict(extras)), cfg, fns)
            bad = [i for i in ids if not _valid_guesses(st.preds.get(i), cfg.top_k)]
            if bad:
                raise RuntimeError(f"stages left invalid or missing guesses for {bad[:5]}")
            for i in ids:
                preds[i] = list(st.preds[i])
            if st.molecules is not None:
                rep = st.molecules[st.molecules["molecule_id"].astype(str).isin(set(ids))]
                reports.append(rep)
        except Exception as exc:  # noqa: BLE001 - isolate the failing molecule
            if len(ids) > 1:
                log.exception("pipeline: batch of %d failed; rerunning one molecule at a time",
                              len(ids))
                for i in ids:
                    one([i])
                return
            mid = ids[0]
            log.exception("pipeline: molecule %s failed; padding only", mid)
            g, row = _fallback(mid, sub, cfg, dbh, extras, f"{type(exc).__name__}: {exc}")
            preds[mid] = g
            fallback_rows.append(row)

    for b, ids in enumerate(batches):
        tb = time.perf_counter()
        one(ids)
        log.info("pipeline: batch %d/%d (%d molecules) %.1f s", b + 1, len(batches), len(ids),
                 time.perf_counter() - tb)
    try:
        dbh.release()
    except Exception:  # noqa: BLE001
        pass
    for mid in sorted(pos):  # belt and braces: every molecule gets a valid list
        if not _valid_guesses(preds.get(mid), cfg.top_k):
            log.error("pipeline: molecule %s has no valid guesses; padding only", mid)
            g, row = _fallback(mid, subset([mid]), cfg, dbh, extras, "missing after run")
            preds[mid] = g
            fallback_rows.append(row)
    parts = [r for r in reports if len(r)]
    if fallback_rows:
        parts.append(pd.DataFrame(fallback_rows))
    report = (pd.concat(parts, ignore_index=True) if parts
              else pd.DataFrame({"molecule_id": pd.Series([], dtype=object)}))
    report["molecule_id"] = report["molecule_id"].astype(str)
    report = (report.drop_duplicates("molecule_id", keep="last")
              .sort_values("molecule_id", kind="mergesort").reset_index(drop=True))
    if "error" in report.columns:
        report["error"] = report["error"].fillna("").astype(str)
    res = PipelineResult(preds={m: preds[m] for m in sorted(preds)}, molecules=report,
                         wall_s=time.perf_counter() - t0)
    for line in _summary_lines(summarize(report)):
        log.info("pipeline: %s", line)
    log.info("pipeline: done in %.1f s", res.wall_s)
    return res


def predict(test_df: pd.DataFrame, cfg: Config | None = None, **kwargs: Any) -> dict[str, list[str]]:
    """molecule_id -> up to ``cfg.top_k`` SMILES, best first (``run(...).preds``)."""
    return run(test_df, cfg, **kwargs).preds


# --- reporting -------------------------------------------------------------------------

def _quantiles(x: pd.Series) -> dict[str, float]:
    v = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if not len(v):
        return {}
    q = np.quantile(v, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {"min": q[0], "p05": q[1], "p25": q[2], "median": q[3], "p75": q[4], "p95": q[5],
            "max": q[6], "mean": float(v.mean())}


def summarize(report: pd.DataFrame) -> dict[str, Any]:
    """Run statistics from the per-molecule report (``PipelineResult.molecules``)."""
    n = len(report)
    out: dict[str, Any] = {"n_molecules": n}
    if not n:
        return out

    def col(c: str) -> pd.Series:
        return (pd.to_numeric(report[c], errors="coerce").fillna(0) if c in report.columns
                else pd.Series(np.zeros(n)))

    if "pool_n_np_first" in report.columns:
        out["frac_empty_np_pool_tol"] = float((col("pool_n_np_first") == 0).mean())
        out["frac_empty_pool_tol"] = float((col("pool_n_first") == 0).mean())
    if "pool_attempt" in report.columns:
        out["pool_attempts"] = {str(k): int(v) for k, v in
                                report["pool_attempt"].fillna("none").value_counts().sort_index().items()}
    out["pool_n"] = _quantiles(col("pool_n"))
    out["pool_n_np"] = _quantiles(col("pool_n_np"))
    pad = col("n_pad_wide") + col("n_pad_nearest") + col("n_last_resort")
    out["n_padded"] = int((pad > 0).sum())
    out["n_pad_wide"] = int((col("n_pad_wide") > 0).sum())
    out["n_pad_nearest"] = int((col("n_pad_nearest") > 0).sum())
    out["n_last_resort"] = int((col("n_last_resort") > 0).sum())
    out["n_guesses"] = _quantiles(col("n_guesses"))
    out["n_full"] = int((col("n_guesses") >= 25).sum())
    out["n_skip_unparseable"] = int(col("n_skip_unparseable").sum())
    out["n_skip_dup_mkey"] = int(col("n_skip_dup_mkey").sum())
    out["n_mkey_missing"] = int(col("n_mkey_missing").sum())
    err = report["error"].fillna("").astype(str) if "error" in report.columns else pd.Series([""] * n)
    out["n_errors"] = int((err != "").sum())
    return out


def _fmt(d: Mapping[str, float]) -> str:
    return ", ".join(f"{k} {v:.4g}" for k, v in d.items())


def _summary_lines(s: Mapping[str, Any]) -> list[str]:
    lines = [f"molecules {s.get('n_molecules', 0)}"]
    if "frac_empty_np_pool_tol" in s:
        lines.append(f"empty NP-tier pool at the standard window: "
                     f"{s['frac_empty_np_pool_tol']:.4f}; empty pool (all tiers queried): "
                     f"{s['frac_empty_pool_tol']:.4f}")
    if "pool_attempts" in s:
        lines.append(f"pool attempts: {s['pool_attempts']}")
    if s.get("pool_n"):
        lines.append(f"pool size: {_fmt(s['pool_n'])}")
    if s.get("pool_n_np"):
        lines.append(f"NP-tier pool size: {_fmt(s['pool_n_np'])}")
    if "n_padded" in s:
        lines.append(f"padded molecules: {s['n_padded']} (widened window {s['n_pad_wide']}, "
                     f"nearest NP {s['n_pad_nearest']}, last resort {s['n_last_resort']})")
    if s.get("n_guesses"):
        lines.append(f"guesses per molecule: {_fmt(s['n_guesses'])}; full lists {s['n_full']}")
    if "n_errors" in s:
        lines.append(f"skipped: unparseable {s['n_skip_unparseable']}, duplicate mkey "
                     f"{s['n_skip_dup_mkey']}; mkey missing (ik14 used) {s['n_mkey_missing']}; "
                     f"molecules with errors {s['n_errors']}")
    return lines


def limit_threads(n_threads: int) -> None:
    """Cap the Arrow thread pool (parquet decoding) at ``n_threads``."""
    import pyarrow as pa

    pa.set_cpu_count(max(1, int(n_threads)))

"""Analog evidence features by mass-shift propagation (ROADMAP R6, PLAN.md 4.3).

Why: in the public 0.34 pipeline (KAGGLE_INTEL 2, RESEARCH 7.1) the core class-2 channel is
"analog propagation": library spectra that match the query under a constant mass shift lend
their structures, and a candidate is supported in proportion to its structural similarity to
those analog structures, weighted by spectral similarity to a power (3 to 4). A class-2 query
has no spectrum of its own in the library, but its biosynthetic relatives (a methyl, a
hydroxyl, a sugar away) usually have one; their fragments and neutral losses survive the edit.

Hits (``search_hits``, ``select_hits``)
---------------------------------------
For each query consensus spectrum of the molecule (``library.QuerySet``: one per polarity and
adduct family), ``Library.search_open`` over the WHOLE library of the same polarity (no mass
window): coarse dot product on the binned fragment + neutral-loss open index, then exact
modified cosine on the best ``cfg.analog_feat_coarse_top`` consensus spectra, best consensus
per structure. The lists of the molecule's consensus spectra are merged per structure (highest
modified cosine, then more matched peaks, then higher coarse score, then lower consensus
index). ``select_hits`` then drops hits with fewer than ``analog_feat_min_matched`` matched
peaks, a modified cosine under ``analog_feat_min_sim`` or (if ``analog_feat_max_dm_da`` > 0)
a mass difference above that limit, and keeps the best ``analog_feat_top_k``. Each hit has
``sim`` (modified cosine), weight ``w = sim ** analog_feat_power``, ``dm`` = query neutral
mass M minus the hit structure mass, and ``delta`` = the edit family whose signed mass matches
``dm`` within ``analog_feat_delta_tol_da`` ("0" for the same mass, "" for none).

Validation exclusions: the caller passes the molecule's spectrum mask (``exclude_mask``, one
bool per library spectrum, as from ``ValidationHooks.library_mask(lib, molecule_id)``); every
consensus spectrum with an excluded member is removed BEFORE the coarse stage, so a class 2/3
query never sees its own spectra (and under C3h never its near neighbours: they are in the
masked group). The stage does this per molecule; ``validation.library_guard`` (when active)
OR-s in the union over the chunk, which is harmless.

Edit deltas (``delta_table``): the distinct signed formula deltas of ``generate.RULES`` (the
74 edit rules of the class-3 generator: methylation, hydroxylation, glycosylation, acylation,
...), with their monoisotopic masses, plus "0".

Features (``FEATURES``; NaN when the molecule has no hit or the candidate has no fingerprint)
--------------------------------------------------------------------------------------------
Fingerprints: Morgan radius 2, 4096 bits on ``chem.fp_mol`` (the NP tier's ``np_fp.npy``
convention); candidates from ``np_row`` where present, else from their SMILES; hit structures
from ``cfg.analog_feat_fp_path`` (optional precomputed array) or their library SMILES
(memoised per process). T(c, h) = Tanimoto, w_h = hit weight.

Every candidate:

- ``an_tan_max``: max_h T; ``an_tan_best``: T to the best hit (hit 1);
- ``an_tan_wmax``: max_h w_h T (public "ap"); ``an_tan_wsum``: sum_h w_h T (ROADMAP R6);
  ``an_tan_wmean``: sum_h w_h T / sum_h w_h (public "mean_tan");
- ``an_n_close``: hits with T >= ``analog_feat_close_tanimoto``;
- ``an_edit_n`` / ``an_edit_w``: hits whose formula differs from the candidate's by exactly
  one rule delta (count / summed weight); ``an_iso_w``: summed weight of hits with the
  candidate's formula; ``an_best_edit``: 1 if the candidate formula minus the best hit's is a
  rule delta or zero;
- molecule context (constant within a molecule): ``an_n_hits``, ``an_best_sim``,
  ``an_best_dm``, ``an_best_delta`` (1 if the best hit's dm matches a non-zero rule delta,
  0 otherwise), ``an_delta_share`` (weight share of hits whose dm matches a non-zero delta),
  ``an_w_total`` (sum of weights).

Top ``analog_feat_struct_top_k`` candidates per molecule by ``pool[analog_feat_rank_col]``
(``fragments.select_top_k`` order; 0 = all), against the best ``analog_feat_struct_hits``
hits (NaN elsewhere):

- ``an_scaf_n`` / ``an_scaf_w``: hits with the candidate's Bemis-Murcko scaffold (count /
  summed weight; 0 for acyclic candidates);
- ``an_sub_n`` / ``an_sub_w``: hits that are substructure-consistent with the candidate:
  the formula difference is one rule delta or zero and, on the element graphs (heavy atoms,
  all bonds single, RDKit MolHash ElementGraph), the hit is a substructure of the candidate
  when the delta only adds heavy atoms (methylation, glycosylation, ...), the candidate a
  substructure of the hit when it only removes them, the two graphs are equal when no heavy
  atom changes (hydrogenation, positional moves keep the formula), and equal after merging
  N and O for the NH/O swaps. This is "candidate = hit structure + that delta" without
  enumerating rule products;
- ``an_sub_best``: 1 if substructure-consistent with the best hit.

Within-molecule normalisation over the rows that carry the feature (all candidates of the
molecule in the pool for the fingerprint features, the top K for the structure features):
``<f>_z`` (z-score) and ``<f>_rank`` (1 = largest, ties share the lowest rank) for
``NORM_COLUMNS``, and ``<f>_gap`` (value minus the molecule maximum) for ``GAP_COLUMNS``.

Pipeline stage ``analogs`` (plugin ``casmi.analogs``), after ``score`` (or any stage that
sets ``analog_feat_rank_col``): library from ``libmatch.get_library`` (no library on disk:
NaN features and a warning), exclusions from ``state.extras["validation"]`` per molecule,
query spectra from ``state.test_df``, M from ``state.molecules``. Adds ``FEATURES`` to
``state.pool``, ``MOLECULE_COLUMNS`` to ``state.molecules`` and the selected hit lists to
``state.extras["analogs"]["hits"]`` ({molecule_id: DataFrame with ``HIT_COLUMNS``}) for the
neighbour network (R10) and the class-3 seeds (R13). Candidate fingerprints are kept in
``state.extras["cand_fp"]`` (``pool_fingerprints``) for later stages.

Inference-path rule: numpy, pandas, scipy, numba, rdkit and casmi modules only.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from numba import njit

from . import paths
from .config import Config

__all__ = [
    "HIT_COLUMNS",
    "RAW_FEATURES",
    "STRUCT_FEATURES",
    "NORM_COLUMNS",
    "GAP_COLUMNS",
    "FEATURES",
    "MOLECULE_COLUMNS",
    "delta_table",
    "match_delta",
    "formula_delta_codes",
    "search_hits",
    "select_hits",
    "StructCache",
    "struct_info",
    "tanimoto_matrix",
    "fingerprint_features",
    "structure_features",
    "normalise",
    "pool_fingerprints",
    "molecule_features",
    "score_pool",
    "features_task",
    "stage_analogs",
    "warmup",
]

log = logging.getLogger(__name__)

HIT_COLUMNS: tuple[str, ...] = (
    "hit_rank", "struct_idx", "ik14", "smiles", "formula", "mono_mass", "sim", "n_matched",
    "coarse", "polarity", "adduct_family", "cons_idx", "dm", "delta", "w",
)
FP_FEATURES: tuple[str, ...] = (
    "an_tan_max", "an_tan_best", "an_tan_wmax", "an_tan_wsum", "an_tan_wmean", "an_n_close",
    "an_edit_n", "an_edit_w", "an_iso_w", "an_best_edit",
)
CONTEXT_FEATURES: tuple[str, ...] = (
    "an_n_hits", "an_best_sim", "an_best_dm", "an_best_delta", "an_delta_share", "an_w_total",
)
STRUCT_FEATURES: tuple[str, ...] = (
    "an_scaf_n", "an_scaf_w", "an_sub_n", "an_sub_w", "an_sub_best",
)
RAW_FEATURES: tuple[str, ...] = FP_FEATURES + CONTEXT_FEATURES + STRUCT_FEATURES
NORM_COLUMNS: tuple[str, ...] = (   # an_tan_wsum is a constant multiple of an_tan_wmean in a molecule
    "an_tan_max", "an_tan_best", "an_tan_wmax", "an_tan_wmean", "an_edit_w", "an_scaf_w",
    "an_sub_w",
)
GAP_COLUMNS: tuple[str, ...] = ("an_tan_wmax", "an_tan_wmean")
FEATURES: tuple[str, ...] = (RAW_FEATURES
                             + tuple(f"{c}_z" for c in NORM_COLUMNS)
                             + tuple(f"{c}_rank" for c in NORM_COLUMNS)
                             + tuple(f"{c}_gap" for c in GAP_COLUMNS))
MOLECULE_COLUMNS: tuple[str, ...] = ("an_n_hits", "an_best_sim", "an_best_ik14", "an_best_dm",
                                     "an_best_delta_family", "an_n_scored_struct")

_N_BYTES = 512                      # packed Morgan 4096
_NAN = float("nan")


# ======================================================================================
# Edit deltas
# ======================================================================================

@lru_cache(maxsize=1)
def delta_table() -> pd.DataFrame:
    """Distinct signed formula deltas of ``generate.RULES`` plus zero: ``code``, ``family``
    (the rule family, "0" for zero), ``sign`` (+1 adds the family's atoms, -1 removes them,
    0), ``mass`` (signed monoisotopic Da), ``delta`` (tuple in ``formula.ELEMENTS`` order),
    ``rules`` (";"-joined rule ids). Sorted by mass, code = row position."""
    from .formula import ELEMENTS, counts_mass
    from .generate import RULES

    acc: dict[tuple[int, ...], list[str]] = {}
    fam: dict[tuple[int, ...], str] = {}
    for r in RULES:
        acc.setdefault(tuple(int(v) for v in r.delta), []).append(r.id)
        fam.setdefault(tuple(int(v) for v in r.delta), r.family)
    zero = (0,) * len(ELEMENTS)
    acc.setdefault(zero, [])
    fam[zero] = "0"
    rows = []
    for d, ids in acc.items():
        heavy = [v for i, v in enumerate(d) if ELEMENTS[i] != "H"]
        allv = list(d)
        sign = 0 if not any(allv) else (1 if all(v >= 0 for v in allv) else
                                        (-1 if all(v <= 0 for v in allv) else 0))
        rows.append({"family": fam[d], "sign": sign, "mass": float(counts_mass(np.asarray(d))),
                     "delta": d, "heavy": tuple(heavy), "rules": ";".join(sorted(ids))})
    out = pd.DataFrame(rows).sort_values(["mass", "family"], kind="mergesort").reset_index(drop=True)
    out.insert(0, "code", np.arange(len(out), dtype=np.int64))
    return out


@lru_cache(maxsize=1)
def _delta_index() -> tuple[np.ndarray, dict[tuple[int, ...], int], int]:
    t = delta_table()
    zero_code = int(t.index[t["family"] == "0"][0])
    return (t["mass"].to_numpy(np.float64), {d: int(c) for d, c in zip(t["delta"], t["code"])},
            zero_code)


@lru_cache(maxsize=1)
def _delta_kind() -> np.ndarray:
    """Per delta code: 0 = no heavy atom changes, 1 = adds heavy atoms only, -1 = removes
    them only, 2 = mixed (N/O swaps)."""
    out = []
    for heavy in delta_table()["heavy"]:
        if not any(heavy):
            out.append(0)
        elif all(v >= 0 for v in heavy):
            out.append(1)
        elif all(v <= 0 for v in heavy):
            out.append(-1)
        else:
            out.append(2)
    return np.asarray(out, dtype=np.int64)


def match_delta(dm: np.ndarray | float, tol: float) -> np.ndarray:
    """Code (``delta_table`` row) of the delta whose signed mass is nearest to each ``dm``
    within ``tol`` Da, else -1."""
    masses, _, _ = _delta_index()
    d = np.atleast_1d(np.asarray(dm, dtype=np.float64))
    out = np.full(d.shape, -1, dtype=np.int64)
    ok = np.isfinite(d)
    if not ok.any():
        return out
    pos = np.searchsorted(masses, d[ok])
    best = np.full(pos.shape, -1, dtype=np.int64)
    err = np.full(pos.shape, np.inf)
    for off in (-1, 0):
        k = np.clip(pos + off, 0, len(masses) - 1)
        e = np.abs(masses[k] - d[ok])
        better = e < err
        best[better] = k[better]
        err[better] = e[better]
    best[err > tol] = -1
    out[ok] = best
    return out


@lru_cache(maxsize=65536)
def _counts(formula: Any) -> tuple[int, ...] | None:
    from .formula import formula_counts

    if not isinstance(formula, str) or not formula:
        return None
    c = formula_counts(formula)
    return None if c is None else tuple(int(v) for v in c)


def formula_delta_codes(cand_formulas: Sequence[Any], hit_formulas: Sequence[Any]) -> np.ndarray:
    """(n_cand, n_hits) int64: ``delta_table`` code of candidate formula minus hit formula
    when that difference is exactly one rule delta or zero, else -1."""
    _, by_delta, _ = _delta_index()
    hc = [_counts(f) for f in hit_formulas]
    out = np.full((len(cand_formulas), len(hit_formulas)), -1, dtype=np.int64)
    memo: dict[Any, np.ndarray] = {}
    for i, f in enumerate(cand_formulas):
        if f not in memo:
            cc = _counts(f)
            row = np.full(len(hc), -1, dtype=np.int64)
            if cc is not None:
                for j, h in enumerate(hc):
                    if h is not None:
                        row[j] = by_delta.get(tuple(a - b for a, b in zip(cc, h)), -1)
            memo[f] = row
        out[i] = memo[f]
    return out


# ======================================================================================
# Hits
# ======================================================================================

def _empty_hits() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype=object) for c in HIT_COLUMNS})
    for c in ("hit_rank", "struct_idx", "n_matched", "polarity", "cons_idx"):
        df[c] = df[c].astype(np.int64)
    for c in ("mono_mass", "sim", "coarse", "dm", "w"):
        df[c] = df[c].astype(np.float64)
    return df


def search_hits(lib: Any, qs: Any, cfg: Config, M: float | None = None,
                exclude_mask: np.ndarray | None = None,
                exclude_cons_mask: np.ndarray | None = None,
                coarse_top: int | None = None) -> pd.DataFrame:
    """Every structure of the modified-cosine shortlist of the molecule's query consensus
    spectra (module docstring), merged per structure and sorted (``HIT_COLUMNS``; ``delta``
    and ``w`` filled by ``select_hits``). ``qs``: a ``library.QuerySet``."""
    ct = int(cfg.analog_feat_coarse_top if coarse_top is None else coarse_top)
    if qs is None or len(qs) == 0 or len(qs.cons_store) == 0 or ct <= 0:
        return _empty_hits()
    cex = None
    if exclude_mask is not None:
        cex = lib.consensus_mask(np.asarray(exclude_mask, dtype=bool))
    if exclude_cons_mask is not None:
        e = np.asarray(exclude_cons_mask, dtype=bool)
        cex = e.copy() if cex is None else (cex | e)
    parts = []
    for i in range(len(qs.cons_store)):
        mz, x = qs.cons_store[i]
        if len(mz) == 0:
            continue
        r = lib.search_open((mz, x, float(qs.cons_precursor_mz[i])), int(qs.cons_polarity[i]),
                            top_k=ct, exclude_mask=None, query_is_tims=bool(
                                qs.cons_instrument_family[i] == "timstof"),
                            coarse_top=ct, unique_structures=True, exclude_cons_mask=cex)
        if len(r):
            parts.append(r.assign(_q=i, polarity=int(qs.cons_polarity[i])))
    if not parts:
        return _empty_hits()
    r = pd.concat(parts, ignore_index=True)
    sim = pd.to_numeric(r["mod_cos"], errors="coerce").to_numpy(np.float64)
    nm = r["n_matched"].to_numpy(np.int64)
    co = r["coarse"].to_numpy(np.float64)
    o = np.lexsort((r["cons_idx"].to_numpy(np.int64), r["_q"].to_numpy(np.int64), -co, -nm,
                    -np.nan_to_num(sim, nan=-1.0)))
    r = r.iloc[o]
    r = r[~r["struct_idx"].duplicated(keep="first")].reset_index(drop=True)
    Mf = float(M) if M is not None and np.isfinite(M) else _NAN
    out = pd.DataFrame({
        "hit_rank": np.arange(1, len(r) + 1, dtype=np.int64),
        "struct_idx": r["struct_idx"].to_numpy(np.int64),
        "ik14": r["ik14"].astype(str).to_numpy(object),
        "smiles": r["smiles"].to_numpy(object),
        "formula": np.asarray(lib.struct_formula, dtype=object)[r["struct_idx"].to_numpy(np.int64)],
        "mono_mass": r["mono_mass"].to_numpy(np.float64),
        "sim": pd.to_numeric(r["mod_cos"], errors="coerce").to_numpy(np.float64),
        "n_matched": r["n_matched"].to_numpy(np.int64),
        "coarse": r["coarse"].to_numpy(np.float64),
        "polarity": r["polarity"].to_numpy(np.int64),
        "adduct_family": r["adduct_family"].astype(str).to_numpy(object),
        "cons_idx": r["cons_idx"].to_numpy(np.int64),
        "dm": Mf - r["mono_mass"].to_numpy(np.float64),
        "delta": "", "w": _NAN,
    })
    return out[list(HIT_COLUMNS)]


def select_hits(hits: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Filter and truncate a ``search_hits`` list (module docstring); fills ``delta``,
    ``w`` and renumbers ``hit_rank``."""
    if hits is None or not len(hits):
        return _empty_hits()
    h = hits
    keep = (h["n_matched"].to_numpy(np.int64) >= int(cfg.analog_feat_min_matched))
    sim = h["sim"].to_numpy(np.float64)
    keep &= np.nan_to_num(sim, nan=-1.0) >= float(cfg.analog_feat_min_sim)
    keep &= np.isfinite(sim)
    if float(cfg.analog_feat_max_dm_da) > 0:
        dm = h["dm"].to_numpy(np.float64)
        keep &= ~(np.abs(dm) > float(cfg.analog_feat_max_dm_da))
    h = h[keep].head(int(cfg.analog_feat_top_k)).copy()
    if not len(h):
        return _empty_hits()
    h["hit_rank"] = np.arange(1, len(h) + 1, dtype=np.int64)
    code = match_delta(h["dm"].to_numpy(np.float64), float(cfg.analog_feat_delta_tol_da))
    fam = delta_table()["family"].to_numpy(object)
    h["delta"] = np.where(code >= 0, fam[np.maximum(code, 0)], "").astype(object)
    h["w"] = np.clip(h["sim"].to_numpy(np.float64), 0.0, None) ** float(cfg.analog_feat_power)
    return h.reset_index(drop=True)


# ======================================================================================
# Structures: fingerprints, scaffolds, element graphs (memoised per process)
# ======================================================================================

@dataclass
class _Struct:
    scaffold: str            # Bemis-Murcko scaffold SMILES ("" if acyclic or unparsable)
    skel: str                # MolHash ElementGraph ("" if unparsable)
    graph: Any               # element-graph Mol for substructure tests (None if unparsable)
    n_heavy: int
    _skel_no: str | None = None

    @property
    def skel_no(self) -> str:
        """Element graph with N mapped to O (NH/O swaps), computed on first use."""
        if self._skel_no is None:
            self._skel_no = ""
            if self.graph is not None:
                from rdkit import Chem, rdBase

                with rdBase.BlockLogs():
                    try:
                        g2 = Chem.RWMol(self.graph)
                        for a in g2.GetAtoms():
                            if a.GetAtomicNum() == 7:
                                a.SetAtomicNum(8)
                        self._skel_no = Chem.MolToSmiles(g2, canonical=True)
                    except Exception:  # noqa: BLE001
                        self._skel_no = ""
        return self._skel_no


_STRUCT_MEMO: "OrderedDict[str, _Struct]" = OrderedDict()
_STRUCT_MEMO_MAX = 200_000


def struct_info(smiles: Any) -> _Struct:
    """Scaffold, element-graph keys and element-graph Mol of a SMILES (memoised)."""
    key = smiles if isinstance(smiles, str) else ""
    hit = _STRUCT_MEMO.get(key)
    if hit is not None:
        _STRUCT_MEMO.move_to_end(key)
        return hit
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdMolHash
    from rdkit.Chem.Scaffolds import MurckoScaffold

    out = _Struct("", "", None, 0)
    if key:
        with rdBase.BlockLogs():
            try:
                mol = Chem.MolFromSmiles(key)
                if mol is not None:
                    scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
                    skel = rdMolHash.MolHash(Chem.Mol(mol), rdMolHash.HashFunction.ElementGraph)
                    g = Chem.MolFromSmiles(skel, sanitize=False) if skel else None
                    if g is not None:
                        g.UpdatePropertyCache(strict=False)
                    out = _Struct(scaf or "", skel or "", g, mol.GetNumAtoms())
            except Exception:  # noqa: BLE001 - one bad structure never fails a run
                out = _Struct("", "", None, 0)
    _STRUCT_MEMO[key] = out
    if len(_STRUCT_MEMO) > _STRUCT_MEMO_MAX:
        _STRUCT_MEMO.popitem(last=False)
    return out


class StructCache:
    """Packed Morgan fingerprints of library (hit) structures by ik14.

    Precomputed rows from ``cfg.analog_feat_fp_path`` (``<x>.npy`` with ``<x>_ik14.parquet``,
    relative paths under ``paths.PROCESSED``; e.g. ``validation/train_fp``, the WP7 train
    fingerprint cache, which uses the same ``chem.morgan_fps``) when present; the rest are
    computed from the SMILES on first use and memoised."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self._memo: dict[str, np.ndarray] = {}
        self._pre: np.ndarray | None = None
        self._pre_row: dict[str, int] = {}
        self.n_pre = self.n_computed = 0
        p = str(self.cfg.analog_feat_fp_path or "").strip()
        if p:
            base = Path(p) if Path(p).is_absolute() else paths.PROCESSED / p
            npy = base.with_suffix(".npy") if base.suffix != ".npy" else base
            ikp = npy.with_name(npy.stem + "_ik14.parquet")
            if npy.exists() and ikp.exists():
                ik = pd.read_parquet(ikp)["ik14"].astype(str).to_numpy(object)
                arr = np.load(npy, mmap_mode="r")
                if arr.ndim == 2 and arr.shape[0] == len(ik) and arr.shape[1] == _N_BYTES:
                    self._pre = arr
                    self._pre_row = {k: i for i, k in enumerate(ik)}
                else:
                    log.warning("analogs: %s not aligned with %s; computing on the fly", npy, ikp)
            else:
                log.warning("analogs: no fingerprint cache at %s; computing on the fly", npy)

    def get(self, ik14: Sequence[str], smiles: Sequence[Any]) -> np.ndarray:
        """(n, 512) uint8 packed fingerprints (zero rows for unparsable structures)."""
        from .chem import morgan_fps

        n = len(ik14)
        out = np.zeros((n, _N_BYTES), dtype=np.uint8)
        todo: list[int] = []
        for i, k in enumerate(ik14):
            k = str(k)
            v = self._memo.get(k)
            if v is not None:
                out[i] = v
                continue
            r = self._pre_row.get(k)
            if r is not None and self._pre is not None:
                v = np.asarray(self._pre[r], dtype=np.uint8)
                self._memo[k] = v
                out[i] = v
                self.n_pre += 1
                continue
            todo.append(i)
        if todo:
            fps, ok = morgan_fps([smiles[i] if isinstance(smiles[i], str) else "" for i in todo],
                                 self.cfg)
            fps = np.asarray(fps, dtype=np.uint8)
            fps[~np.asarray(ok, bool)] = 0
            for j, i in enumerate(todo):
                self._memo[str(ik14[i])] = fps[j]
                out[i] = fps[j]
            self.n_computed += len(todo)
        return out


_CACHES: dict[tuple, StructCache] = {}


def _struct_cache(cfg: Config) -> StructCache:
    key = (str(cfg.analog_feat_fp_path), cfg.fp_morgan_bits, cfg.fp_morgan_radius,
           cfg.fp_inchi_roundtrip)
    c = _CACHES.get(key)
    if c is None:
        c = _CACHES[key] = StructCache(cfg)
    return c


# ======================================================================================
# Tanimoto kernel
# ======================================================================================

_M1 = np.uint64(0x5555555555555555)
_M2 = np.uint64(0x3333333333333333)
_M4 = np.uint64(0x0F0F0F0F0F0F0F0F)
_H01 = np.uint64(0x0101010101010101)
_S1, _S2, _S4, _S56 = np.uint64(1), np.uint64(2), np.uint64(4), np.uint64(56)


@njit(cache=True, nogil=True)
def _pc64(x):
    x = x - ((x >> _S1) & _M1)
    x = (x & _M2) + ((x >> _S2) & _M2)
    x = (x + (x >> _S4)) & _M4
    return (x * _H01) >> _S56


@njit(cache=True, nogil=True)
def _tanimoto_kernel(a, b, out):
    n, w = a.shape
    h = b.shape[0]
    ca = np.zeros(n, np.int64)
    cb = np.zeros(h, np.int64)
    for i in range(n):
        s = 0
        for k in range(w):
            s += _pc64(a[i, k])
        ca[i] = s
    for j in range(h):
        s = 0
        for k in range(w):
            s += _pc64(b[j, k])
        cb[j] = s
    for i in range(n):
        for j in range(h):
            inter = 0
            for k in range(w):
                inter += _pc64(a[i, k] & b[j, k])
            union = ca[i] + cb[j] - inter
            out[i, j] = inter / union if union > 0 else 0.0


def _u64(fp: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(np.asarray(fp, dtype=np.uint8))
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.shape[1] % 8:
        a = np.ascontiguousarray(np.pad(a, ((0, 0), (0, 8 - a.shape[1] % 8))))
    return a.view(np.uint64)


def tanimoto_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(len(a), len(b)) float64 Tanimoto between packed uint8 fingerprint rows (0 where both
    are empty; the same values as ``chem.tanimoto``)."""
    ua, ub = _u64(a), _u64(b)
    out = np.zeros((ua.shape[0], ub.shape[0]), dtype=np.float64)
    if ua.shape[0] and ub.shape[0]:
        _tanimoto_kernel(ua, ub, out)
    return out


# ======================================================================================
# Features of one molecule
# ======================================================================================

def _context(hits: pd.DataFrame) -> dict[str, float]:
    if hits is None or not len(hits):
        return {"an_n_hits": 0.0, "an_best_sim": _NAN, "an_best_dm": _NAN,
                "an_best_delta": _NAN, "an_delta_share": _NAN, "an_w_total": 0.0}
    w = hits["w"].to_numpy(np.float64)
    fam = hits["delta"].astype(str).to_numpy(object)
    nz = (fam != "") & (fam != "0")
    tot = float(w.sum())
    return {"an_n_hits": float(len(hits)), "an_best_sim": float(hits["sim"].iat[0]),
            "an_best_dm": float(hits["dm"].iat[0]), "an_best_delta": float(nz[0]),
            "an_delta_share": float(w[nz].sum() / tot) if tot > 0 else 0.0,
            "an_w_total": tot}


def fingerprint_features(cand_fp: np.ndarray, cand_ok: np.ndarray, cand_formulas: Sequence[Any],
                         hits: pd.DataFrame, hit_fp: np.ndarray, cfg: Config
                         ) -> dict[str, np.ndarray]:
    """``FP_FEATURES`` + ``CONTEXT_FEATURES`` for the candidates of one molecule against its
    selected hits (``select_hits``)."""
    n = len(cand_fp)
    out = {c: np.full(n, np.nan) for c in FP_FEATURES}
    ctx = _context(hits)
    for c, v in ctx.items():
        out[c] = np.full(n, v)
    if n == 0 or hits is None or not len(hits):
        return out
    w = hits["w"].to_numpy(np.float64)
    T = tanimoto_matrix(cand_fp, hit_fp)
    ok = np.asarray(cand_ok, dtype=bool)
    wt = T * w[None, :]
    ws = float(w.sum())
    vals = {
        "an_tan_max": T.max(axis=1), "an_tan_best": T[:, 0], "an_tan_wmax": wt.max(axis=1),
        "an_tan_wsum": wt.sum(axis=1),
        "an_tan_wmean": wt.sum(axis=1) / ws if ws > 0 else np.zeros(n),
        "an_n_close": (T >= float(cfg.analog_feat_close_tanimoto)).sum(axis=1).astype(np.float64),
    }
    for c, v in vals.items():
        out[c] = np.where(ok, v, np.nan)
    codes = formula_delta_codes(list(cand_formulas), list(hits["formula"]))
    zero = _delta_index()[2]
    edit = (codes >= 0) & (codes != zero)
    iso = codes == zero
    fok = np.array([_counts(f) is not None for f in cand_formulas], dtype=bool)
    out["an_edit_n"] = np.where(fok, edit.sum(axis=1).astype(np.float64), np.nan)
    out["an_edit_w"] = np.where(fok, (edit * w[None, :]).sum(axis=1), np.nan)
    out["an_iso_w"] = np.where(fok, (iso * w[None, :]).sum(axis=1), np.nan)
    out["an_best_edit"] = np.where(fok, (codes[:, 0] >= 0).astype(np.float64), np.nan)
    return out


def _consistent(c: _Struct, h: _Struct, code: int) -> bool:
    """Substructure consistency of candidate ``c`` with hit ``h`` under delta ``code``."""
    if code < 0 or c.graph is None or h.graph is None or not c.skel or not h.skel:
        return False
    kind = int(_delta_kind()[code])
    if kind == 0:
        return c.skel == h.skel
    if kind == 1:
        small, big = h, c
    elif kind == -1:
        small, big = c, h
    else:
        return bool(c.skel_no) and c.skel_no == h.skel_no
    if small.n_heavy > big.n_heavy:
        return False
    try:
        return bool(big.graph.HasSubstructMatch(small.graph))
    except Exception:  # noqa: BLE001
        return False


def structure_features(cand_smiles: Sequence[Any], cand_formulas: Sequence[Any],
                       hits: pd.DataFrame, cfg: Config) -> dict[str, np.ndarray]:
    """``STRUCT_FEATURES`` for candidates against the best ``analog_feat_struct_hits`` hits."""
    n = len(cand_smiles)
    out = {c: np.full(n, np.nan) for c in STRUCT_FEATURES}
    if n == 0 or hits is None or not len(hits):
        return out
    hs = hits.head(int(cfg.analog_feat_struct_hits))
    w = hs["w"].to_numpy(np.float64)
    hinfo = [struct_info(s) for s in hs["smiles"]]
    codes = formula_delta_codes(list(cand_formulas), list(hs["formula"]))
    hscaf = np.array([h.scaffold for h in hinfo], dtype=object)
    for i, smi in enumerate(cand_smiles):
        ci = struct_info(smi)
        if ci.graph is None:
            continue
        scaf = (hscaf == ci.scaffold) if ci.scaffold else np.zeros(len(hinfo), bool)
        sub = np.zeros(len(hinfo), bool)
        for j in np.flatnonzero(codes[i] >= 0):
            sub[j] = _consistent(ci, hinfo[j], int(codes[i, j]))
        out["an_scaf_n"][i] = float(scaf.sum())
        out["an_scaf_w"][i] = float(w[scaf].sum())
        out["an_sub_n"][i] = float(sub.sum())
        out["an_sub_w"][i] = float(w[sub].sum())
        out["an_sub_best"][i] = float(sub[0])
    return out


# ======================================================================================
# Normalisation
# ======================================================================================

def _group_z_rank(g: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per group: z-score, rank (1 = largest, ties share the lowest rank) and group max;
    NaN values are ignored and get NaN."""
    n = len(x)
    z = np.full(n, np.nan)
    rank = np.full(n, np.nan)
    gmax = np.full(n, np.nan)
    ok = np.isfinite(x)
    if not ok.any():
        return z, rank, gmax
    gi, xi = g[ok], x[ok]
    ng = int(g.max()) + 1
    cnt = np.bincount(gi, minlength=ng).astype(np.float64)
    mean = np.bincount(gi, weights=xi, minlength=ng) / np.maximum(cnt, 1)
    sd = np.sqrt(np.bincount(gi, weights=(xi - mean[gi]) ** 2, minlength=ng) / np.maximum(cnt, 1))
    s = sd[gi]
    zz = np.where(s > 1e-12, (xi - mean[gi]) / np.where(s > 1e-12, s, 1.0), 0.0)
    mx = np.full(ng, -np.inf)
    np.maximum.at(mx, gi, xi)
    order = np.lexsort((-xi, gi))
    gs, xs = gi[order], xi[order]
    pos = np.arange(len(xs))
    gstart = np.r_[True, gs[1:] != gs[:-1]]
    bstart = gstart | np.r_[True, xs[1:] != xs[:-1]]
    first_g = np.maximum.accumulate(np.where(gstart, pos, 0))
    first_b = np.maximum.accumulate(np.where(bstart, pos, 0))
    rr = np.empty(len(xs))
    rr[order] = first_b - first_g + 1
    idx = np.flatnonzero(ok)
    z[idx], rank[idx], gmax[idx] = zz, rr, mx[gi]
    return z, rank, gmax


def normalise(raw: pd.DataFrame, molecule_ids: Sequence[Any]) -> pd.DataFrame:
    """``FEATURES`` from a frame with ``RAW_FEATURES`` (rows of one or more molecules)."""
    out = raw.copy()
    g = pd.factorize(pd.Series(list(molecule_ids), dtype=object).astype(str), sort=True)[0]
    for c in NORM_COLUMNS:
        z, r, mx = _group_z_rank(g, out[c].to_numpy(dtype=np.float64))
        out[f"{c}_z"] = z
        out[f"{c}_rank"] = r
        if c in GAP_COLUMNS:
            out[f"{c}_gap"] = out[c].to_numpy(dtype=np.float64) - mx
    for c in FEATURES:
        if c not in out.columns:
            out[c] = np.nan
    return out[list(FEATURES)]


# ======================================================================================
# Pool level and the pipeline stage
# ======================================================================================

def pool_fingerprints(pool: pd.DataFrame, cfg: Config, db: Any = None,
                      extras: dict | None = None, n_jobs: int | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """(packed (n, 512) uint8, ok) candidate fingerprints aligned with ``pool`` rows: NP rows
    from ``np_fp.npy`` (``np_row``), the rest from ``chem.morgan_fps`` of their SMILES.
    With ``extras`` (``state.extras``), the result is reused while the pool's ik14 column is
    unchanged and stored as ``extras["cand_fp"] = {"ik14", "fp", "ok"}``."""
    n = len(pool)
    ik = pool["ik14"].astype(str).to_numpy(object) if n else np.zeros(0, object)
    if extras is not None:
        c = extras.get("cand_fp")
        if isinstance(c, dict) and len(c.get("ik14", ())) == n and np.array_equal(c["ik14"], ik):
            return c["fp"], c["ok"]
    fp = np.zeros((n, _N_BYTES), dtype=np.uint8)
    ok = np.zeros(n, dtype=bool)
    if n:
        np_row = (pd.to_numeric(pool["np_row"], errors="coerce").fillna(-1).to_numpy(np.int64)
                  if "np_row" in pool.columns else np.full(n, -1, np.int64))
        has = np_row >= 0
        if has.any():
            try:
                from . import db as cdb

                dbh = db if db is not None else cdb.get_db(cfg)
                fp[has] = dbh.np_fingerprints(np_row[has])
                ok[has] = fp[has].any(axis=1)
            except Exception:  # noqa: BLE001 - fall back to SMILES
                log.debug("analogs: NP fingerprints unavailable", exc_info=True)
                has = np.zeros(n, dtype=bool)
        rest = np.flatnonzero(~has)
        if len(rest):
            from .chem import morgan_fps

            nj = int(cfg.analog_feat_n_jobs if n_jobs is None else n_jobs)
            if nj <= 0:
                nj = int(cfg.n_jobs)
            smi = pool["smiles"].to_numpy(object)[rest]
            f, o = morgan_fps([s if isinstance(s, str) else "" for s in smi], cfg,
                              n_jobs=nj if len(rest) >= 4000 else 1)
            f = np.asarray(f, dtype=np.uint8)
            o = np.asarray(o, dtype=bool)
            f[~o] = 0
            fp[rest] = f
            ok[rest] = o
    if extras is not None:
        extras["cand_fp"] = {"ik14": ik, "fp": fp, "ok": ok}
    return fp, ok


def molecule_features(cand_fp: np.ndarray, cand_ok: np.ndarray, cand_formulas: Sequence[Any],
                      cand_smiles: Sequence[Any], struct_rows: np.ndarray, hits: pd.DataFrame,
                      cfg: Config, cache: StructCache | None = None) -> np.ndarray:
    """(n, len(RAW_FEATURES)) raw features of one molecule's candidates; ``struct_rows``:
    bool mask of the candidates that get ``STRUCT_FEATURES``."""
    n = len(cand_fp)
    out = np.full((n, len(RAW_FEATURES)), np.nan)
    if n == 0:
        return out
    cache = cache if cache is not None else _struct_cache(cfg)
    if hits is not None and len(hits):
        hfp = cache.get(list(hits["ik14"]), list(hits["smiles"]))
    else:
        hfp = np.zeros((0, _N_BYTES), np.uint8)
    f = fingerprint_features(cand_fp, cand_ok, cand_formulas, hits, hfp, cfg)
    sel = np.flatnonzero(np.asarray(struct_rows, dtype=bool))
    s = structure_features([cand_smiles[i] for i in sel], [cand_formulas[i] for i in sel],
                           hits, cfg)
    col = {c: i for i, c in enumerate(RAW_FEATURES)}
    for c, v in f.items():
        out[:, col[c]] = v
    for c, v in s.items():
        out[sel, col[c]] = v
    return out


def _struct_mask(pool: pd.DataFrame, cfg: Config) -> np.ndarray:
    k = int(cfg.analog_feat_struct_top_k)
    if k <= 0:
        return np.ones(len(pool), dtype=bool)
    from .fragments import select_top_k

    return select_top_k(pool, cfg.replace(frag_top_k=k, frag_rank_col=cfg.analog_feat_rank_col))


def score_pool(pool: pd.DataFrame, test_df: pd.DataFrame, lib: Any, cfg: Config | None = None,
               molecules: pd.DataFrame | None = None, hooks: Any = None, db: Any = None,
               extras: dict | None = None, struct_rows: np.ndarray | None = None,
               hits_by_mol: Mapping[str, pd.DataFrame] | None = None
               ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Analog features for every ``pool`` row.

    ``lib``: a ``library.Library`` (None: hits only from ``hits_by_mol``, else none);
    ``molecules``: per-molecule ``M`` (default ``io.mass_tables(test_df)``); ``hooks``: a
    ``validation.ValidationHooks`` (per-molecule ``library_mask``); ``struct_rows``: bool
    mask of rows that get ``STRUCT_FEATURES`` (default the top K by
    ``analog_feat_rank_col``); ``hits_by_mol``: precomputed ``search_hits`` lists (raw,
    unselected) per molecule, used instead of searching.

    Returns (features aligned with ``pool.index``, per-molecule frame with
    ``MOLECULE_COLUMNS``, {molecule_id: selected hits})."""
    from .library import prepare_queries

    cfg = cfg or Config()
    n = len(pool)
    feats = pd.DataFrame(np.nan, index=pool.index, columns=list(FEATURES))
    mol_cols = ["molecule_id", *MOLECULE_COLUMNS]
    if n == 0:
        return feats, pd.DataFrame(columns=mol_cols), {}
    mids = pool["molecule_id"].astype(str).to_numpy(object)
    if molecules is None or "M" not in getattr(molecules, "columns", ()):
        from .io import mass_tables

        molecules = mass_tables(test_df, cfg)[0]
    M_of = dict(zip(molecules["molecule_id"].astype(str), pd.to_numeric(
        molecules["M"], errors="coerce").to_numpy(np.float64)))
    srow = _struct_mask(pool, cfg) if struct_rows is None else np.asarray(struct_rows, bool)
    fp, ok = pool_fingerprints(pool, cfg, db=db, extras=extras)
    formulas = (pool["formula"].to_numpy(object) if "formula" in pool.columns
                else np.full(n, None, object))
    smiles = pool["smiles"].to_numpy(object)
    tdf = test_df.assign(molecule_id=test_df["molecule_id"].astype(str)) if test_df is not None \
        and len(test_df) else None
    tpos = tdf.groupby("molecule_id", sort=True).indices if tdf is not None else {}
    cache = _struct_cache(cfg)
    raw = np.full((n, len(RAW_FEATURES)), np.nan)
    hits_out: dict[str, pd.DataFrame] = {}
    mol_rows = []
    by_mol = pd.Series(np.arange(n)).groupby(mids, sort=True).indices
    for m in sorted(by_mol):
        rows = np.sort(np.asarray(by_mol[m], dtype=np.int64))
        M = float(M_of.get(m, np.nan))
        if hits_by_mol is not None:
            h_raw = hits_by_mol.get(m)
        elif lib is not None and m in tpos:
            sub = tdf.iloc[np.sort(tpos[m])].reset_index(drop=True)
            mask = None
            if hooks is not None:
                try:
                    mask = hooks.library_mask(lib, m)
                except Exception:  # noqa: BLE001 - the guard still applies its own mask
                    log.exception("analogs: validation library_mask failed for %s", m)
            qs = prepare_queries(sub, lib.cfg, M=M if np.isfinite(M) else None)
            h_raw = search_hits(lib, qs, cfg, M=M, exclude_mask=mask)
        else:
            h_raw = None
        hits = select_hits(h_raw, cfg) if h_raw is not None else _empty_hits()
        hits_out[m] = hits
        raw[rows] = molecule_features(fp[rows], ok[rows], list(formulas[rows]), list(smiles[rows]),
                                      srow[rows], hits, cfg, cache)
        mol_rows.append({"molecule_id": m, "an_n_hits": int(len(hits)),
                         "an_best_sim": float(hits["sim"].iat[0]) if len(hits) else _NAN,
                         "an_best_ik14": str(hits["ik14"].iat[0]) if len(hits) else "",
                         "an_best_dm": float(hits["dm"].iat[0]) if len(hits) else _NAN,
                         "an_best_delta_family": str(hits["delta"].iat[0]) if len(hits) else "",
                         "an_n_scored_struct": int(srow[rows].sum()) if len(hits) else 0})
    normed = normalise(pd.DataFrame(raw, columns=list(RAW_FEATURES)), mids)
    feats.iloc[:, :] = normed.to_numpy(dtype=np.float64)
    return feats, pd.DataFrame(mol_rows, columns=mol_cols), hits_out


def features_task(args: tuple) -> list[tuple[str, np.ndarray]]:
    """Worker for batch scoring (``chem.parallel_map``): ``args = (cfg, items)`` with items
    ``(molecule_id, cand_fp, cand_ok, formulas, smiles, struct_rows, raw_hits)``; returns
    ``[(molecule_id, raw feature matrix)]`` (``RAW_FEATURES`` columns, ``select_hits``
    applied to the raw hits)."""
    cfg, items = args
    cache = _struct_cache(cfg)
    out = []
    for mid, fp, ok, form, smi, srow, raw in items:
        hits = select_hits(raw, cfg) if raw is not None else _empty_hits()
        out.append((mid, molecule_features(fp, ok, form, smi, srow, hits, cfg, cache)))
    return out


_WARNED: set[str] = set()


def stage_analogs(state: Any, cfg: Config) -> Any:
    """Pipeline stage "analogs": ``FEATURES`` on ``state.pool`` (module docstring)."""
    pool = state.pool
    if pool is None or not len(pool):
        return state
    t0 = time.perf_counter()
    from .libmatch import get_library

    lib = get_library(state, cfg)
    if lib is None and "nolib" not in _WARNED:
        _WARNED.add("nolib")
        log.warning("analogs: no library; analog features are NaN")
    hooks = state.extras.get("validation")
    feats, mol, hits = score_pool(pool, state.test_df, lib, cfg, molecules=state.molecules,
                                  hooks=hooks, db=state.db, extras=state.extras)
    for c in FEATURES:
        pool[c] = feats[c].to_numpy(dtype=np.float64)
    state.pool = pool
    ex = state.extras.setdefault("analogs", {})
    ex.setdefault("hits", {}).update(hits)
    if state.molecules is not None:
        mols = state.molecules
        ids = mols["molecule_id"].astype(str)
        mm = mol.set_index("molecule_id") if len(mol) else None
        for c in MOLECULE_COLUMNS:
            if mm is None:
                v = pd.Series(np.nan, index=ids.index)
            else:
                v = ids.map(mm[c])
            if c in ("an_n_hits", "an_n_scored_struct"):
                mols[c] = pd.to_numeric(v, errors="coerce").fillna(0).astype(np.int64).to_numpy()
            elif c in ("an_best_ik14", "an_best_delta_family"):
                mols[c] = v.fillna("").astype(str).to_numpy(object)
            else:
                mols[c] = pd.to_numeric(v, errors="coerce").to_numpy(np.float64)
        state.molecules = mols
    log.debug("analogs: %d molecules, %d rows in %.2f s", len(mol), len(pool),
              time.perf_counter() - t0)
    return state


def warmup() -> None:
    """Compile the numba kernel once (cached on disk afterwards)."""
    a = np.zeros((2, _N_BYTES), np.uint8)
    a[0, :3] = 7
    tanimoto_matrix(a, a)


try:  # registration at import, as pipeline plugins expect
    from . import pipeline as _pipeline

    _pipeline.register_stage("analogs", stage_analogs)
except Exception:  # noqa: BLE001 - never fail an import because of the registry
    log.debug("analogs stage not registered", exc_info=True)

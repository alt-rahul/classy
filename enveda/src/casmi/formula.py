"""Molecular formula enumeration, sub-formula explanation score and formula model (WP12).

PLAN.md section 3.4. For LIB and DB candidates the formula is a property of the
candidate; formula inference is needed for the class 3 generator (which formulas should
generated structures have, section 4.7) and as a ranker feature.

Public API:

- ``ELEMENTS``: the ten elements, in the column order of every count array here:
  C, H, N, O, P, S, F, Cl, Br, I.
- ``enumerate_formulas(mass, cfg=None, tol_ppm=None, min_tol_da=None) -> FormulaSet``:
  all formulas over ``ELEMENTS`` inside ``|m - mass| <= max(ppm * mass, min_da)`` with
  integer non-negative RDBE, the Senior checks and the element-ratio rules (below).
  Numba kernel, sorted by (|ppm error|, C, H, N, ...).
- ``prepare_peaks(spectra, cfg=None, tol_da=None, cleaned=False) -> PeakSet``: a
  molecule's cleaned fragment peaks grouped by adduct, ready for ``explain``.
- ``explain(counts, peaks) -> (expl_frac, expl_peaks)``: for each candidate neutral
  formula, the fraction of the pooled cleaned fragment intensity (and of the peak count)
  whose m/z is a sub-formula of the ion formula (formula + adduct) within the fragment
  tolerance.
- ``formula_features(mass, spectra, cfg=None, ...) -> DataFrame``: one row per candidate
  formula with mass error, ``expl_frac``, RDBE, element counts and ratios, DB counts
  (``db.lookup``), ``lib_count`` (hook, parameter).
- ``FormulaModel``: numpy-only L2 logistic regression (``fit``, ``predict_proba``,
  ``save`` / ``load`` JSON at ``models/formula_v1.json``), plus ``FormulaModel.default()``,
  a hand-set placeholder used until the fitted file exists.
- ``rank_formulas(...)``: features plus model probability, best first;
  ``select_targets(ranked, cfg)``: the GEN target formulas (top ``gen_n_formulas`` with
  probability >= ``gen_formula_min_prob``).
- ``fit_formula_model(tables, true_formulas, ...)``: NEEDS_TRAIN hook (labels = formula
  equals the true formula), called once holdout molecules exist.
- helpers: ``hill``, ``formula_counts``, ``counts_mass``, ``rdbe``, ``adduct_delta``.

Rules fixed here (the plan leaves them open):

- Valences for RDBE and the Senior checks are the lowest ones: C 4, H 1, N 3, O 2, P 3,
  S 2, halogens 1. ``RDBE = 1 + C + (N + P) / 2 - (H + X) / 2``; it must be an integer
  >= 0 (this is also the even-valence-sum Senior rule and the nitrogen rule), the valence
  sum must be >= 2 x the largest valence (4, since C >= 1), and RDBE >= 0 is Senior's
  third rule. Hypervalent S and P (sulfates, phosphates) satisfy these at the lowest
  valences anyway.
- Ratio rules use the ``formula_*`` config fields: H/C in [hc_min, hc_max] (H only, not
  halogens), N/C <= nc_max, O/C <= oc_max, P <= p_max, S <= s_max, F + Cl + Br + I <=
  halogen_max, and at least ``formula_c_min`` carbons.
- Fragments: singly charged with the precursor's polarity; fragment formula mass =
  m/z + sign x electron mass. A fragment is explained if some sub-formula ``d`` of the
  ion formula ``U = multiplier x F + adduct delta`` (all counts 0..U, Na and K from the
  adduct included, so [M+Na]+ fragments may or may not keep Na) has
  ``|mass(d) - fragment formula mass| <= tol`` and fragment RDBE >= -0.5, i.e.
  ``H + X + Na + K <= 2C + N + P + 3``. A formula that the adduct cannot be applied to
  (negative ion counts, e.g. no O for [M-H2O+H]+) explains nothing for that adduct's
  spectra and gets ``adduct_ok = False``.
- Sub-formulas are never materialised. The numba kernel enumerates heteroatom
  sub-counts (pruned by the largest peak mass) and, for each combination and each still
  unexplained peak, solves for C and H directly (about U_H x 1.008 / 12 carbon values per
  peak), with an early exit once every peak is explained. This replaced a first version
  that walked all (heteroatoms, C, H) sub-formulas against a peak bin table: same
  results, 16 to 20 times slower above 1,000 Da. Precomputing and caching sorted
  sub-formula mass lists per formula (as PLAN.md 3.4 suggests) would cost more still: a
  C54H90O24 ion has about 77k sub-formulas and an N-rich peptide formula millions. The
  reusable per-molecule part is the ``PeakSet``. Measured: about 0.1 ms per formula at
  450 Da and 0.6 ms at 1,100 Da, so ``formula_features`` explains at most
  ``cfg.formula_expl_max`` formulas per molecule (see there).
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from numba import njit

from . import paths
from .adducts import ABBREVIATIONS, ELECTRON_MASS, element_mass, parse_formula
from .config import Config

__all__ = [
    "ELEMENTS",
    "ION_ELEMENTS",
    "FORMULA_FEATURES",
    "FORMULA_MODEL_PATH",
    "FormulaSet",
    "PeakSet",
    "FormulaModel",
    "element_masses",
    "hill",
    "formula_counts",
    "counts_mass",
    "rdbe",
    "formula_ok",
    "adduct_delta",
    "enumerate_formulas",
    "prepare_peaks",
    "spectra_from_frame",
    "explain",
    "db_formula_counts",
    "formula_features",
    "rank_formulas",
    "select_targets",
    "load_formula_model",
    "fit_formula_model",
]

log = logging.getLogger(__name__)

ELEMENTS: tuple[str, ...] = ("C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I")
ION_ELEMENTS: tuple[str, ...] = ELEMENTS + ("Na", "K")
_EIDX = {e: i for i, e in enumerate(ION_ELEMENTS)}
_HALOGENS = (6, 7, 8, 9)
# Hill order: C, H, then the rest alphabetically (RDKit CalcMolFormula and PubChem agree).
_HILL = ("C", "H") + tuple(sorted(e for e in ION_ELEMENTS if e not in ("C", "H")))

FORMULA_MODEL_PATH: Path = paths.MODELS / "formula_v1.json"


@lru_cache(maxsize=1)
def element_masses() -> np.ndarray:
    """Monoisotopic masses of ``ION_ELEMENTS`` (RDKit most-common isotopes), float64 (12,)."""
    return np.array([element_mass(e) for e in ION_ELEMENTS], dtype=np.float64)


def _em10() -> np.ndarray:
    return element_masses()[:10].copy()


# --------------------------------------------------------------------------------------
# Formula strings and counts
# --------------------------------------------------------------------------------------

def hill(counts: Sequence[int] | np.ndarray | Mapping[str, int]) -> str:
    """Hill formula string of a count vector (``ELEMENTS`` or ``ION_ELEMENTS`` order) or a
    mapping. Zero counts are omitted, counts of 1 have no digit."""
    if isinstance(counts, Mapping):
        d = {k: int(v) for k, v in counts.items() if int(v) != 0}
    else:
        arr = np.asarray(counts).reshape(-1)
        d = {ION_ELEMENTS[i]: int(arr[i]) for i in range(arr.shape[0]) if int(arr[i]) != 0}
    if any(v < 0 for v in d.values()):
        raise ValueError(f"negative count in {d}")
    order = list(_HILL) + sorted(k for k in d if k not in _HILL)
    if "C" not in d:  # Hill: without carbon everything is alphabetical
        order = sorted(d)
    out = []
    for e in order:
        n = d.get(e, 0)
        if n:
            out.append(e if n == 1 else f"{e}{n}")
    return "".join(out)


def formula_counts(formula: str | Mapping[str, int] | None, ion: bool = False
                   ) -> np.ndarray | None:
    """Count vector (int64, ``ELEMENTS`` order, or ``ION_ELEMENTS`` with ``ion=True``) of a
    neutral formula string or mapping. None if it is not a plain formula, carries a charge
    suffix, or has an element outside the alphabet."""
    if formula is None:
        return None
    if isinstance(formula, str):
        d = parse_formula(formula.strip())
        if d is None:
            return None
    else:
        d = {k: int(v) for k, v in formula.items()}
    alphabet = ION_ELEMENTS if ion else ELEMENTS
    out = np.zeros(len(alphabet), dtype=np.int64)
    for e, n in d.items():
        if n == 0:
            continue
        if e not in _EIDX or _EIDX[e] >= len(alphabet):
            return None
        out[_EIDX[e]] = n
    return out


def counts_mass(counts: np.ndarray) -> np.ndarray | float:
    """Monoisotopic mass of count vector(s) (last axis 10 or 12 elements)."""
    c = np.asarray(counts, dtype=np.float64)
    em = element_masses()[: c.shape[-1]]
    out = c @ em
    return float(out) if np.ndim(out) == 0 else out


def rdbe(counts: np.ndarray) -> np.ndarray | float:
    """``1 + C + (N + P) / 2 - (H + X) / 2`` (lowest valences; X = halogens, Na, K)."""
    c = np.asarray(counts, dtype=np.float64)
    x = c[..., 6:10].sum(axis=-1)
    if c.shape[-1] > 10:
        x = x + c[..., 10:].sum(axis=-1)
    out = 1.0 + c[..., 0] + (c[..., 2] + c[..., 4]) / 2.0 - (c[..., 1] + x) / 2.0
    return float(out) if np.ndim(out) == 0 else out


# --------------------------------------------------------------------------------------
# Adduct ion formulas (private adapter: adducts.py has shifts, not element deltas)
# --------------------------------------------------------------------------------------

_TERM_RE = re.compile(r"([+-])(\d*)([A-Za-z][A-Za-z0-9]*)")
_ABBREV_LOWER = {k.lower(): v for k, v in ABBREVIATIONS.items()}


@lru_cache(maxsize=1024)
def adduct_delta(adduct: str) -> tuple[int, tuple[int, ...], int] | None:
    """``(multiplier, delta, charge)`` of an adduct string, where the ion formula is
    ``multiplier x M + delta`` (``delta`` in ``ION_ELEMENTS`` order, signed) and
    ``charge`` is signed. ``[M+Na]+`` -> (1, Na +1, +1); ``[M-H2O-H]-`` -> (1, H -3 O -1,
    -1). None if the string is not ``[kM+A-B...]z`` with terms over ``ION_ELEMENTS``.
    The mass of ``delta`` minus ``charge`` electrons equals ``adducts`` shift (tested)."""
    if not isinstance(adduct, str):
        return None
    text = re.sub(r"\s+", "", adduct)
    m = re.match(r"^\[(\d*)M(.*)\](\d*)([+-])$", text)
    if m is None:
        return None
    mult = int(m.group(1)) if m.group(1) else 1
    body = m.group(2)
    z = int(m.group(3)) if m.group(3) else 1
    charge = z if m.group(4) == "+" else -z
    delta = np.zeros(len(ION_ELEMENTS), dtype=np.int64)
    pos = 0
    for t in _TERM_RE.finditer(body):
        if t.start() != pos:
            return None
        sign, k, token = t.groups()
        formula = ABBREVIATIONS.get(token) or _ABBREV_LOWER.get(token.lower()) or token
        cnt = formula_counts(formula, ion=True)
        if cnt is None:
            return None
        delta += (1 if sign == "+" else -1) * (int(k) if k else 1) * cnt
        pos = t.end()
    if pos != len(body) or mult < 1 or charge == 0:
        return None
    return mult, tuple(int(v) for v in delta), charge


# --------------------------------------------------------------------------------------
# Enumeration
# --------------------------------------------------------------------------------------

@njit(cache=True)
def _enum_kernel(lo, hi, em, c_min, hc_min, hc_max, nc_max, oc_max, p_max, s_max, hal_max):
    mC, mH, mN, mO, mP, mS = em[0], em[1], em[2], em[3], em[4], em[5]
    mF, mCl, mBr, mI = em[6], em[7], em[8], em[9]
    eps = 1e-9
    cap = 256
    out = np.empty((cap, 10), dtype=np.int16)
    n = 0
    for nI in range(hal_max + 1):
        m_i = nI * mI
        if m_i > hi:
            break
        for nBr in range(hal_max - nI + 1):
            m_br = m_i + nBr * mBr
            if m_br > hi:
                break
            for nCl in range(hal_max - nI - nBr + 1):
                m_cl = m_br + nCl * mCl
                if m_cl > hi:
                    break
                for nF in range(hal_max - nI - nBr - nCl + 1):
                    m_f = m_cl + nF * mF
                    if m_f > hi:
                        break
                    nX = nI + nBr + nCl + nF
                    for nS in range(s_max + 1):
                        m_s = m_f + nS * mS
                        if m_s > hi:
                            break
                        for nP in range(p_max + 1):
                            m_p = m_s + nP * mP
                            if m_p > hi:
                                break
                            c_hi = int((hi - m_p) / mC)
                            for nC in range(c_min, c_hi + 1):
                                m_c = m_p + nC * mC
                                h_min = int(math.ceil(hc_min * nC - eps))
                                if h_min < 0:
                                    h_min = 0
                                h_ratio = int(math.floor(hc_max * nC + eps))
                                n_hi = int(math.floor(nc_max * nC + eps))
                                n_mass = int((hi - m_c) / mN)
                                if n_mass < n_hi:
                                    n_hi = n_mass
                                o_ratio = int(math.floor(oc_max * nC + eps))
                                for nN in range(n_hi + 1):
                                    m_n = m_c + nN * mN
                                    h_max = 2 * nC + nN + nP + 2 - nX   # RDBE >= 0
                                    if h_ratio < h_max:
                                        h_max = h_ratio
                                    if h_max < h_min:
                                        continue
                                    o_lo = int(math.ceil((lo - m_n - h_max * mH) / mO - eps))
                                    if o_lo < 0:
                                        o_lo = 0
                                    o_hi = int(math.floor((hi - m_n - h_min * mH) / mO + eps))
                                    if o_ratio < o_hi:
                                        o_hi = o_ratio
                                    for nO in range(o_lo, o_hi + 1):
                                        m_o = m_n + nO * mO
                                        h_lo = int(math.ceil((lo - m_o) / mH - eps))
                                        if h_lo < h_min:
                                            h_lo = h_min
                                        h_hi = int(math.floor((hi - m_o) / mH + eps))
                                        if h_hi > h_max:
                                            h_hi = h_max
                                        if h_lo > h_hi:
                                            continue
                                        # integer RDBE: H + X + N + P even
                                        if (h_lo + nX + nN + nP) % 2 == 1:
                                            h_lo += 1
                                        for nH in range(h_lo, h_hi + 1, 2):
                                            mass = m_o + nH * mH
                                            if mass < lo or mass > hi:
                                                continue
                                            vsum = (4 * nC + nH + 3 * nN + 2 * nO + 3 * nP
                                                    + 2 * nS + nX)
                                            if vsum < 8:         # Senior: >= 2 x max valence
                                                continue
                                            if n == cap:
                                                new = np.empty((cap * 2, 10), dtype=np.int16)
                                                new[:cap] = out
                                                out = new
                                                cap *= 2
                                            out[n, 0] = nC
                                            out[n, 1] = nH
                                            out[n, 2] = nN
                                            out[n, 3] = nO
                                            out[n, 4] = nP
                                            out[n, 5] = nS
                                            out[n, 6] = nF
                                            out[n, 7] = nCl
                                            out[n, 8] = nBr
                                            out[n, 9] = nI
                                            n += 1
    return out[:n]


@dataclass(frozen=True)
class FormulaSet:
    """Candidate formulas of one neutral mass. ``counts`` is (n, 10) int16 in ``ELEMENTS``
    order; ``mass`` the monoisotopic masses; ``ppm`` = (mass - query) / query x 1e6."""

    counts: np.ndarray
    mass: np.ndarray
    ppm: np.ndarray
    query_mass: float

    def __len__(self) -> int:
        return int(self.counts.shape[0])

    @property
    def formulas(self) -> list[str]:
        return [hill(c) for c in self.counts]

    @property
    def rdbe(self) -> np.ndarray:
        return np.asarray(rdbe(self.counts), dtype=np.float64).reshape(-1)

    def index(self, formula: str) -> int:
        """Row of ``formula`` (any element order), -1 if absent."""
        c = formula_counts(formula)
        if c is None or len(self) == 0:
            return -1
        hit = np.flatnonzero((self.counts == c[None, :]).all(axis=1))
        return int(hit[0]) if hit.size else -1


def formula_ok(counts: np.ndarray, cfg: Config | None = None) -> np.ndarray:
    """Vectorised version of the enumeration rules for (n, 10) counts: integer RDBE >= 0,
    Senior valence sum, carbon minimum and the ``formula_*`` ratio and count limits.
    ``enumerate_formulas`` returns exactly the formulas in the window that pass this."""
    cfg = cfg or Config()
    c = np.asarray(counts, dtype=np.int64).reshape(-1, 10)
    C, H, N, O, P, S = (c[:, i] for i in range(6))
    X = c[:, 6:10].sum(axis=1)
    eps = 1e-9
    ok = (c >= 0).all(axis=1) & (C >= max(1, int(cfg.formula_c_min)))
    ok &= ((H + X + N + P) % 2 == 0) & (2 * C + N + P + 2 - X - H >= 0)
    ok &= (4 * C + H + 3 * N + 2 * O + 3 * P + 2 * S + X) >= 8
    Cf = C.astype(np.float64)
    ok &= H >= np.ceil(cfg.formula_hc_min * Cf - eps)
    ok &= H <= np.floor(cfg.formula_hc_max * Cf + eps)
    ok &= N <= np.floor(cfg.formula_nc_max * Cf + eps)
    ok &= O <= np.floor(cfg.formula_oc_max * Cf + eps)
    ok &= (P <= cfg.formula_p_max) & (S <= cfg.formula_s_max) & (X <= cfg.formula_halogen_max)
    return ok


def _window(mass: float, tol_ppm: float, min_tol_da: float) -> tuple[float, float]:
    tol = max(tol_ppm * 1e-6 * mass, min_tol_da)
    return mass - tol, mass + tol


def enumerate_formulas(mass: float, cfg: Config | None = None, tol_ppm: float | None = None,
                       min_tol_da: float | None = None) -> FormulaSet:
    """All formulas over ``ELEMENTS`` within the mass window that pass the RDBE, Senior and
    ratio rules (module docstring). Window defaults: ``cfg.mass_tol_ppm`` and
    ``cfg.mass_tol_min_da``. Sorted by |ppm|, then by counts (C, H, N, ...)."""
    cfg = cfg or Config()
    tol_ppm = cfg.mass_tol_ppm if tol_ppm is None else float(tol_ppm)
    min_tol_da = cfg.mass_tol_min_da if min_tol_da is None else float(min_tol_da)
    try:
        mass = float(mass)
    except (TypeError, ValueError):
        mass = float("nan")
    empty = FormulaSet(np.zeros((0, 10), np.int16), np.zeros(0), np.zeros(0), mass)
    if not (math.isfinite(mass) and mass > 0):
        return empty
    lo, hi = _window(float(mass), tol_ppm, min_tol_da)
    em = _em10()
    counts = _enum_kernel(lo, hi, em, int(max(1, cfg.formula_c_min)), float(cfg.formula_hc_min),
                          float(cfg.formula_hc_max), float(cfg.formula_nc_max),
                          float(cfg.formula_oc_max), int(cfg.formula_p_max),
                          int(cfg.formula_s_max), int(cfg.formula_halogen_max))
    if counts.shape[0] == 0:
        return empty
    masses = counts.astype(np.float64) @ em
    ppm = (masses - mass) / mass * 1e6
    keys = [counts[:, j] for j in range(9, -1, -1)] + [np.round(np.abs(ppm), 9)]
    order = np.lexsort(keys)
    return FormulaSet(np.ascontiguousarray(counts[order]), masses[order], ppm[order],
                      float(mass))


# --------------------------------------------------------------------------------------
# Sub-formula explanation
# --------------------------------------------------------------------------------------

@dataclass
class PeakSet:
    """One molecule's cleaned fragment peaks, grouped by adduct (sorted names).

    ``fmass``: fragment formula masses (m/z + sign x electron mass), sorted inside each
    group; ``ptol``: per-peak tolerance (Da); ``inten``: cleaned intensities (base peak 1
    per spectrum, pooled); ``goff``: group offsets (G + 1); ``gdelta``: (G, 12) adduct
    deltas; ``gmult``: multipliers; ``adducts``: group names.
    """

    fmass: np.ndarray
    ptol: np.ndarray
    inten: np.ndarray
    goff: np.ndarray
    gdelta: np.ndarray
    gmult: np.ndarray
    adducts: tuple[str, ...]
    n_spectra: int = 0
    skipped: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_peaks(self) -> int:
        return int(self.fmass.shape[0])

    @property
    def total_intensity(self) -> float:
        return float(self.inten.sum())


@njit(cache=True)
def _match_peaks(U, fm, ptol, tol_max, em, expl):
    """Mark peaks of one group (``fm`` sorted) explained by a sub-formula of ``U``.

    Heteroatom counts (I, Br, K, Cl, Na, S, P, F, O, N) are enumerated outside, pruned by
    the largest peak mass; for each combination and each still unexplained peak at or
    above its mass, every carbon count that leaves a residual reachable with 0..U_H
    hydrogens is tried and the nearest H count checked against the peak tolerance and the
    fragment RDBE limit (H + X + Na + K <= 2C + N + P + 3). Returns the explained count.
    """
    mC, mH, mN, mO, mP, mS = em[0], em[1], em[2], em[3], em[4], em[5]
    mF, mCl, mBr, mI, mNa, mK = em[6], em[7], em[8], em[9], em[10], em[11]
    npk = fm.shape[0]
    n_expl = 0
    for k in range(npk):
        if expl[k]:
            n_expl += 1
    if n_expl == npk:
        return n_expl
    hi_m = fm[npk - 1] + tol_max
    uC = U[0]
    uH = U[1]
    for nI in range(U[9] + 1):
        m_i = nI * mI
        if m_i > hi_m:
            break
        for nBr in range(U[8] + 1):
            m_br = m_i + nBr * mBr
            if m_br > hi_m:
                break
            for nK in range(U[11] + 1):
                m_k = m_br + nK * mK
                if m_k > hi_m:
                    break
                for nCl in range(U[7] + 1):
                    m_cl = m_k + nCl * mCl
                    if m_cl > hi_m:
                        break
                    for nNa in range(U[10] + 1):
                        m_na = m_cl + nNa * mNa
                        if m_na > hi_m:
                            break
                        for nS in range(U[5] + 1):
                            m_s = m_na + nS * mS
                            if m_s > hi_m:
                                break
                            for nP in range(U[4] + 1):
                                m_p = m_s + nP * mP
                                if m_p > hi_m:
                                    break
                                for nF in range(U[6] + 1):
                                    m_f = m_p + nF * mF
                                    if m_f > hi_m:
                                        break
                                    nX = nI + nBr + nCl + nF + nNa + nK
                                    for nO in range(U[3] + 1):
                                        m_o = m_f + nO * mO
                                        if m_o > hi_m:
                                            break
                                        for nN in range(U[2] + 1):
                                            m_h = m_o + nN * mN
                                            if m_h > hi_m:
                                                break
                                            rd = nN + nP + 3 - nX
                                            start = np.searchsorted(fm, m_h - tol_max)
                                            for p in range(start, npk):
                                                if expl[p]:
                                                    continue
                                                r = fm[p] - m_h
                                                t = ptol[p]
                                                c_lo = int(math.ceil((r - t - mH * uH) / mC))
                                                if c_lo < 0:
                                                    c_lo = 0
                                                c_hi = int(math.floor((r + t) / mC))
                                                if c_hi > uC:
                                                    c_hi = uC
                                                for nC in range(c_lo, c_hi + 1):
                                                    rem = r - nC * mC
                                                    nH = int(math.floor(rem / mH + 0.5))
                                                    if nH < 0 or nH > uH or nH > 2 * nC + rd:
                                                        continue
                                                    if abs(rem - nH * mH) <= t:
                                                        expl[p] = True
                                                        n_expl += 1
                                                        break
                                                if n_expl == npk:
                                                    return n_expl
    return n_expl


@njit(cache=True)
def _explain_kernel(cands, fm, ptol, inten, goff, gdelta, gmult, em):
    n = cands.shape[0]
    g_n = goff.shape[0] - 1
    out_int = np.zeros(n, dtype=np.float64)
    out_cnt = np.zeros(n, dtype=np.int64)
    out_ok = np.ones(n, dtype=np.bool_)
    U = np.zeros(12, dtype=np.int64)
    max_pk = 1
    for g in range(g_n):
        if goff[g + 1] - goff[g] > max_pk:
            max_pk = goff[g + 1] - goff[g]
    expl = np.zeros(max_pk, dtype=np.bool_)
    tmax = np.zeros(g_n)
    for g in range(g_n):
        for p in range(goff[g], goff[g + 1]):
            if ptol[p] > tmax[g]:
                tmax[g] = ptol[p]
    for i in range(n):
        for g in range(g_n):
            p0, p1 = goff[g], goff[g + 1]
            ok = True
            for e in range(12):
                v = gdelta[g, e]
                if e < 10:
                    v += gmult[g] * cands[i, e]
                U[e] = v
                if v < 0:
                    ok = False
            if not ok:
                out_ok[i] = False
                continue
            npk = p1 - p0
            if npk == 0:
                continue
            ex = expl[:npk]
            for k in range(npk):
                ex[k] = False
            cnt = _match_peaks(U, fm[p0:p1], ptol[p0:p1], tmax[g], em, ex)
            s = 0.0
            for k in range(npk):
                if ex[k]:
                    s += inten[p0 + k]
            out_int[i] += s
            out_cnt[i] += cnt
    return out_int, out_cnt, out_ok


def spectra_from_frame(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, float, str]]:
    """``(mz, intensity, precursor_mz, adduct)`` tuples from rows in the test.parquet schema
    (one molecule's spectra), in row order."""
    return [(np.asarray(m, dtype=np.float64), np.asarray(x, dtype=np.float64), float(p), str(a))
            for m, x, p, a in zip(df["ms2_mzs"], df["ms2_normalized_intensities"],
                                  df["precursor_mz"], df["adduct"])]


def prepare_peaks(spectra: Iterable[Any], cfg: Config | None = None, tol_da: float | None = None,
                  cleaned: bool = False, tol_ppm: float = 0.0) -> PeakSet:
    """Group one molecule's spectra by adduct for ``explain``.

    ``spectra``: ``(mz, intensity, precursor_mz, adduct)`` tuples (see
    ``spectra_from_frame``). Peaks are cleaned with ``spectra.clean`` unless ``cleaned``.
    Per-peak tolerance ``max(tol_da, tol_ppm x 1e-6 x m/z)``; ``tol_da`` defaults to
    ``cfg.frag_tol_same_da`` (timsTOF). Spectra whose adduct cannot be written as an
    element delta are skipped (listed in ``skipped``).
    """
    from .spectra import clean

    cfg = cfg or Config()
    tol_abs = float(cfg.frag_tol_same_da if tol_da is None else tol_da)
    groups: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    deltas: dict[str, tuple[int, tuple[int, ...], int]] = {}
    skipped: list[str] = []
    n_spec = 0
    for item in spectra:
        mz, it, prec, add = item[0], item[1], item[2], item[3]
        d = adduct_delta(add)
        if d is None:
            skipped.append(str(add))
            continue
        mz = np.asarray(mz, dtype=np.float64)
        it = np.asarray(it, dtype=np.float64)
        if not cleaned:
            mz, it, _ = clean(mz, it, prec, cfg)
            mz = mz.astype(np.float64)
            it = it.astype(np.float64)
        sign = 1.0 if d[2] > 0 else -1.0
        groups.setdefault(add, []).append((mz + sign * ELECTRON_MASS, it))
        deltas[add] = d
        n_spec += 1
    names = tuple(sorted(groups))
    fms, ints, goff = [], [], [0]
    gdelta = np.zeros((len(names), len(ION_ELEMENTS)), dtype=np.int64)
    gmult = np.ones(len(names), dtype=np.int64)
    for g, name in enumerate(names):
        f = np.concatenate([a for a, _ in groups[name]])
        x = np.concatenate([b for _, b in groups[name]])
        keep = np.isfinite(f) & np.isfinite(x) & (x > 0) & (f > 0)
        f, x = f[keep], x[keep]
        order = np.argsort(f, kind="mergesort")
        fms.append(f[order])
        ints.append(x[order])
        goff.append(goff[-1] + len(f))
        gdelta[g] = deltas[name][1]
        gmult[g] = deltas[name][0]
    fm = np.concatenate(fms) if fms else np.zeros(0)
    inten = np.concatenate(ints) if ints else np.zeros(0)
    ptol = np.maximum(tol_abs, float(tol_ppm) * 1e-6 * np.abs(fm))
    if skipped:
        log.warning("prepare_peaks: skipped spectra with adducts %s", sorted(set(skipped)))
    return PeakSet(fm, ptol, inten, np.asarray(goff, dtype=np.int64), gdelta, gmult, names,
                   n_spec, tuple(skipped))


def explain(counts: np.ndarray | FormulaSet | Sequence[str], peaks: PeakSet
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sub-formula explanation of ``peaks`` for each candidate neutral formula.

    Returns ``(expl_frac, expl_peaks, adduct_ok)``: explained fraction of the pooled
    cleaned intensity, explained fraction of the peak count (both NaN when the molecule
    has no peaks), and whether every adduct gives a valid ion formula.
    """
    if isinstance(counts, FormulaSet):
        c = counts.counts
    elif len(counts) and isinstance(counts[0], str):  # type: ignore[index]
        rows = [formula_counts(s) for s in counts]  # type: ignore[union-attr]
        if any(r is None for r in rows):
            raise ValueError("explain: unparseable formula among candidates")
        c = np.asarray(rows)
    else:
        c = np.asarray(counts)
    c = np.ascontiguousarray(c, dtype=np.int64).reshape(-1, 10)
    tot, cnt, ok = _explain_kernel(c, peaks.fmass, peaks.ptol, peaks.inten, peaks.goff,
                                   peaks.gdelta, peaks.gmult, element_masses())
    total = peaks.total_intensity
    if peaks.n_peaks == 0 or total <= 0:
        nan = np.full(c.shape[0], np.nan)
        return nan, nan.copy(), ok
    return tot / total, cnt / float(peaks.n_peaks), ok


# --------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------

# Model features (FormulaModel). Raw table columns are a superset.
FORMULA_FEATURES: tuple[str, ...] = (
    "abs_ppm", "expl_frac", "expl_peaks", "expl_frac_strict", "rdbe", "hc", "nc", "oc",
    "has_N", "has_P", "has_S", "has_hal", "n_hetero", "log_db_np", "log_db_pubchem",
    "log_lib", "log_n_formulas",
)


def db_formula_counts(db_frame: pd.DataFrame | None) -> dict[str, tuple[int, int]]:
    """``{hill formula: (n NP-tier rows, n PubChem rows)}`` from a ``db.lookup`` frame
    (columns ``formula``, ``in_np``, ``in_pubchem``; ``tier`` as fallback)."""
    out: dict[str, tuple[int, int]] = {}
    if db_frame is None or len(db_frame) == 0 or "formula" not in db_frame.columns:
        return out
    f = db_frame["formula"].astype(object).to_numpy()
    if "in_np" in db_frame.columns:
        is_np = db_frame["in_np"].fillna(False).to_numpy(dtype=bool)
    else:
        is_np = (db_frame.get("tier", pd.Series(["np"] * len(f))) == "np").to_numpy()
    if "in_pubchem" in db_frame.columns:
        is_pc = db_frame["in_pubchem"].fillna(False).to_numpy(dtype=bool)
    else:
        is_pc = (db_frame.get("tier", pd.Series([""] * len(f))) == "pubchem").to_numpy()
    norm: dict[str, str | None] = {}
    acc: dict[str, list[int]] = {}
    for s, a, b in zip(f, is_np, is_pc):
        if not isinstance(s, str):
            continue
        if s not in norm:
            c = formula_counts(s)
            norm[s] = hill(c) if c is not None else None
        h = norm[s]
        if h is None:
            continue
        v = acc.setdefault(h, [0, 0])
        v[0] += int(a)
        v[1] += int(b)
    return {k: (v[0], v[1]) for k, v in sorted(acc.items())}


def _lib_count(lib_counts: Mapping[str, int] | Callable[[str], int] | None, f: str) -> int:
    if lib_counts is None:
        return 0
    if callable(lib_counts):
        return int(lib_counts(f))
    return int(lib_counts.get(f, 0))


def formula_features(mass: float, spectra: Iterable[Any] | PeakSet | None = None,
                     cfg: Config | None = None, *, formulas: FormulaSet | None = None,
                     extra_formulas: Sequence[str] = (), db_frame: pd.DataFrame | None = None,
                     lookup_db: bool = False,
                     lib_counts: Mapping[str, int] | Callable[[str], int] | None = None,
                     require_adduct_ok: bool = True, tol_da: float | None = None,
                     expl_max: int | None = None) -> pd.DataFrame:
    """One row per candidate formula of neutral mass ``mass``.

    Candidates: ``formulas`` (default ``enumerate_formulas(mass, cfg)``) plus
    ``extra_formulas`` (e.g. formulas of DB candidates that fail the ratio rules), deduped.
    ``spectra``: tuples for ``prepare_peaks`` or a ``PeakSet``; None gives NaN explanation.
    DB counts come from ``db_frame`` (a ``db.lookup`` result for this mass) or, with
    ``lookup_db``, from ``db.lookup(mass)`` (zero when the DB is unavailable).
    ``lib_counts``: library structures per Hill formula (NEEDS_TRAIN hook; 0 when None).
    ``require_adduct_ok`` drops formulas the molecule's adducts cannot be applied to.

    Explanation: ``expl_frac`` / ``expl_peaks`` at ``tol_da`` (default
    ``cfg.frag_tol_same_da``) and ``expl_frac_strict`` at max(``formula_strict_tol_da``,
    ``formula_strict_tol_ppm`` x m/z). It is computed for at most ``expl_max`` (default
    ``cfg.formula_expl_max``) formulas per molecule, chosen in this order: formulas with
    DB or library structures first, then fewer P + S + halogen atoms, then fewer N, then
    smaller |ppm| (counts as tie-breaker). The rest get NaN (``expl_done`` False); above
    about 1,000 Da the enumeration has 10k-30k formulas and explaining one costs about
    0.6 ms there.

    Columns: formula, mono_mass, ppm_err, abs_ppm, C..I, rdbe, n_hal, n_hetero, hc, nc,
    oc, has_N, has_P, has_S, has_hal, adduct_ok, db_count_np, db_count_pubchem,
    lib_count, log_db_np, log_db_pubchem, log_lib, in_enumeration, expl_done, expl_frac,
    expl_peaks, expl_frac_strict, n_formulas, log_n_formulas. Row order: as enumerated,
    extras last.
    """
    cfg = cfg or Config()
    fs = formulas if formulas is not None else enumerate_formulas(mass, cfg)
    counts = [fs.counts.astype(np.int64)]
    seen = {tuple(int(v) for v in r) for r in fs.counts}
    n_enum = len(fs)
    extra_rows = []
    for s in extra_formulas:
        c = formula_counts(s)
        if c is None:
            continue
        key = tuple(int(v) for v in c)
        if key in seen:
            continue
        seen.add(key)
        extra_rows.append(c)
    if extra_rows:
        counts.append(np.asarray(extra_rows, dtype=np.int64))
    c = np.concatenate(counts, axis=0) if counts else np.zeros((0, 10), np.int64)
    n = c.shape[0]
    mono = counts_mass(c) if n else np.zeros(0)
    mono = np.asarray(mono, dtype=np.float64).reshape(-1)
    ppm = (mono - mass) / mass * 1e6
    names = [hill(r) for r in c]
    df = pd.DataFrame({"formula": pd.Series(names, dtype=object), "mono_mass": mono,
                       "ppm_err": ppm, "abs_ppm": np.abs(ppm)})
    for j, e in enumerate(ELEMENTS):
        df[e] = c[:, j].astype(np.int64)
    carbons = np.maximum(c[:, 0], 1).astype(np.float64)
    df["rdbe"] = np.asarray(rdbe(c), dtype=np.float64).reshape(-1) if n else np.zeros(0)
    df["n_hal"] = c[:, 6:10].sum(axis=1)
    df["n_hetero"] = c[:, 2:10].sum(axis=1)
    df["hc"] = c[:, 1] / carbons
    df["nc"] = c[:, 2] / carbons
    df["oc"] = c[:, 3] / carbons
    df["has_N"] = (c[:, 2] > 0).astype(np.float64)
    df["has_P"] = (c[:, 4] > 0).astype(np.float64)
    df["has_S"] = (c[:, 5] > 0).astype(np.float64)
    df["has_hal"] = (df["n_hal"].to_numpy() > 0).astype(np.float64)

    # adduct compatibility (every adduct must give non-negative ion counts)
    peaks: PeakSet | None = None
    if spectra is not None:
        peaks = spectra if isinstance(spectra, PeakSet) else prepare_peaks(spectra, cfg, tol_da)
    ok = np.ones(n, dtype=bool)
    if peaks is not None:
        for g in range(len(peaks.adducts)):
            ion = peaks.gmult[g] * c + peaks.gdelta[g][None, :10]
            ok &= (ion >= 0).all(axis=1)
    df["adduct_ok"] = ok

    if db_frame is None and lookup_db:
        try:
            from . import db as _db
            db_frame = _db.lookup(mass)
        except Exception as exc:  # noqa: BLE001 - DB missing must not break inference
            log.warning("formula_features: db.lookup(%.5f) failed: %s", mass, exc)
            db_frame = None
    dbc = db_formula_counts(db_frame)
    df["db_count_np"] = np.array([dbc.get(f, (0, 0))[0] for f in names], dtype=np.int64)
    df["db_count_pubchem"] = np.array([dbc.get(f, (0, 0))[1] for f in names], dtype=np.int64)
    df["lib_count"] = np.array([_lib_count(lib_counts, f) for f in names], dtype=np.int64)
    df["log_db_np"] = np.log1p(df["db_count_np"].to_numpy(dtype=np.float64))
    df["log_db_pubchem"] = np.log1p(df["db_count_pubchem"].to_numpy(dtype=np.float64))
    df["log_lib"] = np.log1p(df["lib_count"].to_numpy(dtype=np.float64))
    df["in_enumeration"] = np.arange(n) < n_enum
    if require_adduct_ok and n:
        keep = df["adduct_ok"].to_numpy()
        df = df[keep].reset_index(drop=True)
        c = c[keep]
        n = c.shape[0]

    ef = np.full(n, np.nan)
    ep = np.full(n, np.nan)
    es = np.full(n, np.nan)
    done = np.zeros(n, dtype=bool)
    if peaks is not None and n:
        cap = int(cfg.formula_expl_max if expl_max is None else expl_max)
        if n > cap:
            support = (df["db_count_np"].to_numpy() + df["db_count_pubchem"].to_numpy()
                       + df["lib_count"].to_numpy()) > 0
            keys = ([c[:, j] for j in range(9, -1, -1)]
                    + [np.round(df["abs_ppm"].to_numpy(), 9), c[:, 2],
                       c[:, 4] + c[:, 5] + c[:, 6:10].sum(axis=1), ~support])
            sel = np.sort(np.lexsort(keys)[:cap])
        else:
            sel = np.arange(n)
        strict = replace(peaks, ptol=np.maximum(float(cfg.formula_strict_tol_da),
                                                 float(cfg.formula_strict_tol_ppm) * 1e-6
                                                 * np.abs(peaks.fmass)))
        a, b, _ = explain(c[sel], peaks)
        ef[sel], ep[sel] = a, b
        es[sel] = explain(c[sel], strict)[0]
        done[sel] = True
    df["expl_done"] = done
    df["expl_frac"] = ef
    df["expl_peaks"] = ep
    df["expl_frac_strict"] = es
    df["n_formulas"] = len(df)
    df["log_n_formulas"] = np.log1p(float(len(df)))
    return df


# --------------------------------------------------------------------------------------
# Logistic model
# --------------------------------------------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


# Placeholder coefficients on raw (unstandardised) features, used until a fitted
# models/formula_v1.json exists. Signs follow the plan's reasoning (small mass error,
# high explanation, many DB structures with the formula, few unusual elements); the
# magnitudes are guesses. NEEDS_TRAIN: replace by fit_formula_model on holdout molecules.
_DEFAULT_COEF: dict[str, float] = {
    "abs_ppm": -0.35, "expl_frac": 2.0, "expl_peaks": 0.5, "expl_frac_strict": 3.0,
    "rdbe": 0.0, "hc": 0.0, "nc": -1.0, "oc": 0.0, "has_N": 0.0, "has_P": -1.5,
    "has_S": -1.0, "has_hal": -2.0,
    "n_hetero": -0.05, "log_db_np": 1.0, "log_db_pubchem": 0.6, "log_lib": 0.5,
    "log_n_formulas": -0.5,
}
_DEFAULT_INTERCEPT = -3.0


class FormulaModel:
    """L2-regularised logistic regression on ``features`` (numpy only).

    ``decision = intercept + coef . ((x - mean) / scale)``; NaN inputs are replaced by
    ``mean`` (contribute 0). ``fit`` standardises with the training mean and std and uses
    Newton (IRLS) steps with an L2 penalty on the coefficients (not the intercept), so it
    is deterministic. ``predict_proba`` returns P(formula is correct), shape (n,).
    """

    def __init__(self, features: Sequence[str] = FORMULA_FEATURES,
                 coef: Sequence[float] | None = None, intercept: float = 0.0,
                 mean: Sequence[float] | None = None, scale: Sequence[float] | None = None,
                 l2: float = 1.0, meta: Mapping[str, Any] | None = None) -> None:
        self.features = tuple(features)
        k = len(self.features)
        self.coef = np.zeros(k) if coef is None else np.asarray(coef, dtype=np.float64)
        self.intercept = float(intercept)
        self.mean = np.zeros(k) if mean is None else np.asarray(mean, dtype=np.float64)
        self.scale = np.ones(k) if scale is None else np.asarray(scale, dtype=np.float64)
        self.l2 = float(l2)
        self.meta: dict[str, Any] = dict(meta or {})
        for name, arr in (("coef", self.coef), ("mean", self.mean), ("scale", self.scale)):
            if arr.shape != (k,):
                raise ValueError(f"FormulaModel: {name} has shape {arr.shape}, expected ({k},)")

    # ---------------------------------------------------------------- construction
    @classmethod
    def default(cls) -> "FormulaModel":
        """Hand-set placeholder (``meta['fitted'] = False``)."""
        feats = FORMULA_FEATURES
        return cls(feats, [_DEFAULT_COEF[f] for f in feats], _DEFAULT_INTERCEPT,
                   meta={"fitted": False, "note": "hand-set placeholder, not fitted"})

    @property
    def fitted(self) -> bool:
        return bool(self.meta.get("fitted", False))

    # ---------------------------------------------------------------- design matrix
    def _matrix(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            missing = [f for f in self.features if f not in X.columns]
            if missing:
                raise KeyError(f"FormulaModel: missing feature columns {missing}")
            A = X[list(self.features)].to_numpy(dtype=np.float64, na_value=np.nan)
        else:
            A = np.asarray(X, dtype=np.float64)
            if A.ndim == 1:
                A = A.reshape(1, -1)
            if A.shape[1] != len(self.features):
                raise ValueError(f"FormulaModel: expected {len(self.features)} columns, "
                                 f"got {A.shape[1]}")
        return A

    def _standardise(self, A: np.ndarray) -> np.ndarray:
        Z = (A - self.mean) / self.scale
        Z[~np.isfinite(Z)] = 0.0
        return Z

    # ---------------------------------------------------------------- fit / predict
    def fit(self, X: pd.DataFrame | np.ndarray, y: Sequence[float] | np.ndarray,
            sample_weight: Sequence[float] | np.ndarray | None = None, l2: float | None = None,
            max_iter: int = 100, tol: float = 1e-8) -> "FormulaModel":
        A = self._matrix(X)
        yv = np.asarray(y, dtype=np.float64).reshape(-1)
        if yv.shape[0] != A.shape[0]:
            raise ValueError("FormulaModel.fit: X and y differ in length")
        sw = (np.ones_like(yv) if sample_weight is None
              else np.asarray(sample_weight, dtype=np.float64).reshape(-1))
        if l2 is not None:
            self.l2 = float(l2)
        mean = np.zeros(A.shape[1])
        scale = np.ones(A.shape[1])
        for j in range(A.shape[1]):
            col = A[:, j]
            fin = np.isfinite(col)
            if fin.any():
                mean[j] = float(np.average(col[fin], weights=sw[fin]))
                sd = float(np.sqrt(np.average((col[fin] - mean[j]) ** 2, weights=sw[fin])))
                scale[j] = sd if sd > 1e-12 else 1.0
        self.mean, self.scale = mean, scale
        Z = np.hstack([np.ones((A.shape[0], 1)), self._standardise(A)])
        k = Z.shape[1]
        w = np.zeros(k)
        pos = float(np.sum(sw * yv)) / max(float(np.sum(sw)), 1e-12)
        if 0.0 < pos < 1.0:
            w[0] = math.log(pos / (1.0 - pos))
        pen = np.full(k, self.l2)
        pen[0] = 0.0
        n_iter = 0
        for n_iter in range(1, max_iter + 1):
            p = _sigmoid(Z @ w)
            grad = Z.T @ (sw * (yv - p)) - pen * w
            hw = sw * p * (1.0 - p) + 1e-12
            H = (Z * hw[:, None]).T @ Z + np.diag(pen)
            step = np.linalg.solve(H, grad)
            w = w + step
            if float(np.max(np.abs(step))) < tol:
                break
        self.intercept = float(w[0])
        self.coef = w[1:].copy()
        p = _sigmoid(Z @ w)
        eps = 1e-15
        ll = float(np.sum(sw * (yv * np.log(p + eps) + (1 - yv) * np.log(1 - p + eps))))
        self.meta.update({"fitted": True, "n_train": int(A.shape[0]),
                          "n_pos": int(np.sum(yv > 0.5)), "n_iter": n_iter,
                          "log_likelihood": ll, "l2": self.l2})
        return self

    def decision_function(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.intercept + self._standardise(self._matrix(X)) @ self.coef

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """P(correct formula) per row, shape (n,)."""
        return _sigmoid(self.decision_function(X))

    # ---------------------------------------------------------------- persistence
    def to_dict(self) -> dict[str, Any]:
        return {"format": "casmi.formula.FormulaModel", "version": 1,
                "features": list(self.features), "coef": self.coef.tolist(),
                "intercept": self.intercept, "mean": self.mean.tolist(),
                "scale": self.scale.tolist(), "l2": self.l2, "meta": self.meta}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FormulaModel":
        if d.get("format") != "casmi.formula.FormulaModel":
            raise ValueError("not a FormulaModel JSON")
        return cls(d["features"], d["coef"], d["intercept"], d["mean"], d["scale"],
                   d.get("l2", 1.0), d.get("meta", {}))

    def save(self, path: str | Path | None = None) -> Path:
        p = Path(path) if path is not None else FORMULA_MODEL_PATH
        paths.ensure_dir(p.parent)
        p.write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path | None = None) -> "FormulaModel":
        p = Path(path) if path is not None else FORMULA_MODEL_PATH
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))


_MODEL_CACHE: dict[str, FormulaModel] = {}


def load_formula_model(path: str | Path | None = None, fallback: bool = True) -> FormulaModel:
    """The fitted model at ``path`` (default ``models/formula_v1.json``), cached per path;
    the hand-set ``FormulaModel.default()`` (with a warning) if the file is absent and
    ``fallback``."""
    p = Path(path) if path is not None else FORMULA_MODEL_PATH
    key = str(p)
    if key not in _MODEL_CACHE:
        if p.is_file():
            _MODEL_CACHE[key] = FormulaModel.load(p)
        elif fallback:
            log.warning("formula model %s not found: using the hand-set default", p)
            _MODEL_CACHE[key] = FormulaModel.default()
        else:
            raise FileNotFoundError(p)
    return _MODEL_CACHE[key]


def rank_formulas(mass: float, spectra: Iterable[Any] | PeakSet | None = None,
                  cfg: Config | None = None, model: FormulaModel | None = None,
                  **feature_kwargs: Any) -> pd.DataFrame:
    """``formula_features`` plus ``prob`` (model probability) and ``rank`` (0 = best),
    sorted by prob descending, ties by abs_ppm then formula string."""
    cfg = cfg or Config()
    df = formula_features(mass, spectra, cfg, **feature_kwargs)
    model = model or load_formula_model()
    df["prob"] = model.predict_proba(df) if len(df) else np.zeros(0)
    df = df.sort_values(["prob", "abs_ppm", "formula"], ascending=[False, True, True],
                        kind="mergesort").reset_index(drop=True)
    df["rank"] = np.arange(len(df))
    return df


def select_targets(ranked: pd.DataFrame, cfg: Config | None = None, min_one: bool = True
                   ) -> list[str]:
    """GEN target formulas: the top ``gen_n_formulas`` rows of a ``rank_formulas`` frame
    with prob >= ``gen_formula_min_prob``; with ``min_one`` the best formula is returned
    even when it is below the threshold (an unfitted model is not calibrated)."""
    cfg = cfg or Config()
    if ranked is None or len(ranked) == 0:
        return []
    top = ranked.head(int(cfg.gen_n_formulas))
    keep = top[top["prob"] >= float(cfg.gen_formula_min_prob)]["formula"].tolist()
    if not keep and min_one:
        keep = [str(ranked["formula"].iloc[0])]
    return keep


def fit_formula_model(tables: Sequence[pd.DataFrame], true_formulas: Sequence[str],
                      l2: float = 1.0, features: Sequence[str] = FORMULA_FEATURES,
                      save_path: str | Path | None = None) -> FormulaModel:
    """NEEDS_TRAIN hook: fit ``FormulaModel`` on held-out molecules.

    ``tables[i]`` is ``formula_features`` of molecule i (cross-fitted inputs: DB and
    library counts must exclude the molecule itself, PLAN.md 5.6), ``true_formulas[i]``
    its label formula. Label = (Hill formula equals the truth). Molecules whose true
    formula is missing from their table still contribute negatives (the miss rate is the
    enumeration recall to report). Saves to ``save_path`` when given
    (``FORMULA_MODEL_PATH`` for the shipped model).
    TODO(WP12, NEEDS_TRAIN): call from the validation harness once holdout molecules and
    library formula counts exist; report top-1/top-3 accuracy in docs/EXPERIMENTS.md.
    """
    if len(tables) != len(true_formulas):
        raise ValueError("fit_formula_model: tables and true_formulas differ in length")
    xs, ys = [], []
    for t, f in zip(tables, true_formulas):
        if t is None or len(t) == 0:
            continue
        c = formula_counts(f)
        truth = hill(c) if c is not None else None
        xs.append(t)
        ys.append((t["formula"].to_numpy(dtype=object) == truth).astype(np.float64))
    if not xs:
        raise ValueError("fit_formula_model: no rows")
    X = pd.concat(xs, ignore_index=True)
    y = np.concatenate(ys)
    model = FormulaModel(features).fit(X, y, l2=l2)
    if save_path is not None:
        model.save(save_path)
    return model

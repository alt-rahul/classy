"""Spectral similarity (numba): greedy cosine, modified cosine, spectral entropy.

PLAN.md section 4.2 (WP5). Own implementations; ``matchms`` 0.34 and ``ms_entropy`` 1.5.3
are used only in tests to cross-check them (tests/test_similarity.py), never imported here.

Conventions
- Peaks are m/z-sorted arrays (any float dtype; computed in float64). Every function
  returns ``(score, n_matched_peaks)``.
- Argument order follows matchms: spectrum "a" is matchms' ``reference`` (its spec1),
  spectrum "b" is matchms' ``query`` (spec2). Scores are symmetric except for ties in the
  greedy assignment. The batch functions put the library spectrum in "a" and the query in
  "b", exactly like ``matchms.calculate_scores(references, queries)``.

Definitions
- ``cosine_greedy``: matchms ``CosineGreedy(tolerance, mz_power, intensity_power)``.
  Weights w = mz**mz_power * intensity**intensity_power (default 0 and 0.5: cosine on
  sqrt intensities). Candidate pairs are all (i, j) with
  mz_a[i] - tol <= mz_b[j] <= mz_a[i] + tol; they are ordered by w_a*w_b descending (ties:
  the pair found later in (i, j) order first, which is matchms' reversed stable argsort)
  and accepted greedily if neither peak is used yet. Score = sum of accepted products /
  (||w_a|| ||w_b||); 0 if either norm is 0.
- ``modified_cosine``: matchms ``ModifiedCosineGreedy`` (0.34's name for ModifiedCosine).
  shift = precursor_a - precursor_b; if |shift| <= tol it is ``cosine_greedy``.
  Otherwise candidate pairs are the unshifted pairs followed by the pairs matching after
  adding ``shift`` to mz_b, ordered and accepted as above.
- ``entropy_similarity``: ms_entropy ``calculate_entropy_similarity`` (``weighted=True``,
  the default, Li et al. 2021 entropy weighting) or
  ``calculate_unweighted_entropy_similarity`` (``weighted=False``), with
  ``clean_spectra=False`` and ``ms2_tolerance_in_da=tol``. Intensities are first
  normalised to sum 1 (ms_entropy expects that from its own ``clean_spectrum``; we use the
  PLAN.md 4.1 cleaning instead, so its centroiding and 1 percent noise filter are not
  applied). Weighting: S = -sum p ln p; if S < 3, p <- p**(0.25 + 0.25 S), renormalised.
  Matching is ms_entropy's two-pointer walk: a peak pair within ``tol`` is matched and
  both pointers advance. Similarity = sum over matched (f(pa + pb) - f(pa) - f(pb)) / 2
  with f(x) = x log2 x; it lies in [0, 1].
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

__all__ = [
    "cosine_greedy",
    "modified_cosine",
    "entropy_similarity",
    "cosine_batch",
    "modified_cosine_batch",
    "entropy_batch",
    "pair_scores_batch",
]


# --------------------------------------------------------------------------------------
# Greedy cosine core
# --------------------------------------------------------------------------------------


@njit(cache=True)
def _weights(mz, inten, mz_power, int_power, out):
    """matchms: (mz ** mz_power) * (intensity ** intensity_power); returns sum of w**2."""
    ss = 0.0
    for i in range(mz.shape[0]):
        w = (float(mz[i]) ** mz_power) * (float(inten[i]) ** int_power)
        out[i] = w
        ss += w ** 2
    return ss


@njit(cache=True)
def _count_matches(mz_a, mz_b, tol, shift):
    lowest = 0
    n = 0
    nb = mz_b.shape[0]
    for i in range(mz_a.shape[0]):
        m = float(mz_a[i])
        low = m - tol
        high = m + tol
        for j in range(lowest, nb):
            m2 = float(mz_b[j]) + shift
            if m2 > high:
                break
            if m2 < low:
                lowest = j + 1
            else:
                n += 1
    return n


@njit(cache=True)
def _fill_matches(mz_a, w_a, mz_b, w_b, tol, shift, bi, bj, bp, start):
    lowest = 0
    n = start
    nb = mz_b.shape[0]
    for i in range(mz_a.shape[0]):
        m = float(mz_a[i])
        low = m - tol
        high = m + tol
        for j in range(lowest, nb):
            m2 = float(mz_b[j]) + shift
            if m2 > high:
                break
            if m2 < low:
                lowest = j + 1
            else:
                bi[n] = i
                bj[n] = j
                bp[n] = w_a[i] * w_b[j]
                n += 1
    return n


@njit(cache=True)
def _greedy(bi, bj, bp, n, na, nb, ss_a, ss_b):
    if n == 0 or ss_a <= 0.0 or ss_b <= 0.0:
        return 0.0, 0
    order = np.argsort(bp[:n], kind="mergesort")
    used_a = np.zeros(na, dtype=np.bool_)
    used_b = np.zeros(nb, dtype=np.bool_)
    score = 0.0
    used = 0
    for t in range(n - 1, -1, -1):
        k = order[t]
        i = bi[k]
        j = bj[k]
        if not used_a[i] and not used_b[j]:
            score += bp[k]
            used_a[i] = True
            used_b[j] = True
            used += 1
    return score / (ss_a ** 0.5 * ss_b ** 0.5), used


@njit(cache=True)
def _cosine_core(mz_a, w_a, ss_a, mz_b, w_b, ss_b, tol, shift, modified):
    """Greedy (modified) cosine for precomputed weights. ``modified`` adds shifted pairs."""
    do_shift = modified and abs(shift) > tol
    n = _count_matches(mz_a, mz_b, tol, 0.0)
    if do_shift:
        n += _count_matches(mz_a, mz_b, tol, shift)
    if n == 0:
        return 0.0, 0
    bi = np.empty(n, dtype=np.int64)
    bj = np.empty(n, dtype=np.int64)
    bp = np.empty(n, dtype=np.float64)
    k = _fill_matches(mz_a, w_a, mz_b, w_b, tol, 0.0, bi, bj, bp, 0)
    if do_shift:
        k = _fill_matches(mz_a, w_a, mz_b, w_b, tol, shift, bi, bj, bp, k)
    return _greedy(bi, bj, bp, k, mz_a.shape[0], mz_b.shape[0], ss_a, ss_b)


@njit(cache=True)
def _cosine_pair(mz_a, int_a, mz_b, int_b, tol, shift, modified, mz_power, int_power):
    w_a = np.empty(mz_a.shape[0], dtype=np.float64)
    w_b = np.empty(mz_b.shape[0], dtype=np.float64)
    ss_a = _weights(mz_a, int_a, mz_power, int_power, w_a)
    ss_b = _weights(mz_b, int_b, mz_power, int_power, w_b)
    return _cosine_core(mz_a, w_a, ss_a, mz_b, w_b, ss_b, tol, shift, modified)


def _arr(x) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float64)


def cosine_greedy(mz_a, int_a, mz_b, int_b, tol: float, mz_power: float = 0.0,
                  intensity_power: float = 0.5) -> tuple[float, int]:
    """matchms CosineGreedy score and number of matched peaks (a = reference, b = query)."""
    s, n = _cosine_pair(_arr(mz_a), _arr(int_a), _arr(mz_b), _arr(int_b), float(tol), 0.0,
                        False, float(mz_power), float(intensity_power))
    return float(s), int(n)


def modified_cosine(mz_a, int_a, prec_a: float, mz_b, int_b, prec_b: float, tol: float,
                    mz_power: float = 0.0, intensity_power: float = 0.5) -> tuple[float, int]:
    """matchms ModifiedCosineGreedy: shift = prec_a - prec_b is added to mz_b."""
    s, n = _cosine_pair(_arr(mz_a), _arr(int_a), _arr(mz_b), _arr(int_b), float(tol),
                        float(prec_a) - float(prec_b), True, float(mz_power),
                        float(intensity_power))
    return float(s), int(n)


# --------------------------------------------------------------------------------------
# Entropy similarity core
# --------------------------------------------------------------------------------------


@njit(cache=True)
def _entropy_prepare(inten, weighted, out):
    """Normalise to sum 1 and apply ms_entropy's entropy weight. Returns False if empty."""
    n = inten.shape[0]
    s = 0.0
    for i in range(n):
        s += float(inten[i])
    if n == 0 or s <= 0.0:
        return False
    for i in range(n):
        out[i] = float(inten[i]) / s
    if weighted:
        ent = 0.0
        for i in range(n):
            p = out[i]
            if p > 0.0:
                ent -= p * math.log(p)
        if ent < 3.0:
            w = 0.25 + 0.25 * ent
            s2 = 0.0
            for i in range(n):
                out[i] = out[i] ** w
                s2 += out[i]
            for i in range(n):
                out[i] = out[i] / s2
    return True


@njit(cache=True)
def _xlog2x(x):
    return x * math.log2(x) if x > 0.0 else 0.0


@njit(cache=True)
def _entropy_core(mz_a, p_a, mz_b, p_b, tol):
    a = 0
    b = 0
    na = mz_a.shape[0]
    nb = mz_b.shape[0]
    sim = 0.0
    n = 0
    while a < na and b < nb:
        d = float(mz_a[a]) - float(mz_b[b])
        if d < -tol:
            a += 1
        elif d > tol:
            b += 1
        else:
            x = p_a[a]
            y = p_b[b]
            sim += _xlog2x(x + y) - _xlog2x(x) - _xlog2x(y)
            n += 1
            a += 1
            b += 1
    return sim / 2.0, n


@njit(cache=True)
def _entropy_pair(mz_a, int_a, mz_b, int_b, tol, weighted):
    p_a = np.empty(mz_a.shape[0], dtype=np.float64)
    p_b = np.empty(mz_b.shape[0], dtype=np.float64)
    if not _entropy_prepare(int_a, weighted, p_a) or not _entropy_prepare(int_b, weighted, p_b):
        return 0.0, 0
    return _entropy_core(mz_a, p_a, mz_b, p_b, tol)


def entropy_similarity(mz_a, int_a, mz_b, int_b, tol: float, weighted: bool = True
                       ) -> tuple[float, int]:
    """ms_entropy (un)weighted entropy similarity with clean_spectra=False semantics."""
    s, n = _entropy_pair(_arr(mz_a), _arr(int_a), _arr(mz_b), _arr(int_b), float(tol),
                         bool(weighted))
    return float(s), int(n)


# --------------------------------------------------------------------------------------
# Batch: one query against many references of a flat store
# --------------------------------------------------------------------------------------


@njit(cache=True)
def _batch_kernel(q_mz, q_int, q_prec, ref_mz, ref_int, ref_off, ref_prec, ref_idx, tols,
                  kind, mz_power, int_power, weighted):
    """kind: 0 cosine, 1 modified cosine, 2 entropy."""
    n = ref_idx.shape[0]
    scores = np.zeros(n, dtype=np.float64)
    matches = np.zeros(n, dtype=np.int32)
    nq = q_mz.shape[0]
    if nq == 0:
        return scores, matches
    qw = np.empty(nq, dtype=np.float64)
    if kind == 2:
        ok = _entropy_prepare(q_int, weighted, qw)
        if not ok:
            return scores, matches
        ss_q = 1.0
    else:
        ss_q = _weights(q_mz, q_int, mz_power, int_power, qw)
    max_len = 0
    for t in range(n):
        s = ref_idx[t]
        L = ref_off[s + 1] - ref_off[s]
        if L > max_len:
            max_len = L
    rw = np.empty(max(max_len, 1), dtype=np.float64)
    for t in range(n):
        s = ref_idx[t]
        a = ref_off[s]
        b = ref_off[s + 1]
        if b == a:
            continue
        rm = ref_mz[a:b]
        if kind == 2:
            if not _entropy_prepare(ref_int[a:b], weighted, rw):
                continue
            sc, nm = _entropy_core(rm, rw[: b - a], q_mz, qw, tols[t])
        else:
            ss_r = _weights(rm, ref_int[a:b], mz_power, int_power, rw)
            shift = ref_prec[s] - q_prec if kind == 1 else 0.0
            sc, nm = _cosine_core(rm, rw[: b - a], ss_r, q_mz, qw, ss_q, tols[t], shift,
                                  kind == 1)
        scores[t] = sc
        matches[t] = nm
    return scores, matches


def _batch_args(q_mz, q_int, ref_mz, ref_int, ref_offsets, ref_idx, tol):
    ref_off = np.ascontiguousarray(ref_offsets, dtype=np.int64)
    if ref_idx is None:
        ref_idx = np.arange(len(ref_off) - 1, dtype=np.int64)
    else:
        ref_idx = np.ascontiguousarray(ref_idx, dtype=np.int64)
    tols = np.ascontiguousarray(np.broadcast_to(np.asarray(tol, dtype=np.float64),
                                                ref_idx.shape))
    return _arr(q_mz), _arr(q_int), np.asarray(ref_mz), np.asarray(ref_int), ref_off, ref_idx, tols


def cosine_batch(q_mz, q_int, ref_mz, ref_int, ref_offsets, ref_idx=None, tol=0.01,
                 mz_power: float = 0.0, intensity_power: float = 0.5):
    """Greedy cosine of one query against store rows ``ref_idx`` (all if None).

    ``ref_mz``/``ref_int``/``ref_offsets`` are the flat store arrays (memmaps are fine);
    ``tol`` is a scalar or one value per reference. Returns (scores float64, matches int32)
    aligned with ``ref_idx``.
    """
    qm, qi, rm, ri, ro, idx, tols = _batch_args(q_mz, q_int, ref_mz, ref_int, ref_offsets,
                                                 ref_idx, tol)
    return _batch_kernel(qm, qi, 0.0, rm, ri, ro, np.zeros(1), idx, tols, 0,
                         float(mz_power), float(intensity_power), False)


def modified_cosine_batch(q_mz, q_int, q_prec: float, ref_mz, ref_int, ref_offsets,
                          ref_prec, ref_idx=None, tol=0.01, mz_power: float = 0.0,
                          intensity_power: float = 0.5):
    """Modified cosine of one query against store rows; ``ref_prec`` is indexed by row."""
    qm, qi, rm, ri, ro, idx, tols = _batch_args(q_mz, q_int, ref_mz, ref_int, ref_offsets,
                                                 ref_idx, tol)
    rp = np.ascontiguousarray(ref_prec, dtype=np.float64)
    return _batch_kernel(qm, qi, float(q_prec), rm, ri, ro, rp, idx, tols, 1,
                         float(mz_power), float(intensity_power), False)


def entropy_batch(q_mz, q_int, ref_mz, ref_int, ref_offsets, ref_idx=None, tol=0.01,
                  weighted: bool = True):
    """Entropy similarity of one query against store rows (see ``entropy_similarity``)."""
    qm, qi, rm, ri, ro, idx, tols = _batch_args(q_mz, q_int, ref_mz, ref_int, ref_offsets,
                                                 ref_idx, tol)
    return _batch_kernel(qm, qi, 0.0, rm, ri, ro, np.zeros(1), idx, tols, 2, 0.0, 0.5,
                         bool(weighted))


@njit(cache=True)
def _pair_scores_kernel(q_mz, q_int, q_off, ref_mz, ref_int, ref_off, pair_q, pair_r, tols,
                        int_power, weighted):
    """Cosine and entropy for explicit (query row, reference row) pairs."""
    n = pair_q.shape[0]
    cos = np.zeros(n, dtype=np.float64)
    ncos = np.zeros(n, dtype=np.int32)
    ent = np.zeros(n, dtype=np.float64)
    nent = np.zeros(n, dtype=np.int32)
    max_len = 1
    for t in range(n):
        s = pair_r[t]
        L = ref_off[s + 1] - ref_off[s]
        if L > max_len:
            max_len = L
        s = pair_q[t]
        L = q_off[s + 1] - q_off[s]
        if L > max_len:
            max_len = L
    wa = np.empty(max_len, dtype=np.float64)
    wb = np.empty(max_len, dtype=np.float64)
    for t in range(n):
        r = pair_r[t]
        q = pair_q[t]
        a0 = ref_off[r]
        a1 = ref_off[r + 1]
        b0 = q_off[q]
        b1 = q_off[q + 1]
        if a1 == a0 or b1 == b0:
            continue
        ma = ref_mz[a0:a1]
        mb = q_mz[b0:b1]
        ss_a = _weights(ma, ref_int[a0:a1], 0.0, int_power, wa)
        ss_b = _weights(mb, q_int[b0:b1], 0.0, int_power, wb)
        c, k = _cosine_core(ma, wa[: a1 - a0], ss_a, mb, wb[: b1 - b0], ss_b, tols[t], 0.0,
                            False)
        cos[t] = c
        ncos[t] = k
        if _entropy_prepare(ref_int[a0:a1], weighted, wa) and \
                _entropy_prepare(q_int[b0:b1], weighted, wb):
            e, k2 = _entropy_core(ma, wa[: a1 - a0], mb, wb[: b1 - b0], tols[t])
            ent[t] = e
            nent[t] = k2
    return cos, ncos, ent, nent


def pair_scores_batch(q_mz, q_int, q_offsets, ref_mz, ref_int, ref_offsets, pair_q, pair_r,
                      tol, intensity_power: float = 0.5, weighted: bool = True):
    """Cosine and entropy for explicit pairs between two flat stores.

    Returns ``(cos, n_matched_cos, entropy, n_matched_entropy)`` aligned with the pairs.
    """
    pq_ = np.ascontiguousarray(pair_q, dtype=np.int64)
    pr_ = np.ascontiguousarray(pair_r, dtype=np.int64)
    tols = np.ascontiguousarray(np.broadcast_to(np.asarray(tol, dtype=np.float64), pq_.shape))
    return _pair_scores_kernel(np.asarray(q_mz), np.asarray(q_int),
                               np.ascontiguousarray(q_offsets, dtype=np.int64),
                               np.asarray(ref_mz), np.asarray(ref_int),
                               np.ascontiguousarray(ref_offsets, dtype=np.int64), pq_, pr_,
                               tols, float(intensity_power), bool(weighted))

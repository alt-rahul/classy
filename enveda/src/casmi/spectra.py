"""Spectrum preprocessing, consensus spectra, sparse binning and a flat spectrum store.

PLAN.md section 4.1 (WP5). Everything that touches peaks goes through ``clean`` so that
library, queries and models see the same spectra. All kernels are numba-compiled and work
on flat arrays (no Python loop per spectrum on the build path).

Definitions (the plan leaves these details open; they are fixed here):

``clean(mz, inten, precursor_mz, cfg) -> (mz, inten, prec_intensity)``
  0. Peaks with a non-finite m/z or intensity, m/z <= 0 or intensity <= 0 are dropped;
     peaks are sorted by m/z (stable).
  1. Peaks with m/z > precursor_mz + ``clean_max_above_precursor_da`` are dropped.
  2. ``prec_intensity`` = highest intensity inside the precursor window
     [precursor_mz - ``precursor_window_low_da``, precursor_mz + ``precursor_window_high_da``]
     divided by the highest intensity of all peaks left after step 1 (0.0 if the window
     is empty, NaN if precursor_mz is NaN). Unless ``keep_precursor``, peaks inside the
     window are then removed.
  3. Relative floor: keep peaks with intensity >= ``intensity_floor`` x the highest
     remaining intensity (the base peak of what is left after step 2, so a dominant
     precursor does not wipe out the fragments).
  4. Keep the ``top_n_peaks`` most intense peaks (ties: lower m/z first), output sorted
     by m/z.
  5. Renormalise to base peak 1.0. Output arrays are float32 (PLAN.md section 7).
  ``clean`` is idempotent: cleaning a cleaned spectrum returns it unchanged.

``merge_consensus(spectra, cfg, precursor_mz=None)``
  Pools the peaks of the (cleaned, base-peak-1) member spectra, sorts them by m/z and
  merges greedily from the most intense peak down: each still-unmerged peak absorbs every
  other unmerged peak within ``consensus_merge_da`` of it (intensity-weighted m/z, summed
  intensity). The merged spectrum then goes through steps 1-5 of ``clean`` (with
  ``precursor_mz`` if given, else without the precursor steps), so a consensus spectrum
  also has at most ``top_n_peaks`` peaks and base peak 1.

``bin_sparse(mz, inten, offsets, precursor_mz, cfg) -> (frag, nl)``
  Two scipy CSR matrices with one row per spectrum: fragment channel (column
  floor(mz / ``analog_bin_da``)) and neutral-loss channel (loss = precursor_mz - mz,
  kept for ``precursor_window_low_da`` <= loss < ``analog_max_mz``), both over
  ceil(``analog_max_mz`` / ``analog_bin_da``) columns, value sqrt(intensity) summed per
  bin, each row L2-normalised per channel (all-zero rows stay zero). ``open_vectors``
  joins the two channels side by side with weight 1/sqrt(2) each, so that the dot product
  of two rows is the mean of the two channel cosines (PLAN.md 4.3, coarse stage).

``SpectrumStore``
  Flat arrays: ``mz`` float32, ``intensity`` float32, ``offsets`` int64 (n + 1, peaks of
  spectrum i are ``offsets[i]:offsets[i + 1]``). ``save`` writes three ``.npy`` files,
  ``load`` memory-maps them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

import numba
import numpy as np
from numba import njit

from .config import Config

__all__ = [
    "Spectrum",
    "SpectrumStore",
    "clean",
    "clean_flat",
    "merge_consensus",
    "merge_groups",
    "bin_sparse",
    "open_vectors",
    "open_query_vector",
    "open_n_bins",
    "list_arrays_to_flat",
]


class Spectrum(NamedTuple):
    """One cleaned spectrum (float32 arrays sorted by m/z) and its precursor m/z."""

    mz: np.ndarray
    intensity: np.ndarray
    precursor_mz: float = float("nan")


# --------------------------------------------------------------------------------------
# Cleaning kernels
# --------------------------------------------------------------------------------------


@njit(cache=True)
def _clean_one(mz, inten, prec, max_above, keep_prec, win_lo, win_hi, floor, top_n,
               out_mz, out_int):
    """Clean one spectrum into ``out_mz`` / ``out_int`` (float64 buffers of len >= n).

    Returns (number of output peaks, prec_intensity)."""
    n = mz.shape[0]
    # Step 0: sort by m/z if needed (stable).
    is_sorted = True
    for i in range(1, n):
        if mz[i] < mz[i - 1]:
            is_sorted = False
            break
    if is_sorted:
        order = np.arange(n)
    else:
        order = np.argsort(mz, kind="mergesort")
    has_prec = not math.isnan(prec)
    lo = prec - win_lo
    hi = prec + win_hi
    limit = prec + max_above
    keep = np.zeros(n, dtype=np.bool_)
    max_all = 0.0
    max_win = 0.0
    for k in range(n):
        i = order[k]
        m = mz[i]
        x = inten[i]
        if not (math.isfinite(m) and math.isfinite(x)) or m <= 0.0 or x <= 0.0:
            continue
        if has_prec and m > limit:          # step 1
            continue
        keep[k] = True
        if x > max_all:
            max_all = x
        if has_prec and m >= lo and m <= hi:
            if x > max_win:
                max_win = x
    if not has_prec:
        prec_int = np.nan
    elif max_all > 0.0:
        prec_int = max_win / max_all
    else:
        prec_int = 0.0
    # Step 2: precursor window.
    base = 0.0
    for k in range(n):
        if not keep[k]:
            continue
        m = mz[order[k]]
        if has_prec and (not keep_prec) and m >= lo and m <= hi:
            keep[k] = False
            continue
        x = inten[order[k]]
        if x > base:
            base = x
    if base <= 0.0:
        return 0, prec_int
    # Step 3: relative floor.
    thr = floor * base
    m_kept = 0
    for k in range(n):
        if keep[k]:
            if inten[order[k]] >= thr:
                m_kept += 1
            else:
                keep[k] = False
    idx = np.empty(m_kept, dtype=np.int64)
    j = 0
    for k in range(n):
        if keep[k]:
            idx[j] = order[k]
            j += 1
    # Step 4: top N by intensity, ties to lower m/z (stable sort on m/z-sorted input).
    if m_kept > top_n:
        neg = np.empty(m_kept, dtype=np.float64)
        for t in range(m_kept):
            neg[t] = -inten[idx[t]]
        sel = np.argsort(neg, kind="mergesort")[:top_n]
        sel = np.sort(sel)            # positions in m/z order
        idx2 = np.empty(top_n, dtype=np.int64)
        for t in range(top_n):
            idx2[t] = idx[sel[t]]
        idx = idx2
        m_kept = top_n
    # Step 5: renormalise to base peak 1.
    for t in range(m_kept):
        out_mz[t] = mz[idx[t]]
        out_int[t] = inten[idx[t]] / base
    return m_kept, prec_int


@njit(cache=True)
def _clean_flat_kernel(mz, inten, offsets, valid, prec, max_above, keep_prec, win_lo,
                       win_hi, floor, top_n):
    n = offsets.shape[0] - 1
    total = 0
    max_len = 0
    for i in range(n):
        L = offsets[i + 1] - offsets[i]
        total += min(L, top_n)
        if L > max_len:
            max_len = L
    out_mz = np.empty(total, dtype=np.float32)
    out_int = np.empty(total, dtype=np.float32)
    out_off = np.zeros(n + 1, dtype=np.int64)
    prec_int = np.full(n, np.nan, dtype=np.float64)
    n_raw = np.zeros(n, dtype=np.int32)
    buf_mz = np.empty(max(max_len, 1), dtype=np.float64)
    buf_int = np.empty(max(max_len, 1), dtype=np.float64)
    pos = 0
    for i in range(n):
        a = offsets[i]
        b = offsets[i + 1]
        if valid[i] and b > a:
            n_raw[i] = b - a
            k, p = _clean_one(mz[a:b], inten[a:b], prec[i], max_above, keep_prec, win_lo,
                              win_hi, floor, top_n, buf_mz, buf_int)
            prec_int[i] = p
            for t in range(k):
                out_mz[pos + t] = buf_mz[t]
                out_int[pos + t] = buf_int[t]
            pos += k
        out_off[i + 1] = pos
    return out_mz[:pos].copy(), out_int[:pos].copy(), out_off, prec_int, n_raw


def _clean_params(cfg: Config) -> tuple:
    return (float(cfg.clean_max_above_precursor_da), bool(cfg.keep_precursor),
            float(cfg.precursor_window_low_da), float(cfg.precursor_window_high_da),
            float(cfg.intensity_floor), int(cfg.top_n_peaks))


def clean(mz: Sequence[float] | np.ndarray, inten: Sequence[float] | np.ndarray,
          precursor_mz: float, cfg: Config | None = None
          ) -> tuple[np.ndarray, np.ndarray, float]:
    """Clean one spectrum (PLAN.md 4.1, see module docstring).

    Returns ``(mz float32, intensity float32 with base peak 1, prec_intensity)``.
    """
    cfg = cfg or Config()
    mz = np.ascontiguousarray(mz, dtype=np.float64)
    inten = np.ascontiguousarray(inten, dtype=np.float64)
    if mz.shape != inten.shape:
        raise ValueError(f"mz and intensity lengths differ: {mz.shape} vs {inten.shape}")
    prec = float("nan") if precursor_mz is None else float(precursor_mz)
    offsets = np.array([0, mz.shape[0]], dtype=np.int64)
    out = _clean_flat_kernel(mz, inten, offsets, np.ones(1, dtype=np.bool_),
                             np.array([prec]), *_clean_params(cfg))
    return out[0], out[1], float(out[3][0])


def clean_flat(mz: np.ndarray, inten: np.ndarray, offsets: np.ndarray,
               precursor_mz: np.ndarray, cfg: Config | None = None,
               valid: np.ndarray | None = None
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``clean`` for many spectra in flat layout.

    Returns ``(mz float32, intensity float32, offsets int64, prec_intensity float64,
    n_peaks_raw int32)``. Rows with ``valid`` False come out empty (n_peaks_raw 0).
    """
    cfg = cfg or Config()
    mz = np.ascontiguousarray(mz, dtype=np.float64)
    inten = np.ascontiguousarray(inten, dtype=np.float64)
    offsets = np.ascontiguousarray(offsets, dtype=np.int64)
    prec = np.ascontiguousarray(precursor_mz, dtype=np.float64)
    n = offsets.shape[0] - 1
    if prec.shape != (n,):
        raise ValueError("precursor_mz must have one value per spectrum")
    valid = (np.ones(n, dtype=np.bool_) if valid is None
             else np.ascontiguousarray(valid, dtype=np.bool_))
    return _clean_flat_kernel(mz, inten, offsets, valid, prec, *_clean_params(cfg))


def list_arrays_to_flat(mz_col, int_col) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """pyarrow list columns (ms2_mzs, ms2_normalized_intensities) -> flat float64 arrays.

    Returns ``(mz, intensity, offsets, valid)``; ``valid`` is False for null lists and for
    rows whose two lists differ in length (those rows get zero peaks).
    """
    import pyarrow as pa

    if isinstance(mz_col, pa.ChunkedArray):
        mz_col = mz_col.combine_chunks()
    if isinstance(int_col, pa.ChunkedArray):
        int_col = int_col.combine_chunks()
    n = len(mz_col)
    mo = np.asarray(mz_col.offsets, dtype=np.int64)
    io = np.asarray(int_col.offsets, dtype=np.int64)
    mv = np.asarray(mz_col.values.to_numpy(zero_copy_only=False), dtype=np.float64)
    iv = np.asarray(int_col.values.to_numpy(zero_copy_only=False), dtype=np.float64)
    valid = np.ones(n, dtype=np.bool_)
    if mz_col.null_count:
        valid &= ~np.asarray(mz_col.is_null().to_numpy(zero_copy_only=False))
    if int_col.null_count:
        valid &= ~np.asarray(int_col.is_null().to_numpy(zero_copy_only=False))
    ml = np.diff(mo)
    il = np.diff(io)
    valid &= ml == il
    lengths = np.where(valid, ml, 0)
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    mz = _gather_segments_f64(mv, mo[:-1], lengths, offsets)
    it = _gather_segments_f64(iv, io[:-1], lengths, offsets)
    return mz, it, offsets, valid


@njit(cache=True)
def _gather_segments_f64(src, starts, lengths, out_offsets):
    out = np.empty(out_offsets[-1], dtype=np.float64)
    for i in range(starts.shape[0]):
        a = starts[i]
        o = out_offsets[i]
        for t in range(lengths[i]):
            out[o + t] = src[a + t]
    return out


# --------------------------------------------------------------------------------------
# Consensus spectra
# --------------------------------------------------------------------------------------


@njit(cache=True)
def _merge_sorted(mz, inten, tol, out_mz, out_int):
    """Greedy merge of m/z-sorted peaks (most intense first). Returns n_out (sorted)."""
    n = mz.shape[0]
    if n == 0:
        return 0
    neg = np.empty(n, dtype=np.float64)
    for i in range(n):
        neg[i] = -inten[i]
    order = np.argsort(neg, kind="mergesort")
    used = np.zeros(n, dtype=np.bool_)
    tmp_mz = np.empty(n, dtype=np.float64)
    tmp_int = np.empty(n, dtype=np.float64)
    k = 0
    for t in range(n):
        i = order[t]
        if used[i]:
            continue
        c = mz[i]
        lo = i
        while lo - 1 >= 0 and c - mz[lo - 1] <= tol:
            lo -= 1
        hi = i
        while hi + 1 < n and mz[hi + 1] - c <= tol:
            hi += 1
        s = 0.0
        w = 0.0
        for j in range(lo, hi + 1):
            if not used[j]:
                s += inten[j]
                w += inten[j] * mz[j]
                used[j] = True
        tmp_mz[k] = w / s
        tmp_int[k] = s
        k += 1
    o2 = np.argsort(tmp_mz[:k], kind="mergesort")
    for t in range(k):
        out_mz[t] = tmp_mz[o2[t]]
        out_int[t] = tmp_int[o2[t]]
    return k


@njit(cache=True)
def _merge_groups_kernel(mz, inten, offsets, member_idx, group_off, group_prec, tol,
                         max_above, keep_prec, win_lo, win_hi, floor, top_n):
    """Consensus of each group g = members member_idx[group_off[g]:group_off[g+1]]."""
    G = group_off.shape[0] - 1
    out_off = np.zeros(G + 1, dtype=np.int64)
    out_mz = np.empty(G * top_n, dtype=np.float32)
    out_int = np.empty(G * top_n, dtype=np.float32)
    pos = 0
    for g in range(G):
        total = 0
        for t in range(group_off[g], group_off[g + 1]):
            s = member_idx[t]
            total += offsets[s + 1] - offsets[s]
        pm = np.empty(total, dtype=np.float64)
        pi = np.empty(total, dtype=np.float64)
        q = 0
        for t in range(group_off[g], group_off[g + 1]):
            s = member_idx[t]
            for p in range(offsets[s], offsets[s + 1]):
                pm[q] = mz[p]
                pi[q] = inten[p]
                q += 1
        o = np.argsort(pm, kind="mergesort")
        sm = pm[o]
        si = pi[o]
        mm = np.empty(max(total, 1), dtype=np.float64)
        mi = np.empty(max(total, 1), dtype=np.float64)
        k = _merge_sorted(sm, si, tol, mm, mi)
        cm = np.empty(max(k, 1), dtype=np.float64)
        ci = np.empty(max(k, 1), dtype=np.float64)
        k2, _ = _clean_one(mm[:k], mi[:k], group_prec[g], max_above, keep_prec, win_lo,
                           win_hi, floor, top_n, cm, ci)
        for t in range(k2):
            out_mz[pos + t] = cm[t]
            out_int[pos + t] = ci[t]
        pos += k2
        out_off[g + 1] = pos
    return out_mz[:pos].copy(), out_int[:pos].copy(), out_off


def merge_groups(store: "SpectrumStore", member_idx: np.ndarray, group_offsets: np.ndarray,
                 group_precursor_mz: np.ndarray, cfg: Config | None = None) -> "SpectrumStore":
    """Consensus spectrum per group of store rows (flat version of ``merge_consensus``).

    Group g has members ``member_idx[group_offsets[g]:group_offsets[g + 1]]`` and reference
    precursor ``group_precursor_mz[g]`` (NaN: no precursor steps).
    """
    cfg = cfg or Config()
    out = _merge_groups_kernel(
        np.asarray(store.mz), np.asarray(store.intensity), np.asarray(store.offsets),
        np.ascontiguousarray(member_idx, dtype=np.int64),
        np.ascontiguousarray(group_offsets, dtype=np.int64),
        np.ascontiguousarray(group_precursor_mz, dtype=np.float64),
        float(cfg.consensus_merge_da), *_clean_params(cfg))
    return SpectrumStore(out[0], out[1], out[2])


def merge_consensus(spectra: Iterable, cfg: Config | None = None,
                    precursor_mz: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Merged spectrum of cleaned member spectra (PLAN.md 4.1, see module docstring).

    ``spectra``: iterable of ``(mz, intensity)`` pairs or ``Spectrum``. Returns float32
    ``(mz, intensity)`` with base peak 1 (empty arrays if there are no peaks).
    """
    members = [(np.asarray(s[0], dtype=np.float32), np.asarray(s[1], dtype=np.float32))
               for s in spectra]
    store = SpectrumStore.from_spectra(members)
    prec = np.nan if precursor_mz is None else float(precursor_mz)
    out = merge_groups(store, np.arange(len(members), dtype=np.int64),
                       np.array([0, len(members)], dtype=np.int64), np.array([prec]), cfg)
    return out.mz, out.intensity


# --------------------------------------------------------------------------------------
# Sparse binning (open search, PLAN.md 4.3)
# --------------------------------------------------------------------------------------


def open_n_bins(cfg: Config) -> int:
    """Columns per channel of the binned vectors."""
    return int(math.ceil(cfg.analog_max_mz / cfg.analog_bin_da))


@njit(cache=True)
def _bin_kernel(mz, inten, offsets, prec, bin_da, n_bins, min_loss, neutral_loss):
    n = offsets.shape[0] - 1
    total = offsets[n] - offsets[0]
    indptr = np.zeros(n + 1, dtype=np.int64)
    indices = np.empty(total, dtype=np.int32)
    data = np.empty(total, dtype=np.float32)
    pos = 0
    for i in range(n):
        a = offsets[i]
        b = offsets[i + 1]
        L = b - a
        cols = np.empty(L, dtype=np.int64)
        vals = np.empty(L, dtype=np.float64)
        m = 0
        for p in range(a, b):
            if neutral_loss:
                x = prec[i] - mz[p]
                if not (x >= min_loss):     # also false for NaN precursor
                    continue
            else:
                x = mz[p]
            c = int(math.floor(x / bin_da))
            if c < 0 or c >= n_bins:
                continue
            cols[m] = c
            vals[m] = math.sqrt(max(inten[p], 0.0))
            m += 1
        if m > 0:
            o = np.argsort(cols[:m], kind="mergesort")
            start = pos
            last = -1
            for t in range(m):
                c = cols[o[t]]
                v = vals[o[t]]
                if c == last:
                    data[pos - 1] += v
                else:
                    indices[pos] = c
                    data[pos] = v
                    pos += 1
                    last = c
            ss = 0.0
            for t in range(start, pos):
                ss += float(data[t]) * float(data[t])
            if ss > 0.0:
                inv = 1.0 / math.sqrt(ss)
                for t in range(start, pos):
                    data[t] = data[t] * inv
        indptr[i + 1] = pos
    return indptr, indices[:pos].copy(), data[:pos].copy()


def bin_sparse(mz: np.ndarray, inten: np.ndarray, offsets: np.ndarray,
               precursor_mz: np.ndarray, cfg: Config | None = None):
    """Fragment and neutral-loss channels as two L2-normalised CSR matrices (float32).

    Shapes: (n_spectra, ``open_n_bins(cfg)``) each. See the module docstring.
    """
    import scipy.sparse as sp

    cfg = cfg or Config()
    mz = np.ascontiguousarray(mz, dtype=np.float32)
    inten = np.ascontiguousarray(inten, dtype=np.float32)
    offsets = np.ascontiguousarray(offsets, dtype=np.int64)
    prec = np.ascontiguousarray(precursor_mz, dtype=np.float64)
    n = offsets.shape[0] - 1
    nb = open_n_bins(cfg)
    out = []
    for nl in (False, True):
        indptr, indices, data = _bin_kernel(mz, inten, offsets, prec, float(cfg.analog_bin_da),
                                            nb, float(cfg.precursor_window_low_da), nl)
        out.append(sp.csr_matrix((data, indices, indptr), shape=(n, nb)))
    return out[0], out[1]


def open_query_vector(mz, inten, precursor_mz: float, cfg: Config | None = None) -> np.ndarray:
    """Dense float32 row of ``open_vectors`` for one spectrum (length 2 x open_n_bins)."""
    cfg = cfg or Config()
    nb = open_n_bins(cfg)
    mz = np.ascontiguousarray(mz, dtype=np.float32)
    inten = np.ascontiguousarray(inten, dtype=np.float32)
    off = np.array([0, mz.size], dtype=np.int64)
    prec = np.array([float(precursor_mz)], dtype=np.float64)
    out = np.zeros(2 * nb, dtype=np.float32)
    w = np.float32(1.0 / math.sqrt(2.0))
    for ch, nl in enumerate((False, True)):
        _, ind, dat = _bin_kernel(mz, inten, off, prec, float(cfg.analog_bin_da), nb,
                                  float(cfg.precursor_window_low_da), nl)
        out[ch * nb + ind] = dat * w
    return out


def open_vectors(frag, nl):
    """[frag | nl] / sqrt(2): rows of the open-search index (unit norm if both channels
    are non-empty). The dot product of two rows is the mean of the two channel cosines."""
    import scipy.sparse as sp

    w = np.float32(1.0 / math.sqrt(2.0))
    m = sp.hstack([frag * w, nl * w], format="csr", dtype=np.float32)
    m.sort_indices()
    return m


# --------------------------------------------------------------------------------------
# Flat spectrum store
# --------------------------------------------------------------------------------------

STORE_NAMES = ("peaks_mz.npy", "peaks_int.npy", "offsets.npy")


@njit(cache=True)
def _take_kernel(mz, inten, offsets, idx):
    n = idx.shape[0]
    out_off = np.zeros(n + 1, dtype=np.int64)
    for t in range(n):
        s = idx[t]
        out_off[t + 1] = out_off[t] + offsets[s + 1] - offsets[s]
    out_mz = np.empty(out_off[n], dtype=np.float32)
    out_int = np.empty(out_off[n], dtype=np.float32)
    for t in range(n):
        s = idx[t]
        a = offsets[s]
        o = out_off[t]
        for p in range(offsets[s + 1] - a):
            out_mz[o + p] = mz[a + p]
            out_int[o + p] = inten[a + p]
    return out_mz, out_int, out_off


@njit(cache=True)
def _take_into_kernel(mz, inten, offsets, idx, out_mz, out_int, out_off, start_row,
                      start_peak):
    """Copy rows ``idx`` into preallocated output arrays (used for memmap gathers)."""
    pos = start_peak
    for t in range(idx.shape[0]):
        s = idx[t]
        a = offsets[s]
        L = offsets[s + 1] - a
        for p in range(L):
            out_mz[pos + p] = mz[a + p]
            out_int[pos + p] = inten[a + p]
        pos += L
        out_off[start_row + t + 1] = pos
    return pos


@dataclass
class SpectrumStore:
    """Flat spectrum arrays: ``mz`` float32, ``intensity`` float32, ``offsets`` int64."""

    mz: np.ndarray
    intensity: np.ndarray
    offsets: np.ndarray

    def __post_init__(self) -> None:
        if self.mz.dtype != np.float32 or self.intensity.dtype != np.float32:
            raise TypeError("SpectrumStore peak arrays must be float32")
        if self.offsets.dtype != np.int64 or self.offsets.ndim != 1 or len(self.offsets) < 1:
            raise TypeError("SpectrumStore offsets must be a 1-d int64 array of length n + 1")
        if self.mz.shape != self.intensity.shape:
            raise ValueError("mz and intensity arrays differ in length")

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        return self.mz[a:b], self.intensity[a:b]

    @property
    def n_peaks(self) -> np.ndarray:
        return np.diff(self.offsets)

    @classmethod
    def empty(cls) -> "SpectrumStore":
        return cls(np.zeros(0, np.float32), np.zeros(0, np.float32), np.zeros(1, np.int64))

    @classmethod
    def from_spectra(cls, spectra: Iterable) -> "SpectrumStore":
        """From an iterable of ``(mz, intensity)`` pairs (or ``Spectrum``)."""
        mzs, ints, lens = [], [], [0]
        for s in spectra:
            m = np.asarray(s[0], dtype=np.float32).ravel()
            x = np.asarray(s[1], dtype=np.float32).ravel()
            if m.shape != x.shape:
                raise ValueError("mz and intensity lengths differ")
            mzs.append(m)
            ints.append(x)
            lens.append(len(m))
        if not mzs:
            return cls.empty()
        return cls(np.concatenate(mzs), np.concatenate(ints),
                   np.cumsum(np.asarray(lens, dtype=np.int64)))

    def take(self, idx: Sequence[int] | np.ndarray) -> "SpectrumStore":
        """New in-memory store with rows ``idx`` (in that order)."""
        idx = np.ascontiguousarray(idx, dtype=np.int64)
        m, x, o = _take_kernel(np.asarray(self.mz), np.asarray(self.intensity),
                               np.asarray(self.offsets), idx)
        return SpectrumStore(m, x, o)

    def save(self, directory: Path | str, names: tuple[str, str, str] = STORE_NAMES) -> list[Path]:
        """Write ``names`` = (mz, intensity, offsets) ``.npy`` files into ``directory``."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        out = []
        for arr, name in zip((self.mz, self.intensity, self.offsets), names):
            p = d / name
            np.save(p, np.asarray(arr))
            out.append(p)
        return out

    @classmethod
    def load(cls, directory: Path | str, names: tuple[str, str, str] = STORE_NAMES,
             mmap: bool = True) -> "SpectrumStore":
        """Read a saved store; ``mmap=True`` memory-maps the arrays read-only."""
        d = Path(directory)
        mode = "r" if mmap else None
        arrs = [np.load(d / n, mmap_mode=mode) for n in names]
        return cls(arrs[0], arrs[1], arrs[2])

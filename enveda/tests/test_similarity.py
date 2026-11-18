"""WP5: similarity functions (PLAN.md 4.2).

The cross-checks against matchms 0.34 (CosineGreedy, ModifiedCosineGreedy) and
ms_entropy 1.5.3 need those packages (dev environment only; skipped when absent).
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pyarrow.parquet as pq
import pytest

from casmi.config import Config
from casmi.similarity import (cosine_batch, cosine_greedy, entropy_batch, entropy_similarity,
                              modified_cosine, modified_cosine_batch, pair_scores_batch)
from casmi.spectra import SpectrumStore, clean_flat, list_arrays_to_flat


@pytest.fixture(scope="module")
def public_clean(public_test_parquet):
    t = pq.read_table(public_test_parquet)
    mz, it, off, valid = list_arrays_to_flat(t["ms2_mzs"], t["ms2_normalized_intensities"])
    prec = t["precursor_mz"].to_numpy()
    cm, ci, co, _, _ = clean_flat(mz, it, off, prec, Config(), valid)
    mol = np.asarray(t["molecule_id"].to_pylist(), dtype=object)
    return SpectrumStore(cm, ci, co), prec, mol


def _pairs(mol: np.ndarray, n_random: int = 1000, n_same: int = 500) -> np.ndarray:
    """1,000 uniformly random pairs plus one pair per molecule with >= 2 spectra (345)."""
    rng = np.random.default_rng(Config().seed)
    rand = rng.integers(0, len(mol), size=(n_random, 2))
    same = []
    for m in sorted(set(mol)):
        ix = np.flatnonzero(mol == m)
        if len(ix) > 1:
            same.append(rng.choice(ix, 2, replace=False))
    same = np.array(same[:n_same]).reshape(-1, 2)
    return np.vstack([rand, same])


def test_identical_spectra_score_one(public_clean):
    store, prec, _ = public_clean
    for i in range(0, len(store), 97):
        m, x = store[i]
        for tol in (0.01, 0.02):
            assert cosine_greedy(m, x, m, x, tol) == (pytest.approx(1.0), len(m))
            assert modified_cosine(m, x, prec[i], m, x, prec[i], tol) == (pytest.approx(1.0), len(m))
            for w in (True, False):
                s, n = entropy_similarity(m, x, m, x, tol, weighted=w)
                assert s == pytest.approx(1.0, abs=1e-12) and n == len(m)


def test_hand_computed_examples():
    a_mz, a_int = np.array([100.0, 200.0, 300.0]), np.array([1.0, 0.25, 0.04])
    b_mz, b_int = np.array([100.005, 200.5, 300.0]), np.array([0.25, 1.0, 1.0])
    s, n = cosine_greedy(a_mz, a_int, b_mz, b_int, 0.01)
    num = 1.0 * 0.5 + 0.2 * 1.0
    den = np.sqrt(1.0 + 0.25 + 0.04) * np.sqrt(0.25 + 1.0 + 1.0)
    assert n == 2 and s == pytest.approx(num / den)
    assert cosine_greedy(a_mz, a_int, b_mz + 50, b_int, 0.01) == (0.0, 0)
    # greedy: one query peak between two reference peaks takes the larger product
    s, n = cosine_greedy([100.0, 100.008], [0.04, 1.0], [100.004], [1.0], 0.01)
    assert n == 1 and s == pytest.approx(1.0 / np.sqrt(1.04))
    # modified cosine: b is a shifted by +14 (precursor 314 vs 300) except one peak
    b2_mz = np.array([114.0, 214.0, 300.0])
    s0, n0 = cosine_greedy(a_mz, a_int, b2_mz, a_int, 0.01)
    s1, n1 = modified_cosine(a_mz, a_int, 300.0, b2_mz, a_int, 314.0, 0.01)
    assert n0 == 1 and n1 == 3 and s1 == pytest.approx(1.0)
    # |shift| <= tol: plain cosine
    assert modified_cosine(a_mz, a_int, 300.0, b_mz, b_int, 300.005, 0.01) == \
        cosine_greedy(a_mz, a_int, b_mz, b_int, 0.01)
    # entropy: two identical one-peak spectra -> 1; half overlap
    assert entropy_similarity([100.0], [1.0], [100.0], [5.0], 0.01, weighted=False) == (1.0, 1)
    s, n = entropy_similarity([100.0, 200.0], [1.0, 1.0], [100.0, 300.0], [1.0, 1.0], 0.01,
                              weighted=False)
    assert n == 1 and s == pytest.approx(0.5)
    assert entropy_similarity([], [], [1.0], [1.0], 0.01) == (0.0, 0)


def test_batch_equals_pairwise(public_clean):
    store, prec, _ = public_clean
    q = 3
    qm, qi = store[q]
    idx = np.arange(0, len(store), 3)
    tols = np.where(idx % 2 == 0, 0.01, 0.02)
    cs, cn = cosine_batch(qm, qi, store.mz, store.intensity, store.offsets, idx, tols)
    ms, mn = modified_cosine_batch(qm, qi, prec[q], store.mz, store.intensity, store.offsets,
                                   prec, idx, tols)
    es, en = entropy_batch(qm, qi, store.mz, store.intensity, store.offsets, idx, tols)
    us, un = entropy_batch(qm, qi, store.mz, store.intensity, store.offsets, idx, tols,
                           weighted=False)
    pc, pcn, pe, pen = pair_scores_batch(store.mz, store.intensity, store.offsets, store.mz,
                                         store.intensity, store.offsets,
                                         np.full(idx.size, q), idx, tols)
    for t, r in enumerate(idx):
        rm, ri = store[r]
        # batch puts the reference first (matchms reference / query order)
        assert (cs[t], cn[t]) == cosine_greedy(rm, ri, qm, qi, tols[t])
        assert (pc[t], pcn[t]) == (cs[t], cn[t])
        assert (ms[t], mn[t]) == modified_cosine(rm, ri, prec[r], qm, qi, prec[q], tols[t])
        e = entropy_similarity(rm, ri, qm, qi, tols[t])
        assert (es[t], en[t]) == e and (pe[t], pen[t]) == e
        assert (us[t], un[t]) == entropy_similarity(rm, ri, qm, qi, tols[t], weighted=False)
    all_s, _ = cosine_batch(qm, qi, store.mz, store.intensity, store.offsets, None, 0.01)
    assert all_s.shape == (len(store),) and all_s[q] == pytest.approx(1.0)


def test_cross_check_matchms_ms_entropy(public_clean):
    """Agreement to 1e-6 on 1,000 random + 345 same-molecule pairs, tol 0.01 and 0.02."""
    matchms = pytest.importorskip("matchms")
    me = pytest.importorskip("ms_entropy")
    from matchms.similarity import CosineGreedy, ModifiedCosineGreedy

    logging.getLogger("matchms").setLevel(logging.ERROR)
    store, prec, mol = public_clean
    pairs = _pairs(mol)
    assert len(pairs) >= 1300        # 1,000 random + one pair per multi-spectrum molecule

    def mspec(i):
        m, x = store[i]
        return matchms.Spectrum(mz=m.astype(np.float64), intensities=x.astype(np.float64),
                                metadata={"precursor_mz": float(prec[i])},
                                metadata_harmonization=False)

    def epeaks(i):
        m, x = store[i]
        return np.column_stack([m, x / x.astype(np.float64).sum()]).astype(np.float32)

    specs = {i: mspec(i) for i in np.unique(pairs)}
    peaks = {i: epeaks(i) for i in np.unique(pairs)}
    worst = {}
    for tol in (0.01, 0.02):
        cg = CosineGreedy(tolerance=tol, intensity_power=0.5)
        mc = ModifiedCosineGreedy(tolerance=tol, intensity_power=0.5)
        err = {"cos": 0.0, "mcos": 0.0, "ent": 0.0, "uent": 0.0}
        n_high = 0
        for a, b in pairs:
            ma, xa = store[a]
            mb, xb = store[b]
            r = cg.pair(specs[a], specs[b])
            s, n = cosine_greedy(ma, xa, mb, xb, tol)
            assert n == int(r["matches"])
            err["cos"] = max(err["cos"], abs(s - float(r["score"])))
            n_high += s > 0.5
            r = mc.pair(specs[a], specs[b])
            s, n = modified_cosine(ma, xa, prec[a], mb, xb, prec[b], tol)
            assert n == int(r["matches"])
            err["mcos"] = max(err["mcos"], abs(s - float(r["score"])))
            e = me.calculate_entropy_similarity(peaks[a], peaks[b], ms2_tolerance_in_da=tol,
                                                clean_spectra=False)
            err["ent"] = max(err["ent"], abs(entropy_similarity(ma, xa, mb, xb, tol)[0] - e))
            e = me.calculate_unweighted_entropy_similarity(
                peaks[a], peaks[b], ms2_tolerance_in_da=tol, clean_spectra=False)
            err["uent"] = max(err["uent"],
                              abs(entropy_similarity(ma, xa, mb, xb, tol, weighted=False)[0] - e))
        assert n_high > 200            # the same-molecule pairs give high scores
        for k, v in err.items():
            assert v < 1e-6, (tol, k, v)
        worst[tol] = err
    print(f"\nmax abs differences vs matchms / ms_entropy: {worst}")


def test_timing_one_query_vs_10k(public_clean):
    """One query against 10,000 references of a flat store (reported, loose bound)."""
    store, prec, _ = public_clean
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(store), 10_000)
    refs = store.take(idx)
    rprec = prec[idx]
    qm, qi = store[0]
    funcs = {
        "cosine": lambda: cosine_batch(qm, qi, refs.mz, refs.intensity, refs.offsets, None, 0.01),
        "modified_cosine": lambda: modified_cosine_batch(qm, qi, prec[0], refs.mz,
                                                         refs.intensity, refs.offsets, rprec,
                                                         None, 0.01),
        "entropy": lambda: entropy_batch(qm, qi, refs.mz, refs.intensity, refs.offsets, None,
                                         0.01),
    }
    out = {}
    for name, f in funcs.items():
        f()                                    # compile / warm up
        t0 = time.perf_counter()
        for _ in range(5):
            f()
        out[name] = (time.perf_counter() - t0) / 5 * 1e3
    print(f"\none query vs 10k refs (ms, mean peaks/ref {refs.n_peaks.mean():.1f}): "
          + ", ".join(f"{k} {v:.1f}" for k, v in out.items()))
    assert max(out.values()) < 2000

"""WP5: spectrum cleaning, consensus spectra, sparse binning, flat store (PLAN.md 4.1)."""

from __future__ import annotations

import math

import numpy as np
import pyarrow.parquet as pq
import pytest

from casmi.config import Config
from casmi.spectra import (SpectrumStore, bin_sparse, clean, clean_flat, list_arrays_to_flat,
                           merge_consensus, merge_groups, open_matrix, open_n_bins,
                           open_query_vector, open_vectors)


@pytest.fixture(scope="module")
def public_flat(public_test_parquet):
    t = pq.read_table(public_test_parquet)
    mz, it, off, valid = list_arrays_to_flat(t["ms2_mzs"], t["ms2_normalized_intensities"])
    return mz, it, off, valid, t["precursor_mz"].to_numpy()


def test_clean_public_test_median_and_base_peak(public_flat):
    mz, it, off, valid, prec = public_flat
    cfg = Config()
    raw_n = np.diff(off)
    assert np.median(raw_n) == 230
    cm, ci, co, pi, nraw = clean_flat(mz, it, off, prec, cfg, valid)
    n = np.diff(co)
    assert np.median(n) <= 100 and n.max() <= cfg.top_n_peaks
    assert (nraw == raw_n).all() and (n > 0).all()
    assert cm.dtype == np.float32 and ci.dtype == np.float32
    for i in range(len(prec)):
        m, x = mz[off[i]:off[i + 1]], it[off[i]:off[i + 1]]
        p = prec[i]
        ok = m <= p + cfg.clean_max_above_precursor_da
        win = (m >= p - cfg.precursor_window_low_da) & (m <= p + cfg.precursor_window_high_da)
        frag = ok & ~win
        base_mz = m[frag][np.argmax(x[frag])]
        om, oi = cm[co[i]:co[i + 1]], ci[co[i]:co[i + 1]]
        j = np.flatnonzero(om == np.float32(base_mz))
        assert j.size == 1 and oi[j[0]] == 1.0, i
        assert np.all(np.diff(om) >= 0)
        # prec_intensity: window max over overall max (peaks <= prec + 2)
        expect = x[ok & win].max() / x[ok].max() if (ok & win).any() else 0.0
        assert pi[i] == pytest.approx(expect)


def test_clean_keep_precursor_keeps_absolute_base_peak(public_flat):
    mz, it, off, valid, prec = public_flat
    cfg = Config().replace(keep_precursor=True)
    cm, ci, co, _, _ = clean_flat(mz, it, off, prec, cfg, valid)
    for i in range(0, len(prec), 7):
        m, x = mz[off[i]:off[i + 1]], it[off[i]:off[i + 1]]
        ok = m <= prec[i] + 2.0
        base_mz = np.float32(m[ok][np.argmax(x[ok])])
        om, oi = cm[co[i]:co[i + 1]], ci[co[i]:co[i + 1]]
        assert oi[np.flatnonzero(om == base_mz)][0] == 1.0


def test_clean_steps_synthetic():
    cfg = Config().replace(top_n_peaks=4, intensity_floor=0.05)
    prec = 300.0
    mz = np.array([303.0, 50.0, 100.0, 120.0, 150.0, 299.8, 300.0, 301.0, 200.0, 250.0,
                   260.0, 270.0, np.nan, 280.0])
    it = np.array([9.0, 4.0, 2.0, 0.1, 2.0, 1.0, 10.0, 0.5, 4.0, 1.0, 0.0, 0.2, 5.0, -1.0])
    m, x, p = clean(mz, it, prec, cfg)
    # 303 > prec + 2 and NaN / non-positive peaks dropped; the precursor window
    # [299.5, 302] is removed (its max 10.0 / overall max 10.0 -> prec_intensity 1.0);
    # floor 0.05 x base 4.0 = 0.2 drops 120 (0.1) and keeps 270 (0.2); the top 4 of
    # {50:4, 100:2, 150:2, 200:4, 250:1, 270:0.2} are 50, 200, 100, 150; base -> 1.
    assert m.dtype == np.float32 and x.dtype == np.float32
    assert m.tolist() == [50.0, 100.0, 150.0, 200.0]
    assert x.tolist() == [1.0, 0.5, 0.5, 1.0]
    assert p == 1.0
    # tie-break: with top 3, 100 (lower m/z) wins over 150
    m3, x3, _ = clean(mz, it, prec, cfg.replace(top_n_peaks=3))
    assert m3.tolist() == [50.0, 100.0, 200.0]
    # unsorted input gives the same result, cleaning is idempotent
    perm = np.random.default_rng(0).permutation(len(mz))
    m2, x2, p2 = clean(mz[perm], it[perm], prec, cfg)
    assert np.array_equal(m, m2) and np.array_equal(x, x2) and p2 == p
    m4, x4, _ = clean(m, x, prec, cfg)
    assert np.array_equal(m, m4) and np.array_equal(x, x4)
    # no precursor: precursor steps skipped, prec_intensity NaN
    m5, _, p5 = clean(mz, it, float("nan"), cfg)
    assert math.isnan(p5) and 303.0 in m5.tolist() and 300.0 in m5.tolist()
    # empty input
    m6, x6, p6 = clean([], [], 100.0, cfg)
    assert m6.size == 0 and x6.size == 0 and p6 == 0.0
    with pytest.raises(ValueError):
        clean([1.0, 2.0], [1.0], 100.0, cfg)


def test_clean_flat_equals_clean(public_flat):
    mz, it, off, valid, prec = public_flat
    cfg = Config()
    cm, ci, co, pi, _ = clean_flat(mz, it, off, prec, cfg, valid)
    for i in range(0, len(prec), 50):
        m, x, p = clean(mz[off[i]:off[i + 1]], it[off[i]:off[i + 1]], prec[i], cfg)
        assert np.array_equal(m, cm[co[i]:co[i + 1]]) and np.array_equal(x, ci[co[i]:co[i + 1]])
        assert p == pi[i]


def test_list_arrays_to_flat_handles_nulls_and_slices():
    import pyarrow as pa

    mzs = pa.array([[1.0, 2.0], None, [3.0], [4.0, 5.0, 6.0]], type=pa.list_(pa.float64()))
    ints = pa.array([[1.0, 1.0], [1.0], [1.0, 2.0], [1.0, 1.0, 1.0]], type=pa.list_(pa.float64()))
    mz, it, off, valid = list_arrays_to_flat(mzs.slice(0), ints)
    assert valid.tolist() == [True, False, False, True]
    assert off.tolist() == [0, 2, 2, 2, 5] and mz.tolist() == [1.0, 2.0, 4.0, 5.0, 6.0]
    mz, it, off, valid = list_arrays_to_flat(mzs.slice(2), ints.slice(2))
    assert valid.tolist() == [False, True] and mz.tolist() == [4.0, 5.0, 6.0]


def test_merge_consensus():
    cfg = Config()
    a = (np.array([100.000, 150.0, 200.0], np.float32), np.array([1.0, 0.5, 0.2], np.float32))
    b = (np.array([100.006, 150.02, 250.0], np.float32), np.array([1.0, 1.0, 0.004], np.float32))
    m, x = merge_consensus([a, b], cfg)
    assert m.dtype == np.float32 and x.max() == 1.0
    # 100.000 and 100.006 merge (intensity-weighted m/z, summed intensity 2.0 -> 1.0);
    # 150.0 and 150.02 are 0.02 apart and stay separate; 250 (0.004 / 2 < 0.5 %) is floored.
    assert len(m) == 5 - 1
    assert m[0] == pytest.approx(100.003, abs=1e-4) and x[0] == 1.0
    assert np.allclose(m[1:], [150.0, 150.02, 200.0], atol=1e-4)
    assert np.allclose(x[1:], [0.25, 0.5, 0.1])
    # greedy from the most intense peak: 100.012 joins 100.006 (1.0), not 100.0 (0.1)
    c = (np.array([100.0, 100.006, 100.0125], np.float32), np.array([0.1, 1.0, 0.5], np.float32))
    m, x = merge_consensus([c], cfg)
    assert len(m) == 1 and m[0] == pytest.approx((0.1 * 100 + 100.006 + 0.5 * 100.0125) / 1.6,
                                                 abs=1e-4)
    # precursor window removed when a precursor is given
    d = (np.array([50.0, 299.9, 300.0], np.float32), np.array([0.5, 0.2, 1.0], np.float32))
    m, x = merge_consensus([d], cfg, precursor_mz=300.0)
    assert m.tolist() == [50.0] and x.tolist() == [1.0]
    # merging one cleaned spectrum alone (peaks > tol apart) returns it unchanged
    s = (np.array([60.0, 80.0, 90.0], np.float32), np.array([0.3, 1.0, 0.6], np.float32))
    m, x = merge_consensus([s], cfg)
    assert np.array_equal(m, s[0]) and np.array_equal(x, s[1])
    m, x = merge_consensus([], cfg)
    assert m.size == 0


def test_merge_groups_matches_merge_consensus(public_flat):
    mz, it, off, valid, prec = public_flat
    cfg = Config()
    cm, ci, co, _, _ = clean_flat(mz, it, off, prec, cfg, valid)
    store = SpectrumStore(cm, ci, co)
    groups = [[0, 1, 2], [5], [10, 11]]
    out = merge_groups(store, np.concatenate(groups), np.array([0, 3, 4, 6]),
                       np.array([prec[0], np.nan, prec[10]]), cfg)
    for g, members in enumerate(groups):
        p = None if g == 1 else prec[members[0]]
        m, x = merge_consensus([store[i] for i in members], cfg, precursor_mz=p)
        assert np.array_equal(out[g][0], m) and np.array_equal(out[g][1], x)


def test_bin_sparse_and_open_vectors():
    cfg = Config()
    nb = open_n_bins(cfg)
    assert nb == 40000
    s1 = (np.array([50.01, 50.04, 100.0], np.float32), np.array([0.25, 0.25, 1.0], np.float32))
    s2 = (np.array([100.01, 150.0], np.float32), np.array([1.0, 0.36], np.float32))
    store = SpectrumStore.from_spectra([s1, s2])
    prec = np.array([200.0, 200.0])
    frag, nl = bin_sparse(store.mz, store.intensity, store.offsets, prec, cfg)
    assert frag.shape == (2, nb) and nl.shape == (2, nb)
    # s1: 50.01 and 50.04 share bin 1000 -> sqrt(.25) + sqrt(.25) = 1; bin 2000 -> 1.
    r = frag.getrow(0).toarray().ravel()
    assert r[1000] == pytest.approx(1 / math.sqrt(2)) and r[2000] == pytest.approx(1 / math.sqrt(2))
    # neutral losses: 200 - 100 = 100 -> bin 2000; 149.99 and 149.96 -> bin 2999
    r = nl.getrow(0).toarray().ravel()
    assert set(np.flatnonzero(r)) == {2000, 2999}
    norms = np.sqrt(np.asarray(frag.multiply(frag).sum(axis=1))).ravel()
    assert np.allclose(norms, 1.0)
    ov = open_vectors(frag, nl)
    assert ov.shape == (2, 2 * nb)
    dot = (ov[0] @ ov[1].T).toarray()[0, 0]
    cf = (frag[0] @ frag[1].T).toarray()[0, 0]
    cn = (nl[0] @ nl[1].T).toarray()[0, 0]
    assert dot == pytest.approx((cf + cn) / 2)
    qv = open_query_vector(*s1, 200.0, cfg)
    assert np.allclose(qv, ov[0].toarray().ravel())
    # losses below precursor_window_low_da and NaN precursors give no loss entries
    f2, nl2 = bin_sparse(np.array([199.8], np.float32), np.array([1.0], np.float32),
                         np.array([0, 1]), np.array([200.0]), cfg)
    assert nl2.nnz == 0 and f2.nnz == 1
    _, nl3 = bin_sparse(store.mz, store.intensity, store.offsets, np.array([np.nan, np.nan]), cfg)
    assert nl3.nnz == 0


def test_open_matrix_equals_open_vectors(public_flat):
    mz, it, off, valid, prec = public_flat
    cm, ci, co, _, _ = clean_flat(mz, it, off, prec, Config(), valid)
    a = open_matrix(cm, ci, co, prec)
    b = open_vectors(*bin_sparse(cm, ci, co, prec))
    assert a.shape == b.shape and np.array_equal(a.indptr, b.indptr)
    assert np.array_equal(a.indices, b.indices) and np.allclose(a.data, b.data, atol=1e-6)
    norms = np.sqrt(np.asarray(a.multiply(a).sum(axis=1))).ravel()
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_store_roundtrip(tmp_path):
    spectra = [(np.array([1.0, 2.0]), np.array([0.5, 1.0])), (np.zeros(0), np.zeros(0)),
               (np.array([3.0]), np.array([1.0]))]
    st = SpectrumStore.from_spectra(spectra)
    assert len(st) == 3 and st.n_peaks.tolist() == [2, 0, 1]
    assert st.mz.dtype == np.float32 and st.offsets.dtype == np.int64
    files = st.save(tmp_path)
    assert [f.name for f in files] == ["peaks_mz.npy", "peaks_int.npy", "offsets.npy"]
    ld = SpectrumStore.load(tmp_path)
    assert isinstance(ld.mz, np.memmap)
    for i in range(3):
        assert np.array_equal(ld[i][0], st[i][0]) and np.array_equal(ld[i][1], st[i][1])
    tk = ld.take([2, 0])
    assert tk[0][0].tolist() == [3.0] and tk[1][1].tolist() == [0.5, 1.0]
    st.save(tmp_path, names=("a.npy", "b.npy", "c.npy"))
    assert len(SpectrumStore.load(tmp_path, names=("a.npy", "b.npy", "c.npy"), mmap=False)) == 3
    with pytest.raises(TypeError):
        SpectrumStore(np.zeros(2), np.zeros(2, np.float32), np.zeros(2, np.int64))

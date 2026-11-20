"""Tests for casmi.fragments (ROADMAP R7): enumeration, matching, features, cache, stage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from casmi import fragments as F
from casmi import pipeline
from casmi.adducts import element_mass
from casmi.config import Config

H = element_mass("H")
C = element_mass("C")
O = element_mass("O")
PROTON = F.PROTON


def _has(fl, mass, bit=None, tol=1e-6):
    i = np.flatnonzero(np.abs(fl.masses - mass) < tol)
    if not len(i):
        return False
    return True if bit is None else bool(fl.masks[i[0]] >> bit & 1)


# --- enumeration ----------------------------------------------------------------------

def test_ethanol_fragments_and_levels():
    fl = F.fragment_list("CCO")
    ch3, ch2, oh = C + 3 * H, C + 2 * H, O + H
    exp = {C * 2 + O + 6 * H: 0, ch3: 1, ch2 + oh: 1, ch3 + ch2: 1, oh: 1, ch2: 2}
    assert len(fl) == len(exp)
    for m, bit in exp.items():
        assert _has(fl, m, bit), m
    assert not _has(fl, ch2, 1)          # CH2 needs both bonds cut
    assert np.all(np.diff(fl.masses) > 0)


def test_ring_bonds_need_two_cuts():
    fl = F.fragment_list("C1CCCCC1")
    # intact + CnH2n pieces from ring cleavage only; no single-cut fragments
    assert len(fl) == 6
    for n in range(1, 6):
        assert _has(fl, n * (C + 2 * H), 3)
        assert not _has(fl, n * (C + 2 * H), 1)
    fl2 = F.fragment_list("CCC1CCCCC1")
    ring = 6 * C + 11 * H
    assert _has(fl2, ring, 1)            # side chain off: one acyclic cut
    assert _has(fl2, C + 3 * H, 1)


def test_one_cut_only_and_pair_cap():
    one = F.fragment_list("CCO", Config(frag_max_cuts=1))
    assert not _has(one, C + 2 * H)
    assert not one.two_cuts
    capped = F.fragment_list("CCCCCCCCCCO", Config(frag_max_pairs=5))
    assert not capped.two_cuts


def test_salt_and_bad_smiles():
    fl = F.fragment_list("CC.O")
    assert _has(fl, 2 * C + 6 * H, 0) and _has(fl, O + 2 * H, 0)
    assert _has(fl, 2 * C + 8 * H + O, 0)
    assert F.fragment_list("not a smiles") is None
    assert F.fragment_list("") is None


def test_carriers():
    assert F.carriers("[M+H]+") == (PROTON,)
    na = F.carriers("[M+Na]+")
    assert len(na) == 2 and abs(na[1] - (element_mass("Na") - F.ELECTRON_MASS)) < 1e-9
    assert F.carriers("[M-H]-") == (-PROTON,)
    assert len(F.carriers("[M+CH2O2-H]-")) == 2
    assert F.carriers("[M+FOO]+", "positive") == (PROTON,)
    assert F.carriers("weird", "negative") == (-PROTON,)


# --- matching -------------------------------------------------------------------------

def _query(mz, it, adduct="[M+H]+", cfg=None, prec=500.0):
    return F.prepare_query([(np.asarray(mz, float), np.asarray(it, float), prec, adduct,
                             "positive" if adduct.endswith("+") else "negative")],
                           cfg or Config(), cleaned=True)


def _feat(qs, smiles, cfg=None):
    cfg = cfg or Config()
    m = F.score_candidates(qs, F.fragment_lists(smiles, cfg), cfg)
    return pd.DataFrame(m, columns=list(F.RAW_FEATURES) + ["frag_log_n"])


def test_credit_one_and_two_cuts():
    ch2, oh = C + 2 * H, O + H
    # peaks: CH2OH+H (one cut, credit 1) and CH2 + H (two cuts only, credit 0.6); without
    # H shifts, so CH2+ cannot be read as CH3 (one cut) minus a hydrogen
    cfg = Config(frag_h_shift_min=0, frag_h_shift_max=0)
    qs = _query([ch2 + oh + PROTON, ch2 + PROTON], [1.0, 1.0], cfg=cfg)
    f = _feat(qs, ["CCO"], cfg).iloc[0]
    assert f["frag_expl_lin"] == pytest.approx(1.0)
    assert f["frag_score"] == pytest.approx((1.0 + 0.6) / 2)
    assert f["frag_n_expl"] == 2 and f["frag_frac_peaks"] == pytest.approx(1.0)
    assert f["frag_mean_cuts"] == pytest.approx(1.5)
    assert f["frag_base_expl"] == pytest.approx(1.0)
    f2 = _feat(qs, ["CCO"], cfg.replace(frag_w_two=1.0)).iloc[0]
    assert f2["frag_score"] == pytest.approx(1.0)
    # with the default H shifts CH2+ is CH3 (one cut) - H: full credit
    f3 = _feat(_query([ch2 + PROTON], [1.0]), ["CCO"]).iloc[0]
    assert f3["frag_score"] == pytest.approx(1.0)


def test_tolerance_offset_and_h_shift():
    base = C + 3 * H + PROTON                    # CH3+ from ethanol
    for d, ok in ((0.004, True), (0.006, False)):
        f = _feat(_query([base + d], [1.0]), ["CCO"]).iloc[0]
        assert (f["frag_expl_lin"] > 0) == ok
    cfg = Config(frag_mz_offset_pos_da=0.002)
    f = _feat(_query([base + 0.006], [1.0], cfg=cfg), ["CCO"], cfg).iloc[0]
    assert f["frag_expl_lin"] == pytest.approx(1.0)
    f = _feat(_query([base + 3 * H], [1.0]), ["CCO"]).iloc[0]       # dh = +3 allowed
    assert f["frag_expl_lin"] == pytest.approx(1.0)
    narrow = Config(frag_h_shift_min=-1, frag_h_shift_max=1)
    f = _feat(_query([base + 3 * H], [1.0], cfg=narrow), ["CCO"], narrow).iloc[0]
    assert f["frag_expl_lin"] == 0.0


def test_negative_mode_and_sodium_carrier():
    ch3o = C + 3 * H + O
    f = _feat(_query([ch3o - PROTON], [1.0], adduct="[M-H]-"), ["COC"]).iloc[0]
    assert f["frag_expl_lin"] == pytest.approx(1.0)
    na = element_mass("Na") - F.ELECTRON_MASS
    f = _feat(_query([ch3o + na], [1.0], adduct="[M+Na]+"), ["COC"]).iloc[0]
    assert f["frag_expl_lin"] == pytest.approx(1.0)
    f = _feat(_query([ch3o + na], [1.0], adduct="[M+H]+"), ["COC"]).iloc[0]
    assert f["frag_expl_lin"] == 0.0


def test_isomers_are_separated():
    # CH2OH (CH3O) is one cut from 1-propanol but two cuts (CHOH + H) from 2-propanol
    ch3o = C + 3 * H + O
    qs = _query([ch3o + PROTON], [1.0])
    f = _feat(qs, ["CCCO", "CC(C)O", "xx"])
    assert f["frag_score"].iloc[0] == pytest.approx(1.0)
    assert f["frag_score"].iloc[1] == pytest.approx(0.6)
    assert f["frag_expl_lin"].iloc[1] == pytest.approx(1.0)
    assert np.isnan(f["frag_score"].iloc[2])


def test_empty_query_gives_nan():
    qs = F.prepare_query([], Config())
    assert qs.n_peaks == 0
    f = _feat(qs, ["CCO"])
    assert f.isna().all(axis=None)


def test_spectra_pooled_equally():
    ch3, oh = C + 3 * H, O + H
    qs = F.prepare_query([
        (np.array([ch3 + PROTON]), np.array([1.0]), 100.0, "[M+H]+"),
        (np.array([oh + PROTON, 77.0]), np.array([1.0, 1.0]), 100.0, "[M+H]+"),
    ], Config(), cleaned=True)
    f = _feat(qs, ["CCO"]).iloc[0]
    assert f["frag_expl_lin"] == pytest.approx(0.5 * 1.0 + 0.5 * 0.5)
    assert f["frag_base_expl"] == pytest.approx(1.0)  # both base peaks (first max) explained


# --- normalisation and selection --------------------------------------------------------

def test_normalise_z_rank_ties_and_formula_groups():
    raw = pd.DataFrame({c: [np.nan] * 5 for c in list(F.RAW_FEATURES) + ["frag_log_n"]})
    raw["frag_score"] = [0.5, 0.5, 0.1, np.nan, 0.9]
    raw["frag_expl_lin"] = [0.2, 0.4, 0.6, np.nan, 0.8]
    raw["frag_base_w"] = [1.0, 0.6, 0.0, np.nan, 1.0]
    out = F.normalise(raw, ["a", "a", "a", "a", "b"], ["X", "X", "Y", "X", "X"])
    assert list(out["frag_score_rank"].iloc[:3]) == [1, 1, 3]
    assert np.isnan(out["frag_score_rank"].iloc[3])
    assert out["frag_score_rank"].iloc[4] == 1 and out["frag_score_z"].iloc[4] == 0.0
    z = out["frag_score_z"].iloc[:3].to_numpy()
    assert z.mean() == pytest.approx(0.0, abs=1e-12) and z[2] < 0
    assert out["frag_score_fz"].iloc[2] == 0.0                     # alone in formula Y
    assert out["frag_score_gap"].iloc[2] == pytest.approx(-0.4)
    assert list(out["frag_base_best"].iloc[:3]) == [1.0, 0.0, 0.0]
    assert list(out.columns) == list(F.FEATURES)


def _pool():
    return pd.DataFrame({
        "molecule_id": ["m1"] * 4 + ["m2"] * 2,
        "ik14": ["A", "B", "C", "D", "E", "F"],
        "smiles": ["CCCO", "CC(C)O", "COCC", "CCCCO", "CCO", "COC"],
        "formula": ["C3H8O", "C3H8O", "C3H8O", "C4H10O", "C2H6O", "C2H6O"],
        "tier": ["np"] * 6, "in_np": [True] * 6, "ppm_err": [0.0] * 6,
        "score": [1.2, 1.9, 1.5, 1.1, 1.0, 1.4],
    })


def _test_df():
    ch3o = C + 3 * H + O
    return pd.DataFrame({
        "molecule_id": ["m1", "m2"],
        "ms2_mzs": [np.array([ch3o + PROTON, C + 3 * H + PROTON]), np.array([31.0178, 15.0230])],
        "ms2_normalized_intensities": [np.array([1.0, 0.5]), np.array([1.0, 0.3])],
        "precursor_mz": [61.0648, 47.0491], "adduct": ["[M+H]+", "[M+H]+"],
        "ionization_mode": ["positive", "positive"],
    })


def test_select_top_k_by_score():
    keep = F.select_top_k(_pool(), Config(frag_top_k=2))
    assert keep.tolist() == [False, True, True, False, True, True]
    pool = _pool().drop(columns="score")
    keep = F.select_top_k(pool, Config(frag_top_k=1))
    assert keep.sum() == 2


def test_score_pool_and_stage(tmp_path):
    cfg = Config(frag_top_k=3, frag_cache_dir=str(tmp_path / "fc"))
    st = pipeline.State(test_df=_test_df())
    st.pool = _pool()
    st.molecules = pd.DataFrame({"molecule_id": ["m1", "m2"]})
    assert "fragments" in pipeline.STAGES
    st = pipeline.get_stage("fragments")(st, cfg)
    p = st.pool
    for c in F.FEATURES:
        assert c in p.columns
    assert np.isnan(p.loc[3, "frag_score"])            # 4th of m1: outside the top 3
    assert p.loc[[0, 1, 2, 4, 5], "frag_score"].notna().all()
    assert p.loc[0, "frag_score"] > p.loc[1, "frag_score"]   # CH2OH+: one cut from 1-propanol
    assert p.loc[0, "frag_score_rank"] == 1
    assert st.molecules["frag_n_scored"].tolist() == [3, 2]
    assert st.molecules["frag_n_spectra"].tolist() == [1, 1]
    assert (tmp_path / "fc" / "fragments.sqlite").exists()
    # second run from the disk cache gives identical features
    st2 = pipeline.State(test_df=_test_df())
    st2.pool = _pool()
    cache = F.FragmentCache(cfg)
    feats, _ = F.score_pool(st2.pool, st2.test_df, cfg, cache=cache)
    assert cache.disk_hits == 5 and cache.misses == 0
    pd.testing.assert_frame_equal(feats.reset_index(drop=True),
                                  p[list(F.FEATURES)].reset_index(drop=True), check_names=False)


def test_score_pool_parallel_matches_serial(tmp_path):
    cfg = Config(frag_top_k=3, frag_cache_dir="none")
    a, _ = F.score_pool(_pool(), _test_df(), cfg, n_jobs=1)
    b, _ = F.score_pool(_pool(), _test_df(), cfg, n_jobs=2)
    pd.testing.assert_frame_equal(a, b)


def test_cache_roundtrip(tmp_path):
    cfg = Config(frag_cache_dir=str(tmp_path))
    c1 = F.FragmentCache(cfg)
    a = c1.get_many(["CCO", "bad((", "CCO"])
    assert a[1] is None and c1.misses == 2
    c1.close()
    c2 = F.FragmentCache(cfg)
    b = c2.get_many(["CCO", "bad(("])
    assert c2.disk_hits == 2 and c2.misses == 0
    assert b[1] is None
    np.testing.assert_array_equal(a[0].masses, b[0].masses)
    np.testing.assert_array_equal(a[0].masks, b[0].masks)
    mem = F.FragmentCache(cfg, directory="none")
    assert mem.path is None and mem.get_many(["CCO"])[0] is not None


def test_peak_residuals():
    base = C + 3 * H + PROTON
    qs = _query([base + 0.0007, 200.0], [1.0, 1.0])
    err, spread = F.peak_residuals(qs, F.fragment_list("CCO"), 0.01)
    assert err[0] == pytest.approx(0.0007, abs=1e-9) and spread[0] < 1e-9
    assert np.isnan(err[1])


def test_stage_on_empty_pool():
    st = pipeline.State(test_df=_test_df())
    st.pool = _pool().iloc[:0]
    out = F.stage_fragments(st, Config())
    assert out.pool is not None and not len(out.pool)

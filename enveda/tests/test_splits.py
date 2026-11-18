"""WP7: holdout sets, folds, eligibility and the training exclusion (PLAN.md 5.2-5.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from casmi.config import Config
from casmi.splits import (ENVEDA_LIBS, HOLDOUT_COLUMNS, MASS_BINS, SETS, TrainingExclusion,
                          assert_no_holdout, ensure_mkeys, hash_uniform, holdout_molecules,
                          load_holdout, make_holdout, mass_quotas, save_holdout,
                          target_mass_density, training_exclusion)


def test_hash_uniform_deterministic():
    a = hash_uniform(["X", "Y", "Z"], 1, "s")
    b = hash_uniform(["Z", "X"], 1, "s")
    assert a[2] == b[0] and a[0] == b[1]
    assert ((a >= 0) & (a < 1)).all()
    assert not np.array_equal(a, hash_uniform(["X", "Y", "Z"], 2, "s"))


def test_mass_quotas_follow_target_and_capacity():
    cfg = Config()
    cdf = target_mass_density(cfg)
    assert cdf(np.array([cfg.val_target_mass_median]))[0] == pytest.approx(0.5)
    lo, hi = cdf(np.array([cfg.val_target_mass_min, cfg.val_target_mass_max]))
    assert lo == pytest.approx(0.01, abs=1e-3) and hi == pytest.approx(0.995, abs=1e-3)
    rng = np.random.default_rng(0)
    m = rng.uniform(150, 1200, 20000)
    b, q = mass_quotas(m, 1000, cfg)
    assert q.sum() == 1000 and len(q) == MASS_BINS
    assert (q <= np.bincount(b[b >= 0], minlength=MASS_BINS)).all()
    edges = np.exp(np.linspace(np.log(cfg.val_target_mass_min), np.log(cfg.val_target_mass_max),
                               MASS_BINS + 1))
    mid = np.sqrt(edges[:-1] * edges[1:])
    median_bin_mass = mid[np.searchsorted(np.cumsum(q), 500)]
    assert 280 < median_bin_mass < 430
    # capacity-limited: only 30 candidates below 300 Da, quota moves elsewhere
    m2 = np.concatenate([rng.uniform(160, 300, 30), rng.uniform(400, 1100, 3000)])
    b2, q2 = mass_quotas(m2, 500, cfg)
    assert q2.sum() == 500 and (q2 <= np.bincount(b2[b2 >= 0], minlength=MASS_BINS)).all()
    _, q3 = mass_quotas(m2[:10], 50, cfg)
    assert q3.sum() == 10


def test_holdout_sets(synthetic_train):
    ho = synthetic_train["holdout"]
    st = synthetic_train["structures"]
    assert list(ho.columns) == list(HOLDOUT_COLUMNS)
    assert ho["ik14"].is_unique
    mols = holdout_molecules(ho)
    assert mols["mkey"].is_unique
    n = mols["set"].value_counts().to_dict()
    np_examples = st.loc[st["n_lib_enveda-np-examples"] > 0, "mkey"].nunique()
    assert n["NP-TIMS"] == np_examples and n["SYN-TIMS"] == 30 and n["NP-OTHER"] == 30
    # every train ik14 of a holdout mkey is listed, and only those
    full = st[st["mkey"].isin(set(mols["mkey"]))]
    assert set(full["ik14"]) == set(ho["ik14"])
    assert (ho.groupby("mkey")["primary"].sum() == 1).all()
    # NP-OTHER: no Enveda spectra anywhere in the group, in the NP tier, query lib allowed
    no = ho[ho["set"] == "NP-OTHER"]
    g = st[st["ik14"].isin(set(no["ik14"]))]
    assert (g[[f"n_lib_{l}" for l in ENVEDA_LIBS]].sum(axis=1) == 0).all()
    assert st.set_index("ik14").loc[no.loc[no["primary"], "ik14"], "in_np"].all()
    assert set(no["query_lib"]) <= set(Config().val_np_other_libs)
    assert set(ho.loc[ho["set"] == "NP-TIMS", "query_lib"]) == {"enveda-np-examples"}
    assert set(ho.loc[ho["set"] == "SYN-TIMS", "query_lib"]) == {"enveda-180"}
    # folds balanced per set
    for s in SETS:
        f = mols.loc[mols["set"] == s, "fold"].value_counts()
        assert set(f.index) <= set(range(5)) and f.max() - f.min() <= 1
    # eligible_c1 = spectra in a library other than the query library
    libc = {c[6:]: c for c in st.columns if c.startswith("n_lib_")}
    sti = st.set_index("ik14")
    for r in mols.itertuples(index=False):
        other = sum(int(sti.loc[list(r.ik14_group), c].sum()) for l, c in libc.items()
                    if l != r.query_lib)
        assert r.eligible_c1 == (other > 0)
    # mass within the target range for SYN-TIMS
    syn = mols[mols["set"] == "SYN-TIMS"]
    cfg = Config()
    assert syn["mono_mass"].between(cfg.val_target_mass_min, cfg.val_target_mass_max).all()


def test_holdout_deterministic_and_roundtrip(synthetic_train, tmp_path):
    st, lc = synthetic_train["structures"], synthetic_train["lib_counts"]
    a, _ = make_holdout(st, lc, Config(), n_syn_tims=30, n_np_other=30)
    pd.testing.assert_frame_equal(a, synthetic_train["holdout"])
    p = save_holdout(a, tmp_path / "h.csv")
    b = load_holdout(p)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b[list(HOLDOUT_COLUMNS)],
                                  check_dtype=False, atol=1e-6)
    # without precomputed mkeys the same split comes out (closure mkeys computed on demand)
    st2 = st.assign(mkey=None)
    c, st_filled = make_holdout(st2, lc, Config(), n_syn_tims=30, n_np_other=30)
    pd.testing.assert_frame_equal(a, c)
    assert st_filled["mkey"].notna().sum() < len(st)       # only the closure was computed


def test_ensure_mkeys_closure(synthetic_train):
    st = synthetic_train["structures"]
    one = st.iloc[[0]]["ik14"].tolist()
    out = ensure_mkeys(st.assign(mkey=None), one, Config())
    f = st.iloc[0]["formula"]
    done = out["mkey"].notna()
    assert set(out.loc[done, "formula"]) == {f}
    assert (out.loc[done, "mkey"] == st.loc[done, "mkey"]).all()


def test_training_exclusion(synthetic_train):
    ho, st = synthetic_train["holdout"], synthetic_train["structures"]
    ex = training_exclusion(ho, st)
    assert isinstance(ex, TrainingExclusion)
    meta = pd.read_parquet(synthetic_train["out"] / "meta.parquet")
    kept = ex.filter(meta)
    assert len(kept) < len(meta)
    assert not kept["ik14"].isin(set(ho["ik14"])).any()
    mk = dict(zip(st["ik14"], st["mkey"]))
    assert not kept["ik14"].map(mk).isin(set(ho["mkey"])).any()
    with pytest.raises(AssertionError):
        ex.assert_clean(meta["ik14"].to_numpy(object))
    with pytest.raises(AssertionError):
        assert_no_holdout([ho["ik14"].iloc[0]], ho, structures=st)
    assert_no_holdout(kept["ik14"].to_numpy(object), ho, structures=st)
    # an alias missing from the file is still caught through the structures table
    alias = ho[~ho["primary"]]
    if len(alias):
        k = alias["ik14"].iloc[0]
        ex2 = training_exclusion(ho[ho["ik14"] != k], st)
        assert k in ex2.ik14
        with pytest.raises(AssertionError):
            ex2.assert_clean([k])
    # by mkey column directly
    with pytest.raises(AssertionError):
        ex.assert_clean(["NOTAKEY"], [ho["mkey"].iloc[0]])

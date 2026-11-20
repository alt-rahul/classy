"""Tests for casmi.libmatch (WP8): LIB candidates, pool merge, fusion rule 1b / demotion,
the "lib" stage end to end on the synthetic library, and the public-test plumbing check.

End-to-end tests use the ``synthetic_train`` fixture (conftest.py: synthetic train, WP5
library, holdout queries, NP-tier-only CandidateDB). The plumbing check (``data``) needs the
real library and WP6's public test truth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from casmi import fusion, io, libmatch, paths, pipeline
from casmi.config import Config, load_config
from casmi.splits import SETS
from casmi.validation import CONDITIONS, evaluate, load_queries, pipeline_predictor, summarize

B1 = paths.CONFIGS / "b1_library.yaml"


def _cfg(**kw) -> Config:
    base = load_config(B1).replace(db_tiers=("np",), n_jobs=1, mkey_cache=False,
                                   batch_molecules=50)
    return base.replace(**kw) if kw else base


# --- registration and config -----------------------------------------------------------

def test_stage_registered_under_own_name_and_config_loads():
    assert pipeline.get_stage("lib") is libmatch.stage_lib
    assert pipeline.get_stage("pool") is pipeline.stage_pool       # nothing replaced
    cfg = load_config(B1)
    assert "casmi.libmatch" in cfg.pipeline_plugins
    st = list(cfg.pipeline_stages)
    assert st.index("pool") < st.index("lib") < st.index("features")
    assert cfg.lib_sim in libmatch.LIB_SIMS
    d = Config()
    assert not d.fusion_lib_weak and not d.fusion_lib_demote      # plan rules by default


def test_lib_dir_resolution():
    assert libmatch.lib_dir(Config()) == paths.PROCESSED / "library"
    assert libmatch.lib_dir(Config(lib_dir="lib_x")) == paths.PROCESSED / "lib_x"
    assert libmatch.lib_dir(Config(lib_dir="/abs/lib")) == Path("/abs/lib")


# --- fusion extension ------------------------------------------------------------------

def _lib_frame():
    return pd.DataFrame([
        {"ik14": "A", "tier": "lib", "lib_best_cos": 0.9, "lib_n_matched": 6},   # rule 1
        {"ik14": "B", "tier": "lib", "lib_best_cos": 0.6, "lib_n_matched": 3},   # 1b
        {"ik14": "C", "tier": "lib", "lib_best_cos": 0.95, "lib_n_matched": 1},  # too few peaks
        {"ik14": "D", "tier": "lib", "lib_best_cos": 0.05, "lib_n_matched": 1},  # contradicted
        {"ik14": "E", "tier": "lib"},                                          # not measured
        {"ik14": "F", "tier": "np", "in_np": True, "pubmed_count": 10 ** 6},
        {"ik14": "G", "tier": "pubchem", "pmid_count": 5},
    ])


def test_extra_rules_off_by_default_keep_plan_scores():
    df = _lib_frame()
    cfg = Config()
    s, z = fusion.fuse(df, cfg, return_z=True)
    assert s[0] == pytest.approx(3.9)
    for i in (1, 2, 3, 4):
        assert s[i] == pytest.approx(1.0 + z[i])                    # rule 2, no literature
    # returns the very same array when both switches are off
    arr = np.arange(3.0)
    assert fusion.lib_extra_rules(arr, arr, arr, arr, np.ones(3, bool), np.zeros(3, bool),
                                  cfg) is arr


def test_weak_rule_and_demotion_order():
    df = _lib_frame()
    cfg = Config(fusion_lib_weak=True, fusion_lib_weak_cos_min=0.5, fusion_lib_weak_min_matched=2,
                 fusion_lib_demote=True, fusion_lib_demote_cos_max=0.2)
    s, z = fusion.fuse(df, cfg, return_z=True)
    cap = cfg.fusion_prior_cap
    assert s[1] == pytest.approx(2.0 + cap + (1.0 - cap) * 0.6)
    assert s[2] == pytest.approx(1.0 + z[2])                        # 1 matched peak: rule 2
    assert s[3] == pytest.approx(z[3]) and s[3] <= 1.0              # demoted below NP
    assert s[4] == pytest.approx(1.0 + z[4])                        # no evidence: rule 2
    # rule 1 > rule 1b > every NP score (max 2 + cap) > demoted LIB
    assert s[0] > s[1] > 2.0 + cap >= s[5] > s[3]
    # weak scores stay below rule 1 even at cosine 1
    one = pd.DataFrame([{"tier": "lib", "lib_best_cos": 1.0, "lib_n_matched": 2}])
    assert fusion.fuse(one, cfg)[0] == pytest.approx(3.0)


def test_additive_weak_rule():
    df = _lib_frame()
    cfg = Config(fusion_lib_cos_min=9.0, fusion_lib_weak=True, fusion_lib_weak_additive=True,
                 fusion_lib_weak_weight=0.5, fusion_lib_weak_cos_min=0.5,
                 fusion_lib_weak_min_matched=2)
    s, z = fusion.fuse(df, cfg, return_z=True)
    assert s[0] == pytest.approx(1.0 + z[0] + 0.5 * (0.9 - 0.5) / 0.5)   # rule 1 off: additive
    assert s[1] == pytest.approx(1.0 + z[1] + 0.5 * (0.6 - 0.5) / 0.5)
    assert s[2] == pytest.approx(1.0 + z[2])                               # 1 peak: plain rule 2
    assert s[5] == pytest.approx(1.0 + z[5] + cfg.fusion_prior_cap)        # NP unchanged
    # prior and similarity both count: a popular NP row can beat a weak LIB match
    assert s[5] > s[1]


# --- similarity choice, pool context, merge ---------------------------------------------

def test_lib_similarity_options():
    r = pd.DataFrame({"cos_max": [0.9, np.nan, 0.2], "ent_max": [0.7, np.nan, 0.6],
                      "cons_cos": [0.8, 0.5, 0.4], "n_matched_best": [5, 0, 2],
                      "cons_n_matched": [4, 3, 6]})
    s, n = libmatch.lib_similarity(r, Config(lib_sim="max_cos_cons"))
    assert s.tolist() == [0.9, 0.5, 0.4] and n.tolist() == [5, 3, 6]
    s, n = libmatch.lib_similarity(r, Config(lib_sim="cons_cos"))
    assert s.tolist() == [0.8, 0.5, 0.4] and n.tolist() == [4, 3, 6]
    s, _ = libmatch.lib_similarity(r, Config(lib_sim="mean_cos_ent"))
    assert s[0] == pytest.approx(0.8) and np.isnan(s[1])
    with pytest.raises(ValueError):
        libmatch.lib_similarity(r, Config(lib_sim="bogus"))
    with pytest.raises(ValueError):                    # graded needs graded_evidence columns
        libmatch.lib_similarity(r, Config(lib_sim="graded"))
    g = r.assign(_g_cos=[0.95, 0.5, np.nan], _g_n=[12, 3, 0])
    s, n = libmatch.lib_similarity(g, Config(lib_sim="graded"))
    assert s[0] == 0.95 and n.tolist() == [12, 3, 0]


def test_grade_orders_rule_then_cosine():
    cfg = Config(fusion_lib_cos_min=0.7, fusion_lib_min_matched=4, fusion_lib_weak_cos_min=0.5,
                 fusion_lib_weak_min_matched=2)
    g = libmatch._grade(np.array([1.0, 0.71, 0.99, 0.5, np.nan]), np.array([2, 4, 1, 2, 0]), cfg)
    # 0.71 with 4 peaks (rule 1) > 1.0 with 2 peaks (1b) > 0.5 with 2 peaks (1b) > 0.99 with 1
    assert g[1] > g[0] > g[3] > g[2] > g[4]


def test_graded_evidence_dominates_search_best_pair(synthetic_train):
    lib = synthetic_train["library"]
    cfg = _cfg()
    st = lib.structures
    multi = st[st["n_spectra"] >= 3].head(15)
    for row in multi.itertuples(index=False):
        a, b = int(row.spec_start), int(row.spec_end)
        qs = lib.queries_from_library(np.arange(a, b), M=float(row.mono_mass))
        r = lib.search_by_mass(float(row.mono_mass), queries=qs)
        gc, gn, gk = libmatch.graded_evidence(lib, qs, r["struct_idx"].to_numpy(np.int64), None,
                                              cfg)
        gg = libmatch._grade(gc, gn, cfg)
        for col_c, col_n in (("cos_max", "n_matched_best"), ("cons_cos", "cons_n_matched")):
            ok = r[col_c].notna().to_numpy()
            ref = libmatch._grade(r[col_c].to_numpy(float), r[col_n].to_numpy(float), cfg)
            assert (gg[ok] >= ref[ok] - 1e-9).all()
        own = r["ik14"].to_numpy() == row.ik14
        assert own.sum() == 1 and gc[own][0] > 0.999
        # excluding all its spectra leaves no evidence for it
        mask = lib.spectra_mask(ik14=[row.ik14])
        gc2, gn2, gk2 = libmatch.graded_evidence(lib, qs, r["struct_idx"].to_numpy(np.int64),
                                                 mask, cfg)
        assert np.isnan(gc2[own][0]) and gn2[own][0] == 0 and np.isnan(gk2[4][own][0])
        # per-k maxima are non-increasing in k and reproduce the graded choice
        ks = libmatch.GRADED_KS
        for i in range(len(ks) - 1):
            a_, b_ = gk[ks[i]], gk[ks[i + 1]]
            both = ~np.isnan(b_)
            assert (a_[both] >= b_[both] - 1e-12).all()
        c4 = gk[cfg.fusion_lib_min_matched]
        r1 = np.nan_to_num(c4, nan=-1) >= cfg.fusion_lib_cos_min
        assert np.allclose(gc[r1], c4[r1])


def test_pool_context():
    df = pd.DataFrame({"molecule_id": ["a", "a", "a", "b"],
                       "lib_best_cos": [0.9, np.nan, 0.4, 0.3]})
    out = libmatch._pool_context(df)
    assert out["lib_pool_n"].tolist() == [3, 3, 3, 1]
    assert out["lib_rank"].tolist() == [1, 3, 2, 1]
    assert out["lib_pool_best"].tolist()[:3] == [0.9] * 3
    assert out["lib_pool_second"].tolist()[:3] == [0.4] * 3 and np.isnan(out["lib_pool_second"][3])
    assert out["lib_gap"][0] == pytest.approx(0.5) and out["lib_gap"][2] == pytest.approx(-0.5)
    assert np.isnan(out["lib_gap"][3])


def _lib_rows(rows):
    base = libmatch.lib_candidates_empty()
    df = pd.DataFrame(rows)
    for c in base.columns:
        if c not in df.columns:
            df[c] = np.nan
    return df[list(base.columns)]


def test_merge_pool_dedupes_and_carries_priors():
    pool = pd.DataFrame({
        "molecule_id": ["m1", "m1", "m2"], "query_mass": [300.0, 300.0, 400.0],
        "ik14": ["AAA", "BBB", "AAA"], "smiles": ["db_a", "db_b", "db_a2"],
        "formula": ["C", "C", "C"], "mono_mass": [300.0, 300.0, 400.0], "ppm_err": [0.1, 0.2, 0.0],
        "tier": ["np", "pubchem", "np"], "in_np": [True, False, True],
        "pubmed_count": pd.array([7, 1, 3], dtype="Int32"),
    })
    lib = _lib_rows([
        {"molecule_id": "m1", "query_mass": 300.0, "ik14": "AAA", "smiles": "lib_a",
         "formula": "C", "mono_mass": 300.0, "ppm_err": 0.1, "tier": "lib", "is_lib": True,
         "lib_best_cos": 0.9, "lib_n_matched": 5},
        {"molecule_id": "m1", "query_mass": 300.0, "ik14": "CCC", "smiles": "lib_c",
         "formula": "C", "mono_mass": 300.0, "ppm_err": 1.0, "tier": "lib", "is_lib": True,
         "lib_best_cos": 0.2, "lib_n_matched": 1},
    ])
    out = libmatch.merge_pool(pool, lib)
    assert len(out) == 4                                            # AAA of m1 merged
    assert out["molecule_id"].tolist() == ["m1", "m1", "m1", "m2"]  # grouped, stable
    m1 = out[out["molecule_id"] == "m1"].set_index("ik14")
    assert m1.loc["AAA", "tier"] == "lib" and m1.loc["AAA", "db_tier"] == "np"
    assert m1.loc["AAA", "smiles"] == "lib_a" and int(m1.loc["AAA", "pubmed_count"]) == 7
    assert bool(m1.loc["AAA", "in_np"]) and bool(m1.loc["AAA", "is_lib"])
    assert pd.isna(m1.loc["CCC", "db_tier"]) and pd.isna(m1.loc["CCC", "pubmed_count"])
    assert m1.loc["BBB", "tier"] == "pubchem" and not bool(m1.loc["BBB", "is_lib"])
    assert np.isnan(m1.loc["BBB", "lib_best_cos"])
    src = dict(zip(out["molecule_id"] + out["ik14"], fusion.candidate_source(out)))
    assert src["m1AAA"] == "lib" and src["m1CCC"] == "lib" and src["m1BBB"] == "pubchem"
    # the merged LIB row keeps its DB prior: it beats the LIB-only row under rule 2
    fe = fusion.add_base_features(out)
    assert fe.set_index(fe["molecule_id"] + fe["ik14"]).loc["m1AAA", "prior"] > 0
    # no LIB rows / no pool
    o2 = libmatch.merge_pool(pool, None)
    assert len(o2) == 3 and not o2["is_lib"].any() and set(libmatch.LIB_COLUMNS) <= set(o2.columns)
    o3 = libmatch.merge_pool(None, lib)
    assert len(o3) == 2 and o3["is_lib"].all()
    o4 = libmatch.merge_pool(pool.iloc[:0], lib)
    assert len(o4) == 2


# --- stage end to end on the synthetic library -------------------------------------------

def test_stage_without_library_keeps_pool(synthetic_train, tmp_path):
    q, _ = load_queries("NP-TIMS", synthetic_train["val_dir"])
    sub = q[q["molecule_id"].isin(sorted(q["molecule_id"].unique())[:3])]
    libmatch.clear_library_cache()
    cfg = _cfg(lib_dir=str(tmp_path / "nothing"))
    base = pipeline.run(sub, cfg.replace(pipeline_stages=("mass", "pool", "features", "score",
                                                           "assemble"), pipeline_plugins=()),
                        db=synthetic_train["db"])
    res = pipeline.run(sub, cfg, db=synthetic_train["db"])
    assert res.preds == base.preds
    assert (res.molecules["lib_n"] == 0).all()


def test_stage_self_match_and_features(synthetic_train):
    q, t = load_queries("NP-TIMS", synthetic_train["val_dir"])
    ids = sorted(q["molecule_id"].unique())[:8]
    sub = q[q["molecule_id"].isin(ids)]
    cfg = _cfg()
    lib = synthetic_train["library"]
    st = pipeline.State(test_df=sub.reset_index(drop=True), db=synthetic_train["db"],
                        extras={"library": lib})
    st = pipeline.run_stages(st, cfg, pipeline.resolve_stages(cfg)[:3])   # mass, pool, lib
    pool = st.pool
    assert set(libmatch.LIB_COLUMNS) <= set(pool.columns)
    lr = pool[pool["is_lib"]]
    assert len(lr) and (lr["tier"] == "lib").all()
    # no duplicate (molecule, ik14) after the merge
    assert not pool.duplicated(["molecule_id", "ik14"]).any()
    truth = dict(zip(t["molecule_id"], t["ik14"]))
    for mid in ids:                                 # no exclusions: the truth matches itself
        r = lr[(lr["molecule_id"] == mid) & (lr["ik14"] == truth[mid])]
        assert len(r) == 1 and r["lib_best_cos"].iloc[0] > 0.99 and r["lib_rank"].iloc[0] == 1
    rep = st.molecules.set_index("molecule_id")
    assert (rep.loc[ids, "lib_n"] >= 1).all() and (rep.loc[ids, "lib_n_rule1"] >= 1).all()
    # deterministic
    a = pipeline.predict(sub, cfg, db=synthetic_train["db"], extras={"library": lib})
    b = pipeline.predict(sub, cfg, db=synthetic_train["db"], extras={"library": lib})
    assert a == b and all(a[m][0] for m in ids)


def test_evaluate_b1_respects_conditions(synthetic_train):
    cfg = _cfg()
    tab, cells = evaluate(pipeline_predictor, cfg, sets=["NP-TIMS", "NP-OTHER"],
                          val_dir=synthetic_train["val_dir"], library=synthetic_train["library"],
                          train_dir=synthetic_train["out"], db=synthetic_train["db"])
    summ = summarize(tab, cfg)
    for s in ("NP-TIMS", "NP-OTHER"):
        c1, c2, c3 = (summ["cells"][f"{s}|{c}"] for c in CONDITIONS)
        assert c1["mrr"] > 0.9 and c1["mrr"] > c2["mrr"] + 0.3     # references elsewhere
        assert c3["mrr"] == 0.0 and c3["recall_any"] == 0.0
        assert c1["recall_main"] == 1.0 and c2["recall_main"] == 1.0
    # every library search was resolved to its own molecule (no union fallbacks)
    assert sum(c.n_union_fallbacks for c in cells) == 0


# --- plumbing check on the public test (real library) -----------------------------------

@pytest.mark.data("data/processed/library/meta.parquet",
                  "data/processed/train/public_test_truth.parquet",
                  "data/processed/db/np/np.parquet")
def test_public_test_plumbing_subset():
    from casmi.metric import reciprocal_ranks

    test = io.read_test(paths.TEST_PARQUET)
    truth = pd.read_parquet(paths.PROCESSED / "train" / "public_test_truth.parquet")
    ids = sorted(truth["molecule_id"].astype(str))[::8]                # 50 molecules
    sub = test[test["molecule_id"].isin(ids)]
    libmatch.clear_library_cache()
    res = pipeline.run(sub, _cfg(n_jobs=1))
    tm = dict(zip(truth["molecule_id"].astype(str), truth["smiles"]))
    rr = reciprocal_ranks(res.preds, {m: tm[m] for m in ids}, k=25, n_jobs=1)
    assert np.mean(list(rr.values())) >= 0.95

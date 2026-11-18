"""WP7: query emulation, conditions, exclusions, scoring and reporting (PLAN.md 5.2-5.5).

Uses the ``synthetic_train`` fixture (conftest.py): synthetic train, WP6 outputs, WP5 library,
holdout and query tables, and an NP-tier-only CandidateDB.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from casmi import paths, pipeline
from casmi.adducts import TEST_ADDUCTS
from casmi.config import Config
from casmi.io import TEST_COLUMNS, read_test
from casmi.splits import SET_CODES, SETS
from casmi.validation import (CONDITIONS, TRUTH_COLUMNS, ExcludingDB, ValidationHooks,
                              append_experiments, ce_class, cco_predictor, condition_exclusions,
                              evaluate, format_cells, library_guard, load_queries,
                              pipeline_predictor, queries_current, ring_closures, run_cell,
                              self_check, set_slug, skeleton_key, summarize, truth_keys,
                              truth_predictor)


@pytest.fixture(scope="module")
def cfg():
    return Config(n_jobs=1, db_tiers=("np",), batch_molecules=50)


@pytest.fixture(scope="module")
def keys(synthetic_train, cfg):
    out = {}
    for s in SETS:
        _, t = load_queries(s, synthetic_train["val_dir"])
        out[s] = truth_keys(t, cfg, db=synthetic_train["db"],
                            cache_path=synthetic_train["val_dir"] / "truth_keys.parquet")
    return out


def test_query_tables_have_test_schema(synthetic_train):
    test_schema = pq.read_schema(paths.TEST_PARQUET)
    meta = pd.read_parquet(synthetic_train["out"] / "meta.parquet").set_index("spectrum_id")
    cfg = Config()
    for s in SETS:
        d = synthetic_train["val_dir"]
        qs = pq.read_schema(d / f"queries_{set_slug(s)}.parquet")
        assert qs.names == list(TEST_COLUMNS) == test_schema.names
        assert [f.type for f in qs] == [f.type for f in test_schema]
        q = read_test(d / f"queries_{set_slug(s)}.parquet")
        t = pd.read_parquet(d / f"truth_{set_slug(s)}.parquet")
        assert list(t.columns) == list(TRUTH_COLUMNS)
        assert set(q["molecule_id"]) == set(t["molecule_id"])
        assert all(m.startswith(SET_CODES[s]) for m in t["molecule_id"])
        n = q.groupby("molecule_id").size()
        assert n.min() >= 1 and n.max() <= cfg.val_max_spectra_per_molecule
        assert (n.reindex(t["molecule_id"]).to_numpy() == t["n_query_spectra"].to_numpy()).all()
        assert set(q["adduct"]) <= set(TEST_ADDUCTS)
        assert set(q["ionization_mode"]) <= {"positive", "negative"}
        for mz, it, p in zip(q["ms2_mzs"], q["ms2_normalized_intensities"], q["precursor_mz"]):
            assert len(mz) and np.all(np.diff(mz) >= 0) and it.max() == pytest.approx(1.0)
            assert mz.max() <= p + cfg.val_query_peak_cut_da
        # every query spectrum comes from the molecule's query library and ik14 group
        tq = t.set_index("molecule_id")
        for mid, sid in zip(q["molecule_id"], q["spectrum_id"]):
            r = meta.loc[sid]
            assert r["ingest_lib"] == tq.loc[mid, "query_lib"]
            assert r["ik14"] in tq.loc[mid, "ik14_group"].split(";") and r["query_ok"]
        assert (t["peaks_source"] == "raw").all()
    assert queries_current(synthetic_train["holdout"], synthetic_train["val_dir"])
    assert not queries_current(synthetic_train["holdout"].iloc[1:], synthetic_train["val_dir"])


def test_ce_class_and_skeleton():
    assert ce_class(1, 20.4) == "20" and ce_class(1, -40) == "40" and ce_class(3, 40) == "merged"
    assert ce_class(0, np.nan) == "unknown" and ce_class(1, 35) == "other"
    assert skeleton_key("CC(=O)CC(C)=O") == skeleton_key("CC(O)=CC(C)=O")
    assert skeleton_key("CC(=O)[O-]") == skeleton_key("CC(=O)O")
    assert skeleton_key("CCCO") != skeleton_key("CC(C)O")
    assert ring_closures("c1ccc2ccccc2c1") == 2 and ring_closures("C[C@@H]1CC1") == 1
    assert ring_closures("[13CH3]C") == 0


def test_truth_keys_contain_groups(synthetic_train, keys):
    for s in SETS:
        _, t = load_queries(s, synthetic_train["val_dir"])
        for mid, g in zip(t["molecule_id"], t["ik14_group"]):
            assert set(g.split(";")) <= keys[s][mid]


def test_self_check(synthetic_train, cfg):
    res = self_check(cfg, val_dir=synthetic_train["val_dir"], library=synthetic_train["library"],
                     train_dir=synthetic_train["out"], db=synthetic_train["db"])
    assert {k for k, v in res.items() if k.startswith("truth")} and \
        all(v == (1.0 if k.startswith("truth") else 0.0) for k, v in res.items())
    assert len(res) == 2 * len(SETS) * len(CONDITIONS)


def test_db_exclusions_per_molecule(synthetic_train, keys):
    q, t = load_queries("NP-TIMS", synthetic_train["val_dir"])
    db = synthetic_train["db"]
    for cond in CONDITIONS:
        excl = condition_exclusions(t, cond, keys["NP-TIMS"])
        hooks = ValidationHooks(cond, excl)
        proxy = ExcludingDB(db, hooks)
        hooks.active = tuple(t["molecule_id"])
        res = proxy.lookup_batch(t["mono_mass"].to_numpy(), tol_ppm=10.0, tiers=("np",),
                                 keys=t["molecule_id"].tolist())
        found = {(m, k) for m, k in zip(res["query"], res["ik14"])}
        hit = sum(any((m, k) in found for k in keys["NP-TIMS"][m]) for m in t["molecule_id"])
        if cond == "C3":
            assert hit == 0
            near = proxy.load_np()
            assert not near["ik14"].isin(set().union(*keys["NP-TIMS"].values())).any()
            assert len(proxy.np_mass) == len(near)
        else:
            assert hit == len(t)                      # truths are NP-tier structures
            assert len(proxy.load_np()) == len(db.load_np())
        assert not hooks.violations
        assert all(hooks.pools[m]["main"] for m in t["molecule_id"])


def test_library_masks(synthetic_train, keys):
    lib = synthetic_train["library"]
    st = synthetic_train["structures"]
    lib_mk = dict(zip(st["ik14"], st["mkey"]))
    for s in SETS:
        _, t = load_queries(s, synthetic_train["val_dir"])
        for cond in CONDITIONS:
            hooks = ValidationHooks(cond, condition_exclusions(t, cond, keys[s]), lib_mkeys=lib_mk)
            chk = hooks.check_library(lib, t)
            assert not hooks.violations
            assert (chk["lib_own_left"] == 0).all()
            if cond in ("C2", "C3"):
                assert (chk["lib_other_left"] == 0).all()
            else:
                el = dict(zip(t["molecule_id"], t["eligible_c1"]))
                assert all((left > 0) == el[m] for m, left in
                           zip(chk["molecule_id"], chk["lib_other_left"]))
            mask = hooks.library_mask(lib, t["molecule_id"].iloc[0])
            assert mask.shape == (lib.n_spectra,) and mask.any()


def _lib_pool(state, cfg):
    """A toy library stage: Baseline 0 pool plus LIB candidates with spectral matches."""
    state = pipeline.stage_pool(state, cfg)
    lib = state.extras["library"]
    rows = [state.pool]
    for mid, M in zip(state.molecules["molecule_id"], state.molecules["M"]):
        sub = state.test_df[state.test_df["molecule_id"] == mid]
        r = lib.search_by_mass(float(M), queries=sub)
        if len(r):
            rows.append(pd.DataFrame({
                "molecule_id": mid, "query_mass": float(M), "ik14": r["ik14"].astype(str),
                "smiles": r["smiles"], "formula": r["formula"], "mono_mass": r["mono_mass"],
                "ppm_err": r["mass_err_ppm"], "tier": "lib", "lib_best_cos": r["cos_max"],
                "lib_n_matched": r["n_matched_best"]}))
    state.pool = (pd.concat(rows, ignore_index=True).sort_values("molecule_id", kind="mergesort")
                  .reset_index(drop=True))
    return state


def _lib_predictor(test_df, cfg, **kw):
    return pipeline.predict(test_df, cfg, stages={"pool": _lib_pool}, **kw)


def test_library_guard_end_to_end(synthetic_train, cfg):
    tab, cells = evaluate(_lib_predictor, cfg, sets=["NP-TIMS", "NP-OTHER"],
                          val_dir=synthetic_train["val_dir"], library=synthetic_train["library"],
                          train_dir=synthetic_train["out"], db=synthetic_train["db"])
    summ = summarize(tab, cfg)
    for s in ("NP-TIMS", "NP-OTHER"):
        c1, c2, c3 = (summ["cells"][f"{s}|{c}"] for c in CONDITIONS)
        assert c1["mrr"] > c2["mrr"] + 0.2           # references from other libraries help
        assert c1["recall_main"] == 1.0 and c3["recall_any"] == 0.0 and c3["mrr"] == 0.0
    # the guard resolved every library call to its molecule (DataFrame queries)
    assert sum(c.n_union_fallbacks for c in cells) == 0
    # the patch is removed afterwards
    from casmi.library import Library
    assert Library.search_by_mass.__qualname__ == "Library.search_by_mass"


def test_guard_union_fallback_and_violations(synthetic_train, keys, cfg):
    q, t = load_queries("NP-TIMS", synthetic_train["val_dir"])
    lib = synthetic_train["library"]
    hooks = ValidationHooks("C2", condition_exclusions(t, "C2", keys["NP-TIMS"]))
    m0 = t.iloc[0]
    with library_guard(hooks):
        hooks.active = tuple(t["molecule_id"])
        r = lib.search_by_mass(float(m0["mono_mass"]))          # no queries: chunk union
        assert m0["ik14"] not in set(r["ik14"]) and hooks.n_union_fallbacks == 1
        ci = lib.candidates_in_window(float(m0["mono_mass"]))
        assert m0["ik14"] not in set(lib.struct_ik14[ci])
    with library_guard(ValidationHooks("C1", condition_exclusions(t, "C1", keys["NP-TIMS"]))) as h1:
        h1.active = (m0["molecule_id"],)
        r = lib.search_by_mass(float(m0["mono_mass"]))
        assert (m0["ik14"] in set(r["ik14"])) == bool(m0["eligible_c1"])
    # a predictor that leaks the truth under C3 fails the cell
    with pytest.raises(AssertionError, match="truth among the predictions"):
        run_cell(truth_predictor(t), q, t, "C3", cfg, keys["NP-TIMS"], db=synthetic_train["db"])
    with pytest.raises(AssertionError, match="excluded ik14"):
        h = ValidationHooks("C3", condition_exclusions(t, "C3", keys["NP-TIMS"]))
        h.check_rows([m0["molecule_id"]], [m0["ik14"]], "test")
        if h.violations:
            raise AssertionError(h.violations[0])


def test_evaluate_pipeline_all_cells(synthetic_train, cfg):
    tab, cells = evaluate(pipeline_predictor, cfg, val_dir=synthetic_train["val_dir"],
                          library=synthetic_train["library"], train_dir=synthetic_train["out"],
                          db=synthetic_train["db"])
    assert len(cells) == 9
    summ = summarize(tab, cfg)
    for s in SETS:
        for c in CONDITIONS:
            d = summ["cells"][f"{s}|{c}"]
            assert d["n"] > 0 and 0 <= d["mrr"] <= 1
            assert d["top1"] <= d["top5"] <= d["top25"]
            if c == "C3":
                assert d["mrr"] == 0.0 and d["recall_any"] == 0.0 and d["truth_in_preds"] == 0
            else:
                assert d["recall_main"] == 1.0
        w = summ["sets"][s]
        assert w["worst"] == min(w["blends"].values())
        assert w["worst_ci"][0] <= w["worst"] + 1e-9 and w["worst_ci"][1] >= w["worst"] - 1e-9
    # Baseline 0 does not use the library: C1 equals C2 on the C1-eligible molecules
    t = tab.pivot_table(index=["set", "molecule_id"], columns="condition", values="rr")
    both = t.dropna(subset=["C1", "C2"])
    assert np.allclose(both["C1"], both["C2"])
    # rank and hit flags agree with rr
    v = tab[tab["in_cell"]]
    assert ((v["rank"] > 0) == (v["rr"] > 0)).all()
    assert np.allclose(v.loc[v["rank"] > 0, "rr"], 1.0 / v.loc[v["rank"] > 0, "rank"])
    assert "NP-TIMS C1/C2/C3" in format_cells(summ)


def test_summarize_hand_values():
    cfg = Config(val_bootstrap_n=200)
    rows = []
    rr = {"C1": [1.0, 0.5, np.nan], "C2": [0.5, 0.0, 0.25], "C3": [0.0, 0.0, 0.1]}
    for c, vals in rr.items():
        for i, v in enumerate(vals):
            rows.append({"set": "NP-TIMS", "condition": c, "molecule_id": f"m{i}", "mkey": f"k{i}",
                         "fold": 0, "in_cell": not np.isnan(v), "rr": v,
                         "rank": int(round(1 / v)) if v and not np.isnan(v) else 0,
                         "top1": v == 1.0, "top5": v >= 0.2, "top25": v > 0, "in_pool_main": True,
                         "in_pool_any": True, "n_pool_main": 5, "n_pool_any": 5, "n_guesses": 25,
                         "truth_in_preds": False, "pubmed0": i == 0, "mass_tercile": i,
                         "n_query_spectra": 2})
    s = summarize(pd.DataFrame(rows), cfg)
    m1, m2, m3 = 0.75, 0.25, 0.1 / 3
    assert s["cells"]["NP-TIMS|C1"]["mrr"] == pytest.approx(m1)
    assert s["cells"]["NP-TIMS|C1"]["n"] == 2
    assert s["cells"]["NP-TIMS|C2"]["mrr_pubmed0"] == pytest.approx(0.5)
    want = {mix: m1 * mix[0] + m2 * mix[1] + m3 * mix[2] for mix in cfg.val_blend_mixes}
    got = s["sets"]["NP-TIMS"]["blends"]
    for mix, v in want.items():
        assert got[",".join(f"{w:g}" for w in mix)] == pytest.approx(v)
    assert s["primary"] == pytest.approx(min(want.values()))
    lo, hi = s["primary_ci"]
    assert lo <= s["primary"] <= hi


def test_append_experiments(tmp_path):
    p = tmp_path / "EXPERIMENTS.md"
    p.write_text("# Experiments\n\nintro\n\n| date | tag | config | git commit | cells | notes |\n"
                 "|---|---|---|---|---|---|\n| 2026-11-18 | b0_x | c | g | none | old row |\n\n"
                 "## Notes and spikes\n\ntext\n")
    summ = {"cells": {"NP-TIMS|C1": {"n": 2, "mrr": 0.5, "recall_any": 1.0}}, "sets": {}}
    append_experiments(summ, "b0_test", "configs/x.yaml", p, notes="n", git="abc")
    append_experiments(summ, "b0_test2", "configs/x.yaml", p, notes="n", git="abc")
    lines = p.read_text().splitlines()
    i = next(i for i, l in enumerate(lines) if l.startswith("| date |"))
    assert lines[i + 1].startswith("|---")
    assert "old row" in lines[i + 2]
    assert "b0_test |" in lines[i + 3] and "b0_test2 |" in lines[i + 4]
    assert lines[i + 3].count("|") == lines[i].count("|")
    assert lines[i + 5] == "" and lines[i + 6].startswith("## Notes and spikes")


def test_predictor_signatures(synthetic_train):
    q, t = load_queries("SYN-TIMS", synthetic_train["val_dir"])
    assert all(v == ["CCO"] for v in cco_predictor(q).values())
    tp = truth_predictor(t)(q)
    assert set(tp) == set(t["molecule_id"])

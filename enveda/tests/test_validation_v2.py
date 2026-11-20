"""R1: holdout v2, protocol v2 cells, LB-proxy, stored pools, probes and the class-mix fit.

Hand-made tables test the arithmetic; the ``synthetic_train`` fixture (conftest.py: synthetic
train, WP6 outputs, library, v1 holdout and queries, NP-tier-only DB) runs the v2 harness end
to end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from casmi import pipeline, validation as V
from casmi.chem import morgan_fps, tanimoto
from casmi.config import Config
from casmi.splits import (HOLDOUT_COLUMNS, HOLDOUT_V2_COLUMNS, load_holdout, load_holdout_v2,
                          make_holdout_v2, pmid_bins, save_holdout_v2, w_obs_weights,
                          zen_holdout_rows)


# --------------------------------------------------------------------------------------
# hand-made tables
# --------------------------------------------------------------------------------------

def _attrs(rows):
    a = pd.DataFrame(rows)
    a = a.assign(fold=0, eligible_c1=True, has_public_spectra=False, n_ext_lib=0, fast=True,
                 w_obs=1.0, in_np=True, in_pubchem=True)
    return a.set_index(a["mkey"])


def _row(cell, s, m, rr):
    rank = int(round(1 / rr)) if rr > 0 else 0
    return {"cell": cell, "set": s, "condition": V.CELL_CONDITION[cell], "molecule_id": m,
            "mkey": m, "fold": 0, "in_cell": True, "rr": rr, "rank": rank, "top1": rank == 1,
            "top5": 0 < rank <= 5, "top25": rank > 0, "in_pool_main": True, "in_pool_any": True,
            "n_pool_main": 5, "n_pool_any": 5, "n_guesses": 25, "truth_in_preds": False,
            "pubmed0": False, "mass_tercile": 0, "n_query_spectra": 2, "structural": False}


@pytest.fixture()
def hand():
    cfg = Config(val_bootstrap_n=200)
    attrs = _attrs([
        {"mkey": "a", "set": "NP-TIMS", "pmid_count": 0.0, "c2_cell": "NP", "w_fast": 1.0},
        {"mkey": "b", "set": "NP-TIMS", "pmid_count": 99.0, "c2_cell": "NP", "w_fast": 1.0},
        {"mkey": "c", "set": "NP-OTHER", "pmid_count": 3.0, "c2_cell": "NP", "w_fast": 2.0},
        {"mkey": "d", "set": "SYN-TIMS", "pmid_count": 0.0, "c2_cell": "PC", "w_fast": 4.0},
        {"mkey": "e", "set": "SYN-TIMS", "pmid_count": 0.0, "c2_cell": "PC", "w_fast": 4.0},
    ])
    rows = [_row("C1", "NP-TIMS", "a", 1.0), _row("C1", "NP-TIMS", "b", 0.5),
            _row("C1", "NP-OTHER", "c", 1.0),
            _row("C2-NP", "NP-TIMS", "a", 0.5), _row("C2-NP", "NP-TIMS", "b", 1.0),
            _row("C2-NP", "NP-OTHER", "c", 0.25),
            _row("C2-PC", "SYN-TIMS", "d", 0.5), _row("C2-PC", "SYN-TIMS", "e", 0.0),
            _row("C3h", "NP-TIMS", "a", 0.0), _row("C3h", "NP-TIMS", "b", 0.2),
            _row("C3h", "NP-OTHER", "c", 0.0)]
    return cfg, attrs, pd.DataFrame(rows)


def test_w_obs_and_bins():
    w = w_obs_weights([0, 1, 99, None], 1.0)
    assert np.allclose(w, [1.0, 0.5, 0.01, 1.0])
    assert pmid_bins([0, 1, 9, 10, 5000, None], (1, 10, 100, 1000)).tolist() == [0, 1, 1, 2, 4, 0]


def test_proxy_hand_values(hand):
    cfg, attrs, t = hand
    alpha = 0.5
    meta = {"alpha": {"A": alpha, "B": alpha, "C": alpha}}
    rep = V.proxy_report(t, attrs, cfg, panel="fast", meta=meta, n_boot=100)
    wa, wb, wc = 1.0, 100 ** -alpha, 2.0 * 4 ** -alpha
    c1 = (wa * 1.0 + wb * 0.5) / (wa + wb)                  # NP-TIMS only
    c1o = 1.0
    c2np = (wa * 0.5 + wb * 1.0 + wc * 0.25) / (wa + wb + wc)
    c2pc = 0.25                                             # equal w_fast
    c3h = (0.0 + 0.2 + 2.0 * 0.0) / 4.0                     # w_fast only
    a = rep["mixes"]["A"]
    assert a["cells"]["C1"]["mrr"] == pytest.approx(c1)
    assert a["cells"]["C1o"]["mrr"] == pytest.approx(c1o)
    assert a["cells"]["C2-NP"]["mrr"] == pytest.approx(c2np)
    assert a["cells"]["C2-PC"]["mrr"] == pytest.approx(c2pc)
    assert a["cells"]["C3h"]["mrr"] == pytest.approx(c3h)
    pi = V.v2_mixes(cfg)["A"]
    assert a["P"] == pytest.approx(pi @ [c1, c2np, c2pc, c3h])
    lo, hi = a["P_ci"]
    assert lo <= a["P"] + 1e-9 and hi >= a["P"] - 1e-9
    # top-1 + tail = P
    assert a["P_top1"] + a["P_tail"] == pytest.approx(a["P"])
    # full panel ignores w_fast
    full = V.proxy_report(t, attrs, cfg, panel="full", meta=meta, n_boot=0)
    assert full["mixes"]["A"]["cells"]["C3h"]["mrr"] == pytest.approx(0.2 / 3)


def test_calibrate_alpha_hits_target(hand):
    cfg, attrs, t = hand
    full = V.proxy_report(t, attrs, cfg, panel="full", meta={"alpha": {}}, n_boot=0)
    p0 = full["mixes"]["A"]["P"]
    target = 0.17 * 0.9 + 0.30 * 0.5   # between alpha 0 and alpha max for this table
    cal = V.calibrate_alpha(t, attrs, cfg, panel="full", target=target, mixes=["A"])
    a = cal["alpha"]["A"]
    assert 0 < a <= cfg.val_v2_alpha_max and cal["flags"]["A"] in ("ok", "approx")
    rep = V.proxy_report(t[t["cell"].isin(["C1", "C2-NP"])], attrs, cfg, panel="full",
                         meta={"alpha": cal["alpha"]}, n_boot=0, mixes=["A"])
    assert rep["mixes"]["A"]["P"] == pytest.approx(target, abs=2e-3)
    assert p0 != pytest.approx(target)


def test_paired_report_and_adoption(hand):
    cfg, attrs, t = hand
    meta = {"alpha": {"A": 0.5, "B": 0.5, "C": 0.5}}
    same = V.paired_report(t, t, attrs, cfg, meta=meta, n_boot=200)
    assert same["mixes"]["A"]["diff"] == pytest.approx(0.0)
    assert same["mixes"]["A"]["ci"] == [pytest.approx(0.0), pytest.approx(0.0)]
    assert same["adopt"] is False
    better = t.copy()
    better.loc[better["cell"] == "C3h", "rr"] = 1.0
    better.loc[better["cell"] == "C3h", "top1"] = True
    rep = V.paired_report(t, better, attrs, cfg, meta=meta, n_boot=200)
    a = rep["mixes"]["A"]
    assert a["diff"] > 0.1 and a["ci"][0] > 0
    assert a["cells"]["C3h"]["diff"] > 0
    assert rep["adopt"] is True
    worse = t.copy()
    worse.loc[worse["cell"] == "C1", "rr"] = 0.0
    worse.loc[worse["cell"] == "C1", "top1"] = False
    rep2 = V.paired_report(better, worse, attrs, cfg, meta=meta, n_boot=200)
    assert rep2["adopt"] is False


def test_v1_to_v2(hand):
    cfg, attrs, _ = hand
    v1 = pd.DataFrame([{**_row("C1", "NP-TIMS", "a", 1.0), "condition": "C1"},
                       {**_row("C1", "SYN-TIMS", "d", 1.0), "condition": "C1"},
                       {**_row("C1", "NP-TIMS", "a", 0.5), "condition": "C2"},
                       {**_row("C1", "SYN-TIMS", "d", 0.5), "condition": "C2"},
                       {**_row("C1", "NP-TIMS", "a", 0.0), "condition": "C3"}]).drop(columns=["cell", "structural"])
    v2 = V.v1_to_v2(v1, attrs)
    assert v2["cell"].tolist() == ["C1", "C2-NP", "C2-PC", "C3e", "C3h"]
    assert v2["structural"].tolist() == [False] * 4 + [True]


def test_tanimoto_index_matches_brute_force():
    rng = np.random.default_rng(0)
    fps = (rng.random((3000, 512)) < 0.02).astype(np.uint8) * rng.integers(0, 256, (3000, 512), dtype=np.uint8)
    q = fps[17].copy()
    fps[5] = q
    fps[6] = q
    fps[6, :3] ^= 0xFF
    idx = V.TanimotoIndex(fps, chunk=257)
    for thr in (0.3, 0.75, 0.99):
        got, sims = idx.query(q, thr)
        ref = tanimoto(q, fps)
        want = np.flatnonzero(ref >= thr - 1e-6)
        assert sorted(got.tolist()) == sorted(want.tolist())
        assert np.allclose(np.sort(sims), np.sort(ref[want]), atol=1e-6)
    assert 5 in idx.query(q, 0.999)[0] and 17 in idx.query(q, 0.999)[0]


def test_exclusions_v2_c3h():
    truth = pd.DataFrame({"molecule_id": ["m1"], "mkey": ["K"], "ik14_group": ["K;K2"],
                          "query_lib": ["enveda-np-examples"], "eligible_c1": [True]})
    keys = {"m1": frozenset({"K", "K2", "KDB"})}
    near = {"K": {"train": frozenset({"N1", "K"}), "np": frozenset({"N2"})}}
    e3e = V.exclusions_v2(truth, "C3e", keys)["m1"]
    e3h = V.exclusions_v2(truth, "C3h", keys, near)["m1"]
    e2 = V.exclusions_v2(truth, "C2-NP", keys)["m1"]
    e1 = V.exclusions_v2(truth, "C1", keys)["m1"]
    assert e1.lib_libs == frozenset({"enveda-np-examples"}) and e2.lib_libs is None
    assert e2.db_ik14 == frozenset() and e3e.db_ik14 == keys["m1"]
    assert e3h.group == frozenset({"K", "K2", "N1"})
    assert e3h.db_ik14 == frozenset({"K", "K2", "KDB", "N1", "N2"})
    assert e3h.truth == keys["m1"]
    with pytest.raises(ValueError):
        V.exclusions_v2(truth, "C3h", keys)
    cnt = V.c3h_removed_counts(truth.assign(set="NP-TIMS"), keys, near)
    assert cnt["n_neighbours"].iloc[0] == 2 and cnt["n_removed"].iloc[0] == 5


def test_blind_queries():
    cfg = Config()
    q = pd.DataFrame({"molecule_id": ["a", "b"], "ms2_mzs": [np.array([100.0, 150.0]), np.array([80.0])],
                      "ms2_normalized_intensities": [np.array([1.0, 0.5]), np.array([1.0])],
                      "precursor_mz": [200.0, 300.0], "adduct": ["[M+H]+", "[M-H]-"]})
    b = V.blind_queries(q, cfg)
    assert all(list(x) == list(cfg.val_v2_blind_mz) for x in b["ms2_mzs"])
    assert b["precursor_mz"].tolist() == [200.0, 300.0] and b["adduct"].tolist() == q["adduct"].tolist()
    assert list(q["ms2_mzs"].iloc[0]) == [100.0, 150.0]


def test_ranks_from_pools_dedupe_and_fallback():
    cfg = Config(top_k=3)
    rows = []
    # molecule x: decoy, tautomer-free duplicate ik14 rows of the same smiles, then the truth
    for ik, smi, sc, truth in (("D1", "CCO", 3.0, False), ("D2", "OCC", 2.5, False),
                               ("T", "CCN", 2.0, True), ("D3", "CCC", 1.0, False)):
        rows.append({"set": "S", "cell": "C2-NP", "molecule_id": "x", "ik14": ik, "smiles": smi,
                     "tier": "np", "in_np": True, "ppm_err": 0.0, "score": sc, "is_truth": truth})
    # molecule y: truth not stored (came from padding in the original run)
    rows.append({"set": "S", "cell": "C2-NP", "molecule_id": "y", "ik14": "Q", "smiles": "CCCC",
                 "tier": "np", "in_np": True, "ppm_err": 0.0, "score": 1.0, "is_truth": False})
    pools = pd.DataFrame(rows)
    fb = pd.DataFrame({"set": ["S", "S"], "cell": ["C2-NP", "C2-NP"], "molecule_id": ["x", "y"],
                       "rr": [0.5, 0.25]})
    r = V.ranks_from_pools(pools, cfg, exact=True, fallback=fb).set_index("molecule_id")
    assert r.loc["x", "rank"] == 2          # CCO and OCC are one metric key
    assert r.loc["y", "rr"] == 0.25          # padding fallback
    r2 = V.ranks_from_pools(pools, cfg, exact=False).set_index("molecule_id")
    assert r2.loc["x", "rank"] == 3 and r2.loc["y", "rr"] == 0.0
    r3 = V.ranks_from_pools(pools.assign(s2=-pools["score"]), cfg, score="s2").set_index("molecule_id")
    assert r3.loc["x", "rank"] == 2          # D3 first, then the truth
    cfg1 = Config(top_k=1)
    assert V.ranks_from_pools(pools, cfg1).set_index("molecule_id").loc["x", "rank"] == 0


def test_probe_mask():
    from casmi.config import Config as C
    from casmi.probes import probe_mask

    pool = pd.DataFrame({"tier": ["lib", "lib", "np", "pubchem", "gen", "lib"],
                         "in_np": [True, False, True, False, False, False],
                         "in_pubchem": [True, True, True, True, False, True],
                         "lib_best_cos": [0.9, 0.5, np.nan, np.nan, np.nan, 0.8],
                         "lib_n_matched": [6, 9, np.nan, np.nan, np.nan, 3]})
    assert probe_mask(pool, C(probe_rule="lib_hits")).tolist() == [True, False, False, False, False, False]
    assert probe_mask(pool, C(probe_rule="np_only")).tolist() == [True, False, True, False, False, False]
    assert probe_mask(pool, C(probe_rule="pubchem_only")).tolist() == [False, False, False, True, False, False]
    assert probe_mask(pool, C(probe_rule="novel")).tolist() == [False, False, False, False, True, False]
    assert probe_mask(pool, C()).all()
    with pytest.raises(ValueError):
        probe_mask(pool, C(probe_rule="nope"))


def test_class_mix_fit_recovers_shares(hand):
    cfg, attrs, t = hand
    cfg = cfg.replace(val_v2_board_se=0.001)
    # three probe "pipelines" with different cell profiles; boards from a known mix
    t1 = t.copy()
    t2 = t.copy()
    t2.loc[t2["cell"].isin(["C1", "C2-NP"]), "rr"] = 0.0
    t3 = t.copy()
    t3.loc[t3["cell"] != "C3h", "rr"] = 0.0
    t3.loc[t3["cell"] == "C3h", "rr"] = 0.5
    t4 = t.copy()
    t4.loc[t4["cell"] != "C1", "rr"] = 0.0
    tables = {"p1": t1, "p2": t2, "p3": t3, "p4": t4}
    pi = V.v2_mixes(cfg)["A"]
    meta_alpha = 0.0
    boards = {}
    for k, tab in tables.items():
        rep = V.proxy_report(tab, attrs, cfg, panel="full", meta={"alpha": {"A": meta_alpha}},
                             n_boot=0, mixes=["A"])
        boards[k] = rep["mixes"]["A"]["P"]
    probes = pd.DataFrame({"probe": list(tables), "tag": list(tables), "board": [boards[k] for k in tables],
                           "role": "share"})
    res = V.class_mix_fit(probes, tables, attrs, cfg, panel="full", n_boot=50, alpha0=meta_alpha)
    assert np.allclose(res["pi"], pi, atol=0.02)
    assert res["mix_consistent"]["A"]


# --------------------------------------------------------------------------------------
# synthetic train: v2 holdout and the harness end to end
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def v2(synthetic_train, tmp_path_factory):
    cfg = Config(n_jobs=1, db_tiers=("np",), batch_molecules=50, val_bootstrap_n=100,
                 val_v2_fast_sizes=(("NP-TIMS", 0), ("NP-OTHER", 12), ("SYN-TIMS", 10), ("C2-ZEN", 5)))
    st = synthetic_train["structures"]
    ho1 = synthetic_train["holdout"]
    ho2 = make_holdout_v2(ho1, st, cfg, alpha=0.5)
    d = tmp_path_factory.mktemp("v2")
    p = save_holdout_v2(ho2, d / "holdout_v2.csv", {"alpha": {"A": 0.5, "B": 0.4, "C": 0.6}})
    return {"cfg": cfg, "ho2": ho2, "path": p, "dir": d}


def test_holdout_v2_keeps_v1_and_adds_columns(synthetic_train, v2, tmp_path):
    ho1 = load_holdout(synthetic_train["holdout"])
    back = load_holdout_v2(v2["path"])
    assert list(back.columns) == list(HOLDOUT_V2_COLUMNS)
    # v1 rows and columns unchanged (byte-identical after the same formatting)
    from casmi.splits import save_holdout

    a = save_holdout(ho1, tmp_path / "a.csv").read_text()
    b = save_holdout(back[back["set"].isin(ho1["set"].unique())][list(HOLDOUT_COLUMNS)], tmp_path / "b.csv").read_text()
    assert a == b
    mols = V.v2_attributes(back)
    assert set(mols["c2_cell"]) <= {"NP", "PC", "none"}
    np_sets = mols["set"].isin(["NP-TIMS", "NP-OTHER"])
    assert (mols.loc[np_sets & mols["in_np"], "c2_cell"] == "NP").all()
    syn = mols[mols["set"] == "SYN-TIMS"]
    assert (syn.loc[syn["has_public_spectra"], "c2_cell"] == "none").all()
    assert (mols.loc[~np_sets, "w_obs"] == 1.0).all()
    w = w_obs_weights(mols.loc[np_sets, "pmid_count"], 0.5)
    assert np.allclose(mols.loc[np_sets, "w_obs"], w)
    # fast panel: sizes, inverse-probability weights sum to the population
    f = mols[mols["fast"]]
    assert (f["set"] == "NP-TIMS").sum() == (mols["set"] == "NP-TIMS").sum()
    no = mols[mols["set"] == "NP-OTHER"]
    assert (f["set"] == "NP-OTHER").sum() == min(12, len(no))
    assert f.loc[f["set"] == "NP-OTHER", "w_fast"].sum() == pytest.approx(len(no))
    assert (mols.loc[~mols["fast"], "w_fast"] == 0).all()
    # deterministic
    again = make_holdout_v2(ho1, synthetic_train["structures"], v2["cfg"], alpha=0.5)
    assert again.equals(v2["ho2"])


def test_zen_rows_and_training_exclusion(synthetic_train, tmp_path):
    zt = pd.DataFrame({"molecule_id": ["ZE00000"], "mkey": ["ZZZZZZZZZZZZZZ"], "ik14": ["ZZZZZZZZZZZZZZ"],
                       "ik14_group": ["ZZZZZZZZZZZZZZ"], "fold": [3], "mono_mass": [300.1],
                       "formula": ["C10H10O10"], "query_lib": ["ext_enveda180"], "in_np": [False],
                       "in_pubchem": [True], "pubmed_count": pd.array([pd.NA], dtype="Int32")})
    rows = zen_holdout_rows(zt)
    assert rows["set"].tolist() == ["C2-ZEN"] and bool(rows["primary"].iloc[0])
    ho2 = make_holdout_v2(synthetic_train["holdout"], synthetic_train["structures"], Config(), zen_truth=zt)
    z = ho2[ho2["set"] == "C2-ZEN"]
    assert len(z) == 1 and z["c2_cell"].iloc[0] == "PC" and bool(z["fast"].iloc[0])
    from casmi.splits import training_exclusion

    p = save_holdout_v2(ho2, tmp_path / "h.csv")
    ex = training_exclusion(p, synthetic_train["structures"])
    assert "ZZZZZZZZZZZZZZ" in ex.ik14


def test_evaluate_v2_end_to_end(synthetic_train, v2, monkeypatch):
    cfg = v2["cfg"]
    s = synthetic_train
    # C3h neighbours: the synthetic train structures and a small slice of the NP tier
    ik_np, fps_np = V._np_tier_fps(None)
    monkeypatch.setattr(V, "_np_tier_fps", lambda db: (ik_np[:5000], np.asarray(fps_np[:5000])))
    kw = dict(holdout=v2["path"], val_dir=s["val_dir"], library=s["library"], train_dir=s["out"],
              db=s["db"], n_jobs=1)
    sc = V.self_check_v2(cfg, panel="full", **kw)
    assert any(k.startswith("truth|") and k.endswith("|C3h") for k in sc)
    store = V.PoolStore(keep=50)
    tab, cells = V.evaluate_v2(V.pool_capture_predictor(store), cfg, panel="full", store=store, **kw)
    assert set(tab.columns) == set(V.RESULT_V2_COLUMNS)
    assert set(tab["cell"]) == {"C1", "C2-NP", "C2-PC", "C3h", "C3e"}
    c3 = tab[tab["cell"].isin(["C3h", "C3e"])]
    assert c3["structural"].all() and (c3["rr"] == 0).all()
    run = tab[~tab["structural"]]
    assert ((run["rank"] > 0) == (run["rr"] > 0)).all()
    # Baseline 0 does not use spectra: C1 equals C2-NP on molecules in both
    p = run.pivot_table(index="mkey", columns="cell", values="rr")
    both = p.dropna(subset=["C1", "C2-NP"])
    assert len(both) and np.allclose(both["C1"], both["C2-NP"])
    attrs = V.v2_attributes(load_holdout_v2(v2["path"]))
    rep = V.proxy_report(tab, attrs, cfg, panel="full", meta={"alpha": {"A": 0.5, "B": 0.4, "C": 0.6}})
    assert set(rep["mixes"]) == {"A", "B", "C"} and 0 <= rep["mixes"]["A"]["P"] <= 1
    assert "P(A)" in V.format_proxy(rep)
    # stored pools reproduce the pipeline's ranks exactly with the stored score ...
    pools_dir = v2["dir"] / "pools"
    store.save(pools_dir)
    pools = V.load_pools(pools_dir)
    assert pools["is_truth"].any()
    ranks = V.ranks_from_pools(pools, cfg, fallback=tab)
    resc = V.rescore_table(tab, ranks)
    m = resc.merge(tab, on=["set", "cell", "molecule_id"], suffixes=("", "_orig"))
    assert np.allclose(m["rr"], m["rr_orig"])
    # ... and with fusion.fuse recomputed from the stored features
    r2 = V.rescore_table(tab, V.ranks_from_pools(pools, cfg, V.fusion_scorer(), fallback=tab))
    m2 = r2.merge(tab, on=["set", "cell", "molecule_id"], suffixes=("", "_orig"))
    assert np.allclose(m2["rr"], m2["rr_orig"])
    # blind audit: same pools, prior-only pipeline gives the same ranks
    tb, _ = V.evaluate_v2(V.pipeline_predictor, cfg, panel="fast", blind=True, cells=("C2-NP",), **kw)
    mb = tb.merge(tab, on=["set", "cell", "molecule_id"], suffixes=("", "_real"))
    assert len(mb) and np.allclose(mb["rr"], mb["rr_real"])


def test_probe_stage_cuts_padding(synthetic_train):
    from casmi import probes  # noqa: F401  (registers probe_assemble)

    q, _ = V.load_queries("NP-TIMS", synthetic_train["val_dir"])
    q = q[q["molecule_id"].isin(sorted(set(q["molecule_id"]))[:5])]
    cfg = Config(n_jobs=1, db_tiers=("np",), pipeline_plugins=("casmi.probes",),
                 pipeline_stages=("mass", "pool", "features", "score", "probe_assemble"),
                 probe_rule="pubchem_only")
    preds = pipeline.predict(q, cfg, db=synthetic_train["db"])
    assert all(v == [pipeline.LAST_RESORT_SMILES] for v in preds.values())
    cfg2 = cfg.replace(probe_rule="np_only")
    preds2 = pipeline.predict(q, cfg2, db=synthetic_train["db"])
    base = pipeline.predict(q, cfg.replace(probe_rule="", pipeline_stages=("mass", "pool", "features",
                                                                         "score", "assemble")),
                            db=synthetic_train["db"])
    for m in preds2:
        assert preds2[m] == base[m][:len(preds2[m])]

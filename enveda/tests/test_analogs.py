"""Tests for casmi.analogs (ROADMAP R6): deltas, hits, exclusion masks, features, stage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from casmi import analogs as A
from casmi import chem, metric, pipeline
from casmi.config import Config
from casmi.library import Library

PROTON = 1.007276
QUERCETIN = "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12"
MORIN = "O=c1c(O)c(-c2ccc(O)cc2O)oc2cc(O)cc(O)c12"
TRICETIN = "O=c1cc(-c2cc(O)c(O)c(O)c2)oc2cc(O)cc(O)c12"
ISORHAMNETIN = "COc1cc(-c2oc3cc(O)cc(O)c3c(=O)c2O)ccc1O"
Q3G = "OCC1OC(Oc2c(-c3ccc(O)c(O)c3)oc3cc(O)cc(O)c3c2=O)C(O)C(O)C1O"
KAEMPFEROL = "O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12"
CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
THEOBROMINE = "Cn1cnc2c1c(=O)[nH]c(=O)n2C"
LAURIC = "CCCCCCCCCCCC(=O)O"

FLAV_CORE = [69.0335, 109.0284, 121.0284, 137.0233, 153.0182]
FLAV_LOSS = [18.0106, 46.0055, 64.0160, 92.0109, 120.0059]
ALK_CORE = [42.0338, 69.0447, 110.0349, 138.0298]
ALK_LOSS = [57.0215, 85.0164]


def _spectrum(smiles: str, core, losses, extra: float, scale: float = 1.0):
    prec = chem.mono_mass(smiles) + PROTON
    mz = list(core) + [prec - x for x in losses] + [extra]
    it = [1.0, 0.8, 0.6, 0.5, 0.4, 0.9, 0.7, 0.3, 0.35, 0.25, 0.2, 0.15][:len(mz)]
    it = [min(1.0, v * scale) for v in it]
    o = np.argsort(mz)
    return np.asarray(mz)[o], np.asarray(it)[o], prec


def _lib_rows():
    spec = [  # smiles, family, extra peak, library, copies
        (QUERCETIN, "flav", 181.0, "gnps", 2), (QUERCETIN, "flav", 181.0, "massbank", 1),
        (ISORHAMNETIN, "flav", 195.0, "gnps", 1), (Q3G, "flav", 211.0, "gnps", 1),
        (KAEMPFEROL, "flav", 165.0, "massbank", 1),
        (CAFFEINE, "alk", 99.0, "gnps", 1), (THEOBROMINE, "alk", 97.0, "gnps", 1),
        (LAURIC, "lip", 0.0, "gnps", 1),
    ]
    rows = []
    for smi, fam, extra, lib, copies in spec:
        for c in range(copies):
            if fam == "flav":
                mz, it, prec = _spectrum(smi, FLAV_CORE, FLAV_LOSS, extra, 1.0 - 0.05 * c)
            elif fam == "alk":
                mz, it, prec = _spectrum(smi, ALK_CORE, ALK_LOSS, extra)
            else:
                prec = chem.mono_mass(smi) + PROTON
                mz, it = np.array([57.07, 71.085, 85.101, 99.117]), np.array([1.0, 0.7, 0.5, 0.3])
            rows.append({"ms2_mzs": mz, "ms2_normalized_intensities": it, "precursor_mz": prec,
                         "inchikey14": metric.ik14(smi), "normalized_smiles": smi,
                         "spectrum_id": f"{lib}_{metric.ik14(smi)}_{c}", "adduct": "[M+H]+",
                         "ionization_mode": "positive", "ingest_lib": lib,
                         "instrument_type": "qTof", "mono_mass": chem.mono_mass(smi),
                         "molecular_formula": chem.formula(smi)})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    out = tmp_path_factory.mktemp("r6lib") / "library"
    return Library.build(_lib_rows(), Config(), out_dir=out, batch_rows=100)


def _query(smiles=QUERCETIN, mid="q1"):
    mz, it, prec = _spectrum(smiles, FLAV_CORE, FLAV_LOSS, 181.0, 0.97)
    return pd.DataFrame({"molecule_id": [mid], "spectrum_id": [f"{mid}_s1"], "ms2_mzs": [mz],
                         "ms2_normalized_intensities": [it], "precursor_mz": [prec],
                         "adduct": ["[M+H]+"], "ionization_mode": ["positive"]})


def _pool(mid="q1"):
    rows = []
    for i, smi in enumerate((QUERCETIN, MORIN, TRICETIN)):
        rows.append({"molecule_id": mid, "ik14": metric.ik14(smi), "smiles": smi,
                     "formula": chem.formula(smi), "mono_mass": chem.mono_mass(smi),
                     "tier": "np", "score": 1.0 - 0.1 * i})
    return pd.DataFrame(rows)


def _cfg(**kw):
    base = dict(analog_feat_min_matched=1, analog_feat_struct_top_k=0, analog_feat_top_k=10)
    base.update(kw)
    return Config().replace(**base)


def _hooks(lib_libs=None, condition="C2"):
    from casmi.validation import MoleculeExclusion, ValidationHooks

    ik = metric.ik14(QUERCETIN)
    e = MoleculeExclusion(molecule_id="q1", mkey=ik, group=frozenset([ik]), lib_libs=lib_libs,
                          db_ik14=frozenset(), truth=frozenset([ik]))
    return ValidationHooks(condition, {"q1": e})


# --- deltas and kernels ----------------------------------------------------------------

def test_delta_table_and_matching():
    t = A.delta_table()
    assert (t["family"] == "0").sum() == 1
    assert t["mass"].is_monotonic_increasing
    fam = t["family"].to_numpy(object)
    codes = A.match_delta(np.array([14.01565, -162.05282, 0.0003, 7.0, np.nan]), 0.005)
    assert fam[codes[0]] == "CH2" and t["sign"].iat[codes[0]] == 1
    assert fam[codes[1]] == "C6H10O5" and t["sign"].iat[codes[1]] == -1
    assert fam[codes[2]] == "0"
    assert codes[3] == -1 and codes[4] == -1
    # CO (27.9949) and C2H4 (28.0313) are told apart
    assert fam[A.match_delta(27.9949, 0.01)[0]] == "CO"
    assert fam[A.match_delta(28.0313, 0.01)[0]] == "C2H4"


def test_formula_delta_codes():
    fam = A.delta_table()["family"].to_numpy(object)
    c = A.formula_delta_codes(["C15H10O7", "C16H12O7", None], ["C16H12O7", "C15H10O7", "C9H8O4"])
    assert fam[c[0, 0]] == "CH2" and fam[c[0, 1]] == "0" and c[0, 2] == -1
    assert fam[c[1, 1]] == "CH2"
    assert (c[2] == -1).all()


def test_tanimoto_matrix_matches_chem():
    rng = np.random.default_rng(0)
    a = (rng.random((7, 512)) < 0.05).astype(np.uint8) * rng.integers(1, 255, (7, 512)).astype(np.uint8)
    b = (rng.random((5, 512)) < 0.05).astype(np.uint8) * rng.integers(1, 255, (5, 512)).astype(np.uint8)
    b[4] = 0
    T = A.tanimoto_matrix(a, b)
    for j in range(5):
        np.testing.assert_allclose(T[:, j], chem.tanimoto(b[j], a), rtol=1e-6, atol=1e-7)
    assert (T[:, 4] == 0).all()
    assert A.tanimoto_matrix(a, b[:0]).shape == (7, 0)


def test_structure_consistency():
    q, mo, tr, iso, g, k = (A.struct_info(s) for s in (QUERCETIN, MORIN, TRICETIN, ISORHAMNETIN,
                                                       Q3G, KAEMPFEROL))
    code = A.formula_delta_codes([chem.formula(QUERCETIN)],
                                 [chem.formula(s) for s in (ISORHAMNETIN, Q3G, KAEMPFEROL)])[0]
    assert (code >= 0).all()
    assert A._consistent(q, iso, int(code[0]))        # quercetin = isorhamnetin - CH2
    assert A._consistent(q, g, int(code[1]))          # quercetin = glucoside - hexose
    assert A._consistent(q, k, int(code[2]))          # quercetin = kaempferol + O
    assert not A._consistent(mo, iso, int(code[0]))   # morin: wrong ring positions
    assert not A._consistent(tr, g, int(code[1]))
    assert A._consistent(mo, k, int(code[2]))         # kaempferol is inside morin too
    assert not A._consistent(q, iso, -1)
    assert q.scaffold and q.scaffold == mo.scaffold == tr.scaffold == k.scaffold
    assert g.scaffold != q.scaffold                   # the sugar ring is part of its scaffold
    assert A.struct_info("not a smiles").graph is None


# --- hits and the exclusion-mask guarantee ---------------------------------------------

def _hits(lib, cfg, mask=None):
    from casmi.library import prepare_queries

    q = _query()
    M = chem.mono_mass(QUERCETIN)
    qs = prepare_queries(q, lib.cfg, M=M)
    return A.search_hits(lib, qs, cfg, M=M, exclude_mask=mask)


def test_hits_without_mask_find_the_structure_itself(lib):
    cfg = _cfg()
    h = A.select_hits(_hits(lib, cfg), cfg)
    assert list(h.columns) == list(A.HIT_COLUMNS)
    assert h["ik14"].iat[0] == metric.ik14(QUERCETIN)
    assert h["sim"].iat[0] > 0.95 and h["delta"].iat[0] == "0"
    assert h["hit_rank"].tolist() == list(range(1, len(h) + 1))
    flav = {metric.ik14(s) for s in (ISORHAMNETIN, Q3G, KAEMPFEROL)}
    assert flav <= set(h["ik14"])
    fam = dict(zip(h["ik14"], h["delta"]))
    assert fam[metric.ik14(ISORHAMNETIN)] == "CH2"
    assert fam[metric.ik14(Q3G)] == "C6H10O5"
    assert fam[metric.ik14(KAEMPFEROL)] == "O"
    assert np.allclose(h["w"], h["sim"] ** cfg.analog_feat_power)


def test_exclusion_mask_hides_own_spectra(lib):
    """A class 2/3 query never sees its own spectra (C2: every library); C1 hides only the
    query library."""
    cfg = _cfg()
    ik = metric.ik14(QUERCETIN)
    m2 = _hooks().library_mask(lib, "q1")
    assert m2 is not None and m2.sum() == 3
    h2 = A.select_hits(_hits(lib, cfg, m2), cfg)
    assert ik not in set(h2["ik14"]) and len(h2)
    assert h2["ik14"].iat[0] != ik
    # every consensus row of the structure is gone, not only the spectra
    cons = set(np.flatnonzero(lib.cons_struct == int(np.flatnonzero(lib.struct_ik14 == ik)[0])))
    assert not cons & set(_hits(lib, cfg, m2)["cons_idx"])
    m1 = _hooks(lib_libs=frozenset(["gnps"]), condition="C1").library_mask(lib, "q1")
    assert m1.sum() == 2
    h1 = A.select_hits(_hits(lib, cfg, m1), cfg)
    assert h1["ik14"].iat[0] == ik                    # massbank spectrum still visible


def test_score_pool_with_hooks_and_guard(lib):
    from casmi.validation import library_guard

    cfg = _cfg()
    ik = metric.ik14(QUERCETIN)
    for guard in (False, True):
        hooks = _hooks()
        hooks.active = ("q1",)
        if guard:
            with library_guard(hooks):
                feats, mol, hits = A.score_pool(_pool(), _query(), lib, cfg, hooks=hooks)
            assert not hooks.violations
        else:
            feats, mol, hits = A.score_pool(_pool(), _query(), lib, cfg, hooks=hooks)
        assert ik not in set(hits["q1"]["ik14"])
        assert mol["an_best_ik14"].iat[0] != ik
        assert feats["an_tan_max"].max() < 1.0      # no Tanimoto-1 neighbour left
    feats_open, mol_open, _ = A.score_pool(_pool(), _query(), lib, cfg)
    assert mol_open["an_best_ik14"].iat[0] == ik
    assert feats_open["an_tan_best"].iat[0] == pytest.approx(1.0)


# --- features --------------------------------------------------------------------------

def test_features_rank_the_truth_first(lib):
    cfg = _cfg()
    pool = _pool()
    feats, mol, hits = A.score_pool(pool, _query(), lib, cfg, hooks=_hooks())
    assert list(feats.columns) == list(A.FEATURES)
    f = feats.reset_index(drop=True)
    # quercetin is the hit structures minus or plus one edit; morin and tricetin are not
    assert f.loc[0, "an_sub_n"] == 3 and f.loc[1, "an_sub_n"] == 1 and f.loc[2, "an_sub_n"] == 0
    assert f.loc[0, "an_sub_best"] == 1.0
    assert f.loc[0, "an_tan_wmean"] > max(f.loc[1, "an_tan_wmean"], f.loc[2, "an_tan_wmean"])
    assert f.loc[0, "an_tan_wmean_rank"] == 1 and f.loc[0, "an_sub_w_rank"] == 1
    assert f.loc[0, "an_tan_wmax_gap"] == 0.0
    # every candidate shares the flavone scaffold with isorhamnetin and kaempferol (the
    # glucoside's scaffold includes its sugar ring)
    assert (f["an_scaf_n"] == 2).all()
    # formula level: all three are C15H10O7, so the formula features cannot separate them
    assert f["an_edit_n"].nunique() == 1 and f.loc[0, "an_edit_n"] == 3
    # molecule context
    assert f["an_n_hits"].nunique() == 1 and f.loc[0, "an_best_sim"] == mol["an_best_sim"].iat[0]
    assert np.isclose(f["an_tan_wsum"] / f["an_tan_wmean"], f.loc[0, "an_w_total"]).all()
    assert mol["an_n_hits"].iat[0] == len(hits["q1"])


def test_struct_top_k_and_selection_settings(lib):
    cfg = _cfg(analog_feat_struct_top_k=1)
    pool = _pool()
    feats, mol, _ = A.score_pool(pool, _query(), lib, cfg, hooks=_hooks())
    assert feats["an_sub_n"].notna().tolist() == [True, False, False]   # best score only
    assert feats["an_tan_max"].notna().all()
    assert mol["an_n_scored_struct"].iat[0] == 1
    few = _cfg(analog_feat_top_k=2)
    _, m2, h2 = A.score_pool(pool, _query(), lib, few, hooks=_hooks())
    assert len(h2["q1"]) == 2
    strict = _cfg(analog_feat_min_matched=50)          # more peaks than any spectrum has
    f3, m3, h3 = A.score_pool(pool, _query(), lib, strict, hooks=_hooks())
    assert len(h3["q1"]) == 0 and f3["an_tan_max"].isna().all() and (f3["an_n_hits"] == 0).all()
    near = _cfg(analog_feat_max_dm_da=20.0)
    _, _, h4 = A.score_pool(pool, _query(), lib, near, hooks=_hooks())
    assert (h4["q1"]["dm"].abs() <= 20.0).all() and metric.ik14(Q3G) not in set(h4["q1"]["ik14"])


def test_deterministic_and_fp_cache(lib, tmp_path):
    cfg = _cfg()
    a, _, ha = A.score_pool(_pool(), _query(), lib, cfg, hooks=_hooks())
    b, _, hb = A.score_pool(_pool(), _query(), lib, cfg, hooks=_hooks())
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(ha["q1"], hb["q1"])
    # a precomputed fingerprint file gives the same features as computing on the fly
    ik = list(lib.struct_ik14)
    fps, _ = chem.morgan_fps(list(lib.struct_smiles))
    np.save(tmp_path / "fp.npy", fps)
    pd.DataFrame({"ik14": ik}).to_parquet(tmp_path / "fp_ik14.parquet")
    c2 = cfg.replace(analog_feat_fp_path=str(tmp_path / "fp"))
    sc = A.StructCache(c2)
    assert len(sc._pre_row) == lib.n_structures
    c, _, _ = A.score_pool(_pool(), _query(), lib, c2, hooks=_hooks())
    pd.testing.assert_frame_equal(a, c)


def test_no_library_and_empty_pool():
    cfg = _cfg()
    feats, mol, hits = A.score_pool(_pool(), _query(), None, cfg)
    assert feats["an_tan_max"].isna().all() and (feats["an_n_hits"] == 0).all()
    assert hits["q1"].empty
    st = pipeline.State(test_df=_query())
    st.pool = _pool().iloc[:0]
    assert A.stage_analogs(st, cfg).pool.empty


# --- stage through pipeline.run --------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_db(tmp_path_factory):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from casmi.db import CandidateDB

    root = tmp_path_factory.mktemp("r6db")
    rows = []
    for smi, pmid in ((QUERCETIN, 10), (MORIN, 1000), (TRICETIN, 100)):
        rows.append(dict(ik14=metric.ik14(smi), smiles=smi, formula=chem.formula(smi),
                         mono_mass=chem.mono_mass(smi), in_coconut=True, in_npatlas=False,
                         in_lotus=False, in_pubchemlite=False, coconut_annotation_level=None,
                         n_organisms=None, n_citations=None, n_collections=1,
                         pubmed_count=pmid, patent_count=None, anno_type_count=None, cid=None))
    df = pd.DataFrame(rows).sort_values(["mono_mass", "ik14"], kind="mergesort")
    for c in ("n_organisms", "n_citations", "n_collections", "pubmed_count", "patent_count",
              "anno_type_count"):
        df[c] = df[c].astype("Int32")
    df["coconut_annotation_level"] = df["coconut_annotation_level"].astype("Int8")
    df["cid"] = df["cid"].astype("Int64")
    p = root / "np.parquet"
    pq.write_table(pa.Table.from_pandas(df.reset_index(drop=True), preserve_index=False), p)
    return CandidateDB(np_path=p, pubchem_dir=root / "no_pubchem")


def test_pipeline_run_with_analogs_stage(tiny_db, lib):
    cfg = Config().replace(db_tiers=("np",), n_jobs=1, mkey_cache=False, mkey_timeout_s=0.0,
                           pipeline_plugins=("casmi.analogs",),
                           pipeline_stages=("mass", "pool", "features", "score", "analogs",
                                            "assemble"),
                           analog_feat_min_matched=1)
    seen = {}

    def spy(state, c):
        seen["pool"] = state.pool.copy()
        seen["hits"] = dict(state.extras["analogs"]["hits"])
        return pipeline.stage_assemble(state, c)

    hooks = _hooks()
    hooks.active = ("q1",)
    res = pipeline.run(_query(), cfg, db=tiny_db, stages={"assemble": spy},
                       extras={"library": lib, "validation": hooks})
    pool = seen["pool"]
    assert len(pool) == 3 and pool["an_tan_wmean"].notna().all()
    assert pool["an_sub_n"].notna().all()               # top 100 covers the 3 rows
    best = pool.loc[pool["an_sub_w"].idxmax(), "smiles"]
    assert best == QUERCETIN
    assert metric.ik14(QUERCETIN) not in set(seen["hits"]["q1"]["ik14"])
    for c in A.MOLECULE_COLUMNS:
        assert c in res.molecules.columns
    assert res.molecules["an_n_hits"].iat[0] >= 3
    assert len(res.preds["q1"]) >= 1


_BLOCKED = r"""
import importlib.abc, sys
BLOCKED = ("torch", "polars", "duckdb", "sklearn")
class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError("blocked on the inference path: " + name)
        return None
sys.meta_path.insert(0, _Blocker())
import casmi.analogs as A
A.warmup()
A.delta_table()
A.struct_info("CCO")
bad = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
assert not bad, bad
from casmi import pipeline
assert "analogs" in pipeline.STAGES
print("ok")
"""


def test_inference_path_imports():
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-c", _BLOCKED], capture_output=True, text=True,
                         timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.strip().endswith("ok")

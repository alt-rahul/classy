"""Tests for casmi.va (ROADMAP R13): keys, one-edit products, the shard build and merge, the
VA tier in db lookups, and the reach evaluation, on small synthetic fixtures."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from casmi import db as cdb
from casmi import va
from casmi.config import Config
from casmi.metric import ik14

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
SALICYLIC = "O=C(O)c1ccccc1O"
ARBUTIN = "OCC1OC(Oc2ccc(O)cc2)C(O)C(O)C1O"


def test_ik14_keys_and_membership():
    k = va.ik14_keys(["AAAAAAAAAAAAAA", "BSYNRYMUTXBXSQ", None, "abc", "BSYNRYMUTXBXSq"])
    assert k[0] == 0 and k[2] == va.BAD_KEY and k[3] == va.BAD_KEY and k[4] == va.BAD_KEY
    sliced = va.ik14_keys(pa.array(["ZZZZZZZZZZZZZZ", "BSYNRYMUTXBXSQ"]).slice(1))
    assert sliced[0] == k[1]
    ref = va.sorted_keys([k, k])
    assert va.BAD_KEY not in ref and len(ref) == 2
    assert va.in_sorted(k, ref).tolist() == [True, True, False, False, False]
    assert va.in_sorted(np.zeros(0, np.uint64), ref).shape == (0,)


def test_keys_from_parquet(tmp_path):
    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    pq.write_table(pa.table({"ik14": ["CCCCCCCCCCCCCC", "AAAAAAAAAAAAAA"]}), a)
    pq.write_table(pa.table({"ik14": ["AAAAAAAAAAAAAA", None]}), b)
    keys = va.keys_from_parquet([a, b])
    assert len(keys) == 2 and np.all(np.diff(keys.astype(np.float64)) > 0)


def test_seed_products_one_edit():
    prods, per_rule, cut = va.seed_products(ASPIRIN)
    assert not cut and prods and per_rule
    iks = [p[0] for p in prods]
    assert iks == sorted(iks) and len(set(iks)) == len(iks)
    assert ik14(SALICYLIC) in iks          # deacetylation
    assert ik14(ASPIRIN) not in iks        # the seed itself is never a product
    cfg = Config()
    assert all(cfg.va_mass_min <= p[3] <= cfg.va_mass_max for p in prods)
    assert va.seed_products("not a smiles") == ([], {}, False)
    assert va.seed_products("[Na+].[Cl-]")[0] == []


def test_seed_table_ids_and_priority():
    np_df = pd.DataFrame({"ik14": ["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"], "smiles": ["CCO", "CCN"],
                          "formula": ["C2H6O", "C2H7N"], "mono_mass": [46.0, 45.0],
                          "sid_count": [5, 50]})
    train = pd.DataFrame({"ik14": ["AAAAAAAAAAAAAA", "CCCCCCCCCCCCCC", "DDDDDDDDDDDDDD"],
                          "smiles": ["CCO", "CCC", "CCCC"], "formula": ["C2H6O", "C3H8", "C4H10"],
                          "mono_mass": [46.0, 44.0, 58.0], "libs": ["gnps", "gnps;enveda-180", "enveda-180"],
                          "n_spectra": [3, 2, 9], "rdkit_ok": True, "np_score": 0.1, "pmid_count": 0})
    s = va.seed_table(np_df, train)
    assert list(s.columns) == list(va.SEED_COLUMNS)
    assert s["seed_id"].tolist() == [0, 1, 2]           # DDDD only in enveda-180: not a seed
    assert s["source"].tolist() == ["np", "np", "train"]
    assert s.loc[0, "in_train"] and s.loc[0, "n_spectra"] == 3
    assert s["pri"].tolist() == [0, 2, 1]               # spectra first, then SID count


def _key_files(tmp_path, pubchem=(), npk=(), train=()):
    out = {}
    for name, vals in (("pubchem", pubchem), ("np", npk), ("train", train)):
        p = tmp_path / f"{name}.npy"
        np.save(p, va.sorted_keys(va.ik14_keys(list(vals))))
        out[name] = p
    return out


def _seeds(smiles, start=0):
    return pd.DataFrame({"seed_id": np.arange(start, start + len(smiles), dtype=np.int32),
                         "ik14": [ik14(s) for s in smiles], "smiles": smiles,
                         "pri": np.arange(start, start + len(smiles), dtype=np.int32)})


def test_build_merge_and_hits(tmp_path):
    from casmi.validation import skeleton_key

    prods = {p[0] for p in va.seed_products(ASPIRIN)[0]}
    known_pc = sorted(prods)[:3]
    sal = ik14(SALICYLIC)
    kf = _key_files(tmp_path, pubchem=known_pc, npk=[sal], train=[sal])
    hold = va.holdout_lookup(pd.DataFrame({"ik14": [sal], "mkey": [sal]}),
                             pd.DataFrame({"mkey": [sal], "formula": ["C7H6O3"],
                                           "skeleton": [skeleton_key(SALICYLIC)]}))
    i0 = va.build_products(_seeds([ASPIRIN, "CCCCCC"]), kf, hold, tmp_path / "p0", n_jobs=1, chunk=1)
    i1 = va.build_products(_seeds([ASPIRIN.replace("CC(=O)O", "CC(=O)O", 1), "OCCCCO"], start=10)
                           .assign(smiles=["O=C(O)c1ccccc1OC(C)=O", "OCCCCO"]),
                           kf, hold, tmp_path / "p1", n_jobs=1, chunk=1)
    assert i0["n_seeds_done"] == 2 and i1["n_seeds_done"] == 2
    hits = pq.read_table(tmp_path / "p0" / "holdout_hits.parquet").to_pandas()
    assert list(hits.columns) == list(va.HIT_COLUMNS)
    assert (hits["ik14"] == sal).any() and (hits.loc[hits["ik14"] == sal, "seed_id"] == 0).all()
    t0 = pa.concat_tables([pq.read_table(p) for p in sorted((tmp_path / "p0").glob("mass_*.parquet"))])
    got = set(t0.column("ik14").to_pylist())
    assert not got & set(known_pc) and sal not in got and got
    assert np.all(t0.column("n_seeds").to_numpy() >= 1)
    res = va.merge_parts([tmp_path / "p0", tmp_path / "p1"], tmp_path / "va")
    m = pa.concat_tables([pq.read_table(p) for p in sorted((tmp_path / "va").glob("mass_*.parquet"))]).to_pandas()
    assert list(m.columns) == list(va.VA_COLUMNS)
    assert m["ik14"].is_unique and res["n_rows"] == len(m)
    # aspirin is a seed in both shards (same ik14 under two seed ids): n_seeds adds up
    shared = got & set(m["ik14"])
    assert shared and (m.loc[m["ik14"].isin(shared), "n_seeds"] >= 2).any()
    # provenance from the seed with the lowest priority value
    assert (m.loc[m["ik14"].isin(shared) & (m["n_seeds"] == 2), "seed_id"] == 0).all()
    mass = m["mono_mass"].to_numpy()
    assert np.all(np.diff(mass) >= 0)
    # filtered merge
    res2 = va.merge_parts([tmp_path / "p0", tmp_path / "p1"], tmp_path / "va2",
                          row_filter=lambda t: t.column("n_seeds").to_numpy() >= 2)
    assert 0 < res2["n_rows"] < res["n_rows"]


def _np_tier(d):
    d.mkdir(parents=True, exist_ok=True)
    rows = [("NPAAAAAAAAAAAA", "CCO", "C2H6O", 180.0001), ("NPBBBBBBBBBBBB", "CCN", "C2H7N", 180.0600),
            ("VADUPAAAAAAAAA", "CCC", "C3H8", 180.0603)]
    df = pd.DataFrame(rows, columns=["ik14", "smiles", "formula", "mono_mass"])
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), d / "np.parquet")
    np.save(d / "np_fp.npy", np.zeros((len(df), 512), np.uint8))
    return d / "np.parquet"


def _va_tier(d):
    d.mkdir(parents=True, exist_ok=True)
    rows = [("VAAAAAAAAAAAAA", "CCOC", "C3H8O", 180.0599, 1, 0, 3, 4),
            ("VADUPAAAAAAAAA", "CCC", "C3H8", 180.0603, 0, 4, 1, 1),
            ("VABBBBBBBBBBBB", "CCCO", "C3H8O", 185.0, 2, 1, 1, 1)]
    t = pa.table({c: [r[i] for r in rows] for i, c in enumerate(va.VA_COLUMNS)}, schema=va._VA_SCHEMA)
    va.write_shards(t, d)
    seeds = pd.DataFrame({"seed_id": np.arange(3, dtype=np.int32), "ik14": ["NPAAAAAAAAAAAA", "NPBBBBBBBBBBBB", "TRAAAAAAAAAAAA"],
                          "smiles": ["CCO", "CCN", "CC"], "formula": ["C2H6O", "C2H7N", "C2H6"],
                          "mono_mass": [180.0001, 180.06, 30.0], "source": ["np", "np", "train"],
                          "np_row": np.array([0, 1, -1], np.int32), "in_train": [False, True, True],
                          "n_spectra": np.array([0, 4, 2], np.int32), "sid_count": np.array([7, -1, -1], np.int32),
                          "pubmed_count": np.array([-1, 3, 0], np.int32),
                          "np_likeness": np.array([1.0, np.nan, 0.2], np.float32), "pri": np.array([2, 0, 1], np.int32)})
    seeds.to_parquet(d / "seeds.parquet", index=False)
    (d / "_index.json").write_text(json.dumps({"rule_stats": {"n_products": [10, 10, 0, 0, 5],
                                                              "n_np": [5, 1, 0, 0, 0]}}))
    return d


def test_db_lookup_with_va_tier(tmp_path):
    npp = _np_tier(tmp_path / "np")
    vd = _va_tier(tmp_path / "va")
    d = cdb.CandidateDB(np_path=npp, pubchem_dir=tmp_path / "nopc")
    d.va_path = vd
    before = d.lookup(180.06, tol_ppm=10)
    assert list(before.columns) == list(cdb.OUT_COLUMNS)
    both = d.lookup(180.06, tol_ppm=10, tiers=("np", "va"))
    assert list(both.columns) == [*cdb.OUT_COLUMNS, *va.LOOKUP_COLUMNS]
    assert sorted(both["tier"]) == ["np", "np", "va"]  # 180.0599 (va), 180.0600 and 180.0603 (np)
    assert np.all(np.diff(both["mono_mass"].to_numpy()) >= 0)
    vrow = both[both["tier"] == "va"].iloc[0]
    assert vrow["ik14"] == "VAAAAAAAAAAAAA" and vrow["np_row"] == -1 and not vrow["in_np"]
    assert vrow["va_n_seeds"] == 3 and vrow["va_seed_ik14"] == "NPBBBBBBBBBBBB"
    assert vrow["va_seed_tier"] == "np" and vrow["va_seed_n_spectra"] == 4
    assert pd.isna(vrow["va_seed_sid_count"]) and vrow["va_seed_pubmed_count"] == 3
    assert vrow["va_rule_np_rate"] == pytest.approx(0.5)
    assert vrow["va_rule_name"] == "O_methylation"
    # the VA duplicate of an NP row is dropped; NP rows have NA provenance
    assert (both["ik14"] == "VADUPAAAAAAAAA").sum() == 1
    assert both.loc[both["tier"] == "np", "va_seed_id"].isna().all()
    # the default and ("np", "pubchem") lookups are unchanged by the presence of the VA tier
    pd.testing.assert_frame_equal(before, d.lookup(180.06, tol_ppm=10, tiers=("np", "pubchem")))
    only = d.lookup_batch([180.06, 185.0, 500.0], tol_ppm=10, tiers=("va",), keys=["a", "b", "c"])
    assert only["query"].tolist() == ["a", "a", "b"] and set(only["tier"]) == {"va"}  # NP not queried: no dedupe
    assert np.allclose(only["ppm_err"], (only["mono_mass"] - only["query_mass"]) / only["query_mass"] * 1e6)
    d.release()
    d.va_path = tmp_path / "missing"
    empty = d.lookup(180.06, tol_ppm=10, tiers=("np", "va"))
    assert set(empty["tier"]) == {"np"} and "va_seed_id" in empty.columns


def test_tiers_validation_still_strict():
    with pytest.raises(ValueError):
        cdb._norm_tiers(("np", "nope"), ("np",))
    assert cdb._norm_tiers(None, ("np", "pubchem")) == ("np", "pubchem")
    assert "va" in cdb.TIERS


def test_reach_table_masks_seeds():
    seeds = pd.DataFrame({"seed_id": [0, 1, 2], "ik14": ["SAAAAAAAAAAAAA", "SBBBBBBBBBBBBB", "SCCCCCCCCCCCCC"],
                          "pri": [2, 1, 0], "source": "np"})
    hits = pd.DataFrame({"ik14": ["TAAAAAAAAAAAAA", "TAAAAAAAAAAAAA", "TBBBBBBBBBBBBB", "XXXXXXXXXXXXXX"],
                         "mkey": ["TAAAAAAAAAAAAA", "TAAAAAAAAAAAAA", "TBBBBBBBBBBBBB", "TCCCCCCCCCCCCC"],
                         "match": ["ik14", "ik14", "ik14", "mkey"], "seed_id": [0, 1, 2, 0],
                         "rule": [3, 5, 1, 2], "smiles": ["C", "C", "C", "C"]})
    mols = pd.DataFrame({"molecule_id": ["m1", "m2", "m3"], "mkey": ["TAAAAAAAAAAAAA", "TBBBBBBBBBBBBB", "TCCCCCCCCCCCCC"],
                         "ik14": ["TAAAAAAAAAAAAA", "TBBBBBBBBBBBBB", "TCCCCCCCCCCCCC"],
                         "ik14_group": ["TAAAAAAAAAAAAA", "TBBBBBBBBBBBBB", "TCCCCCCCCCCCCC"],
                         "smiles": ["CCO", "CCN", "CCC"], "formula": ["C2H6O", "C2H7N", "C3H8"],
                         "mono_mass": [46.0, 45.0, 44.0]})
    excl = {"m1": {"SBBBBBBBBBBBBB"}, "m2": {"SCCCCCCCCCCCCC"}, "m3": set()}
    tab = va.reach_table(mols, excl, hits, seeds).set_index("molecule_id")
    assert tab.loc["m1", "va_reach"] and tab.loc["m1", "va_n_seeds"] == 1 and tab.loc["m1", "va_rule"] == 3
    assert tab.loc["m1", "va_n_seeds_all"] == 2
    assert not tab.loc["m2", "va_reach"] and not tab.loc["m2", "reach"]
    assert tab.loc["m3", "va_reach"]                   # matched by metric key (tautomer)
    s = va.summarize_reach(tab.reset_index())
    assert s["n"] == 3 and s["va_reach"] == pytest.approx(2 / 3, abs=1e-4)


def test_reach_gen_two_edits():
    # truth: 4-O-methyl salicylic acid methyl ester is two edits from salicylic acid
    truth = "COC(=O)c1ccc(OC)cc1O"
    mols = pd.DataFrame({"molecule_id": ["m1"], "mkey": [ik14(truth)], "ik14": [ik14(truth)],
                         "ik14_group": [ik14(truth)], "smiles": [truth], "formula": ["C9H10O4"],
                         "mono_mass": [182.0579]})
    seeds = pd.DataFrame({"seed_id": [0], "ik14": ["SAAAAAAAAAAAAA"], "pri": [0], "source": "np"})
    hits = pd.DataFrame(columns=list(va.HIT_COLUMNS))
    gen_seeds = {"m1": [(ik14("OC(=O)c1ccc(O)cc1O"), "OC(=O)c1ccc(O)cc1O")]}
    tab = va.reach_table(mols, {"m1": set()}, hits, seeds, gen_seeds=gen_seeds)
    assert bool(tab["gen_reach"].iloc[0]) and bool(tab["reach"].iloc[0])
    assert tab["gen_n_products"].iloc[0] > 0 and "methylation" in tab["gen_edits"].iloc[0]
    tab2 = va.reach_table(mols, {"m1": {ik14("OC(=O)c1ccc(O)cc1O")}}, hits, seeds, gen_seeds=gen_seeds)
    assert not bool(tab2["gen_reach"].iloc[0])        # the only seed is excluded


def test_kaggle_job_notebook_offline():
    import ast
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import kaggle_job

    args = kaggle_job._parse(["push", "--slug", "casmi-va-build-0", "--code", "casmi-job-code",
                              "--dataset", "casmi-db-pubchem", "--dataset", "other/ds-x",
                              "--kernel-source", "casmi-va-build-1", "--env", "A=1",
                              "--cmd", 'python "$CODE/scripts/build_va.py" build --shard 0',
                              "--cmd", "ls $DS_CASMI_DB_PUBCHEM"])
    args.slug = args.slug[0]
    job = kaggle_job.make_job("me", args)
    assert job["datasets"] == ["me/casmi-db-pubchem", "other/ds-x"]
    assert job["ds_env"]["other/ds-x"] == "DS_DS_X" and job["ks_env"]["me/casmi-va-build-1"] == "KS_CASMI_VA_BUILD_1"
    assert job["code"] == "me/casmi-job-code" and job["env"] == {"A": "1"} and job["stop_on_error"]
    nb = kaggle_job.job_notebook(job)
    text = json.dumps(nb)
    for bad in ("/home/", "/root/"):
        assert bad not in text
    assert chr(0x2014) not in text
    code = "\n\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "code")
    ast.parse(code)
    ns: dict = {}
    exec(nb["cells"][1]["source"], ns)  # the settings cell round-trips the job
    assert ns["JOB"]["cmds"] == job["cmds"] and ns["OWNER"] == "me"
    assert "job_status.json" in code and "PYTHONPATH" in code

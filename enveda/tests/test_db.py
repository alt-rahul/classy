"""Tests for casmi.db (WP2): NP-tier and PubChem-tier lookup on synthetic fixtures,
plus acceptance checks on the real NP tier when it has been built."""

from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from casmi import db as cdb
from casmi.config import Config
from casmi.db import OUT_COLUMNS, CandidateDB

NP_ROWS = [
    # ik14, smiles, formula, mono_mass, in_coconut, cid, pubmed_count
    ("NPLOWMASSAAAAA", "CCCC(=O)O", "C4H8O2", 100.0015, True, None, None),
    ("AAAAAAAAAAAAAA", "CCO", "C2H6O", 199.9990, True, None, 3),
    ("BBBBBBBBBBBBBB", "CCN", "C2H7N", 200.0005, False, 77, None),
    ("CCCCCCCCCCCCCC", "CCCl", "C2H5Cl", 250.0000, True, None, None),
    ("REFJWTPEDVJJIY", "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12", "C15H10O7", 302.0427, True, 5280343, 10),
]
# PubChem tier rows: (shard bucket, ik14, mono_mass, cid, pmid_count, n_cids)
PC_ROWS = [
    (190, "PCLOWAAAAAAAAA", 191.0, 1, 0, 1),
    (190, "PCLOWBBBBBBBBB", 195.0, 2, 0, 1),
    (190, "PCLOWCCCCCCCCC", 198.0, 3, 0, 1),
    (190, "PPPPPPPPPPPPPP", 199.9985, 10, 1, 1),
    (190, "AAAAAAAAAAAAAA", 199.9990, 11, 5, 2),
    (200, "BBBBBBBBBBBBBB", 200.0005, 22, 7, 3),
    (200, "QQQQQQQQQQQQQQ", 200.0010, 23, 0, 1),
    (200, "PCHIGHAAAAAAAA", 205.0, 24, 0, 1),
    (200, "PCHIGHBBBBBBBB", 209.0, 25, 0, 1),
]


def _write_np(d):
    df = pd.DataFrame(NP_ROWS, columns=["ik14", "smiles", "formula", "mono_mass", "in_coconut",
                                        "cid", "pubmed_count"])
    df["in_npatlas"] = False
    df["in_lotus"] = df["in_coconut"]
    df["in_pubchemlite"] = df["pubmed_count"].notna()
    df["cid"] = df["cid"].astype("Int64")
    df["pubmed_count"] = df["pubmed_count"].astype("Int32")
    p = d / "np.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), p)
    rng = np.random.default_rng(0)
    fp = rng.integers(0, 256, size=(len(df), 512), dtype=np.uint8)
    np.save(d / "np_fp.npy", fp)
    return p, fp


def _write_pubchem(d, with_index=True):
    d.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([("ik14", pa.string()), ("smiles", pa.string()), ("formula", pa.string()),
                        ("mono_mass", pa.float64()), ("cid", pa.int32()), ("pmid_count", pa.int32()),
                        ("n_cids", pa.int16())])
    for b in sorted({r[0] for r in PC_ROWS}):
        rows = sorted([r for r in PC_ROWS if r[0] == b], key=lambda r: r[2])
        tbl = pa.table({
            "ik14": [r[1] for r in rows], "smiles": ["C" * (i + 1) for i in range(len(rows))],
            "formula": ["CX"] * len(rows), "mono_mass": [r[2] for r in rows],
            "cid": [r[3] for r in rows], "pmid_count": [r[4] for r in rows],
            "n_cids": [r[5] for r in rows]}, schema=schema)
        pq.write_table(tbl, d / cdb.shard_name(b), row_group_size=2)
    if with_index:
        (d / "_index.json").write_text(json.dumps({"n_shards": 2}))
    return d


@pytest.fixture()
def dbs(tmp_path):
    npd = tmp_path / "np"
    npd.mkdir()
    np_path, fp = _write_np(npd)
    pcd = _write_pubchem(tmp_path / "pubchem")
    db = CandidateDB(np_path=np_path, pubchem_dir=pcd, cfg=Config())
    return db, fp, tmp_path


def test_helpers():
    assert cdb.shard_bucket(199.99) == 190
    assert cdb.shard_bucket(200.0) == 200
    assert cdb.shard_name(190) == "mass_0190.parquet"
    lo, hi = cdb.window(100.0, 10.0, 0.002)
    assert hi - lo == pytest.approx(0.004)
    lo, hi = cdb.window(1000.0, 10.0, 0.002)
    assert hi - lo == pytest.approx(0.02)
    packed = np.packbits(np.eye(1, 4096, 9, dtype=np.uint8), axis=1)
    assert packed[0, 1] == 0x40  # bit 9 -> byte 1, mask 0x80 >> 1 (bit order "big")
    assert cdb.unpack_fp(packed)[0, 9] == 1


def test_np_only(dbs):
    db, _, _ = dbs
    r = db.lookup(200.0, 10.0, tiers=("np",))
    assert list(r.columns) == list(OUT_COLUMNS)
    assert list(r["ik14"]) == ["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"]
    assert (r["tier"] == "np").all() and r["in_np"].all() and not r["in_pubchem"].any()
    assert list(r["np_row"]) == [1, 2]
    assert r["ppm_err"].to_numpy() == pytest.approx([-5.0, 2.5])
    assert r["in_coconut"].tolist() == [True, False]
    assert r["cid"].dtype == "Int64" and r["pmid_count"].isna().all()


def test_union_np_wins_and_spans_shards(dbs):
    db, _, _ = dbs
    r = db.lookup(200.0, 10.0, tiers=("np", "pubchem"))
    assert list(r.columns) == list(OUT_COLUMNS)
    assert r["ik14"].is_unique
    assert list(r["ik14"]) == ["PPPPPPPPPPPPPP", "AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB", "QQQQQQQQQQQQQQ"]
    assert list(r["tier"]) == ["pubchem", "np", "np", "pubchem"]
    a = r.set_index("ik14")
    # NP rows win and inherit the PubChem counts and (missing) cid.
    assert a.at["AAAAAAAAAAAAAA", "smiles"] == "CCO"
    assert bool(a.at["AAAAAAAAAAAAAA", "in_pubchem"])
    assert a.at["AAAAAAAAAAAAAA", "pmid_count"] == 5 and a.at["AAAAAAAAAAAAAA", "n_cids"] == 2
    assert a.at["AAAAAAAAAAAAAA", "cid"] == 11
    assert a.at["BBBBBBBBBBBBBB", "cid"] == 77  # NP cid kept
    assert a.at["BBBBBBBBBBBBBB", "pmid_count"] == 7
    assert a.at["PPPPPPPPPPPPPP", "np_row"] == -1
    assert not bool(a.at["PPPPPPPPPPPPPP", "in_coconut"])
    assert pd.isna(a.at["PPPPPPPPPPPPPP", "pubmed_count"])
    assert r["mono_mass"].is_monotonic_increasing
    # Only row groups whose mass range meets the window are read: 2 of 3 in mass_0190
    # (199.998-199.9985 and 199.9990) and 1 of 2 in mass_0200.
    assert db.n_row_groups_read == 3
    db.lookup(200.0, 10.0)
    assert db.n_row_groups_read == 3  # cached
    db.release()
    db.lookup(200.0, 10.0)
    assert db.n_row_groups_read == 6


def test_pubchem_only_and_dtypes(dbs):
    db, _, _ = dbs
    r = db.lookup(205.0, 10.0, tiers="pubchem")
    assert list(r["ik14"]) == ["PCHIGHAAAAAAAA"]
    assert r["cid"].dtype == "Int64" and r["pmid_count"].dtype == "Int32"
    both = db.lookup(200.0, 10.0)
    np_only = db.lookup(250.0, 10.0)
    assert dict(both.dtypes) == dict(np_only.dtypes)


def test_missing_pubchem_degrades(tmp_path, caplog):
    npd = tmp_path / "np"
    npd.mkdir()
    np_path, _ = _write_np(npd)
    db = CandidateDB(np_path=np_path, pubchem_dir=tmp_path / "nothing_here")
    with caplog.at_level(logging.WARNING, logger="casmi.db"):
        r = db.lookup(200.0, 10.0, tiers=("np", "pubchem"))
        db.lookup(200.0, 10.0, tiers=("np", "pubchem"))
    assert list(r["ik14"]) == ["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"]
    msgs = [m for m in caplog.messages if "PubChem tier not found" in m]
    assert len(msgs) == 1
    assert not db.pubchem_available()


def test_incomplete_pubchem_warns(tmp_path, caplog):
    npd = tmp_path / "np"
    npd.mkdir()
    np_path, _ = _write_np(npd)
    pcd = _write_pubchem(tmp_path / "pubchem", with_index=False)
    db = CandidateDB(np_path=np_path, pubchem_dir=pcd)
    with caplog.at_level(logging.WARNING, logger="casmi.db"):
        r = db.lookup(200.0, 10.0)
    assert len(r) == 4
    assert any("no _index.json" in m for m in caplog.messages)


def test_batch_matches_single(dbs):
    db, _, _ = dbs
    masses = [250.0, 200.0, 700.0, 199.9990]
    keys = ["m1", "m2", "m3", "m4"]
    b = db.lookup_batch(masses, 10.0, keys=keys)
    assert list(b.columns) == ["query", "query_mass", *OUT_COLUMNS]
    assert list(dict.fromkeys(b["query"])) == ["m1", "m2", "m4"]
    for k, m in zip(keys, masses):
        single = db.lookup(m, 10.0)
        part = b[b["query"] == k].drop(columns=["query", "query_mass"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(part, single)
    bi = db.lookup_batch(masses, 10.0, tiers=("np",))
    assert bi["query"].dtype == np.int64 and set(bi["query"]) == {0, 1, 3}


def test_empty_and_min_tol(dbs):
    db, _, _ = dbs
    e = db.lookup(1000.0, 10.0)
    assert len(e) == 0 and list(e.columns) == list(OUT_COLUMNS)
    eb = db.lookup_batch([1000.0], 10.0)
    assert len(eb) == 0 and list(eb.columns) == ["query", "query_mass", *OUT_COLUMNS]
    # At 100 Da, 10 ppm is 0.001 Da; the 0.002 Da floor reaches the row at +0.0015.
    assert list(db.lookup(100.0, 10.0, tiers="np")["ik14"]) == ["NPLOWMASSAAAAA"]
    assert len(db.lookup(100.0, 10.0, tiers="np", min_tol_da=0.0)) == 0
    with pytest.raises(ValueError):
        db.lookup(200.0, 10.0, tiers=("np", "chembl"))


def test_np_fingerprints(dbs):
    db, fp, _ = dbs
    r = db.lookup(200.0, 10.0)
    rows = r.loc[r["tier"] == "np", "np_row"].to_numpy()
    got = db.np_fingerprints(rows)
    assert got.dtype == np.uint8 and got.shape == (2, 512)
    assert np.array_equal(got, fp[rows])
    bits = db.np_fingerprints([4], unpack=True)
    assert bits.shape == (1, 4096)
    assert np.array_equal(bits, np.unpackbits(fp[[4]], axis=1))
    with pytest.raises(IndexError):
        db.np_fingerprints([-1])


def test_module_level_api(dbs, monkeypatch):
    db, _, _ = dbs
    monkeypatch.setattr(cdb, "_DEFAULT", db)
    assert list(cdb.lookup(302.0427, 10.0, tiers="np")["ik14"]) == ["REFJWTPEDVJJIY"]
    assert len(cdb.lookup_batch([302.0427, 250.0], 10.0)) == 2
    assert cdb.np_fingerprints([0]).shape == (1, 512)
    assert len(cdb.np_table()) == len(NP_ROWS)
    cdb.release()


REAL_NP = "data/processed/db/np/np.parquet"
REAL_FP = "data/processed/db/np/np_fp.npy"


@pytest.mark.data(REAL_NP, REAL_FP)
def test_real_np_tier():
    db = CandidateDB(pubchem_dir="/nonexistent_pubchem_dir_for_test")
    t = db.load_np()
    assert 600_000 <= len(t) <= 1_500_000
    assert t["ik14"].is_unique
    r = db.lookup(302.0427, 10.0, tiers=("np",))
    hit = r[r["ik14"] == "REFJWTPEDVJJIY"]
    assert len(hit) == 1 and bool(hit["in_coconut"].iloc[0])
    t0 = time.perf_counter()
    for m in np.linspace(150, 1150, 200):
        db.lookup(float(m), 10.0, tiers=("np",))
    assert (time.perf_counter() - t0) / 200 < 0.05
    fp = db.np_fingerprints(hit["np_row"].to_numpy())
    assert fp.shape == (1, 512) and fp.any()

"""WP6: train ingestion, structure table, mkey pass, public test join and audit.

Runs on the synthetic train file of ``scripts/make_synthetic_train.py`` (the ``synthetic_train``
fixture in conftest.py: NP tier only, no PubChem scan). Tests marked ``data`` run on the real
``data/raw/train.parquet`` when it exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from casmi import paths
from casmi.adducts import TEST_ADDUCTS
from casmi.config import Config
from casmi.spectra import clean


def test_drop_reasons_and_flags(synthetic_train):
    info = synthetic_train["info"]
    tot = {}
    for lib, d in info["drops"].items():
        for k, v in d.items():
            tot[k] = tot.get(k, 0) + v
    # planted by Maker._bad_rows
    assert tot == {"null_label": 2, "bad_precursor": 1, "bad_peak_arrays": 2, "no_raw_peaks": 1,
                   "no_polarity": 1, "duplicate_spectrum_id": 3, "empty_after_cleaning": 1}
    for lib, n_in in info["rows_in"].items():
        assert n_in == info["rows_kept"].get(lib, 0) + sum(info["drops"].get(lib, {}).values())
    flags = {}
    for d in info["flags"].values():
        for k, v in d.items():
            flags[k] = flags.get(k, 0) + v
    assert flags["adduct_unparsed"] == 4          # four planted unparseable adducts (kept)
    assert flags["abs_ppm_gt_20"] == 5            # five planted +800 ppm rows (kept)
    assert flags["peaks_above_precursor"] > 0 and flags["bpi_null"] > 0
    assert info["kept_rows"] == info["input_rows"] - sum(tot.values())


def test_spectra_files(synthetic_train):
    out, ingest = synthetic_train["out"], synthetic_train["ingest"]
    files = sorted((out / "spectra").glob("spectra_*.parquet"))
    info = synthetic_train["info"]
    assert {f.name for f in files} == {ingest.lib_file_name(l) for l in info["rows_kept"]}
    assert not list(out.glob("spectra_*.parquet"))          # only in the spectra/ subdir
    meta = pd.read_parquet(out / "meta.parquet")
    assert list(meta.columns) == list(ingest.META_COLUMNS)
    assert len(meta) == info["kept_rows"] and not meta["spectrum_id"].duplicated().any()
    raw = pq.read_table(synthetic_train["train"]).to_pandas()
    cfg = Config()
    for f in files:
        t = pq.read_table(f)
        names = t.schema.names
        assert names[:len(ingest.TRAIN_COLUMNS)] == list(ingest.TRAIN_COLUMNS)
        assert set(ingest.DERIVED) <= set(names)
        assert t.schema.field("ms2_mzs").type == pa.list_(pa.float32())
        d = t.to_pandas()
        lib = d["ingest_lib"].iloc[0]
        assert (d["ingest_lib"] == lib).all()
        m = meta[meta["ingest_lib"] == lib].sort_values("file_row")
        assert (m["file_row"].to_numpy() == np.arange(len(d))).all()
        assert (m["spectrum_id"].to_numpy() == d["spectrum_id"].to_numpy()).all()
        for i in range(0, len(d), 37):
            r = d.iloc[i]
            src = raw.iloc[int(r["source_row"])]
            cm, ci, _ = clean(src["ms2_mzs"], src["ms2_normalized_intensities"],
                              src["precursor_mz"], cfg)
            assert np.array_equal(np.asarray(r["ms2_mzs"], np.float32), cm)
            assert np.array_equal(np.asarray(r["ms2_normalized_intensities"], np.float32), ci)
            assert len(cm) <= cfg.top_n_peaks and r["n_peaks"] == len(cm)
            assert r["n_peaks_raw"] == len(src["ms2_mzs"])
            assert (np.asarray(r["ms2_mzs"]) <= r["precursor_mz"] + cfg.clean_max_above_precursor_da).all()
        ta = d["adduct_test"].dropna()
        assert set(ta) <= set(TEST_ADDUCTS)


def test_streaming_matches_single_batch(synthetic_train, tmp_path):
    """Row-group streaming in small batches gives the same outputs as one big batch."""
    import logging

    from casmi import db as cdb

    ingest = synthetic_train["ingest"]

    class Args:
        pass

    a = Args()
    a.out, a.test, a.input, a.batch_rows, a.n_jobs = (tmp_path / "big", None,
                                                      synthetic_train["train"], 10 ** 6, 1)
    empty = tmp_path / "nopc"
    empty.mkdir()
    mp = pytest.MonkeyPatch()
    mp.setattr(cdb, "pubchem_dir", lambda: empty)
    try:
        ingest.stage_ingest(a, synthetic_train["cfg"], logging.getLogger("t"))
    finally:
        mp.undo()
    small, big = synthetic_train["out"], tmp_path / "big"
    for f in sorted((small / "spectra").glob("*.parquet")):
        ta = pq.read_table(f).to_pandas()
        tb = pq.read_table(big / "spectra" / f.name).to_pandas()
        pd.testing.assert_frame_equal(ta, tb)
    sa = pd.read_parquet(small / "structures.parquet").drop(columns=["mkey", "mkey_status"])
    sb = pd.read_parquet(big / "structures.parquet").drop(columns=["mkey", "mkey_status"])
    pd.testing.assert_frame_equal(sa, sb)
    ia = json.loads((small / "ingest_info.json").read_text())
    ib = json.loads((big / "ingest_info.json").read_text())
    assert ia["drops"] == ib["drops"] and ia["rows_kept"] == ib["rows_kept"]


def test_structures_table(synthetic_train):
    st = synthetic_train["structures"]
    meta = pd.read_parquet(synthetic_train["out"] / "meta.parquet")
    assert st["ik14"].is_unique and set(st["ik14"]) == set(meta["ik14"])
    assert st["n_spectra"].sum() == len(meta)
    libc = [c for c in st.columns if c.startswith("n_lib_")]
    assert (st[libc].sum(axis=1) == st["n_spectra"]).all()
    instc = [f"n_inst_{f}" for f in ("timstof", "qtof", "orbitrap", "other")]
    assert (st[instc].sum(axis=1) == st["n_spectra"]).all()
    addc = [c for c in st.columns if c.startswith("n_add_")]
    assert len(addc) == 12 and (st[addc].sum(axis=1) == st["n_spectra"]).all()
    assert (st["n_pos"] + st["n_neg"] == st["n_spectra"]).all()
    # labels are NP-tier structures (tautomer variants and decoys aside)
    assert st["in_np"].mean() > 0.9
    assert st["in_pubchem"].isna().all()                 # PubChem scan disabled in the fixture
    assert st["rdkit_ok"].all() and np.isfinite(st["np_score"]).all()
    assert (np.abs(st["mono_mass"] - st["mass_given"]) < 1e-3).all()
    assert st["mkey"].notna().all() and (st["mkey_status"] == "ok").all()
    # tautomer variants: another ik14 with the same mkey
    lab = synthetic_train["labels"]
    n_taut = int(lab["taut_smiles"].notna().sum())
    shared = st.groupby("mkey")["ik14"].nunique()
    assert n_taut > 0 and (shared > 1).sum() >= n_taut
    lc = synthetic_train["lib_counts"]
    assert lc.groupby("ik14")["n_spectra"].sum().reindex(st["ik14"]).to_numpy().tolist() == \
        st["n_spectra"].tolist()


def test_instrument_family_of(synthetic_train):
    f = synthetic_train["ingest"].instrument_family_of
    assert f("anything", "enveda-180") == "timstof"
    assert f("Bruker timsTOF Pro", "gnps") == "timstof"
    assert f("Q Exactive Plus Orbitrap Thermo Scientific", "mona") == "orbitrap"
    assert f("LC-ESI-ITFT", "massbank") == "orbitrap"
    assert f("LC-ESI-QTOF", "riken") == "qtof"
    assert f("Maxis II HD Q-TOF Bruker", "gnps") == "qtof"
    assert f(None, "pluskal_ms2") == "orbitrap"
    assert f(None, "drug_plus") == "other" and f("LC-ESI-QQ", "massbank") == "other"
    assert f.__doc__ and "timstof" in f.__doc__


def test_public_test_join(synthetic_train):
    out = synthetic_train["out"]
    m = pd.read_parquet(out / "public_test_matches.parquet")
    truth = pd.read_parquet(out / "public_test_truth.parquet")
    best = m[m["best"]]
    lab = synthetic_train["labels"]
    assert len(best) >= 1150 and best["test_spectrum_id"].is_unique
    assert best["by_id"].sum() > 100 and (~best["by_id"]).sum() > 100
    assert len(truth) == len(lab) and truth["consistent"].all()
    want = dict(zip(lab["molecule_id"], [k[:14] for k in lab["inchikey"]]))
    assert all(want[m_] == k for m_, k in zip(truth["molecule_id"], truth["ik14"]))
    # copies with extra peaks above precursor + 2 still match; the cut evidence is recorded
    assert (best["n_dropped_above"] > 0).any()
    assert best["max_kept_excess"].max() < best["min_dropped_excess"].min()
    assert np.allclose(best["int_ratio_median"], 1.0)
    tol = synthetic_train["ingest"].tolerance_recommendation(
        pd.read_parquet(out / "meta.parquet"), m)
    assert tol["recommended_ppm"] >= 5 and tol["public_n"] == len(truth)


def test_audit(synthetic_train, tmp_path):
    import logging

    ingest = synthetic_train["ingest"]
    p = tmp_path / "DATA_AUDIT.md"
    ingest.write_audit(synthetic_train["out"], p, Config(), logging.getLogger("t"))
    text = p.read_text()
    for h in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.", "## 7.", "## 8.", "## 9.",
              "## 10.", "## 11.", "## 12."):
        assert h in text, h
    assert "Recommended `mass_tol_ppm`" in text and "SYNTHETIC" not in text.split("\n")[2]
    assert chr(0x2014) not in text
    # every markdown table is rectangular (no stray "|" inside a cell)
    width = None
    for line in text.splitlines() + [""]:
        if line.startswith("|"):
            width = line.count("|") if width is None else width
            assert line.count("|") == width, line
        else:
            width = None


def test_library_from_ingest_outputs(synthetic_train):
    lib = synthetic_train["library"]
    info = synthetic_train["info"]
    assert lib.n_spectra == info["kept_rows"]
    assert lib.n_structures == len(synthetic_train["structures"])
    fam = dict(zip(lib.meta["spectrum_id"], lib.meta["instrument_family"]))
    meta = pd.read_parquet(synthetic_train["out"] / "meta.parquet")
    assert all(fam[s] == f for s, f in zip(meta["spectrum_id"], meta["instrument_family"]))


@pytest.mark.data
def test_real_train_schema(train_parquet):
    ingest = __import__("conftest").load_script("ingest_train")
    pf = pq.ParquetFile(train_parquet)
    missing = [c for c in ingest.TRAIN_COLUMNS if c not in pf.schema_arrow.names]
    assert not missing, missing

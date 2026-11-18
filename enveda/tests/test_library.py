"""WP5: library build, load, mass-restricted and open search (PLAN.md 4.2, 4.3).

Code-phase tests build a library from the public test file with fake labels
(``library.fake_labels_from_test``); tests marked ``data`` need train.parquet.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from casmi import paths
from casmi.config import Config
from casmi.library import (SEARCH_BY_MASS_COLUMNS, SEARCH_OPEN_COLUMNS, SPECTRUM_COLUMNS, Library,
                           adduct_family, fake_labels_from_test, instrument_family,
                           prepare_queries)


@pytest.fixture(scope="module")
def fake_table(public_test_parquet) -> pd.DataFrame:
    return fake_labels_from_test(pd.read_parquet(public_test_parquet))


@pytest.fixture(scope="module")
def lib(fake_table, tmp_path_factory) -> Library:
    out = tmp_path_factory.mktemp("lib") / "library"
    return Library.build(fake_table, Config(), out_dir=out, batch_rows=250)


def test_files_and_formats(lib):
    d = lib.dir
    for name in ("peaks_mz.npy", "peaks_int.npy", "offsets.npy", "meta.parquet",
                 "structures.parquet", "consensus_mz.npy", "consensus_int.npy",
                 "consensus_offsets.npy", "consensus_meta.parquet", "open_index_pos.npz",
                 "open_index_neg.npz", "library_info.json"):
        assert (d / name).is_file(), name
    assert not list(d.glob("_tmp*"))
    assert isinstance(lib.store.mz, np.memmap) and lib.store.mz.dtype == np.float32
    assert lib.store.offsets.dtype == np.int64 and lib.store.offsets[-1] == len(lib.store.mz)
    assert lib.n_spectra == 1213 and lib.n_structures == 400
    info = json.loads((d / "library_info.json").read_text())
    assert info["n_spectra"] == 1213 and info["n_consensus"] == lib.n_consensus
    assert list(lib.meta.columns) == SPECTRUM_COLUMNS + ["source_row"]
    assert (lib.meta["spec_idx"].to_numpy() == np.arange(lib.n_spectra)).all()
    # sorted by structure mass, spectra of a structure contiguous, consensus too
    assert np.all(np.diff(lib.struct_mass) >= 0)
    assert np.all(np.diff(lib.spec_struct) >= 0) and np.all(np.diff(lib.cons_struct) >= 0)
    s = lib.structures
    assert (s["spec_end"] - s["spec_start"]).sum() == lib.n_spectra
    assert (s["cons_end"] - s["cons_start"]).sum() == lib.n_consensus
    # every spectrum's consensus belongs to its structure and contains it
    c = lib.cons_meta
    ci = lib.spec_cons
    assert (lib.cons_struct[ci] == lib.spec_struct).all()
    assert ((c["spec_start"].to_numpy()[ci] <= np.arange(lib.n_spectra))
            & (np.arange(lib.n_spectra) < c["spec_end"].to_numpy()[ci])).all()
    for p in (1, -1):
        m = lib.open_index(p)
        assert m.shape == ((lib.cons_pol == p).sum(), 80000) and m.dtype == np.float32
    # peaks equal a direct clean of the source rows
    from casmi.spectra import clean
    src = pd.read_parquet(paths.TEST_PARQUET)
    for i in range(0, lib.n_spectra, 101):
        row = src.iloc[int(lib.meta["source_row"].iloc[i])]
        m, x, _ = clean(row.ms2_mzs, row.ms2_normalized_intensities, row.precursor_mz)
        assert np.array_equal(lib.store[i][0], m) and np.array_equal(lib.store[i][1], x)


def test_self_search_by_mass_every_spectrum(lib):
    """Each library spectrum, used as a query, is its own best reference (cosine 1)."""
    t0 = time.perf_counter()
    for i in range(lib.n_spectra):
        r = lib.search_by_mass(lib.struct_mass[lib.spec_struct[i]],
                               queries=lib.queries_from_library([i]))
        top = r.iloc[0]
        assert top.struct_idx == lib.spec_struct[i] and top.best_ref_idx == i, i
        assert top.cos_max == pytest.approx(1.0) and top.ent_max == pytest.approx(1.0)
        assert top.n_matched_best == lib.store.n_peaks[i]
    dt = time.perf_counter() - t0
    print(f"\nself-search by mass: {lib.n_spectra} queries in {dt:.2f} s")


def test_molecule_search_from_test_schema(lib, fake_table):
    """Raw test-schema rows of a molecule: own structure first, consensus identical."""
    t0 = time.perf_counter()
    n = 0
    for mol, g in fake_table.groupby("molecule_id", sort=True):
        r = lib.search_by_mass(float(g["mono_mass"].iloc[0]), queries=g)
        assert list(r.columns) == SEARCH_BY_MASS_COLUMNS
        top = r.iloc[0]
        assert top.ik14 == g["inchikey14"].iloc[0], mol
        assert top.cos_max == pytest.approx(1.0) and top.cons_cos == pytest.approx(1.0)
        assert top.cos_mean == pytest.approx(1.0) and top.has_tims
        assert top.n_refs == len(g) and top.ce_dist_best == 0.0
        assert abs(top.mass_err_ppm) < 1e-6
        n += 1
    print(f"\nmass-restricted search: {n} molecules in {time.perf_counter() - t0:.2f} s")


def test_exclusion_masks(lib, fake_table):
    g = fake_table[fake_table["molecule_id"] == fake_table["molecule_id"].iloc[0]]
    ik = g["inchikey14"].iloc[0]
    M = float(g["mono_mass"].iloc[0])
    base = lib.search_by_mass(M, queries=g)
    assert base.ik14.iloc[0] == ik
    # C2-like: all spectra of the structure removed -> structure absent, nothing else lost
    mask = lib.spectra_mask(ik14=[ik])
    assert mask.sum() == len(g)
    r = lib.search_by_mass(M, queries=g, exclude_mask=mask)
    assert ik not in set(r.ik14) and len(r) == len(base) - 1
    # partial removal (one spectrum id): the structure stays with one reference fewer,
    # and its consensus spectra containing that spectrum are removed
    sid = g["spectrum_id"].iloc[0]
    m1 = lib.spectra_mask(spectrum_id=[sid])
    r1 = lib.search_by_mass(M, queries=g, exclude_mask=m1)
    row = r1[r1.ik14 == ik].iloc[0]
    assert row.n_refs == len(g) - 1 and lib.spec_id[row.best_ref_idx] != sid
    cm = lib.consensus_mask(m1)
    assert cm.sum() == 1 and cm[lib.spec_cons[np.flatnonzero(m1)[0]]]
    # library mask by ingest_lib
    assert lib.spectra_mask(ingest_lib=["public_test"]).all()
    assert not lib.spectra_mask(ik14=[ik], ingest_lib=["other_lib"]).any()
    # masks do not change the library
    assert lib.search_by_mass(M, queries=g).equals(base)
    with pytest.raises(ValueError):
        lib.search_by_mass(M, queries=g, exclude_mask=np.zeros(3, bool))


def test_search_without_queries_and_polarity(lib, fake_table):
    g = fake_table[fake_table["ionization_mode"] == "negative"]
    g = g[g["molecule_id"] == g["molecule_id"].iloc[0]]
    M = float(g["mono_mass"].iloc[0])
    r = lib.search_by_mass(M)
    assert len(r) >= 1 and r["cos_max"].isna().all() and (r["n_refs"] > 0).all()
    assert (np.abs(r["mass_err_ppm"]) <= 10).all()
    rp = lib.search_by_mass(M, queries=g, polarity="positive")
    assert rp["cos_max"].isna().all()                    # negative queries only
    rn = lib.search_by_mass(M, queries=g, polarity="negative")
    assert rn.cos_max.iloc[0] == pytest.approx(1.0)
    assert lib.search_by_mass(float("nan")).empty
    assert len(lib.search_by_mass(1e5)) == 0


def test_open_self_search_every_consensus(lib):
    t0 = time.perf_counter()
    for i in range(lib.n_consensus):
        r = lib.search_open(lib.consensus_spectrum(i), int(lib.cons_pol[i]), top_k=5,
                            family=lib.cons_family[i], unique_structures=False)
        assert list(r.columns) == SEARCH_OPEN_COLUMNS
        assert r.cons_idx.iloc[0] == i and r.mod_cos.iloc[0] == pytest.approx(1.0), i
        assert r["rank"].tolist() == list(range(1, len(r) + 1))
    print(f"\nopen self-search: {lib.n_consensus} queries in {time.perf_counter() - t0:.2f} s")


def test_open_search_masks_and_uniqueness(lib):
    i = int(np.flatnonzero(lib.cons_pol == 1)[0])
    q = lib.consensus_spectrum(i)
    st = lib.cons_struct[i]
    r = lib.search_open(q, "positive", top_k=50)
    assert r.struct_idx.is_unique and len(r) <= 50
    assert (np.diff(r.mod_cos.to_numpy()) <= 0).all()
    mask = lib.spectra_mask(ik14=[lib.struct_ik14[st]])
    r2 = lib.search_open(q, "positive", top_k=50, exclude_mask=mask)
    assert st not in set(r2.struct_idx)
    cons_mask = np.zeros(lib.n_consensus, bool)
    cons_mask[i] = True
    r3 = lib.search_open(q, 1, exclude_cons_mask=cons_mask, unique_structures=False)
    assert i not in set(r3.cons_idx)
    assert lib.search_open((np.zeros(0, np.float32), np.zeros(0, np.float32), 300.0), 1).empty


def test_query_consensus_and_families(fake_table):
    cfg = Config()
    g = fake_table[fake_table["molecule_id"] == fake_table["molecule_id"].iloc[0]]
    qs = prepare_queries(g, cfg)
    assert len(qs) == len(g) and set(qs.instrument_family) == {"timstof"}
    c = qs.consensus(1)
    assert c is not None and c.intensity.max() == 1.0
    assert adduct_family("[M+NH4]+", 1) == "[M+H]+" and adduct_family("[M+CH2O2-H]-", -1) == "[M-H]-"
    assert adduct_family("[M+FA-H]-", -1) == "[M-H]-" and adduct_family("[M+Na]+") == "[M+Na]+"
    assert adduct_family("[M+2H]2+") == "[M+2H]2+" and adduct_family(None) == ""
    assert instrument_family("timsTOF") == "timstof"
    assert instrument_family(None, "enveda-180") == "timstof"
    assert instrument_family("LC-ESI-QTOF") == "qtof" and instrument_family("Maxis II HD Q-TOF Bruker") == "qtof"
    assert instrument_family("Orbitrap") == "orbitrap" and instrument_family("LC-ESI-ITFT") == "orbitrap"
    assert instrument_family("Q Exactive Plus") == "orbitrap" and instrument_family("ESI-QFT") == "orbitrap"
    assert instrument_family(None, "pluskal_ms2") == "orbitrap"
    assert instrument_family("LC-ESI-QQ") == "other" and instrument_family(None) == "other"


def _synthetic_train(n_struct: int = 40, seed: int = 0) -> pd.DataFrame:
    """Train-schema table: formulas, several libraries / instruments / adducts, bad rows."""
    from casmi.adducts import TEST_ADDUCTS, formula_mass

    rng = np.random.default_rng(seed)
    formulas = ["C6H12O6", "C9H11NO2", "C15H10O5", "C21H20O11", "C10H16N5O13P3",
                "C17H19NO3", "C8H10N4O2", "C20H24N2O2"]
    rows = []
    libs = [("gnps", "ESI-QTOF"), ("pluskal_ms2", None), ("enveda-180", "timsTOF"),
            ("massbank", "LC-ESI-ITFT")]
    adducts = [("[M+H]+", "positive"), ("[M+NH4]+", "positive"), ("[M+Na]+", "positive"),
               ("[M-H]-", "negative"), ("[M+CH2O2-H]-", "negative")]
    for s in range(n_struct):
        f = formulas[s % len(formulas)]
        M = formula_mass(f)
        ik = f"SYNTH{s:09d}"
        base = np.sort(rng.uniform(50, M - 5, 30))
        for k in range(int(rng.integers(1, 6))):
            lib, itype = libs[int(rng.integers(0, len(libs)))]
            add, mode = adducts[int(rng.integers(0, len(adducts)))]
            prec = TEST_ADDUCTS[add].precursor_mz(M) * (1 + rng.normal(0, 2e-6))
            mz = np.sort(np.concatenate([base + rng.normal(0, 0.002, base.size),
                                         rng.uniform(50, prec + 5, 40)]))
            it = rng.uniform(0, 1, mz.size) ** 3
            it = it / it.max()
            rows.append(dict(spectrum_id=f"sp{s:04d}_{k}", ms2_mzs=mz,
                             ms2_normalized_intensities=it, precursor_mz=prec, adduct=add,
                             ionization_mode=mode, instrument_type=itype, ingest_lib=lib,
                             collision_energy_ev=[float(rng.choice([20.0, 40.0]))],
                             base_peak_intensity=1e5, precursor_error_ppm=1.0,
                             inchikey14=ik, normalized_smiles=f"C{s}", molecular_formula=f,
                             inchikey=ik + "-XXX"))
    rows.append({**rows[0], "spectrum_id": "no_ik", "inchikey14": None})
    rows.append({**rows[0], "spectrum_id": "empty", "ms2_mzs": np.zeros(0),
                 "ms2_normalized_intensities": np.zeros(0)})
    rows.append({**rows[0], "spectrum_id": "bad_formula", "inchikey14": "BADFORMULA0000",
                 "molecular_formula": "Xx?"})
    return pd.DataFrame(rows)


def test_build_train_schema_streaming_and_determinism(tmp_path):
    df = _synthetic_train()
    f = tmp_path / "train_like.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), f, row_group_size=17)
    a = Library.build(f, Config(), out_dir=tmp_path / "a", batch_rows=7)
    b = Library.build(df.sample(frac=1.0, random_state=1), Config(), out_dir=tmp_path / "b")
    assert a.info["n_input_rows"] == len(df)
    # drops: missing ik14 and empty spectrum; the bad formula falls back to the precursor
    assert a.info["drops"] == {"missing ik14": 1, "fewer than 1 peaks after cleaning": 1}
    assert a.n_structures == 41
    # same library whatever the input order and batching
    for x, y in ((a.store.mz, b.store.mz), (a.store.intensity, b.store.intensity),
                 (a.store.offsets, b.store.offsets), (a.cons_store.mz, b.cons_store.mz)):
        assert np.array_equal(x, y)
    cols = [c for c in SPECTRUM_COLUMNS if c != "spec_idx"]
    pd.testing.assert_frame_equal(a.meta[cols], b.meta[cols])
    pd.testing.assert_frame_equal(a.structures, b.structures)
    # formula masses, families, instrument families
    from casmi.adducts import formula_mass
    s = a.structures.set_index("ik14")
    assert s.loc["SYNTH000000000", "mono_mass"] == pytest.approx(formula_mass("C6H12O6"))
    bad = s.loc["BADFORMULA0000"]
    assert abs(bad["mono_mass"] - formula_mass("C6H12O6")) < 0.01
    m = a.meta
    assert set(m["instrument_family"]) <= {"qtof", "orbitrap", "timstof"}
    assert (m.loc[m["ingest_lib"] == "pluskal_ms2", "instrument_family"] == "orbitrap").all()
    assert set(m["adduct_family"]) <= {"[M+H]+", "[M+Na]+", "[M-H]-"}
    # consensus key includes ingest_lib: no consensus mixes libraries
    c = a.cons_meta
    for _, row in c.iterrows():
        libs = set(m["ingest_lib"].iloc[row.spec_start:row.spec_end])
        assert libs == {row.ingest_lib}
    # reference precursor of an [M+H]+ family consensus is M + 1.007276
    h = c[c["adduct_family"] == "[M+H]+"].iloc[0]
    assert h.precursor_mz == pytest.approx(h.mono_mass + 1.007276, abs=1e-5)
    # self-search on a few spectra
    for i in range(0, a.n_spectra, 9):
        r = a.search_by_mass(a.struct_mass[a.spec_struct[i]], queries=a.queries_from_library([i]))
        assert r.best_ref_idx.iloc[0] == i


@pytest.mark.data
def test_train_sample_build_and_self_search(train_parquet, tmp_path):
    """Build from the first 100k train rows; 1,000 random spectra find themselves."""
    pf = pq.ParquetFile(train_parquet)
    tables, n = [], 0
    for batch in pf.iter_batches(batch_size=20_000):
        tables.append(pa.Table.from_batches([batch]))
        n += batch.num_rows
        if n >= 100_000:
            break
    lib = Library.build(pa.concat_tables(tables), Config(), out_dir=tmp_path / "lib")
    rng = np.random.default_rng(Config().seed)
    idx = rng.choice(lib.n_spectra, size=min(1000, lib.n_spectra), replace=False)
    t0 = time.perf_counter()
    for i in idx:
        r = lib.search_by_mass(lib.struct_mass[lib.spec_struct[i]],
                               queries=lib.queries_from_library([i]))
        assert r.cos_max.iloc[0] == pytest.approx(1.0)
        assert r.best_ref_idx.iloc[0] == i or lib.store.n_peaks[i] == lib.store.n_peaks[r.best_ref_idx.iloc[0]]
    print(f"\ntrain sample: {lib.n_spectra} spectra, {len(idx)} self-searches in "
          f"{time.perf_counter() - t0:.1f} s")


# --- raw-spectrum metadata and precleaned input (integration fix) -----------------------


def _by_id(lib: Library) -> pd.DataFrame:
    m = lib.meta.copy()
    m["mz"] = [lib.store[i][0] for i in range(lib.n_spectra)]
    m["it"] = [lib.store[i][1] for i in range(lib.n_spectra)]
    return m.set_index("spectrum_id").sort_index()


def _cleaned_like_wp6(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, str]:
    """Clean every row as scripts/ingest_train.py does (float32 peaks, raw n_peaks_raw and
    prec_intensity columns) and add a peak at float32(0.005) relative intensity to one
    spectrum: a valid cleaned peak that a second cleaning drops (0.0049999999 < 0.005)."""
    from casmi.spectra import clean

    out = df.copy()
    mzs, its, nraw, pint = [], [], [], []
    for r in df.itertuples(index=False):
        m, x, p = clean(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz, cfg)
        mzs.append(m), its.append(x), nraw.append(len(r.ms2_mzs)), pint.append(p)
    k = next(i for i, m in enumerate(mzs) if len(m) >= 2 and df["inchikey14"].iloc[i])
    mid = np.float32((float(mzs[k][0]) + float(mzs[k][1])) / 2)
    mzs[k] = np.insert(mzs[k], 1, mid)
    its[k] = np.insert(its[k], 1, np.float32(0.005))
    out["ms2_mzs"], out["ms2_normalized_intensities"] = mzs, its
    out["n_peaks_raw"] = np.asarray(nraw, np.int32)
    out["prec_intensity"] = np.asarray(pint, np.float32)
    return out, str(df["spectrum_id"].iloc[k])


def test_build_from_raw_computes_raw_metadata(tmp_path):
    from casmi.spectra import clean

    df = _synthetic_train()
    lib = Library.build(df, Config(), out_dir=tmp_path / "raw")
    assert lib.info["precleaned"] is False
    m = _by_id(lib)
    src = df.set_index("spectrum_id")
    for sid, row in m.iterrows():
        r = src.loc[sid]
        assert row.n_peaks_raw == len(r.ms2_mzs)
        p = clean(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz)[2]
        assert row.prec_intensity == pytest.approx(p, abs=1e-6)
    assert (m["n_peaks_raw"] > m["n_peaks"]).all() and (m["prec_intensity"] > 0).any()


def test_build_from_cleaned_keeps_raw_metadata(tmp_path):
    cfg = Config()
    df = _synthetic_train()
    raw = _by_id(Library.build(df, cfg, out_dir=tmp_path / "raw"))
    cleaned, sid = _cleaned_like_wp6(df, cfg)
    pre_lib = Library.build(cleaned, cfg, out_dir=tmp_path / "pre", batch_rows=11, precleaned=True)
    assert pre_lib.info["precleaned"] is True
    pre = _by_id(pre_lib)
    assert pre.index.equals(raw.index)
    # raw-spectrum metadata survive (they equal a build from the raw spectra)
    np.testing.assert_array_equal(pre["n_peaks_raw"], raw["n_peaks_raw"])
    np.testing.assert_array_equal(pre["prec_intensity"], raw["prec_intensity"])
    # stored peaks are used as given: identical to the raw build, plus the floor peak
    src = cleaned.set_index("spectrum_id")
    for s in pre.index:
        assert np.array_equal(pre.at[s, "mz"], np.asarray(src.at[s, "ms2_mzs"], np.float32))
        assert np.array_equal(pre.at[s, "it"], np.asarray(src.at[s, "ms2_normalized_intensities"],
                                                        np.float32))
        if s != sid:
            assert np.array_equal(pre.at[s, "mz"], raw.at[s, "mz"])
    assert pre.at[sid, "n_peaks"] == raw.at[sid, "n_peaks"] + 1
    # cleaning the cleaned peaks again is not idempotent (the floor peak goes), but the
    # raw-spectrum columns are still kept
    again = _by_id(Library.build(cleaned, cfg, out_dir=tmp_path / "again"))
    assert again.at[sid, "n_peaks"] == raw.at[sid, "n_peaks"]
    np.testing.assert_array_equal(again["n_peaks_raw"], raw["n_peaks_raw"])
    np.testing.assert_array_equal(again["prec_intensity"], raw["prec_intensity"])
    # precleaned input without the two columns: stored count and NaN
    bare = _by_id(Library.build(cleaned.drop(columns=["n_peaks_raw", "prec_intensity"]), cfg,
                                out_dir=tmp_path / "bare", precleaned=True))
    assert (bare["n_peaks_raw"] == bare["n_peaks"]).all() and bare["prec_intensity"].isna().all()
    # raw spectra declared precleaned are refused
    with pytest.raises(ValueError, match="precleaned"):
        Library.build(df, cfg, out_dir=tmp_path / "bad", precleaned=True)


def test_ingest_library_stage_keeps_raw_metadata(synthetic_train, caplog):
    """scripts/ingest_train.py --stage library builds from WP6's cleaned files without
    re-cleaning, so the library carries the ingested n_peaks_raw / prec_intensity and
    the fingerprint-model loader does not warn about lost raw metadata."""
    import logging

    import casmi.fpmodel as fm

    lib = synthetic_train["library"]
    assert lib.info["precleaned"] is True
    meta = pd.read_parquet(synthetic_train["out"] / "meta.parquet").set_index("spectrum_id")
    m = lib.meta.set_index("spectrum_id").loc[meta.index]
    np.testing.assert_array_equal(m["n_peaks_raw"], meta["n_peaks_raw"])
    np.testing.assert_array_equal(m["prec_intensity"], meta["prec_intensity"])
    np.testing.assert_array_equal(m["n_peaks"], meta["n_peaks"])
    with caplog.at_level(logging.WARNING, logger="casmi.fpmodel"):
        fm.TrainingData.from_library(lib.dir, synthetic_train["cfg"], targets_cache=None)
    assert not [r for r in caplog.records if "already-cleaned" in r.getMessage()]

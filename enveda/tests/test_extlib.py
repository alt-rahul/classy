"""Tests for casmi.extlib (ROADMAP R2): parsers, adduct mapping, filters, dedup, library build."""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from casmi import extlib
from casmi.adducts import TEST_ADDUCTS
from casmi.chem import mono_mass, std_mol
from casmi.config import Config

KNOWN = {"[M+H]+": 100, "[M-H]-": 50, "[M+Na]+": 20, "[M+CH2O2-H]-": 10, "[M+2H]2+": 3,
         "[M-H2O+H]+": 5, "[2M+H]+": 4, "[M]+": 2}


def test_iter_json_array_small_chunks():
    data = [{"a": "x]y,[z", "b": [1, 2, {"c": "}"}]}, {"peaks_json": "[[1.0,2.0],[3.0,4.0]]"}, {}]
    text = json.dumps(data, indent=4)
    for chunk in (3, 7, 64, 1 << 20):
        got = list(extlib.iter_json_array(io.StringIO(text), chunk_chars=chunk))
        assert got == data
    assert list(extlib.iter_json_array(io.StringIO("[]"))) == []


def test_adduct_mapper():
    m = extlib.AdductMapper(KNOWN)
    assert [n for n, _ in m.candidates("M+H", "positive")] == ["[M+H]+"]
    assert [n for n, _ in m.candidates("[M+HCOO]-", None)] == ["[M+CH2O2-H]-"]
    assert [n for n, _ in m.candidates("M+FA-H", "negative")] == ["[M+CH2O2-H]-"]
    assert [n for n, _ in m.candidates("[M+H-H2O]+", None)] == ["[M-H2O+H]+"]
    # charge-less "M+2H" may be charge 1 (unknown here) or 2
    assert "[M+2H]2+" in [n for n, _ in m.candidates("M+2H", "positive")]
    assert m.candidates("M+K", "positive") == []
    assert m.candidates("not an adduct", "positive") is None


def test_parse_collision_energy():
    pce = extlib.parse_collision_energy
    assert pce("20(NCE)", 500.0) == ([20.0], "NCE")
    assert pce("40 eV", 300.0) == ([40.0], "eV")
    assert pce("10V", 300.0) == ([10.0], "V")
    assert pce("Ramp 20-40 eV", 300.0) == ([20.0, 40.0], "eV")
    assert pce("20,40,60", 300.0, units_hint="eV") == ([20.0, 40.0, 60.0], "eV")
    assert pce("30", 250.0, units_hint="NCE") == ([15.0], "NCE")
    assert pce("30%", 250.0, charge=2) == ([13.5], "NCE")
    assert pce(None, 300.0) == (None, "unknown")
    assert pce("HCD", 300.0) == (None, "unknown")


def test_resolve_structure():
    st, smi, ik, f, m = extlib.resolve_structure(("smiles", "OC(=O)c1ccccc1O"))
    assert st == "ok" and ik.startswith("YGSDEFSMJLZEOE") and f == "C7H6O3"
    assert abs(m - 138.0317) < 1e-3
    assert extlib.resolve_structure(("smiles", "CC(=O)[O-].[Na+]"))[0] == "multi_component"
    assert extlib.resolve_structure(("smiles", "C[N+](C)(C)C"))[0] == "charged"
    assert extlib.resolve_structure(("smiles", "[13CH3]O"))[0] == "isotope_label"
    assert extlib.resolve_structure(("smiles", "C1CC"))[0] == "bad_structure"
    assert extlib.resolve_structure(("smiles", "CC(=O)[O-]"))[0] == "ok"   # neutralised
    st, smi, *_ = extlib.resolve_structure(("inchi", "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"))
    assert st == "ok" and smi == "CCO"


def _rec(sid, smiles, adduct, mode, prec=None, n_peaks=6, drop=None, shift=0.0, **kw):
    m = mono_mass(std_mol(smiles))
    a = TEST_ADDUCTS[adduct]
    p = a.precursor_mz(m) + shift if prec is None else prec
    mz = np.linspace(50.0, p - 5.0, n_peaks)
    it = np.linspace(10.0, 1000.0, n_peaks)
    return extlib.RawRecord(source_id=sid, sub_lib="lib", smiles=smiles, adduct=adduct, ionization_mode=mode,
                            precursor_mz=p, instrument_type="LC-ESI-QTOF", ms_level=2,
                            collision_energy="20 eV", mzs=mz, intensities=it, licence="CC0", drop=drop, **kw)


def test_stager_filters_and_schema(tmp_path):
    cfg = Config()
    m = extlib.AdductMapper(KNOWN)
    recs = [
        _rec("a", "OC(=O)c1ccccc1O", "[M+H]+", "positive"),
        _rec("b", "OC(=O)c1ccccc1O", "[M-H]-", "negative"),
        _rec("c", "OC(=O)c1ccccc1O", "[M+H]+", "positive", shift=0.05),          # ppm
        _rec("d", "OC(=O)c1ccccc1O", "[M+H]+", "positive", n_peaks=2),           # peaks
        _rec("e", "OC(=O)c1ccccc1O", "[M+H]+", "positive", drop="nist"),
        _rec("f", "CC(=O)[O-].[Na+]", "[M+H]+", "positive", prec=83.0),
        _rec("g", "OC(=O)c1ccccc1O", "[M+H]+", "negative"),                       # polarity
        _rec("a", "OC(=O)c1ccccc1O", "[M+H]+", "positive"),                       # duplicate
        _rec("h", "OC(=O)c1ccccc1O", "[M+K]+", "positive", prec=177.0),
    ]
    recs.append(extlib.RawRecord(source_id="i", smiles=None, adduct="[M+H]+", ionization_mode="positive",
                                 precursor_mz=100.0, mzs=np.array([1.0, 2.0, 3.0]),
                                 intensities=np.array([1.0, 2.0, 3.0])))
    with extlib.Stager("unit", cfg, m, tmp_path / "st", n_jobs=1) as st:
        st.add_all(recs)
    info = st.info()
    d = info["drops"]
    assert info["n_staged"] == 2
    assert d["ppm_gt_max"] == 1 and d["lt_min_peaks"] == 1 and d["nist"] == 1
    assert d["multi_component"] == 1 and d["polarity_mismatch"] == 1
    assert d["duplicate_in_source"] == 1 and d["adduct_unknown"] == 1 and d["no_structure"] == 1
    t = pq.read_table(tmp_path / "st" / "part-000.parquet")
    assert t.schema.equals(extlib.STAGE_SCHEMA)
    df = t.to_pandas()
    assert list(df["adduct"]) == ["[M+H]+", "[M-H]-"]
    assert (df["ingest_lib"] == "ext_unit").all()
    assert (df["precursor_error_ppm"] < 1.0).all()
    assert (df["inchikey14"] == "YGSDEFSMJLZEOE").all()
    assert df["spectrum_id"].tolist() == ["ext_unit:a", "ext_unit:b"]
    assert df["collision_energy_orig_units"].tolist() == ["eV", "eV"]
    assert df["base_peak_intensity"].isna().all()           # max 1000 = normalised scale
    assert df["instrument_family"].tolist() == ["qtof", "qtof"]
    assert all(abs(max(x) - 1.0) < 1e-6 for x in df["ms2_normalized_intensities"])


def test_nist_flag():
    assert extlib.nist_flag("NIST14", library="GNPS-LIBRARY")
    assert not extlib.nist_flag("Commercial", library="GNPS-LIBRARY")
    assert not extlib.nist_flag("NIST14", library="GNPS-NIST14-MATCHES")
    assert extlib.nist_flag(None, library="NIST Glycan Reference Library")


def test_massbank_msp_parser(tmp_path):
    text = """Name: Salicylic acid
DB#: MSBNK-TEST-XX000001
InChIKey: YGSDEFSMJLZEOE-UHFFFAOYSA-N
SMILES: OC(=O)c1ccccc1O
Precursor_type: [M-H]-
Spectrum_type: MS2
PrecursorMZ: 137.0244
Instrument_type: LC-ESI-QTOF
Instrument: maXis
Ion_mode: NEGATIVE
Collision_energy: 20(NCE)
Comments: Parent=137.0244
Num Peaks: 3
65.0 10
93.0344 999
137.0244 50

Name: EI record
DB#: MSBNK-TEST-XX000002
SMILES: CCO
Spectrum_type: MS1
Num Peaks: 1
31.0 999
"""
    p = tmp_path / "mb.msp"
    p.write_text(text)
    recs = list(extlib.iter_massbank_msp(p, {"MSBNK-TEST-XX000001": "CC BY"}))
    assert len(recs) == 2
    r = recs[0]
    assert r.source_id == "MSBNK-TEST-XX000001" and r.sub_lib == "TEST" and r.licence == "CC BY"
    assert r.adduct == "[M-H]-" and r.ms_level == 2 and r.instrument_type == "LC-ESI-QTOF; maXis"
    assert np.allclose(r.mzs, [65.0, 93.0344, 137.0244]) and r.intensities[1] == 999
    assert recs[1].ms_level == 1


def test_mona_and_gnps_parsers(tmp_path):
    rec = {"id": "X1", "compound": [{"kind": "biological", "inchi": "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
                                     "metaData": [{"name": "SMILES", "value": "CCO", "computed": False}]}],
           "metaData": [{"name": "ms level", "value": "MS2"}, {"name": "precursor type", "value": "[M+H]+"},
                        {"name": "precursor m/z", "value": "47.0491"}, {"name": "ionization mode", "value": "positive"},
                        {"name": "instrument type", "value": "LC-ESI-QTOF"}, {"name": "collision energy", "value": "10 eV"}],
           "spectrum": "29.0:10 31.0:100 47.0:5", "tags": [{"text": "Lib1", "ruleBased": False}],
           "library": {"library": "Lib1"}}
    skip = dict(rec, id="X2", tags=[{"text": "Enveda-180", "ruleBased": False}])
    insil = dict(rec, id="X3", tags=[{"text": "In-Silico", "ruleBased": False}])
    nist = dict(rec, id="X4", library={"library": "NIST Glycan Reference Library"})
    zp = tmp_path / "m.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("m.json", "[\n" + ",\n".join(json.dumps(r, separators=(",", ":")) for r in (rec, skip, insil, nist)) + "\n]")
    out = list(extlib.iter_mona_json(zp))
    assert [r.drop for r in out] == [None, "skipped_subset", "in_silico", "nist"]
    assert out[0].smiles == "CCO" and out[0].ms_level == 2 and out[0].sub_lib == "Lib1"
    assert np.allclose(out[0].mzs, [29.0, 31.0, 47.0])
    g = [{"spectrum_id": "CCMSLIB1", "library_membership": "GNPS-LIBRARY", "Smiles": "CCO", "INCHI": "N/A",
          "Adduct": "M+H", "Ion_Mode": "Positive", "Precursor_MZ": "47.0491", "Instrument": "qTof",
          "ms_level": "2", "Compound_Source": "Commercial", "peaks_json": "[[29.0,10.0],[31.0,100.0]]"},
         {"spectrum_id": "CCMSLIB2", "library_membership": "GNPS-LIBRARY", "Compound_Source": "NIST14",
          "peaks_json": "[]"}]
    gp = tmp_path / "g.json"
    gp.write_text(json.dumps(g, indent=2))
    out = list(extlib.iter_gnps_json(gp))
    assert out[0].adduct == "M+H" and out[0].inchi is None and out[0].licence.startswith("CC0")
    assert np.allclose(out[0].intensities, [10.0, 100.0])
    assert out[1].drop == "nist"


def test_dedup_rules():
    cfg = Config()
    base_mz = np.array([50.0, 80.0, 120.0], np.float32)
    other_mz = np.array([55.0, 85.0, 125.0], np.float32)
    it = np.array([0.2, 1.0, 0.5], np.float32)
    peaks_by_row = {0: base_mz, 1: base_mz, 2: other_mz, 3: base_mz, 4: base_mz, 5: other_mz}
    meta = pd.DataFrame({
        "row": [0, 1, 2, 3, 4, 5],
        "source": ["s1", "s1", "s1", "s2", "s1", "s2"],
        "inchikey14": ["T", "T", "T", "N", "N", "N"],
        "adduct": ["[M+H]+"] * 6,
        "instrument_family": ["qtof", "orbitrap", "orbitrap", "qtof", "qtof", "qtof"],
        "priority": [0, 0, 0, 1, 0, 1],
    })

    def peaks(rows, sources):
        return extlib._flat([peaks_by_row[int(r)] for r in rows], [it for _ in rows])

    tpk = {"T|[M+H]+": extlib._flat([base_mz], [it])}
    dec = extlib.dedup(meta, peaks, {"T|[M+H]+|qtof"}, {"T|[M+H]+"}, {"T"}, lambda prs: tpk, cfg)
    # row 0: covered; row 1: same peaks as train -> dup_train; row 2: new peaks -> keep;
    # rows 3 and 4 (structure N, same peaks): the earlier source keeps it; row 5 differs -> keep
    assert dec.tolist() == ["train_covered", "dup_train", "keep", "dup_ext", "keep", "keep"]


def test_final_file_builds_library(tmp_path):
    from casmi.library import Library

    cfg = Config()
    m = extlib.AdductMapper(KNOWN)
    recs = [_rec(f"r{i}", s, a, "positive" if a.endswith("+") else "negative", n_peaks=8 + i)
            for i, (s, a) in enumerate([("OC(=O)c1ccccc1O", "[M+H]+"), ("CCCCCCCCCC(=O)O", "[M-H]-"),
                                        ("c1ccc2[nH]ccc2c1", "[M+H]+"), ("OC(=O)c1ccccc1O", "[M-H]-")])]
    with extlib.Stager("unit", cfg, m, tmp_path / "st", n_jobs=1) as st:
        st.add_all(recs)
    t = pq.read_table(tmp_path / "st" / "part-000.parquet")
    import pyarrow as pa

    t = t.append_column("in_train", pa.array([False] * t.num_rows)).append_column(
        "mkey", pa.array([None] * t.num_rows, pa.string()))
    t = t.select(list(extlib.FINAL_COLUMNS))
    d = tmp_path / "final"
    d.mkdir()
    pq.write_table(t, d / "extlib_unit.parquet")
    for pre in (True, False):
        lib = Library.build(d, cfg, out_dir=tmp_path / f"lib_{pre}", precleaned=pre)
        assert lib.n_spectra == 4 and lib.n_structures == 3
        assert set(lib.meta["ingest_lib"]) == {"ext_unit"}
        assert lib.meta["spectrum_id"].str.startswith("ext_unit:").all()

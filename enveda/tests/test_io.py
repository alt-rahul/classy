"""WP1: read_test, molecule_table (PLAN 3.2), write_submission, validate_submission."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from casmi import io, paths
from casmi.adducts import TEST_ADDUCTS
from casmi.config import Config
from casmi.io import SubmissionError, validate_submission, write_submission


@pytest.fixture(scope="module")
def test_df(public_test_parquet):
    return io.read_test(public_test_parquet)


def test_read_test_types(test_df):
    assert len(test_df) == 1213 and test_df["molecule_id"].nunique() == 400
    assert list(test_df.columns) == list(io.TEST_COLUMNS)
    assert test_df["precursor_mz"].dtype == np.float64
    assert test_df["ms2_mzs"].iloc[0].dtype == np.float32
    assert test_df["ms2_normalized_intensities"].iloc[0].dtype == np.float32
    assert test_df["collision_energy_ev"].iloc[0].dtype == np.float64
    assert set(test_df["ionization_mode"]) == {"positive", "negative"}
    keys = list(zip(test_df["molecule_id"], test_df["spectrum_id"]))
    assert keys == sorted(keys)


def test_molecule_table_public(test_df):
    mol = io.molecule_table(test_df, Config())
    assert len(mol) == 400 and mol["molecule_id"].is_unique
    assert mol["molecule_id"].tolist() == sorted(mol["molecule_id"])
    assert 246.10 <= mol["M"].min() < 246.11 and 439.10 < mol["M"].max() <= 439.11
    assert mol["m_spread_ppm"].max() == pytest.approx(5.5, abs=0.1)
    assert (mol["n_adducts"] > 1).sum() == 108
    assert (mol["has_pos"] & mol["has_neg"]).sum() == 97
    assert mol["n_spectra"].sum() == 1213
    assert not mol["mass_clustered"].any() and (mol["n_outliers"] == 0).all()
    assert (mol["n_unknown_adduct"] == 0).all()
    spec = io.spectrum_table(test_df)
    assert spec.index.equals(test_df.index)
    assert spec["adduct_known"].all() and not spec["mass_outlier"].any()
    assert spec["m_dev_ppm"].abs().max() < 6


def _toy(rows):
    return pd.DataFrame(rows, columns=["molecule_id", "spectrum_id", "adduct", "ionization_mode",
                                       "precursor_mz", "base_peak_intensity"])


def test_molecule_table_cluster_fallback():
    M = 302.042653  # quercetin
    h, na, mh = (TEST_ADDUCTS[a].shift for a in ("[M+H]+", "[M+Na]+", "[M-H]-"))
    pos, neg = "positive", "negative"
    df = _toy([
        ("a", "s1", "[M+H]+", pos, M + h, 1e5),
        ("a", "s2", "[M+H]+", pos, M * (1 + 2e-6) + h, 1e5),
        ("a", "s3", "[M-H]-", neg, M + mh, 1e5),
        ("a", "s4", "[M+Na]+", pos, M + h, 1e5),       # mislabelled: really [M+H]+
        ("b", "s5", "[M+Na]+", pos, 200.0 + na, 1e9),  # two disagreeing spectra:
        ("b", "s6", "[M-H]-", neg, 200.01 + mh, 1e3),  # the [M-H]- cluster wins the tie
        ("c", "s7", "[M+Foo]+", pos, 150.0 + h, 1e4),  # unknown adduct -> [M+H]+ fallback
        ("d", "s8", "[M+2H]2+", pos, (180.0 + 2 * h) / 2, 1e4),  # general parser
    ])
    mol, spec = io.mass_tables(df)
    mol = mol.set_index("molecule_id")
    assert mol.loc["a", "M"] == pytest.approx(M, rel=2e-6)
    assert bool(mol.loc["a", "mass_clustered"]) and mol.loc["a", "n_outliers"] == 1
    assert mol.loc["a", "m_spread_ppm"] == pytest.approx(2.0, abs=0.01)
    assert spec["mass_outlier"].tolist() == [False, False, False, True, True, False, False, False]
    assert mol.loc["a", "n_adducts"] == 3 and mol.loc["a", "adducts"] == "[M+H]+;[M+Na]+;[M-H]-"
    assert bool(mol.loc["a", "has_pos"]) and bool(mol.loc["a", "has_neg"])
    assert mol.loc["b", "M"] == pytest.approx(200.01, abs=1e-9)
    assert mol.loc["c", "M"] == pytest.approx(150.0, abs=1e-9)
    assert mol.loc["c", "n_unknown_adduct"] == 1 and not spec["adduct_known"].iloc[6]
    assert mol.loc["d", "M"] == pytest.approx(180.0, abs=1e-9)
    assert spec["adduct_known"].iloc[7] and spec["charge"].iloc[7] == 2
    # within 15 ppm of the median: no clustering, nothing flagged
    df2 = _toy([("x", "t1", "[M+H]+", pos, M + h, 1.0),
                ("x", "t2", "[M+H]+", pos, M * (1 + 2e-5) + h, 1.0)])
    m2 = io.molecule_table(df2)
    assert not m2["mass_clustered"].iloc[0]
    assert m2["m_spread_ppm"].iloc[0] == pytest.approx(20.0, abs=0.01)


def test_sample_submission_is_valid(test_df):
    rep = validate_submission(paths.SAMPLE_SUBMISSION, test_df=test_df)
    assert rep["n_rows"] == 400 and rep["max_guesses"] <= 25 and rep["warnings"] == []


def _good(test_df):
    ids = sorted(test_df["molecule_id"].unique())
    return {m: ["CCO", "c1ccccc1", "OC(=O)C"] for m in ids}


def test_write_submission_deterministic(tmp_path, test_df):
    preds = _good(test_df)
    preds[sorted(preds)[0]] = ["C[C@H](N)C(=O)O"] * 25
    p1, p2 = tmp_path / "a.csv", tmp_path / "b.csv"
    write_submission(preds, p1, test_df=test_df)
    shuffled = dict(sorted(preds.items(), reverse=True))
    write_submission(shuffled, p2, test_df=test_df)
    b1 = p1.read_bytes()
    assert b1 == p2.read_bytes()
    lines = b1.decode().split("\n")
    assert lines[0] == "molecule_id,smiles" and lines[-1] == ""
    assert lines[1] == f"{sorted(preds)[0]},{';'.join(['C[C@H](N)C(=O)O'] * 25)}"
    assert len(lines) == 402
    assert b"\r" not in b1
    rep = validate_submission(p1, test_df=test_df)
    assert rep["n_rows"] == 400
    # DataFrame input and ";"-joined strings give the same bytes
    df = pd.DataFrame({"molecule_id": list(preds), "smiles": [";".join(v) for v in preds.values()]})
    write_submission(df, tmp_path / "c.csv")
    assert (tmp_path / "c.csv").read_bytes() == b1


def test_write_submission_refuses_invalid(tmp_path, test_df):
    out = tmp_path / "bad.csv"
    for preds in ({"m1": []}, {"m1": ["C", ""]}, {"m1": ["C"] * 26}, {"m1": ["C,C"]}):
        with pytest.raises(SubmissionError):
            write_submission(preds, out)
        assert not out.exists()
    with pytest.raises(SubmissionError):
        write_submission({"m_005e53": ["C"]}, out, test_df=test_df)   # 399 ids missing


def _csv(tmp_path, text, name="s.csv"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_validate_rejections(tmp_path, test_df):
    ids = sorted(test_df["molecule_id"].unique())
    good = "molecule_id,smiles\n" + "".join(f"{m},CCO;CC\n" for m in ids)
    assert validate_submission(_csv(tmp_path, good), test_df=test_df)["n_rows"] == 400

    bad = {
        "missing column": "molecule_id,smile\nm1,CCO\n",
        "only id column": "molecule_id\nm1\n",
        "empty file": "",
        "header only": "molecule_id,smiles\n",
        "empty cell": "molecule_id,smiles\nm1,CCO\nm2,\n",
        "null id": "molecule_id,smiles\nm1,CCO\n,CCO\n",
        "repeated id": "molecule_id,smiles\nm1,CCO\nm1,CC\n",
        "26 guesses": "molecule_id,smiles\nm1," + ";".join(["C"] * 26) + "\n",
    }
    for case, text in bad.items():
        with pytest.raises(SubmissionError):
            validate_submission(_csv(tmp_path, text, case.replace(" ", "_") + ".csv"))
    assert validate_submission(_csv(tmp_path, "molecule_id,smiles\nm1," + ";".join(["C"] * 25)
                                    + "\n"))["max_guesses"] == 25

    # ids versus the test file
    missing_one = "molecule_id,smiles\n" + "".join(f"{m},CCO\n" for m in ids[1:])
    with pytest.raises(SubmissionError, match="missing"):
        validate_submission(_csv(tmp_path, missing_one), test_df=test_df)
    extra_one = good + "m_zzzzzz,CCO\n"
    with pytest.raises(SubmissionError, match="not in the test file"):
        validate_submission(_csv(tmp_path, extra_one), test_df=test_df)
    with pytest.raises(SubmissionError):
        validate_submission(pd.DataFrame({"molecule_id": [], "smiles": []}))

    # several problems are all reported
    with pytest.raises(SubmissionError) as exc:
        validate_submission(_csv(tmp_path, bad["repeated id"] + "m2," + ";".join(["C"] * 30)
                                 + "\n"))
    assert len(exc.value.errors) == 2
    # malformed CSV (a comma inside an unquoted cell)
    with pytest.raises(SubmissionError, match="CSV"):
        validate_submission(_csv(tmp_path, "molecule_id,smiles\nm1,CCO\nm2,C,C\n"))


def test_validate_warnings_only(tmp_path):
    text = "molecule_id,smiles,extra\nm1,CCO;not_a_smiles;C1CC,1\nm2,C;;CC,2\n"
    rep = validate_submission(_csv(tmp_path, text))
    assert rep["n_unparseable"] == 2
    assert len(rep["warnings"]) == 3   # unparseable, empty guess, extra column

"""WP1: standardisation, formula, mass, fingerprints, NP-likeness, batch helpers."""

from __future__ import annotations

import gzip
import importlib.util
import os
import pickle

import numpy as np
import pytest
from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

from casmi import chem
from casmi.config import Config

QUERCETIN = "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"


def _nci(n: int) -> list[str]:
    path = os.path.join(RDConfig.RDDataDir, "NCI", "first_5K.smi")
    with open(path) as fh:
        return [line.split()[0] for _, line in zip(range(n), fh)]


def test_std_mol_uncharges_strips_stereo_no_tautomers():
    cases = {
        "CC(=O)[O-]": "CC(=O)O",                         # uncharge
        "C[NH3+]": "CN",
        "C[N+](C)(C)CC(=O)[O-]": "C[N+](C)(C)CC(=O)[O-]",  # zwitterion stays (net 0)
        "C[C@H](O)CC": "CCC(C)O",                        # stereo stripped
        "C/C=C/C": "CC=CC",
        "[H]OC([H])([H])C": "CCO",                       # explicit H removed
        "Oc1ccccn1": "Oc1ccccn1",                        # no tautomer canonicalisation
        "CC(=O)CC(C)=O": "CC(=O)CC(C)=O",
    }
    for smi, want in cases.items():
        assert chem.std_smiles(smi) == want, smi
    # metal disconnection (Cleanup): two components
    out = chem.std_mol("CC(=O)O[Na]")
    assert out is not None and len(Chem.GetMolFrags(out)) == 2
    for bad in ("xyz", "", None, "C1CC"):
        assert chem.std_mol(bad) is None
        assert chem.std_smiles(bad) is None
    # accepts a Mol and does not modify it
    m = Chem.MolFromSmiles("C[C@H](N)C(=O)[O-]")
    before = Chem.MolToSmiles(m)
    assert chem.std_smiles(m) == "CC(N)C(=O)O"
    assert Chem.MolToSmiles(m) == before


def test_canonical_smiles():
    assert chem.canonical_smiles("C[C@H](N)C(=O)O") == "CC(N)C(=O)O"
    assert chem.canonical_smiles("C[C@H](N)C(=O)O", stereo=True) == "C[C@H](N)C(=O)O"
    assert chem.canonical_smiles("OC(=O)C") == "CC(=O)O"
    assert chem.canonical_smiles("bad") is None


def test_formula_and_mass():
    assert chem.formula(CAFFEINE) == "C8H10N4O2"
    assert chem.mono_mass(CAFFEINE) == pytest.approx(194.080376, abs=1e-6)
    assert chem.formula("OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl") == "C14H11Cl2NO2"
    assert chem.mono_mass("OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl") == pytest.approx(295.016684, abs=1e-6)
    assert chem.formula(None) is None
    assert np.isnan(chem.mono_mass("bad"))


def test_morgan_fp_layout_and_chirality_off():
    m = Chem.MolFromSmiles(QUERCETIN)
    fp = chem.morgan_fp(m)
    assert fp.dtype == np.uint8 and fp.shape == (512,)
    ref = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096).GetFingerprint(m)
    bits = chem.unpack_fp(fp, 4096)
    assert bits.dtype == bool and bits.shape == (4096,)
    assert list(np.flatnonzero(bits)) == list(ref.GetOnBits())
    # documented bit order: bit i is (fp[i // 8] >> (7 - i % 8)) & 1
    i = int(np.flatnonzero(bits)[0])
    assert (fp[i // 8] >> (7 - i % 8)) & 1 == 1
    assert np.array_equal(chem.morgan_fp(m, packed=False), bits)
    l_ala, d_ala = "C[C@@H](N)C(=O)O", "C[C@H](N)C(=O)O"
    assert np.array_equal(chem.morgan_fp(l_ala), chem.morgan_fp(d_ala))
    small = chem.morgan_fp(m, Config().replace(fp_morgan_bits=1024))
    assert small.shape == (128,)
    assert chem.morgan_fp("bad") is None


def test_maccs_fp():
    m = Chem.MolFromSmiles(QUERCETIN)
    fp = chem.maccs_fp(m)
    assert fp.dtype == np.uint8 and fp.shape == (21,)
    bits = chem.unpack_fp(fp, chem.MACCS_BITS)
    assert list(np.flatnonzero(bits)) == list(MACCSkeys.GenMACCSKeys(m).GetOnBits())
    assert np.array_equal(chem.maccs_fp(m, packed=False), bits)


def test_tanimoto_matches_rdkit():
    smis = _nci(40)
    fps, ok = chem.morgan_fps(smis)
    assert ok.all()
    got = chem.tanimoto(fps[0], fps)
    bvs = []
    for f in fps:
        bv = DataStructs.ExplicitBitVect(4096)
        bv.SetBitsFromList([int(i) for i in np.flatnonzero(chem.unpack_fp(f, 4096))])
        bvs.append(bv)
    ref = np.array(DataStructs.BulkTanimotoSimilarity(bvs[0], bvs))
    assert np.allclose(got, ref, atol=1e-6)
    assert got[0] == pytest.approx(1.0)
    assert chem.tanimoto(np.zeros(512, np.uint8), np.zeros((2, 512), np.uint8)).tolist() == [0, 0]


def test_np_score_equals_rdkit_contrib():
    directory = os.path.join(RDConfig.RDContribDir, "NP_Score")
    spec = importlib.util.spec_from_file_location("np_ref", os.path.join(directory, "npscorer.py"))
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    with gzip.open(os.path.join(directory, "publicnp.model.gz"), "rb") as fh:
        model = pickle.load(fh)
    for smi in (QUERCETIN, CAFFEINE, "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl"):
        assert chem.np_score(smi) == ref.scoreMol(Chem.MolFromSmiles(smi), model)
    assert chem.np_score(QUERCETIN) > 1.0 > chem.np_score(CAFFEINE)
    assert np.isnan(chem.np_score("bad"))


def test_featurize_parallel_equals_serial_and_keeps_order():
    smis = _nci(300) + ["bad", "", "CC(=O)[O-].[Na+]"]
    smis = smis[::-1]
    one = chem.featurize(smis, fields=chem.FEATURE_FIELDS)
    two = chem.featurize(smis, fields=chem.FEATURE_FIELDS, n_jobs=2, chunk_size=64)
    assert set(one) == set(two) == {"ok", *chem.FEATURE_FIELDS}
    for key in one:
        a, b = one[key], two[key]
        if isinstance(a, list):
            assert a == b, key
        else:
            assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f"), key
    n = len(smis)
    assert one["morgan"].shape == (n, 512) and one["maccs"].shape == (n, 21)
    assert one["ok"].sum() == n - 2
    i_bad = smis.index("bad")
    assert not one["ok"][i_bad] and one["formula"][i_bad] is None
    assert np.isnan(one["mono_mass"][i_bad]) and not one["morgan"][i_bad].any()
    i_salt = smis.index("CC(=O)[O-].[Na+]")
    assert one["charge"][i_salt] == 0 and one["n_components"][i_salt] == 2
    # row-wise agreement with the single-molecule functions
    j = smis.index(_nci(1)[0])
    std = chem.std_mol(smis[j])
    assert one["std_smiles"][j] == Chem.MolToSmiles(std)
    assert np.array_equal(one["morgan"][j], chem.morgan_fp(std))
    assert one["np_score"][j] == chem.np_score(std)
    # thin wrappers
    fps, ok = chem.morgan_fps(_nci(5), packed=False)
    assert fps.shape == (5, 4096) and fps.dtype == bool and ok.all()
    mk, ok = chem.maccs_fps(_nci(5) + ["bad"], packed=False)
    assert mk.shape == (6, 167) and ok.tolist() == [True] * 5 + [False]
    assert chem.std_smiles_batch(["C[NH3+]", "bad"]) == ["CN", None]
    assert np.allclose(chem.np_scores([QUERCETIN]), [chem.np_score(QUERCETIN)])
    empty = chem.featurize([], fields=("formula", "morgan"))
    assert empty["morgan"].shape == (0, 512) and empty["formula"] == []
    with pytest.raises(ValueError):
        chem.featurize(["C"], fields=("nope",))

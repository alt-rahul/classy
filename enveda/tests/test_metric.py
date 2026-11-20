"""WP1: metric key (mkey), batch keys with timeout and disk cache, MRR@25."""

from __future__ import annotations

import os
import subprocess
import sys

import pandas as pd
import pytest
from rdkit import RDConfig, rdBase

from casmi import metric
from casmi.metric import ik14, mkey, mkeys, mrr_at_25, reciprocal_ranks

QUERCETIN = "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12"
QUERCETIN_ALT = "Oc1cc(O)c2c(c1)oc(-c1ccc(O)c(O)c1)c(O)c2=O"
# About 200 ms in TautomerEnumerator().Canonicalize on this machine.
SLOW = "CC(=O)CC(=O)CC(=O)CC(=O)CC(=O)CC(=O)CC(=O)CC(=O)C"


def _nci(n: int) -> list[str]:
    path = os.path.join(RDConfig.RDDataDir, "NCI", "first_5K.smi")
    with open(path) as fh:
        return [line.split()[0] for _, line in zip(range(n), fh)]


@pytest.fixture
def fresh_cache(tmp_path):
    metric._CACHES.clear()
    yield tmp_path / "cache" / "mkey.parquet"
    metric._CACHES.clear()


def test_keto_enol_same_key():
    keto, enol = "CC(=O)CC(=O)C", "CC(O)=CC(=O)C"
    assert ik14(keto) != ik14(enol)            # plain InChI keeps them apart
    assert mkey(keto) == mkey(enol) == "YRKCREAYFQTBPV"
    assert mkey("O=c1cccc[nH]1") == mkey("Oc1ccccn1")


def test_stereoisomers_same_key():
    assert mkey("C[C@@H](N)C(=O)O") == mkey("C[C@H](N)C(=O)O") == mkey("CC(N)C(=O)O")
    with_stereo = "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"
    no_stereo = "OCC1OC(O)C(O)C(O)C1O"
    assert mkey(with_stereo) == mkey(no_stereo) == "WQZGKKKJIJFFOK"
    assert mkey("C/C=C/C") == mkey("C/C=C\\C")


def test_bad_input():
    for bad in ("not a smiles", "", "C1CC"):
        assert mkey(bad) is None
        assert ik14(bad) is None
    assert metric.mkey_mol(None) is None


def test_mkeys_matches_mkey_and_caches(fresh_cache, monkeypatch):
    smis = _nci(60) + [QUERCETIN, QUERCETIN, "bad", "", None]
    ref = [mkey(s) if isinstance(s, str) else None for s in smis]
    got = mkeys(smis, n_jobs=2, timeout_s=10.0, cache_path=fresh_cache)
    assert got == ref
    df = pd.read_parquet(fresh_cache)
    assert list(df.columns) == ["smiles", "mkey", "status", "timeout_s", "rdkit"]
    assert df["smiles"].is_unique and df["smiles"].tolist() == sorted(df["smiles"])
    assert set(df["rdkit"]) == {rdBase.rdkitVersion}
    assert set(df["status"]) <= {"ok", "fail"} and len(df) == len(set(_nci(60))) + 2
    st = dict(zip(df["smiles"], df["status"]))
    assert st["bad"] == "fail"

    # A new process (simulated: empty in-memory cache) reads the file and computes nothing.
    metric._CACHES.clear()

    def boom(*a, **k):
        raise AssertionError("computed although cached")

    monkeypatch.setattr(metric, "_compute", boom)
    monkeypatch.setattr(metric, "_run_workers", boom)
    assert mkeys(smis, n_jobs=2, timeout_s=10.0, cache_path=fresh_cache) == ref
    assert mkeys(smis, n_jobs=1, timeout_s=None, cache_path=fresh_cache) == ref


def test_mkeys_in_process_without_cache():
    smis = _nci(20) + ["bad"]
    assert mkeys(smis, n_jobs=1, timeout_s=None, cache=False) == [mkey(s) for s in smis]
    res = mkeys(["CCO", "bad", None], n_jobs=1, timeout_s=None, cache=False, return_status=True)
    assert res == [(mkey("CCO"), "ok"), (None, "fail"), (None, "fail")]


def test_timeout_falls_back_to_plain_inchikey(fresh_cache):
    items = ["CCO", SLOW, "CC(=O)CC(=O)C"]
    res = mkeys(items, n_jobs=1, timeout_s=0.02, cache_path=fresh_cache, return_status=True)
    assert res[0] == (mkey("CCO"), "ok")
    assert res[1] == (ik14(SLOW), "timeout")
    assert res[2] == (mkey("CC(=O)CC(=O)C"), "ok")   # the replacement worker carried on
    # The fallback is reused for the same (or a shorter) timeout ...
    assert mkeys([SLOW], n_jobs=1, timeout_s=0.01, cache_path=fresh_cache,
                 return_status=True) == [(ik14(SLOW), "timeout")]
    # ... but recomputed when we are willing to wait longer, and then replaced on disk.
    assert mkeys([SLOW], n_jobs=1, timeout_s=None, cache_path=fresh_cache,
                 return_status=True) == [(mkey(SLOW), "ok")]
    metric._CACHES.clear()
    assert mkeys([SLOW], n_jobs=1, timeout_s=0.02, cache_path=fresh_cache,
                 return_status=True) == [(mkey(SLOW), "ok")]
    df = pd.read_parquet(fresh_cache).set_index("smiles")
    assert df.loc[SLOW, "status"] == "ok"


def test_deferred_flush(fresh_cache):
    mkeys(["CCO"], n_jobs=1, timeout_s=None, cache_path=fresh_cache, flush=False)
    assert not fresh_cache.exists()
    metric.flush_mkey_cache(fresh_cache)
    assert pd.read_parquet(fresh_cache)["smiles"].tolist() == ["CCO"]


def test_concurrent_writers_lose_nothing(fresh_cache):
    """Two processes flushing into one cache file: every entry survives."""
    a, b = _nci(120)[:60], _nci(120)[60:]
    code = ("import sys, json; from casmi.metric import mkeys; "
            "mkeys(json.loads(sys.argv[2]), n_jobs=1, timeout_s=None, cache_path=sys.argv[1], "
            "flush_every=5)")
    import json

    procs = [subprocess.Popen([sys.executable, "-c", code, str(fresh_cache), json.dumps(x)])
             for x in (a, b)]
    assert [p.wait(timeout=300) for p in procs] == [0, 0]
    df = pd.read_parquet(fresh_cache)
    assert set(df["smiles"]) == set(a) | set(b)
    assert not list(fresh_cache.parent.glob("*.tmp"))


# --- MRR@25 on a 5-molecule fixture -----------------------------------------------------

WRONG = ["C" * k for k in range(1, 30)]          # methane, ethane, ... (never the truth)

TRUTH = {
    "m1": QUERCETIN,
    "m2": "CC(=O)CC(=O)C",                        # keto form
    "m3": "C[C@@H](N)C(=O)O",                     # L-alanine
    "m4": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",           # caffeine, absent from pred
    "m5": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
}
PRED = {
    # rank 1, written differently -> 1
    "m1": [QUERCETIN_ALT] + WRONG[:5],
    # enol form at rank 3 after a wrong and an unparseable guess (both occupy ranks) -> 1/3
    "m2": "CCO;not_a_smiles;CC(O)=CC(=O)C;CC(=O)CC(=O)C",
    # D-alanine at rank 26: only the first 25 guesses count -> 0
    "m3": WRONG[:25] + ["C[C@H](N)C(=O)O"],
    # glucose without stereo at rank 25 -> 1/25
    "m5": WRONG[:24] + ["OCC1OC(O)C(O)C(O)C1O", "OCC1OC(O)C(O)C(O)C1O"],
    # not in truth: ignored
    "m6": [QUERCETIN],
}


def test_mrr_fixture():
    rr = reciprocal_ranks(PRED, TRUTH)
    assert rr == pytest.approx({"m1": 1.0, "m2": 1 / 3, "m3": 0.0, "m4": 0.0, "m5": 0.04})
    want = (1 + 1 / 3 + 0 + 0 + 1 / 25) / 5
    assert mrr_at_25(PRED, TRUTH) == pytest.approx(want, abs=1e-12)
    assert mrr_at_25(PRED, TRUTH) == pytest.approx(0.2746666667, abs=1e-9)
    # same with precomputed keys and with a process pool
    keys = {s: mkey(s) for s in [QUERCETIN_ALT, "CCO"]}
    assert mrr_at_25(PRED, TRUTH, keys=keys) == pytest.approx(want, abs=1e-12)
    assert mrr_at_25(PRED, TRUTH, n_jobs=2) == pytest.approx(want, abs=1e-12)


def test_mrr_edge_cases():
    assert mrr_at_25({"a": ["CCO"]}, {"a": "OCC"}) == 1.0
    assert mrr_at_25({"a": ["CCC"]}, {"a": "OCC"}) == 0.0
    assert mrr_at_25({}, {"a": "OCC", "b": "C"}) == 0.0
    assert mrr_at_25({"a": "bad;OCC"}, {"a": "OCC"}) == 0.5
    assert mrr_at_25({"a": ["OCC"]}, {"a": "unparseable truth"}) == 0.0
    assert reciprocal_ranks({"a": WRONG[:9] + ["OCC"]}, {"a": "CCO"}, k=9) == {"a": 0.0}
    with pytest.raises(ValueError):
        mrr_at_25({"a": ["C"]}, {})
    assert metric.split_guesses(" CCO ; C ") == ["CCO", "C"]
    assert metric.split_guesses(None) == [] and metric.split_guesses(float("nan")) == []


# --- parity with the official metric (docs/METRIC_NOTEBOOK.md) ------------------------

def _official_ik14(smiles):
    """Verbatim logic of canonicalize_smiles_to_ik14 from the official metric notebook."""
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    with rdBase.BlockLogs():
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            Chem.RemoveStereochemistry(mol)
            canonical = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
            return Chem.MolToInchiKey(canonical)[:14]
        except (ValueError, RuntimeError, TypeError):
            return None


def test_official_stereo_pinned_enone_regression():
    # From the official notebook's doctests: before stereo was stripped first these two
    # canonicalised to different tautomers (UMCVTKGFWFSNGJ vs PUSIBJUZNMNGHT).
    a = mkey("C=C(C)C1CCC2CCC(=O)C=C2C1")
    b = mkey("C=C(C)[C@@H]1CCC2CCC(=O)C=C2C1")
    assert a == b == "UMCVTKGFWFSNGJ"
    assert mkey("C[C@H](N)C(=O)O") == mkey("CC(N)C(=O)O")


def test_mkey_does_not_mutate_input_mol():
    from rdkit import Chem

    mol = Chem.MolFromSmiles("C=C(C)[C@@H]1CCC2CCC(=O)C=C2C1")
    before = Chem.MolToSmiles(mol)
    metric.mkey_mol(mol)
    assert Chem.MolToSmiles(mol) == before


def test_parity_with_official_metric_nci():
    smiles = _nci(400)
    mismatches = [s for s in smiles if mkey(s) != _official_ik14(s)]
    assert not mismatches, mismatches[:5]


@pytest.mark.data("data/processed/db/np/np.parquet")
def test_parity_with_official_metric_natural_products():
    import pyarrow.parquet as pq

    from casmi import paths

    tab = pq.read_table(paths.PROCESSED / "db" / "np" / "np.parquet", columns=["smiles"])
    smiles = tab.column("smiles").to_pylist()[::2000][:400]  # stereo-rich, deterministic
    mismatches = [s for s in smiles if mkey(s) != _official_ik14(s)]
    assert not mismatches, mismatches[:5]

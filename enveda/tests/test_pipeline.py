"""Tests for casmi.pipeline (WP4): stage registry, Baseline 0 stages, assembly rules,
failure isolation, determinism, the inference-path import rule and scripts/predict.py.

Most tests run on a synthetic candidate database written to tmp_path (real SMILES, so
canonicalisation and metric keys are exercised); tests marked ``data`` use the real NP
tier and the public test file.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from casmi import chem, io, metric, paths, pipeline
from casmi.adducts import precursor_mz
from casmi.config import Config
from casmi.db import CandidateDB, shard_name

QUERCETIN = "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12"
MORIN = "O=c1c(O)c(-c2ccc(O)cc2O)oc2cc(O)cc(O)c12"
TRICETIN = "O=c1cc(-c2cc(O)c(O)c(O)c2)oc2cc(O)cc(O)c12"
ISO_PC = "O=c1c(O)c(-c2cccc(O)c2O)oc2cc(O)cc(O)c12"       # same formula, PubChem-only
ISO_PC2 = "O=c1c(O)c(-c2ccccc2)oc2c(O)c(O)c(O)c(O)c12"    # same formula, PubChem-only
KETO = "CC(=O)Cc1ccccc1"                                  # tautomers: same metric key,
ENOL = "CC(O)=Cc1ccccc1"                                  # different InChIKey
CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
IBUPROFEN_R = "CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1"

NP_SPECIAL = [
    # smiles, priors
    (QUERCETIN, dict(in_lotus=True, in_pubchemlite=True, coconut_annotation_level=5,
                     pubmed_count=10000, n_organisms=300, n_citations=50)),
    (MORIN, dict(coconut_annotation_level=4, pubmed_count=500)),
    (TRICETIN, dict(coconut_annotation_level=2, pubmed_count=20)),
    (KETO, dict(pubmed_count=5)),
    (ENOL, dict()),
    (CAFFEINE, dict(pubmed_count=30000)),
    (IBUPROFEN_R, dict(pubmed_count=8000)),
]
# Filler structures spread over 60-600 Da (padding source); none within 20 ppm of the
# masses the tests query.
FILLER = ([("C" * n) + "O" for n in range(3, 41)] + [("C" * n) + "C(=O)O" for n in range(2, 41)]
          + ["c1ccccc1" + "C" * n for n in range(1, 41)] + [("C" * n) + "N" for n in range(3, 41)])

TEST_CFG = Config().replace(db_tiers=("np",), n_jobs=1, mkey_cache=False, mkey_timeout_s=0.0,
                            batch_molecules=2)


def _np_table() -> pd.DataFrame:
    rows = []
    for smi, pri in NP_SPECIAL + [(s, {}) for s in FILLER]:
        r = dict(ik14=metric.ik14(smi), smiles=smi, formula=chem.formula(smi),
                 mono_mass=chem.mono_mass(smi), in_coconut=True, in_npatlas=False,
                 in_lotus=False, in_pubchemlite=False, coconut_annotation_level=None,
                 n_organisms=None, n_citations=None, n_collections=1, pubmed_count=None,
                 patent_count=None, anno_type_count=None, cid=None)
        r.update(pri)
        rows.append(r)
    df = pd.DataFrame(rows).sort_values(["mono_mass", "ik14"], kind="mergesort")
    for c in ("n_organisms", "n_citations", "n_collections", "pubmed_count", "patent_count",
              "anno_type_count"):
        df[c] = df[c].astype("Int32")
    df["coconut_annotation_level"] = df["coconut_annotation_level"].astype("Int8")
    df["cid"] = df["cid"].astype("Int64")
    assert df["ik14"].is_unique
    return df.reset_index(drop=True)


def _write_pubchem(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    rows = [(QUERCETIN, 1, 50000, 3), (ISO_PC, 2, 10 ** 6, 1), (ISO_PC2, 3, 0, 1)]
    tbl = pd.DataFrame({
        "ik14": [metric.ik14(s) for s, *_ in rows], "smiles": [s for s, *_ in rows],
        "formula": [chem.formula(s) for s, *_ in rows],
        "mono_mass": [chem.mono_mass(s) for s, *_ in rows],
        "cid": np.array([r[1] for r in rows], dtype=np.int32),
        "pmid_count": np.array([r[2] for r in rows], dtype=np.int32),
        "n_cids": np.array([r[3] for r in rows], dtype=np.int16),
    }).sort_values(["mono_mass", "cid"], kind="mergesort")
    pq.write_table(pa.Table.from_pandas(tbl, preserve_index=False), d / shard_name(300))
    (d / "_index.json").write_text(json.dumps({"n_shards": 1}))
    return d


@pytest.fixture(scope="module")
def fixture_dirs(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("wp4db")
    npd = root / "processed" / "db" / "np"
    npd.mkdir(parents=True)
    pq.write_table(pa.Table.from_pandas(_np_table(), preserve_index=False), npd / "np.parquet")
    pc = _write_pubchem(root / "pubchem_fixture")
    return {"root": root, "processed": root / "processed", "np": npd / "np.parquet",
            "pubchem": pc, "no_pubchem": root / "no_pubchem"}


def _db(dirs, pubchem: bool = False) -> CandidateDB:
    return CandidateDB(np_path=dirs["np"], pubchem_dir=dirs["pubchem" if pubchem else "no_pubchem"])


def _spec(mid: str, k: int, mass: float, adduct: str, mode: str, mz: float | None = None) -> dict:
    return {
        "molecule_id": mid, "spectrum_id": f"{mid}_s{k}",
        "ms2_mzs": np.array([51.0, 77.0, 105.0], dtype=np.float32),
        "ms2_normalized_intensities": np.array([0.2, 0.5, 1.0], dtype=np.float32),
        "base_peak_intensity": 1.0e4, "adduct": adduct, "ionization_mode": mode,
        "instrument_type": "timsTOF",
        "precursor_mz": float(precursor_mz(mass, adduct) if mz is None else mz),
        "collision_energy_orig": "20", "collision_energy_ev": np.array([20.0]),
        "collision_energy_orig_units": "eV",
    }


def _test_df(extra: list[dict] | None = None) -> pd.DataFrame:
    m = chem.mono_mass
    rows = [
        _spec("m_quer", 0, m(QUERCETIN), "[M+H]+", "positive"),
        _spec("m_quer", 1, m(QUERCETIN), "[M-H]-", "negative"),
        _spec("m_taut", 0, m(KETO), "[M+H]+", "positive"),
        # labelled [M+Na]+ but measured as [M+H]+: empty pool until the adduct retry
        _spec("m_wrong", 0, m(CAFFEINE), "[M+Na]+", "positive", mz=m(CAFFEINE) + 1.007276),
        _spec("m_stereo", 0, m(IBUPROFEN_R), "[M-H]-", "negative"),
        _spec("m_far", 0, 1100.0, "[M+H]+", "positive"),
    ] + (extra or [])
    return pd.DataFrame(rows)


def _check_lists(preds: dict, ids, k: int = 25) -> None:
    assert sorted(preds) == sorted(ids)
    for mid, g in preds.items():
        assert 1 <= len(g) <= k, mid
        keys = []
        for s in g:
            assert isinstance(s, str) and s and ";" not in s and "," not in s
            assert "@" not in s and "/" not in s and "\\" not in s, s       # no stereo
            assert chem.canonical_smiles(s) == s                            # canonical, re-parses
            keys.append(metric.mkey(s))
        assert len(set(keys)) == len(keys), mid                             # unique metric keys


@pytest.fixture()
def registry():
    saved = dict(pipeline.STAGES)
    yield pipeline.STAGES
    pipeline.STAGES.clear()
    pipeline.STAGES.update(saved)


# --- registry ----------------------------------------------------------------------------

def test_default_registry():
    assert pipeline.DEFAULT_STAGES == Config().pipeline_stages
    assert set(pipeline.DEFAULT_STAGES) <= set(pipeline.stage_names())
    assert pipeline.get_stage("pool") is pipeline.stage_pool
    with pytest.raises(KeyError):
        pipeline.get_stage("no_such_stage")


def test_register_replace_add_and_order(registry, fixture_dirs):
    with pytest.raises(ValueError):
        pipeline.register_stage("pool", pipeline.stage_pool, replace=False)

    @pipeline.register_stage("tag_rows")
    def tag_rows(state, cfg):
        state.pool["seen_by_tag_rows"] = True
        state.extras.setdefault("tag_rows", []).append(len(state.pool))
        return state

    seen = []

    def score_and_record(state, cfg):
        assert state.pool["seen_by_tag_rows"].all()
        seen.append(sorted(state.molecule_ids))
        return pipeline.stage_score(state, cfg)

    pipeline.register_stage("score", score_and_record)
    cfg = TEST_CFG.replace(pipeline_stages=("mass", "pool", "tag_rows", "features", "score",
                                            "assemble"))
    res = pipeline.run(_test_df(), cfg, db=_db(fixture_dirs))
    assert sorted(m for b in seen for m in b) == sorted(_test_df()["molecule_id"].unique())
    _check_lists(res.preds, _test_df()["molecule_id"].unique())

    with pytest.raises(KeyError):  # unknown stage name: configuration error, before any work
        pipeline.run(_test_df(), TEST_CFG.replace(pipeline_stages=("mass", "nope")),
                     db=_db(fixture_dirs))


def test_plugins_are_imported(registry, fixture_dirs, tmp_path, monkeypatch):
    (tmp_path / "wp4_fake_plugin.py").write_text(
        "from casmi import pipeline\n"
        "CALLS = []\n"
        "@pipeline.register_stage('features')\n"
        "def features(state, cfg):\n"
        "    CALLS.append(len(state.molecule_ids))\n"
        "    return pipeline.stage_features(state, cfg)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    cfg = TEST_CFG.replace(pipeline_plugins=("wp4_fake_plugin",), batch_molecules=50)
    pipeline.run(_test_df(), cfg, db=_db(fixture_dirs))
    import wp4_fake_plugin

    assert wp4_fake_plugin.CALLS == [5]
    with pytest.raises(ModuleNotFoundError):
        pipeline.run(_test_df(), TEST_CFG.replace(pipeline_plugins=("no_such_plugin_xyz",)),
                     db=_db(fixture_dirs))


# --- Baseline 0 behaviour ----------------------------------------------------------------

@pytest.fixture(scope="module")
def np_run(fixture_dirs):
    return pipeline.run(_test_df(), TEST_CFG, db=_db(fixture_dirs))


def test_lists_are_valid(np_run):
    _check_lists(np_run.preds, _test_df()["molecule_id"].unique())
    assert all(len(g) == 25 for g in np_run.preds.values())


def test_prior_order_and_padding(np_run):
    g = np_run.preds["m_quer"]
    assert g[:3] == [chem.canonical_smiles(s) for s in (QUERCETIN, MORIN, TRICETIN)]
    rep = np_run.molecules.set_index("molecule_id")
    r = rep.loc["m_quer"]
    assert (r.pool_attempt, r.pool_n, r.n_from_pool, r.n_pad_wide, r.n_pad_nearest) == \
        ("tol", 3, 3, 0, 22)
    assert rep.loc["m_far", "pool_n"] == 0 and rep.loc["m_far", "n_pad_nearest"] == 25


def test_tautomers_are_deduplicated(np_run):
    g = np_run.preds["m_taut"]
    both = {chem.canonical_smiles(KETO), chem.canonical_smiles(ENOL)}
    assert len(both & set(g)) == 1
    assert both & {g[0]}                      # the pool's only structure comes first
    assert np_run.molecules.set_index("molecule_id").loc["m_taut", "n_skip_dup_mkey"] >= 1


def test_wrong_adduct_retry(np_run):
    r = np_run.molecules.set_index("molecule_id").loc["m_wrong"]
    assert r.pool_n_first == 0 and r.pool_attempt == "default_adduct"
    assert abs(r.M_pool - chem.mono_mass(CAFFEINE)) < 1e-4
    assert np_run.preds["m_wrong"][0] == chem.canonical_smiles(CAFFEINE)


def test_stereo_removed(np_run):
    assert np_run.preds["m_stereo"][0] == "CC(C)Cc1ccc(C(C)C(=O)O)cc1"


def test_deterministic_and_batch_invariant(np_run, fixture_dirs, tmp_path):
    again = pipeline.run(_test_df(), TEST_CFG.replace(batch_molecules=50), db=_db(fixture_dirs))
    single = pipeline.run(_test_df(), TEST_CFG.replace(batch_molecules=1), db=_db(fixture_dirs))
    assert again.preds == np_run.preds == single.preds
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    io.write_submission(np_run.preds, a, test_df=_test_df())
    io.write_submission(again.preds, b, test_df=_test_df())
    assert a.read_bytes() == b.read_bytes()
    io.validate_submission(a, test_df=_test_df())


def test_exception_falls_back_to_padding(np_run, fixture_dirs):
    bad = [_spec("m_bad", 0, chem.mono_mass(QUERCETIN), "[M+H]+", "positive")]

    def features(state, cfg):
        if "m_bad" in state.molecule_ids:
            raise RuntimeError("injected failure")
        return pipeline.stage_features(state, cfg)

    res = pipeline.run(_test_df(bad), TEST_CFG, db=_db(fixture_dirs), stages={"features": features})
    _check_lists(res.preds, list(_test_df()["molecule_id"].unique()) + ["m_bad"])
    for mid, g in np_run.preds.items():
        assert res.preds[mid] == g
    r = res.molecules.set_index("molecule_id").loc["m_bad"]
    assert r.pool_attempt == "fallback" and "injected failure" in r.error
    # padding = widened window first, so the quercetin isomers still come first
    assert res.preds["m_bad"][0] == chem.canonical_smiles(QUERCETIN)
    assert (res.molecules.set_index("molecule_id").drop(index="m_bad")["error"] == "").all()


def test_never_empty_without_database(tmp_path):
    empty = tmp_path / "np_empty.parquet"
    pq.write_table(pa.Table.from_pandas(_np_table().iloc[:0], preserve_index=False), empty)
    for npp in (empty, tmp_path / "missing.parquet"):
        db = CandidateDB(np_path=npp, pubchem_dir=tmp_path / "none")
        res = pipeline.run(_test_df(), TEST_CFG, db=db)
        assert all(g == [pipeline.LAST_RESORT_SMILES] for g in res.preds.values())
        assert len(res.preds) == 5


def test_both_tiers_np_before_pubchem_only(fixture_dirs):
    cfg = TEST_CFG.replace(db_tiers=("np", "pubchem"))
    res = pipeline.run(_test_df(), cfg, db=_db(fixture_dirs, pubchem=True))
    _check_lists(res.preds, _test_df()["molecule_id"].unique())
    g = res.preds["m_quer"]
    assert g[:5] == [chem.canonical_smiles(s) for s in (QUERCETIN, MORIN, TRICETIN, ISO_PC, ISO_PC2)]
    r = res.molecules.set_index("molecule_id").loc["m_quer"]
    assert (r.pool_n, r.pool_n_np) == (5, 3)


def test_summary(np_run):
    s = pipeline.summarize(np_run.molecules)
    assert s["n_molecules"] == 5 and s["n_errors"] == 0
    assert s["frac_empty_np_pool_tol"] == pytest.approx(2 / 5)    # m_wrong, m_far
    assert s["n_padded"] == 5 and s["pool_attempts"]["default_adduct"] == 1


# --- inference-path import rule (PLAN 7) -------------------------------------------------

_BLOCKED_RUN = r"""
import importlib.abc, sys
BLOCKED = ("torch", "polars", "duckdb", "sklearn")
class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError("blocked on the inference path: " + name)
        return None
sys.meta_path.insert(0, _Blocker())
import numpy as np, pandas as pd
import casmi.pipeline as pl
from casmi.config import Config
from casmi.db import CandidateDB
df = pd.DataFrame({"molecule_id": ["m1"], "spectrum_id": ["s1"],
                   "ms2_mzs": [np.array([100.0], np.float32)],
                   "ms2_normalized_intensities": [np.array([1.0], np.float32)],
                   "adduct": ["[M+H]+"], "ionization_mode": ["positive"],
                   "precursor_mz": [303.04993]})
cfg = Config().replace(db_tiers=("np",), n_jobs=1, mkey_cache=False, mkey_timeout_s=0.0)
preds = pl.predict(df, cfg, db=CandidateDB(np_path=sys.argv[1], pubchem_dir=sys.argv[2]))
assert len(preds["m1"]) == 25, preds
bad = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
assert not bad, bad
print("ok")
"""


def test_inference_path_imports_no_training_libraries(fixture_dirs):
    out = subprocess.run([sys.executable, "-c", _BLOCKED_RUN, str(fixture_dirs["np"]),
                          str(fixture_dirs["no_pubchem"])], capture_output=True, text=True,
                         timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.strip().endswith("ok")


# --- scripts/predict.py ------------------------------------------------------------------

def test_predict_script(fixture_dirs, tmp_path):
    test_pq = tmp_path / "test.parquet"
    _test_df().to_parquet(test_pq, index=False)
    cfg_path = tmp_path / "b0_fixture.yaml"
    cfg_path.write_text("submission_tag: b0_fixture\ndb_tiers: [np]\nn_jobs: 1\n"
                        "mkey_cache: false\nmkey_timeout_s: 0\n")
    out = tmp_path / "sub.csv"
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASMI_")}
    env.update(CASMI_PROCESSED=str(fixture_dirs["processed"]), CASMI_LOGS=str(tmp_path / "logs"))
    r = subprocess.run([sys.executable, str(paths.ROOT / "scripts" / "predict.py"), "--test",
                        str(test_pq), "--config", str(cfg_path), "--out", str(out)],
                       capture_output=True, text=True, timeout=300, env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    rep = io.validate_submission(out, test_df=_test_df())
    assert rep["n_rows"] == 5 and rep["max_guesses"] == 25
    logs = list((tmp_path / "logs").glob("predict_b0_fixture_*.log"))
    assert len(logs) == 1
    text = logs[0].read_text()
    assert "SUCCESS" in text and "sha256" in text and "peak RSS" in text
    assert (tmp_path / "logs" / "sub_molecules.csv").is_file()


# --- real data ---------------------------------------------------------------------------

@pytest.mark.data("data/processed/db/np/np.parquet")
def test_real_np_tier_quercetin_rank1():
    df = pd.DataFrame([_spec("q", 0, chem.mono_mass(QUERCETIN), "[M+H]+", "positive")])
    preds = pipeline.predict(df, TEST_CFG)
    assert metric.mkey(preds["q"][0]) == metric.mkey(QUERCETIN)


@pytest.mark.data("data/processed/db/np/np.parquet")
def test_real_np_tier_public_subset(public_test_parquet, tmp_path):
    df = io.read_test(public_test_parquet)
    ids = sorted(df["molecule_id"].unique())[:8]
    df = df[df["molecule_id"].isin(ids)].reset_index(drop=True)
    res = pipeline.run(df, TEST_CFG.replace(batch_molecules=5))
    _check_lists(res.preds, ids)
    out = tmp_path / "sub.csv"
    io.write_submission(res.preds, out, test_df=df)
    assert io.validate_submission(out, test_df=df)["n_rows"] == 8
    assert (res.molecules["error"] == "").all()

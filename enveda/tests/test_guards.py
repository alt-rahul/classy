"""Tests for casmi.guards (ROADMAP R9): stage wrappers, budgets, deadlines, progress,
and the supervised child process with its fallback. Uses the synthetic NP tier of
tests/test_pipeline.py."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from casmi import guards, io, pipeline
from test_pipeline import TEST_CFG, _check_lists, _db, _test_df, fixture_dirs, registry  # noqa: F401

IDS = sorted(_test_df()["molecule_id"].unique())


def _plain(dirs, cfg=TEST_CFG):
    return pipeline.run(_test_df(), cfg, db=_db(dirs)).preds


def test_passthrough_is_identical_and_inactive_wrappers_do_nothing(registry, fixture_dirs, tmp_path):
    want = _plain(fixture_dirs)
    res, rep = guards.run(_test_df(), TEST_CFG, db=_db(fixture_dirs),
                          progress_path=tmp_path / "p.jsonl")
    assert res.preds == want
    assert not rep["tripped"]
    assert set(rep["stages"]) == set(TEST_CFG.pipeline_stages)
    st = rep["stages"]["assemble"]
    assert st["molecules"] == len(IDS) and st["calls"] >= 3 and st["total_s"] > 0
    assert getattr(pipeline.get_stage("pool"), "__guarded__") is pipeline.stage_pool
    assert guards.install(list(TEST_CFG.pipeline_stages)) == []  # idempotent
    # wrappers outside an active guard are pass-throughs
    assert guards.current() is None and pipeline.run(_test_df(), TEST_CFG, db=_db(fixture_dirs)).preds == want
    assert guards.read_progress(tmp_path / "p.jsonl", 25) == want


def test_failing_optional_stage_keeps_b0_answer(registry, fixture_dirs):
    want = _plain(fixture_dirs)

    def boom(state, cfg):
        state.pool["half_done"] = 1.0  # partial change before the error
        if "m_quer" in state.molecule_ids:
            raise RuntimeError("boom")
        state.pool["half_done"] = 2.0
        return state

    pipeline.register_stage("boom", boom)
    cfg = TEST_CFG.replace(pipeline_stages=("mass", "pool", "boom", "features", "score", "assemble"))
    res, rep = guards.run(_test_df(), cfg, db=_db(fixture_dirs))
    assert res.preds == want  # m_quer got the full b0 list, not padding only
    assert rep["stages"]["boom"]["n_errors"] == 1 and rep["stages"]["boom"]["n_skipped"] == 1
    assert (res.molecules["error"] == "").all()
    # without guards the molecule is reported as failed (padding only)
    plain = pipeline.run(_test_df(), cfg, db=_db(fixture_dirs))
    assert plain.molecules.set_index("molecule_id").loc["m_quer", "error"].startswith("RuntimeError")


def test_failing_core_stage_uses_fallback(registry, fixture_dirs):
    want = _plain(fixture_dirs)

    def bad_score(state, cfg):
        if "m_taut" in state.molecule_ids:
            raise ValueError("ranker exploded")
        return pipeline.stage_score(state, cfg)

    pipeline.register_stage("score", bad_score)
    res, rep = guards.run(_test_df(), TEST_CFG, db=_db(fixture_dirs))
    assert res.preds == want
    assert rep["stages"]["score"]["n_fallback"] == 1 and rep["stages"]["score"]["n_errors"] == 1


def test_budget_trips_slow_stage(registry, fixture_dirs):
    calls = []

    def slow(state, cfg):
        calls.append(len(state.molecule_ids))
        time.sleep(0.2)
        return state

    pipeline.register_stage("slow", slow)
    cfg = TEST_CFG.replace(pipeline_stages=("mass", "pool", "slow", "features", "score", "assemble"),
                           guard_stage_budgets=("slow=0.01",), guard_stage_grace_s=0.0,
                           batch_molecules=1)
    res, rep = guards.run(_test_df(), cfg, db=_db(fixture_dirs))
    _check_lists(res.preds, IDS)
    s = rep["stages"]["slow"]
    assert len(calls) == 1 and s["n_skipped"] == len(IDS) - 1
    assert s["tripped"].startswith("budget") and "slow" in rep["tripped"]
    assert not rep["stages"]["assemble"]["tripped"]


def test_soft_deadline_degrades_every_stage(registry, fixture_dirs):
    seen = []
    pipeline.register_stage("extra", lambda st, cfg: seen.append(1) or st)
    cfg = TEST_CFG.replace(pipeline_stages=("mass", "pool", "extra", "features", "score", "assemble"))
    res, rep = guards.run(_test_df(), cfg, db=_db(fixture_dirs), soft_deadline=time.time() - 1)
    assert not seen
    assert set(rep["tripped"]) == {"pool", "extra", "features", "score", "assemble"}
    assert rep["stages"]["assemble"]["n_fallback"] >= 1
    for g in res.preds.values():  # ik14 dedup only, still valid lists
        assert guards.valid_guesses(g, 25)


def test_read_progress_ignores_truncated_and_invalid(tmp_path):
    p = tmp_path / "p.jsonl"
    p.write_text(json.dumps({"preds": {"a": ["CCO"], "b": ["C;C"]}}) + "\n"
                 + json.dumps({"preds": {"a": ["CCN"]}}) + "\n" + '{"preds": {"c": ["CC')
    assert guards.read_progress(p, 25) == {"a": ["CCN"]}
    assert guards.read_progress(tmp_path / "none.jsonl", 25) == {}


# --- supervised child process ------------------------------------------------------------

def _env(dirs, tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASMI_")}
    src = str(Path(guards.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), src, env.get("PYTHONPATH", "")])
    env.update(CASMI_PROCESSED=str(dirs["processed"]), CASMI_LOGS=str(tmp_path / "logs"),
               CASMI_ROOT=str(tmp_path))
    return env


_BASE_YAML = "db_tiers: [np]\nn_jobs: 1\nmkey_cache: false\nmkey_timeout_s: 0\nbatch_molecules: 1\n"


def test_supervise_ok_equals_in_process(fixture_dirs, tmp_path):
    test_pq = tmp_path / "test.parquet"
    _test_df().to_parquet(test_pq, index=False)
    cfg_path = tmp_path / "plain.yaml"
    cfg_path.write_text(_BASE_YAML)
    out = tmp_path / "sub.csv"
    lines = []
    rep = guards.supervise(test_pq, cfg_path, out, work_dir=tmp_path / "work",
                           hard_deadline=time.time() + 240, env=_env(fixture_dirs, tmp_path),
                           poll_s=0.5, relay=lines.append)
    assert rep["sources"] == {"main": len(IDS)} and rep["phases"][0]["status"] == "ok"
    ref = tmp_path / "ref.csv"
    io.write_submission(_plain(fixture_dirs, TEST_CFG.replace(batch_molecules=1)), ref)
    assert out.read_bytes() == ref.read_bytes()
    assert any("guard: assemble" in ln for ln in lines)  # child log relayed
    assert rep["phases"][0]["report"]["stages"]["pool"]["calls"] == len(IDS)


def test_supervise_kills_hung_child_and_fills_from_fallback(fixture_dirs, tmp_path):
    (tmp_path / "r9_hang_plugin.py").write_text(
        "import time\nfrom casmi import pipeline\n"
        "@pipeline.register_stage('hang')\n"
        "def hang(state, cfg):\n"
        "    if 'm_far' in state.molecule_ids:\n"
        "        time.sleep(3600)\n"
        "    return state\n")
    test_pq = tmp_path / "test.parquet"
    _test_df().to_parquet(test_pq, index=False)
    main_cfg = tmp_path / "hang.yaml"
    main_cfg.write_text(_BASE_YAML + "pipeline_plugins: [r9_hang_plugin]\n"
                        "pipeline_stages: [mass, pool, hang, features, score, assemble]\n")
    fb_cfg = tmp_path / "fb.yaml"
    fb_cfg.write_text(_BASE_YAML)
    out = tmp_path / "sub.csv"
    t0 = time.time()
    rep = guards.supervise(test_pq, main_cfg, out, work_dir=tmp_path / "work",
                           hard_deadline=t0 + 30, fallback_config=fb_cfg,
                           fallback_deadline=t0 + 200, env=_env(fixture_dirs, tmp_path),
                           poll_s=0.5, relay=lambda s: None)
    assert rep["phases"][0]["status"] == "timeout"
    assert rep["phases"][1]["status"] == "ok"
    # m_far (the heaviest molecule) is the last batch: the four before it come from the
    # main run's progress file, m_far from the fallback run
    assert rep["sources"] == {"main_partial": len(IDS) - 1, "fallback": 1}
    sub = pd.read_csv(out)
    assert sorted(sub["molecule_id"]) == IDS
    want = _plain(fixture_dirs, TEST_CFG.replace(batch_molecules=1))
    got = {m: s.split(";") for m, s in zip(sub["molecule_id"], sub["smiles"])}
    assert got == want
    assert (tmp_path / "work" / "fallback_ids.txt").read_text().split() == ["m_far"]

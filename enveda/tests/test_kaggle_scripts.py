"""Offline tests for the WP14 Kaggle scripts (no network)."""

from __future__ import annotations

import ast
import datetime as dt
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import kaggle_publish  # noqa: E402
import kaggle_submit  # noqa: E402
import package_kaggle  # noqa: E402

UTC = dt.timezone.utc


def _subs(*times: str) -> list[dict]:
    out = [{"date": t, "_t": kaggle_submit._parse_time(t)} for t in times]
    return sorted(out, key=lambda s: s["_t"], reverse=True)


def test_parse_time_variants():
    a = kaggle_submit._parse_time("2026-11-20T10:00:00.123456789Z")
    b = kaggle_submit._parse_time("2026-11-20T10:00:00Z")
    c = kaggle_submit._parse_time("2026-11-20T10:00:00")
    assert a.tzinfo is not None and b == c
    assert abs((a - b).total_seconds() - 0.123456) < 1e-5


def test_pacing_gap_and_daily_cap():
    now = dt.datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)
    assert kaggle_submit.pacing([], now)["wait_s"] == 0
    p = kaggle_submit.pacing(_subs("2026-11-20T11:30:00Z"), now)
    assert p["wait_s"] == pytest.approx(25 * 60)
    p = kaggle_submit.pacing(_subs("2026-11-20T11:05:00Z"), now)
    assert p["wait_s"] == 0 and p["n_today"] == 1
    five = _subs(*[f"2026-11-20T0{h}:00:00Z" for h in range(1, 6)])
    p = kaggle_submit.pacing(five, now)
    assert p["n_today"] == 5 and p["wait_s"] == pytest.approx(12 * 3600)
    # yesterday's submissions do not count toward today's cap
    p = kaggle_submit.pacing(_subs(*[f"2026-11-19T1{h}:00:00Z" for h in range(5)]), now)
    assert p["n_today"] == 0 and p["wait_s"] == 0


def test_collect_and_tree(tmp_path):
    (tmp_path / "pkg" / "sub").mkdir(parents=True)
    (tmp_path / "pkg" / "a.py").write_text("a")
    (tmp_path / "pkg" / "sub" / "b.py").write_text("b")
    (tmp_path / "pkg" / "__pycache__").mkdir()
    (tmp_path / "pkg" / "__pycache__" / "a.cpython.pyc").write_text("x")
    (tmp_path / "top.toml").write_text("t")
    files = kaggle_publish.collect([f"{tmp_path / 'pkg'}=casmi", str(tmp_path / "top.toml")], [])
    assert [k for k, _ in files] == ["casmi/a.py", "casmi/sub/b.py", "top.toml"]
    root, dirs = kaggle_publish._tree({k: f"tok-{k}" for k, _ in files})
    assert root == [{"token": "tok-top.toml"}]
    assert dirs == [{"name": "casmi", "files": [{"token": "tok-casmi/a.py"}],
                     "directories": [{"name": "sub", "files": [{"token": "tok-casmi/sub/b.py"}],
                                      "directories": []}]}]


def test_inference_notebook_is_offline_and_parses():
    nb = package_kaggle.inference_notebook("b0_npdb", "someone", dict(package_kaggle.ARTEFACTS),
                                           [], "abc1234")
    text = json.dumps(nb)
    for bad in ("/home/", "/root/", "/tmp/"):
        assert bad not in text
    code = "\n\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "code")
    ast.parse(code)
    assert "CONFIG = 'b0_npdb'" in code
    assert "NUMBA_CACHE_DIR" in code and "validate_submission" in code
    assert chr(0x2014) not in text  # no em-dash


def test_inference_notebook_is_supervised_and_debug_kernels_are_separate(tmp_path):
    nb = package_kaggle.inference_notebook("b1_library", "someone", dict(package_kaggle.ARTEFACTS),
                                           [], "abc1234")
    code = "\n\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "code")
    assert "guards.supervise(" in code and "TEST_DATASET = None" in code
    assert "DEADLINE_OVERRIDES = {}" in code and "write_fallback" in code
    assert package_kaggle.ARTEFACTS["processed/library"] == "casmi-library"
    dbg = package_kaggle.inference_notebook("b1_library", "someone", {}, [], "abc1234",
                                            test_dataset="casmi-extreme-test",
                                            deadlines={"hard": 60.0})
    dcode = "\n\n".join(c["source"] for c in dbg["cells"] if c["cell_type"] == "code")
    ast.parse(dcode)
    assert "TEST_DATASET = 'casmi-extreme-test'" in dcode and '"hard": 60.0' in dcode
    assert package_kaggle._deadlines("soft=1,hard=2.5") == {"soft": 1.0, "hard": 2.5}
    # a debug notebook can never be pushed as the submission kernel
    with pytest.raises(SystemExit):
        package_kaggle.main(["push", "--config", "b1_library", "--owner", "x",
                             "--test-dataset", "casmi-extreme-test",
                             "--out", str(tmp_path / "nb.ipynb")])

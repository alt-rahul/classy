"""Smoke tests for the WP0 scaffolding: paths, config, logging."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import casmi
from casmi import paths
from casmi.config import Config, load_config, save_config


def _run(code: str, **env: str) -> str:
    """Run ``code`` in a fresh interpreter with CASMI_* env vars cleared, then ``env`` set."""
    base = {k: v for k, v in os.environ.items() if not k.startswith("CASMI_")}
    base.update(env)
    out = subprocess.run([sys.executable, "-c", code], env=base, capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_default_paths():
    root = Path(casmi.__file__).resolve().parents[2]
    got = _run("import casmi; print(casmi.paths.ROOT); print(casmi.paths.RAW); "
               "print(casmi.paths.MODELS)").splitlines()
    assert got == [str(root), str(root / "data" / "raw"), str(root / "models")]


def test_env_overrides_create_nothing(tmp_path):
    root, art = tmp_path / "root", tmp_path / "art"
    root.mkdir()
    got = _run("import casmi.paths as p; print(p.RAW); print(p.PROCESSED); print(p.MODELS); "
               "print(p.LOGS)", CASMI_ROOT=str(root), CASMI_ARTEFACTS=str(art),
               CASMI_LOGS=str(tmp_path / "lg")).splitlines()
    assert got == [str(root / "data" / "raw"), str(art / "processed"), str(art / "models"),
                   str(tmp_path / "lg")]
    assert list(root.iterdir()) == [] and not art.exists()


def test_config_defaults():
    cfg = Config()
    assert cfg.seed == 20261118
    assert (cfg.mass_tol_ppm, cfg.mass_tol_min_da, cfg.mass_cluster_ppm) == (10.0, 0.002, 15.0)
    assert (cfg.frag_tol_same_da, cfg.frag_tol_cross_da) == (0.01, 0.02)
    assert (cfg.intensity_floor, cfg.top_n_peaks, cfg.keep_precursor) == (0.005, 100, False)
    assert (cfg.top_k, cfg.pool_cap) == (25, 30000)
    assert cfg.overrides() == {}


def test_load_config(tmp_path):
    assert load_config(None) == Config()
    assert load_config(paths.CONFIGS / "default.yaml") == Config()

    f = tmp_path / "c.yaml"
    f.write_text("mass_tol_ppm: 8\nfpm_lr: 1e-4\ndb_tiers: [np]\nkeep_precursor: true\n")
    cfg = load_config(f)
    assert cfg.mass_tol_ppm == 8.0 and isinstance(cfg.mass_tol_ppm, float)
    assert cfg.fpm_lr == 1e-4 and cfg.db_tiers == ("np",) and cfg.keep_precursor is True
    assert cfg.overrides() == {"mass_tol_ppm": 8.0, "db_tiers": ["np"],
                               "keep_precursor": True, "fpm_lr": 1e-4}

    save_config(cfg, tmp_path / "out.yaml")
    assert load_config(tmp_path / "out.yaml") == cfg

    f.write_text("mass_tol_pmm: 8\n")
    with pytest.raises(KeyError):
        load_config(f)
    f.write_text("top_n_peaks: 1.5\n")
    with pytest.raises(TypeError):
        load_config(f)
    with pytest.raises(KeyError):
        Config().replace(no_such_field=1)
    assert Config().replace(top_k=10).top_k == 10


def test_logger_writes_file(tmp_path):
    code = ("from casmi.log import get_logger, log_run, log_file\n"
            "lg = get_logger('smoke')\n"
            "log_run(lg, {'x': 1})\n"
            "lg.info('hello')\n"
            "print(log_file())\n")
    path = Path(_run(code, CASMI_LOGS=str(tmp_path)))
    assert path.parent == tmp_path and path.name.startswith("smoke_")
    text = path.read_text()
    assert "git commit:" in text and "hello" in text and "peak RSS" in text

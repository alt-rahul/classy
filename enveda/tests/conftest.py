"""Shared pytest set-up (WP0). Other WPs may append fixtures at the end of this file.

Large-data tests (PLAN.md section 7):
- ``@pytest.mark.data`` marks a test that needs ``data/raw/train.parquet``; it is skipped
  when that file is absent. ``@pytest.mark.data("data/processed/db/np/np.parquet", ...)``
  names other required files instead (relative to ROOT or absolute).
- The ``train_parquet`` fixture returns the path or skips. ``data/raw/test.parquet``
  (5 MB, always present) needs no marker; use the ``public_test_parquet`` fixture.
- Deselect all of them with ``pytest -m "not data"``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from casmi import paths


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "data(*files): needs large local data (default data/raw/train.parquet); "
        "skipped when any required file is absent",
    )


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else paths.ROOT / p


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        for mark in item.iter_markers(name="data"):
            required = [_resolve(a) for a in mark.args] or [paths.TRAIN_PARQUET]
            missing = [str(p) for p in required if not p.exists()]
            if missing:
                item.add_marker(pytest.mark.skip(reason=f"missing data: {', '.join(missing)}"))


@pytest.fixture(scope="session")
def train_parquet() -> Path:
    """Path of data/raw/train.parquet; skips the test when it is absent."""
    if not paths.TRAIN_PARQUET.is_file():
        pytest.skip(f"missing data: {paths.TRAIN_PARQUET}")
    return paths.TRAIN_PARQUET


@pytest.fixture(scope="session")
def public_test_parquet() -> Path:
    """Path of data/raw/test.parquet (public test); skips the test when it is absent."""
    if not paths.TEST_PARQUET.is_file():
        pytest.skip(f"missing data: {paths.TEST_PARQUET}")
    return paths.TEST_PARQUET

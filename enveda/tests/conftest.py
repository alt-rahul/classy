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


# --- WP6 / WP7: synthetic train, ingestion outputs, library, split and queries ----------

def load_script(name: str):
    """Import ``scripts/<name>.py`` as a module (scripts are not a package)."""
    import importlib.util

    p = paths.ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"casmi_script_{name}", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def synthetic_train(tmp_path_factory):
    """Synthetic train file (scripts/make_synthetic_train.py) and everything WP6/WP7 build
    from it, in a temp dir, with the NP tier only (no PubChem scan or lookups).

    Keys: root, train (parquet), out (ingest dir), labels, info (ingest_info), cfg, db
    (NP-only CandidateDB), library, holdout, structures, val_dir, queries {set: (q, truth)},
    ingest (the script module), synth (the script module).
    """
    import json
    import logging

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    np_path = paths.PROCESSED / "db" / "np" / "np.parquet"
    if not np_path.exists():
        pytest.skip(f"missing data: {np_path}")
    from casmi import db as cdb
    from casmi.config import Config
    from casmi.splits import make_holdout
    from casmi.validation import build_queries

    root = tmp_path_factory.mktemp("wp6")
    lg = logging.getLogger("synthetic_train")
    synth = load_script("make_synthetic_train")
    ingest = load_script("ingest_train")
    cfg = Config(n_jobs=1)
    test = pd.read_parquet(paths.TEST_PARQUET).sort_values(
        ["molecule_id", "spectrum_id"], kind="mergesort").reset_index(drop=True)
    mk = synth.Maker(cfg, cfg.seed, 0)
    labels = mk.labels(test, lg)
    df = mk.build(test, labels, lg)
    train = root / "train_synth.parquet"
    tbl = pa.Table.from_pandas(df, schema=synth.TRAIN_SCHEMA, preserve_index=False)
    with pq.ParquetWriter(train, synth.TRAIN_SCHEMA) as w:
        for a in range(0, tbl.num_rows, 500):
            w.write_table(tbl.slice(a, 500))
    empty = root / "no_pubchem"
    empty.mkdir()
    out = root / "train"

    class Args:
        pass

    args = Args()
    args.out, args.test, args.input, args.batch_rows, args.n_jobs = out, paths.TEST_PARQUET, train, 700, 1
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cdb, "pubchem_dir", lambda: empty)
        info = ingest.stage_ingest(args, cfg, lg)
    ingest.stage_mkey(out, cfg, 1, lg)
    lib = ingest.stage_library(out, root / "library", cfg, lg)
    from casmi.library import Library

    library = Library.load(root / "library", cfg)
    st = pd.read_parquet(out / "structures.parquet")
    lc = pd.read_parquet(out / "structure_lib_counts.parquet")
    holdout, _ = make_holdout(st, lc, cfg, n_syn_tims=30, n_np_other=30)
    val_dir = root / "validation"
    queries = build_queries(holdout, out, cfg, raw=train, test_path=paths.TEST_PARQUET,
                            out_dir=val_dir)
    db = cdb.CandidateDB(pubchem_dir=empty, cfg=cfg)
    return {"root": root, "train": train, "out": out, "labels": labels, "frame": df,
            "info": json.loads((out / "ingest_info.json").read_text()), "cfg": cfg, "db": db,
            "library": library, "library_info": lib, "holdout": holdout, "structures": st,
            "lib_counts": lc, "val_dir": val_dir, "queries": queries, "ingest": ingest,
            "synth": synth, "info_stage": info}

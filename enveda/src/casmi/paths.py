"""Filesystem layout. Every path in the project comes from here (PLAN.md section 7).

Resolution order for each directory constant below:
1. Its own environment variable, ``CASMI_<NAME>`` (e.g. ``CASMI_RAW``, ``CASMI_MODELS``).
2. For the artefact directories PROCESSED and MODELS only: ``$CASMI_ARTEFACTS/processed``
   and ``$CASMI_ARTEFACTS/models`` (Kaggle: point it at the unpacked artefact bundle).
3. ``ROOT/<default subdir>``. ROOT is ``$CASMI_ROOT`` if set, else the checkout root
   found from this file's location (``src/casmi/paths.py`` -> two levels up).

On Kaggle the code dataset is read-only, so set ``CASMI_ROOT=/kaggle/working`` (writable
outputs: logs, submissions) and ``CASMI_ARTEFACTS`` / ``CASMI_RAW`` before importing casmi.

Values are read once, at import. Importing this module creates nothing on disk; call
``ensure_dir`` before writing.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "ROOT",
    "RAW",
    "EXTERNAL",
    "PROCESSED",
    "MODELS",
    "SUBMISSIONS",
    "LOGS",
    "CONFIGS",
    "SPLITS",
    "DOCS",
    "DIST",
    "TRAIN_PARQUET",
    "TEST_PARQUET",
    "SAMPLE_SUBMISSION",
    "ensure_dir",
    "as_dict",
]


def _env(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    return Path(value).expanduser().resolve()


def _dir(name: str, default: Path, artefact_sub: str | None = None) -> Path:
    own = _env(f"CASMI_{name}")
    if own is not None:
        return own
    if artefact_sub is not None:
        artefacts = _env("CASMI_ARTEFACTS")
        if artefacts is not None:
            return artefacts / artefact_sub
    return default


ROOT: Path = _env("CASMI_ROOT") or Path(__file__).resolve().parents[2]

RAW: Path = _dir("RAW", ROOT / "data" / "raw")
EXTERNAL: Path = _dir("EXTERNAL", ROOT / "data" / "external")
PROCESSED: Path = _dir("PROCESSED", ROOT / "data" / "processed", artefact_sub="processed")
MODELS: Path = _dir("MODELS", ROOT / "models", artefact_sub="models")
SUBMISSIONS: Path = _dir("SUBMISSIONS", ROOT / "submissions")
LOGS: Path = _dir("LOGS", ROOT / "logs")
CONFIGS: Path = _dir("CONFIGS", ROOT / "configs")
SPLITS: Path = _dir("SPLITS", ROOT / "splits")
DOCS: Path = _dir("DOCS", ROOT / "docs")
DIST: Path = _dir("DIST", ROOT / "dist")

# Competition files.
TRAIN_PARQUET: Path = RAW / "train.parquet"
TEST_PARQUET: Path = RAW / "test.parquet"
SAMPLE_SUBMISSION: Path = RAW / "sample_submission.csv"


def ensure_dir(path: Path | str) -> Path:
    """Create ``path`` (and parents) if missing and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def as_dict() -> dict[str, str]:
    """Resolved directory layout, for logging."""
    names = ["ROOT", "RAW", "EXTERNAL", "PROCESSED", "MODELS", "SUBMISSIONS",
             "LOGS", "CONFIGS", "SPLITS", "DOCS", "DIST"]
    g = globals()
    return {n: str(g[n]) for n in names}

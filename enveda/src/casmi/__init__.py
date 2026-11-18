"""casmi - CASMI 2026 MS/MS spectrum-to-structure pipeline.

See docs/PLAN.md. Submodules are owned by work packages (PLAN.md section 7);
this file only exposes the WP0 scaffolding so that `import casmi` is enough to
reach `casmi.paths`, `casmi.config` and `casmi.log`. Keep it free of heavy
imports: the inference path must stay importable without torch, polars or duckdb.
"""

from __future__ import annotations

from . import config, log, paths

__version__ = "0.1.0"

__all__ = ["config", "log", "paths", "__version__"]

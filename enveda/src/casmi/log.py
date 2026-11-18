"""Logging set-up (PLAN.md section 7).

Scripts call ``get_logger(<script name>)`` once at start, then ``log_run(logger, args)``.
That installs a console handler and a file handler ``logs/<name>_<YYYYmmdd_HHMMSS>.log``
on the root logger, so library modules only need ``logging.getLogger(__name__)`` and
their messages land in the script's log file. ``log_run`` logs the command line,
arguments, git commit and resolved paths, and at exit the wall time and peak RSS.
Row counts in and out of each stage are logged by the stages themselves.
"""

from __future__ import annotations

import atexit
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from . import paths

__all__ = ["get_logger", "log_run", "git_commit", "peak_rss_mb", "log_file"]

_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_MARK = "_casmi_handler"
_log_file: Path | None = None


def get_logger(name: str, level: int = logging.INFO, to_file: bool = True,
               log_dir: Path | str | None = None) -> logging.Logger:
    """Logger ``name``; on first call also install console + file handlers on the root.

    The file is ``<log_dir or paths.LOGS>/<name>_<timestamp>.log``. Later calls (any name)
    reuse the same handlers and file. ``to_file=False`` gives console only.
    """
    global _log_file
    root = logging.getLogger()
    ours = [h for h in root.handlers if getattr(h, _MARK, False)]
    fmt = logging.Formatter(_FMT, _DATEFMT)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in ours):
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(fmt)
        setattr(console, _MARK, True)
        root.addHandler(console)
    if to_file and _log_file is None:
        directory = paths.ensure_dir(log_dir if log_dir is not None else paths.LOGS)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
        _log_file = directory / f"{safe}_{stamp}.log"
        fh = logging.FileHandler(_log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        setattr(fh, _MARK, True)
        root.addHandler(fh)
    if root.level == logging.NOTSET or root.level > level:
        root.setLevel(level)
    return logging.getLogger(name)


def log_file() -> Path | None:
    """Path of the current log file, or None if no file handler was installed."""
    return _log_file


def git_commit(root: Path | str | None = None) -> str:
    """Short commit hash of the checkout, with ``-dirty`` if tracked files changed.

    Read-only (``--no-optional-locks``). Returns "unknown" outside a git checkout.
    """
    cwd = str(root if root is not None else paths.ROOT)
    try:
        sha = subprocess.run(["git", "-C", cwd, "rev-parse", "--short=12", "HEAD"],
                             capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        status = subprocess.run(["git", "--no-optional-locks", "-C", cwd, "status",
                                 "--porcelain", "--untracked-files=no", "--", "."],
                                capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return sha + ("-dirty" if status.strip() else "") if sha else "unknown"


def peak_rss_mb(children: bool = False) -> float:
    """Peak resident set size in MB of this process (or its largest waited-for child)."""
    try:
        import resource
    except ImportError:  # Windows
        return float("nan")
    who = resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF
    kb = resource.getrusage(who).ru_maxrss
    if sys.platform == "darwin":  # bytes on macOS, KB on Linux
        kb = kb / 1024
    return kb / 1024


def log_run(logger: logging.Logger, args: Any = None) -> dict[str, Any]:
    """Log run metadata now and wall time + peak RSS at exit. Returns the metadata.

    ``args`` may be an argparse Namespace, a dict or None. Also logs uncaught exceptions
    to the log file before the default hook prints them.
    """
    t0 = time.perf_counter()
    if args is not None and not isinstance(args, dict):
        args = vars(args)
    info = {
        "argv": list(sys.argv),
        "args": args,
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "host": platform.node(),
        "pid": os.getpid(),
        "log_file": str(_log_file) if _log_file else None,
    }
    logger.info("command: %s", " ".join(info["argv"]))
    logger.info("args: %s", args)
    logger.info("git commit: %s | python %s | host %s | pid %d",
                info["git_commit"], info["python"], info["host"], info["pid"])
    logger.info("paths: %s", paths.as_dict())

    def _at_exit() -> None:
        logger.info("done: wall %.1f s | peak RSS %.0f MB (largest child %.0f MB)",
                    time.perf_counter() - t0, peak_rss_mb(), peak_rss_mb(children=True))

    atexit.register(_at_exit)

    previous_hook = sys.excepthook

    def _hook(exc_type, exc, tb):  # type: ignore[no-untyped-def]
        logger.critical("uncaught exception", exc_info=(exc_type, exc, tb))
        previous_hook(exc_type, exc, tb)

    sys.excepthook = _hook
    return info

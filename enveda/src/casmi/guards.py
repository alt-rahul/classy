"""Inference guards for the Kaggle kernel (ROADMAP R9, continues WP14).

Two layers, both outside the stage code of other packages.

1. Stage wrappers (in process, soft)
------------------------------------
``install()`` wraps every registered pipeline stage and registers the wrapper under the
same name (``pipeline.register_stage``); the original stays on ``wrapper.__guarded__``.
A wrapper is a pure pass-through unless a guard is active (``run`` activates one), so
importing this module never changes other runs. Under an active guard every stage call
is timed and logged with the process RSS, and a stage TRIPS when

- its cumulative time exceeds ``budget x molecules it has seen + cfg.guard_stage_grace_s``
  (budget: ``cfg.guard_stage_budgets`` entry "name=seconds per molecule", else
  ``cfg.guard_stage_budget_s``; 0 = no budget), or
- the soft deadline (an epoch time, ``run(soft_deadline=...)``) has passed.

A tripped stage runs its fallback for the rest of the run:

- ``pool``: ``pipeline.stage_pool`` on the NP tier only (same empty-pool retries);
- ``features`` / ``score``: the Baseline 0 stages (``pipeline.stage_features`` /
  ``stage_score``); tripping either trips both, so a ranker never sees b0 features;
- ``assemble``: ``pipeline.stage_assemble`` with metric keys off (ik14 dedup);
- ``mass``: never trips;
- any other (optional) stage: skipped, the state passes through unchanged.

Exceptions: a stage that raises on a batch of several molecules re-raises (the pipeline
then reruns the batch one molecule at a time); on a single molecule an optional stage is
skipped (columns it added are dropped) and a core stage gets its fallback, so the molecule
keeps a Baseline-0-like answer instead of padding only. If the fallback raises too, the
pipeline's own per-molecule fallback takes over.

Progress: after the last stage of every batch, the batch's valid guess lists are appended
to a JSONL file (``progress_path``), the partial result ``supervise`` uses.

2. Supervisor (hard, separate process)
--------------------------------------
``supervise(...)`` (the Kaggle notebook calls it) runs the pipeline in a child process
(``python -m casmi.guards child ...``, its log relayed line by line) and kills its whole
process group at the hard deadline. On a timeout, crash or OOM kill it keeps the child's
finished batches (progress JSONL), runs ``fallback_config`` on the molecules still missing
(child again, own deadline), fills whatever is left with ``pipeline.LAST_RESORT_SMILES``
and always writes a valid submission. Each phase reports its wall time, return code and
peak RSS (child's own report, else the largest reaped child), plus the cgroup peak when
the kernel exposes one (Kaggle: the whole notebook container).

Deadlines are absolute epoch seconds; the notebook derives them from its start time and
``cfg.guard_soft_deadline_s`` / ``guard_hard_deadline_s`` / ``guard_fallback_deadline_s``.

Inference-path rule (PLAN 7): numpy, pandas and the standard library only.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import pandas as pd

from . import pipeline
from .config import Config, load_config

__all__ = [
    "CORE_STAGES",
    "StageStats",
    "Guard",
    "install",
    "active",
    "current",
    "run",
    "valid_guesses",
    "read_progress",
    "rss_mb",
    "cgroup_peak_mb",
    "supervise",
    "main",
]

log = logging.getLogger(__name__)

CORE_STAGES: tuple[str, ...] = ("mass", "pool", "features", "score", "assemble")
_LINKED = {"features": ("score",), "score": ("features",)}


# --- memory -----------------------------------------------------------------------------

def rss_mb() -> float:
    """Current resident set size of this process in MB (Linux /proc; else peak RSS)."""
    try:
        with open("/proc/self/statm", "rb") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 2 ** 20
    except (OSError, ValueError, IndexError, AttributeError):
        return _peak_mb()


def _peak_mb(children: bool = False) -> float:
    try:
        import resource
    except ImportError:  # Windows
        return float("nan")
    kb = resource.getrusage(resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF).ru_maxrss
    return kb / 1024 / (1024 if sys.platform == "darwin" else 1)


def cgroup_peak_mb() -> float | None:
    """Peak memory of this container's cgroup in MB, if the kernel exposes it (v2 or v1).
    On a shared machine this includes other processes; on Kaggle it is the notebook."""
    for p in ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        try:
            return int(Path(p).read_text().strip()) / 2 ** 20
        except (OSError, ValueError):
            continue
    return None


# --- stage wrappers ---------------------------------------------------------------------

def valid_guesses(g: Any, k: int) -> bool:
    """1 to ``k`` non-empty SMILES strings without ";", ",", quotes or newlines."""
    if not isinstance(g, (list, tuple)) or not 1 <= len(g) <= k:
        return False
    return all(isinstance(s, str) and s.strip() and not any(c in s for c in ";,\n\"") for s in g)


def _fb_pool(state: pipeline.State, cfg: Config) -> pipeline.State:
    return pipeline.stage_pool(state, cfg.replace(db_tiers=("np",)))


def _fb_assemble(state: pipeline.State, cfg: Config) -> pipeline.State:
    state.extras["assemble_use_mkeys"] = False
    return pipeline.stage_assemble(state, cfg)


FALLBACKS: dict[str, Callable[[pipeline.State, Config], pipeline.State]] = {
    "pool": _fb_pool,
    "features": pipeline.stage_features,
    "score": pipeline.stage_score,
    "assemble": _fb_assemble,
}


@dataclass
class StageStats:
    calls: int = 0
    molecules: int = 0
    total_s: float = 0.0
    max_s: float = 0.0
    n_skipped: int = 0          # calls passed through (optional stage tripped or failed)
    n_fallback: int = 0         # calls that ran the core fallback
    n_errors: int = 0           # single-molecule calls that raised
    tripped: str = ""           # why the stage tripped ("" = never)
    rss_max_mb: float = 0.0     # largest RSS right after a call
    peak_rss_mb: float = 0.0    # process peak RSS after the last call


class _Snapshot:
    """What a failed optional stage may have changed: frames (and their columns), preds."""

    def __init__(self, st: pipeline.State) -> None:
        self.pool, self.pool_cols = st.pool, None if st.pool is None else list(st.pool.columns)
        self.mols, self.mol_cols = st.molecules, None if st.molecules is None else list(st.molecules.columns)
        self.spectra = st.spectra
        self.preds = dict(st.preds)

    @staticmethod
    def _back(df: pd.DataFrame | None, cols: list | None) -> pd.DataFrame | None:
        if df is None or cols is None:
            return df
        extra = [c for c in df.columns if c not in set(cols)]
        return df.drop(columns=extra) if extra else df

    def restore(self, st: pipeline.State) -> None:
        st.pool = self._back(self.pool, self.pool_cols)
        st.molecules = self._back(self.mols, self.mol_cols)
        st.spectra = self.spectra
        st.preds = self.preds


def _parse_budgets(items: Sequence[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for it in items:
        name, sep, val = str(it).partition("=")
        if not sep or not name.strip():
            raise ValueError(f"guard_stage_budgets entry {it!r}: expected 'name=seconds'")
        out[name.strip()] = float(val)
    return out


class Guard:
    """Per-run guard state: stage statistics, trips, deadlines, progress file."""

    def __init__(self, cfg: Config, soft_deadline: float | None = None,
                 progress_path: Path | str | None = None) -> None:
        self.cfg = cfg
        self.soft_deadline = soft_deadline
        self.progress_path = Path(progress_path) if progress_path else None
        self.last_stage = cfg.pipeline_stages[-1] if cfg.pipeline_stages else None
        self.budgets = _parse_budgets(cfg.guard_stage_budgets)
        self.stats: dict[str, StageStats] = {}
        self.t0 = time.time()
        self.n_progress = 0

    # trips
    def _reason(self, name: str, st: StageStats) -> str:
        if name == "mass":
            return ""
        if self.soft_deadline is not None and time.time() > self.soft_deadline:
            return "soft deadline passed"
        b = self.budgets.get(name, self.cfg.guard_stage_budget_s)
        limit = b * st.molecules + self.cfg.guard_stage_grace_s
        if b > 0 and st.total_s > limit:
            return f"budget: {st.total_s:.0f} s > {b:g} s/molecule x {st.molecules} + {self.cfg.guard_stage_grace_s:g} s"
        return ""

    def trip(self, name: str, reason: str) -> None:
        st = self.stats.setdefault(name, StageStats())
        if st.tripped or name == "mass":
            return
        st.tripped = reason
        log.warning("guard: stage %s tripped (%s); %s from now on", name, reason,
                    "its fallback" if name in CORE_STAGES else "skipped")
        for other in _LINKED.get(name, ()):
            self.trip(other, f"linked to {name}")

    def _degraded(self, name: str, fn: Callable, state: pipeline.State, cfg: Config,
                  st: StageStats, exc: BaseException | None = None) -> pipeline.State:
        if name not in CORE_STAGES:
            st.n_skipped += 1
            return state
        fb = FALLBACKS.get(name)
        if fb is None or (exc is not None and fb is fn):
            if exc is not None:
                raise exc
            return fn(state, cfg)  # mass, or a default stage with nothing cheaper
        st.n_fallback += 1
        return fb(state, cfg)

    def call(self, name: str, fn: Callable, state: pipeline.State, cfg: Config) -> pipeline.State:
        st = self.stats.setdefault(name, StageStats())
        ids = state.molecule_ids
        reason = self._reason(name, st)
        if reason:
            self.trip(name, reason)
        t0 = time.perf_counter()
        try:
            if st.tripped:
                out = self._degraded(name, fn, state, cfg, st)
            else:
                snap = _Snapshot(state) if len(ids) == 1 else None
                try:
                    out = fn(state, cfg)
                except Exception as exc:  # noqa: BLE001 - one molecule must not sink the stage
                    if snap is None:
                        raise
                    st.n_errors += 1
                    log.exception("guard: stage %s failed on molecule %s; %s", name, ids[0],
                                  "fallback" if name in CORE_STAGES else "skipped")
                    snap.restore(state)
                    out = self._degraded(name, fn, state, cfg, st, exc)
        finally:
            dt = time.perf_counter() - t0
            st.calls += 1
            st.molecules += len(ids)
            st.total_s += dt
            st.max_s = max(st.max_s, dt)
            r = rss_mb()
            st.rss_max_mb = max(st.rss_max_mb, r)
            st.peak_rss_mb = _peak_mb()
            log.info("guard: %-10s %4d molecules %7.2f s (total %7.1f s) rss %6.0f MB peak %6.0f MB%s",
                     name, len(ids), dt, st.total_s, r, st.peak_rss_mb,
                     f" [{'fallback' if name in CORE_STAGES else 'skipped'}]" if st.tripped else "")
        if name == self.last_stage and self.progress_path is not None:
            self._progress(out, ids, cfg)
        return out

    def _progress(self, state: pipeline.State, ids: Sequence[str], cfg: Config) -> None:
        good = {m: list(state.preds[m]) for m in ids if valid_guesses(state.preds.get(m), cfg.top_k)}
        if not good:
            return
        try:
            self.progress_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.progress_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"t": round(time.time() - self.t0, 2), "preds": good},
                                    sort_keys=True) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            self.n_progress += len(good)
        except OSError:
            log.exception("guard: cannot write progress to %s", self.progress_path)

    def report(self) -> dict[str, Any]:
        order = list(self.cfg.pipeline_stages) + sorted(set(self.stats) - set(self.cfg.pipeline_stages))
        stages = {n: asdict(self.stats[n]) for n in order if n in self.stats}
        return {"wall_s": round(time.time() - self.t0, 2), "stages": stages,
                "peak_rss_mb": round(_peak_mb(), 1),
                "peak_rss_children_mb": round(_peak_mb(children=True), 1),
                "cgroup_peak_mb": cgroup_peak_mb(),
                "tripped": {n: s.tripped for n, s in self.stats.items() if s.tripped}}

    def log_summary(self) -> None:
        log.info("guard: %-10s %6s %9s %8s %8s %5s %5s %4s %8s  %s", "stage", "calls", "molecules",
                 "total_s", "max_s", "skip", "fallb", "err", "rss_max", "tripped")
        for n, s in self.report()["stages"].items():
            log.info("guard: %-10s %6d %9d %8.1f %8.2f %5d %5d %4d %8.0f  %s", n, s["calls"],
                     s["molecules"], s["total_s"], s["max_s"], s["n_skipped"], s["n_fallback"],
                     s["n_errors"], s["rss_max_mb"], s["tripped"] or "-")
        r = self.report()
        log.info("guard: wall %.1f s; peak RSS %.0f MB (largest child %.0f MB); cgroup peak %s MB",
                 r["wall_s"], r["peak_rss_mb"], r["peak_rss_children_mb"],
                 "n/a" if r["cgroup_peak_mb"] is None else f"{r['cgroup_peak_mb']:.0f}")


_ACTIVE: Guard | None = None


def current() -> Guard | None:
    """The active guard, or None."""
    return _ACTIVE


@contextlib.contextmanager
def active(guard: Guard) -> Iterator[Guard]:
    global _ACTIVE
    prev, _ACTIVE = _ACTIVE, guard
    try:
        yield guard
    finally:
        _ACTIVE = prev


def _wrap(name: str, fn: Callable) -> Callable:
    def guarded(state: pipeline.State, cfg: Config) -> pipeline.State:
        g = _ACTIVE
        if g is None:
            return fn(state, cfg)
        return g.call(name, fn, state, cfg)

    guarded.__guarded__ = fn  # type: ignore[attr-defined]
    guarded.__name__ = guarded.__qualname__ = f"guarded_{name}"
    guarded.__doc__ = f"casmi.guards wrapper of {getattr(fn, '__module__', '?')}." \
                      f"{getattr(fn, '__qualname__', repr(fn))}"
    return guarded


def install(names: Sequence[str] | None = None) -> list[str]:
    """Wrap the registered stages ``names`` (default: all) in place; idempotent.
    Returns the names newly wrapped."""
    done = []
    for name in sorted(pipeline.STAGES) if names is None else list(names):
        fn = pipeline.get_stage(name)
        if getattr(fn, "__guarded__", None) is None:
            pipeline.register_stage(name, _wrap(name, fn))
            done.append(name)
    return done


def run(test_df: pd.DataFrame, cfg: Config, *, soft_deadline: float | None = None,
        progress_path: Path | str | None = None, extras: Mapping[str, Any] | None = None,
        db: Any = None) -> tuple[pipeline.PipelineResult, dict[str, Any]]:
    """``pipeline.run`` with every stage of ``cfg`` guarded. Returns (result, guard report).
    With no trip the predictions equal ``pipeline.run(test_df, cfg)`` exactly."""
    pipeline.load_plugins(cfg)
    install(list(dict.fromkeys(cfg.pipeline_stages)))
    g = Guard(cfg, soft_deadline=soft_deadline, progress_path=progress_path)
    with active(g):
        res = pipeline.run(test_df, cfg, db=db, extras=extras)
    g.log_summary()
    rep = g.report()
    rep["pipeline_wall_s"] = round(res.wall_s, 2)
    rep["n_progress"] = g.n_progress
    return res, rep


# --- child process ----------------------------------------------------------------------

def _child(a: argparse.Namespace) -> int:
    from . import io
    from . import log as clog

    lg = clog.get_logger(f"guard_{a.tag}")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lg.info("guard child %s: test %s, config %s, soft deadline in %s s", a.tag, a.test, a.config,
            "-" if a.soft_deadline is None else f"{a.soft_deadline - time.time():.0f}")
    cfg = load_config(a.config)
    lg.info("config overrides: %s", cfg.overrides())
    pipeline.limit_threads(cfg.n_jobs)
    test_df = io.read_test(a.test)
    if a.ids:
        want = set(Path(a.ids).read_text().split())
        test_df = test_df[test_df["molecule_id"].isin(want)].reset_index(drop=True)
        lg.info("restricted to %d molecules (%d spectra)", test_df["molecule_id"].nunique(), len(test_df))
    extras = {"assemble_use_mkeys": False} if a.no_mkeys else None
    res, rep = run(test_df, cfg, soft_deadline=a.soft_deadline, progress_path=out / "progress.jsonl",
                   extras=extras)
    io.write_submission(res.preds, out / "submission.csv", test_df=test_df, max_guesses=cfg.top_k)
    res.molecules.to_csv(out / "molecules.csv", index=False, lineterminator="\n")
    rep["n_molecules"] = len(res.preds)
    rep["n_errors"] = int((res.molecules.get("error", pd.Series(dtype=str)).fillna("") != "").sum())
    (out / "guard_report.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    lg.info("guard child %s done: %d molecules in %.1f s", a.tag, len(res.preds), res.wall_s)
    return 0


# --- supervisor -------------------------------------------------------------------------

def read_progress(path: Path | str, k: int) -> dict[str, list[str]]:
    """Valid guess lists from a progress JSONL (the last line wins per molecule);
    a truncated last line (process killed while writing) is ignored."""
    out: dict[str, list[str]] = {}
    p = Path(path)
    if not p.is_file():
        return out
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        for m, g in (d.get("preds") or {}).items():
            if valid_guesses(g, k):
                out[str(m)] = list(g)
    return out


def _read_submission(path: Path, k: int) -> dict[str, list[str]]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    out = {}
    for m, cell in zip(df["molecule_id"], df["smiles"]):
        g = [s for s in str(cell).split(";") if s.strip()]
        if valid_guesses(g, k):
            out[str(m)] = g
    return out


def _relay(path: Path, pos: int, prefix: str, relay: Callable[[str], Any]) -> int:
    try:
        with open(path, "rb") as fh:
            fh.seek(pos)
            data = fh.read()
    except OSError:
        return pos
    if not data:
        return pos
    cut = data.rfind(b"\n") + 1  # whole lines only; the rest comes next time
    if cut == 0:
        return pos
    for line in data[:cut].decode("utf-8", errors="replace").splitlines():
        relay(f"{prefix} {line}")
    return pos + cut


def _kill(p: subprocess.Popen, grace_s: float = 15.0) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(p.pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                p.send_signal(sig)
            except (ProcessLookupError, OSError):
                pass
        try:
            p.wait(timeout=grace_s)
            return
        except subprocess.TimeoutExpired:
            continue


def _run_child(tag: str, args: list[str], deadline: float, work: Path, *, python: str,
               env: Mapping[str, str] | None, poll_s: float,
               relay: Callable[[str], Any]) -> dict[str, Any]:
    out = work / tag
    out.mkdir(parents=True, exist_ok=True)
    logp = out / "child.log"
    cmd = [python, "-m", "casmi.guards", "child", "--tag", tag, "--out-dir", str(out), *args]
    t0 = time.time()
    phase: dict[str, Any] = {"phase": tag, "cmd": cmd[3:], "budget_s": round(deadline - t0, 1)}
    if deadline - t0 < 5:
        phase.update(status="skipped (no time left)", rc=None, wall_s=0.0)
        return phase
    relay(f"[guard] {tag}: starting child, deadline in {deadline - t0:.0f} s")
    with open(logp, "wb") as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True,
                             env=dict(env) if env is not None else None)
    pos = 0
    status = "ok"
    while True:
        rc = p.poll()
        pos = _relay(logp, pos, f"[{tag}]", relay)
        if rc is not None:
            break
        if time.time() > deadline:
            relay(f"[guard] {tag}: hard deadline reached after {time.time() - t0:.0f} s; killing")
            _kill(p)
            rc = p.returncode
            status = "timeout"
            break
        time.sleep(poll_s)
    pos = _relay(logp, pos, f"[{tag}]", relay)
    if status == "ok" and rc != 0:
        status = f"failed rc {rc}" + (" (SIGKILL: out of memory?)" if rc == -signal.SIGKILL else "")
    phase.update(status=status, rc=rc, wall_s=round(time.time() - t0, 1),
                 peak_rss_children_mb=round(_peak_mb(children=True), 1),
                 cgroup_peak_mb=cgroup_peak_mb(), out_dir=str(out))
    rep = out / "guard_report.json"
    if rep.is_file():
        try:
            phase["report"] = json.loads(rep.read_text())
        except ValueError:
            pass
    relay(f"[guard] {tag}: {status} in {phase['wall_s']:.0f} s")
    return phase


def supervise(test_path: Path | str, config_path: Path | str, out_csv: Path | str, *,
              work_dir: Path | str, hard_deadline: float, soft_deadline: float | None = None,
              fallback_config: Path | str | None = None, fallback_deadline: float | None = None,
              python: str | None = None, env: Mapping[str, str] | None = None,
              poll_s: float = 5.0, relay: Callable[[str], Any] = print) -> dict[str, Any]:
    """Run ``config_path`` on ``test_path`` in a guarded child process and always write a
    valid ``out_csv`` (see the module docstring). Returns the run report."""
    t_start = time.time()
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    python = python or sys.executable
    ids = sorted(set(pd.read_parquet(test_path, columns=["molecule_id"])["molecule_id"].astype(str)))
    k = load_config(config_path).top_k
    report: dict[str, Any] = {"n_molecules": len(ids), "phases": []}
    preds: dict[str, list[str]] = {}
    source: dict[str, str] = {}

    def take(got: Mapping[str, list[str]], label: str) -> None:
        for m in ids:
            if m not in preds and m in got:
                preds[m] = got[m]
                source[m] = label

    args = ["--test", str(test_path), "--config", str(config_path)]
    if soft_deadline is not None:
        args += ["--soft-deadline", f"{soft_deadline:.3f}"]
    ph = _run_child("main", args, hard_deadline, work, python=python, env=env, poll_s=poll_s,
                    relay=relay)
    report["phases"].append(ph)
    sub = work / "main" / "submission.csv"
    if ph["status"] == "ok" and sub.is_file():
        take(_read_submission(sub, k), "main")
    take(read_progress(work / "main" / "progress.jsonl", k), "main_partial")

    missing = [m for m in ids if m not in preds]
    if missing and fallback_config is not None:
        fb_deadline = fallback_deadline if fallback_deadline is not None else hard_deadline
        (work / "fallback_ids.txt").write_text("\n".join(missing) + "\n")
        left = fb_deadline - time.time()
        fargs = ["--test", str(test_path), "--config", str(fallback_config),
                 "--ids", str(work / "fallback_ids.txt"),
                 "--soft-deadline", f"{time.time() + 0.5 * max(left, 0):.3f}"]
        relay(f"[guard] {len(missing)} molecules missing after the main run; fallback {fallback_config}")
        ph = _run_child("fallback", fargs, fb_deadline, work, python=python, env=env,
                        poll_s=poll_s, relay=relay)
        report["phases"].append(ph)
        fsub = work / "fallback" / "submission.csv"
        if ph["status"] == "ok" and fsub.is_file():
            take(_read_submission(fsub, k), "fallback")
        take(read_progress(work / "fallback" / "progress.jsonl", k), "fallback_partial")

    rest = [m for m in ids if m not in preds]
    take({m: [pipeline.LAST_RESORT_SMILES] for m in rest}, "last_resort")
    from . import io

    io.write_submission(preds, out_csv, test_df=pd.DataFrame({"molecule_id": ids}), max_guesses=k)
    counts = pd.Series(source).value_counts().to_dict() if source else {}
    report.update(sources={str(a): int(b) for a, b in counts.items()},
                  wall_s=round(time.time() - t_start, 1), cgroup_peak_mb=cgroup_peak_mb(),
                  peak_rss_children_mb=round(_peak_mb(children=True), 1), out_csv=str(out_csv))
    (work / "supervise_report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    relay(f"[guard] submission written: {len(ids)} molecules, sources {report['sources']}, "
          f"{report['wall_s']:.0f} s")
    return report


# --- command line -----------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m casmi.guards", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("child", help="run one guarded pipeline (what supervise starts)")
    c.add_argument("--test", required=True)
    c.add_argument("--config", required=True)
    c.add_argument("--out-dir", required=True)
    c.add_argument("--tag", default="main")
    c.add_argument("--ids", default=None, help="file with the molecule_ids to run (default: all)")
    c.add_argument("--soft-deadline", type=float, default=None, help="epoch seconds")
    c.add_argument("--no-mkeys", action="store_true", help="assemble without metric keys")
    s = sub.add_parser("supervise", help="supervised run with deadlines in seconds from now")
    s.add_argument("--test", required=True)
    s.add_argument("--config", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--work-dir", required=True)
    s.add_argument("--soft", type=float, default=None)
    s.add_argument("--hard", type=float, required=True)
    s.add_argument("--fallback-config", default=None)
    s.add_argument("--fallback", type=float, default=None)
    a = p.parse_args(argv)
    if a.cmd == "child":
        return _child(a)
    now = time.time()
    rep = supervise(a.test, a.config, a.out, work_dir=a.work_dir, hard_deadline=now + a.hard,
                    soft_deadline=None if a.soft is None else now + a.soft,
                    fallback_config=a.fallback_config,
                    fallback_deadline=None if a.fallback is None else now + a.fallback)
    print(json.dumps({k: v for k, v in rep.items() if k != "phases"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    from casmi import guards as _g  # one module object, so the wrappers see one _ACTIVE

    sys.exit(_g.main())

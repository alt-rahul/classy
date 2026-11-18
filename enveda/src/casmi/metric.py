"""Competition metric: the metric key ``mkey`` and MRR@25 (PLAN.md section 5.1).

AUTHORITY: the official metric notebook
(https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank) could not be read from
this machine. If ``docs/METRIC_NOTEBOOK.md`` appears, it is the authority: make
``mkey_mol`` below identical to it (it is the single place that defines the key) and add
its examples to ``tests/test_metric.py``. Until then the key is, as in COMPETITION.md:
``Chem.MolFromSmiles`` -> ``rdMolStandardize.TautomerEnumerator().Canonicalize`` (default
limits) -> ``Chem.MolToInchiKey(...)[:14]``, None when parsing or key generation fails.

Public API:
- ``mkey(smiles)``: the key of one SMILES (exact, no timeout, in-process, memoised).
- ``mkeys(smiles, ...)``: many SMILES with a process pool, a per-molecule timeout
  (``Config.mkey_timeout_s``) that falls back to the plain InChIKey first block, and a
  parquet disk cache (``paths.PROCESSED / "cache" / "mkey.parquet"``, keyed by SMILES).
- ``ik14(smiles)``: plain InChIKey first block (the timeout fallback).
- ``reciprocal_ranks(pred, truth)`` and ``mrr_at_25(pred, truth)``.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize

from . import paths
from .chem import mol_from_smiles, mp_context
from .config import Config

__all__ = [
    "mkey",
    "mkey_mol",
    "ik14",
    "mkeys",
    "flush_mkey_cache",
    "default_cache_path",
    "split_guesses",
    "reciprocal_ranks",
    "mrr_at_25",
    "STATUSES",
]

log = logging.getLogger(__name__)

# Per-SMILES status returned by mkeys(..., return_status=True) and stored in the cache:
#   ok      - key computed
#   fail    - SMILES does not parse (key None)
#   error   - parsed, but canonicalisation or InChI failed, or the worker crashed (key None)
#   timeout - exceeded the timeout; key is the plain InChIKey first block (or None)
STATUSES = ("ok", "fail", "error", "timeout")

_UNSET: Any = object()


# --- the key --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _enumerator() -> Any:
    # A cached enumerator gives identical results to a fresh one per call (checked on 300
    # NCI molecules) and saves about 1 ms per molecule.
    return rdMolStandardize.TautomerEnumerator()


def mkey_mol(mol: Chem.Mol | None) -> str | None:
    """THE metric key of a parsed molecule. Keep every metric detail in this function."""
    if mol is None:
        return None
    with rdBase.BlockLogs():
        try:
            canon = _enumerator().Canonicalize(mol)
            key = Chem.MolToInchiKey(canon)
        except Exception:  # noqa: BLE001
            return None
    return key[:14] if key else None


@lru_cache(maxsize=1 << 18)
def mkey(smiles: str) -> str | None:
    """Metric key of a SMILES: tautomer-canonical InChIKey first block, None on failure."""
    return mkey_mol(mol_from_smiles(smiles))


def ik14(smiles: str | Chem.Mol | None) -> str | None:
    """Plain InChIKey first block (no tautomer canonicalisation); None on failure."""
    mol = smiles if isinstance(smiles, Chem.Mol) else mol_from_smiles(smiles)
    if mol is None:
        return None
    with rdBase.BlockLogs():
        try:
            key = Chem.MolToInchiKey(mol)
        except Exception:  # noqa: BLE001
            return None
    return key[:14] if key else None


def _compute(smiles: str) -> tuple[str | None, str]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None, "fail"
    key = mkey_mol(mol)
    return (key, "ok") if key is not None else (None, "error")


# --- worker processes with a per-molecule timeout -------------------------------------

def _worker_loop(conn: Any) -> None:
    """Child: say "ready", then receive chunks [(i, smiles), ...] and send (i, key, status)
    per item, then None per chunk."""
    rdBase.DisableLog("rdApp.*")
    _enumerator()
    conn.send("ready")
    while True:
        try:
            chunk = conn.recv()
        except (EOFError, OSError):
            return
        if chunk is None:
            return
        for i, s in chunk:
            key, status = _compute(s)
            conn.send((i, key, status))
        conn.send(None)


class _Worker:
    def __init__(self, ctx: Any) -> None:
        parent, child = ctx.Pipe(duplex=True)
        self.proc = ctx.Process(target=_worker_loop, args=(child,), daemon=True)
        self.proc.start()
        child.close()
        self.conn = parent
        self.todo: deque[tuple[int, str]] = deque()
        self.t_last = time.monotonic()
        self.ready = False

    def wait_ready(self, timeout_s: float = 120.0) -> None:
        """Block until the child has started (start-up must not count as molecule time)."""
        if not self.ready:
            if not self.conn.poll(timeout_s) or self.conn.recv() != "ready":
                raise RuntimeError("mkey worker did not start")
            self.ready = True

    def send(self, chunk: list[tuple[int, str]]) -> None:
        self.wait_ready()
        self.todo = deque(chunk)
        self.t_last = time.monotonic()
        self.conn.send(chunk)

    def kill(self) -> None:
        try:
            self.proc.kill()
            self.proc.join(timeout=5)
        finally:
            self.conn.close()

    def stop(self) -> None:
        try:
            self.conn.send(None)
        except (OSError, ValueError):
            pass
        self.proc.join(timeout=5)
        if self.proc.is_alive():
            self.proc.kill()
            self.proc.join(timeout=5)
        self.conn.close()


def _run_workers(items: Sequence[str], n_workers: int, timeout_s: float | None,
                 chunk_size: int = 8) -> list[tuple[str | None, str]]:
    """Keys for ``items`` in worker processes; a molecule running longer than
    ``timeout_s`` gets its worker killed, the plain ik14 fallback and status "timeout"."""
    from multiprocessing.connection import wait

    results: list[tuple[str | None, str] | None] = [None] * len(items)
    indexed = list(enumerate(items))
    queue: deque[list[tuple[int, str]]] = deque(
        indexed[i:i + chunk_size] for i in range(0, len(indexed), chunk_size))
    if not queue:
        return []
    ctx = mp_context()
    idle: list[_Worker] = []
    busy: dict[Any, _Worker] = {}

    def retire(w: _Worker, status: str) -> None:
        del busy[w.conn]
        if w.todo:
            i, s = w.todo.popleft()
            results[i] = (ik14(s), "timeout") if status == "timeout" else (None, "error")
            if status == "timeout":
                log.warning("mkey timeout after %.3g s, using plain InChIKey: %s",
                            timeout_s, s[:200])
            else:
                log.warning("mkey worker died on: %s", s[:200])
            if w.todo:
                queue.appendleft(list(w.todo))
        w.kill()
        if queue:
            idle.append(_Worker(ctx))

    try:
        idle.extend(_Worker(ctx) for _ in range(max(1, min(n_workers, len(queue)))))
        for w in idle:  # started together, waited for together
            w.wait_ready()
        while queue or busy:
            while queue and idle:
                w = idle.pop()
                w.send(queue.popleft())
                busy[w.conn] = w
            wait_s = None
            if timeout_s is not None:
                deadline = min(w.t_last for w in busy.values()) + timeout_s
                wait_s = max(0.0, deadline - time.monotonic())
            for conn in wait(list(busy), timeout=wait_s):
                w = busy[conn]
                try:
                    msg = conn.recv()
                except (EOFError, OSError):
                    retire(w, "error")
                    continue
                if msg is None:
                    del busy[conn]
                    idle.append(w)
                    continue
                i, key, status = msg
                results[i] = (key, status)
                if w.todo and w.todo[0][0] == i:
                    w.todo.popleft()
                w.t_last = time.monotonic()
            if timeout_s is not None:
                now = time.monotonic()
                for w in [w for w in busy.values() if w.todo and now - w.t_last >= timeout_s]:
                    if not w.conn.poll(0):  # a result already in the pipe is not a timeout
                        retire(w, "timeout")
    finally:
        for w in idle:
            w.stop()
        for w in list(busy.values()):
            w.kill()
    assert all(r is not None for r in results)
    return results  # type: ignore[return-value]


# --- disk cache -----------------------------------------------------------------------

def default_cache_path() -> Path:
    return paths.PROCESSED / "cache" / "mkey.parquet"


class _Cache:
    """In-memory view of one parquet cache file, plus entries not yet written."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.mem: dict[str, tuple[str | None, str, float]] = {}
        self.pending: dict[str, tuple[str | None, str, float]] = {}
        self.loaded = False
        self.read_only = False

    def _read_file(self) -> dict[str, tuple[str | None, str, float]]:
        if not self.path.is_file():
            return {}
        import pyarrow.parquet as pq

        t = pq.read_table(self.path).to_pydict()
        version = rdBase.rdkitVersion
        return {s: (k, st, float(to)) for s, k, st, to, v in
                zip(t["smiles"], t["mkey"], t["status"], t["timeout_s"], t["rdkit"])
                if v == version}

    def load(self) -> None:
        if not self.loaded:
            try:
                self.mem.update(self._read_file())
            except Exception as exc:  # noqa: BLE001 - a broken cache must not stop a run
                log.warning("cannot read mkey cache %s: %s", self.path, exc)
            self.loaded = True

    def get(self, smiles: str, timeout_s: float | None) -> str | None | Any:
        hit = self.mem.get(smiles)
        if hit is None:
            return _UNSET
        key, status, stored_timeout = hit
        if status == "timeout":
            # Reuse a fallback only if we would not wait longer now than when it timed out.
            if timeout_s is None or timeout_s > stored_timeout:
                return _UNSET
        return key, status

    def put(self, smiles: str, key: str | None, status: str, timeout_s: float | None) -> None:
        entry = (key, status, float("inf") if timeout_s is None else float(timeout_s))
        self.mem[smiles] = entry
        self.pending[smiles] = entry

    def flush(self) -> None:
        if not self.pending or self.read_only:
            self.pending.clear()
            return
        import pandas as pd

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with _FileLock(self.path.with_name(self.path.name + ".lock")):
                try:
                    merged = self._read_file()
                except OSError:
                    raise
                except Exception as exc:  # noqa: BLE001 - corrupt file: rebuild it
                    log.warning("mkey cache %s unreadable (%s); rewriting it", self.path, exc)
                    merged = {}
                for s, entry in self.pending.items():
                    old = merged.get(s)
                    if old is None or _better(entry, old):
                        merged[s] = entry
                smiles = sorted(merged)
                df = pd.DataFrame({
                    "smiles": pd.Series(smiles, dtype=object),
                    "mkey": pd.Series([merged[s][0] for s in smiles], dtype=object),
                    "status": pd.Series([merged[s][1] for s in smiles], dtype=object),
                    "timeout_s": np.array([merged[s][2] for s in smiles], dtype=np.float64),
                    "rdkit": pd.Series([rdBase.rdkitVersion] * len(smiles), dtype=object),
                })
                fd, tmp = tempfile.mkstemp(prefix=".mkey.", suffix=".tmp", dir=self.path.parent)
                os.close(fd)
                try:
                    df.to_parquet(tmp, index=False, compression="zstd")
                    os.replace(tmp, self.path)
                finally:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
            for s, entry in merged.items():
                cur = self.mem.get(s)
                if cur is None or _better(entry, cur):
                    self.mem[s] = entry
            log.info("mkey cache %s: %d entries (+%d)", self.path, len(merged), len(self.pending))
        except OSError as exc:
            log.warning("mkey cache %s not writable (%s); keeping results in memory only",
                        self.path, exc)
            self.read_only = True
        self.pending.clear()


def _better(new: tuple[str | None, str, float], old: tuple[str | None, str, float]) -> bool:
    """Final results beat timeout fallbacks; among fallbacks the longer timeout wins."""
    new_final, old_final = new[1] != "timeout", old[1] != "timeout"
    if new_final != old_final:
        return new_final
    if not new_final:
        return new[2] > old[2]
    return False


class _FileLock:
    """Exclusive advisory lock (fcntl) on a side file; no-op where fcntl is missing."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.fh: Any = None

    def __enter__(self) -> "_FileLock":
        try:
            import fcntl
        except ImportError:
            return self
        self.fh = open(self.path, "a+")  # noqa: SIM115
        fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.fh is not None:
            import fcntl

            fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
            self.fh.close()


_CACHES: dict[Path, _Cache] = {}


def _cache_for(path: Path | str | None) -> _Cache:
    p = Path(path) if path is not None else default_cache_path()
    p = p.resolve()
    if p not in _CACHES:
        _CACHES[p] = _Cache(p)
    c = _CACHES[p]
    c.load()
    return c


def flush_mkey_cache(cache_path: Path | str | None = None) -> None:
    """Write pending cache entries (only needed after ``mkeys(..., flush=False)``)."""
    _cache_for(cache_path).flush()


# --- batch keys -----------------------------------------------------------------------

def mkeys(smiles: Iterable[str | None], cfg: Config | None = None, *, n_jobs: int | None = None,
          timeout_s: float | None = _UNSET, cache: bool = True,
          cache_path: Path | str | None = None, flush: bool = True,
          flush_every: int = 20000, return_status: bool = False) -> list[Any]:
    """Metric keys for many SMILES, in input order.

    - ``n_jobs``: worker processes (default ``cfg.n_jobs``).
    - ``timeout_s``: per-molecule limit (default ``cfg.mkey_timeout_s``; None disables it).
      A molecule over the limit gets the plain InChIKey first block (status "timeout").
      With a timeout, keys are computed in worker processes even for ``n_jobs=1``;
      without one and with ``n_jobs <= 1`` they are computed in this process.
    - ``cache``: read and write the parquet cache at ``cache_path`` (default
      ``default_cache_path()``). Several processes may share it: writes happen under a
      file lock, via a temp file and an atomic rename. Entries are tagged with the RDKit
      version and ignored under another version. A read-only location (Kaggle) only
      logs a warning. ``flush=False`` defers writing until ``flush_mkey_cache``.
    - ``return_status``: return ``(key, status)`` pairs (``STATUSES``) instead of keys.
    """
    cfg = cfg or Config()
    if n_jobs is None:
        n_jobs = cfg.n_jobs
    if timeout_s is _UNSET:
        timeout_s = cfg.mkey_timeout_s
    if timeout_s is not None and timeout_s <= 0:
        timeout_s = None
    items = list(smiles)
    store = _cache_for(cache_path) if cache else None

    found: dict[str, tuple[str | None, str]] = {}
    todo: list[str] = []
    for s in sorted({s for s in items if isinstance(s, str) and s}):
        hit = store.get(s, timeout_s) if store is not None else _UNSET
        if hit is _UNSET:
            todo.append(s)
        else:
            found[s] = hit
    if todo:
        t0 = time.perf_counter()
        block = max(1, flush_every) if store is not None else max(1, len(todo))
        for start in range(0, len(todo), block):
            part = todo[start:start + block]
            if timeout_s is None and n_jobs <= 1:
                res = [_compute(s) for s in part]
            else:
                res = _run_workers(part, n_jobs, timeout_s)
            for s, (key, status) in zip(part, res):
                found[s] = (key, status)
                if store is not None:
                    store.put(s, key, status, timeout_s)
            if store is not None and flush:
                store.flush()
        dt = time.perf_counter() - t0
        n_to = sum(1 for s in todo if found[s][1] == "timeout")
        log.log(logging.INFO if len(todo) >= 1000 or n_to else logging.DEBUG,
                "mkeys: %d unique, %d computed in %.1f s (%d timeouts), %d from cache",
                len(found), len(todo), dt, n_to, len(found) - len(todo))
    out = []
    for s in items:
        r = found.get(s) if isinstance(s, str) and s else None
        if r is None:
            r = (None, "fail")
        out.append(r if return_status else r[0])
    return out


# --- MRR@25 ---------------------------------------------------------------------------

def split_guesses(value: str | Sequence[str] | None) -> list[str]:
    """A submission cell (``"A;B;C"``) or a list -> list of stripped guesses (order kept,
    empty guesses kept as "" so that they still occupy a rank)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        return [g.strip() for g in value.split(";")]
    return [str(g).strip() if g is not None else "" for g in value]


def reciprocal_ranks(pred: Mapping[str, str | Sequence[str]], truth: Mapping[str, str],
                     k: int = 25, keys: Mapping[str, str | None] | None = None,
                     n_jobs: int = 1, cache: bool = False, timeout_s: float | None = None,
                     cache_path: Path | str | None = None) -> dict[str, float]:
    """1/rank of the first of the first ``k`` guesses whose mkey equals the truth's mkey,
    0 if none, for every molecule in ``truth`` (missing from ``pred``: 0).

    ``keys`` (SMILES -> mkey) short-cuts key computation for SMILES it contains; the rest
    are computed with ``mkeys`` (defaults: exact, in-process, no cache, no timeout).
    A truth whose mkey is None scores 0 (logged).
    """
    guesses = {mid: split_guesses(pred.get(mid))[:k] for mid in truth}
    needed = set(truth.values()) | {g for gs in guesses.values() for g in gs}
    needed = {s for s in needed if isinstance(s, str) and s}
    known: dict[str, str | None] = {}
    if keys is not None:
        known = {s: keys[s] for s in needed if s in keys}
    missing = sorted(needed - set(known))
    if missing:
        known.update(zip(missing, mkeys(missing, n_jobs=n_jobs, timeout_s=timeout_s,
                                        cache=cache, cache_path=cache_path)))
    out: dict[str, float] = {}
    bad_truth = 0
    for mid in sorted(truth):
        tkey = known.get(truth[mid]) if isinstance(truth[mid], str) else None
        rr = 0.0
        if tkey is None:
            bad_truth += 1
        else:
            for rank, g in enumerate(guesses[mid], start=1):
                if known.get(g) == tkey:
                    rr = 1.0 / rank
                    break
        out[mid] = rr
    if bad_truth:
        log.warning("reciprocal_ranks: %d truth structures have no mkey (scored 0)", bad_truth)
    return out


def mrr_at_25(pred: Mapping[str, str | Sequence[str]], truth: Mapping[str, str],
              **kwargs: Any) -> float:
    """MRR@25 (PLAN 5.1): mean over all ``truth`` molecules of ``reciprocal_ranks``.

    ``pred``: molecule_id -> list of SMILES (best first) or a ";"-joined string.
    ``truth``: molecule_id -> SMILES. Keyword arguments go to ``reciprocal_ranks``.
    """
    if not truth:
        raise ValueError("mrr_at_25: empty truth")
    rr = reciprocal_ranks(pred, truth, **kwargs)
    return float(np.mean([rr[m] for m in sorted(rr)]))

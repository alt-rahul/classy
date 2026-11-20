#!/usr/bin/env python3
"""WP3 - build the PubChem candidate tier.

Output: data/processed/db/pubchem/mass_XXXX.parquet (one shard per 10 Da of
monoisotopic mass, 100-1300 Da) plus _index.json, schema as in docs/PLAN.md 2.3:

    ik14 str, smiles str, formula str, mono_mass f64, cid i32, pmid_count i32, n_cids i16

Stages (each resumable, markers in data/external/pubchem/.stage_*):

  1  stream the four PubChem Extras files (CID-SMILES, CID-Mass, CID-InChI-Key,
     CID-PMID) over HTTPS with range-resume, gunzip in memory and write parquet
     keyed by CID. Nothing raw is stored on disk. md5 is checked against the
     .md5 file published next to each source.
  2  DuckDB join on CID in CID-range chunks (memory_limit 7GB, 2 threads),
     string-level filters of PLAN.md 2.2 with row counts logged after every
     filter, write partitioned by floor(mono_mass / 10).
  3  per shard: group by ik14, keep the lowest-CID row, max pmid_count,
     n_cids = group size, sort by mono_mass, zstd level 9.
  priors   R3 documentation priors: stream CID-InChI-Key (all CIDs, for the
           ik14 groups), CID-SID, CID-Synonym-filtered, CID-Patent, CID-MeSH and
           Substance SID-Map (+ RDF source categories), count distinct items per
           ik14 over all its CIDs and append sid_count, synonym_count,
           pc_patent_count, mesh_count, n_vendors, n_depositors to every shard
           (same rows, order and row groups). Also writes np_priors.parquet in
           the work dir for build_npdb.py --stage priors. Sub-steps resumable.
  check    acceptance checks (row totals, ik14 uniqueness, caffeine and
           paclitaxel at 10 ppm, 20 random public-test masses timed).
  cleanup  delete intermediates (stage 1 and 2 parquet, DuckDB temp dir).
  docs     write or refresh the WP3 section of docs/ARTEFACTS.md.

Usage:
    nice -n 10 python3 scripts/build_pubchem.py                 # all stages
    python3 scripts/build_pubchem.py --stage check              # one stage
    python3 scripts/build_pubchem.py --test --max-bytes 30000000  # small dry run in a scratch dir

RDKit parsing is deliberately not done here (PLAN.md 2.2: checked lazily at
inference for retrieved candidates only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import resource
import shutil
import subprocess
import sys
import time
import warnings
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.compute as pc
import pyarrow.parquet as pq

try:  # duckdb is a build-time dependency only
    import duckdb
except ImportError:  # pragma: no cover
    sys.exit("duckdb is required: pip install duckdb")


# ---------------------------------------------------------------- paths

def _paths() -> tuple:
    """Directories from casmi.paths when importable, else the same defaults derived
    from this file's location (the script must also run before casmi is installed)."""
    here = Path(__file__).resolve().parents[1]
    try:
        src = str(here / "src")
        if src not in sys.path:
            sys.path.append(src)
        import casmi.paths as cp  # type: ignore

        return (Path(cp.ROOT), Path(cp.EXTERNAL), Path(cp.PROCESSED), Path(cp.LOGS),
                Path(cp.TEST_PARQUET), Path(cp.DOCS))
    except Exception:
        return (here, here / "data/external", here / "data/processed", here / "logs",
                here / "data/raw/test.parquet", here / "docs")


ROOT, EXTERNAL, PROCESSED, LOGS, TEST_PARQUET, DOCS = _paths()
BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras"

# name -> (file, column names, pyarrow types, columns kept)
SOURCES = {
    "smiles": ("CID-SMILES.gz", ["cid", "smiles"], {"cid": pa.int32(), "smiles": pa.string()},
               ["cid", "smiles"]),
    "mass": ("CID-Mass.gz", ["cid", "formula", "mono_mass", "exact_mass"],
             {"cid": pa.int32(), "formula": pa.string(), "mono_mass": pa.float64()},
             ["cid", "formula", "mono_mass"]),
    "ik": ("CID-InChI-Key.gz", ["cid", "inchi", "inchikey"],
           {"cid": pa.int32(), "inchikey": pa.string()}, ["cid", "inchikey"]),
    "pmid": ("CID-PMID.gz", ["cid", "pmid", "type"],
             {"cid": pa.int32(), "pmid": pa.int32(), "type": pa.int8()}, ["cid", "pmid", "type"]),
}
STAGE1_ORDER = ["mass", "smiles", "pmid", "ik"]

MASS_MIN, MASS_MAX, BIN_DA = 100.0, 1300.0, 10
ELEMENT_RE = r"^(Cl|Br|C|H|N|O|P|S|F|I|[0-9])+$"  # whitelist C,H,N,O,P,S,F,Cl,Br,I
ISOTOPE_RE = r"\[[0-9]"  # [13C], [2H], [18O] ...
OUT_SCHEMA = pa.schema([
    ("ik14", pa.string()), ("smiles", pa.string()), ("formula", pa.string()),
    ("mono_mass", pa.float64()), ("cid", pa.int32()), ("pmid_count", pa.int32()),
    ("n_cids", pa.int16()),
])

# PLAN.md 3.1: precursor_mz = M + shift
ADDUCT_SHIFT = {
    "[M+H]+": 1.007276, "[M+NH4]+": 18.033826, "[M-H2O+H]+": -17.003288,
    "[M-2H2O+H]+": -35.013853, "[M+Na]+": 22.989221, "[M+K]+": 38.963158,
    "[M-H]-": -1.007276, "[M-H2O-H]-": -19.017841, "[M+CH2O2-H]-": 44.998203,
    "[M+Cl]-": 34.969401,
}
SEED = 20261118

log = logging.getLogger("build_pubchem")
warnings.filterwarnings("ignore", message="Arrow output on a DB-API result")  # small tables only


class Cfg:
    def __init__(self, args):
        if args.test:
            base = Path(args.work_dir or (EXTERNAL / "pubchem_test"))
            self.work = base
            self.out = Path(args.out_dir or (base / "out"))
        else:
            self.work = Path(args.work_dir or (EXTERNAL / "pubchem"))
            self.out = Path(args.out_dir or (PROCESSED / "db" / "pubchem"))
        self.s1 = self.work / "stage1"
        self.s2 = self.work / "stage2"
        self.tmp = Path(args.tmp_dir or (EXTERNAL / "tmp" / ("pubchem_test" if args.test else "pubchem")))
        self.stats_path = self.work / "build_stats.json"
        self.max_bytes = args.max_bytes
        self.chunk_cids = args.chunk_cids
        self.memory_limit = args.memory_limit
        self.threads = args.threads
        self.keep_intermediates = args.keep_intermediates
        self.test = args.test
        for d in (self.work, self.out):  # stage dirs are created by the stages that use them
            d.mkdir(parents=True, exist_ok=True)

    def marker(self, name: str) -> Path:
        return self.work / f".stage_{name}"


# ---------------------------------------------------------------- helpers

def setup_logging(stage: str) -> Path:
    logs = LOGS
    logs.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = logs / f"build_pubchem_{stage}_{ts}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(path)):
        h.setFormatter(fmt)
        log.addHandler(h)
    return path


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def anon_rss_mb() -> float:
    """Current anonymous RSS (excludes file-backed pages of memory-mapped arrays)."""
    try:
        with open("/proc/self/status") as fh:
            for ln in fh:
                if ln.startswith("RssAnon:"):
                    return int(ln.split()[1]) / 1024.0
    except OSError:  # pragma: no cover
        pass
    return float("nan")


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def load_stats(cfg: Cfg) -> dict:
    if cfg.stats_path.exists():
        return json.loads(cfg.stats_path.read_text())
    return {}


def save_stats(cfg: Cfg, stats: dict) -> None:
    tmp = cfg.stats_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(stats, indent=2, sort_keys=True))
    tmp.replace(cfg.stats_path)


def update_stats(cfg: Cfg, key: str, value) -> None:
    stats = load_stats(cfg)
    stats[key] = value
    save_stats(cfg, stats)


def dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def fmt_n(n: int) -> str:
    return f"{n:,}"


def http_head(url: str) -> dict:
    out = subprocess.run(["curl", "-sSIL", "--fail", "--max-time", "60", url],
                         capture_output=True, text=True, check=True).stdout
    info = {}
    for line in out.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            info[k.strip().lower()] = v.strip()
    return info


def http_text(url: str) -> str:
    return subprocess.run(["curl", "-sSL", "--fail", "--max-time", "60", url],
                          capture_output=True, text=True, check=True).stdout


def duck(cfg: Cfg, memory_limit: str | None = None):
    cfg.tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit or cfg.memory_limit}'")
    con.execute(f"SET threads={cfg.threads}")
    con.execute(f"SET temp_directory='{cfg.tmp}'")
    con.execute("SET preserve_insertion_order=true")
    return con


# ---------------------------------------------------------------- stage 1: resumable gz stream

class GzHttpStream:
    """Read-only file object yielding the gunzipped bytes of a remote .gz file.

    The compressed bytes are fetched with curl (which already knows the proxy
    and CA bundle); when a transfer dies, curl is restarted with a byte range
    from the last offset and the same zlib state continues, so nothing is lost
    and nothing is written to disk. md5 of the compressed stream is computed on
    the fly. With max_bytes, the stream stops early at a line boundary (test).
    """

    def __init__(self, url: str, size: int, hdr_path: Path, max_bytes: int | None = None,
                 chunk: int = 1 << 20, max_restarts: int = 200):
        self.url, self.size, self.hdr_path = url, size, hdr_path
        self.max_bytes = max_bytes
        self.chunk = chunk
        self.max_restarts = max_restarts
        self.restarts = 0
        self.offset = 0
        self.md5 = hashlib.md5()
        self._d = zlib.decompressobj(31)
        self._buf = b""
        self._pos = 0
        self._eof = False
        self._proc = None
        self.out_bytes = 0
        self.closed = False
        self._open()

    # -- transport
    def _open(self):
        cmd = ["curl", "-sS", "--fail", "-L", "--connect-timeout", "60",
               "--speed-limit", "20000", "--speed-time", "120",
               "-D", str(self.hdr_path), "-r", f"{self.offset}-", self.url]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      bufsize=self.chunk)
        self._checked_status = False
        self._status_tries = 0
        self._resume_offset = self.offset

    def _check_status(self):
        # Early detection only: the last HTTP status line in the dumped headers
        # should be 206 (partial content). curl may not have flushed the header
        # file yet; then try again on later reads and finally rely on the md5.
        self._status_tries += 1
        try:
            lines = [ln for ln in self.hdr_path.read_text(errors="replace").splitlines()
                     if ln.startswith("HTTP/")]
        except FileNotFoundError:
            lines = []
        if not lines:
            if self._status_tries > 50:
                self._checked_status = True
            return
        status = lines[-1].split()[1]
        if status not in ("206", "200"):
            raise RuntimeError(f"unexpected HTTP status {status} for {self.url}")
        if self._resume_offset > 0 and status != "206":
            raise RuntimeError(f"server ignored Range on resume (status {status})")
        self._checked_status = True

    def _restart(self, reason: str):
        self.restarts += 1
        if self.restarts > self.max_restarts:
            raise RuntimeError(f"too many restarts for {self.url}: {reason}")
        wait = min(60, 2 ** min(self.restarts, 6))
        log.warning("stream %s cut at %s/%s bytes (%s); resuming in %ss (restart %d)",
                    self.url.rsplit("/", 1)[-1], fmt_n(self.offset), fmt_n(self.size),
                    reason.strip()[:200], wait, self.restarts)
        time.sleep(wait)
        self._open()

    def _feed(self, raw: bytes) -> bytes:
        out = []
        data = raw
        while data:
            if self._d.eof:  # multi-member gzip: next member starts here
                self._d = zlib.decompressobj(31)
            out.append(self._d.decompress(data))
            data = self._d.unused_data if self._d.eof else b""
        return b"".join(out)

    def _fill(self):
        """Refill self._buf with at least one byte unless EOF."""
        while not self._eof:
            if self.max_bytes is not None and self.offset >= self.max_bytes:
                self._kill()
                data = self._pending_tail
                cut = data.rfind(b"\n")
                self._set_buf(data[: cut + 1] if cut >= 0 else b"")
                self._pending_tail = b""
                self._eof = True
                return
            raw = self._proc.stdout.read(self.chunk)
            if raw:
                if not self._checked_status:
                    self._check_status()
                self.offset += len(raw)
                self.md5.update(raw)
                out = self._feed(raw)
                if self.max_bytes is not None:
                    # keep a tail so that the early stop ends on a newline
                    data = self._pending_tail + out
                    cut = data.rfind(b"\n")
                    self._pending_tail = data[cut + 1:]
                    out = data[: cut + 1]
                if out:
                    self._set_buf(out)
                    return
                continue
            # curl ended
            rc = self._proc.wait()
            err = self._proc.stderr.read().decode(errors="replace")
            if rc == 0 and self.offset >= self.size:
                tail = self._d.flush()
                if not self._d.eof:
                    raise RuntimeError(f"gzip stream of {self.url} is truncated (no end of member)")
                self._eof = True
                if tail:
                    self._set_buf(tail)
                return
            self._restart(f"curl rc={rc} {err}")

    _pending_tail = b""

    def _set_buf(self, data: bytes):
        self._buf = data
        self._pos = 0
        self.out_bytes += len(data)

    def _kill(self):
        if self._proc and self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait()

    # -- file protocol used by pyarrow.PythonFile
    def read(self, n: int = -1) -> bytes:
        parts = []
        want = n if n is not None and n >= 0 else float("inf")
        got = 0
        while got < want:
            if self._pos >= len(self._buf):
                if self._eof:
                    break
                self._fill()
                if self._pos >= len(self._buf):
                    break
            take = int(min(want - got, len(self._buf) - self._pos))
            parts.append(self._buf[self._pos:self._pos + take])
            self._pos += take
            got += take
        return b"".join(parts)

    def readable(self):
        return True

    def seekable(self):
        return False

    def writable(self):
        return False

    def tell(self):
        return self.out_bytes - (len(self._buf) - self._pos)

    def close(self):
        self._kill()
        self.closed = True

    @property
    def complete(self) -> bool:
        return self._eof and self.max_bytes is None and self.offset >= self.size


def stage1_file(cfg: Cfg, name: str) -> dict:
    fname, colnames, types, keep = SOURCES[name]
    url = f"{BASE_URL}/{fname}"
    out = cfg.s1 / f"{name}_raw.parquet" if name == "pmid" else cfg.s1 / f"{name}.parquet"
    head = http_head(url)
    size = int(head["content-length"])
    md5_expected = None
    try:
        md5_expected = http_text(url + ".md5").split()[0]
    except Exception as e:  # pragma: no cover
        log.warning("no md5 for %s: %s", fname, e)
    log.info("stage1 %s: %s, %s bytes, last-modified %s, md5 %s", name, url, fmt_n(size),
             head.get("last-modified"), md5_expected)

    t0 = time.time()
    stream = GzHttpStream(url, size, cfg.tmp / f"{name}.hdr", max_bytes=cfg.max_bytes)
    bad_rows = []

    def on_invalid(row):
        if len(bad_rows) < 20:
            bad_rows.append(f"{row.actual_columns} cols: {row.text[:120]!r}")
        else:
            bad_rows.append(None)
        return "skip"

    reader = pacsv.open_csv(
        pa.PythonFile(stream, mode="r"),
        read_options=pacsv.ReadOptions(column_names=colnames, block_size=64 << 20, use_threads=True),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False, escape_char=False,
                                         newlines_in_values=False, invalid_row_handler=on_invalid),
        convert_options=pacsv.ConvertOptions(column_types=types, include_columns=keep,
                                             strings_can_be_null=False, quoted_strings_can_be_null=False),
    )
    if name == "ik":
        schema = pa.schema([("cid", pa.int32()), ("ik14", pa.string())])
    else:
        schema = pa.schema([(c, types[c]) for c in keep])
    tmp_out = out.with_suffix(".parquet.part")
    writer = pq.ParquetWriter(tmp_out, schema, compression="zstd", compression_level=3)
    n = 0
    n_null_cid = 0
    n_badkey = 0
    sorted_ok = True
    prev_max = -1
    last_log = time.time()
    first_rows = None
    try:
        for batch in reader:
            tbl = pa.Table.from_batches([batch])
            if first_rows is None:
                first_rows = tbl.slice(0, 3).to_pylist()
            null_cid = tbl["cid"].null_count
            if null_cid:
                n_null_cid += null_cid
                tbl = tbl.filter(pc.is_valid(tbl["cid"]))
            if name == "ik":
                key = tbl["inchikey"]
                ok = pc.equal(pc.utf8_length(key), 27)
                n_badkey += len(key) - pc.sum(ok).as_py() if len(key) else 0
                tbl = tbl.filter(ok)
                tbl = pa.table({"cid": tbl["cid"], "ik14": pc.utf8_slice_codeunits(tbl["inchikey"], 0, 14)})
            if len(tbl) == 0:
                continue
            mn, mx = pc.min_max(tbl["cid"]).values()
            mn, mx = mn.as_py(), mx.as_py()
            if mn <= prev_max:
                sorted_ok = False
            # also check order inside the batch (cheap enough)
            if sorted_ok and name != "pmid":
                c = tbl["cid"].combine_chunks() if hasattr(tbl["cid"], "combine_chunks") else tbl["cid"]
                if len(c) > 1 and pc.any(pc.less_equal(c[1:], c[:-1])).as_py():
                    sorted_ok = False
            prev_max = max(prev_max, mx)
            writer.write_table(tbl.cast(schema))
            n += len(tbl)
            if time.time() - last_log > 60:
                el = time.time() - t0
                log.info("  %s: %s rows, %.1f%% of %s compressed bytes, %.1f MB/s, restarts %d, rss %.0f MB",
                         name, fmt_n(n), 100.0 * stream.offset / size, fmt_n(size),
                         stream.offset / el / 1e6, stream.restarts, peak_rss_mb())
                last_log = time.time()
    finally:
        writer.close()
        stream.close()
    if cfg.max_bytes is None:
        if not stream.complete:
            raise RuntimeError(f"stream for {fname} ended early at {stream.offset} of {size}")
        md5_got = stream.md5.hexdigest()
        if md5_expected and md5_got != md5_expected:
            raise RuntimeError(f"md5 mismatch for {fname}: {md5_got} != {md5_expected}")
    tmp_out.replace(out)
    el = time.time() - t0
    n_bad = len(bad_rows)
    info = {
        "url": url, "bytes": size, "last_modified": head.get("last-modified"),
        "md5_expected": md5_expected, "md5_ok": (stream.md5.hexdigest() == md5_expected) if cfg.max_bytes is None else None,
        "rows": n, "invalid_rows_skipped": n_bad, "invalid_row_examples": [b for b in bad_rows if b][:5],
        "null_cid_rows": n_null_cid, "bad_inchikey_rows": n_badkey, "sorted_by_cid": sorted_ok,
        "max_cid": prev_max, "restarts": stream.restarts, "seconds": round(el, 1),
        "decompressed_bytes": stream.out_bytes, "parquet_bytes": out.stat().st_size,
        "columns_in_file": colnames, "first_rows": first_rows,
    }
    log.info("stage1 %s done: %s rows (skipped %s invalid, %s null cid, %s bad keys), sorted=%s, "
             "max cid %s, %.0f s, %.1f MB/s, parquet %.2f GB, restarts %d",
             name, fmt_n(n), fmt_n(n_bad), fmt_n(n_null_cid), fmt_n(n_badkey), sorted_ok,
             fmt_n(prev_max), el, stream.offset / max(el, 1e-9) / 1e6, out.stat().st_size / 1e9, stream.restarts)
    if n_bad:
        log.info("  invalid row examples: %s", info["invalid_row_examples"])
    log.info("  first rows: %s", first_rows)
    return info


def stage1(cfg: Cfg):
    cfg.s1.mkdir(parents=True, exist_ok=True)
    cfg.tmp.mkdir(parents=True, exist_ok=True)
    for name in STAGE1_ORDER:
        m = cfg.marker(f"1_{name}")
        if m.exists():
            log.info("stage1 %s: marker present, skipping", name)
            continue
        info = stage1_file(cfg, name)
        if name == "pmid":
            info.update(stage1_pmid_agg(cfg))
        update_stats(cfg, f"stage1_{name}", info)
        m.write_text(json.dumps({"rows": info["rows"], "done": datetime.now(timezone.utc).isoformat()}))
    if not cfg.marker("1").exists():
        cfg.marker("1").write_text(datetime.now(timezone.utc).isoformat())


def stage1_pmid_agg(cfg: Cfg) -> dict:
    """CID-PMID is not sorted by CID: aggregate to (cid, pmid_count) sorted by CID."""
    t0 = time.time()
    raw = cfg.s1 / "pmid_raw.parquet"
    out = cfg.s1 / "pmid.parquet"
    con = duck(cfg)
    con.execute(f"""
        COPY (
            SELECT cid, CAST(count(DISTINCT pmid) AS INTEGER) AS pmid_count
            FROM read_parquet('{raw}') GROUP BY cid ORDER BY cid
        ) TO '{out}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)
    """)
    n, mx, tot = con.execute(f"SELECT count(*), max(pmid_count), sum(pmid_count) FROM read_parquet('{out}')").fetchone()
    by_type = con.execute(f"SELECT type, count(*) FROM read_parquet('{raw}') GROUP BY type ORDER BY type").fetchall()
    con.close()
    raw.unlink()
    log.info("stage1 pmid aggregated: %s CIDs with >= 1 PMID, max pmid_count %s, rows by type %s, %.0f s",
             fmt_n(n), fmt_n(mx), by_type, time.time() - t0)
    return {"cids_with_pmid": n, "max_pmid_count": mx, "sum_pmid_count": int(tot or 0),
            "rows_by_type": {str(t): c for t, c in by_type}}


# ---------------------------------------------------------------- stage 2: join and filter

FILTER_STEPS = [
    ("cids_in_CID-Mass", "TRUE"),
    ("has_smiles", "smiles IS NOT NULL AND smiles <> ''"),
    ("has_inchikey", "ik14 IS NOT NULL"),
    ("neutral_formula_no_charge_suffix",
     "formula IS NOT NULL AND formula <> '' AND NOT contains(formula, '+') AND NOT contains(formula, '-')"),
    ("elements_CHNOPSFClBrI", f"regexp_full_match(formula, '{ELEMENT_RE}')"),
    ("at_least_3_C", "n_c >= 3"),
    ("mass_100_1300", f"mono_mass >= {MASS_MIN} AND mono_mass < {MASS_MAX}"),
    ("single_component_no_dot", "NOT contains(smiles, '.')"),
    ("no_isotope_spec", f"NOT regexp_matches(smiles, '{ISOTOPE_RE}')"),
]
N_C_SQL = ("CASE WHEN regexp_matches(formula, '^C([0-9]|[A-Z]|$)') "
           "THEN coalesce(TRY_CAST(nullif(regexp_extract(formula, '^C([0-9]*)', 1), '') AS INTEGER), 1) "
           "ELSE 0 END")


def stage2(cfg: Cfg):
    if cfg.marker("2").exists():
        log.info("stage2: marker present, skipping")
        return
    t0 = time.time()
    cfg.s2.mkdir(parents=True, exist_ok=True)
    con = duck(cfg)
    s1 = cfg.s1
    max_cid = con.execute(f"SELECT max(cid) FROM read_parquet('{s1 / 'mass.parquet'}')").fetchone()[0]
    for name in ("smiles", "ik"):
        mc = con.execute(f"SELECT max(cid) FROM read_parquet('{s1 / (name + '.parquet')}')").fetchone()[0]
        max_cid = max(max_cid, mc)
    n_chunks = max_cid // cfg.chunk_cids + 1
    log.info("stage2: max cid %s, %d chunks of %s CIDs, memory_limit %s, threads %d",
             fmt_n(max_cid), n_chunks, fmt_n(cfg.chunk_cids), cfg.memory_limit, cfg.threads)
    for i in range(n_chunks):
        lo, hi = i * cfg.chunk_cids, (i + 1) * cfg.chunk_cids
        final = cfg.s2 / f"chunk_{i:03d}"
        done = cfg.s2 / f"chunk_{i:03d}.json"
        if done.exists() and final.exists():
            continue
        if final.exists():
            shutil.rmtree(final)
        tc = time.time()
        rng = f"cid >= {lo} AND cid < {hi}"
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE j AS
            SELECT m.cid, m.formula, m.mono_mass, s.smiles, k.ik14,
                   coalesce(p.pmid_count, 0) AS pmid_count, {N_C_SQL} AS n_c
            FROM (SELECT cid, formula, mono_mass FROM read_parquet('{s1 / 'mass.parquet'}') WHERE {rng}) m
            LEFT JOIN (SELECT cid, smiles FROM read_parquet('{s1 / 'smiles.parquet'}') WHERE {rng}) s USING (cid)
            LEFT JOIN (SELECT cid, ik14 FROM read_parquet('{s1 / 'ik.parquet'}') WHERE {rng}) k USING (cid)
            LEFT JOIN (SELECT cid, pmid_count FROM read_parquet('{s1 / 'pmid.parquet'}') WHERE {rng}) p USING (cid)
        """)
        cum = []
        conds = []
        for step, cond in FILTER_STEPS:
            conds.append(f"({cond})")
            cum.append(f"count(*) FILTER (WHERE {' AND '.join(conds)}) AS \"{step}\"")
        row = con.execute(f"SELECT {', '.join(cum)}, count(DISTINCT cid) AS distinct_cid FROM j").fetchone()
        counts = dict(zip([s for s, _ in FILTER_STEPS] + ["distinct_cid"], row))
        if counts["distinct_cid"] != counts["cids_in_CID-Mass"]:
            raise RuntimeError(f"chunk {i}: duplicate CIDs after join: {counts}")
        part = cfg.s2 / f"chunk_{i:03d}.part"
        if part.exists():
            shutil.rmtree(part)
        all_cond = " AND ".join(f"({c})" for _, c in FILTER_STEPS)
        con.execute(f"""
            COPY (
                SELECT CAST(cid AS INTEGER) AS cid, ik14, smiles, formula, mono_mass,
                       CAST(pmid_count AS INTEGER) AS pmid_count,
                       CAST(floor(mono_mass / {BIN_DA}) AS INTEGER) AS bucket
                FROM j WHERE {all_cond}
            ) TO '{part}' (FORMAT parquet, PARTITION_BY (bucket), COMPRESSION zstd)
        """)
        con.execute("DROP TABLE j")
        part.rename(final)
        counts["seconds"] = round(time.time() - tc, 1)
        done.write_text(json.dumps(counts))
        log.info("stage2 chunk %d/%d [%s, %s): %s -> %s rows, %.0f s, rss %.0f MB", i + 1, n_chunks,
                 fmt_n(lo), fmt_n(hi), fmt_n(counts["cids_in_CID-Mass"]),
                 fmt_n(counts[FILTER_STEPS[-1][0]]), counts["seconds"], peak_rss_mb())
    con.close()
    # totals
    tot = {s: 0 for s, _ in FILTER_STEPS}
    for f in sorted(cfg.s2.glob("chunk_*.json")):
        c = json.loads(f.read_text())
        for s, _ in FILTER_STEPS:
            tot[s] += c[s]
    log.info("stage2 filter cascade (rows remaining after each filter, cumulative):")
    prev = None
    for s, _ in FILTER_STEPS:
        drop = "" if prev is None else f"  (-{fmt_n(prev - tot[s])})"
        log.info("  %-36s %15s%s", s, fmt_n(tot[s]), drop)
        prev = tot[s]
    el = time.time() - t0
    update_stats(cfg, "stage2", {"filter_cascade": tot, "seconds": round(el, 1), "n_chunks": n_chunks,
                                 "chunk_cids": cfg.chunk_cids, "max_cid": max_cid,
                                 "stage2_bytes": dir_size(cfg.s2), "peak_rss_mb": round(peak_rss_mb())})
    if not cfg.test and not (40e6 < tot[FILTER_STEPS[-1][0]] < 200e6):
        raise RuntimeError(f"stage2 row total implausible: {tot}")
    cfg.marker("2").write_text(json.dumps(tot))
    log.info("stage2 done in %.0f s; stage2 size %.2f GB", el, dir_size(cfg.s2) / 1e9)
    if not cfg.keep_intermediates:
        n1 = dir_size(cfg.s1)
        shutil.rmtree(cfg.s1)
        log.info("deleted stage1 parquet (%.2f GB) to keep the disk peak low", n1 / 1e9)


# ---------------------------------------------------------------- stage 3: per-shard dedup

def schema_ok(sch: pa.Schema) -> bool:
    """Same column names, order and types as OUT_SCHEMA, or OUT_SCHEMA followed by the
    R3 prior columns (PRIOR_SCHEMA, after --stage priors); nullability ignored."""
    got = [(f.name, f.type) for f in sch]
    return got in ([(f.name, f.type) for f in OUT_SCHEMA], [(f.name, f.type) for f in PRIOR_SCHEMA])


def shard_name(bucket: int) -> str:
    return f"mass_{bucket * BIN_DA:04d}.parquet"


def stage3(cfg: Cfg):
    if cfg.marker("3").exists():
        log.info("stage3: marker present, skipping")
        return
    t0 = time.time()
    con = duck(cfg)
    b_lo, b_hi = int(MASS_MIN // BIN_DA), int(math.ceil(MASS_MAX / BIN_DA))
    shard_meta_dir = cfg.s2 / "_shards"
    shard_meta_dir.mkdir(parents=True, exist_ok=True)
    for b in range(b_lo, b_hi):
        meta_path = shard_meta_dir / f"{b:04d}.json"
        out = cfg.out / shard_name(b)
        if meta_path.exists() and out.exists():
            continue
        files = sorted(cfg.s2.glob(f"chunk_*/bucket={b}/*.parquet"))
        if not files:
            log.warning("stage3: bucket %d has no input rows", b)
            continue
        ts = time.time()
        flist = "[" + ", ".join(f"'{f}'" for f in files) + "]"
        tmp = out.with_suffix(".parquet.part")
        con.execute(f"""
            COPY (
                SELECT ik14,
                       arg_min(smiles, cid) AS smiles,
                       arg_min(formula, cid) AS formula,
                       arg_min(mono_mass, cid) AS mono_mass,
                       CAST(min(cid) AS INTEGER) AS cid,
                       CAST(max(pmid_count) AS INTEGER) AS pmid_count,
                       CAST(least(count(*), 32767) AS SMALLINT) AS n_cids
                FROM read_parquet({flist}, hive_partitioning=false)
                GROUP BY ik14
                ORDER BY mono_mass, cid
            ) TO '{tmp}' (FORMAT parquet, COMPRESSION zstd, COMPRESSION_LEVEL 9, ROW_GROUP_SIZE 100000)
        """)
        n_in, n_conflict = con.execute(f"""
            SELECT sum(n), count(*) FILTER (WHERE nf > 1) FROM (
                SELECT count(*) AS n, count(DISTINCT formula) AS nf
                FROM read_parquet({flist}, hive_partitioning=false) GROUP BY ik14)
        """).fetchone()
        n, n_ik, mn, mx = con.execute(f"""
            SELECT count(*), count(DISTINCT ik14), min(mono_mass), max(mono_mass)
            FROM read_parquet('{tmp}')
        """).fetchone()
        if n != n_ik:
            raise RuntimeError(f"bucket {b}: ik14 not unique ({n} rows, {n_ik} distinct)")
        sch = pq.read_schema(tmp)
        if not schema_ok(sch):
            raise RuntimeError(f"bucket {b}: schema mismatch\n{sch}\nexpected\n{OUT_SCHEMA}")
        tmp.replace(out)
        meta = {"file": out.name, "rows": n, "rows_before_dedup": int(n_in), "min_mass": mn, "max_mass": mx,
                "ik14_with_several_formulas": n_conflict,
                "bytes": out.stat().st_size, "sha256": sha256_file(out),
                "row_groups": pq.ParquetFile(out).metadata.num_row_groups}
        meta_path.write_text(json.dumps(meta))
        log.info("stage3 %s: %s -> %s rows (%d ik14 with >1 formula), %.1f MB, %.1f s", out.name,
                 fmt_n(int(n_in)), fmt_n(n), n_conflict, meta["bytes"] / 1e6, time.time() - ts)
    con.close()
    shards = [json.loads(p.read_text()) for p in sorted(shard_meta_dir.glob("*.json"))]
    total = sum(s["rows"] for s in shards)
    total_in = sum(s["rows_before_dedup"] for s in shards)
    size = sum(s["bytes"] for s in shards)
    stats = load_stats(cfg)
    el = time.time() - t0
    stats["stage3"] = {"seconds": round(el, 1), "rows_before_dedup": total_in, "rows": total,
                       "n_shards": len(shards), "bytes": size,
                       "ik14_with_several_formulas": sum(s.get("ik14_with_several_formulas", 0) for s in shards)}
    save_stats(cfg, stats)
    write_index(cfg, shards, stats)
    log.info("stage3 done: %d shards, %s rows before ik14 dedup, %s after, %.2f GB, %.0f s",
             len(shards), fmt_n(total_in), fmt_n(total), size / 1e9, el)
    cfg.marker("3").write_text(json.dumps({"rows": total, "shards": len(shards)}))


def ordered_cascade(casc: dict | None) -> dict:
    """Filter counts in the order the filters are applied (stats JSON is key-sorted)."""
    casc = casc or {}
    return {s: casc[s] for s, _ in FILTER_STEPS if s in casc}


def write_index(cfg: Cfg, shards: list, stats: dict):
    sources = {}
    for name in STAGE1_ORDER:
        s = stats.get(f"stage1_{name}", {})
        sources[SOURCES[name][0]] = {k: s.get(k) for k in ("url", "bytes", "last_modified", "md5_expected",
                                                           "md5_ok", "rows", "columns_in_file")}
    idx = {
        "description": "PubChem candidate tier (docs/PLAN.md 2.3), built by scripts/build_pubchem.py",
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "bin_da": BIN_DA, "mass_min": MASS_MIN, "mass_max": MASS_MAX,
        "shard_pattern": "mass_XXXX.parquet, XXXX = 10 * floor(mono_mass / 10), zero padded to 4 digits",
        "sort": "mono_mass, then cid; row groups of 100000 rows, zstd level 9",
        "schema": {f.name: str(f.type) for f in OUT_SCHEMA},
        "total_rows": sum(s["rows"] for s in shards),
        "total_bytes": sum(s["bytes"] for s in shards),
        "n_shards": len(shards),
        "filter_cascade": ordered_cascade(stats.get("stage2", {}).get("filter_cascade")),
        "rows_before_ik14_dedup": sum(s["rows_before_dedup"] for s in shards),
        "sources": sources,
        "shards": {s["file"]: {k: s[k] for k in ("rows", "min_mass", "max_mass", "bytes", "sha256")}
                   for s in shards},
    }
    (cfg.out / "_index.json").write_text(json.dumps(idx, indent=1))


# ---------------------------------------------------------------- priors (R3): documentation counts

# Appended to every shard by --stage priors (docs/ROADMAP.md R3). One ik14 group = every
# PubChem CID whose InChIKey (CID-InChI-Key, all CIDs, no structure filter) starts with the
# ik14; counts are of DISTINCT items over the whole group, so a SID, name or patent linked
# to two stereo, isotope or charge variants of the ik14 is counted once.
PRIOR_FIELDS = [
    ("sid_count", pa.int32()),        # distinct SIDs over CID-SID types 1 (standardized) and 2 (component)
    ("synonym_count", pa.int32()),    # distinct case-folded names of CID-Synonym-filtered
    ("pc_patent_count", pa.int32()),  # distinct publication numbers of CID-Patent
    ("mesh_count", pa.int32()),       # distinct MeSH terms of CID-MeSH
    ("n_vendors", pa.int16()),        # distinct depositors in RDF category Chemical_Vendors (SID-Map)
    ("n_depositors", pa.int16()),     # distinct depositors of any category except NIST sources (SID-Map)
]
PRIOR_NAMES = [n for n, _ in PRIOR_FIELDS]
PRIOR_SCHEMA = pa.schema(list(OUT_SCHEMA) + [pa.field(n, t) for n, t in PRIOR_FIELDS])
SUBSTANCE_URL = "https://ftp.ncbi.nlm.nih.gov/pubchem/Substance/Extras"
SOURCE_RDF_URL = "https://ftp.ncbi.nlm.nih.gov/pubchem/RDF/source/pc_source.ttl.gz"
NIST_RE = re.compile(r"\bNIST\b|National Institute of Standards", re.I)
VENDOR_CATEGORY = "concept:Chemical_Vendors"
PART_BITS = 20                     # pair files hold 2**20 consecutive groups each
ITEM_BITS = 64 - PART_BITS         # packed key = (group within part) << ITEM_BITS | item
PRIOR_STEPS = ["map", "sid", "synonym", "patent", "sidmap", "mesh", "write"]
NP_PRIORS_NAME = "np_priors.parquet"
PRIOR_CHECK = (("caffeine", "RYYVLZVUVIJVGH", 194.0804), ("quercetin", "REFJWTPEDVJJIY", 302.0427),
               ("paclitaxel", "RCINICONZNJXQF", 853.3310))
LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]


def mem_available_gb() -> float:
    try:
        with open("/proc/meminfo") as fh:
            for ln in fh:
                if ln.startswith("MemAvailable:"):
                    return int(ln.split()[1]) / 1024.0 ** 2
    except OSError:  # pragma: no cover
        pass
    return float("inf")


def wait_for_memory(need_gb: float, what: str, max_wait_s: int = 3 * 3600) -> float:
    """Block until MemAvailable >= need_gb (the machine is shared; never risk an OOM kill)."""
    t0 = time.time()
    while True:
        avail = mem_available_gb()
        if avail >= need_gb:
            log.info("%s: %.1f GB available (need %.1f GB)", what, avail, need_gb)
            return avail
        if time.time() - t0 > max_wait_s:
            raise RuntimeError(f"{what}: only {avail:.1f} GB available after {max_wait_s} s")
        log.warning("%s: %.1f GB available < %.1f GB needed; waiting 60 s", what, avail, need_gb)
        time.sleep(60)


def prior_memory_limit(cfg: Cfg) -> str:
    """DuckDB memory_limit for the priors stage: min(--memory-limit, 6 GB, available - 1 GB)."""
    m = re.match(r"^\s*([0-9.]+)\s*([GM])B?\s*$", str(cfg.memory_limit), re.I)
    cap = float(m.group(1)) / (1024.0 if m.group(2).upper() == "M" else 1.0) if m else 6.0
    lim = max(0.5, min(cap, 6.0, mem_available_gb() - 1.0))
    return f"{lim:.1f}GB"


def prior_stream(cfg: Cfg, name: str, url: str, colnames: list, types: dict, keep: list, info: dict,
                 delimiter: str = "\t"):
    """Yield RecordBatches of a remote gzipped TSV (GzHttpStream: resumable, md5 checked at the
    end, nothing written to disk). ``info`` receives bytes, rows, md5 and timing."""
    head = http_head(url)
    size = int(head["content-length"])
    md5_expected = None
    try:
        md5_expected = http_text(url + ".md5").split()[0]
    except Exception as e:  # pragma: no cover
        log.warning("no md5 for %s: %s", url, e)
    log.info("priors %s: %s, %s bytes, last-modified %s, md5 %s", name, url, fmt_n(size),
             head.get("last-modified"), md5_expected)
    cfg.tmp.mkdir(parents=True, exist_ok=True)
    stream = GzHttpStream(url, size, cfg.tmp / f"prior_{name}.hdr", max_bytes=cfg.max_bytes)
    n_bad = [0]

    def on_invalid(row):
        n_bad[0] += 1
        return "skip"

    reader = pacsv.open_csv(
        pa.PythonFile(stream, mode="r"),
        read_options=pacsv.ReadOptions(column_names=colnames, block_size=64 << 20, use_threads=True),
        parse_options=pacsv.ParseOptions(delimiter=delimiter, quote_char=False, escape_char=False,
                                         newlines_in_values=False, invalid_row_handler=on_invalid),
        convert_options=pacsv.ConvertOptions(column_types=types, include_columns=keep,
                                             strings_can_be_null=False, quoted_strings_can_be_null=False),
    )
    t0 = last = time.time()
    n = 0
    try:
        for batch in reader:
            n += batch.num_rows
            yield batch
            if time.time() - last > 120:
                el = time.time() - t0
                log.info("  %s: %s rows, %.1f%% of %s bytes, %.1f MB/s, restarts %d, anon rss %.0f MB, "
                         "arrow pool %.0f MB, avail %.1f GB", name, fmt_n(n), 100.0 * stream.offset / size,
                         fmt_n(size), stream.offset / el / 1e6, stream.restarts, anon_rss_mb(),
                         pa.total_allocated_bytes() / 1e6, mem_available_gb())
                last = time.time()
    finally:
        stream.close()
    md5_got = stream.md5.hexdigest()
    if cfg.max_bytes is None:
        if not stream.complete:
            raise RuntimeError(f"stream for {url} ended early at {stream.offset} of {size}")
        if md5_expected and md5_got != md5_expected:
            raise RuntimeError(f"md5 mismatch for {url}: {md5_got} != {md5_expected}")
    el = time.time() - t0
    info.update({"url": url, "bytes": size, "last_modified": head.get("last-modified"),
                 "md5_expected": md5_expected,
                 "md5_ok": (md5_got == md5_expected) if cfg.max_bytes is None else None,
                 "rows": n, "invalid_rows_skipped": n_bad[0], "restarts": stream.restarts,
                 "decompressed_bytes": stream.out_bytes, "seconds": round(el, 1)})
    log.info("priors %s streamed: %s rows (%s invalid skipped), %.0f s, %.1f MB/s, restarts %d", name,
             fmt_n(n), fmt_n(n_bad[0]), el, stream.offset / max(el, 1e-9) / 1e6, stream.restarts)


class PairCounter:
    """Distinct items per group. Rows of single-CID groups are distinct by construction
    (one line per (CID, item) in the PubChem files) and are counted directly; items of
    multi-CID groups (or every item, for counters fed only through ``add_pairs``) are
    packed into uint64 keys, spilled to one file per 2**PART_BITS groups, and reduced
    with np.unique part by part."""

    def __init__(self, pw: Path, name: str, n_groups: int):
        self.dir = pw / f"pairs_{name}"
        shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir(parents=True)
        self.name = name
        self.n_groups = n_groups
        self.counts = np.zeros(n_groups, dtype=np.int32)
        self.fh: dict[int, object] = {}
        self.n_pairs_written = 0
        self.n_single_rows = 0

    def add_rows(self, g: np.ndarray) -> None:
        if len(g):
            np.add.at(self.counts, g, 1)
            self.n_single_rows += len(g)

    def add_pairs(self, g: np.ndarray, item: np.ndarray, dedupe: bool = False) -> None:
        if not len(g):
            return
        g = g.astype(np.uint64)
        key = ((g & np.uint64((1 << PART_BITS) - 1)) << np.uint64(ITEM_BITS)) | \
            (item.astype(np.uint64) & np.uint64((1 << ITEM_BITS) - 1))
        part = (g >> np.uint64(PART_BITS)).astype(np.int64)
        order = np.argsort(part, kind="stable")
        part, key = part[order], key[order]
        cut = np.flatnonzero(np.diff(part)) + 1
        for s, e in zip(np.r_[0, cut], np.r_[cut, len(part)]):
            seg = key[s:e]
            if dedupe:
                seg = np.unique(seg)
            p = int(part[s])
            fh = self.fh.get(p)
            if fh is None:
                fh = self.fh[p] = open(self.dir / f"{p:04d}.bin", "ab")
            fh.write(seg.tobytes())
            self.n_pairs_written += len(seg)

    CHUNK_KEYS = 20_000_000   # np.unique in one go up to this many keys (about 0.5 GB of temporaries)
    SLICE_KEYS = 8_000_000    # streamed slices when a file must be split first

    def _count_file(self, path: Path, c_out: np.ndarray, depth: int = 0) -> None:
        """Add distinct-key counts per local group of one key file to c_out. Files above
        CHUNK_KEYS are first split 8 ways on the next 3 bits of the local group id, in
        streamed slices (memory stays bounded however skewed the groups are)."""
        n = path.stat().st_size // 8
        if n == 0:
            return
        keys = np.memmap(path, dtype=np.uint64, mode="r")
        shift = ITEM_BITS + PART_BITS - 3 * (depth + 1)
        if n <= self.CHUNK_KEYS or shift < ITEM_BITS:
            u = np.unique(np.asarray(keys))
            c_out += np.bincount((u >> np.uint64(ITEM_BITS)).astype(np.int64), minlength=len(c_out))
            del keys, u
            return
        subs = [path.with_name(f"{path.stem}_{k}.bin") for k in range(8)]
        fhs = [open(q, "wb") for q in subs]
        try:
            for s0 in range(0, n, self.SLICE_KEYS):
                v = np.array(keys[s0:s0 + self.SLICE_KEYS])
                sub = ((v >> np.uint64(shift)) & np.uint64(7)).astype(np.int8)
                order = np.argsort(sub, kind="stable")
                v, sub = v[order], sub[order]
                cut = np.searchsorted(sub, np.arange(1, 8, dtype=np.int8))
                for k, part in enumerate(np.split(v, cut)):
                    fhs[k].write(part.tobytes())
                del v, sub, order
        finally:
            for fh in fhs:
                fh.close()
        del keys
        for q in subs:
            self._count_file(q, c_out, depth + 1)
            q.unlink()

    def reduce(self) -> np.ndarray:
        for fh in self.fh.values():
            fh.close()
        self.fh = {}
        nloc = 1 << PART_BITS
        for f in sorted(self.dir.glob("[0-9][0-9][0-9][0-9].bin")):
            p = int(f.stem)
            c = np.zeros(nloc, dtype=np.int64)
            self._count_file(f, c)
            base = p << PART_BITS
            hi = min(base + nloc, self.n_groups)
            self.counts[base:hi] += c[:hi - base].astype(np.int32)
            f.unlink()
        shutil.rmtree(self.dir, ignore_errors=True)
        return self.counts


def _letter_writers(root: Path, schema: pa.Schema) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    return {L: pq.ParquetWriter(root / f"{L}.parquet", schema, compression="zstd", compression_level=1)
            for L in LETTERS}


def _write_by_letter(tbl: pa.Table, writers: dict) -> int:
    """Split rows by the first letter of ik14 (InChIKey blocks are A-Z only)."""
    first = pc.utf8_slice_codeunits(tbl["ik14"], 0, 1)
    n = 0
    for L in LETTERS:
        part = tbl.filter(pc.equal(first, L))
        if len(part):
            writers[L].write_table(part)
            n += len(part)
    return n


def _close_writers(writers: dict) -> None:
    for w in writers.values():
        w.close()


def _np_tier_path() -> Path:
    return PROCESSED / "db" / "np" / "np.parquet"


def priors_map(cfg: Cfg, pw: Path) -> dict:
    """Group ids: g = global row of the ik14 in the tier (shards in mass order, rows in file
    order), then NP-tier ik14s absent from the tier (sorted by ik14). Writes cid2g.npy
    (int32, -1 = no group), n_all.npy (CIDs per group), np_groups.parquet (NP ik14 -> g)."""
    t0 = time.time()
    idx = json.loads((cfg.out / "_index.json").read_text())
    names = sorted(idx["shards"])
    by = pw / "by_letter"
    shutil.rmtree(by, ignore_errors=True)
    tier_w = _letter_writers(by / "tier", pa.schema([("ik14", pa.string()), ("g", pa.int32())]))
    offsets = {}
    off = 0
    max_cid = 0
    for name in names:
        t = pq.read_table(cfg.out / name, columns=["ik14", "cid"])
        n = len(t)
        if n:
            max_cid = max(max_cid, pc.max(t["cid"]).as_py())
        offsets[name] = [off, n]
        _write_by_letter(pa.table({"ik14": t["ik14"], "g": pa.array(np.arange(off, off + n, dtype=np.int32))}),
                         tier_w)
        off += n
    _close_writers(tier_w)
    n_tier = off
    np_w = _letter_writers(by / "np", pa.schema([("ik14", pa.string())]))
    n_np = 0
    if _np_tier_path().exists():
        t = pq.read_table(_np_tier_path(), columns=["ik14"])
        t = t.filter(pc.equal(pc.utf8_length(t["ik14"]), 14))
        n_np = _write_by_letter(t, np_w)
    else:
        log.warning("priors map: %s missing; NP-only ik14 groups skipped", _np_tier_path())
    _close_writers(np_w)
    log.info("priors map: tier %s rows in %d shards, NP tier %s ik14 split by first letter", fmt_n(n_tier),
             len(names), fmt_n(n_np))

    wait_for_memory(2.0, "priors map (CID-InChI-Key stream)")
    cid_w = _letter_writers(by / "cid", pa.schema([("cid", pa.int32()), ("ik14", pa.string())]))
    info: dict = {}
    n_badkey = 0
    max_cid_file = 0
    for b in prior_stream(cfg, "ik", f"{BASE_URL}/CID-InChI-Key.gz", ["cid", "inchi", "inchikey"],
                          {"cid": pa.int32(), "inchikey": pa.string()}, ["cid", "inchikey"], info):
        key = b.column("inchikey")
        ok = pc.and_(pc.equal(pc.utf8_length(key), 27), pc.is_valid(b.column("cid")))
        n_badkey += len(key) - pc.sum(ok).as_py() if len(key) else 0
        b = b.filter(ok)
        if not b.num_rows:
            continue
        max_cid_file = max(max_cid_file, pc.max(b.column("cid")).as_py())
        _write_by_letter(pa.table({"cid": b.column("cid"), "ik14": pc.utf8_slice_codeunits(b.column("inchikey"), 0, 14)}),
                         cid_w)
    _close_writers(cid_w)
    info["bad_inchikey_rows"] = n_badkey
    info["max_cid"] = max_cid_file
    max_cid = max(max_cid, max_cid_file)

    wait_for_memory(1.5, "priors map (per-letter joins)")
    con = duck(cfg, prior_memory_limit(cfg))
    cid2g = np.lib.format.open_memmap(pw / "cid2g.part.npy", mode="w+", dtype=np.int32, shape=(max_cid + 1,))
    cid2g[:] = -1
    n_extra = 0
    np_groups = []
    n_join = 0
    dup_cids = 0
    for L in LETTERS:
        tf, nf, cf = (by / s / f"{L}.parquet" for s in ("tier", "np", "cid"))
        ex = con.execute(f"SELECT DISTINCT n.ik14 FROM read_parquet('{nf}') n "
                         f"ANTI JOIN read_parquet('{tf}') t USING (ik14) ORDER BY n.ik14").to_arrow_table()
        ex = ex.read_all() if hasattr(ex, "read_all") else ex
        ex_t = pa.table({"ik14": ex.column("ik14"),
                         "g": pa.array(np.arange(n_tier + n_extra, n_tier + n_extra + len(ex), dtype=np.int32))})
        n_extra += len(ex)
        con.register("ex_l", ex_t)
        grp = f"(SELECT ik14, g FROM read_parquet('{tf}') UNION ALL SELECT ik14, g FROM ex_l)"
        nrow, ndist = con.execute(f"SELECT count(*), count(DISTINCT cid) FROM read_parquet('{cf}')").fetchone()
        dup_cids += nrow - ndist
        j = con.execute(f"SELECT k.cid, t.g FROM read_parquet('{cf}') k JOIN {grp} t USING (ik14)").fetchnumpy()
        cid2g[np.asarray(j["cid"], dtype=np.int64)] = np.asarray(j["g"], dtype=np.int32)
        n_join += len(j["cid"])
        npg = con.execute(f"SELECT n.ik14, t.g FROM read_parquet('{nf}') n JOIN {grp} t USING (ik14)").to_arrow_table()
        np_groups.append(npg.read_all() if hasattr(npg, "read_all") else npg)
        con.unregister("ex_l")
    con.close()
    n_groups = n_tier + n_extra
    # Every shard row keeps at least its own (lowest, filtered) CID: a CID missing from the
    # newer CID-InChI-Key is re-attached to its shard row; a CID whose key moved to another
    # ik14 follows the newer file (counted, not changed).
    n_missing = n_moved = 0
    for name, (o, n) in offsets.items():
        cid = pq.read_table(cfg.out / name, columns=["cid"])["cid"].to_numpy().astype(np.int64)
        g = np.arange(o, o + n, dtype=np.int32)
        cur = cid2g[cid]
        miss = cur < 0
        n_missing += int(miss.sum())
        n_moved += int(((cur >= 0) & (cur != g)).sum())
        cid2g[cid[miss]] = g[miss]
    cid2g.flush()
    n_all = np.zeros(n_groups, dtype=np.int32)
    step = 20_000_000
    for s in range(0, len(cid2g), step):
        v = np.asarray(cid2g[s:s + step])
        np.add.at(n_all, v[v >= 0], 1)
    del cid2g
    (pw / "cid2g.part.npy").replace(pw / "cid2g.npy")
    np.save(pw / "n_all.npy", n_all)
    npt = pa.concat_tables(np_groups) if np_groups else pa.table({"ik14": pa.array([], pa.string()),
                                                                   "g": pa.array([], pa.int32())})
    npt = npt.sort_by("ik14")
    pq.write_table(npt, pw / "np_groups.parquet", compression="zstd")
    tier_all = n_all[:n_tier]
    meta = {"n_tier": n_tier, "n_np_extra": n_extra, "n_groups": n_groups, "max_cid": max_cid,
            "offsets": offsets, "cid_rows": info.get("rows"), "cids_mapped": n_join, "dup_cids": dup_cids,
            "tier_rows_without_inchikey_cid": int((tier_all == 0).sum()),
            "shard_cids_missing_in_new_inchikey_file": n_missing, "shard_cids_moved_to_other_ik14": n_moved,
            "multi_cid_groups": int((n_all > 1).sum()), "cids_in_multi_groups": int(n_all[n_all > 1].sum()),
            "max_cids_per_group": int(n_all.max()) if n_groups else 0,
            "np_ik14_rows": len(npt), "np_ik14_in_pubchem": int((n_all[npt["g"].to_numpy()] > 0).sum()) if len(npt) else 0,
            "source": info, "seconds": round(time.time() - t0, 1)}
    (pw / "groups.json").write_text(json.dumps(meta, indent=1))
    shutil.rmtree(by, ignore_errors=True)
    log.info("priors map: %s groups (%s tier + %s NP-only), %s CIDs mapped, %s multi-CID groups holding %s CIDs, "
             "max %s CIDs per group, shard CIDs re-attached %s, moved %s, %.0f s", fmt_n(n_groups), fmt_n(n_tier),
             fmt_n(n_extra), fmt_n(n_join), fmt_n(meta["multi_cid_groups"]), fmt_n(meta["cids_in_multi_groups"]),
             fmt_n(meta["max_cids_per_group"]), fmt_n(n_missing), fmt_n(n_moved), time.time() - t0)
    return meta


class GroupMap:
    """cid -> group id and multi-CID flag, memory-mapped (file-backed pages, reclaimable)."""

    def __init__(self, pw: Path):
        self.meta = json.loads((pw / "groups.json").read_text())
        self.cid2g = np.load(pw / "cid2g.npy", mmap_mode="r")
        self.multi = np.load(pw / "n_all.npy", mmap_mode="r") > 1
        self.n_groups = int(self.meta["n_groups"])

    def lookup(self, cid: np.ndarray) -> np.ndarray:
        cid = np.asarray(cid, dtype=np.int64)
        g = np.full(len(cid), -1, dtype=np.int32)
        ok = (cid >= 0) & (cid < len(self.cid2g))
        if ok.all():
            g[:] = self.cid2g[cid]
        else:
            g[ok] = self.cid2g[cid[ok]]
        return g


def _hash_strings(con, arr: pa.Array, lower: bool) -> np.ndarray:
    con.register("hs_t", pa.table({"s": arr}))
    expr = "hash(lower(s))" if lower else "hash(s)"
    h = con.execute(f"SELECT {expr} AS h FROM hs_t").fetchnumpy()["h"]
    con.unregister("hs_t")
    return np.asarray(h, dtype=np.uint64) >> np.uint64(PART_BITS)


def priors_count(cfg: Cfg, pw: Path, step: str, gm: GroupMap) -> dict:
    """Stream one source and write <column>.npy (int32 per group). See PRIOR_FIELDS."""
    t0 = time.time()
    info: dict = {}
    con = duck(cfg, "1GB")
    con.execute("SET threads=1")
    if step in ("sid", "synonym", "patent"):
        fname, cols, types, keep, col = {
            "sid": ("CID-SID.gz", ["cid", "sid", "type"], {"cid": pa.int32(), "sid": pa.int64(), "type": pa.int8()},
                    ["cid", "sid", "type"], "sid_count"),
            "synonym": ("CID-Synonym-filtered.gz", ["cid", "name"], {"cid": pa.int32(), "name": pa.string()},
                        ["cid", "name"], "synonym_count"),
            "patent": ("CID-Patent.gz", ["cid", "patent"], {"cid": pa.int32(), "patent": pa.string()},
                       ["cid", "patent"], "pc_patent_count"),
        }[step]
        ctr = PairCounter(pw, step, gm.n_groups)
        n_in = 0
        by_type = {}
        for b in prior_stream(cfg, step, f"{BASE_URL}/{fname}", cols, types, keep, info):
            g = gm.lookup(b.column("cid").to_numpy(zero_copy_only=False))
            ok = g >= 0
            n_in += int(ok.sum())
            if step == "sid":
                tv, tc = np.unique(b.column("type").to_numpy(zero_copy_only=False), return_counts=True)
                for v, c in zip(tv.tolist(), tc.tolist()):
                    by_type[str(v)] = by_type.get(str(v), 0) + c
            mul = np.zeros(len(g), dtype=bool)
            mul[ok] = gm.multi[g[ok]]
            ctr.add_rows(g[ok & ~mul])
            if mul.any():
                if step == "sid":
                    item = b.column("sid").to_numpy(zero_copy_only=False)[mul].astype(np.uint64)
                else:
                    item = _hash_strings(con, b.column(keep[1]).filter(pa.array(mul)), lower=(step == "synonym"))
                ctr.add_pairs(g[mul], item)
        info["rows_in_tier_or_np_groups"] = n_in
        info["pairs_spilled"] = ctr.n_pairs_written
        pa.default_memory_pool().release_unused()
        log.info("priors %s: %s pairs spilled for multi-CID groups; reducing (anon rss %.0f MB, avail %.1f GB)",
                 step, fmt_n(ctr.n_pairs_written), anon_rss_mb(), mem_available_gb())
        wait_for_memory(1.5, f"priors {step} reduce")
        info["single_group_rows"] = ctr.n_single_rows
        if by_type:
            info["rows_by_type"] = by_type
        counts = {col: ctr.reduce()}
    elif step == "sidmap":
        vend, nist, rdf_info = source_categories(cfg)
        info["source_categories"] = rdf_info
        ctr_v = PairCounter(pw, "vendor", gm.n_groups)
        ctr_d = PairCounter(pw, "depositor", gm.n_groups)
        src_id: dict[str, int] = {}
        src_sids: list[int] = []
        n_with_cid = n_in = 0
        for b in prior_stream(cfg, "sidmap", f"{SUBSTANCE_URL}/SID-Map.gz", ["line"], {"line": pa.string()},
                              ["line"], info, delimiter="\x1f"):
            parts = pc.split_pattern(b.column("line"), "\t")
            parts = parts.filter(pc.greater_equal(pc.list_value_length(parts), 4))
            if not len(parts):
                continue
            cid_s = pc.list_element(parts, 3)
            has = pc.match_substring_regex(cid_s, "^[0-9]{1,10}$")
            parts, cid_s = parts.filter(has), cid_s.filter(has)
            n_with_cid += len(cid_s)
            src = pc.list_element(parts, 1).dictionary_encode()
            lut = np.empty(len(src.dictionary), dtype=np.int64)
            for i, nm in enumerate(src.dictionary.to_pylist()):
                j = src_id.get(nm)
                if j is None:
                    j = src_id[nm] = len(src_id)
                    src_sids.append(0)
                lut[i] = j
            sid_src = lut[src.indices.to_numpy(zero_copy_only=False)]
            for j, c in zip(*np.unique(sid_src, return_counts=True)):
                src_sids[int(j)] += int(c)
            g = gm.lookup(pc.cast(cid_s, pa.int64()).to_numpy(zero_copy_only=False))
            ok = g >= 0
            n_in += int(ok.sum())
            is_v = np.array([nm in vend for nm in src_id], dtype=bool)
            is_n = np.array([nm in nist or bool(NIST_RE.search(nm)) for nm in src_id], dtype=bool)
            dep = ok & ~is_n[sid_src]
            ctr_d.add_pairs(g[dep], sid_src[dep].astype(np.uint64), dedupe=True)
            ven = ok & is_v[sid_src] & ~is_n[sid_src]
            ctr_v.add_pairs(g[ven], sid_src[ven].astype(np.uint64), dedupe=True)
        names = list(src_id)
        top = sorted(range(len(names)), key=lambda i: -src_sids[i])
        info.update({
            "sids_with_standardized_cid": n_with_cid, "sids_in_groups": n_in, "n_sources_seen": len(names),
            "n_vendor_sources_seen": int(sum(nm in vend for nm in names)),
            "sids_of_vendor_sources": int(sum(src_sids[i] for i in range(len(names)) if names[i] in vend)),
            "nist_sources_excluded": sorted(nm for nm in names if nm in nist or NIST_RE.search(nm)),
            "top_sources": [[names[i], src_sids[i], names[i] in vend] for i in top[:40]],
            "pairs_spilled": {"vendor": ctr_v.n_pairs_written, "depositor": ctr_d.n_pairs_written},
        })
        pa.default_memory_pool().release_unused()
        wait_for_memory(1.5, "priors sidmap reduce")
        log.info("priors sidmap: %s sources seen (%s vendors), %s SIDs with a CID, %s in groups; NIST excluded: %s",
                 fmt_n(len(names)), info["n_vendor_sources_seen"], fmt_n(n_with_cid), fmt_n(n_in),
                 info["nist_sources_excluded"])
        for nm, c, v in info["top_sources"][:15]:
            log.info("  source %-60s %12s SIDs vendor=%s", nm[:60], fmt_n(c), v)
        counts = {"n_vendors": ctr_v.reduce(), "n_depositors": ctr_d.reduce()}
    elif step == "mesh":
        url = f"{BASE_URL}/CID-MeSH"
        head = http_head(url)
        raw = subprocess.run(["curl", "-sS", "--fail", "-L", "--max-time", "600", url], capture_output=True,
                             check=True).stdout
        md5_expected = http_text(url + ".md5").split()[0]
        md5_got = hashlib.md5(raw).hexdigest()
        if md5_got != md5_expected:
            raise RuntimeError(f"md5 mismatch for CID-MeSH: {md5_got} != {md5_expected}")
        cnt = np.zeros(gm.n_groups, dtype=np.int32)
        multi_sets: dict[int, set] = {}
        n_lines = n_in = 0
        for ln in raw.decode("utf-8", errors="replace").splitlines():
            parts = ln.split("\t")
            if len(parts) < 2 or not parts[0].isdigit():
                continue
            n_lines += 1
            g = int(gm.lookup(np.array([int(parts[0])]))[0])
            if g < 0:
                continue
            n_in += 1
            terms = {t.strip().lower() for t in parts[1:] if t.strip()}
            if gm.multi[g]:
                multi_sets.setdefault(g, set()).update(terms)
            else:
                cnt[g] += len(terms)
        for g, terms in multi_sets.items():
            cnt[g] += len(terms)
        info.update({"url": url, "bytes": len(raw), "last_modified": head.get("last-modified"),
                     "md5_expected": md5_expected, "md5_ok": True, "rows": n_lines, "rows_in_groups": n_in,
                     "seconds": round(time.time() - t0, 1)})
        counts = {"mesh_count": cnt}
    else:  # pragma: no cover
        raise ValueError(step)
    con.close()
    for col, arr in counts.items():
        mx = np.iinfo(np.int32).max
        np.save(pw / f"{col}.npy", np.minimum(arr, mx).astype(np.int32))
        t = arr[:gm.meta["n_tier"]]
        log.info("priors %s: %s on tier rows: nonzero %.4f, mean %.2f, max %s", step, col,
                 float((t > 0).mean()) if len(t) else 0.0, float(t.mean()) if len(t) else 0.0,
                 fmt_n(int(t.max())) if len(t) else 0)
    info["seconds_total"] = round(time.time() - t0, 1)
    return info


def _ttl_records(text: str) -> dict:
    """Minimal reader for PubChem's pc_source.ttl: {subject: {predicate: [objects]}}."""
    recs: dict = {}
    subj = pred = None
    for line in text.splitlines():
        if not line.strip() or line.startswith("@prefix"):
            continue
        if not line.startswith(("\t", " ")):
            parts = line.split("\t", 2)
            if len(parts) < 3:
                continue
            subj, pred, rest = parts
        else:
            rest = line
        rest = rest.strip()
        if rest.endswith((",", ".", ";")):
            rest = rest[:-1].rstrip()
        if subj is None:
            continue
        recs.setdefault(subj, {}).setdefault(pred, []).append(rest)
    return recs


def _ttl_literal(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == '"':
        s = s[1:s.rindex('"')]
        s = s.replace('\\"', '"').replace("\\\\", "\\")
    return s


def source_categories(cfg: Cfg) -> tuple[set, set, dict]:
    """Names (as they appear in SID-Map) of vendor sources and of NIST sources.

    Categories come from the PubChem RDF source file (dcterms:subject); a source is
    known by its id (dcterms:title), its display name (dcterms:alternative) and every
    display name it has had (Substance/Extras/Source-Names)."""
    ttl = zlib.decompress(subprocess.run(["curl", "-sS", "--fail", "-L", "--max-time", "120", SOURCE_RDF_URL],
                                         capture_output=True, check=True).stdout, 31).decode("utf-8")
    names_txt = http_text(f"{SUBSTANCE_URL}/Source-Names")
    alias: dict[str, set] = {}
    for ln in names_txt.splitlines():
        p = ln.split("\t")
        if len(p) >= 2:
            alias.setdefault(p[0], set()).add(p[1])
            alias.setdefault(p[1], set()).add(p[0])
    vend, nist = set(), set()
    n_vendor_records = 0
    for subj, r in _ttl_records(ttl).items():
        cats = r.get("dcterms:subject", [])
        nm = {_ttl_literal(x) for x in r.get("dcterms:title", []) + r.get("dcterms:alternative", [])}
        for x in list(nm):
            nm |= alias.get(x, set())
        if any(NIST_RE.search(x) for x in nm):
            nist |= nm
        if VENDOR_CATEGORY in cats:
            n_vendor_records += 1
            vend |= nm
    vend -= nist
    info = {"rdf_url": SOURCE_RDF_URL, "rdf_bytes": len(ttl), "vendor_records": n_vendor_records,
            "vendor_names": len(vend), "nist_names": sorted(nist)}
    log.info("source categories: %d vendor records (%d names incl. aliases); NIST names %s", n_vendor_records,
             len(vend), sorted(nist))
    return vend, nist, info


def _quantiles(x: np.ndarray, qs=(0.5, 0.9, 0.99, 0.999), chunk: int = 10_000_000) -> dict:
    """Quantiles of a non-negative integer array (or memmap) via its histogram, in chunks."""
    n = len(x)
    if not n:
        return {}
    mx = max(int(np.max(x[s:s + chunk])) for s in range(0, n, chunk))
    h = np.zeros(mx + 1, dtype=np.int64)
    tot = 0
    for s in range(0, n, chunk):
        v = np.asarray(x[s:s + chunk])
        h += np.bincount(v, minlength=mx + 1)
        tot += int(v.sum(dtype=np.int64))
    cdf = np.cumsum(h) / n
    out = {f"p{q * 100:g}": int(np.searchsorted(cdf, q)) for q in qs}
    out.update({"mean": round(tot / n, 3), "max": mx, "frac_zero": round(float(h[0] / n), 4)})
    return out


def priors_write(cfg: Cfg, pw: Path) -> dict:
    """Rewrite every shard with the prior columns appended (row groups copied one by one,
    so rows, order and row-group boundaries are unchanged), refresh _index.json and write
    np_priors.parquet (NP-tier ik14 -> prior columns) for build_npdb.py --stage priors."""
    t0 = time.time()
    meta = json.loads((pw / "groups.json").read_text())
    arrs = {c: np.load(pw / f"{c}.npy", mmap_mode="r") for c in PRIOR_NAMES}
    idx = json.loads((cfg.out / "_index.json").read_text())
    wdir = pw / "write"
    wdir.mkdir(exist_ok=True)
    for name, (off, n) in sorted(meta["offsets"].items()):
        mk = wdir / f"{name}.json"
        path = cfg.out / name
        if mk.exists() and pq.read_schema(path).names == PRIOR_SCHEMA.names:
            sh = json.loads(mk.read_text())
            idx["shards"].setdefault(name, {}).update({k: sh[k] for k in ("rows", "bytes", "sha256")})
            continue
        pf = pq.ParquetFile(path)
        tmp = path.with_suffix(".parquet.part")
        w = pq.ParquetWriter(tmp, PRIOR_SCHEMA, compression="zstd", compression_level=9,
                             use_dictionary=["formula", "pmid_count", "n_cids", *PRIOR_NAMES],
                             write_statistics=True)
        pos = off
        rg_rows = []
        for i in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(i)
            t = t.select(OUT_SCHEMA.names)
            k = len(t)
            cols = {}
            for c, typ in PRIOR_FIELDS:
                v = np.asarray(arrs[c][pos:pos + k])
                if typ == pa.int16():
                    v = np.minimum(v, np.iinfo(np.int16).max).astype(np.int16)
                cols[c] = pa.array(v, type=typ)
            for c, a in cols.items():
                t = t.append_column(c, a)
            w.write_table(t.cast(PRIOR_SCHEMA), row_group_size=max(k, 1))
            rg_rows.append(k)
            pos += k
        w.close()
        if pos - off != n or sum(rg_rows) != pf.metadata.num_rows:
            raise RuntimeError(f"{name}: wrote {pos - off} rows, expected {n}")
        md = pq.read_metadata(tmp)
        if md.num_rows != n or [md.row_group(i).num_rows for i in range(md.num_row_groups)] != rg_rows:
            raise RuntimeError(f"{name}: row groups changed")
        tmp.replace(path)
        sh = {"rows": n, "bytes": path.stat().st_size, "sha256": sha256_file(path), "row_groups": len(rg_rows)}
        mk.write_text(json.dumps(sh))
        e = idx["shards"].setdefault(name, {})
        if e.get("rows") not in (None, n):
            raise RuntimeError(f"{name}: row count {n} differs from _index.json {e.get('rows')}")
        e.update({k: sh[k] for k in ("rows", "bytes", "sha256")})
    # stats over the tier rows and over NP ik14
    n_tier = int(meta["n_tier"])
    stats = {c: _quantiles(np.asarray(arrs[c][:n_tier])) for c in PRIOR_NAMES}
    n_all = np.load(pw / "n_all.npy")
    npg = pq.read_table(pw / "np_groups.parquet")
    g = npg["g"].to_numpy()
    np_tbl = pa.table({"ik14": npg["ik14"], **{
        c: pa.array(np.minimum(np.asarray(arrs[c])[g], np.iinfo(np.int16 if t == pa.int16() else np.int32).max)
                    .astype(np.int16 if t == pa.int16() else np.int32), type=t) for c, t in PRIOR_FIELDS},
        "n_pubchem_cids": pa.array(n_all[g].astype(np.int32))})
    np_out = cfg.work / NP_PRIORS_NAME
    pq.write_table(np_tbl, np_out.with_suffix(".parquet.part"), compression="zstd")
    np_out.with_suffix(".parquet.part").replace(np_out)
    checks = {}
    for nm, ik, mass in PRIOR_CHECK:
        t = query_mass(cfg.out, mass, 10.0)
        hit = t.filter(pc.equal(t["ik14"], ik)).to_pylist()
        checks[nm] = {k: v for k, v in hit[0].items() if k not in ("smiles",)} if hit else None
        log.info("priors check %s: %s", nm, checks[nm])
    total_rows = sum(s["rows"] for s in idx["shards"].values())
    srcs = {k: v for k, v in load_stats(cfg).get("priors", {}).get("sources", {}).items()}
    idx.update({
        "schema": {f.name: str(f.type) for f in PRIOR_SCHEMA},
        "total_bytes": sum(s["bytes"] for s in idx["shards"].values()),
        "priors": {
            "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": git_commit(),
            "group": "all PubChem CIDs whose InChIKey (CID-InChI-Key) starts with the ik14; distinct items per group",
            "columns": {"sid_count": "distinct SIDs linked by CID-SID (type 1 standardized form, type 2 component)",
                        "synonym_count": "distinct case-folded names in CID-Synonym-filtered",
                        "pc_patent_count": "distinct publication numbers in CID-Patent",
                        "mesh_count": "distinct MeSH terms in CID-MeSH",
                        "n_vendors": "distinct depositors (SID-Map, standardized CID) in RDF category Chemical_Vendors",
                        "n_depositors": "distinct depositors (SID-Map, standardized CID), NIST sources excluded"},
            "stats_tier_rows": stats, "groups": {k: v for k, v in meta.items() if k != "offsets"},
            "sources": srcs, "checks": checks,
        },
    })
    if total_rows != idx.get("total_rows"):
        raise RuntimeError(f"total rows {total_rows} != _index.json {idx.get('total_rows')}")
    (cfg.out / "_index.json").write_text(json.dumps(idx, indent=1))
    log.info("priors write: %d shards, %s rows, %.2f GB, np_priors %s rows, %.0f s", len(idx["shards"]),
             fmt_n(total_rows), idx["total_bytes"] / 1e9, fmt_n(len(np_tbl)), time.time() - t0)
    for c in PRIOR_NAMES:
        log.info("  %-16s %s", c, stats[c])
    return {"stats_tier_rows": stats, "checks": checks, "np_priors_rows": len(np_tbl),
            "np_priors_sha256": sha256_file(np_out), "total_bytes": idx["total_bytes"],
            "index_sha256": sha256_file(cfg.out / "_index.json"), "seconds": round(time.time() - t0, 1)}


def stage_priors(cfg: Cfg):
    if cfg.marker("priors").exists():
        log.info("priors: marker present, skipping")
        return
    if not (cfg.out / "_index.json").exists():
        raise RuntimeError("priors: no _index.json; run stages 1-3 first")
    pw = cfg.work / "priors"
    pw.mkdir(parents=True, exist_ok=True)
    gj = pw / "groups.json"
    if gj.exists():  # a group map of older shards is useless: start over
        rows_now = {k: v["rows"] for k, v in json.loads((cfg.out / "_index.json").read_text())["shards"].items()}
        rows_map = {k: v[1] for k, v in json.loads(gj.read_text())["offsets"].items()}
        if rows_now != rows_map:
            log.warning("priors: group map does not match the current shards; rebuilding from the map step")
            shutil.rmtree(pw)
            pw.mkdir(parents=True)
    stats = load_stats(cfg)
    ps = stats.setdefault("priors", {})
    gm = None
    for step in PRIOR_STEPS:
        mk = pw / f".done_{step}"
        if mk.exists():
            log.info("priors %s: done, skipping", step)
            continue
        ts = time.time()
        if step == "map":
            res = priors_map(cfg, pw)
            ps.setdefault("sources", {})["CID-InChI-Key.gz"] = res.pop("source")
            ps["map"] = {k: v for k, v in res.items() if k != "offsets"}
        elif step == "write":
            ps["write"] = priors_write(cfg, pw)
        else:
            wait_for_memory(2.0, f"priors {step}")
            if gm is None:
                gm = GroupMap(pw)
            info = priors_count(cfg, pw, step, gm)
            ps.setdefault("sources", {})[step] = info
        ps.setdefault("timing_seconds", {})[step] = round(time.time() - ts, 1)
        ps.setdefault("peak_rss_mb", {})[step] = round(peak_rss_mb())
        stats["priors"] = ps
        save_stats(cfg, stats)
        if step == "write":  # _index.json now carries the sources too
            idx = json.loads((cfg.out / "_index.json").read_text())
            idx["priors"]["sources"] = ps.get("sources", {})
            (cfg.out / "_index.json").write_text(json.dumps(idx, indent=1))
            ps["write"]["index_sha256"] = sha256_file(cfg.out / "_index.json")
            save_stats(cfg, stats)
        mk.write_text(datetime.now(timezone.utc).isoformat())
        pa.default_memory_pool().release_unused()
        log.info("priors %s finished in %.0f s (peak rss %.0f MB incl. mapped files, anon rss now %.0f MB, "
                 "available %.1f GB)", step, time.time() - ts, peak_rss_mb(), anon_rss_mb(), mem_available_gb())
    if not cfg.keep_intermediates:
        freed = 0
        for p in list(pw.glob("*.npy")) + list(pw.glob("pairs_*")):
            freed += dir_size(p)
            shutil.rmtree(p) if p.is_dir() else p.unlink()
        log.info("priors: deleted %.2f GB of intermediates (kept groups.json, np_groups.parquet, markers)",
                 freed / 1e9)
    cfg.marker("priors").write_text(datetime.now(timezone.utc).isoformat())
    if not cfg.test:
        stage_docs(cfg)  # WP3 row and section: new _index.json sha256 and size
        priors_docs(cfg)


PRIORS_START = "<!-- R3:pubchem-priors:start -->"
PRIORS_END = "<!-- R3:pubchem-priors:end -->"


def priors_docs(cfg: Cfg):
    """Write the R3 PubChem-priors section of docs/ARTEFACTS.md (after the WP3 section)."""
    ps = load_stats(cfg).get("priors", {})
    idx = json.loads((cfg.out / "_index.json").read_text())
    pr = idx.get("priors", {})
    tim = ps.get("timing_seconds", {})
    rss = max(ps.get("peak_rss_mb", {}).values() or [0])
    gm = pr.get("groups", {})
    lines = [PRIORS_START, "## PubChem documentation priors (R3)", "",
             f"- Built {pr.get('built_utc')} by `nice -n 10 python3 scripts/build_pubchem.py --stage priors` "
             f"(git {pr.get('git_commit')}), {sum(tim.values()) / 60:.0f} min wall "
             f"({', '.join(f'{k} {tim[k] / 60:.1f}' for k in PRIOR_STEPS if k in tim)} min), 2 cores, peak RSS "
             f"{rss / 1024:.1f} GB including memory-mapped files (anonymous RSS at most about 3 GB), DuckDB "
             "memory_limit at most min(6 GB, available - 1 GB). Resumable through "
             "`data/external/pubchem/priors/.done_*`; intermediates (cid2g.npy, pair files) are deleted at the end.",
             "- Adds to every shard, after the WP3 columns: " + ", ".join(
                 f"`{n}` {t}" for n, t in PRIOR_FIELDS) + ". Rows, row order and row groups of every shard are "
             "unchanged (row groups copied one by one; checked against `_index.json`).",
             f"- Group of an ik14: {pr.get('group')}. Groups: {fmt_n(gm.get('n_tier', 0))} tier rows + "
             f"{fmt_n(gm.get('n_np_extra', 0))} NP-tier ik14 not in the tier; {fmt_n(gm.get('cids_mapped', 0))} CIDs "
             f"mapped; {fmt_n(gm.get('multi_cid_groups', 0))} groups with more than one CID (max "
             f"{fmt_n(gm.get('max_cids_per_group', 0))}). Tier rows with no CID in the newer CID-InChI-Key: "
             f"{fmt_n(gm.get('tier_rows_without_inchikey_cid', 0))} (join coverage "
             f"{100.0 * (1 - gm.get('tier_rows_without_inchikey_cid', 0) / max(gm.get('n_tier', 1), 1)):.3f} percent).",
             "- Definitions: " + "; ".join(f"`{k}`: {v}" for k, v in pr.get("columns", {}).items()) + ". "
             "No depositor-category flag is stored; NIST sources are excluded from `n_depositors` and are not "
             "vendors. `pmid_count` is the WP3 column, unchanged.",
             f"- `np_priors.parquet` (`data/external/pubchem/`, NP ik14 -> the same columns + `n_pubchem_cids`), "
             f"{fmt_n(ps.get('write', {}).get('np_priors_rows', 0))} rows, sha256 "
             f"`{ps.get('write', {}).get('np_priors_sha256')}`; joined into np.parquet by "
             "`build_npdb.py --stage priors`.", "",
             "Sources (streamed, never stored; md5 checked):", "",
             "| step | file | Last-Modified | bytes | lines | seconds streamed |", "|---|---|---|---|---|---|"]
    srcs = ps.get("sources", {})
    for k in ["CID-InChI-Key.gz", *PRIOR_STEPS]:
        if k in srcs:
            s = srcs[k]
            lines.append(f"| {'map' if k.endswith('.gz') else k} | {s.get('url', '').rsplit('/', 1)[-1]} | "
                         f"{s.get('last_modified')} | {fmt_n(s.get('bytes') or 0)} | {fmt_n(s.get('rows') or 0)} | "
                         f"{s.get('seconds')} |")
    sc = ps.get("sources", {}).get("sidmap", {}).get("source_categories", {})
    if sc:
        lines += ["", f"Vendor categories: `{SOURCE_RDF_URL}` ({sc.get('vendor_records')} Chemical_Vendors "
                      f"records, {sc.get('vendor_names')} names with aliases from `Substance/Extras/Source-Names`)."]
    lines += ["", "Distribution over the tier rows (90.7M):", "",
              "| column | p50 | p90 | p99 | p99.9 | max | mean | fraction zero |", "|---|---|---|---|---|---|---|---|"]
    for c, s in pr.get("stats_tier_rows", {}).items():
        lines.append(f"| {c} | {s.get('p50')} | {s.get('p90')} | {s.get('p99')} | {s.get('p99.9')} | "
                     f"{fmt_n(s.get('max', 0))} | {s.get('mean')} | {s.get('frac_zero')} |")
    lines += ["", "Spot checks (tier rows):", "", "| compound | cid | pmid_count | n_cids | " +
              " | ".join(PRIOR_NAMES) + " |", "|---" * (3 + len(PRIOR_NAMES)) + "|---|"]
    for nm, r in pr.get("checks", {}).items():
        if r:
            lines.append(f"| {nm} | {r['cid']} | {fmt_n(r['pmid_count'])} | {r['n_cids']} | " +
                         " | ".join(fmt_n(r[c]) for c in PRIOR_NAMES) + " |")
    lines += ["", f"sha256 of `_index.json` (lists the per-shard sha256): `{sha256_file(cfg.out / '_index.json')}`; "
                  f"tier size {idx.get('total_bytes', 0) / 1e9:.2f} GB.", PRIORS_END, ""]
    section = "\n".join(lines)
    path = DOCS / "ARTEFACTS.md"
    text = path.read_text()
    if PRIORS_START in text and PRIORS_END in text:
        a, b = text.index(PRIORS_START), text.index(PRIORS_END) + len(PRIORS_END)
        text = text[:a] + section.rstrip("\n") + text[b:]
    else:
        anchor = SECTION_END
        i = text.index(anchor) + len(anchor) if anchor in text else len(text)
        text = text[:i] + "\n\n" + section.rstrip("\n") + text[i:]
    path.write_text(text)
    log.info("docs: wrote the R3 priors section to %s", path)


# ---------------------------------------------------------------- self-check

def query_mass(db_dir: Path, mass: float, tol_ppm: float = 10.0) -> pa.Table:
    """Rows of the PubChem tier with |mono_mass - mass| <= tol_ppm * mass * 1e-6.

    Small stand-in for casmi.db.lookup (WP2), used only for the WP3 self-check.
    """
    tol = mass * tol_ppm * 1e-6
    lo, hi = mass - tol, mass + tol
    parts = []
    for b in range(int(lo // BIN_DA), int(hi // BIN_DA) + 1):
        p = Path(db_dir) / shard_name(b)
        if p.exists():
            parts.append(pq.read_table(p, filters=[("mono_mass", ">=", lo), ("mono_mass", "<=", hi)]))
    if not parts:
        return OUT_SCHEMA.empty_table()
    return pa.concat_tables(parts)


def test_masses(n: int = 20, seed: int = SEED) -> list[tuple[str, float]]:
    t = pq.read_table(TEST_PARQUET, columns=["molecule_id", "adduct", "precursor_mz"]).to_pylist()
    per = {}
    for r in t:
        shift = ADDUCT_SHIFT.get(r["adduct"])
        if shift is None:
            continue
        per.setdefault(r["molecule_id"], []).append(r["precursor_mz"] - shift)
    mols = sorted(per)
    rng = random.Random(seed)
    pick = rng.sample(mols, min(n, len(mols)))
    out = []
    for m in pick:
        v = sorted(per[m])
        k = len(v)
        med = v[k // 2] if k % 2 else 0.5 * (v[k // 2 - 1] + v[k // 2])
        out.append((m, med))
    return out


def stage_check(cfg: Cfg) -> dict:
    t0 = time.time()
    db = cfg.out
    idx = json.loads((db / "_index.json").read_text())
    shards = sorted(db.glob("mass_*.parquet"))
    res = {"ok": True, "problems": []}

    def fail(msg):
        res["ok"] = False
        res["problems"].append(msg)
        log.error("CHECK FAILED: %s", msg)

    # schema, uniqueness, sortedness
    con = duck(cfg)
    total = 0
    for p in shards:
        sch = pq.read_schema(p)
        if not schema_ok(sch):
            fail(f"{p.name} schema {sch}")
        n, n_ik = con.execute(f"SELECT count(*), count(DISTINCT ik14) FROM read_parquet('{p}')").fetchone()
        mm = pq.read_table(p, columns=["mono_mass"])["mono_mass"].combine_chunks()
        n_unsorted = pc.sum(pc.less(mm[1:], mm[:-1])).as_py() if len(mm) > 1 else 0
        total += n
        if n != n_ik:
            fail(f"{p.name}: ik14 not unique ({n} vs {n_ik})")
        if n_unsorted:
            fail(f"{p.name}: {n_unsorted} rows out of mass order")
        if idx["shards"].get(p.name, {}).get("rows") != n:
            fail(f"{p.name}: row count differs from _index.json")
    con.close()
    size = sum(p.stat().st_size for p in db.iterdir() if p.is_file())
    res.update({"total_rows": total, "n_shards": len(shards), "bytes": size})
    log.info("check: %d shards, %s rows, %.2f GB on disk", len(shards), fmt_n(total), size / 1e9)
    if not cfg.test:
        if not (60e6 <= total <= 110e6):
            fail(f"total rows {total} outside [60M, 110M]")
        if size >= 6e9:
            fail(f"size {size} >= 6 GB")
    # known compounds
    for name, ik14, mass in (("caffeine", "RYYVLZVUVIJVGH", 194.0804), ("paclitaxel", "RCINICONZNJXQF", 853.3310)):
        ts = time.time()
        t = query_mass(db, mass, 10.0)
        dt = time.time() - ts
        hit = [r for r in t.filter(pc.equal(t["ik14"], ik14)).to_pylist()]
        res[name] = {"n_candidates": len(t), "found": bool(hit), "row": hit[0] if hit else None, "seconds": round(dt, 3)}
        log.info("check %s (M %.4f, 10 ppm): %s candidates in %.3f s, found=%s %s", name, mass, fmt_n(len(t)),
                 dt, bool(hit), hit[0] if hit else "")
        if not hit and not cfg.test:
            fail(f"{name} not found")
    # 20 random public-test masses
    timings = []
    for mol, m in test_masses(20):
        ts = time.time()
        t = query_mass(db, m, 10.0)
        dt = time.time() - ts
        timings.append({"molecule_id": mol, "M": round(m, 5), "n_candidates": len(t), "seconds": round(dt, 3)})
        log.info("check test %s M=%.5f: %s candidates in %.3f s", mol, m, fmt_n(len(t)), dt)
        if dt >= 2.0:
            fail(f"query for {mol} took {dt:.2f} s")
    res["test_queries"] = timings
    cnts = sorted(x["n_candidates"] for x in timings)
    secs = [x["seconds"] for x in timings]
    res["test_summary"] = {"min": cnts[0], "median": cnts[len(cnts) // 2], "max": cnts[-1],
                           "max_seconds": max(secs), "mean_seconds": round(sum(secs) / len(secs), 3)}
    log.info("check test masses: candidates min %s median %s max %s; query time max %.3f s mean %.3f s",
             *[fmt_n(res["test_summary"][k]) for k in ("min", "median", "max")],
             res["test_summary"]["max_seconds"], res["test_summary"]["mean_seconds"])
    res["seconds"] = round(time.time() - t0, 1)
    update_stats(cfg, "check", res)
    idx["self_check"] = {k: res[k] for k in ("ok", "caffeine", "paclitaxel", "test_summary")}
    idx["filter_cascade"] = ordered_cascade(idx.get("filter_cascade"))
    (db / "_index.json").write_text(json.dumps(idx, indent=1))
    if res["ok"]:
        cfg.marker("check").write_text(json.dumps({"rows": total}))
        log.info("check: all acceptance checks passed")
    return res


# ---------------------------------------------------------------- cleanup and docs

def stage_cleanup(cfg: Cfg):
    freed = 0
    for d in (cfg.s1, cfg.s2, cfg.tmp):
        if d.exists():
            freed += dir_size(d)
            shutil.rmtree(d)
    for p in cfg.work.glob("*.part"):
        freed += p.stat().st_size
        p.unlink()
    log.info("cleanup: freed %.2f GB of intermediates; kept %s and stage markers/stats in %s",
             freed / 1e9, cfg.out, cfg.work)
    cfg.marker("cleanup").write_text(datetime.now(timezone.utc).isoformat())


SECTION_START = "<!-- WP3:pubchem:start -->"
SECTION_END = "<!-- WP3:pubchem:end -->"


def stage_docs(cfg: Cfg):
    stats = load_stats(cfg)
    idx = json.loads((cfg.out / "_index.json").read_text())
    casc = ordered_cascade(idx["filter_cascade"])
    timing = stats.get("timing_seconds", {})
    wall_min = sum(v for k, v in timing.items() if k in ("1", "2", "3")) / 60.0
    stage_min = ", ".join(f"stage {k} {timing[k] / 60:.1f} min" for k in ("1", "2", "3") if k in timing)
    peak_disk = (sum(stats.get(f"stage1_{n}", {}).get("parquet_bytes", 0) for n in STAGE1_ORDER)
                 + stats.get("stage2", {}).get("stage2_bytes", 0))
    rss = [v for k, v in stats.get("peak_rss_mb_by_stage", {}).items() if k in ("1", "2", "3")] or \
        [stats.get("stage2", {}).get("peak_rss_mb", 0)]
    lines = [SECTION_START, "## PubChem tier (WP3)", ""]
    lines += [
        f"- Built {idx['built_utc']} by `scripts/build_pubchem.py` (git {idx['git_commit']}). "
        f"Rebuild: `nice -n 10 python3 scripts/build_pubchem.py`; measured {wall_min:.0f} min wall "
        f"({stage_min}; downloads ran at about 20 MB/s), 2 cores, peak RSS {max(rss) / 1024:.1f} GB "
        f"(DuckDB memory_limit 7GB), peak disk about {peak_disk / 1e9:.0f} GB of intermediates (stage 1 parquet "
        "is deleted after stage 2, stage 2 after the checks). Resumable via `data/external/pubchem/.stage_*`; "
        "`--stage 1|2|3|check|cleanup|docs` runs one stage, `--force` rebuilds from scratch.",
        f"- Location: `data/processed/db/pubchem/`, {idx['n_shards']} shards `mass_0100.parquet` ... "
        f"`mass_1290.parquet` (shard XXXX holds 10 * floor(mono_mass / 10) = XXXX), plus `_index.json` "
        "(per shard: rows, min and max mass, bytes, sha256).",
        f"- Rows: {fmt_n(idx['total_rows'])} (one per ik14 within a shard; ik14 is unique across the tier "
        "because one ik14 implies one formula). Size: "
        f"{idx['total_bytes'] / 1e9:.2f} GB. sha256 of `_index.json`: `{sha256_file(cfg.out / '_index.json')}`.",
        "- Schema (PLAN.md 2.3): `ik14` string, `smiles` string (PubChem isomeric SMILES of the lowest CID), "
        "`formula` string (PubChem Hill formula), `mono_mass` float64 (PubChem monoisotopic mass), `cid` int32 "
        "(lowest CID in the ik14 group), `pmid_count` int32 (max over the group of distinct PMIDs per CID, 0 if none), "
        "`n_cids` int16 (CIDs collapsed into the ik14, capped at 32767). Rows sorted by mono_mass then cid, "
        "row groups of 100,000 rows, zstd level 9, so a pyarrow `filters=[('mono_mass', '>=', lo), ...]` read "
        "touches one or two row groups.",
        "- Not done here: RDKit parsing (lazy at inference, PLAN.md 2.2), fingerprints, NP-likeness.",
    ]
    if idx.get("priors"):
        lines.append("- R3 documentation-prior columns (" + ", ".join(f"`{n}`" for n in PRIOR_NAMES) + ") are "
                     "appended by `--stage priors`; see the PubChem documentation priors section below.")
    lines += [
        "",
        "Sources (PubChem FTP `Compound/Extras`, streamed, never stored; md5 checked against the published `.md5`):",
        "",
        "| File | Last-Modified | Bytes | Lines parsed | Columns found (tab separated, no header) |",
        "|---|---|---|---|---|",
    ]
    fmts = {
        "CID-SMILES.gz": "CID, isomeric SMILES",
        "CID-Mass.gz": "CID, formula (charged species carry a suffix: `C9H18NO4+`, `C6H5O7-3`), monoisotopic mass, exact mass",
        "CID-InChI-Key.gz": "CID, InChI, InChIKey (27 chars); only CID and key[:14] kept",
        "CID-PMID.gz": "CID, PMID, type (1 depositor, 2 MeSH, 3 publisher, 4 BioAssay); not sorted by CID; pmid_count = distinct PMIDs per CID over all types",
    }
    for f, s in idx["sources"].items():
        lines.append(f"| {f} | {s.get('last_modified')} | {fmt_n(s.get('bytes') or 0)} | "
                     f"{fmt_n(s.get('rows') or 0)} | {fmts.get(f, '')} |")
    lines += ["", "Filter cascade (rows remaining after each filter, applied in this order, then ik14 dedup):", "",
              "| Step | Rows | Removed |", "|---|---|---|"]
    prev = None
    for k, v in casc.items():
        lines.append(f"| {k} | {fmt_n(v)} | {'' if prev is None else fmt_n(prev - v)} |")
        prev = v
    lines.append(f"| dedup on ik14 (lowest CID kept) | {fmt_n(idx['total_rows'])} | "
                 f"{fmt_n((prev or 0) - idx['total_rows'])} |")
    chk = stats.get("check", {})
    if chk:
        ts = chk.get("test_summary", {})
        lines += ["", f"Self-check: caffeine found={chk.get('caffeine', {}).get('found')} "
                      f"({fmt_n(chk.get('caffeine', {}).get('n_candidates', 0))} candidates at 10 ppm), "
                      f"paclitaxel found={chk.get('paclitaxel', {}).get('found')} "
                      f"({fmt_n(chk.get('paclitaxel', {}).get('n_candidates', 0))} candidates); "
                      f"20 public-test masses at 10 ppm: candidates min {fmt_n(ts.get('min', 0))}, "
                      f"median {fmt_n(ts.get('median', 0))}, max {fmt_n(ts.get('max', 0))}, "
                      f"slowest query {ts.get('max_seconds')} s (pyarrow, one shard file read with row-group pruning)."]
    lines += [SECTION_END, ""]
    section = "\n".join(lines)
    dates = ", ".join(f"{f} {(s.get('last_modified') or '')[5:16]}" for f, s in idx["sources"].items())
    row = (f"| PubChem tier ({idx['n_shards']} shards) | `data/processed/db/pubchem/mass_XXXX.parquet`, "
           f"`_index.json` | {fmt_n(idx['total_rows'])} | {idx['total_bytes'] / 1e9:.2f} GB | "
           f"`{sha256_file(cfg.out / '_index.json')}` (`_index.json`) | "
           f"`nice -n 10 python3 scripts/build_pubchem.py`; PubChem FTP Compound/Extras ({dates}); "
           f"about {wall_min:.0f} min wall, 2 cores, 7 GB RAM"
           + ("; stage `priors` (R3, six more FTP files) adds the prior columns, see the R3 section" if idx.get("priors")
              else "") + " |")
    path = DOCS / "ARTEFACTS.md"
    if path.exists():
        text = path.read_text()
        if SECTION_START in text and SECTION_END in text:
            a = text.index(SECTION_START)
            b = text.index(SECTION_END) + len(SECTION_END)
            text = text[:a] + section.rstrip("\n") + text[b:]
        else:
            text = text.rstrip("\n") + "\n\n" + section
    else:
        text = "# Artefacts\n\n" + section
    text = upsert_table_row(text, row, key="| PubChem tier (")
    path.write_text(text)
    log.info("docs: wrote WP3 table row and section to %s", path)


def upsert_table_row(text: str, row: str, key: str) -> str:
    """Replace the line starting with key in the artefact table, or append the row
    to the first table whose header starts with '| artefact'."""
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        if ln.startswith(key):
            lines[i] = row
            return "\n".join(lines)
    for i, ln in enumerate(lines):
        if ln.lower().startswith("| artefact"):
            j = i + 1
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            lines.insert(j, row)
            return "\n".join(lines)
    return text


# ---------------------------------------------------------------- main

def reconcile_markers(cfg: Cfg) -> None:
    """Drop stage markers whose outputs are gone, so that a rerun rebuilds instead of
    skipping (e.g. data/processed was wiped but data/external/pubchem survived)."""
    idx_path = cfg.out / "_index.json"
    if cfg.marker("priors").exists() and idx_path.exists():
        files = [cfg.out / f for f in json.loads(idx_path.read_text()).get("shards", {})]
        if any(f.exists() and pq.read_schema(f).names != PRIOR_SCHEMA.names for f in files):
            log.warning("priors marker present but some shards lack the prior columns: redoing the write step")
            cfg.marker("priors").unlink()
            (cfg.work / "priors" / ".done_write").unlink(missing_ok=True)
            shutil.rmtree(cfg.work / "priors" / "write", ignore_errors=True)
    if cfg.marker("3").exists():
        ok = idx_path.exists()
        if ok:
            idx = json.loads(idx_path.read_text())
            ok = all((cfg.out / f).exists() for f in idx.get("shards", {}))
        if ok:
            return
        log.warning("stage 3 marker present but shards or _index.json missing: rebuilding from stage 1")
        for m in cfg.work.glob(".stage_*"):
            m.unlink()
        return
    if cfg.marker("2").exists() and not cfg.s2.exists():
        log.warning("stage 2 marker present but stage 2 output missing: rebuilding from stage 1")
        for m in cfg.work.glob(".stage_*"):
            m.unlink()
        return
    if not cfg.marker("2").exists():
        for name in STAGE1_ORDER:
            out = cfg.s1 / f"{name}.parquet"
            if cfg.marker(f"1_{name}").exists() and not out.exists():
                log.warning("stage 1 marker for %s present but %s missing: redoing it", name, out)
                cfg.marker(f"1_{name}").unlink()
                cfg.marker("1").unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=["all", "1", "2", "3", "priors", "check", "cleanup", "docs"])
    ap.add_argument("--test", action="store_true", help="small dry run in data/external/pubchem_test")
    ap.add_argument("--max-bytes", type=int, default=None, help="stop each download after this many compressed bytes (test)")
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tmp-dir", default=None)
    ap.add_argument("--chunk-cids", type=int, default=10_000_000)
    ap.add_argument("--memory-limit", default="7GB")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--keep-intermediates", action="store_true")
    ap.add_argument("--force", action="store_true", help="remove stage markers and outputs first")
    args = ap.parse_args()
    if args.max_bytes is not None and not args.test:
        ap.error("--max-bytes only with --test")

    logpath = setup_logging(args.stage)
    pa.set_cpu_count(max(1, args.threads - 1))
    pa.set_io_thread_count(2)
    cfg = Cfg(args)
    log.info("build_pubchem args=%s root=%s git=%s log=%s", vars(args), ROOT, git_commit(), logpath)
    if args.force:
        for m in cfg.work.glob(".stage_*"):
            m.unlink()
        for d in (cfg.s1, cfg.s2, cfg.work / "priors"):
            shutil.rmtree(d, ignore_errors=True)
        for p in cfg.out.glob("mass_*.parquet"):
            p.unlink()
    t0 = time.time()
    reconcile_markers(cfg)
    stages = ["1", "2", "3", "priors", "check", "cleanup", "docs"] if args.stage == "all" else [args.stage]
    for st in stages:
        ts = time.time()
        if st == "1":
            stage1(cfg)
        elif st == "2":
            stage2(cfg)
        elif st == "3":
            stage3(cfg)
        elif st == "priors":
            stage_priors(cfg)
        elif st == "check":
            res = stage_check(cfg)
            if not res["ok"]:
                log.error("acceptance checks failed: %s", res["problems"])
                if args.stage == "all":
                    log.error("not cleaning up so that the failure can be investigated")
                    sys.exit(2)
        elif st == "cleanup":
            if cfg.keep_intermediates:
                log.info("cleanup skipped (--keep-intermediates)")
            else:
                stage_cleanup(cfg)
        elif st == "docs":
            if cfg.test:
                log.info("docs skipped in --test mode")
            else:
                stage_docs(cfg)
                if cfg.marker("priors").exists():
                    priors_docs(cfg)
        stats = load_stats(cfg)
        stats.setdefault("timing_seconds", {})[st] = round(time.time() - ts, 1)
        if st in ("1", "2", "3", "priors"):
            stats.setdefault("peak_rss_mb_by_stage", {})[st] = round(peak_rss_mb())
        save_stats(cfg, stats)
        log.info("stage %s finished in %.0f s (peak rss %.0f MB)", st, time.time() - ts, peak_rss_mb())
    log.info("all requested stages done in %.0f s, peak rss %.0f MB", time.time() - t0, peak_rss_mb())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("build_pubchem failed")
        sys.exit(1)

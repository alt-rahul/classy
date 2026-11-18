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
import zlib
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.compute as pc
import pyarrow.parquet as pq

try:  # duckdb is a build-time dependency only
    import duckdb
except ImportError:  # pragma: no cover
    sys.exit("duckdb is required: pip install duckdb")


# ---------------------------------------------------------------- paths

def _root() -> Path:
    try:  # prefer the package if it exists and is importable
        import casmi.paths as cp  # type: ignore

        return Path(cp.ROOT)
    except Exception:
        return Path(__file__).resolve().parents[1]


ROOT = _root()
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


class Cfg:
    def __init__(self, args):
        if args.test:
            base = Path(args.work_dir or (ROOT / "data/external/pubchem_test"))
            self.work = base
            self.out = Path(args.out_dir or (base / "out"))
        else:
            self.work = Path(args.work_dir or (ROOT / "data/external/pubchem"))
            self.out = Path(args.out_dir or (ROOT / "data/processed/db/pubchem"))
        self.s1 = self.work / "stage1"
        self.s2 = self.work / "stage2"
        self.tmp = Path(args.tmp_dir or (ROOT / "data/external/tmp" / ("pubchem_test" if args.test else "pubchem")))
        self.stats_path = self.work / "build_stats.json"
        self.max_bytes = args.max_bytes
        self.chunk_cids = args.chunk_cids
        self.memory_limit = args.memory_limit
        self.threads = args.threads
        self.keep_intermediates = args.keep_intermediates
        self.test = args.test
        for d in (self.work, self.s1, self.s2, self.tmp, self.out):
            d.mkdir(parents=True, exist_ok=True)

    def marker(self, name: str) -> Path:
        return self.work / f".stage_{name}"


# ---------------------------------------------------------------- helpers

def setup_logging(stage: str) -> Path:
    logs = ROOT / "logs"
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


def duck(cfg: Cfg):
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{cfg.memory_limit}'")
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

    def _check_status(self):
        # Last HTTP status line in the dumped headers must be 206 (partial content).
        try:
            lines = [ln for ln in self.hdr_path.read_text(errors="replace").splitlines()
                     if ln.startswith("HTTP/")]
        except FileNotFoundError:
            lines = []
        status = lines[-1].split()[1] if lines else "?"
        if status not in ("206", "200"):
            raise RuntimeError(f"unexpected HTTP status {status} for {self.url}")
        if self.offset > 0 and status != "206":
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
            out.append(self._d.decompress(data))
            if self._d.eof:  # multi-member gzip
                data = self._d.unused_data
                self._d = zlib.decompressobj(31)
            else:
                data = b""
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
             fmt_n(prev_max), el, size / max(el, 1e-9) / 1e6, out.stat().st_size / 1e9, stream.restarts)
    if n_bad:
        log.info("  invalid row examples: %s", info["invalid_row_examples"])
    log.info("  first rows: %s", first_rows)
    return info


def stage1(cfg: Cfg):
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
        cfg.s1.mkdir(exist_ok=True)
        log.info("deleted stage1 parquet (%.2f GB) to keep the disk peak low", n1 / 1e9)


# ---------------------------------------------------------------- stage 3: per-shard dedup

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
    shard_meta_dir.mkdir(exist_ok=True)
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
        n_in = con.execute(f"SELECT count(*) FROM read_parquet({flist}, hive_partitioning=false)").fetchone()[0]
        n, n_ik, mn, mx, n_formula_conflict = con.execute(f"""
            SELECT count(*), count(DISTINCT ik14), min(mono_mass), max(mono_mass), 0
            FROM read_parquet('{tmp}')
        """).fetchone()
        if n != n_ik:
            raise RuntimeError(f"bucket {b}: ik14 not unique ({n} rows, {n_ik} distinct)")
        sch = pq.read_schema(tmp)
        if not sch.remove_metadata().equals(OUT_SCHEMA):
            raise RuntimeError(f"bucket {b}: schema mismatch\n{sch}\nexpected\n{OUT_SCHEMA}")
        tmp.replace(out)
        meta = {"file": out.name, "rows": n, "rows_before_dedup": n_in, "min_mass": mn, "max_mass": mx,
                "bytes": out.stat().st_size, "sha256": sha256_file(out),
                "row_groups": pq.ParquetFile(out).metadata.num_row_groups}
        meta_path.write_text(json.dumps(meta))
        log.info("stage3 %s: %s -> %s rows, %.1f MB, %.1f s", out.name, fmt_n(n_in), fmt_n(n),
                 meta["bytes"] / 1e6, time.time() - ts)
    con.close()
    shards = [json.loads(p.read_text()) for p in sorted(shard_meta_dir.glob("*.json"))]
    total = sum(s["rows"] for s in shards)
    total_in = sum(s["rows_before_dedup"] for s in shards)
    size = sum(s["bytes"] for s in shards)
    stats = load_stats(cfg)
    el = time.time() - t0
    stats["stage3"] = {"seconds": round(el, 1), "rows_before_dedup": total_in, "rows": total,
                       "n_shards": len(shards), "bytes": size}
    save_stats(cfg, stats)
    write_index(cfg, shards, stats)
    log.info("stage3 done: %d shards, %s rows before ik14 dedup, %s after, %.2f GB, %.0f s",
             len(shards), fmt_n(total_in), fmt_n(total), size / 1e9, el)
    cfg.marker("3").write_text(json.dumps({"rows": total, "shards": len(shards)}))


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
        "filter_cascade": stats.get("stage2", {}).get("filter_cascade"),
        "rows_before_ik14_dedup": sum(s["rows_before_dedup"] for s in shards),
        "sources": sources,
        "shards": {s["file"]: {k: s[k] for k in ("rows", "min_mass", "max_mass", "bytes", "sha256")}
                   for s in shards},
    }
    (cfg.out / "_index.json").write_text(json.dumps(idx, indent=1))


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
    t = pq.read_table(ROOT / "data/raw/test.parquet", columns=["molecule_id", "adduct", "precursor_mz"]).to_pylist()
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
        sch = pq.read_schema(p).remove_metadata()
        if not sch.equals(OUT_SCHEMA):
            fail(f"{p.name} schema {sch}")
        n, n_ik, n_unsorted = con.execute(f"""
            SELECT count(*), count(DISTINCT ik14),
                   count(*) FILTER (WHERE mono_mass < prev)
            FROM (SELECT ik14, mono_mass, lag(mono_mass) OVER () AS prev FROM read_parquet('{p}'))
        """).fetchone()
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
    casc = idx["filter_cascade"] or {}
    lines = [SECTION_START, "## PubChem tier (WP3)", ""]
    lines += [
        f"- Built {idx['built_utc']} by `scripts/build_pubchem.py` (git {idx['git_commit']}). "
        f"Rebuild: `nice -n 10 python3 scripts/build_pubchem.py` (about 1 to 2 hours, 2 cores, under 7 GB RAM, "
        "about 11 GB peak disk; resumable via `data/external/pubchem/.stage_*`).",
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
        "CID-PMID.gz": "CID, PMID, type (1 depositor, 2 MeSH, 3 publisher); not sorted by CID; pmid_count = distinct PMIDs per CID over all types",
    }
    for f, s in idx["sources"].items():
        lines.append(f"| {f} | {s.get('last_modified')} | {fmt_n(s.get('bytes') or 0)} | "
                     f"{fmt_n(s.get('rows') or 0)} | {fmts.get(f, '')} |")
    lines += ["", "Filter cascade (rows remaining after each filter, applied in this order, then ik14 dedup):", "",
              "| Step | Rows |", "|---|---|"]
    for k, v in casc.items():
        lines.append(f"| {k} | {fmt_n(v)} |")
    lines.append(f"| dedup on ik14 (lowest CID kept) | {fmt_n(idx['total_rows'])} |")
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
    path = ROOT / "docs/ARTEFACTS.md"
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
    path.write_text(text)
    log.info("docs: wrote WP3 section to %s", path)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=["all", "1", "2", "3", "check", "cleanup", "docs"])
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
        for d in (cfg.s1, cfg.s2):
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)
        for p in cfg.out.glob("mass_*.parquet"):
            p.unlink()
    t0 = time.time()
    stages = ["1", "2", "3", "check", "cleanup", "docs"] if args.stage == "all" else [args.stage]
    for st in stages:
        ts = time.time()
        if st == "1":
            stage1(cfg)
        elif st == "2":
            stage2(cfg)
        elif st == "3":
            stage3(cfg)
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
        stats = load_stats(cfg)
        stats.setdefault("timing_seconds", {})[st] = round(time.time() - ts, 1)
        save_stats(cfg, stats)
        log.info("stage %s finished in %.0f s (peak rss %.0f MB)", st, time.time() - ts, peak_rss_mb())
    log.info("all requested stages done in %.0f s, peak rss %.0f MB", time.time() - t0, peak_rss_mb())


if __name__ == "__main__":
    main()

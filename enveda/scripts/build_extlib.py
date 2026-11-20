"""External spectral libraries (ROADMAP R2): download, stage, deduplicate against train, key,
index, calibrate, evaluate, and build the C2-ZEN query set.

Stages (``--stage``; ``all`` runs fetch+stage for every source in SOURCE_ORDER, then the rest):

``fetch``     download one ``--source`` into ``data/external/extlib/<source>/`` (sha256 and
              size in ``downloads.json``). GNPS sources are fetched library by library inside
              ``stage`` (download, convert, delete) to keep the disk peak low.
``stage``     parse + standardise + filter + clean (casmi.extlib.Stager) into
              ``data/processed/extlib/staging/<source>/part-*.parquet``; raw downloads larger
              than ``--keep-raw-mb`` are deleted afterwards. enveda180 also writes the raw
              peaks of structures absent from train (``c2zen_raw.parquet``, for C2-ZEN).
``finalize``  deduplicate (casmi.extlib.dedup) and write ``data/processed/extlib/extlib_<source>.parquet``.
``keys``      mkey per record (train mkey for train structures, casmi.metric.mkeys otherwise,
              cache ``data/processed/extlib/mkey_cache.parquet``), NP-tier and PubChem-tier
              membership; rewrites the final files with ``mkey``; ``structures.parquet``.
``library``   ``Library.build`` of the final files into ``data/processed/extlib/library/``
              (precleaned) and a small ``precleaned=False`` smoke build in a scratch dir.
``calibrate`` per source: cosine / matched-peak thresholds giving ``extlib_calib_precision``
              rank-1 precision for timsTOF train queries of structures present in train and
              in the source (staged spectra, mass-window library of that source only).
``evaluate``  library-only MRR on NP-OTHER molecules with external spectra, all train spectra
              hidden; NP-TIMS C1 with and without the external library.
``c2zen``     C2-ZEN validation queries (test schema) from Enveda-180 structures absent from
              train: ``data/processed/validation/queries_C2-ZEN.parquet`` + truth.
``holdout``   ``n_ext_lib`` per holdout_v1 molecule (``data/processed/extlib/holdout_ext.parquet``).
``report``    ``data/processed/extlib/report.json`` + a markdown table on stdout.
``crosscheck`` structure overlap with the public Kaggle dumps (taeyangg4, takumuhata, samartalwar).

Downloads are untrusted data: they are only parsed (json, csv, parquet, msp text), never run.
Resources: ``--n-jobs`` RDKit processes (default 2), run under ``nice -n 10``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from casmi import extlib, paths
from casmi.config import Config, load_config
from casmi.log import get_logger, log_run, peak_rss_mb

RAW_ROOT = paths.EXTERNAL / "extlib"
GNPS_BASE = "https://external.gnps2.org"
MONA_RETRIEVE = "https://mona.fiehnlab.ucdavis.edu/rest/downloads/retrieve/"
MONA_EXPERIMENTAL_ID = "19a23fd5-4e06-4122-ae9d-169198ee9794"   # "All Spectra - Experimental Spectra" JSON
MASSBANK_TAG = "2026.11"
DOWNLOADS: dict[str, list[tuple[str, str]]] = {
    "massbank": [
        ("MassBank_NISTformat.msp",
         f"https://github.com/MassBank/MassBank-data/releases/download/{MASSBANK_TAG}/MassBank_NISTformat.msp"),
        ("MassBank.json",
         f"https://github.com/MassBank/MassBank-data/releases/download/{MASSBANK_TAG}/MassBank.json"),
    ],
    "mona": [("MoNA-export-Experimental_Spectra-json.zip", MONA_RETRIEVE + MONA_EXPERIMENTAL_ID)],
    "msnlib": [("msnlib_ms2.parquet", "https://zenodo.org/api/records/23494858/files/msnlib_ms2.parquet/content")],
    "enveda180": [("enveda-180.jsonl.gz", "https://zenodo.org/api/records/21346580/files/enveda-180.jsonl.gz/content")],
    "massspecgym": [("MassSpecGym1.5.tsv",
                     "https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/resolve/main/data/MassSpecGym1.5.tsv")],
    "kaggle_gnpsfull": [(f, f"https://www.kaggle.com/api/v1/datasets/download/taeyangg4/casmi26-extlib-gnps-full/{f}")
                        for f in ("ext_meta.parquet", "ext_mz.npy", "ext_in.npy", "BUILD.json")],
}
KAGGLE_AUTH = "Basic ZHVtbXk6ZHVtbXk="   # dummy; the session proxy injects the real credentials
MULTIPLEX_LIBS = tuple(f"MULTIPLEX-SYNTHESIS-LIBRARY-ALL-PARTITION-{i}" for i in range(1, 7)) + (
    "GNPS-SYNTHESIS-CARNITINES",)
# GNPS libraries of type GNPS or IMPORT (external.gnps2.org/gnpslibrary, 2026-11-20). The
# propagated types (suspect lists, IIMN, drug analogs, conjugated metabolome, bile-acid isomers,
# REFRAME) carry inferred labels and are not used; the multiplex synthesis library is its own
# source (gnps_multiplex).
GNPS_TYPES = ("GNPS", "IMPORT")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _curl(url: str, dest: Path, lg, kaggle: bool = False, tries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    cmd = ["curl", "-sSL", "--fail", "--retry", "3", "--retry-delay", "5", "-o", str(tmp)]
    if kaggle:
        cmd += ["-H", f"Authorization: {KAGGLE_AUTH}"]
    cmd.append(url)
    for k in range(tries):
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            tmp.rename(dest)
            lg.info("downloaded %s (%.1f MB, %.0f s)", dest.name, dest.stat().st_size / 1e6,
                    time.perf_counter() - t0)
            return
        lg.warning("download failed (%d/%d) %s: %s", k + 1, tries, url, r.stderr.strip()[-300:])
        time.sleep(10)
    raise RuntimeError(f"download failed: {url}")


def _record_download(source: str, name: str, url: str, path: Path, deleted: bool = False) -> None:
    rec_path = RAW_ROOT / source / "downloads.json"
    data = json.loads(rec_path.read_text()) if rec_path.exists() else {}
    data[name] = {"url": url, "bytes": path.stat().st_size if path.exists() else data.get(name, {}).get("bytes"),
                  "sha256": _sha256(path) if path.exists() else data.get(name, {}).get("sha256"),
                  "downloaded": time.strftime("%Y-%m-%d %H:%M:%S"), "deleted_after_staging": deleted}
    rec_path.write_text(json.dumps(data, indent=2))


def _mark_deleted(source: str, name: str) -> None:
    rec_path = RAW_ROOT / source / "downloads.json"
    data = json.loads(rec_path.read_text()) if rec_path.exists() else {}
    if name in data:
        data[name]["deleted_after_staging"] = True
        rec_path.write_text(json.dumps(data, indent=2))


def stage_fetch(source: str, lg) -> None:
    for name, url in DOWNLOADS.get(source, []):
        dest = RAW_ROOT / source / name
        if dest.exists() and dest.stat().st_size > 0:
            lg.info("%s exists (%.1f MB)", dest, dest.stat().st_size / 1e6)
        else:
            _curl(url, dest, lg, kaggle="kaggle.com" in url)
        _record_download(source, name, url, dest)



def _join(*cols) -> pa.ChunkedArray:
    """Element-wise "a|b|c" of string columns (any string type), as large_string."""
    cs = [pc.cast(c, pa.large_string()) for c in cols]
    return pc.binary_join_element_wise(*cs, pa.scalar("|", pa.large_string()))


def _isin(col, values) -> np.ndarray:
    return pc.is_in(pc.cast(col, pa.large_string()), value_set=pc.cast(values, pa.large_string())).to_numpy(
        zero_copy_only=False)


# --------------------------------------------------------------------------------------
# train context
# --------------------------------------------------------------------------------------

def train_dir() -> Path:
    return paths.PROCESSED / "train"


def train_ik14_set() -> set:
    t = pq.read_table(train_dir() / "structures.parquet", columns=["ik14"])
    return set(t.column("ik14").to_pylist())


def _gnps_library_list(lg) -> list[tuple[str, str]]:
    """(library name, type) from the GNPS library page (embedded JSON)."""
    tmp = RAW_ROOT / "gnps" / "gnpslibrary.html"
    _curl(GNPS_BASE + "/gnpslibrary", tmp, lg)
    text = tmp.read_text(errors="replace")
    out = []
    for m in re.finditer(r"(\[\{\"[a-z_]+link.*?\}\])", text, re.S):
        try:
            arr = json.loads(m.group(1))
        except ValueError:
            continue
        for d in arr:
            if d.get("type") and d.get("jsonlink", "").startswith("/gnpslibrary/"):
                out.append((d["libraryname"], d["type"]))
    lg.info("GNPS library page: %d libraries", len(out))
    return out


# --------------------------------------------------------------------------------------
# stage
# --------------------------------------------------------------------------------------

class RawSink:
    """Raw peaks of staged rows whose structure is absent from train (C2-ZEN source)."""

    def __init__(self, path: Path, train_ik14: set):
        self.path = path
        self.train = train_ik14
        self.writer = None
        self.schema = pa.schema([("spectrum_id", pa.string()), ("inchikey14", pa.string()),
                                 ("raw_mzs", pa.list_(pa.float64())),
                                 ("raw_intensities", pa.list_(pa.float64()))])
        self.n = 0

    def __call__(self, rows: list[dict]) -> None:
        rows = [r for r in rows if r["inchikey14"] not in self.train]
        if not rows:
            return
        tab = pa.Table.from_pydict({
            "spectrum_id": [r["spectrum_id"] for r in rows],
            "inchikey14": [r["inchikey14"] for r in rows],
            "raw_mzs": [np.asarray(r["mzs"], np.float64) for r in rows],
            "raw_intensities": [np.asarray(r["intensities"], np.float64) for r in rows]},
            schema=self.schema)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")
        self.writer.write_table(tab)
        self.n += len(rows)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def _stage_records(source: str, part: str, records, cfg: Config, mapper, n_jobs: int, lg,
                   cache: dict, raw_sink=None) -> dict:
    out = extlib.staging_dir(source)
    with extlib.Stager(source, cfg, mapper, out, n_jobs=n_jobs, part=part,
                       struct_cache=cache, raw_sink=raw_sink) as st:
        st.add_all(records)
    info = st.info()
    lg.info("%s/%s: %d in, %d staged; drops %s", source, part, info["n_in"], info["n_staged"],
            {k: v for k, v in info["drops"].items() if k != "staged"})
    return info


def stage_source(source: str, cfg: Config, n_jobs: int, keep_raw_mb: float, lg, resume: bool = False) -> dict:
    d = extlib.staging_dir(source)
    if d.exists() and not resume:
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    mapper = extlib.AdductMapper(extlib.known_adducts())
    cache: dict = {}
    infos: list[dict] = []
    raw = RAW_ROOT / source
    t0 = time.perf_counter()
    if source in ("gnps", "gnps_multiplex"):
        libs = _gnps_library_list(lg)
        if source == "gnps":
            names = [n for n, t in libs if t in GNPS_TYPES and n not in MULTIPLEX_LIBS
                     and not n.startswith("ALL_GNPS")]
        else:
            names = [n for n, _ in libs if n in MULTIPLEX_LIBS]
        lg.info("%s: %d libraries: %s", source, len(names), names)
        import threading

        def fetch(u: str, dest: Path) -> None:
            if dest.exists() and dest.stat().st_size > 0:
                return
            try:
                _curl(u, dest, lg, tries=2)
            except RuntimeError as exc:
                lg.warning("skipping GNPS library %s: %s", dest.stem, exc)

        nxt = None   # prefetch of the next library (one download ahead, bounded disk)
        for i, name in enumerate(names):
            part = f"part-{i:03d}-{name}"
            done = d / f"{part}.info.json"
            f = raw / f"{name}.json"
            url = f"{GNPS_BASE}/gnpslibrary/{name}.json"
            if nxt is not None:
                nxt.join()
                nxt = None
            if done.exists():
                infos.append(json.loads(done.read_text()))
                continue
            fetch(url, f)
            if i + 1 < len(names) and not (d / f"part-{i + 1:03d}-{names[i + 1]}.info.json").exists():
                nxt = threading.Thread(target=fetch, args=(f"{GNPS_BASE}/gnpslibrary/{names[i + 1]}.json",
                                                           raw / f"{names[i + 1]}.json"), daemon=True)
                nxt.start()
            if not f.exists():
                info = {"source": source, "part": part, "n_in": 0, "n_staged": 0, "drops": {},
                        "by_sub_lib": {}, "skipped": "download failed (listed on the GNPS page, HTTP 404)"}
                done.write_text(json.dumps(info))
                infos.append(info)
                continue
            _record_download(source, f.name, url, f)
            (d / f"{part}.parquet").unlink(missing_ok=True)
            info = _stage_records(source, part, extlib.iter_gnps_json(f, name), cfg, mapper, n_jobs, lg, cache)
            done.write_text(json.dumps(info))
            infos.append(info)
            if f.stat().st_size > keep_raw_mb * 1e6:
                f.unlink()
                _mark_deleted(source, f.name)
    elif source == "massbank":
        lic = extlib.massbank_licences(raw / "MassBank.json")
        lg.info("MassBank licences: %d accessions, %s", len(lic),
                Counter(extlib.short_licence(v) for v in lic.values()).most_common())
        lic = {k: extlib.short_licence(v) for k, v in lic.items()}
        infos.append(_stage_records(source, "part-000", extlib.iter_massbank_msp(raw / "MassBank_NISTformat.msp", lic),
                                    cfg, mapper, n_jobs, lg, cache))
    elif source == "mona":
        zp = raw / "MoNA-export-Experimental_Spectra-json.zip"
        if zp.exists():
            infos.append(_stage_records(source, "part-000", extlib.iter_mona_json(zp), cfg, mapper, n_jobs, lg, cache))
        else:
            # streamed: curl -> streaming unzip -> parser (the 3.9 GB zip never touches disk)
            url = DOWNLOADS["mona"][0][1]
            proc = subprocess.Popen(["curl", "-sSL", "--fail", url], stdout=subprocess.PIPE)
            h = hashlib.sha256()

            class _Tee:
                def read(self, n: int) -> bytes:
                    b = proc.stdout.read(n)
                    h.update(b)
                    self.n = getattr(self, "n", 0) + len(b)
                    return b

            tee = _Tee()
            try:
                infos.append(_stage_records(source, "part-000", extlib.iter_mona_json(extlib.open_zip_stream(tee)),
                                            cfg, mapper, n_jobs, lg, cache))
            finally:
                rest = proc.stdout.read()
                h.update(rest)
                proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"MoNA download failed (curl exit {proc.returncode})")
            rec = RAW_ROOT / source / "downloads.json"
            rec.parent.mkdir(parents=True, exist_ok=True)
            data = json.loads(rec.read_text()) if rec.exists() else {}
            data[zp.name] = {"url": url, "bytes": getattr(tee, "n", 0) + len(rest), "sha256": h.hexdigest(),
                             "downloaded": time.strftime("%Y-%m-%d %H:%M:%S"), "streamed_not_stored": True}
            rec.write_text(json.dumps(data, indent=2))
    elif source == "msnlib":
        infos.append(_stage_records(source, "part-000", extlib.iter_msg_table(raw / "msnlib_ms2.parquet", "msnlib"),
                                    cfg, mapper, n_jobs, lg, cache))
    elif source == "massspecgym":
        infos.append(_stage_records(source, "part-000", extlib.iter_msg_table(raw / "MassSpecGym1.5.tsv", "massspecgym"),
                                    cfg, mapper, n_jobs, lg, cache))
    elif source == "enveda180":
        sink = RawSink(extlib.extlib_dir() / "c2zen_raw.parquet", train_ik14_set())
        try:
            infos.append(_stage_records(source, "part-000", extlib.iter_enveda_jsonl(raw / "enveda-180.jsonl.gz"),
                                        cfg, mapper, n_jobs, lg, cache, raw_sink=sink))
        finally:
            sink.close()
        lg.info("C2-ZEN raw peaks: %d spectra of structures absent from train", sink.n)
    elif source == "kaggle_gnpsfull":
        infos.append(_stage_records(source, "part-000",
                                    extlib.iter_kaggle_npy(raw / "ext_meta.parquet", raw / "ext_mz.npy", raw / "ext_in.npy"),
                                    cfg, mapper, n_jobs, lg, cache))
    else:
        raise ValueError(f"unknown source {source}")
    total = Counter()
    for inf in infos:
        total["n_in"] += inf["n_in"]
        total["n_staged"] += inf["n_staged"]
        for k, v in inf["drops"].items():
            total[f"drop_{k}"] += v
    info = {"source": source, "seconds": round(time.perf_counter() - t0, 1), "totals": dict(total),
            "parts": infos, "n_structures_cached": len(cache)}
    (d / "stage_info.json").write_text(json.dumps(info, indent=2))
    lg.info("%s staged: %s in %.0f s", source, dict(total), info["seconds"])
    if source not in ("gnps", "gnps_multiplex"):
        for name, _ in DOWNLOADS.get(source, []):
            f = raw / name
            if f.exists() and f.stat().st_size > keep_raw_mb * 1e6:
                f.unlink()
                _mark_deleted(source, name)
                lg.info("deleted raw %s", f)
    return info


# --------------------------------------------------------------------------------------
# finalize
# --------------------------------------------------------------------------------------

def _staged_sources() -> list[str]:
    return [s for s in extlib.SOURCE_ORDER if list(extlib.staging_dir(s).glob("part-*.parquet"))]


def _staged_files(source: str) -> list[Path]:
    return sorted(extlib.staging_dir(source).glob("part-*.parquet"))


def _read_staged(source: str, columns: list[str]) -> pa.Table:
    tabs = [pq.read_table(f, columns=columns) for f in _staged_files(source)]
    return pa.concat_tables(tabs) if tabs else pa.table({c: [] for c in columns})


class StagedPeaks:
    """Cleaned peaks of selected staged rows (global row number per source)."""

    def __init__(self, want: dict[str, np.ndarray]):
        self.data: dict[str, tuple] = {}
        for src, rows in want.items():
            rows = np.unique(np.asarray(rows, np.int64))
            mzs, its, got = [], [], []
            base = 0
            for f in _staged_files(src):
                pf = pq.ParquetFile(f)
                for rg in range(pf.num_row_groups):
                    n = pf.metadata.row_group(rg).num_rows
                    a, b = np.searchsorted(rows, [base, base + n])
                    if b > a:
                        t = pf.read_row_group(rg, columns=["ms2_mzs", "ms2_normalized_intensities"])
                        t = t.take(pa.array(rows[a:b] - base))
                        mzs.extend(t.column(0).to_numpy(zero_copy_only=False))
                        its.extend(t.column(1).to_numpy(zero_copy_only=False))
                        got.append(rows[a:b])
                    base += n
            got_rows = np.concatenate(got) if got else np.zeros(0, np.int64)
            mz, it, off = extlib._flat(mzs, its)
            self.data[src] = (got_rows, mz, it, off)

    def __call__(self, rows: np.ndarray, sources: np.ndarray) -> tuple:
        out_mz, out_int, lens = [], [], []
        for r, s in zip(rows, sources):
            got, mz, it, off = self.data[s]
            k = int(np.searchsorted(got, r))
            out_mz.append(mz[off[k]:off[k + 1]])
            out_int.append(it[off[k]:off[k + 1]])
        return extlib._flat(out_mz, out_int)


def train_peaks_for(pairs: list[str], lg) -> dict[str, tuple]:
    """(ik14|adduct) -> cleaned train peaks (flat) from WP6's spectra files."""
    want = pa.array(sorted(set(pairs)), pa.large_string())
    acc: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    t0 = time.perf_counter()
    n = 0
    for f in sorted((train_dir() / "spectra").glob("*.parquet")):
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=50000, columns=["inchikey14", "adduct", "ms2_mzs",
                                                             "ms2_normalized_intensities"]):
            key = _join(b.column(0), b.column(1))
            hit = pc.is_in(key, value_set=want)
            if not pc.any(hit).as_py():
                continue
            sel = pa.Table.from_batches([b]).filter(hit)
            keys = _join(sel.column(0), sel.column(1)).to_pylist()
            mzs = sel.column(2).to_numpy(zero_copy_only=False)
            its = sel.column(3).to_numpy(zero_copy_only=False)
            for k, m, x in zip(keys, mzs, its):
                acc[k][0].append(m)
                acc[k][1].append(x)
                n += 1
    lg.info("train peaks: %d spectra for %d of %d (ik14, adduct) pairs (%.0f s)", n, len(acc),
            len(want), time.perf_counter() - t0)
    return {k: extlib._flat(v[0], v[1]) for k, v in acc.items()}


def stage_finalize(cfg: Config, lg) -> dict:
    t0 = time.perf_counter()
    sources = _staged_sources()
    lg.info("finalize sources (priority order): %s", sources)
    tm = pq.read_table(train_dir() / "meta.parquet", columns=["ik14", "adduct", "instrument_family"])
    combos = pc.unique(_join(tm["ik14"], tm["adduct"], tm["instrument_family"]))
    pairs_t = pc.unique(_join(tm["ik14"], tm["adduct"]))
    train_ik = pc.unique(pc.cast(tm["ik14"], pa.large_string()))
    del tm
    lg.info("train: %d structures, %d (ik14, adduct) pairs, %d (ik14, adduct, instrument) combos",
            len(train_ik), len(pairs_t), len(combos))
    cols = ["inchikey14", "adduct", "instrument_family"]
    metas = []
    for pri, s in enumerate(sources):
        t = _read_staged(s, cols)
        k3 = _join(t["inchikey14"], t["adduct"], t["instrument_family"])
        k2 = _join(t["inchikey14"], t["adduct"])
        intr = _isin(t["inchikey14"], train_ik)
        cov = _isin(k3, combos)
        tpair = _isin(k2, pairs_t)
        alive = ~cov
        df = pd.DataFrame({
            "row": np.flatnonzero(alive).astype(np.int64),
            "source": s, "priority": pri,
            "inchikey14": np.asarray(t["inchikey14"].to_numpy(zero_copy_only=False), object)[alive],
            "adduct": np.asarray(t["adduct"].to_numpy(zero_copy_only=False), object)[alive],
            "instrument_family": np.asarray(t["instrument_family"].to_numpy(zero_copy_only=False), object)[alive],
            "in_train": intr[alive], "train_pair": tpair[alive],
        })
        lg.info("%s: %d staged, %d covered by train (ik14, adduct, instrument), %d left (%d of train structures)",
                s, t.num_rows, int(cov.sum()), len(df), int(df["in_train"].sum()))
        metas.append((s, t.num_rows, int(cov.sum()), df))
    meta = pd.concat([m[3] for m in metas], ignore_index=True)
    meta["source"] = meta["source"].astype(object)
    # rows that need peaks: train-pair rows (dup_train check) and multi-row (ik14, adduct) groups
    meta["pair"] = meta["inchikey14"] + "|" + meta["adduct"]
    gsize = meta.groupby("pair")["row"].transform("size")
    need = (meta["in_train"] & meta["train_pair"]) | (gsize > 1)
    want = {s: meta.loc[need & (meta["source"] == s), "row"].to_numpy() for s in sources}
    lg.info("rows needing peak comparison: %d (%s)", int(need.sum()), {s: len(v) for s, v in want.items()})
    peaks = StagedPeaks(want)
    lg.info("staged peaks loaded; RSS peak %.0f MB", peak_rss_mb())
    tpairs = sorted(set(meta.loc[meta["in_train"] & meta["train_pair"], "pair"]))
    tp = train_peaks_for(tpairs, lg)

    dmeta = meta[["row", "source", "inchikey14", "adduct", "instrument_family", "priority"]]
    # covered rows were removed already; dedup sees train_combos empty, train pairs/ik14 sets
    dec = extlib.dedup(dmeta, peaks, set(), set(tpairs), set(meta.loc[meta["in_train"], "inchikey14"]),
                       lambda prs: tp, cfg)
    meta["decision"] = dec.to_numpy()
    del peaks, tp
    counts = {}
    out_dir = paths.ensure_dir(extlib.extlib_dir())
    for s, n_staged, n_cov, _ in metas:
        keep_rows = np.sort(meta.loc[(meta["source"] == s) & (meta["decision"] == "keep"), "row"].to_numpy())
        tabs = []
        base = 0
        for f in _staged_files(s):
            pf = pq.ParquetFile(f)
            for rg in range(pf.num_row_groups):
                n = pf.metadata.row_group(rg).num_rows
                a, b = np.searchsorted(keep_rows, [base, base + n])
                if b > a:
                    tabs.append(pf.read_row_group(rg).take(pa.array(keep_rows[a:b] - base)))
                base += n
        tab = pa.concat_tables(tabs) if tabs else pq.read_table(_staged_files(s)[0]).slice(0, 0)
        intr = pa.array(_isin(tab["inchikey14"], train_ik))
        tab = tab.append_column("in_train", intr).append_column("mkey", pa.nulls(tab.num_rows, pa.string()))
        tab = tab.select(list(extlib.FINAL_COLUMNS))
        # structures absent from train first, then by structure and spectrum id
        order = pc.sort_indices(tab, sort_keys=[("in_train", "ascending"), ("inchikey14", "ascending"),
                                                ("spectrum_id", "ascending")])
        tab = tab.take(order)
        path = out_dir / f"extlib_{s}.parquet"
        pq.write_table(tab, path, compression="zstd", row_group_size=50000)
        dsub = meta.loc[meta["source"] == s, "decision"].value_counts().to_dict()
        counts[s] = {"staged": n_staged, "train_covered": n_cov, **{k: int(v) for k, v in dsub.items()},
                     "written": tab.num_rows,
                     "structures": int(pc.count_distinct(tab["inchikey14"]).as_py()),
                     "structures_new_ik14": int(pc.count_distinct(tab.filter(pc.invert(tab["in_train"]))["inchikey14"]).as_py())}
        lg.info("%s: %s -> %s", s, counts[s], path)
    info = {"sources": sources, "counts": counts, "seconds": round(time.perf_counter() - t0, 1)}
    (out_dir / "finalize_info.json").write_text(json.dumps(info, indent=2))
    return info


# --------------------------------------------------------------------------------------
# keys: mkey, tiers, structures table
# --------------------------------------------------------------------------------------

def final_files() -> list[Path]:
    return [extlib.extlib_dir() / f"extlib_{s}.parquet" for s in extlib.SOURCE_ORDER
            if (extlib.extlib_dir() / f"extlib_{s}.parquet").exists()]


def stage_keys(cfg: Config, n_jobs: int, lg) -> dict:
    from casmi import db, metric

    t0 = time.perf_counter()
    files = final_files()
    st = pq.read_table(train_dir() / "structures.parquet", columns=["ik14", "mkey"]).to_pandas()
    train_mk = dict(zip(st["ik14"], st["mkey"]))
    train_mkeys = set(st["mkey"].dropna())
    rows = []
    for f in files:
        t = pq.read_table(f, columns=["inchikey14", "normalized_smiles", "ingest_lib", "in_train",
                                      "instrument_family", "adduct", "licence"]).to_pandas()
        rows.append(t)
    allr = pd.concat(rows, ignore_index=True)
    # one SMILES per ik14 (most frequent, then lexicographic)
    sm = (allr.groupby(["inchikey14", "normalized_smiles"]).size().rename("n").reset_index()
          .sort_values(["inchikey14", "n", "normalized_smiles"], ascending=[True, False, True], kind="mergesort")
          .drop_duplicates("inchikey14"))
    new = sm[~sm["inchikey14"].isin(train_mk.keys())]
    lg.info("final structures: %d (%d absent from train by ik14); computing mkeys for those", len(sm), len(new))
    res = metric.mkeys(new["normalized_smiles"].tolist(), cfg, n_jobs=n_jobs,
                       cache_path=extlib.extlib_dir() / "mkey_cache.parquet", return_status=True)
    sm["mkey"] = sm["inchikey14"].map(train_mk)
    sm["mkey_status"] = np.where(sm["mkey"].notna(), "train", None)
    sm.loc[new.index, "mkey"] = [r[0] for r in res]
    sm.loc[new.index, "mkey_status"] = [r[1] for r in res]
    # a failed key falls back to the ik14 (same rule as metric's timeout fallback)
    miss = sm["mkey"].isna()
    sm.loc[miss, "mkey"] = sm.loc[miss, "inchikey14"]
    sm["in_train"] = sm["inchikey14"].isin(train_mk.keys())
    sm["mkey_in_train"] = sm["mkey"].isin(train_mkeys)
    # NP tier and PubChem tier membership
    npt = pq.read_table(db.np_dir() / "np.parquet", columns=["ik14"]).column("ik14").to_pylist()
    sm["in_np"] = sm["inchikey14"].isin(set(npt))
    hit = np.zeros(len(sm), bool)
    pmid = np.full(len(sm), np.nan)
    idx = pd.Index(sm["inchikey14"].to_numpy(object))
    shards = sorted(db.pubchem_dir().glob("mass_*.parquet"))
    for p in shards:
        t = pq.read_table(p, columns=["ik14", "pmid_count"])
        pos = pd.Index(t.column("ik14").to_numpy(zero_copy_only=False)).get_indexer(idx)
        found = pos >= 0
        if found.any():
            hit |= found
            pmid[found] = t.column("pmid_count").to_numpy(zero_copy_only=False)[pos[found]]
    sm["in_pubchem"] = hit
    sm["pmid_count"] = pmid
    # per-structure source lists
    src = allr.groupby("inchikey14")["ingest_lib"].agg(lambda x: ";".join(sorted(set(x))))
    nsp = allr.groupby("inchikey14").size()
    sm["sources"] = sm["inchikey14"].map(src)
    sm["n_spectra"] = sm["inchikey14"].map(nsp).astype(int)
    sm = sm.drop(columns=["n"]).rename(columns={"inchikey14": "ik14", "normalized_smiles": "smiles"})
    sm = sm.sort_values("ik14", kind="mergesort").reset_index(drop=True)
    sm.to_parquet(extlib.extlib_dir() / "structures.parquet", index=False)
    mkmap = dict(zip(sm["ik14"], sm["mkey"]))
    for f in files:
        t = pq.read_table(f)
        mkc = pa.array([mkmap.get(k) for k in t.column("inchikey14").to_pylist()], pa.string())
        t = t.set_column(t.schema.get_field_index("mkey"), "mkey", mkc)
        tmp = f.with_suffix(".tmp")
        pq.write_table(t, tmp, compression="zstd", row_group_size=50000)
        tmp.replace(f)
    lg.info("keys done in %.0f s: %d structures, %d new by ik14, %d new by mkey; in NP tier %d, in PubChem %d",
            time.perf_counter() - t0, len(sm), int((~sm["in_train"]).sum()),
            int((~sm["mkey_in_train"]).sum()), int(sm["in_np"].sum()), int(sm["in_pubchem"].sum()))
    return {"n_structures": int(len(sm))}


# --------------------------------------------------------------------------------------
# library
# --------------------------------------------------------------------------------------

def stage_library(cfg: Config, lg, scratch: Path) -> dict:
    from casmi.library import Library

    files = final_files()
    out = extlib.extlib_dir() / "library"
    lib = Library.build([str(f) for f in files], cfg, out_dir=out, precleaned=True)
    lg.info("ext library: %s", {k: lib.info[k] for k in ("n_spectra", "n_structures", "n_consensus", "n_peaks", "drops")})
    # smoke build with precleaned=False on a small slice of every file
    sc = paths.ensure_dir(scratch / "extlib_smoke")
    parts = []
    for f in files:
        t = pq.read_table(f).slice(0, 500)
        p = sc / f.name
        pq.write_table(t, p)
        parts.append(p)
    small = Library.build(sc, cfg, out_dir=sc / "library", precleaned=False)
    lg.info("smoke build (precleaned=False) from %d files: %s", len(parts),
            {k: small.info[k] for k in ("n_spectra", "n_structures", "n_consensus", "drops")})
    size = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    info = {"library": {k: lib.info[k] for k in ("n_spectra", "n_structures", "n_consensus", "n_peaks", "drops")},
            "bytes": size, "library_info_sha256": _sha256(out / "library_info.json"),
            "smoke": {k: small.info[k] for k in ("n_spectra", "n_structures", "n_consensus", "drops")}}
    (extlib.extlib_dir() / "library_build_info.json").write_text(json.dumps(info, indent=2))
    return info


# --------------------------------------------------------------------------------------
# helpers for calibrate / evaluate
# --------------------------------------------------------------------------------------

def _window_rows(masses: np.ndarray, struct_mass: np.ndarray, cfg: Config) -> np.ndarray:
    """Boolean over struct_mass: within the search window of any of ``masses``."""
    m = np.sort(np.asarray(masses, np.float64))
    if m.size == 0:
        return np.zeros(len(struct_mass), bool)
    tol = np.maximum(cfg.mass_tol_ppm * 1e-6 * struct_mass, cfg.mass_tol_min_da) * 1.5
    lo = np.searchsorted(m, struct_mass - tol, side="left")
    hi = np.searchsorted(m, struct_mass + tol, side="right")
    return hi > lo


def _train_rows(sel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Rows of WP6 spectra files by (ingest_lib, file_row)."""
    from casmi.validation import _lib_file

    parts = []
    for lib, g in sel.groupby("ingest_lib"):
        f = train_dir() / "spectra" / _lib_file(lib)
        pf = pq.ParquetFile(f)
        rows = np.sort(g["file_row"].to_numpy(np.int64))
        base = 0
        for rg in range(pf.num_row_groups):
            n = pf.metadata.row_group(rg).num_rows
            a, b = np.searchsorted(rows, [base, base + n])
            if b > a:
                t = pf.read_row_group(rg, columns=cols).take(pa.array(rows[a:b] - base))
                parts.append(t)
            base += n
    return pa.concat_tables(parts).to_pandas() if parts else pd.DataFrame(columns=cols)


TRAIN_LIB_COLS = ["spectrum_id", "inchikey14", "normalized_smiles", "molecular_formula", "ionization_mode",
                  "instrument_type", "ingest_lib", "adduct", "precursor_mz", "ms2_mzs",
                  "ms2_normalized_intensities", "collision_energy_ev", "base_peak_intensity",
                  "precursor_error_ppm", "mono_mass", "instrument_family", "n_peaks_raw", "prec_intensity"]


_LIB_TYPES = {"ms2_mzs": pa.list_(pa.float32()), "ms2_normalized_intensities": pa.list_(pa.float32()),
              "collision_energy_ev": pa.list_(pa.float64()), "precursor_mz": pa.float64(),
              "base_peak_intensity": pa.float64(), "precursor_error_ppm": pa.float64(),
              "mono_mass": pa.float64(), "n_peaks_raw": pa.float64(), "prec_intensity": pa.float64()}


def _build_small_library(frames: list[pd.DataFrame], cfg: Config, out: Path):
    from casmi.library import Library

    schema = pa.schema([(c, _LIB_TYPES.get(c, pa.string())) for c in TRAIN_LIB_COLS])
    tabs = []
    for df in frames:
        df = df.reindex(columns=TRAIN_LIB_COLS).copy()
        for c in ("ms2_mzs", "ms2_normalized_intensities"):
            df[c] = [np.asarray(x, np.float32) for x in df[c]]
        df["collision_energy_ev"] = [None if x is None or (isinstance(x, float) and math.isnan(x))
                                     else np.asarray(x, np.float64) for x in df["collision_energy_ev"]]
        for c in ("n_peaks_raw", "prec_intensity"):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float64)
        tabs.append(pa.Table.from_pandas(df, schema=schema, preserve_index=False))
    tab = pa.concat_tables(tabs)
    return Library.build(tab, cfg, out_dir=out, precleaned=True)


def _queries_from_rows(df: pd.DataFrame, mol_col: str) -> pd.DataFrame:
    """Test-shaped query rows from cleaned library-format rows."""
    q = pd.DataFrame({
        "molecule_id": df[mol_col].astype(str).to_numpy(),
        "spectrum_id": df["spectrum_id"].astype(str).to_numpy(),
        "ms2_mzs": [np.asarray(x, np.float64) for x in df["ms2_mzs"]],
        "ms2_normalized_intensities": [np.asarray(x, np.float64) for x in df["ms2_normalized_intensities"]],
        "base_peak_intensity": df.get("base_peak_intensity", np.nan),
        "adduct": df["adduct"].to_numpy(), "ionization_mode": df["ionization_mode"].to_numpy(),
        "instrument_type": df["instrument_type"].fillna("unknown").to_numpy(),
        "precursor_mz": df["precursor_mz"].to_numpy(np.float64),
        "collision_energy_ev": df["collision_energy_ev"].to_numpy(),
    })
    return q


def _molecule_mass(q: pd.DataFrame) -> float:
    from casmi.adducts import parse_adduct

    ms = []
    for p, a, m in zip(q["precursor_mz"], q["adduct"], q["ionization_mode"]):
        ad = parse_adduct(a, m)
        if ad is not None:
            ms.append(ad.neutral_mass(float(p)))
    return float(np.median(ms)) if ms else float("nan")


# --------------------------------------------------------------------------------------
# calibrate
# --------------------------------------------------------------------------------------

def _threshold_table(res: pd.DataFrame, cfg: Config) -> dict:
    """Per min-matched k: lowest cosine c with precision >= target among queries whose top-1
    has cos >= c and n_matched >= k (at least extlib_calib_min_hits accepted)."""
    target = float(cfg.extlib_calib_precision)
    out = {"n_queries": int(len(res)), "top1_correct_rate": float(res["correct"].mean()) if len(res) else float("nan"),
           "by_k": {}}
    best = None
    grid = np.round(np.arange(0.30, 1.0001, 0.01), 2)
    for k in (1, 2, 3, 4, 5, 6, 8, 10):
        rk = res[res["n_matched"] >= k]
        chosen = None
        for c in grid:
            acc = rk[rk["cos"] >= c]
            if len(acc) < int(cfg.extlib_calib_min_hits):
                break
            prec = float(acc["correct"].mean())
            if prec >= target:
                chosen = {"cos": float(c), "precision": prec, "accepted": int(len(acc)),
                          "recall": float(acc["correct"].sum() / max(len(res), 1))}
                break
        out["by_k"][str(k)] = chosen
        if chosen and (best is None or chosen["recall"] > best["recall"]):
            best = {"min_matched": k, **chosen}
    out["best"] = best
    return out


def stage_calibrate(cfg: Config, lg, scratch: Path, sources: list[str] | None = None) -> dict:
    t0 = time.perf_counter()
    meta = pq.read_table(train_dir() / "meta.parquet", columns=[
        "spectrum_id", "ingest_lib", "file_row", "ik14", "instrument_family", "query_ok", "mono_mass"]).to_pandas()
    tq = meta[meta["query_ok"] & (meta["instrument_family"] == "timstof")]
    del meta
    rng = np.random.default_rng(cfg.seed)
    out = {}
    for s in (sources or _staged_sources()):
        if s == "enveda180":
            out[s] = {"skipped": "same instrument, lab and spectra as train enveda-180; use the train timsTOF thresholds"}
            continue
        t = _read_staged(s, ["inchikey14", "mono_mass"])
        ik = pd.Series(t["inchikey14"].to_numpy(zero_copy_only=False))
        shared = sorted(set(ik) & set(tq["ik14"]))
        if not shared:
            out[s] = {"n_queries": 0}
            continue
        pick = sorted(rng.choice(shared, size=min(len(shared), int(cfg.extlib_calib_max_molecules)), replace=False))
        qmeta = tq[tq["ik14"].isin(set(pick))].sort_values(["ik14", "spectrum_id"], kind="mergesort")
        qmeta = qmeta.groupby("ik14", sort=True).head(3)
        qrows = _train_rows(qmeta, TRAIN_LIB_COLS)
        # library of the source: staged spectra of structures near the query masses
        smass = t["mono_mass"].to_numpy(zero_copy_only=False)
        near = _window_rows(qmeta.groupby("ik14")["mono_mass"].first().to_numpy(), smass, cfg)
        rows = np.flatnonzero(near)
        lib_rows = _read_staged_rows(s, rows)
        lib = _build_small_library([lib_rows], cfg, scratch / f"calib_{s}")
        res = []
        for ikey, g in qrows.groupby("inchikey14", sort=True):
            q = _queries_from_rows(g, "inchikey14")
            M = _molecule_mass(q)
            r = lib.search_by_mass(M, queries=q)
            if not len(r) or not np.isfinite(r["cos_max"].iloc[0]):
                res.append({"ik14": ikey, "cos": 0.0, "n_matched": 0, "correct": False, "n_cand": len(r)})
                continue
            top = r.iloc[0]
            res.append({"ik14": ikey, "cos": float(top["cos_max"]), "n_matched": int(top["n_matched_best"]),
                        "correct": top["ik14"] == ikey, "n_cand": len(r)})
        res = pd.DataFrame(res)
        out[s] = _threshold_table(res, cfg)
        out[s]["n_library_spectra"] = int(lib.n_spectra)
        out[s]["median_candidates"] = float(res["n_cand"].median()) if len(res) else float("nan")
        lg.info("calibration %s: %s", s, json.dumps(out[s])[:600])
        shutil.rmtree(scratch / f"calib_{s}", ignore_errors=True)
    (extlib.extlib_dir() / "calibration.json").write_text(json.dumps(out, indent=2))
    lg.info("calibration done in %.0f s", time.perf_counter() - t0)
    return out


def _read_staged_rows(source: str, rows: np.ndarray) -> pd.DataFrame:
    rows = np.unique(np.asarray(rows, np.int64))
    tabs = []
    base = 0
    for f in _staged_files(source):
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            n = pf.metadata.row_group(rg).num_rows
            a, b = np.searchsorted(rows, [base, base + n])
            if b > a:
                tabs.append(pf.read_row_group(rg, columns=TRAIN_LIB_COLS).take(pa.array(rows[a:b] - base)))
            base += n
    return pa.concat_tables(tabs).to_pandas() if tabs else pd.DataFrame(columns=TRAIN_LIB_COLS)


# --------------------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------------------

def _rr(res: pd.DataFrame, truth_mkey: str, mk: dict) -> tuple[float, int]:
    if not len(res):
        return 0.0, 0
    keys = [mk.get(i) for i in res["ik14"]]
    seen = []
    for k in keys:
        if k in seen:
            continue
        seen.append(k)
        if k == truth_mkey:
            r = len(seen)
            return (1.0 / r if r <= 25 else 0.0), r
    return 0.0, 0


def _independent(extra: pd.DataFrame, trows: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Boolean over ``extra`` rows: not near-identical (cosine >= extlib_dup_cos, same ik14 and
    adduct) to any train spectrum of that structure (the train spectra the queries come from)."""
    ok = np.ones(len(extra), bool)
    if not len(extra):
        return ok
    tk = (trows["inchikey14"] + "|" + trows["adduct"]).to_numpy(object)
    groups = pd.Series(np.arange(len(trows))).groupby(tk).apply(list).to_dict()
    ek = (extra["inchikey14"] + "|" + extra["adduct"]).to_numpy(object)
    tmz, tint = trows["ms2_mzs"].to_numpy(object), trows["ms2_normalized_intensities"].to_numpy(object)
    emz, eint = extra["ms2_mzs"].to_numpy(object), extra["ms2_normalized_intensities"].to_numpy(object)
    for i, k in enumerate(ek):
        g = groups.get(k)
        if not g:
            continue
        r_mz, r_int, r_off = extlib._flat([tmz[j] for j in g], [tint[j] for j in g])
        q = np.asarray(emz[i], np.float32)
        cos, _ = extlib._cos_pairs(q, np.asarray(eint[i], np.float32), np.array([0, len(q)], np.int64),
                                   r_mz, r_int, r_off, np.zeros(len(g), np.int64), np.arange(len(g)),
                                   float(cfg.extlib_dup_tol_da))
        if cos.size and float(cos.max()) >= float(cfg.extlib_dup_cos):
            ok[i] = False
    return ok


def _eval_set(set_name: str, truth: pd.DataFrame, queries: pd.DataFrame, hide: str, cfg: Config,
              ext_files: list[Path], mk: dict, lg, scratch: Path, extra: pd.DataFrame | None = None) -> dict:
    """Library-only MRR with and without the external library for the molecules in ``truth``.

    Candidates: structures of the library within the mass window (search_by_mass, cos_max
    order), deduplicated by mkey. ``hide``: "all_train" removes every train spectrum of the
    molecule's ik14 group, "query_lib" only those of its query library (C1). ``extra``: staged
    external spectra of the query molecules; only those independent of the molecule's train
    spectra (``_independent``) are added, and ``truth`` is restricted to molecules with one.
    """
    tmeta = pq.read_table(train_dir() / "meta.parquet", columns=[
        "spectrum_id", "ingest_lib", "file_row", "ik14", "mono_mass"]).to_pandas()
    masses = truth["mono_mass"].to_numpy(np.float64)
    st = tmeta.groupby("ik14")["mono_mass"].first()
    near_ik = set(st.index[_window_rows(masses, st.to_numpy(np.float64), cfg)])
    tsel = tmeta[tmeta["ik14"].isin(near_ik)]
    del tmeta
    trows = _train_rows(tsel, TRAIN_LIB_COLS)
    lg.info("%s: %d molecules, train reference spectra in windows: %d of %d structures",
            set_name, len(truth), len(trows), len(near_ik))
    erows = []
    for f in ext_files:
        t = pq.read_table(f, columns=TRAIN_LIB_COLS)
        keep = _window_rows(masses, t["mono_mass"].to_numpy(zero_copy_only=False), cfg)
        if keep.any():
            erows.append(t.filter(pa.array(keep)).to_pandas())
    ext = pd.concat(erows, ignore_index=True) if erows else pd.DataFrame(columns=TRAIN_LIB_COLS)
    info: dict[str, Any] = {}
    if extra is not None:
        ind = _independent(extra, trows, cfg)
        extra = extra[ind]
        info = {"extra_staged_rows": int(len(ind)), "extra_independent_rows": int(ind.sum())}
        groups = truth["ik14_group"].astype(str).str.split(";")
        has = groups.apply(lambda g: bool(set(g) & set(extra["inchikey14"])))
        truth = truth[has.to_numpy()].reset_index(drop=True)
        queries = queries[queries["molecule_id"].isin(set(truth["molecule_id"]))]
        ext = pd.concat([ext, extra], ignore_index=True)
        info["n_molecules"] = int(len(truth))
    lib_t = _build_small_library([trows], cfg, scratch / f"eval_{set_name}_train")
    lib_te = _build_small_library([trows, ext], cfg, scratch / f"eval_{set_name}_both")
    out: dict[str, Any] = dict(info)
    for name, lib in (("train_only", lib_t), ("train_plus_ext", lib_te)):
        meta = lib.meta
        is_ext = meta["ingest_lib"].astype(str).str.startswith("ext_").to_numpy()
        spec_ik = meta["ik14"].astype(str).to_numpy()
        spec_lib = meta["ingest_lib"].astype(str).to_numpy()
        rrs, top1, by_mol = [], 0, []
        for r in truth.itertuples(index=False):
            group = set(str(r.ik14_group).split(";")) if isinstance(r.ik14_group, str) else {r.ik14}
            in_group = np.isin(spec_ik, list(group))
            if hide == "all_train":
                mask = in_group & ~is_ext
            else:   # C1: the query library's own spectra
                mask = in_group & (spec_lib == r.query_lib)
            q = queries[queries["molecule_id"] == r.molecule_id]
            res = lib.search_by_mass(_molecule_mass(q), queries=q, exclude_mask=mask)
            rr, rank = _rr(res, r.mkey, mk)
            rrs.append(rr)
            top1 += int(rank == 1)
            by_mol.append((r.molecule_id, rr))
        out[name] = {"n": len(rrs), "mrr": float(np.mean(rrs)) if rrs else float("nan"),
                     "top1": top1 / max(len(rrs), 1), "rr": rrs}
    for d in ("eval_%s_train" % set_name, "eval_%s_both" % set_name):
        shutil.rmtree(scratch / d, ignore_errors=True)
    lg.info("%s (%s hidden): %s", set_name, hide, {k: ({kk: vv for kk, vv in v.items() if kk != "rr"}
                                                       if isinstance(v, dict) else v) for k, v in out.items()})
    return out


def _eval_chunked(set_name: str, truth: pd.DataFrame, queries: pd.DataFrame, hide: str, cfg: Config,
                  ext_files: list[Path], mk: dict, lg, scratch: Path, extra: pd.DataFrame | None = None,
                  chunk: int = 400) -> dict:
    """``_eval_set`` over mass-sorted chunks of molecules (bounded memory), pooled."""
    truth = truth.sort_values(["mono_mass", "molecule_id"], kind="mergesort").reset_index(drop=True)
    acc: dict[str, list] = {"train_only": [], "train_plus_ext": []}
    extra_info: Counter = Counter()
    for i in range(0, len(truth), chunk):
        t = truth.iloc[i:i + chunk]
        q = queries[queries["molecule_id"].isin(set(t["molecule_id"]))]
        e = None
        if extra is not None:
            grp = set(";".join(t["ik14_group"].astype(str)).split(";"))
            e = extra[extra["inchikey14"].isin(grp)]
        r = _eval_set(set_name, t, q, hide, cfg, ext_files, mk, lg, scratch, extra=e)
        for k in acc:
            acc[k].extend(r[k]["rr"])
        for k, v in r.items():
            if not isinstance(v, dict):
                extra_info[k] += v
    out: dict[str, Any] = dict(extra_info)
    for k, rr in acc.items():
        a = np.asarray(rr, np.float64)
        out[k] = {"n": int(a.size), "mrr": float(a.mean()) if a.size else float("nan"),
                  "top1": float((a == 1.0).mean()) if a.size else float("nan")}
    lg.info("%s pooled (%s hidden): %s", set_name, hide, out)
    return out


def _staged_rows_for(ik14: set) -> pd.DataFrame:
    """Staged (filtered, not deduplicated) spectra of the given structures, all sources."""
    parts = []
    for s in _staged_sources():
        t = _read_staged(s, ["inchikey14"])
        rows = np.flatnonzero(_isin(t["inchikey14"], pa.array(sorted(ik14))))
        if rows.size:
            parts.append(_read_staged_rows(s, rows))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=TRAIN_LIB_COLS)


def stage_evaluate(cfg: Config, lg, scratch: Path) -> dict:
    from casmi.validation import load_queries

    vdir = paths.PROCESSED / "validation"
    ext_files = final_files()
    es = pd.read_parquet(extlib.extlib_dir() / "structures.parquet")
    st = pq.read_table(train_dir() / "structures.parquet", columns=["ik14", "mkey"]).to_pandas()
    mk = dict(zip(st["ik14"], st["mkey"]))
    mk.update(dict(zip(es["ik14"], es["mkey"])))
    ext_mkeys = set(es["mkey"].dropna())
    out = {}
    q, truth = load_queries("NP-OTHER", vdir)
    # (a) the final, deduplicated library: molecules whose mkey has spectra in it
    sel = truth[truth["mkey"].isin(ext_mkeys)].reset_index(drop=True)
    out["NP-OTHER_final_ext"] = _eval_chunked("NP-OTHER", sel, q[q["molecule_id"].isin(set(sel["molecule_id"]))],
                                          "all_train", cfg, ext_files, mk, lg, scratch)
    out["NP-OTHER_final_ext"]["n_molecules_with_ext"] = int(len(sel))
    # (b) every staged external spectrum of the molecule that is independent of its train
    # spectra (the deduplication keeps only new instrument/adduct coverage of train
    # structures; for a structure absent from train all its public spectra would be kept)
    groups = set(";".join(truth["ik14_group"].astype(str)).split(";"))
    extra = _staged_rows_for(groups)
    out["NP-OTHER_independent_ext"] = _eval_chunked("NP-OTHER", truth, q, "all_train", cfg, ext_files, mk, lg,
                                                scratch, extra=extra)
    # NP-TIMS C1 (query library hidden) with and without the external library
    q, truth = load_queries("NP-TIMS", vdir)
    out["NP-TIMS_C1"] = _eval_chunked("NP-TIMS", truth, q, "query_lib", cfg, ext_files, mk, lg, scratch)
    (extlib.extlib_dir() / "evaluation.json").write_text(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------------------
# C2-ZEN queries
# --------------------------------------------------------------------------------------

def stage_c2zen(cfg: Config, lg, max_molecules: int = 1500) -> dict:
    """C2-ZEN (ROADMAP 2.3 item 3): Enveda-180 Zenodo structures absent from train (by mkey),
    queries emulated like ``validation.build_queries`` (k spectra per molecule from the public
    test distribution, CE-class weighting, test cleaning of the RAW peaks, test adducts only,
    base peak >= val_min_base_peak_intensity, |ppm| <= val_query_max_abs_ppm). At most
    ``max_molecules`` molecules (hash order), folds 0-4 by hash of the mkey."""
    from casmi.adducts import TEST_ADDUCTS
    from casmi.io import TEST_COLUMNS, read_test
    from casmi.splits import hash_uniform
    from casmi.validation import TRUTH_COLUMNS, _choose, _test_clean, _write_queries, set_slug, test_reference

    set_name = "C2-ZEN"
    ed = extlib.extlib_dir()
    raw = pq.read_table(ed / "c2zen_raw.parquet").to_pandas()
    cols = ["spectrum_id", "inchikey14", "normalized_smiles", "molecular_formula", "mono_mass", "adduct",
            "ionization_mode", "instrument_type", "precursor_mz", "precursor_error_ppm", "base_peak_intensity",
            "collision_energy_ev", "collision_energy_orig", "collision_energy_orig_units", "mkey", "in_train"]
    lib = pq.read_table(ed / "extlib_enveda180.parquet", columns=cols).to_pandas()
    es = pd.read_parquet(ed / "structures.parquet")
    new_mk = set(es.loc[~es["mkey_in_train"], "mkey"].dropna())
    n_struct = int(lib.loc[~lib["in_train"], "inchikey14"].nunique())
    lib = lib[(~lib["in_train"]) & lib["mkey"].isin(new_mk) & lib["adduct"].isin(set(TEST_ADDUCTS))]
    lib = lib[~(lib["base_peak_intensity"] < cfg.val_min_base_peak_intensity)
              & (lib["precursor_error_ppm"] <= cfg.val_query_max_abs_ppm)]
    lib = lib.merge(raw[["spectrum_id", "raw_mzs", "raw_intensities"]], on="spectrum_id", how="inner")
    lib["collision_energy"] = [float(np.mean(x)) if x is not None and len(x) else np.nan
                               for x in lib["collision_energy_ev"]]
    lib["n_energies"] = [len(x) if x is not None else 0 for x in lib["collision_energy_ev"]]
    lib["instrument_family"] = "timstof"
    mol = (lib.groupby("mkey").agg(ik14=("inchikey14", "first"), n=("spectrum_id", "size"),
                                  mono_mass=("mono_mass", "first"), formula=("molecular_formula", "first"))
           .reset_index())
    u = hash_uniform(mol["mkey"], cfg.seed, "c2zen")
    mol = mol.iloc[np.lexsort((mol["mkey"].astype(str).to_numpy(), u))].head(max_molecules).reset_index(drop=True)
    mol["molecule_id"] = [f"ZE{i:05d}" for i in range(len(mol))]
    mol["fold"] = np.minimum((hash_uniform(mol["mkey"], cfg.seed, "fold") * cfg.val_n_folds).astype(int),
                             cfg.val_n_folds - 1)
    ref = test_reference(read_test(None))
    by_mk = {k: g.sort_values("spectrum_id", kind="mergesort").reset_index(drop=True)
             for k, g in lib[lib["mkey"].isin(set(mol["mkey"]))].groupby("mkey")}
    qrows = []
    for r in mol.itertuples(index=False):
        c = by_mk[r.mkey]
        for p in c.iloc[_choose(c, ref, cfg, set_name, r.mkey)].itertuples(index=False):
            mz, it = _test_clean(p.raw_mzs, p.raw_intensities, float(p.precursor_mz), cfg.val_query_peak_cut_da)
            if not mz:
                continue
            ev = p.collision_energy_ev
            qrows.append({"molecule_id": r.molecule_id, "spectrum_id": p.spectrum_id, "ms2_mzs": mz,
                          "ms2_normalized_intensities": it,
                          "base_peak_intensity": float(p.base_peak_intensity),
                          "adduct": p.adduct, "ionization_mode": p.ionization_mode,
                          "instrument_type": p.instrument_type or "timsTOF", "precursor_mz": float(p.precursor_mz),
                          "collision_energy_orig": p.collision_energy_orig,
                          "collision_energy_ev": None if ev is None else [float(x) for x in ev],
                          "collision_energy_orig_units": p.collision_energy_orig_units})
    q = pd.DataFrame(qrows, columns=list(TEST_COLUMNS))
    mol["n_query_spectra"] = mol["molecule_id"].map(q.groupby("molecule_id").size()).fillna(0).astype(int)
    mol = mol[mol["n_query_spectra"] > 0].reset_index(drop=True)
    info = es.drop_duplicates("mkey").set_index("mkey").reindex(mol["mkey"])
    mass = mol["mono_mass"].to_numpy(np.float64)
    terc = np.full(len(mol), -1)
    if len(mol) >= 3:
        terc = np.searchsorted(np.quantile(mass, [1 / 3, 2 / 3]), mass, side="right")
    truth = pd.DataFrame({
        "molecule_id": mol["molecule_id"], "set": set_name, "fold": mol["fold"].astype(int),
        "mkey": mol["mkey"], "ik14": mol["ik14"], "ik14_group": mol["ik14"],
        "smiles": info["smiles"].to_numpy(object), "formula": mol["formula"].to_numpy(object), "mono_mass": mass,
        "eligible_c1": False, "query_lib": "ext_enveda180", "n_query_spectra": mol["n_query_spectra"],
        "n_available": mol["n"].astype(int).to_numpy(),
        "pubmed_count": pd.array([pd.NA] * len(mol), dtype="Int32"),
        "in_np": info["in_np"].fillna(False).astype(bool).to_numpy(),
        "in_pubchem": info["in_pubchem"].fillna(False).astype(bool).to_numpy(),
        "np_score": np.nan, "mass_tercile": terc.astype(int), "peaks_source": "raw",
    }, columns=list(TRUTH_COLUMNS))
    q = (q[q["molecule_id"].isin(set(truth["molecule_id"]))]
         .sort_values(["molecule_id", "spectrum_id"], kind="mergesort").reset_index(drop=True))
    vdir = paths.ensure_dir(paths.PROCESSED / "validation")
    qp, tp = vdir / f"queries_{set_slug(set_name)}.parquet", vdir / f"truth_{set_slug(set_name)}.parquet"
    _write_queries(q, qp)
    truth.to_parquet(tp, index=False)
    res = {"enveda180_structures_absent_from_train_ik14": n_struct,
           "eligible_molecules_new_mkey": int(lib["mkey"].nunique()), "molecules": int(len(truth)),
           "spectra": int(len(q)), "in_np": int(truth["in_np"].sum()), "in_pubchem": int(truth["in_pubchem"].sum()),
           "spectra_per_molecule_mean": float(len(q) / max(len(truth), 1)),
           "folds": truth["fold"].value_counts().sort_index().to_dict(),
           "queries_sha256": _sha256(qp), "truth_sha256": _sha256(tp)}
    (ed / "c2zen_info.json").write_text(json.dumps(res, indent=2))
    lg.info("C2-ZEN: %s", res)
    return res


# --------------------------------------------------------------------------------------
# holdout n_ext_lib
# --------------------------------------------------------------------------------------

def stage_holdout(lg) -> dict:
    hold = pd.read_csv(paths.SPLITS / "holdout_v1.csv", dtype=str)
    st = pq.read_table(train_dir() / "structures.parquet", columns=["ik14", "mkey"]).to_pandas()
    mk = dict(zip(st["ik14"], st["mkey"]))
    # staged (pre-dedup, post-filter) spectra per (ik14, source): public spectra outside train
    acc = []
    for s in _staged_sources():
        t = _read_staged(s, ["inchikey14"]).to_pandas()
        c = t.groupby("inchikey14").size().rename("n").reset_index()
        c["source"] = s
        acc.append(c)
    allc = pd.concat(acc, ignore_index=True)
    allc["mkey"] = allc["inchikey14"].map(mk)
    fin = []
    for f in final_files():
        t = pq.read_table(f, columns=["inchikey14", "mkey", "ingest_lib"]).to_pandas()
        fin.append(t)
    fin = pd.concat(fin, ignore_index=True) if fin else pd.DataFrame(columns=["inchikey14", "mkey", "ingest_lib"])
    rows = []
    by_mkey_staged = allc.dropna(subset=["mkey"]).groupby("mkey")
    st_src = by_mkey_staged["source"].agg(lambda x: sorted(set(x)))
    st_n = by_mkey_staged["n"].sum()
    fin_src = fin.dropna(subset=["mkey"]).groupby("mkey")["ingest_lib"].agg(lambda x: sorted(set(x)))
    fin_n = fin.dropna(subset=["mkey"]).groupby("mkey").size()
    out = hold[["ik14", "mkey", "set"]].copy()
    out["ext_sources_any"] = out["mkey"].map(st_src).apply(lambda x: ";".join(x) if isinstance(x, list) else "")
    out["n_ext_lib_any"] = out["mkey"].map(st_src).apply(lambda x: len(x) if isinstance(x, list) else 0)
    out["n_ext_spectra_any"] = out["mkey"].map(st_n).fillna(0).astype(int)
    out["n_ext_lib"] = out["mkey"].map(fin_src).apply(lambda x: len(x) if isinstance(x, list) else 0)
    out["n_ext_spectra"] = out["mkey"].map(fin_n).fillna(0).astype(int)
    out.to_parquet(extlib.extlib_dir() / "holdout_ext.parquet", index=False)
    res = out.groupby("set").agg(n=("ik14", "size"), any_ext=("n_ext_lib_any", lambda x: int((x > 0).sum())),
                                 final_ext=("n_ext_lib", lambda x: int((x > 0).sum()))).to_dict("index")
    lg.info("holdout ext coverage: %s", res)
    return res


# --------------------------------------------------------------------------------------
# cross-check with public dumps, report
# --------------------------------------------------------------------------------------

def stage_crosscheck(lg) -> dict:
    es = pd.read_parquet(extlib.extlib_dir() / "structures.parquet")
    ours = set(es["ik14"])
    ours_new = set(es.loc[~es["in_train"], "ik14"])
    tr = train_ik14_set()
    out = {}
    dumps = {
        "taeyangg4/casmi26-extlib-gnps-full": RAW_ROOT / "kaggle_taeyangg4" / "ext_meta.parquet.dl",
        "takumuhata/casmi26-extlib": RAW_ROOT / "kaggle_takumuhata" / "ext_meta.parquet.dl",
    }
    sam = RAW_ROOT / "kaggle_samartalwar" / "spectra.parquet.dl"
    if sam.exists():
        d = pd.read_parquet(sam, columns=["spectrum_id", "inchikey", "is_casmi_subset"])
        d = d[~d["is_casmi_subset"].astype(bool)]
        theirs = set(d["inchikey"].dropna().str[:14])
        mb = _read_staged("massbank", ["source_id"]).column("source_id").to_pylist() if "massbank" in _staged_sources() else []
        out["samartalwar/casmi-2026-spectral-library-massbankharmonized"] = {
            "spectra": int(len(d)), "structures": len(theirs), "absent_from_train": len(theirs - tr),
            "absent_from_train_and_ours": len(theirs - tr - ours),
            "accessions_not_in_our_massbank_2026_11_staged": int(len(set(d["spectrum_id"]) - set(mb)))}
    for name, p in dumps.items():
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["inchikey14"])
        theirs = set(d["inchikey14"])
        out[name] = {"spectra": int(len(d)), "structures": len(theirs),
                     "absent_from_train": len(theirs - tr),
                     "absent_from_train_and_ours": len(theirs - tr - ours),
                     "ours_new_not_in_dump": len(ours_new - theirs)}
    lg.info("crosscheck: %s", out)
    (extlib.extlib_dir() / "crosscheck.json").write_text(json.dumps(out, indent=2))
    return out


def stage_report(lg) -> dict:
    ed = extlib.extlib_dir()
    es = pd.read_parquet(ed / "structures.parquet")
    fin = json.loads((ed / "finalize_info.json").read_text())
    rep = {"sources": {}, "total": {}}
    for s in fin["sources"]:
        si = json.loads((extlib.staging_dir(s) / "stage_info.json").read_text())
        f = ed / f"extlib_{s}.parquet"
        t = pq.read_table(f, columns=["inchikey14", "mkey", "in_train", "instrument_family", "licence"]).to_pandas()
        e = es.set_index("ik14").reindex(t["inchikey14"].unique())
        new = e[~e["in_train"]]
        new_mk = e[~e["mkey_in_train"]]
        rep["sources"][s] = {
            "spectra_in": si["totals"].get("n_in", 0), "after_filters": si["totals"].get("n_staged", 0),
            "after_dedup": int(len(t)), "structures": int(len(e)),
            "new_ik14": int(len(new)), "new_mkey": int(new_mk["mkey"].nunique()),
            "new_in_np": int(new["in_np"].sum()), "new_in_pubchem": int(new["in_pubchem"].sum()),
            "instruments": t["instrument_family"].value_counts().to_dict(),
            "licences": t["licence"].value_counts().head(6).to_dict(),
            "drops": {k.replace("drop_", ""): v for k, v in si["totals"].items() if k.startswith("drop_") and k != "drop_staged"},
            "dedup": fin["counts"][s],
        }
    new = es[~es["in_train"]]
    rep["total"] = {"structures": int(len(es)), "new_ik14": int(len(new)),
                    "new_mkey": int(es.loc[~es["mkey_in_train"], "mkey"].nunique()),
                    "new_in_np": int(new["in_np"].sum()), "new_in_pubchem": int(new["in_pubchem"].sum()),
                    "spectra": int(sum(v["after_dedup"] for v in rep["sources"].values()))}
    (ed / "report.json").write_text(json.dumps(rep, indent=2))
    lines = ["| source | in | after filters | after dedup | structures | new ik14 | new mkey | new in NP | new in PubChem | instruments |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for s, v in rep["sources"].items():
        inst = ", ".join(f"{k} {n:,}" for k, n in v["instruments"].items())
        lines.append(f"| {s} | {v['spectra_in']:,} | {v['after_filters']:,} | {v['after_dedup']:,} | {v['structures']:,} | "
                     f"{v['new_ik14']:,} | {v['new_mkey']:,} | {v['new_in_np']:,} | {v['new_in_pubchem']:,} | {inst} |")
    t = rep["total"]
    lines.append(f"| total | | | {t['spectra']:,} | {t['structures']:,} | {t['new_ik14']:,} | {t['new_mkey']:,} | "
                 f"{t['new_in_np']:,} | {t['new_in_pubchem']:,} | |")
    print("\n".join(lines))
    return rep


# --------------------------------------------------------------------------------------

STAGES = ("fetch", "stage", "finalize", "keys", "library", "calibrate", "evaluate", "c2zen",
          "holdout", "crosscheck", "report")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True, choices=STAGES + ("all",))
    ap.add_argument("--source", action="append", default=None, help="source(s) for fetch/stage/calibrate")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--keep-raw-mb", type=float, default=300.0,
                    help="raw downloads larger than this are deleted after staging")
    ap.add_argument("--scratch", type=Path, default=None, help="scratch dir for test builds")
    ap.add_argument("--resume", action="store_true", help="stage: keep finished GNPS parts")
    args = ap.parse_args(argv)
    lg = get_logger("build_extlib")
    log_run(lg, args)
    cfg = load_config(args.config)
    scratch = paths.ensure_dir(args.scratch or (extlib.extlib_dir() / "scratch"))
    sources = args.source or list(extlib.SOURCE_ORDER)
    stages = STAGES if args.stage == "all" else (args.stage,)
    for st in stages:
        if st == "fetch":
            for s in sources:
                if s not in ("gnps", "gnps_multiplex"):
                    stage_fetch(s, lg)
        elif st == "stage":
            for s in sources:
                stage_source(s, cfg, args.n_jobs, args.keep_raw_mb, lg, resume=args.resume)
        elif st == "finalize":
            stage_finalize(cfg, lg)
        elif st == "keys":
            stage_keys(cfg, args.n_jobs, lg)
        elif st == "library":
            stage_library(cfg, lg, scratch)
        elif st == "calibrate":
            stage_calibrate(cfg, lg, scratch, args.source)
        elif st == "evaluate":
            stage_evaluate(cfg, lg, scratch)
        elif st == "c2zen":
            stage_c2zen(cfg, lg)
        elif st == "holdout":
            stage_holdout(lg)
        elif st == "crosscheck":
            stage_crosscheck(lg)
        elif st == "report":
            stage_report(lg)
    return 0


if __name__ == "__main__":
    sys.exit(main())

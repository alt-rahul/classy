"""WP6: train ingestion, structure table, mkey pass and data audit (PLAN.md section 6, WP6).

Stages (``--stage``, default ``all`` = ingest + audit; ``mkey`` is a separate background job):

``ingest``  Stream ``--input`` (default ``data/raw/train.parquet``) batch by batch with pyarrow
            (never the whole file in memory), count and drop bad rows by reason, clean the
            peaks with ``casmi.spectra.clean_flat`` and write, under ``--out`` (default
            ``data/processed/train``):
            - ``spectra/spectra_<lib>.parquet``: one file per ingest_lib, the 20 train columns
              (peaks replaced by the cleaned float32 lists) plus derived columns (DERIVED).
              They sit in their own subdirectory so that
              ``scripts/build_library.py --source data/processed/train/spectra/`` reads only
              spectra (Library.build globs every parquet file of a directory).
            - ``meta.parquet``: one row per kept spectrum, metadata only (META_COLUMNS), with
              ``file_row`` (row in its spectra file) and ``source_row`` (row in the input).
            - ``structures.parquet``: one row per ik14 (STRUCTURE_COLUMNS): label SMILES, RDKit
              formula and mass (on ``chem.std_mol``), counts per library / instrument family /
              adduct, NP-tier flags and priors, PubChem-tier membership, NP-likeness, element
              and filter (PLAN 2.2) flags; ``mkey`` stays null until the mkey stage.
            - ``structure_lib_counts.parquet``: one row per (ik14, ingest_lib) with the counts
              the splits need (test-adduct and query-usable spectra, per polarity, QTOF).
            - ``public_test_matches.parquet`` / ``public_test_truth.parquet``: public test
              spectra found in train (PLAN 5.7) and the per-molecule label.
            - ``ingest_info.json``: row counts, drop reasons and flags per library, timings,
              rows per second, peak RSS.
``mkey``    Metric key (``casmi.metric.mkeys``: process pool, per-molecule timeout, disk cache
            ``data/processed/cache/mkey.parquet``) of every structure's label SMILES; writes the
            ``mkey`` and ``mkey_status`` columns into structures.parquet. Resumable through the
            cache. Run it in the background: ``nohup nice -n 10 python scripts/ingest_train.py
            --stage mkey --mkey-jobs 4 > logs/mkey.out 2>&1 &``.
``audit``   Writes ``--audit-out`` (default ``docs/DATA_AUDIT.md``) from the files above; rerun
            after the mkey stage to fill the mkey-dependent items.
``library`` Builds the WP5 library index (``--library-out``, default data/processed/library) from
            ``spectra/`` with ``Library.build``, working around an all-null-column concat
            error in ``library._Builder._pass1`` (see ``_permissive_concat``). Equivalent to
            ``scripts/build_library.py --source data/processed/train/spectra/`` once that is fixed.

Drop reasons (first failing check wins, in this order): null_label (no inchikey14 or no
normalized_smiles), bad_precursor (null, NaN or <= 0), bad_peak_arrays (null list or the two
lists differ in length), no_raw_peaks, no_polarity (neither ionization_mode nor the adduct sign
gives one), duplicate_spectrum_id (the first occurrence is kept), empty_after_cleaning.
Kept rows are flagged (counted, not dropped): adduct_unparsed (``adducts.parse_adduct`` fails;
the library keeps them, PLAN 3.1), abs_ppm_gt_<fpm_max_precursor_error_ppm>, peaks_above_precursor
(raw peaks above precursor + ``clean_max_above_precursor_da``), bpi_null, ce_unresolved.

Instrument family: ``instrument_family_of`` (documented below; the same mapping the WP5
library uses, so library, models and splits agree).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from casmi import paths
from casmi.config import Config, load_config
from casmi.log import get_logger, log_run, peak_rss_mb

# docs/COMPETITION.md, "Training libraries": (spectra, unique structures).
EXPECTED: dict[str, tuple[int, int]] = {
    "enveda-180": (1_153_785, 182_941),
    "pluskal_ms2": (527_581, 46_821),
    "riken": (347_171, 15_892),
    "gnps": (220_849, 45_750),
    "massbank": (101_727, 9_180),
    "mona": (92_416, 11_681),
    "spectraverse": (50_933, 9_631),
    "msdial": (40_765, 9_127),
    "drug_plus": (2_545, 2_539),
    "enveda-np-examples": (1_151, 250),
    "masaryk": (652, 416),
}
ENVEDA_LIBS = ("enveda-180", "enveda-np-examples")
NP_HEAVY_LIBS = ("enveda-np-examples", "riken", "gnps", "msdial", "massbank", "mona")
DROP_REASONS = ("null_label", "bad_precursor", "bad_peak_arrays", "no_raw_peaks",
                "no_polarity", "duplicate_spectrum_id", "empty_after_cleaning")
INSTRUMENT_FAMILIES = ("timstof", "qtof", "orbitrap", "other")

TRAIN_COLUMNS = (
    "molecule_id", "spectrum_id", "ms2_mzs", "ms2_normalized_intensities",
    "base_peak_intensity", "adduct", "ionization_mode", "instrument_type", "precursor_mz",
    "collision_energy_orig", "collision_energy_ev", "collision_energy_orig_units",
    "normalized_smiles", "inchikey", "inchikey14", "molecular_formula", "ingest_lib",
    "adduct_orig", "precursor_error_ppm", "num_peaks",
)
# Derived per-spectrum columns written next to the train columns in spectra_<lib>.parquet.
DERIVED = ("source_row", "polarity", "adduct_test", "adduct_known", "instrument_family",
           "mono_mass", "m_spec", "m_err_ppm", "collision_energy", "n_energies", "n_peaks_raw",
           "n_peaks", "n_above_precursor", "prec_intensity", "query_ok")
META_COLUMNS = ("spectrum_id", "molecule_id", "ingest_lib", "file_row", "source_row", "ik14",
                "adduct", "adduct_test", "adduct_known", "multiplier", "charge", "polarity",
                "instrument_type", "instrument_family", "precursor_mz", "m_spec", "mono_mass",
                "m_err_ppm", "precursor_error_ppm", "base_peak_intensity", "collision_energy",
                "n_energies", "ce_units", "num_peaks", "n_peaks_raw", "n_peaks",
                "n_above_precursor", "prec_intensity", "query_ok")


def lib_file_name(lib: str) -> str:
    """``spectra_<lib>.parquet`` with characters outside [A-Za-z0-9._-] replaced by "_"."""
    return "spectra_" + re.sub(r"[^A-Za-z0-9._-]", "_", lib) + ".parquet"


def instrument_family_of(instrument_type: str | None, ingest_lib: str | None = None) -> str:
    """Instrument family of a spectrum from free-text ``instrument_type`` and the library.

    Returns one of "timstof", "qtof" (other QTOF / TOF), "orbitrap", "other" (anything
    else, including unknown / null). Rules, first match wins (case-insensitive):

    1. ingest_lib enveda-180 or enveda-np-examples -> timstof (uniformly timsTOF).
    2. text contains "tims" -> timstof.
    3. Orbitrap family: orbitrap, exactive, lumos, fusion, eclipse, astral, exploris, ITFT,
       QFT, LTQ-FT, hybrid FT, FTMS, FT-ICR, a bare "FT" token -> orbitrap.
    4. TOF family: Q-TOF / QTOF / TOF, maXis, impact, synapt, xevo, TripleTOF, Zeno,
       Agilent 65xx, compact -> qtof.
    5. no match: pluskal_ms2 (MSnLib, Orbitrap) -> orbitrap; else other.

    This is ``casmi.library.instrument_family`` (WP5), re-exported here so that ingestion,
    the library index, the fingerprint model and the splits use one mapping. The audit lists
    every free-text value with the family it maps to.
    """
    from casmi.library import instrument_family

    return instrument_family(instrument_type, ingest_lib)


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------

def _str_np(col: pa.Array | pa.ChunkedArray | None, n: int) -> np.ndarray:
    if col is None:
        return np.full(n, None, dtype=object)
    return np.asarray(col.to_pylist(), dtype=object)


def _f64_np(col, n: int) -> np.ndarray:
    if col is None:
        return np.full(n, np.nan)
    return np.asarray(pc.cast(col, pa.float64()).to_numpy(zero_copy_only=False), dtype=np.float64)


def _list_mean(col) -> tuple[np.ndarray, np.ndarray]:
    """Mean of the finite values per list and their count (NaN / 0 for null or empty)."""
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    n = len(col)
    if n == 0:
        return np.zeros(0), np.zeros(0, np.int16)
    off = np.asarray(col.offsets, dtype=np.int64)
    vals = np.asarray(pc.cast(col.values, pa.float64()).to_numpy(zero_copy_only=False))
    fin = np.isfinite(vals)
    v0 = np.where(fin, vals, 0.0)
    cs = np.concatenate([[0.0], np.cumsum(v0)])
    cc = np.concatenate([[0], np.cumsum(fin)])
    s = cs[off[1:]] - cs[off[:-1]]
    c = cc[off[1:]] - cc[off[:-1]]
    if col.null_count:
        c = np.where(np.asarray(col.is_null().to_numpy(zero_copy_only=False)), 0, c)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    return mean, c.astype(np.int16)


@lru_cache(maxsize=200_000)
def _formula_mass(formula: str | None) -> float:
    from casmi.adducts import formula_mass

    if not isinstance(formula, str) or not formula.strip():
        return float("nan")
    f = re.sub(r"[+-]\d*$", "", formula.strip())
    try:
        return float(formula_mass(f))
    except (ValueError, KeyError):
        return float("nan")


@lru_cache(maxsize=100_000)
def _adduct_info(adduct: str | None, mode: str | None) -> tuple[int, float, int, str | None, bool]:
    """(multiplier, shift, charge, test adduct name or None, parsed) for an adduct string."""
    from casmi.adducts import TEST_ADDUCTS, match_test_adduct, parse_adduct

    a = TEST_ADDUCTS.get(adduct) if isinstance(adduct, str) else None
    if a is None:
        a = parse_adduct(adduct, mode)
    if a is None:
        return 0, float("nan"), 0, None, False
    return a.multiplier, a.shift, a.charge, match_test_adduct(a), True


@lru_cache(maxsize=100_000)
def _polarity(mode: str | None, adduct: str | None) -> int:
    from casmi.library import polarity_sign

    return polarity_sign(mode, adduct)


@lru_cache(maxsize=100_000)
def _family(itype: str | None, lib: str | None) -> str:
    return instrument_family_of(itype, lib)


def _map(func, *cols: np.ndarray) -> np.ndarray:
    out = np.empty(len(cols[0]), dtype=object)
    for i, k in enumerate(zip(*cols)):
        out[i] = func(*k)
    return out


def _float32_lists(values: np.ndarray, offsets: np.ndarray) -> pa.ListArray:
    return pa.ListArray.from_arrays(pa.array(offsets.astype(np.int32)),
                                    pa.array(values.astype(np.float32, copy=False)))


def _rss_now_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


# --------------------------------------------------------------------------------------
# public test join (PLAN 5.7)
# --------------------------------------------------------------------------------------

class TestJoin:
    """Find public test spectra in train while streaming: by spectrum_id when shared, else by
    precursor m/z (within ``prec_tol`` Da) plus peak arrays (at least ``min_frac`` of the test
    peaks have a train peak within ``mz_tol`` Da; same polarity)."""

    def __init__(self, test: pd.DataFrame, prec_tol: float = 1e-4, mz_tol: float = 1e-5,
                 min_frac: float = 0.9):
        self.test = test.reset_index(drop=True)
        self.ids = {s: i for i, s in enumerate(self.test["spectrum_id"].astype(str))}
        p = self.test["precursor_mz"].to_numpy(np.float64)
        self.order = np.argsort(p, kind="mergesort")
        self.sorted_prec = p[self.order]
        self.mz = [np.asarray(x, np.float64) for x in self.test["ms2_mzs"]]
        self.it = [np.asarray(x, np.float64) for x in self.test["ms2_normalized_intensities"]]
        self.pol = np.array([1 if str(m).lower().startswith("pos") else -1
                             for m in self.test["ionization_mode"]])
        self.prec = p
        self.prec_tol, self.mz_tol, self.min_frac = prec_tol, mz_tol, min_frac
        self.rows: list[dict] = []

    def scan(self, sid: np.ndarray, prec: np.ndarray, pol: np.ndarray, mz: np.ndarray,
             it: np.ndarray, off: np.ndarray, info: dict[str, np.ndarray]) -> None:
        pairs: set[tuple[int, int]] = set()
        for r, s in enumerate(sid):
            t = self.ids.get(s) if isinstance(s, str) else None
            if t is not None:
                pairs.add((r, t))
        ok = np.isfinite(prec)
        lo = np.searchsorted(self.sorted_prec, np.where(ok, prec - self.prec_tol, np.inf), "left")
        hi = np.searchsorted(self.sorted_prec, np.where(ok, prec + self.prec_tol, -np.inf), "right")
        for r in np.flatnonzero(hi > lo):
            for k in range(lo[r], hi[r]):
                pairs.add((int(r), int(self.order[k])))
        has_label = np.array([isinstance(k, str) and bool(k) for k in info["ik14"]])
        for r, t in sorted(pairs):
            if not has_label[r]:
                continue
            by_id = sid[r] == self.test["spectrum_id"].iat[t]
            if pol[r] != self.pol[t] and not by_id:
                continue
            rm = mz[off[r]:off[r + 1]]
            ri = it[off[r]:off[r + 1]]
            o = np.argsort(rm, kind="mergesort")
            rm, ri = rm[o], ri[o]
            tm, ti = self.mz[t], self.it[t]
            if rm.size == 0 or tm.size == 0:
                continue
            j = np.clip(np.searchsorted(rm, tm), 1, rm.size - 1) if rm.size > 1 else np.zeros(tm.size, int)
            if rm.size > 1:
                j = np.where(np.abs(rm[j - 1] - tm) < np.abs(rm[j] - tm), j - 1, j)
            hit = np.abs(rm[j] - tm) <= self.mz_tol
            frac_t = float(hit.mean())
            if frac_t < self.min_frac and not by_id:
                continue
            used = np.zeros(rm.size, bool)
            used[j[hit]] = True
            p = prec[r]
            dropped = rm[~used]
            above = dropped[dropped > p] - p
            ratio = ti[hit] / np.maximum(ri[j[hit]], 1e-30)
            rec = {"test_spectrum_id": self.test["spectrum_id"].iat[t],
                   "molecule_id": self.test["molecule_id"].iat[t],
                   "by_id": bool(by_id), "frac_test_matched": frac_t,
                   "frac_train_matched": float(used.mean()), "n_test_peaks": int(tm.size),
                   "n_train_peaks": int(rm.size), "prec_diff": float(self.prec[t] - p),
                   "max_kept_excess": float(np.max(tm - self.prec[t])),
                   "min_dropped_excess": float(above.min()) if above.size else float("nan"),
                   "n_dropped_above": int(above.size),
                   "int_ratio_median": float(np.median(ratio)) if ratio.size else float("nan"),
                   "int_ratio_spread": float(np.ptp(ratio)) if ratio.size else float("nan")}
            for k, v in info.items():
                rec[k] = v[r]
            self.rows.append(rec)

    def result(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        cols = ["test_spectrum_id", "molecule_id", "train_spectrum_id", "source_row",
                "ingest_lib", "ik14", "smiles", "adduct", "precursor_error_ppm", "m_err_ppm",
                "test_m_err_ppm", "by_id", "frac_test_matched", "frac_train_matched",
                "n_test_peaks", "n_train_peaks", "prec_diff", "max_kept_excess",
                "min_dropped_excess", "n_dropped_above", "int_ratio_median",
                "int_ratio_spread", "best"]
        if not self.rows:
            return pd.DataFrame(columns=cols), pd.DataFrame(
                columns=["molecule_id", "ik14", "smiles", "n_spectra", "n_matched", "n_labels",
                         "source_libs", "consistent"])
        m = pd.DataFrame(self.rows)
        m["enveda"] = m["ingest_lib"].isin(ENVEDA_LIBS)
        m = m.sort_values(["test_spectrum_id", "by_id", "frac_test_matched", "frac_train_matched",
                           "enveda", "source_row"],
                          ascending=[True, False, False, False, False, True], kind="mergesort")
        m["best"] = ~m.duplicated("test_spectrum_id", keep="first")
        # M error of the test spectrum against the train label (after organisers' cleaning)
        tmz = dict(zip(self.test["spectrum_id"], self.test["precursor_mz"]))
        tad = dict(zip(self.test["spectrum_id"], self.test["adduct"]))
        tmo = dict(zip(self.test["spectrum_id"], self.test["ionization_mode"]))
        errs = []
        for s, mass in zip(m["test_spectrum_id"], m["mono_mass"]):
            mult, shift, charge, _, okp = _adduct_info(tad[s], tmo[s])
            if okp and np.isfinite(mass):
                ms = (tmz[s] * abs(charge) - shift) / mult
                errs.append((ms - mass) / mass * 1e6)
            else:
                errs.append(np.nan)
        m["test_m_err_ppm"] = errs
        best = m[m["best"]]
        n_spec = self.test.groupby("molecule_id").size()
        rows = []
        for mid, g in best.groupby("molecule_id", sort=True):
            cnt = Counter(g["ik14"])
            top, _ = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            rows.append({"molecule_id": mid, "ik14": top,
                         "smiles": g.loc[g["ik14"] == top, "smiles"].iloc[0],
                         "n_spectra": int(n_spec.get(mid, 0)), "n_matched": int(len(g)),
                         "n_labels": int(len(cnt)),
                         "source_libs": ";".join(sorted(set(g["ingest_lib"]))),
                         "consistent": len(cnt) == 1})
        truth = pd.DataFrame(rows)
        return m[[c for c in cols if c in m.columns]].reset_index(drop=True), truth


# --------------------------------------------------------------------------------------
# stage: ingest (pass 1)
# --------------------------------------------------------------------------------------

class Ingest:
    def __init__(self, cfg: Config, src: Path, out: Path, batch_rows: int, test: pd.DataFrame | None,
                 lg):
        self.cfg, self.src, self.out, self.batch_rows, self.lg = cfg, src, out, batch_rows, lg
        self.spec_dir = out / "spectra"
        self.writers: dict[str, pq.ParquetWriter] = {}
        self.file_rows: Counter = Counter()
        self.seen: set[str] = set()
        self.n_in: Counter = Counter()
        self.drops: dict[str, Counter] = defaultdict(Counter)
        self.flags: dict[str, Counter] = defaultdict(Counter)
        self.meta_parts: list[pa.Table] = []
        self.label_parts: list[pd.DataFrame] = []
        self.join = TestJoin(test) if test is not None else None
        self.ppm_flag = float(cfg.fpm_max_precursor_error_ppm)
        self.n_raw_peaks = 0
        self.n_clean_peaks = 0

    def _writer(self, lib: str, schema: pa.Schema) -> pq.ParquetWriter:
        w = self.writers.get(lib)
        if w is None:
            w = pq.ParquetWriter(self.spec_dir / (lib_file_name(lib) + ".tmp"), schema,
                                 compression="zstd")
            self.writers[lib] = w
        return w

    def run(self) -> dict:
        from casmi.spectra import clean_flat, list_arrays_to_flat

        cfg = self.cfg
        paths.ensure_dir(self.spec_dir)
        for f in self.spec_dir.glob("spectra_*.parquet*"):
            f.unlink()
        pf = pq.ParquetFile(self.src, buffer_size=1 << 22, pre_buffer=False)
        md = pf.metadata
        names = pf.schema_arrow.names
        missing = [c for c in TRAIN_COLUMNS if c not in names]
        if missing:
            raise ValueError(f"{self.src}: missing train columns {missing}")
        self.lg.info("input %s: %d rows, %d row groups, %d columns; schema:\n%s", self.src,
                     md.num_rows, md.num_row_groups, md.num_columns, pf.schema_arrow)
        t0 = time.perf_counter()
        row0 = 0
        rss_max = 0.0
        cut = float(cfg.clean_max_above_precursor_da)
        for bi, batch in enumerate(pf.iter_batches(batch_size=self.batch_rows,
                                                   columns=list(TRAIN_COLUMNS))):
            n = batch.num_rows
            if n == 0:
                continue
            g = lambda c: batch.column(c)  # noqa: E731
            sid = _str_np(g("spectrum_id"), n)
            lib = _str_np(g("ingest_lib"), n)
            lib = np.array([x if isinstance(x, str) and x else "unknown" for x in lib], dtype=object)
            ik = _str_np(g("inchikey14"), n)
            smi = _str_np(g("normalized_smiles"), n)
            form = _str_np(g("molecular_formula"), n)
            adduct = _str_np(g("adduct"), n)
            mode = _str_np(g("ionization_mode"), n)
            itype = _str_np(g("instrument_type"), n)
            prec = _f64_np(g("precursor_mz"), n)
            ppm = _f64_np(g("precursor_error_ppm"), n)
            bpi = _f64_np(g("base_peak_intensity"), n)
            mz, it, off, valid = list_arrays_to_flat(g("ms2_mzs"), g("ms2_normalized_intensities"))
            n_raw = np.diff(off)
            self.n_raw_peaks += int(off[-1])
            for x in lib:
                self.n_in[x] += 1
            # ---- drop reasons
            reason = np.full(n, None, dtype=object)

            def mark(bad: np.ndarray, name: str) -> None:
                sel = bad & (reason == None)  # noqa: E711
                reason[sel] = name

            mark(np.array([not (isinstance(a, str) and a) or not (isinstance(b, str) and b)
                           for a, b in zip(ik, smi)]), "null_label")
            mark(~(np.isfinite(prec) & (prec > 0)), "bad_precursor")
            mark(~valid, "bad_peak_arrays")
            mark(n_raw == 0, "no_raw_peaks")
            pol = np.array([_polarity(m, a) for m, a in zip(mode, adduct)], dtype=np.int8)
            mark(pol == 0, "no_polarity")
            dup = np.zeros(n, bool)
            for r in range(n):
                if reason[r] is None:
                    s = sid[r]
                    if s in self.seen:
                        dup[r] = True
                    else:
                        self.seen.add(s)
            mark(dup, "duplicate_spectrum_id")
            # public test join on raw peaks, before cleaning
            info = {"train_spectrum_id": sid, "source_row": np.arange(row0, row0 + n),
                    "ingest_lib": lib, "ik14": ik, "smiles": smi, "adduct": adduct,
                    "precursor_error_ppm": ppm}
            # ---- clean
            ok_pre = reason == None  # noqa: E711
            cm, ci, co, pint, nraw = clean_flat(mz, it, off, np.where(np.isfinite(prec), prec, 0.0),
                                                cfg, ok_pre)
            n_clean = np.diff(co)
            mark(ok_pre & (n_clean == 0), "empty_after_cleaning")
            keep = reason == None  # noqa: E711
            # ---- derived
            ainfo = [_adduct_info(a, m) for a, m in zip(adduct, mode)]
            mult = np.array([x[0] for x in ainfo], dtype=np.int16)
            shift = np.array([x[1] for x in ainfo], dtype=np.float64)
            charge = np.array([x[2] for x in ainfo], dtype=np.int8)
            atest = np.array([x[3] for x in ainfo], dtype=object)
            aknown = np.array([x[4] for x in ainfo], dtype=bool)
            with np.errstate(invalid="ignore", divide="ignore"):
                m_spec = np.where(aknown, (prec * np.abs(charge) - shift) / np.maximum(mult, 1),
                                  np.nan)
            mass = np.array([_formula_mass(f) for f in form], dtype=np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                m_err = (m_spec - mass) / mass * 1e6
            mass_info = {"mono_mass": mass, "m_err_ppm": m_err}
            if self.join is not None:
                info.update(mass_info)
                self.join.scan(sid, prec, pol, mz, it, off, info)
            fam = np.array([_family(t, l) for t, l in zip(itype, lib)], dtype=object)
            ce, nce = _list_mean(g("collision_energy_ev"))
            above = np.zeros(n, np.int32)
            for r in np.flatnonzero(keep):
                seg = mz[off[r]:off[r + 1]]
                above[r] = int((seg > prec[r] + cut).sum())
            q_ok = (keep & (atest != None) & aknown  # noqa: E711
                    & (~np.isfinite(bpi) | (bpi >= cfg.val_min_base_peak_intensity))
                    & np.isfinite(m_err) & (np.abs(np.nan_to_num(m_err, nan=1e9))
                                            <= cfg.val_query_max_abs_ppm))
            # ---- counts
            for r in np.flatnonzero(~keep):
                self.drops[lib[r]][reason[r]] += 1
            kl = lib[keep]
            for name, flag in (("adduct_unparsed", ~aknown),
                               (f"abs_ppm_gt_{self.ppm_flag:g}",
                                np.abs(np.nan_to_num(ppm, nan=0.0)) > self.ppm_flag),
                               ("peaks_above_precursor", above > 0),
                               ("bpi_null", ~np.isfinite(bpi)),
                               ("ce_unresolved", nce == 0)):
                for x in kl[flag[keep]]:
                    self.flags[x][name] += 1
            # ---- write per library
            kidx = np.flatnonzero(keep)
            if kidx.size:
                self.n_clean_peaks += int(n_clean[kidx].sum())
                base = batch.select(list(TRAIN_COLUMNS)).take(pa.array(kidx))
                derived = {
                    "source_row": np.arange(row0, row0 + n, dtype=np.int64)[kidx],
                    "polarity": pol[kidx],
                    "adduct_test": atest[kidx],
                    "adduct_known": aknown[kidx],
                    "instrument_family": fam[kidx],
                    "mono_mass": mass[kidx],
                    "m_spec": m_spec[kidx],
                    "m_err_ppm": m_err[kidx],
                    "collision_energy": ce[kidx],
                    "n_energies": nce[kidx].astype(np.int16),
                    "n_peaks_raw": nraw[kidx].astype(np.int32),
                    "n_peaks": n_clean[kidx].astype(np.int32),
                    "n_above_precursor": above[kidx],
                    "prec_intensity": pint[kidx].astype(np.float32),
                    "query_ok": q_ok[kidx],
                }
                kl_arr = lib[kidx]
                for L in sorted(set(kl_arr.tolist())):
                    sel = np.flatnonzero(kl_arr == L)
                    rows = kidx[sel]
                    lens = n_clean[rows]
                    starts = co[rows]
                    idx = np.concatenate([np.arange(s, s + k) for s, k in zip(starts, lens)]) \
                        if rows.size else np.zeros(0, np.int64)
                    loff = np.zeros(rows.size + 1, np.int64)
                    np.cumsum(lens, out=loff[1:])
                    t = base.take(pa.array(sel))
                    cols = {}
                    for c in TRAIN_COLUMNS:
                        if c == "ms2_mzs":
                            cols[c] = _float32_lists(cm[idx], loff)
                        elif c == "ms2_normalized_intensities":
                            cols[c] = _float32_lists(ci[idx], loff)
                        else:
                            cols[c] = t.column(c)
                    for c, v in derived.items():
                        vv = v[sel]
                        cols[c] = (pa.array(vv, type=pa.string(), from_pandas=True)
                                   if vv.dtype == object else pa.array(vv))
                    tbl = pa.table(cols)
                    w = self._writer(L, tbl.schema)
                    w.write_table(tbl)
                    frow = np.arange(self.file_rows[L], self.file_rows[L] + rows.size, dtype=np.int64)
                    self.file_rows[L] += rows.size
                    meta = {
                        "spectrum_id": sid[rows], "molecule_id": _str_np(t.column("molecule_id"), rows.size),
                        "ingest_lib": lib[rows], "file_row": frow,
                        "source_row": derived["source_row"][sel], "ik14": ik[rows],
                        "adduct": adduct[rows], "adduct_test": atest[rows],
                        "adduct_known": aknown[rows], "multiplier": mult[rows],
                        "charge": charge[rows], "polarity": pol[rows],
                        "instrument_type": itype[rows], "instrument_family": fam[rows],
                        "precursor_mz": prec[rows], "m_spec": m_spec[rows], "mono_mass": mass[rows],
                        "m_err_ppm": m_err[rows], "precursor_error_ppm": ppm[rows],
                        "base_peak_intensity": bpi[rows], "collision_energy": ce[rows],
                        "n_energies": nce[rows].astype(np.int16),
                        "ce_units": _str_np(t.column("collision_energy_orig_units"), rows.size),
                        "num_peaks": _f64_np(t.column("num_peaks"), rows.size),
                        "n_peaks_raw": nraw[rows].astype(np.int32),
                        "n_peaks": n_clean[rows].astype(np.int32),
                        "n_above_precursor": above[rows], "prec_intensity": pint[rows].astype(np.float32),
                        "query_ok": q_ok[rows],
                    }
                    self.meta_parts.append(pa.table({k: (pa.array(v, type=pa.string(), from_pandas=True)
                                                         if v.dtype == object else pa.array(v))
                                                     for k, v in meta.items()}))
                lab = pd.DataFrame({"ik14": ik[kidx], "smiles": smi[kidx], "formula": form[kidx]})
                self.label_parts.append(lab.groupby(["ik14", "smiles", "formula"], dropna=False,
                                                    sort=False).size().reset_index(name="n"))
            row0 += n
            rss_max = max(rss_max, _rss_now_mb())
            if bi % 10 == 0:
                el = time.perf_counter() - t0
                self.lg.info("  batch %d: %d rows read (%.0f rows/s), %d kept, RSS %.0f MB",
                             bi, row0, row0 / max(el, 1e-9), sum(self.file_rows.values()),
                             _rss_now_mb())
        for L, w in self.writers.items():
            w.close()
            tmp = self.spec_dir / (lib_file_name(L) + ".tmp")
            tmp.replace(self.spec_dir / lib_file_name(L))
        t_pass = time.perf_counter() - t0
        self.lg.info("pass 1: %d rows in %.1f s (%.0f rows/s), RSS max %.0f MB", row0, t_pass,
                     row0 / max(t_pass, 1e-9), rss_max)
        meta = pa.concat_tables(self.meta_parts, promote_options="permissive").to_pandas() \
            if self.meta_parts else pd.DataFrame(columns=list(META_COLUMNS))
        self.meta_parts = []
        meta = meta[list(META_COLUMNS)]
        meta.to_parquet(self.out / "meta.parquet", index=False)
        labels = (pd.concat(self.label_parts, ignore_index=True)
                  .groupby(["ik14", "smiles", "formula"], dropna=False, sort=True)["n"].sum()
                  .reset_index()) if self.label_parts else pd.DataFrame(columns=["ik14", "smiles", "formula", "n"])
        if self.join is not None:
            matches, truth = self.join.result()
            matches.to_parquet(self.out / "public_test_matches.parquet", index=False)
            truth.to_parquet(self.out / "public_test_truth.parquet", index=False)
            self.lg.info("public test join: %d of %d spectra matched (%d by id), %d molecules",
                         int(matches["best"].sum()) if len(matches) else 0, len(self.join.test),
                         int(matches.loc[matches["best"], "by_id"].sum()) if len(matches) else 0,
                         len(truth))
        info = {
            "input": str(self.src), "input_rows": int(row0), "input_row_groups": md.num_row_groups,
            "batch_rows": self.batch_rows, "kept_rows": int(len(meta)),
            "rows_in": dict(sorted(self.n_in.items())),
            "rows_kept": {L: int(v) for L, v in sorted(self.file_rows.items())},
            "drops": {L: dict(sorted(c.items())) for L, c in sorted(self.drops.items())},
            "flags": {L: dict(sorted(c.items())) for L, c in sorted(self.flags.items())},
            "n_raw_peaks": int(self.n_raw_peaks), "n_clean_peaks": int(self.n_clean_peaks),
            "pass1_seconds": round(t_pass, 2), "pass1_rows_per_s": round(row0 / max(t_pass, 1e-9), 1),
            "rss_max_mb_pass1": round(rss_max, 1),
            "clean_cfg": {k: getattr(cfg, k) for k in (
                "clean_max_above_precursor_da", "keep_precursor", "precursor_window_low_da",
                "precursor_window_high_da", "intensity_floor", "top_n_peaks")},
        }
        return {"meta": meta, "labels": labels, "info": info}


# --------------------------------------------------------------------------------------
# stage: structures
# --------------------------------------------------------------------------------------

_ISOTOPE_RE = re.compile(r"\[\d+[A-Z]")
_ELEM_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def _elements(formula: Any) -> str:
    if not isinstance(formula, str):
        return ""
    f = re.sub(r"[+-]\d*$", "", formula)
    return ",".join(sorted({e for e, _ in _ELEM_RE.findall(f)}))


def _n_carbon(formula: Any) -> int:
    if not isinstance(formula, str):
        return 0
    for e, k in _ELEM_RE.findall(formula):
        if e == "C":
            return int(k) if k else 1
    return 0


def _pubchem_membership(ik14: pd.Series, lg) -> tuple[np.ndarray | None, np.ndarray | None]:
    """(in_pubchem bool, pmid_count float) by ik14 over all PubChem shards; (None, None) if
    the tier is absent."""
    from casmi import db

    d = db.pubchem_dir()
    shards = sorted(d.glob("mass_*.parquet")) if d.is_dir() else []
    if not shards:
        lg.warning("PubChem tier not found in %s: in_pubchem left null", d)
        return None, None
    idx = pd.Index(ik14.astype(str).to_numpy(object))
    hit = np.zeros(len(idx), bool)
    pmid = np.full(len(idx), np.nan)
    t0 = time.perf_counter()
    for p in shards:
        t = pq.read_table(p, columns=["ik14", "pmid_count"])
        keys = t.column("ik14").to_numpy(zero_copy_only=False)
        pos = pd.Index(keys).get_indexer(idx)
        found = pos >= 0
        if found.any():
            hit |= found
            pm = t.column("pmid_count").to_numpy(zero_copy_only=False)
            pmid[found] = pm[pos[found]]
    lg.info("PubChem membership over %d shards: %d of %d structures (%.1f s)", len(shards),
            int(hit.sum()), len(idx), time.perf_counter() - t0)
    return hit, pmid


def build_structures(meta: pd.DataFrame, labels: pd.DataFrame, cfg: Config, n_jobs: int,
                     lg) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    from casmi import chem, db

    t0 = time.perf_counter()
    lab = labels.sort_values(["ik14", "n", "smiles"], ascending=[True, False, True], kind="mergesort")
    first = lab.drop_duplicates("ik14", keep="first")
    nvar = lab.groupby("ik14")["smiles"].nunique()
    fgiven = (lab.dropna(subset=["formula"]).sort_values(["ik14", "n", "formula"],
                                                          ascending=[True, False, True], kind="mergesort")
              .drop_duplicates("ik14"))
    st = pd.DataFrame({"ik14": first["ik14"].to_numpy(object),
                       "smiles": first["smiles"].to_numpy(object)})
    st["n_smiles_variants"] = st["ik14"].map(nvar).fillna(0).astype(np.int32).to_numpy()
    st["formula_given"] = st["ik14"].map(dict(zip(fgiven["ik14"], fgiven["formula"])))
    st["mass_given"] = [_formula_mass(f) for f in st["formula_given"]]
    lg.info("structures: %d (from %d label variants)", len(st), len(labels))
    # RDKit properties on std_mol (formula, mass), NP-likeness, charge and components as parsed.
    t1 = time.perf_counter()
    feat = chem.featurize(st["smiles"].tolist(), cfg, n_jobs=n_jobs,
                          fields=("formula", "mono_mass", "np_score", "charge", "n_components"))
    st["rdkit_ok"] = feat["ok"]
    st["formula"] = feat["formula"]
    st["mono_mass"] = feat["mono_mass"]
    st["charge"] = feat["charge"].astype(np.int16)
    st["n_components"] = feat["n_components"].astype(np.int16)
    st["np_score"] = feat["np_score"].astype(np.float32)
    t_feat = time.perf_counter() - t1
    lg.info("RDKit featurize: %d structures in %.1f s on %d jobs (%.0f /s)", len(st), t_feat,
            n_jobs, len(st) / max(t_feat, 1e-9))
    st["has_isotope"] = st["smiles"].map(lambda s: bool(_ISOTOPE_RE.search(s)) if isinstance(s, str) else False)
    st["elements"] = [_elements(f if isinstance(f, str) else g)
                      for f, g in zip(st["formula"], st["formula_given"])]
    allowed = set(cfg.db_elements)
    st["elements_ok"] = [bool(e) and set(e.split(",")) <= allowed for e in st["elements"]]
    st["n_carbon"] = [_n_carbon(f if isinstance(f, str) else g)
                      for f, g in zip(st["formula"], st["formula_given"])]
    m = st["mono_mass"].to_numpy(np.float64)
    st["db_filter_ok"] = (st["rdkit_ok"] & st["elements_ok"] & (st["charge"] == 0)
                          & (st["n_components"] == 1) & ~st["has_isotope"]
                          & (st["n_carbon"] >= cfg.db_min_carbons)
                          & (m >= cfg.db_mass_min) & (m <= cfg.db_mass_max))
    # Counts per library, instrument family, adduct, polarity.
    mt = meta[["ik14", "ingest_lib", "instrument_family", "adduct_test", "adduct_known",
               "polarity", "query_ok"]].copy()
    mt["add_cat"] = np.where(mt["adduct_test"].notna(), mt["adduct_test"].astype(object),
                             np.where(mt["adduct_known"], "other", "unparsed"))
    total = mt.groupby("ik14").size()
    st["n_spectra"] = st["ik14"].map(total).fillna(0).astype(np.int64)
    libs = sorted(meta["ingest_lib"].unique().tolist())

    def wide(col: str, prefix: str, values: Iterable[str]) -> None:
        ct = mt.groupby(["ik14", col]).size().unstack(fill_value=0)
        for v in values:
            s = ct[v] if v in ct.columns else pd.Series(dtype=np.int64)
            st[f"{prefix}{v}"] = st["ik14"].map(s).fillna(0).astype(np.int32).to_numpy()

    from casmi.adducts import TEST_ADDUCT_NAMES

    wide("ingest_lib", "n_lib_", libs)
    wide("instrument_family", "n_inst_", INSTRUMENT_FAMILIES)
    wide("add_cat", "n_add_", list(TEST_ADDUCT_NAMES) + ["other", "unparsed"])
    st["n_pos"] = st["ik14"].map(mt[mt["polarity"] > 0].groupby("ik14").size()).fillna(0).astype(np.int32)
    st["n_neg"] = st["ik14"].map(mt[mt["polarity"] < 0].groupby("ik14").size()).fillna(0).astype(np.int32)
    st["n_query_ok"] = st["ik14"].map(mt[mt["query_ok"]].groupby("ik14").size()).fillna(0).astype(np.int32)
    lib_sets = mt.groupby("ik14")["ingest_lib"].agg(lambda s: ";".join(sorted(set(s))))
    st["libs"] = st["ik14"].map(lib_sets)
    st["n_libs"] = st["libs"].fillna("").map(lambda s: len(s.split(";")) if s else 0).astype(np.int16)
    # NP tier and PubChem tier.
    try:
        npt = db.np_table()
        npcols = ["ik14", "in_coconut", "in_npatlas", "in_lotus", "in_pubchemlite",
                  "coconut_annotation_level", "pubmed_count", "n_citations"]
        npd = npt[npcols].drop_duplicates("ik14").set_index("ik14")
        npd.index = npd.index.astype(object)
        j = npd.reindex(st["ik14"].to_numpy(object))
        st["in_np"] = j["in_coconut"].notna().to_numpy()
        for c in ("in_coconut", "in_npatlas", "in_lotus", "in_pubchemlite"):
            st[c] = j[c].fillna(False).astype(bool).to_numpy()
        st["coconut_annotation_level"] = pd.array(j["coconut_annotation_level"].to_numpy(), dtype="Int8")
        st["pubmed_count"] = pd.array(j["pubmed_count"].to_numpy(), dtype="Int32")
        st["n_citations"] = pd.array(j["n_citations"].to_numpy(), dtype="Int32")
    except FileNotFoundError as exc:
        lg.warning("NP tier missing (%s): NP flags left null", exc)
        for c in ("in_np", "in_coconut", "in_npatlas", "in_lotus", "in_pubchemlite"):
            st[c] = pd.array([pd.NA] * len(st), dtype="boolean")
    hit, pmid = _pubchem_membership(st["ik14"], lg)
    st["in_pubchem"] = (pd.array(hit, dtype="boolean") if hit is not None
                        else pd.array([pd.NA] * len(st), dtype="boolean"))
    st["pmid_count"] = (pd.array(np.where(np.isfinite(pmid), pmid, np.nan), dtype="Float64").astype("Int32")
                        if pmid is not None else pd.array([pd.NA] * len(st), dtype="Int32"))
    st["mkey"] = pd.Series([None] * len(st), dtype=object)
    st["mkey_status"] = pd.Series([None] * len(st), dtype=object)
    st = st.sort_values("ik14", kind="mergesort").reset_index(drop=True)
    # Long table per (ik14, lib).
    q = mt.assign(qtof=meta["instrument_family"].eq("qtof"),
                  test_add=meta["adduct_test"].notna())
    q["q_pos"] = q["query_ok"] & (q["polarity"] > 0)
    q["q_neg"] = q["query_ok"] & (q["polarity"] < 0)
    q["q_qtof"] = q["query_ok"] & q["qtof"]
    lc = q.groupby(["ik14", "ingest_lib"], sort=True).agg(
        n_spectra=("polarity", "size"), n_test_adduct=("test_add", "sum"),
        n_query_ok=("query_ok", "sum"), n_query_ok_pos=("q_pos", "sum"),
        n_query_ok_neg=("q_neg", "sum"), n_query_ok_qtof=("q_qtof", "sum")).reset_index()
    for c in lc.columns[2:]:
        lc[c] = lc[c].astype(np.int32)
    info = {"n_structures": int(len(st)), "featurize_seconds": round(t_feat, 2),
            "featurize_per_s": round(len(st) / max(t_feat, 1e-9), 1), "featurize_jobs": n_jobs,
            "structures_seconds": round(time.perf_counter() - t0, 2)}
    return st, lc, info


# --------------------------------------------------------------------------------------
# stage: mkey
# --------------------------------------------------------------------------------------

def stage_mkey(out: Path, cfg: Config, n_jobs: int, lg) -> dict:
    from casmi.metric import mkeys

    path = out / "structures.parquet"
    st = pd.read_parquet(path)
    t0 = time.perf_counter()
    res = mkeys(st["smiles"].tolist(), cfg, n_jobs=n_jobs, cache=True, return_status=True)
    st["mkey"] = pd.Series([k for k, _ in res], dtype=object)
    st["mkey_status"] = pd.Series([s for _, s in res], dtype=object)
    dt = time.perf_counter() - t0
    tmp = path.with_name(path.name + ".tmp")
    st.to_parquet(tmp, index=False)
    tmp.replace(path)
    stats = dict(Counter(st["mkey_status"]))
    diff = int((st["mkey"].notna() & (st["mkey"] != st["ik14"])).sum())
    lg.info("mkey: %d structures in %.1f s on %d jobs; status %s; mkey != ik14 for %d",
            len(st), dt, n_jobs, stats, diff)
    info_path = out / "ingest_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    info["mkey"] = {"seconds": round(dt, 1), "jobs": n_jobs, "status": stats,
                    "mkey_ne_ik14": diff, "done_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    info_path.write_text(json.dumps(info, indent=2))
    return info["mkey"]


# --------------------------------------------------------------------------------------
# stage: audit
# --------------------------------------------------------------------------------------

def _fmt(x: Any, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    if isinstance(x, (float, np.floating)):
        return f"{float(x):,.{nd}f}"
    return str(x)


def _table(df: pd.DataFrame, nd: int = 2) -> str:
    cols = [str(c).replace("|", "/") for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v, nd).replace("|", "/") for v in row) + " |")
    return "\n".join(lines)


def tolerance_recommendation(meta: pd.DataFrame, matches: pd.DataFrame, floor_ppm: float = 5.0,
                             q: float = 99.5) -> dict:
    """PLAN 3.3: per-molecule median |M error| (ppm) over test-adduct spectra of the two Enveda
    libraries, and over the public test spectra joined to their train labels; window =
    q-th percentile, rounded up, at least ``floor_ppm``."""
    out: dict[str, Any] = {}
    e = meta[meta["ingest_lib"].isin(ENVEDA_LIBS) & meta["adduct_test"].notna()]
    for name, col in (("m_err", "m_err_ppm"), ("given", "precursor_error_ppm")):
        med = e.groupby(["ingest_lib", "ik14"])[col].median().abs().dropna()
        out[f"enveda_{name}_n"] = int(len(med))
        out[f"enveda_{name}_p{q:g}"] = float(np.percentile(med, q)) if len(med) else float("nan")
        out[f"enveda_{name}_max"] = float(med.max()) if len(med) else float("nan")
    if matches is not None and len(matches):
        b = matches[matches["best"]]
        per_mol = b.groupby("molecule_id")["test_m_err_ppm"].median().abs().dropna()
        out["public_n"] = int(len(per_mol))
        out[f"public_p{q:g}"] = float(np.percentile(per_mol, q)) if len(per_mol) else float("nan")
        out["public_max"] = float(per_mol.max()) if len(per_mol) else float("nan")
        spec = b["test_m_err_ppm"].abs().dropna()
        out["public_spectrum_max"] = float(spec.max()) if len(spec) else float("nan")
    cands = [out.get(f"enveda_m_err_p{q:g}"), out.get(f"public_p{q:g}")]
    cands = [c for c in cands if c is not None and math.isfinite(c)]
    out["recommended_ppm"] = float(max([floor_ppm] + [math.ceil(c) for c in cands]))
    return out


def write_audit(out: Path, audit_path: Path, cfg: Config, lg) -> None:
    from casmi.adducts import TEST_ADDUCT_NAMES

    info = json.loads((out / "ingest_info.json").read_text())
    meta = pd.read_parquet(out / "meta.parquet")
    st = pd.read_parquet(out / "structures.parquet")
    mpath = out / "public_test_matches.parquet"
    matches = pd.read_parquet(mpath) if mpath.exists() else None
    tpath = out / "public_test_truth.parquet"
    truth = pd.read_parquet(tpath) if tpath.exists() else None
    libs = sorted(set(info["rows_in"]) | set(EXPECTED))
    L: list[str] = []
    synthetic = "synthetic" in str(info.get("input", ""))
    L.append("# Data audit (WP6)")
    L.append("")
    L.append(f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/ingest_train.py --stage audit` "
             f"from `{info['input']}` ({info['input_rows']:,} rows, {info['input_row_groups']} row groups). "
             "Every number below is recomputed by that command; PLAN.md section numbers in brackets.")
    if synthetic:
        L.append("")
        L.append("**SYNTHETIC INPUT** (scripts/make_synthetic_train.py): numbers exercise the code "
                 "only and say nothing about the real data.")
    mk_done = st["mkey"].notna().any()
    if not mk_done:
        L.append("")
        L.append("mkey pass not run yet: mkey-dependent items are marked pending. Run "
                 "`scripts/ingest_train.py --stage mkey` then `--stage audit`.")
    # 1. row counts
    L += ["", "## 1. Rows and structures per library (WP6 acceptance)", ""]
    rows = []
    nst = meta.groupby("ingest_lib")["ik14"].nunique()
    for lib in libs:
        exp_s, exp_u = EXPECTED.get(lib, (None, None))
        rin = info["rows_in"].get(lib, 0)
        kept = info["rows_kept"].get(lib, 0)
        dr = sum(info["drops"].get(lib, {}).values())
        u = int(nst.get(lib, 0))
        dev = (kept - exp_s) / exp_s * 100 if exp_s else float("nan")
        devu = (u - exp_u) / exp_u * 100 if exp_u else float("nan")
        rows.append([lib, rin, dr, kept, exp_s, dev, u, exp_u, devu])
    df = pd.DataFrame(rows, columns=["library", "rows in", "dropped", "kept", "expected spectra",
                                     "kept vs expected %", "structures", "expected structures",
                                     "structures vs expected %"])
    L.append(_table(df, 2))
    within = df.dropna(subset=["expected spectra"])
    ok = bool((within["kept vs expected %"].abs() <= 1.0).all()) if len(within) else False
    L.append("")
    L.append(f"Within 1 percent of COMPETITION.md for every library: **{ok}** "
             f"(total kept {info['kept_rows']:,} of {info['input_rows']:,}).")
    L += ["", "Drop reasons (first failing check, order as in the ingest docstring):", ""]
    dd = pd.DataFrame([[lib] + [info["drops"].get(lib, {}).get(r, 0) for r in DROP_REASONS]
                       for lib in libs], columns=["library", *DROP_REASONS])
    L.append(_table(dd))
    fl_names = sorted({k for v in info["flags"].values() for k in v})
    L += ["", "Flags on kept rows (not dropped):", ""]
    ff = pd.DataFrame([[lib] + [info["flags"].get(lib, {}).get(k, 0) for k in fl_names]
                       for lib in libs], columns=["library", *fl_names])
    L.append(_table(ff))
    # 2. adducts
    L += ["", "## 2. Adducts per library [3.1]", ""]
    cat = np.where(meta["adduct_test"].notna(), meta["adduct_test"].astype(object),
                   np.where(meta["adduct_known"], "other parsed", "unparsed"))
    ct = pd.crosstab(pd.Series(cat, name="adduct"), meta["ingest_lib"])
    order = [a for a in list(TEST_ADDUCT_NAMES) + ["other parsed", "unparsed"] if a in ct.index]
    ct = ct.reindex(order).fillna(0).astype(int)
    ct.insert(0, "adduct", ct.index)
    L.append(_table(ct.reset_index(drop=True)))
    other = meta.loc[meta["adduct_test"].isna(), ["adduct", "adduct_known"]]
    top = other.groupby(["adduct", "adduct_known"], dropna=False).size().sort_values(ascending=False).head(25)
    L += ["", "Most frequent adduct strings outside the ten test adducts (spelling as in `adduct`):", ""]
    L.append(_table(pd.DataFrame({"adduct": [a for a, _ in top.index],
                                  "parsed": [bool(k) for _, k in top.index], "spectra": top.values})))
    # 3. instruments
    L += ["", "## 3. Instrument family mapping [4.4, `instrument_family_of`]", ""]
    L.append("Rules: Enveda libraries -> timstof; text with `tims` -> timstof; Orbitrap/Exactive/"
             "Lumos/Fusion/Eclipse/Astral/Exploris/ITFT/QFT/FT names -> orbitrap; Q-TOF/TOF/maXis/"
             "impact/Synapt/Xevo/TripleTOF/Zeno/Agilent 65xx/compact -> qtof; else pluskal_ms2 -> "
             "orbitrap, otherwise other.")
    fam = pd.crosstab(meta["instrument_family"], meta["ingest_lib"])
    fam.insert(0, "family", fam.index)
    L.append("")
    L.append(_table(fam.reset_index(drop=True)))
    it = meta.assign(itype=meta["instrument_type"].fillna("<null>")).groupby(
        ["itype", "instrument_family"]).agg(spectra=("ik14", "size"),
                                            libraries=("ingest_lib", lambda s: ";".join(sorted(set(s)))))
    it = it.sort_values("spectra", ascending=False).reset_index().head(60)
    L += ["", f"Free-text values with their family (top {len(it)} of "
          f"{meta['instrument_type'].fillna('<null>').nunique()} distinct texts, by spectra):", ""]
    L.append(_table(it.rename(columns={"itype": "instrument_type", "instrument_family": "family"})))
    # 4. ppm
    L += ["", "## 4. precursor_error_ppm per library [3.3, risk 10]", ""]
    rows = []
    for lib, g in meta.groupby("ingest_lib"):
        e = g["precursor_error_ppm"].to_numpy(np.float64)
        a = np.abs(e[np.isfinite(e)])
        me = np.abs(g["m_err_ppm"].to_numpy(np.float64))
        me = me[np.isfinite(me)]
        qs = np.percentile(a, [50, 90, 99, 99.5]) if a.size else [np.nan] * 4
        rows.append([lib, int(e.size), int((~np.isfinite(e)).sum()),
                     float(np.nanmedian(e)) if a.size else np.nan, *qs,
                     float(a.max()) if a.size else np.nan,
                     int((a > cfg.fpm_max_precursor_error_ppm).sum()),
                     float(np.percentile(me, 99.5)) if me.size else np.nan])
    L.append(_table(pd.DataFrame(rows, columns=[
        "library", "spectra", "NaN", "median (signed)", "abs p50", "abs p90", "abs p99",
        "abs p99.5", "abs max", f"abs > {cfg.fpm_max_precursor_error_ppm:g}",
        "abs M err p99.5 (recomputed)"])))
    L.append("")
    L.append("Columns `abs ...` are quantiles of the absolute given error. `abs M err` is recomputed here: neutral mass from precursor m/z and the parsed adduct "
             "versus the formula mass of `molecular_formula`, in ppm of M (the quantity the "
             "retrieval window applies to).")
    # 5. overlap
    L += ["", "## 5. Structure overlap between libraries (by inchikey14)", ""]
    pairs = meta[["ik14", "ingest_lib"]].drop_duplicates()
    sets = {lib: set(g["ik14"]) for lib, g in pairs.groupby("ingest_lib")}
    names = sorted(sets)
    ov = pd.DataFrame([[a] + [len(sets[a] & sets[b]) for b in names] for a in names],
                      columns=["library", *names])
    L.append(_table(ov))
    if mk_done:
        mk = dict(zip(st["ik14"], st["mkey"]))
        msets = {lib: {mk.get(k) for k in s} - {None} for lib, s in sets.items()}
        L += ["", "Same matrix by mkey:", ""]
        L.append(_table(pd.DataFrame([[a] + [len(msets[a] & msets[b]) for b in names] for a in names],
                                     columns=["library", *names])))
    # 6. C1 eligibility
    L += ["", "## 6. C1 eligibility [5.3]: Enveda structures also in other libraries", ""]
    rows = []
    for lib in ENVEDA_LIBS:
        s = sets.get(lib, set())
        others = set().union(*[v for k, v in sets.items() if k != lib]) if sets else set()
        public = set().union(*[v for k, v in sets.items() if k not in ENVEDA_LIBS]) if sets else set()
        rows.append([lib, len(s), len(s & others), len(s & others) / max(len(s), 1),
                     len(s & public), len(s & public) / max(len(s), 1)])
    L.append(_table(pd.DataFrame(rows, columns=["library", "structures", "in any other library",
                                                "fraction", "in a non-Enveda library", "fraction"]), 3))
    # 7. DB recall
    L += ["", "## 7. Candidate recall ceiling: train structures in the NP and PubChem tiers [2, 5.5]", ""]
    rows = []
    lk = pairs.merge(st[["ik14", "in_np", "in_pubchem", "db_filter_ok"]], on="ik14", how="left")
    for lib, g in lk.groupby("ingest_lib"):
        inp = g["in_np"].fillna(False).astype(bool)
        inpc = g["in_pubchem"].astype("boolean")
        pc_ok = inpc.notna().all()
        inpc = inpc.fillna(False).astype(bool)
        filt = g["db_filter_ok"].astype(bool)
        rows.append([lib, len(g), inp.mean(), inpc.mean() if pc_ok else np.nan,
                     (inp | inpc).mean(), filt.mean(),
                     ((inp | inpc) & filt).sum() / max(filt.sum(), 1)])
    L.append(_table(pd.DataFrame(rows, columns=["library", "structures", "in NP tier",
                                                "in PubChem tier", "in either",
                                                "passes 2.2 filters", "in either | passes filters"]), 3))
    L.append("")
    L.append("Membership is by inchikey14 (train label vs the tier's ik14). A structure that fails "
             "the 2.2 filters cannot be retrieved from either tier by construction.")
    # 8. elements
    L += ["", "## 8. Element coverage and the other 2.2 filters", ""]
    sl = pairs.merge(st[["ik14", "elements", "elements_ok", "charge", "n_components",
                         "has_isotope", "n_carbon", "mono_mass", "rdkit_ok"]], on="ik14", how="left")
    allowed = set(cfg.db_elements)
    rows = []
    extra_all: Counter = Counter()
    for lib, g in sl.groupby("ingest_lib"):
        extra: Counter = Counter()
        for e in g["elements"].fillna(""):
            for x in set(e.split(",")) - allowed - {""}:
                extra[x] += 1
        extra_all.update(extra)
        m = g["mono_mass"]
        rows.append([lib, len(g), 1 - g["elements_ok"].mean(), (g["charge"] != 0).mean(),
                     (g["n_components"] > 1).mean(), g["has_isotope"].mean(),
                     (g["n_carbon"] < cfg.db_min_carbons).mean(),
                     ((m < cfg.db_mass_min) | (m > cfg.db_mass_max)).mean(), (~g["rdkit_ok"]).mean(),
                     ", ".join(f"{k} {v}" for k, v in extra.most_common(6))])
    L.append(_table(pd.DataFrame(rows, columns=[
        "library", "structures", "elements outside set", "charged", "multi-component",
        "isotope label", f"< {cfg.db_min_carbons} C", "mass outside range", "RDKit fails",
        "extra elements (structures)"]), 4))
    nph = sl[sl["ingest_lib"].isin(NP_HEAVY_LIBS)].drop_duplicates("ik14")
    rec = []
    for el in ("B", "Si", "Se"):
        frac = nph["elements"].fillna("").map(lambda e: el in e.split(",")).mean() if len(nph) else 0.0
        rec.append(f"{el}: {frac * 100:.3f} percent")
    widen = [el for el in ("B", "Si", "Se") if len(nph) and
             nph["elements"].fillna("").map(lambda e: el in e.split(",")).mean() > 0.005]
    L.append("")
    L.append(f"NP-heavy libraries ({', '.join(NP_HEAVY_LIBS)}; {len(nph):,} structures): "
             + "; ".join(rec) + ". Rule (2.2): widen the element set only above 0.5 percent. "
             f"Recommendation: **{'widen with ' + ', '.join(widen) if widen else 'keep the element set'}**.")
    # 9. mkey vs ik14
    L += ["", "## 9. mkey versus inchikey14 [5.1]", ""]
    if mk_done:
        has = st["mkey"].notna()
        diff = has & (st["mkey"] != st["ik14"])
        grp = st[has].groupby("mkey")["ik14"].nunique()
        L.append(f"mkey computed for {int(has.sum()):,} of {len(st):,} structures "
                 f"(status: {dict(Counter(st['mkey_status'].fillna('none')))}). "
                 f"mkey != ik14 for {int(diff.sum()):,} ({diff.sum() / max(has.sum(), 1) * 100:.2f} percent). "
                 f"mkeys shared by more than one ik14: {int((grp > 1).sum()):,} "
                 f"(covering {int(grp[grp > 1].sum()):,} ik14).")
        pl = pairs.merge(st[["ik14", "mkey"]], on="ik14")
        pl = pl[pl["mkey"].notna()]
        r2 = pl.assign(d=pl["mkey"] != pl["ik14"]).groupby("ingest_lib")["d"].mean().reset_index()
        L.append("")
        L.append(_table(r2.rename(columns={"ingest_lib": "library", "d": "fraction mkey != ik14"}), 4))
    else:
        L.append("Pending (mkey stage not run).")
    # 10. public test join
    L += ["", "## 10. Public test join [5.7]", ""]
    tol = tolerance_recommendation(meta, matches)
    if matches is not None and len(matches):
        b = matches[matches["best"]]
        n_test = int(info.get("public_test_spectra", 0)) or len(set(matches["test_spectrum_id"]))
        L.append(f"Public test spectra matched to a train row: {len(b):,} of {n_test:,} "
                 f"({int(b['by_id'].sum()):,} by shared spectrum_id, {int((~b['by_id']).sum()):,} by "
                 f"precursor m/z plus peak arrays). Molecules with a label: {len(truth):,}; "
                 f"label consistent across a molecule's spectra: {int(truth['consistent'].sum()):,}.")
        src = b.groupby("ingest_lib").agg(spectra=("test_spectrum_id", "size"),
                                          molecules=("molecule_id", "nunique")).reset_index()
        L.append("")
        L.append(_table(src.rename(columns={"ingest_lib": "source library"})))
        e = b["test_m_err_ppm"].abs().dropna()
        L.append("")
        if len(e):
            L.append("M error of the public test spectra against their train label (ppm, after the "
                     f"organisers' cleaning): p50 {np.percentile(e, 50):.2f}, p90 {np.percentile(e, 90):.2f}, "
                     f"p99 {np.percentile(e, 99):.2f}, max {e.max():.2f}; per molecule (median over its "
                     f"spectra) p99.5 {tol.get('public_p99.5', float('nan')):.2f}, max {tol.get('public_max', float('nan')):.2f}.")
        ke = b["max_kept_excess"].max()
        de = b["min_dropped_excess"].min()
        L.append("")
        L.append(f"Peak cut evidence: largest kept peak above precursor {ke:+.4f} Da; smallest "
                 f"train peak above precursor missing from the test spectrum {de:+.4f} Da; "
                 f"intensity ratio test/train on matched peaks: median "
                 f"{b['int_ratio_median'].median():.4f}, max spread {b['int_ratio_spread'].max():.3g} "
                 "(1 and 0 mean the intensities were not renormalised).")
        ad = b.merge(truth[["molecule_id"]], on="molecule_id")
        cnt = b.groupby("molecule_id").size()
        L.append("")
        L.append(f"Matched spectra per molecule: mean {cnt.mean():.2f}, max {cnt.max()}. Train "
                 "spectra available for the same structure in the source library versus spectra "
                 "the test kept (query emulation, 5.2):")
        avail = meta.groupby(["ik14", "ingest_lib"]).size()
        rat = []
        for mid, g in b.groupby("molecule_id"):
            k = (g["ik14"].iloc[0], g["ingest_lib"].iloc[0])
            rat.append((len(g), int(avail.get(k, 0))))
        ra = pd.DataFrame(rat, columns=["kept", "available"])
        L.append("")
        L.append(f"kept / available: mean {np.mean(ra['kept'] / ra['available'].clip(lower=1)):.3f}; "
                 f"available median {ra['available'].median():.0f}, max {ra['available'].max()}.")
        _ = ad
    else:
        L.append("No public test spectrum was found in train.")
    # 11. tolerance
    L += ["", "## 11. Retrieval tolerance recommendation [3.3]", ""]
    L.append("Per-molecule median |M error| over the ten test adducts, 99.5th percentile:")
    L.append("")
    L.append(f"- enveda libraries, recomputed M error: {tol.get('enveda_m_err_p99.5', float('nan')):.2f} ppm "
             f"over {tol.get('enveda_m_err_n', 0):,} (library, structure) groups (max "
             f"{tol.get('enveda_m_err_max', float('nan')):.2f}); with the given precursor_error_ppm: "
             f"{tol.get('enveda_given_p99.5', float('nan')):.2f} ppm.")
    L.append(f"- public test joined to train labels: {tol.get('public_p99.5', float('nan')):.2f} ppm over "
             f"{tol.get('public_n', 0):,} molecules (single spectrum max "
             f"{tol.get('public_spectrum_max', float('nan')):.2f} ppm).")
    L.append("")
    L.append(f"Recommended `mass_tol_ppm`: **{tol['recommended_ppm']:g}** (larger of the two, rounded up, "
             f"floor 5; current default {cfg.mass_tol_ppm:g}). `val_query_max_abs_ppm` (query "
             "emulation, organisers' precursor cut) should not be below the public single-spectrum "
             "maximum above.")
    # 12. structures table summary
    L += ["", "## 12. structures.parquet", ""]
    L.append(f"{len(st):,} structures; RDKit parse ok {int(st['rdkit_ok'].sum()):,}; label SMILES "
             f"variants per ik14 > 1: {int((st['n_smiles_variants'] > 1).sum()):,}; RDKit mass vs "
             f"formula mass differs by more than 0.001 Da: "
             f"{int(((st['mono_mass'] - st['mass_given']).abs() > 1e-3).sum()):,}; NP-likeness median "
             f"{st['np_score'].median():.2f}.")
    L.append("")
    L.append(f"Ingestion: pass 1 {info['pass1_seconds']:.1f} s ({info['pass1_rows_per_s']:,.0f} rows/s), "
             f"peak RSS {info.get('rss_max_mb_pass1', float('nan')):,.0f} MB during pass 1; structures "
             f"{info.get('structures', {}).get('structures_seconds', float('nan'))} s "
             f"(RDKit {info.get('structures', {}).get('featurize_per_s', float('nan'))} structures/s on "
             f"{info.get('structures', {}).get('featurize_jobs', '?')} jobs).")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    tol_path = out / "tolerance.json"
    tol_path.write_text(json.dumps(tol, indent=2))
    lg.info("audit written to %s (tolerance recommendation %s ppm)", audit_path, tol["recommended_ppm"])


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def stage_ingest(args, cfg: Config, lg) -> dict:
    out = paths.ensure_dir(args.out)
    test = pd.read_parquet(args.test) if args.test and Path(args.test).exists() else None
    if test is None:
        lg.warning("no public test file: skipping the public test join")
    ing = Ingest(cfg, Path(args.input), out, args.batch_rows, test, lg)
    res = ing.run()
    info = res["info"]
    info["public_test_spectra"] = int(len(test)) if test is not None else 0
    st, lc, sinfo = build_structures(res["meta"], res["labels"], cfg, args.n_jobs, lg)
    info["structures"] = sinfo
    st.to_parquet(out / "structures.parquet", index=False)
    lc.to_parquet(out / "structure_lib_counts.parquet", index=False)
    info["peak_rss_mb"] = round(peak_rss_mb(), 1)
    info["done_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / "ingest_info.json").write_text(json.dumps(info, indent=2))
    lg.info("ingest done: %d kept rows, %d structures, peak RSS %.0f MB", info["kept_rows"],
            len(st), peak_rss_mb())
    return info


class _permissive_concat:
    """Make ``pyarrow.concat_tables`` promote null-typed columns while WP5's library builds.

    ``library._Builder._pass1`` concatenates per-batch metadata tables; a batch whose text
    column is entirely null (drug_plus has no instrument_type) yields a null-typed column and
    the default concat refuses it. Workaround until ``_pass1`` passes
    ``promote_options="permissive"`` itself.
    """

    def __enter__(self):
        self._orig = pa.concat_tables

        def concat(tables, *args, **kwargs):
            kwargs.setdefault("promote_options", "permissive")
            return self._orig(tables, *args, **kwargs)

        pa.concat_tables = concat
        return self

    def __exit__(self, *exc):
        pa.concat_tables = self._orig


def stage_library(out: Path, lib_out: Path, cfg: Config, lg) -> dict:
    """WP5 library index from the WP6 spectra files (``Library.build``)."""
    from casmi.library import Library

    with _permissive_concat():
        lib = Library.build(out / "spectra", cfg, out_dir=lib_out)
    info = lib.info
    size = sum(f.stat().st_size for f in lib.dir.iterdir() if f.is_file())
    lg.info("library %s: %d spectra, %d structures, %d consensus, %.1f MB; drops %s", lib.dir,
            info["n_spectra"], info["n_structures"], info["n_consensus"], size / 1e6, info["drops"])
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=("ingest", "mkey", "audit", "library", "all"), default="all")
    ap.add_argument("--library-out", type=Path, default=paths.PROCESSED / "library",
                    help="output of --stage library (WP5 index built from the spectra files)")
    ap.add_argument("--input", type=Path, default=paths.TRAIN_PARQUET)
    ap.add_argument("--out", type=Path, default=paths.PROCESSED / "train")
    ap.add_argument("--test", type=Path, default=paths.TEST_PARQUET,
                    help="public test file for the join of PLAN 5.7")
    ap.add_argument("--audit-out", type=Path, default=paths.DOCS / "DATA_AUDIT.md")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--batch-rows", type=int, default=20000, help="rows per streamed batch")
    ap.add_argument("--n-jobs", type=int, default=2, help="processes for the RDKit structure pass")
    ap.add_argument("--mkey-jobs", type=int, default=4, help="processes for the mkey stage")
    args = ap.parse_args(argv)
    lg = get_logger(f"ingest_train_{args.stage}")
    log_run(lg, args)
    cfg = load_config(args.config)
    if args.stage in ("ingest", "all"):
        if not Path(args.input).exists():
            lg.error("input not found: %s", args.input)
            return 2
        stage_ingest(args, cfg, lg)
    if args.stage == "mkey":
        stage_mkey(Path(args.out), cfg, args.mkey_jobs, lg)
    if args.stage in ("audit", "all"):
        write_audit(Path(args.out), Path(args.audit_out), cfg, lg)
    if args.stage == "library":
        stage_library(Path(args.out), Path(args.library_out), cfg, lg)
    return 0


if __name__ == "__main__":
    sys.exit(main())

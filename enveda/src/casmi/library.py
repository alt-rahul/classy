"""Spectral library index: build, load, mass-restricted search and open (analog) search.

PLAN.md sections 4.2 and 4.3 (WP5).

Input schema (``Library.build``)
--------------------------------
One row per spectrum, the columns of ``train.parquet`` (docs/COMPETITION.md); WP6's
``data/processed/train/spectra_<lib>.parquet`` (same columns, peaks already cleaned) works
too, because ``spectra.clean`` is idempotent. Columns read:

=============================  =========================  =================================
column                         type                       use
=============================  =========================  =================================
ms2_mzs                        list<double|float>         required, fragment m/z
ms2_normalized_intensities     list<double|float>         required, aligned intensities
precursor_mz                   double                     required
inchikey14 (or ik14)           string                     required, structure key
spectrum_id                    string                     id (row number if absent)
adduct                         string                     adduct family, same-adduct pairs
ionization_mode                string                     polarity (else the adduct sign)
normalized_smiles (or smiles)  string                     carried to the candidate tables
molecular_formula              string                     structure mass (casmi.adducts)
instrument_type                string, free text          instrument family
ingest_lib                     string                     source library, exclusion masks
collision_energy_ev            list<double>               mean eV and number of energies
base_peak_intensity            double                     carried
precursor_error_ppm            double                     carried (label-quality feature)
mono_mass                      double, optional           overrides the formula mass
instrument_family              string, optional           overrides the mapping below
molecule_id                    string, optional           carried (test-derived libraries)
=============================  =========================  =================================

Structure mass: ``mono_mass`` (median per structure) if present, else the monoisotopic
mass of the most frequent ``molecular_formula`` (charge suffix stripped), else the median
over the structure's spectra of the neutral mass implied by precursor m/z and adduct.
Structures without a finite mass, spectra without ik14 or polarity, and spectra with fewer
than ``cfg.lib_min_peaks`` peaks after cleaning are dropped (counts in library_info.json).

Instrument family (``instrument_family``): "timstof" for the two Enveda libraries or
"tims" in the text; "orbitrap" for Orbitrap/Exactive/Lumos/Fusion/Astral/ITFT/QFT/FT
names (or pluskal_ms2 with no text); "qtof" for QTOF/TOF/Maxis/Impact/Synapt/Xevo/
TripleTOF/Agilent 65xx; else "other".

Adduct family (``ADDUCT_FAMILY``): the ion type whose fragments the spectrum shows.
[M+H]+ family = [M+H]+, [M+NH4]+, [M-H2O+H]+, [M-2H2O+H]+; [M-H]- family = [M-H]-,
[M-H2O-H]-, [M+CH2O2-H]-; [M+Na]+, [M+K]+, [M+Cl]- are their own families; any other
adduct is its own family named by its adduct string. A consensus spectrum's reference
precursor (for neutral losses and the modified-cosine shift) is the family's base adduct
at the structure mass (median member precursor m/z for "other" families).

Output files (``paths.PROCESSED / "library"``)
----------------------------------------------
- ``peaks_mz.npy`` float32, ``peaks_int.npy`` float32 (base peak 1), ``offsets.npy``
  int64 (n_spectra + 1): cleaned spectra, ordered by structure (mono_mass, ik14), then
  polarity, adduct family, instrument family, ingest_lib, adduct, collision energy,
  spectrum_id. The spectra of a structure are contiguous.
- ``meta.parquet``: one row per spectrum in store order (``SPECTRUM_COLUMNS``).
- ``structures.parquet``: one row per structure, sorted by (mono_mass, ik14):
  struct_idx, ik14, smiles, formula, mono_mass, spec_start, spec_end, cons_start,
  cons_end, n_spectra, n_consensus.
- ``consensus_mz.npy``, ``consensus_int.npy``, ``consensus_offsets.npy``: one consensus
  spectrum per (structure, polarity, adduct family, instrument family, ingest_lib),
  members are contiguous store rows. ``ingest_lib`` is part of the key so that the
  validation masks (which remove a molecule's spectra per library) never split a
  consensus.
- ``consensus_meta.parquet``: cons_idx, struct_idx, ik14, polarity, adduct_family,
  instrument_family, ingest_lib, precursor_mz (reference), mono_mass, spec_start,
  spec_end, n_members, n_peaks, open_row.
- ``open_index_pos.npz``, ``open_index_neg.npz``: ``scipy.sparse.save_npz`` (uncompressed)
  CSR float32, one row per consensus of that polarity in consensus order (row =
  ``open_row``), columns [fragment bins | neutral-loss bins] (``spectra.open_vectors``).
- ``library_info.json``: counts, drop reasons, the cleaning settings used, build time.

Exclusion masks (validation conditions, PLAN.md 5.3) are boolean arrays over store rows
(True = removed) passed per call; ``spectra_mask`` builds them. A consensus spectrum is
removed if any of its members is removed.
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from numba import njit

from . import paths
from .config import Config
from .similarity import modified_cosine_batch, pair_scores_batch
from .spectra import (STORE_NAMES, Spectrum, SpectrumStore, _take_into_kernel, clean_flat,
                      list_arrays_to_flat, merge_groups, open_matrix, open_query_vector)

__all__ = [
    "Library",
    "QuerySet",
    "prepare_queries",
    "instrument_family",
    "adduct_family",
    "polarity_sign",
    "fake_labels_from_test",
    "SEARCH_BY_MASS_COLUMNS",
    "SEARCH_OPEN_COLUMNS",
    "SPECTRUM_COLUMNS",
    "LIBRARY_DIR",
]

log = logging.getLogger(__name__)

LIBRARY_VERSION = 1
CONSENSUS_NAMES = ("consensus_mz.npy", "consensus_int.npy", "consensus_offsets.npy")
INSTRUMENT_FAMILIES = ("timstof", "qtof", "orbitrap", "other")

# Test adduct -> family base adduct (see module docstring).
ADDUCT_FAMILY: dict[str, str] = {
    "[M+H]+": "[M+H]+", "[M+NH4]+": "[M+H]+", "[M-H2O+H]+": "[M+H]+",
    "[M-2H2O+H]+": "[M+H]+", "[M+Na]+": "[M+Na]+", "[M+K]+": "[M+K]+",
    "[M-H]-": "[M-H]-", "[M-H2O-H]-": "[M-H]-", "[M+CH2O2-H]-": "[M-H]-",
    "[M+Cl]-": "[M+Cl]-",
}

SPECTRUM_COLUMNS = [
    "spec_idx", "spectrum_id", "struct_idx", "cons_idx", "ik14", "precursor_mz",
    "adduct", "adduct_family", "polarity", "instrument_family", "instrument_type",
    "ingest_lib", "collision_energy", "n_energies", "base_peak_intensity",
    "precursor_error_ppm", "prec_intensity", "n_peaks_raw", "n_peaks", "molecule_id",
]

SEARCH_BY_MASS_COLUMNS = [
    "struct_idx", "ik14", "smiles", "formula", "mono_mass", "mass_err_ppm",
    "n_refs", "n_refs_pol", "n_refs_tims", "has_tims", "n_cons",
    "cos_max", "cos_mean", "ent_max", "ent_mean",
    "cos_max_tims", "cos_mean_tims", "ent_max_tims", "ent_mean_tims",
    "cos_max_same_adduct", "cos_mean_same_adduct", "ent_max_same_adduct",
    "ent_mean_same_adduct",
    "cons_cos", "cons_ent", "cons_n_matched",
    "n_matched_best", "best_ref_idx", "best_query_idx", "best_ref_lib", "ce_dist_best",
]

SEARCH_OPEN_COLUMNS = [
    "rank", "cons_idx", "struct_idx", "ik14", "smiles", "mono_mass", "precursor_mz",
    "prec_diff", "adduct_family", "instrument_family", "ingest_lib", "n_members",
    "coarse", "mod_cos", "n_matched",
]

LIBRARY_DIR: Path = paths.PROCESSED / "library"

_ENVEDA_LIBS = ("enveda-180", "enveda-np-examples")
_LIB_DEFAULT_FAMILY = {"pluskal_ms2": "orbitrap"}
_RE_ORBI = re.compile(r"orbitrap|exactive|lumos|fusion|eclipse|astral|exploris|itft|qft|"
                      r"ltq.?ft|hybrid.?ft|ftms|ft-?icr|fticr|\bft\b")
_RE_QTOF = re.compile(r"q.?tof|tof|maxis|impact|synapt|xevo|triple.?tof|zeno|\b65\d\d\b|"
                      r"compact")


# --------------------------------------------------------------------------------------
# Small helpers: polarity, adduct family, instrument family
# --------------------------------------------------------------------------------------


def polarity_sign(ionization_mode: Any, adduct: Any = None) -> int:
    """+1 positive, -1 negative, 0 unknown (from the mode, else the adduct's last sign)."""
    if isinstance(ionization_mode, str):
        m = ionization_mode.strip().lower()
        if m.startswith("pos") or m == "+":
            return 1
        if m.startswith("neg") or m == "-":
            return -1
    if isinstance(adduct, str):
        s = adduct.strip()
        if s.endswith("+"):
            return 1
        if s.endswith("-"):
            return -1
    if isinstance(ionization_mode, (int, np.integer)) and int(ionization_mode) in (1, -1):
        return int(ionization_mode)
    return 0


@lru_cache(maxsize=65536)
def adduct_family(adduct: str | None, polarity: int = 0) -> str:
    """Adduct family name (see module docstring); "" if ``adduct`` is missing."""
    if not isinstance(adduct, str) or not adduct.strip():
        return ""
    from .adducts import match_test_adduct

    mode = "positive" if polarity > 0 else ("negative" if polarity < 0 else None)
    name = match_test_adduct(adduct, mode)
    if name is not None:
        return ADDUCT_FAMILY[name]
    return adduct.strip()


def _family_precursor(family: str, mass: float) -> float:
    """Precursor m/z of the family's base adduct at neutral mass; NaN if not a known family."""
    if family not in ADDUCT_FAMILY.values() or not math.isfinite(mass):
        return float("nan")
    from .adducts import TEST_ADDUCTS

    return float(TEST_ADDUCTS[family].precursor_mz(mass))


@lru_cache(maxsize=65536)
def instrument_family(instrument_type: str | None, ingest_lib: str | None = None) -> str:
    """Map free-text ``instrument_type`` (and the library) to one of INSTRUMENT_FAMILIES."""
    lib = ingest_lib if isinstance(ingest_lib, str) else ""
    if lib in _ENVEDA_LIBS:
        return "timstof"
    s = instrument_type.strip().lower() if isinstance(instrument_type, str) else ""
    if "tims" in s:
        return "timstof"
    if s and _RE_ORBI.search(s):
        return "orbitrap"
    if s and _RE_QTOF.search(s):
        return "qtof"
    return _LIB_DEFAULT_FAMILY.get(lib, "other")


def _map_unique(func, *cols: np.ndarray) -> np.ndarray:
    """Apply ``func`` once per unique tuple of values (None for missing)."""
    frame = pd.DataFrame({f"c{i}": pd.Series(c, dtype=object) for i, c in enumerate(cols)})
    frame = frame.astype(object).where(frame.notna(), None)
    keys = list(zip(*[frame[c].tolist() for c in frame.columns]))
    cache: dict = {}
    out = np.empty(len(keys), dtype=object)
    for i, k in enumerate(keys):
        v = cache.get(k)
        if v is None and k not in cache:
            v = func(*k)
            cache[k] = v
        out[i] = v
    return out


def _formula_mass_safe(formula: Any) -> float:
    from .adducts import formula_mass

    if not isinstance(formula, str) or not formula.strip():
        return float("nan")
    f = re.sub(r"[+-]\d*$", "", formula.strip())
    try:
        return float(formula_mass(f))
    except (ValueError, KeyError):
        return float("nan")


def _neutral_mass_safe(prec: float, adduct: Any, polarity: int) -> float:
    from .adducts import TEST_ADDUCTS, parse_adduct

    if not isinstance(adduct, str) or not math.isfinite(prec):
        return float("nan")
    a = TEST_ADDUCTS.get(adduct)
    if a is None:
        mode = "positive" if polarity > 0 else ("negative" if polarity < 0 else None)
        a = parse_adduct(adduct, mode)
    if a is None:
        return float("nan")
    return float(a.neutral_mass(prec))


def _rank_codes(values: Sequence | np.ndarray) -> np.ndarray:
    """Lexicographic rank codes of strings (missing first, as -1)."""
    codes, _ = pd.factorize(pd.Series(values, dtype=object), sort=True)
    return codes.astype(np.int64)


def _list_mean(col) -> tuple[np.ndarray, np.ndarray]:
    """Mean of finite values per list (NaN if none) and their count, for a pyarrow list column."""
    import pyarrow as pa

    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    n = len(col)
    if n == 0:
        return np.zeros(0), np.zeros(0, np.int16)
    off = np.asarray(col.offsets, dtype=np.int64)
    vals = np.asarray(col.values.to_numpy(zero_copy_only=False), dtype=np.float64)
    vals = vals[off[0]:off[-1]] if len(vals) >= off[-1] else vals
    off = off - off[0]
    valid = np.ones(n, dtype=bool)
    if col.null_count:
        valid = ~np.asarray(col.is_null().to_numpy(zero_copy_only=False))
    fin = np.isfinite(vals)
    v0 = np.where(fin, vals, 0.0)
    cs = np.concatenate([[0.0], np.cumsum(v0)])
    cc = np.concatenate([[0], np.cumsum(fin)])
    s = cs[off[1:]] - cs[off[:-1]]
    c = (cc[off[1:]] - cc[off[:-1]]) * valid
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    return mean, c.astype(np.int16)


# --------------------------------------------------------------------------------------
# Source iteration (streaming by row groups)
# --------------------------------------------------------------------------------------

_SOURCE_COLUMNS = [
    "spectrum_id", "ms2_mzs", "ms2_normalized_intensities", "precursor_mz", "adduct",
    "ionization_mode", "instrument_type", "ingest_lib", "collision_energy_ev",
    "base_peak_intensity", "precursor_error_ppm", "inchikey14", "ik14",
    "normalized_smiles", "smiles", "molecular_formula", "mono_mass", "instrument_family",
    "molecule_id",
]


def _iter_source(source: Any, batch_rows: int) -> Iterator[Any]:
    """Yield pyarrow RecordBatches / Tables from a DataFrame, Table, file, dir or list."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if isinstance(source, pd.DataFrame):
        source = pa.Table.from_pandas(source, preserve_index=False)
    if isinstance(source, pa.Table):
        cols = [c for c in _SOURCE_COLUMNS if c in source.column_names]
        yield from source.select(cols).to_batches(max_chunksize=batch_rows)
        return
    if isinstance(source, (str, Path)):
        p = Path(source)
        files = sorted(p.glob("*.parquet")) if p.is_dir() else [p]
    else:
        files = [Path(s) for s in source]
    if not files:
        raise FileNotFoundError(f"no parquet files in {source}")
    for f in files:
        pf = pq.ParquetFile(f)
        cols = [c for c in _SOURCE_COLUMNS if c in pf.schema_arrow.names]
        log.info("reading %s: %d rows, %d row groups", f, pf.metadata.num_rows,
                 pf.metadata.num_row_groups)
        yield from pf.iter_batches(batch_size=batch_rows, columns=cols)


def _col(batch, *names):
    for n in names:
        if n in batch.schema.names:
            return batch.column(n)
    return None


def _np_col(batch, n_rows: int, *names, dtype=np.float64) -> np.ndarray:
    c = _col(batch, *names)
    if c is None:
        return np.full(n_rows, np.nan, dtype=dtype)
    import pyarrow.compute as pc

    c = pc.cast(c, "float64")
    return np.asarray(c.to_numpy(zero_copy_only=False), dtype=dtype)


def _str_col(batch, n_rows: int, *names) -> np.ndarray:
    c = _col(batch, *names)
    if c is None:
        return np.full(n_rows, None, dtype=object)
    return np.asarray(c.to_pylist(), dtype=object)


# --------------------------------------------------------------------------------------
# Query sets
# --------------------------------------------------------------------------------------


@dataclass
class QuerySet:
    """Cleaned query spectra of one molecule plus its consensus spectra.

    Consensus: one per (polarity, adduct family) present, merged from those spectra, with
    reference precursor = the family's base adduct at ``M`` (``M`` given, else the median
    neutral mass of the spectra; median precursor m/z for unknown families).
    """

    store: SpectrumStore
    precursor_mz: np.ndarray
    polarity: np.ndarray               # int8, +1 / -1
    adduct: np.ndarray                 # object (str)
    adduct_family: np.ndarray          # object (str)
    instrument_family: np.ndarray      # object (str)
    collision_energy: np.ndarray       # float64, mean eV (NaN unknown)
    spectrum_id: np.ndarray            # object (str)
    cons_store: SpectrumStore = field(default_factory=SpectrumStore.empty)
    cons_polarity: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int8))
    cons_family: np.ndarray = field(default_factory=lambda: np.zeros(0, object))
    cons_precursor_mz: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cons_instrument_family: np.ndarray = field(default_factory=lambda: np.zeros(0, object))

    def __len__(self) -> int:
        return len(self.store)

    def consensus(self, polarity: int | str, family: str | None = None) -> Spectrum | None:
        """Consensus Spectrum for (polarity, family); family None: the first one of that
        polarity (base families [M+H]+ / [M-H]- sort first). None if absent."""
        pol = polarity_sign(polarity) if isinstance(polarity, str) else int(polarity)
        for i in range(len(self.cons_store)):
            if self.cons_polarity[i] == pol and (family is None or self.cons_family[i] == family):
                m, x = self.cons_store[i]
                return Spectrum(m, x, float(self.cons_precursor_mz[i]))
        return None

    @classmethod
    def from_arrays(cls, store: SpectrumStore, precursor_mz, polarity, adduct,
                    instrument_family_=None, collision_energy=None, spectrum_id=None,
                    M: float | None = None, cfg: Config | None = None) -> "QuerySet":
        """From already cleaned spectra (e.g. library rows)."""
        cfg = cfg or Config()
        n = len(store)
        prec = np.asarray(precursor_mz, dtype=np.float64).reshape(n)
        pol = np.asarray([polarity_sign(p) if isinstance(p, str) else int(p)
                          for p in np.asarray(polarity, dtype=object).reshape(n)], dtype=np.int8)
        add = np.asarray(adduct, dtype=object).reshape(n)
        fam = np.asarray([adduct_family(a, int(p)) for a, p in zip(add, pol)], dtype=object)
        inst = (np.full(n, "timstof", dtype=object) if instrument_family_ is None
                else np.asarray(instrument_family_, dtype=object).reshape(n))
        ce = (np.full(n, np.nan) if collision_energy is None
              else np.asarray(collision_energy, dtype=np.float64).reshape(n))
        sid = (np.asarray([str(i) for i in range(n)], dtype=object) if spectrum_id is None
               else np.asarray(spectrum_id, dtype=object).reshape(n))
        qs = cls(store, prec, pol, add, fam, inst, ce, sid)
        qs._build_consensus(M, cfg)
        return qs

    def _build_consensus(self, M: float | None, cfg: Config) -> None:
        n = len(self.store)
        if n == 0:
            return
        if M is None or not math.isfinite(M):
            ms = np.array([_neutral_mass_safe(p, a, int(s)) for p, a, s in
                           zip(self.precursor_mz, self.adduct, self.polarity)])
            M = float(np.nanmedian(ms)) if np.isfinite(ms).any() else float("nan")
        keys = sorted({(int(-p), f) for p, f in zip(self.polarity, self.adduct_family)},
                      key=lambda k: (k[0], _family_sort_key(k[1])))
        members, goff, gprec, gpol, gfam, ginst = [], [0], [], [], [], []
        for negpol, fam in keys:
            pol = -negpol
            idx = np.flatnonzero((self.polarity == pol) & (self.adduct_family == fam))
            members.append(idx)
            goff.append(goff[-1] + len(idx))
            ref = _family_precursor(fam, M)
            if not math.isfinite(ref):
                ref = float(np.median(self.precursor_mz[idx]))
            gprec.append(ref)
            gpol.append(pol)
            gfam.append(fam)
            insts = sorted(set(self.instrument_family[idx].tolist()))
            ginst.append(insts[0] if len(insts) == 1 else "other")
        self.cons_store = merge_groups(self.store, np.concatenate(members),
                                       np.asarray(goff, dtype=np.int64), np.asarray(gprec), cfg)
        self.cons_polarity = np.asarray(gpol, dtype=np.int8)
        self.cons_family = np.asarray(gfam, dtype=object)
        self.cons_precursor_mz = np.asarray(gprec, dtype=np.float64)
        self.cons_instrument_family = np.asarray(ginst, dtype=object)


def _family_sort_key(fam: str) -> tuple:
    base = ("[M+H]+", "[M-H]-", "[M+Na]+", "[M+K]+", "[M+Cl]-")
    return (base.index(fam), "") if fam in base else (len(base), fam)


def prepare_queries(df: pd.DataFrame, cfg: Config | None = None, M: float | None = None
                    ) -> QuerySet:
    """QuerySet from rows in the ``test.parquet`` schema (one molecule's spectra).

    Peaks are cleaned with ``spectra.clean`` (PLAN.md 4.1). ``instrument_type`` missing
    means timsTOF (the test instrument).
    """
    import pyarrow as pa

    cfg = cfg or Config()
    tab = pa.Table.from_pandas(df.reset_index(drop=True), preserve_index=False)
    n = tab.num_rows
    mz, it, off, valid = list_arrays_to_flat(tab.column("ms2_mzs"),
                                             tab.column("ms2_normalized_intensities"))
    prec = _np_col(tab, n, "precursor_mz")
    cm, ci, co, _, _ = clean_flat(mz, it, off, prec, cfg, valid)
    store = SpectrumStore(cm, ci, co)
    adduct = _str_col(tab, n, "adduct")
    mode = _str_col(tab, n, "ionization_mode")
    pol = np.array([polarity_sign(m, a) for m, a in zip(mode, adduct)], dtype=np.int8)
    itype = _str_col(tab, n, "instrument_type")
    lib = _str_col(tab, n, "ingest_lib")
    inst = np.array([instrument_family(t, l) if (t is not None or l is not None) else "timstof"
                     for t, l in zip(itype, lib)], dtype=object)
    if "collision_energy_ev" in tab.column_names:
        ce, _ = _list_mean(tab.column("collision_energy_ev"))
    else:
        ce = np.full(n, np.nan)
    sid = _str_col(tab, n, "spectrum_id")
    return QuerySet.from_arrays(store, prec, pol, adduct, inst, ce, sid, M=M, cfg=cfg)


# --------------------------------------------------------------------------------------
# Library
# --------------------------------------------------------------------------------------


@njit(cache=True)
def _expand_ranges(starts, ends):
    n = 0
    for i in range(starts.shape[0]):
        n += ends[i] - starts[i]
    out = np.empty(n, dtype=np.int64)
    owner = np.empty(n, dtype=np.int64)
    k = 0
    for i in range(starts.shape[0]):
        for j in range(starts[i], ends[i]):
            out[k] = j
            owner[k] = i
            k += 1
    return out, owner


def _agg_max_mean(key: np.ndarray, val: np.ndarray, nq: int, nc: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Per candidate: max over queries and mean over queries of (max over its refs)."""
    if key.size == 0:
        return np.full(nc, np.nan), np.full(nc, np.nan)
    m = np.full(nq * nc, -np.inf)
    np.maximum.at(m, key, val)
    m = m.reshape(nq, nc)
    have = np.isfinite(m)
    cnt = have.sum(axis=0)
    mx = np.where(cnt > 0, np.where(have, m, -np.inf).max(axis=0), np.nan)
    mean = np.where(cnt > 0, np.where(have, m, 0.0).sum(axis=0) / np.maximum(cnt, 1), np.nan)
    return mx, mean


class Library:
    """A built spectral library (see the module docstring for files and layout)."""

    def __init__(self, directory: Path | str, cfg: Config | None = None, mmap: bool = True):
        self.dir = Path(directory)
        self.cfg = cfg or Config()
        info_path = self.dir / "library_info.json"
        self.info: dict = json.loads(info_path.read_text()) if info_path.exists() else {}
        self.store = SpectrumStore.load(self.dir, STORE_NAMES, mmap=mmap)
        self.cons_store = SpectrumStore.load(self.dir, CONSENSUS_NAMES, mmap=mmap)
        self.meta = pd.read_parquet(self.dir / "meta.parquet")
        self.structures = pd.read_parquet(self.dir / "structures.parquet")
        self.cons_meta = pd.read_parquet(self.dir / "consensus_meta.parquet")
        self._open: dict[int, Any] = {}
        self._check_cfg()
        # Hot arrays.
        m = self.meta
        self.spec_struct = m["struct_idx"].to_numpy(np.int64)
        self.spec_cons = m["cons_idx"].to_numpy(np.int64)
        self.spec_pol = m["polarity"].to_numpy(np.int8)
        self.spec_prec = m["precursor_mz"].to_numpy(np.float64)
        self.spec_ce = m["collision_energy"].to_numpy(np.float64)
        self.spec_tims = (m["instrument_family"] == "timstof").to_numpy(bool)
        self.spec_adduct = m["adduct"].astype(object).to_numpy()
        self.spec_lib = m["ingest_lib"].astype(object).to_numpy()
        self.spec_inst = m["instrument_family"].astype(object).to_numpy()
        self.spec_id = m["spectrum_id"].astype(object).to_numpy()
        s = self.structures
        self.struct_ik14 = s["ik14"].astype(object).to_numpy()
        self.struct_smiles = s["smiles"].astype(object).to_numpy()
        self.struct_formula = s["formula"].astype(object).to_numpy()
        self.struct_mass = s["mono_mass"].to_numpy(np.float64)
        self.struct_spec = np.stack([s["spec_start"].to_numpy(np.int64),
                                     s["spec_end"].to_numpy(np.int64)], axis=1)
        self.struct_cons = np.stack([s["cons_start"].to_numpy(np.int64),
                                     s["cons_end"].to_numpy(np.int64)], axis=1)
        c = self.cons_meta
        self.cons_struct = c["struct_idx"].to_numpy(np.int64)
        self.cons_pol = c["polarity"].to_numpy(np.int8)
        self.cons_family = c["adduct_family"].astype(object).to_numpy()
        self.cons_prec = c["precursor_mz"].to_numpy(np.float64)
        self.cons_tims = (c["instrument_family"] == "timstof").to_numpy(bool)
        self.cons_open_row = c["open_row"].to_numpy(np.int64)
        self.cons_inst = c["instrument_family"].astype(object).to_numpy()
        self.cons_lib = c["ingest_lib"].astype(object).to_numpy()
        self.cons_n_members = c["n_members"].to_numpy(np.int64)

    # ---- loading ---------------------------------------------------------------------

    @classmethod
    def load(cls, directory: Path | str | None = None, cfg: Config | None = None,
             mmap: bool = True) -> "Library":
        """Open a built library; peak arrays are memory-mapped (``mmap=True``)."""
        return cls(directory if directory is not None else LIBRARY_DIR, cfg, mmap)

    def _check_cfg(self) -> None:
        built = self.info.get("clean_cfg", {})
        now = _clean_cfg_dict(self.cfg)
        diff = {k: (built.get(k), v) for k, v in now.items() if k in built and built[k] != v}
        if diff:
            log.warning("library %s was built with other cleaning settings (built, now): %s",
                        self.dir, diff)

    @property
    def n_spectra(self) -> int:
        return len(self.store)

    @property
    def n_structures(self) -> int:
        return len(self.structures)

    @property
    def n_consensus(self) -> int:
        return len(self.cons_store)

    def spectrum(self, i: int) -> Spectrum:
        m, x = self.store[i]
        return Spectrum(m, x, float(self.spec_prec[i]))

    def consensus_spectrum(self, i: int) -> Spectrum:
        m, x = self.cons_store[i]
        return Spectrum(m, x, float(self.cons_prec[i]))

    def open_index(self, polarity: int | str):
        """CSR open-search matrix for a polarity (loaded on first use)."""
        import scipy.sparse as sp

        pol = polarity_sign(polarity) if isinstance(polarity, str) else int(polarity)
        if pol not in self._open:
            name = "open_index_pos.npz" if pol > 0 else "open_index_neg.npz"
            self._open[pol] = sp.load_npz(self.dir / name).tocsr()
        return self._open[pol]

    # ---- masks -----------------------------------------------------------------------

    def spectra_mask(self, ik14: Iterable[str] | None = None,
                     ingest_lib: Iterable[str] | None = None,
                     spectrum_id: Iterable[str] | None = None) -> np.ndarray:
        """Boolean mask over store rows: True where the spectrum's structure is in ``ik14``
        (all structures if None) AND its library is in ``ingest_lib`` (all if None), OR its
        id is in ``spectrum_id``. Use as ``exclude_mask``."""
        mask = np.zeros(self.n_spectra, dtype=bool)
        if ik14 is not None or ingest_lib is not None:
            sel = np.ones(self.n_spectra, dtype=bool)
            if ik14 is not None:
                keys = set(ik14)
                sidx = np.flatnonzero(self.structures["ik14"].isin(keys).to_numpy())
                hit = np.zeros(self.n_structures, dtype=bool)
                hit[sidx] = True
                sel &= hit[self.spec_struct]
            if ingest_lib is not None:
                sel &= self.meta["ingest_lib"].isin(set(ingest_lib)).to_numpy()
            mask |= sel
        if spectrum_id is not None:
            mask |= self.meta["spectrum_id"].isin(set(spectrum_id)).to_numpy()
        return mask

    def consensus_mask(self, exclude_mask: np.ndarray | None) -> np.ndarray:
        """Consensus rows removed by a spectrum mask (any member removed)."""
        ex = np.zeros(self.n_consensus, dtype=bool)
        if exclude_mask is not None:
            em = np.asarray(exclude_mask, dtype=bool)
            if em.shape != (self.n_spectra,):
                raise ValueError("exclude_mask must have one entry per library spectrum")
            ex[self.spec_cons[em]] = True
        return ex

    # ---- queries ---------------------------------------------------------------------

    def queries_from_library(self, spec_idx: Sequence[int], M: float | None = None) -> QuerySet:
        """QuerySet made of library spectra (self-search tests, simulated queries)."""
        idx = np.asarray(spec_idx, dtype=np.int64)
        return QuerySet.from_arrays(self.store.take(idx), self.spec_prec[idx],
                                    self.spec_pol[idx], self.spec_adduct[idx],
                                    self.spec_inst[idx], self.spec_ce[idx], self.spec_id[idx],
                                    M=M, cfg=self.cfg)

    def _tolerances(self, q_tims: bool, ref_tims: np.ndarray) -> np.ndarray:
        same, cross = float(self.cfg.frag_tol_same_da), float(self.cfg.frag_tol_cross_da)
        return np.where(ref_tims & q_tims, same, cross)

    # ---- mass-restricted search (PLAN.md 4.2) ----------------------------------------

    def candidates_in_window(self, M: float, tol_ppm: float | None = None,
                             min_da: float | None = None) -> np.ndarray:
        """struct_idx of structures with |mono_mass - M| <= max(tol_ppm * M, min_da)."""
        if M is None or not math.isfinite(M):
            return np.zeros(0, dtype=np.int64)
        tol_ppm = self.cfg.mass_tol_ppm if tol_ppm is None else tol_ppm
        min_da = self.cfg.mass_tol_min_da if min_da is None else min_da
        tol = max(tol_ppm * 1e-6 * M, min_da)
        lo = np.searchsorted(self.struct_mass, M - tol, side="left")
        hi = np.searchsorted(self.struct_mass, M + tol, side="right")
        return np.arange(lo, hi, dtype=np.int64)

    def search_by_mass(self, M: float, tol_ppm: float | None = None,
                       polarity: int | str | None = None,
                       exclude_mask: np.ndarray | None = None,
                       queries: QuerySet | pd.DataFrame | None = None,
                       min_da: float | None = None) -> pd.DataFrame:
        """LIB candidates within the mass window and their spectral features (PLAN.md 4.2).

        One row per structure with at least one non-excluded reference spectrum, columns
        ``SEARCH_BY_MASS_COLUMNS``, sorted by cos_max desc (NaN last), |mass_err_ppm| asc,
        ik14 asc. With ``queries`` None only the structural columns and counts are filled.
        Pairs are (query spectrum, reference) of the same polarity; ``polarity`` restricts
        queries and references to one polarity. Tolerance: ``frag_tol_same_da`` when query
        and reference are both timsTOF, else ``frag_tol_cross_da``.

        Features: cos_* / ent_* = max (or mean over query spectra) of the best reference
        per query spectrum, over all refs, timsTOF refs, and same-adduct refs; cons_* =
        best query consensus vs reference consensus of the same (polarity, family);
        n_matched_best, best_ref_idx, best_query_idx, best_ref_lib, ce_dist_best describe
        the best cosine pair (ce_dist in eV, NaN if unknown).
        """
        cand = self.candidates_in_window(M, tol_ppm, min_da)
        pol_filter = (None if polarity is None else
                      (polarity_sign(polarity) if isinstance(polarity, str) else int(polarity)))
        ex = None if exclude_mask is None else np.asarray(exclude_mask, dtype=bool)
        if ex is not None and ex.shape != (self.n_spectra,):
            raise ValueError("exclude_mask must have one entry per library spectrum")
        if cand.size == 0:
            return pd.DataFrame({c: [] for c in SEARCH_BY_MASS_COLUMNS})
        refs, owner = _expand_ranges(self.struct_spec[cand, 0], self.struct_spec[cand, 1])
        if ex is not None:
            keep = ~ex[refs]
            refs, owner = refs[keep], owner[keep]
        nc = cand.size
        n_refs = np.bincount(owner, minlength=nc)
        present = n_refs > 0
        if isinstance(queries, pd.DataFrame):
            queries = prepare_queries(queries, self.cfg, M=M)
        q_pols = ({1, -1} if queries is None else set(int(p) for p in queries.polarity))
        if pol_filter is not None:
            q_pols &= {pol_filter}
        ref_pol = self.spec_pol[refs]
        in_pol = np.isin(ref_pol, list(q_pols)) if q_pols else np.zeros(refs.size, bool)
        cols: dict[str, Any] = {}
        cols["n_refs"] = n_refs
        cols["n_refs_pol"] = np.bincount(owner[in_pol], minlength=nc)
        cols["n_refs_tims"] = np.bincount(owner[self.spec_tims[refs]], minlength=nc)
        cols["has_tims"] = cols["n_refs_tims"] > 0
        cons_ex = self.consensus_mask(ex) if ex is not None else None
        cr, cown = _expand_ranges(self.struct_cons[cand, 0], self.struct_cons[cand, 1])
        if cons_ex is not None:
            k = ~cons_ex[cr]
            cr, cown = cr[k], cown[k]
        cols["n_cons"] = np.bincount(cown, minlength=nc)
        nan = np.full(nc, np.nan)
        for name in SEARCH_BY_MASS_COLUMNS[11:]:
            cols[name] = nan.copy()
        cols["best_ref_idx"] = np.full(nc, -1, np.int64)
        cols["best_query_idx"] = np.full(nc, -1, np.int64)
        cols["n_matched_best"] = np.zeros(nc, np.int64)
        cols["cons_n_matched"] = np.zeros(nc, np.int64)
        cols["best_ref_lib"] = np.full(nc, None, object)
        if queries is not None and len(queries) and in_pol.any():
            self._spectrum_features(queries, pol_filter, refs, owner, nc, cols)
            self._consensus_features(queries, pol_filter, cr, cown, nc, cols)
        cols["struct_idx"] = cand
        cols["ik14"] = self.struct_ik14[cand]
        cols["smiles"] = self.struct_smiles[cand]
        cols["formula"] = self.struct_formula[cand]
        cols["mono_mass"] = self.struct_mass[cand]
        cols["mass_err_ppm"] = (self.struct_mass[cand] - M) / M * 1e6
        cm = cols["cos_max"]
        order = np.lexsort((cols["ik14"].astype(str), np.abs(cols["mass_err_ppm"]),
                            np.where(np.isnan(cm), np.inf, -cm)))
        order = order[present[order]]
        return pd.DataFrame({c: np.asarray(cols[c])[order] for c in SEARCH_BY_MASS_COLUMNS})

    def _spectrum_features(self, qs: QuerySet, pol_filter, refs, owner, nc, cols) -> None:
        pq_list, pr_list, pc_list, tol_list = [], [], [], []
        ref_pol = self.spec_pol[refs]
        for q in range(len(qs)):
            p = int(qs.polarity[q])
            if (pol_filter is not None and p != pol_filter) or qs.store.n_peaks[q] == 0:
                continue
            sel = np.flatnonzero(ref_pol == p)
            if sel.size == 0:
                continue
            pq_list.append(np.full(sel.size, q, np.int64))
            pr_list.append(refs[sel])
            pc_list.append(owner[sel])
            tol_list.append(self._tolerances(qs.instrument_family[q] == "timstof",
                                             self.spec_tims[refs[sel]]))
        if not pq_list:
            return
        pq_, pr_ = np.concatenate(pq_list), np.concatenate(pr_list)
        pc_, tols = np.concatenate(pc_list), np.concatenate(tol_list)
        cos, ncos, ent, _ = pair_scores_batch(
            qs.store.mz, qs.store.intensity, qs.store.offsets, self.store.mz,
            self.store.intensity, self.store.offsets, pq_, pr_, tols,
            weighted=self.cfg.entropy_weighted)
        nq = len(qs)
        key = pq_ * nc + pc_
        subsets = {
            "": np.ones(pq_.size, bool),
            "_tims": self.spec_tims[pr_],
            "_same_adduct": self.spec_adduct[pr_] == qs.adduct[pq_],
        }
        for suffix, msk in subsets.items():
            for stem, val in (("cos", cos), ("ent", ent)):
                mx, mean = _agg_max_mean(key[msk], val[msk], nq, nc)
                cols[f"{stem}_max{suffix}"] = mx
                cols[f"{stem}_mean{suffix}"] = mean
        # Best cosine pair per candidate: cos desc, ent desc, ref asc, query asc.
        order = np.lexsort((pq_, pr_, -ent, -cos, pc_))
        first = np.ones(order.size, bool)
        first[1:] = pc_[order][1:] != pc_[order][:-1]
        b = order[first]
        c = pc_[b]
        cols["n_matched_best"][c] = ncos[b]
        cols["best_ref_idx"][c] = pr_[b]
        cols["best_query_idx"][c] = pq_[b]
        cols["best_ref_lib"][c] = self.spec_lib[pr_[b]]
        cols["ce_dist_best"][c] = np.abs(qs.collision_energy[pq_[b]] - self.spec_ce[pr_[b]])

    def _consensus_features(self, qs: QuerySet, pol_filter, cr, cown, nc, cols) -> None:
        pq_list, pr_list, pc_list, tol_list = [], [], [], []
        for q in range(len(qs.cons_store)):
            p = int(qs.cons_polarity[q])
            if pol_filter is not None and p != pol_filter:
                continue
            sel = np.flatnonzero((self.cons_pol[cr] == p) & (self.cons_family[cr] == qs.cons_family[q]))
            if sel.size == 0:
                continue
            pq_list.append(np.full(sel.size, q, np.int64))
            pr_list.append(cr[sel])
            pc_list.append(cown[sel])
            tol_list.append(self._tolerances(qs.cons_instrument_family[q] == "timstof",
                                             self.cons_tims[cr[sel]]))
        if not pq_list:
            return
        pq_, pr_ = np.concatenate(pq_list), np.concatenate(pr_list)
        pc_, tols = np.concatenate(pc_list), np.concatenate(tol_list)
        cs = self.cons_store
        cos, ncos, ent, _ = pair_scores_batch(
            qs.cons_store.mz, qs.cons_store.intensity, qs.cons_store.offsets, cs.mz,
            cs.intensity, cs.offsets, pq_, pr_, tols, weighted=self.cfg.entropy_weighted)
        m = np.full(nc, -np.inf)
        np.maximum.at(m, pc_, cos)
        e = np.full(nc, -np.inf)
        np.maximum.at(e, pc_, ent)
        cols["cons_cos"] = np.where(np.isfinite(m), m, np.nan)
        cols["cons_ent"] = np.where(np.isfinite(e), e, np.nan)
        order = np.lexsort((pr_, -cos, pc_))
        first = np.ones(order.size, bool)
        first[1:] = pc_[order][1:] != pc_[order][:-1]
        b = order[first]
        cols["cons_n_matched"][pc_[b]] = ncos[b]

    # ---- open search (PLAN.md 4.3) ---------------------------------------------------

    def search_open(self, query: Spectrum | tuple, polarity: int | str,
                    top_k: int | None = None, exclude_mask: np.ndarray | None = None,
                    family: str | None = None, query_is_tims: bool = True,
                    coarse_top: int | None = None, unique_structures: bool | None = None,
                    exclude_cons_mask: np.ndarray | None = None) -> pd.DataFrame:
        """Analog hits for a (consensus) query spectrum, no mass restriction.

        Stage 1: dot product of the binned query (``spectra.open_vectors``) with every
        library consensus of the polarity (and of adduct ``family`` if given); excluded
        consensus rows and rows with coarse score 0 are skipped; the best
        ``analog_coarse_top`` go to stage 2. Stage 2: exact modified cosine (query
        precursor vs the consensus reference precursor). Sorted by mod_cos desc, coarse
        desc, cons_idx asc; with ``unique_structures`` the best consensus per structure is
        kept. Returns up to ``top_k`` (default ``analog_top_k``) rows, columns
        ``SEARCH_OPEN_COLUMNS``; ``prec_diff`` = query precursor - hit precursor.
        """
        cfg = self.cfg
        top_k = cfg.analog_top_k if top_k is None else int(top_k)
        coarse_top = cfg.analog_coarse_top if coarse_top is None else int(coarse_top)
        uniq = cfg.analog_unique_structures if unique_structures is None else unique_structures
        pol = polarity_sign(polarity) if isinstance(polarity, str) else int(polarity)
        q_mz = np.asarray(query[0], dtype=np.float32)
        q_int = np.asarray(query[1], dtype=np.float32)
        q_prec = float(query[2]) if len(query) > 2 else float("nan")
        empty = pd.DataFrame({c: [] for c in SEARCH_OPEN_COLUMNS})
        rows = np.flatnonzero(self.cons_pol == pol)        # consensus order == open_row order
        if rows.size == 0 or q_mz.size == 0:
            return empty
        R = self.open_index(pol)
        qv = open_query_vector(q_mz, q_int, q_prec, cfg)
        coarse = np.asarray(R @ qv).ravel()
        ok = coarse > 0
        if family is not None:
            ok &= self.cons_family[rows] == family
        if exclude_mask is not None or exclude_cons_mask is not None:
            cex = (self.consensus_mask(exclude_mask) if exclude_mask is not None
                   else np.zeros(self.n_consensus, bool))
            if exclude_cons_mask is not None:
                cex |= np.asarray(exclude_cons_mask, dtype=bool)
            ok &= ~cex[rows]
        cand = np.flatnonzero(ok)
        if cand.size == 0:
            return empty
        if cand.size > coarse_top:
            part = np.argpartition(-coarse[cand], coarse_top - 1)[:coarse_top]
            thr = coarse[cand[part]].min()
            cand = cand[coarse[cand] >= thr]          # keep ties at the threshold
        o = np.lexsort((cand, -coarse[cand]))[:coarse_top]
        cand = cand[o]
        cons = rows[cand]
        tols = self._tolerances(query_is_tims, self.cons_tims[cons])
        cs = self.cons_store
        mc, nm = modified_cosine_batch(q_mz, q_int, q_prec, cs.mz, cs.intensity, cs.offsets,
                                       self.cons_prec, cons, tols)
        co = coarse[cand].astype(np.float64)
        o = np.lexsort((cons, -co, -mc))
        cons, co, mc, nm = cons[o], co[o], mc[o], nm[o]
        struct = self.cons_struct[cons]
        if uniq:
            _, first = np.unique(struct, return_index=True)
            keep = np.sort(first)
            cons, co, mc, nm, struct = cons[keep], co[keep], mc[keep], nm[keep], struct[keep]
        k = slice(0, top_k)
        cons, co, mc, nm, struct = cons[k], co[k], mc[k], nm[k], struct[k]
        out = {
            "rank": np.arange(1, cons.size + 1), "cons_idx": cons, "struct_idx": struct,
            "ik14": self.struct_ik14[struct], "smiles": self.struct_smiles[struct],
            "mono_mass": self.struct_mass[struct], "precursor_mz": self.cons_prec[cons],
            "prec_diff": q_prec - self.cons_prec[cons],
            "adduct_family": self.cons_family[cons],
            "instrument_family": self.cons_inst[cons], "ingest_lib": self.cons_lib[cons],
            "n_members": self.cons_n_members[cons], "coarse": co, "mod_cos": mc,
            "n_matched": nm,
        }
        return pd.DataFrame(out, columns=SEARCH_OPEN_COLUMNS)

    # ---- build -----------------------------------------------------------------------

    @classmethod
    def build(cls, source: Any, cfg: Config | None = None, out_dir: Path | str | None = None,
              batch_rows: int | None = None) -> "Library":
        """Build the library from a spectra table and return it loaded.

        ``source``: a parquet file, a directory of parquet files, a list of files, a
        pandas DataFrame or a pyarrow Table with the columns in the module docstring.
        Files are streamed in batches of ``batch_rows`` (default ``lib_build_batch_rows``);
        cleaned peaks go to temporary float32 files on disk, so RAM holds only the
        per-spectrum metadata (about 0.5 KB per spectrum) plus one batch. The library is
        written to ``out_dir`` (default ``paths.PROCESSED / "library"``) via a temporary
        sibling directory that replaces it at the end.
        """
        cfg = cfg or Config()
        batch_rows = int(batch_rows or cfg.lib_build_batch_rows)
        out = Path(out_dir) if out_dir is not None else LIBRARY_DIR
        tmp = out.parent / (out.name + ".building")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        t0 = time.perf_counter()
        _Builder(cfg, tmp, batch_rows).run(source)
        if out.exists():
            shutil.rmtree(out)
        tmp.rename(out)
        log.info("library written to %s in %.1f s", out, time.perf_counter() - t0)
        return cls.load(out, cfg)


def _rss_mb() -> tuple[float, float]:
    """(current RSS, peak RSS) of this process in MB (Linux /proc; NaN elsewhere)."""
    try:
        txt = Path("/proc/self/status").read_text()
    except OSError:
        return float("nan"), float("nan")
    vals = {}
    for line in txt.splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            k, v = line.split(":", 1)
            vals[k] = float(v.split()[0]) / 1024
    return vals.get("VmRSS", float("nan")), vals.get("VmHWM", float("nan"))


def _clean_cfg_dict(cfg: Config) -> dict:
    keys = ("clean_max_above_precursor_da", "keep_precursor", "precursor_window_low_da",
            "precursor_window_high_da", "intensity_floor", "top_n_peaks",
            "consensus_merge_da", "analog_bin_da", "analog_max_mz", "lib_min_peaks")
    return {k: getattr(cfg, k) for k in keys}


class _Builder:
    """Streaming library build (see Library.build)."""

    def __init__(self, cfg: Config, out: Path, batch_rows: int):
        self.cfg = cfg
        self.out = out
        self.batch_rows = batch_rows
        self.drops: dict[str, int] = {}

    def run(self, source: Any) -> None:
        t0 = time.perf_counter()
        meta, n_peaks_total = self._pass1(source)
        t1 = time.perf_counter()
        log.info("pass 1 (read + clean): %d spectra, %d peaks, %.1f s", len(meta),
                 n_peaks_total, t1 - t0)
        log.info("  RSS %.0f MB, peak %.0f MB", *_rss_mb())
        meta, structures = self._index(meta)
        self._write_spectra(meta)
        t2 = time.perf_counter()
        log.info("spectra sorted and written: %d spectra, %d structures, %.1f s", len(meta),
                 len(structures), t2 - t1)
        log.info("  RSS %.0f MB, peak %.0f MB", *_rss_mb())
        cons_meta = self._consensus(meta, structures)
        t3 = time.perf_counter()
        log.info("consensus: %d spectra, %.1f s", len(cons_meta), t3 - t2)
        log.info("  RSS %.0f MB, peak %.0f MB", *_rss_mb())
        self._open_index(cons_meta)
        t4 = time.perf_counter()
        log.info("open index: %.1f s", t4 - t3)
        log.info("  RSS %.0f MB, peak %.0f MB", *_rss_mb())
        meta = meta.rename(columns={"row": "source_row"})
        meta[SPECTRUM_COLUMNS + ["source_row"]].to_parquet(self.out / "meta.parquet",
                                                           index=False)
        structures.to_parquet(self.out / "structures.parquet", index=False)
        cons_meta.to_parquet(self.out / "consensus_meta.parquet", index=False)
        info = {
            "version": LIBRARY_VERSION,
            "n_spectra": int(len(meta)),
            "n_structures": int(len(structures)),
            "n_consensus": int(len(cons_meta)),
            "n_peaks": int(meta["n_peaks"].sum()),
            "n_consensus_peaks": int(np.load(self.out / CONSENSUS_NAMES[2], mmap_mode="r")[-1]),
            "n_input_rows": int(self.n_input),
            "drops": self.drops,
            "clean_cfg": _clean_cfg_dict(self.cfg),
            "build_seconds": round(time.perf_counter() - t0, 1),
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (self.out / "library_info.json").write_text(json.dumps(info, indent=2))
        for f in ("_tmp_mz.f32", "_tmp_int.f32"):
            (self.out / f).unlink(missing_ok=True)

    # pass 1: stream, clean, spill peaks to disk, keep metadata
    def _pass1(self, source: Any) -> tuple[pd.DataFrame, int]:
        import pyarrow as pa

        cfg = self.cfg
        parts: list[pa.Table] = []
        n_rows = 0
        n_peaks = 0
        with open(self.out / "_tmp_mz.f32", "wb") as fm, open(self.out / "_tmp_int.f32", "wb") as fi:
            for batch in _iter_source(source, self.batch_rows):
                n = batch.num_rows
                if n == 0:
                    continue
                mz, it, off, valid = list_arrays_to_flat(batch.column("ms2_mzs"),
                                                         batch.column("ms2_normalized_intensities"))
                prec = _np_col(batch, n, "precursor_mz")
                cm, ci, co, pint, nraw = clean_flat(mz, it, off, prec, cfg, valid)
                fm.write(cm.tobytes())
                fi.write(ci.tobytes())
                ce, nce = (_list_mean(batch.column("collision_energy_ev"))
                           if "collision_energy_ev" in batch.schema.names
                           else (np.full(n, np.nan), np.zeros(n, np.int16)))
                cols: dict[str, Any] = {
                    "row": np.arange(n_rows, n_rows + n, dtype=np.int64),
                    "spectrum_id": (_str_col(batch, n, "spectrum_id")
                                    if _col(batch, "spectrum_id") is not None
                                    else np.array([str(i) for i in range(n_rows, n_rows + n)],
                                                  dtype=object)),
                    "ik14": _str_col(batch, n, "inchikey14", "ik14"),
                    "smiles": _str_col(batch, n, "normalized_smiles", "smiles"),
                    "formula": _str_col(batch, n, "molecular_formula"),
                    "precursor_mz": prec,
                    "adduct": _str_col(batch, n, "adduct"),
                    "ionization_mode": _str_col(batch, n, "ionization_mode"),
                    "instrument_type": _str_col(batch, n, "instrument_type"),
                    "instrument_family": _str_col(batch, n, "instrument_family"),
                    "ingest_lib": _str_col(batch, n, "ingest_lib"),
                    "collision_energy": ce,
                    "n_energies": nce,
                    "base_peak_intensity": _np_col(batch, n, "base_peak_intensity"),
                    "precursor_error_ppm": _np_col(batch, n, "precursor_error_ppm"),
                    "in_mono_mass": _np_col(batch, n, "mono_mass"),
                    "molecule_id": _str_col(batch, n, "molecule_id"),
                    "prec_intensity": pint.astype(np.float32),
                    "n_peaks_raw": nraw,
                    "n_peaks": np.diff(co).astype(np.int32),
                }
                parts.append(pa.table({k: pa.array(v, from_pandas=True) if v.dtype == object
                                       else pa.array(v) for k, v in cols.items()}))
                n_rows += n
                n_peaks += int(co[-1])
                if len(parts) % 20 == 0:
                    log.info("  %d rows read, %d clean peaks", n_rows, n_peaks)
        self.n_input = n_rows
        if not parts:
            raise ValueError("empty source")
        # permissive: a batch whose text column is all null (e.g. drug_plus
        # instrument_type) arrives typed as null and must unify with string.
        meta = pa.concat_tables(parts, promote_options="permissive").to_pandas()
        off = np.zeros(n_rows + 1, dtype=np.int64)
        np.cumsum(meta["n_peaks"].to_numpy(np.int64), out=off[1:])
        np.save(self.out / "_tmp_offsets.npy", off)
        return meta, n_peaks

    def _drop(self, meta: pd.DataFrame, bad: np.ndarray, reason: str) -> pd.DataFrame:
        k = int(bad.sum())
        if k:
            self.drops[reason] = self.drops.get(reason, 0) + k
            log.info("dropped %d spectra: %s", k, reason)
        return meta[~bad]

    # structure table, families, sort order
    def _index(self, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        cfg = self.cfg
        ik = meta["ik14"]
        meta = self._drop(meta, (ik.isna() | (ik.fillna("").astype(str).str.len() == 0)
                                 ).to_numpy(), "missing ik14")
        meta = self._drop(meta, ~np.isfinite(meta["precursor_mz"].to_numpy()),
                          "missing precursor_mz")
        pol = _map_unique(polarity_sign, meta["ionization_mode"].to_numpy(object),
                          meta["adduct"].to_numpy(object)).astype(np.int8)
        meta = meta.assign(polarity=pol)
        meta = self._drop(meta, pol == 0, "unknown polarity")
        meta = self._drop(meta, meta["n_peaks"].to_numpy() < cfg.lib_min_peaks,
                          f"fewer than {cfg.lib_min_peaks} peaks after cleaning")
        meta = meta.reset_index(drop=True)
        meta["adduct_family"] = _map_unique(lambda a, p: adduct_family(a, int(p)),
                                            meta["adduct"].to_numpy(object),
                                            meta["polarity"].to_numpy())
        fam_given = meta["instrument_family"]
        mapped = _map_unique(instrument_family, meta["instrument_type"].to_numpy(object),
                             meta["ingest_lib"].to_numpy(object))
        meta["instrument_family"] = np.where(fam_given.notna(), fam_given.to_numpy(object), mapped)
        meta["ingest_lib"] = meta["ingest_lib"].fillna("")
        # Structures.
        ik = meta["ik14"].astype(str).to_numpy(object)
        smiles = _mode_per_key(ik, meta["smiles"].to_numpy(object))
        formula = _mode_per_key(ik, meta["formula"].to_numpy(object))
        keys = np.array(sorted(smiles.keys()), dtype=object)
        st = pd.DataFrame({"ik14": keys,
                           "smiles": [smiles[k] for k in keys],
                           "formula": [formula[k] for k in keys]})
        m_in = pd.Series(meta["in_mono_mass"].to_numpy(np.float64)).groupby(ik).median()
        mass = st["ik14"].map(m_in).to_numpy(np.float64)
        f_mass = _map_unique(_formula_mass_safe, st["formula"].to_numpy(object)).astype(np.float64)
        mass = np.where(np.isfinite(mass), mass, f_mass)
        if not np.isfinite(mass).all():
            nm = _map_unique(lambda p, a, s: _neutral_mass_safe(p, a, int(s)),
                             meta["precursor_mz"].to_numpy(), meta["adduct"].to_numpy(object),
                             meta["polarity"].to_numpy()).astype(np.float64)
            m_prec = pd.Series(nm).groupby(ik).median()
            mass = np.where(np.isfinite(mass), mass, st["ik14"].map(m_prec).to_numpy(np.float64))
        st["mono_mass"] = mass
        good = np.isfinite(mass)
        if not good.all():
            bad_keys = set(st["ik14"][~good])
            meta = self._drop(meta, meta["ik14"].isin(bad_keys).to_numpy(),
                              "structure without a mass")
            meta = meta.reset_index(drop=True)
            st = st[good]
        st = st.iloc[np.lexsort((_rank_codes(st["ik14"].to_numpy(object)),
                                 st["mono_mass"].to_numpy()))].reset_index(drop=True)
        st["struct_idx"] = np.arange(len(st), dtype=np.int64)
        sidx = pd.Series(st["struct_idx"].to_numpy(), index=st["ik14"].to_numpy(object))
        meta["struct_idx"] = sidx.reindex(meta["ik14"].astype(str).to_numpy(object)).to_numpy(np.int64)
        # Spectrum order (stable, explicit tie-breakers).
        fam_rank = _rank_codes(meta["adduct_family"].map(
            lambda f: "%02d%s" % _family_sort_key(f)).to_numpy(object))
        inst_rank = meta["instrument_family"].map(
            {k: i for i, k in enumerate(INSTRUMENT_FAMILIES)}).fillna(len(INSTRUMENT_FAMILIES)).to_numpy(np.int64)
        ce = meta["collision_energy"].to_numpy(np.float64)
        order = np.lexsort((
            meta["row"].to_numpy(),
            _rank_codes(meta["spectrum_id"].astype(str).to_numpy(object)),
            np.where(np.isnan(ce), np.inf, ce),
            _rank_codes(meta["adduct"].to_numpy(object)),
            _rank_codes(meta["ingest_lib"].to_numpy(object)),
            inst_rank,
            fam_rank,
            -meta["polarity"].to_numpy(np.int64),
            meta["struct_idx"].to_numpy(),
        ))
        meta = meta.iloc[order].reset_index(drop=True)
        meta["spec_idx"] = np.arange(len(meta), dtype=np.int64)
        # Structure ranges.
        s_idx = meta["struct_idx"].to_numpy()
        st["spec_start"] = np.searchsorted(s_idx, st["struct_idx"].to_numpy(), side="left")
        st["spec_end"] = np.searchsorted(s_idx, st["struct_idx"].to_numpy(), side="right")
        st["n_spectra"] = st["spec_end"] - st["spec_start"]
        return meta, st

    def _write_spectra(self, meta: pd.DataFrame) -> None:
        tmp_off = np.load(self.out / "_tmp_offsets.npy")
        n_tmp = int(tmp_off[-1])
        src_mz = (np.memmap(self.out / "_tmp_mz.f32", dtype=np.float32, mode="r", shape=(n_tmp,))
                  if n_tmp else np.zeros(0, np.float32))
        src_int = (np.memmap(self.out / "_tmp_int.f32", dtype=np.float32, mode="r", shape=(n_tmp,))
                   if n_tmp else np.zeros(0, np.float32))
        rows = meta["row"].to_numpy(np.int64)
        lens = tmp_off[rows + 1] - tmp_off[rows]
        total = int(lens.sum())
        mz_out = np.lib.format.open_memmap(self.out / STORE_NAMES[0], mode="w+",
                                           dtype=np.float32, shape=(total,))
        int_out = np.lib.format.open_memmap(self.out / STORE_NAMES[1], mode="w+",
                                            dtype=np.float32, shape=(total,))
        off_out = np.zeros(len(rows) + 1, dtype=np.int64)
        chunk = 200_000
        pos = 0
        for a in range(0, len(rows), chunk):
            pos = _take_into_kernel(src_mz, src_int, tmp_off, rows[a:a + chunk], mz_out,
                                    int_out, off_out, a, pos)
        mz_out.flush()
        int_out.flush()
        del mz_out, int_out, src_mz, src_int
        np.save(self.out / STORE_NAMES[2], off_out)
        (self.out / "_tmp_offsets.npy").unlink()

    def _consensus(self, meta: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        key_cols = ["struct_idx", "polarity", "adduct_family", "instrument_family", "ingest_lib"]
        k = meta[key_cols]
        change = np.zeros(len(meta), dtype=bool)
        if len(meta):
            change[0] = True
            for c in key_cols:
                v = k[c].to_numpy()
                change[1:] |= v[1:] != v[:-1]
        starts = np.flatnonzero(change)
        ends = np.append(starts[1:], len(meta))
        G = len(starts)
        gid = np.cumsum(change) - 1
        meta["cons_idx"] = gid.astype(np.int64)
        first = meta.iloc[starts]
        fam = first["adduct_family"].to_numpy(object)
        mass = st["mono_mass"].to_numpy()[first["struct_idx"].to_numpy()]
        ref = np.array([_family_precursor(f, m) for f, m in zip(fam, mass)], dtype=np.float64)
        prec = meta["precursor_mz"].to_numpy()
        for g in np.flatnonzero(~np.isfinite(ref)):
            ref[g] = float(np.median(prec[starts[g]:ends[g]]))
        store = SpectrumStore.load(self.out, STORE_NAMES, mmap=True)
        parts = []
        chunk = 100_000
        for a in range(0, G, chunk):
            b = min(G, a + chunk)
            members = np.arange(starts[a], ends[b - 1], dtype=np.int64)
            goff = np.append(starts[a:b], ends[b - 1]) - starts[a]
            parts.append(merge_groups(store, members, goff, ref[a:b], cfg))
        mzs = [p.mz for p in parts]
        ints = [p.intensity for p in parts]
        offs = [np.zeros(1, np.int64)]
        base = 0
        for p in parts:
            offs.append(p.offsets[1:] + base)
            base += int(p.offsets[-1])
        cons = SpectrumStore(np.concatenate(mzs) if mzs else np.zeros(0, np.float32),
                             np.concatenate(ints) if ints else np.zeros(0, np.float32),
                             np.concatenate(offs))
        cons.save(self.out, CONSENSUS_NAMES)
        cm = pd.DataFrame({
            "cons_idx": np.arange(G, dtype=np.int64),
            "struct_idx": first["struct_idx"].to_numpy(np.int64),
            "ik14": first["ik14"].to_numpy(object),
            "polarity": first["polarity"].to_numpy(np.int8),
            "adduct_family": fam,
            "instrument_family": first["instrument_family"].to_numpy(object),
            "ingest_lib": first["ingest_lib"].to_numpy(object),
            "precursor_mz": ref,
            "mono_mass": mass,
            "spec_start": starts.astype(np.int64),
            "spec_end": ends.astype(np.int64),
            "n_members": (ends - starts).astype(np.int64),
            "n_peaks": np.diff(cons.offsets).astype(np.int32),
        })
        pol = cm["polarity"].to_numpy()
        open_row = np.full(G, -1, dtype=np.int64)
        for p in (1, -1):
            sel = np.flatnonzero(pol == p)
            open_row[sel] = np.arange(sel.size)
        cm["open_row"] = open_row
        c_struct = cm["struct_idx"].to_numpy()
        sidx = st["struct_idx"].to_numpy()
        st["cons_start"] = np.searchsorted(c_struct, sidx, side="left")
        st["cons_end"] = np.searchsorted(c_struct, sidx, side="right")
        st["n_consensus"] = st["cons_end"] - st["cons_start"]
        return cm

    def _open_index(self, cm: pd.DataFrame) -> None:
        import scipy.sparse as sp

        cons = SpectrumStore.load(self.out, CONSENSUS_NAMES, mmap=False)
        pol = cm["polarity"].to_numpy()
        prec = cm["precursor_mz"].to_numpy()
        for p, name in ((1, "open_index_pos.npz"), (-1, "open_index_neg.npz")):
            sel = np.flatnonzero(pol == p)
            sub = cons.take(sel)
            m = open_matrix(sub.mz, sub.intensity, sub.offsets, prec[sel], self.cfg)
            del sub
            sp.save_npz(self.out / name, m, compressed=False)
            log.info("open index %s: %d rows, %d non-zeros", name, m.shape[0], m.nnz)
            del m


def _mode_per_key(keys: np.ndarray, values: np.ndarray) -> dict:
    """Most frequent non-null value per key (ties: lexicographically smallest); None if
    the key has only nulls."""
    df = pd.DataFrame({"k": keys, "v": values})
    out = {k: None for k in pd.unique(keys)}
    d = df[df["v"].notna()]
    if len(d):
        cnt = d.groupby(["k", "v"], sort=True).size().reset_index(name="n")
        cnt = cnt.sort_values(["k", "n", "v"], ascending=[True, False, True], kind="mergesort")
        first = cnt.drop_duplicates("k", keep="first")
        out.update(dict(zip(first["k"], first["v"])))
    return out


# --------------------------------------------------------------------------------------
# Test-derived library (code phase, before train exists)
# --------------------------------------------------------------------------------------


def fake_labels_from_test(df: pd.DataFrame, per_spectrum: bool = False) -> pd.DataFrame:
    """Give ``test.parquet`` rows fake train labels so a library can be built from them.

    Structure = molecule_id (or spectrum_id with ``per_spectrum``): ``inchikey14`` is a
    14-character fake key, ``normalized_smiles`` the id itself (not a SMILES),
    ``mono_mass`` the median neutral mass implied by precursor m/z and adduct,
    ``ingest_lib`` "public_test".
    """
    out = df.copy()
    key = out["spectrum_id" if per_spectrum else "molecule_id"].astype(str)
    out["inchikey14"] = ("FAKE" + key.str.replace(r"[^0-9A-Za-z]", "", regex=True).str.upper()
                         ).str.ljust(14, "X").str[:14]
    out["normalized_smiles"] = key
    pol = [polarity_sign(m, a) for m, a in zip(out["ionization_mode"], out["adduct"])]
    nm = np.array([_neutral_mass_safe(p, a, s) for p, a, s in
                   zip(out["precursor_mz"], out["adduct"], pol)], dtype=np.float64)
    out["mono_mass"] = pd.Series(nm, index=out.index).groupby(key).transform("median")
    out["ingest_lib"] = "public_test"
    return out

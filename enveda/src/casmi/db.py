"""Candidate database lookup by neutral monoisotopic mass (PLAN.md 2.3, WP2 API).

Two tiers share one lookup:

- NP tier: ``PROCESSED/db/np/np.parquet`` (sorted by mono_mass) plus the row-aligned
  packed Morgan fingerprints ``np_fp.npy``. Loaded once, kept in memory.
- PubChem tier: ``PROCESSED/db/pubchem/mass_XXXX.parquet``, one shard per 10 Da
  (XXXX = 10 * floor(mono_mass / 10)), rows sorted by mono_mass. Read on demand at
  row-group granularity (only the row groups whose mono_mass range meets the window),
  kept in a small LRU cache and released with ``release()``. If no shard exists a
  warning is logged once and lookups return the NP tier only.

Window: ``|mono_mass - mass| <= max(tol_ppm * 1e-6 * mass, min_tol_da)``, both ends
inclusive. Defaults come from ``Config`` (mass_tol_ppm, mass_tol_min_da, db_tiers).

Result frame (one row per ik14 per query; NP-tier rows win over PubChem rows with the
same ik14, and inherit their ``cid`` when missing, ``pmid_count``, ``n_cids`` and
``in_pubchem=True``). Columns, in order (``OUT_COLUMNS``):

    ik14, smiles, formula, mono_mass, ppm_err, tier ("np" | "pubchem"), in_np,
    in_pubchem, in_coconut, in_npatlas, in_lotus, in_pubchemlite, np_row,
    cid, coconut_annotation_level, coconut_np_likeness, n_organisms, n_citations,
    n_collections, pubmed_count, patent_count, anno_type_count, pmid_count, n_cids,
    smiles_source, coconut_id, npaid

``ppm_err = (mono_mass - mass) / mass * 1e6``. ``np_row`` is the row index into
np.parquet / np_fp.npy (-1 for PubChem-only rows). ``in_pubchem`` is True only when the
PubChem tier was queried and holds the ik14 inside the window. Count and prior columns
are pandas nullable dtypes (NA when the tier or source has no value). Rows are sorted by
(mono_mass, ik14). ``lookup_batch`` adds ``query`` (the key or input position) and
``query_mass`` in front and sorts by query position first.

Fingerprints: ``np_fingerprints(rows)`` returns packed uint8 rows (n, 512) of the NP
tier; ``unpack_fp`` gives (n, 4096) 0/1 uint8. Packing is ``np.packbits`` with bit order
"big" (bit j -> byte j // 8, mask 0x80 >> (j % 8)); see docs/ARTEFACTS.md for the RDKit
generator call.

Only numpy, pandas and pyarrow are used (inference-path safe).
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import paths
from .config import Config

__all__ = [
    "CandidateDB", "OUT_COLUMNS", "TIERS", "FP_NBITS", "FP_BITORDER",
    "np_dir", "pubchem_dir", "shard_bucket", "shard_name", "window",
    "get_db", "lookup", "lookup_batch", "np_fingerprints", "np_table", "release", "unpack_fp",
]

log = logging.getLogger(__name__)

TIERS = ("np", "pubchem")
SHARD_DA = 10
FP_NBITS = 4096
FP_BITORDER = "big"

# Output columns and their dtypes (strings keep the dtype pandas gives them).
_STR = ("ik14", "smiles", "formula", "tier", "smiles_source", "coconut_id", "npaid")
_DTYPES: dict[str, str] = {
    "mono_mass": "float64", "ppm_err": "float64",
    "in_np": "bool", "in_pubchem": "bool", "in_coconut": "bool", "in_npatlas": "bool",
    "in_lotus": "bool", "in_pubchemlite": "bool", "np_row": "int64",
    "cid": "Int64", "coconut_annotation_level": "Int8", "coconut_np_likeness": "Float32",
    "n_organisms": "Int32", "n_citations": "Int32", "n_collections": "Int32",
    "pubmed_count": "Int32", "patent_count": "Int32", "anno_type_count": "Int32",
    "pmid_count": "Int32", "n_cids": "Int32",
}
OUT_COLUMNS: tuple[str, ...] = (
    "ik14", "smiles", "formula", "mono_mass", "ppm_err", "tier", "in_np", "in_pubchem",
    "in_coconut", "in_npatlas", "in_lotus", "in_pubchemlite", "np_row",
    "cid", "coconut_annotation_level", "coconut_np_likeness", "n_organisms", "n_citations",
    "n_collections", "pubmed_count", "patent_count", "anno_type_count", "pmid_count", "n_cids",
    "smiles_source", "coconut_id", "npaid",
)
_NP_REQUIRED = ("ik14", "smiles", "formula", "mono_mass")
_PC_COLUMNS = ("ik14", "smiles", "formula", "mono_mass", "cid", "pmid_count", "n_cids")
_FROM_PUBCHEM = ("pmid_count", "n_cids")


def np_dir() -> Path:
    """Directory of the NP tier (``PROCESSED/db/np``)."""
    return paths.PROCESSED / "db" / "np"


def pubchem_dir() -> Path:
    """Directory of the PubChem tier shards (``PROCESSED/db/pubchem``)."""
    return paths.PROCESSED / "db" / "pubchem"


def shard_bucket(mass: float) -> int:
    """Shard key of a mass: 10 * floor(mass / 10) (``mass_0190.parquet`` holds 190-200 Da)."""
    return int(math.floor(mass / SHARD_DA)) * SHARD_DA


def shard_name(bucket: int) -> str:
    return f"mass_{int(bucket):04d}.parquet"


def window(mass: float, tol_ppm: float, min_tol_da: float) -> tuple[float, float]:
    """Inclusive mass window ``mass +- max(tol_ppm * 1e-6 * mass, min_tol_da)``."""
    half = max(float(tol_ppm) * 1e-6 * float(mass), float(min_tol_da))
    return float(mass) - half, float(mass) + half


def unpack_fp(packed: np.ndarray, n_bits: int = FP_NBITS) -> np.ndarray:
    """Packed (n, n_bits/8) uint8 -> (n, n_bits) uint8 0/1 (inverse of ``np.packbits``)."""
    return np.unpackbits(np.asarray(packed, dtype=np.uint8), axis=-1, count=n_bits,
                         bitorder=FP_BITORDER)


def _norm_tiers(tiers: str | Iterable[str] | None, default: Sequence[str]) -> tuple[str, ...]:
    if tiers is None:
        tiers = default
    if isinstance(tiers, str):
        tiers = (tiers,)
    out = tuple(dict.fromkeys(tiers))
    bad = [t for t in out if t not in TIERS]
    if bad:
        raise ValueError(f"unknown tier(s) {bad}; expected a subset of {TIERS}")
    return out


_STR_DTYPE = None


def _str_dtype():
    """The pandas dtype that Arrow strings convert to (``str`` on pandas 3, object before)."""
    global _STR_DTYPE
    if _STR_DTYPE is None:
        _STR_DTYPE = pa.table({"x": pa.array(["a"])}).to_pandas()["x"].dtype
    return _STR_DTYPE


def _na_column(name: str, n: int):
    """A length-``n`` column of the output dtype of ``name`` filled with its missing value."""
    dt = _DTYPES.get(name)
    if dt is None:  # string column
        return pd.Series([None] * n, dtype=_str_dtype()).array
    if dt == "bool":
        return np.zeros(n, dtype=bool)
    if dt == "int64":
        return np.full(n, -1, dtype=np.int64)
    if dt == "float64":
        return np.full(n, np.nan, dtype=np.float64)
    return pd.array([pd.NA] * n, dtype=dt)


def _str_column(values, n: int):
    return pd.Series(values if n else [], dtype=_str_dtype()).array


_EMPTY: pd.DataFrame | None = None


def _empty_out() -> pd.DataFrame:
    global _EMPTY
    if _EMPTY is None:
        cols = {"_q": np.zeros(0, dtype=np.int64), "query_mass": np.zeros(0, dtype=np.float64)}
        for c in OUT_COLUMNS:
            cols[c] = _na_column(c, 0)
        _EMPTY = pd.DataFrame(cols)
    return _EMPTY.copy()


class CandidateDB:
    """NP tier in memory plus on-demand PubChem shards. See the module docstring.

    ``max_cached_row_groups``: PubChem row groups kept in memory (WP3 writes 100k-row
    groups, about 10 MB each in Arrow form); least recently used ones are dropped.
    """

    def __init__(self, np_path: Path | str | None = None, pubchem_dir: Path | str | None = None,
                 cfg: Config | None = None, max_cached_row_groups: int = 32,
                 np_fp_path: Path | str | None = None):
        self.cfg = cfg if cfg is not None else Config()
        self.np_path = Path(np_path) if np_path is not None else np_dir() / "np.parquet"
        if np_fp_path is not None:
            self.np_fp_path = Path(np_fp_path)
        else:
            self.np_fp_path = self.np_path.with_name("np_fp.npy")
        self.pubchem_path = Path(pubchem_dir) if pubchem_dir is not None else globals()["pubchem_dir"]()
        self.max_cached_row_groups = int(max_cached_row_groups)
        self._np: pd.DataFrame | None = None
        self._np_mass: np.ndarray | None = None
        self._fp: np.ndarray | None = None
        self._shards: dict[int, Path] | None = None
        self._shard_meta: dict[int, tuple] = {}
        self._rg_cache: OrderedDict = OrderedDict()
        self._warned: set[str] = set()
        self.n_row_groups_read = 0

    # ------------------------------------------------------------------ NP tier
    def load_np(self) -> pd.DataFrame:
        """Load (once) and return the NP tier with the output columns except ppm_err."""
        if self._np is not None:
            return self._np
        if not self.np_path.exists():
            raise FileNotFoundError(f"NP tier not found: {self.np_path} (run scripts/build_npdb.py)")
        tbl = pq.read_table(self.np_path)
        missing = [c for c in _NP_REQUIRED if c not in tbl.column_names]
        if missing:
            raise ValueError(f"{self.np_path}: missing column(s) {missing}")
        # One chunk per column: Arrow ``take`` on a multi-chunk column costs O(rows).
        df = tbl.combine_chunks().to_pandas(types_mapper=_nullable_mapper)
        del tbl
        n = len(df)
        mass = df["mono_mass"].to_numpy(dtype=np.float64)
        if n > 1 and not np.all(mass[1:] >= mass[:-1]):
            raise ValueError(f"{self.np_path}: not sorted by mono_mass")
        out = {}
        for c in OUT_COLUMNS:
            if c == "ppm_err":
                continue
            if c == "tier":
                out[c] = _str_column(["np"] * n, n)
            elif c == "in_np":
                out[c] = np.ones(n, dtype=bool)
            elif c == "in_pubchem":
                out[c] = np.zeros(n, dtype=bool)
            elif c == "np_row":
                out[c] = np.arange(n, dtype=np.int64)
            elif c in df.columns:
                s = df[c]
                dt = _DTYPES.get(c)
                if dt == "bool":
                    s = s.fillna(False).astype(bool) if s.dtype != bool else s
                elif dt is not None and str(s.dtype) != dt:
                    s = s.astype(dt)
                elif dt is None and s.dtype != _str_dtype():
                    s = pd.Series(s.to_numpy(dtype=object), dtype=_str_dtype())
                out[c] = s.reset_index(drop=True).array
            else:
                out[c] = _na_column(c, n)
        self._np = pd.DataFrame(out)
        self._np_mass = mass
        log.info("NP tier loaded: %d rows from %s", n, self.np_path)
        return self._np

    @property
    def np_mass(self) -> np.ndarray:
        self.load_np()
        return self._np_mass  # type: ignore[return-value]

    def np_fingerprints(self, rows: Sequence[int] | np.ndarray, unpack: bool = False) -> np.ndarray:
        """Packed fingerprints (len(rows), 512) uint8 of NP rows (``np_row`` values).

        ``unpack=True`` returns (len(rows), 4096) 0/1 uint8 instead.
        """
        if self._fp is None:
            if not self.np_fp_path.exists():
                raise FileNotFoundError(f"NP fingerprints not found: {self.np_fp_path}")
            fp = np.load(self.np_fp_path, mmap_mode="r")
            n = len(self.load_np())
            if fp.ndim != 2 or fp.shape[0] != n:
                raise ValueError(f"{self.np_fp_path}: shape {fp.shape} not aligned to {n} NP rows")
            self._fp = fp
        idx = np.asarray(rows, dtype=np.int64).reshape(-1)
        if idx.size and (idx.min() < 0 or idx.max() >= self._fp.shape[0]):
            raise IndexError("np_fingerprints: row index out of range (PubChem rows have np_row -1)")
        out = np.asarray(self._fp[idx])
        return unpack_fp(out, self._fp.shape[1] * 8) if unpack else out

    # ------------------------------------------------------------------ PubChem tier
    def _pubchem_shards(self) -> dict[int, Path]:
        if self._shards is None:
            shards = {}
            d = self.pubchem_path
            if d.is_dir():
                for p in sorted(d.glob("mass_*.parquet")):
                    try:
                        shards[int(p.stem.split("_")[1])] = p
                    except (IndexError, ValueError):
                        continue
            self._shards = shards
            if not shards:
                self._warn_once("absent", "PubChem tier not found (no mass_*.parquet in %s); "
                                "returning the NP tier only", d)
            elif not (d / "_index.json").exists():
                self._warn_once("incomplete", "PubChem tier at %s has %d shards but no _index.json "
                                "(build still running?); using the shards present", d, len(shards))
            else:
                log.info("PubChem tier: %d shards in %s", len(shards), d)
        return self._shards

    def pubchem_available(self) -> bool:
        return bool(self._pubchem_shards())

    def _warn_once(self, key: str, msg: str, *args) -> None:
        if key not in self._warned:
            self._warned.add(key)
            log.warning(msg, *args)

    def _meta(self, bucket: int):
        m = self._shard_meta.get(bucket)
        if m is None:
            path = self._pubchem_shards()[bucket]
            md = pq.ParquetFile(path).metadata
            names = [md.schema.column(i).name for i in range(md.num_columns)]
            ci = names.index("mono_mass")
            lo, hi = [], []
            for r in range(md.num_row_groups):
                st = md.row_group(r).column(ci).statistics
                if st is None or not st.has_min_max:
                    lo, hi = None, None
                    break
                lo.append(st.min)
                hi.append(st.max)
            m = (path, md.num_row_groups,
                 None if lo is None else np.asarray(lo, dtype=np.float64),
                 None if hi is None else np.asarray(hi, dtype=np.float64))
            self._shard_meta[bucket] = m
        return m

    def _row_group(self, bucket: int, rg: int | None):
        key = (bucket, rg)
        hit = self._rg_cache.get(key)
        if hit is not None:
            self._rg_cache.move_to_end(key)
            return hit
        path = self._meta(bucket)[0]
        pf = pq.ParquetFile(path)
        cols = [c for c in _PC_COLUMNS if c in pf.schema_arrow.names]
        tbl = pf.read(columns=cols) if rg is None else pf.read_row_group(rg, columns=cols)
        mass = tbl.column("mono_mass").to_numpy()
        self.n_row_groups_read += 1
        self._rg_cache[key] = (tbl, mass)
        while len(self._rg_cache) > self.max_cached_row_groups:
            self._rg_cache.popitem(last=False)
        return tbl, mass

    def _pubchem_window(self, lo: float, hi: float) -> pd.DataFrame | None:
        shards = self._pubchem_shards()
        if not shards:
            return None
        parts = []
        for b in range(shard_bucket(lo), shard_bucket(hi) + SHARD_DA, SHARD_DA):
            if b not in shards:
                continue
            _, n_rg, rg_lo, rg_hi = self._meta(b)
            if rg_lo is None:
                groups: list[int | None] = [None]
            else:
                groups = [int(r) for r in np.nonzero((rg_hi >= lo) & (rg_lo <= hi))[0]]
            for rg in groups:
                tbl, mass = self._row_group(b, rg)
                i0 = int(np.searchsorted(mass, lo, side="left"))
                i1 = int(np.searchsorted(mass, hi, side="right"))
                if i1 > i0:
                    parts.append(tbl.slice(i0, i1 - i0))
        if not parts:
            return None
        frames = [p.to_pandas(types_mapper=_nullable_mapper) for p in parts]
        return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)

    def release(self, np_tier: bool = False) -> None:
        """Drop cached PubChem row groups (and the NP tier and fingerprints if asked)."""
        self._rg_cache.clear()
        self._shard_meta.clear()
        self._shards = None
        if np_tier:
            self._np = None
            self._np_mass = None
            self._fp = None

    # ------------------------------------------------------------------ lookup
    def lookup(self, mass: float, tol_ppm: float | None = None,
               tiers: str | Iterable[str] | None = None,
               min_tol_da: float | None = None) -> pd.DataFrame:
        """Candidates with |mono_mass - mass| <= max(tol_ppm * 1e-6 * mass, min_tol_da)."""
        out = self._lookup_many(np.asarray([mass], dtype=np.float64), tol_ppm, tiers, min_tol_da)
        return out.drop(columns=["_q", "query_mass"])

    def lookup_batch(self, masses: Sequence[float] | np.ndarray, tol_ppm: float | None = None,
                     tiers: str | Iterable[str] | None = None, min_tol_da: float | None = None,
                     keys: Sequence | None = None) -> pd.DataFrame:
        """``lookup`` for many masses in one frame with ``query`` and ``query_mass`` in front.

        ``query`` is ``keys[i]`` (e.g. molecule_id) or the input position ``i``. Masses are
        processed in sorted order so each PubChem row group is read once.
        """
        m = np.asarray(masses, dtype=np.float64).reshape(-1)
        if keys is not None and len(keys) != len(m):
            raise ValueError("lookup_batch: keys and masses differ in length")
        out = self._lookup_many(m, tol_ppm, tiers, min_tol_da)
        q = out.pop("_q").to_numpy()
        if keys is not None:
            karr = np.asarray(list(keys), dtype=object)
            out.insert(0, "query", karr[q] if len(q) else pd.Series([], dtype=object))
        else:
            out.insert(0, "query", q.astype(np.int64))
        qm = out.pop("query_mass")
        out.insert(1, "query_mass", qm)
        return out

    def _lookup_many(self, masses: np.ndarray, tol_ppm, tiers, min_tol_da) -> pd.DataFrame:
        cfg = self.cfg
        tol_ppm = cfg.mass_tol_ppm if tol_ppm is None else float(tol_ppm)
        min_tol_da = cfg.mass_tol_min_da if min_tol_da is None else float(min_tol_da)
        tiers = _norm_tiers(tiers, cfg.db_tiers)
        if not np.all(np.isfinite(masses)):
            raise ValueError("lookup: masses must be finite")
        half = np.maximum(tol_ppm * 1e-6 * masses, min_tol_da)
        lo, hi = masses - half, masses + half
        frames = []
        np_part = None
        if "np" in tiers:
            npdf = self.load_np()
            nm = self._np_mass
            i0 = np.searchsorted(nm, lo, side="left")
            i1 = np.searchsorted(nm, hi, side="right")
            cnt = np.maximum(i1 - i0, 0)
            if cnt.sum():
                if len(masses) == 1:  # contiguous: a slice, no gather
                    np_part = npdf.iloc[int(i0[0]):int(i1[0])].reset_index(drop=True)
                else:
                    idx = np.concatenate([np.arange(a, b) for a, b in zip(i0, i1) if b > a])
                    np_part = npdf.iloc[idx].reset_index(drop=True)
                np_part.insert(0, "_q", np.repeat(np.arange(len(masses)), cnt))
        pc_part = None
        if "pubchem" in tiers and self._pubchem_shards():
            pcs = []
            for qi in np.argsort(masses, kind="stable"):
                w = self._pubchem_window(lo[qi], hi[qi])
                if w is not None and len(w):
                    w.insert(0, "_q", int(qi))
                    pcs.append(w)
            if pcs:
                pc_part = self._pubchem_to_out(pd.concat(pcs, ignore_index=True))
        if np_part is not None and pc_part is not None:
            np_part, pc_part = self._merge_tiers(np_part, pc_part)
        for p in (np_part, pc_part):
            if p is not None and len(p):
                frames.append(p)
        if not frames:
            return _empty_out()
        if len(frames) == 1:
            # The NP part is already in (query, mono_mass, ik14) order; sort PubChem only.
            out = frames[0]
            if out is pc_part:
                out = out.sort_values(["_q", "mono_mass", "ik14"], kind="mergesort",
                                      na_position="last").reset_index(drop=True)
        else:
            out = pd.concat(frames, ignore_index=True)
            out = out.sort_values(["_q", "mono_mass", "ik14", "tier"], kind="mergesort",
                                  na_position="last").reset_index(drop=True)
        q = out["_q"].to_numpy()
        qm = masses[q]
        out.insert(1, "query_mass", qm)
        mm = out["mono_mass"].to_numpy(dtype=np.float64)
        out.insert(out.columns.get_loc("mono_mass") + 1, "ppm_err", (mm - qm) / qm * 1e6)
        return out[["_q", "query_mass", *OUT_COLUMNS]]

    @staticmethod
    def _pubchem_to_out(pc: pd.DataFrame) -> pd.DataFrame:
        # One row per (query, ik14): lowest mass then lowest cid wins (shards are unique
        # per ik14 already; this only guards against duplicates across shards).
        pc = pc.sort_values(["_q", "mono_mass", "cid"], kind="mergesort")
        has_ik = pc["ik14"].notna()
        dup = pc.duplicated(["_q", "ik14"], keep="first") & has_ik
        pc = pc[~dup].reset_index(drop=True)
        n = len(pc)
        out = {"_q": pc["_q"].to_numpy(dtype=np.int64)}
        for c in OUT_COLUMNS:
            if c == "ppm_err":
                continue
            if c == "tier":
                out[c] = _str_column(["pubchem"] * n, n)
            elif c == "in_pubchem":
                out[c] = np.ones(n, dtype=bool)
            elif c in ("ik14", "smiles", "formula"):
                out[c] = _str_column(pc[c].to_numpy(dtype=object), n)
            elif c == "mono_mass":
                out[c] = pc[c].to_numpy(dtype=np.float64)
            elif c in ("cid", "pmid_count", "n_cids") and c in pc.columns:
                out[c] = pc[c].astype(_DTYPES[c]).array
            else:
                out[c] = _na_column(c, n)
        return pd.DataFrame(out)

    @staticmethod
    def _merge_tiers(np_part: pd.DataFrame, pc_part: pd.DataFrame):
        """Drop PubChem rows whose (query, ik14) is in the NP part; copy their PubChem info."""
        k_np = pd.MultiIndex.from_arrays([np_part["_q"].to_numpy(), np_part["ik14"].astype(object)])
        k_pc = pd.MultiIndex.from_arrays([pc_part["_q"].to_numpy(), pc_part["ik14"].astype(object)])
        dup = k_pc.isin(k_np) & pc_part["ik14"].notna().to_numpy()
        if dup.any():
            d = pc_part.loc[dup, ["_q", "ik14", "cid", "pmid_count", "n_cids"]].copy()
            d["ik14"] = d["ik14"].astype(object)
            left = pd.DataFrame({"_q": np_part["_q"].to_numpy(),
                                 "ik14": np_part["ik14"].astype(object).to_numpy()})
            j = left.merge(d, how="left", on=["_q", "ik14"], indicator=True, validate="one_to_one")
            both = (j["_merge"] == "both").to_numpy()
            np_part = np_part.copy()
            np_part["in_pubchem"] = np_part["in_pubchem"].to_numpy() | both
            for c in _FROM_PUBCHEM:
                np_part[c] = pd.array(j[c].to_numpy(), dtype=_DTYPES[c])
            pc_cid = pd.Series(pd.array(j["cid"].to_numpy(), dtype="Int64"), index=np_part.index)
            np_part["cid"] = np_part["cid"].fillna(pc_cid)
            pc_part = pc_part[~dup].reset_index(drop=True)
        return np_part, pc_part


def _nullable_mapper(t: pa.DataType):
    """Arrow -> pandas nullable dtypes for integer and float32 columns."""
    if pa.types.is_int8(t):
        return pd.Int8Dtype()
    if pa.types.is_int16(t):
        return pd.Int16Dtype()
    if pa.types.is_int32(t):
        return pd.Int32Dtype()
    if pa.types.is_int64(t):
        return pd.Int64Dtype()
    if pa.types.is_float32(t):
        return pd.Float32Dtype()
    return None


# ---------------------------------------------------------------------- module-level API
_DEFAULT: CandidateDB | None = None


def get_db(cfg: Config | None = None) -> CandidateDB:
    """Process-wide default ``CandidateDB`` (paths from ``casmi.paths``)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = CandidateDB(cfg=cfg)
    elif cfg is not None:
        _DEFAULT.cfg = cfg
    return _DEFAULT


def lookup(mass: float, tol_ppm: float | None = None, tiers: str | Iterable[str] | None = None,
           min_tol_da: float | None = None) -> pd.DataFrame:
    """Candidates of the default DB around ``mass``. See ``CandidateDB.lookup``."""
    return get_db().lookup(mass, tol_ppm, tiers, min_tol_da)


def lookup_batch(masses: Sequence[float] | np.ndarray, tol_ppm: float | None = None,
                 tiers: str | Iterable[str] | None = None, min_tol_da: float | None = None,
                 keys: Sequence | None = None) -> pd.DataFrame:
    """Batch lookup on the default DB. See ``CandidateDB.lookup_batch``."""
    return get_db().lookup_batch(masses, tol_ppm, tiers, min_tol_da, keys)


def np_fingerprints(row_indices: Sequence[int] | np.ndarray, unpack: bool = False) -> np.ndarray:
    """Packed (n, 512) uint8 Morgan fingerprints of NP rows (``np_row`` of a lookup)."""
    return get_db().np_fingerprints(row_indices, unpack=unpack)


def np_table() -> pd.DataFrame:
    """The whole NP tier (output columns without ppm_err). Treat as read-only."""
    return get_db().load_np()


def release(np_tier: bool = False) -> None:
    """Free cached PubChem row groups (and the NP tier if ``np_tier``) of the default DB."""
    if _DEFAULT is not None:
        _DEFAULT.release(np_tier=np_tier)

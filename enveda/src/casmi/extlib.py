"""External spectral libraries: public reference spectra outside train (ROADMAP R2).

Class 1 means "the structure has public reference spectra", not "it is in train". This module
turns public library dumps into files in the train-ingestion layout, keeps only what train does
not already hold, and records per-record provenance (source id, licence, mkey).

Flow (``scripts/build_extlib.py`` drives it; all paths come from ``paths``)
-------------------------------------------------------------------------
1. ``stage``: a parser per source yields raw records (``RawRecord``); ``Stager`` standardises
   the structure (``chem.std_mol``, cached per input string, RDKit in a small process pool),
   applies the quality rules, cleans the peaks with ``spectra.clean_flat`` and appends the
   survivors to ``PROCESSED/extlib/staging/<source>/part-*.parquet`` (STAGE_SCHEMA). Drop
   counts by reason go to ``staging/<source>/stage_info.json``.
2. ``finalize``: deduplicates the staged spectra of every source against train and against
   sources earlier in ``SOURCE_ORDER`` (``dedup``), and writes
   ``PROCESSED/extlib/extlib_<source>.parquet`` (FINAL_COLUMNS: the train.parquet columns plus
   provenance and the derived columns ``Library.build`` reads).

Quality rules (first failing rule wins; reasons in ``DROP_REASONS``)
-------------------------------------------------------------------
NIST-attributed (any provenance field names NIST, except the host-approved GNPS-NIST14-MATCHES
library) or in-silico records are dropped; MS level must be 2 when given; the structure must
parse (SMILES, else InChI), be one component before and after standardisation, carry net
charge 0 after ``std_mol`` (which neutralises acids and bases) and no isotope labels; the
adduct must parse (``adducts.parse_adduct``, with the record's ionization mode) to the
numbers (multiplier, charge, shift within 1e-5 Da) of an adduct string that occurs in train
(``known_adducts``), whose string is then written to ``adduct`` (``adduct_orig`` keeps the
source text; an uncharged ``M+2H`` style string may also resolve to charge 2 or 3 when only
that charge fits the precursor); the polarity of the adduct must match the record's mode; the
precursor m/z must be within ``extlib_max_abs_ppm`` of the value implied by the standardised
structure and the adduct; at least ``extlib_min_peaks`` peaks must survive ``spectra.clean``.

Columns written
---------------
``normalized_smiles`` is ``Chem.MolToSmiles(chem.std_mol(...))`` (no stereo), ``inchikey`` /
``inchikey14`` / ``molecular_formula`` come from the same molecule. ``precursor_error_ppm``
is recomputed: ``abs(precursor_mz - theoretical) / theoretical * 1e6`` with the electron
mass included (casmi.adducts). It is an absolute value like train's column (which is almost
never negative), but train's values come from the host's own formula, which ignores the
electron mass (about +1.6 ppm at m/z 330) and is off by about 2,000 ppm for
``[M+CH2O2-H]-``; ours are not on that scale. Peaks are stored CLEANED
(``spectra.clean``, float32, base peak 1, at most ``top_n_peaks``), exactly like WP6's
``data/processed/train/spectra/`` files, with ``n_peaks_raw`` and ``prec_intensity`` of the
raw spectrum; ``num_peaks`` is the raw peak count. So ``Library.build(files,
precleaned=True)`` stores them as they are, alone or together with WP6's train files, and
``precleaned=False`` cleans them again (nearly idempotent). ``base_peak_intensity`` is the
largest raw intensity when the source ships raw counts, else null (sources whose maximum
is a normalisation constant such as 1, 100, 999, 1000 or 10000). ``collision_energy_ev`` is
a list (several values = merged or ramped); NCE is converted as NCE * precursor_mz / 500 *
charge factor (1, 0.9, 0.85, 0.8, 0.75 for |z| = 1, 2, 3, 4, 5+), V is read as eV.

Inference-path rule (PLAN 7) does not apply: this is build code (numpy, pandas, pyarrow,
rdkit only, no network).
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import math
import re
import time
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import paths
from .config import Config

__all__ = [
    "RawRecord",
    "SOURCE_ORDER",
    "SOURCE_INFO",
    "DROP_REASONS",
    "STAGE_SCHEMA",
    "FINAL_COLUMNS",
    "TRAIN_COLUMNS",
    "extlib_dir",
    "staging_dir",
    "known_adducts",
    "AdductMapper",
    "parse_collision_energy",
    "resolve_structure",
    "iter_json_array",
    "iter_gnps_json",
    "iter_massbank_msp",
    "massbank_licences",
    "iter_mona_json",
    "iter_msg_table",
    "iter_enveda_jsonl",
    "iter_kaggle_npy",
    "Stager",
    "dedup",
    "nist_flag",
]

log = logging.getLogger(__name__)

# Priority for cross-source deduplication: a spectrum is kept by the first source that has it.
SOURCE_ORDER: tuple[str, ...] = ("massbank", "gnps", "mona", "msnlib", "gnps_multiplex",
                                 "enveda180", "massspecgym", "kaggle_gnpsfull")

SOURCE_INFO: dict[str, dict[str, str]] = {
    "massbank": {"licence": "per record (MassBank.json 'license'; CC BY, CC BY-SA, CC BY-NC, "
                            "CC BY-NC-SA, CC0, dl-de/by-2-0)",
                 "url": "https://github.com/MassBank/MassBank-data/releases/tag/2026.11"},
    "gnps": {"licence": "CC0 for GNPS deposits; imports keep their source licence",
             "url": "https://external.gnps2.org/gnpslibrary (per-library JSON, type GNPS and IMPORT)"},
    "gnps_multiplex": {"licence": "CC0 (GNPS deposit; propagated-by-mass annotations)",
                       "url": "https://external.gnps2.org/gnpslibrary "
                              "(MULTIPLEX-SYNTHESIS-LIBRARY-ALL-PARTITION-1..6, GNPS-SYNTHESIS-CARNITINES)"},
    "mona": {"licence": "per record (MoNA metaData 'license'), CC BY 4.0 where absent",
             "url": "https://mona.fiehnlab.ucdavis.edu/downloads (Experimental Spectra, JSON)"},
    "msnlib": {"licence": "CC BY 4.0",
               "url": "https://zenodo.org/records/23494858 (MSnLib v8 MS2, MassSpecGym format)"},
    "enveda180": {"licence": "CC BY 4.0",
                  "url": "https://zenodo.org/records/21346580 (enveda-180.jsonl.gz)"},
    "massspecgym": {"licence": "MIT (dataset card)",
                    "url": "https://huggingface.co/datasets/roman-bushuiev/MassSpecGym "
                           "(data/MassSpecGym1.5.tsv)"},
    "kaggle_gnpsfull": {"licence": "CC BY-NC-SA 4.0 (Kaggle dataset; underlying GNPS CC0)",
                        "url": "https://www.kaggle.com/datasets/taeyangg4/casmi26-extlib-gnps-full"},
}

DROP_REASONS: tuple[str, ...] = (
    "nist", "in_silico", "skipped_subset", "not_ms2", "no_precursor", "bad_peaks",
    "no_structure", "bad_structure", "multi_component", "charged", "isotope_label",
    "adduct_unparsed", "adduct_unknown", "polarity_mismatch", "ppm_gt_max",
    "lt_min_peaks", "duplicate_in_source",
)

# train.parquet columns (docs/COMPETITION.md), in train order.
TRAIN_COLUMNS: tuple[str, ...] = (
    "ingest_lib", "normalized_smiles", "inchikey", "inchikey14", "molecular_formula",
    "ionization_mode", "instrument_type", "adduct", "adduct_orig", "precursor_mz",
    "precursor_error_ppm", "ms2_mzs", "ms2_normalized_intensities", "num_peaks",
    "base_peak_intensity", "collision_energy_ev", "collision_energy_orig",
    "collision_energy_orig_units",
)

_STAGE_FIELDS = [
    ("spectrum_id", pa.string()), ("source", pa.string()), ("source_id", pa.string()),
    ("sub_lib", pa.string()), ("licence", pa.string()),
    ("ingest_lib", pa.string()), ("normalized_smiles", pa.string()), ("inchikey", pa.string()),
    ("inchikey14", pa.string()), ("molecular_formula", pa.string()),
    ("ionization_mode", pa.string()), ("instrument_type", pa.string()), ("adduct", pa.string()),
    ("adduct_orig", pa.string()), ("precursor_mz", pa.float64()),
    ("precursor_error_ppm", pa.float64()),
    ("ms2_mzs", pa.list_(pa.float32())), ("ms2_normalized_intensities", pa.list_(pa.float32())),
    ("num_peaks", pa.int64()), ("base_peak_intensity", pa.float64()),
    ("collision_energy_ev", pa.list_(pa.float64())), ("collision_energy_orig", pa.string()),
    ("collision_energy_orig_units", pa.string()),
    ("polarity", pa.int8()), ("instrument_family", pa.string()), ("mono_mass", pa.float64()),
    ("n_peaks_raw", pa.int32()), ("prec_intensity", pa.float32()), ("n_peaks", pa.int32()),
    ("is_test_adduct", pa.bool_()),
]
STAGE_SCHEMA = pa.schema(_STAGE_FIELDS)

# Final extlib_<source>.parquet: train columns first, then provenance and derived columns.
FINAL_COLUMNS: tuple[str, ...] = TRAIN_COLUMNS + (
    "spectrum_id", "source_id", "sub_lib", "licence", "mkey", "in_train",
    "polarity", "instrument_family", "mono_mass", "n_peaks_raw", "prec_intensity", "n_peaks",
)


def extlib_dir() -> Path:
    return paths.PROCESSED / "extlib"


def staging_dir(source: str | None = None) -> Path:
    d = extlib_dir() / "staging"
    return d / source if source else d


# --------------------------------------------------------------------------------------
# Raw records
# --------------------------------------------------------------------------------------


@dataclass
class RawRecord:
    """One spectrum as a parser reads it (strings as found; peaks as float64 arrays)."""

    source_id: str
    sub_lib: str = ""
    smiles: str | None = None
    inchi: str | None = None
    adduct: str | None = None
    ionization_mode: str | None = None
    precursor_mz: float = float("nan")
    instrument_type: str | None = None
    ms_level: int | None = None
    collision_energy: str | None = None
    ce_units_hint: str | None = None       # "NCE" / "eV" when the source fixes the unit
    mzs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    intensities: np.ndarray = field(default_factory=lambda: np.zeros(0))
    licence: str = ""
    raw_intensities: bool | None = None    # True raw counts, False normalised, None: infer
    drop: str | None = None                # parser-level drop reason (nist, in_silico, ...)


_NIST_RE = re.compile(r"\bNIST", re.IGNORECASE)
_NIST_ALLOWED = ("GNPS-NIST14-MATCHES",)
_INSILICO_RE = re.compile(r"in[- ]?silico|predicted|cfm-?id|lipidblast|\bsimulated\b",
                          re.IGNORECASE)


def nist_flag(*fields: Any, library: str | None = None) -> bool:
    """True if any provenance field names NIST (the host-approved GNPS-NIST14-MATCHES library
    is exempt: its spectra are GNPS data annotated by NIST14 matching)."""
    if library in _NIST_ALLOWED:
        return False
    return any(isinstance(f, str) and _NIST_RE.search(f) for f in (*fields, library))


def _insilico(*fields: Any) -> bool:
    return any(isinstance(f, str) and _INSILICO_RE.search(f) for f in fields)


def _na(x: Any) -> str | None:
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    s = str(x).strip()
    if not s or s.upper() in ("N/A", "NA", "NAN", "NONE", "NULL", "-", " "):
        return None
    return s


def _float(x: Any) -> float:
    try:
        v = float(str(x).strip())
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _peaks_from_pairs(pairs: Any) -> tuple[np.ndarray, np.ndarray]:
    try:
        a = np.asarray(pairs, dtype=np.float64)
    except (TypeError, ValueError):
        return np.zeros(0), np.zeros(0)
    if a.ndim != 2 or a.shape[1] < 2:
        return np.zeros(0), np.zeros(0)
    return a[:, 0].copy(), a[:, 1].copy()


def _peaks_from_text(text: str, pair_sep: str | None = None, mz_sep: str = ":") -> tuple[np.ndarray, np.ndarray]:
    """'mz:int mz:int ...' (MoNA) -> arrays."""
    if not text:
        return np.zeros(0), np.zeros(0)
    try:
        flat = np.array(text.replace(mz_sep, " ").split(pair_sep) if pair_sep else
                        text.replace(mz_sep, " ").split(), dtype=np.float64)
    except ValueError:
        return np.zeros(0), np.zeros(0)
    if flat.size % 2:
        return np.zeros(0), np.zeros(0)
    return flat[0::2].copy(), flat[1::2].copy()


def _csv_floats(text: Any) -> np.ndarray:
    if text is None:
        return np.zeros(0)
    if isinstance(text, (list, tuple, np.ndarray)):
        return np.asarray(text, dtype=np.float64)
    try:
        return np.array(str(text).split(","), dtype=np.float64)
    except ValueError:
        return np.zeros(0)


# --------------------------------------------------------------------------------------
# Streaming readers
# --------------------------------------------------------------------------------------

_WS = re.compile(r"[\s,]*")


def iter_json_array(fh: io.TextIOBase, chunk_chars: int = 1 << 23) -> Iterator[Any]:
    """Yield the elements of a top-level JSON array from a text stream, with bounded memory
    (``json.JSONDecoder.raw_decode`` over a sliding buffer; elements must be objects or
    arrays, which is the case for every library dump read here)."""
    dec = json.JSONDecoder()
    buf = ""
    pos = 0
    eof = False
    # find the opening bracket
    while True:
        i = buf.find("[")
        if i >= 0:
            pos = i + 1
            break
        more = fh.read(chunk_chars)
        if not more:
            return
        buf += more
    while True:
        m = _WS.match(buf, pos)
        pos = m.end()
        if pos >= len(buf):
            if eof:
                return
            more = fh.read(chunk_chars)
            if not more:
                eof = True
            buf = buf[pos:] + more
            pos = 0
            continue
        if buf[pos] == "]":
            return
        try:
            obj, end = dec.raw_decode(buf, pos)
        except json.JSONDecodeError:
            if eof:
                raise
            more = fh.read(chunk_chars)
            if not more:
                eof = True
            buf = buf[pos:] + more
            pos = 0
            continue
        yield obj
        pos = end
        if pos > chunk_chars:
            buf = buf[pos:]
            pos = 0


def _open_text(path: Path) -> io.TextIOBase:
    path = Path(path)
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _open_zip_member(path: Path) -> io.TextIOBase:
    zf = zipfile.ZipFile(path)
    names = [n for n in zf.namelist() if not n.endswith("/")]
    if len(names) != 1:
        raise ValueError(f"{path}: expected one member, found {names[:5]}")
    return io.TextIOWrapper(zf.open(names[0]), encoding="utf-8", errors="replace")


# --------------------------------------------------------------------------------------
# Parsers, one per source format
# --------------------------------------------------------------------------------------


def iter_gnps_json(path: Path, library: str | None = None) -> Iterator[RawRecord]:
    """GNPS library JSON export (``/gnpslibrary/<LIB>.json``): a pretty-printed array of flat
    records with ``peaks_json``."""
    with _open_text(path) as fh:
        for r in iter_json_array(fh):
            lib = r.get("library_membership") or library or ""
            rec = RawRecord(
                source_id=str(r.get("spectrum_id") or r.get("SpectrumID") or ""),
                sub_lib=str(lib),
                smiles=_na(r.get("Smiles")),
                inchi=_na(r.get("INCHI")),
                adduct=_na(r.get("Adduct")),
                ionization_mode=_na(r.get("Ion_Mode")),
                precursor_mz=_float(r.get("Precursor_MZ")),
                instrument_type=_na(r.get("Instrument")),
                ms_level=int(_float(r.get("ms_level"))) if math.isfinite(_float(r.get("ms_level"))) else None,
                licence=("CC0 (GNPS)" if not str(lib).startswith(("MASSBANK", "MONA", "HMDB", "RESPECT",
                                                                    "CASMI", "SUMNER", "BMDMS",
                                                                    "BIRMINGHAM"))
                         else f"GNPS import of {lib} (original licence)"),
                raw_intensities=None,
            )
            if nist_flag(r.get("Compound_Source"), r.get("PI"), r.get("Data_Collector"), library=str(lib)):
                rec.drop = "nist"
            elif _insilico(r.get("Instrument"), r.get("Ion_Source"), r.get("Compound_Source")):
                rec.drop = "in_silico"
            else:
                try:
                    pk = json.loads(r.get("peaks_json") or "[]")
                except (TypeError, ValueError):
                    pk = []
                rec.mzs, rec.intensities = _peaks_from_pairs(pk)
            yield rec


_MSP_KEYS = {"name", "db#", "inchikey", "inchi", "smiles", "precursor_type", "spectrum_type",
             "precursormz", "instrument_type", "instrument", "ion_mode", "collision_energy",
             "formula", "comments", "num peaks"}


def iter_massbank_msp(path: Path, licences: dict[str, str] | None = None) -> Iterator[RawRecord]:
    """MassBank ``MassBank_NISTformat.msp`` (the format is named after NIST; the data is
    MassBank's). Licence per accession from ``massbank_licences``."""
    licences = licences or {}

    def emit(d: dict, mz: list, it: list) -> RawRecord:
        acc = d.get("db#", "")
        st = (d.get("spectrum_type") or "").upper()
        rec = RawRecord(
            source_id=acc, sub_lib=acc.split("-")[1] if acc.count("-") >= 2 else "",
            smiles=_na(d.get("smiles")), inchi=_na(d.get("inchi")),
            adduct=_na(d.get("precursor_type")), ionization_mode=_na(d.get("ion_mode")),
            precursor_mz=_float(d.get("precursormz")),
            instrument_type=_na("; ".join(x for x in (d.get("instrument_type"), d.get("instrument")) if x)),
            ms_level=(2 if st == "MS2" else (int(st[2:]) if st.startswith("MS") and st[2:].isdigit()
                                              else (1 if st == "MS" else None))),
            collision_energy=_na(d.get("collision_energy")),
            mzs=np.asarray(mz, np.float64), intensities=np.asarray(it, np.float64),
            licence=licences.get(acc, "unknown (MassBank record)"), raw_intensities=False,
        )
        if nist_flag(d.get("comments"), d.get("instrument")):
            rec.drop = "nist"
        return rec

    d: dict[str, str] = {}
    mz: list[float] = []
    it: list[float] = []
    n_peaks = -1
    with _open_text(path) as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not line.strip():
                if d:
                    yield emit(d, mz, it)
                d, mz, it, n_peaks = {}, [], [], -1
                continue
            if n_peaks >= 0:
                parts = line.replace("\t", " ").split()
                if len(parts) >= 2:
                    try:
                        mz.append(float(parts[0]))
                        it.append(float(parts[1]))
                    except ValueError:
                        pass
                continue
            k, sep, v = line.partition(":")
            if not sep:
                continue
            k = k.strip().lower()
            v = v.strip()
            if k == "num peaks":
                n_peaks = int(_float(v)) if math.isfinite(_float(v)) else 0
                continue
            if k in _MSP_KEYS and k not in d:
                d[k] = v
        if d:
            yield emit(d, mz, it)


def massbank_licences(path: Path) -> dict[str, str]:
    """Accession -> licence URL from the MassBank Bioschemas JSON (``MassBank.json``)."""
    out: dict[str, str] = {}
    with _open_text(path) as fh:
        for r in iter_json_array(fh):
            if isinstance(r, dict) and r.get("@type") == "Dataset" and r.get("identifier"):
                lic = r.get("license")
                if isinstance(lic, list):
                    lic = ";".join(str(x) for x in lic)
                out[str(r["identifier"])] = str(lic or "")
    return out


_LICENCE_SHORT = [
    (re.compile(r"publicdomain/zero|cc0", re.I), "CC0"),
    (re.compile(r"by-nc-sa", re.I), "CC BY-NC-SA"),
    (re.compile(r"by-nc", re.I), "CC BY-NC"),
    (re.compile(r"by-sa", re.I), "CC BY-SA"),
    (re.compile(r"licenses/by/|cc by\b|cc-by\b", re.I), "CC BY"),
    (re.compile(r"dl-de", re.I), "dl-de/by-2-0"),
]


def short_licence(s: str) -> str:
    for rx, name in _LICENCE_SHORT:
        if s and rx.search(s):
            return name
    return s or "unknown"


def _mona_meta(items: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in items or []:
        try:
            k = str(m.get("name", "")).strip().lower()
            v = m.get("value")
        except AttributeError:
            continue
        if k and k not in out and v is not None:
            out[k] = str(v)
    return out


def iter_mona_json(path: Path, skip_tags: Sequence[str] = ("Enveda-180",)) -> Iterator[RawRecord]:
    """MoNA JSON export (zip with one JSON file, one record per line). Records carrying a tag
    in ``skip_tags`` are yielded as drop="skipped_subset" without decoding (Enveda-180 is
    staged from its own Zenodo release)."""
    needles = [f'"text":"{t}"' for t in skip_tags]
    fh = _open_zip_member(path) if str(path).endswith(".zip") else _open_text(path)
    with fh:
        for line in fh:
            line = line.strip()
            if line.endswith(","):
                line = line[:-1]
            if not line.startswith("{"):
                continue
            if any(n in line for n in needles):
                yield RawRecord(source_id="", drop="skipped_subset")
                continue
            try:
                r = json.loads(line)
            except ValueError:
                yield RawRecord(source_id="", drop="bad_peaks")
                continue
            md = _mona_meta(r.get("metaData"))
            comp = None
            for c in r.get("compound") or []:
                if c.get("kind") == "biological":
                    comp = c
                    break
            if comp is None and r.get("compound"):
                comp = r["compound"][0]
            cmd = _mona_meta(comp.get("metaData")) if comp else {}
            smiles = None
            if comp:
                for m in comp.get("metaData") or []:
                    if str(m.get("name", "")).lower() == "smiles" and not m.get("computed"):
                        smiles = m.get("value")
                        break
                if smiles is None:
                    smiles = cmd.get("smiles")
            tags = [str(t.get("text", "")) for t in r.get("tags") or [] if isinstance(t, dict)]
            libd = r.get("library") or {}
            lib = str(libd.get("library") or "") if isinstance(libd, dict) else ""
            sub = lib or next((t for t in tags if t not in ("LC-MS", "GC-MS", "LC-MS/MS", "MS2")), "")
            msl = (md.get("ms level") or "").upper().replace("MS", "")
            rec = RawRecord(
                source_id=str(r.get("id") or ""), sub_lib=sub,
                smiles=_na(smiles), inchi=_na(comp.get("inchi")) if comp else None,
                adduct=_na(md.get("precursor type")), ionization_mode=_na(md.get("ionization mode")),
                precursor_mz=_float(md.get("precursor m/z")),
                instrument_type=_na("; ".join(x for x in (md.get("instrument type"), md.get("instrument")) if x)),
                ms_level=int(msl) if msl.isdigit() else None,
                collision_energy=_na(md.get("collision energy")),
                licence=short_licence(md.get("license") or "") if md.get("license") else "CC BY 4.0 (MoNA default)",
                raw_intensities=None,
            )
            if nist_flag(lib, *tags, md.get("author"), md.get("copyright"), md.get("license"),
                         md.get("publication")):
                rec.drop = "nist"
            elif "In-Silico" in tags or _insilico(lib, *tags):
                rec.drop = "in_silico"
            elif "GC-MS" in tags and "LC-MS" not in tags:
                rec.drop = "not_ms2"
            else:
                rec.mzs, rec.intensities = _peaks_from_text(r.get("spectrum") or "")
            yield rec


def iter_msg_table(path: Path, source: str) -> Iterator[RawRecord]:
    """MassSpecGym-format tables: ``msnlib_ms2.parquet`` (MSnLib v8) or ``MassSpecGym*.tsv``.
    Peaks are comma-separated strings; MSnLib collision energies are NCE (Orbitrap)."""
    p = Path(path)
    if p.suffix == ".parquet":
        pf = pq.ParquetFile(p)
        want = [c for c in ("identifier", "mzs", "intensities", "smiles", "inchi", "adduct",
                            "precursor_mz", "instrument_type", "collision_energy", "ionmode",
                            "mslevel", "charge", "dataset_id", "acquisition", "source_instrument",
                            "fragmentation_method", "spectype") if c in pf.schema_arrow.names]
        batches = (b.to_pandas() for b in pf.iter_batches(batch_size=20000, columns=want))
    else:
        batches = pd.read_csv(p, sep="\t", chunksize=20000, dtype=str, keep_default_na=False)
    for df in batches:
        cols = set(df.columns)
        for r in df.itertuples(index=False):
            g = r._asdict()
            mode = _na(g.get("ionmode"))
            add = _na(g.get("adduct"))
            if mode is None and add:
                mode = "positive" if add.strip().endswith("+") else ("negative" if add.strip().endswith("-") else None)
            itype = _na(g.get("instrument_type"))
            if "source_instrument" in cols and _na(g.get("source_instrument")):
                itype = "; ".join(x for x in (itype, _na(g.get("source_instrument"))) if x)
            msl = _float(g.get("mslevel"))
            rec = RawRecord(
                source_id=str(g.get("identifier") or ""),
                sub_lib=str(_na(g.get("dataset_id")) or _na(g.get("acquisition")) or ""),
                smiles=_na(g.get("smiles")), inchi=_na(g.get("inchi")), adduct=add,
                ionization_mode=mode, precursor_mz=_float(g.get("precursor_mz")),
                instrument_type=itype, ms_level=int(msl) if math.isfinite(msl) else None,
                collision_energy=_na(g.get("collision_energy")),
                ce_units_hint="NCE" if source == "msnlib" else None,
                licence=SOURCE_INFO.get(source, {}).get("licence", ""),
                raw_intensities=False,
            )
            rec.mzs = _csv_floats(g.get("mzs"))
            rec.intensities = _csv_floats(g.get("intensities"))
            if source == "massspecgym" and _insilico(g.get("simulation_challenge") if "simulation_challenge" in cols else None):
                rec.drop = "in_silico"
            yield rec


def iter_enveda_jsonl(path: Path) -> Iterator[RawRecord]:
    """Enveda-180 Zenodo release, ``enveda-180.jsonl.gz`` (one JSON record per line). Key
    names are resolved case-insensitively from a list of aliases (checked against the file
    by ``scripts/build_extlib.py --stage headers``)."""
    alias = {
        "id": ("spectrum_id", "spectrumid", "id", "identifier", "scan_id", "feature_id"),
        "smiles": ("smiles", "normalized_smiles", "canonical_smiles", "structure_smiles"),
        "inchi": ("inchi",),
        "adduct": ("adduct", "precursor_type", "ion_type", "precursortype"),
        "mode": ("ionization_mode", "ion_mode", "ionmode", "polarity", "mode"),
        "prec": ("precursor_mz", "precursormz", "pepmass", "precursor"),
        "inst": ("instrument_type", "instrument", "source_instrument"),
        "ce": ("collision_energy", "collision_energy_ev", "ce", "collisionenergy"),
        "mzs": ("ms2_mzs", "mzs", "mz", "peaks_mz", "mz_array"),
        "ints": ("ms2_intensities", "intensities", "intensity", "peaks_intensity",
                 "intensity_array", "ms2_normalized_intensities"),
        "peaks": ("peaks", "spectrum", "peaks_json"),
        "level": ("ms_level", "mslevel"),
    }
    keymap: dict[str, str] | None = None
    with _open_text(path) as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                r = json.loads(line)
            except ValueError:
                yield RawRecord(source_id="", drop="bad_peaks")
                continue
            if keymap is None:
                low = {k.lower(): k for k in r}
                keymap = {}
                for a, names in alias.items():
                    for n in names:
                        if n in low:
                            keymap[a] = low[n]
                            break
                log.info("enveda-180 jsonl keys %s -> %s", sorted(r)[:40], keymap)

            def g(a: str) -> Any:
                k = keymap.get(a) if keymap else None
                return r.get(k) if k else None

            ce = g("ce")
            if isinstance(ce, (list, tuple)):
                ce = ",".join(str(x) for x in ce)
            mode = _na(g("mode"))
            if mode in ("+", "1", "pos"):
                mode = "positive"
            elif mode in ("-", "-1", "neg"):
                mode = "negative"
            lv = _float(g("level"))
            rec = RawRecord(
                source_id=str(g("id") if g("id") is not None else ""), sub_lib="enveda-180",
                smiles=_na(g("smiles")), inchi=_na(g("inchi")), adduct=_na(g("adduct")),
                ionization_mode=mode, precursor_mz=_float(g("prec")),
                instrument_type=_na(g("inst")) or "timsTOF",
                ms_level=int(lv) if math.isfinite(lv) else None,
                collision_energy=_na(ce), ce_units_hint="eV",
                licence="CC BY 4.0", raw_intensities=None,
            )
            mz, it = g("mzs"), g("ints")
            if mz is not None and it is not None:
                rec.mzs, rec.intensities = _csv_floats(mz), _csv_floats(it)
            elif g("peaks") is not None:
                pk = g("peaks")
                if isinstance(pk, str):
                    try:
                        pk = json.loads(pk)
                    except ValueError:
                        pk = None
                rec.mzs, rec.intensities = _peaks_from_pairs(pk)
            yield rec


def iter_kaggle_npy(meta_path: Path, mz_path: Path, in_path: Path) -> Iterator[RawRecord]:
    """``taeyangg4/casmi26-extlib-gnps-full``: ext_meta.parquet + ext_mz/ext_in float32
    [N, 128] (m/z ascending, sqrt(intensity / max), zero padded); intensities are squared
    back."""
    meta = pd.read_parquet(meta_path)
    mz = np.load(mz_path, mmap_mode="r")
    it = np.load(in_path, mmap_mode="r")
    for i, r in enumerate(meta.itertuples(index=False)):
        m = np.asarray(mz[i], np.float64)
        x = np.asarray(it[i], np.float64)
        ok = (m > 0) & (x > 0)
        lib = str(r.source_library)
        rec = RawRecord(
            source_id=str(r.source_id), sub_lib=lib, smiles=_na(r.normalized_smiles),
            adduct=_na(r.adduct), ionization_mode=_na(r.ionization_mode),
            precursor_mz=float(r.precursor_mz), instrument_type=_na(r.instrument_type),
            mzs=m[ok], intensities=x[ok] ** 2, licence=str(r.licence), raw_intensities=False,
        )
        if nist_flag(lib.split(":")[-1], library=lib.split(":")[-1]):
            rec.drop = "nist"
        elif lib.startswith("ms2deepscore"):
            rec.drop = "skipped_subset"   # training-set copies without upstream ids
        yield rec


# --------------------------------------------------------------------------------------
# Adducts
# --------------------------------------------------------------------------------------


def known_adducts(train_meta: Path | None = None) -> dict[str, int]:
    """Adduct strings of train (WP6 meta.parquet ``adduct``) with their spectrum counts."""
    import pyarrow.compute as pc

    p = train_meta or (paths.PROCESSED / "train" / "meta.parquet")
    t = pq.read_table(p, columns=["adduct"])
    vc = pc.value_counts(t["adduct"]).to_pylist()
    return {str(d["values"]): int(d["counts"]) for d in vc if d["values"] is not None}


class AdductMapper:
    """Map a source adduct string to the train spelling with the same numbers."""

    def __init__(self, known: dict[str, int]):
        from .adducts import parse_adduct

        self._parse = parse_adduct
        self.table: dict[tuple, str] = {}
        for name, n in sorted(known.items(), key=lambda kv: (-kv[1], kv[0])):
            a = parse_adduct(name)
            if a is None:
                continue
            key = self._key(a)
            self.table.setdefault(key, name)
        self._cache: dict[tuple, Any] = {}

    @staticmethod
    def _key(a) -> tuple:
        return (int(a.multiplier), int(a.charge), round(float(a.shift), 4))

    def candidates(self, s: str | None, mode: str | None) -> list[tuple[str, Any]] | None:
        """[(train adduct name, Adduct)] for ``s`` (several only for charge-less strings
        with multiple H, e.g. "M+2H", which may be charge 1 or 2); [] if it parses to no
        known form; None if it does not parse."""
        from .adducts import ELECTRON_MASS, Adduct

        key = (s, mode)
        if key in self._cache:
            return self._cache[key]
        out: list[tuple[str, Any]] | None
        a = self._parse(s, mode) if isinstance(s, str) else None
        if a is None and isinstance(s, str):
            # common GNPS forms that the general parser rejects
            t = s.strip().replace(" ", "")
            t = re.sub(r"^\[?M\]?\+$", "[M]+", t)
            t = re.sub(r"^\[?M\]?-$", "[M]-", t)
            a = self._parse(t, mode)
        if a is None:
            out = None
        else:
            out = []
            cands = [a]
            explicit = self._parse(s, None) is not None   # charge written in the string
            if not explicit:
                terms = a.shift + a.charge * ELECTRON_MASS
                sign = 1 if a.charge > 0 else -1
                for z in (2, 3):
                    cands.append(Adduct(a.multiplier, terms - sign * z * ELECTRON_MASS, sign * z))
            for c in cands:
                name = self.table.get(self._key(c))
                if name is not None:
                    out.append((name, c))
        self._cache[key] = out
        return out


# --------------------------------------------------------------------------------------
# Collision energy
# --------------------------------------------------------------------------------------

_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_CHARGE_FACTOR = {1: 1.0, 2: 0.9, 3: 0.85, 4: 0.8}


def parse_collision_energy(text: Any, precursor_mz: float, charge: int = 1,
                           units_hint: str | None = None) -> tuple[list[float] | None, str]:
    """(eV list or None, units in {eV, NCE, V, unknown}) for a collision-energy string."""
    s = _na(text)
    if s is None:
        return None, "unknown"
    low = s.lower()
    nums = [float(x) for x in _NUM_RE.findall(low)]
    nums = [abs(v) for v in nums if math.isfinite(v)]
    if not nums:
        return None, "unknown"
    if units_hint == "NCE" or "nce" in low or "%" in low or "nominal" in low or "normalized" in low:
        units = "NCE"
    elif "ev" in low:
        units = "eV"
    elif re.search(r"\d\s*v\b", low) or low.endswith("v"):
        units = "V"
    elif units_hint:
        units = units_hint
    elif re.fullmatch(r"[\d.,;\s-]+", low):
        units = "eV"
    else:
        return None, "unknown"
    if units == "NCE":
        if not math.isfinite(precursor_mz):
            return None, units
        f = _CHARGE_FACTOR.get(abs(int(charge)) or 1, 0.75)
        ev = [round(v * precursor_mz / 500.0 * f, 1) for v in nums]
    else:
        ev = [float(v) for v in nums]
    if any(v <= 0 or v > 500 for v in ev):
        return None, units
    return ev, units


# --------------------------------------------------------------------------------------
# Structures
# --------------------------------------------------------------------------------------


def resolve_structure(key: tuple[str, str]) -> tuple:
    """(status, smiles, inchikey, formula, mono_mass) for ("smiles"|"inchi", text).

    status is "ok" or a drop reason (bad_structure, multi_component, charged, isotope_label).
    """
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdMolDescriptors

    from .chem import mol_from_smiles, std_mol

    kind, text = key
    with rdBase.BlockLogs():
        try:
            if kind == "smiles":
                mol = mol_from_smiles(text)
            else:
                mol = Chem.MolFromInchi(text)
        except Exception:  # noqa: BLE001
            mol = None
        if mol is None or mol.GetNumAtoms() == 0:
            return ("bad_structure", None, None, None, float("nan"))
        if len(Chem.GetMolFrags(mol)) > 1:
            return ("multi_component", None, None, None, float("nan"))
        if any(a.GetIsotope() for a in mol.GetAtoms()):
            return ("isotope_label", None, None, None, float("nan"))
        std = std_mol(mol)
        if std is None:
            return ("bad_structure", None, None, None, float("nan"))
        if len(Chem.GetMolFrags(std)) > 1:
            return ("multi_component", None, None, None, float("nan"))
        if sum(a.GetFormalCharge() for a in std.GetAtoms()) != 0:
            return ("charged", None, None, None, float("nan"))
        try:
            smi = Chem.MolToSmiles(std)
            ik = Chem.MolToInchiKey(std)
            formula = rdMolDescriptors.CalcMolFormula(std)
            mass = float(rdMolDescriptors.CalcExactMolWt(std))
        except Exception:  # noqa: BLE001
            return ("bad_structure", None, None, None, float("nan"))
    if not smi or not ik or len(ik) < 14:
        return ("bad_structure", None, None, None, float("nan"))
    return ("ok", smi, ik, formula, mass)


def _resolve_chunk(keys: list) -> list:
    return [resolve_structure(k) for k in keys]


# --------------------------------------------------------------------------------------
# Staging
# --------------------------------------------------------------------------------------

_NORMALISED_MAX = (1.0, 100.0, 999.0, 1000.0, 10000.0, 100000.0)


class Stager:
    """Standardise, filter, clean and write the records of one source (module docstring)."""

    def __init__(self, source: str, cfg: Config, mapper: AdductMapper, out_dir: Path,
                 n_jobs: int = 1, batch_records: int = 20000, part: str = "part-000",
                 struct_cache: dict | None = None, raw_sink: Any = None):
        from .adducts import TEST_ADDUCTS
        from .library import instrument_family

        self.source = source
        self.cfg = cfg
        self.mapper = mapper
        self.out_dir = paths.ensure_dir(out_dir)
        self.n_jobs = max(1, int(n_jobs))
        self.batch_records = int(batch_records)
        self.part = part
        self.cache: dict = struct_cache if struct_cache is not None else {}
        self.counts: Counter = Counter()
        self.sub_counts: dict[str, Counter] = {}
        self.instrument_family = instrument_family
        self.test_adducts = set(TEST_ADDUCTS)
        self.ingest_lib = f"ext_{source}"
        self._writer: pq.ParquetWriter | None = None
        self._buf: list[RawRecord] = []
        self._seen: set = set()
        self._ids: Counter = Counter()
        self._pool = None
        self.n_in = 0
        self.n_out = 0
        self.raw_sink = raw_sink     # callable(rows: list[dict]) for raw peaks of kept rows

    # -- lifecycle ---------------------------------------------------------------------
    def __enter__(self) -> "Stager":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._buf:
            self._flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def add(self, rec: RawRecord) -> None:
        self.n_in += 1
        if rec.drop:
            self._count(rec, rec.drop)
            return
        self._buf.append(rec)
        if len(self._buf) >= self.batch_records:
            self._flush()

    def add_all(self, records: Iterable[RawRecord], log_every: int = 200000) -> None:
        t0 = time.perf_counter()
        for rec in records:
            self.add(rec)
            if self.n_in % log_every == 0:
                log.info("%s/%s: %d records read, %d staged (%.0f rec/s), cache %d structures",
                         self.source, self.part, self.n_in, self.n_out,
                         self.n_in / max(time.perf_counter() - t0, 1e-9), len(self.cache))

    def _count(self, rec: RawRecord | None, reason: str, n: int = 1) -> None:
        self.counts[reason] += n
        if rec is not None:
            self.sub_counts.setdefault(rec.sub_lib or "", Counter())[reason] += n

    # -- structures ----------------------------------------------------------------------
    def _structures(self, keys: list[tuple[str, str]]) -> None:
        todo = sorted({k for k in keys if k not in self.cache})
        if not todo:
            return
        if self.n_jobs > 1 and len(todo) >= 200:
            if self._pool is None:
                from .chem import mp_context

                self._pool = mp_context().Pool(self.n_jobs)
            step = max(50, len(todo) // (self.n_jobs * 8))
            chunks = [todo[i:i + step] for i in range(0, len(todo), step)]
            res: list = []
            for part in self._pool.imap(_resolve_chunk, chunks):
                res.extend(part)
        else:
            res = _resolve_chunk(todo)
        for k, r in zip(todo, res):
            self.cache[k] = r

    # -- one batch -----------------------------------------------------------------------
    def _flush(self) -> None:
        from .spectra import clean_flat

        recs, self._buf = self._buf, []
        keys: list[tuple[str, str] | None] = []
        for r in recs:
            if r.smiles:
                keys.append(("smiles", r.smiles))
            elif r.inchi:
                keys.append(("inchi", r.inchi))
            else:
                keys.append(None)
        self._structures([k for k in keys if k is not None])
        # InChI fallback when the SMILES fails
        alt = [("inchi", r.inchi) for r, k in zip(recs, keys)
               if k is not None and k[0] == "smiles" and r.inchi and self.cache[k][0] == "bad_structure"]
        if alt:
            self._structures(alt)
        keep: list[tuple[RawRecord, tuple, str, Any, float, int]] = []
        for r, k in zip(recs, keys):
            if r.ms_level is not None and r.ms_level != 2:
                self._count(r, "not_ms2")
                continue
            if not math.isfinite(r.precursor_mz) or r.precursor_mz <= 0:
                self._count(r, "no_precursor")
                continue
            if r.mzs.size == 0 or r.mzs.shape != r.intensities.shape:
                self._count(r, "bad_peaks")
                continue
            if k is None:
                self._count(r, "no_structure")
                continue
            st = self.cache[k]
            if st[0] == "bad_structure" and k[0] == "smiles" and r.inchi:
                st = self.cache.get(("inchi", r.inchi), st)
            if st[0] != "ok":
                self._count(r, st[0])
                continue
            mode = r.ionization_mode
            sign = 0
            if isinstance(mode, str):
                ml = mode.strip().lower()
                sign = 1 if ml.startswith("p") or ml == "+" else (-1 if ml.startswith("n") or ml == "-" else 0)
            cands = self.mapper.candidates(r.adduct, "positive" if sign > 0 else ("negative" if sign < 0 else None))
            if cands is None:
                self._count(r, "adduct_unparsed")
                continue
            if not cands:
                self._count(r, "adduct_unknown")
                continue
            best = None
            for name, a in cands:
                theo = a.precursor_mz(st[4])
                ppm = (r.precursor_mz - theo) / theo * 1e6
                if best is None or abs(ppm) < abs(best[2]):
                    best = (name, a, ppm)
            name, a, ppm = best
            pol = 1 if a.charge > 0 else -1
            if sign != 0 and sign != pol:
                self._count(r, "polarity_mismatch")
                continue
            if abs(ppm) > float(self.cfg.extlib_max_abs_ppm):
                self._count(r, "ppm_gt_max")
                continue
            keep.append((r, st, name, a, ppm, pol))
        if not keep:
            return
        # peaks: flat arrays, clean in one call
        lens = np.array([len(x[0].mzs) for x in keep], np.int64)
        off = np.zeros(len(keep) + 1, np.int64)
        np.cumsum(lens, out=off[1:])
        mz = np.concatenate([x[0].mzs for x in keep]).astype(np.float64)
        it = np.concatenate([x[0].intensities for x in keep]).astype(np.float64)
        prec = np.array([x[0].precursor_mz for x in keep], np.float64)
        cm, ci, co, pint, nraw = clean_flat(mz, it, off, prec, self.cfg)
        ncl = np.diff(co)
        rows: list[dict] = []
        raw_rows: list[dict] = []
        for j, (r, st, name, a, ppm, pol) in enumerate(keep):
            if ncl[j] < int(self.cfg.extlib_min_peaks):
                self._count(r, "lt_min_peaks")
                continue
            pm = cm[co[j]:co[j + 1]]
            pi = ci[co[j]:co[j + 1]]
            ik14 = st[2][:14]
            sig = hashlib.blake2b(
                (f"{ik14}|{name}|{r.precursor_mz:.3f}|".encode()
                 + np.round(pm, 2).astype(np.float32).tobytes()
                 + np.round(pi, 2).astype(np.float32).tobytes()), digest_size=12).digest()
            if sig in self._seen:
                self._count(r, "duplicate_in_source")
                continue
            self._seen.add(sig)
            raw_it = r.intensities[np.isfinite(r.intensities)]
            mx = float(raw_it.max()) if raw_it.size else float("nan")
            if r.raw_intensities is True:
                bpi = mx
            elif r.raw_intensities is False:
                bpi = float("nan")
            else:
                bpi = float("nan") if (not math.isfinite(mx) or any(abs(mx - v) < 1e-6 for v in _NORMALISED_MAX)
                                       or mx <= 1.0) else mx
            ev, units = parse_collision_energy(r.collision_energy, r.precursor_mz, a.charge, r.ce_units_hint)
            sid = r.source_id or f"row{self.n_in}"
            self._ids[sid] += 1
            if self._ids[sid] > 1:
                sid = f"{sid}#{self._ids[sid]}"
            rows.append({
                "spectrum_id": f"{self.ingest_lib}:{sid}", "source": self.source,
                "source_id": r.source_id, "sub_lib": r.sub_lib, "licence": r.licence,
                "ingest_lib": self.ingest_lib, "normalized_smiles": st[1], "inchikey": st[2],
                "inchikey14": ik14, "molecular_formula": st[3],
                "ionization_mode": "positive" if pol > 0 else "negative",
                "instrument_type": r.instrument_type, "adduct": name, "adduct_orig": r.adduct,
                "precursor_mz": float(r.precursor_mz), "precursor_error_ppm": abs(float(ppm)),
                "ms2_mzs": pm, "ms2_normalized_intensities": pi, "num_peaks": int(lens[j]),
                "base_peak_intensity": bpi, "collision_energy_ev": ev,
                "collision_energy_orig": _na(r.collision_energy),
                "collision_energy_orig_units": units, "polarity": pol,
                "instrument_family": self.instrument_family(r.instrument_type, self.ingest_lib),
                "mono_mass": float(st[4]), "n_peaks_raw": int(nraw[j]),
                "prec_intensity": float(pint[j]), "n_peaks": int(ncl[j]),
                "is_test_adduct": name in self.test_adducts,
            })
            if self.raw_sink is not None:
                raw_rows.append({"spectrum_id": rows[-1]["spectrum_id"], "inchikey14": ik14,
                                 "mzs": r.mzs, "intensities": r.intensities})
            self.sub_counts.setdefault(r.sub_lib or "", Counter())["staged"] += 1
        if not rows:
            return
        self.n_out += len(rows)
        self.counts["staged"] += len(rows)
        tab = pa.Table.from_pylist(rows, schema=STAGE_SCHEMA)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.out_dir / f"{self.part}.parquet", STAGE_SCHEMA,
                                            compression="zstd")
        self._writer.write_table(tab, row_group_size=50000)
        if self.raw_sink is not None and raw_rows:
            self.raw_sink(raw_rows)

    def info(self) -> dict:
        return {"source": self.source, "part": self.part, "n_in": self.n_in, "n_staged": self.n_out,
                "drops": {k: int(v) for k, v in sorted(self.counts.items())},
                "by_sub_lib": {s: {k: int(v) for k, v in sorted(c.items())}
                               for s, c in sorted(self.sub_counts.items())}}


# --------------------------------------------------------------------------------------
# Deduplication (finalize)
# --------------------------------------------------------------------------------------


def _cos_pairs(a_mz, a_int, a_off, b_mz, b_int, b_off, pa_, pb_, tol: float) -> tuple[np.ndarray, np.ndarray]:
    from .similarity import pair_scores_batch

    if len(pa_) == 0:
        return np.zeros(0), np.zeros(0, np.int32)
    cos, n, _, _ = pair_scores_batch(b_mz, b_int, b_off, a_mz, a_int, a_off,
                                     np.asarray(pb_, np.int64), np.asarray(pa_, np.int64), tol)
    return cos, n


def _flat(lists_mz: Sequence, lists_int: Sequence) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lens = np.array([len(x) for x in lists_mz], np.int64)
    off = np.zeros(len(lens) + 1, np.int64)
    np.cumsum(lens, out=off[1:])
    if len(lens) == 0 or off[-1] == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32), off
    mz = np.concatenate([np.asarray(x, np.float32) for x in lists_mz])
    it = np.concatenate([np.asarray(x, np.float32) for x in lists_int])
    return mz, it, off


def dedup(meta: pd.DataFrame, peaks: Any, train_combos: set, train_pairs: set,
          train_ik14: set, train_peaks: Any, cfg: Config) -> pd.Series:
    """Decision per staged spectrum (index aligned with ``meta``): "keep" or a drop reason.

    ``meta`` columns: row (position for ``peaks``), source, inchikey14, adduct,
    instrument_family, priority (``SOURCE_ORDER`` position). ``peaks(rows) -> (mz, int, off)``
    returns cleaned peaks of staged rows; ``train_peaks(pairs) -> dict[(ik14, adduct)] ->
    (mz, int, off)`` those of train spectra with that structure and adduct.

    Rules: a structure in train keeps a spectrum only if train has no spectrum with the same
    (ik14, adduct, instrument family) ("train_covered" otherwise) and the spectrum is not
    near-identical (cosine >= ``extlib_dup_cos`` at ``extlib_dup_tol_da``) to a train spectrum
    of the same (ik14, adduct) ("dup_train"); any spectrum near-identical to a kept spectrum
    of an earlier source (or an earlier row of its own source) with the same (ik14, adduct)
    is dropped ("dup_ext").
    """
    thr = float(cfg.extlib_dup_cos)
    tol = float(cfg.extlib_dup_tol_da)
    m = meta
    decision = pd.Series("keep", index=m.index, dtype=object)
    in_train = m["inchikey14"].isin(train_ik14).to_numpy()
    combo = (m["inchikey14"] + "|" + m["adduct"] + "|" + m["instrument_family"]).to_numpy(object)
    covered = in_train & np.fromiter((c in train_combos for c in combo), bool, len(m))
    decision[covered] = "train_covered"
    # train near-duplicates
    pair = (m["inchikey14"] + "|" + m["adduct"]).to_numpy(object)
    need_t = (~covered) & in_train & np.fromiter((p in train_pairs for p in pair), bool, len(m))
    if need_t.any():
        idx = np.flatnonzero(need_t)
        tp = train_peaks(sorted(set(pair[idx])))
        q_mz, q_int, q_off = peaks(m["row"].to_numpy()[idx], m["source"].to_numpy()[idx])
        for j, i in enumerate(idx):
            ref = tp.get(pair[i])
            if ref is None:
                continue
            r_mz, r_int, r_off = ref
            nref = len(r_off) - 1
            qm = q_mz[q_off[j]:q_off[j + 1]]
            qi = q_int[q_off[j]:q_off[j + 1]]
            cos, _ = _cos_pairs(qm, qi, np.array([0, len(qm)], np.int64), r_mz, r_int, r_off,
                                np.zeros(nref, np.int64), np.arange(nref), tol)
            if cos.size and float(cos.max()) >= thr:
                decision.iloc[i] = "dup_train"
    # cross-source / within-source near duplicates among the remaining rows
    alive = (decision == "keep").to_numpy()
    sub = m[alive].copy()
    sub["pair"] = pair[alive]
    sub["pos"] = np.flatnonzero(alive)
    grp_n = sub.groupby("pair")["pos"].transform("size")
    multi = sub[grp_n > 1].sort_values(["pair", "priority", "row"], kind="mergesort")
    if len(multi):
        q_mz, q_int, q_off = peaks(multi["row"].to_numpy(), multi["source"].to_numpy())
        pos = multi["pos"].to_numpy()
        pairs_arr = multi["pair"].to_numpy(object)
        start = 0
        n = len(multi)
        while start < n:
            end = start
            while end < n and pairs_arr[end] == pairs_arr[start]:
                end += 1
            kept: list[int] = []
            for j in range(start, end):
                if kept:
                    pa_ = np.full(len(kept), j, np.int64)
                    pb_ = np.asarray(kept, np.int64)
                    cos, _ = _cos_pairs(q_mz, q_int, q_off, q_mz, q_int, q_off, pa_, pb_, tol)
                    if float(cos.max()) >= thr:
                        decision.iloc[pos[j]] = "dup_ext"
                        continue
                kept.append(j)
            start = end
    return decision

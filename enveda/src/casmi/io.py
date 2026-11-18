"""Test-file reading, per-molecule neutral mass, submission writing and validation (WP1).

PLAN.md sections 3.2 (aggregating a molecule's spectra into M) and 4.8 (final list),
COMPETITION.md (submission format and rejection rules).

Public API:
- ``read_test(path=None) -> DataFrame``: test-shaped parquet, typed and sorted.
- ``mass_tables(test_df, cfg=None) -> (molecules, spectra)`` and its two halves
  ``molecule_table`` / ``spectrum_table``: neutral mass M per molecule (3.2).
- ``write_submission(preds, path, test_df=None)``: deterministic CSV, validated first.
- ``validate_submission(path_or_df, test_df=None) -> dict``: raises ``SubmissionError``
  for anything Kaggle rejects, returns a report with warnings otherwise.
"""

from __future__ import annotations

import csv
import io as _io
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import paths
from .adducts import TEST_ADDUCTS, Adduct, resolve_adduct
from .chem import mol_from_smiles
from .config import Config

__all__ = [
    "TEST_COLUMNS",
    "read_test",
    "mass_tables",
    "molecule_table",
    "spectrum_table",
    "SubmissionError",
    "write_submission",
    "validate_submission",
    "submission_frame",
]

log = logging.getLogger(__name__)

TEST_COLUMNS: tuple[str, ...] = (
    "molecule_id", "spectrum_id", "ms2_mzs", "ms2_normalized_intensities",
    "base_peak_intensity", "adduct", "ionization_mode", "instrument_type", "precursor_mz",
    "collision_energy_orig", "collision_energy_ev", "collision_energy_orig_units",
)
# Needed by the pipeline; the rest are passed through when present.
_REQUIRED = ("molecule_id", "spectrum_id", "ms2_mzs", "ms2_normalized_intensities",
             "adduct", "ionization_mode", "precursor_mz")

SUBMISSION_COLUMNS = ("molecule_id", "smiles")


# --- reading ----------------------------------------------------------------------------

def _float_array(x: Any, dtype: Any) -> np.ndarray:
    if x is None:
        return np.zeros(0, dtype=dtype)
    return np.asarray(x, dtype=dtype)


def read_test(path: str | Path | None = None) -> pd.DataFrame:
    """Read a test-shaped parquet (default ``paths.TEST_PARQUET``).

    Types: ids, adduct, ionization_mode and other text columns are str; precursor_mz and
    base_peak_intensity float64; ``ms2_mzs`` / ``ms2_normalized_intensities`` become
    float32 numpy arrays, ``collision_energy_ev`` float64 arrays (empty if missing).
    ``ionization_mode`` is lower-cased. Rows sorted by (molecule_id, spectrum_id), index
    reset. ValueError if a required column is missing or ids repeat.
    """
    p = Path(path) if path is not None else paths.TEST_PARQUET
    df = pd.read_parquet(p)
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{p}: missing columns {missing}")
    df = df.copy()
    if df["molecule_id"].isna().any() or df["spectrum_id"].isna().any():
        raise ValueError(f"{p}: null molecule_id or spectrum_id")
    df["molecule_id"] = df["molecule_id"].astype(str)
    df["spectrum_id"] = df["spectrum_id"].astype(str)
    df["ionization_mode"] = df["ionization_mode"].str.strip().str.lower()
    df["precursor_mz"] = df["precursor_mz"].astype(np.float64)
    if "base_peak_intensity" in df.columns:
        df["base_peak_intensity"] = df["base_peak_intensity"].astype(np.float64)
    df["ms2_mzs"] = [_float_array(x, np.float32) for x in df["ms2_mzs"]]
    df["ms2_normalized_intensities"] = [_float_array(x, np.float32)
                                        for x in df["ms2_normalized_intensities"]]
    if "collision_energy_ev" in df.columns:
        df["collision_energy_ev"] = [_float_array(x, np.float64) for x in df["collision_energy_ev"]]
    if df["spectrum_id"].duplicated().any():
        raise ValueError(f"{p}: repeated spectrum_id")
    df = df.sort_values(["molecule_id", "spectrum_id"], kind="mergesort").reset_index(drop=True)
    unknown = sorted({a for a in df["adduct"] if a not in TEST_ADDUCTS}, key=str)
    if unknown:
        log.warning("%s: adducts outside the ten test adducts: %s", p, unknown)
    log.info("read %s: %d spectra, %d molecules", p, len(df), df["molecule_id"].nunique())
    return df


# --- neutral mass per molecule (PLAN 3.2) ------------------------------------------------

def _clusters(masses: np.ndarray, tol_ppm: float) -> list[np.ndarray]:
    """Single-linkage clusters of masses: split sorted values where the gap exceeds
    ``tol_ppm`` of the lower value. Returns index arrays into ``masses``."""
    order = np.argsort(masses, kind="mergesort")
    m = masses[order]
    gaps = np.diff(m) / m[:-1] * 1e6
    cuts = np.flatnonzero(gaps > tol_ppm) + 1
    return [order[a:b] for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(m)])]


def mass_tables(test_df: pd.DataFrame, cfg: Config | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Neutral mass per molecule and per spectrum (PLAN 3.2).

    Per spectrum: M_i = neutral mass from precursor_mz and adduct (``resolve_adduct``:
    test table, general parser, else the polarity default with ``adduct_known=False``).
    Per molecule: M = median M_i; if some |M_i - M| / M >= ``cfg.mass_cluster_ppm``,
    cluster the M_i by single linkage at that ppm, keep the largest cluster (ties: the
    one containing ``[M+H]+`` or ``[M-H]-``, then the larger summed base_peak_intensity,
    then the lower mass), M = its median, the other spectra get ``mass_outlier=True``.

    Returns ``(molecules, spectra)``.
    molecules (one row per molecule_id, sorted): molecule_id, M, m_spread_ppm (max - min
    of the kept M_i over M, in ppm), n_spectra, n_kept, n_outliers, n_adducts (distinct
    adduct strings), adducts (sorted, ";"-joined), has_pos, has_neg, n_unknown_adduct,
    mass_clustered (True if the cluster fallback was used).
    spectra (aligned to ``test_df``'s index): spectrum_id, molecule_id, adduct,
    adduct_known, multiplier, shift, charge, m_spec (M_i), m_dev_ppm ((M_i - M) / M), and
    mass_outlier.
    """
    cfg = cfg or Config()
    tol = cfg.mass_cluster_ppm
    n = len(test_df)
    mult = np.empty(n, dtype=np.int64)
    shift = np.empty(n, dtype=np.float64)
    charge = np.empty(n, dtype=np.int64)
    known = np.empty(n, dtype=bool)
    cache: dict[tuple[Any, Any], tuple[Adduct, bool]] = {}
    for i, (a, mode) in enumerate(zip(test_df["adduct"], test_df["ionization_mode"])):
        k = (a, mode)
        if k not in cache:
            cache[k] = resolve_adduct(a, mode)
        add, ok = cache[k]
        mult[i], shift[i], charge[i], known[i] = add.multiplier, add.shift, add.charge, ok
    mz = test_df["precursor_mz"].to_numpy(dtype=np.float64)
    m_spec = (mz * np.abs(charge) - shift) / mult
    if "base_peak_intensity" in test_df.columns:
        bpi = np.nan_to_num(test_df["base_peak_intensity"].to_numpy(dtype=np.float64), nan=0.0)
    else:
        bpi = np.zeros(n)
    modes = test_df["ionization_mode"].to_numpy(dtype=object)
    pos = np.array([(md == "positive") if md in ("positive", "negative") else c > 0
                    for md, c in zip(modes, charge)], dtype=bool)
    adduct_str = test_df["adduct"].to_numpy(dtype=object)
    is_default = np.isin(adduct_str.astype(str), ["[M+H]+", "[M-H]-"])

    outlier = np.zeros(n, dtype=bool)
    m_mol = np.empty(n, dtype=np.float64)
    rows = []
    mol_ids = test_df["molecule_id"].to_numpy(dtype=object)
    order = np.argsort(mol_ids.astype(str), kind="mergesort")
    sorted_ids = mol_ids[order]
    bounds = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        ms = m_spec[idx]
        M = float(np.median(ms))
        kept = idx
        clustered = False
        if np.any(np.abs(ms - M) / M * 1e6 >= tol):
            clustered = True
            best = None
            for c in _clusters(ms, tol):
                members = idx[c]
                key = (len(members), bool(is_default[members].any()),
                       float(bpi[members].sum()), -float(np.median(m_spec[members])))
                if best is None or key > best[0]:
                    best = (key, members)
            assert best is not None
            kept = best[1]
            M = float(np.median(m_spec[kept]))
            outlier[np.setdiff1d(idx, kept)] = True
        m_mol[idx] = M
        mk = m_spec[kept]
        rows.append({
            "molecule_id": str(sorted_ids[a]),
            "M": M,
            "m_spread_ppm": float((mk.max() - mk.min()) / M * 1e6),
            "n_spectra": int(len(idx)),
            "n_kept": int(len(kept)),
            "n_outliers": int(len(idx) - len(kept)),
            "n_adducts": int(len(set(adduct_str[idx].astype(str)))),
            "adducts": ";".join(sorted(set(adduct_str[idx].astype(str)))),
            "has_pos": bool(pos[idx].any()),
            "has_neg": bool((~pos[idx]).any()),
            "n_unknown_adduct": int((~known[idx]).sum()),
            "mass_clustered": clustered,
        })
    molecules = pd.DataFrame(rows, columns=[
        "molecule_id", "M", "m_spread_ppm", "n_spectra", "n_kept", "n_outliers", "n_adducts",
        "adducts", "has_pos", "has_neg", "n_unknown_adduct", "mass_clustered"])
    spectra = pd.DataFrame({
        "spectrum_id": test_df["spectrum_id"].to_numpy(dtype=object),
        "molecule_id": mol_ids,
        "adduct": adduct_str,
        "adduct_known": known,
        "multiplier": mult,
        "shift": shift,
        "charge": charge,
        "m_spec": m_spec,
        "m_dev_ppm": (m_spec - m_mol) / m_mol * 1e6,
        "mass_outlier": outlier,
    }, index=test_df.index)
    n_out = int(outlier.sum())
    if n_out:
        log.warning("mass_tables: %d outlier spectra in %d molecules", n_out,
                    int(molecules["n_outliers"].gt(0).sum()))
    return molecules, spectra


def molecule_table(test_df: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Molecule-level half of ``mass_tables`` (one row per molecule_id, sorted)."""
    return mass_tables(test_df, cfg)[0]


def spectrum_table(test_df: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Spectrum-level half of ``mass_tables`` (M_i, adduct_known, mass_outlier ...)."""
    return mass_tables(test_df, cfg)[1]


# --- submissions -----------------------------------------------------------------------

class SubmissionError(ValueError):
    """A submission Kaggle would reject; ``errors`` lists every problem found."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = list(errors)
        super().__init__("invalid submission:\n  - " + "\n  - ".join(self.errors))


def _guesses(cell: str) -> list[str]:
    return cell.split(";")


def submission_frame(preds: Mapping[str, Sequence[str] | str] | pd.DataFrame) -> pd.DataFrame:
    """preds (dict molecule_id -> list of SMILES or ";"-joined string, or a DataFrame with
    molecule_id and smiles) -> DataFrame(molecule_id, smiles) sorted by molecule_id."""
    if isinstance(preds, pd.DataFrame):
        missing = [c for c in SUBMISSION_COLUMNS if c not in preds.columns]
        if missing:
            raise SubmissionError([f"missing columns: {missing}"])
        items = list(zip(preds["molecule_id"], preds["smiles"]))
    else:
        items = list(preds.items())
    ids, cells = [], []
    for mid, value in items:
        ids.append(str(mid))
        if isinstance(value, str):
            cells.append(value)
        elif value is None:
            cells.append(None)
        else:
            cells.append(";".join(str(v) for v in value))
    df = pd.DataFrame({"molecule_id": pd.Series(ids, dtype=object),
                       "smiles": pd.Series(cells, dtype=object)})
    return df.sort_values("molecule_id", kind="mergesort").reset_index(drop=True)


def write_submission(preds: Mapping[str, Sequence[str] | str] | pd.DataFrame,
                     path: str | Path, test_df: pd.DataFrame | None = None,
                     max_guesses: int = 25) -> dict[str, Any]:
    """Write ``molecule_id,smiles`` CSV: rows sorted by molecule_id, guesses joined by ";"
    in the given order, UTF-8, "\\n" line ends, no index, so equal input gives equal bytes.

    The frame is validated first (``validate_submission``, plus: no empty guess and no
    ";" or "," inside a SMILES); nothing is written if it fails. The file is written to a
    temp file and renamed. Returns the validation report.
    """
    df = submission_frame(preds)
    errors = []
    for mid, cell in zip(df["molecule_id"], df["smiles"]):
        if isinstance(cell, str) and cell:
            if any(g == "" for g in _guesses(cell)):
                errors.append(f"{mid}: empty guess")
            if "," in cell or "\n" in cell or '"' in cell:
                errors.append(f"{mid}: SMILES contains a comma, quote or newline")
    if errors:
        raise SubmissionError(errors)
    report = validate_submission(df, test_df=test_df, max_guesses=max_guesses)
    buf = _io.StringIO()
    writer = csv.writer(buf, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(SUBMISSION_COLUMNS)
    for mid, cell in zip(df["molecule_id"], df["smiles"]):
        writer.writerow([mid, cell])
    data = buf.getvalue().encode("utf-8")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{out.name}.", suffix=".tmp", dir=out.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    log.info("wrote %s: %d molecules, %d guesses", out, report["n_rows"], report["n_guesses"])
    return report


def validate_submission(path_or_df: str | Path | pd.DataFrame, test_df: pd.DataFrame | None = None,
                        max_guesses: int = 25, check_smiles: bool = True) -> dict[str, Any]:
    """Check a submission against the rules in COMPETITION.md.

    Errors (raise ``SubmissionError`` listing all of them): missing ``molecule_id`` or
    ``smiles`` column; no rows; null or empty cells; repeated molecule_id; more than
    ``max_guesses`` guesses in a cell; with ``test_df``: ids missing from the submission
    and ids not in the test file. A file is read like Kaggle would (``pd.read_csv``
    defaults), so an empty cell is a null.

    Warnings only (returned in the report and logged): guesses that RDKit cannot parse,
    empty guesses inside a cell (``"A;;B"``), extra columns.

    Returns {"n_rows", "n_guesses", "max_guesses", "n_unparseable", "warnings"}.
    """
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df
    else:
        try:
            df = pd.read_csv(path_or_df)
        except pd.errors.EmptyDataError:
            raise SubmissionError(["file is empty"]) from None
        except (pd.errors.ParserError, UnicodeDecodeError) as exc:
            raise SubmissionError([f"cannot parse CSV: {exc}"]) from None
    errors: list[str] = []
    warnings: list[str] = []
    missing_cols = [c for c in SUBMISSION_COLUMNS if c not in df.columns]
    if missing_cols:
        raise SubmissionError([f"missing columns: {missing_cols}"])
    extra = [c for c in df.columns if c not in SUBMISSION_COLUMNS]
    if extra:
        warnings.append(f"extra columns ignored: {extra}")
    if len(df) == 0:
        raise SubmissionError(["no rows"])

    ids = df["molecule_id"]
    if ids.isna().any():
        errors.append(f"{int(ids.isna().sum())} null molecule_id")
    id_str = ids[ids.notna()].astype(str)
    dup = sorted(set(id_str[id_str.duplicated()]))
    if dup:
        errors.append(f"repeated molecule_id: {dup[:10]}{' ...' if len(dup) > 10 else ''}")

    n_guesses = 0
    most = 0
    n_bad = 0
    bad_examples: list[str] = []
    empty_guess_ids: list[str] = []
    null_ids: list[str] = []
    too_many: list[str] = []
    for mid, cell in zip(ids, df["smiles"]):
        if cell is None or (isinstance(cell, float) and np.isnan(cell)) or (
                isinstance(cell, str) and cell.strip() == ""):
            null_ids.append(str(mid))
            continue
        guesses = _guesses(str(cell))
        n_guesses += len(guesses)
        most = max(most, len(guesses))
        if len(guesses) > max_guesses:
            too_many.append(f"{mid} ({len(guesses)})")
        if any(g.strip() == "" for g in guesses):
            empty_guess_ids.append(str(mid))
        if check_smiles:
            for g in guesses:
                if g.strip() and mol_from_smiles(g.strip()) is None:
                    n_bad += 1
                    if len(bad_examples) < 5:
                        bad_examples.append(f"{mid}: {g[:80]}")
    if null_ids:
        errors.append(f"{len(null_ids)} null or empty smiles cells: {null_ids[:10]}")
    if too_many:
        errors.append(f"more than {max_guesses} guesses: {too_many[:10]}")
    if empty_guess_ids:
        warnings.append(f"{len(empty_guess_ids)} cells contain an empty guess: {empty_guess_ids[:10]}")
    if n_bad:
        warnings.append(f"{n_bad} guesses do not parse in RDKit, e.g. {bad_examples}")

    if test_df is not None:
        want = set(test_df["molecule_id"].astype(str))
        have = set(id_str)
        missing = sorted(want - have)
        extra_ids = sorted(have - want)
        if missing:
            errors.append(f"{len(missing)} test molecule_ids missing: {missing[:10]}")
        if extra_ids:
            errors.append(f"{len(extra_ids)} molecule_ids not in the test file: {extra_ids[:10]}")

    if errors:
        raise SubmissionError(errors)
    for w in warnings:
        log.warning("submission: %s", w)
    return {"n_rows": int(len(df)), "n_guesses": int(n_guesses), "max_guesses": int(most),
            "n_unparseable": int(n_bad), "warnings": warnings}

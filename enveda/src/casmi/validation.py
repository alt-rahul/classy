"""Validation harness: query emulation, conditions C1/C2/C3, scoring and reporting (WP7).

PLAN.md sections 5.2 (queries), 5.3 (conditions), 5.5 (reporting), 5.6 (pubmed_count = 0 split,
mass terciles).

Query tables (``build_queries``)
--------------------------------
Per holdout set, ``PROCESSED/validation/queries_<set>.parquet`` has exactly the 12 columns of
``test.parquet`` (``io.TEST_COLUMNS``, same Arrow types) so that ``scripts/predict.py`` and
``pipeline.predict`` run on it unchanged; the truth is in a separate file
``truth_<set>.parquet`` (TRUTH_COLUMNS), never in the query table. Emulation (5.2):

- candidate spectra of a molecule: its primary ik14 and aliases (same mkey) in the query
  library, ``query_ok`` from WP6 (one of the ten test adducts, base peak >= 1000 where known,
  |M error| <= ``val_query_max_abs_ppm``);
- number of spectra k: drawn from the public test distribution of spectra per molecule,
  capped by what is available and by ``val_max_spectra_per_molecule``;
- which k: weighted sampling without replacement (Efraimidis-Spirakis) with weight
  p_test(ce class) / p_molecule(ce class), ce class in {20, 40, 60, merged, other, unknown}
  (nearest of 20/40/60 eV within 2.5 eV for single energies); NP-OTHER spectra from QTOF
  instruments get 4x weight ("prefer QTOF");
- peaks: the raw train peaks (``--raw``, default data/raw/train.parquet, streamed) with the
  test cleaning: peaks above precursor + ``val_query_peak_cut_da`` dropped, intensities
  renormalised to a base peak of 1. Without the raw file the cleaned WP6 peaks are used
  (logged; ``peaks_source`` in the truth table);
- adduct written as the canonical test name, ionization_mode "positive" / "negative",
  null instrument_type as "unknown";
- molecule ids ``<NT|ST|NO><00000>`` in sha1(seed, mkey) order (anonymous), spectrum ids
  from train.

Conditions (``condition_exclusions``), per molecule, at lookup time
--------------------------------------------------------------------
- C1: library spectra of the molecule's ik14 group from its query library are removed
  (``Library.spectra_mask(ik14=group, ingest_lib=[query_lib])``). Molecules without
  remaining spectra elsewhere are not C1-eligible (checked against the library when one is
  given, else ``eligible_c1`` from the split).
- C2: all library spectra of the group are removed.
- C3: as C2, and every DB row whose ik14 is in the molecule's truth keys is removed: the
  group's ik14 plus every NP / PubChem row with the same formula, the same heavy-atom
  skeleton (RDKit ``MolHash`` ElementGraph, stereo removed) and the truth's mkey
  (``truth_keys``; tautomers keep the skeleton, so the skeleton prefilter loses nothing and
  replaces the plan's Tanimoto prefilter; the result is cached in
  ``PROCESSED/validation/truth_keys.parquet``).

How the exclusions reach the pipeline: ``pipeline.run(..., db=ExcludingDB(...),
extras={"validation": hooks, "library": lib})``.

- ``ExcludingDB`` wraps ``casmi.db.CandidateDB``: ``lookup_batch`` drops the excluded ik14 of
  each row's own query key (the pipeline passes molecule ids as keys, so this is exact per
  molecule); ``lookup`` (no key), ``load_np`` / ``np_mass`` (nearest-mass padding) drop the
  union over the molecules of the current chunk. It records every candidate ik14 it returns
  per molecule (candidate recall) and checks every result (violations fail the run).
- ``library_guard`` patches ``Library.search_by_mass``, ``search_open`` and
  ``candidates_in_window`` for the duration of a cell: the molecule is resolved from the
  ``queries`` argument (molecule_id column or the QuerySet's spectrum ids), else the union over
  the chunk is used. Library stages (WP8+) may instead call
  ``state.extras["validation"].library_mask(lib, molecule_id)`` and pass the mask themselves
  (OR-ed with the guard's mask, so doing both is harmless).
- Molecules are sent to the predictor in chunks of ``batch`` (default
  ``cfg.batch_molecules``) sorted by mass; ``batch=1`` makes even the union fallbacks exact.

Checks (raise AssertionError): every lookup result is free of the molecule's exclusions;
under C2 and C3 no library spectrum of the molecule remains (every library structure whose
mkey is the truth mkey is in the masked group); under C3 the truth is absent from every
retrieved pool and, unless a generator stage is configured, from the predictions; the truth
predictor scores exactly 1.0 and the CCO predictor 0.0 in every cell (``self_check``).

Reporting (``summarize``)
-------------------------
Per (set, condition) cell: n, MRR@25, top-1/5/25 hit rates, candidate recall (truth key among
the candidates the predictor retrieved: ``recall_main`` at ``cfg.mass_tol_ppm``,
``recall_any`` over every lookup, padding included), C2 split by pubmed_count = 0 (5.6a), MRR
by mass tercile. Per set: blended MRR for every mix in ``cfg.val_blend_mixes`` (C1/C2/C3
weights), the worst-case blend and its bootstrap interval over molecules
(``cfg.val_bootstrap_n`` resamples, ``cfg.val_bootstrap_ci``; C1 averages over the eligible
molecules of each resample). The primary number is the NP-TIMS worst-case blend, NP-OTHER
breaks ties.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import paths
from .config import Config
from .io import TEST_COLUMNS
from .splits import SET_CODES, SETS, hash_uniform, holdout_molecules

__all__ = [
    "CONDITIONS",
    "TRUTH_COLUMNS",
    "RESULT_COLUMNS",
    "validation_dir",
    "set_slug",
    "ce_class",
    "test_reference",
    "build_queries",
    "load_queries",
    "skeleton_key",
    "truth_keys",
    "MoleculeExclusion",
    "condition_exclusions",
    "ExcludingDB",
    "ValidationHooks",
    "library_guard",
    "truth_predictor",
    "cco_predictor",
    "pipeline_predictor",
    "run_cell",
    "evaluate",
    "summarize",
    "self_check",
    "format_cells",
    "append_experiments",
]

log = logging.getLogger(__name__)

CONDITIONS: tuple[str, ...] = ("C1", "C2", "C3")
CE_CLASSES = ("20", "40", "60", "merged", "other", "unknown")
TRUTH_COLUMNS = ("molecule_id", "set", "fold", "mkey", "ik14", "ik14_group", "smiles", "formula",
                 "mono_mass", "eligible_c1", "query_lib", "n_query_spectra", "n_available",
                 "pubmed_count", "in_np", "in_pubchem", "np_score", "mass_tercile",
                 "peaks_source")
RESULT_COLUMNS = ("set", "condition", "molecule_id", "mkey", "fold", "in_cell", "rr", "rank",
                  "top1", "top5", "top25", "in_pool_main", "in_pool_any", "n_pool_main",
                  "n_pool_any", "n_guesses", "truth_in_preds", "pubmed0", "mass_tercile",
                  "n_query_spectra")
_EMPTY: frozenset = frozenset()


def validation_dir() -> Path:
    return paths.PROCESSED / "validation"


def set_slug(name: str) -> str:
    """"NP-TIMS" -> "np_tims" (file names)."""
    return name.lower().replace("-", "_")


# --------------------------------------------------------------------------------------
# reference distributions from the public test file
# --------------------------------------------------------------------------------------

def ce_class(n_energies: int, ce: float) -> str:
    """Collision-energy class of a spectrum (see module docstring)."""
    if n_energies is None or n_energies <= 0 or not np.isfinite(ce):
        return "unknown"
    if n_energies > 1:
        return "merged"
    ce = abs(ce)
    for v in (20.0, 40.0, 60.0):
        if abs(ce - v) <= 2.5:
            return f"{v:g}"
    return "other"


def test_reference(test_df: pd.DataFrame) -> dict[str, Any]:
    """Spectra-per-molecule pmf and CE-class pmf of a test-shaped frame."""
    n = test_df.groupby("molecule_id").size().to_numpy()
    counts = np.bincount(n)
    pmf = counts / counts.sum()
    cls = []
    for ev in test_df["collision_energy_ev"]:
        a = np.asarray(ev if ev is not None else [], np.float64)
        a = np.abs(a[np.isfinite(a)])
        cls.append(ce_class(len(a), float(a.mean()) if len(a) else float("nan")))
    c = pd.Series(cls).value_counts(normalize=True)
    return {"spectra_pmf": pmf.tolist(), "ce_pmf": {k: float(c.get(k, 0.0)) for k in CE_CLASSES},
            "n_molecules": int(len(n)), "n_spectra": int(len(test_df))}


# --------------------------------------------------------------------------------------
# query emulation
# --------------------------------------------------------------------------------------

def _mol_rng(cfg: Config, *salt: Any) -> np.random.Generator:
    h = hashlib.sha1("|".join(str(s) for s in (cfg.seed, *salt)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


def _choose(cands: pd.DataFrame, ref: dict, cfg: Config, set_name: str, mkey: str) -> np.ndarray:
    """Row positions (into ``cands``) of the query spectra of one molecule."""
    rng = _mol_rng(cfg, set_name, mkey, "queries")
    pmf = np.asarray(ref["spectra_pmf"], np.float64)
    k = int(rng.choice(len(pmf), p=pmf)) if pmf.sum() > 0 else 3
    k = max(1, min(k, len(cands), int(cfg.val_max_spectra_per_molecule)))
    cls = np.array([ce_class(int(a), float(b)) for a, b in
                    zip(cands["n_energies"], cands["collision_energy"])], dtype=object)
    p_test = np.array([ref["ce_pmf"].get(c, 0.0) for c in cls])
    vals, inv, cnt = np.unique(cls, return_inverse=True, return_counts=True)
    p_mol = (cnt / cnt.sum())[inv]
    w = np.where(p_test > 0, p_test, 0.05) / p_mol
    if set_name == "NP-OTHER":
        w = w * np.where(cands["instrument_family"].to_numpy(object) == "qtof", 4.0, 1.0)
    u = rng.random(len(cands))
    keys = np.log(np.maximum(u, 1e-300)) / w
    order = np.lexsort((cands["spectrum_id"].astype(str).to_numpy(), -keys))
    return np.sort(order[:k])


def _test_clean(mz: np.ndarray, it: np.ndarray, prec: float, cut: float) -> tuple[list, list]:
    mz = np.asarray(mz if mz is not None else [], np.float64)
    it = np.asarray(it if it is not None else [], np.float64)
    ok = np.isfinite(mz) & np.isfinite(it) & (it > 0) & (mz > 0) & (mz <= prec + cut)
    mz, it = mz[ok], it[ok]
    o = np.argsort(mz, kind="mergesort")
    mz, it = mz[o], it[o]
    if it.size:
        it = it / it.max()
    return mz.tolist(), it.tolist()


_RAW_COLS = ("spectrum_id", "ms2_mzs", "ms2_normalized_intensities", "base_peak_intensity",
             "ionization_mode", "instrument_type", "precursor_mz", "collision_energy_orig",
             "collision_energy_ev", "collision_energy_orig_units")


def _fetch_raw(raw: Path, rows: np.ndarray, batch_rows: int = 50000) -> pd.DataFrame:
    """Rows ``rows`` (global positions) of a train-schema parquet, streamed by batch."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    want = np.unique(np.asarray(rows, np.int64))
    pf = pq.ParquetFile(raw, buffer_size=1 << 22, pre_buffer=False)
    # the real train file has no spectrum_id (WP6 ids are t_<row>); only peaks are used here
    cols = [c for c in _RAW_COLS if c in pf.schema_arrow.names]
    parts, row0 = [], 0
    for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
        n = batch.num_rows
        a, b = np.searchsorted(want, [row0, row0 + n])
        if b > a:
            t = pa.Table.from_batches([batch]).take(pa.array(want[a:b] - row0))
            d = t.to_pandas()
            d["source_row"] = want[a:b]
            parts.append(d)
        row0 += n
        if not len(want) or row0 > want[-1]:
            break
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[*_RAW_COLS, "source_row"])
    return out.set_index("source_row")


def _fetch_clean(train_dir: Path, sel: pd.DataFrame) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.parquet as pq

    parts = []
    for lib, g in sel.groupby("ingest_lib"):
        f = train_dir / "spectra" / _lib_file(lib)
        t = pq.read_table(f, columns=list(_RAW_COLS)).take(pa.array(g["file_row"].to_numpy(np.int64)))
        d = t.to_pandas()
        d["source_row"] = g["source_row"].to_numpy()
        parts.append(d)
    return pd.concat(parts, ignore_index=True).set_index("source_row")


def _lib_file(lib: str) -> str:
    import re

    return "spectra_" + re.sub(r"[^A-Za-z0-9._-]", "_", lib) + ".parquet"


def build_queries(holdout: pd.DataFrame, train_dir: Path | str | None = None,
                  cfg: Config | None = None, raw: Path | str | None = None,
                  test_path: Path | str | None = None, out_dir: Path | str | None = None,
                  sets: Sequence[str] = SETS) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Query and truth tables per set (module docstring); written to ``out_dir``."""
    from .io import read_test

    cfg = cfg or Config()
    t0 = time.perf_counter()
    train_dir = Path(train_dir) if train_dir is not None else paths.PROCESSED / "train"
    out = paths.ensure_dir(out_dir if out_dir is not None else validation_dir())
    raw = Path(raw) if raw is not None else paths.TRAIN_PARQUET
    ref = test_reference(read_test(test_path))
    mols = holdout_molecules(holdout)
    mols = mols[mols["set"].isin(sets)].reset_index(drop=True)
    st = pd.read_parquet(train_dir / "structures.parquet")
    st_i = st.set_index("ik14")
    meta = pd.read_parquet(train_dir / "meta.parquet", columns=[
        "spectrum_id", "ingest_lib", "file_row", "source_row", "ik14", "adduct_test", "polarity",
        "instrument_type", "instrument_family", "collision_energy", "n_energies", "query_ok"])
    meta = meta[meta["query_ok"]]
    grp = mols[["mkey", "query_lib", "ik14_group"]].explode("ik14_group").rename(
        columns={"ik14_group": "ik14"})
    cand = meta.merge(grp, left_on=["ik14", "ingest_lib"], right_on=["ik14", "query_lib"])
    by_key = {k: g.reset_index(drop=True) for k, g in cand.groupby("mkey", sort=False)}
    chosen, n_avail = [], {}
    for r in mols.itertuples(index=False):
        c = by_key.get(r.mkey)
        n_avail[r.mkey] = 0 if c is None else len(c)
        if c is None or not len(c):
            continue
        c = c.sort_values(["spectrum_id"], kind="mergesort").reset_index(drop=True)
        pick = _choose(c, ref, cfg, r.set, r.mkey)
        chosen.append(c.iloc[pick].assign(set=r.set))
    sel = pd.concat(chosen, ignore_index=True) if chosen else meta.iloc[:0].assign(set=[], mkey=[])
    if raw.exists():
        peaks = _fetch_raw(raw, sel["source_row"].to_numpy())
        source = "raw"
    else:
        log.warning("raw train file %s not found: query peaks come from the cleaned WP6 spectra", raw)
        peaks = _fetch_clean(train_dir, sel)
        source = "cleaned"
    result = {}
    info = {"reference": ref, "peaks_source": source, "sets": {}}
    for s in sets:
        ms = mols[mols["set"] == s].copy()
        u = hash_uniform(ms["mkey"], cfg.seed, "qid")
        ms = ms.iloc[np.lexsort((ms["mkey"].astype(str).to_numpy(), u))].reset_index(drop=True)
        ms["molecule_id"] = [f"{SET_CODES[s]}{i:05d}" for i in range(len(ms))]
        idmap = dict(zip(ms["mkey"], ms["molecule_id"]))
        ss = sel[sel["set"] == s]
        qrows = []
        for r in ss.itertuples(index=False):
            p = peaks.loc[r.source_row]
            mz, it = _test_clean(p["ms2_mzs"], p["ms2_normalized_intensities"], float(p["precursor_mz"]),
                                 cfg.val_query_peak_cut_da)
            if not mz:
                continue
            ev = p["collision_energy_ev"]
            itype = p["instrument_type"]
            qrows.append({
                "molecule_id": idmap[r.mkey], "spectrum_id": str(r.spectrum_id),
                "ms2_mzs": mz, "ms2_normalized_intensities": it,
                "base_peak_intensity": float(p["base_peak_intensity"]) if p["base_peak_intensity"] is not None else np.nan,
                "adduct": r.adduct_test, "ionization_mode": "positive" if r.polarity > 0 else "negative",
                "instrument_type": itype if isinstance(itype, str) and itype else "unknown",
                "precursor_mz": float(p["precursor_mz"]),
                "collision_energy_orig": p["collision_energy_orig"],
                "collision_energy_ev": None if ev is None else [float(x) for x in ev],
                "collision_energy_orig_units": p["collision_energy_orig_units"],
            })
        q = pd.DataFrame(qrows, columns=list(TEST_COLUMNS))
        nq = q.groupby("molecule_id").size()
        ms["n_query_spectra"] = ms["molecule_id"].map(nq).fillna(0).astype(int)
        dropped = int((ms["n_query_spectra"] == 0).sum())
        ms = ms[ms["n_query_spectra"] > 0].reset_index(drop=True)
        pr = st_i.reindex(ms["ik14"])
        terc = np.full(len(ms), -1)
        if len(ms) >= 3:
            edges = np.quantile(ms["mono_mass"], [1 / 3, 2 / 3])
            terc = np.searchsorted(edges, ms["mono_mass"].to_numpy(), side="right")
        truth = pd.DataFrame({
            "molecule_id": ms["molecule_id"], "set": s, "fold": ms["fold"].astype(int),
            "mkey": ms["mkey"], "ik14": ms["ik14"],
            "ik14_group": [";".join(g) for g in ms["ik14_group"]],
            "smiles": pr["smiles"].to_numpy(object), "formula": ms["formula"],
            "mono_mass": ms["mono_mass"].astype(float), "eligible_c1": ms["eligible_c1"].astype(bool),
            "query_lib": ms["query_lib"], "n_query_spectra": ms["n_query_spectra"],
            "n_available": ms["mkey"].map(n_avail).fillna(0).astype(int).to_numpy(),
            "pubmed_count": (pd.array(pr["pubmed_count"].to_numpy(), dtype="Int32")
                             if "pubmed_count" in pr.columns else pd.array([pd.NA] * len(ms), dtype="Int32")),
            "in_np": pr["in_np"].fillna(False).astype(bool).to_numpy() if "in_np" in pr.columns else False,
            "in_pubchem": (pr["in_pubchem"].astype("boolean").fillna(False).astype(bool).to_numpy()
                           if "in_pubchem" in pr.columns else False),
            "np_score": pr["np_score"].to_numpy(np.float64) if "np_score" in pr.columns else np.nan,
            "mass_tercile": terc.astype(int), "peaks_source": source,
        }, columns=list(TRUTH_COLUMNS))
        q = q[q["molecule_id"].isin(set(truth["molecule_id"]))]
        q = q.sort_values(["molecule_id", "spectrum_id"], kind="mergesort").reset_index(drop=True)
        _write_queries(q, out / f"queries_{set_slug(s)}.parquet")
        truth.to_parquet(out / f"truth_{set_slug(s)}.parquet", index=False)
        result[s] = (q, truth)
        info["sets"][s] = {"molecules": int(len(truth)), "spectra": int(len(q)),
                           "dropped_no_spectra": dropped,
                           "spectra_per_molecule_mean": float(len(q) / max(len(truth), 1)),
                           "eligible_c1": int(truth["eligible_c1"].sum())}
        log.info("queries %s: %d molecules, %d spectra (%.2f per molecule), %d without spectra",
                 s, len(truth), len(q), len(q) / max(len(truth), 1), dropped)
    info["holdout_sha256"] = _holdout_hash(holdout)
    info["seconds"] = round(time.perf_counter() - t0, 1)
    (out / "queries_info.json").write_text(json.dumps(info, indent=2))
    return result


def _write_queries(q: pd.DataFrame, path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([
        ("molecule_id", pa.large_string()), ("spectrum_id", pa.large_string()),
        ("ms2_mzs", pa.list_(pa.float64())), ("ms2_normalized_intensities", pa.list_(pa.float64())),
        ("base_peak_intensity", pa.float64()), ("adduct", pa.large_string()),
        ("ionization_mode", pa.large_string()), ("instrument_type", pa.large_string()),
        ("precursor_mz", pa.float64()), ("collision_energy_orig", pa.large_string()),
        ("collision_energy_ev", pa.list_(pa.float64())),
        ("collision_energy_orig_units", pa.large_string())])
    pq.write_table(pa.Table.from_pandas(q, schema=schema, preserve_index=False), path)


def _holdout_hash(holdout: pd.DataFrame) -> str:
    cols = [c for c in ("ik14", "mkey", "set", "fold", "eligible_c1", "query_lib", "primary")
            if c in holdout.columns]
    h = holdout[cols].astype(str).sort_values(cols).to_csv(index=False)
    return hashlib.sha256(h.encode()).hexdigest()


def load_queries(set_name: str, val_dir: Path | str | None = None
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(queries as ``io.read_test`` returns them, truth) of one set."""
    from .io import read_test

    d = Path(val_dir) if val_dir is not None else validation_dir()
    q = read_test(d / f"queries_{set_slug(set_name)}.parquet")
    t = pd.read_parquet(d / f"truth_{set_slug(set_name)}.parquet")
    return q, t


def queries_current(holdout: pd.DataFrame, val_dir: Path | str | None = None,
                    sets: Sequence[str] = SETS) -> bool:
    """True if query files exist for ``sets`` and were built from this holdout."""
    d = Path(val_dir) if val_dir is not None else validation_dir()
    p = d / "queries_info.json"
    if not p.exists():
        return False
    info = json.loads(p.read_text())
    if info.get("holdout_sha256") != _holdout_hash(holdout):
        return False
    return all((d / f"queries_{set_slug(s)}.parquet").exists() for s in sets)


# --------------------------------------------------------------------------------------
# truth keys: DB rows that are the truth by mkey
# --------------------------------------------------------------------------------------

def skeleton_key(smiles: str | None) -> str | None:
    """Heavy-atom graph with element labels, bond orders, charges, H and stereo ignored
    (RDKit MolHash ElementGraph). Tautomers and protonation variants share it."""
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdMolHash

    from .chem import mol_from_smiles

    m = mol_from_smiles(smiles)
    if m is None:
        return None
    with rdBase.BlockLogs():
        try:
            m = Chem.Mol(m)
            Chem.RemoveStereochemistry(m)
            return rdMolHash.MolHash(m, rdMolHash.HashFunction.ElementGraph)
        except Exception:  # noqa: BLE001
            return None


_BRACKET_RE = None
_RING_RE = None


def ring_closures(smiles: str) -> int:
    """Ring-closure pairs in a SMILES (digits and %nn outside brackets, halved). For a
    connected molecule this is the cycle rank of the heavy-atom graph, which tautomers
    share, so it is a sound string-level prefilter for ``skeleton_key``."""
    import re

    global _BRACKET_RE, _RING_RE
    if _BRACKET_RE is None:
        _BRACKET_RE = re.compile(r"\[[^\]]*\]")
        _RING_RE = re.compile(r"%\d\d|\d")
    if not isinstance(smiles, str):
        return -1
    return len(_RING_RE.findall(_BRACKET_RE.sub("", smiles))) // 2


def _skeleton_chunk(smiles: list[str]) -> list[str | None]:
    return [skeleton_key(s) for s in smiles]


def _formula_counts(f: Any) -> tuple | None:
    import re

    from .adducts import parse_formula

    if not isinstance(f, str):
        return None
    c = parse_formula(re.sub(r"[+-]\d*$", "", f.strip()))
    return tuple(sorted(c.items())) if c else None


def _truth_key_rows(dbh: Any, todo: pd.DataFrame, tiers: tuple, n_jobs: int) -> tuple[pd.DataFrame, dict]:
    """Same-formula, same-ring-count, same-skeleton DB rows for the molecules of ``todo``
    (columns query = truth mkey, ik14, smiles) and the funnel counts."""
    from .chem import parallel_map

    res = dbh.lookup_batch(todo["mono_mass"].to_numpy(np.float64), tol_ppm=1.0, tiers=tiers,
                           min_tol_da=0.001, keys=todo["mkey"].tolist())
    n_window = len(res)
    tf = dict(zip(todo["mkey"], [_formula_counts(f) for f in todo["formula"]]))
    tsk = dict(zip(todo["mkey"], [skeleton_key(s) for s in todo["smiles"]]))
    trc = dict(zip(todo["mkey"], [ring_closures(s) for s in todo["smiles"]]))
    fmap = {f: _formula_counts(f) for f in pd.unique(res["formula"].astype(object))}
    q = res["query"].astype(object).to_numpy(object)
    ok = np.fromiter((fmap.get(f) is not None and fmap.get(f) == tf.get(k)
                      for k, f in zip(q, res["formula"].astype(object))), bool, len(res))
    res = res.loc[ok, ["query", "ik14", "smiles"]]
    n_formula = len(res)
    q = res["query"].astype(object).to_numpy(object)
    ok = np.fromiter((ring_closures(sm) == trc.get(k) for k, sm in
                      zip(q, res["smiles"].astype(object))), bool, len(res))
    res = res[ok]
    n_ring = len(res)
    uniq = sorted(set(res["smiles"].astype(object)) - {None})
    chunks = [uniq[i:i + 5000] for i in range(0, len(uniq), 5000)]
    skm = dict(zip(uniq, [k for part in parallel_map(_skeleton_chunk, chunks, n_jobs)
                          for k in part]))
    res = res[[skm.get(sm) is not None and skm.get(sm) == tsk.get(k)
               for k, sm in zip(res["query"], res["smiles"])]].reset_index(drop=True)
    return res, {"window": n_window, "formula": n_formula, "ring": n_ring,
                 "unique_smiles": len(uniq), "skeleton": len(res)}


def truth_keys(truth: pd.DataFrame, cfg: Config | None = None, db: Any = None, n_jobs: int = 1,
               cache_path: Path | str | None = None, chunk: int = 200) -> dict[str, frozenset]:
    """molecule_id -> ik14 that are the truth: the ik14 group plus DB rows (all tiers
    present) with the truth's formula, skeleton and mkey. Cached by (mkey, tiers).

    Molecules are processed ``chunk`` at a time in mass order and the cache is written after
    every chunk: the 1 ppm window (at least 0.001 Da) holds about 6,000 PubChem rows per
    molecule, so one lookup for thousands of molecules does not fit in RAM (13 GB and an
    OOM kill for 3,000 SYN-TIMS molecules)."""
    from . import db as cdb
    from .metric import mkeys

    cfg = cfg or Config()
    dbh = db if db is not None else cdb.get_db(cfg)
    tiers = ("np", "pubchem") if dbh.pubchem_available() else ("np",)
    tiers_s = ",".join(tiers)
    cp = Path(cache_path) if cache_path is not None else validation_dir() / "truth_keys.parquet"
    cache: dict[str, frozenset] = {}
    if cp.exists():
        c = pd.read_parquet(cp)
        c = c[c["tiers"] == tiers_s]
        cache = {k: frozenset(v.split(";")) if v else _EMPTY for k, v in zip(c["mkey"], c["db_ik14"])}
    todo_all = truth[~truth["mkey"].isin(set(cache))].drop_duplicates("mkey")
    todo_all = todo_all.iloc[np.lexsort((todo_all["mkey"].astype(str).to_numpy(),
                                         todo_all["mono_mass"].to_numpy(np.float64)))]
    step = max(1, int(chunk))
    t0 = time.perf_counter()
    tot: dict[str, int] = defaultdict(int)
    for start in range(0, len(todo_all), step):
        todo = todo_all.iloc[start:start + step]
        res, cnt = _truth_key_rows(dbh, todo, tiers, n_jobs)
        for k, v in cnt.items():
            tot[k] += v
        keys = mkeys(res["smiles"].tolist(), cfg, n_jobs=n_jobs, cache=True) if len(res) else []
        res = res.assign(_mk=keys)
        hit = res[res["_mk"] == res["query"]]
        tot["hit"] += len(hit)
        found = hit.groupby("query")["ik14"].agg(lambda s: frozenset(map(str, s))).to_dict()
        new_rows = []
        for k in todo["mkey"]:
            v = found.get(k, _EMPTY)
            cache[k] = v
            new_rows.append({"mkey": k, "db_ik14": ";".join(sorted(v)), "tiers": tiers_s})
        old = pd.read_parquet(cp) if cp.exists() else pd.DataFrame(columns=["mkey", "db_ik14", "tiers"])
        allc = pd.concat([old, pd.DataFrame(new_rows)], ignore_index=True)
        allc = allc.drop_duplicates(["mkey", "tiers"], keep="last").sort_values(["tiers", "mkey"])
        cp.parent.mkdir(parents=True, exist_ok=True)
        allc.to_parquet(cp, index=False)
        log.info("truth keys: %d of %d molecules; window %d, same formula %d, same ring count %d "
                 "(%d unique SMILES), same skeleton %d, truth mkey %d (%.1f s)",
                 min(start + step, len(todo_all)), len(todo_all), tot["window"], tot["formula"],
                 tot["ring"], tot["unique_smiles"], tot["skeleton"], tot["hit"],
                 time.perf_counter() - t0)
    out = {}
    for mid, key, grp in zip(truth["molecule_id"], truth["mkey"], truth["ik14_group"]):
        g = frozenset(str(grp).split(";")) if isinstance(grp, str) else frozenset(grp)
        out[str(mid)] = g | cache.get(key, _EMPTY)
    return out


# --------------------------------------------------------------------------------------
# exclusions
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class MoleculeExclusion:
    """What a condition removes for one query molecule."""

    molecule_id: str
    mkey: str
    group: frozenset                 # train ik14 of the molecule (library structures)
    lib_libs: frozenset | None       # libraries whose spectra of ``group`` are removed (None: all)
    db_ik14: frozenset               # DB rows removed (C3: truth keys)
    truth: frozenset                 # truth keys (recall, leak checks)


def condition_exclusions(truth: pd.DataFrame, condition: str,
                         keys: Mapping[str, frozenset]) -> dict[str, MoleculeExclusion]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    out = {}
    for r in truth.itertuples(index=False):
        g = frozenset(str(r.ik14_group).split(";"))
        tk = keys.get(str(r.molecule_id), g)
        out[str(r.molecule_id)] = MoleculeExclusion(
            molecule_id=str(r.molecule_id), mkey=str(r.mkey), group=g,
            lib_libs=frozenset([r.query_lib]) if condition == "C1" else None,
            db_ik14=tk if condition == "C3" else _EMPTY, truth=tk)
    return out


class ValidationHooks:
    """Per-cell state shared by ``ExcludingDB``, ``library_guard`` and cooperating stages
    (``state.extras["validation"]``)."""

    def __init__(self, condition: str, excl: Mapping[str, MoleculeExclusion],
                 spectrum_to_molecule: Mapping[str, str] | None = None,
                 main_tol_ppm: float = 10.0, lib_mkeys: Mapping[str, str] | None = None):
        self.condition = condition
        self.excl = dict(excl)
        self.spec2mol = dict(spectrum_to_molecule or {})
        self.main_tol = float(main_tol_ppm)
        self.lib_mkeys = lib_mkeys
        self.active: tuple[str, ...] = ()
        self.pools: dict[str, dict[str, set]] = defaultdict(lambda: {"main": set(), "any": set()})
        self.pool_summary: dict[str, dict] = {}     # filled by compact()
        self.violations: list[str] = []
        self._lib_cache: dict[tuple[int, str], np.ndarray] = {}   # excluded spectrum indices
        self._lib_index: dict[int, dict] = {}
        self.n_union_fallbacks = 0

    # -- molecules
    def resolve(self, queries: Any = None) -> tuple[str, ...]:
        """Molecule ids a library call is about: from ``queries`` when possible, else the
        active chunk."""
        if isinstance(queries, pd.DataFrame) and "molecule_id" in queries.columns:
            ids = tuple(sorted(set(queries["molecule_id"].astype(str)) & set(self.excl)))
            if ids:
                return ids
        sid = getattr(queries, "spectrum_id", None)
        if sid is not None and self.spec2mol:
            ids = tuple(sorted({self.spec2mol[s] for s in np.asarray(sid, dtype=object).tolist()
                                if s in self.spec2mol}))
            if ids:
                return ids
        if len(self.active) > 1:
            self.n_union_fallbacks += 1
        return self.active

    def db_exclude(self, molecule_id: str) -> frozenset:
        e = self.excl.get(str(molecule_id))
        return e.db_ik14 if e is not None else _EMPTY

    def db_union(self, ids: Iterable[str] | None = None) -> frozenset:
        ids = self.active if ids is None else ids
        out: set = set()
        for m in ids:
            out |= self.db_exclude(m)
        return frozenset(out)

    # -- library
    def _index(self, lib: Any) -> dict:
        k = id(lib)
        ix = self._lib_index.get(k)
        if ix is None:
            ix = {"struct": {str(s): i for i, s in enumerate(lib.struct_ik14)},
                  "lib": np.asarray(lib.spec_lib, dtype=object)}
            self._lib_index[k] = ix
        return ix

    def excluded_spectra(self, lib: Any, molecule_id: str) -> np.ndarray:
        """Library store rows of ``molecule_id`` removed under this condition (sorted int64;
        cached per molecule, small)."""
        key = (id(lib), str(molecule_id))
        idx = self._lib_cache.get(key)
        if idx is None:
            e = self.excl.get(str(molecule_id))
            parts = []
            if e is not None:
                ix = self._index(lib)
                for k in sorted(e.group):
                    s = ix["struct"].get(k)
                    if s is None:
                        continue
                    a, b = (int(v) for v in lib.struct_spec[s])
                    r = np.arange(a, b, dtype=np.int64)
                    if e.lib_libs is not None:
                        r = r[np.isin(ix["lib"][a:b], list(e.lib_libs))]
                    parts.append(r)
            idx = np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)
            self._lib_cache[key] = idx
        return idx

    def library_mask(self, lib: Any, molecule_id: str | Iterable[str]) -> np.ndarray | None:
        """Spectra of ``molecule_id`` (or the union over several ids) removed under this
        condition (True = excluded, one entry per library spectrum), or None if nothing is
        removed. Built on demand from ``excluded_spectra`` (not cached)."""
        ids = [molecule_id] if isinstance(molecule_id, str) else list(molecule_id)
        idx = [self.excluded_spectra(lib, m) for m in ids if str(m) in self.excl]
        idx = [i for i in idx if i.size]
        if not idx:
            return None
        mask = np.zeros(lib.n_spectra, dtype=bool)
        for i in idx:
            mask[i] = True
        return mask

    def check_library(self, lib: Any, truth: pd.DataFrame) -> pd.DataFrame:
        """Assert the C2/C3 rule for every molecule; return per-molecule remaining spectra
        counts (own query library / other libraries) after the mask."""
        rows = []
        mk = np.asarray([self.lib_mkeys.get(str(s)) if self.lib_mkeys else None
                         for s in lib.struct_ik14], dtype=object)
        by_mkey: dict[str, list[int]] = defaultdict(list)
        for i, k in enumerate(mk):
            if isinstance(k, str):
                by_mkey[k].append(i)
        ix = self._index(lib)
        for r in truth.itertuples(index=False):
            e = self.excl[str(r.molecule_id)]
            gone = self.excluded_spectra(lib, e.molecule_id)
            idx = {ix["struct"][k] for k in e.group if k in ix["struct"]}
            alias = set(by_mkey.get(e.mkey, []))
            if self.lib_mkeys is not None and not alias <= idx:
                stray = sorted(str(lib.struct_ik14[i]) for i in alias - idx)
                self.violations.append(f"{e.molecule_id}: library structures {stray[:3]} share the "
                                       "truth mkey but are outside the masked group")
                idx |= alias
            own = other = 0
            for s in sorted(idx):
                a, b = (int(v) for v in lib.struct_spec[s])
                keep = ~np.isin(np.arange(a, b), gone)
                libs = ix["lib"][a:b]
                own += int((keep & (libs == r.query_lib)).sum())
                other += int((keep & (libs != r.query_lib)).sum())
            if self.condition in ("C2", "C3") and own + other:
                self.violations.append(f"{e.molecule_id}: {own + other} library spectra of the "
                                       f"molecule remain under {self.condition}")
            if self.condition == "C1" and own:
                self.violations.append(f"{e.molecule_id}: {own} query-library spectra remain under C1")
            rows.append({"molecule_id": e.molecule_id, "lib_own_left": own, "lib_other_left": other})
        return pd.DataFrame(rows)

    # -- pools
    def record(self, molecule_id: str, ik14: Iterable[str], main: bool) -> None:
        p = self.pools[str(molecule_id)]
        s = {str(x) for x in ik14 if isinstance(x, str)}
        p["any"] |= s
        if main:
            p["main"] |= s

    def compact(self, ids: Iterable[str], keys: Mapping[str, frozenset]) -> None:
        """Replace the candidate sets of finished molecules by what the result table needs
        (sizes and whether a truth key is in the pool). PubChem pools hold about 10,000 ik14
        per molecule, so keeping every set of a 3,000-molecule cell costs gigabytes. A
        molecule recorded again after compaction (not expected: lookups are per chunk) gets
        the union of the counts and hits."""
        for m in ids:
            m = str(m)
            p = self.pools.pop(m, None)
            if p is None:
                continue
            tk = keys.get(m, frozenset())
            new = {"n_main": len(p["main"]), "n_any": len(p["any"]),
                   "hit_main": bool(p["main"] & tk), "hit_any": bool(p["any"] & tk)}
            old = self.pool_summary.get(m)
            if old is not None:
                new = {"n_main": old["n_main"] + new["n_main"], "n_any": old["n_any"] + new["n_any"],
                       "hit_main": old["hit_main"] or new["hit_main"],
                       "hit_any": old["hit_any"] or new["hit_any"]}
            self.pool_summary[m] = new

    def pool_info(self, molecule_id: str, keys: Mapping[str, frozenset]) -> dict:
        """Pool sizes and truth hits of one molecule (compacting it first)."""
        self.compact([molecule_id], keys)
        return self.pool_summary.get(str(molecule_id),
                                     {"n_main": 0, "n_any": 0, "hit_main": False, "hit_any": False})

    def check_rows(self, ids: Sequence[str], ik14: Sequence[Any], where: str) -> None:
        """Every row (molecule id, ik14) must be free of that molecule's DB exclusions; and
        under C3 of its truth keys."""
        for m, k in zip(ids, ik14):
            e = self.excl.get(str(m))
            if e is None:
                continue
            if k in e.db_ik14 or (self.condition == "C3" and k in e.truth):
                self.violations.append(f"{m}: excluded ik14 {k} returned by {where}")


class ExcludingDB:
    """``casmi.db.CandidateDB`` wrapper applying a cell's DB exclusions (module docstring)."""

    def __init__(self, base: Any, hooks: ValidationHooks):
        self._base = base
        self._hooks = hooks
        self._np_cache: tuple[frozenset, pd.DataFrame, np.ndarray] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def lookup_batch(self, masses, tol_ppm=None, tiers=None, min_tol_da=None, keys=None):
        h = self._hooks
        out = self._base.lookup_batch(masses, tol_ppm, tiers, min_tol_da, keys)
        main = tol_ppm is None or abs(float(tol_ppm) - h.main_tol) < 1e-9
        if keys is not None:
            q = out["query"].astype(str).to_numpy(object)
            ik = out["ik14"].astype(object).to_numpy(object)
            if len(out):
                drop = np.fromiter((k in h.db_exclude(m) for m, k in zip(q, ik)), bool, len(out))
                if drop.any():
                    out = out[~drop].reset_index(drop=True)
                    q, ik = q[~drop], ik[~drop]
            h.check_rows(q, ik, "lookup_batch")
            for m in sorted(set(q)):
                h.record(m, ik[q == m], main)
        else:
            u = h.db_union()
            if u and len(out):
                out = out[~out["ik14"].isin(u)].reset_index(drop=True)
            if len(h.active) == 1:
                h.check_rows([h.active[0]] * len(out), out["ik14"].tolist(), "lookup_batch")
                h.record(h.active[0], out["ik14"], main)
        return out

    def lookup(self, mass, tol_ppm=None, tiers=None, min_tol_da=None):
        h = self._hooks
        out = self._base.lookup(mass, tol_ppm, tiers, min_tol_da)
        u = h.db_union()
        if u and len(out):
            out = out[~out["ik14"].isin(u)].reset_index(drop=True)
        if len(h.active) == 1:
            main = tol_ppm is None or abs(float(tol_ppm) - h.main_tol) < 1e-9
            h.check_rows([h.active[0]] * len(out), out["ik14"].tolist(), "lookup")
            h.record(h.active[0], out["ik14"], main)
        return out

    def _np_view(self) -> tuple[pd.DataFrame, np.ndarray]:
        u = self._hooks.db_union()
        base_df = self._base.load_np()
        if not u:
            return base_df, self._base.np_mass
        if self._np_cache is None or self._np_cache[0] != u:
            keep = ~base_df["ik14"].isin(u).to_numpy()
            self._np_cache = (u, base_df[keep].reset_index(drop=True),
                              np.asarray(self._base.np_mass)[keep])
        return self._np_cache[1], self._np_cache[2]

    def load_np(self) -> pd.DataFrame:
        return self._np_view()[0]

    @property
    def np_mass(self) -> np.ndarray:
        return self._np_view()[1]

    def release(self, np_tier: bool = False) -> None:
        self._base.release(np_tier=np_tier)


@contextlib.contextmanager
def library_guard(hooks: ValidationHooks):
    """Patch ``Library`` search methods to apply the cell's exclusions (module docstring)."""
    from . import library as L

    cls = L.Library
    orig = {n: getattr(cls, n) for n in ("search_by_mass", "search_open", "candidates_in_window")}
    sigs = {n: inspect.signature(f) for n, f in orig.items()}

    def _merge(mask, extra):
        if extra is None:
            return mask
        if mask is None:
            return extra
        return np.asarray(mask, bool) | extra

    def search_by_mass(self, *args, **kwargs):
        ba = sigs["search_by_mass"].bind(self, *args, **kwargs)
        ba.apply_defaults()
        ids = hooks.resolve(ba.arguments.get("queries"))
        ba.arguments["exclude_mask"] = _merge(ba.arguments.get("exclude_mask"),
                                              hooks.library_mask(self, ids))
        out = orig["search_by_mass"](*ba.args, **ba.kwargs)
        main = ba.arguments.get("tol_ppm") is None or abs(float(ba.arguments["tol_ppm"]) - hooks.main_tol) < 1e-9
        for m in ids:
            hooks.record(m, out["ik14"].tolist(), main)
            if hooks.condition in ("C2", "C3"):
                e = hooks.excl.get(m)
                bad = [k for k in out["ik14"] if e is not None and k in e.truth]
                if bad:
                    hooks.violations.append(f"{m}: library candidate {bad[0]} returned under "
                                            f"{hooks.condition}")
        return out

    def search_open(self, *args, **kwargs):
        ba = sigs["search_open"].bind(self, *args, **kwargs)
        ba.apply_defaults()
        ids = hooks.active
        if len(ids) > 1:
            hooks.n_union_fallbacks += 1
        ba.arguments["exclude_mask"] = _merge(ba.arguments.get("exclude_mask"),
                                              hooks.library_mask(self, ids))
        out = orig["search_open"](*ba.args, **ba.kwargs)
        if hooks.condition in ("C2", "C3"):
            for m in ids:
                e = hooks.excl.get(m)
                bad = [k for k in out["ik14"] if e is not None and k in e.truth]
                if bad:
                    hooks.violations.append(f"{m}: analog hit {bad[0]} returned under {hooks.condition}")
        return out

    def candidates_in_window(self, *args, **kwargs):
        out = orig["candidates_in_window"](self, *args, **kwargs)
        if hooks.condition not in ("C2", "C3") or not len(out):
            return out
        drop = set()
        for m in hooks.active:
            e = hooks.excl.get(m)
            if e is not None:
                drop |= e.truth
        if not drop:
            return out
        keep = np.array([str(self.struct_ik14[i]) not in drop for i in out], dtype=bool)
        return out[keep]

    cls.search_by_mass = search_by_mass
    cls.search_open = search_open
    cls.candidates_in_window = candidates_in_window
    try:
        yield hooks
    finally:
        for n, f in orig.items():
            setattr(cls, n, f)


# --------------------------------------------------------------------------------------
# predictors
# --------------------------------------------------------------------------------------

Predictor = Callable[..., Mapping[str, Sequence[str]]]


def truth_predictor(truth: pd.DataFrame) -> Predictor:
    """A cheating predictor that returns the truth SMILES (self check: MRR exactly 1.0)."""
    smi = dict(zip(truth["molecule_id"].astype(str), truth["smiles"]))

    def predict(test_df: pd.DataFrame, cfg: Config | None = None, **_: Any) -> dict[str, list[str]]:
        return {m: [smi[m]] for m in sorted(set(test_df["molecule_id"].astype(str)))}

    return predict


def cco_predictor(test_df: pd.DataFrame, cfg: Config | None = None, **_: Any) -> dict[str, list[str]]:
    """Ethanol for every molecule (self check: MRR exactly 0.0)."""
    return {m: ["CCO"] for m in sorted(set(test_df["molecule_id"].astype(str)))}


def pipeline_predictor(test_df: pd.DataFrame, cfg: Config | None = None, **kwargs: Any
                       ) -> dict[str, list[str]]:
    """``casmi.pipeline.predict`` (imported lazily)."""
    from . import pipeline

    return pipeline.predict(test_df, cfg, **kwargs)


# --------------------------------------------------------------------------------------
# running one cell
# --------------------------------------------------------------------------------------

@dataclass
class CellResult:
    set: str
    condition: str
    table: pd.DataFrame
    seconds: float
    n_union_fallbacks: int = 0
    lib_check: pd.DataFrame | None = None


def _has_generator(cfg: Config) -> bool:
    return any("gen" in str(s).lower() for s in getattr(cfg, "pipeline_stages", ()))


def run_cell(predict_fn: Predictor, queries: pd.DataFrame, truth: pd.DataFrame, condition: str,
             cfg: Config, keys: Mapping[str, frozenset], *, db: Any = None, library: Any = None,
             lib_mkeys: Mapping[str, str] | None = None, batch: int | None = None,
             n_jobs: int = 1, strict: bool = True, leak_check: bool = True) -> CellResult:
    """Run ``predict_fn`` on one (set, condition) cell and score it (per-molecule table)."""
    from . import db as cdb
    from .metric import mkeys, reciprocal_ranks

    t0 = time.perf_counter()
    set_name = str(truth["set"].iloc[0]) if len(truth) else "?"
    excl = condition_exclusions(truth, condition, keys)
    spec2mol = dict(zip(queries["spectrum_id"].astype(str), queries["molecule_id"].astype(str)))
    hooks = ValidationHooks(condition, excl, spec2mol, cfg.mass_tol_ppm, lib_mkeys)
    in_cell = pd.Series(True, index=truth["molecule_id"].astype(str))
    lib_check = None
    if library is not None:
        lib_check = hooks.check_library(library, truth)
        if condition == "C1":
            left = dict(zip(lib_check["molecule_id"], lib_check["lib_other_left"]))
            in_cell = pd.Series([bool(e) and left.get(m, 0) > 0 for m, e in
                                 zip(truth["molecule_id"].astype(str), truth["eligible_c1"])],
                                index=in_cell.index)
    elif condition == "C1":
        in_cell = pd.Series(truth["eligible_c1"].astype(bool).to_numpy(), index=in_cell.index)
    cell_ids = [m for m in in_cell.index if in_cell[m]]
    base_db = db if db is not None else cdb.get_db(cfg)
    proxy = ExcludingDB(base_db, hooks)
    extras: dict[str, Any] = {"validation": hooks}
    if library is not None:
        extras["library"] = library
    mass = dict(zip(truth["molecule_id"].astype(str), truth["mono_mass"]))
    order = sorted(cell_ids, key=lambda m: (mass.get(m, 0.0), m))
    size = max(1, int(batch or cfg.batch_molecules))
    qpos = queries.groupby(queries["molecule_id"].astype(str)).indices
    preds: dict[str, list[str]] = {}
    with library_guard(hooks):
        for a in range(0, len(order), size):
            chunk = order[a:a + size]
            rows = np.sort(np.concatenate([qpos[m] for m in chunk if m in qpos]))
            sub = queries.iloc[rows].reset_index(drop=True)
            hooks.active = tuple(sorted(chunk))
            out = predict_fn(sub, cfg, db=proxy, extras=extras)
            for m in chunk:
                g = out.get(m)
                preds[m] = list(g) if g is not None else []
            hooks.compact(chunk, keys)
    hooks.active = ()
    tsmi = dict(zip(truth["molecule_id"].astype(str), truth["smiles"]))
    truth_cell = {m: tsmi[m] for m in cell_ids}
    rr = reciprocal_ranks(preds, truth_cell, k=cfg.top_k, n_jobs=n_jobs, cache=True,
                          timeout_s=cfg.mkey_timeout_s) if truth_cell else {}
    # truth-in-predictions leak check (C3)
    leak = {m: False for m in cell_ids}
    if condition == "C3" and cell_ids:
        guesses = sorted({g for m in cell_ids for g in preds[m] if isinstance(g, str) and g})
        gk = dict(zip(guesses, mkeys(guesses, cfg, n_jobs=n_jobs, cache=True,
                                     timeout_s=cfg.mkey_timeout_s))) if guesses else {}
        tk = dict(zip(truth["molecule_id"].astype(str), truth["mkey"]))
        for m in cell_ids:
            leak[m] = any(gk.get(g) == tk[m] for g in preds[m])
        n_leak = sum(leak.values())
        if n_leak and leak_check and not _has_generator(cfg):
            hooks.violations.append(f"C3: truth among the predictions of {n_leak} molecules "
                                    "without a generator stage")
    rows = []
    tr = truth.set_index(truth["molecule_id"].astype(str))
    for m in truth["molecule_id"].astype(str):
        r = tr.loc[m]
        pool = hooks.pool_info(m, keys)
        v = float(rr.get(m, np.nan)) if in_cell[m] else np.nan
        rank = int(round(1.0 / v)) if isinstance(v, float) and v > 0 else 0
        pc = r["pubmed_count"]
        rows.append({
            "set": set_name, "condition": condition, "molecule_id": m, "mkey": r["mkey"],
            "fold": int(r["fold"]), "in_cell": bool(in_cell[m]), "rr": v, "rank": rank,
            "top1": 0 < rank <= 1, "top5": 0 < rank <= 5, "top25": 0 < rank <= 25,
            "in_pool_main": pool["hit_main"], "in_pool_any": pool["hit_any"],
            "n_pool_main": pool["n_main"], "n_pool_any": pool["n_any"],
            "n_guesses": len(preds.get(m, [])), "truth_in_preds": bool(leak.get(m, False)),
            "pubmed0": bool(pd.isna(pc) or int(pc) == 0), "mass_tercile": int(r["mass_tercile"]),
            "n_query_spectra": int(r["n_query_spectra"]),
        })
    table = pd.DataFrame(rows, columns=list(RESULT_COLUMNS))
    if hooks.violations:
        msg = f"{set_name} {condition}: {len(hooks.violations)} protocol violations: {hooks.violations[:5]}"
        if strict:
            raise AssertionError(msg)
        log.error(msg)
    dt = time.perf_counter() - t0
    log.info("cell %s %s: %d molecules in cell, MRR %.4f, recall %.3f, %.1f s", set_name, condition,
             len(cell_ids), float(np.nanmean(table.loc[table["in_cell"], "rr"])) if cell_ids else float("nan"),
             float(table.loc[table["in_cell"], "in_pool_any"].mean()) if cell_ids else float("nan"), dt)
    return CellResult(set_name, condition, table, dt, hooks.n_union_fallbacks, lib_check)


# --------------------------------------------------------------------------------------
# evaluation, summary, self check
# --------------------------------------------------------------------------------------

def _lib_mkeys(library: Any, train_dir: Path | None) -> dict[str, str] | None:
    if library is None:
        return None
    p = (train_dir or paths.PROCESSED / "train") / "structures.parquet"
    if not p.exists():
        log.warning("no %s: library mkey check limited to the ik14 groups", p)
        return None
    st = pd.read_parquet(p, columns=["ik14", "mkey"]).dropna()
    want = set(map(str, library.struct_ik14))
    st = st[st["ik14"].isin(want)]
    return dict(zip(st["ik14"].astype(str), st["mkey"].astype(str)))


def evaluate(predict_fn: Predictor, cfg: Config, sets: Sequence[str] = SETS,
             conditions: Sequence[str] = CONDITIONS, val_dir: Path | str | None = None,
             library: Any = None, train_dir: Path | str | None = None, batch: int | None = None,
             n_jobs: int = 1, max_molecules: int | None = None, strict: bool = True,
             db: Any = None, leak_check: bool = True) -> tuple[pd.DataFrame, list[CellResult]]:
    """Run every (set, condition) cell; returns the per-molecule table and the cells."""
    val_dir = Path(val_dir) if val_dir is not None else validation_dir()
    train_dir = Path(train_dir) if train_dir is not None else None
    lib_mk = _lib_mkeys(library, train_dir)
    cells = []
    for s in sets:
        q, t = load_queries(s, val_dir)
        if max_molecules:
            keep = sorted(t["molecule_id"].astype(str))[:max_molecules]
            t = t[t["molecule_id"].astype(str).isin(keep)].reset_index(drop=True)
            q = q[q["molecule_id"].isin(keep)].reset_index(drop=True)
        if not len(t):
            log.warning("set %s has no molecules", s)
            continue
        keys = truth_keys(t, cfg, db=db, n_jobs=n_jobs, cache_path=val_dir / "truth_keys.parquet")
        for c in conditions:
            cells.append(run_cell(predict_fn, q, t, c, cfg, keys, db=db, library=library,
                                  lib_mkeys=lib_mk, batch=batch, n_jobs=n_jobs, strict=strict,
                                  leak_check=leak_check))
    table = pd.concat([c.table for c in cells], ignore_index=True) if cells else \
        pd.DataFrame(columns=list(RESULT_COLUMNS))
    return table, cells


def _bootstrap_worst(rr: np.ndarray, mixes: np.ndarray, n_boot: int, ci: float, seed: int
                     ) -> tuple[float, float]:
    """rr: (n_molecules, 3) with NaN outside a cell. Interval of min over mixes of blends."""
    n = rr.shape[0]
    if n == 0 or n_boot <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b0 in range(0, n_boot, 200):
        b1 = min(n_boot, b0 + 200)
        idx = rng.integers(0, n, size=(b1 - b0, n))
        x = rr[idx]                                   # (B, n, 3)
        with np.errstate(invalid="ignore"):
            cnt = np.isfinite(x).sum(axis=1)
            m = np.where(cnt > 0, np.nansum(x, axis=1) / np.maximum(cnt, 1), np.nan)   # (B, 3)
        blends = m @ mixes.T                          # (B, n_mix)
        vals[b0:b1] = blends.min(axis=1)
    lo = (1 - ci) / 2
    return float(np.nanquantile(vals, lo)), float(np.nanquantile(vals, 1 - lo))


def summarize(table: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    """Cells, blends, worst case and bootstrap interval (module docstring)."""
    out: dict[str, Any] = {"cells": {}, "sets": {}}
    mixes = np.asarray(cfg.val_blend_mixes, np.float64)
    for (s, c), g in table.groupby(["set", "condition"], sort=False):
        g = g[g["in_cell"]]
        d = {"n": int(len(g))}
        if len(g):
            d.update(mrr=float(g["rr"].mean()), top1=float(g["top1"].mean()),
                     top5=float(g["top5"].mean()), top25=float(g["top25"].mean()),
                     recall_main=float(g["in_pool_main"].mean()),
                     recall_any=float(g["in_pool_any"].mean()),
                     pool_median=float(g["n_pool_main"].median()),
                     truth_in_preds=int(g["truth_in_preds"].sum()))
            if c == "C2":
                p0 = g[g["pubmed0"]]
                p1 = g[~g["pubmed0"]]
                d.update(n_pubmed0=int(len(p0)), mrr_pubmed0=float(p0["rr"].mean()) if len(p0) else None,
                         mrr_pubmed_pos=float(p1["rr"].mean()) if len(p1) else None)
            d["mrr_by_mass_tercile"] = {int(k): float(v) for k, v in
                                        g.groupby("mass_tercile")["rr"].mean().items()}
        out["cells"][f"{s}|{c}"] = d
    for s, g in table.groupby("set", sort=False):
        ids = sorted(set(g["molecule_id"]))
        pos = {m: i for i, m in enumerate(ids)}
        rr = np.full((len(ids), 3), np.nan)
        have = set()
        for r in g[g["in_cell"]].itertuples(index=False):
            rr[pos[r.molecule_id], CONDITIONS.index(r.condition)] = r.rr
            have.add(r.condition)
        cm = np.array([np.nanmean(rr[:, j]) if np.isfinite(rr[:, j]).any() else np.nan for j in range(3)])
        blends = {",".join(f"{w:g}" for w in mix): float(cm @ mix) for mix in mixes}
        complete = all(c in have for c in CONDITIONS)
        worst = float(min(blends.values())) if complete else float("nan")
        lo, hi = (_bootstrap_worst(rr, mixes, cfg.val_bootstrap_n, cfg.val_bootstrap_ci, cfg.seed)
                  if complete else (float("nan"), float("nan")))
        out["sets"][s] = {"blends": blends if complete else {}, "worst": worst, "worst_ci": [lo, hi],
                          "n_molecules": len(ids)}
    p = out["sets"].get("NP-TIMS", {})
    out["primary"] = p.get("worst", float("nan"))
    out["primary_ci"] = p.get("worst_ci", [float("nan"), float("nan")])
    out["tie_breaker"] = out["sets"].get("NP-OTHER", {}).get("worst", float("nan"))
    return out


def self_check(cfg: Config, sets: Sequence[str] = SETS, conditions: Sequence[str] = CONDITIONS,
               val_dir: Path | str | None = None, library: Any = None,
               train_dir: Path | str | None = None, n_jobs: int = 1,
               max_molecules: int | None = None, db: Any = None) -> dict[str, Any]:
    """Truth predictor scores exactly 1.0 and CCO exactly 0.0 in every cell (asserted),
    through the same harness (exclusions, library checks) as a real run."""
    val_dir = Path(val_dir) if val_dir is not None else validation_dir()
    res = {}
    for s in sets:
        _, t = load_queries(s, val_dir)
        for name, fn, want in (("truth", truth_predictor(t), 1.0), ("cco", cco_predictor, 0.0)):
            tab, _ = evaluate(fn, cfg, [s], conditions, val_dir, library, train_dir, None, n_jobs,
                              max_molecules, True, db, leak_check=name != "truth")
            for c, g in tab.groupby("condition"):
                g = g[g["in_cell"]]
                if not len(g):
                    continue
                v = float(g["rr"].mean())
                assert v == want, f"self check {name} {s} {c}: MRR {v} != {want}"
                res[f"{name}|{s}|{c}"] = v
    log.info("self check passed: truth predictor 1.0 and CCO 0.0 in %d cells", len(res) // 2)
    return res


def _f(x: Any, nd: int = 3) -> str:
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def format_cells(summary: Mapping[str, Any]) -> str:
    """One-line cell summary for docs/EXPERIMENTS.md."""
    parts = []
    for s in SETS:
        cs = [summary["cells"].get(f"{s}|{c}") for c in CONDITIONS]
        if not any(cs):
            continue
        mrr = "/".join(_f(c.get("mrr")) if c else "-" for c in cs)
        rec = "/".join(_f(c.get("recall_any"), 2) if c else "-" for c in cs)
        n = "/".join(str(c.get("n", 0)) if c else "0" for c in cs)
        w = summary["sets"].get(s, {})
        lo, hi = w.get("worst_ci", [float("nan")] * 2)
        parts.append(f"{s} C1/C2/C3 {mrr} (n {n}, recall {rec}), worst {_f(w.get('worst'))} "
                     f"[{_f(lo)}, {_f(hi)}]")
    return "; ".join(parts)


def append_experiments(summary: Mapping[str, Any], tag: str, config: str, path: Path | str | None = None,
                       notes: str = "", git: str | None = None) -> Path:
    """Insert one row at the end of the first table of docs/EXPERIMENTS.md."""
    from .log import git_commit

    p = Path(path) if path is not None else paths.DOCS / "EXPERIMENTS.md"
    cells = format_cells(summary).replace("|", "/")
    row = (f"| {time.strftime('%Y-%m-%d')} | {tag} | {config} | {git or git_commit()} | {cells} | "
           f"{notes.replace('|', '/')} |")
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("| date | tag | config | git commit | cells | notes |\n|---|---|---|---|---|---|\n")
    lines = p.read_text(encoding="utf-8").splitlines()
    hdr = next((i for i, l in enumerate(lines) if l.startswith("| date |")), None)
    if hdr is None:
        lines += ["", "| date | tag | config | git commit | cells | notes |", "|---|---|---|---|---|---|"]
        hdr = len(lines) - 2
    end = hdr + 1
    while end + 1 < len(lines) and lines[end + 1].startswith("|"):
        end += 1
    lines.insert(end + 1, row)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p

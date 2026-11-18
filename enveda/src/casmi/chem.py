"""Chemistry core: standardisation, formula, mass, fingerprints, NP-likeness (WP1).

PLAN.md sections 2.2-2.4 and 4.4. Shared by the DB builds (WP2, WP3), train ingestion
(WP6), the fingerprint model (WP9) and candidate features (WP10), so that targets and
candidates are processed identically.

Conventions:
- Functions that take a molecule accept an RDKit ``Mol`` or a SMILES string and return
  None (or NaN / an all-zero row in batch helpers) on failure; they never raise for bad
  input. RDKit log output is suppressed while they run.
- ``std_mol``: RDKit ``rdMolStandardize.Cleanup`` (RemoveHs, metal disconnection,
  normalisation, reionisation), then ``Uncharger``, then stereo removal. No tautomer
  canonicalisation (too slow for millions of candidates) and no fragment selection.
  It uses cached standardiser objects (twice as fast as calling ``Cleanup``) and
  sanitises the result; on the 4,991 parseable NCI molecules shipped with RDKit it gives
  the same canonical SMILES as ``Cleanup`` + ``Uncharger`` + ``RemoveStereochemistry``
  for all but three Fe/Cu complexes (outside the element filter of PLAN 2.2 anyway).
- Fingerprints are returned PACKED by default: ``np.packbits`` of the 0/1 bit vector,
  dtype uint8, ``bitorder="big"`` (numpy default), so bit ``i`` is
  ``(fp[i // 8] >> (7 - i % 8)) & 1``. Morgan: 4096 bits -> 512 bytes (radius 2,
  chirality off, RDKit default ECFP-like invariants and bond types). MACCS: RDKit's 167
  keys (bit 0 is always 0) -> 21 bytes (the last bit of the last byte is padding).
  ``packed=False`` gives a bool vector. ``unpack_fp`` inverts.
- Batch helpers (``featurize``, ``morgan_fps``, ``maccs_fps``, ``np_scores``,
  ``std_smiles_batch``) keep input order, process in chunks, and use a process pool when
  ``n_jobs > 1`` (forkserver where available, else spawn: scripts need the usual
  ``if __name__ == "__main__":`` guard).
"""

from __future__ import annotations

import importlib.util
import logging
import multiprocessing as mp
import os
from functools import lru_cache
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

from .config import Config

__all__ = [
    "MACCS_BITS",
    "mol_from_smiles",
    "std_mol",
    "std_smiles",
    "canonical_smiles",
    "formula",
    "mono_mass",
    "morgan_fp",
    "maccs_fp",
    "unpack_fp",
    "tanimoto",
    "np_score",
    "FEATURE_FIELDS",
    "featurize",
    "std_smiles_batch",
    "morgan_fps",
    "maccs_fps",
    "np_scores",
    "parallel_map",
    "mp_context",
]

log = logging.getLogger(__name__)

MACCS_BITS = 167


@lru_cache(maxsize=1)
def _default_cfg() -> Config:
    return Config()


# --- parsing and standardisation ------------------------------------------------------

def mol_from_smiles(smiles: str | None) -> Chem.Mol | None:
    """``Chem.MolFromSmiles`` with RDKit logging blocked; None for None, "" or bad SMILES."""
    if not isinstance(smiles, str) or not smiles:
        return None
    with rdBase.BlockLogs():
        try:
            return Chem.MolFromSmiles(smiles)
        except Exception:  # noqa: BLE001 - RDKit can raise on pathological input
            return None


def _as_mol(m: Chem.Mol | str | None) -> Chem.Mol | None:
    if m is None:
        return None
    if isinstance(m, Chem.Mol):
        return m
    return mol_from_smiles(m)


class _Standardizer:
    """Per-process cache of RDKit standardiser objects (they are expensive to build)."""

    def __init__(self) -> None:
        self.disconnector = rdMolStandardize.MetalDisconnector()
        self.normalizer = rdMolStandardize.Normalizer()
        self.reionizer = rdMolStandardize.Reionizer()
        self.uncharger = rdMolStandardize.Uncharger()

    def __call__(self, mol: Chem.Mol) -> Chem.Mol:
        m = Chem.RemoveHs(mol)
        m = self.disconnector.Disconnect(m)
        m = self.normalizer.normalize(m)
        m = self.reionizer.reionize(m)
        m = self.uncharger.uncharge(m)
        Chem.RemoveStereochemistry(m)
        # Normalisation of multi-fragment inputs can leave ring info unset; sanitising
        # rebuilds it (a no-op otherwise) and rejects anything RDKit cannot handle.
        Chem.SanitizeMol(m)
        return m


@lru_cache(maxsize=1)
def _standardizer() -> _Standardizer:
    return _Standardizer()


def std_mol(m: Chem.Mol | str | None) -> Chem.Mol | None:
    """Light standardisation shared by targets and candidates (PLAN 4.4).

    Cleanup (RemoveHs, disconnect metals, normalise, reionise), uncharge, strip stereo.
    No tautomer canonicalisation. Returns a new Mol, or None on failure.
    """
    mol = _as_mol(m)
    if mol is None:
        return None
    with rdBase.BlockLogs():
        try:
            out = _standardizer()(mol)
        except Exception:  # noqa: BLE001
            return None
    return out if out is not None and out.GetNumAtoms() > 0 else None


def std_smiles(m: Chem.Mol | str | None) -> str | None:
    """Canonical SMILES of ``std_mol(m)`` (no stereo), or None."""
    mol = std_mol(m)
    return Chem.MolToSmiles(mol) if mol is not None else None


def canonical_smiles(m: Chem.Mol | str | None, stereo: bool = False) -> str | None:
    """Canonical RDKit SMILES, stereo removed by default, validated by re-parsing (4.8).

    No standardisation: the structure is emitted as given. None if it does not parse or
    the output does not re-parse.
    """
    mol = _as_mol(m)
    if mol is None:
        return None
    with rdBase.BlockLogs():
        try:
            if not stereo:
                mol = Chem.Mol(mol)
                Chem.RemoveStereochemistry(mol)
            out = Chem.MolToSmiles(mol, isomericSmiles=stereo)
        except Exception:  # noqa: BLE001
            return None
    if not out or mol_from_smiles(out) is None:
        return None
    return out


# --- descriptors ---------------------------------------------------------------------

def formula(m: Chem.Mol | str | None) -> str | None:
    """Hill formula (RDKit ``CalcMolFormula``; charged molecules get a +/- suffix)."""
    mol = _as_mol(m)
    if mol is None:
        return None
    return rdMolDescriptors.CalcMolFormula(mol)


def mono_mass(m: Chem.Mol | str | None) -> float:
    """Monoisotopic mass (RDKit ``CalcExactMolWt``; electron mass is subtracted for
    charged molecules). NaN on failure."""
    mol = _as_mol(m)
    if mol is None:
        return float("nan")
    return float(rdMolDescriptors.CalcExactMolWt(mol))


# --- fingerprints --------------------------------------------------------------------

@lru_cache(maxsize=8)
def _morgan_generator(radius: int, n_bits: int):  # type: ignore[no-untyped-def]
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits,
                                                     includeChirality=False)


def morgan_fp(m: Chem.Mol | str | None, cfg: Config | None = None,
              packed: bool = True) -> np.ndarray | None:
    """Morgan fingerprint, ``cfg.fp_morgan_bits`` bits (4096), radius ``cfg.fp_morgan_radius``
    (2), chirality off. Packed uint8 (512,) by default, bool (4096,) with ``packed=False``.

    Computed on the molecule as given: call ``std_mol`` first (the batch helpers do).
    """
    mol = _as_mol(m)
    if mol is None:
        return None
    cfg = cfg or _default_cfg()
    bits = _morgan_generator(cfg.fp_morgan_radius, cfg.fp_morgan_bits).GetFingerprintAsNumPy(mol)
    bits = bits.astype(bool, copy=False)
    return np.packbits(bits) if packed else bits


def maccs_fp(m: Chem.Mol | str | None, packed: bool = True) -> np.ndarray | None:
    """RDKit MACCS keys, 167 bits (bit 0 unused). Packed uint8 (21,) or bool (167,)."""
    mol = _as_mol(m)
    if mol is None:
        return None
    with rdBase.BlockLogs():
        fp = MACCSkeys.GenMACCSKeys(mol)
    bits = np.zeros(MACCS_BITS, dtype=bool)
    on = list(fp.GetOnBits())
    if on:
        bits[on] = True
    return np.packbits(bits) if packed else bits


def unpack_fp(packed: np.ndarray, n_bits: int) -> np.ndarray:
    """Inverse of the packing used here: (..., ceil(n_bits / 8)) uint8 -> (..., n_bits) bool."""
    return np.unpackbits(np.asarray(packed, dtype=np.uint8), axis=-1, count=n_bits).astype(bool)


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def tanimoto(query: np.ndarray, fps: np.ndarray) -> np.ndarray:
    """Tanimoto between one packed fingerprint (n_bytes,) and packed rows (n, n_bytes).

    Returns float32 (n,); 0 where both are empty.
    """
    q = np.asarray(query, dtype=np.uint8)
    f = np.atleast_2d(np.asarray(fps, dtype=np.uint8))
    inter = _POPCOUNT[f & q].sum(axis=1, dtype=np.int64)
    union = _POPCOUNT[f | q].sum(axis=1, dtype=np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    return out.astype(np.float32)


# --- NP-likeness ---------------------------------------------------------------------

@lru_cache(maxsize=1)
def _np_scorer() -> tuple[Any, dict]:
    """(RDKit Contrib npscorer module, model dict), loaded once per process."""
    import gzip
    import pickle

    directory = os.path.join(RDConfig.RDContribDir, "NP_Score")
    script = os.path.join(directory, "npscorer.py")
    model_path = os.path.join(directory, "publicnp.model.gz")
    if not (os.path.isfile(script) and os.path.isfile(model_path)):
        raise FileNotFoundError(f"RDKit Contrib NP_Score not found in {directory}")
    spec = importlib.util.spec_from_file_location("casmi_rdkit_npscorer", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with gzip.open(model_path, "rb") as fh:
        model = pickle.load(fh)  # noqa: S301 - model shipped with RDKit
    return module, model


def np_score(m: Chem.Mol | str | None) -> float:
    """Ertl NP-likeness (RDKit Contrib ``NP_Score``, ``publicnp.model.gz``), about -5..5.

    Exactly ``npscorer.scoreMol(mol, model)``; the model is loaded once per process.
    NaN on failure.
    """
    mol = _as_mol(m)
    if mol is None or mol.GetNumAtoms() == 0:
        return float("nan")
    module, model = _np_scorer()
    try:
        return float(module.scoreMol(mol, model))
    except Exception:  # noqa: BLE001
        return float("nan")


# --- batch helpers -------------------------------------------------------------------

FEATURE_FIELDS: tuple[str, ...] = ("std_smiles", "formula", "mono_mass", "charge",
                                   "n_components", "morgan", "maccs", "np_score")


def _featurize_chunk(args: tuple[list, tuple, bool, int, int]) -> dict[str, Any]:
    smiles, fields, standardize, radius, n_bits = args
    n = len(smiles)
    cfg = _default_cfg()
    if (radius, n_bits) != (cfg.fp_morgan_radius, cfg.fp_morgan_bits):
        cfg = cfg.replace(fp_morgan_radius=radius, fp_morgan_bits=n_bits)
    out: dict[str, Any] = {"ok": np.zeros(n, dtype=bool)}
    if "std_smiles" in fields:
        out["std_smiles"] = [None] * n
    if "formula" in fields:
        out["formula"] = [None] * n
    if "mono_mass" in fields:
        out["mono_mass"] = np.full(n, np.nan, dtype=np.float64)
    if "charge" in fields:
        out["charge"] = np.zeros(n, dtype=np.int16)
    if "n_components" in fields:
        out["n_components"] = np.zeros(n, dtype=np.int16)
    if "morgan" in fields:
        out["morgan"] = np.zeros((n, (n_bits + 7) // 8), dtype=np.uint8)
    if "maccs" in fields:
        out["maccs"] = np.zeros((n, (MACCS_BITS + 7) // 8), dtype=np.uint8)
    if "np_score" in fields:
        out["np_score"] = np.full(n, np.nan, dtype=np.float64)

    for i, s in enumerate(smiles):
        raw = s if isinstance(s, Chem.Mol) else mol_from_smiles(s)
        if raw is None:
            continue
        if "charge" in out:
            out["charge"][i] = Chem.GetFormalCharge(raw)
        if "n_components" in out:
            out["n_components"][i] = len(Chem.GetMolFrags(raw))
        mol = std_mol(raw) if standardize else raw
        if mol is None:
            continue
        out["ok"][i] = True
        with rdBase.BlockLogs():
            if "std_smiles" in out:
                out["std_smiles"][i] = Chem.MolToSmiles(mol)
            if "formula" in out:
                out["formula"][i] = rdMolDescriptors.CalcMolFormula(mol)
            if "mono_mass" in out:
                out["mono_mass"][i] = rdMolDescriptors.CalcExactMolWt(mol)
            if "morgan" in out:
                out["morgan"][i] = morgan_fp(mol, cfg, packed=True)
            if "maccs" in out:
                out["maccs"][i] = maccs_fp(mol, packed=True)
            if "np_score" in out:
                out["np_score"][i] = np_score(mol)
    return out


def mp_context() -> Any:
    """Multiprocessing context: forkserver where available (safe with threads in the
    parent, unlike fork), else spawn. RDKit and casmi are preloaded in the server."""
    methods = mp.get_all_start_methods()
    if "forkserver" in methods:
        ctx = mp.get_context("forkserver")
        try:
            ctx.set_forkserver_preload(["rdkit.Chem", "casmi.chem", "casmi.metric"])
        except Exception:  # noqa: BLE001 - best effort
            pass
        return ctx
    return mp.get_context("spawn")


def parallel_map(func: Callable[[Any], Any], items: Sequence[Any], n_jobs: int = 1) -> list[Any]:
    """``[func(x) for x in items]``, in order, in a process pool when ``n_jobs > 1``.

    ``func`` must be a picklable module-level function. Use for coarse items (chunks).
    """
    items = list(items)
    if n_jobs <= 1 or len(items) <= 1:
        return [func(x) for x in items]
    with mp_context().Pool(min(n_jobs, len(items))) as pool:
        return list(pool.imap(func, items, chunksize=1))


def featurize(smiles: Iterable[str | Chem.Mol], cfg: Config | None = None, n_jobs: int = 1,
              standardize: bool = True, fields: Sequence[str] = ("formula", "mono_mass"),
              chunk_size: int = 2000) -> dict[str, Any]:
    """Compute several properties per SMILES in one parse, keeping input order.

    ``fields`` is a subset of ``FEATURE_FIELDS``. Returns a dict with ``"ok"`` (bool,
    parse and standardisation succeeded) and one entry per field:
    ``std_smiles``/``formula`` (lists, None on failure), ``mono_mass``/``np_score``
    (float64, NaN), ``charge``/``n_components`` (int16, of the molecule as parsed, before
    standardisation, for the filters of PLAN 2.2), ``morgan`` (uint8 (n, 512) packed)
    and ``maccs`` (uint8 (n, 21) packed), zero rows on failure. Everything except
    ``charge`` and ``n_components`` is computed on ``std_mol`` when ``standardize``.
    """
    cfg = cfg or _default_cfg()
    fields = tuple(fields)
    unknown = sorted(set(fields) - set(FEATURE_FIELDS))
    if unknown:
        raise ValueError(f"unknown fields: {unknown}")
    items = list(smiles)
    chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    if any(isinstance(x, Chem.Mol) for x in items) and n_jobs > 1:
        n_jobs = 1  # keep Mol objects in-process (pickling them loses properties)
    args = [(c, fields, standardize, cfg.fp_morgan_radius, cfg.fp_morgan_bits) for c in chunks]
    parts = parallel_map(_featurize_chunk, args, n_jobs) if args else []
    if not parts:
        parts = [_featurize_chunk(([], fields, standardize, cfg.fp_morgan_radius,
                                   cfg.fp_morgan_bits))]
    out: dict[str, Any] = {}
    for key in parts[0]:
        if isinstance(parts[0][key], list):
            out[key] = [v for p in parts for v in p[key]]
        else:
            out[key] = np.concatenate([p[key] for p in parts], axis=0)
    return out


def std_smiles_batch(smiles: Iterable[str], n_jobs: int = 1) -> list[str | None]:
    """``std_smiles`` for many SMILES, in order."""
    return featurize(smiles, n_jobs=n_jobs, fields=("std_smiles",))["std_smiles"]


def morgan_fps(smiles: Iterable[str], cfg: Config | None = None, n_jobs: int = 1,
               standardize: bool = True, packed: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(fingerprints, ok): packed uint8 (n, 512) or bool (n, 4096); zero rows where not ok."""
    cfg = cfg or _default_cfg()
    res = featurize(smiles, cfg, n_jobs=n_jobs, standardize=standardize, fields=("morgan",))
    fps = res["morgan"] if packed else unpack_fp(res["morgan"], cfg.fp_morgan_bits)
    return fps, res["ok"]


def maccs_fps(smiles: Iterable[str], n_jobs: int = 1, standardize: bool = True,
              packed: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(fingerprints, ok): packed uint8 (n, 21) or bool (n, 167); zero rows where not ok."""
    res = featurize(smiles, n_jobs=n_jobs, standardize=standardize, fields=("maccs",))
    fps = res["maccs"] if packed else unpack_fp(res["maccs"], MACCS_BITS)
    return fps, res["ok"]


def np_scores(smiles: Iterable[str], n_jobs: int = 1, standardize: bool = True) -> np.ndarray:
    """NP-likeness per SMILES (float64, NaN on failure)."""
    return featurize(smiles, n_jobs=n_jobs, standardize=standardize, fields=("np_score",))["np_score"]

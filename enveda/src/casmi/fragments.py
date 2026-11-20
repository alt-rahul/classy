"""MetFrag-lite in-silico fragment features and fragment mass calibration (ROADMAP R7).

Why: when the truth is in the list but not first, the winner has the same formula about 98
percent of the time (KAGGLE_INTEL 5). Formula-level evidence (``casmi.formula.explain``) is
identical for isomers by construction; structure-level fragment evidence is not. This module
scores how well the bond-disconnection fragments of each candidate structure explain the
molecule's cleaned query spectra, with the settings that held out of sample in public
(1-2 bond cuts, two-bond cleavages credited 0.6, linear intensity, 0.005 Da, H shifts -2..+3;
3-bond cuts and hand-written bond propensities lost and are not implemented).

Fragments (``fragment_list``; numba kernel, cached per SMILES on disk)
----------------------------------------------------------------------
Heavy-atom graph of the RDKit molecule; each atom carries its monoisotopic mass plus its
implicit and explicit hydrogens. A fragment is a connected piece left after removing bonds:

- level 0 ("intact"): every connected piece of the molecule (normally the molecule itself;
  also the sum of the pieces for salts);
- level 1 ("one"): both sides of one acyclic bond (a bridge of the graph);
- level 2 ("two"): every piece after removing two acyclic bonds (the middle piece is new);
- level 3 ("ring"): ring cleavage as in MetFrag. Removing a single ring bond never splits a
  molecule, so a ring only fragments when two of its bonds go; pairs of ring bonds of the same
  ring system (rings sharing at least two atoms) are tried and kept when the graph splits. A
  ring cleavage is one fragmentation step of two bonds: it counts 2 in the cut count and has
  its own credit ``cfg.frag_w_ring`` (default 0.6, the same as two acyclic cuts).

Pairs that mix an acyclic and a ring bond give nothing new (the ring bond alone does not
disconnect) and are skipped. Molecules with more than ``cfg.frag_max_pairs`` bond pairs get
level 0 and 1 only. Neutral fragment masses are de-duplicated (1e-6 Da); each unique mass keeps
a bit mask of the levels that produce it, so credits and cut counts are applied at match time
and the cache does not depend on scoring settings.

Matching (``prepare_query``, ``score_candidates``)
--------------------------------------------------
Query: the molecule's spectra cleaned with ``spectra.clean`` (precursor window removed),
pooled. Each spectrum counts equally: linear weights are intensity / spectrum total / number of
spectra (sqrt weights likewise), so pooled sums are means over spectra. Peak ``mz`` minus the
per-polarity calibration offset (``cfg.frag_mz_offset_pos_da`` / ``_neg_da``) is matched to
``fragment + dh x m(H) + carrier`` for dh in [``frag_h_shift_min``, ``frag_h_shift_max``] and
the charge carriers its adduct allows (``carriers``: H+ always, plus Na+, K+, NH4+, Cl-,
HCOO- for those adducts), within ``max(frag_tol_da, frag_tol_ppm x mz)``. A peak's credit is
the best credit among its explanations (1 for levels 0 and 1, ``frag_w_two``, ``frag_w_ring``).

Features (``FEATURES``; NaN for rows outside the top K, for unparseable SMILES and for
molecules without peaks):

- ``frag_score``: credited share of the (linear) intensity, mean over spectra (the MetFrag-like
  score of the public ablation);
- ``frag_expl_lin`` / ``frag_expl_sqrt``: explained share of the linear / sqrt intensity
  (any explanation, uncredited);
- ``frag_n_expl`` (explained peaks, pooled count), ``frag_frac_peaks`` (share of peaks);
- ``frag_mean_cuts``: intensity-weighted mean of the fewest bond cuts over explained peaks
  (parsimony; NaN if nothing is explained); ``frag_err_mda``: intensity-weighted mean |error|;
- ``frag_base_expl``: share of spectra whose base peak is explained; ``frag_base_w``: mean
  credit of the base peaks; ``frag_base_best``: 1 if ``frag_base_w`` > 0 and is the best in the
  molecule's scored set;
- ``frag_log_n``: log1p(number of unique fragment masses), the chance-match confounder;
- within the molecule's scored set (top K): ``<f>_z`` (z-score) and ``<f>_rank`` (1 = best,
  ties share the lowest rank) for ``frag_score``, ``frag_expl_lin``, ``frag_expl_sqrt``,
  ``frag_n_expl``; within the formula group of the scored set: ``frag_score_fz``,
  ``frag_score_frank``, ``frag_expl_lin_fz``; ``frag_score_gap`` = score minus the best
  score of the molecule.

Pipeline stage ``fragments`` (plugin ``casmi.fragments``): the top ``cfg.frag_top_k``
candidates per molecule by ``pool[cfg.frag_rank_col]`` (the current score; ``fusion.fuse`` when
the column is absent, so the stage can run before ``score``; never library similarity alone)
get the columns above; ``state.molecules`` gets ``frag_n_spectra``, ``frag_n_peaks``,
``frag_n_scored``. ``cfg.frag_n_jobs`` processes (0 = ``cfg.n_jobs``).

Cache (``FragmentCache``): SQLite file ``fragments.sqlite`` in ``cfg.frag_cache_dir`` (default
``PROCESSED/cache/fragments``; "none" = memory only; a read-only location falls back to memory),
one table per enumeration setting, key = the SMILES string, value = zlib(mass array, mask array).

Calibration helper (``peak_residuals``): signed error of each peak against the nearest
explanation by the true structure's fragments, used by ``scripts/fragment_calibration.py``.

Inference-path rule: numpy, pandas, numba, rdkit and the standard library only.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from numba import njit

from . import paths
from .adducts import ELECTRON_MASS, element_mass, match_test_adduct, parse_adduct
from .config import Config

__all__ = [
    "ENUM_VERSION",
    "LEVEL_NAMES",
    "RAW_FEATURES",
    "NORM_FEATURES",
    "FEATURES",
    "MOLECULE_COLUMNS",
    "FragmentList",
    "QuerySpectra",
    "FragmentCache",
    "carriers",
    "mol_graph",
    "fragment_list",
    "fragment_lists",
    "prepare_query",
    "query_from_frame",
    "score_candidates",
    "normalise",
    "select_top_k",
    "score_pool",
    "stage_fragments",
    "peak_residuals",
    "warmup",
]

log = logging.getLogger(__name__)

ENUM_VERSION = 1
LEVEL_NAMES = ("intact", "one", "two", "ring")
_LEVEL_CUTS = np.array([0, 1, 2, 2], dtype=np.int64)

H_MASS = element_mass("H")
PROTON = H_MASS - ELECTRON_MASS

RAW_FEATURES: tuple[str, ...] = (
    "frag_score", "frag_expl_lin", "frag_expl_sqrt", "frag_n_expl", "frag_frac_peaks",
    "frag_mean_cuts", "frag_err_mda", "frag_base_expl", "frag_base_w",
)
_N_RAW = len(RAW_FEATURES)
_Z_COLS = ("frag_score", "frag_expl_lin", "frag_expl_sqrt", "frag_n_expl")
_FZ_COLS = ("frag_score", "frag_expl_lin")
NORM_FEATURES: tuple[str, ...] = (
    tuple(f"{c}_z" for c in _Z_COLS) + tuple(f"{c}_rank" for c in _Z_COLS)
    + ("frag_score_fz", "frag_score_frank", "frag_expl_lin_fz", "frag_score_gap", "frag_base_best")
)
FEATURES: tuple[str, ...] = RAW_FEATURES + ("frag_log_n",) + NORM_FEATURES
MOLECULE_COLUMNS: tuple[str, ...] = ("frag_n_spectra", "frag_n_peaks", "frag_n_scored")


# ======================================================================================
# Charge carriers
# ======================================================================================

def _carrier_table() -> dict[str, tuple[float, ...]]:
    e, h = ELECTRON_MASS, H_MASS
    na, k, n = element_mass("Na"), element_mass("K"), element_mass("N")
    cl, c, o = element_mass("Cl"), element_mass("C"), element_mass("O")
    return {
        "[M+H]+": (PROTON,),
        "[M-H2O+H]+": (PROTON,),
        "[M-2H2O+H]+": (PROTON,),
        "[M+NH4]+": (PROTON, n + 4 * h - e),
        "[M+Na]+": (PROTON, na - e),
        "[M+K]+": (PROTON, k - e),
        "[M-H]-": (-PROTON,),
        "[M-H2O-H]-": (-PROTON,),
        "[M+CH2O2-H]-": (-PROTON, c + h + 2 * o + e),   # HCOO- kept
        "[M+Cl]-": (-PROTON, cl + e),
    }


_CARRIERS = _carrier_table()


def _polarity(adduct: Any, ionization_mode: Any = None) -> int:
    """+1 / -1 from the adduct's charge, else from the ionization mode (default +1)."""
    a = parse_adduct(adduct, ionization_mode) if isinstance(adduct, str) else None
    if a is not None and a.charge != 0:
        return 1 if a.charge > 0 else -1
    m = str(ionization_mode or "").strip().lower()
    if m.startswith("neg") or m == "-":
        return -1
    if isinstance(adduct, str) and adduct.strip().endswith("-"):
        return -1
    return 1


def carriers(adduct: Any, ionization_mode: Any = None) -> tuple[float, ...]:
    """Ion mass added to a neutral fragment (electron included) for each charge carrier a
    fragment of a precursor with ``adduct`` may keep. Unknown adducts: H+ or -H+ by polarity."""
    name = match_test_adduct(adduct, ionization_mode) if isinstance(adduct, str) else None
    if name is not None:
        return _CARRIERS[name]
    return (PROTON,) if _polarity(adduct, ionization_mode) > 0 else (-PROTON,)


# ======================================================================================
# Molecule graph and enumeration
# ======================================================================================

@dataclass(frozen=True)
class FragmentList:
    """Unique neutral fragment masses (sorted, float64) and their level bit masks (uint8;
    bit b set = produced at level b, ``LEVEL_NAMES``)."""

    masses: np.ndarray
    masks: np.ndarray
    n_atoms: int = 0
    n_bonds: int = 0
    two_cuts: bool = True

    def __len__(self) -> int:
        return int(self.masses.shape[0])


def mol_graph(smiles: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                    np.ndarray, int] | None:
    """``(atom_mass, bond_a, bond_b, bridges, ring_pairs (P, 2), n_components)`` of the
    heavy-atom graph, or None if the SMILES does not parse. ``atom_mass`` includes the
    atom's hydrogens; ``bridges`` are the acyclic bonds; ``ring_pairs`` are the pairs of
    ring bonds inside one ring system (rings sharing >= 2 atoms)."""
    from rdkit import Chem

    from .chem import mol_from_smiles

    mol = smiles if isinstance(smiles, Chem.Mol) else mol_from_smiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    if any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        try:
            mol = Chem.RemoveHs(mol, sanitize=False)
        except Exception:  # noqa: BLE001 - keep explicit H atoms as graph atoms then
            pass
    pt = Chem.GetPeriodicTable()
    n = mol.GetNumAtoms()
    w = np.empty(n, dtype=np.float64)
    for a in mol.GetAtoms():
        z = a.GetAtomicNum()
        iso = a.GetIsotope()
        m = pt.GetMassForIsotope(z, iso) if iso else pt.GetMostCommonIsotopeMass(z)
        w[a.GetIdx()] = m + a.GetTotalNumHs() * H_MASS
    bonds = list(mol.GetBonds())
    ea = np.fromiter((b.GetBeginAtomIdx() for b in bonds), dtype=np.int64, count=len(bonds))
    eb = np.fromiter((b.GetEndAtomIdx() for b in bonds), dtype=np.int64, count=len(bonds))
    ri = mol.GetRingInfo()
    try:
        ri.NumRings()
    except RuntimeError:  # ring info not initialised (unsanitised input)
        Chem.FastFindRings(mol)
        ri = mol.GetRingInfo()
    in_ring = np.fromiter((ri.NumBondRings(b.GetIdx()) > 0 for b in bonds), dtype=bool,
                          count=len(bonds))
    bridges = np.flatnonzero(~in_ring).astype(np.int64)
    pairs = np.zeros((0, 2), dtype=np.int64)
    atom_rings = [set(r) for r in ri.AtomRings()]
    bond_rings = ri.BondRings()
    if atom_rings:
        nr = len(atom_rings)
        parent = list(range(nr))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(nr):
            for j in range(i + 1, nr):
                if len(atom_rings[i] & atom_rings[j]) >= 2:
                    ri_, rj_ = find(i), find(j)
                    if ri_ != rj_:
                        parent[ri_] = rj_
        systems: dict[int, set] = {}
        for i, br in enumerate(bond_rings):
            systems.setdefault(find(i), set()).update(br)
        chunks = []
        for root in sorted(systems):
            bl = np.array(sorted(systems[root]), dtype=np.int64)
            if len(bl) >= 2:
                iu, ju = np.triu_indices(len(bl), k=1)
                chunks.append(np.stack([bl[iu], bl[ju]], axis=1))
        if chunks:
            pairs = np.concatenate(chunks, axis=0)
    ncomp = len(Chem.GetMolFrags(mol, sanitizeFrags=False))
    return w, ea, eb, bridges, pairs, ncomp


@njit(cache=True, nogil=True)
def _uf_find(parent, a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


@njit(cache=True, nogil=True)
def _cut(n, ea, eb, s1, s2, parent):
    """Union-find over all bonds except s1 and s2; returns the number of components."""
    for a in range(n):
        parent[a] = a
    nc = n
    for e in range(ea.shape[0]):
        if e == s1 or e == s2:
            continue
        ra = _uf_find(parent, ea[e])
        rb = _uf_find(parent, eb[e])
        if ra != rb:
            parent[ra] = rb
            nc -= 1
    return nc


@njit(cache=True, nogil=True)
def _emit(n, w, parent, acc, out_m, out_l, cnt, level):
    for a in range(n):
        acc[a] = 0.0
    for a in range(n):
        acc[_uf_find(parent, a)] += w[a]
    for a in range(n):
        if parent[a] == a:
            out_m[cnt] = acc[a]
            out_l[cnt] = level
            cnt += 1
    return cnt


@njit(cache=True, nogil=True)
def _enumerate_kernel(w, ea, eb, bridges, ring_a, ring_b, two_cuts):
    n = w.shape[0]
    nbr = bridges.shape[0]
    nrp = ring_a.shape[0] if two_cuts else 0
    parent = np.empty(n, np.int64)
    acc = np.empty(n, np.float64)
    base = _cut(n, ea, eb, -1, -1, parent)
    npair = nbr * (nbr - 1) // 2 if two_cuts else 0
    cap = (base + 1) + nbr * (base + 1) + npair * (base + 2) + nrp * (base + 1)
    out_m = np.empty(cap, np.float64)
    out_l = np.empty(cap, np.uint8)
    cnt = _emit(n, w, parent, acc, out_m, out_l, 0, 0)
    if base > 1:  # salts / mixtures: the whole input also counts as intact
        tot = 0.0
        for a in range(n):
            tot += w[a]
        out_m[cnt] = tot
        out_l[cnt] = 0
        cnt += 1
    for i in range(nbr):
        _cut(n, ea, eb, bridges[i], -1, parent)
        cnt = _emit(n, w, parent, acc, out_m, out_l, cnt, 1)
    if two_cuts:
        for i in range(nbr):
            for j in range(i + 1, nbr):
                _cut(n, ea, eb, bridges[i], bridges[j], parent)
                cnt = _emit(n, w, parent, acc, out_m, out_l, cnt, 2)
        for p in range(nrp):
            nc = _cut(n, ea, eb, ring_a[p], ring_b[p], parent)
            if nc > base:
                cnt = _emit(n, w, parent, acc, out_m, out_l, cnt, 3)
    return out_m[:cnt], out_l[:cnt]


def _dedupe(m: np.ndarray, lv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    key = np.round(m, 6)
    order = np.argsort(key, kind="mergesort")
    key, m, lv = key[order], m[order], lv[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    bits = (np.uint8(1) << lv.astype(np.uint8)).astype(np.uint8)
    masks = np.bitwise_or.reduceat(bits, starts).astype(np.uint8)
    return np.ascontiguousarray(m[starts], dtype=np.float64), masks


def fragment_list(smiles: Any, cfg: Config | None = None) -> FragmentList | None:
    """Fragments of one structure (no cache). None if the SMILES does not parse."""
    cfg = cfg or Config()
    g = mol_graph(smiles)
    if g is None:
        return None
    w, ea, eb, bridges, pairs, _ = g
    nbr = len(bridges)
    two = int(cfg.frag_max_cuts) >= 2 and nbr * (nbr - 1) // 2 + len(pairs) <= int(cfg.frag_max_pairs)
    m, lv = _enumerate_kernel(w, ea, eb, bridges, np.ascontiguousarray(pairs[:, 0]),
                              np.ascontiguousarray(pairs[:, 1]), two)
    masses, masks = _dedupe(m, lv)
    return FragmentList(masses, masks, int(len(w)), int(len(ea)), bool(two))


# ======================================================================================
# Disk cache
# ======================================================================================

def _cache_table(cfg: Config) -> str:
    return f"frag_v{ENUM_VERSION}_c{int(cfg.frag_max_cuts)}_p{int(cfg.frag_max_pairs)}"


def _encode(fl: FragmentList | None) -> bytes:
    if fl is None:
        return zlib.compress(np.array([-1, 0, 0, 0], dtype=np.int32).tobytes())
    head = np.array([len(fl), fl.n_atoms, fl.n_bonds, int(fl.two_cuts)], dtype=np.int32)
    return zlib.compress(head.tobytes() + fl.masses.astype(np.float64).tobytes()
                         + fl.masks.astype(np.uint8).tobytes(), 1)


def _decode(blob: bytes) -> FragmentList | None:
    raw = zlib.decompress(blob)
    head = np.frombuffer(raw[:16], dtype=np.int32)
    n = int(head[0])
    if n < 0:
        return None
    masses = np.frombuffer(raw[16:16 + 8 * n], dtype=np.float64).copy()
    masks = np.frombuffer(raw[16 + 8 * n:16 + 9 * n], dtype=np.uint8).copy()
    return FragmentList(masses, masks, int(head[1]), int(head[2]), bool(head[3]))


class FragmentCache:
    """Fragment lists per SMILES: a bounded in-memory LRU over an SQLite file (module
    docstring). Safe to use from several processes (one connection per process)."""

    def __init__(self, cfg: Config | None = None, directory: str | Path | None = None,
                 max_memory: int = 20000):
        self.cfg = cfg or Config()
        d = self.cfg.frag_cache_dir if directory is None else str(directory)
        if d.strip().lower() == "none":
            self.path: Path | None = None
        elif d.strip():
            p = Path(d)
            self.path = (p if p.is_absolute() else paths.PROCESSED / p) / "fragments.sqlite"
        else:
            self.path = paths.PROCESSED / "cache" / "fragments" / "fragments.sqlite"
        self.table = _cache_table(self.cfg)
        self.max_memory = int(max_memory)
        self.mem: OrderedDict[str, FragmentList | None] = OrderedDict()
        self._conn: sqlite3.Connection | None = None
        self._pid = -1
        self.hits = self.misses = self.disk_hits = 0

    # -- disk
    def _db(self) -> sqlite3.Connection | None:
        if self.path is None:
            return None
        if self._conn is not None and self._pid == os.getpid():
            return self._conn
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.path), timeout=60.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(f"CREATE TABLE IF NOT EXISTS {self.table} "
                         "(key TEXT PRIMARY KEY, data BLOB NOT NULL)")
            conn.commit()
        except (sqlite3.Error, OSError) as exc:
            log.warning("fragment cache %s unusable (%s): memory only", self.path, exc)
            self.path = None
            return None
        self._conn, self._pid = conn, os.getpid()
        return conn

    def _disk_get(self, keys: list[str]) -> dict[str, FragmentList | None]:
        conn = self._db()
        out: dict[str, FragmentList | None] = {}
        if conn is None or not keys:
            return out
        try:
            for a in range(0, len(keys), 500):
                part = keys[a:a + 500]
                q = f"SELECT key, data FROM {self.table} WHERE key IN ({','.join('?' * len(part))})"
                for k, blob in conn.execute(q, part):
                    out[k] = _decode(blob)
        except sqlite3.Error as exc:
            log.warning("fragment cache read failed (%s)", exc)
        return out

    def _disk_put(self, items: dict[str, FragmentList | None]) -> None:
        conn = self._db()
        if conn is None or not items:
            return
        try:
            conn.executemany(f"INSERT OR IGNORE INTO {self.table} (key, data) VALUES (?, ?)",
                             [(k, _encode(v)) for k, v in sorted(items.items())])
            conn.commit()
        except sqlite3.Error as exc:
            log.warning("fragment cache write failed (%s)", exc)

    # -- memory
    def _remember(self, k: str, v: FragmentList | None) -> None:
        self.mem[k] = v
        self.mem.move_to_end(k)
        while len(self.mem) > self.max_memory:
            self.mem.popitem(last=False)

    def get_many(self, smiles: Sequence[Any]) -> list[FragmentList | None]:
        """Fragment lists for ``smiles`` (in order); computes and stores the missing ones."""
        keys = [s if isinstance(s, str) else "" for s in smiles]
        out: dict[str, FragmentList | None] = {}
        todo = []
        for k in dict.fromkeys(keys):
            if k in self.mem:
                out[k] = self.mem[k]
                self.mem.move_to_end(k)
                self.hits += 1
            else:
                todo.append(k)
        if todo:
            disk = self._disk_get([k for k in todo if k])
            self.disk_hits += len(disk)
            new: dict[str, FragmentList | None] = {}
            for k in todo:
                if k in disk:
                    v = disk[k]
                else:
                    self.misses += 1
                    try:
                        v = fragment_list(k, self.cfg) if k else None
                    except Exception:  # noqa: BLE001 - one bad structure must not stop a molecule
                        log.debug("fragment enumeration failed for %r", k, exc_info=True)
                        v = None
                    if k:
                        new[k] = v
                out[k] = v
                self._remember(k, v)
            self._disk_put(new)
        return [out[k] for k in keys]

    def close(self) -> None:
        if self._conn is not None and self._pid == os.getpid():
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
        self._conn = None


def fragment_lists(smiles: Sequence[Any], cfg: Config | None = None,
                   cache: FragmentCache | None = None) -> list[FragmentList | None]:
    """Fragment lists for many SMILES through ``cache`` (memory-only cache if None)."""
    cache = cache if cache is not None else FragmentCache(cfg, directory="none")
    return cache.get_many(smiles)


# ======================================================================================
# Query spectra
# ======================================================================================

@dataclass
class QuerySpectra:
    """One molecule's cleaned peaks, pooled over spectra, with the matching targets.

    Peaks: ``mz`` (calibrated), ``spec`` (spectrum index), ``w_lin`` / ``w_sqrt`` (sum to 1
    over all peaks; each spectrum 1 / n_spectra), ``base`` (index of each spectrum's base
    peak). Targets (sorted by ``q``): neutral fragment mass ``q`` a peak would need for one
    (carrier, H shift), the peak it belongs to and its tolerance."""

    n_spectra: int
    mz: np.ndarray
    spec: np.ndarray
    w_lin: np.ndarray
    w_sqrt: np.ndarray
    base: np.ndarray
    q: np.ndarray
    q_peak: np.ndarray
    q_tol: np.ndarray
    tol_max: float

    @property
    def n_peaks(self) -> int:
        return int(self.mz.shape[0])


def prepare_query(spectra: Iterable[Sequence[Any]], cfg: Config | None = None,
                  cleaned: bool = False) -> QuerySpectra:
    """``spectra``: ``(mz, intensity, precursor_mz, adduct[, ionization_mode])`` tuples of one
    molecule. Cleaned with ``spectra.clean`` unless ``cleaned``; empty spectra are skipped."""
    from .spectra import clean

    cfg = cfg or Config()
    dh = np.arange(int(cfg.frag_h_shift_min), int(cfg.frag_h_shift_max) + 1, dtype=np.float64)
    mzs, sps, wl, ws, bases, qs, qp = [], [], [], [], [], [], []
    off = 0
    k = 0
    for item in spectra:
        mz, it, prec, add = item[0], item[1], item[2], item[3]
        mode = item[4] if len(item) > 4 else None
        mz = np.asarray(mz, dtype=np.float64).ravel()
        it = np.asarray(it, dtype=np.float64).ravel()
        if not cleaned:
            mz, it, _ = clean(mz, it, float(prec) if prec is not None else float("nan"), cfg)
            mz = mz.astype(np.float64)
            it = it.astype(np.float64)
        ok = np.isfinite(mz) & np.isfinite(it) & (it > 0) & (mz > 0)
        mz, it = mz[ok], it[ok]
        if not len(mz):
            continue
        pol = _polarity(add, mode)
        mz = mz - (cfg.frag_mz_offset_pos_da if pol > 0 else cfg.frag_mz_offset_neg_da)
        car = np.asarray(carriers(add, mode), dtype=np.float64)
        shifts = (car[:, None] + dh[None, :] * H_MASS).ravel()
        n = len(mz)
        mzs.append(mz)
        sps.append(np.full(n, k, dtype=np.int64))
        wl.append(it / it.sum())
        sq = np.sqrt(it)
        ws.append(sq / sq.sum())
        bases.append(off + int(np.argmax(it)))
        qs.append((mz[None, :] - shifts[:, None]).ravel())
        qp.append(np.tile(np.arange(off, off + n, dtype=np.int64), len(shifts)))
        off += n
        k += 1
    if k == 0:
        z = np.zeros(0)
        zi = np.zeros(0, dtype=np.int64)
        return QuerySpectra(0, z, zi, z, z.copy(), zi.copy(), z.copy(), zi.copy(), z.copy(),
                            float(cfg.frag_tol_da))
    mz = np.concatenate(mzs)
    q = np.concatenate(qs)
    q_peak = np.concatenate(qp)
    order = np.argsort(q, kind="mergesort")
    q, q_peak = q[order], q_peak[order]
    tol = np.maximum(float(cfg.frag_tol_da), float(cfg.frag_tol_ppm) * 1e-6 * mz)
    q_tol = tol[q_peak]
    return QuerySpectra(k, mz, np.concatenate(sps), np.concatenate(wl) / k,
                        np.concatenate(ws) / k, np.asarray(bases, dtype=np.int64),
                        np.ascontiguousarray(q), np.ascontiguousarray(q_peak),
                        np.ascontiguousarray(q_tol), float(tol.max()))


def query_from_frame(df: pd.DataFrame, cfg: Config | None = None) -> QuerySpectra:
    """``prepare_query`` on one molecule's rows in the test.parquet schema."""
    modes = df["ionization_mode"] if "ionization_mode" in df.columns else [None] * len(df)
    items = list(zip(df["ms2_mzs"], df["ms2_normalized_intensities"], df["precursor_mz"],
                     df["adduct"], modes))
    return prepare_query(items, cfg)


# ======================================================================================
# Scoring
# ======================================================================================

def _level_tables(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Credit and fewest cuts for each 4-bit level mask."""
    credit = np.array([1.0, 1.0, float(cfg.frag_w_two), float(cfg.frag_w_ring)])
    wtab = np.zeros(16)
    ctab = np.full(16, 99, dtype=np.int64)
    for m in range(1, 16):
        bits = [b for b in range(4) if m >> b & 1]
        wtab[m] = max(credit[b] for b in bits)
        ctab[m] = min(_LEVEL_CUTS[b] for b in bits)
    return wtab, ctab


@njit(cache=True, nogil=True)
def _score_kernel(fm, fmask, foff, valid, q, q_peak, q_tol, tol_max, w_lin, w_sqrt, base,
                  n_spec, wtab, ctab, out):
    n_cand = foff.shape[0] - 1
    npk = w_lin.shape[0]
    nq = q.shape[0]
    best_w = np.zeros(npk)
    best_c = np.zeros(npk, np.int64)
    best_e = np.zeros(npk)
    hit = np.zeros(npk, np.bool_)
    for c in range(n_cand):
        if not valid[c] or npk == 0:
            for f in range(out.shape[1]):
                out[c, f] = np.nan
            continue
        for p in range(npk):
            best_w[p] = 0.0
            best_c[p] = 99
            best_e[p] = 1e9
            hit[p] = False
        a = foff[c]
        b = foff[c + 1]
        lo = a
        for t in range(nq):
            qt = q[t]
            while lo < b and fm[lo] < qt - tol_max:
                lo += 1
            j = lo
            while j < b and fm[j] <= qt + tol_max:
                err = abs(fm[j] - qt)
                if err <= q_tol[t]:
                    p = q_peak[t]
                    m = fmask[j]
                    hit[p] = True
                    if wtab[m] > best_w[p]:
                        best_w[p] = wtab[m]
                    if ctab[m] < best_c[p]:
                        best_c[p] = ctab[m]
                    if err < best_e[p]:
                        best_e[p] = err
                j += 1
        score = 0.0
        el = 0.0
        es = 0.0
        ne = 0
        cuts = 0.0
        errs = 0.0
        for p in range(npk):
            if hit[p]:
                score += w_lin[p] * best_w[p]
                el += w_lin[p]
                es += w_sqrt[p]
                ne += 1
                cuts += w_lin[p] * best_c[p]
                errs += w_lin[p] * best_e[p]
        bexp = 0.0
        bw = 0.0
        for s in range(n_spec):
            p = base[s]
            if hit[p]:
                bexp += 1.0
                bw += best_w[p]
        out[c, 0] = score
        out[c, 1] = el
        out[c, 2] = es
        out[c, 3] = ne
        out[c, 4] = ne / npk
        out[c, 5] = cuts / el if el > 0 else np.nan
        out[c, 6] = 1000.0 * errs / el if el > 0 else np.nan
        out[c, 7] = bexp / n_spec
        out[c, 8] = bw / n_spec


def score_candidates(query: QuerySpectra, frags: Sequence[FragmentList | None],
                     cfg: Config | None = None) -> np.ndarray:
    """Raw features (``RAW_FEATURES`` + ``frag_log_n``) for each fragment list; NaN rows for
    None (unparseable) and for a query without peaks."""
    cfg = cfg or Config()
    n = len(frags)
    valid = np.array([f is not None for f in frags], dtype=np.bool_)
    lens = np.array([len(f) if f is not None else 0 for f in frags], dtype=np.int64)
    foff = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lens, out=foff[1:])
    if lens.sum():
        fm = np.concatenate([f.masses for f in frags if f is not None and len(f)])
        fmask = np.concatenate([f.masks for f in frags if f is not None and len(f)])
    else:
        fm = np.zeros(0)
        fmask = np.zeros(0, dtype=np.uint8)
    wtab, ctab = _level_tables(cfg)
    out = np.empty((n, _N_RAW), dtype=np.float64)
    if n:
        _score_kernel(fm, fmask, foff, valid, query.q, query.q_peak, query.q_tol, query.tol_max,
                      query.w_lin, query.w_sqrt, query.base, query.n_spectra, wtab, ctab, out)
    logn = np.where(valid, np.log1p(lens.astype(np.float64)), np.nan)
    if query.n_peaks == 0:
        logn = np.full(n, np.nan)
    return np.column_stack([out, logn])


# ======================================================================================
# Pool normalisation
# ======================================================================================

def _codes(keys: Sequence[Any]) -> np.ndarray:
    return pd.factorize(pd.Series(list(keys), dtype=object).astype(str), sort=True)[0]


def _group_z_rank(g: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per group ``g``: z-score, rank (1 = largest, ties share the lowest rank) and the group
    max of ``x``; NaN values are ignored and get NaN."""
    n = len(x)
    z = np.full(n, np.nan)
    rank = np.full(n, np.nan)
    gmax = np.full(n, np.nan)
    ok = np.isfinite(x)
    if not ok.any():
        return z, rank, gmax
    gi = g[ok]
    xi = x[ok]
    ng = int(g.max()) + 1
    cnt = np.bincount(gi, minlength=ng).astype(np.float64)
    s1 = np.bincount(gi, weights=xi, minlength=ng)
    mean = s1 / np.maximum(cnt, 1)
    s2 = np.bincount(gi, weights=(xi - mean[gi]) ** 2, minlength=ng)
    sd = np.sqrt(s2 / np.maximum(cnt, 1))
    zz = np.where(sd[gi] > 1e-12, (xi - mean[gi]) / np.where(sd[gi] > 1e-12, sd[gi], 1.0), 0.0)
    mx = np.full(ng, -np.inf)
    np.maximum.at(mx, gi, xi)
    order = np.lexsort((-xi, gi))
    gs, xs = gi[order], xi[order]
    pos = np.arange(len(xs))
    gstart = np.r_[True, gs[1:] != gs[:-1]]
    bstart = gstart | np.r_[True, xs[1:] != xs[:-1]]
    first_g = np.maximum.accumulate(np.where(gstart, pos, 0))
    first_b = np.maximum.accumulate(np.where(bstart, pos, 0))
    rr = np.empty(len(xs))
    rr[order] = first_b - first_g + 1
    idx = np.flatnonzero(ok)
    z[idx] = zz
    rank[idx] = rr
    gmax[idx] = mx[gi]
    return z, rank, gmax


def normalise(raw: pd.DataFrame, molecule_ids: Sequence[Any],
              formulas: Sequence[Any] | None = None) -> pd.DataFrame:
    """Add ``NORM_FEATURES`` to a frame of raw features (one row per scored candidate)."""
    out = raw.copy()
    gm = _codes(molecule_ids)
    if formulas is None:
        gf = gm
    else:
        fs = [f if isinstance(f, str) and f else "?" for f in formulas]
        gf = _codes([f"{m}|{f}" for m, f in zip(pd.Series(list(molecule_ids)).astype(str), fs)])
    for c in _Z_COLS:
        z, r, mx = _group_z_rank(gm, out[c].to_numpy(dtype=np.float64))
        out[f"{c}_z"] = z
        out[f"{c}_rank"] = r
        if c == "frag_score":
            out["frag_score_gap"] = out[c].to_numpy(dtype=np.float64) - mx
    for c in _FZ_COLS:
        z, r, _ = _group_z_rank(gf, out[c].to_numpy(dtype=np.float64))
        out[f"{c}_fz"] = z
        if c == "frag_score":
            out["frag_score_frank"] = r
    bw = out["frag_base_w"].to_numpy(dtype=np.float64)
    _, _, bmx = _group_z_rank(gm, bw)
    best = np.where(np.isfinite(bw), ((bw > 0) & (bw >= bmx - 1e-12)).astype(np.float64), np.nan)
    out["frag_base_best"] = best
    return out[list(FEATURES)]


# ======================================================================================
# Pool scoring and the pipeline stage
# ======================================================================================

def select_top_k(pool: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Boolean mask of the ``cfg.frag_top_k`` best rows per molecule by the current score
    (``pool[cfg.frag_rank_col]``, else ``fusion.fuse``) in ``fusion.sort_pool`` order."""
    from . import fusion

    n = len(pool)
    k = int(cfg.frag_top_k)
    if n == 0 or k <= 0:
        return np.zeros(n, dtype=bool)
    tmp = pool.copy()
    tmp["_row"] = np.arange(n)
    col = cfg.frag_rank_col
    if col and col in tmp.columns and col != "score":
        tmp["score"] = pd.to_numeric(tmp[col], errors="coerce")
    elif "score" not in tmp.columns:
        tmp["score"] = fusion.fuse(tmp, cfg)
    if "molecule_id" not in tmp.columns:
        tmp["molecule_id"] = "0"
    srt = fusion.sort_pool(tmp, cfg=cfg)
    pos = srt.groupby(srt["molecule_id"].astype(str), sort=False).cumcount().to_numpy()
    keep = np.zeros(n, dtype=bool)
    keep[srt["_row"].to_numpy()[pos < k]] = True
    return keep


def _molecule_spectra(test_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if test_df is None or not len(test_df):
        return {}
    mids = test_df["molecule_id"].astype(str)
    return {m: test_df.iloc[idx] for m, idx in mids.groupby(mids, sort=True).indices.items()}


def _score_task(args: tuple) -> list[tuple[str, np.ndarray, int, int]]:
    """Worker: ``[(molecule_id, raw feature matrix, n_spectra, n_peaks)]`` for a chunk."""
    cfg, items, cache_dir = args
    cache = FragmentCache(cfg, directory=cache_dir)
    out = []
    try:
        for mid, spec_df, smiles in items:
            qs = query_from_frame(spec_df, cfg)
            frags = cache.get_many(smiles)
            out.append((mid, score_candidates(qs, frags, cfg), qs.n_spectra, qs.n_peaks))
    finally:
        cache.close()
    return out


def score_pool(pool: pd.DataFrame, test_df: pd.DataFrame, cfg: Config | None = None,
               rows: np.ndarray | None = None, cache: FragmentCache | None = None,
               n_jobs: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fragment features for ``pool`` rows.

    ``rows``: boolean mask of the rows to score (default ``select_top_k``). Returns
    ``(features aligned with pool.index (NaN elsewhere), per-molecule frame)``."""
    cfg = cfg or Config()
    n = len(pool)
    feats = pd.DataFrame(np.nan, index=pool.index, columns=list(FEATURES))
    mol_cols = ["molecule_id", *MOLECULE_COLUMNS]
    if n == 0:
        return feats, pd.DataFrame(columns=mol_cols)
    sel = select_top_k(pool, cfg) if rows is None else np.asarray(rows, dtype=bool)
    mids = pool["molecule_id"].astype(str).to_numpy(dtype=object)
    smiles = pool["smiles"].to_numpy(dtype=object)
    spec = _molecule_spectra(test_df)
    idx_sel = np.flatnonzero(sel)
    by_mol: dict[str, np.ndarray] = {}
    if len(idx_sel):
        s = pd.Series(idx_sel)
        by_mol = {m: v.to_numpy() for m, v in s.groupby(mids[idx_sel], sort=True)}
    items = []
    empty = pd.DataFrame(columns=["ms2_mzs", "ms2_normalized_intensities", "precursor_mz",
                                  "adduct", "ionization_mode"])
    for m in sorted(by_mol):
        items.append((m, spec.get(m, empty), [smiles[i] for i in by_mol[m]]))
    nj = int(cfg.frag_n_jobs if n_jobs is None else n_jobs)
    if nj <= 0:
        nj = int(cfg.n_jobs)
    results: list[tuple[str, np.ndarray, int, int]] = []
    if nj > 1 and len(items) > 1:
        from .chem import parallel_map

        cache_dir = cfg.frag_cache_dir if cache is None or cache.path is not None else "none"
        if cache is not None and cache.path is not None:
            cache_dir = str(cache.path.parent)
        chunks = [items[i::nj] for i in range(nj)]
        chunks = [c for c in chunks if c]
        for part in parallel_map(_score_task, [(cfg, c, cache_dir) for c in chunks], nj):
            results.extend(part)
        results.sort(key=lambda r: r[0])
    else:
        own = cache is None
        cache = cache if cache is not None else FragmentCache(cfg)
        try:
            for mid, spec_df, smi in items:
                qs = query_from_frame(spec_df, cfg)
                results.append((mid, score_candidates(qs, cache.get_many(smi), cfg),
                                qs.n_spectra, qs.n_peaks))
        finally:
            if own:
                cache.close()
    raw = np.full((len(idx_sel), _N_RAW + 1), np.nan)
    order_rows = []
    mol_rows = []
    pos = 0
    for mid, mat, ns, npk in results:
        r = by_mol[mid]
        raw[pos:pos + len(r)] = mat
        order_rows.append(r)
        pos += len(r)
        mol_rows.append({"molecule_id": mid, "frag_n_spectra": ns, "frag_n_peaks": npk,
                         "frag_n_scored": len(r)})
    rows_all = np.concatenate(order_rows) if order_rows else np.zeros(0, dtype=np.int64)
    rawdf = pd.DataFrame(raw[:pos], columns=list(RAW_FEATURES) + ["frag_log_n"])
    form = pool["formula"].to_numpy(dtype=object)[rows_all] if "formula" in pool.columns else None
    normed = normalise(rawdf, mids[rows_all], form)
    feats.iloc[rows_all] = normed.to_numpy(dtype=np.float64)
    return feats, pd.DataFrame(mol_rows, columns=mol_cols)


def stage_fragments(state: Any, cfg: Config) -> Any:
    """Pipeline stage "fragments": ``FEATURES`` on the top ``cfg.frag_top_k`` rows per molecule
    (module docstring); other rows get NaN."""
    pool = state.pool
    if pool is None or not len(pool):
        return state
    t0 = time.perf_counter()
    feats, mol = score_pool(pool, state.test_df, cfg)
    for c in FEATURES:
        pool[c] = feats[c].to_numpy(dtype=np.float64)
    state.pool = pool
    if state.molecules is not None:
        mols = state.molecules
        ids = mols["molecule_id"].astype(str)
        m = mol.set_index("molecule_id") if len(mol) else None
        for c in MOLECULE_COLUMNS:
            v = ids.map(m[c]) if m is not None else pd.Series(np.nan, index=ids.index)
            mols[c] = pd.to_numeric(v, errors="coerce").fillna(0).astype(np.int64).to_numpy()
        state.molecules = mols
    log.debug("fragments: %d rows scored in %.2f s", int(mol["frag_n_scored"].sum()) if len(mol)
              else 0, time.perf_counter() - t0)
    return state


# ======================================================================================
# Calibration helper
# ======================================================================================

@njit(cache=True, nogil=True)
def _residual_kernel(fm, q, q_peak, window, npk, out_err, out_spread):
    best = np.full(npk, 1e9)
    lo_d = np.full(npk, 1e9)
    hi_d = np.full(npk, -1e9)
    for p in range(npk):
        out_err[p] = np.nan
        out_spread[p] = np.nan
    lo = 0
    nf = fm.shape[0]
    for t in range(q.shape[0]):
        qt = q[t]
        while lo < nf and fm[lo] < qt - window:
            lo += 1
        j = lo
        while j < nf and fm[j] <= qt + window:
            d = qt - fm[j]          # observed minus theoretical
            p = q_peak[t]
            if abs(d) < best[p]:
                best[p] = abs(d)
                out_err[p] = d
            if d < lo_d[p]:
                lo_d[p] = d
            if d > hi_d[p]:
                hi_d[p] = d
            j += 1
    for p in range(npk):
        if hi_d[p] >= lo_d[p]:
            out_spread[p] = hi_d[p] - lo_d[p]


def peak_residuals(query: QuerySpectra, frags: FragmentList, window_da: float = 0.02
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Per query peak: signed error (observed m/z minus theoretical ion m/z, Da) of the nearest
    explanation by ``frags`` within ``window_da`` (NaN if none), and the spread (Da) between
    the most negative and most positive explanation in the window (0 = every explanation has
    the same theoretical mass, i.e. the assignment is unambiguous). Uses the query's carriers,
    H shifts and calibration offsets."""
    err = np.empty(query.n_peaks)
    spread = np.empty(query.n_peaks)
    if query.n_peaks:
        _residual_kernel(frags.masses, query.q, query.q_peak, float(window_da), query.n_peaks,
                         err, spread)
    return err, spread


def warmup() -> None:
    """Compile the numba kernels once (they are cached on disk afterwards)."""
    cfg = Config()
    fl = fragment_list("OC(=O)c1ccccc1OC(C)=O", cfg)
    qs = prepare_query([(np.array([121.0284, 163.0390, 93.0335]), np.array([1.0, 0.5, 0.2]),
                         181.0495, "[M+H]+", "positive")], cfg, cleaned=True)
    score_candidates(qs, [fl, None], cfg)
    if fl is not None:
        peak_residuals(qs, fl)


try:  # registration at import, as pipeline plugins expect
    from . import pipeline as _pipeline

    _pipeline.register_stage("fragments", stage_fragments)
except Exception:  # noqa: BLE001 - never fail an import because of the registry
    log.debug("fragments stage not registered", exc_info=True)


"""Contrastive peak-token transformer spectrum encoder (RESEARCH.md roadmap item 1).

One module, two halves, like ``casmi.fpmodel``:

- numpy half (importable without torch): ``EncoderSpec``, the training ``Bundle`` (flat
  arrays on disk, built by ``scripts/build_encoder_bundle.py``), molecule-level query
  construction, evaluation packs and metrics, and a tiny synthetic bundle for tests.
- torch half: every ``torch`` import is inside a function (network, GPU batch sampler,
  losses, training loop, export, ``EncoderModel`` inference, the pipeline stage), so
  ``import casmi.encoder`` works with torch absent or blocked.

Model (``EncoderSpec``; defaults from ``Config.enc_*``)
--------------------------------------------------------
A query is a set of 1 to ``max_members`` spectra of one molecule with the same polarity
and adduct family. Tokens:

- peak tokens: the ``max_tokens`` most intense peaks over the member spectra (each member
  normalised to base peak 1). Features: Fourier features of m/z (sin and cos at
  ``n_freq`` wavelengths log-spaced in [lambda_min, lambda_max] Da), Fourier features of
  the neutral loss to the member's own precursor (zeroed when the loss is negative),
  sqrt(intensity), intensity, and the member's collision energy (CE / 100, missing flag,
  merged-energies flag), through a two-layer MLP to ``d_model``;
- one precursor/metadata token: adduct (ten test adducts + other), polarity, instrument
  family (timstof, qtof, orbitrap, other), a simulated-or-real flag (always 0 here,
  reserved for roadmap item 2), Fourier features and value of the neutral mass M, number
  of members, mean CE and merged fraction.

Pre-norm transformer encoder (``n_layers`` x ``d_model``, ``n_heads``, ``d_ff``, GELU),
key padding mask. Pooling: [meta token output, masked mean of peak outputs] -> Linear ->
GELU -> LayerNorm = h. Heads: ``fp_head`` Linear(h) -> 4096 Morgan radius 2 logits;
``proj`` Linear(h) -> ``emb_dim`` (L2-normalised spectrum embedding). Molecule tower
(``use_tower``): 4096 candidate bits -> Linear -> GELU -> LayerNorm -> Linear ->
``emb_dim`` (L2-normalised). Contrastive score = cosine(spectrum, molecule) x learned
scale (1 / temperature).

Losses (``ARMS``): fingerprint loss "bce" or "focal" (gamma ``enc_focal_gamma``), summed
over bits / ``enc_fp_loss_scale``, mean over the batch; candidate-contrastive InfoNCE
over [true structure, ``enc_n_neg`` same-mass negatives of the structure, the other
queries' true structures in the batch] (same-structure and identical-fingerprint
entries masked).

Candidate scores (``EncoderModel.score``, all ranker features): ``con`` (cosine in the
contrastive space; tower arms only), ``fpcos`` (cosine between the fingerprint
probabilities and the candidate bits), ``bern`` (Bernoulli log-likelihood of the bits,
probabilities clipped to [fp_prob_min, fp_prob_max]).

Molecule level (``build_queries``): per (polarity, adduct family) group of a molecule,
one merged query of all its spectra (chunks of ``max_members`` when larger), plus one
query per spectrum when the group has more than one. ``mode`` "all" averages over both
kinds, "groups" / "singles" over one kind: L2-normalised embeddings are averaged (then
renormalised), fingerprint logits are averaged.

Bundle (``Bundle``, directory of .npy files plus ``info.json`` and ``structures.parquet``)
-----------------------------------------------------------------------------------------
- spectra (cleaned with ``spectra.clean``, the WP5 library store): ``spec_mz`` float32,
  ``spec_int`` float16, ``spec_off`` int64 (n + 1), and per spectrum ``spec_prec``
  float64, ``spec_mass`` float64 (neutral mass from precursor and adduct), ``spec_ce``
  float32 (NaN unknown), ``spec_ne`` int8 (number of energies), ``spec_add`` int8
  (adduct code), ``spec_pol`` int8 (+1 / -1), ``spec_inst`` int8, ``spec_lib`` int8,
  ``spec_sim`` int8 (0 real, 1 simulated), ``spec_struct`` int32.
- groups (one per structure x library x polarity x adduct family x instrument family):
  ``grp_off`` int64, ``grp_mem`` int32 (spectrum rows, deterministic shuffled order),
  ``grp_struct`` int32, ``grp_w`` float64 (sampling weight, sums to 1), ``grp_lib`` int8.
- structures: ``st_fp`` uint8 (n, 512) packed Morgan of ``chem.fp_mol`` of the label
  SMILES, ``st_mass`` float64, ``st_uni`` int32 (the structure's own row in the
  candidate universe), ``structures.parquet`` (ik14, mkey, in_np).
- candidate universe = NP tier rows 0..n_np-1 (``np_fp.npy`` of the NP tier, not copied;
  ``info.json`` records its sha256) followed by ``uni_extra_fp`` uint8 (train structures
  that are not in the NP tier and not holdout). ``neg_off`` int64 / ``neg_idx`` int32:
  per structure, universe rows within ``enc_neg_window_ppm`` of its mass, excluding its
  own ik14 and identical fingerprints, subsampled to ``enc_neg_max`` and topped up with
  the nearest masses to ``enc_neg_min``.
- evaluation packs ``eval_<set>/`` (NP-TIMS, NP-OTHER): cleaned query spectra (``q_*``
  arrays like the spectrum arrays plus ``q_mol``), ``mol.parquet`` (molecule_id, mkey,
  M, n_pool, n_pos) and per molecule the NP-tier pool within ``mass_tol_ppm`` of M
  (``pool_off``, ``pool_idx`` int32 NP rows, ``pool_pos`` bool: the row's ik14 is a
  truth key of the molecule). The class 2 condition: the structure stays in the pool,
  the encoder never trained on it (holdout excluded via ``splits.training_exclusion``).

Metrics (``rank_metrics``): rank of the best-scoring positive among the pool (ties count
against it), MRR@25, top-1/5/25 hit rates, recall (truth in pool); molecules without
the truth in the pool count 0.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import paths
from .adducts import TEST_ADDUCT_NAMES
from .config import Config

__all__ = [
    "BUNDLE_VERSION",
    "ARMS",
    "INSTRUMENT_FAMILIES",
    "ADDUCT_FAMILY_CODE",
    "EncoderSpec",
    "Bundle",
    "EvalPack",
    "QuerySet",
    "build_queries",
    "rank_metrics",
    "lib_weights",
    "synthetic_bundle",
    "unpack_bits",
    "build_net",
    "train",
    "EncoderModel",
    "evaluate_packs",
    "stage_encoder",
    "register",
]

log = logging.getLogger(__name__)

BUNDLE_VERSION = 1
N_BITS = 4096
N_BYTES = N_BITS // 8
INSTRUMENT_FAMILIES: tuple[str, ...] = ("timstof", "qtof", "orbitrap", "other")
N_ADDUCTS = len(TEST_ADDUCT_NAMES) + 1          # ten test adducts + other
ADDUCT_OTHER = len(TEST_ADDUCT_NAMES)
# Adduct family code per adduct code (casmi.library.ADDUCT_FAMILY): [M+H]+ family 0,
# [M+Na]+ 1, [M+K]+ 2, [M-H]- family 3, [M+Cl]- 4, other 5.
_FAMILY_OF = {"[M+H]+": 0, "[M+NH4]+": 0, "[M-H2O+H]+": 0, "[M-2H2O+H]+": 0, "[M+Na]+": 1,
              "[M+K]+": 2, "[M-H]-": 3, "[M-H2O-H]-": 3, "[M+CH2O2-H]-": 3, "[M+Cl]-": 4}
ADDUCT_FAMILY_CODE = np.array([_FAMILY_OF[a] for a in TEST_ADDUCT_NAMES] + [5], dtype=np.int8)

# Training arms: fingerprint loss, fingerprint weight, contrastive weight, molecule tower.
ARMS: dict[str, dict[str, Any]] = {
    "bce": {"fp_loss": "bce", "fp_weight": 1.0, "con_weight": 0.0, "use_tower": False},
    "focal": {"fp_loss": "focal", "fp_weight": 1.0, "con_weight": 0.0, "use_tower": False},
    "focal_con": {"fp_loss": "focal", "fp_weight": 1.0, "con_weight": 1.0, "use_tower": True},
    "con": {"fp_loss": "focal", "fp_weight": 0.0, "con_weight": 1.0, "use_tower": True},
}

SPEC_ARRAYS = ("spec_prec", "spec_mass", "spec_ce", "spec_ne", "spec_add", "spec_pol",
               "spec_inst", "spec_lib", "spec_sim", "spec_struct")
GROUP_ARRAYS = ("grp_off", "grp_mem", "grp_struct", "grp_w", "grp_lib")
STRUCT_ARRAYS = ("st_fp", "st_mass", "st_uni")
QUERY_ARRAYS = ("q_mz", "q_int", "q_off", "q_prec", "q_ce", "q_ne", "q_add", "q_pol",
                "q_inst", "q_mol")


# ======================================================================================
# Specification
# ======================================================================================


@dataclass(frozen=True)
class EncoderSpec:
    """Architecture of one encoder (stored in every checkpoint)."""

    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    n_freq: int = 64
    lambda_min: float = 0.01
    lambda_max: float = 2000.0
    max_tokens: int = 150
    max_members: int = 4
    max_peaks: int = 100
    n_bits: int = N_BITS
    emb_dim: int = 512
    tower_hidden: int = 1024
    use_tower: bool = True
    n_adducts: int = N_ADDUCTS
    n_instruments: int = len(INSTRUMENT_FAMILIES)
    version: int = 1

    @classmethod
    def from_config(cls, cfg: Config, **over: Any) -> "EncoderSpec":
        base = dict(d_model=cfg.enc_d_model, n_layers=cfg.enc_n_layers,
                    n_heads=cfg.enc_n_heads, d_ff=cfg.enc_d_ff, dropout=cfg.enc_dropout,
                    n_freq=cfg.enc_n_freq, lambda_min=cfg.enc_lambda_min,
                    lambda_max=cfg.enc_lambda_max, max_tokens=cfg.enc_max_tokens,
                    max_members=cfg.enc_max_members, max_peaks=cfg.enc_max_peaks,
                    n_bits=cfg.fp_morgan_bits, emb_dim=cfg.enc_emb_dim,
                    tower_hidden=cfg.enc_tower_hidden)
        base.update(over)
        return cls(**base)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EncoderSpec":
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


# ======================================================================================
# Bundle (numpy)
# ======================================================================================


def _load_arrays(d: Path, names: Iterable[str], mmap: bool) -> dict[str, np.ndarray]:
    mode = "r" if mmap else None
    return {n: np.load(d / f"{n}.npy", mmap_mode=mode) for n in names}


def _save_arrays(d: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, int]:
    d.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for n, a in arrays.items():
        p = d / f"{n}.npy"
        np.save(p, np.ascontiguousarray(a))
        sizes[n] = int(p.stat().st_size)
    return sizes


@dataclass
class EvalPack:
    """Class 2 evaluation molecules of one holdout set (see module docstring)."""

    name: str
    q: dict[str, np.ndarray]          # QUERY_ARRAYS
    mol: pd.DataFrame                 # molecule_id, mkey, M, n_pool, n_pos (row = q_mol code)
    pool_off: np.ndarray              # int64 (n_mol + 1)
    pool_idx: np.ndarray              # int32 NP-tier rows
    pool_pos: np.ndarray              # bool

    def save(self, d: Path) -> dict[str, int]:
        d.mkdir(parents=True, exist_ok=True)
        sizes = _save_arrays(d, {**self.q, "pool_off": self.pool_off, "pool_idx": self.pool_idx,
                                 "pool_pos": self.pool_pos})
        self.mol.to_parquet(d / "mol.parquet", index=False)
        return sizes

    @classmethod
    def load(cls, d: Path, name: str | None = None) -> "EvalPack":
        q = _load_arrays(d, QUERY_ARRAYS, mmap=False)
        rest = _load_arrays(d, ("pool_off", "pool_idx", "pool_pos"), mmap=False)
        mol = pd.read_parquet(d / "mol.parquet")
        return cls(name or d.name, q, mol, rest["pool_off"], rest["pool_idx"], rest["pool_pos"])

    def subset(self, n: int, seed: int) -> "EvalPack":
        """At most ``n`` molecules (deterministic random choice), same layout."""
        m = len(self.mol)
        if n <= 0 or n >= m:
            return self
        keep = np.sort(np.random.default_rng(seed).choice(m, n, replace=False))
        remap = np.full(m, -1, np.int64)
        remap[keep] = np.arange(n)
        qm = self.q["q_mol"].astype(np.int64)
        sel = np.flatnonzero(remap[qm] >= 0)
        off = self.q["q_off"]
        lens = (off[1:] - off[:-1])[sel]
        new_off = np.zeros(len(sel) + 1, np.int64)
        np.cumsum(lens, out=new_off[1:])
        peaks = np.concatenate([np.arange(off[i], off[i + 1]) for i in sel]) if len(sel) else \
            np.zeros(0, np.int64)
        q = {k: (v[sel] if k not in ("q_mz", "q_int", "q_off") else v) for k, v in self.q.items()}
        q["q_mz"] = self.q["q_mz"][peaks]
        q["q_int"] = self.q["q_int"][peaks]
        q["q_off"] = new_off
        q["q_mol"] = remap[qm[sel]].astype(np.int32)
        po = self.pool_off
        plen = (po[1:] - po[:-1])[keep]
        new_po = np.zeros(n + 1, np.int64)
        np.cumsum(plen, out=new_po[1:])
        rows = np.concatenate([np.arange(po[i], po[i + 1]) for i in keep]) if n else \
            np.zeros(0, np.int64)
        return EvalPack(self.name, q, self.mol.iloc[keep].reset_index(drop=True), new_po,
                        self.pool_idx[rows], self.pool_pos[rows])


@dataclass
class Bundle:
    """Encoder training bundle (see module docstring). Arrays are memory-mapped."""

    root: Path
    info: dict[str, Any]
    spec_mz: np.ndarray
    spec_int: np.ndarray
    spec_off: np.ndarray
    s: dict[str, np.ndarray]          # SPEC_ARRAYS
    g: dict[str, np.ndarray]          # GROUP_ARRAYS
    st: dict[str, np.ndarray]         # STRUCT_ARRAYS
    structures: pd.DataFrame
    uni_extra_fp: np.ndarray
    neg_off: np.ndarray
    neg_idx: np.ndarray

    @classmethod
    def load(cls, root: Path | str, mmap: bool = True) -> "Bundle":
        root = Path(root)
        info = json.loads((root / "info.json").read_text())
        if int(info.get("version", 0)) != BUNDLE_VERSION:
            raise ValueError(f"{root}: bundle version {info.get('version')} != {BUNDLE_VERSION}")
        a = _load_arrays(root, ("spec_mz", "spec_int", "spec_off", "uni_extra_fp", "neg_off",
                                "neg_idx"), mmap)
        return cls(root, info, a["spec_mz"], a["spec_int"], a["spec_off"],
                   _load_arrays(root, SPEC_ARRAYS, mmap), _load_arrays(root, GROUP_ARRAYS, mmap),
                   _load_arrays(root, STRUCT_ARRAYS, mmap),
                   pd.read_parquet(root / "structures.parquet"), a["uni_extra_fp"],
                   a["neg_off"], a["neg_idx"])

    @property
    def n_np(self) -> int:
        return int(self.info["n_np"])

    def eval_packs(self) -> dict[str, EvalPack]:
        out = {}
        for name in self.info.get("eval_sets", []):
            d = self.root / f"eval_{_slug(name)}"
            if d.is_dir():
                out[name] = EvalPack.load(d, name)
        return out


def _slug(name: str) -> str:
    return name.lower().replace("-", "_")


def save_bundle(root: Path | str, *, spec_mz: np.ndarray, spec_int: np.ndarray,
                spec_off: np.ndarray, s: Mapping[str, np.ndarray], g: Mapping[str, np.ndarray],
                st: Mapping[str, np.ndarray], structures: pd.DataFrame,
                uni_extra_fp: np.ndarray, neg_off: np.ndarray, neg_idx: np.ndarray,
                info: Mapping[str, Any], packs: Mapping[str, EvalPack] | None = None
                ) -> dict[str, Any]:
    """Write a bundle directory; returns ``info`` with array sizes added."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    missing = [n for n in SPEC_ARRAYS if n not in s] + [n for n in GROUP_ARRAYS if n not in g] \
        + [n for n in STRUCT_ARRAYS if n not in st]
    if missing:
        raise ValueError(f"save_bundle: missing arrays {missing}")
    sizes = _save_arrays(root, {"spec_mz": spec_mz.astype(np.float32),
                                "spec_int": spec_int.astype(np.float16),
                                "spec_off": spec_off.astype(np.int64),
                                **{n: s[n] for n in SPEC_ARRAYS},
                                **{n: g[n] for n in GROUP_ARRAYS},
                                **{n: st[n] for n in STRUCT_ARRAYS},
                                "uni_extra_fp": uni_extra_fp.astype(np.uint8),
                                "neg_off": neg_off.astype(np.int64),
                                "neg_idx": neg_idx.astype(np.int32)})
    structures.to_parquet(root / "structures.parquet", index=False)
    out = dict(info)
    out["version"] = BUNDLE_VERSION
    out["eval_sets"] = sorted(packs) if packs else []
    for name, pack in (packs or {}).items():
        sz = pack.save(root / f"eval_{_slug(name)}")
        sizes.update({f"eval_{_slug(name)}/{k}": v for k, v in sz.items()})
    out["array_bytes"] = sizes
    out["total_bytes"] = int(sum(sizes.values()))
    (root / "info.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    return out


def lib_weights(grp_lib_names: np.ndarray, grp_struct: np.ndarray,
                shares: Sequence[Sequence[Any]]) -> np.ndarray:
    """Group sampling weights (sum 1): each library gets its share (``shares`` pairs; the
    remainder goes to unnamed libraries in proportion to n_groups^0.75); inside a library
    a group weighs 1 / sqrt(groups of its structure in that library)."""
    names = np.asarray(grp_lib_names, dtype=object)
    st = np.asarray(grp_struct, dtype=np.int64)
    n = len(names)
    w = np.zeros(n, np.float64)
    if n == 0:
        return w
    df = pd.DataFrame({"lib": names, "st": st})
    per = df.groupby(["lib", "st"], sort=False)["st"].transform("size").to_numpy(np.float64)
    base = 1.0 / np.sqrt(per)
    libs = sorted(set(names.tolist()))
    counts = {l: int((names == l).sum()) for l in libs}
    named = {str(k): float(v) for k, v in shares}
    present_named = {l: v for l, v in named.items() if l in counts}
    rest = [l for l in libs if l not in present_named]
    total_named = sum(present_named.values())
    left = max(0.0, 1.0 - total_named)
    lib_share = dict(present_named)
    if rest:
        z = sum(counts[l] ** 0.75 for l in rest)
        for l in rest:
            lib_share[l] = left * counts[l] ** 0.75 / z
    tot = sum(lib_share.values())
    for l in libs:
        m = names == l
        bw = base[m]
        w[m] = lib_share[l] / tot * bw / bw.sum()
    return w / w.sum()


# ======================================================================================
# Queries (numpy, shared by evaluation and inference)
# ======================================================================================


@dataclass
class QuerySet:
    """Queries of many molecules over one cleaned spectrum store.

    ``members`` (n_q, max_members) spectrum rows (-1 = none), ``mol`` (n_q,) molecule code,
    ``kind`` (n_q,) 0 = group query, 1 = single-spectrum query.
    """

    members: np.ndarray
    mol: np.ndarray
    kind: np.ndarray

    def __len__(self) -> int:
        return len(self.mol)


def build_queries(mol_code: np.ndarray, polarity: np.ndarray, adduct_code: np.ndarray,
                  n_peaks: np.ndarray, max_members: int = 4,
                  collision_energy: np.ndarray | None = None) -> QuerySet:
    """Molecule-level queries (module docstring). Spectra with no peaks are skipped unless
    a molecule has nothing else (then it gets one metadata-only query). Deterministic:
    spectra are ordered by (molecule, polarity, family, CE, row)."""
    mol_code = np.asarray(mol_code, np.int64)
    n = len(mol_code)
    pol = np.asarray(polarity, np.int64)
    fam = ADDUCT_FAMILY_CODE[np.clip(np.asarray(adduct_code, np.int64), 0, ADDUCT_OTHER)]
    ce = (np.nan_to_num(np.asarray(collision_energy, np.float64), nan=1e9)
          if collision_energy is not None else np.zeros(n))
    npk = np.asarray(n_peaks, np.int64)
    rows = np.arange(n)
    order = np.lexsort((rows, ce, fam, -pol, mol_code))
    members: list[list[int]] = []
    mols: list[int] = []
    kinds: list[int] = []
    i = 0
    while i < n:
        m = mol_code[order[i]]
        j = i
        while j < n and mol_code[order[j]] == m:
            j += 1
        idx = order[i:j]
        usable = idx[npk[idx] > 0]
        if len(usable) == 0:
            members.append([int(idx[0])])
            mols.append(int(m))
            kinds.append(0)
            i = j
            continue
        keys = list(zip(pol[usable].tolist(), fam[usable].tolist()))
        a = 0
        while a < len(usable):
            b = a
            while b < len(usable) and keys[b] == keys[a]:
                b += 1
            grp = usable[a:b]
            n_chunks = int(math.ceil(len(grp) / max_members))
            for c in range(n_chunks):
                chunk = grp[c::n_chunks]          # round robin spreads energies over chunks
                members.append([int(x) for x in chunk])
                mols.append(int(m))
                kinds.append(0)
            if len(grp) > 1:
                for x in grp:
                    members.append([int(x)])
                    mols.append(int(m))
                    kinds.append(1)
            a = b
        i = j
    out = np.full((len(members), max_members), -1, np.int64)
    for r, mem in enumerate(members):
        out[r, :len(mem)] = mem[:max_members]
    return QuerySet(out, np.asarray(mols, np.int64), np.asarray(kinds, np.int8))


def query_mask(qs: QuerySet, mode: str) -> np.ndarray:
    """Which queries take part in the molecule aggregate for ``mode``; a molecule without
    a query of the requested kind falls back to all of its queries."""
    if mode == "all":
        return np.ones(len(qs), bool)
    want = 0 if mode == "groups" else 1 if mode == "singles" else None
    if want is None:
        raise ValueError(f"unknown query mode {mode!r}")
    sel = qs.kind == want
    has = np.zeros(int(qs.mol.max()) + 1 if len(qs) else 0, bool)
    has[qs.mol[sel]] = True
    return sel | ~has[qs.mol]


def rank_metrics(scores: np.ndarray, pos: np.ndarray, off: np.ndarray, k: int = 25
                 ) -> dict[str, Any]:
    """Per molecule rank of the best positive (ties against it) over CSR pools.

    Returns n, mrr, top1, top5, top25, recall and the ``rank`` array (0 = truth absent).
    """
    n = len(off) - 1
    rank = np.zeros(n, np.int64)
    for i in range(n):
        a, b = int(off[i]), int(off[i + 1])
        p = pos[a:b]
        if not p.any():
            continue
        s = scores[a:b]
        best = s[p].max()
        rank[i] = 1 + int(((s >= best) & ~p).sum())
    ok = rank > 0
    rr = np.where(ok & (rank <= k), 1.0 / np.maximum(rank, 1), 0.0)
    return {"n": int(n), "mrr": float(rr.mean()) if n else float("nan"),
            "top1": float((ok & (rank <= 1)).mean()) if n else float("nan"),
            "top5": float((ok & (rank <= 5)).mean()) if n else float("nan"),
            "top25": float((ok & (rank <= 25)).mean()) if n else float("nan"),
            "recall": float(ok.mean()) if n else float("nan"), "rank": rank}


def unpack_bits(packed: np.ndarray, n_bits: int = N_BITS) -> np.ndarray:
    """Packed big-endian uint8 (..., n_bits / 8) -> (..., n_bits) uint8 0/1."""
    return np.unpackbits(np.asarray(packed, np.uint8), axis=-1, count=n_bits)


def fp_hash(packed: np.ndarray) -> np.ndarray:
    """64-bit hash per packed fingerprint row (identity test)."""
    a = np.ascontiguousarray(packed, np.uint8)
    if a.shape[1] % 8:
        a = np.pad(a, ((0, 0), (0, 8 - a.shape[1] % 8)))
    v = a.view(np.uint64)
    h = np.full(len(a), np.uint64(1469598103934665603), np.uint64)
    with np.errstate(over="ignore"):
        for j in range(v.shape[1]):
            h = (h ^ v[:, j]) * np.uint64(1099511628211)
            h ^= h >> np.uint64(29)
    return h


# ======================================================================================
# Synthetic bundle (tests)
# ======================================================================================


def synthetic_bundle(root: Path | str, n_struct: int = 48, n_np: int = 200,
                     spectra_per_struct: int = 4, seed: int = 0, n_eval: int = 8
                     ) -> tuple[Path, Path]:
    """Tiny learnable bundle: fragment m/z are a deterministic function of fingerprint
    bits, so a model can learn spectrum -> structure. Returns (bundle dir, np_fp path)."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    n_bits = N_BITS
    # Universe: n_np "NP tier" structures; training structures are the first n_struct of
    # them plus a few train-only ones; evaluation structures are other NP rows.
    def rand_fp(n: int) -> np.ndarray:
        bits = np.zeros((n, n_bits), np.uint8)
        for i in range(n):
            bits[i, rng.choice(64, 12, replace=False)] = 1          # few informative bits
            bits[i, 64 + rng.choice(n_bits - 64, 20, replace=False)] = 1
        return np.packbits(bits, axis=1)

    np_fp = rand_fp(n_np)
    np_mass = np.sort(rng.uniform(200, 400, n_np))
    # Mass clusters so that same-mass negatives exist: snap to 20 values.
    np_mass = np.round(np_mass / 10) * 10 + rng.uniform(0, 1e-3, n_np)
    order = np.argsort(np_mass, kind="mergesort")
    np_mass, np_fp = np_mass[order], np_fp[order]
    n_extra = 8
    extra_fp = rand_fp(n_extra)
    extra_mass = rng.uniform(200, 400, n_extra)
    eval_rows = np.arange(n_np - n_eval, n_np)
    train_np_rows = np.arange(n_struct - n_extra)
    st_fp = np.concatenate([np_fp[train_np_rows], extra_fp])
    st_mass = np.concatenate([np_mass[train_np_rows], extra_mass])
    st_uni = np.concatenate([train_np_rows, n_np + np.arange(n_extra)]).astype(np.int32)

    def spectrum(fp_packed: np.ndarray, mass: float, ce: float) -> tuple[np.ndarray, np.ndarray]:
        bits = np.flatnonzero(unpack_bits(fp_packed[None])[0][:64])
        mz = 50.0 + bits * 2.5 + rng.normal(0, 0.002, len(bits))
        it = rng.uniform(0.2, 1.0, len(bits)) * (1.0 + ce / 100.0)
        noise = rng.uniform(30, mass, 3)
        mz = np.concatenate([mz, noise])
        it = np.concatenate([it, rng.uniform(0.01, 0.05, 3)])
        o = np.argsort(mz)
        return mz[o].astype(np.float32), (it[o] / it.max()).astype(np.float32)

    mzs, its, prec, mass_s, ce_s, struct_s = [], [], [], [], [], []
    for s_i in range(n_struct):
        for k in range(spectra_per_struct):
            ce = float(rng.choice([20.0, 40.0, 60.0]))
            m, x = spectrum(st_fp[s_i], st_mass[s_i], ce)
            mzs.append(m)
            its.append(x)
            prec.append(st_mass[s_i] + 1.007276)
            mass_s.append(st_mass[s_i])
            ce_s.append(ce)
            struct_s.append(s_i)
    n_spec = len(mzs)
    off = np.zeros(n_spec + 1, np.int64)
    np.cumsum([len(m) for m in mzs], out=off[1:])
    s = {"spec_prec": np.asarray(prec, np.float64), "spec_mass": np.asarray(mass_s, np.float64),
         "spec_ce": np.asarray(ce_s, np.float32), "spec_ne": np.ones(n_spec, np.int8),
         "spec_add": np.zeros(n_spec, np.int8), "spec_pol": np.ones(n_spec, np.int8),
         "spec_inst": np.zeros(n_spec, np.int8), "spec_lib": np.zeros(n_spec, np.int8),
         "spec_sim": np.zeros(n_spec, np.int8), "spec_struct": np.asarray(struct_s, np.int32)}
    grp_struct = np.arange(n_struct, dtype=np.int32)
    grp_off = np.arange(0, n_spec + 1, spectra_per_struct, dtype=np.int64)
    g = {"grp_off": grp_off, "grp_mem": np.arange(n_spec, dtype=np.int32),
         "grp_struct": grp_struct, "grp_w": np.full(n_struct, 1.0 / n_struct),
         "grp_lib": np.zeros(n_struct, np.int8)}
    # Negatives: universe rows within 2 Da (synthetic window), excluding self.
    uni_mass = np.concatenate([np_mass, extra_mass])
    neg_lists = []
    for i in range(n_struct):
        c = np.flatnonzero(np.abs(uni_mass - st_mass[i]) < 2.0)
        c = c[c != st_uni[i]]
        if len(c) < 4:
            c = np.argsort(np.abs(uni_mass - st_mass[i]), kind="mergesort")[1:5]
        neg_lists.append(c.astype(np.int32))
    neg_off = np.zeros(n_struct + 1, np.int64)
    np.cumsum([len(c) for c in neg_lists], out=neg_off[1:])
    neg_idx = np.concatenate(neg_lists)
    structures = pd.DataFrame({"ik14": [f"S{i:05d}" for i in range(n_struct)],
                               "mkey": [f"S{i:05d}" for i in range(n_struct)],
                               "in_np": st_uni < n_np})
    # Evaluation pack: each eval structure has 2 spectra; pool = NP rows within 2 Da.
    q_mz, q_int, q_prec, q_ce, q_mol, q_n = [], [], [], [], [], []
    mol_rows = []
    pools, poss = [], []
    for j, r in enumerate(eval_rows):
        for k in range(2):
            ce = [20.0, 40.0][k]
            m, x = spectrum(np_fp[r], np_mass[r], ce)
            q_mz.append(m)
            q_int.append(x)
            q_prec.append(np_mass[r] + 1.007276)
            q_ce.append(ce)
            q_mol.append(j)
        pool = np.flatnonzero(np.abs(np_mass - np_mass[r]) < 2.0)
        pools.append(pool.astype(np.int32))
        poss.append(pool == r)
        mol_rows.append({"molecule_id": f"E{j:03d}", "mkey": f"E{j:03d}", "M": float(np_mass[r]),
                         "n_pool": len(pool), "n_pos": 1})
    nq = len(q_mz)
    q_off = np.zeros(nq + 1, np.int64)
    np.cumsum([len(m) for m in q_mz], out=q_off[1:])
    q = {"q_mz": np.concatenate(q_mz).astype(np.float32),
         "q_int": np.concatenate(q_int).astype(np.float16), "q_off": q_off,
         "q_prec": np.asarray(q_prec, np.float64), "q_ce": np.asarray(q_ce, np.float32),
         "q_ne": np.ones(nq, np.int8), "q_add": np.zeros(nq, np.int8),
         "q_pol": np.ones(nq, np.int8), "q_inst": np.zeros(nq, np.int8),
         "q_mol": np.asarray(q_mol, np.int32)}
    pool_off = np.zeros(len(pools) + 1, np.int64)
    np.cumsum([len(p) for p in pools], out=pool_off[1:])
    pack = EvalPack("SYN", q, pd.DataFrame(mol_rows), pool_off,
                    np.concatenate(pools).astype(np.int32), np.concatenate(poss))
    np_fp_path = root / "np_fp.npy"
    np.save(np_fp_path, np_fp)
    save_bundle(root, spec_mz=np.concatenate(mzs), spec_int=np.concatenate(its), spec_off=off,
                s=s, g=g, st={"st_fp": st_fp, "st_mass": st_mass, "st_uni": st_uni},
                structures=structures, uni_extra_fp=extra_fp, neg_off=neg_off, neg_idx=neg_idx,
                info={"n_np": n_np, "synthetic": True, "libs": ["syn"]}, packs={"SYN": pack})
    return root, np_fp_path


# ======================================================================================
# torch half: network
# ======================================================================================


def _torch():
    import torch  # noqa: PLC0415 - lazy by design (module docstring)

    return torch


def build_net(spec: EncoderSpec):
    """The encoder ``nn.Module`` for ``spec`` (torch imported here)."""
    torch = _torch()
    nn = torch.nn
    F = torch.nn.functional

    class Fourier(nn.Module):
        def __init__(self, n_freq: int, lmin: float, lmax: float):
            super().__init__()
            lam = np.exp(np.linspace(math.log(lmin), math.log(lmax), n_freq))
            self.register_buffer("w", torch.tensor(2 * math.pi / lam, dtype=torch.float32),
                                 persistent=False)

        def forward(self, x):
            a = x.float().unsqueeze(-1) * self.w
            return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)

    class Encoder(nn.Module):
        def __init__(self, sp: EncoderSpec):
            super().__init__()
            self.spec = sp
            d = sp.d_model
            nf2 = 2 * sp.n_freq
            self.fourier = Fourier(sp.n_freq, sp.lambda_min, sp.lambda_max)
            self.peak_mlp = nn.Sequential(nn.Linear(2 * nf2 + 5, d), nn.GELU(), nn.Linear(d, d))
            self.add_emb = nn.Embedding(sp.n_adducts, d)
            self.pol_emb = nn.Embedding(3, d)
            self.inst_emb = nn.Embedding(sp.n_instruments, d)
            self.sim_emb = nn.Embedding(2, d)
            self.meta_mlp = nn.Sequential(nn.Linear(nf2 + 4, d), nn.GELU(), nn.Linear(d, d))
            self.type_emb = nn.Parameter(torch.zeros(2, d))
            layer = nn.TransformerEncoderLayer(d, sp.n_heads, sp.d_ff, sp.dropout,
                                               activation="gelu", batch_first=True,
                                               norm_first=True)
            self.blocks = nn.TransformerEncoder(layer, sp.n_layers, enable_nested_tensor=False)
            self.out_norm = nn.LayerNorm(d)
            self.pool = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.LayerNorm(d))
            self.fp_head = nn.Linear(d, sp.n_bits)
            self.proj = nn.Linear(d, sp.emb_dim)
            if sp.use_tower:
                self.tower = nn.Sequential(nn.Linear(sp.n_bits, sp.tower_hidden), nn.GELU(),
                                           nn.LayerNorm(sp.tower_hidden),
                                           nn.Linear(sp.tower_hidden, sp.emb_dim))
            else:
                self.tower = None
            self.log_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.05)))
            with torch.no_grad():
                self.fp_head.bias.fill_(-4.0)       # about 1.5 percent of bits are on

        def encode(self, tok):
            """tok: dict from ``assemble_tokens`` -> (h (B, d), valid tokens)."""
            sp = self.spec
            mz, it, prec, ce, ne = tok["mz"], tok["int"], tok["prec"], tok["ce"], tok["ne"]
            valid = tok["valid"]
            nl = prec - mz
            fm = self.fourier(mz)
            fl = self.fourier(nl) * (nl > 0).unsqueeze(-1).float()
            ce_ok = torch.isfinite(ce)
            extra = torch.stack([torch.sqrt(it.clamp(min=0)), it,
                                 torch.where(ce_ok, ce, torch.zeros_like(ce)) / 100.0,
                                 (~ce_ok).float(), (ne > 1).float()], dim=-1)
            x = torch.cat([fm, fl, extra], dim=-1)
            x = x * valid.unsqueeze(-1).float()
            peaks = self.peak_mlp(x) + self.type_emb[1]
            fmass = self.fourier(tok["mass"])
            mextra = torch.stack([tok["mass"] / 1000.0, tok["n_members"] / sp.max_members,
                                  tok["mean_ce"] / 100.0, tok["merged_frac"]], dim=-1)
            meta = (self.meta_mlp(torch.cat([fmass, mextra], dim=-1)) + self.add_emb(tok["adduct"])
                    + self.pol_emb(tok["polarity"]) + self.inst_emb(tok["inst"])
                    + self.sim_emb(tok["sim"]) + self.type_emb[0])
            seq = torch.cat([meta.unsqueeze(1), peaks], dim=1)
            pad = torch.cat([torch.zeros_like(valid[:, :1]), ~valid], dim=1)
            out = self.blocks(seq, src_key_padding_mask=pad)
            out = self.out_norm(out)
            vf = valid.unsqueeze(-1).to(out.dtype)
            mean = (out[:, 1:] * vf).sum(1) / vf.sum(1).clamp(min=1.0)
            return self.pool(torch.cat([out[:, 0], mean], dim=-1))

        def forward(self, tok):
            h = self.encode(tok)
            z = F.normalize(self.proj(h).float(), dim=-1)
            return self.fp_head(h).float(), z

        def embed_molecules(self, bits):
            """bits: (N, n_bits) float 0/1 -> (N, emb_dim) L2-normalised."""
            if self.tower is None:
                raise RuntimeError("this encoder has no molecule tower")
            return F.normalize(self.tower(bits).float(), dim=-1)

        def scale(self):
            return self.log_scale.exp().clamp(max=100.0)

    return Encoder(spec)


# ======================================================================================
# torch half: tokens (shared by training and inference)
# ======================================================================================


def _gather_members(t, mz_flat, int_flat, off, members, max_peaks):
    """Padded member peaks (B, M, P) from flat arrays; members (B, M) rows, -1 = none."""
    torch = t
    B, M = members.shape
    mvalid = members >= 0
    rows = members.clamp(min=0)
    start = off[rows]
    length = (off[rows + 1] - start).clamp(max=max_peaks)
    ar = torch.arange(max_peaks, device=members.device)
    idx = start.unsqueeze(-1) + ar
    valid = (ar < length.unsqueeze(-1)) & mvalid.unsqueeze(-1)
    idx = torch.where(valid, idx, torch.zeros_like(idx))
    mz = mz_flat[idx].float() * valid
    it = int_flat[idx].float() * valid
    return mz, it, valid


def assemble_tokens(mz, it, valid, member_meta: Mapping[str, Any], query_meta: Mapping[str, Any],
                    max_tokens: int, aug: Mapping[str, Any] | None = None, gen=None):
    """Token tensors from padded member peaks.

    mz, it, valid: (B, M, P); member_meta: prec, ce, ne (B, M) float, mvalid (B, M) bool;
    query_meta: adduct, polarity (0 neg, 1 pos, 2 unknown), inst, sim (B,) long, mass (B,)
    float. ``aug`` (training): mz_ppm, peak_drop, int_sigma, noise_prob, noise_max,
    noise_int_min, noise_int_max. Returns the dict ``build_net(...).encode`` reads.
    """
    torch = _torch()
    B, M, P = mz.shape
    dev = mz.device
    prec = member_meta["prec"].float()
    mvalid = member_meta["mvalid"]
    if aug:
        if aug.get("mz_ppm", 0) > 0:
            mz = mz * (1.0 + torch.randn(mz.shape, device=dev, generator=gen) * aug["mz_ppm"] * 1e-6)
        if aug.get("peak_drop", 0) > 0:
            top = it >= it.amax(dim=-1, keepdim=True)
            drop = torch.rand(it.shape, device=dev, generator=gen) < aug["peak_drop"]
            valid = valid & ~(drop & ~top)
        if aug.get("int_sigma", 0) > 0:
            it = it * torch.exp(torch.randn(it.shape, device=dev, generator=gen) * aug["int_sigma"])
        nz = int(aug.get("noise_max", 0))
        if nz > 0:
            use = torch.rand((B, M, 1), device=dev, generator=gen) < aug["noise_prob"]
            k = torch.randint(0, nz + 1, (B, M, 1), device=dev, generator=gen)
            nvalid = (torch.arange(nz, device=dev) < k) & use & mvalid.unsqueeze(-1)
            hi = (prec - 0.5).clamp(min=21.0).unsqueeze(-1)
            nmz = 20.0 + torch.rand((B, M, nz), device=dev, generator=gen) * (hi - 20.0)
            lo_i, hi_i = float(aug["noise_int_min"]), float(aug["noise_int_max"])
            nit = lo_i + torch.rand((B, M, nz), device=dev, generator=gen) * (hi_i - lo_i)
            mz = torch.cat([mz, nmz * nvalid], dim=-1)
            valid = torch.cat([valid, nvalid], dim=-1)
            base = torch.where(valid[..., :P], it, torch.zeros_like(it)).amax(-1, keepdim=True)
            it = torch.cat([it, nit * base * nvalid], dim=-1)
    it = torch.where(valid, it, torch.zeros_like(it))
    it = it / it.amax(dim=-1, keepdim=True).clamp(min=1e-12)
    Pn = mz.shape[-1]
    flat_score = torch.where(valid, it, torch.full_like(it, -1.0)).reshape(B, M * Pn)
    T = min(max_tokens, M * Pn)
    sc, pos = torch.topk(flat_score, T, dim=1)
    tvalid = sc > 0
    member = torch.div(pos, Pn, rounding_mode="floor")
    tmz = mz.reshape(B, M * Pn).gather(1, pos) * tvalid
    tit = it.reshape(B, M * Pn).gather(1, pos) * tvalid
    ce = member_meta["ce"].float()
    ne = member_meta["ne"].float()
    tprec = prec.gather(1, member)
    tce = ce.gather(1, member)
    tne = ne.gather(1, member)
    nm = mvalid.float().sum(1)
    ce_ok = torch.isfinite(ce) & mvalid
    mean_ce = torch.where(ce_ok, ce, torch.zeros_like(ce)).sum(1) / ce_ok.float().sum(1).clamp(min=1)
    merged = ((ne > 1) & mvalid).float().sum(1) / nm.clamp(min=1)
    return {"mz": tmz, "int": tit, "prec": tprec, "ce": tce, "ne": tne, "valid": tvalid,
            "mass": query_meta["mass"].float(), "n_members": nm, "mean_ce": mean_ce,
            "merged_frac": merged, "adduct": query_meta["adduct"], "polarity": query_meta["polarity"],
            "inst": query_meta["inst"], "sim": query_meta["sim"]}


def _unpack_t(t, packed):
    """(N, n_bytes) uint8 tensor -> (N, 8 * n_bytes) float 0/1 (big-endian bits)."""
    torch = t
    shifts = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    bits = (packed.unsqueeze(-1) >> shifts) & 1
    return bits.reshape(*packed.shape[:-1], packed.shape[-1] * 8)


def _pol_code(t, pol):
    """+1 -> 1, -1 -> 0, anything else -> 2."""
    torch = t
    return torch.where(pol > 0, torch.ones_like(pol), torch.where(pol < 0, torch.zeros_like(pol),
                                                                  torch.full_like(pol, 2)))


# ======================================================================================
# torch half: device data and batch sampler (training)
# ======================================================================================


class DeviceData:
    """Bundle arrays as tensors on one device plus a GPU-side batch sampler."""

    def __init__(self, bundle: Bundle, np_fp: np.ndarray, device: str, cfg: Config):
        torch = _torch()
        self.torch = torch
        self.dev = torch.device(device)
        self.cfg = cfg
        T = lambda a, dt=None: torch.as_tensor(np.ascontiguousarray(a), device=self.dev,  # noqa: E731
                                               dtype=dt)
        self.mz = T(bundle.spec_mz, torch.float32)
        self.it = T(bundle.spec_int, torch.float16)
        self.off = T(bundle.spec_off, torch.int64)
        s = bundle.s
        self.prec = T(s["spec_prec"], torch.float32)
        self.mass = T(s["spec_mass"], torch.float32)
        self.ce = T(s["spec_ce"], torch.float32)
        self.ne = T(s["spec_ne"].astype(np.float32), torch.float32)
        self.add = T(s["spec_add"].astype(np.int64))
        self.pol = T(s["spec_pol"].astype(np.int64))
        self.inst = T(s["spec_inst"].astype(np.int64))
        self.sim = T(s["spec_sim"].astype(np.int64))
        g = bundle.g
        self.goff = T(g["grp_off"], torch.int64)
        self.gmem = T(g["grp_mem"].astype(np.int64))
        self.gst = T(g["grp_struct"].astype(np.int64))
        w = np.asarray(g["grp_w"], np.float64)
        cw = np.cumsum(w / w.sum())
        cw[-1] = 1.0
        self.gcum = T(cw, torch.float64)
        self.st_fp = T(bundle.st["st_fp"], torch.uint8)
        self.st_uni = T(bundle.st["st_uni"].astype(np.int64))
        self.noff = T(bundle.neg_off, torch.int64)
        self.nidx = T(bundle.neg_idx.astype(np.int64))
        uni = np.concatenate([np.asarray(np_fp, np.uint8), np.asarray(bundle.uni_extra_fp, np.uint8)])
        self.uni_fp = T(uni, torch.uint8)
        self.n_np = bundle.n_np
        kp = np.asarray(cfg.enc_k_probs, np.float64)
        self.k_probs = T(kp / kp.sum(), torch.float32)

    def sample(self, B: int, spec: EncoderSpec, n_neg: int, gen, aug: Mapping[str, Any] | None):
        torch = self.torch
        dev = self.dev
        M = spec.max_members
        u = torch.rand(B, device=dev, generator=gen, dtype=torch.float64)
        g = torch.searchsorted(self.gcum, u).clamp(max=len(self.gst) - 1)
        g0 = self.goff[g]
        ng = self.goff[g + 1] - g0
        k = torch.multinomial(self.k_probs, B, replacement=True, generator=gen) + 1
        k = torch.minimum(k, ng)
        start = torch.floor(torch.rand(B, device=dev, generator=gen) * ng).long()
        stride = 1 + torch.floor(torch.rand(B, device=dev, generator=gen) *
                                 (ng - 1).clamp(min=0)).long()
        j = torch.arange(M, device=dev)
        posn = (start.unsqueeze(1) + j * stride.unsqueeze(1)) % ng.unsqueeze(1).clamp(min=1)
        mvalid = j < k.unsqueeze(1)
        dup = (posn.unsqueeze(2) == posn.unsqueeze(1)) & (j.unsqueeze(1) < j.unsqueeze(0)).unsqueeze(0)
        mvalid = mvalid & ~(dup & mvalid.unsqueeze(1)).any(1)
        rows = self.gmem[(g0.unsqueeze(1) + posn).clamp(max=len(self.gmem) - 1)]
        members = torch.where(mvalid, rows, torch.full_like(rows, -1))
        mz, it, valid = _gather_members(torch, self.mz, self.it, self.off, members, spec.max_peaks)
        r0 = members[:, 0].clamp(min=0)
        rr = members.clamp(min=0)
        mm = {"prec": self.prec[rr], "ce": torch.where(mvalid, self.ce[rr], torch.full_like(self.ce[rr], float("nan"))),
              "ne": self.ne[rr] * mvalid, "mvalid": mvalid}
        qm = {"adduct": self.add[r0].clamp(0, spec.n_adducts - 1),
              "polarity": _pol_code(torch, self.pol[r0]), "inst": self.inst[r0],
              "sim": self.sim[r0], "mass": self.mass[r0]}
        tok = assemble_tokens(mz, it, valid, mm, qm, spec.max_tokens, aug, gen)
        st = self.gst[g]
        pos_fp = self.st_fp[st]
        y = _unpack_t(torch, pos_fp).float()
        out = {"tok": tok, "y": y, "st": st, "pos_packed": pos_fp}
        if n_neg > 0:
            n0 = self.noff[st]
            nn_ = self.noff[st + 1] - n0
            r = torch.floor(torch.rand((B, n_neg), device=dev, generator=gen) *
                            nn_.unsqueeze(1).clamp(min=1)).long()
            nrow = self.nidx[(n0.unsqueeze(1) + r).clamp(max=max(len(self.nidx) - 1, 0))]
            out["neg_valid"] = (nn_ > 0).unsqueeze(1).expand(B, n_neg)
            out["neg_bits"] = _unpack_t(torch, self.uni_fp[nrow])
        return out


# ======================================================================================
# torch half: losses
# ======================================================================================


def fp_loss(logits, y, kind: str, gamma: float, scale: float):
    torch = _torch()
    F = torch.nn.functional
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    if kind == "focal":
        p = torch.sigmoid(logits)
        pt = p * y + (1 - p) * (1 - y)
        bce = bce * (1 - pt).pow(gamma)
    elif kind != "bce":
        raise ValueError(f"unknown fingerprint loss {kind!r}")
    return bce.sum(1).mean() / scale


def contrastive_loss(net, z, batch):
    """InfoNCE over [positive, sampled negatives, in-batch positives]."""
    torch = _torch()
    F = torch.nn.functional
    B = z.shape[0]
    zp = net.embed_molecules(batch["y"])
    s_pos = (z * zp).sum(-1, keepdim=True)
    parts = [s_pos]
    if "neg_bits" in batch:
        K = batch["neg_bits"].shape[1]
        zn = net.embed_molecules(batch["neg_bits"].reshape(B * K, -1).float()).reshape(B, K, -1)
        s_neg = torch.einsum("bd,bkd->bk", z, zn)
        s_neg = s_neg.masked_fill(~batch["neg_valid"], float("-inf"))
        parts.append(s_neg)
    s_in = z @ zp.T
    same = batch["st"].unsqueeze(0) == batch["st"].unsqueeze(1)
    pp = batch["pos_packed"]
    ident = (pp.unsqueeze(0) == pp.unsqueeze(1)).all(-1) if B <= 512 else same
    s_in = s_in.masked_fill(same | ident, float("-inf"))
    parts.append(s_in)
    logits = torch.cat(parts, dim=1) * net.scale()
    target = torch.zeros(B, dtype=torch.long, device=z.device)
    loss = F.cross_entropy(logits, target)
    with torch.no_grad():
        acc = (logits.argmax(1) == 0).float().mean()
    return loss, acc


# ======================================================================================
# torch half: inference model
# ======================================================================================


class EncoderModel:
    """A trained encoder for inference (torch, CPU or GPU)."""

    def __init__(self, net, spec: EncoderSpec, info: Mapping[str, Any] | None = None,
                 device: str = "cpu"):
        self.net = net
        self.spec = spec
        self.info = dict(info or {})
        self.device = device

    # ---- io
    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> "EncoderModel":
        torch = _torch()
        ck = torch.load(Path(path), map_location=device, weights_only=True)
        spec = EncoderSpec.from_dict(json.loads(ck["spec_json"]))
        net = build_net(spec).to(device)
        net.load_state_dict(ck["state_dict"])
        net.eval()
        info = json.loads(ck.get("info_json", "{}"))
        return cls(net, spec, info, device)

    def save(self, path: Path | str) -> Path:
        return export_model(self.net, self.spec, path, self.info)

    # ---- encoding
    def encode_store(self, mz_flat: np.ndarray, int_flat: np.ndarray, off: np.ndarray,
                     meta: Mapping[str, np.ndarray], qs: QuerySet, mol_mass: np.ndarray,
                     batch: int = 256) -> tuple[np.ndarray, np.ndarray]:
        """(z (n_q, emb) float32 normalised, logits (n_q, n_bits) float32) per query.

        ``meta``: per spectrum prec, ce, ne, add, pol, inst (numpy); ``mol_mass`` per
        molecule code (neutral mass M of the query's molecule).
        """
        torch = _torch()
        dev = torch.device(self.device)
        sp = self.spec
        mzt = torch.as_tensor(np.ascontiguousarray(mz_flat, np.float32), device=dev)
        itt = torch.as_tensor(np.ascontiguousarray(int_flat, np.float32), device=dev)
        offt = torch.as_tensor(np.ascontiguousarray(off, np.int64), device=dev)
        prec = torch.as_tensor(np.asarray(meta["prec"], np.float32), device=dev)
        ce = torch.as_tensor(np.asarray(meta["ce"], np.float32), device=dev)
        ne = torch.as_tensor(np.asarray(meta["ne"], np.float32), device=dev)
        add = torch.as_tensor(np.clip(np.asarray(meta["add"], np.int64), 0, sp.n_adducts - 1), device=dev)
        pol = torch.as_tensor(np.asarray(meta["pol"], np.int64), device=dev)
        inst = torch.as_tensor(np.clip(np.asarray(meta["inst"], np.int64), 0, sp.n_instruments - 1),
                               device=dev)
        mass = torch.as_tensor(np.asarray(mol_mass, np.float32), device=dev)
        zs, ls = [], []
        self.net.eval()
        with torch.no_grad():
            for a in range(0, len(qs), batch):
                mem = torch.as_tensor(qs.members[a:a + batch, :sp.max_members], device=dev)
                if mem.shape[1] < sp.max_members:
                    mem = torch.cat([mem, torch.full((mem.shape[0], sp.max_members - mem.shape[1]),
                                                     -1, dtype=mem.dtype, device=dev)], 1)
                mz, it, valid = _gather_members(torch, mzt, itt, offt, mem, sp.max_peaks)
                mv = mem >= 0
                rr = mem.clamp(min=0)
                r0 = rr[:, 0]
                mm = {"prec": prec[rr], "ce": torch.where(mv, ce[rr], torch.full_like(ce[rr], float("nan"))),
                      "ne": ne[rr] * mv, "mvalid": mv}
                mol = torch.as_tensor(qs.mol[a:a + batch], device=dev)
                qm = {"adduct": add[r0], "polarity": _pol_code(torch, pol[r0]), "inst": inst[r0],
                      "sim": torch.zeros_like(r0), "mass": mass[mol]}
                tok = assemble_tokens(mz, it, valid, mm, qm, sp.max_tokens)
                logits, z = self.net(tok)
                zs.append(z.float().cpu().numpy())
                ls.append(logits.float().cpu().numpy())
        if not zs:
            return np.zeros((0, sp.emb_dim), np.float32), np.zeros((0, sp.n_bits), np.float32)
        return np.concatenate(zs), np.concatenate(ls)

    @staticmethod
    def aggregate(z: np.ndarray, logits: np.ndarray, qs: QuerySet, n_mol: int, mode: str = "all"
                  ) -> tuple[np.ndarray, np.ndarray]:
        """Molecule-level (embedding normalised mean, mean logits) for ``mode``."""
        sel = query_mask(qs, mode)
        mol = qs.mol[sel]
        cnt = np.bincount(mol, minlength=n_mol).astype(np.float64)[:, None]
        ze = np.zeros((n_mol, z.shape[1]), np.float64)
        np.add.at(ze, mol, z[sel])
        le = np.zeros((n_mol, logits.shape[1]), np.float64)
        np.add.at(le, mol, logits[sel])
        ze /= np.maximum(cnt, 1)
        le /= np.maximum(cnt, 1)
        ze /= np.maximum(np.linalg.norm(ze, axis=1, keepdims=True), 1e-12)
        return ze.astype(np.float32), le.astype(np.float32)

    def encode_frame(self, df: pd.DataFrame, cfg: Config | None = None,
                     molecules: pd.DataFrame | None = None, mode: str | None = None
                     ) -> dict[str, Any]:
        """Molecule embeddings and fingerprint probabilities for a test-schema frame.

        Returns {"molecule_id" (sorted), "z" (n, emb), "logits" (n, n_bits), "proba"}.
        ``molecules``: ``io.mass_tables`` molecule table (computed when None).
        """
        from .fpmodel import spectra_from_frame  # noqa: PLC0415

        cfg = cfg or Config()
        mode = mode or cfg.enc_query_mode
        if molecules is None:
            from .io import mass_tables  # noqa: PLC0415

            molecules = mass_tables(df, cfg)[0]
        df = df.reset_index(drop=True)
        store, meta = spectra_from_frame(df, cfg)
        ids = molecules["molecule_id"].astype(str).to_numpy()
        code = {m: i for i, m in enumerate(ids)}
        mol_code = np.array([code[str(m)] for m in df["molecule_id"]], np.int64)
        qs = build_queries(mol_code, meta.polarity, meta.adduct_code, store.n_peaks,
                           self.spec.max_members, meta.collision_energy)
        m = {"prec": meta.precursor_mz, "ce": meta.collision_energy, "ne": meta.n_energies,
             "add": meta.adduct_code, "pol": meta.polarity, "inst": meta.instrument_code}
        z, lg = self.encode_store(store.mz, store.intensity, store.offsets, m, qs,
                                  molecules["M"].to_numpy(np.float64), cfg.enc_batch_queries)
        ze, le = self.aggregate(z, lg, qs, len(ids), mode)
        return {"molecule_id": ids, "z": ze, "logits": le, "proba": 1.0 / (1.0 + np.exp(-le))}

    # ---- candidates
    def embed_candidates(self, packed: np.ndarray, batch: int = 8192) -> np.ndarray | None:
        """Molecule-tower embeddings (n, emb) of packed fingerprints; None without a tower."""
        if self.net.tower is None:
            return None
        torch = _torch()
        dev = torch.device(self.device)
        out = []
        with torch.no_grad():
            for a in range(0, len(packed), batch):
                p = torch.as_tensor(np.ascontiguousarray(packed[a:a + batch], np.uint8), device=dev)
                out.append(self.net.embed_molecules(_unpack_t(torch, p).float()).cpu().numpy())
        if not out:
            return np.zeros((0, self.spec.emb_dim), np.float32)
        return np.concatenate(out)

    def score(self, z: np.ndarray, proba: np.ndarray, packed: np.ndarray,
              cfg: Config | None = None, cand_emb: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Scores of candidates (packed fps, n x 512) for ONE molecule (z, proba rows)."""
        cfg = cfg or Config()
        bits = unpack_bits(packed, self.spec.n_bits).astype(np.float32)
        p = np.asarray(proba, np.float64)
        pc = np.clip(p, cfg.fp_prob_min, cfg.fp_prob_max)
        w = np.log(pc) - np.log1p(-pc)
        bern = bits @ w.astype(np.float32) + float(np.log1p(-pc).sum())
        nb = np.sqrt(bits.sum(1))
        fpcos = (bits @ p.astype(np.float32)) / np.maximum(nb * np.linalg.norm(p), 1e-12)
        out = {"bern": bern.astype(np.float64), "fpcos": fpcos.astype(np.float64)}
        if cand_emb is None:
            cand_emb = self.embed_candidates(packed)
        if cand_emb is not None:
            out["con"] = (cand_emb @ np.asarray(z, np.float32)).astype(np.float64)
        return out


def export_model(net, spec: EncoderSpec, path: Path | str, info: Mapping[str, Any] | None = None,
                 half: bool = False) -> Path:
    """Save {spec_json, info_json, state_dict} (loadable with ``weights_only=True``)."""
    torch = _torch()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sd = {k: (v.detach().cpu().half() if half and v.is_floating_point() else v.detach().cpu())
          for k, v in net.state_dict().items()}
    torch.save({"spec_json": json.dumps(spec.to_dict()), "info_json": json.dumps(dict(info or {}),
                default=str), "state_dict": sd}, p)
    return p


# ======================================================================================
# torch half: evaluation on packs
# ======================================================================================


def evaluate_packs(model: EncoderModel, packs: Mapping[str, EvalPack], np_fp: np.ndarray,
                   cfg: Config | None = None, modes: Sequence[str] = ("all", "groups", "singles"),
                   return_ranks: bool = False) -> dict[str, Any]:
    """Class 2 MRR of each score x query mode on each pack (see module docstring)."""
    torch = _torch()
    cfg = cfg or Config()
    dev = torch.device(model.device)
    out: dict[str, Any] = {}
    for name, pack in packs.items():
        t0 = time.time()
        q = pack.q
        n_mol = len(pack.mol)
        npk = np.diff(q["q_off"])
        qs = build_queries(q["q_mol"], q["q_pol"], q["q_add"], npk, model.spec.max_members, q["q_ce"])
        meta = {"prec": q["q_prec"], "ce": q["q_ce"], "ne": q["q_ne"], "add": q["q_add"],
                "pol": q["q_pol"], "inst": q["q_inst"]}
        z, lg = model.encode_store(q["q_mz"], q["q_int"].astype(np.float32), q["q_off"], meta, qs,
                                   pack.mol["M"].to_numpy(np.float64), cfg.enc_batch_queries)
        uniq, inv = np.unique(pack.pool_idx, return_inverse=True)
        cand_packed = np.asarray(np_fp[uniq], np.uint8)
        cand_emb = model.embed_candidates(cand_packed)
        cand_t = torch.as_tensor(cand_packed, device=dev)
        res: dict[str, Any] = {}
        for mode in modes:
            ze, le = model.aggregate(z, lg, qs, n_mol, mode)
            p = 1.0 / (1.0 + np.exp(-le.astype(np.float64)))
            pc = np.clip(p, cfg.fp_prob_min, cfg.fp_prob_max)
            w = (np.log(pc) - np.log1p(-pc)).astype(np.float32)
            scores = {k: np.zeros(len(pack.pool_idx), np.float64) for k in
                      (["con"] if cand_emb is not None else []) + ["fpcos", "bern"]}
            off = pack.pool_off
            with torch.no_grad():
                for i in range(n_mol):
                    a, b = int(off[i]), int(off[i + 1])
                    if a == b:
                        continue
                    ci = inv[a:b]
                    bits = _unpack_t(torch, cand_t[torch.as_tensor(ci, device=dev)]).float()
                    pt = torch.as_tensor(p[i], dtype=torch.float32, device=dev)
                    scores["bern"][a:b] = (bits @ torch.as_tensor(w[i], device=dev)).cpu().numpy()
                    nb = bits.sum(1).sqrt().clamp(min=1e-6)
                    scores["fpcos"][a:b] = ((bits @ pt) / (nb * pt.norm())).cpu().numpy()
                    if cand_emb is not None:
                        scores["con"][a:b] = cand_emb[ci] @ ze[i]
            for k, s in scores.items():
                m = rank_metrics(s, pack.pool_pos, off)
                if return_ranks:
                    res[f"{k}/{mode}/rank"] = m["rank"]
                res[f"{k}/{mode}"] = {kk: v for kk, v in m.items() if kk != "rank"}
        res["n_queries"] = int(len(qs))
        res["seconds"] = round(time.time() - t0, 1)
        res["recall"] = float((np.add.reduceat(pack.pool_pos.astype(np.int64), off[:-1])
                               [np.diff(off) > 0] > 0).sum() / max(n_mol, 1)) if len(pack.pool_pos) else 0.0
        res["median_pool"] = float(np.median(np.diff(off))) if n_mol else 0.0
        out[name] = res
    return out


# ======================================================================================
# torch half: training
# ======================================================================================


def _param_groups(net, wd: float):
    decay, no_decay = [], []
    for n, p in net.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "emb" in n or "norm" in n or n.endswith(".bias") or "log_scale" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def _lr_factor(step: int, frac: float, cfg: Config) -> float:
    warm = min(1.0, (step + 1) / max(1, cfg.enc_warmup_steps))
    cos = cfg.enc_lr_min_frac + (1 - cfg.enc_lr_min_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, frac)))
    return warm * cos


def _flat_metrics(ev: Mapping[str, Any]) -> dict[str, float]:
    out = {}
    for name, res in ev.items():
        for k, v in res.items():
            if isinstance(v, dict) and "mrr" in v:
                out[f"{name}:{k}"] = round(float(v["mrr"]), 4)
    return out


def train(bundle_dir: Path | str, np_fp_path: Path | str, out_dir: Path | str,
          cfg: Config | None = None, arm: str = "focal_con", hours: float | None = None,
          device: str | None = None, max_steps: int | None = None, seed: int | None = None,
          eval_max_mol: int = 0, resume: bool = True, spec_over: Mapping[str, Any] | None = None
          ) -> dict[str, Any]:
    """Train one arm (``ARMS``) under a wall-time budget; see the module docstring.

    Writes into ``out_dir``: ``train_log.csv`` (every ``enc_log_every_steps``),
    ``eval.jsonl`` (class 2 metrics every ``enc_eval_every_min`` and at the end),
    ``checkpoint.pt`` (model + optimiser, for resuming), ``encoder_<arm>.pt`` (final
    export) and ``summary.json``.
    """
    torch = _torch()
    cfg = cfg or Config()
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; one of {sorted(ARMS)}")
    a = ARMS[arm]
    hours = cfg.enc_hours if hours is None else float(hours)
    seed = cfg.seed if seed is None else int(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    t_start = time.time()
    bundle = Bundle.load(bundle_dir, mmap=True)
    np_fp = np.load(np_fp_path, mmap_mode="r")
    if np_fp.shape[0] != bundle.n_np:
        raise ValueError(f"np_fp rows {np_fp.shape[0]} != bundle n_np {bundle.n_np}")
    data = DeviceData(bundle, np_fp, device, cfg)
    packs = bundle.eval_packs()
    if eval_max_mol > 0:
        packs = {k: v.subset(eval_max_mol, seed) for k, v in packs.items()}
    spec = EncoderSpec.from_config(cfg, use_tower=a["use_tower"], **dict(spec_over or {}))
    net = build_net(spec).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(_param_groups(net, cfg.enc_weight_decay), lr=cfg.enc_lr,
                            betas=(0.9, 0.98), eps=1e-6)
    use_amp = data.dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    gen = torch.Generator(device=data.dev)
    gen.manual_seed(seed)
    step, train_seconds = 0, 0.0
    ck_path = out / "checkpoint.pt"
    if resume and ck_path.exists():
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        net.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, train_seconds = int(ck["step"]), float(ck["train_seconds"])
        gen.manual_seed(seed + step)
        log.info("resumed from %s at step %d (%.0f s trained)", ck_path, step, train_seconds)
    budget = hours * 3600.0
    aug = {"mz_ppm": cfg.enc_aug_mz_ppm, "peak_drop": cfg.enc_aug_peak_drop,
           "int_sigma": cfg.enc_aug_intensity_sigma, "noise_prob": cfg.enc_aug_noise_prob,
           "noise_max": cfg.enc_aug_noise_max, "noise_int_min": cfg.enc_aug_noise_int_min,
           "noise_int_max": cfg.enc_aug_noise_int_max}
    n_neg = cfg.enc_n_neg if a["con_weight"] > 0 else 0
    info = {"arm": arm, **a, "n_params": int(n_params), "device": device, "seed": seed,
            "bundle": str(bundle_dir), "bundle_info": {k: bundle.info.get(k) for k in
                                                       ("holdout_sha256", "n_np", "n_structures",
                                                        "n_spectra", "n_groups", "created")},
            "config_overrides": cfg.overrides()}
    log.info("encoder %s: %d parameters, device %s, budget %.2f h, %d groups, %d structures",
             arm, n_params, device, hours, len(bundle.g["grp_struct"]), len(bundle.st["st_mass"]))
    log_path = out / "train_log.csv"
    new_log = not log_path.exists() or step == 0
    logf = log_path.open("w" if new_log else "a", encoding="utf-8")
    if new_log:
        logf.write("step,train_s,lr,loss,fp_loss,con_loss,con_acc,scale,samples_per_s\n")
    evalf = (out / "eval.jsonl").open("a", encoding="utf-8")
    model = EncoderModel(net, spec, info, device)

    def run_eval(tag: str) -> dict[str, Any]:
        if not packs:
            return {}
        ev = evaluate_packs(model, packs, np_fp, cfg)
        net.train()
        rec = {"tag": tag, "step": step, "train_s": round(train_seconds, 1), **_flat_metrics(ev),
               "detail": ev}
        evalf.write(json.dumps(rec, default=float) + "\n")
        evalf.flush()
        log.info("eval %s step %d: %s", tag, step, _flat_metrics(ev))
        return ev

    def save_ck() -> None:
        tmp = ck_path.with_suffix(".tmp")
        torch.save({"state_dict": net.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "step": step, "train_seconds": train_seconds,
                    "spec_json": json.dumps(spec.to_dict()), "info_json": json.dumps(info, default=str)},
                   tmp)
        os.replace(tmp, ck_path)

    net.train()
    acc_loss = acc_fp = acc_con = acc_acc = 0.0
    acc_n = 0
    t_log = time.time()
    t_eval = time.time()
    B = cfg.enc_batch_size
    while True:
        if train_seconds >= budget or (max_steps is not None and step >= max_steps):
            break
        t0 = time.time()
        frac = train_seconds / budget if budget > 0 else 1.0
        for gp in opt.param_groups:
            gp["lr"] = cfg.enc_lr * _lr_factor(step, frac, cfg)
        batch = data.sample(B, spec, n_neg, gen, aug)
        with torch.autocast(device_type=data.dev.type, dtype=torch.float16, enabled=use_amp):
            logits, z = net(batch["tok"])
            loss = torch.zeros((), device=data.dev)
            lf = fp_loss(logits.float(), batch["y"], a["fp_loss"], cfg.enc_focal_gamma,
                         cfg.enc_fp_loss_scale)
            if a["fp_weight"] > 0:
                loss = loss + a["fp_weight"] * lf
            if a["con_weight"] > 0:
                lc, cacc = contrastive_loss(net, z, batch)
                loss = loss + a["con_weight"] * lc
            else:
                lc, cacc = torch.zeros(()), torch.zeros(())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.enc_grad_clip)
        scaler.step(opt)
        scaler.update()
        step += 1
        acc_loss += float(loss.detach())
        acc_fp += float(lf.detach())
        acc_con += float(lc.detach())
        acc_acc += float(cacc)
        acc_n += 1
        if data.dev.type == "cuda" and step % 50 == 0:
            torch.cuda.synchronize()
        train_seconds += time.time() - t0
        if step % cfg.enc_log_every_steps == 0:
            dt = time.time() - t_log
            logf.write(f"{step},{train_seconds:.1f},{opt.param_groups[0]['lr']:.3e},"
                       f"{acc_loss / acc_n:.4f},{acc_fp / acc_n:.4f},{acc_con / acc_n:.4f},"
                       f"{acc_acc / acc_n:.4f},{float(net.scale()):.2f},"
                       f"{acc_n * B / max(dt, 1e-9):.1f}\n")
            logf.flush()
            if step % (cfg.enc_log_every_steps * 10) == 0:
                log.info("step %d (%.0f s): loss %.4f fp %.4f con %.4f acc %.3f, %.0f samples/s",
                         step, train_seconds, acc_loss / acc_n, acc_fp / acc_n, acc_con / acc_n,
                         acc_acc / acc_n, acc_n * B / max(dt, 1e-9))
            acc_loss = acc_fp = acc_con = acc_acc = 0.0
            acc_n = 0
            t_log = time.time()
        if time.time() - t_eval >= cfg.enc_eval_every_min * 60.0:
            save_ck()
            run_eval("periodic")
            t_eval = time.time()
            t_log = time.time()
    save_ck()
    final = run_eval("final")
    info.update({"steps": step, "train_seconds": round(train_seconds, 1),
                 "wall_seconds": round(time.time() - t_start, 1)})
    model.info = info
    export_path = export_model(net, spec, out / f"encoder_{arm}.pt", info)
    summary = {**info, "export": str(export_path), "final": _flat_metrics(final) if final else {},
               "final_detail": final}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    logf.close()
    evalf.close()
    log.info("encoder %s done: %d steps, %.0f s training, export %s", arm, step, train_seconds,
             export_path)
    return summary


# ======================================================================================
# Pipeline stage (registered by ``register()``; torch imported only when it runs)
# ======================================================================================

_MODEL_CACHE: dict[tuple[str, str], EncoderModel] = {}


def _model_path(cfg: Config) -> Path:
    p = Path(cfg.enc_model_path)
    return p if p.is_absolute() else paths.MODELS / p


def get_model(cfg: Config) -> EncoderModel:
    key = (str(_model_path(cfg)), cfg.enc_device)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = EncoderModel.load(key[0], cfg.enc_device)
    return _MODEL_CACHE[key]


ENCODER_COLUMNS = ("enc_con", "enc_fpcos", "enc_bern", "enc_con_rank", "enc_bern_rank")


def stage_encoder(state, cfg: Config):
    """Pipeline stage "encoder": adds ENCODER_COLUMNS to ``state.pool``.

    Candidate fingerprints: NP-tier rows from ``np_fp.npy`` (``np_row``), other rows
    from ``chem.morgan_fps`` of their SMILES (InChI round trip, as for targets). Ranks
    are 1-based within the molecule's pool (higher score first, ties by ik14).
    """
    pool = state.pool
    if pool is None or not len(pool) or not cfg.enc_model_path:
        return state
    model = get_model(cfg)
    enc = model.encode_frame(state.test_df, cfg, state.molecules)
    row_of = {m: i for i, m in enumerate(enc["molecule_id"])}
    n = len(pool)
    packed = np.zeros((n, N_BYTES), np.uint8)
    np_row = pool["np_row"].to_numpy(np.int64) if "np_row" in pool.columns else np.full(n, -1)
    has = np_row >= 0
    if has.any():
        dbh = state.db if state.db is not None else __import__("casmi.db", fromlist=["get_db"]).get_db(cfg)
        packed[has] = dbh.np_fingerprints(np_row[has])
    if (~has).any():
        from .chem import morgan_fps  # noqa: PLC0415

        fps, _ = morgan_fps(pool.loc[~has, "smiles"].astype(object).tolist(), cfg)
        packed[~has] = fps
    cand_emb = model.embed_candidates(packed)
    cols = {c: np.full(n, np.nan) for c in ENCODER_COLUMNS}
    mids = pool["molecule_id"].astype(str).to_numpy()
    for m, idx in pd.Series(np.arange(n)).groupby(mids, sort=True):
        i = row_of.get(m)
        if i is None:
            continue
        rows = idx.to_numpy()
        sc = model.score(enc["z"][i], enc["proba"][i], packed[rows], cfg,
                         None if cand_emb is None else cand_emb[rows])
        cols["enc_bern"][rows] = sc["bern"]
        cols["enc_fpcos"][rows] = sc["fpcos"]
        ik = pool["ik14"].astype(str).to_numpy()[rows]
        for key, col in (("bern", "enc_bern_rank"), ("con", "enc_con_rank")):
            if key not in sc:
                continue
            o = np.lexsort((ik, -sc[key]))
            r = np.empty(len(rows))
            r[o] = np.arange(1, len(rows) + 1)
            cols[col][rows] = r
        if "con" in sc:
            cols["enc_con"][rows] = sc["con"]
    for c, v in cols.items():
        pool[c] = v
    state.pool = pool
    return state


def register() -> None:
    """Register the "encoder" pipeline stage (add it to ``cfg.pipeline_stages`` and
    ``casmi.encoder`` to ``cfg.pipeline_plugins``)."""
    from . import pipeline  # noqa: PLC0415

    pipeline.register_stage("encoder", stage_encoder, replace=True)


try:  # registration at import, as pipeline plugins expect; pipeline imports no torch
    register()
except Exception:  # noqa: BLE001 - never fail an import because of the registry
    log.debug("encoder stage not registered", exc_info=True)

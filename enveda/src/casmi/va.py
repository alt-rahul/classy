"""Class 3 virtual-analog (VA) tier: one-edit products of every known natural product,
stored as a database tier (ROADMAP R13).

Why: class 3 molecules are in neither PubChem nor COCONUT; the host calls them
"hypothesised natural products" and "natural product analogs". Public generators edit only
the 6 to 20 nearest spectral parents, so their first limit is reach (KAGGLE_INTEL 5: 46
percent with close analogs, 16.5 percent without). Applying the edit rules of
``casmi.generate`` once, at every matching site, to every NP-tier structure and every
NP-like train-library structure turns one-edit generation into a mass lookup with no
choice of parents; a second edit is still applied on demand to the best seeds (GEN).

Seeds (``seed_table``)
----------------------
``seed_id`` = NP-tier row (``np_row`` of ``np.parquet``) for the NP tier, then the train
structures that are not in the NP tier and have spectra in at least one library outside
``cfg.va_exclude_libs`` (enveda-180 and drug_plus: synthetic and drug libraries), in ik14
order. Columns ``SEED_COLUMNS``; ``pri`` is the provenance priority (0 = preferred): seeds
with library spectra first, then by SID count, then seed_id.

Products (``seed_products``, ``build_products``)
------------------------------------------------
``generate.prepare_seed``, then every rule of ``generate.RULES`` at every matching site
(``generate._products``: sanitised, single fragment, seed itself excluded). Kept when the
net charge is 0, every element is in ``formula.ELEMENTS``, and the monoisotopic mass is in
[``cfg.va_mass_min``, ``cfg.va_mass_max``]. Per seed, products are deduplicated by ik14
(plain InChIKey first block): ``rule`` = lowest rule code that makes it, ``n_rules`` = how
many rules do. A product is novel when its ik14 is in none of the known key sets (PubChem
tier, NP tier, train structures); only novel products enter the tier. Per rule the build
counts products and how many are known NP-tier structures (``rule_np_rate``: how often the
rule maps a natural product onto another known natural product, a plausibility prior).

Holdout hits: a product whose ik14 is a holdout ik14 (or a DB row with a holdout's metric
key), or whose formula and heavy-atom skeleton match a holdout molecule and whose metric
key (``metric.mkey_mol``) is the holdout's, is written to the side table ``HIT_COLUMNS``
whatever its novelty (holdout truths are train structures, so the tier itself never holds
them). The reach evaluation masks seeds per molecule on this table (no rebuild).

Keys: ik14 strings are mapped to uint64 (``ik14_keys``: base-26 polynomial modulo 2**64;
a false match needs two keys exactly k * 2**64 apart, about 1e-11 per product against
90M PubChem keys). Sorted key arrays answer membership by ``np.searchsorted``.

Shards
------
``data/processed/db/va/mass_XXXX.parquet`` (10 Da, as the PubChem tier), rows sorted by
(mono_mass, ik14), columns ``VA_COLUMNS``: ik14, smiles, formula, mono_mass, seed_id (of
the preferred producing seed), rule (code in ``generate.RULES``), n_seeds (distinct seeds
producing it: a plausibility feature), n_pairs (seed-rule pairs). Next to them:
``seeds.parquet`` (``SEED_COLUMNS``) and ``_index.json`` (counts, rule table and statistics,
build provenance). ``db.CandidateDB`` serves the tier as ``tiers=(..., "va")`` (appended
section of db.py); ``VATier`` is the reader.

Reach (``reach_table``)
-----------------------
Per holdout molecule and its excluded-structure set (truth keys, plus near neighbours for
C3h): the truth is in VA when an allowed seed (ik14 not excluded) produces it (hit table);
in GEN when ``generate.generate`` with two edits from the best ``cfg.va_gen_n_seeds``
allowed seeds (e.g. spectral analog hits) at the truth's formula produces it.

Only numpy, pandas and pyarrow at import; RDKit and ``casmi.generate`` are imported inside
the functions that need them (the reader is inference-path safe).
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import paths
from .config import Config

__all__ = [
    "VA_COLUMNS", "SEED_COLUMNS", "HIT_COLUMNS", "PRODUCT_COLUMNS", "LOOKUP_COLUMNS", "LOOKUP_DTYPES",
    "SHARD_DA", "va_dir", "ik14_keys", "sorted_keys", "in_sorted", "keys_from_parquet",
    "seed_table", "seed_products", "build_products", "dedupe", "merge_parts", "write_shards",
    "VATier", "seed_mask", "reach_table", "summarize_reach", "window_counts",
]

log = logging.getLogger(__name__)

SHARD_DA = 10
BAD_KEY = np.uint64(0xFFFFFFFFFFFFFFFF)
_P26 = np.array([26 ** (13 - i) for i in range(14)], dtype=np.uint64)

VA_COLUMNS: tuple[str, ...] = ("ik14", "smiles", "formula", "mono_mass", "seed_id", "rule",
                               "n_seeds", "n_pairs")
# Rows produced by the build workers (before deduplication across seeds).
PRODUCT_COLUMNS: tuple[str, ...] = ("ik14", "smiles", "formula", "mono_mass", "seed_id",
                                    "seed_pri", "rule", "n_rules")
SEED_COLUMNS: tuple[str, ...] = ("seed_id", "ik14", "smiles", "formula", "mono_mass", "source",
                                 "np_row", "in_train", "n_spectra", "sid_count", "pubmed_count",
                                 "np_likeness", "pri")
HIT_COLUMNS: tuple[str, ...] = ("ik14", "mkey", "match", "seed_id", "rule", "smiles")
# Provenance columns ``db`` appends to lookups that include the VA tier (NA on other rows).
LOOKUP_DTYPES: dict[str, str] = {
    "va_seed_id": "Int32", "va_rule": "Int16", "va_rule_name": "string", "va_n_seeds": "Int32",
    "va_n_pairs": "Int32", "va_rule_np_rate": "Float32", "va_seed_tier": "string",
    "va_seed_ik14": "string", "va_seed_np_row": "Int64", "va_seed_mass": "Float64",
    "va_seed_in_train": "boolean", "va_seed_n_spectra": "Int32", "va_seed_sid_count": "Int32",
    "va_seed_pubmed_count": "Int32", "va_seed_np_likeness": "Float32",
}
LOOKUP_COLUMNS: tuple[str, ...] = tuple(LOOKUP_DTYPES)

_PRODUCT_SCHEMA = pa.schema([("ik14", pa.string()), ("smiles", pa.string()), ("formula", pa.string()),
                             ("mono_mass", pa.float64()), ("seed_id", pa.int32()),
                             ("seed_pri", pa.int32()), ("rule", pa.int16()), ("n_rules", pa.int16())])
_MERGE_SCHEMA = pa.schema([("ik14", pa.string()), ("smiles", pa.string()), ("formula", pa.string()),
                           ("mono_mass", pa.float64()), ("seed_id", pa.int32()), ("seed_pri", pa.int32()),
                           ("rule", pa.int16()), ("n_seeds", pa.int32()), ("n_pairs", pa.int32())])
_VA_SCHEMA = pa.schema([("ik14", pa.string()), ("smiles", pa.string()), ("formula", pa.string()),
                        ("mono_mass", pa.float64()), ("seed_id", pa.int32()), ("rule", pa.int16()),
                        ("n_seeds", pa.int32()), ("n_pairs", pa.int32())])
_HIT_SCHEMA = pa.schema([("ik14", pa.string()), ("mkey", pa.string()), ("match", pa.string()),
                         ("seed_id", pa.int32()), ("rule", pa.int16()), ("smiles", pa.string())])


def va_dir() -> Path:
    """Directory of the VA tier (``PROCESSED/db/va``)."""
    return paths.PROCESSED / "db" / "va"


def shard_bucket(mass: float) -> int:
    return int(math.floor(mass / SHARD_DA)) * SHARD_DA


def shard_name(bucket: int) -> str:
    return f"mass_{int(bucket):04d}.parquet"


# --------------------------------------------------------------------------------------
# ik14 keys
# --------------------------------------------------------------------------------------

def ik14_keys(values: Any) -> np.ndarray:
    """uint64 key per ik14 (module docstring); ``BAD_KEY`` for nulls and anything that is not
    14 upper-case letters. Accepts an Arrow (chunked) string array or any sequence."""
    if isinstance(values, pa.ChunkedArray):
        values = values.combine_chunks() if values.num_chunks != 1 else values.chunk(0)
    if not isinstance(values, pa.Array):
        values = pa.array([v if isinstance(v, str) else None for v in values], type=pa.string())
    if pa.types.is_large_string(values.type):
        values = values.cast(pa.string())
    n = len(values)
    out = np.full(n, BAD_KEY, dtype=np.uint64)
    if n == 0:
        return out
    bufs = values.buffers()
    off = np.frombuffer(bufs[1], dtype=np.int32)[values.offset:values.offset + n + 1]
    data = np.frombuffer(bufs[2], dtype=np.uint8) if bufs[2] is not None else np.zeros(0, np.uint8)
    lens = np.diff(off)
    valid = lens == 14
    if values.null_count:
        valid &= values.is_valid().to_numpy(zero_copy_only=False)
    if valid.all() and off[-1] - off[0] == 14 * n:
        mat = data[off[0]:off[-1]].reshape(n, 14)
        rows = np.arange(n)
    else:
        rows = np.flatnonzero(valid)
        mat = data[off[:-1][rows, None] + np.arange(14)[None, :]]
    letters = mat.astype(np.int64) - 65
    ok = ((letters >= 0) & (letters < 26)).all(axis=1)
    with np.errstate(over="ignore"):
        k = (letters.astype(np.uint64) * _P26[None, :]).sum(axis=1, dtype=np.uint64)
    out[rows[ok]] = k[ok]
    return out


def sorted_keys(keys: Iterable[np.ndarray] | np.ndarray) -> np.ndarray:
    """Sorted unique keys without ``BAD_KEY``."""
    if isinstance(keys, np.ndarray):
        a = keys.astype(np.uint64, copy=True)
    else:
        parts = [np.asarray(k, dtype=np.uint64) for k in keys]
        a = np.concatenate(parts) if parts else np.zeros(0, np.uint64)
    a.sort()
    a = a[a != BAD_KEY]
    if len(a) > 1:
        keep = np.empty(len(a), bool)
        keep[0] = True
        np.not_equal(a[1:], a[:-1], out=keep[1:])
        a = a[keep]
    return a


def in_sorted(keys: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Boolean mask: ``keys`` present in the sorted unique array ``ref``."""
    keys = np.asarray(keys, dtype=np.uint64)
    if len(ref) == 0 or len(keys) == 0:
        return np.zeros(len(keys), bool)
    i = np.searchsorted(ref, keys)
    i[i >= len(ref)] = len(ref) - 1
    return (ref[i] == keys) & (keys != BAD_KEY)


def keys_from_parquet(files: Sequence[Path | str], column: str = "ik14") -> np.ndarray:
    """Sorted unique keys of the ``column`` of every parquet file, read one file at a time
    into one preallocated array (peak memory about twice the result: 1.5 GB for 90M keys)."""
    files = list(files)
    n = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    out = np.empty(n, np.uint64)
    pos = 0
    for f in files:
        k = ik14_keys(pq.read_table(f, columns=[column]).column(column))
        out[pos:pos + len(k)] = k
        pos += len(k)
        del k
    out = out[:pos]
    out.sort()
    if not len(out):
        return out
    keep = np.empty(len(out), bool)
    keep[0] = True
    np.not_equal(out[1:], out[:-1], out=keep[1:])
    keep &= out != BAD_KEY
    return out[keep]


# --------------------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------------------

def seed_table(np_df: pd.DataFrame, train: pd.DataFrame | None, cfg: Config | None = None) -> pd.DataFrame:
    """The seed list (module docstring). ``np_df``: np.parquet in row order (ik14, smiles,
    formula, mono_mass and optionally coconut_np_likeness, pubmed_count, sid_count);
    ``train``: train structures.parquet (ik14, smiles, formula, mono_mass, libs, n_spectra,
    rdkit_ok, np_score, pmid_count) or None."""
    cfg = cfg or Config()
    n_np = len(np_df)

    def col(df, c, dtype, fill):
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(fill).to_numpy().astype(dtype)
        return np.full(len(df), fill, dtype=dtype)

    np_ik = np_df["ik14"].astype(str).to_numpy(object)
    nps = pd.DataFrame({
        "seed_id": np.arange(n_np, dtype=np.int32), "ik14": np_ik,
        "smiles": np_df["smiles"].astype(object).to_numpy(object),
        "formula": np_df["formula"].astype(object).to_numpy(object),
        "mono_mass": np_df["mono_mass"].to_numpy(np.float64), "source": "np",
        "np_row": np.arange(n_np, dtype=np.int32),
        "sid_count": col(np_df, "sid_count", np.int32, -1),
        "pubmed_count": col(np_df, "pubmed_count", np.int32, -1),
        "np_likeness": col(np_df, "coconut_np_likeness", np.float32, np.nan),
    })
    nps["in_train"] = False
    nps["n_spectra"] = np.int32(0)
    tr = pd.DataFrame(columns=list(SEED_COLUMNS))
    if train is not None and len(train):
        t = train.copy()
        if "rdkit_ok" in t.columns:
            t = t[t["rdkit_ok"].astype(bool)]
        t = t[t["smiles"].notna()].drop_duplicates("ik14").sort_values("ik14", kind="mergesort")
        nspec = dict(zip(t["ik14"].astype(str), col(t, "n_spectra", np.int32, 0)))
        in_np = nps["ik14"].isin(set(t["ik14"].astype(str))).to_numpy()
        nps["in_train"] = in_np
        nps["n_spectra"] = np.array([nspec.get(k, 0) if f else 0 for k, f in zip(np_ik, in_np)],
                                    dtype=np.int32)
        bad = set(cfg.va_exclude_libs)
        libs = t["libs"].fillna("").astype(str).str.split(";") if "libs" in t.columns else None
        keep = ~t["ik14"].astype(str).isin(set(np_ik)).to_numpy()
        if libs is not None:
            keep &= np.fromiter((bool(set(x) - bad - {""}) for x in libs), bool, len(t))
        t = t[keep].reset_index(drop=True)
        tr = pd.DataFrame({
            "seed_id": (n_np + np.arange(len(t))).astype(np.int32),
            "ik14": t["ik14"].astype(str).to_numpy(object),
            "smiles": t["smiles"].astype(object).to_numpy(object),
            "formula": t["formula"].astype(object).to_numpy(object),
            "mono_mass": pd.to_numeric(t["mono_mass"], errors="coerce").to_numpy(np.float64),
            "source": "train", "np_row": np.int32(-1), "in_train": True,
            "n_spectra": col(t, "n_spectra", np.int32, 0),
            "sid_count": np.int32(-1),
            "pubmed_count": col(t, "pmid_count", np.int32, -1),
            "np_likeness": col(t, "np_score", np.float32, np.nan),
        })
    seeds = pd.concat([nps, tr], ignore_index=True) if len(tr) else nps
    has = (seeds["n_spectra"].to_numpy() > 0).astype(np.int64)
    order = np.lexsort((seeds["seed_id"].to_numpy(), -seeds["sid_count"].to_numpy(np.int64), -has))
    pri = np.empty(len(seeds), np.int32)
    pri[order] = np.arange(len(seeds), dtype=np.int32)
    seeds["pri"] = pri
    for c, dt in (("seed_id", np.int32), ("np_row", np.int32), ("n_spectra", np.int32),
                  ("sid_count", np.int32), ("pubmed_count", np.int32), ("np_likeness", np.float32)):
        seeds[c] = seeds[c].astype(dt)
    seeds["in_train"] = seeds["in_train"].astype(bool)
    return seeds[list(SEED_COLUMNS)]


# --------------------------------------------------------------------------------------
# Products of one seed (RDKit)
# --------------------------------------------------------------------------------------

def seed_products(smiles: str, cfg: Config | None = None, time_limit_s: float | None = None
                  ) -> tuple[list[tuple[str, str, str, float, int, int]], dict[int, int], bool]:
    """One-edit products of a seed: ``[(ik14, smiles, formula, mono_mass, rule, n_rules)]``
    deduplicated by ik14 and sorted by ik14 (module docstring), the number of distinct
    products per rule code (before the mass / charge filters: ``rule -> n``), and whether
    the time limit cut the seed short. Empty when the seed does not prepare."""
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdMolDescriptors

    from . import generate as gen
    from .chem import mol_from_smiles

    cfg = cfg or Config()
    mol = gen.prepare_seed(smiles)
    if mol is None or Chem.GetFormalCharge(mol) != 0 or len(Chem.GetMolFrags(mol)) != 1:
        return [], {}, False
    if mol.GetNumHeavyAtoms() > int(cfg.va_seed_max_heavy) or gen.mol_counts(mol) is None:
        return [], {}, False
    seed_smi = Chem.MolToSmiles(mol, isomericSmiles=False)
    t0 = time.perf_counter()
    by_smi: dict[str, list[int]] = {}
    per_rule: dict[int, int] = {}
    cut = False
    for code, rule in enumerate(gen.RULES):
        prods = gen._products(rule, mol, seed_smi)
        if prods:
            per_rule[code] = len(prods)
        for p in prods:
            by_smi.setdefault(p, []).append(code)
        if time_limit_s is not None and time.perf_counter() - t0 > time_limit_s:
            cut = code < len(gen.RULES) - 1
            break
    lo, hi = float(cfg.va_mass_min), float(cfg.va_mass_max)
    best: dict[str, tuple[str, str, str, float, int, int]] = {}
    for smi in sorted(by_smi):
        pm = mol_from_smiles(smi)
        if pm is None or Chem.GetFormalCharge(pm) != 0:
            continue
        mass = float(rdMolDescriptors.CalcExactMolWt(pm))
        if not lo <= mass <= hi or gen.mol_counts(pm) is None:
            continue
        with rdBase.BlockLogs():
            try:
                key = Chem.MolToInchiKey(pm)
            except Exception:  # noqa: BLE001
                key = ""
        if not key:
            continue
        ik = key[:14]
        codes = by_smi[smi]
        row = (ik, smi, rdMolDescriptors.CalcMolFormula(pm), mass, min(codes), len(set(codes)))
        old = best.get(ik)
        if old is None:
            best[ik] = row
        else:  # same ik14 from two SMILES (InChI merges them): lowest rule, rules counted once
            best[ik] = (old[0], old[1], old[2], old[3], min(old[4], row[4]), old[5] + row[5])
    return [best[k] for k in sorted(best)], per_rule, cut


# Worker state (set by ``_init_worker``): known key arrays (memory-mapped), holdout lookups.
_W: dict[str, Any] = {}


def _init_worker(key_files: Mapping[str, str], holdout: Mapping[str, Any], cfg_dict: Mapping[str, Any]) -> None:
    _W.clear()
    for name, f in key_files.items():
        _W[name] = np.load(f, mmap_mode="r")
    _W["hold_ik"] = dict(holdout.get("ik14", {}))           # ik14 -> mkey
    _W["hold_formula"] = {f: {sk: set(m) for sk, m in d.items()}
                          for f, d in holdout.get("formula", {}).items()}   # formula -> skeleton -> mkeys
    _W["cfg"] = Config(**dict(cfg_dict)) if cfg_dict else Config()


def _skeleton(mol: Any) -> str | None:
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdMolHash

    with rdBase.BlockLogs():
        try:
            m = Chem.Mol(mol)
            Chem.RemoveStereochemistry(m)
            return rdMolHash.MolHash(m, rdMolHash.HashFunction.ElementGraph)
        except Exception:  # noqa: BLE001
            return None


def _process_chunk(task: tuple[np.ndarray, list[str], np.ndarray]) -> dict[str, Any]:
    """Products of a chunk of seeds (``build_products`` worker)."""
    from .chem import mol_from_smiles
    from .metric import mkey_mol

    seed_ids, smiles, pris = task
    cfg: Config = _W["cfg"]
    t0 = time.perf_counter()
    rows: list[tuple] = []
    hits: list[tuple] = []
    per_rule_all: dict[int, int] = {}
    n_fail = n_cut = 0
    for sid, smi, pri in zip(seed_ids, smiles, pris):
        try:
            prods, per_rule, cut = seed_products(smi, cfg, cfg.va_seed_time_s)
        except Exception:  # noqa: BLE001 - a broken seed must not stop the shard
            prods, per_rule, cut = [], {}, False
        if not prods and not per_rule:
            n_fail += 1
        n_cut += int(cut)
        for c, n in per_rule.items():
            per_rule_all[c] = per_rule_all.get(c, 0) + n
        for p in prods:
            rows.append((*p, int(sid), int(pri)))
    n = len(rows)
    ik = pa.array([r[0] for r in rows], type=pa.string())
    keys = ik14_keys(ik)
    in_pc = in_sorted(keys, _W["pubchem"]) if "pubchem" in _W else np.zeros(n, bool)
    in_np = in_sorted(keys, _W["np"]) if "np" in _W else np.zeros(n, bool)
    in_tr = in_sorted(keys, _W["train"]) if "train" in _W else np.zeros(n, bool)
    known = in_pc | in_np | in_tr
    rule = np.fromiter((r[4] for r in rows), np.int64, n)
    nr = len(_rule_names())
    stats = {
        "n_products": np.bincount(rule, minlength=nr), "n_np": np.bincount(rule[in_np], minlength=nr),
        "n_pubchem": np.bincount(rule[in_pc], minlength=nr), "n_train": np.bincount(rule[in_tr], minlength=nr),
        "n_novel": np.bincount(rule[~known], minlength=nr),
        "n_raw": np.bincount(np.fromiter(per_rule_all.keys(), np.int64, len(per_rule_all)),
                             weights=np.fromiter(per_rule_all.values(), np.float64, len(per_rule_all)),
                             minlength=nr) if per_rule_all else np.zeros(nr),
    }
    hold_ik, hold_f = _W["hold_ik"], _W["hold_formula"]
    for i, r in enumerate(rows):
        mk = hold_ik.get(r[0])
        if mk is not None:
            hits.append((r[0], mk, "ik14", r[6], r[4], r[1]))
            continue
        sks = hold_f.get(r[2])
        if sks:
            m = mol_from_smiles(r[1])
            sk = _skeleton(m) if m is not None else None
            if sk is not None and sk in sks:
                k = mkey_mol(m)
                if k is not None and k in sks[sk]:
                    hits.append((r[0], k, "mkey", r[6], r[4], r[1]))
    nov = np.flatnonzero(~known)
    sel = [rows[i] for i in nov]
    tbl = pa.table({
        "ik14": pa.array([r[0] for r in sel], type=pa.string()),
        "smiles": pa.array([r[1] for r in sel], type=pa.string()),
        "formula": pa.array([r[2] for r in sel], type=pa.string()),
        "mono_mass": pa.array([r[3] for r in sel], type=pa.float64()),
        "seed_id": pa.array([r[6] for r in sel], type=pa.int32()),
        "seed_pri": pa.array([r[7] for r in sel], type=pa.int32()),
        "rule": pa.array([r[4] for r in sel], type=pa.int16()),
        "n_rules": pa.array([r[5] for r in sel], type=pa.int16()),
    }, schema=_PRODUCT_SCHEMA)
    return {"table": tbl, "hits": hits, "stats": stats, "n_seeds": len(seed_ids), "n_fail": n_fail,
            "n_cut": n_cut, "n_rows": n, "seconds": time.perf_counter() - t0,
            "first": int(seed_ids[0]) if len(seed_ids) else -1}


def _rule_names() -> list[str]:
    from . import generate as gen

    return [r.id for r in gen.RULES]


def holdout_lookup(hold_ik: pd.DataFrame, hold_mol: pd.DataFrame) -> dict[str, Any]:
    """Worker holdout lookups from ``holdout_ik14`` (ik14, mkey) and ``holdout_mols`` (mkey,
    formula, skeleton) tables (build_va.py ``inputs``)."""
    ik = {str(a): str(b) for a, b in zip(hold_ik["ik14"], hold_ik["mkey"])}
    f: dict[str, dict[str, list[str]]] = {}
    for mk, fo, sk in zip(hold_mol["mkey"], hold_mol["formula"], hold_mol["skeleton"]):
        if isinstance(fo, str) and isinstance(sk, str):
            f.setdefault(fo, {}).setdefault(sk, []).append(str(mk))
    return {"ik14": ik, "formula": f}


def build_products(seeds: pd.DataFrame, key_files: Mapping[str, Path | str], holdout: Mapping[str, Any],
                   out_dir: Path | str, cfg: Config | None = None, n_jobs: int = 1,
                   chunk: int = 200, budget_s: float | None = None, part_rows: int = 3_000_000,
                   log_every_s: float = 300.0) -> dict[str, Any]:
    """Apply every rule once to every seed in ``seeds`` (rows: seed_id, smiles, pri) and write
    the novel products, deduplicated by ik14 across these seeds, as per-bucket files
    ``out_dir/mass_XXXX.parquet`` (``_MERGE_SCHEMA``), the holdout hits
    ``out_dir/holdout_hits.parquet``, per-rule statistics and ``out_dir/build_info.json``.

    ``key_files``: name -> .npy of sorted uint64 known keys ("pubchem", "np", "train").
    ``budget_s``: stop scheduling chunks after this many seconds (finished chunks are kept;
    ``build_info.json`` lists the seed ids done). Workers: ``chem.mp_context`` processes."""
    from .chem import mp_context

    cfg = cfg or Config()
    out = paths.ensure_dir(out_dir)
    part_dir = paths.ensure_dir(out / "parts")
    seeds = seeds.sort_values("seed_id", kind="mergesort").reset_index(drop=True)
    sid = seeds["seed_id"].to_numpy(np.int32)
    smi = seeds["smiles"].astype(object).tolist()
    pri = seeds["pri"].to_numpy(np.int32)
    tasks = [(sid[a:a + chunk], smi[a:a + chunk], pri[a:a + chunk]) for a in range(0, len(sid), chunk)]
    kf = {k: str(v) for k, v in key_files.items()}
    cfg_d = cfg.to_dict()
    names = _rule_names()
    stats = {k: np.zeros(len(names), np.float64) for k in
             ("n_raw", "n_products", "n_np", "n_pubchem", "n_train", "n_novel")}
    hits: list[tuple] = []
    buf: list[pa.Table] = []
    buf_rows = 0
    parts: list[Path] = []
    done_first: list[int] = []
    tot = {"n_seeds": 0, "n_fail": 0, "n_cut": 0, "n_rows": 0, "worker_seconds": 0.0}
    t0 = time.perf_counter()
    last = t0
    stopped = False

    def flush() -> None:
        nonlocal buf, buf_rows
        if not buf:
            return
        p = part_dir / f"part_{len(parts):05d}.parquet"
        pq.write_table(pa.concat_tables(buf), p, compression="zstd")
        parts.append(p)
        buf, buf_rows = [], 0

    def take(res: dict[str, Any]) -> None:
        nonlocal buf_rows, last
        buf.append(res["table"])
        buf_rows += res["table"].num_rows
        hits.extend(res["hits"])
        for k in stats:
            stats[k] += res["stats"][k]
        for k in ("n_seeds", "n_fail", "n_cut", "n_rows"):
            tot[k] += res[k]
        tot["worker_seconds"] += res["seconds"]
        done_first.append(res["first"])
        if buf_rows >= part_rows:
            flush()
        now = time.perf_counter()
        if now - last > log_every_s:
            last = now
            el = now - t0
            log.info("va build: %d/%d seeds (%.1f%%), %d products, %d novel, %d hits, %.0f s "
                     "(%.1f seeds/s)", tot["n_seeds"], len(sid), 100 * tot["n_seeds"] / max(1, len(sid)),
                     tot["n_rows"], int(stats["n_novel"].sum()), len(hits), el, tot["n_seeds"] / max(el, 1e-9))

    if n_jobs <= 1:
        _init_worker(kf, holdout, cfg_d)
        for t in tasks:
            if budget_s is not None and time.perf_counter() - t0 > budget_s:
                stopped = True
                break
            take(_process_chunk(t))
    else:
        ctx = mp_context()
        with ctx.Pool(n_jobs, initializer=_init_worker, initargs=(kf, holdout, cfg_d)) as pool:
            it = pool.imap_unordered(_process_chunk, tasks, chunksize=1)
            for res in it:
                take(res)
                if budget_s is not None and time.perf_counter() - t0 > budget_s:
                    stopped = True
                    pool.terminate()
                    break
    flush()
    wall = time.perf_counter() - t0
    log.info("va build: products done, %d seeds in %.0f s; deduplicating %d part files",
             tot["n_seeds"], wall, len(parts))
    tables = [pq.read_table(p) for p in parts]
    allp = pa.concat_tables(tables) if tables else _PRODUCT_SCHEMA.empty_table()
    del tables
    allp = allp.append_column("n_seeds", pa.array(np.ones(allp.num_rows, np.int32)))
    allp = allp.rename_columns([c if c != "n_rules" else "n_pairs" for c in allp.column_names])
    allp = allp.select(list(_MERGE_SCHEMA.names)).cast(_MERGE_SCHEMA)
    ded = dedupe(allp)
    del allp
    files = write_shards(ded, out, schema=_MERGE_SCHEMA, keep_pri=True)
    for p in parts:
        p.unlink()
    try:
        part_dir.rmdir()
    except OSError:
        pass
    hit_tbl = pa.table({c: pa.array([h[i] for h in hits], type=_HIT_SCHEMA.field(c).type)
                        for i, c in enumerate(HIT_COLUMNS)}, schema=_HIT_SCHEMA)
    pq.write_table(hit_tbl, out / "holdout_hits.parquet", compression="zstd")
    done = sorted(set(done_first))
    info = {
        "n_seeds_input": int(len(sid)), "n_seeds_done": int(tot["n_seeds"]),
        "n_seeds_failed": int(tot["n_fail"]), "n_seeds_cut": int(tot["n_cut"]),
        "n_products": int(tot["n_rows"]), "n_novel_rows": int(stats["n_novel"].sum()),
        "n_unique_novel": int(ded.num_rows), "n_hits": len(hits), "wall_seconds": round(wall, 1),
        "worker_seconds": round(tot["worker_seconds"], 1), "budget_stopped": bool(stopped),
        "chunk": int(chunk), "chunks_done_first_seed": done, "n_jobs": int(n_jobs),
        "rules": names, "rule_stats": {k: [int(x) for x in v] for k, v in stats.items()},
        "n_files": len(files),
    }
    (out / "build_info.json").write_text(json.dumps(info, indent=1) + "\n")
    return info


# --------------------------------------------------------------------------------------
# Deduplication, merging, shards
# --------------------------------------------------------------------------------------

def dedupe(tbl: pa.Table) -> pa.Table:
    """One row per ik14 (``_MERGE_SCHEMA`` rows): provenance from the row with the lowest
    (seed_pri, seed_id); n_seeds and n_pairs summed. Sorted by (mono_mass, ik14)."""
    n = tbl.num_rows
    if n == 0:
        return tbl
    keys = ik14_keys(tbl.column("ik14"))
    pri = tbl.column("seed_pri").to_numpy()
    sid = tbl.column("seed_id").to_numpy()
    order = np.lexsort((sid, pri, keys))
    ks = keys[order]
    start = np.ones(n, bool)
    start[1:] = ks[1:] != ks[:-1]
    first = order[start]
    gid = np.cumsum(start) - 1
    n_seeds = np.bincount(gid, weights=tbl.column("n_seeds").to_numpy()[order]).astype(np.int32)
    n_pairs = np.bincount(gid, weights=tbl.column("n_pairs").to_numpy()[order]).astype(np.int32)
    out = tbl.take(pa.array(first))
    out = out.set_column(out.column_names.index("n_seeds"), "n_seeds", pa.array(n_seeds))
    out = out.set_column(out.column_names.index("n_pairs"), "n_pairs", pa.array(n_pairs))
    import pyarrow.compute as pc

    o2 = pc.sort_indices(out, sort_keys=[("mono_mass", "ascending"), ("ik14", "ascending")])
    return out.take(o2)


def write_shards(tbl: pa.Table, out_dir: Path | str, schema: pa.Schema = _VA_SCHEMA,
                 keep_pri: bool = False, row_group_size: int = 100_000) -> list[Path]:
    """Write rows sorted by (mono_mass, ik14) as 10 Da files ``mass_XXXX.parquet``."""
    out = paths.ensure_dir(out_dir)
    if tbl.num_rows == 0:
        return []
    cols = [c for c in schema.names if keep_pri or c != "seed_pri"]
    tbl = tbl.select(cols).cast(pa.schema([schema.field(c) for c in cols]))
    mass = tbl.column("mono_mass").to_numpy()
    b = (np.floor(mass / SHARD_DA) * SHARD_DA).astype(np.int64)
    files = []
    edges = np.flatnonzero(np.r_[True, b[1:] != b[:-1], True])
    for a, z in zip(edges[:-1], edges[1:]):
        p = out / shard_name(int(b[a]))
        pq.write_table(tbl.slice(int(a), int(z - a)), p, compression="zstd", row_group_size=row_group_size)
        files.append(p)
    return files


def merge_parts(part_dirs: Sequence[Path | str], out_dir: Path | str,
                row_filter: Any = None) -> dict[str, Any]:
    """Merge per-shard build outputs into the final tier: per 10 Da bucket, rows of every
    part directory deduplicated across shards (``dedupe``; seeds are disjoint between
    shards, so summing n_seeds is exact), optionally filtered by ``row_filter(table) ->
    bool mask``, written as ``VA_COLUMNS`` shards. Returns counts per bucket."""
    out = paths.ensure_dir(out_dir)
    buckets: dict[int, list[Path]] = {}
    for d in part_dirs:
        for p in sorted(Path(d).glob("mass_*.parquet")):
            buckets.setdefault(int(p.stem.split("_")[1]), []).append(p)
    rows, raw, kept_bytes = {}, 0, 0
    for b in sorted(buckets):
        t = pa.concat_tables([pq.read_table(p).cast(_MERGE_SCHEMA) for p in buckets[b]])
        raw += t.num_rows
        t = dedupe(t)
        if row_filter is not None and t.num_rows:
            t = t.filter(pa.array(np.asarray(row_filter(t), bool)))
        if t.num_rows:
            p = out / shard_name(b)
            pq.write_table(t.select(list(VA_COLUMNS)).cast(_VA_SCHEMA), p, compression="zstd",
                           row_group_size=100_000)
            kept_bytes += p.stat().st_size
        rows[b] = int(t.num_rows)
    return {"rows_per_bucket": rows, "n_rows": int(sum(rows.values())), "n_rows_before_merge": int(raw),
            "bytes": int(kept_bytes)}


# --------------------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------------------

class VATier:
    """Reader of the VA shards (row-group granularity, small LRU cache) and its seed table."""

    def __init__(self, directory: Path | str | None = None, max_cached_row_groups: int = 32):
        self.dir = Path(directory) if directory is not None else va_dir()
        self.max_cached = int(max_cached_row_groups)
        self._shards: dict[int, Path] | None = None
        self._meta: dict[int, tuple] = {}
        self._cache: OrderedDict = OrderedDict()
        self._seeds: dict[str, np.ndarray] | None = None
        self._index: dict[str, Any] | None = None
        self._warned = False

    def shards(self) -> dict[int, Path]:
        if self._shards is None:
            self._shards = {}
            if self.dir.is_dir():
                for p in sorted(self.dir.glob("mass_*.parquet")):
                    try:
                        self._shards[int(p.stem.split("_")[1])] = p
                    except (IndexError, ValueError):
                        continue
            if not self._shards and not self._warned:
                self._warned = True
                log.warning("VA tier not found (no mass_*.parquet in %s)", self.dir)
        return self._shards

    def available(self) -> bool:
        return bool(self.shards())

    def index(self) -> dict[str, Any]:
        if self._index is None:
            p = self.dir / "_index.json"
            self._index = json.loads(p.read_text()) if p.exists() else {}
        return self._index

    def rule_np_rate(self) -> np.ndarray:
        """Per rule code: share of its products that are known NP-tier structures."""
        st = self.index().get("rule_stats", {})
        prod = np.asarray(st.get("n_products", []), np.float64)
        npk = np.asarray(st.get("n_np", []), np.float64)
        if not len(prod):
            return np.zeros(0, np.float32)
        return np.where(prod > 0, npk / np.maximum(prod, 1), np.nan).astype(np.float32)

    def _shard_meta(self, b: int):
        m = self._meta.get(b)
        if m is None:
            path = self.shards()[b]
            md = pq.ParquetFile(path).metadata
            names = [md.schema.column(i).name for i in range(md.num_columns)]
            ci = names.index("mono_mass")
            lo, hi = [], []
            for r in range(md.num_row_groups):
                st = md.row_group(r).column(ci).statistics
                lo.append(st.min if st is not None and st.has_min_max else -np.inf)
                hi.append(st.max if st is not None and st.has_min_max else np.inf)
            m = (path, np.asarray(lo, np.float64), np.asarray(hi, np.float64))
            self._meta[b] = m
        return m

    def _row_group(self, b: int, rg: int):
        key = (b, rg)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        tbl = pq.ParquetFile(self._shard_meta(b)[0]).read_row_group(rg)
        val = (tbl, tbl.column("mono_mass").to_numpy())
        self._cache[key] = val
        while len(self._cache) > self.max_cached:
            self._cache.popitem(last=False)
        return val

    def window(self, lo: float, hi: float) -> pa.Table | None:
        """VA rows with lo <= mono_mass <= hi (``VA_COLUMNS``), or None."""
        shards = self.shards()
        parts = []
        for b in range(shard_bucket(lo), shard_bucket(hi) + SHARD_DA, SHARD_DA):
            if b not in shards:
                continue
            _, rlo, rhi = self._shard_meta(b)
            for rg in np.flatnonzero((rhi >= lo) & (rlo <= hi)):
                tbl, mass = self._row_group(b, int(rg))
                i0 = int(np.searchsorted(mass, lo, side="left"))
                i1 = int(np.searchsorted(mass, hi, side="right"))
                if i1 > i0:
                    parts.append(tbl.slice(i0, i1 - i0))
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else pa.concat_tables(parts)

    def windows(self, lo: np.ndarray, hi: np.ndarray) -> pd.DataFrame | None:
        """Rows for many windows with ``_q`` (query position), in query order."""
        frames = []
        for qi in np.argsort(np.asarray(lo), kind="stable"):
            t = self.window(float(lo[qi]), float(hi[qi]))
            if t is not None and t.num_rows:
                df = t.to_pandas()
                df.insert(0, "_q", int(qi))
                frames.append(df)
        if not frames:
            return None
        out = pd.concat(frames, ignore_index=True)
        return out.sort_values(["_q", "mono_mass", "ik14"], kind="mergesort").reset_index(drop=True)

    def seeds(self) -> dict[str, np.ndarray]:
        """Seed columns as arrays indexed by seed_id (``seeds.parquet``)."""
        if self._seeds is None:
            p = self.dir / "seeds.parquet"
            cols = ["seed_id", "ik14", "source", "np_row", "mono_mass", "in_train", "n_spectra",
                    "sid_count", "pubmed_count", "np_likeness"]
            if not p.exists():
                self._seeds = {}
                return self._seeds
            t = pq.read_table(p, columns=cols)
            sid = t.column("seed_id").to_numpy()
            n = int(sid.max()) + 1 if len(sid) else 0
            out: dict[str, np.ndarray] = {}
            for c in cols[1:]:
                v = t.column(c).to_numpy(zero_copy_only=False)
                if v.dtype == object:
                    arr = np.empty(n, dtype=object)
                else:
                    arr = np.zeros(n, dtype=v.dtype)
                arr[sid] = v
                out[c] = arr
            self._seeds = out
        return self._seeds

    def provenance(self, seed_id: np.ndarray, rule: np.ndarray, n_seeds: np.ndarray,
                   n_pairs: np.ndarray) -> dict[str, Any]:
        """``LOOKUP_COLUMNS`` values for VA rows (pandas arrays)."""
        from . import generate as gen

        s = self.seeds()
        sid = np.asarray(seed_id, np.int64)
        rl = np.asarray(rule, np.int64)
        names = np.asarray([r.id for r in gen.RULES], dtype=object)
        rate = self.rule_np_rate()
        n = len(sid)

        def pick(c, dtype, na=None):
            if c not in s or not n:
                return pd.array([na] * n, dtype=dtype)
            v = s[c][np.clip(sid, 0, len(s[c]) - 1)]
            if dtype in ("Int32", "Int64"):
                v = pd.array(v.astype(np.int64), dtype=dtype)
                return v
            return pd.array(v, dtype=dtype)

        sidc = pick("sid_count", "Int32")
        pmc = pick("pubmed_count", "Int32")
        if n:
            sidc[np.asarray(sidc.fillna(-1)) < 0] = pd.NA
            pmc[np.asarray(pmc.fillna(-1)) < 0] = pd.NA
        return {
            "va_seed_id": pd.array(sid, dtype="Int32"), "va_rule": pd.array(rl, dtype="Int16"),
            "va_rule_name": pd.array(names[np.clip(rl, 0, len(names) - 1)] if n else [], dtype="string"),
            "va_n_seeds": pd.array(np.asarray(n_seeds, np.int64), dtype="Int32"),
            "va_n_pairs": pd.array(np.asarray(n_pairs, np.int64), dtype="Int32"),
            "va_rule_np_rate": pd.array(rate[np.clip(rl, 0, len(rate) - 1)] if len(rate) and n
                                        else [np.nan] * n, dtype="Float32"),
            "va_seed_tier": pick("source", "string"), "va_seed_ik14": pick("ik14", "string"),
            "va_seed_np_row": pick("np_row", "Int64"), "va_seed_mass": pick("mono_mass", "Float64"),
            "va_seed_in_train": pick("in_train", "boolean"),
            "va_seed_n_spectra": pick("n_spectra", "Int32"), "va_seed_sid_count": sidc,
            "va_seed_pubmed_count": pmc, "va_seed_np_likeness": pick("np_likeness", "Float32"),
        }

    def release(self) -> None:
        self._cache.clear()
        self._meta.clear()
        self._shards = None


# --------------------------------------------------------------------------------------
# Validation: seed masks, reach
# --------------------------------------------------------------------------------------

def seed_mask(seed_ik14: np.ndarray, excluded: Iterable[str]) -> np.ndarray:
    """True for seeds that stay allowed (ik14 not in ``excluded``)."""
    ex = ik14_keys(list(excluded))
    ex = sorted_keys(ex)
    return ~in_sorted(ik14_keys(pa.array(list(seed_ik14), type=pa.string())
                                if not isinstance(seed_ik14, (pa.Array, pa.ChunkedArray)) else seed_ik14), ex)


def _gen_truth(task: tuple) -> tuple[str, bool, int, int, str]:
    """GEN reach of one molecule: (molecule_id, hit, n products, n seeds used, edits)."""
    from .chem import mol_from_smiles
    from .generate import generate
    from .metric import mkey_mol

    mid, seeds, formula, truth_ik, mkey, skel, cfg_d, max_edits = task
    cfg = Config(**cfg_d) if cfg_d else Config()
    if not seeds or not isinstance(formula, str):
        return mid, False, 0, len(seeds), ""
    g = generate(list(seeds), [formula], cfg, max_edits=max_edits)
    if not len(g):
        return mid, False, 0, len(seeds), ""
    hit = g[g["ik14"].isin(set(truth_ik))]
    if len(hit):
        return mid, True, len(g), len(seeds), str(hit["edits"].iloc[0])
    if skel:
        for smi, ed in zip(g["smiles"], g["edits"]):
            m = mol_from_smiles(smi)
            if m is not None and _skeleton(m) == skel and mkey_mol(m) == mkey:
                return mid, True, len(g), len(seeds), str(ed)
    return mid, False, len(g), len(seeds), ""


def reach_table(molecules: pd.DataFrame, excluded: Mapping[str, Iterable[str]], hits: pd.DataFrame,
                seeds: pd.DataFrame, *, gen_seeds: Mapping[str, Sequence[str]] | None = None,
                truth_ik14: Mapping[str, Iterable[str]] | None = None, cfg: Config | None = None,
                n_jobs: int = 1, max_edits: int = 2) -> pd.DataFrame:
    """Reach of the VA tier (+ GEN) per holdout molecule (module docstring).

    ``molecules``: molecule_id, mkey, smiles, formula, mono_mass (+ ik14_group);
    ``excluded``: molecule_id -> ik14 removed from every tier and from the seeds (C3e: the
    truth keys; C3h: plus every structure with Tanimoto >= 0.75); ``hits``: the build's
    holdout hit table (``HIT_COLUMNS``); ``seeds``: the seed table (seed_id, ik14);
    ``gen_seeds``: molecule_id -> SMILES of the best seeds for the on-demand second edit
    (already filtered by the caller or filtered here by ``excluded`` when given as
    (ik14, smiles) pairs); ``truth_ik14``: molecule_id -> ik14 that are the truth (default:
    ik14_group plus every hit with the molecule's mkey). Columns: molecule_id, va_reach,
    va_n_seeds (allowed seeds producing the truth), va_rule (rule code of the preferred one),
    va_n_seeds_all (before the mask), gen_reach, gen_n_products, gen_n_seeds, gen_edits,
    reach."""
    cfg = cfg or Config()
    sid2ik = dict(zip(seeds["seed_id"].astype(int), seeds["ik14"].astype(str)))
    sid2pri = dict(zip(seeds["seed_id"].astype(int), seeds["pri"].astype(int))) if "pri" in seeds else {}
    h = hits.copy()
    h["seed_ik14"] = [sid2ik.get(int(s), "") for s in h["seed_id"]]
    by_mkey = {k: g for k, g in h.groupby(h["mkey"].astype(str))}
    by_ik = {k: g for k, g in h.groupby(h["ik14"].astype(str))}
    rows, tasks = [], []
    cfg_d = cfg.to_dict()
    from .validation import skeleton_key

    for r in molecules.itertuples(index=False):
        mid = str(r.molecule_id)
        tset = set(truth_ik14.get(mid, ())) if truth_ik14 is not None else set()
        grp = getattr(r, "ik14_group", None)
        if isinstance(grp, str):
            tset |= set(grp.split(";"))
        tset.add(str(getattr(r, "ik14", "")))
        parts = [by_mkey[str(r.mkey)]] if str(r.mkey) in by_mkey else []
        parts += [by_ik[k] for k in sorted(tset) if k in by_ik]
        cand = pd.concat(parts, ignore_index=True).drop_duplicates(["seed_id", "ik14"]) if parts else h.iloc[:0]
        ex = set(map(str, excluded.get(mid, ())))
        allowed = cand[~cand["seed_ik14"].isin(ex)]
        best_rule = -1
        if len(allowed):
            pr = [sid2pri.get(int(s), int(s)) for s in allowed["seed_id"]]
            best_rule = int(allowed["rule"].to_numpy()[int(np.argmin(pr))])
        rows.append({"molecule_id": mid, "va_reach": bool(len(allowed)),
                     "va_n_seeds": int(allowed["seed_id"].nunique()),
                     "va_n_seeds_all": int(cand["seed_id"].nunique()), "va_rule": best_rule})
        if gen_seeds is not None:
            gs = gen_seeds.get(mid, [])
            sm = []
            for s in gs:
                if isinstance(s, tuple):
                    if str(s[0]) in ex:
                        continue
                    s = s[1]
                sm.append(s)
            sm = sm[:int(cfg.va_gen_n_seeds)]
            tasks.append((mid, sm, r.formula, sorted(tset), str(r.mkey), skeleton_key(r.smiles),
                          cfg_d, int(max_edits)))
    out = pd.DataFrame(rows)
    if gen_seeds is not None:
        from .chem import parallel_map

        res = parallel_map(_gen_chunk, [tasks[i:i + 8] for i in range(0, len(tasks), 8)], n_jobs)
        g = pd.DataFrame([x for part in res for x in part],
                         columns=["molecule_id", "gen_reach", "gen_n_products", "gen_n_seeds", "gen_edits"])
        out = out.merge(g, on="molecule_id", how="left")
    else:
        out["gen_reach"] = False
        out["gen_n_products"] = 0
        out["gen_n_seeds"] = 0
        out["gen_edits"] = ""
    out["gen_reach"] = out["gen_reach"].fillna(False).astype(bool)
    out["reach"] = out["va_reach"] | out["gen_reach"]
    return out


def _gen_chunk(tasks: list[tuple]) -> list[tuple]:
    return [_gen_truth(t) for t in tasks]


def window_counts(tier: VATier, masses: np.ndarray, cfg: Config | None = None) -> np.ndarray:
    """VA rows inside each mass window (``cfg.mass_tol_ppm``, at least ``mass_tol_min_da``)."""
    cfg = cfg or Config()
    m = np.asarray(masses, np.float64)
    half = np.maximum(cfg.mass_tol_ppm * 1e-6 * m, cfg.mass_tol_min_da)
    out = np.zeros(len(m), np.int64)
    for i in np.argsort(m, kind="stable"):
        t = tier.window(float(m[i] - half[i]), float(m[i] + half[i]))
        out[i] = 0 if t is None else t.num_rows
    return out


def summarize_reach(tab: pd.DataFrame) -> dict[str, Any]:
    """Shares of molecules reached (VA, GEN, either) and window statistics when present."""
    n = len(tab)
    out: dict[str, Any] = {"n": int(n)}
    for c in ("va_reach", "gen_reach", "reach"):
        if c in tab.columns and n:
            out[c] = round(float(tab[c].mean()), 4)
    if "va_window" in tab.columns and n:
        w = tab["va_window"].to_numpy(np.float64)
        out["va_window_median"] = float(np.median(w))
        out["va_window_p95"] = float(np.percentile(w, 95))
        out["va_window_mean"] = round(float(w.mean()), 1)
    if "gen_n_products" in tab.columns and n:
        g = tab["gen_n_products"].fillna(0).to_numpy(np.float64)
        out["gen_products_median"] = float(np.median(g))
        out["gen_products_p95"] = float(np.percentile(g, 95))
    return out


def worker_count(n_jobs: int | None) -> int:
    """``n_jobs`` or all CPUs of this machine (cgroup quota not considered)."""
    return int(n_jobs) if n_jobs else (os.cpu_count() or 1)

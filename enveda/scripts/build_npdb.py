#!/usr/bin/env python3
"""WP2 - build the natural-product (NP) candidate tier.

Output (docs/PLAN.md 2.3), in ``casmi.paths.PROCESSED / db / np``:

- ``np.parquet``: one row per ik14, sorted by (mono_mass, ik14). Union of COCONUT,
  NPAtlas, LOTUS and PubChemLite after the filters of PLAN 2.2, SMILES from the
  best source (COCONUT > NPAtlas > LOTUS > PubChemLite), formula and mono_mass
  recomputed with RDKit, membership flags and priors.
- ``np_fp.npy``: uint8 array (n_rows, 512), Morgan radius 2, 4096 bits, packed with
  ``np.packbits`` (bit order "big"), row-aligned to np.parquet. Exact recipe in
  ``morgan_packed`` below and in docs/ARTEFACTS.md.
- ``build_report.json``: every count, check and checksum of the build.
- ``mass_disagreements.csv``: rows whose RDKit mass differs from the source mass by
  more than 0.001 Da.

Stages (``--stage``, default ``all``):

  download  fetch the four sources into data/external/<source>/ (skip if present,
            md5 checked where the publisher gives one)
  headers   print each source's header and fail loudly on a missing expected column
  table     read, filter, RDKit formula/mass, dedup, priors -> np.parquet
  fp        fingerprints for np.parquet rows -> np_fp.npy (re-runnable on its own)
  check     acceptance checks (row count, ik14 unique, mass agreement, quercetin,
            lookup speed, public-test pool sizes)

Usage:
    nice -n 10 python -I scripts/build_npdb.py                       # everything
    nice -n 10 python -I scripts/build_npdb.py --stage fp            # redo fingerprints
    python -I scripts/build_npdb.py --limit 20000 --out <scratch dir>  # quick dry run

Resources: at most ``--workers`` (default 2) RDKit worker processes, peak RAM about 3 GB.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import multiprocessing as mp
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from casmi import paths
from casmi.config import Config
from casmi.log import get_logger, log_run

log = logging.getLogger("build_npdb")

# ----------------------------------------------------------------------------- sources

SOURCES: dict[str, dict] = {
    "coconut": {
        "file": "coconut/coconut_csv-11-2026.zip",
        "url": "https://coconut.s3.uni-jena.de/prod/downloads/2026-11/coconut_csv-11-2026.zip",
        "md5": None,
        "member": "coconut_csv-11-2026.csv",
        "sep": ",",
        "required": ["identifier", "canonical_smiles", "standard_inchi_key", "annotation_level",
                     "exact_molecular_weight", "molecular_formula", "formal_charge", "np_likeness",
                     "organisms", "dois", "collections"],
    },
    "npatlas": {
        "file": "npatlas/NPAtlas_download.tsv",
        "url": "https://www.npatlas.org/static/downloads/NPAtlas_download.tsv",
        "md5": None,
        "sep": "\t",
        "required": ["npaid", "compound_inchikey", "compound_smiles", "compound_accurate_mass",
                     "compound_molecular_formula"],
    },
    "lotus": {
        "file": "lotus/structure_metadata.tsv.gz",
        "url": "https://zenodo.org/api/records/7534077/files/structure_metadata.tsv.gz/content",
        "md5": "a791309a82df1d7ac538624f8f7856dc",
        "sep": "\t",
        "required": ["structureCleanedSmiles", "structureCleanedInchikey", "structureCleaned_exactMass",
                     "structureCleaned_molecularFormula", "structureCleaned_cid"],
    },
    "pubchemlite": {
        "file": "pubchemlite/PubChemLite_CCSbase_20260925.csv",
        "url": "https://zenodo.org/api/records/22953182/files/PubChemLite_CCSbase_20260925.csv/content",
        "md5": "fd1629808c2cf2896acc502e26e352f0",
        "sep": ",",
        "required": ["Identifier", "FirstBlock", "InChIKey", "SMILES", "MonoisotopicMass",
                     "MolecularFormula", "PubMed_Count", "Patent_Count", "AnnoTypeCount"],
    },
}
# SMILES preference across sources (PLAN 2.3: COCONUT preferred).
SOURCE_ORDER = ["coconut", "npatlas", "lotus", "pubchemlite"]

# Public-test adduct shifts (PLAN 3.1): precursor_mz = M + shift.
ADDUCT_SHIFT = {
    "[M+H]+": 1.007276, "[M+NH4]+": 18.033826, "[M-H2O+H]+": -17.003288,
    "[M-2H2O+H]+": -35.013853, "[M+Na]+": 22.989221, "[M+K]+": 38.963158,
    "[M-H]-": -1.007276, "[M-H2O-H]-": -19.017841, "[M+CH2O2-H]-": 44.998203,
    "[M+Cl]-": 34.969401,
}
QUERCETIN_IK14 = "REFJWTPEDVJJIY"
QUERCETIN_M = 302.0427
MASS_AGREE_DA = 0.001

IK_FULL_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
IK14_RE = re.compile(r"^[A-Z]{14}$")
ISOTOPE_SMILES_RE = r"\[[0-9]+[A-Za-z*]"   # [13C], [2H], [18O]... (PLAN 2.2)

OUT_SCHEMA = pa.schema([
    ("ik14", pa.string()), ("smiles", pa.string()), ("formula", pa.string()),
    ("mono_mass", pa.float64()),
    ("in_coconut", pa.bool_()), ("in_npatlas", pa.bool_()), ("in_lotus", pa.bool_()),
    ("in_pubchemlite", pa.bool_()),
    ("coconut_annotation_level", pa.int8()), ("coconut_np_likeness", pa.float32()),
    ("n_organisms", pa.int32()), ("n_citations", pa.int32()), ("n_collections", pa.int32()),
    ("pubmed_count", pa.int32()), ("patent_count", pa.int32()), ("anno_type_count", pa.int32()),
    ("cid", pa.int64()),
    ("smiles_source", pa.string()), ("coconut_id", pa.string()), ("npaid", pa.string()),
])


class Ctx:
    def __init__(self, args: argparse.Namespace, cfg: Config):
        self.args = args
        self.cfg = cfg
        self.ext = paths.EXTERNAL
        self.out = Path(args.out) if args.out else paths.PROCESSED / "db" / "np"
        self.workers = max(1, int(args.workers))
        self.limit = args.limit

    def src_path(self, name: str) -> Path:
        return self.ext / SOURCES[name]["file"]

    @property
    def np_parquet(self) -> Path:
        return self.out / "np.parquet"

    @property
    def np_fp(self) -> Path:
        return self.out / "np_fp.npy"

    @property
    def report_path(self) -> Path:
        return self.out / "build_report.json"

    def load_report(self) -> dict:
        if self.report_path.exists():
            return json.loads(self.report_path.read_text())
        return {}

    def save_report(self, rep: dict) -> None:
        paths.ensure_dir(self.out)
        tmp = self.report_path.with_suffix(".json.part")
        tmp.write_text(json.dumps(rep, indent=1, default=_json_default))
        tmp.replace(self.report_path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def sha256_file(p: Path, block: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def md5_file(p: Path, block: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- download

def stage_download(ctx: Ctx) -> None:
    for name in SOURCES:
        spec = SOURCES[name]
        dest = ctx.src_path(name)
        paths.ensure_dir(dest.parent)
        if dest.exists() and dest.stat().st_size > 0:
            if spec["md5"] and md5_file(dest) != spec["md5"]:
                log.warning("download %s: md5 mismatch, downloading again", name)
                dest.unlink()
            else:
                log.info("download %s: present (%s, %.1f MB)", name, dest, dest.stat().st_size / 1e6)
                continue
        log.info("download %s: %s -> %s", name, spec["url"], dest)
        part = dest.with_name(dest.name + ".part")
        cmd = ["curl", "-sS", "--fail", "-L", "--retry", "5", "--retry-delay", "10", "-C", "-",
               "-o", str(part), spec["url"]]
        subprocess.run(cmd, check=True)
        if spec["md5"]:
            got = md5_file(part)
            if got != spec["md5"]:
                raise SystemExit(f"download {name}: md5 {got} != expected {spec['md5']}")
        part.replace(dest)
        log.info("download %s: ok, %.1f MB", name, dest.stat().st_size / 1e6)


# ----------------------------------------------------------------------------- headers

def _read_header(ctx: Ctx, name: str) -> list[str]:
    spec = SOURCES[name]
    p = ctx.src_path(name)
    if not p.exists():
        raise SystemExit(f"source {name}: file missing: {p} (run --stage download)")
    if p.suffix == ".zip":
        import io
        import zipfile
        with zipfile.ZipFile(p) as z:
            members = z.namelist()
            if spec["member"] not in members:
                raise SystemExit(f"source {name}: member {spec['member']!r} not in zip ({members})")
            with z.open(spec["member"]) as fh:
                line = io.TextIOWrapper(fh, encoding="utf-8", newline="").readline()
    elif p.suffix == ".gz":
        import gzip
        with gzip.open(p, "rt", encoding="utf-8", newline="") as fh:
            line = fh.readline()
    else:
        with open(p, encoding="utf-8", newline="") as fh:
            line = fh.readline()
    return next(csv.reader([line.rstrip("\r\n")], delimiter=spec["sep"]))


def stage_headers(ctx: Ctx) -> dict[str, list[str]]:
    found = {}
    problems = []
    for name in SOURCES:
        cols = _read_header(ctx, name)
        found[name] = cols
        print(f"[{name}] {len(cols)} columns: {cols}", flush=True)
        log.info("header %s (%d columns): %s", name, len(cols), cols)
        missing = [c for c in SOURCES[name]["required"] if c not in cols]
        if missing:
            problems.append(f"{name}: missing expected column(s) {missing}")
    if problems:
        for p in problems:
            log.error("HEADER CHECK FAILED: %s", p)
        raise SystemExit("header check failed: " + "; ".join(problems))
    rep = ctx.load_report()
    rep["columns_found"] = found
    ctx.save_report(rep)
    return found


# ----------------------------------------------------------------------------- readers
# Each reader returns a frame with the common columns
#   src_row (int), src_id (str), smiles (str), ik_full (str), ik14 (str or ""),
#   src_mass (float), src_formula (str)
# plus source-specific prior columns, and (COCONUT) the exploded organism / doi /
# collection lists for set-union aggregation over stereoisomers.

def _ik14_from_full(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.strip()
    ok = s.str.match(IK_FULL_RE.pattern)
    return s.str.slice(0, 14).where(ok, "")


def _num(s: pd.Series, kind: str = "float") -> pd.Series:
    v = pd.to_numeric(s.replace("", np.nan), errors="coerce")
    return v.astype("float64") if kind == "float" else v


def _split_lists(ids: pd.Series, text: pd.Series) -> pd.DataFrame:
    """(src_row, item) pairs from '|'-separated text, items stripped, empty dropped."""
    lists = text.fillna("").astype(str).str.split("|")
    ex = pd.DataFrame({"src_row": ids.to_numpy(), "item": lists}).explode("item")
    ex["item"] = ex["item"].astype(str).str.strip()
    ex = ex[ex["item"].str.len() > 0]
    return ex.reset_index(drop=True)


def read_coconut(ctx: Ctx):
    spec = SOURCES["coconut"]
    usecols = spec["required"]
    frames, orgs, dois, colls = [], [], [], []
    reader = pd.read_csv(ctx.src_path("coconut"), compression="zip", sep=",", usecols=usecols,
                         dtype=str, keep_default_na=False, chunksize=100_000, nrows=ctx.limit)
    start = 0
    for ch in reader:
        ch = ch.reset_index(drop=True)
        ids = pd.Series(np.arange(start, start + len(ch)), dtype="int64")
        start += len(ch)
        orgs.append(_split_lists(ids, ch["organisms"]))
        dois.append(_split_lists(ids, ch["dois"]))
        colls.append(_split_lists(ids, ch["collections"]))
        frames.append(pd.DataFrame({
            "src_row": ids.to_numpy(),
            "src_id": ch["identifier"].astype(str),
            "smiles": ch["canonical_smiles"].astype(str).str.strip(),
            "ik_full": ch["standard_inchi_key"].astype(str),
            "src_mass": _num(ch["exact_molecular_weight"]),
            "src_formula": ch["molecular_formula"].astype(str),
            "src_charge": _num(ch["formal_charge"]),
            "annotation_level": _num(ch["annotation_level"]),
            "np_likeness": _num(ch["np_likeness"]),
        }))
    df = pd.concat(frames, ignore_index=True)
    df["ik14"] = _ik14_from_full(df["ik_full"])
    lists = {"organisms": pd.concat(orgs, ignore_index=True),
             "dois": pd.concat(dois, ignore_index=True),
             "collections": pd.concat(colls, ignore_index=True)}
    return df, lists


def read_npatlas(ctx: Ctx):
    spec = SOURCES["npatlas"]
    ch = pd.read_csv(ctx.src_path("npatlas"), sep="\t", usecols=spec["required"], dtype=str,
                     keep_default_na=False, quoting=csv.QUOTE_NONE, nrows=ctx.limit)
    df = pd.DataFrame({
        "src_row": np.arange(len(ch), dtype="int64"),
        "src_id": ch["npaid"].astype(str),
        "smiles": ch["compound_smiles"].astype(str).str.strip(),
        "ik_full": ch["compound_inchikey"].astype(str),
        "src_mass": _num(ch["compound_accurate_mass"]),
        "src_formula": ch["compound_molecular_formula"].astype(str),
    })
    df["ik14"] = _ik14_from_full(df["ik_full"])
    return df, {}


def read_lotus(ctx: Ctx):
    spec = SOURCES["lotus"]
    ch = pd.read_csv(ctx.src_path("lotus"), sep="\t", usecols=spec["required"], dtype=str,
                     keep_default_na=False, quoting=csv.QUOTE_NONE, nrows=ctx.limit)
    df = pd.DataFrame({
        "src_row": np.arange(len(ch), dtype="int64"),
        "src_id": ch["structureCleanedInchikey"].astype(str),
        "smiles": ch["structureCleanedSmiles"].astype(str).str.strip(),
        "ik_full": ch["structureCleanedInchikey"].astype(str),
        "src_mass": _num(ch["structureCleaned_exactMass"]),
        "src_formula": ch["structureCleaned_molecularFormula"].astype(str),
        "cid": _num(ch["structureCleaned_cid"]),
    })
    df["ik14"] = _ik14_from_full(df["ik_full"])
    return df, {}


def read_pubchemlite(ctx: Ctx):
    spec = SOURCES["pubchemlite"]
    ch = pd.read_csv(ctx.src_path("pubchemlite"), sep=",", usecols=spec["required"], dtype=str,
                     keep_default_na=False, nrows=ctx.limit)
    fb = ch["FirstBlock"].astype(str).str.strip()
    ik14_full = _ik14_from_full(ch["InChIKey"])
    fb_ok = fb.str.match(IK14_RE.pattern)
    n_conflict = int(((ik14_full != "") & fb_ok & (ik14_full != fb)).sum())
    if n_conflict:
        log.warning("pubchemlite: %d rows where FirstBlock != InChIKey[:14] (FirstBlock used)", n_conflict)
    df = pd.DataFrame({
        "src_row": np.arange(len(ch), dtype="int64"),
        "src_id": ch["Identifier"].astype(str),
        "smiles": ch["SMILES"].astype(str).str.strip(),
        "ik_full": ch["InChIKey"].astype(str),
        "src_mass": _num(ch["MonoisotopicMass"]),
        "src_formula": ch["MolecularFormula"].astype(str),
        "cid": _num(ch["Identifier"]),
        "pubmed_count": _num(ch["PubMed_Count"]),
        "patent_count": _num(ch["Patent_Count"]),
        "anno_type_count": _num(ch["AnnoTypeCount"]),
    })
    df["ik14"] = fb.where(fb_ok, ik14_full)
    return df, {}


READERS = {"coconut": read_coconut, "npatlas": read_npatlas, "lotus": read_lotus,
           "pubchemlite": read_pubchemlite}


# ----------------------------------------------------------------------------- RDKit workers

_FP_GEN = None


def _worker_init() -> None:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")


def props_chunk(args):
    """RDKit properties for a chunk of SMILES.

    Returns (ok, n_iso_atoms, charge, elems_ok, n_carbon, formula, mono_mass, inchikey) lists.
    ``elements`` is the allowed element set; ``want_key`` flags rows needing an InChIKey.
    """
    smiles, want_key, elements = args
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    try:  # WP1 helpers when present (same RDKit calls on the molecule as given)
        from casmi.chem import formula as _formula, mono_mass as _mono_mass
    except Exception:  # noqa: BLE001
        _formula, _mono_mass = rdMolDescriptors.CalcMolFormula, Descriptors.ExactMolWt

    allowed = set(elements)
    ok, iso, chg, eok, nc, form, mass, keys = [], [], [], [], [], [], [], []
    for smi, wk in zip(smiles, want_key):
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            ok.append(False); iso.append(0); chg.append(0); eok.append(False); nc.append(0)
            form.append(""); mass.append(np.nan); keys.append("")
            continue
        n_iso = 0
        n_c = 0
        e_ok = True
        for a in mol.GetAtoms():
            if a.GetIsotope():
                n_iso += 1
            z = a.GetAtomicNum()
            if z == 6:
                n_c += 1
            if a.GetSymbol() not in allowed:
                e_ok = False
        ok.append(True)
        iso.append(n_iso)
        chg.append(Chem.GetFormalCharge(mol))
        eok.append(e_ok)
        nc.append(n_c)
        form.append(_formula(mol))
        mass.append(float(_mono_mass(mol)))
        key = ""
        if wk:
            try:
                key = Chem.MolToInchiKey(mol) or ""
            except Exception:  # noqa: BLE001 - InChI failures only lose the key
                key = ""
        keys.append(key)
    return ok, iso, chg, eok, nc, form, mass, keys


def morgan_packed(mol, radius: int, n_bits: int) -> np.ndarray:
    """Packed Morgan bits of one RDKit Mol (the exact recipe of np_fp.npy).

    ``rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)`` with all
    other options at their defaults (includeChirality=False, useBondTypes=True, no count
    simulation, standard connectivity invariants), ``GetFingerprintAsNumPy(mol)`` (uint8
    0/1 vector), then ``np.packbits(bits)`` (bitorder "big": bit j -> byte j // 8, mask
    0x80 >> (j % 8)). ``mol`` is ``Chem.MolFromSmiles(smiles)`` of the stored SMILES,
    sanitised, no explicit hydrogens, no further standardisation.
    """
    global _FP_GEN
    if _FP_GEN is None or _FP_GEN[0] != (radius, n_bits):
        from rdkit.Chem import rdFingerprintGenerator
        _FP_GEN = ((radius, n_bits), rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits))
    bits = _FP_GEN[1].GetFingerprintAsNumPy(mol)
    return np.packbits(bits.astype(np.uint8, copy=False))


def _chem():
    try:
        from casmi import chem
        if hasattr(chem, "std_mol") and hasattr(chem, "morgan_fp"):
            return chem
    except Exception:  # noqa: BLE001
        pass
    return None


def fp_chunk(args):
    """Packed fingerprints of a chunk of stored SMILES.

    With WP1's ``casmi.chem``: ``chem.morgan_fp(chem.std_mol(smiles), cfg)`` (the same
    as ``chem.morgan_fps(..., standardize=True)``), falling back to the unstandardised
    molecule where ``std_mol`` fails. Without it: ``morgan_packed`` on the parsed SMILES.
    Returns (packed array, rows that failed to parse, rows where std_mol failed).
    """
    smiles, radius, n_bits, use_chem = args
    from rdkit import Chem

    chem = _chem() if use_chem else None
    cfg = None
    if chem is not None:
        cfg = Config().replace(fp_morgan_radius=radius, fp_morgan_bits=n_bits)
    out = np.zeros((len(smiles), n_bits // 8), dtype=np.uint8)
    bad, std_fail = [], []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            bad.append(i)
            continue
        if chem is not None:
            sm = chem.std_mol(mol)
            if sm is None:
                std_fail.append(i)
                sm = mol
            out[i] = chem.morgan_fp(sm, cfg, packed=True)
        else:
            out[i] = morgan_packed(mol, radius, n_bits)
    return out, bad, std_fail


def _pool(ctx: Ctx):
    return mp.get_context("spawn").Pool(ctx.workers, initializer=_worker_init)


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield i, seq[i:i + size]


# ----------------------------------------------------------------------------- table

def _cascade(rep: dict, src: str, step: str, n: int) -> None:
    rep.setdefault("filter_cascade", {}).setdefault(src, []).append([step, int(n)])
    log.info("filter %-12s %-28s %9d rows", src, step, n)


def _dedup_lists(ex: pd.DataFrame, row_to_ik: pd.Series, name: str) -> pd.Series:
    """Number of distinct items per ik14 (set union over the source's rows)."""
    if ex.empty:
        return pd.Series(dtype="int64", name=name)
    ex = ex.assign(ik14=row_to_ik.reindex(ex["src_row"].to_numpy()).to_numpy())
    ex = ex[ex["ik14"].notna() & (ex["ik14"] != "")]
    return ex.drop_duplicates(["ik14", "item"]).groupby("ik14").size().rename(name)


def stage_table(ctx: Ctx) -> None:
    cfg = ctx.cfg
    t0 = time.time()
    rep = ctx.load_report()
    rep.pop("filter_cascade", None)
    rep["config"] = {"db_mass_min": cfg.db_mass_min, "db_mass_max": cfg.db_mass_max,
                     "db_elements": list(cfg.db_elements), "db_min_carbons": cfg.db_min_carbons}
    rep["limit"] = ctx.limit
    raw = {}
    lists = {}
    for name in SOURCE_ORDER:
        ts = time.time()
        df, ls = READERS[name](ctx)
        df.insert(0, "src", name)
        raw[name] = df
        lists[name] = ls
        _cascade(rep, name, "rows_read", len(df))
        n_noik = int((df["ik14"] == "").sum())
        log.info("read %s: %d rows, %d without a valid source InChIKey, %.1f s",
                 name, len(df), n_noik, time.time() - ts)
        rep.setdefault("rows_without_source_ik", {})[name] = n_noik

    # String-level filters (PLAN 2.2) before RDKit.
    keep_str = {}
    for name in SOURCE_ORDER:
        df = raw[name]
        m = df["smiles"].str.len() > 0
        _cascade(rep, name, "smiles_nonempty", m.sum())
        m &= ~df["smiles"].str.contains(".", regex=False)
        _cascade(rep, name, "single_component", m.sum())
        m &= ~df["smiles"].str.contains(ISOTOPE_SMILES_RE, regex=True)
        _cascade(rep, name, "no_isotope_label_smiles", m.sum())
        keep_str[name] = m.to_numpy()

    # RDKit properties once per distinct SMILES string.
    allsmi = pd.concat([raw[n].loc[keep_str[n], ["smiles", "ik14"]] for n in SOURCE_ORDER],
                       ignore_index=True)
    need_key = allsmi.loc[allsmi["ik14"] == "", "smiles"]
    uniq = np.sort(np.asarray(allsmi["smiles"].unique(), dtype=object))
    want = pd.Index(uniq).isin(pd.Index(need_key.unique())).astype(bool)
    log.info("RDKit pass: %d distinct SMILES (%d need an InChIKey), %d workers",
             len(uniq), int(want.sum()), ctx.workers)
    ts = time.time()
    res = {k: [] for k in ("ok", "iso", "chg", "eok", "nc", "form", "mass", "key")}
    chunk = 5000
    tasks = [(list(c), list(want[i:i + chunk]), tuple(cfg.db_elements)) for i, c in _chunks(uniq, chunk)]
    with _pool(ctx) as pool:
        for j, out in enumerate(pool.imap(props_chunk, tasks)):
            for k, v in zip(res, out):
                res[k].extend(v)
            if (j + 1) % 40 == 0:
                done = min((j + 1) * chunk, len(uniq))
                log.info("RDKit pass: %d / %d (%.0f s)", done, len(uniq), time.time() - ts)
    props = pd.DataFrame({
        "smiles": uniq,
        "rd_ok": np.array(res["ok"], dtype=bool),
        "rd_iso": np.array(res["iso"], dtype=np.int32),
        "rd_charge": np.array(res["chg"], dtype=np.int32),
        "rd_elems_ok": np.array(res["eok"], dtype=bool),
        "rd_nc": np.array(res["nc"], dtype=np.int32),
        "formula": np.array(res["form"], dtype=object),
        "mono_mass": np.array(res["mass"], dtype=np.float64),
        "rd_key": np.array(res["key"], dtype=object),
    }).set_index("smiles")
    del res, tasks
    log.info("RDKit pass done: %d distinct SMILES in %.0f s (%d failed to parse)",
             len(props), time.time() - ts, int((~props["rd_ok"]).sum()))
    rep["rdkit_distinct_smiles"] = len(props)
    rep["rdkit_parse_failures"] = int((~props["rd_ok"]).sum())

    # Apply RDKit filters, fill missing ik14, pick per-source winners.
    winners = []
    agg = {}
    for name in SOURCE_ORDER:
        df = raw[name]
        p = props.reindex(df["smiles"].to_numpy())
        p.index = df.index
        for c in ("rd_ok", "rd_elems_ok"):
            p[c] = p[c].fillna(False).astype(bool)
        m = keep_str[name] & p["rd_ok"].to_numpy()
        _cascade(rep, name, "rdkit_parses", m.sum())
        m &= (p["rd_iso"].fillna(1).to_numpy() == 0)
        _cascade(rep, name, "no_isotope_atoms", m.sum())
        m &= (p["rd_charge"].fillna(1).to_numpy() == 0)
        _cascade(rep, name, "net_charge_zero", m.sum())
        m &= p["rd_elems_ok"].to_numpy()
        _cascade(rep, name, "elements_allowed", m.sum())
        m &= (p["rd_nc"].fillna(0).to_numpy() >= cfg.db_min_carbons)
        _cascade(rep, name, f"carbons_ge_{cfg.db_min_carbons}", m.sum())
        mm = p["mono_mass"].to_numpy()
        m &= (mm >= cfg.db_mass_min) & (mm <= cfg.db_mass_max)
        _cascade(rep, name, f"mass_{cfg.db_mass_min:g}_{cfg.db_mass_max:g}", m.sum())

        fill = (df["ik14"] == "") & p["rd_key"].fillna("").astype(str).str.match(IK_FULL_RE.pattern)
        n_fill = int(fill.sum())
        df.loc[fill, "ik14"] = p.loc[fill, "rd_key"].astype(str).str.slice(0, 14)
        m &= (df["ik14"] != "").to_numpy()
        _cascade(rep, name, "has_ik14", m.sum())
        rep.setdefault("ik14_filled_by_rdkit", {})[name] = n_fill

        # Aggregates over every row of the source with an ik14 (pre-filter): membership
        # flags and priors describe the compound in that source, whatever row passed.
        a = df[df["ik14"] != ""]
        g = a.groupby("ik14", sort=True)
        ag = pd.DataFrame(index=g.size().index)
        if name == "coconut":
            ag["coconut_annotation_level"] = g["annotation_level"].max()
            ag["coconut_np_likeness"] = g["np_likeness"].max()
            row_to_ik = df.set_index("src_row")["ik14"]
            ag["n_organisms"] = _dedup_lists(lists[name]["organisms"], row_to_ik, "n").reindex(ag.index).fillna(0)
            ag["n_citations"] = _dedup_lists(lists[name]["dois"], row_to_ik, "n").reindex(ag.index).fillna(0)
            ag["n_collections"] = _dedup_lists(lists[name]["collections"], row_to_ik, "n").reindex(ag.index).fillna(0)
        if name == "pubchemlite":
            ag["pubmed_count"] = g["pubmed_count"].max()
            ag["patent_count"] = g["patent_count"].max()
            ag["anno_type_count"] = g["anno_type_count"].max()
        if "cid" in a.columns:
            ag[f"cid_{name}"] = g["cid"].min()
        agg[name] = ag
        rep.setdefault("distinct_ik14_all_rows", {})[name] = len(ag)

        w = df.loc[m].copy()
        w["formula"] = p.loc[m, "formula"].to_numpy()
        w["mono_mass"] = p.loc[m, "mono_mass"].to_numpy()
        if name == "coconut":
            w = w.sort_values(["ik14", "annotation_level", "src_id"], ascending=[True, False, True],
                              kind="mergesort", na_position="last")
        elif name in ("lotus", "pubchemlite"):
            w = w.sort_values(["ik14", "cid", "smiles"], kind="mergesort", na_position="last")
        else:
            w = w.sort_values(["ik14", "src_id"], kind="mergesort")
        w = w.drop_duplicates("ik14", keep="first")
        _cascade(rep, name, "dedup_ik14", len(w))
        rep.setdefault("rows_after_filters", {})[name] = len(w)
        winners.append(w[["src", "src_id", "ik14", "smiles", "formula", "mono_mass", "src_mass",
                          "src_formula"]])
    del raw, lists, props

    # Cross-source union, COCONUT SMILES preferred.
    u = pd.concat(winners, ignore_index=True)
    del winners
    u = u.drop_duplicates("ik14", keep="first").set_index("ik14")
    rep["union_rows"] = len(u)
    rep["smiles_source_counts"] = u["src"].value_counts().to_dict()
    log.info("union: %d rows (smiles source counts %s)", len(u), rep["smiles_source_counts"])

    out = pd.DataFrame(index=u.index)
    out["smiles"] = u["smiles"]
    out["formula"] = u["formula"]
    out["mono_mass"] = u["mono_mass"].astype("float64")
    for name in SOURCE_ORDER:
        out[f"in_{name}"] = out.index.isin(agg[name].index)
    co = agg["coconut"].reindex(out.index)
    out["coconut_annotation_level"] = co["coconut_annotation_level"].round().astype("Int8")
    out["coconut_np_likeness"] = co["coconut_np_likeness"].astype("Float32")
    for c in ("n_organisms", "n_citations", "n_collections"):
        out[c] = co[c].round().astype("Int32")
    pl = agg["pubchemlite"].reindex(out.index)
    for c in ("pubmed_count", "patent_count", "anno_type_count"):
        out[c] = pl[c].round().astype("Int32")
    cid = agg["pubchemlite"]["cid_pubchemlite"].reindex(out.index)
    cid = cid.fillna(agg["lotus"]["cid_lotus"].reindex(out.index))
    out["cid"] = cid.round().astype("Int64")
    out["smiles_source"] = u["src"]
    out["coconut_id"] = u["src_id"].where(u["src"] == "coconut")
    out["npaid"] = u["src_id"].where(u["src"] == "npatlas")
    # coconut_id / npaid also for rows whose SMILES came from another source is not
    # needed by the ranker; keep only the winning row's id (documented).

    # Mass agreement with the source that supplied the SMILES.
    diff = (u["mono_mass"] - u["src_mass"]).abs()
    has = u["src_mass"].notna()
    bad = has & (diff > MASS_AGREE_DA)
    n_has = int(has.sum())
    frac = 1.0 - bad.sum() / max(n_has, 1)
    rep["mass_check"] = {"rows_with_source_mass": n_has, "rows_without_source_mass": int((~has).sum()),
                         "disagree_gt_0p001": int(bad.sum()), "agree_fraction": frac,
                         "by_source": {s: int(bad[u["src"] == s].sum()) for s in SOURCE_ORDER}}
    fbad = u["formula"] != u["src_formula"].astype(str).str.replace(" ", "", regex=False)
    rep["formula_differs_from_source"] = {s: int(fbad[u["src"] == s].sum()) for s in SOURCE_ORDER}
    log.info("mass check: %d / %d rows disagree by > %.3f Da (agree fraction %.5f), by source %s",
             int(bad.sum()), n_has, MASS_AGREE_DA, frac, rep["mass_check"]["by_source"])
    log.info("formula differs from the source formula: %s", rep["formula_differs_from_source"])
    paths.ensure_dir(ctx.out)
    dis = u.loc[bad, ["src", "src_id", "smiles", "formula", "src_formula", "mono_mass", "src_mass"]].copy()
    dis["diff_da"] = (dis["mono_mass"] - dis["src_mass"])
    dis = dis.sort_values(["src", "src_id"], kind="mergesort")
    dis.to_csv(ctx.out / "mass_disagreements.csv")
    for row in dis.head(15).itertuples():
        log.info("  mass disagreement %s %s %s rdkit %.5f source %.5f (%s vs %s)", row.src, row.src_id,
                 row.Index, row.mono_mass, row.src_mass, row.formula, row.src_formula)

    out = out.reset_index().rename(columns={"index": "ik14"})
    out = out.sort_values(["mono_mass", "ik14"], kind="mergesort").reset_index(drop=True)
    assert out["ik14"].is_unique
    tbl = pa.Table.from_pandas(out[OUT_SCHEMA.names], schema=OUT_SCHEMA, preserve_index=False)
    tmp = ctx.np_parquet.with_suffix(".parquet.part")
    pq.write_table(tbl, tmp, compression="zstd", compression_level=9, row_group_size=100_000)
    tmp.replace(ctx.np_parquet)
    rep["np_parquet"] = {"rows": len(out), "bytes": ctx.np_parquet.stat().st_size,
                         "sha256": sha256_file(ctx.np_parquet)}
    rep["membership_counts"] = {f"in_{s}": int(out[f"in_{s}"].sum()) for s in SOURCE_ORDER}
    rep["only_in"] = {s: int((out[f"in_{s}"] & ~out[[f"in_{t}" for t in SOURCE_ORDER if t != s]].any(axis=1)).sum())
                      for s in SOURCE_ORDER}
    rep["table_seconds"] = round(time.time() - t0, 1)
    ctx.save_report(rep)
    log.info("wrote %s: %d rows, %.1f MB, membership %s, only-in %s, %.0f s", ctx.np_parquet, len(out),
             ctx.np_parquet.stat().st_size / 1e6, rep["membership_counts"], rep["only_in"],
             time.time() - t0)


# ----------------------------------------------------------------------------- fingerprints

def _fp_backend_note(use_chem: bool) -> str:
    if use_chem:
        return "casmi.chem: morgan_fp(std_mol(smiles)) (= chem.morgan_fps(standardize=True))"
    return "RDKit direct: morgan_packed(MolFromSmiles(smiles)), no standardisation"


def stage_fp(ctx: Ctx) -> None:
    cfg = ctx.cfg
    t0 = time.time()
    smiles = pq.read_table(ctx.np_parquet, columns=["smiles"]).column("smiles").to_pylist()
    n = len(smiles)
    n_bits, radius = int(cfg.fp_morgan_bits), int(cfg.fp_morgan_radius)
    if n_bits % 8:
        raise SystemExit("fp_morgan_bits must be a multiple of 8")
    use_chem = _chem() is not None and not ctx.args.fp_raw
    note = _fp_backend_note(use_chem)
    log.info("fp: %d rows, Morgan r=%d %d bits, %d workers (%s)", n, radius, n_bits, ctx.workers, note)
    tmp = ctx.np_fp.with_name("np_fp.part.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(n, n_bits // 8))
    chunk = 5000
    tasks = [(c, radius, n_bits, use_chem) for _, c in _chunks(smiles, chunk)]
    bad_total, std_fail_total = [], []
    with _pool(ctx) as pool:
        for j, (out, bad, std_fail) in enumerate(pool.imap(fp_chunk, tasks)):
            i0 = j * chunk
            arr[i0:i0 + len(out)] = out
            bad_total.extend(i0 + b for b in bad)
            std_fail_total.extend(i0 + b for b in std_fail)
            if (j + 1) % 40 == 0:
                log.info("fp: %d / %d (%.0f s)", min(i0 + chunk, n), n, time.time() - t0)
    arr.flush()
    del arr
    if bad_total:
        raise SystemExit(f"fp: {len(bad_total)} stored SMILES failed to parse, e.g. rows {bad_total[:5]}")
    tmp.replace(ctx.np_fp)
    if std_fail_total:
        log.warning("fp: std_mol failed for %d rows (fingerprinted unstandardised), e.g. rows %s",
                    len(std_fail_total), std_fail_total[:10])
    rep = ctx.load_report()
    rep["np_fp"] = {"shape": [n, n_bits // 8], "dtype": "uint8", "bitorder": "big",
                    "radius": radius, "n_bits": n_bits, "bytes": ctx.np_fp.stat().st_size,
                    "sha256": sha256_file(ctx.np_fp), "seconds": round(time.time() - t0, 1),
                    "backend": note, "standardized": use_chem,
                    "std_mol_failures": len(std_fail_total),
                    "std_mol_failure_rows": std_fail_total[:1000]}
    ctx.save_report(rep)
    log.info("wrote %s: shape (%d, %d), %.1f MB, %.0f s", ctx.np_fp, n, n_bits // 8,
             ctx.np_fp.stat().st_size / 1e6, time.time() - t0)


# ----------------------------------------------------------------------------- check

def public_test_masses() -> pd.DataFrame:
    t = pd.read_parquet(paths.TEST_PARQUET, columns=["molecule_id", "adduct", "precursor_mz"])
    t["shift"] = t["adduct"].map(ADDUCT_SHIFT)
    unknown = t["shift"].isna()
    if unknown.any():
        log.warning("public test: %d spectra with an adduct outside PLAN 3.1, ignored", int(unknown.sum()))
    t = t[~unknown]
    t["M"] = t["precursor_mz"] - t["shift"]
    return t.groupby("molecule_id", sort=True)["M"].median().rename("M").reset_index()


def stage_check(ctx: Ctx) -> None:
    from casmi.db import CandidateDB

    cfg = ctx.cfg
    rep = ctx.load_report()
    res = {"ok": True, "problems": []}

    def fail(msg: str) -> None:
        res["ok"] = False
        res["problems"].append(msg)
        log.error("CHECK FAILED: %s", msg)

    df = pd.read_parquet(ctx.np_parquet)
    n = len(df)
    res["rows"] = n
    if not (600_000 <= n <= 1_500_000):
        (fail if ctx.limit is None else log.warning)(f"row count {n} outside [0.6M, 1.5M]")
    res["ik14_unique"] = bool(df["ik14"].is_unique)
    if not res["ik14_unique"]:
        fail("ik14 not unique")
    if not df["mono_mass"].is_monotonic_increasing:
        fail("np.parquet not sorted by mono_mass")
    mc = rep.get("mass_check", {})
    res["mass_agree_fraction"] = mc.get("agree_fraction")
    if mc.get("agree_fraction", 0.0) < 0.999:
        fail(f"RDKit vs source mass agreement {mc.get('agree_fraction')} < 0.999")
    fp = np.load(ctx.np_fp, mmap_mode="r")
    res["fp_shape"] = list(fp.shape)
    if fp.shape != (n, cfg.fp_morgan_bits // 8):
        fail(f"np_fp.npy shape {fp.shape} != ({n}, {cfg.fp_morgan_bits // 8})")

    # Fingerprint spot check against a fresh RDKit computation, and InChIKey agreement.
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    rng = np.random.default_rng(cfg.seed)
    sample = np.sort(rng.choice(n, size=min(5000, n), replace=False))
    fpinfo = rep.get("np_fp", {})
    std_failed = set(fpinfo.get("std_mol_failure_rows", []))
    fp_sample = [int(i) for i in sample[::5] if int(i) not in std_failed]
    if fpinfo.get("standardized"):
        from casmi import chem
        ref, ok = chem.morgan_fps([df.at[i, "smiles"] for i in fp_sample], cfg, n_jobs=1,
                                  standardize=True)
        n_fp_bad = int(sum(not np.array_equal(ref[k], fp[i]) for k, i in enumerate(fp_sample)))
    else:
        n_fp_bad = int(sum(not np.array_equal(morgan_packed(Chem.MolFromSmiles(df.at[i, "smiles"]),
                                                            cfg.fp_morgan_radius, cfg.fp_morgan_bits), fp[i])
                           for i in fp_sample))
    res["fp_spot_check_rows"] = len(fp_sample)
    n_key_same = n_key_total = 0
    for i in sample:
        mol = Chem.MolFromSmiles(df.at[i, "smiles"])
        key = Chem.MolToInchiKey(mol)
        n_key_total += 1
        n_key_same += int(key[:14] == df.at[i, "ik14"])
    res["fp_spot_check_mismatches"] = n_fp_bad
    if n_fp_bad:
        fail(f"{n_fp_bad} fingerprint spot-check mismatches")
    res["rdkit_inchikey14_agrees_with_source"] = n_key_same / max(n_key_total, 1)
    log.info("RDKit InChIKey[:14] equals the source ik14 for %.4f of %d sampled rows",
             res["rdkit_inchikey14_agrees_with_source"], n_key_total)

    db = CandidateDB(np_path=ctx.np_parquet, pubchem_dir=ctx.out / "_no_pubchem_", cfg=cfg)
    tl = time.perf_counter()
    db.load_np()
    res["np_load_seconds"] = round(time.perf_counter() - tl, 2)
    q = db.lookup(QUERCETIN_M, 10.0, tiers=("np",))
    hit = q[q["ik14"] == QUERCETIN_IK14]
    res["quercetin_pool"] = len(q)
    res["quercetin_found"] = bool(len(hit))
    res["quercetin_in_coconut"] = bool(len(hit) and bool(hit["in_coconut"].iloc[0]))
    if not res["quercetin_in_coconut"]:
        (fail if ctx.limit is None else log.warning)("quercetin REFJWTPEDVJJIY not found with in_coconut true")
    else:
        log.info("quercetin: found in a pool of %d, row %s", len(q), hit.iloc[0].to_dict())

    tm = public_test_masses()
    masses = tm["M"].to_numpy()
    # Timing: single lookups at the 400 public-test masses plus 600 random ones.
    rnd = rng.uniform(150, 1200, size=600)
    times = []
    sizes = []
    for m in np.concatenate([masses, rnd]):
        t1 = time.perf_counter()
        r = db.lookup(float(m), cfg.mass_tol_ppm, tiers=("np",))
        times.append((time.perf_counter() - t1) * 1000)
        sizes.append(len(r))
    times = np.array(times)
    res["lookup_ms"] = {"mean": float(times.mean()), "p50": float(np.median(times)),
                        "p99": float(np.percentile(times, 99)), "max": float(times.max())}
    if np.median(times) >= 50:
        fail(f"median NP lookup {np.median(times):.1f} ms >= 50 ms")
    log.info("NP lookup time (ms): %s", res["lookup_ms"])

    tb = time.perf_counter()
    bt = db.lookup_batch(masses, cfg.mass_tol_ppm, tiers=("np",))
    res["batch_400_seconds"] = round(time.perf_counter() - tb, 3)
    pool = np.array(sizes[:len(masses)])
    if len(bt) != pool.sum():
        fail(f"lookup_batch rows {len(bt)} != sum of single lookups {pool.sum()}")
    pools = {}
    for tol in (5.0, cfg.mass_tol_ppm, 20.0):
        s = np.array([len(db.lookup(float(m), tol, tiers=("np",))) for m in masses])
        pools[f"{tol:g}ppm"] = {"median": float(np.median(s)), "p95": float(np.percentile(s, 95)),
                                "mean": float(s.mean()), "max": int(s.max()),
                                "frac_empty": float((s == 0).mean()), "n": int(len(s))}
        log.info("public-test NP pool at %g ppm: median %.0f, p95 %.0f, max %d, empty %.3f",
                 tol, np.median(s), np.percentile(s, 95), s.max(), (s == 0).mean())
    res["public_test_pool"] = pools
    rep["check"] = res
    rep["raw_files"] = {}
    for name in SOURCES:
        p = ctx.src_path(name)
        if p.exists():
            rep["raw_files"][name] = {"path": str(p.relative_to(paths.ROOT)) if p.is_relative_to(paths.ROOT) else str(p),
                                      "url": SOURCES[name]["url"], "bytes": p.stat().st_size,
                                      "sha256": sha256_file(p)}
    ctx.save_report(rep)
    if not res["ok"]:
        raise SystemExit("acceptance checks failed: " + "; ".join(res["problems"]))
    log.info("all checks passed")


# ----------------------------------------------------------------------------- main

STAGES = ["download", "headers", "table", "fp", "check"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=STAGES + ["all"], default="all")
    ap.add_argument("--config", default=None, help="YAML config overrides (configs/*.yaml)")
    ap.add_argument("--workers", type=int, default=2, help="RDKit worker processes (default 2)")
    ap.add_argument("--limit", type=int, default=None, help="read only the first N rows per source (dry run)")
    ap.add_argument("--out", default=None, help="output directory (default PROCESSED/db/np)")
    ap.add_argument("--fp-raw", action="store_true",
                    help="fingerprint the stored SMILES without casmi.chem.std_mol")
    args = ap.parse_args(argv)
    lg = get_logger("build_npdb")
    log_run(lg, args)
    from casmi.config import load_config
    cfg = load_config(args.config)
    ctx = Ctx(args, cfg)
    stages = STAGES if args.stage == "all" else [args.stage]
    for st in stages:
        t = time.time()
        log.info("=== stage %s ===", st)
        globals()[f"stage_{st}"](ctx)
        log.info("=== stage %s done in %.0f s ===", st, time.time() - t)
    return 0


if __name__ == "__main__":
    sys.exit(main())

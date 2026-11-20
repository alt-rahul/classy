"""Holdout molecules, folds and the training exclusion (PLAN.md 5.2, 5.4; WP7).

A holdout molecule is a metric key (``mkey``). Every train structure (ik14) whose mkey equals a
holdout mkey belongs to that molecule, so the holdout file has one row per ik14:

    ik14, mkey, set, fold, eligible_c1, query_lib, primary, mono_mass, formula

(``splits/holdout_v1.csv``, tracked in git). ``primary`` marks the ik14 that was sampled (its
spectra are the query spectra); the other rows of the same mkey are aliases (tautomers or
protonation variants shipped under another inchikey14).

Sets (``SETS``):

- NP-TIMS: every enveda-np-examples structure with at least one query-usable spectrum there
  (``query_ok`` in WP6 meta: one of the ten test adducts, base peak >= 1000 where known,
  |M error| <= ``val_query_max_abs_ppm``). Query library enveda-np-examples.
- NP-OTHER: structures in the NP tier with no spectra in the two Enveda libraries (checked
  over the whole mkey group) and at least ``val_np_other_min_spectra`` query-usable spectra
  in one library of ``val_np_other_libs``; query library = the one with the most QTOF
  query-usable spectra, then the most query-usable spectra, then name order.
- SYN-TIMS: enveda-180 structures with a query-usable enveda-180 spectrum, mass in
  [``val_target_mass_min``, ``val_target_mass_max``]; ``val_syn_np_fraction`` of them are the
  highest NP-likeness (``chem.np_score``) and the rest random, both following the hidden
  mass distribution as far as the library allows (``mass_quotas``).

An mkey sampled by several sets goes to the first in the order NP-TIMS, NP-OTHER, SYN-TIMS;
two sampled ik14 with the same mkey make one molecule. Selection is deterministic: each ik14
gets a uniform number from sha1(seed, set, ik14), so the choice does not depend on row order
and changes little when the candidate pool changes.

Target mass distribution (``target_mass_density``): the hidden test is 157 to 1,159 Da with
median 348 (COMPETITION.md); modelled as a two-piece log-normal with median 348 whose lower
half puts 157 at the 1st percentile and whose upper half puts 1,159 at the 99.5th, truncated
to the range. ``mass_quotas`` splits a sample size over ``MASS_BINS`` log-spaced bins in
proportion to it, moving the quota a bin cannot fill to the other bins pro rata.

``fold`` 0 to ``val_n_folds - 1`` by molecule, stratified by set (rank of sha1(seed, mkey)
within the set, modulo the number of folds). ``eligible_c1``: the molecule has spectra in a
library other than its query library (PLAN 5.3).

Training exclusion (WP7 acceptance): every training script calls
``training_exclusion(holdout, structures)`` and passes the data it hands to a model through
``TrainingExclusion.filter`` (or checks it with ``assert_clean``); both raise AssertionError if
a holdout ik14 or mkey would remain.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from . import paths
from .config import Config

__all__ = [
    "SETS",
    "SET_CODES",
    "QUERY_LIBS",
    "ENVEDA_LIBS",
    "HOLDOUT_COLUMNS",
    "MASS_BINS",
    "default_holdout_path",
    "hash_uniform",
    "target_mass_density",
    "mass_quotas",
    "ensure_mkeys",
    "make_holdout",
    "save_holdout",
    "load_holdout",
    "holdout_molecules",
    "TrainingExclusion",
    "training_exclusion",
    "assert_no_holdout",
]

log = logging.getLogger(__name__)

SETS: tuple[str, ...] = ("NP-TIMS", "SYN-TIMS", "NP-OTHER")
SET_CODES = {"NP-TIMS": "NT", "SYN-TIMS": "ST", "NP-OTHER": "NO"}
QUERY_LIBS = {"NP-TIMS": "enveda-np-examples", "SYN-TIMS": "enveda-180"}
ENVEDA_LIBS = ("enveda-180", "enveda-np-examples")
PRIORITY = ("NP-TIMS", "NP-OTHER", "SYN-TIMS")
HOLDOUT_COLUMNS = ("ik14", "mkey", "set", "fold", "eligible_c1", "query_lib", "primary",
                   "mono_mass", "formula")
MASS_BINS = 24


def default_holdout_path() -> Path:
    return paths.SPLITS / "holdout_v1.csv"


# --------------------------------------------------------------------------------------
# deterministic randomness and the mass target
# --------------------------------------------------------------------------------------

def hash_uniform(keys: Iterable[str], *salt: Any) -> np.ndarray:
    """Uniform [0, 1) per key from sha1(salt..., key); independent of order and platform."""
    pre = "|".join(str(s) for s in salt) + "|"
    out = []
    for k in keys:
        h = hashlib.sha1((pre + str(k)).encode("utf-8")).digest()
        out.append(int.from_bytes(h[:8], "big") / 2.0 ** 64)
    return np.asarray(out, dtype=np.float64)


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    from scipy.special import ndtr

    return ndtr(z)


def target_mass_density(cfg: Config) -> Callable[[np.ndarray], np.ndarray]:
    """CDF (unnormalised over the truncation range) of the target mass distribution."""
    m0, lo, hi = cfg.val_target_mass_median, cfg.val_target_mass_min, cfg.val_target_mass_max
    s_lo = (math.log(m0) - math.log(lo)) / 2.3263   # lo at the 1st percentile
    s_hi = (math.log(hi) - math.log(m0)) / 2.5758   # hi at the 99.5th percentile

    def cdf(m: np.ndarray) -> np.ndarray:
        x = np.log(np.clip(np.asarray(m, np.float64), 1e-9, None)) - math.log(m0)
        return np.where(x < 0, _norm_cdf(x / s_lo), 0.5 + (_norm_cdf(x / s_hi) - 0.5))

    return cdf


def _bins(cfg: Config) -> np.ndarray:
    return np.exp(np.linspace(math.log(cfg.val_target_mass_min), math.log(cfg.val_target_mass_max),
                              MASS_BINS + 1))


def mass_quotas(masses: np.ndarray, n: int, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """(bin index per mass (-1 outside the range), quota per bin) for a sample of ``n``.

    Quotas follow ``target_mass_density`` (largest remainder rounding); the part of a quota a
    bin cannot fill (fewer candidates) moves to the bins with spare candidates, in proportion
    to their target share, until everything is placed or no candidate is left.
    """
    edges = _bins(cfg)
    m = np.asarray(masses, np.float64)
    b = np.searchsorted(edges, m, side="right") - 1
    b = np.where((m >= edges[0]) & (m <= edges[-1]), np.minimum(b, MASS_BINS - 1), -1)
    cap = np.bincount(b[b >= 0], minlength=MASS_BINS)
    cdf = target_mass_density(cfg)
    p = np.diff(cdf(edges))
    p = p / p.sum()
    quota = np.zeros(MASS_BINS, np.int64)
    left = min(int(n), int(cap.sum()))
    while left > 0:
        room = cap - quota
        w = np.where(room > 0, p, 0.0)
        if w.sum() <= 0:
            break
        raw = left * w / w.sum()
        add = np.minimum(np.floor(raw).astype(np.int64), room)
        rem = left - int(add.sum())
        if rem > 0:
            frac = np.where(room - add > 0, raw - np.floor(raw), -1.0)
            for i in np.lexsort((np.arange(MASS_BINS), -frac))[:rem]:
                if frac[i] >= 0 and add[i] < room[i]:
                    add[i] += 1
        if add.sum() == 0:
            i = int(np.argmax(np.where(room > 0, p, -1.0)))
            add[i] = 1
        quota += add
        left -= int(add.sum())
    return b, quota


# --------------------------------------------------------------------------------------
# mkeys for a closure of structures
# --------------------------------------------------------------------------------------

def ensure_mkeys(structures: pd.DataFrame, ik14: Iterable[str], cfg: Config, n_jobs: int = 1
                 ) -> pd.DataFrame:
    """Fill missing ``mkey`` for ``ik14`` and every structure with the same RDKit formula.

    Two ik14 with the same mkey are tautomers or protonation variants: same heavy-atom
    skeleton and, after ``chem.std_mol`` (uncharging), the same formula. So computing mkeys
    over the same-formula closure finds every alias of a holdout molecule without the full
    mkey pass. Returns ``structures`` with the column filled where computed (copy).
    """
    from .metric import mkeys

    st = structures.copy()
    if "mkey" not in st.columns:
        st["mkey"] = None
    st["mkey"] = st["mkey"].astype(object)
    sel = st["ik14"].isin(set(ik14))
    forms = set(st.loc[sel, "formula"].dropna())
    clos = sel | st["formula"].isin(forms)
    need = clos & st["mkey"].isna() & st["smiles"].notna()
    if need.any():
        keys = mkeys(st.loc[need, "smiles"].tolist(), cfg, n_jobs=n_jobs, cache=True)
        st.loc[need, "mkey"] = pd.Series(keys, index=st.index[need], dtype=object)
        log.info("ensure_mkeys: %d computed (closure of %d structures over %d formulas)",
                 int(need.sum()), int(sel.sum()), len(forms))
    return st


# --------------------------------------------------------------------------------------
# candidates and sampling
# --------------------------------------------------------------------------------------

def _lib_cols(st: pd.DataFrame) -> dict[str, str]:
    return {c[len("n_lib_"):]: c for c in st.columns if c.startswith("n_lib_")}


def _candidates(st: pd.DataFrame, lc: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    """Per set: candidate ik14 with query_lib, n_query, mono_mass, np_score."""
    base = st[st["rdkit_ok"].astype(bool) & st["smiles"].notna()
              & np.isfinite(st["mono_mass"].to_numpy(np.float64))]
    base = base[["ik14", "mono_mass", "np_score", "formula"]]
    out = {}
    for s, lib in QUERY_LIBS.items():
        q = lc[(lc["ingest_lib"] == lib) & (lc["n_query_ok"] > 0)][["ik14", "n_query_ok"]]
        c = base.merge(q, on="ik14").assign(query_lib=lib)
        if s == "SYN-TIMS":
            m = c["mono_mass"]
            c = c[(m >= cfg.val_target_mass_min) & (m <= cfg.val_target_mass_max)]
        out[s] = c.rename(columns={"n_query_ok": "n_query"})
    # NP-OTHER
    libc = _lib_cols(st)
    env = np.zeros(len(st), np.int64)
    for lib in ENVEDA_LIBS:
        if lib in libc:
            env += st[libc[lib]].to_numpy(np.int64)
    in_np = st["in_np"].fillna(False).astype(bool) if "in_np" in st.columns else pd.Series(False, index=st.index)
    ok_ik = set(st.loc[(env == 0) & in_np.to_numpy(), "ik14"])
    q = lc[lc["ingest_lib"].isin(cfg.val_np_other_libs) & lc["ik14"].isin(ok_ik)
           & (lc["n_query_ok"] >= cfg.val_np_other_min_spectra)].copy()
    q["lib_rank"] = q["ingest_lib"].map({l: i for i, l in enumerate(sorted(cfg.val_np_other_libs))})
    q = q.sort_values(["ik14", "n_query_ok_qtof", "n_query_ok", "lib_rank"],
                      ascending=[True, False, False, True], kind="mergesort")
    q = q.drop_duplicates("ik14", keep="first")[["ik14", "ingest_lib", "n_query_ok"]]
    c = base.merge(q, on="ik14").rename(columns={"ingest_lib": "query_lib", "n_query_ok": "n_query"})
    out["NP-OTHER"] = c
    return out


def _select(c: pd.DataFrame, n: int, cfg: Config, set_name: str, excluded: set[str],
            mass_match: bool, np_fraction: float = 0.0) -> list[str]:
    """Deterministic selection of ``n`` ik14 from candidates ``c`` (see module docstring)."""
    c = c[~c["ik14"].isin(excluded)]
    if n <= 0 or not len(c):
        return []
    ik = c["ik14"].to_numpy(object)
    u = hash_uniform(ik, cfg.seed, set_name)
    if not mass_match:
        order = np.lexsort((ik.astype(str), u))
        return [str(x) for x in ik[order[:n]]]
    m = c["mono_mass"].to_numpy(np.float64)
    npsc = np.nan_to_num(c["np_score"].to_numpy(np.float64), nan=-np.inf)
    chosen: list[str] = []
    taken = np.zeros(len(c), bool)
    n_np = int(round(n * np_fraction))
    for part, k in (("np", n_np), ("rand", n - n_np)):
        if k <= 0:
            continue
        avail = ~taken
        b, quota = mass_quotas(np.where(avail, m, np.nan), k, cfg)
        for bi in range(MASS_BINS):
            if quota[bi] <= 0:
                continue
            idx = np.flatnonzero((b == bi) & avail)
            if part == "np":
                o = np.lexsort((ik[idx].astype(str), u[idx], -npsc[idx]))
            else:
                o = np.lexsort((ik[idx].astype(str), u[idx]))
            pick = idx[o[:quota[bi]]]
            taken[pick] = True
            chosen.extend(str(x) for x in ik[pick])
    return chosen


def make_holdout(structures: pd.DataFrame, lib_counts: pd.DataFrame, cfg: Config | None = None,
                 n_np_tims: int | None = None, n_syn_tims: int | None = None,
                 n_np_other: int | None = None, n_jobs: int = 1, max_rounds: int = 8
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the holdout table (one row per ik14, HOLDOUT_COLUMNS) and return it with the
    structures table (mkeys filled for the closures used).

    ``structures`` / ``lib_counts``: WP6 ``structures.parquet`` and
    ``structure_lib_counts.parquet``. Sizes default to all NP-TIMS candidates,
    ``cfg.val_n_syn_tims`` and ``cfg.val_n_np_other``.
    """
    cfg = cfg or Config()
    sizes = {"NP-TIMS": n_np_tims, "SYN-TIMS": n_syn_tims if n_syn_tims is not None else cfg.val_n_syn_tims,
             "NP-OTHER": n_np_other if n_np_other is not None else cfg.val_n_np_other}
    st = structures.copy()
    if "mkey" not in st.columns:
        st["mkey"] = None
    cands = _candidates(st, lib_counts, cfg)
    for s in SETS:
        log.info("holdout candidates %s: %d structures", s, len(cands[s]))
    libc = _lib_cols(st)
    taken_mkeys: dict[str, str] = {}
    picked: dict[str, list[str]] = {}
    for s in PRIORITY:
        c = cands[s]
        n = len(c) if sizes[s] is None or sizes[s] <= 0 else int(sizes[s])
        mass_match = s == "SYN-TIMS" or (s == "NP-OTHER" and cfg.val_np_other_mass_match)
        frac = cfg.val_syn_np_fraction if s == "SYN-TIMS" else 0.0
        excluded: set[str] = set()
        sel: list[str] = []
        for _ in range(max_rounds):
            sel = _select(c, n, cfg, s, excluded, mass_match, frac) if s != "NP-TIMS" or sizes[s] \
                else sorted(c["ik14"].astype(str))
            st = ensure_mkeys(st, sel, cfg, n_jobs)
            mk = dict(zip(st["ik14"], st["mkey"]))
            env_by_mkey: dict[str, int] = {}
            if s == "NP-OTHER":
                ecols = [libc[l] for l in ENVEDA_LIBS if l in libc]
                keys_sel = {mk.get(k) for k in sel} - {None}
                sub = st[st["mkey"].isin(keys_sel)]
                env_by_mkey = (sub[ecols].sum(axis=1).groupby(sub["mkey"]).sum().astype(int).to_dict()
                               if ecols and len(sub) else {})
            bad: set[str] = set()
            seen: set[str] = set()
            for k in sel:
                key = mk.get(k)
                if not isinstance(key, str) or key in taken_mkeys or key in seen:
                    bad.add(k)
                    continue
                if env_by_mkey.get(key, 0):
                    bad.add(k)
                    continue
                seen.add(key)
            if not bad:
                break
            excluded |= bad
            sel = [k for k in sel if k not in bad]
            if s == "NP-TIMS" and not sizes[s]:
                c = c[~c["ik14"].isin(excluded)]
        picked[s] = sel
        for k in sel:
            taken_mkeys[mk[k]] = s
        log.info("holdout %s: %d molecules (target %s)", s, len(sel), n)
    # closure rows and columns
    st = ensure_mkeys(st, [k for v in picked.values() for k in v], cfg, n_jobs)
    mk = dict(zip(st["ik14"], st["mkey"]))
    rows = []
    qlib = {s: dict(zip(cands[s]["ik14"], cands[s]["query_lib"])) for s in SETS}
    all_keys = {mk[k] for v in picked.values() for k in v}
    groups = {key: g for key, g in st[st["mkey"].isin(all_keys)].groupby("mkey", sort=False)}
    for s in SETS:
        keys = [mk[k] for k in picked[s]]
        u = hash_uniform(keys, cfg.seed, "fold")
        order = np.lexsort((np.asarray(keys, dtype=object).astype(str), u))
        fold = np.empty(len(keys), np.int64)
        fold[order] = np.arange(len(keys)) % cfg.val_n_folds
        for k, key, f in zip(picked[s], keys, fold):
            grp = groups[key]
            ql = qlib[s][k]
            other = 0
            for lib, col in libc.items():
                if lib != ql:
                    other += int(grp[col].sum())
            for r in grp.itertuples(index=False):
                rows.append({"ik14": r.ik14, "mkey": key, "set": s, "fold": int(f),
                             "eligible_c1": other > 0, "query_lib": ql, "primary": r.ik14 == k,
                             "mono_mass": float(r.mono_mass), "formula": r.formula})
    ho = pd.DataFrame(rows, columns=list(HOLDOUT_COLUMNS))
    ho = ho.sort_values(["set", "mkey", "primary", "ik14"], ascending=[True, True, False, True],
                        kind="mergesort").reset_index(drop=True)
    assert not ho["ik14"].duplicated().any(), "an ik14 in two holdout molecules"
    return ho, st


def save_holdout(ho: pd.DataFrame, path: Path | str | None = None) -> Path:
    p = Path(path) if path is not None else default_holdout_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    out = ho[list(HOLDOUT_COLUMNS)].copy()
    out["mono_mass"] = out["mono_mass"].map(lambda x: f"{x:.6f}")
    out.to_csv(p, index=False, lineterminator="\n")
    return p


def load_holdout(path: Path | str | pd.DataFrame | None = None) -> pd.DataFrame:
    """Read a holdout CSV (default ``splits/holdout_v1.csv``) with proper dtypes."""
    if isinstance(path, pd.DataFrame):
        ho = path.copy()
    else:
        p = Path(path) if path is not None else default_holdout_path()
        ho = pd.read_csv(p, dtype=str, keep_default_na=False, na_values=[""])
    for c in ("ik14", "mkey", "set", "query_lib", "formula"):
        if c in ho.columns:
            ho[c] = ho[c].astype(object)
    if "fold" in ho.columns:
        ho["fold"] = ho["fold"].astype(np.int64)
    for c in ("eligible_c1", "primary"):
        if c in ho.columns and ho[c].dtype != bool:
            ho[c] = ho[c].astype(str).str.lower().isin(("true", "1"))
    if "mono_mass" in ho.columns:
        ho["mono_mass"] = ho["mono_mass"].astype(np.float64)
    return ho


def holdout_molecules(ho: pd.DataFrame) -> pd.DataFrame:
    """One row per holdout molecule (mkey): set, fold, eligible_c1, query_lib, ik14 (primary),
    ik14_group (tuple of every ik14 of the mkey), mono_mass, formula."""
    rows = []
    for key, g in ho.groupby("mkey", sort=True):
        prim = g[g["primary"]] if "primary" in g.columns and g["primary"].any() else g.iloc[:1]
        p = prim.iloc[0]
        rows.append({"mkey": key, "set": p["set"], "fold": int(p["fold"]),
                     "eligible_c1": bool(p["eligible_c1"]), "query_lib": p["query_lib"],
                     "ik14": p["ik14"], "ik14_group": tuple(sorted(g["ik14"].astype(str))),
                     "mono_mass": float(p["mono_mass"]), "formula": p["formula"]})
    out = pd.DataFrame(rows)
    return out.sort_values(["set", "mkey"], kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# training exclusion
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingExclusion:
    """Holdout keys that must never reach model training (PLAN 5.3, last paragraph)."""

    ik14: frozenset
    mkey: frozenset

    def mask(self, ik14: Sequence[Any], mkey: Sequence[Any] | None = None) -> np.ndarray:
        """True where a row belongs to a holdout molecule (by ik14, or by mkey if given)."""
        ik = pd.Series(np.asarray(ik14, dtype=object))
        m = ik.isin(self.ik14).to_numpy()
        if mkey is not None:
            m = m | pd.Series(np.asarray(mkey, dtype=object)).isin(self.mkey).to_numpy()
        return m

    def assert_clean(self, ik14: Sequence[Any], mkey: Sequence[Any] | None = None) -> None:
        """AssertionError if any row belongs to a holdout molecule."""
        bad = self.mask(ik14, mkey)
        if bad.any():
            ex = sorted({str(x) for x in np.asarray(ik14, dtype=object)[bad]})[:5]
            raise AssertionError(f"holdout leak: {int(bad.sum())} training rows of holdout "
                                 f"molecules, e.g. {ex}")

    def filter(self, df: pd.DataFrame, ik14_col: str | None = None,
               mkey_col: str | None = None) -> pd.DataFrame:
        """``df`` without holdout rows; asserts the result is clean. ``ik14_col`` defaults to
        "ik14" or "inchikey14"; ``mkey_col`` to "mkey" when present."""
        if ik14_col is None:
            ik14_col = "ik14" if "ik14" in df.columns else "inchikey14"
        if mkey_col is None and "mkey" in df.columns:
            mkey_col = "mkey"
        mk = df[mkey_col].to_numpy(object) if mkey_col else None
        keep = ~self.mask(df[ik14_col].to_numpy(object), mk)
        out = df[keep]
        self.assert_clean(out[ik14_col].to_numpy(object),
                          out[mkey_col].to_numpy(object) if mkey_col else None)
        log.info("training exclusion: %d of %d rows removed (%d holdout ik14, %d mkeys)",
                 int((~keep).sum()), len(df), len(self.ik14), len(self.mkey))
        return out


def training_exclusion(holdout: pd.DataFrame | Path | str | None = None,
                       structures: pd.DataFrame | Path | str | None = None) -> TrainingExclusion:
    """The keys training must exclude: every holdout ik14 and mkey, plus every structure
    whose mkey is a holdout mkey (from ``structures``, WP6 structures.parquet, if given;
    default ``PROCESSED/train/structures.parquet`` when it exists). ``holdout`` None: the v2
    file when it exists (v1 rows plus C2-ZEN), else v1 (``default_exclusion_holdout``).

    Every training script (WP9 fingerprint model, WP11 ranker, WP12 formula model) must call
    this and pass its training table through ``.filter`` (or ``.assert_clean``).
    """
    ho = load_holdout(holdout if holdout is not None else default_exclusion_holdout())
    ik = set(ho["ik14"].dropna().astype(str))
    mk = set(ho["mkey"].dropna().astype(str))
    if structures is None:
        p = paths.PROCESSED / "train" / "structures.parquet"
        structures = p if p.exists() else None
    if structures is not None:
        st = structures if isinstance(structures, pd.DataFrame) else pd.read_parquet(
            structures, columns=["ik14", "mkey"])
        if "mkey" in st.columns:
            ik |= set(st.loc[st["mkey"].isin(mk), "ik14"].astype(str))
    return TrainingExclusion(frozenset(ik), frozenset(mk))


def assert_no_holdout(ik14: Sequence[Any], holdout: pd.DataFrame | Path | str | None = None,
                      mkey: Sequence[Any] | None = None,
                      structures: pd.DataFrame | Path | str | None = None) -> None:
    """AssertionError if any of ``ik14`` (or ``mkey``) belongs to a holdout molecule."""
    training_exclusion(holdout, structures).assert_clean(ik14, mkey)


# --------------------------------------------------------------------------------------
# holdout v2 (ROADMAP 2.3; R1)
# --------------------------------------------------------------------------------------
#
# ``splits/holdout_v2.csv`` keeps every row of holdout_v1 unchanged (HOLDOUT_COLUMNS, same
# formatting) and appends the molecule-level columns ``V2_COLUMNS`` (repeated on every ik14
# row of a molecule) plus the rows of set C2-ZEN (Enveda-180 Zenodo structures absent from
# train, built by package R2: ``scripts/build_extlib.py --stage c2zen`` writes
# ``PROCESSED/validation/truth_c2_zen.parquet``; ``make_holdout_v2(zen_truth=...)`` takes it).
#
# - in_np, in_pubchem: any ik14 of the molecule in that tier (structures.parquet flags);
#   C2-ZEN: from the R2 truth table.
# - pmid_count: max over the molecule's ik14 of max(pmid_count, pubmed_count), the literature
#   count the fusion prior reads (PubChem PMIDs and PubChemLite PubMed); NaN if both absent.
# - sid_count: PubChem substance count (R3); NaN until R3 has filled the tiers.
# - n_ext_lib: external libraries (R2, final deduplicated files) holding spectra of the
#   molecule (``PROCESSED/extlib/holdout_ext.parquet``), 0 until R2; for C2-ZEN its own
#   Zenodo source does not count (it is the query library, as for eligible_c1).
# - has_public_spectra = eligible_c1 or n_ext_lib > 0.
# - c2_cell: "NP" for NP-TIMS / NP-OTHER molecules in the NP tier (the C2-NP cell); "PC" for
#   SYN-TIMS and C2-ZEN molecules without public spectra whose structure is in a tier (C2-PC),
#   and for NP-TIMS / NP-OTHER molecules in PubChem only; else "none".
# - w_obs: (1 + pmid_count) ** -alpha on molecules of the NP sets (NP-TIMS, NP-OTHER), 1 on
#   the others; alpha (per mix, ``holdout_v2.json``) from ``validation.calibrate_alpha``. The
#   column holds the weights of mix A (the default); the evaluator recomputes them per mix.
# - fast, w_fast: the fast panel (``cfg.val_v2_fast_sizes``: all NP-TIMS, 750 NP-OTHER, 500
#   SYN-TIMS of C2-PC, 300 C2-ZEN), stratified by (fold, pmid bin, eligible_c1) (coarser if
#   there are more strata than draws) with one molecule per stratum and the rest allocated in
#   proportion to stratum size; w_fast = stratum size / sampled in the stratum (inverse
#   inclusion probability, so fast-panel cell means estimate the full-panel ones), 0 off the
#   panel.

ZEN_SET = "C2-ZEN"
SETS_V2: tuple[str, ...] = SETS + (ZEN_SET,)
SET_CODES.setdefault(ZEN_SET, "ZE")
NP_SETS = ("NP-TIMS", "NP-OTHER")
PC_SETS = ("SYN-TIMS", ZEN_SET)
V2_COLUMNS = ("in_np", "in_pubchem", "pmid_count", "sid_count", "n_ext_lib",
              "has_public_spectra", "c2_cell", "w_obs", "fast", "w_fast")
HOLDOUT_V2_COLUMNS = HOLDOUT_COLUMNS + V2_COLUMNS
C2_CELL_VALUES = ("NP", "PC", "none")
ZEN_QUERY_LIB = "ext_enveda180"


def default_holdout_v2_path() -> Path:
    return paths.SPLITS / "holdout_v2.csv"


def holdout_v2_meta_path(path: Path | str | None = None) -> Path:
    """The JSON next to a v2 holdout file (alpha per mix, calibration record)."""
    p = Path(path) if path is not None else default_holdout_v2_path()
    return p.with_suffix(".json")


def w_obs_weights(pmid: Any, alpha: float) -> np.ndarray:
    """(1 + pmid) ** -alpha, missing pmid counts as 0."""
    x = pd.to_numeric(pd.Series(np.asarray(pmid, dtype=object)), errors="coerce").to_numpy(np.float64)
    x = np.clip(np.nan_to_num(x, nan=0.0), 0.0, None)
    return np.power(1.0 + x, -float(alpha))


def pmid_bins(pmid: Any, edges: Sequence[float]) -> np.ndarray:
    """Bin index per molecule: 0 below edges[0], i for edges[i-1] <= pmid < edges[i]."""
    x = pd.to_numeric(pd.Series(np.asarray(pmid, dtype=object)), errors="coerce").to_numpy(np.float64)
    x = np.nan_to_num(x, nan=0.0)
    return np.searchsorted(np.asarray(edges, np.float64), x, side="right").astype(np.int64)


def _group_any(st: pd.DataFrame, groups: Mapping[str, Sequence[str]], col: str) -> dict[str, bool]:
    if col not in st.columns:
        return {k: False for k in groups}
    v = st[col].astype("boolean").fillna(False).astype(bool)
    d = dict(zip(st["ik14"].astype(str), v))
    return {k: any(d.get(i, False) for i in g) for k, g in groups.items()}


def _group_max(st: pd.DataFrame, groups: Mapping[str, Sequence[str]], cols: Sequence[str]) -> dict[str, float]:
    vals = np.full(len(st), np.nan)
    for c in cols:
        if c in st.columns:
            v = pd.to_numeric(st[c], errors="coerce").to_numpy(np.float64, na_value=np.nan)
            vals = np.fmax(vals, v)
    d = dict(zip(st["ik14"].astype(str), vals))
    out = {}
    for k, g in groups.items():
        x = [d.get(i, np.nan) for i in g]
        x = [v for v in x if np.isfinite(v)]
        out[k] = float(max(x)) if x else float("nan")
    return out


def c2_cell_of(set_name: str, in_np: bool, in_pubchem: bool, has_public: bool) -> str:
    """C2 cell of a molecule (module notes above)."""
    if set_name in NP_SETS:
        return "NP" if in_np else ("PC" if in_pubchem else "none")
    if set_name in PC_SETS:
        return "PC" if (not has_public and (in_pubchem or in_np)) else "none"
    return "none"


def fast_panel(mols: pd.DataFrame, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """(fast, w_fast) per row of ``mols`` (one row per molecule: mkey, set, fold, pmid_count,
    eligible_c1, c2_cell). Deterministic (hash of seed, set, mkey); see module notes."""
    sizes = {str(s): int(n) for s, n in cfg.val_v2_fast_sizes}
    fast = np.zeros(len(mols), bool)
    wf = np.zeros(len(mols), np.float64)
    pb = pmid_bins(mols["pmid_count"], cfg.val_v2_pmid_edges)
    for s, n in sizes.items():
        elig = (mols["set"] == s).to_numpy().copy()
        if s in PC_SETS:
            elig = elig & (mols["c2_cell"] == "PC").to_numpy()
        idx = np.flatnonzero(elig)
        if not len(idx):
            continue
        if n <= 0 or n >= len(idx):
            fast[idx] = True
            wf[idx] = 1.0
            continue
        fold = mols["fold"].to_numpy()[idx]
        elig = mols["eligible_c1"].astype(bool).to_numpy()[idx]
        # finest stratification that gives every non-empty stratum at least one molecule
        for parts in ((fold, pb[idx], elig), (pb[idx], elig), (pb[idx],), ()):
            strata = list(zip(*parts)) if parts else [()] * len(idx)
            keys = sorted(set(strata))
            if len(keys) <= n:
                break
        pos = {t: j for j, t in enumerate(keys)}
        sid = np.array([pos[t] for t in strata])
        cnt = np.bincount(sid, minlength=len(keys))
        take = np.ones(len(keys), int)
        raw = (n - len(keys)) * cnt / cnt.sum()
        take += np.floor(raw).astype(int)
        rem = n - int(take.sum())
        frac = raw - np.floor(raw)
        for j in np.lexsort((np.arange(len(keys)), -frac))[:rem]:
            take[j] += 1
        take = np.minimum(take, cnt)
        u = hash_uniform(mols["mkey"].to_numpy(object)[idx], cfg.seed, "fast", s)
        for j in range(len(keys)):
            if take[j] <= 0:
                continue
            members = np.flatnonzero(sid == j)
            o = members[np.lexsort((mols["mkey"].to_numpy(object)[idx][members].astype(str), u[members]))]
            pick = idx[o[:take[j]]]
            fast[pick] = True
            wf[pick] = cnt[j] / take[j]
    return fast, wf


def zen_holdout_rows(zen_truth: pd.DataFrame | None) -> pd.DataFrame:
    """C2-ZEN rows (HOLDOUT_COLUMNS) from R2's ``truth_c2_zen.parquet`` (one row per
    molecule; ik14_group may list aliases separated by ";")."""
    if zen_truth is None or not len(zen_truth):
        return pd.DataFrame(columns=list(HOLDOUT_COLUMNS))
    rows = []
    for r in zen_truth.itertuples(index=False):
        grp = [g for g in str(getattr(r, "ik14_group", r.ik14) or r.ik14).split(";") if g]
        for k in sorted(set(grp) | {str(r.ik14)}):
            rows.append({"ik14": k, "mkey": str(r.mkey), "set": ZEN_SET, "fold": int(r.fold),
                         "eligible_c1": False, "query_lib": str(getattr(r, "query_lib", ZEN_QUERY_LIB)),
                         "primary": k == str(r.ik14), "mono_mass": float(r.mono_mass),
                         "formula": r.formula})
    return pd.DataFrame(rows, columns=list(HOLDOUT_COLUMNS))


def make_holdout_v2(ho_v1: pd.DataFrame, structures: pd.DataFrame, cfg: Config | None = None,
                    ext: pd.DataFrame | None = None, zen_truth: pd.DataFrame | None = None,
                    sid: Mapping[str, float] | None = None, alpha: float = 0.0) -> pd.DataFrame:
    """holdout_v2 table (HOLDOUT_V2_COLUMNS): v1 rows unchanged, v2 columns, C2-ZEN rows.

    ``ext``: R2's holdout_ext table (columns mkey, n_ext_lib) or None (all 0). ``zen_truth``:
    R2's C2-ZEN truth table or None. ``sid``: mkey -> sid_count (R3) or None (NaN).
    ``alpha``: w_obs exponent of mix A (``validation.calibrate_alpha``; 0 = unweighted).
    """
    cfg = cfg or Config()
    ho = load_holdout(ho_v1)
    zen = zen_holdout_rows(zen_truth)
    if len(zen):
        clash = set(zen["mkey"]) & set(ho["mkey"])
        assert not clash, f"C2-ZEN molecules already in holdout_v1: {sorted(clash)[:3]}"
        zen = load_holdout(zen)
        ho = pd.concat([ho, zen], ignore_index=True)
    groups = {k: sorted(g["ik14"].astype(str)) for k, g in ho.groupby("mkey", sort=True)}
    st = structures
    in_np = _group_any(st, groups, "in_np")
    in_pc = _group_any(st, groups, "in_pubchem")
    pmid = _group_max(st, groups, ("pmid_count", "pubmed_count"))
    if zen_truth is not None and len(zen_truth):
        zt = zen_truth.set_index(zen_truth["mkey"].astype(str))
        for k in zt.index:
            in_np[k] = bool(zt.at[k, "in_np"]) if "in_np" in zt.columns else False
            in_pc[k] = bool(zt.at[k, "in_pubchem"]) if "in_pubchem" in zt.columns else False
            if "pubmed_count" in zt.columns and pd.notna(zt.at[k, "pubmed_count"]):
                pmid[k] = float(zt.at[k, "pubmed_count"])
            elif "pmid_count" in zt.columns and pd.notna(zt.at[k, "pmid_count"]):
                pmid[k] = float(zt.at[k, "pmid_count"])
    n_ext: dict[str, int] = {}
    if ext is not None and len(ext):
        e = ext.dropna(subset=["mkey"]).groupby(ext["mkey"].astype(str))["n_ext_lib"].max()
        n_ext = {str(k): int(v) for k, v in e.items()}
    mols = holdout_molecules(ho)
    mk = mols["mkey"].astype(str)
    mols["in_np"] = mk.map(in_np).fillna(False).astype(bool)
    mols["in_pubchem"] = mk.map(in_pc).fillna(False).astype(bool)
    mols["pmid_count"] = mk.map(pmid).astype(np.float64)
    mols["sid_count"] = mk.map(sid or {}).astype(np.float64) if sid else np.nan
    mols["n_ext_lib"] = mk.map(n_ext).fillna(0).astype(np.int64)
    mols["has_public_spectra"] = mols["eligible_c1"].astype(bool) | (mols["n_ext_lib"] > 0)
    mols["c2_cell"] = [c2_cell_of(s, a, b, h) for s, a, b, h in
                       zip(mols["set"], mols["in_np"], mols["in_pubchem"], mols["has_public_spectra"])]
    w = w_obs_weights(mols["pmid_count"], alpha)
    mols["w_obs"] = np.where(mols["set"].isin(NP_SETS), w, 1.0)
    mols["fast"], mols["w_fast"] = fast_panel(mols, cfg)
    cols = [c for c in V2_COLUMNS]
    out = ho.merge(mols[["mkey", *cols]], on="mkey", how="left", validate="many_to_one")
    out = out[list(HOLDOUT_V2_COLUMNS)]
    # v1 rows first, in v1 order; then C2-ZEN rows
    return out.reset_index(drop=True)


def save_holdout_v2(ho2: pd.DataFrame, path: Path | str | None = None,
                    meta: Mapping[str, Any] | None = None) -> Path:
    """Write the v2 CSV (v1 columns formatted exactly as ``save_holdout``) and its JSON."""
    import json

    p = Path(path) if path is not None else default_holdout_v2_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    out = ho2[list(HOLDOUT_V2_COLUMNS)].copy()
    out["mono_mass"] = out["mono_mass"].map(lambda x: f"{x:.6f}")
    for c in ("pmid_count", "sid_count"):
        out[c] = out[c].map(lambda x: "" if not np.isfinite(x) else f"{x:.0f}")
    out["w_obs"] = out["w_obs"].map(lambda x: f"{x:.8g}")
    out["w_fast"] = out["w_fast"].map(lambda x: f"{x:.8g}")
    out.to_csv(p, index=False, lineterminator="\n")
    if meta is not None:
        holdout_v2_meta_path(p).write_text(json.dumps(meta, indent=2, default=float) + "\n")
    return p


def load_holdout_v2(path: Path | str | pd.DataFrame | None = None) -> pd.DataFrame:
    """Read holdout_v2.csv with proper dtypes (v1 columns as ``load_holdout``)."""
    ho = load_holdout(path if path is not None else default_holdout_v2_path())
    for c in ("in_np", "in_pubchem", "has_public_spectra", "fast"):
        if c in ho.columns and ho[c].dtype != bool:
            ho[c] = ho[c].astype(str).str.lower().isin(("true", "1"))
    for c in ("pmid_count", "sid_count", "w_obs", "w_fast"):
        if c in ho.columns:
            ho[c] = pd.to_numeric(ho[c], errors="coerce").astype(np.float64)
    if "n_ext_lib" in ho.columns:
        ho["n_ext_lib"] = pd.to_numeric(ho["n_ext_lib"], errors="coerce").fillna(0).astype(np.int64)
    if "c2_cell" in ho.columns:
        ho["c2_cell"] = ho["c2_cell"].fillna("none").astype(object)
    return ho


def load_holdout_v2_meta(path: Path | str | None = None) -> dict[str, Any]:
    import json

    p = holdout_v2_meta_path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def holdout_v2_molecules(ho2: pd.DataFrame) -> pd.DataFrame:
    """One row per molecule with the v1 molecule columns and V2_COLUMNS."""
    mols = holdout_molecules(ho2)
    first = ho2.drop_duplicates("mkey").set_index("mkey")
    for c in V2_COLUMNS:
        if c in first.columns:
            mols[c] = mols["mkey"].map(first[c]).to_numpy()
    return mols


def default_exclusion_holdout() -> Path:
    """Holdout file training must exclude: holdout_v2 (a superset of v1, adds C2-ZEN) when it
    exists, else holdout_v1."""
    p = default_holdout_v2_path()
    return p if p.exists() else default_holdout_path()


def tier_counts(ho: pd.DataFrame, column: str = "sid_count") -> dict[str, float]:
    """mkey -> max of ``column`` over the molecule's ik14 in the NP tier and PubChem tier files
    (R3 prior columns), read straight from the tier parquet files; {} while the column is
    absent from both (before R3)."""
    import pyarrow.parquet as pq

    from . import db as cdb

    ik2mk = dict(zip(ho["ik14"].astype(str), ho["mkey"].astype(str)))
    want = sorted(ik2mk)
    best: dict[str, float] = {}

    def take(df: pd.DataFrame) -> None:
        v = pd.to_numeric(df[column], errors="coerce").to_numpy(np.float64)
        for k, x in zip(df["ik14"].astype(str), v):
            if np.isfinite(x):
                m = ik2mk[k]
                best[m] = max(best.get(m, -np.inf), float(x))

    npf = cdb.np_dir() / "np.parquet"
    if npf.exists() and column in pq.read_schema(npf).names:
        take(pq.read_table(npf, columns=["ik14", column], filters=[("ik14", "in", want)]).to_pandas())
    pdir = cdb.pubchem_dir()
    masses = ho.drop_duplicates("mkey")["mono_mass"].astype(float).to_numpy()
    for b in sorted({cdb.shard_bucket(m) for m in masses if np.isfinite(m)}):
        f = pdir / cdb.shard_name(b)
        if not f.exists() or column not in pq.read_schema(f).names:
            continue
        take(pq.read_table(f, columns=["ik14", column], filters=[("ik14", "in", want)]).to_pandas())
    return best


__all__ += ["ZEN_SET", "SETS_V2", "NP_SETS", "PC_SETS", "V2_COLUMNS", "HOLDOUT_V2_COLUMNS",
            "default_holdout_v2_path", "holdout_v2_meta_path", "w_obs_weights", "pmid_bins",
            "c2_cell_of", "fast_panel", "zen_holdout_rows", "make_holdout_v2", "save_holdout_v2",
            "load_holdout_v2", "load_holdout_v2_meta", "holdout_v2_molecules",
            "default_exclusion_holdout", "tier_counts"]

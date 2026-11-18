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
    default ``PROCESSED/train/structures.parquet`` when it exists).

    Every training script (WP9 fingerprint model, WP11 ranker, WP12 formula model) must call
    this and pass its training table through ``.filter`` (or ``.assert_clean``).
    """
    ho = load_holdout(holdout)
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

"""Build the contrastive-encoder training bundle (RESEARCH.md roadmap item 1).

    nice -n 10 python scripts/build_encoder_bundle.py --n-jobs 2

Inputs (all local artefacts): the WP5 library store (cleaned peaks, ``data/processed/
library``), WP6 ``meta.parquet`` and ``structures.parquet`` (``data/processed/train``), the
holdout ``splits/holdout_v1.csv`` (excluded with ``splits.training_exclusion``), the NP tier
(``np.parquet``, ``np_fp.npy``) and the validation queries / truth of NP-TIMS and NP-OTHER
(``data/processed/validation``). Output: ``data/processed/encoder/bundle_v1/`` in the layout
of ``casmi.encoder.Bundle`` (module docstring there), plus ``info.json`` with counts,
filters, input checksums and diagnostics.

Training spectra: library rows with at least ``cfg.min_peaks`` cleaned peaks, label mass
in [fpm_mass_min, fpm_mass_max], a parseable adduct, |precursor error| <=
fpm_max_precursor_error_ppm (NaN passes), RDKit-parseable label SMILES, and not a holdout
molecule (by ik14 and mkey). Targets: ``chem.morgan_fps`` (std_mol + InChI round trip) of
the label SMILES. Negatives: see ``negatives``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from casmi import encoder as E  # noqa: E402
from casmi import paths  # noqa: E402
from casmi.adducts import TEST_ADDUCT_NAMES, match_test_adduct  # noqa: E402
from casmi.config import Config, load_config  # noqa: E402
from casmi.log import get_logger, git_commit, log_run  # noqa: E402

log = get_logger("build_encoder_bundle")

LIB_COLS = ["spec_idx", "spectrum_id", "ik14", "precursor_mz", "adduct", "adduct_family",
            "polarity", "instrument_family", "ingest_lib", "collision_energy", "n_energies",
            "precursor_error_ppm", "n_peaks"]
EVAL_SETS = ("NP-TIMS", "NP-OTHER")


def sha256(path: Path, block: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hash_key(values, salt: str) -> np.ndarray:
    """Deterministic uint64 per string (sha1 of salt|value), order independent."""
    out = np.empty(len(values), np.uint64)
    for i, v in enumerate(values):
        out[i] = int.from_bytes(hashlib.sha1(f"{salt}|{v}".encode()).digest()[:8], "big")
    return out


def adduct_codes(adduct: np.ndarray, polarity: np.ndarray) -> np.ndarray:
    keys = pd.Series([f"{a}|{p}" for a, p in zip(adduct, polarity)], dtype=object)
    codes, uniq = pd.factorize(keys)
    lut = np.full(len(uniq), E.ADDUCT_OTHER, np.int8)
    for u, k in enumerate(uniq):
        a, p = k.rsplit("|", 1)
        mode = "positive" if int(p) > 0 else "negative" if int(p) < 0 else None
        name = match_test_adduct(a, mode) if a not in ("None", "nan", "") else None
        if name is not None:
            lut[u] = TEST_ADDUCT_NAMES.index(name)
    return lut[codes]


def instrument_codes(fam: np.ndarray) -> np.ndarray:
    lut = {f: i for i, f in enumerate(E.INSTRUMENT_FAMILIES)}
    return pd.Series(fam, dtype=object).map(lambda v: lut.get(v, lut["other"])).to_numpy(np.int8)


# --------------------------------------------------------------------------------------
# training spectra
# --------------------------------------------------------------------------------------

def load_training_table(cfg: Config, lib_dir: Path, train_dir: Path, holdout: Path
                        ) -> tuple[pd.DataFrame, dict]:
    from casmi.splits import training_exclusion

    t0 = time.time()
    lib = pq.read_table(lib_dir / "meta.parquet", columns=LIB_COLS).to_pandas()
    if not np.array_equal(lib["spec_idx"].to_numpy(np.int64), np.arange(len(lib))):
        raise ValueError("library meta spec_idx is not the store row number")
    tm = pq.read_table(train_dir / "meta.parquet",
                       columns=["spectrum_id", "mono_mass", "m_spec", "adduct_known"]).to_pandas()
    df = lib.merge(tm, on="spectrum_id", how="left", validate="one_to_one")
    st = pq.read_table(train_dir / "structures.parquet",
                       columns=["ik14", "smiles", "rdkit_ok", "mkey", "in_np"]).to_pandas()
    df = df.merge(st, on="ik14", how="left", validate="many_to_one")
    counts = {"library_rows": int(len(df))}
    keep = df["n_peaks"].to_numpy() >= cfg.min_peaks
    counts["after_min_peaks"] = int(keep.sum())
    m = df["mono_mass"].to_numpy(np.float64)
    keep &= (m >= cfg.fpm_mass_min) & (m <= cfg.fpm_mass_max)
    counts["after_mass"] = int(keep.sum())
    keep &= df["adduct_known"].fillna(False).to_numpy(bool)
    counts["after_adduct"] = int(keep.sum())
    err = df["precursor_error_ppm"].to_numpy(np.float64)
    keep &= ~(np.abs(err) > cfg.fpm_max_precursor_error_ppm)
    counts["after_ppm"] = int(keep.sum())
    keep &= df["rdkit_ok"].fillna(False).to_numpy(bool) & df["smiles"].notna().to_numpy()
    counts["after_rdkit"] = int(keep.sum())
    df = df[keep].reset_index(drop=True)
    excl = training_exclusion(holdout, train_dir / "structures.parquet")
    df = excl.filter(df, ik14_col="ik14", mkey_col="mkey").reset_index(drop=True)
    counts["after_holdout_exclusion"] = int(len(df))
    counts["holdout_ik14_excluded"] = int(len(excl.ik14))
    log.info("training spectra: %s (%.0f s)", counts, time.time() - t0)
    return df, counts


# --------------------------------------------------------------------------------------
# negatives
# --------------------------------------------------------------------------------------

def negatives(st_mass, st_uni, st_hash, uni_mass, uni_hash, ppm, min_da, n_max, n_min, seed):
    """``casmi.encoder.same_mass_negatives`` (kept here under its original name)."""
    return E.same_mass_negatives(st_mass, st_uni, st_hash, uni_mass, uni_hash, ppm, min_da,
                                 n_max, n_min, seed)


# --------------------------------------------------------------------------------------
# evaluation packs
# --------------------------------------------------------------------------------------

def eval_pack(name: str, cfg: Config, val_dir: Path, np_ik14: np.ndarray, np_mass: np.ndarray,
              truth_keys: pd.DataFrame | None) -> E.EvalPack:
    from casmi.fpmodel import spectra_from_frame
    from casmi.io import mass_tables
    from casmi.validation import load_queries

    q, truth = load_queries(name, val_dir)
    q = q.sort_values(["molecule_id", "spectrum_id"], kind="mergesort").reset_index(drop=True)
    store, meta = spectra_from_frame(q, cfg)
    mols = mass_tables(q, cfg)[0].sort_values("molecule_id", kind="mergesort").reset_index(drop=True)
    code = {m: i for i, m in enumerate(mols["molecule_id"].astype(str))}
    q_mol = np.array([code[str(m)] for m in q["molecule_id"]], np.int32)
    tr = truth.set_index("molecule_id")
    keys_by_mkey: dict[str, set] = {}
    if truth_keys is not None and len(truth_keys):
        for mk, ks in zip(truth_keys["mkey"].astype(str), truth_keys["db_ik14"].astype(str)):
            keys_by_mkey.setdefault(mk, set()).update(k for k in ks.split(";") if k)
    pools, poss, rows = [], [], []
    for mid, M in zip(mols["molecule_id"].astype(str), mols["M"].to_numpy(np.float64)):
        t = tr.loc[mid]
        tol = max(cfg.mass_tol_ppm * 1e-6 * M, cfg.mass_tol_min_da)
        a, b = np.searchsorted(np_mass, [M - tol, M + tol + 1e-12])
        pool = np.arange(a, b, dtype=np.int64)
        group = t["ik14_group"]
        keys = set(group.split(";")) if isinstance(group, str) else set(map(str, group))
        keys.add(str(t["ik14"]))
        keys |= keys_by_mkey.get(str(t["mkey"]), set())
        pos = np.isin(np_ik14[pool], list(keys))
        pools.append(pool.astype(np.int32))
        poss.append(pos)
        rows.append({"molecule_id": mid, "mkey": str(t["mkey"]), "ik14": str(t["ik14"]), "M": float(M),
                     "mono_mass": float(t["mono_mass"]), "n_pool": int(len(pool)), "n_pos": int(pos.sum()),
                     "pubmed_count": t.get("pubmed_count"), "mass_tercile": t.get("mass_tercile"),
                     "fold": int(t["fold"])})
    off = np.zeros(len(pools) + 1, np.int64)
    np.cumsum([len(p) for p in pools], out=off[1:])
    qd = {"q_mz": store.mz.astype(np.float32), "q_int": store.intensity.astype(np.float16),
          "q_off": store.offsets.astype(np.int64), "q_prec": meta.precursor_mz.astype(np.float64),
          "q_ce": meta.collision_energy.astype(np.float32), "q_ne": meta.n_energies.astype(np.int8),
          "q_add": meta.adduct_code.astype(np.int8), "q_pol": meta.polarity.astype(np.int8),
          "q_inst": meta.instrument_code.astype(np.int8), "q_mol": q_mol}
    mol = pd.DataFrame(rows)
    mol["pubmed_count"] = pd.to_numeric(mol["pubmed_count"], errors="coerce")
    pack = E.EvalPack(name, qd, mol, off, np.concatenate(pools) if pools else np.zeros(0, np.int32),
                      np.concatenate(poss) if poss else np.zeros(0, bool))
    log.info("eval pack %s: %d molecules, %d spectra, median pool %.0f, truth in pool %.3f",
             name, len(mol), len(q), float(np.median(mol["n_pool"])) if len(mol) else 0,
             float((mol["n_pos"] > 0).mean()) if len(mol) else 0)
    return pack


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def build(cfg: Config, out: Path, n_jobs: int, lib_dir: Path, train_dir: Path, holdout: Path,
          np_dir: Path, val_dir: Path) -> dict:
    from casmi.chem import morgan_fps
    from casmi.spectra import SpectrumStore

    t0 = time.time()
    df, counts = load_training_table(cfg, lib_dir, train_dir, holdout)
    # structures
    ik_sorted = np.array(sorted(set(df["ik14"].astype(str))), dtype=object)
    st_code = {k: i for i, k in enumerate(ik_sorted)}
    df["st"] = df["ik14"].astype(str).map(st_code).to_numpy(np.int64)
    first = df.drop_duplicates("ik14").set_index("ik14")
    st_smiles = first.loc[ik_sorted, "smiles"].astype(object).tolist()
    st_mkey = first.loc[ik_sorted, "mkey"].astype(object).tolist()
    t1 = time.time()
    st_fp, ok = morgan_fps(st_smiles, cfg, n_jobs=n_jobs)
    log.info("targets: %d structures, %d ok, %.0f s", len(st_fp), int(ok.sum()), time.time() - t1)
    if not ok.all():
        bad = set(ik_sorted[~ok])
        df = df[~df["ik14"].isin(bad)].reset_index(drop=True)
        log.warning("dropped %d structures whose fingerprint failed", len(bad))
    st_mass = pq.read_table(train_dir / "structures.parquet", columns=["ik14", "mono_mass"]).to_pandas() \
        .set_index("ik14").loc[ik_sorted, "mono_mass"].to_numpy(np.float64)
    # groups and spectrum order
    df["g_key"] = df.groupby(["st", "ingest_lib", "polarity", "adduct_family", "instrument_family"],
                             sort=True, dropna=False).ngroup()
    df["shuffle"] = hash_key(df["spectrum_id"].astype(str).tolist(), f"{cfg.seed}|member")
    df = df.sort_values(["g_key", "shuffle"], kind="mergesort").reset_index(drop=True)
    gk = df["g_key"].to_numpy(np.int64)
    bounds = np.flatnonzero(np.r_[True, gk[1:] != gk[:-1], True])
    grp_off = bounds.astype(np.int64)
    gfirst = df.iloc[bounds[:-1]]
    libs = sorted(set(df["ingest_lib"].astype(str)))
    lib_code = {l: i for i, l in enumerate(libs)}
    grp_lib_names = gfirst["ingest_lib"].astype(str).to_numpy(object)
    grp_struct = gfirst["st"].to_numpy(np.int64)
    grp_w = E.lib_weights(grp_lib_names, grp_struct, cfg.enc_lib_shares)
    share = pd.Series(grp_w).groupby(grp_lib_names).sum().round(4).to_dict()
    log.info("groups: %d; library sample shares %s", len(grp_struct), share)
    # peaks
    store = SpectrumStore.load(lib_dir)
    sub = store.take(df["spec_idx"].to_numpy(np.int64))
    pol = df["polarity"].to_numpy(np.int64)
    s = {"spec_prec": df["precursor_mz"].to_numpy(np.float64),
         "spec_mass": np.where(np.isfinite(df["m_spec"].to_numpy(np.float64)),
                               df["m_spec"].to_numpy(np.float64), df["mono_mass"].to_numpy(np.float64)),
         "spec_ce": df["collision_energy"].to_numpy(np.float32),
         "spec_ne": np.clip(df["n_energies"].fillna(0).to_numpy(np.int64), 0, 100).astype(np.int8),
         "spec_add": adduct_codes(df["adduct"].astype(object).to_numpy(), pol),
         "spec_pol": np.sign(pol).astype(np.int8),
         "spec_inst": instrument_codes(df["instrument_family"].astype(object).to_numpy()),
         "spec_lib": df["ingest_lib"].astype(str).map(lib_code).to_numpy(np.int8),
         "spec_sim": np.zeros(len(df), np.int8),
         "spec_struct": df["st"].to_numpy(np.int32)}
    g = {"grp_off": grp_off, "grp_mem": np.arange(len(df), dtype=np.int32),
         "grp_struct": grp_struct.astype(np.int32), "grp_w": grp_w.astype(np.float64),
         "grp_lib": pd.Series(grp_lib_names).map(lib_code).to_numpy(np.int8)}
    # universe and negatives
    npt = pq.read_table(np_dir / "np.parquet", columns=["ik14", "mono_mass"]).to_pandas()
    np_ik14 = npt["ik14"].astype(str).to_numpy(object)
    np_mass = npt["mono_mass"].to_numpy(np.float64)
    np_fp = np.load(np_dir / "np_fp.npy", mmap_mode="r")
    n_np = len(npt)
    np_row = pd.Series(np.arange(n_np), index=np_ik14)
    in_np = pd.Series(ik_sorted).isin(np_row.index).to_numpy()
    extra = np.flatnonzero(~in_np)
    st_uni = np.empty(len(ik_sorted), np.int64)
    st_uni[in_np] = np_row.loc[ik_sorted[in_np]].to_numpy(np.int64)
    st_uni[extra] = n_np + np.arange(len(extra))
    uni_extra_fp = st_fp[extra]
    t1 = time.time()
    np_hash = E.fp_hash(np.asarray(np_fp))
    st_hash = E.fp_hash(st_fp)
    uni_hash = np.concatenate([np_hash, st_hash[extra]])
    uni_mass = np.concatenate([np_mass, st_mass[extra]])
    same_fp = float((np_hash[st_uni[in_np]] == st_hash[in_np]).mean()) if in_np.any() else float("nan")
    neg_off, neg_idx, neg_stats = negatives(st_mass, st_uni, st_hash, uni_mass, uni_hash,
                                            cfg.enc_neg_window_ppm, cfg.mass_tol_min_da,
                                            cfg.enc_neg_max, cfg.enc_neg_min, cfg.seed)
    neg_stats["np_fraction"] = float((neg_idx < n_np).mean()) if len(neg_idx) else 0.0
    log.info("negatives: %s (%.0f s); target fp == NP-tier fp for %.4f of %d NP structures",
             neg_stats, time.time() - t1, same_fp, int(in_np.sum()))
    structures = pd.DataFrame({"ik14": ik_sorted.astype(str), "mkey": pd.Series(st_mkey, dtype=object).astype(str),
                               "in_np": in_np, "n_spectra": np.bincount(df["st"].to_numpy(np.int64),
                                                                        minlength=len(ik_sorted))})
    # evaluation packs
    tk_path = val_dir / "truth_keys.parquet"
    truth_keys = pd.read_parquet(tk_path) if tk_path.exists() else None
    packs = {name: eval_pack(name, cfg, val_dir, np_ik14, np_mass, truth_keys) for name in EVAL_SETS}
    # leak check: no evaluation molecule is a training structure
    for name, pack in packs.items():
        leak = set(pack.mol["mkey"]) & set(structures["mkey"]) | set(pack.mol["ik14"]) & set(structures["ik14"])
        if leak:
            raise AssertionError(f"{name}: {len(leak)} evaluation molecules among training structures")
    info = {
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "git_commit": git_commit(),
        "n_np": int(n_np), "n_structures": int(len(ik_sorted)), "n_spectra": int(len(df)),
        "n_groups": int(len(grp_struct)), "n_peaks": int(len(sub.mz)), "n_extra_universe": int(len(extra)),
        "libs": libs, "adducts": list(TEST_ADDUCT_NAMES) + ["other"],
        "instrument_families": list(E.INSTRUMENT_FAMILIES),
        "library_sample_share": share, "filters": counts, "negatives": neg_stats,
        "target_fp_equals_np_fp": same_fp,
        "holdout_path": str(holdout), "holdout_sha256": sha256(holdout),
        "np_fp_sha256": json.loads((np_dir / "build_report.json").read_text()).get("np_fp", {}).get("sha256")
        if (np_dir / "build_report.json").exists() else None,
        "truth_keys_rows": int(len(truth_keys)) if truth_keys is not None else 0,
        "config": {k: v for k, v in cfg.to_dict().items() if k.startswith(("enc_", "fp_", "min_peaks",
                                                                          "fpm_mass", "fpm_max", "mass_tol",
                                                                          "seed", "top_n", "intensity_floor"))},
        "eval": {n: {"molecules": int(len(p.mol)), "spectra": int(len(p.q["q_mol"])),
                     "median_pool": float(np.median(p.mol["n_pool"])) if len(p.mol) else 0.0,
                     "truth_in_pool": float((p.mol["n_pos"] > 0).mean()) if len(p.mol) else 0.0}
                 for n, p in packs.items()},
    }
    res = E.save_bundle(out, spec_mz=sub.mz, spec_int=sub.intensity, spec_off=sub.offsets, s=s, g=g,
                        st={"st_fp": st_fp, "st_mass": st_mass, "st_uni": st_uni.astype(np.int32)},
                        structures=structures, uni_extra_fp=uni_extra_fp, neg_off=neg_off,
                        neg_idx=neg_idx, info=info, packs=packs)
    log.info("bundle %s: %.1f MB in %.0f s", out, res["total_bytes"] / 1e6, time.time() - t0)
    return res


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="YAML config (default: Config defaults)")
    p.add_argument("--out", default=None, help="bundle directory (default PROCESSED/encoder/bundle_v1)")
    p.add_argument("--n-jobs", type=int, default=2)
    p.add_argument("--holdout", default=None)
    args = p.parse_args(argv)
    log_run(log, args)
    cfg = load_config(args.config)
    out = Path(args.out) if args.out else paths.PROCESSED / "encoder" / "bundle_v1"
    holdout = Path(args.holdout) if args.holdout else paths.SPLITS / "holdout_v1.csv"
    build(cfg, out, args.n_jobs, paths.PROCESSED / "library", paths.PROCESSED / "train", holdout,
          paths.PROCESSED / "db" / "np", paths.PROCESSED / "validation")


if __name__ == "__main__":
    main()

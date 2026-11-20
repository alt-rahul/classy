"""Fragment mass calibration of train spectra against theoretical fragment masses (ROADMAP R7).

For sampled spectra of each library, the true structure's MetFrag-lite fragments
(``casmi.fragments``) are matched to the spectrum's cleaned peaks within ``--window`` Da; a
peak counts when every explanation in the window has the same theoretical ion mass
(unambiguous). Reports the signed residual (observed minus theoretical, mDa) per library and
polarity, per adduct, per m/z bin and per spectrum (which spectra carry an offset), next to the
precursor residual (m_spec - mono_mass) of the same spectra. The offset estimate is the median
of the "core" residuals (|residual| <= 3 mDa, peaks >= 5 percent of the base peak): the rest of
the window is dominated by chance assignments, whose distribution is not symmetric.

The host reported that positive-mode enveda-180 fragments read +0.49 mDa high (a calibrant
with a neutral Na mass, "mass of an electron" error, corrected in the rerun test).

Usage:
    nice -n 10 python scripts/fragment_calibration.py --per-lib 30000
    python scripts/fragment_calibration.py --libs enveda-np-examples --per-lib 2000 --no-save
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from casmi import paths
from casmi.log import get_logger, log_run, peak_rss_mb

COLS = ["ms2_mzs", "ms2_normalized_intensities", "precursor_mz", "adduct_test", "polarity",
        "normalized_smiles", "inchikey14", "mono_mass", "m_spec", "instrument_type",
        "collision_energy", "n_energies"]
MZ_BINS = (0, 100, 150, 200, 300, 400, 600, 2000)


def sample_library(lib: str, n: int, seed: int, train_dir: Path) -> pd.DataFrame:
    """Up to ``n`` spectra of ``lib`` spread over its row groups (deterministic)."""
    import pyarrow.parquet as pq

    f = pq.ParquetFile(train_dir / "spectra" / f"spectra_{lib}.parquet")
    ng = f.metadata.num_row_groups
    rng = np.random.default_rng(seed)
    per = int(np.ceil(n / ng))
    parts = []
    for g in range(ng):
        t = f.read_row_group(g, columns=COLS).to_pandas()
        t = t[t["adduct_test"].notna() & t["normalized_smiles"].notna()]
        if len(t) > per:
            t = t.iloc[np.sort(rng.choice(len(t), per, replace=False))]
        parts.append(t)
    out = pd.concat(parts, ignore_index=True)
    if len(out) > n:
        out = out.iloc[np.sort(rng.choice(len(out), n, replace=False))].reset_index(drop=True)
    out["ingest_lib"] = lib
    return out


def residuals(df: pd.DataFrame, window: float, cfg) -> pd.DataFrame:
    from casmi import fragments as F

    cache: dict[str, object] = {}
    rows = []
    for i, r in enumerate(df.itertuples(index=False)):
        smi = r.normalized_smiles
        if smi not in cache:
            cache[smi] = F.fragment_list(smi, cfg)
        fl = cache[smi]
        if fl is None:
            continue
        mode = "positive" if r.polarity > 0 else "negative"
        qs = F.prepare_query([(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz,
                               r.adduct_test, mode)], cfg, cleaned=True)
        if qs.n_peaks == 0:
            continue
        err, spread = F.peak_residuals(qs, fl, window)
        ok = np.isfinite(err) & (spread < 1e-5)
        n_ok = int(ok.sum())
        if not n_ok:
            continue
        it = np.asarray(r.ms2_normalized_intensities, dtype=np.float64)
        rows.append(pd.DataFrame({
            "spec": i, "lib": r.ingest_lib, "polarity": int(r.polarity), "adduct": r.adduct_test,
            "instrument": r.instrument_type, "ce": r.collision_energy, "mz": qs.mz[ok],
            "intensity": it[ok] if len(it) == qs.n_peaks else np.nan,
            "err_mda": 1000.0 * err[ok],
            "prec_err_mda": 1000.0 * (float(r.m_spec) - float(r.mono_mass)),
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


CORE_MDA = 3.0        # residuals inside +-3 mDa: true assignments dominate (window tails are chance)
CORE_MIN_INT = 0.05   # and peaks at >= 5 percent of the base peak


def stats(e: np.ndarray, inten: np.ndarray | None = None) -> dict:
    ok = np.isfinite(e)
    e = e[ok]
    if not len(e):
        return {"n": 0}
    keep = np.abs(e) <= CORE_MDA
    if inten is not None:
        keep &= np.nan_to_num(inten[ok], nan=1.0) >= CORE_MIN_INT
    core = e[keep]
    return {"n": int(len(e)), "median": float(np.median(e)),
            "n_core": int(len(core)),
            "core_median": float(np.median(core)) if len(core) else float("nan"),
            "core_mean": float(core.mean()) if len(core) else float("nan"),
            "core_se": float(1.2533 * core.std() / np.sqrt(len(core))) if len(core) > 1 else float("nan"),
            "iqr": float(np.subtract(*np.percentile(e, [75, 25]))),
            "p10": float(np.percentile(e, 10)), "p90": float(np.percentile(e, 90))}


def _st(g: pd.DataFrame) -> dict:
    return stats(g["err_mda"].to_numpy(), g["intensity"].to_numpy(dtype=np.float64))


def summarise(res: pd.DataFrame) -> dict:
    out: dict = {"by_lib_polarity": {}, "by_adduct": {}, "by_mz": {}, "per_spectrum": {},
                 "precursor": {}}
    for (lib, pol), g in res.groupby(["lib", "polarity"], sort=True):
        key = f"{lib}|{'+' if pol > 0 else '-'}"
        out["by_lib_polarity"][key] = _st(g)
        out["by_lib_polarity"][key]["n_spectra"] = int(g["spec"].nunique())
        sp = g.groupby("spec")
        prec = sp["prec_err_mda"].first().to_numpy()
        out["precursor"][key] = stats(prec)
        gc = g[(g["err_mda"].abs() <= CORE_MDA)]
        per = gc.groupby("spec")["err_mda"].agg(["median", "size"])
        per = per[per["size"] >= 5]["median"].to_numpy()
        if len(per):
            out["per_spectrum"][key] = {
                "n_spectra_ge5_peaks": int(len(per)),
                "quantiles_mda": [float(x) for x in np.percentile(per, [5, 25, 50, 75, 95])],
                "frac_gt_0.25": float((per > 0.25).mean()),
                "frac_lt_-0.25": float((per < -0.25).mean()),
            }
        for (add,), ga in g.groupby(["adduct"], sort=True):
            out["by_adduct"][f"{key}|{add}"] = _st(ga)
        bins = pd.cut(g["mz"], MZ_BINS, right=False)
        for b, gb in g.groupby(bins, observed=True, sort=True):
            out["by_mz"][f"{key}|{b.left:g}-{b.right:g}"] = _st(gb)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--libs", nargs="+",
                    default=["enveda-180", "enveda-np-examples", "pluskal_ms2", "gnps", "massbank",
                             "riken"])
    ap.add_argument("--per-lib", type=int, default=30000, help="spectra sampled per library")
    ap.add_argument("--window", type=float, default=0.01, help="explanation window (Da)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--train-dir", type=Path, default=paths.PROCESSED / "train")
    ap.add_argument("--out", type=Path, default=paths.PROCESSED / "fragments")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)
    lg = get_logger("fragment_calibration")
    log_run(lg, args)
    from casmi.config import Config
    from casmi import fragments as F

    cfg = Config()
    seed = cfg.seed if args.seed is None else args.seed
    F.warmup()
    parts = []
    for lib in args.libs:
        t0 = time.perf_counter()
        df = sample_library(lib, args.per_lib, seed, args.train_dir)
        r = residuals(df, args.window, cfg)
        lg.info("%s: %d spectra sampled, %d unambiguous peaks, %.1f s, peak RSS %.0f MB", lib,
                len(df), len(r), time.perf_counter() - t0, peak_rss_mb())
        parts.append(r)
        del df
    res = pd.concat([p for p in parts if len(p)], ignore_index=True)
    summ = summarise(res)
    summ["settings"] = {"window_da": args.window, "per_lib": args.per_lib, "seed": seed,
                        "libs": args.libs}
    for k, v in summ["by_lib_polarity"].items():
        p = summ["precursor"][k]
        ps = summ["per_spectrum"].get(k, {})
        lg.info("%-26s peaks %7d spectra %6d | fragment residual core median %+.3f mDa (se %.3f, "
                "n %d; all-window median %+.3f) | precursor median %+.3f mDa | per-spectrum "
                "medians p25/p50/p75 %s", k, v["n"], v["n_spectra"], v["core_median"],
                v["core_se"], v["n_core"], v["median"], p["median"],
                [round(x, 2) for x in ps.get("quantiles_mda", [])[1:4]])
    for k, v in summ["by_mz"].items():
        if k.startswith("enveda"):
            lg.info("  m/z %-34s n %7d core median %+.3f", k, v["n_core"], v["core_median"])
    for k, v in summ["by_adduct"].items():
        if k.startswith("enveda"):
            lg.info("  adduct %-40s n %7d core median %+.3f", k, v["n_core"], v["core_median"])
    if not args.no_save:
        paths.ensure_dir(args.out)
        res.to_parquet(args.out / "calibration_residuals.parquet", index=False)
        (args.out / "calibration.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
        lg.info("wrote %s", args.out / "calibration.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Synthetic extreme test file for the R9 robustness checks (ROADMAP R9, KAGGLE_INTEL 5).

    python scripts/make_extreme_test.py                       # -> data/processed/extreme_test/
    python scripts/make_extreme_test.py --replicate-visible 2 # + 2 renamed copies of the visible test

Writes ``test.parquet`` with exactly the schema of the competition's ``test.parquet`` (so
``io.read_test`` and the notebook read it unchanged) and ``cases.json`` (one entry per
molecule: what it tests). Spectra are built from the visible test (real timsTOF peak
lists as templates, re-labelled and moved to new masses) plus synthetic peaks where a
case needs them. Deterministic (``Config.seed``).

Cases (molecule_id prefix ``x_``):

- ``x_big_1158``: a 1,158.47 Da acylated glycoside (C54H78O27, in the NP tier) with
  sugar and tigloyl losses, [M+Na]+, [M+NH4]+ and [M-H]-; ``x_big_1159``: the same
  spectra moved to 1,159.00 Da (no NP-tier row, a sparse PubChem pool).
- ``x_16_spectra``: one molecule with 16 spectra (4 adducts x 3 energies + 4 merged).
- ``x_adduct_<i>``: one molecule per test adduct (all ten), each from its own template.
- ``x_nodb_300``: M = 300.550 Da (a mass defect no molecule in either tier has; checked: no
  row within 20 ppm from 300.34 to 300.73 Da): empty pools in every tier, so the whole
  padding path runs. (PubChem has rows within 20 ppm of every 0.01 Da step from 1,158.3 to
  1,160 Da, so a no-database molecule cannot be made at 1,159 Da.)
- ``x_above_db``: M = 1,420 Da (above ``db_mass_max``); ``x_low_mass``: M = 102 Da.
- ``x_one_peak``: a single spectrum with one peak; ``x_three_peaks``: a single 3-peak spectrum.
- ``x_zero_peaks``: a spectrum with no peaks next to a normal one.
- ``x_peaks_6000``: a spectrum with 6,000 peaks (plus a normal one); ``x_peaks_3500_only``:
  a single 3,500-peak spectrum.
- ``x_merged``: merged-energy spectra only (``collision_energy_ev`` lists of 3 and 5 values,
  ``"20,40,60"`` / ``"-60,-40,-20"`` / 5 energies), and one with an empty energy list.
- ``x_neg_only``: negative mode only ([M-H]-, [M+Cl]-, [M+CH2O2-H]-, [M-H2O-H]-).
- ``x_both_pol``: positive and negative spectra whose masses disagree by 40 ppm.
- ``x_dup_a`` / ``x_dup_b``: two molecules with identical spectra.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from casmi import log as clog  # noqa: E402
from casmi import paths  # noqa: E402
from casmi.adducts import TEST_ADDUCTS, formula_mass, precursor_mz  # noqa: E402
from casmi.config import Config  # noqa: E402

HEXOSE = 162.052823   # C6H10O5
TIGLOYL = 82.041865   # C5H6O (tigloyl ester loss)
WATER = 18.010565


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test", type=Path, default=paths.TEST_PARQUET, help="template (visible test)")
    p.add_argument("--out-dir", type=Path, default=paths.PROCESSED / "extreme_test")
    p.add_argument("--replicate-visible", type=int, default=0,
                   help="also append N renamed copies of the visible test (runtime at scale)")
    return p.parse_args(argv)


class Maker:
    def __init__(self, tmpl: pd.DataFrame, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.tmpl = tmpl
        self.rows: list[dict] = []
        self.manifest: dict[str, str] = {}
        self.n = 0
        # templates: spectra with at least 20 peaks, by polarity, deterministic order
        npk = tmpl["ms2_mzs"].map(len)
        self.pos = tmpl[(tmpl["ionization_mode"] == "positive") & (npk >= 20)].reset_index(drop=True)
        self.neg = tmpl[(tmpl["ionization_mode"] == "negative") & (npk >= 20)].reset_index(drop=True)
        self.k_pos = 0
        self.k_neg = 0

    def template(self, mode: str) -> pd.Series:
        if mode == "negative":
            r = self.neg.iloc[(self.k_neg * 7) % len(self.neg)]
            self.k_neg += 1
        else:
            r = self.pos.iloc[(self.k_pos * 13) % len(self.pos)]
            self.k_pos += 1
        return r

    def add(self, mid: str, adduct: str, M: float, mzs: np.ndarray, ints: np.ndarray,
            energies: list[float], what: str | None = None) -> None:
        a = TEST_ADDUCTS[adduct]
        mode = "positive" if a.charge > 0 else "negative"
        mzs = np.asarray(mzs, dtype=np.float64)
        ints = np.asarray(ints, dtype=np.float64)
        if len(ints):
            ints = ints / ints.max()
        order = np.argsort(mzs, kind="mergesort")
        sign = 1 if mode == "positive" else -1
        orig = ",".join(str(int(sign * e)) for e in (energies if sign > 0 else sorted(energies, reverse=True)))
        self.n += 1
        self.rows.append({
            "molecule_id": mid, "spectrum_id": f"xs_{self.n:05d}",
            "ms2_mzs": [float(x) for x in mzs[order]],
            "ms2_normalized_intensities": [float(x) for x in ints[order]],
            "base_peak_intensity": float(self.rng.integers(10 ** 4, 10 ** 7)),
            "adduct": adduct, "ionization_mode": mode, "instrument_type": "timsTOF",
            "precursor_mz": round(float(precursor_mz(M, adduct)), 4),
            "collision_energy_orig": orig, "collision_energy_ev": [float(e) for e in energies],
            "collision_energy_orig_units": "eV",
        })
        if what:
            self.manifest[mid] = what

    def moved(self, r: pd.Series, M: float, adduct: str) -> tuple[np.ndarray, np.ndarray]:
        """Template peaks shifted so the fragments sit below the new precursor."""
        mz = np.asarray(r["ms2_mzs"], dtype=np.float64)
        it = np.asarray(r["ms2_normalized_intensities"], dtype=np.float64)
        new_prec = precursor_mz(M, adduct)
        shift = new_prec - float(r["precursor_mz"])
        mz = mz + shift if shift < 0 else mz  # heavier targets keep the template's low fragments
        keep = (mz > 30) & (mz < new_prec + 1.0)
        mz, it = mz[keep], it[keep]
        mz = mz + self.rng.normal(0, 0.0005, len(mz))
        if not len(mz):
            mz, it = np.array([new_prec * 0.5]), np.array([1.0])
        return mz, it


def build(tmpl: pd.DataFrame, cfg: Config, replicate: int) -> tuple[pd.DataFrame, dict]:
    mk = Maker(tmpl, cfg.seed)
    rng = mk.rng

    # 1. large acylated glycoside (NP tier entry, 1,158.473 Da) and a no-DB twin at 1,159.0
    M_big = formula_mass("C54H78O27")
    for mid, M, what in (("x_big_1158", M_big, "1,158.47 Da glycoside C54H78O27 (NP tier)"),
                         ("x_big_1159", 1159.0, "1,159.00 Da: no NP-tier row, sparse PubChem pool")):
        for adduct in ("[M+Na]+", "[M+NH4]+", "[M-H]-"):
            prec = precursor_mz(M, adduct)
            losses = [h * HEXOSE + t * TIGLOYL + w * WATER for h in range(4) for t in range(4)
                      for w in range(2)]
            mz = np.array(sorted({prec - x for x in losses if prec - x > 150}))
            it = rng.uniform(0.05, 1.0, len(mz))
            low = rng.uniform(60, 400, 40)
            mk.add(mid, adduct, M, np.r_[mz, low], np.r_[it, rng.uniform(0.01, 0.3, 40)],
                   [40.0] if adduct != "[M-H]-" else [20.0, 40.0, 60.0], what)

    # 2. sixteen spectra for one molecule (4 adducts x 3 energies, + 4 merged)
    r = mk.template("positive")
    M16 = float(r["precursor_mz"]) - TEST_ADDUCTS["[M+H]+"].shift
    for adduct in ("[M+H]+", "[M+Na]+", "[M+NH4]+", "[M-H2O+H]+"):
        for e in (20.0, 40.0, 60.0):
            mk.add("x_16_spectra", adduct, M16, *mk.moved(r, M16, adduct), [e],
                   "16 spectra: 4 adducts x 3 energies + 4 merged")
        mk.add("x_16_spectra", adduct, M16, *mk.moved(r, M16, adduct), [20.0, 40.0, 60.0])

    # 3. every test adduct, one molecule each
    for i, adduct in enumerate(TEST_ADDUCTS):
        mode = "positive" if TEST_ADDUCTS[adduct].charge > 0 else "negative"
        r = mk.template(mode)
        src = TEST_ADDUCTS[str(r["adduct"])] if str(r["adduct"]) in TEST_ADDUCTS else None
        M = float(r["precursor_mz"]) - (src.shift if src else TEST_ADDUCTS["[M+H]+"].shift)
        mk.add(f"x_adduct_{i:02d}", adduct, M, *mk.moved(r, M, adduct), [40.0], f"adduct {adduct}")

    # 4. masses outside every tier
    r = mk.template("positive")
    mk.add("x_nodb_300", "[M+H]+", 300.550, *mk.moved(r, 300.550, "[M+H]+"), [40.0],
           "M 300.550 Da: no row in any tier within 20 ppm, empty pools")
    mk.add("x_above_db", "[M+H]+", 1420.0, *mk.moved(mk.template("positive"), 1420.0, "[M+H]+"), [60.0],
           "M 1,420 Da: above db_mass_max")
    mk.add("x_low_mass", "[M+H]+", 102.068, np.array([45.03, 57.07, 85.06]),
           np.array([0.3, 1.0, 0.5]), [20.0], "M 102 Da: smallest molecules")

    # 5. peak-count extremes
    r = mk.template("positive")
    M = float(r["precursor_mz"]) - TEST_ADDUCTS["[M+H]+"].shift
    mk.add("x_one_peak", "[M+H]+", M, np.array([M * 0.6]), np.array([1.0]), [40.0], "single 1-peak spectrum")
    mk.add("x_three_peaks", "[M+H]+", M + 14.01565, np.array([91.0542, 119.0491, M * 0.7]),
           np.array([1.0, 0.4, 0.2]), [40.0], "single 3-peak spectrum")
    mk.add("x_zero_peaks", "[M+H]+", M + 28.0313, np.zeros(0), np.zeros(0), [40.0],
           "a spectrum with no peaks plus a normal one")
    mk.add("x_zero_peaks", "[M+H]+", M + 28.0313, *mk.moved(r, M + 28.0313, "[M+H]+"), [20.0])
    prec = precursor_mz(M + 2.01565, "[M+H]+")
    mk.add("x_peaks_6000", "[M+H]+", M + 2.01565, rng.uniform(40, prec, 6000),
           rng.pareto(2.0, 6000) + 0.001, [20.0, 40.0, 60.0], "6,000-peak spectrum plus a normal one")
    mk.add("x_peaks_6000", "[M+H]+", M + 2.01565, *mk.moved(r, M + 2.01565, "[M+H]+"), [40.0])
    prec = precursor_mz(M + 15.9949, "[M-H]-")
    mk.add("x_peaks_3500_only", "[M-H]-", M + 15.9949, rng.uniform(40, prec, 3500),
           rng.pareto(2.0, 3500) + 0.001, [40.0], "single 3,500-peak spectrum")

    # 6. merged energies only
    r = mk.template("positive")
    M = float(r["precursor_mz"]) - TEST_ADDUCTS["[M+H]+"].shift
    mk.add("x_merged", "[M+H]+", M, *mk.moved(r, M, "[M+H]+"), [20.0, 40.0, 60.0],
           "merged-energy spectra only (3 and 5 energies, negative, empty list)")
    mk.add("x_merged", "[M+H]+", M, *mk.moved(r, M, "[M+H]+"), [10.0, 20.0, 30.0, 40.0, 50.0])
    mk.add("x_merged", "[M-H]-", M, *mk.moved(mk.template("negative"), M, "[M-H]-"), [20.0, 40.0, 60.0])
    mk.add("x_merged", "[M+Na]+", M, *mk.moved(r, M, "[M+Na]+"), [])

    # 7. negative mode only
    r = mk.template("negative")
    M = float(r["precursor_mz"]) + 1.007276
    for adduct, e in (("[M-H]-", [20.0]), ("[M-H]-", [40.0]), ("[M+Cl]-", [40.0]),
                      ("[M+CH2O2-H]-", [60.0]), ("[M-H2O-H]-", [20.0, 40.0, 60.0])):
        mk.add("x_neg_only", adduct, M, *mk.moved(r, M, adduct), e, "negative mode only, 4 adducts")

    # 8. both polarities, masses 40 ppm apart (cluster logic)
    r = mk.template("positive")
    M = float(r["precursor_mz"]) - TEST_ADDUCTS["[M+H]+"].shift
    mk.add("x_both_pol", "[M+H]+", M, *mk.moved(r, M, "[M+H]+"), [40.0], "both polarities, M 40 ppm apart")
    mk.add("x_both_pol", "[M-H]-", M * (1 + 40e-6), *mk.moved(mk.template("negative"), M, "[M-H]-"), [40.0])

    # 9. two molecules with identical spectra
    r = mk.template("positive")
    M = float(r["precursor_mz"]) - TEST_ADDUCTS["[M+H]+"].shift
    mz, it = mk.moved(r, M, "[M+H]+")
    for mid in ("x_dup_a", "x_dup_b"):
        mk.add(mid, "[M+H]+", M, mz, it, [40.0], "identical spectra in two molecules")

    df = pd.DataFrame(mk.rows)
    if replicate > 0:
        reps = []
        for k in range(replicate):
            t = tmpl.copy()
            t["molecule_id"] = [f"v{k}_{m}" for m in t["molecule_id"]]
            t["spectrum_id"] = [f"v{k}_{s}" for s in t["spectrum_id"]]
            reps.append(t)
        df = pd.concat([df] + reps, ignore_index=True)
        mk.manifest["v<k>_*"] = f"{replicate} renamed copies of the visible test"
    return df, mk.manifest


def main(argv: list[str] | None = None) -> int:
    a = _parse(argv)
    lg = clog.get_logger("make_extreme_test")
    clog.log_run(lg, a)
    cfg = Config()
    tmpl = pd.read_parquet(a.test)
    schema = pq.read_schema(a.test)
    df, manifest = build(tmpl, cfg, a.replicate_visible)
    df = df[list(schema.names)]
    tbl = pa.Table.from_pandas(df, schema=schema.remove_metadata(), preserve_index=False)
    out = paths.ensure_dir(a.out_dir)
    pq.write_table(tbl, out / "test.parquet")
    n_mol = df["molecule_id"].nunique()
    npk = df["ms2_mzs"].map(len)
    info = {"n_spectra": len(df), "n_molecules": n_mol,
            "max_spectra_per_molecule": int(df.groupby("molecule_id").size().max()),
            "adducts": sorted(df["adduct"].unique().tolist()),
            "peaks_min": int(npk.min()), "peaks_max": int(npk.max()), "cases": manifest}
    (out / "cases.json").write_text(json.dumps(info, indent=1) + "\n")
    lg.info("wrote %s: %d spectra, %d molecules, peaks %d to %d, adducts %d",
            out / "test.parquet", len(df), n_mol, info["peaks_min"], info["peaks_max"],
            len(info["adducts"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

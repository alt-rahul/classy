"""Synthetic ``train.parquet`` for code-phase runs of WP6 and WP7 (before the real file lands).

Writes ``data/processed/synthetic/train_synth.parquet`` (default) with the exact train schema
(docs/COMPETITION.md: the 12 test columns plus normalized_smiles, inchikey, inchikey14,
molecular_formula, ingest_lib, adduct_orig, precursor_error_ppm, num_peaks) and the Arrow
types of ``test.parquet`` (large_string, list<double>, double; num_peaks int64).

Construction (deterministic, ``Config.seed``):
- Labels: each public test molecule gets a real NP-tier structure whose RDKit mass is within
  5 ppm of the molecule's neutral mass (closest first); molecules without one are skipped.
  A second structure in the same window becomes a "decoy" label for a few jittered copies,
  so library mass windows hold isomers. A few labels are also shipped as a different
  tautomer (different inchikey14, same metric key) in another library.
- Libraries (names from COMPETITION.md): exact copies of the public test spectra go to
  enveda-180 or enveda-np-examples (half keep the test spectrum_id, the rest get a new id;
  some get extra peaks above precursor + 2 Da, which the test cleaning removed), plus
  jittered copies (m/z jitter, intensity noise, dropped and added peaks) in the other
  libraries, with library-typical instrument text, CE units (eV, NCE, V, unknown, none)
  and null base_peak_intensity where the real library ships normalised intensities.
- Adducts: the ten test adducts plus other spellings, multimers, doubly charged and
  other adducts; precursor m/z is computed from the label mass, so precursor_error_ppm is
  realistic.
- Deliberately bad rows: huge precursor error, unparseable adduct, no polarity, null
  label, null or mismatched peak arrays, empty spectrum, NaN precursor, duplicated
  spectrum_id.
- Rows are shuffled and written in row groups of ``--row-group-size`` so streaming code
  paths see several row groups.

Usage:
    python scripts/make_synthetic_train.py [--out PATH] [--scale K] [--row-group-size N]

``--scale K`` adds K extra rounds of jittered replicate copies (new spectrum ids, same
labels), for throughput tests.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from casmi import paths
from casmi.config import load_config
from casmi.log import get_logger, log_run

TRAIN_COLUMNS = (
    "molecule_id", "spectrum_id", "ms2_mzs", "ms2_normalized_intensities",
    "base_peak_intensity", "adduct", "ionization_mode", "instrument_type", "precursor_mz",
    "collision_energy_orig", "collision_energy_ev", "collision_energy_orig_units",
    "normalized_smiles", "inchikey", "inchikey14", "molecular_formula", "ingest_lib",
    "adduct_orig", "precursor_error_ppm", "num_peaks",
)

TRAIN_SCHEMA = pa.schema([
    ("molecule_id", pa.large_string()),
    ("spectrum_id", pa.large_string()),
    ("ms2_mzs", pa.list_(pa.float64())),
    ("ms2_normalized_intensities", pa.list_(pa.float64())),
    ("base_peak_intensity", pa.float64()),
    ("adduct", pa.large_string()),
    ("ionization_mode", pa.large_string()),
    ("instrument_type", pa.large_string()),
    ("precursor_mz", pa.float64()),
    ("collision_energy_orig", pa.large_string()),
    ("collision_energy_ev", pa.list_(pa.float64())),
    ("collision_energy_orig_units", pa.large_string()),
    ("normalized_smiles", pa.large_string()),
    ("inchikey", pa.large_string()),
    ("inchikey14", pa.large_string()),
    ("molecular_formula", pa.large_string()),
    ("ingest_lib", pa.large_string()),
    ("adduct_orig", pa.large_string()),
    ("precursor_error_ppm", pa.float64()),
    ("num_peaks", pa.int64()),
])

PUBLIC_NP_LIBS = ("gnps", "massbank", "mona", "riken", "msdial")

# Per library: instrument texts (None = null), CE convention, whether bpi is shipped.
LIB_STYLE: dict[str, dict[str, Any]] = {
    "enveda-180": {"inst": ["timsTOF"], "ce": "ev_list", "bpi": True},
    "enveda-np-examples": {"inst": ["timsTOF"], "ce": "ev_list", "bpi": True},
    "pluskal_ms2": {"inst": ["Orbitrap ID-X Tribrid", "Orbitrap Exploris 480", None],
                    "ce": "nce", "bpi": True},
    "gnps": {"inst": ["Orbitrap", "qTof", "ESI-QTOF", "Maxis II HD Q-TOF Bruker",
                      "LC-ESI-qTof", None], "ce": "mixed", "bpi": False},
    "massbank": {"inst": ["LC-ESI-QTOF", "LC-ESI-ITFT", "LC-ESI-QFT", "LC-ESI-QQ"],
                 "ce": "ev", "bpi": False},
    "mona": {"inst": ["Q Exactive Plus Orbitrap Thermo Scientific", "LC-ESI-QTOF",
                      "Agilent 6530 Q-TOF"], "ce": "volt", "bpi": False},
    "riken": {"inst": ["LC-ESI-QTOF", "LC-ESI-QFT", "UPLC-ESI-QTOF"], "ce": "unknown",
              "bpi": False},
    "msdial": {"inst": ["Agilent 6530", "Sciex TripleTOF 6600", "Thermo Q Exactive HF"],
               "ce": "ev", "bpi": False},
    "spectraverse": {"inst": ["Thermo Q Exactive HF", "Waters Xevo G2-XS QTof", None],
                     "ce": "mixed", "bpi": True},
    "drug_plus": {"inst": [None], "ce": "none", "bpi": False},
    "masaryk": {"inst": ["Orbitrap Exploris 240"], "ce": "nce", "bpi": True},
}

# Alternative adducts for jittered copies: (adduct as harmonised, adduct_orig as recorded).
POS_ALT = [("[M+Na]+", "M+Na"), ("[M+NH4]+", "[M+NH4]+"), ("[M+H-H2O]+", "M+H-H2O"),
           ("[2M+H]+", "[2M+H]+"), ("[M+2H]2+", "[M+2H]2+"), ("[M+ACN+H]+", "M+ACN+H"),
           ("[M+K]+", "[M+K]+"), ("[M+H]+", "M+H"), ("[M-2H2O+H]+", "[M-2H2O+H]+"),
           ("[2M+Na]+", "2M+Na")]
NEG_ALT = [("[M+FA-H]-", "M+FA-H"), ("[2M-H]-", "[2M-H]-"), ("[M+Cl]-", "M+Cl"),
           ("[M-H2O-H]-", "[M-H2O-H]-"), ("[M-H]-", "M-H"), ("[M+Hac-H]-", "[M+CH3COO]-")]


def _ms(x: Any) -> np.ndarray:
    return np.asarray(x if x is not None else [], dtype=np.float64)


class Maker:
    def __init__(self, cfg, seed: int, scale: int):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.scale = scale
        self.rows: list[dict[str, Any]] = []
        self.n_new_ids = 0

    # ---------------------------------------------------------------- labels
    def labels(self, test: pd.DataFrame, lg) -> pd.DataFrame:
        from rdkit import Chem
        from rdkit.Chem.MolStandardize import rdMolStandardize

        from casmi import db
        from casmi.chem import formula, mol_from_smiles, mono_mass, std_mol
        from casmi.io import molecule_table
        from casmi.metric import ik14, mkey

        mols = molecule_table(test, self.cfg)
        res = db.lookup_batch(mols["M"].to_numpy(), tol_ppm=5.0, tiers=("np",),
                              min_tol_da=0.0005, keys=mols["molecule_id"].tolist())
        rows = []
        used: set[str] = set()
        for mid, grp in res.groupby("query", sort=True):
            g = grp.assign(absppm=grp["ppm_err"].abs()).sort_values(["absppm", "ik14"],
                                                                     kind="mergesort")
            picked = []
            for smi, key in zip(g["smiles"], g["ik14"]):
                if key in used:
                    continue
                m = std_mol(smi)
                if m is None:
                    continue
                norm = Chem.MolToSmiles(m)
                ikey = Chem.MolToInchiKey(m)
                if not ikey or ikey[:14] in used:
                    continue
                picked.append((norm, ikey, formula(m), mono_mass(m)))
                used.add(ikey[:14])
                if len(picked) == 2:
                    break
            if not picked:
                continue
            rec = {"molecule_id": mid, "smiles": picked[0][0], "inchikey": picked[0][1],
                   "formula": picked[0][2], "mono_mass": picked[0][3]}
            if len(picked) > 1:
                rec.update(decoy_smiles=picked[1][0], decoy_inchikey=picked[1][1],
                           decoy_formula=picked[1][2], decoy_mass=picked[1][3])
            rows.append(rec)
        lab = pd.DataFrame(rows)
        lg.info("labels: %d of %d test molecules got an NP-tier structure within 5 ppm "
                "(%d with a decoy)", len(lab), len(mols), lab["decoy_smiles"].notna().sum())
        # Tautomer variants (different ik14, same metric key).
        te = rdMolStandardize.TautomerEnumerator()
        taut = {}
        for i in self.rng.permutation(len(lab)):
            if len(taut) >= 8:
                break
            s = lab["smiles"].iloc[i]
            m = mol_from_smiles(s)
            if m is None or m.GetNumAtoms() > 40:
                continue
            try:
                variants = sorted({Chem.MolToSmiles(t) for t in te.Enumerate(m)})
            except Exception:  # noqa: BLE001
                continue
            k0, mk0 = ik14(s), mkey(s)
            for v in variants:
                kv = ik14(v)
                if kv and kv != k0 and kv not in used and mkey(v) == mk0:
                    taut[lab["molecule_id"].iloc[i]] = (v, Chem.MolToInchiKey(mol_from_smiles(v)))
                    used.add(kv)
                    break
        lab["taut_smiles"] = lab["molecule_id"].map(lambda x: taut.get(x, (None, None))[0])
        lab["taut_inchikey"] = lab["molecule_id"].map(lambda x: taut.get(x, (None, None))[1])
        lg.info("tautomer variants: %d", len(taut))
        return lab

    # ---------------------------------------------------------------- spectra
    def _new_id(self) -> str:
        self.n_new_ids += 1
        return f"sp_{self.n_new_ids:07d}"

    def _jitter(self, mz: np.ndarray, it: np.ndarray, strength: float = 1.0
                ) -> tuple[np.ndarray, np.ndarray]:
        r = self.rng
        keep = r.random(mz.size) > 0.15 * strength
        keep[np.argmax(it)] = True
        mz2 = mz[keep] + r.normal(0.0, 0.002 * strength, keep.sum())
        it2 = it[keep] * np.exp(r.normal(0.0, 0.3 * strength, keep.sum()))
        n_noise = int(r.integers(0, 8))
        if n_noise and mz.size:
            lo, hi = float(mz.min()), float(mz.max())
            mz2 = np.concatenate([mz2, r.uniform(lo, max(hi, lo + 1), n_noise)])
            it2 = np.concatenate([it2, r.uniform(0.002, 0.03, n_noise) * it2.max()])
        o = np.argsort(mz2, kind="mergesort")
        mz2, it2 = np.round(mz2[o], 4), it2[o]
        return mz2, it2 / it2.max()

    def _ce(self, lib: str, ev: np.ndarray, prec: float) -> tuple[Any, Any, Any]:
        """(collision_energy_orig, collision_energy_ev, units) in the library's style."""
        style = LIB_STYLE[lib]["ce"]
        r = self.rng
        if style == "ev_list":
            return ",".join(f"{v:g}" for v in ev), [float(v) for v in ev], "eV"
        if style == "none":
            return None, None, "unknown"
        val = float(r.choice([10, 20, 30, 35, 40, 45, 60]))
        if style == "nce" or (style == "mixed" and r.random() < 0.3):
            return f"{val:g}", [val * prec / 500.0], "NCE"
        if style == "unknown" or (style == "mixed" and r.random() < 0.3):
            return f"{val:g} (nominal)", None, "unknown"
        if style == "volt":
            return f"{val:g} V", [val], "V"
        return f"{val:g}", [val], "eV"

    def add(self, *, lib: str, label: dict, mz: np.ndarray, it: np.ndarray, adduct: str,
            adduct_orig: str, mode: str, ev: np.ndarray, bpi: float, molecule_id: str,
            spectrum_id: str | None = None, exact_prec: float | None = None,
            err_ppm: float | None = None) -> None:
        from casmi.adducts import parse_adduct

        a = parse_adduct(adduct, mode)
        mass = label["mono_mass"]
        if exact_prec is not None:
            prec = float(exact_prec)
        else:
            theo = a.precursor_mz(mass)
            sd = 1.5 if LIB_STYLE[lib]["inst"][0] == "timsTOF" else 3.0
            e = self.rng.normal(0.0, sd) if err_ppm is None else err_ppm
            prec = round(theo * (1 + e * 1e-6), 5)
        perr = ((prec - a.precursor_mz(mass)) / a.precursor_mz(mass) * 1e6) if a else np.nan
        inst = LIB_STYLE[lib]["inst"][int(self.rng.integers(len(LIB_STYLE[lib]["inst"])))]
        orig, ce_ev, units = self._ce(lib, ev, prec)
        self.rows.append({
            "molecule_id": molecule_id,
            "spectrum_id": spectrum_id or self._new_id(),
            "ms2_mzs": [float(x) for x in mz],
            "ms2_normalized_intensities": [float(x) for x in it],
            "base_peak_intensity": float(bpi) if LIB_STYLE[lib]["bpi"] else None,
            "adduct": adduct,
            "ionization_mode": mode,
            "instrument_type": inst,
            "precursor_mz": prec,
            "collision_energy_orig": orig,
            "collision_energy_ev": ce_ev,
            "collision_energy_orig_units": units,
            "normalized_smiles": label["smiles"],
            "inchikey": label["inchikey"],
            "inchikey14": label["inchikey"][:14] if label["inchikey"] else None,
            "molecular_formula": label["formula"],
            "ingest_lib": lib,
            "adduct_orig": adduct_orig,
            "precursor_error_ppm": float(perr),
            "num_peaks": int(len(mz)),
        })

    def _mol_id(self, lib: str, ikey: str) -> str:
        return "tm_" + hashlib.sha1(f"{lib}|{ikey}".encode()).hexdigest()[:8]

    def _alt_adduct(self, mode: str, adduct: str) -> tuple[str, str]:
        if self.rng.random() < 0.7:
            return adduct, adduct
        table = POS_ALT if mode == "positive" else NEG_ALT
        return table[int(self.rng.integers(len(table)))]

    def jittered(self, lib: str, label: dict, spec: pd.Series, strength: float = 1.0,
                 alt_adduct: bool = True) -> None:
        mz, it = self._jitter(_ms(spec["ms2_mzs"]), _ms(spec["ms2_normalized_intensities"]),
                              strength)
        adduct, orig = (self._alt_adduct(spec["ionization_mode"], spec["adduct"]) if alt_adduct
                        else (spec["adduct"], spec["adduct"]))
        ev = _ms(spec["collision_energy_ev"])
        if LIB_STYLE[lib]["ce"] == "ev_list" and self.rng.random() < 0.5:
            ev = np.array([float(self.rng.choice([20.0, 40.0, 60.0]))])
        mode = spec["ionization_mode"]
        if lib == "gnps" and self.rng.random() < 0.2:
            mode = mode.capitalize()
        bpi = float(spec["base_peak_intensity"]) * float(np.exp(self.rng.normal(0, 0.5)))
        self.add(lib=lib, label=label, mz=mz, it=it, adduct=adduct, adduct_orig=orig, mode=mode,
                 ev=ev, bpi=bpi, molecule_id=self._mol_id(lib, label["inchikey"]))

    def exact(self, lib: str, label: dict, spec: pd.Series) -> None:
        mz = _ms(spec["ms2_mzs"])
        it = _ms(spec["ms2_normalized_intensities"])
        if self.rng.random() < 0.4:  # peaks the test cleaning removed
            k = int(self.rng.integers(1, 4))
            extra = spec["precursor_mz"] + self.rng.uniform(2.5, 30.0, k)
            mz = np.concatenate([mz, np.round(extra, 4)])
            it = np.concatenate([it, self.rng.uniform(0.01, 0.2, k)])
            o = np.argsort(mz, kind="mergesort")
            mz, it = mz[o], it[o]
        sid = spec["spectrum_id"] if self.rng.random() < 0.5 else None
        self.add(lib=lib, label=label, mz=mz, it=it, adduct=spec["adduct"],
                 adduct_orig=spec["adduct"].strip("[]+-") if self.rng.random() < 0.3 else spec["adduct"],
                 mode=spec["ionization_mode"], ev=_ms(spec["collision_energy_ev"]),
                 bpi=spec["base_peak_intensity"], molecule_id=spec["molecule_id"],
                 spectrum_id=sid, exact_prec=spec["precursor_mz"])

    # ---------------------------------------------------------------- assembly
    def build(self, test: pd.DataFrame, lab: pd.DataFrame, lg) -> pd.DataFrame:
        r = self.rng
        by_mol = {m: g for m, g in test.groupby("molecule_id", sort=True)}
        mids = lab["molecule_id"].to_numpy(object)
        order = r.permutation(len(mids))
        n = len(mids)
        n_np = min(50, n // 6)
        n_syn = int(0.55 * n)
        groups = {}
        for rank, i in enumerate(order):
            groups[mids[i]] = "np" if rank < n_np else ("syn" if rank < n_np + n_syn else "pub")
        labels = {row.molecule_id: row._asdict() for row in lab.itertuples(index=False)}
        for mid in sorted(labels):
            label = {"smiles": labels[mid]["smiles"], "inchikey": labels[mid]["inchikey"],
                     "formula": labels[mid]["formula"], "mono_mass": labels[mid]["mono_mass"]}
            specs = by_mol[mid]
            grp = groups[mid]
            for _, s in specs.iterrows():
                if grp == "np":
                    self.exact("enveda-np-examples", label, s)
                    if r.random() < 0.7:
                        for lib in r.choice(PUBLIC_NP_LIBS, size=int(r.integers(1, 3)),
                                            replace=False):
                            self.jittered(str(lib), label, s)
                    if r.random() < 0.1:
                        self.jittered("enveda-180", label, s, alt_adduct=False)
                elif grp == "syn":
                    self.exact("enveda-180", label, s)
                    for _ in range(int(r.integers(1, 3))):
                        self.jittered("enveda-180", label, s, strength=0.5, alt_adduct=False)
                    if r.random() < 0.25:
                        self.jittered("pluskal_ms2", label, s)
                    if r.random() < 0.1:
                        self.jittered("gnps", label, s)
                else:
                    libs = r.choice(PUBLIC_NP_LIBS, size=int(r.integers(1, 4)), replace=False)
                    self.exact(str(libs[0]), label, s)   # the public test is made of train rows
                    for lib in libs:
                        for _ in range(int(r.integers(1, 3))):
                            self.jittered(str(lib), label, s)
                    if r.random() < 0.15:
                        self.jittered("spectraverse", label, s)
                    if r.random() < 0.05:
                        self.jittered("masaryk", label, s)
            if grp == "pub" and r.random() < 0.15:
                s = specs.iloc[0]
                mz, it = self._jitter(_ms(s["ms2_mzs"]), _ms(s["ms2_normalized_intensities"]))
                self.add(lib="drug_plus", label=label, mz=mz, it=it, adduct=s["adduct"],
                         adduct_orig=s["adduct"], mode=s["ionization_mode"], ev=np.zeros(0),
                         bpi=np.nan, molecule_id=self._mol_id("drug_plus", label["inchikey"]))
            # decoy isomer: heavier jitter under another structure in the same window
            if isinstance(labels[mid].get("decoy_smiles"), str) and r.random() < 0.4:
                dl = {"smiles": labels[mid]["decoy_smiles"], "inchikey": labels[mid]["decoy_inchikey"],
                      "formula": labels[mid]["decoy_formula"], "mono_mass": labels[mid]["decoy_mass"]}
                lib = "enveda-180" if grp == "syn" else "gnps"
                for _, s in specs.head(2).iterrows():
                    self.jittered(lib, dl, s, strength=2.5, alt_adduct=False)
            # tautomer variant in another library
            if isinstance(labels[mid].get("taut_smiles"), str):
                tl = dict(label)
                tl["smiles"], tl["inchikey"] = labels[mid]["taut_smiles"], labels[mid]["taut_inchikey"]
                for _, s in specs.head(2).iterrows():
                    self.jittered("spectraverse" if grp != "pub" else "mona", tl, s)
        base_rows = list(self.rows)
        for _ in range(self.scale):  # throughput tests: more replicate rounds
            for row in base_rows:
                mz, it = self._jitter(np.asarray(row["ms2_mzs"]),
                                      np.asarray(row["ms2_normalized_intensities"]), 0.5)
                new = dict(row)
                new.update(spectrum_id=self._new_id(), ms2_mzs=mz.tolist(),
                           ms2_normalized_intensities=it.tolist(), num_peaks=int(mz.size))
                self.rows.append(new)
        self._bad_rows(lg)
        df = pd.DataFrame(self.rows, columns=list(TRAIN_COLUMNS))
        return df.iloc[r.permutation(len(df))].reset_index(drop=True)

    def _bad_rows(self, lg) -> None:
        """Rows that ingestion must count and drop or flag."""
        r = self.rng
        good = [row for row in self.rows if row["ingest_lib"] in ("gnps", "enveda-180")]
        pick = [good[int(i)] for i in r.choice(len(good), size=22, replace=False)]
        bad = []

        def variant(row, **kw):
            new = dict(row)
            new["spectrum_id"] = self._new_id()
            new.update(kw)
            bad.append(new)

        for row in pick[:5]:     # huge precursor error (kept, flagged)
            p = row["precursor_mz"] * (1 + 800e-6)
            variant(row, precursor_mz=p, precursor_error_ppm=row["precursor_error_ppm"] + 800.0)
        for row, a in zip(pick[5:9], ["[M+?]+", "Unk", "[M+H]+ in-source", "[M+X]-"]):
            variant(row, adduct=a, adduct_orig=a)   # unparseable adduct (kept, flagged)
        variant(pick[9], adduct="[M+?]", adduct_orig="?", ionization_mode=None)  # no polarity
        variant(pick[10], inchikey14=None, inchikey=None)            # null label
        variant(pick[11], normalized_smiles=None)                      # null label
        variant(pick[12], ms2_mzs=None, ms2_normalized_intensities=None, num_peaks=0)
        m = pick[13]["ms2_mzs"]
        variant(pick[13], ms2_normalized_intensities=pick[13]["ms2_normalized_intensities"][:-1]
                if len(m) > 1 else [])                                 # mismatched arrays
        variant(pick[14], ms2_mzs=[], ms2_normalized_intensities=[], num_peaks=0)
        variant(pick[15], precursor_mz=float("nan"))
        mz = list(pick[16]["ms2_mzs"]) + [pick[16]["precursor_mz"] + 50.0] * 3
        it = list(pick[16]["ms2_normalized_intensities"]) + [0.5, 0.4, 0.3]
        variant(pick[16], ms2_mzs=mz, ms2_normalized_intensities=it, num_peaks=len(mz))
        for row in pick[17:20]:  # duplicated spectrum_id
            new = dict(row)
            bad.append(new)
        variant(pick[20], ms2_normalized_intensities=[-x for x in pick[20]["ms2_normalized_intensities"]])
        self.rows.extend(bad)
        lg.info("bad rows added: %d", len(bad))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--test", type=Path, default=paths.TEST_PARQUET)
    ap.add_argument("--out", type=Path,
                    default=paths.PROCESSED / "synthetic" / "train_synth.parquet")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--scale", type=int, default=0, help="extra replicate rounds")
    ap.add_argument("--row-group-size", type=int, default=500)
    args = ap.parse_args(argv)
    lg = get_logger("make_synthetic_train")
    log_run(lg, args)
    cfg = load_config(args.config)

    test = pd.read_parquet(args.test)
    test = test.sort_values(["molecule_id", "spectrum_id"], kind="mergesort").reset_index(drop=True)
    mk = Maker(cfg, cfg.seed, args.scale)
    lab = mk.labels(test, lg)
    df = mk.build(test, lab, lg)
    tbl = pa.Table.from_pandas(df, schema=TRAIN_SCHEMA, preserve_index=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    with pq.ParquetWriter(tmp, TRAIN_SCHEMA, compression="zstd") as w:
        for a in range(0, tbl.num_rows, args.row_group_size):
            w.write_table(tbl.slice(a, args.row_group_size))
    tmp.replace(args.out)
    md = pq.ParquetFile(args.out).metadata
    lg.info("wrote %s: %d rows, %d row groups, %d structures, libraries %s", args.out,
            md.num_rows, md.num_row_groups, df["inchikey14"].nunique(),
            df["ingest_lib"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())

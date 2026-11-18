"""Fingerprint prediction model: spectrum -> Morgan + MACCS bit probabilities (WP9).

PLAN.md section 4.4. One module, two halves:

- numpy half (inference path, importable without torch): featurisation, metadata vector,
  training-data containers, sampler weights, augmentation, ``predict_numpy`` on exported
  ``.npz`` weights, molecule-level aggregation, per-bit calibration, candidate scores.
- torch half (training only): every ``torch`` import is inside a function
  (``train``, ``export_npz``, ``torch_predict``, ``benchmark_throughput``), so
  ``import casmi.fpmodel`` works with torch absent or blocked.

Training input interface (what WP6 / WP5 must provide)
------------------------------------------------------
``TrainingData`` = flat spectrum store + per-spectrum metadata + one target row per
structure. Build it with ``TrainingData.from_parts(store, meta_df, targets)``, or let a
loader do it: ``TrainingData.from_parquet(path)`` (train.parquet schema, a file or a
directory such as WP6's ``data/processed/train/spectra/``, streamed and cleaned here;
the default source) or ``TrainingData.from_library(dir)`` (the WP5 library index: the
same arrays, but a library built from already-cleaned spectra has lost ``n_peaks_raw``
and ``prec_intensity``, which is logged as a warning).

1. ``store``: ``casmi.spectra.SpectrumStore`` (``mz`` float32, ``intensity`` float32,
   ``offsets`` int64 of length n + 1) holding CLEANED spectra (``spectra.clean`` /
   ``clean_flat`` with the run's Config: precursor window removed, floor, top 100,
   base peak 1). Row i of the store is row i of ``meta_df``.
2. ``meta_df``: pandas DataFrame, one row per spectrum. Columns (aliases in brackets):

   =====================  =============  ===================================================
   column                 dtype          meaning
   =====================  =============  ===================================================
   ik14                   str            structure key [inchikey14]; joins to ``targets``
   precursor_mz           float64        measured precursor m/z
   adduct                 str            any spelling; matched to the ten test adducts with
                                         ``adducts.match_test_adduct``, else "other"
   polarity               int (+1/-1)    [ionization_mode str "positive"/"negative"]
   collision_energy       float64        mean eV, NaN if unknown; with ``n_energies`` (int,
                                         0 if unknown) [collision_energy_ev list<double>]
   instrument_family      str            one of INSTRUMENT_FAMILIES [instrument_type free
                                         text + ingest_lib, mapped by
                                         ``library.instrument_family``; absent: timstof]
   base_peak_intensity    float64        raw counts of the base peak, NaN if unknown
   n_peaks_raw            int            peaks before cleaning (NaN/absent: cleaned count)
   prec_intensity         float64        optional, from ``clean_flat`` (NaN if unknown)
   mono_mass              float64        optional, label neutral mass M (default: from
                                         precursor m/z and adduct); used for the 100-1300 Da
                                         filter and as the "mass" input when the adduct
                                         cannot be parsed
   precursor_error_ppm    float64        optional, label-quality filter (NaN passes)
   ingest_lib             str            optional, logging only
   =====================  =============  ===================================================

3. ``targets``: ``FpTargets`` - sorted unique ``ik14`` (object array), ``morgan`` uint8
   (n, 512) and ``maccs`` uint8 (n, 21), packed exactly like ``chem.morgan_fp`` /
   ``chem.maccs_fp`` (big bit order), computed on ``chem.std_mol`` of the label SMILES
   (``build_targets(ik14, smiles)`` does it, with a disk cache). ``ok`` False rows (SMILES
   that failed) are excluded from training.

Eligibility (``TrainingData.eligible``, PLAN 4.4): label mass in [fpm_mass_min,
fpm_mass_max], at least ``min_peaks`` cleaned peaks, adduct parseable, |precursor error|
<= ``fpm_max_precursor_error_ppm``, target ok. Holdout exclusion is by ik14 and by mkey
(``holdout_exclusion``).

Model input (``featurize_sparse``, ``meta_matrix``)
---------------------------------------------------
- Sparse: fragment bins ``floor(mz / bin)`` for 0 <= mz < fpm_mz_max (13,000 bins at
  0.1 Da) and neutral-loss bins ``n_frag + floor((precursor_mz - mz) / bin)`` for
  0 <= loss < fpm_nl_max (7,000 bins), value sqrt(intensity). Entries of a spectrum are
  contiguous (fragments, then losses); duplicate bins are summed by the embedding bag.
- Dense: ``META_FIELDS`` (28 values): adduct one-hot (10 test adducts + other), positive
  polarity, CE / 100 and missing flag, merged-energies flag, number of energies / 3,
  instrument family one-hot (4), M / 1000, precursor m/z / 1000, log10(base peak
  intensity) / 6 and missing flag, log10(1 + raw peaks) / 3, log10(1 + clean peaks) / 2,
  precursor intensity and missing flag.

Architecture (``ModelSpec``): ``EmbeddingBag(n_in, H, sum)`` with per-sample weights, plus
a metadata branch ``Linear(28, 64) -> GELU -> Linear(64, H)`` added to the bag output
(this equals concatenating the 64-unit metadata projection and applying one linear
layer, with that layer folded into the embedding table), then LayerNorm and GELU, then
``n_blocks`` residual blocks ``h + Dropout(GELU(LayerNorm(Linear(h))))``, then
``Linear(H, 4263)``. GELU is the tanh approximation in torch and numpy alike, so the
numpy forward pass reproduces torch to float32 rounding.

Exported ``.npz`` (``export_npz`` / ``FpModel.load``): arrays ``emb`` (n_in, H),
``meta_w1`` (28, 64), ``meta_b1``, ``meta_w2`` (64, H), ``meta_b2``, ``ln0_g``, ``ln0_b``,
``blk{k}_w`` (H, H), ``blk{k}_b``, ``blk{k}_g``, ``blk{k}_beta``, ``out_w`` (H, 4263),
``out_b`` (weights stored as (in, out) for ``x @ W``; float32 or float16), and
``spec_json`` (a unicode scalar: ``ModelSpec`` plus training info). No pickles.

Training (``train``, torch): BCE with per-bit positive weights n_neg / n_pos under the
sampling distribution clipped to [1, fpm_pos_weight_max], plus fpm_cosine_loss_weight x
(1 - cosine(sigmoid(logits), target)); sampler 1 / sqrt(spectra of the structure), x2 for
timsTOF test-adduct spectra (``sampling_weights``); augmentation ``augment_batch``; AdamW
(fused kernel) with linear warm-up and cosine decay; time budget, checkpoints, loss curve.
The script ``scripts/train_fpmodel.py`` wires ``load_training_source`` ->
``split_training_data`` -> ``train`` -> ``calibrate``.

Molecule level and candidates
-----------------------------
``molecule_logits``: weighted mean of spectrum logits, weight clip(log10 bpi, 3, 6) / 3
(missing bpi: weight 1), over all spectra and per polarity. ``candidate_scores``: cosine,
Bernoulli log-likelihood (p clipped to [fp_prob_min, fp_prob_max], optional
``BitCalibration``), soft Tanimoto, the same three on MACCS bits only, and the rank under
the log-likelihood, from one probability vector and candidate fingerprints (packed uint8
as ``chem`` produces them, or bool). Typical use (WP10)::

    model = FpModel.load(paths.MODELS / "fp_v1_holdout.npz")
    cal = BitCalibration.load(paths.MODELS / "fp_v1_calibration.npz")
    mp = predict_molecules(model, test_df, cfg)          # all / pos / neg logits
    s = candidate_scores(mp.proba()[mp.row(mol_id)], morgan_packed, maccs_packed, cfg, cal)
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import paths
from .adducts import TEST_ADDUCT_NAMES, match_test_adduct, parse_adduct
from .config import Config
from .spectra import SpectrumStore, clean_flat, list_arrays_to_flat, merge_groups

__all__ = [
    "INSTRUMENT_FAMILIES",
    "META_FIELDS",
    "MODEL_VERSION",
    "ModelSpec",
    "SparseBatch",
    "SpectrumMeta",
    "FpTargets",
    "TrainingData",
    "FpModel",
    "BitCalibration",
    "MoleculePredictions",
    "featurize_sparse",
    "meta_matrix",
    "spectra_from_frame",
    "build_targets",
    "holdout_exclusion",
    "sampling_weights",
    "epoch_indices",
    "pos_weights",
    "augment_batch",
    "make_batch",
    "predict_numpy",
    "random_params",
    "save_params",
    "molecule_logits",
    "predict_molecules",
    "fit_calibration",
    "candidate_scores",
    "bit_dot",
    "sigmoid",
    "synthetic_training_data",
    "synthetic_test_frame",
    "load_training_source",
    "split_training_data",
    "calibrate",
    "fingerprint_consistency",
    "train",
    "export_npz",
    "torch_predict",
    "benchmark_throughput",
]

log = logging.getLogger(__name__)

MODEL_VERSION = 1
# Same tuple as casmi.library.INSTRUMENT_FAMILIES (asserted in tests/test_fpmodel.py).
INSTRUMENT_FAMILIES: tuple[str, ...] = ("timstof", "qtof", "orbitrap", "other")
N_TEST_ADDUCTS = len(TEST_ADDUCT_NAMES)           # adduct code 0..9; 10 = other
ADDUCT_OTHER = N_TEST_ADDUCTS

META_FIELDS: tuple[str, ...] = (
    *(f"adduct={a}" for a in TEST_ADDUCT_NAMES), "adduct=other",
    "positive", "ce", "ce_missing", "ce_merged", "n_energies",
    *(f"inst={f}" for f in INSTRUMENT_FAMILIES),
    "mass", "precursor_mz", "log_bpi", "bpi_missing", "log_n_peaks_raw", "log_n_peaks",
    "prec_intensity", "prec_intensity_missing",
)
_MF = {name: i for i, name in enumerate(META_FIELDS)}
_GELU_C = math.sqrt(2.0 / math.pi)


# ======================================================================================
# Model specification
# ======================================================================================


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to featurise and run a model; stored in the ``.npz``."""

    bin_da: float = 0.1
    mz_max: float = 1300.0
    nl_max: float = 700.0
    intensity_transform: str = "sqrt"
    hidden: int = 1024
    meta_hidden: int = 64
    n_blocks: int = 2
    dropout: float = 0.1
    n_morgan: int = 4096
    n_maccs: int = 167
    meta_fields: tuple = META_FIELDS
    ln_eps: float = 1e-5
    version: int = MODEL_VERSION
    clean: Mapping[str, Any] = field(default_factory=dict)   # cleaning settings used

    @property
    def n_frag(self) -> int:
        return int(math.ceil(self.mz_max / self.bin_da - 1e-9))

    @property
    def n_nl(self) -> int:
        return int(math.ceil(self.nl_max / self.bin_da - 1e-9))

    @property
    def n_in(self) -> int:
        return self.n_frag + self.n_nl

    @property
    def n_out(self) -> int:
        return self.n_morgan + self.n_maccs

    @property
    def n_meta(self) -> int:
        return len(self.meta_fields)

    @classmethod
    def from_config(cls, cfg: Config | None = None, **overrides: Any) -> "ModelSpec":
        cfg = cfg or Config()
        kw = dict(bin_da=float(cfg.fpm_mz_bin_da), mz_max=float(cfg.fpm_mz_max),
                  nl_max=float(cfg.fpm_nl_max), intensity_transform=cfg.intensity_transform,
                  hidden=int(cfg.fpm_hidden), meta_hidden=int(cfg.fpm_meta_hidden),
                  n_blocks=int(cfg.fpm_n_blocks), dropout=float(cfg.fpm_dropout),
                  n_morgan=int(cfg.fp_morgan_bits),
                  n_maccs=167 if cfg.fp_use_maccs else 0, clean=_clean_settings(cfg))
        kw.update(overrides)
        return cls(**kw)

    def to_json(self, extra: Mapping[str, Any] | None = None) -> str:
        d = dataclasses.asdict(self)
        d["meta_fields"] = list(self.meta_fields)
        d["clean"] = dict(self.clean)
        if extra:
            d["extra"] = dict(extra)
        return json.dumps(d, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> tuple["ModelSpec", dict]:
        d = json.loads(text)
        extra = d.pop("extra", {})
        d["meta_fields"] = tuple(d["meta_fields"])
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known}), extra


def _clean_settings(cfg: Config) -> dict:
    return {k: getattr(cfg, k) for k in (
        "clean_max_above_precursor_da", "keep_precursor", "precursor_window_low_da",
        "precursor_window_high_da", "intensity_floor", "top_n_peaks")}


# ======================================================================================
# Featurisation
# ======================================================================================


@dataclass
class SparseBatch:
    """Embedding-bag input: entries ``offsets[i]:offsets[i+1]`` belong to spectrum i."""

    indices: np.ndarray     # int64 bin ids in [0, n_in)
    weights: np.ndarray     # float32 values
    offsets: np.ndarray     # int64, n + 1

    def __len__(self) -> int:
        return len(self.offsets) - 1


def featurize_sparse(store: SpectrumStore, precursor_mz: np.ndarray,
                     spec: ModelSpec) -> SparseBatch:
    """Fragment + neutral-loss bins of every spectrum in ``store`` (vectorised, no loop).

    See the module docstring for the bin definitions. NaN precursor: no loss entries.
    """
    off = np.asarray(store.offsets, dtype=np.int64)
    n = len(off) - 1
    prec = np.asarray(precursor_mz, dtype=np.float64).reshape(n)
    npk = np.diff(off)
    sid = np.repeat(np.arange(n, dtype=np.int64), npk)
    x = np.asarray(store.mz[off[0]:off[-1]], dtype=np.float64)
    it = np.asarray(store.intensity[off[0]:off[-1]], dtype=np.float32)
    if spec.intensity_transform == "sqrt":
        val = np.sqrt(np.maximum(it, 0.0), dtype=np.float32)
    elif spec.intensity_transform == "none":
        val = it.astype(np.float32, copy=True)
    else:
        raise ValueError(f"unknown intensity_transform {spec.intensity_transform!r}")
    with np.errstate(invalid="ignore"):
        fcol = np.floor(x / spec.bin_da)
        fok = (x >= 0.0) & (fcol < spec.n_frag) & np.isfinite(fcol)
        loss = prec[sid] - x
        ncol = np.floor(loss / spec.bin_da)
        nok = (loss >= 0.0) & (ncol < spec.n_nl) & np.isfinite(ncol)
    fsel = np.flatnonzero(fok)
    nsel = np.flatnonzero(nok)
    fs, ns = sid[fsel], sid[nsel]
    cf = np.bincount(fs, minlength=n).astype(np.int64)
    cn = np.bincount(ns, minlength=n).astype(np.int64)
    out_off = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(cf + cn, out=out_off[1:])
    f_start = np.concatenate([[0], np.cumsum(cf)[:-1]]) if n else np.zeros(0, np.int64)
    n_start = np.concatenate([[0], np.cumsum(cn)[:-1]]) if n else np.zeros(0, np.int64)
    pos_f = out_off[fs] + (np.arange(len(fsel), dtype=np.int64) - f_start[fs])
    pos_n = out_off[ns] + cf[ns] + (np.arange(len(nsel), dtype=np.int64) - n_start[ns])
    total = int(out_off[-1])
    indices = np.empty(total, dtype=np.int64)
    weights = np.empty(total, dtype=np.float32)
    indices[pos_f] = fcol[fsel].astype(np.int64)
    weights[pos_f] = val[fsel]
    indices[pos_n] = spec.n_frag + ncol[nsel].astype(np.int64)
    weights[pos_n] = val[nsel]
    return SparseBatch(indices, weights, out_off)


def active_bins(batch: SparseBatch, spec: ModelSpec) -> tuple[np.ndarray, np.ndarray]:
    """Distinct active (fragment, loss) bins per spectrum, for diagnostics."""
    n = len(batch)
    sid = np.repeat(np.arange(n, dtype=np.int64), np.diff(batch.offsets))
    key = np.unique(sid * spec.n_in + batch.indices)
    s, col = key // spec.n_in, key % spec.n_in
    frag = np.bincount(s[col < spec.n_frag], minlength=n)
    nl = np.bincount(s[col >= spec.n_frag], minlength=n)
    return frag, nl


# ======================================================================================
# Per-spectrum metadata
# ======================================================================================


@dataclass
class SpectrumMeta:
    """Per-spectrum metadata arrays (one entry per store row). See ``from_frame``."""

    precursor_mz: np.ndarray        # float64
    adduct_code: np.ndarray         # int8, index into TEST_ADDUCT_NAMES, 10 = other
    adduct_known: np.ndarray        # bool, adduct string parseable
    polarity: np.ndarray            # int8, +1 / -1 (0 unknown)
    collision_energy: np.ndarray    # float64 mean eV, NaN unknown
    n_energies: np.ndarray          # int16, 0 unknown
    instrument_code: np.ndarray     # int8, index into INSTRUMENT_FAMILIES
    mass: np.ndarray                # float64 neutral mass M used as input (NaN unknown)
    base_peak_intensity: np.ndarray  # float64, NaN unknown
    n_peaks_raw: np.ndarray         # float64 (NaN unknown)
    prec_intensity: np.ndarray      # float64, NaN unknown

    def __len__(self) -> int:
        return len(self.precursor_mz)

    def take(self, idx: np.ndarray) -> "SpectrumMeta":
        return SpectrumMeta(**{f.name: getattr(self, f.name)[idx]
                               for f in dataclasses.fields(self)})

    @classmethod
    def concat(cls, parts: Sequence["SpectrumMeta"]) -> "SpectrumMeta":
        return SpectrumMeta(**{f.name: np.concatenate([getattr(p, f.name) for p in parts])
                               for f in dataclasses.fields(cls)})

    @classmethod
    def from_frame(cls, df: pd.DataFrame, n_peaks_raw: np.ndarray | None = None,
                   prec_intensity: np.ndarray | None = None) -> "SpectrumMeta":
        """From a DataFrame with the columns of the training interface (module docstring).

        ``n_peaks_raw`` / ``prec_intensity`` override the frame's columns (``clean_flat``
        outputs).
        """
        n = len(df)
        prec = _num(df, "precursor_mz", n)
        adduct = _obj(df, "adduct", n)
        if "polarity" in df.columns:
            pol = np.nan_to_num(_num(df, "polarity", n)).astype(np.int8)
            pol = np.sign(pol).astype(np.int8)
        else:
            pol = _polarity_from_mode(_obj(df, "ionization_mode", n), adduct)
        code, known, mult, shift, charge = _adduct_codes(adduct, pol)
        if "collision_energy" in df.columns:
            ce = _num(df, "collision_energy", n)
            ne = (np.nan_to_num(_num(df, "n_energies", n)) if "n_energies" in df.columns
                  else np.isfinite(ce).astype(np.float64))
        elif "collision_energy_ev" in df.columns:
            ce, ne = _list_mean_count(df["collision_energy_ev"])
        else:
            ce, ne = np.full(n, np.nan), np.zeros(n)
        inst = _instrument_codes(df, n)
        with np.errstate(invalid="ignore", divide="ignore"):
            derived = (prec * np.abs(charge) - shift) / mult
        label = _num(df, "mono_mass", n) if "mono_mass" in df.columns else np.full(n, np.nan)
        mass = np.where(known & np.isfinite(derived), derived, label)
        bpi = _num(df, "base_peak_intensity", n)
        nraw = (np.asarray(n_peaks_raw, dtype=np.float64) if n_peaks_raw is not None
                else _num(df, "n_peaks_raw", n))
        pint = (np.asarray(prec_intensity, dtype=np.float64) if prec_intensity is not None
                else _num(df, "prec_intensity", n))
        return cls(prec, code, known, pol, ce, ne.astype(np.int16), inst, mass, bpi, nraw,
                   pint)


def _num(df: pd.DataFrame, col: str, n: int) -> np.ndarray:
    if col not in df.columns:
        return np.full(n, np.nan)
    return np.array(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64,
                                                                     na_value=np.nan))


def _obj(df: pd.DataFrame, col: str, n: int) -> np.ndarray:
    if col not in df.columns:
        return np.full(n, None, dtype=object)
    s = df[col].astype(object)
    return np.array(s.where(s.notna(), None).to_numpy(dtype=object))


def _polarity_from_mode(mode: np.ndarray, adduct: np.ndarray) -> np.ndarray:
    """+1 / -1 / 0 from ionization_mode, else the adduct's trailing sign."""
    codes, uniq = pd.factorize(pd.Series([f"{m}|{a}" for m, a in zip(mode, adduct)]))
    mo = pd.Series(mode).astype(object)
    ad = pd.Series(adduct).astype(object)
    first = pd.Series(np.arange(len(codes))).groupby(codes).first().to_numpy()
    vals = np.zeros(len(uniq), dtype=np.int8)
    for u, i in enumerate(first):
        m, a = mo.iloc[i], ad.iloc[i]
        s = m.strip().lower() if isinstance(m, str) else ""
        if s.startswith("pos") or s == "+":
            vals[u] = 1
        elif s.startswith("neg") or s == "-":
            vals[u] = -1
        elif isinstance(a, str) and a.strip().endswith("+"):
            vals[u] = 1
        elif isinstance(a, str) and a.strip().endswith("-"):
            vals[u] = -1
    return vals[codes] if len(codes) else np.zeros(0, np.int8)


def _adduct_codes(adduct: np.ndarray, pol: np.ndarray):
    """Per spectrum: test-adduct code, parseable flag, multiplier, shift, charge."""
    n = len(adduct)
    keys = pd.Series([f"{a}|{p}" for a, p in zip(adduct, pol)], dtype=object)
    codes, uniq = pd.factorize(keys)
    first = pd.Series(np.arange(n)).groupby(codes).first().to_numpy() if n else []
    u_code = np.full(len(uniq), ADDUCT_OTHER, dtype=np.int8)
    u_known = np.zeros(len(uniq), dtype=bool)
    u_mult = np.ones(len(uniq))
    u_shift = np.full(len(uniq), np.nan)
    u_charge = np.ones(len(uniq))
    for u, i in enumerate(first):
        a, p = adduct[i], int(pol[i])
        mode = "positive" if p > 0 else ("negative" if p < 0 else None)
        if not isinstance(a, str):
            continue
        name = match_test_adduct(a, mode)
        if name is not None:
            u_code[u] = TEST_ADDUCT_NAMES.index(name)
        parsed = parse_adduct(a, mode)
        if parsed is not None:
            u_known[u] = True
            u_mult[u], u_shift[u], u_charge[u] = parsed.multiplier, parsed.shift, parsed.charge
    if n == 0:
        return (np.zeros(0, np.int8), np.zeros(0, bool), np.zeros(0), np.zeros(0),
                np.zeros(0))
    return u_code[codes], u_known[codes], u_mult[codes], u_shift[codes], u_charge[codes]


def _instrument_codes(df: pd.DataFrame, n: int) -> np.ndarray:
    if "instrument_family" in df.columns:
        fam = _obj(df, "instrument_family", n)
    elif "instrument_type" in df.columns or "ingest_lib" in df.columns:
        from .library import instrument_family  # lazy: WP5 module

        itype = _obj(df, "instrument_type", n)
        lib = _obj(df, "ingest_lib", n)
        keys = pd.Series([f"{t}|{l}" for t, l in zip(itype, lib)], dtype=object)
        codes, uniq = pd.factorize(keys)
        first = pd.Series(np.arange(n)).groupby(codes).first().to_numpy()
        ufam = np.array([instrument_family(itype[i], lib[i]) for i in first], dtype=object)
        fam = ufam[codes] if n else np.zeros(0, object)
    else:
        fam = np.full(n, "timstof", dtype=object)
    lut = {f: i for i, f in enumerate(INSTRUMENT_FAMILIES)}
    s = pd.Series(fam, dtype=object)
    return s.map(lambda v: lut.get(v, lut["other"])).to_numpy(dtype=np.int8)


def _list_mean_count(col: Any) -> tuple[np.ndarray, np.ndarray]:
    """Mean of finite values and their count per list (NaN / 0 for null or empty)."""
    import pyarrow as pa

    arr = pa.array(col) if not isinstance(col, (pa.Array, pa.ChunkedArray)) else col
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    n = len(arr)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    arr = arr.cast(pa.list_(pa.float64()))
    off = np.asarray(arr.offsets, dtype=np.int64)
    vals = np.asarray(arr.values.to_numpy(zero_copy_only=False), dtype=np.float64)
    vals = vals[off[0]:off[-1]]
    off = off - off[0]
    fin = np.isfinite(vals)
    cs = np.concatenate([[0.0], np.cumsum(np.where(fin, vals, 0.0))])
    cc = np.concatenate([[0], np.cumsum(fin)])
    s = cs[off[1:]] - cs[off[:-1]]
    c = (cc[off[1:]] - cc[off[:-1]]).astype(np.float64)
    if arr.null_count:
        c = c * ~np.asarray(arr.is_null().to_numpy(zero_copy_only=False))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    return mean, c


def meta_matrix(meta: SpectrumMeta, n_peaks: np.ndarray, spec: ModelSpec | None = None
                ) -> np.ndarray:
    """Dense float32 (n, len(META_FIELDS)) metadata input (module docstring)."""
    if spec is not None and tuple(spec.meta_fields) != META_FIELDS:
        raise ValueError("model was trained with other metadata fields; "
                         f"model {list(spec.meta_fields)} vs code {list(META_FIELDS)}")
    n = len(meta)
    x = np.zeros((n, len(META_FIELDS)), dtype=np.float32)
    rows = np.arange(n)
    code = np.clip(meta.adduct_code.astype(np.int64), 0, ADDUCT_OTHER)
    x[rows, code] = 1.0
    x[:, _MF["positive"]] = meta.polarity > 0
    ce = meta.collision_energy
    ce_ok = np.isfinite(ce)
    x[:, _MF["ce"]] = np.where(ce_ok, ce, 0.0) / 100.0
    x[:, _MF["ce_missing"]] = ~ce_ok
    ne = meta.n_energies.astype(np.float64)
    x[:, _MF["ce_merged"]] = ne > 1
    x[:, _MF["n_energies"]] = np.minimum(ne, 10.0) / 3.0
    inst = np.clip(meta.instrument_code.astype(np.int64), 0, len(INSTRUMENT_FAMILIES) - 1)
    x[rows, _MF["inst=timstof"] + inst] = 1.0
    x[:, _MF["mass"]] = np.nan_to_num(meta.mass, nan=0.0) / 1000.0
    x[:, _MF["precursor_mz"]] = np.nan_to_num(meta.precursor_mz, nan=0.0) / 1000.0
    bpi = meta.base_peak_intensity
    bpi_ok = np.isfinite(bpi) & (bpi > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        x[:, _MF["log_bpi"]] = np.where(bpi_ok, np.log10(np.where(bpi_ok, bpi, 1.0)), 0) / 6.0
    x[:, _MF["bpi_missing"]] = ~bpi_ok
    npk = np.asarray(n_peaks, dtype=np.float64)
    raw = np.where(np.isfinite(meta.n_peaks_raw), meta.n_peaks_raw, npk)
    x[:, _MF["log_n_peaks_raw"]] = np.log10(1.0 + np.maximum(raw, 0.0)) / 3.0
    x[:, _MF["log_n_peaks"]] = np.log10(1.0 + npk) / 2.0
    pi = meta.prec_intensity
    pi_ok = np.isfinite(pi)
    x[:, _MF["prec_intensity"]] = np.where(pi_ok, pi, 0.0)
    x[:, _MF["prec_intensity_missing"]] = ~pi_ok
    return x


def spectra_from_frame(df: pd.DataFrame, cfg: Config | None = None
                       ) -> tuple[SpectrumStore, SpectrumMeta]:
    """Clean the peak lists of a test- or train-schema frame and build its metadata.

    Rows with null or mismatched peak lists get zero peaks (they still get a prediction,
    from metadata only). Order is preserved.
    """
    import pyarrow as pa

    cfg = cfg or Config()
    tab = pa.Table.from_pandas(df[["ms2_mzs", "ms2_normalized_intensities"]],
                               preserve_index=False)
    mz, it, off, valid = list_arrays_to_flat(tab.column("ms2_mzs"),
                                             tab.column("ms2_normalized_intensities"))
    prec = pd.to_numeric(df["precursor_mz"], errors="coerce").to_numpy(np.float64)
    cm, ci, co, pint, nraw = clean_flat(mz, it, off, prec, cfg, valid)
    meta = SpectrumMeta.from_frame(df.reset_index(drop=True), n_peaks_raw=nraw,
                                   prec_intensity=pint)
    return SpectrumStore(cm, ci, co), meta


# ======================================================================================
# Targets
# ======================================================================================


@dataclass
class FpTargets:
    """One target row per structure: packed Morgan (n, 512) and MACCS (n, 21) bits."""

    ik14: np.ndarray                 # object, sorted unique
    morgan: np.ndarray               # uint8 (n, ceil(n_morgan / 8))
    maccs: np.ndarray                # uint8 (n, ceil(n_maccs / 8))
    ok: np.ndarray                   # bool, fingerprint computed
    mono_mass: np.ndarray            # float64, mass of std_mol (NaN unknown)
    n_morgan: int = 4096
    n_maccs: int = 167

    def __len__(self) -> int:
        return len(self.ik14)

    @property
    def n_bits(self) -> int:
        return self.n_morgan + self.n_maccs

    def rows(self, keys: Iterable[str]) -> np.ndarray:
        """Row of each key (-1 if absent)."""
        keys = np.asarray(list(keys) if not isinstance(keys, np.ndarray) else keys,
                          dtype=object)
        idx = pd.Index(self.ik14)
        return idx.get_indexer(keys).astype(np.int64)

    def dense(self, rows: np.ndarray) -> np.ndarray:
        """float32 (len(rows), n_bits) 0/1 matrix [Morgan | MACCS]."""
        m = np.unpackbits(self.morgan[rows], axis=1, count=self.n_morgan)
        if self.n_maccs:
            k = np.unpackbits(self.maccs[rows], axis=1, count=self.n_maccs)
            m = np.concatenate([m, k], axis=1)
        return m.astype(np.float32)

    def take(self, rows: np.ndarray) -> "FpTargets":
        return FpTargets(self.ik14[rows], self.morgan[rows], self.maccs[rows], self.ok[rows],
                         self.mono_mass[rows], self.n_morgan, self.n_maccs)

    @classmethod
    def from_arrays(cls, ik14, morgan, maccs, ok=None, mono_mass=None, n_morgan=4096,
                    n_maccs=167) -> "FpTargets":
        """Sorts by ik14 and checks uniqueness."""
        ik = np.asarray(ik14, dtype=object)
        order = np.argsort(ik.astype(str), kind="stable")
        ik = ik[order]
        if len(ik) > 1 and (ik[1:] == ik[:-1]).any():
            raise ValueError("duplicate ik14 in targets")
        n = len(ik)
        ok = np.ones(n, bool) if ok is None else np.asarray(ok, bool)[order]
        mm = np.full(n, np.nan) if mono_mass is None else np.asarray(mono_mass, float)[order]
        return cls(ik, np.ascontiguousarray(np.asarray(morgan, np.uint8)[order]),
                   np.ascontiguousarray(np.asarray(maccs, np.uint8)[order]), ok, mm,
                   n_morgan, n_maccs)


def _smiles_hash(smiles: Sequence[str | None]) -> np.ndarray:
    out = np.zeros(len(smiles), dtype=np.uint64)
    for i, s in enumerate(smiles):
        if isinstance(s, str):
            out[i] = int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(),
                                    "little")
    return out


def build_targets(ik14: Sequence[str], smiles: Sequence[str | None], cfg: Config | None = None,
                  n_jobs: int = 1, cache: Path | str | None = None) -> FpTargets:
    """Fingerprint targets (``chem.featurize`` on ``std_mol``) for unique structures.

    ``cache``: an ``.npz`` reused for (ik14, SMILES) pairs already computed and extended
    with the new ones (None: no cache).
    """
    from .chem import featurize

    cfg = cfg or Config()
    ik = np.asarray(ik14, dtype=object)
    smi = list(smiles)
    if len(set(ik.tolist())) != len(ik):
        raise ValueError("build_targets expects unique ik14")
    h = _smiles_hash(smi)
    n = len(ik)
    nb_m, nb_k = (cfg.fp_morgan_bits + 7) // 8, (167 + 7) // 8
    morgan = np.zeros((n, nb_m), np.uint8)
    maccs = np.zeros((n, nb_k), np.uint8)
    ok = np.zeros(n, bool)
    mass = np.full(n, np.nan)
    todo = np.ones(n, bool)
    cache_p = Path(cache) if cache is not None else None
    old = None
    if cache_p is not None and cache_p.is_file():
        with np.load(cache_p, allow_pickle=False) as z:
            old = {k: z[k] for k in z.files}
        if int(old["n_morgan"]) == cfg.fp_morgan_bits and int(old["radius"]) == cfg.fp_morgan_radius:
            key_old = pd.Index([f"{a}|{b}" for a, b in zip(old["ik14"], old["smiles_hash"])])
            pos = key_old.get_indexer([f"{a}|{b}" for a, b in zip(ik, h)])
            hit = pos >= 0
            morgan[hit] = old["morgan"][pos[hit]]
            maccs[hit] = old["maccs"][pos[hit]]
            ok[hit] = old["ok"][pos[hit]]
            mass[hit] = old["mono_mass"][pos[hit]]
            todo = ~hit
        else:
            old = None
    idx = np.flatnonzero(todo)
    log.info("targets: %d structures, %d from cache, %d to compute", n, n - len(idx), len(idx))
    if len(idx):
        res = featurize([smi[i] for i in idx], cfg, n_jobs=n_jobs,
                        fields=("morgan", "maccs", "mono_mass"))
        morgan[idx], maccs[idx], ok[idx], mass[idx] = (res["morgan"], res["maccs"], res["ok"],
                                                       res["mono_mass"])
        if cache_p is not None:
            all_ik, all_h = ik, h
            all_m, all_k, all_ok, all_mass = morgan, maccs, ok, mass
            if old is not None:
                keep = ~pd.Index([f"{a}|{b}" for a, b in zip(old["ik14"], old["smiles_hash"])]
                                 ).isin([f"{a}|{b}" for a, b in zip(ik, h)])
                all_ik = np.concatenate([old["ik14"][keep].astype(object), ik])
                all_h = np.concatenate([old["smiles_hash"][keep], h])
                all_m = np.concatenate([old["morgan"][keep], morgan])
                all_k = np.concatenate([old["maccs"][keep], maccs])
                all_ok = np.concatenate([old["ok"][keep], ok])
                all_mass = np.concatenate([old["mono_mass"][keep], mass])
            paths.ensure_dir(cache_p.parent)
            tmp = cache_p.with_name(cache_p.name + f".tmp{os.getpid()}.npz")
            np.savez(tmp, ik14=np.asarray(all_ik, dtype=str), smiles_hash=all_h, morgan=all_m,
                     maccs=all_k, ok=all_ok, mono_mass=all_mass,
                     n_morgan=np.int64(cfg.fp_morgan_bits),
                     radius=np.int64(cfg.fp_morgan_radius))
            os.replace(tmp, cache_p)
    log.info("targets: %d ok, %d failed", int(ok.sum()), int((~ok).sum()))
    return FpTargets.from_arrays(ik, morgan, maccs, ok, mass, cfg.fp_morgan_bits,
                                 167 if cfg.fp_use_maccs else 0)


def default_targets_cache() -> Path:
    return paths.PROCESSED / "fp" / "targets_cache.npz"


# ======================================================================================
# Training data
# ======================================================================================


@dataclass
class TrainingData:
    """Cleaned spectra + metadata + structure targets (the WP6/WP5 -> WP9 interface)."""

    store: SpectrumStore
    meta: SpectrumMeta
    target_row: np.ndarray           # int64 row in ``targets`` per spectrum (-1: none)
    targets: FpTargets
    ik14: np.ndarray                 # object, per spectrum
    label_mass: np.ndarray           # float64, per spectrum (structure mass)
    precursor_error_ppm: np.ndarray  # float64, NaN unknown
    ingest_lib: np.ndarray           # object (None unknown)
    _groups: Any = None              # consensus partner index (lazy)

    def __len__(self) -> int:
        return len(self.store)

    # ---- construction ---------------------------------------------------------------

    @classmethod
    def from_parts(cls, store: SpectrumStore, meta_df: pd.DataFrame, targets: FpTargets
                   ) -> "TrainingData":
        """The documented interface: cleaned store + metadata frame + targets."""
        if len(store) != len(meta_df):
            raise ValueError(f"store has {len(store)} rows, meta {len(meta_df)}")
        df = meta_df.reset_index(drop=True)
        key = "ik14" if "ik14" in df.columns else "inchikey14"
        ik = _obj(df, key, len(df))
        meta = SpectrumMeta.from_frame(df)
        trow = targets.rows(ik)
        label = (_num(df, "mono_mass", len(df)) if "mono_mass" in df.columns
                 else np.full(len(df), np.nan))
        tmass = np.where(trow >= 0, targets.mono_mass[np.maximum(trow, 0)], np.nan)
        label = np.where(np.isfinite(label), label, tmass)
        return cls(store, meta, trow, targets, ik, label,
                   _num(df, "precursor_error_ppm", len(df)), _obj(df, "ingest_lib", len(df)))

    @classmethod
    def from_library(cls, directory: Path | str | None = None, cfg: Config | None = None,
                     targets: FpTargets | None = None, n_jobs: int = 1,
                     targets_cache: Path | str | None = "default",
                     rows: np.ndarray | None = None) -> "TrainingData":
        """From a WP5 library index directory (``peaks_*.npy``, ``offsets.npy``,
        ``meta.parquet``, ``structures.parquet``). ``rows``: subset of store rows."""
        cfg = cfg or Config()
        d = Path(directory) if directory is not None else paths.PROCESSED / "library"
        store = SpectrumStore.load(d, mmap=True)
        cols = ["ik14", "struct_idx", "precursor_mz", "adduct", "polarity",
                "instrument_family", "collision_energy", "n_energies", "base_peak_intensity",
                "precursor_error_ppm", "prec_intensity", "n_peaks_raw", "n_peaks", "ingest_lib"]
        meta = pd.read_parquet(d / "meta.parquet", columns=cols)
        same = float((meta["n_peaks_raw"] == meta["n_peaks"]).mean()) if len(meta) else 0.0
        if same > 0.9 and float(meta["prec_intensity"].fillna(0).abs().max() or 0) == 0.0:
            log.warning("library %s: n_peaks_raw equals the cleaned count for %.0f%% of spectra "
                        "and prec_intensity is 0 everywhere, so it was probably built from "
                        "already-cleaned spectra; these two model inputs will not match test "
                        "time. Prefer WP6's spectra files as the training source.", d, 100 * same)
        meta = meta.drop(columns=["n_peaks"])
        st = pd.read_parquet(d / "structures.parquet", columns=["ik14", "smiles", "mono_mass"])
        meta["mono_mass"] = st["mono_mass"].to_numpy()[meta["struct_idx"].to_numpy()]
        if rows is not None:
            rows = np.asarray(rows, dtype=np.int64)
            store = store.take(rows)
            meta = meta.iloc[rows].reset_index(drop=True)
        else:
            store = SpectrumStore(np.asarray(store.mz), np.asarray(store.intensity),
                                  np.asarray(store.offsets))
        if targets is None:
            used = np.unique(meta["struct_idx"].to_numpy())
            cache = default_targets_cache() if targets_cache == "default" else targets_cache
            targets = build_targets(st["ik14"].to_numpy(object)[used],
                                    st["smiles"].to_numpy(object)[used], cfg, n_jobs, cache)
        log.info("library %s: %d spectra, %d structures", d, len(meta), len(targets))
        return cls.from_parts(store, meta.drop(columns=["struct_idx"]), targets)

    @classmethod
    def from_parquet(cls, source: Path | str | Sequence[Path | str], cfg: Config | None = None,
                     n_jobs: int = 1, targets_cache: Path | str | None = "default",
                     batch_rows: int = 50000, max_rows: int | None = None) -> "TrainingData":
        """Stream train-schema parquet file(s) (or a directory of them, e.g. WP6's
        ``data/processed/train/spectra/``), clean (idempotent on cleaned peaks), build.
        ``n_peaks_raw`` / ``prec_intensity`` columns, when present, win over the values
        recomputed here."""
        import pyarrow.parquet as pq

        cfg = cfg or Config()
        if isinstance(source, (str, Path)):
            p = Path(source)
            files = sorted(p.glob("*.parquet")) if p.is_dir() else [p]
        else:
            files = [Path(s) for s in source]
        want = ["ms2_mzs", "ms2_normalized_intensities", "precursor_mz", "adduct",
                "ionization_mode", "instrument_type", "instrument_family", "ingest_lib",
                "collision_energy_ev", "base_peak_intensity", "precursor_error_ppm",
                "inchikey14", "ik14", "normalized_smiles", "smiles", "mono_mass",
                "n_peaks_raw", "prec_intensity"]
        import pyarrow as pa

        stores, metas, smiles_parts = [], [], []
        n_read = 0
        peak_cols = ("ms2_mzs", "ms2_normalized_intensities", "collision_energy_ev")
        for f in files:
            if max_rows is not None and n_read >= max_rows:
                break
            pf = pq.ParquetFile(f)
            cols = [c for c in want if c in pf.schema_arrow.names]
            for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
                if max_rows is not None:
                    batch = batch.slice(0, max(0, max_rows - n_read))
                if batch.num_rows == 0:
                    break
                n_read += batch.num_rows
                mz, it, off, valid = list_arrays_to_flat(batch.column("ms2_mzs"),
                                                         batch.column("ms2_normalized_intensities"))
                tab = pa.Table.from_batches([batch])
                m = tab.drop_columns([c for c in peak_cols if c in tab.column_names]).to_pandas()
                prec = pd.to_numeric(m["precursor_mz"], errors="coerce").to_numpy(np.float64)
                cm, ci, co, pint, nraw = clean_flat(mz, it, off, prec, cfg, valid)
                key = "ik14" if "ik14" in m.columns else "inchikey14"
                smi_col = "normalized_smiles" if "normalized_smiles" in m.columns else "smiles"
                m = m.rename(columns={key: "ik14"})
                # already-cleaned input (WP6 spectra files) carries the raw values
                if "n_peaks_raw" not in m.columns:
                    m["n_peaks_raw"] = nraw
                if "prec_intensity" not in m.columns:
                    m["prec_intensity"] = pint
                if "collision_energy_ev" in tab.column_names:
                    m["collision_energy"], m["n_energies"] = _list_mean_count(
                        tab.column("collision_energy_ev"))
                smiles_parts.append(m[["ik14", smi_col]].rename(columns={smi_col: "smiles"}))
                metas.append(m.drop(columns=[smi_col]))
                stores.append(SpectrumStore(cm, ci, co))
                if max_rows is not None and n_read >= max_rows:
                    break
        store = _concat_stores(stores)
        meta = pd.concat(metas, ignore_index=True)
        smi = pd.concat(smiles_parts, ignore_index=True).dropna()
        smi = smi.sort_values(["ik14", "smiles"], kind="stable").drop_duplicates("ik14")
        cache = default_targets_cache() if targets_cache == "default" else targets_cache
        targets = build_targets(smi["ik14"].to_numpy(object), smi["smiles"].to_numpy(object),
                                cfg, n_jobs, cache)
        log.info("parquet %s: %d spectra, %d structures", [str(f) for f in files], len(meta),
                 len(targets))
        return cls.from_parts(store, meta, targets)

    # ---- selection ------------------------------------------------------------------

    def eligible(self, cfg: Config | None = None) -> tuple[np.ndarray, dict[str, int]]:
        """Training filter of PLAN 4.4; returns (mask, rows removed per reason, in order)."""
        cfg = cfg or Config()
        mask = np.ones(len(self), bool)
        drops: dict[str, int] = {}

        def apply(name: str, ok: np.ndarray) -> None:
            nonlocal mask
            drops[name] = int((mask & ~ok).sum())
            mask &= ok

        tok = (self.target_row >= 0) & self.targets.ok[np.maximum(self.target_row, 0)]
        apply("no_target", tok)
        lm = self.label_mass
        apply("mass_range", np.isfinite(lm) & (lm >= cfg.fpm_mass_min) & (lm <= cfg.fpm_mass_max))
        apply("min_peaks", self.store.n_peaks >= cfg.min_peaks)
        apply("adduct_unparsed", self.meta.adduct_known)
        err = self.precursor_error_ppm
        apply("precursor_error", ~(np.abs(err) > cfg.fpm_max_precursor_error_ppm))
        return mask, drops

    def subset(self, rows: np.ndarray) -> "TrainingData":
        rows = np.asarray(rows, dtype=np.int64)
        return TrainingData(self.store.take(rows), self.meta.take(rows), self.target_row[rows],
                            self.targets, self.ik14[rows], self.label_mass[rows],
                            self.precursor_error_ppm[rows], self.ingest_lib[rows])

    # ---- consensus partner groups (augmentation) -------------------------------------

    def groups(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(order, group_start, group_end): spectra of the same structure, test adduct and
        instrument family, sorted by collision energy; per spectrum the slice of ``order``
        holding its group. Non-test adducts and unknown targets form singleton groups."""
        if self._groups is None:
            n = len(self)
            m = self.meta
            struct = self.target_row.copy()
            alone = (m.adduct_code >= ADDUCT_OTHER) | (struct < 0)
            uniq_key = np.where(alone, -(np.arange(n, dtype=np.int64) + 1), struct)
            order = np.lexsort((np.nan_to_num(m.collision_energy, nan=-1.0),
                                m.instrument_code, m.adduct_code, uniq_key))
            k1, k2, k3 = uniq_key[order], m.adduct_code[order], m.instrument_code[order]
            brk = np.ones(n, bool)
            if n > 1:
                brk[1:] = (k1[1:] != k1[:-1]) | (k2[1:] != k2[:-1]) | (k3[1:] != k3[:-1])
            gid_sorted = np.cumsum(brk) - 1
            starts = np.flatnonzero(brk)
            ends = np.append(starts[1:], n)
            gs = np.empty(n, np.int64)
            ge = np.empty(n, np.int64)
            gs[order] = starts[gid_sorted]
            ge[order] = ends[gid_sorted]
            self._groups = (order.astype(np.int64), gs, ge)
        return self._groups


def _concat_stores(stores: Sequence[SpectrumStore]) -> SpectrumStore:
    if not stores:
        return SpectrumStore.empty()
    mz = np.concatenate([s.mz for s in stores])
    it = np.concatenate([s.intensity for s in stores])
    offs = [np.asarray(stores[0].offsets, np.int64)]
    for s in stores[1:]:
        offs.append(np.asarray(s.offsets[1:], np.int64) - s.offsets[0] + offs[-1][-1])
    return SpectrumStore(mz, it, np.concatenate(offs))


def holdout_exclusion(train_ik14: np.ndarray, holdout: pd.DataFrame | Path | str,
                      structures: pd.DataFrame | None = None) -> tuple[np.ndarray, dict]:
    """Mask over ``train_ik14`` (True = held out) by ik14 and by mkey (PLAN 5.2).

    ``holdout``: splits/holdout_v1.csv (columns ik14, mkey, ...). ``structures``: a table
    with ik14 and mkey for the training structures (WP6 structures.parquet); without it
    only ik14 matching is possible (logged as a warning).
    """
    if not isinstance(holdout, pd.DataFrame):
        holdout = pd.read_csv(holdout, dtype=str)
    ik = pd.Series(np.asarray(train_ik14, dtype=object))
    ho_ik = set(holdout["ik14"].dropna().astype(str))
    mask = np.array(ik.isin(ho_ik).to_numpy(), dtype=bool)
    info = {"holdout_structures": len(ho_ik), "by_ik14": int(mask.sum()), "by_mkey": 0,
            "mkey_checked": False}
    if "mkey" in holdout.columns and structures is not None and "mkey" in structures.columns:
        ho_mk = set(holdout["mkey"].dropna().astype(str))
        st = structures[["ik14", "mkey"]].dropna().astype(str)
        bad_ik = set(st.loc[st["mkey"].isin(ho_mk), "ik14"])
        by_mk = ik.isin(bad_ik).to_numpy()
        info["by_mkey"] = int((by_mk & ~mask).sum())
        mask |= by_mk
        info["mkey_checked"] = True
        # Assertion required by WP7: no training structure shares a holdout mkey.
        remaining = set(ik[~mask]) & bad_ik
        assert not remaining, f"holdout mkey leak: {sorted(remaining)[:5]}"
    else:
        log.warning("holdout exclusion by ik14 only (no structures table with mkey)")
    return mask, info


# ======================================================================================
# Sampling
# ======================================================================================


def sampling_weights(data: TrainingData, cfg: Config | None = None) -> np.ndarray:
    """1 / sqrt(spectra of the structure) x fpm_test_like_weight for timsTOF test-adduct
    spectra (PLAN 4.4). float64, one per spectrum."""
    cfg = cfg or Config()
    tr = data.target_row
    cnt = np.bincount(np.maximum(tr, 0), minlength=len(data.targets)).astype(np.float64)
    w = 1.0 / np.sqrt(np.maximum(cnt[np.maximum(tr, 0)], 1.0))
    test_like = (data.meta.adduct_code < ADDUCT_OTHER) & (data.meta.instrument_code == 0)
    w = np.where(test_like, w * cfg.fpm_test_like_weight, w)
    return np.where(tr >= 0, w, 0.0)


def epoch_indices(weights: np.ndarray, size: int, seed: int, epoch: int) -> np.ndarray:
    """``size`` spectrum indices drawn with replacement with probability ~ ``weights``;
    deterministic in (seed, epoch)."""
    cw = np.cumsum(weights, dtype=np.float64)
    rng = np.random.default_rng([int(seed), 7, int(epoch)])
    u = rng.random(size) * cw[-1]
    return np.minimum(np.searchsorted(cw, u, side="right"), len(weights) - 1).astype(np.int64)


def pos_weights(data: TrainingData, weights: np.ndarray, cfg: Config | None = None,
                chunk: int = 4096) -> np.ndarray:
    """Per-bit positive weight n_neg / n_pos under the sampling distribution, clipped to
    [1, fpm_pos_weight_max] (bits never on get the maximum). float32 (n_bits,)."""
    cfg = cfg or Config()
    t = data.targets
    w_struct = np.bincount(np.maximum(data.target_row, 0), weights=weights,
                           minlength=len(t))
    rows = np.flatnonzero(w_struct > 0)
    pos = np.zeros(t.n_bits)
    for a in range(0, len(rows), chunk):
        r = rows[a:a + chunk]
        pos += w_struct[r].astype(np.float32) @ t.dense(r)
    total = w_struct.sum()
    freq = pos / max(total, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        pw = np.where(freq > 0, (1.0 - freq) / freq, np.inf)
    return np.clip(pw, 1.0, cfg.fpm_pos_weight_max).astype(np.float32)


# ======================================================================================
# Augmentation (training only, numpy + numba)
# ======================================================================================


def _numba_partner_kernel():
    from numba import njit

    @njit(cache=True)
    def pick(sel, n_extra, u, gstart, gend, order, ce, tol):
        n = sel.shape[0]
        members = np.empty(n * 3, np.int64)
        off = np.zeros(n + 1, np.int64)
        chosen = np.empty(3, np.float64)
        pos = 0
        for t in range(n):
            i = sel[t]
            members[pos] = i
            pos += 1
            k = 1
            chosen[0] = ce[i]
            g0 = gstart[i]
            L = gend[i] - g0
            if L > 1 and np.isfinite(ce[i]):
                start = int(u[t] * L) % L
                for r in range(L):
                    if k > n_extra[t]:
                        break
                    j = order[g0 + (start + r) % L]
                    c = ce[j]
                    if j == i or not np.isfinite(c):
                        continue
                    good = True
                    for q in range(k):
                        if abs(c - chosen[q]) <= tol:
                            good = False
                            break
                    if good:
                        members[pos] = j
                        pos += 1
                        chosen[k] = c
                        k += 1
            off[t + 1] = pos
        return members[:pos].copy(), off

    return pick


_PARTNER_KERNEL = None


def _pick_partners(*args):
    global _PARTNER_KERNEL
    if _PARTNER_KERNEL is None:
        _PARTNER_KERNEL = _numba_partner_kernel()
    return _PARTNER_KERNEL(*args)


def augment_batch(data: TrainingData, idx: np.ndarray, rng: np.random.Generator,
                  cfg: Config | None = None) -> tuple[SpectrumStore, SpectrumMeta]:
    """On-the-fly augmentation of spectra ``idx`` (PLAN 4.4), in this order:

    1. with probability ``fpm_aug_consensus_prob``: replace the spectrum by the consensus
       (``spectra.merge_groups``) of itself and 1 to ``fpm_aug_consensus_members_max`` - 1
       other spectra of the same structure, test adduct and instrument family at different
       collision energies (|dCE| > 0.5 eV); skipped when no such partner exists;
    2. drop each peak with probability ``fpm_aug_peak_drop`` (the base peak is kept);
    3. multiply intensities by exp(N(0, ``fpm_aug_intensity_sigma``)), renormalise to 1;
    4. non-timsTOF spectra: add U{0..``fpm_aug_noise_peaks_max``} noise peaks, m/z uniform
       in [``fpm_aug_noise_mz_min``, precursor - ``precursor_window_low_da``], intensity
       uniform in [``fpm_aug_noise_int_min``, ``fpm_aug_noise_int_max``].
    """
    cfg = cfg or Config()
    idx = np.asarray(idx, dtype=np.int64)
    b = len(idx)
    meta = data.meta.take(idx)
    # 1. consensus
    want = rng.random(b) < cfg.fpm_aug_consensus_prob
    sel = np.flatnonzero(want)
    n_extra = rng.integers(1, max(cfg.fpm_aug_consensus_members_max - 1, 1) + 1,
                           size=len(sel)).astype(np.int64)
    u = rng.random(len(sel))
    store = data.store.take(idx)
    if len(sel):
        order, gs, ge = data.groups()
        members, moff = _pick_partners(idx[sel], n_extra, u, gs, ge, order,
                                       data.meta.collision_energy, 0.5)
        size = np.diff(moff)
        merged = np.flatnonzero(size >= 2)
        if len(merged):
            g_members = np.concatenate([members[moff[t]:moff[t + 1]] for t in merged])
            g_off = np.zeros(len(merged) + 1, np.int64)
            np.cumsum(size[merged], out=g_off[1:])
            rows_b = sel[merged]
            cons = merge_groups(data.store, g_members, g_off,
                                data.meta.precursor_mz[idx[rows_b]], cfg)
            # metadata of the consensus spectra
            mm = data.meta.take(g_members)
            starts = g_off[:-1]
            ce = np.where(np.isfinite(mm.collision_energy), mm.collision_energy, 0.0)
            ce_n = np.isfinite(mm.collision_energy).astype(np.float64)
            s_ce = np.add.reduceat(ce, starts)
            s_n = np.add.reduceat(ce_n, starts)
            with np.errstate(invalid="ignore", divide="ignore"):
                meta.collision_energy[rows_b] = np.where(s_n > 0, s_ce / np.maximum(s_n, 1),
                                                         np.nan)
            meta.n_energies[rows_b] = np.add.reduceat(
                np.maximum(mm.n_energies.astype(np.int64), 1), starts).astype(np.int16)
            meta.base_peak_intensity[rows_b] = np.fmax.reduceat(mm.base_peak_intensity, starts)
            meta.n_peaks_raw[rows_b] = np.add.reduceat(np.nan_to_num(mm.n_peaks_raw), starts)
            meta.prec_intensity[rows_b] = np.fmax.reduceat(mm.prec_intensity, starts)
            # splice: rows_b come from ``cons``, the others from ``store``
            plain = np.setdiff1d(np.arange(b), rows_b)
            src = np.empty(b, np.int64)
            src[plain] = np.arange(len(plain))
            src[rows_b] = len(plain) + np.arange(len(rows_b))
            store = _concat_stores([store.take(plain), cons]).take(src)
    # 2-4. peak dropout, intensity jitter, noise peaks (vectorised over peaks)
    off = store.offsets
    npk = np.diff(off)
    sid = np.repeat(np.arange(b, dtype=np.int64), npk)
    mz = np.asarray(store.mz, dtype=np.float32)
    it = np.asarray(store.intensity, dtype=np.float32)
    base = it >= np.float32(1.0 - 1e-6)
    keep = (rng.random(len(it)) >= cfg.fpm_aug_peak_drop) | base
    if cfg.fpm_aug_intensity_sigma > 0:
        it = it * np.exp(rng.normal(0.0, cfg.fpm_aug_intensity_sigma, len(it))).astype(np.float32)
    mz, it, sid = mz[keep], it[keep], sid[keep]
    cnt = np.bincount(sid, minlength=b)
    nz = np.flatnonzero(cnt)
    starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    mx = np.ones(b, np.float32)
    if len(it):
        mx[nz] = np.maximum.reduceat(it, starts[nz])
    it = it / mx[sid]
    if cfg.fpm_aug_noise_peaks_max > 0:
        lo = np.float64(cfg.fpm_aug_noise_mz_min)
        hi = meta.precursor_mz - cfg.precursor_window_low_da
        can = (meta.instrument_code != 0) & np.isfinite(hi) & (hi > lo)
        k = np.where(can, rng.integers(0, cfg.fpm_aug_noise_peaks_max + 1, size=b), 0)
        ns = np.repeat(np.arange(b, dtype=np.int64), k)
        if len(ns):
            nm = (lo + rng.random(len(ns)) * (hi[ns] - lo)).astype(np.float32)
            ni = rng.uniform(cfg.fpm_aug_noise_int_min, cfg.fpm_aug_noise_int_max,
                             len(ns)).astype(np.float32)
            sid = np.concatenate([sid, ns])
            mz = np.concatenate([mz, nm])
            it = np.concatenate([it, ni])
            o = np.argsort(sid, kind="stable")
            sid, mz, it = sid[o], mz[o], it[o]
    new_off = np.zeros(b + 1, np.int64)
    np.cumsum(np.bincount(sid, minlength=b), out=new_off[1:])
    return SpectrumStore(mz.astype(np.float32), it.astype(np.float32), new_off), meta


def make_batch(data: TrainingData, idx: np.ndarray, spec: ModelSpec, cfg: Config | None = None,
               rng: np.random.Generator | None = None, augment: bool = True
               ) -> tuple[SparseBatch, np.ndarray, np.ndarray]:
    """(sparse input, metadata float32 (b, n_meta), targets float32 (b, n_out))."""
    cfg = cfg or Config()
    idx = np.asarray(idx, dtype=np.int64)
    if augment:
        store, meta = augment_batch(data, idx, rng if rng is not None else
                                    np.random.default_rng(cfg.seed), cfg)
    else:
        store, meta = data.store.take(idx), data.meta.take(idx)
    sp = featurize_sparse(store, meta.precursor_mz, spec)
    x = meta_matrix(meta, store.n_peaks, spec)
    y = data.targets.dense(data.target_row[idx])
    return sp, x, y


# ======================================================================================
# numpy model
# ======================================================================================


def _param_names(spec: ModelSpec) -> list[str]:
    names = ["emb", "meta_w1", "meta_b1", "meta_w2", "meta_b2", "ln0_g", "ln0_b"]
    for k in range(spec.n_blocks):
        names += [f"blk{k}_w", f"blk{k}_b", f"blk{k}_g", f"blk{k}_beta"]
    return names + ["out_w", "out_b"]


def _param_shapes(spec: ModelSpec) -> dict[str, tuple[int, ...]]:
    H, mh = spec.hidden, spec.meta_hidden
    shapes = {"emb": (spec.n_in, H), "meta_w1": (spec.n_meta, mh), "meta_b1": (mh,),
              "meta_w2": (mh, H), "meta_b2": (H,), "ln0_g": (H,), "ln0_b": (H,)}
    for k in range(spec.n_blocks):
        shapes.update({f"blk{k}_w": (H, H), f"blk{k}_b": (H,), f"blk{k}_g": (H,),
                       f"blk{k}_beta": (H,)})
    shapes.update({"out_w": (H, spec.n_out), "out_b": (spec.n_out,)})
    return shapes


def random_params(spec: ModelSpec, seed: int = 0, scale: float = 1.0) -> dict[str, np.ndarray]:
    """Random float32 weights (numpy only): for tests and the no-torch import check."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, shp in _param_shapes(spec).items():
        if name.endswith("_g"):
            out[name] = (1.0 + 0.1 * rng.standard_normal(shp)).astype(np.float32)
        elif len(shp) == 2:
            out[name] = (rng.standard_normal(shp) * scale / math.sqrt(shp[0])).astype(np.float32)
        else:
            out[name] = (0.05 * rng.standard_normal(shp)).astype(np.float32)
    out["emb"] *= np.float32(math.sqrt(spec.n_in / 150.0))  # ~150 active bins
    return out


def save_params(params: Mapping[str, np.ndarray], spec: ModelSpec, path: Path | str,
                extra: Mapping[str, Any] | None = None, float16: bool = False) -> Path:
    """Write the ``.npz`` (atomic). ``float16`` halves the size (inference casts back)."""
    p = Path(path)
    paths.ensure_dir(p.parent)
    shapes = _param_shapes(spec)
    arrs = {}
    for name in _param_names(spec):
        a = np.asarray(params[name])
        if a.shape != shapes[name]:
            raise ValueError(f"{name}: shape {a.shape}, expected {shapes[name]}")
        arrs[name] = a.astype(np.float16 if float16 else np.float32)
    arrs["spec_json"] = np.array(spec.to_json(extra))
    tmp = p.with_name(p.name + f".tmp{os.getpid()}.npz")
    np.savez(tmp, **arrs)
    os.replace(tmp, p)
    return p


def _layer_norm(x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    xc = x - mu
    var = np.mean(xc * xc, axis=1, keepdims=True)
    return xc / np.sqrt(var + np.float32(eps)) * g + b


def _gelu(x: np.ndarray) -> np.ndarray:
    return np.float32(0.5) * x * (np.float32(1.0) + np.tanh(
        np.float32(_GELU_C) * (x + np.float32(0.044715) * x * x * x)))


def _embedding_bag(E: np.ndarray, sp: SparseBatch, s0: int, s1: int,
                   max_entries: int = 16384) -> np.ndarray:
    """sum_j w_j E[idx_j] per spectrum s0..s1-1 (numpy, bounded memory)."""
    H = E.shape[1]
    out = np.zeros((s1 - s0, H), dtype=np.float32)
    off = sp.offsets
    s = s0
    while s < s1:
        # largest t with off[t] - off[s] <= max_entries (at least one spectrum)
        t = int(np.searchsorted(off, off[s] + max_entries, side="right")) - 1
        t = min(max(t, s + 1), s1)
        a, b = int(off[s]), int(off[t])
        if b > a:
            rows = E[sp.indices[a:b]].astype(np.float32, copy=False)
            rows = rows * sp.weights[a:b, None]
            lens = np.diff(off[s:t + 1])
            nz = np.flatnonzero(lens > 0)
            starts = (off[s:t] - a)[nz]
            out[s - s0 + nz] = np.add.reduceat(rows, starts, axis=0)
        s = t
    return out


def predict_numpy(params: Mapping[str, np.ndarray], spec: ModelSpec, sp: SparseBatch,
                  meta_x: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """Logits float32 (n, n_out) of the exported model, numpy only (no dropout)."""
    n = len(sp)
    out = np.empty((n, spec.n_out), dtype=np.float32)
    P = {k: np.asarray(v, dtype=np.float32) for k, v in params.items()
         if not k.startswith("spec")}
    meta_x = np.asarray(meta_x, dtype=np.float32)
    for s0 in range(0, n, chunk):
        s1 = min(n, s0 + chunk)
        h = _embedding_bag(P["emb"], sp, s0, s1)
        m = _gelu(meta_x[s0:s1] @ P["meta_w1"] + P["meta_b1"])
        h = h + (m @ P["meta_w2"] + P["meta_b2"])
        h = _gelu(_layer_norm(h, P["ln0_g"], P["ln0_b"], spec.ln_eps))
        for k in range(spec.n_blocks):
            z = h @ P[f"blk{k}_w"] + P[f"blk{k}_b"]
            h = h + _gelu(_layer_norm(z, P[f"blk{k}_g"], P[f"blk{k}_beta"], spec.ln_eps))
        out[s0:s1] = h @ P["out_w"] + P["out_b"]
    return out


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    with np.errstate(over="ignore"):
        return (1.0 / (1.0 + np.exp(-x))).astype(x.dtype if x.dtype.kind == "f" else np.float64)


class FpModel:
    """An exported model (``.npz``) with numpy inference."""

    def __init__(self, params: Mapping[str, np.ndarray], spec: ModelSpec,
                 extra: Mapping[str, Any] | None = None):
        self.spec = spec
        self.extra = dict(extra or {})
        self.params = {k: np.asarray(v, dtype=np.float32) for k, v in params.items()
                       if k != "spec_json"}

    @classmethod
    def load(cls, path: Path | str) -> "FpModel":
        with np.load(path, allow_pickle=False) as z:
            spec, extra = ModelSpec.from_json(str(z["spec_json"]))
            params = {k: z[k] for k in z.files if k != "spec_json"}
        return cls(params, spec, extra)

    def check_cfg(self, cfg: Config) -> None:
        """Warn when the cleaning settings differ from those the model was trained with."""
        now = _clean_settings(cfg)
        diff = {k: (self.spec.clean.get(k), v) for k, v in now.items()
                if k in self.spec.clean and self.spec.clean[k] != v}
        if diff:
            log.warning("fingerprint model trained with other cleaning settings: %s", diff)

    def predict_logits(self, store: SpectrumStore, meta: SpectrumMeta) -> np.ndarray:
        sp = featurize_sparse(store, meta.precursor_mz, self.spec)
        x = meta_matrix(meta, store.n_peaks, self.spec)
        return predict_numpy(self.params, self.spec, sp, x)

    def predict_frame(self, df: pd.DataFrame, cfg: Config | None = None) -> np.ndarray:
        """Spectrum logits for a test-schema frame (cleaned here), in row order."""
        cfg = cfg or Config()
        self.check_cfg(cfg)
        store, meta = spectra_from_frame(df, cfg)
        return self.predict_logits(store, meta)


# ======================================================================================
# Molecule-level aggregation
# ======================================================================================


@dataclass
class MoleculePredictions:
    """Weighted mean spectrum logits per molecule (rows follow ``molecule_id``, sorted)."""

    molecule_id: np.ndarray        # object, sorted unique
    logits: np.ndarray             # float32 (m, n_out), all spectra
    logits_pos: np.ndarray         # float32 (m, n_out), NaN rows if no positive spectrum
    logits_neg: np.ndarray         # float32 (m, n_out), NaN rows if no negative spectrum
    n_spectra: np.ndarray          # int64
    n_pos: np.ndarray
    n_neg: np.ndarray

    def proba(self, which: str = "all") -> np.ndarray:
        z = {"all": self.logits, "pos": self.logits_pos, "neg": self.logits_neg}[which]
        return sigmoid(z)

    def row(self, molecule_id: str) -> int:
        i = int(np.searchsorted(self.molecule_id.astype(str), str(molecule_id)))
        if i >= len(self.molecule_id) or str(self.molecule_id[i]) != str(molecule_id):
            raise KeyError(molecule_id)
        return i


def spectrum_weights(base_peak_intensity: np.ndarray, cfg: Config | None = None) -> np.ndarray:
    """clip(log10 bpi, fpm_bpi_log_min, fpm_bpi_log_max) / 3; missing or <= 0: the minimum."""
    cfg = cfg or Config()
    bpi = np.asarray(base_peak_intensity, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        lg = np.log10(np.where(np.isfinite(bpi) & (bpi > 0), bpi, 1.0))
    lg = np.where(np.isfinite(bpi) & (bpi > 0), lg, cfg.fpm_bpi_log_min)
    return np.clip(lg, cfg.fpm_bpi_log_min, cfg.fpm_bpi_log_max) / 3.0


def molecule_logits(logits: np.ndarray, molecule_id: Sequence[str],
                    base_peak_intensity: np.ndarray, polarity: np.ndarray,
                    cfg: Config | None = None) -> MoleculePredictions:
    """PLAN 4.4 molecule-level prediction: weighted average of spectrum logits."""
    logits = np.asarray(logits, dtype=np.float32)
    mid = np.asarray(molecule_id, dtype=object)
    w = spectrum_weights(base_peak_intensity, cfg)
    pol = np.asarray(polarity)
    if pol.dtype.kind in "OUS":
        pol = _polarity_from_mode(pol, np.full(len(pol), None, dtype=object))
    codes, uniq = pd.factorize(pd.Series(mid.astype(str)), sort=True)
    m = len(uniq)
    order = np.argsort(codes, kind="stable")
    sc = codes[order]
    starts = np.flatnonzero(np.r_[True, sc[1:] != sc[:-1]]) if len(sc) else np.zeros(0, int)
    zs = logits[order].astype(np.float64)
    out = []
    counts = []
    for sel in (np.ones(len(mid), bool), pol > 0, pol < 0):
        ww = np.where(sel, w, 0.0)[order]
        num = np.add.reduceat(zs * ww[:, None], starts, axis=0) if len(sc) else \
            np.zeros((0, logits.shape[1]))
        den = np.bincount(codes, weights=np.where(sel, w, 0.0), minlength=m)
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(np.where(den[:, None] > 0, num / den[:, None], np.nan).astype(np.float32))
        counts.append(np.bincount(codes[sel], minlength=m).astype(np.int64))
    return MoleculePredictions(np.asarray(uniq, dtype=object), out[0], out[1], out[2],
                               counts[0], counts[1], counts[2])


def predict_molecules(model: FpModel, df: pd.DataFrame, cfg: Config | None = None
                      ) -> MoleculePredictions:
    """Test-schema frame -> molecule-level logits (the call WP10 needs)."""
    cfg = cfg or Config()
    model.check_cfg(cfg)
    store, meta = spectra_from_frame(df, cfg)
    z = model.predict_logits(store, meta)
    return molecule_logits(z, df["molecule_id"].astype(str).to_numpy(object),
                           meta.base_peak_intensity, meta.polarity, cfg)


# ======================================================================================
# Per-bit calibration
# ======================================================================================


@dataclass
class BitCalibration:
    """Per-bit Platt scaling: p' = sigmoid(a_j * logit(p_j) + b_j)."""

    a: np.ndarray     # float64 (n_bits,)
    b: np.ndarray

    def apply_logits(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=np.float64) * self.a + self.b

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
        return sigmoid(self.apply_logits(np.log(p) - np.log1p(-p)))

    @classmethod
    def identity(cls, n_bits: int) -> "BitCalibration":
        return cls(np.ones(n_bits), np.zeros(n_bits))

    def save(self, path: Path | str, extra: Mapping[str, Any] | None = None) -> Path:
        p = Path(path)
        paths.ensure_dir(p.parent)
        np.savez(p, a=self.a, b=self.b, info_json=np.array(json.dumps(dict(extra or {}))))
        return p

    @classmethod
    def load(cls, path: Path | str) -> "BitCalibration":
        with np.load(path, allow_pickle=False) as z:
            return cls(z["a"].astype(np.float64), z["b"].astype(np.float64))


def fit_calibration(logits: np.ndarray, targets: np.ndarray, l2: float = 1.0,
                    n_iter: int = 30, chunk: int = 512) -> BitCalibration:
    """Fit per-bit Platt scaling on held-out (logits, 0/1 targets) by Newton's method.

    Minimises sum_i BCE + l2 / 2 ((a - 1)^2 + b^2) per bit, so rare bits stay close to the
    identity. ``logits`` and ``targets``: (n, n_bits) (molecule level, from holdout).
    """
    z_all = np.asarray(logits, dtype=np.float64)
    y_all = np.asarray(targets, dtype=np.float64)
    K = z_all.shape[1]
    a = np.ones(K)
    b = np.zeros(K)
    for c0 in range(0, K, chunk):
        z, y = z_all[:, c0:c0 + chunk], y_all[:, c0:c0 + chunk]
        aa, bb = a[c0:c0 + chunk].copy(), b[c0:c0 + chunk].copy()
        for _ in range(n_iter):
            p = sigmoid(z * aa + bb)
            r = p - y
            s = p * (1 - p)
            ga = (r * z).sum(0) + l2 * (aa - 1)
            gb = r.sum(0) + l2 * bb
            haa = (s * z * z).sum(0) + l2
            hab = (s * z).sum(0)
            hbb = s.sum(0) + l2
            det = haa * hbb - hab * hab
            da = (hbb * ga - hab * gb) / det
            db = (haa * gb - hab * ga) / det
            aa -= da
            bb -= db
            if max(np.abs(da).max(), np.abs(db).max()) < 1e-6:
                break
        a[c0:c0 + chunk], b[c0:c0 + chunk] = aa, bb
    return BitCalibration(a, b)


# ======================================================================================
# Candidate scores
# ======================================================================================

_POPCOUNT8 = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(float)


def _as_bits(fp: np.ndarray, n_bits: int) -> tuple[np.ndarray, bool]:
    """(array, packed?) for packed uint8 (n, ceil(n_bits/8)) or 0/1 (n, n_bits) input."""
    a = np.atleast_2d(np.asarray(fp))
    nbytes = (n_bits + 7) // 8
    if a.shape[1] == n_bits and (a.dtype == bool or n_bits != nbytes):
        return a, False
    if a.shape[1] == nbytes and a.dtype == np.uint8:
        return a, True
    raise ValueError(f"fingerprint matrix of shape {a.shape} / {a.dtype} does not match "
                     f"{n_bits} bits (packed: {nbytes} uint8 columns)")


def bit_dot(fp: np.ndarray, weights: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """fp @ weights for packed (big bit order) or 0/1 fingerprints; float64 (n, k).

    ``weights``: (n_bits,) or (n_bits, k). Rows are unpacked in chunks and multiplied in
    float64 (30k Morgan rows, k = 2: about 0.17 s; exact for 0/1 input).
    """
    w = np.asarray(weights, dtype=np.float64)
    vec = w.ndim == 1
    if vec:
        w = w[:, None]
    n_bits, k = w.shape
    a, packed = _as_bits(fp, n_bits)
    n = a.shape[0]
    out = np.empty((n, k), dtype=np.float64)
    for s in range(0, n, chunk):
        blk = a[s:s + chunk]
        bits = np.unpackbits(blk, axis=1, count=n_bits) if packed else blk
        out[s:s + chunk] = bits.astype(np.float64) @ w
    return out[:, 0] if vec else out


def _popcount(fp: np.ndarray, n_bits: int) -> np.ndarray:
    a, packed = _as_bits(fp, n_bits)
    if packed:
        return _POPCOUNT8[a].sum(axis=1)
    return a.astype(bool).sum(axis=1).astype(np.float64)


def candidate_scores(p: np.ndarray, morgan: np.ndarray, maccs: np.ndarray | None,
                     cfg: Config | None = None, calibration: BitCalibration | None = None,
                     n_morgan: int | None = None) -> dict[str, np.ndarray]:
    """Fingerprint-agreement features for one molecule's pool (PLAN 4.4).

    ``p``: predicted probabilities (n_morgan + 167,) = [Morgan | MACCS] (e.g.
    ``MoleculePredictions.proba()[row]``). ``morgan``: (n, 512) packed uint8 as
    ``chem.morgan_fp`` / ``db.np_fingerprints`` give, or (n, 4096) bool. ``maccs``: (n, 21)
    packed or (n, 167) bool, or None: then the full-vector scores use Morgan bits only and
    the ``maccs_*`` scores are NaN (pass MACCS for features with a fixed meaning).

    Returns float64 arrays (n,): ``cos``, ``ll`` (Bernoulli log-likelihood with p clipped
    to [fp_prob_min, fp_prob_max] after the optional calibration), ``tanimoto`` (soft
    Tanimoto), ``maccs_cos``, ``maccs_ll``, ``maccs_tanimoto``, and ``ll_rank`` (1 = best
    in the pool; ties keep input order).
    """
    cfg = cfg or Config()
    nm = int(n_morgan if n_morgan is not None else cfg.fp_morgan_bits)
    p = np.asarray(p, dtype=np.float64).ravel()
    if calibration is not None:
        p = calibration.transform(p)
    n_maccs = len(p) - nm
    if n_maccs < 0:
        raise ValueError(f"p has {len(p)} values, fewer than {nm} Morgan bits")
    pc = np.clip(p, cfg.fp_prob_min, cfg.fp_prob_max)
    lr = np.log(pc) - np.log1p(-pc)          # c=1 minus c=0 contribution
    l0 = np.log1p(-pc)
    parts = [("morgan", morgan, slice(0, nm))]
    if maccs is not None:
        if n_maccs <= 0:
            raise ValueError("MACCS fingerprints given but p has no MACCS bits")
        parts.append(("maccs", maccs, slice(nm, nm + n_maccs)))
    sums = {}
    for name, fp, sl in parts:
        d = bit_dot(fp, np.stack([p[sl], lr[sl]], axis=1))
        sums[name] = dict(pc=d[:, 0], lr=d[:, 1], pop=_popcount(fp, sl.stop - sl.start),
                          sp=p[sl].sum(), sp2=(p[sl] ** 2).sum(), l0=l0[sl].sum())
    n = len(sums["morgan"]["pc"])

    def scores(keys: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pc_ = sum(sums[k]["pc"] for k in keys)
        pop = sum(sums[k]["pop"] for k in keys)
        sp = sum(sums[k]["sp"] for k in keys)
        sp2 = sum(sums[k]["sp2"] for k in keys)
        ll = sum(sums[k]["l0"] + sums[k]["lr"] for k in keys)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = np.where(pop > 0, pc_ / (np.sqrt(sp2) * np.sqrt(pop) + 1e-12), 0.0)
            tan = pc_ / np.maximum(sp + pop - pc_, 1e-12)
        return cos, ll, tan

    out: dict[str, np.ndarray] = {}
    out["cos"], out["ll"], out["tanimoto"] = scores(list(sums))
    if "maccs" in sums:
        out["maccs_cos"], out["maccs_ll"], out["maccs_tanimoto"] = scores(["maccs"])
    else:
        out["maccs_cos"] = out["maccs_ll"] = out["maccs_tanimoto"] = np.full(n, np.nan)
    rank = np.empty(n, dtype=np.float64)
    rank[np.argsort(-out["ll"], kind="stable")] = np.arange(1, n + 1)
    out["ll_rank"] = rank
    return out


# ======================================================================================
# Synthetic data (tests, smoke runs, benchmarks)
# ======================================================================================


def synthetic_test_frame(n_spectra: int = 64, n_peaks: tuple[int, int] = (60, 250),
                         seed: int = 0, n_molecules: int | None = None) -> pd.DataFrame:
    """A test.parquet-shaped frame of random spectra (``molecule_id`` groups of ~3)."""
    rng = np.random.default_rng(seed)
    n_mol = n_molecules or max(1, n_spectra // 3)
    mol = rng.integers(0, n_mol, n_spectra)
    mass = rng.uniform(150, 900, n_mol)
    names = list(TEST_ADDUCT_NAMES)
    from .adducts import TEST_ADDUCTS

    rows = []
    for i in range(n_spectra):
        a = names[rng.choice(len(names), p=_ADDUCT_P)]
        prec = TEST_ADDUCTS[a].precursor_mz(mass[mol[i]])
        k = int(rng.integers(n_peaks[0], n_peaks[1] + 1))
        mz = np.sort(rng.uniform(20.0, prec + 1.5, k))
        it = rng.lognormal(0, 1.5, k)
        it = it / it.max()
        merged = rng.random() < 0.25
        ce = [20.0, 40.0, 60.0] if merged else [float(rng.choice([10, 20, 40, 60]))]
        rows.append(dict(molecule_id=f"m_{mol[i]:06x}", spectrum_id=f"s_{i:08x}",
                         ms2_mzs=mz.tolist(), ms2_normalized_intensities=it.tolist(),
                         base_peak_intensity=float(10 ** rng.uniform(3, 7)), adduct=a,
                         ionization_mode="positive" if a.endswith("+") else "negative",
                         instrument_type="timsTOF", precursor_mz=float(prec),
                         collision_energy_orig=",".join(str(int(c)) for c in ce),
                         collision_energy_ev=ce, collision_energy_orig_units="eV"))
    return pd.DataFrame(rows)


_ADDUCT_P = np.array([0.6, 0.03, 0.03, 0.01, 0.04, 0.01, 0.2, 0.02, 0.05, 0.01])
_ADDUCT_P = _ADDUCT_P / _ADDUCT_P.sum()


def synthetic_training_data(n_structures: int = 64, spectra_per_structure: int = 4,
                            n_peaks: tuple[int, int] = (20, 100), seed: int = 0,
                            cfg: Config | None = None, morgan_on: float = 0.012,
                            maccs_on: float = 0.25, noise_peaks: int = 0) -> TrainingData:
    """Random structures (random target bits, a characteristic fragment set each) and
    spectra sampled from them, cleaned with ``clean_flat``. Learnable by memorisation:
    a structure's spectra share most fragments."""
    from .adducts import TEST_ADDUCTS

    cfg = cfg or Config()
    rng = np.random.default_rng(seed)
    S = n_structures
    mass = rng.uniform(150.0, 900.0, S)
    nm, nk = cfg.fp_morgan_bits, 167
    morgan = np.packbits(rng.random((S, nm)) < morgan_on, axis=1)
    maccs = np.packbits(rng.random((S, nk)) < maccs_on, axis=1)
    ik = np.array([f"SYN{i:011d}" for i in range(S)], dtype=object)
    frag_sets = [np.sort(rng.uniform(30.0, mass[s] - 1.0, int(rng.integers(n_peaks[0],
                 n_peaks[1] + 1)))) for s in range(S)]
    frag_int = [rng.lognormal(0, 1.2, len(f)) for f in frag_sets]
    names = list(TEST_ADDUCT_NAMES)
    insts = np.array(INSTRUMENT_FAMILIES, dtype=object)
    mzs, its, lens, rows = [], [], [], []
    for s in range(S):
        for r in range(spectra_per_structure):
            a = names[rng.choice(len(names), p=_ADDUCT_P)]
            prec = TEST_ADDUCTS[a].precursor_mz(mass[s])
            keep = rng.random(len(frag_sets[s])) < 0.85
            m = frag_sets[s][keep] + rng.normal(0, 0.002, keep.sum())
            x = frag_int[s][keep] * np.exp(rng.normal(0, 0.2, keep.sum()))
            if noise_peaks:
                m = np.concatenate([m, rng.uniform(20, prec, noise_peaks)])
                x = np.concatenate([x, rng.uniform(0.001, 0.02, noise_peaks) * x.max()])
            o = np.argsort(m)
            m, x = m[o], x[o] / x.max()
            mzs.append(m)
            its.append(x)
            lens.append(len(m))
            ce = float(rng.choice([10.0, 20.0, 30.0, 40.0, 60.0]))
            rows.append(dict(ik14=ik[s], precursor_mz=prec, adduct=a,
                             polarity=1 if a.endswith("+") else -1,
                             collision_energy=ce if rng.random() > 0.05 else np.nan,
                             n_energies=1,
                             instrument_family=insts[rng.choice(4, p=[0.5, 0.2, 0.2, 0.1])],
                             base_peak_intensity=(float(10 ** rng.uniform(3, 7))
                                                  if rng.random() > 0.2 else np.nan),
                             mono_mass=mass[s], precursor_error_ppm=rng.normal(0, 3),
                             ingest_lib="synthetic"))
    off = np.zeros(len(lens) + 1, np.int64)
    np.cumsum(lens, out=off[1:])
    mz = np.concatenate(mzs)
    it = np.concatenate(its)
    meta = pd.DataFrame(rows)
    cm, ci, co, pint, nraw = clean_flat(mz, it, off, meta["precursor_mz"].to_numpy(), cfg)
    meta["n_peaks_raw"] = nraw
    meta["prec_intensity"] = pint
    targets = FpTargets.from_arrays(ik, morgan, maccs, None, mass, nm, nk)
    return TrainingData.from_parts(SpectrumStore(cm, ci, co), meta, targets)


# ======================================================================================
# Training-run helpers (used by scripts/train_fpmodel.py; numpy only)
# ======================================================================================


def load_training_source(source: Path | str | None, cfg: Config | None = None, n_jobs: int = 1,
                         synthetic: int | None = None, max_rows: int | None = None,
                         targets_cache: Path | str | None = "default") -> TrainingData:
    """TrainingData from a WP5 library directory, train-schema parquet, or synthetic.

    ``source`` None: WP6's ``paths.PROCESSED/train/spectra/*.parquet`` if present (they
    carry the raw peak counts and precursor intensities), else the WP5 library
    ``paths.PROCESSED/library``, else ``paths.TRAIN_PARQUET``.
    ``synthetic``: number of synthetic spectra (4 per structure, ~100 peaks), no files.
    ``max_rows``: library - a random row subset (deterministic); parquet - the first rows.
    ``targets_cache``: "default" = ``default_targets_cache()``, None = no cache.
    """
    cfg = cfg or Config()
    if synthetic:
        return synthetic_training_data(max(1, synthetic // 4), 4, n_peaks=(40, 160),
                                       seed=cfg.seed, cfg=cfg, noise_peaks=20)
    if source is None:
        wp6 = paths.PROCESSED / "train" / "spectra"
        lib = paths.PROCESSED / "library"
        if wp6.is_dir() and any(wp6.glob("*.parquet")):
            source = wp6
        elif (lib / "meta.parquet").is_file():
            source = lib
        else:
            source = paths.TRAIN_PARQUET
        log.info("training source: %s", source)
    p = Path(source)
    if p.is_dir() and (p / "meta.parquet").is_file() and (p / "offsets.npy").is_file():
        rows = None
        if max_rows is not None:
            n = len(np.load(p / "offsets.npy", mmap_mode="r")) - 1
            if n > max_rows:
                rng = np.random.default_rng([cfg.seed, 31])
                rows = np.sort(rng.choice(n, max_rows, replace=False))
        return TrainingData.from_library(p, cfg, n_jobs=n_jobs, rows=rows,
                                         targets_cache=targets_cache)
    if not p.exists():
        raise FileNotFoundError(f"training source {p} not found (train data not ingested?)")
    return TrainingData.from_parquet(p, cfg, n_jobs=n_jobs, max_rows=max_rows,
                                     targets_cache=targets_cache)


def _cap_per_group(keys: np.ndarray, rows: np.ndarray, cap: int, seed: int) -> np.ndarray:
    """At most ``cap`` of ``rows`` per key (random, deterministic), sorted."""
    if len(rows) == 0:
        return rows
    rng = np.random.default_rng([seed, 13])
    r = rows[rng.permutation(len(rows))]
    k = pd.Series(keys[r].astype(str))
    rank = k.groupby(k).cumcount().to_numpy()
    return np.sort(r[rank < cap])


def split_training_data(data: TrainingData, cfg: Config | None = None, *,
                        exclude_holdout: bool = True,
                        holdout: pd.DataFrame | Path | str | None = None,
                        structures: pd.DataFrame | Path | str | None = None,
                        max_spectra: int | None = None
                        ) -> tuple[TrainingData, TrainingData | None, TrainingData | None, dict]:
    """Eligibility filter + holdout exclusion -> (train, val, calib, info).

    ``val``: up to ``fpm_val_max_spectra`` eligible holdout spectra (monitoring);
    ``calib``: holdout spectra with test adducts, at most
    ``fpm_calib_max_spectra_per_molecule`` per structure (per-bit calibration). Both None
    with ``exclude_holdout=False``. ``holdout`` None with synthetic data: 5 percent of the
    structures. ``max_spectra``: random training subset (smoke runs).
    """
    cfg = cfg or Config()
    ok, drops = data.eligible(cfg)
    info: dict[str, Any] = {"n_input": len(data), "drops": drops, "n_eligible": int(ok.sum())}
    log.info("eligible %d of %d spectra; removed: %s", ok.sum(), len(data), drops)
    ho = np.zeros(len(data), bool)
    if exclude_holdout:
        if holdout is None or (isinstance(holdout, (str, Path)) and not Path(holdout).is_file()):
            if not all(str(k).startswith("SYN") for k in data.targets.ik14[:100]):
                raise FileNotFoundError(f"holdout file {holdout} not found (run WP7 make_splits)")
            keys = data.targets.ik14
            pick = np.random.default_rng([cfg.seed, 17]).random(len(keys)) < 0.05
            holdout = pd.DataFrame({"ik14": keys[pick].astype(str)})
            log.info("synthetic data: %d of %d structures held out", pick.sum(), len(keys))
        st = structures
        if isinstance(st, (str, Path)):
            st = pd.read_parquet(st, columns=["ik14", "mkey"]) if Path(st).is_file() else None
        ho, hinfo = holdout_exclusion(data.ik14, holdout, st)
        info["holdout"] = hinfo
    train_rows = np.flatnonzero(ok & ~ho)
    if max_spectra is not None and len(train_rows) > max_spectra:
        rng = np.random.default_rng([cfg.seed, 19])
        train_rows = np.sort(rng.choice(train_rows, max_spectra, replace=False))
    info["n_train"] = len(train_rows)
    train = data.subset(train_rows)
    val = calib = None
    if exclude_holdout:
        ho_rows = np.flatnonzero(ok & ho)
        rng = np.random.default_rng([cfg.seed, 23])
        v = ho_rows if len(ho_rows) <= cfg.fpm_val_max_spectra else np.sort(
            rng.choice(ho_rows, cfg.fpm_val_max_spectra, replace=False))
        c = ho_rows[data.meta.adduct_code[ho_rows] < ADDUCT_OTHER]
        c = _cap_per_group(data.ik14, c, cfg.fpm_calib_max_spectra_per_molecule, cfg.seed)
        val = data.subset(v) if len(v) else None
        calib = data.subset(c) if len(c) else None
        info["n_val"], info["n_calib"] = len(v), len(c)
        info["n_calib_structures"] = int(len(set(data.ik14[c].tolist())))
    lib = pd.Series(train.ingest_lib).fillna("unknown").value_counts()
    log.info("training spectra per library: %s", lib.to_dict())
    log.info("split: train %d, val %s, calib %s", info["n_train"], info.get("n_val"),
             info.get("n_calib"))
    return train, val, calib, info


def calibrate(model: "FpModel", data: TrainingData, cfg: Config | None = None
              ) -> tuple[BitCalibration, dict]:
    """Fit per-bit calibration on held-out structures (molecule = ik14, spectra aggregated
    as at inference) and report molecule-level metrics, with a 2-fold estimate of the
    calibration gain in Bernoulli log-likelihood."""
    cfg = cfg or Config()
    z = model.predict_logits(data.store, data.meta)
    mp = molecule_logits(z, data.ik14, data.meta.base_peak_intensity, data.meta.polarity, cfg)
    rows = data.targets.rows(mp.molecule_id)
    y = data.targets.dense(rows).astype(np.float64)
    zl = mp.logits.astype(np.float64)

    def ll(p: np.ndarray, sel: np.ndarray | slice = slice(None)) -> np.ndarray:
        pc = np.clip(p[sel], cfg.fp_prob_min, cfg.fp_prob_max)
        yy = y[sel]
        return (yy * np.log(pc) + (1 - yy) * np.log1p(-pc)).sum(1)

    p0 = sigmoid(zl)
    cos = (p0 * y).sum(1) / np.maximum(np.linalg.norm(p0, axis=1) * np.linalg.norm(y, axis=1),
                                       1e-12)
    fold = np.random.default_rng([cfg.seed, 29]).random(len(rows)) < 0.5
    ll_cv = np.empty(len(rows))
    for f in (True, False):
        tr, te = fold == f, fold != f
        if tr.sum() and te.sum():
            c = fit_calibration(zl[tr], y[tr], cfg.fpm_calib_l2)
            ll_cv[te] = ll(sigmoid(c.apply_logits(zl)), te)
        else:
            ll_cv[te] = ll(p0, te)
    cal = fit_calibration(zl, y, cfg.fpm_calib_l2)
    ll0 = ll(p0)
    metrics = {"n_molecules": int(len(rows)), "n_spectra": int(len(data)),
               "mean_cosine": float(cos.mean()), "mean_ll": float(ll0.mean()),
               "mean_ll_calibrated_2fold": float(ll_cv.mean()),
               "mean_ll_calibrated_insample": float(ll(sigmoid(cal.apply_logits(zl))).mean())}
    log.info("calibration on %d holdout molecules: %s", len(rows), metrics)
    return cal, metrics


def _inchi_roundtrip_fp(smiles: Sequence[str | None], cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Morgan fingerprints after std_mol + MolFromInchi(MolToInchi(.)) (mobile-H normalised)."""
    from rdkit import Chem, rdBase

    from .chem import morgan_fp, std_mol

    n = len(smiles)
    out = np.zeros((n, (cfg.fp_morgan_bits + 7) // 8), np.uint8)
    ok = np.zeros(n, bool)
    with rdBase.BlockLogs():
        for i, s in enumerate(smiles):
            m = std_mol(s)
            if m is None:
                continue
            try:
                inchi = Chem.MolToInchi(m)
                r = Chem.MolFromInchi(inchi) if inchi else None
            except Exception:  # noqa: BLE001
                r = None
            fp = morgan_fp(r if r is not None else m, cfg)
            if fp is not None:
                out[i], ok[i] = fp, True
    return out, ok


def fingerprint_consistency(smiles_a: Sequence[str | None], smiles_b: Sequence[str | None],
                            cfg: Config | None = None, n_jobs: int = 1,
                            inchi_roundtrip: bool = True) -> pd.DataFrame:
    """Tanimoto between the target-style Morgan fingerprints (``chem.std_mol``) of two
    SMILES of the same structure (PLAN 4.4 "known weakness": train SMILES vs PubChem
    SMILES for the same ik14). Columns: ``tanimoto``, ``identical`` and, with
    ``inchi_roundtrip``, the same after an InChI round trip (``tanimoto_inchi``,
    ``identical_inchi``); NaN where a SMILES failed."""
    from .chem import morgan_fps, tanimoto

    cfg = cfg or Config()
    a, oka = morgan_fps(list(smiles_a), cfg, n_jobs=n_jobs)
    b, okb = morgan_fps(list(smiles_b), cfg, n_jobs=n_jobs)

    def pairwise(x: np.ndarray, y: np.ndarray, ok: np.ndarray) -> np.ndarray:
        t = np.array([tanimoto(x[i], y[i:i + 1])[0] for i in range(len(x))], dtype=np.float64)
        return np.where(ok, t, np.nan)

    out = pd.DataFrame({"tanimoto": pairwise(a, b, oka & okb)})
    out["identical"] = np.where(oka & okb, (a == b).all(axis=1), np.nan)
    if inchi_roundtrip:
        ai, oai = _inchi_roundtrip_fp(list(smiles_a), cfg)
        bi, obi = _inchi_roundtrip_fp(list(smiles_b), cfg)
        out["tanimoto_inchi"] = pairwise(ai, bi, oai & obi)
        out["identical_inchi"] = np.where(oai & obi, (ai == bi).all(axis=1), np.nan)
    return out


# ======================================================================================
# torch half (training only): every torch import is local
# ======================================================================================


def _torch_module(spec: ModelSpec, sparse_embedding: bool = False):
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self, h: int, p: float):
            super().__init__()
            self.lin = nn.Linear(h, h)
            self.ln = nn.LayerNorm(h, eps=spec.ln_eps)
            self.act = nn.GELU(approximate="tanh")
            self.drop = nn.Dropout(p)

        def forward(self, x):
            return x + self.drop(self.act(self.ln(self.lin(x))))

    class FpNet(nn.Module):
        def __init__(self):
            super().__init__()
            H = spec.hidden
            self.emb = nn.EmbeddingBag(spec.n_in, H, mode="sum", sparse=sparse_embedding)
            nn.init.normal_(self.emb.weight, 0.0, 1.0 / math.sqrt(150.0))
            self.meta1 = nn.Linear(spec.n_meta, spec.meta_hidden)
            self.meta2 = nn.Linear(spec.meta_hidden, H)
            self.ln0 = nn.LayerNorm(H, eps=spec.ln_eps)
            self.act = nn.GELU(approximate="tanh")
            self.blocks = nn.ModuleList([Block(H, spec.dropout) for _ in range(spec.n_blocks)])
            self.out = nn.Linear(H, spec.n_out)

        def forward(self, idx, w, off, meta):
            h = self.emb(idx, off, per_sample_weights=w)
            h = h + self.meta2(self.act(self.meta1(meta)))
            h = self.act(self.ln0(h))
            for blk in self.blocks:
                h = blk(h)
            return self.out(h)

    return FpNet()


def _to_torch(sp: SparseBatch, x: np.ndarray, y: np.ndarray | None = None):
    import torch

    t = (torch.from_numpy(sp.indices), torch.from_numpy(sp.weights),
         torch.from_numpy(sp.offsets[:-1].copy()), torch.from_numpy(x))
    return t + ((torch.from_numpy(y),) if y is not None else ())


def _torch_params_to_numpy(net, spec: ModelSpec) -> dict[str, np.ndarray]:
    sd = {k: v.detach().float().cpu().numpy() for k, v in net.state_dict().items()}
    out = {"emb": sd["emb.weight"], "meta_w1": sd["meta1.weight"].T, "meta_b1": sd["meta1.bias"],
           "meta_w2": sd["meta2.weight"].T, "meta_b2": sd["meta2.bias"],
           "ln0_g": sd["ln0.weight"], "ln0_b": sd["ln0.bias"],
           "out_w": sd["out.weight"].T, "out_b": sd["out.bias"]}
    for k in range(spec.n_blocks):
        out[f"blk{k}_w"] = sd[f"blocks.{k}.lin.weight"].T
        out[f"blk{k}_b"] = sd[f"blocks.{k}.lin.bias"]
        out[f"blk{k}_g"] = sd[f"blocks.{k}.ln.weight"]
        out[f"blk{k}_beta"] = sd[f"blocks.{k}.ln.bias"]
    return {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in out.items()}


def export_npz(net, spec: ModelSpec, path: Path | str, extra: Mapping[str, Any] | None = None,
               float16: bool = False) -> Path:
    """Torch model -> ``.npz`` for ``FpModel.load`` / ``predict_numpy``."""
    return save_params(_torch_params_to_numpy(net, spec), spec, path, extra, float16)


def torch_predict(net, sp: SparseBatch, x: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """Logits from the torch model in eval mode (float32), for the equality test."""
    import torch

    net.eval()
    outs = []
    with torch.no_grad():
        for s0 in range(0, len(sp), chunk):
            s1 = min(len(sp), s0 + chunk)
            a, b = sp.offsets[s0], sp.offsets[s1]
            sub = SparseBatch(sp.indices[a:b], sp.weights[a:b], sp.offsets[s0:s1 + 1] - a)
            outs.append(net(*_to_torch(sub, x[s0:s1])).float().numpy())
    return np.concatenate(outs) if outs else np.zeros((0, net.out.out_features), np.float32)


def _loss(logits, y, pos_weight, cos_weight: float):
    import torch
    import torch.nn.functional as F

    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight,
                                             reduction="none").mean(dim=1)
    p = torch.sigmoid(logits)
    cos = (p * y).sum(1) / (p.norm(dim=1) * y.norm(dim=1)).clamp_min(1e-8)
    loss = bce + cos_weight * (1.0 - cos)
    return loss.mean(), bce.mean(), cos.mean()


def _lr_at(step: int, total: int | None, cfg: Config) -> float:
    warm = max(int(cfg.fpm_warmup_steps), 1)
    lr = cfg.fpm_lr * min(1.0, (step + 1) / warm)
    if total is not None and total > 0:
        prog = min(max(step - warm, 0) / max(total - warm, 1), 1.0)
        lr *= cfg.fpm_lr_min_frac + (1 - cfg.fpm_lr_min_frac) * 0.5 * (1 + math.cos(math.pi * prog))
    return lr


@dataclass
class TrainResult:
    steps: int
    total_steps: int
    epochs: float
    seconds: float
    spectra_per_s: float
    final_loss: float
    val: dict
    npz_path: Path | None
    curve: list


def _adamw(params, cfg: Config):
    """AdamW; the fused CPU kernel when ``fpm_fused_adamw`` (6x faster step on 27M
    parameters here), falling back to the default implementation if unsupported."""
    import torch

    params = list(params)
    if cfg.fpm_fused_adamw:
        try:
            return torch.optim.AdamW(params, lr=cfg.fpm_lr, weight_decay=cfg.fpm_weight_decay,
                                     fused=True)
        except (RuntimeError, TypeError, ValueError) as exc:  # older torch
            log.warning("fused AdamW unavailable (%s); using the default", exc)
    return torch.optim.AdamW(params, lr=cfg.fpm_lr, weight_decay=cfg.fpm_weight_decay)


def _make_optimizers(net, cfg: Config, sparse_embedding: bool):
    import torch

    if sparse_embedding:
        emb = [net.emb.weight]
        rest = [p for n, p in net.named_parameters() if not n.startswith("emb.")]
        return [torch.optim.SparseAdam(emb, lr=cfg.fpm_lr), _adamw(rest, cfg)]
    return [_adamw(net.parameters(), cfg)]


def evaluate(net, data: TrainingData, spec: ModelSpec, cfg: Config, pos_weight,
             rows: np.ndarray | None = None) -> dict:
    """Loss and mean cosine on ``data`` (no augmentation, eval mode)."""
    import torch

    rows = np.arange(len(data)) if rows is None else np.asarray(rows, np.int64)
    net.eval()
    tot = {"loss": 0.0, "bce": 0.0, "cos": 0.0}
    n = 0
    with torch.no_grad():
        for s in range(0, len(rows), cfg.fpm_batch_size):
            r = rows[s:s + cfg.fpm_batch_size]
            sp, x, y = make_batch(data, r, spec, cfg, augment=False)
            idx, w, off, xm, yt = _to_torch(sp, x, y)
            loss, bce, cos = _loss(net(idx, w, off, xm), yt, pos_weight,
                                   cfg.fpm_cosine_loss_weight)
            k = len(r)
            tot["loss"] += float(loss) * k
            tot["bce"] += float(bce) * k
            tot["cos"] += float(cos) * k
            n += k
    net.train()
    return {k: v / max(n, 1) for k, v in tot.items()} | {"n": n}


def train(data: TrainingData, cfg: Config | None = None, *, spec: ModelSpec | None = None,
          hours: float | None = None, steps: int | None = None, epochs: float | None = None,
          val: TrainingData | None = None, out_npz: Path | str | None = None,
          checkpoint: Path | str | None = None, resume: bool = False,
          curve_csv: Path | str | None = None, augment: bool = True,
          threads: int | None = None, log_every: int | None = None,
          extra_info: Mapping[str, Any] | None = None, float16_export: bool = False,
          return_net: bool = False, measure_steps: int = 100):
    """Train on ``data`` (already filtered) with the PLAN 4.4 recipe.

    Length: ``steps`` (exact), else ``epochs`` x len(data) / batch, else the ``hours``
    budget (default ``cfg.fpm_hours``): the throughput over ``measure_steps`` steps
    (51k spectra at batch 512, after 10 warm-up steps) fixes the total step count (logged,
    so ``steps=`` reproduces the schedule), and training also stops at the deadline. LR:
    linear warm-up then cosine decay over the total. Checkpoints (and validation on
    ``val``) every ``fpm_checkpoint_minutes`` and at the end; ``resume`` continues from
    ``checkpoint`` (a finished run just re-exports). ``TrainResult.final_loss`` is the
    mean training loss of the last 20 steps. Returns ``TrainResult`` (and the net if asked).
    """
    import torch

    cfg = cfg or Config()
    spec = spec or ModelSpec.from_config(cfg)
    threads = int(threads or cfg.fpm_threads)
    torch.set_num_threads(threads)
    torch.manual_seed(cfg.seed)
    bs = int(cfg.fpm_batch_size)
    log_every = int(log_every or cfg.fpm_log_every_steps)
    sparse_emb = bool(cfg.fpm_sparse_embedding)
    net = _torch_module(spec, sparse_emb)
    opts = _make_optimizers(net, cfg, sparse_emb)
    weights = sampling_weights(data, cfg)
    pw_np = pos_weights(data, weights, cfg)
    pos_weight = torch.from_numpy(pw_np)
    epoch_size = int(cfg.fpm_epoch_size) if cfg.fpm_epoch_size > 0 else len(data)
    batches_per_epoch = max(1, epoch_size // bs)
    total: int | None = None
    if steps is not None:
        total = int(steps)
    elif epochs is not None:
        total = max(1, int(round(epochs * batches_per_epoch)))
    budget_s = None if total is not None else float(hours if hours is not None
                                                    else cfg.fpm_hours) * 3600.0
    step, epoch, batch_in_epoch = 0, 0, 0
    curve: list[dict] = []
    elapsed_before = 0.0
    ckpt = Path(checkpoint) if checkpoint is not None else None
    if resume and ckpt is not None and ckpt.is_file():
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        net.load_state_dict(st["model"])
        for o, s in zip(opts, st["opts"]):
            o.load_state_dict(s)
        step, epoch, batch_in_epoch = st["step"], st["epoch"], st["batch_in_epoch"]
        total = st["total_steps"] if total is None else total
        curve = st.get("curve", [])
        elapsed_before = st.get("elapsed", 0.0)
        torch.set_rng_state(st["torch_rng"])
        log.info("resumed from %s at step %d (epoch %d, batch %d)", ckpt, step, epoch,
                 batch_in_epoch)
    log.info("train: %d spectra, %d structures, batch %d, %d batches/epoch, threads %d, "
             "params %.1fM, bf16 %s, sparse embedding %s, steps %s, budget %s s",
             len(data), int((np.bincount(np.maximum(data.target_row, 0)) > 0).sum()), bs,
             batches_per_epoch, threads, sum(p.numel() for p in net.parameters()) / 1e6,
             cfg.fpm_bf16, sparse_emb, total, budget_s)
    log.info("pos weights: min %.2f median %.2f max %.2f (%d bits at the cap)",
             pw_np.min(), float(np.median(pw_np)), pw_np.max(),
             int((pw_np >= cfg.fpm_pos_weight_max).sum()))
    curve_f = None
    if curve_csv is not None:
        cp = Path(curve_csv)
        paths.ensure_dir(cp.parent)
        new = not (resume and cp.is_file())
        curve_f = open(cp, "w" if new else "a", encoding="utf-8")
        if new:
            curve_f.write("step,epoch,elapsed_s,lr,loss,bce,train_cos,spectra_per_s,"
                          "val_loss,val_cos\n")
    t0 = time.perf_counter()
    deadline = None if budget_s is None else t0 + budget_s - elapsed_before
    last_ckpt = t0
    ep_idx = epoch_indices(weights, epoch_size, cfg.seed, epoch)
    acc = {"loss": 0.0, "bce": 0.0, "cos": 0.0, "n": 0}
    recent: list[float] = []            # losses of the last 20 steps (final_loss)
    t_log = time.perf_counter()
    n_log = 0
    t_measure = None
    measured_from = step

    def next_batch(ep: int, bi: int, eidx: np.ndarray):
        rng = np.random.default_rng([int(cfg.seed), 11, ep, bi])
        r = eidx[bi * bs:(bi + 1) * bs]
        return make_batch(data, r, spec, cfg, rng, augment)

    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(1) if cfg.fpm_prefetch else None
    fut = pool.submit(next_batch, epoch, batch_in_epoch, ep_idx) if pool else None
    net.train()
    stop_reason = "steps"
    val_res: dict = {}

    def save_ckpt() -> None:
        if ckpt is None:
            return
        paths.ensure_dir(ckpt.parent)
        tmp = ckpt.with_name(ckpt.name + ".tmp")
        torch.save({"model": net.state_dict(), "opts": [o.state_dict() for o in opts],
                    "step": step, "epoch": epoch, "batch_in_epoch": batch_in_epoch,
                    "total_steps": total, "curve": curve, "torch_rng": torch.get_rng_state(),
                    "elapsed": elapsed_before + time.perf_counter() - t0,
                    "spec": spec.to_json(), "cfg": cfg.to_dict()}, tmp)
        os.replace(tmp, ckpt)
        log.info("checkpoint %s (step %d)", ckpt, step)

    try:
        while True:
            if total is not None and step >= total:
                break
            now = time.perf_counter()
            if deadline is not None and now >= deadline:
                stop_reason = "deadline"
                break
            sp, x, y = fut.result() if fut is not None else next_batch(epoch, batch_in_epoch,
                                                                         ep_idx)
            # advance the epoch cursor and prefetch the next batch
            batch_in_epoch += 1
            if batch_in_epoch >= batches_per_epoch:
                epoch += 1
                batch_in_epoch = 0
                ep_idx = epoch_indices(weights, epoch_size, cfg.seed, epoch)
            if pool is not None:
                fut = pool.submit(next_batch, epoch, batch_in_epoch, ep_idx)
            lr = _lr_at(step, total, cfg)
            for o in opts:
                for g in o.param_groups:
                    g["lr"] = lr
            idx, w, off, xm, yt = _to_torch(sp, x, y)
            for o in opts:
                o.zero_grad(set_to_none=True)
            if cfg.fpm_bf16:
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    logits = net(idx, w, off, xm)
            else:
                logits = net(idx, w, off, xm)
            loss, bce, cos = _loss(logits, yt, pos_weight, cfg.fpm_cosine_loss_weight)
            loss.backward()
            for o in opts:
                o.step()
            step += 1
            recent = (recent + [float(loss.detach())])[-20:]
            acc["loss"] += recent[-1]
            acc["bce"] += float(bce.detach())
            acc["cos"] += float(cos.detach())
            acc["n"] += 1
            n_log += len(sp)
            # fix the total step count from measured throughput (time budget mode)
            if total is None:
                if step - measured_from == 10:
                    t_measure = time.perf_counter()
                elif t_measure is not None and step - measured_from == 10 + measure_steps:
                    rate = measure_steps / (time.perf_counter() - t_measure)
                    remaining = deadline - time.perf_counter()
                    total = step + max(1, int(rate * remaining * 0.97))
                    log.info("throughput %.2f steps/s (%.0f spectra/s): total steps set to %d "
                             "(%.2f epochs); rerun with steps=%d to reproduce", rate, rate * bs,
                             total, total / batches_per_epoch, total)
            if step % log_every == 0 or (total is not None and step >= total):
                dt = time.perf_counter() - t_log
                rec = {"step": step, "epoch": round(step / batches_per_epoch, 4),
                       "elapsed_s": round(elapsed_before + time.perf_counter() - t0, 1),
                       "lr": lr, "loss": acc["loss"] / acc["n"], "bce": acc["bce"] / acc["n"],
                       "train_cos": acc["cos"] / acc["n"],
                       "spectra_per_s": n_log / max(dt, 1e-9), "val_loss": "", "val_cos": ""}
                if time.perf_counter() - last_ckpt >= cfg.fpm_checkpoint_minutes * 60:
                    if val is not None:
                        val_res = evaluate(net, val, spec, cfg, pos_weight)
                        rec["val_loss"], rec["val_cos"] = val_res["loss"], val_res["cos"]
                    save_ckpt()
                    last_ckpt = time.perf_counter()
                curve.append(rec)
                log.info("step %d ep %.3f lr %.2e loss %.4f bce %.4f cos %.4f | %.0f spectra/s%s",
                         step, rec["epoch"], lr, rec["loss"], rec["bce"], rec["train_cos"],
                         rec["spectra_per_s"],
                         f" | val loss {rec['val_loss']:.4f} cos {rec['val_cos']:.4f}"
                         if rec["val_loss"] != "" else "")
                if curve_f is not None:
                    curve_f.write(",".join(str(rec[k]) for k in (
                        "step", "epoch", "elapsed_s", "lr", "loss", "bce", "train_cos",
                        "spectra_per_s", "val_loss", "val_cos")) + "\n")
                    curve_f.flush()
                acc = {"loss": 0.0, "bce": 0.0, "cos": 0.0, "n": 0}
                t_log = time.perf_counter()
                n_log = 0
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    seconds = time.perf_counter() - t0
    log.info("training stopped (%s) at step %d / %s after %.1f s", stop_reason, step, total,
             seconds)
    if val is not None:
        val_res = evaluate(net, val, spec, cfg, pos_weight)
        log.info("final val: loss %.4f bce %.4f cos %.4f (%d spectra)", val_res["loss"],
                 val_res["bce"], val_res["cos"], val_res["n"])
    final_loss = float(np.mean(recent)) if recent else float("nan")
    save_ckpt()
    if curve_f is not None:
        curve_f.close()
    npz = None
    if out_npz is not None:
        info = {"steps": step, "total_steps": total, "epochs": step / batches_per_epoch,
                "n_train_spectra": len(data), "seconds": round(seconds, 1),
                "final_train_loss": final_loss, "val": val_res, "seed": cfg.seed,
                "cfg_overrides": cfg.overrides(), "stop": stop_reason}
        info.update(dict(extra_info or {}))
        npz = export_npz(net, spec, out_npz, info, float16_export)
        log.info("exported %s (%.1f MB)", npz, npz.stat().st_size / 1e6)
    res = TrainResult(step, int(total or step), step / batches_per_epoch, seconds,
                      step * bs / max(seconds, 1e-9), final_loss, val_res, npz, curve)
    return (res, net) if return_net else res


def benchmark_throughput(cfg: Config | None = None, seconds: float = 60.0,
                         n_spectra: int = 20000, threads: int | None = None,
                         spec_overrides: Mapping[str, Any] | None = None,
                         augment: bool = True, seed: int = 0, peaks: int = 100) -> dict:
    """Training spectra per second of the configured model on synthetic batches of about
    ``peaks`` cleaned peaks per spectrum (100: the plan's shape; the public test has a
    mean of about 32 after cleaning).

    Runs warm-up steps, then times full training steps (augmentation, featurisation,
    forward, backward, optimiser) for ``seconds``. Data preparation runs on the main
    thread unless ``cfg.fpm_prefetch``; its share is reported separately.
    """
    import torch

    cfg = cfg or Config()
    spec = ModelSpec.from_config(cfg, **dict(spec_overrides or {}))
    data = synthetic_training_data(n_structures=max(1, n_spectra // 4), spectra_per_structure=4,
                                   n_peaks=(max(2, round(1.1 * peaks)), max(3, round(1.6 * peaks))),
                                   seed=seed, cfg=cfg, noise_peaks=round(0.4 * peaks))
    torch.set_num_threads(int(threads or cfg.fpm_threads))
    torch.manual_seed(cfg.seed)
    net = _torch_module(spec, cfg.fpm_sparse_embedding)
    opts = _make_optimizers(net, cfg, cfg.fpm_sparse_embedding)
    w = sampling_weights(data, cfg)
    pos_weight = torch.from_numpy(pos_weights(data, w, cfg))
    bs = cfg.fpm_batch_size
    order = epoch_indices(w, 10 ** 6, seed, 0)
    t_prep = t_step = 0.0
    n_done = 0
    k = 0
    t_end = None
    warm = 5
    entries = []
    while True:
        r = order[(k * bs) % len(order):(k * bs) % len(order) + bs]
        a = time.perf_counter()
        sp, x, y = make_batch(data, r, spec, cfg, np.random.default_rng(k), augment)
        b = time.perf_counter()
        idx, wt, off, xm, yt = _to_torch(sp, x, y)
        for o in opts:
            o.zero_grad(set_to_none=True)
        if cfg.fpm_bf16:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                logits = net(idx, wt, off, xm)
        else:
            logits = net(idx, wt, off, xm)
        loss, _, _ = _loss(logits, yt, pos_weight, cfg.fpm_cosine_loss_weight)
        loss.backward()
        for o in opts:
            o.step()
        c = time.perf_counter()
        k += 1
        if k == warm:
            t_end = c + seconds
        elif k > warm:
            t_prep += b - a
            t_step += c - b
            n_done += len(r)
            entries.append(np.diff(sp.offsets).mean())
            if c >= t_end:
                break
    tot = t_prep + t_step
    sps = n_done / tot
    sync = n_done / t_step
    return {"spectra_per_s": sps, "spectra_per_s_model_only": sync,
            "prep_share": t_prep / tot, "steps": k - warm, "batch": bs,
            "entries_per_spectrum": float(np.mean(entries)),
            "peaks_per_spectrum": float(np.mean(data.store.n_peaks)), "hidden": spec.hidden,
            "n_blocks": spec.n_blocks, "n_in": spec.n_in, "n_out": spec.n_out,
            "bin_da": spec.bin_da, "bf16": cfg.fpm_bf16, "sparse_embedding":
            cfg.fpm_sparse_embedding, "threads": int(threads or cfg.fpm_threads),
            "params_m": sum(p.numel() for p in net.parameters()) / 1e6,
            "epoch_minutes_2_3m": 2.3e6 / sps / 60.0}

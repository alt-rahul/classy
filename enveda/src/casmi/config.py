"""Run configuration: every tunable of the pipeline with its default (PLAN.md section 7).

Rules:
- Modules read tunables from the ``Config`` passed to them and never keep their own
  copy of a default listed here. Section numbers in comments refer to docs/PLAN.md.
- A YAML file in ``configs/`` overrides fields by name (flat keys). Unknown keys and
  wrong types are errors, so a typo cannot silently fall back to a default.
- ``Config`` is frozen: derive variants with ``cfg.replace(field=value)``.
- To add a tunable, append a field to the section that owns it (with the plan section
  in a comment). Do not rename or remove fields other modules use.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

__all__ = ["Config", "load_config", "save_config"]


@dataclass(frozen=True)
class Config:
    # --- General, determinism (7) ---------------------------------------------------
    seed: int = 20261118
    n_jobs: int = 4                          # worker processes / threads for heavy steps
    batch_molecules: int = 50                # molecules per inference batch (risk 14)

    # --- Neutral mass and tolerances (3.2, 3.3) --------------------------------------
    mass_tol_ppm: float = 10.0               # window: |m - M| <= max(ppm * M * 1e-6, min_da)
    mass_tol_min_da: float = 0.002
    mass_tol_wide_ppm: float = 20.0          # retry / padding window (4.8, risk 11)
    mass_cluster_ppm: float = 15.0           # linkage for clustering per-spectrum M (3.2)
    frag_tol_same_da: float = 0.01           # fragment matching, timsTOF vs timsTOF
    frag_tol_cross_da: float = 0.02          # fragment matching across instruments

    # --- Candidate databases (2.2, 2.3) -----------------------------------------------
    db_mass_min: float = 100.0
    db_mass_max: float = 1300.0
    db_elements: tuple = ("C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I")
    db_min_carbons: int = 3
    db_tiers: tuple = ("np", "pubchem")      # tiers queried at lookup ("with/without PubChem")

    # --- Spectrum preprocessing (4.1) -------------------------------------------------
    clean_max_above_precursor_da: float = 2.0   # drop peaks with m/z > precursor + this
    keep_precursor: bool = False             # False: exclude the precursor window below
    precursor_window_low_da: float = 0.5     # excluded window [prec - low, prec + high]
    precursor_window_high_da: float = 2.0
    intensity_floor: float = 0.005           # relative to base peak (0.5 percent)
    top_n_peaks: int = 100
    intensity_transform: str = "sqrt"        # "sqrt" or "none"
    consensus_merge_da: float = 0.01         # merge tolerance for consensus spectra
    min_peaks: int = 5                       # min peaks after cleaning (training filter 4.4)

    # --- Library matching and similarity (4.2, WP5) -----------------------------------
    entropy_weighted: bool = True            # ms_entropy weighted entropy similarity (Li 2021)
    lib_min_peaks: int = 1                   # library keeps spectra with >= this many clean peaks
    lib_build_batch_rows: int = 20000        # rows per streamed parquet batch in Library.build

    # --- Analog search (4.3) ----------------------------------------------------------
    analog_bin_da: float = 0.05
    analog_coarse_top: int = 200             # exact modified cosine on this many coarse hits
    analog_top_k: int = 20
    analog_max_mz: float = 2000.0            # binned range (Da) of fragment and loss channels
    analog_unique_structures: bool = True    # open search keeps the best consensus per structure

    # --- Fingerprints (4.4) -----------------------------------------------------------
    fp_morgan_bits: int = 4096
    fp_morgan_radius: int = 2
    fp_use_maccs: bool = True                # append 167 MACCS keys
    fp_prob_min: float = 0.01                # clip for the Bernoulli log-likelihood
    fp_prob_max: float = 0.99

    # --- Fingerprint model (4.4) ------------------------------------------------------
    fpm_mz_bin_da: float = 0.1
    fpm_mz_max: float = 1300.0
    fpm_nl_max: float = 700.0
    fpm_hidden: int = 1024
    fpm_meta_hidden: int = 64
    fpm_n_blocks: int = 2
    fpm_dropout: float = 0.1
    fpm_pos_weight_max: float = 10.0         # per-bit positive weights clipped to [1, max]
    fpm_cosine_loss_weight: float = 0.5
    fpm_lr: float = 1e-3
    fpm_batch_size: int = 512
    fpm_threads: int = 3
    fpm_hours: float = 4.0                   # wall budget, 3 to 5 hours
    fpm_max_precursor_error_ppm: float = 20.0   # label-quality filter
    fpm_test_like_weight: float = 2.0        # test adduct and timsTOF spectra
    fpm_aug_peak_drop: float = 0.1
    fpm_aug_intensity_sigma: float = 0.3
    fpm_aug_noise_peaks_max: int = 10
    fpm_aug_noise_int_min: float = 0.005
    fpm_aug_noise_int_max: float = 0.02
    fpm_aug_consensus_prob: float = 0.3
    fpm_bpi_log_min: float = 3.0             # molecule-level weight: clip(log10 bpi) / 3
    fpm_bpi_log_max: float = 6.0

    # --- Pool and ranker (4.5) --------------------------------------------------------
    pool_cap: int = 30000
    rank_neg_per_group: int = 2000
    rank_keep_top_fp: int = 200
    rank_n_trees: int = 500                  # 300 to 800
    rank_lr: float = 0.05
    rank_num_leaves: int = 31
    rank_min_data_in_leaf: int = 50
    rank_feature_fraction: float = 0.8
    rank_truncation_level: int = 25
    rank_class_weights: tuple = (0.4, 0.4, 0.2)   # group weights for C1 / C2 / C3
    rank_use_count_priors: bool = True       # False: the no-count-priors variant (5.6)

    # --- Deterministic fallback fusion (4.6) ------------------------------------------
    fusion_lib_cos_min: float = 0.7
    fusion_lib_min_matched: int = 4
    fusion_prior_coef: float = 0.1           # x log1p(pubmed_count)
    fusion_prior_cap: float = 0.3
    fusion_gen_factor: float = 0.5

    # --- Formula enumeration (3.4) ----------------------------------------------------
    formula_hc_min: float = 0.2
    formula_hc_max: float = 3.1
    formula_nc_max: float = 1.3
    formula_oc_max: float = 1.2
    formula_p_max: int = 2
    formula_s_max: int = 3
    formula_halogen_max: int = 4

    # --- Class 3 generator (4.7) ------------------------------------------------------
    gen_enabled: bool = True
    gen_n_formulas: int = 3
    gen_formula_min_prob: float = 0.05
    gen_n_analog_seeds: int = 20
    gen_n_db_seeds: int = 50
    gen_max_edits: int = 2
    gen_max_products: int = 3000
    gen_skip_lib_cos: float = 0.9

    # --- Final list assembly (4.8, risk 13) -------------------------------------------
    top_k: int = 25
    mkey_timeout_s: float = 10.0             # not fixed by the plan; initial guess

    # --- Validation (5) ---------------------------------------------------------------
    val_n_folds: int = 5
    val_n_syn_tims: int = 3000
    val_n_np_other: int = 3000
    val_max_spectra_per_molecule: int = 16
    val_min_base_peak_intensity: float = 1000.0
    val_c3_max_mkey: int = 5000              # above this, mkey only for Tanimoto >= min
    val_c3_tanimoto_min: float = 0.5
    val_blend_mixes: tuple = ((0.5, 0.3, 0.2), (0.3, 0.5, 0.2), (0.2, 0.4, 0.4),
                              (0.34, 0.33, 0.33))
    val_bootstrap_ci: float = 0.9
    val_bootstrap_n: int = 1000              # not fixed by the plan

    def __post_init__(self) -> None:
        for f in dataclasses.fields(self):
            value = _coerce(f.name, f.default, getattr(self, f.name))
            object.__setattr__(self, f.name, value)

    def replace(self, **changes: Any) -> "Config":
        """Copy with some fields changed (validated like a YAML override)."""
        _check_keys(changes)
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Plain dict with lists instead of tuples (YAML/JSON friendly)."""
        return {f.name: _to_plain(getattr(self, f.name)) for f in dataclasses.fields(self)}

    def overrides(self) -> dict[str, Any]:
        """Fields that differ from the defaults, for logs and docs/EXPERIMENTS.md."""
        return {f.name: _to_plain(getattr(self, f.name))
                for f in dataclasses.fields(self) if getattr(self, f.name) != f.default}


_FIELD_NAMES = frozenset(f.name for f in dataclasses.fields(Config))


def _check_keys(keys: Mapping[str, Any]) -> None:
    unknown = sorted(set(keys) - _FIELD_NAMES)
    if unknown:
        raise KeyError(f"unknown config key(s): {unknown}")


def _to_tuple(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_to_tuple(v) for v in value)
    return value


def _to_plain(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


def _coerce(name: str, default: Any, value: Any) -> Any:
    """Check ``value`` against the type of the field default; int -> float, list -> tuple."""
    def bad() -> TypeError:
        return TypeError(f"config key {name!r}: expected {type(default).__name__}, "
                         f"got {type(value).__name__} ({value!r})")

    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise bad()
        return value
    if isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise bad()
        return value
    if isinstance(default, float):
        if isinstance(value, str):  # PyYAML reads "1e-3" (no dot) as a string
            try:
                return float(value)
            except ValueError:
                raise bad() from None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise bad()
        return float(value)
    if isinstance(default, str):
        if not isinstance(value, str):
            raise bad()
        return value
    if isinstance(default, tuple):
        if not isinstance(value, (list, tuple)):
            raise bad()
        return _to_tuple(value)
    return value


def load_config(path: str | Path | None = None) -> Config:
    """Defaults overridden by the YAML mapping in ``path`` (None or empty file: defaults)."""
    if path is None:
        return Config()
    import yaml  # lazy: keeps `import casmi` free of optional imports

    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    _check_keys(data)
    return Config(**data)


def save_config(cfg: Config, path: str | Path) -> Path:
    """Write the full config as YAML (e.g. models/config_final.yaml for Kaggle)."""
    import yaml

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
    return p

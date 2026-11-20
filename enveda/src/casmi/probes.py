"""Probe submissions for the class-mix estimate (ROADMAP 2.6; package R1).

A probe is a submission whose lists come from one candidate source only, so its public
score is a linear equation in the class shares: board(probe) = sum over cells of pi_c x
MRR_c(probe), with MRR_c measured locally by ``scripts/evaluate.py`` on the same config.
``scripts/class_mix.py`` solves the equations for the shares.

The stage ``probe_assemble`` (register by listing this module in ``pipeline_plugins`` and
the stage as the LAST entry of ``pipeline_stages``) keeps only the pool rows of
``cfg.probe_rule``, runs the registered ``assemble`` stage and then cuts every list to the
guesses that came from the filtered pool (``n_from_pool``): the padding windows would
otherwise bring in candidates of other sources. A molecule with nothing left gets
``pipeline.LAST_RESORT_SMILES`` (one wrong guess, scores 0). Rules:

- ``lib_hits``: LIB candidates with lib_best_cos >= ``probe_lib_cos_min`` and lib_n_matched
  >= ``probe_lib_min_matched`` (probe P1).
- ``np_only``: candidates in the NP tier (``in_np``, LIB rows included when the structure is
  in the NP tier; P3).
- ``pubchem_only``: PubChem-only candidates: not LIB, not GEN, not in the NP tier (P4).
- ``novel``: candidates in no tier and without library spectra (GEN rows: the class-3
  generator, R13; P5). Without a generator stage every list is the last-resort guess.
- ``""``: no filter (the stage then equals ``assemble``).

Configs: ``configs/probe_p*.yaml`` (designs and the expected local numbers in
docs/EXPERIMENTS.md, "Protocol v2"). Nothing here submits anything.

Inference-path rule (PLAN 7): numpy, pandas and casmi modules only.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import fusion, pipeline
from .config import Config

__all__ = ["PROBE_RULES", "probe_mask", "stage_probe_assemble"]

log = logging.getLogger(__name__)

PROBE_RULES = ("", "lib_hits", "np_only", "pubchem_only", "novel")


def _flag(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(df), bool)
    return df[col].astype("boolean").fillna(False).astype(bool).to_numpy()


def probe_mask(pool: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Rows of ``pool`` that the probe ``cfg.probe_rule`` keeps."""
    rule = str(cfg.probe_rule)
    if rule not in PROBE_RULES:
        raise ValueError(f"unknown probe_rule {rule!r}; known {PROBE_RULES}")
    n = len(pool)
    if rule == "" or n == 0:
        return np.ones(n, bool)
    src = fusion.candidate_source(pool)
    in_np = _flag(pool, "in_np")
    if rule == "lib_hits":
        cos = pd.to_numeric(pool.get("lib_best_cos", pd.Series(np.nan, index=pool.index)),
                            errors="coerce").to_numpy(np.float64)
        nm = pd.to_numeric(pool.get("lib_n_matched", pd.Series(np.nan, index=pool.index)),
                           errors="coerce").to_numpy(np.float64)
        with np.errstate(invalid="ignore"):
            return (src == "lib") & (cos >= cfg.probe_lib_cos_min) & (nm >= cfg.probe_lib_min_matched)
    if rule == "np_only":
        return in_np | (src == "np")
    if rule == "pubchem_only":
        return src == "pubchem"
    in_pc = _flag(pool, "in_pubchem")
    return (src == "gen") & ~in_np & ~in_pc


def stage_probe_assemble(state: pipeline.State, cfg: Config) -> pipeline.State:
    """Filter the pool by ``cfg.probe_rule``, assemble, keep pool-derived guesses only."""
    rule = str(cfg.probe_rule)
    if state.pool is not None and len(state.pool):
        keep = probe_mask(state.pool, cfg)
        state.pool = state.pool[keep].reset_index(drop=True)
    state = pipeline.get_stage("assemble")(state, cfg)
    if rule == "":
        return state
    mols = state.molecules
    nfp = dict(zip(mols["molecule_id"].astype(str), mols["n_from_pool"].astype(int)))
    n_empty = 0
    for m in list(state.preds):
        k = nfp.get(str(m), 0)
        g = list(state.preds[m])[:k]
        if not g:
            g = [pipeline.LAST_RESORT_SMILES]
            n_empty += 1
        state.preds[m] = g
    mols = mols.copy()
    mols["n_guesses"] = [len(state.preds[str(m)]) for m in mols["molecule_id"]]
    mols["probe_empty"] = [len(state.preds[str(m)]) == 1 and nfp.get(str(m), 0) == 0
                           for m in mols["molecule_id"]]
    state.molecules = mols
    log.info("probe %s: %d of %d molecules have no candidate of that source", rule, n_empty,
             len(state.preds))
    return state


pipeline.register_stage("probe_assemble", stage_probe_assemble)

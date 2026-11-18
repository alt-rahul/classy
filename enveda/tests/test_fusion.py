"""Tests for casmi.fusion (WP4): PLAN 4.6 rules, the Baseline 0 prior, ordering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from casmi import fusion
from casmi.config import Config

CFG = Config()


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_candidate_source_labels_and_missing_columns():
    df = _frame([
        {"tier": "np", "in_np": True},
        {"tier": "pubchem", "in_np": False},
        {"tier": "pubchem", "in_np": True},      # merged row: NP evidence wins
        {"tier": "lib", "in_np": True},          # LIB first
        {"tier": "gen", "in_np": False},
        {"tier": None, "in_np": None, "is_lib": True},
    ])
    assert list(fusion.candidate_source(df)) == ["np", "pubchem", "np", "lib", "gen", "lib"]
    # No columns at all: everything is "PubChem-only" (no evidence of anything else).
    assert list(fusion.candidate_source(pd.DataFrame(index=range(2)))) == ["pubchem", "pubchem"]


def test_prior_is_monotone_in_every_input():
    base = {"tier": "np", "in_np": True, "in_coconut": True, "in_npatlas": False,
            "in_lotus": False, "in_pubchemlite": False, "coconut_annotation_level": 2,
            "pubmed_count": 3, "pmid_count": 0, "n_citations": 1, "n_organisms": 2,
            "patent_count": 5, "anno_type_count": 1, "n_collections": 1}
    p0 = fusion.prior_score(_frame([base]), CFG)[0]
    bumps = {"in_npatlas": True, "in_lotus": True, "in_pubchemlite": True,
             "coconut_annotation_level": 3, "pubmed_count": 30, "pmid_count": 100,
             "n_citations": 5, "n_organisms": 9, "patent_count": 50, "anno_type_count": 4,
             "n_collections": 3}
    for col, val in bumps.items():
        p1 = fusion.prior_score(_frame([{**base, col: val}]), CFG)[0]
        assert p1 > p0, col
    # PubChem-only prior: pmid_count and n_cids.
    pc = {"tier": "pubchem", "in_np": False, "pmid_count": 1, "n_cids": 1}
    q0 = fusion.prior_score(_frame([pc]), CFG)[0]
    assert fusion.prior_score(_frame([{**pc, "pmid_count": 10}]), CFG)[0] > q0
    assert fusion.prior_score(_frame([{**pc, "n_cids": 10}]), CFG)[0] > q0


def test_prior_handles_missing_values_and_columns():
    df = pd.DataFrame({
        "tier": ["np", "np"], "in_np": [True, True], "in_coconut": [True, True],
        "pubmed_count": pd.array([pd.NA, 10], dtype="Int32"),
        "coconut_annotation_level": pd.array([pd.NA, pd.NA], dtype="Int8"),
    })
    p = fusion.prior_score(df, CFG)
    assert np.isfinite(p).all()
    assert p[0] == pytest.approx(CFG.fusion_w_np_sources)
    assert p[1] == pytest.approx(CFG.fusion_w_np_sources + CFG.fusion_w_literature * np.log1p(10))
    with pytest.raises(ValueError):
        fusion.prior_score(df, CFG.replace(fusion_w_literature=-1.0))


def test_rank_normalise():
    z = fusion.rank_normalise(np.array([1.0, 3.0, 2.0, 5.0, np.nan, 7.0, 7.0]),
                              np.array([0, 0, 0, 1, 1, 2, 2]))
    assert z[:3].tolist() == [0.0, 1.0, 0.5]
    assert z[3] == 1.0 and np.isnan(z[4])      # single valued group -> 1
    assert z[5] == z[6] == 0.5                 # ties -> average rank
    assert fusion.rank_normalise(np.zeros(0), np.zeros(0)).shape == (0,)


def test_np_tier_always_above_pubchem_only():
    rows = [{"molecule_id": "m1", "tier": "np", "in_np": True, "in_lotus": True,
             "ik14": f"NP{i:012d}", "ppm_err": 1.0} for i in range(3)]
    rows += [{"molecule_id": "m1", "tier": "pubchem", "in_np": False, "pmid_count": 10 ** (i + 3),
              "n_cids": 50, "ik14": f"PC{i:012d}", "ppm_err": 0.1} for i in range(3)]
    df = _frame(rows)
    s = fusion.fuse(df, CFG)
    assert s[:3].min() >= 1.0 >= s[3:].max()
    out = fusion.sort_pool(df.assign(score=s))
    assert list(out["source"]) == ["np"] * 3 + ["pubchem"] * 3
    # Among PubChem-only candidates the larger pmid_count wins.
    assert list(out["ik14"][3:]) == ["PC000000000002", "PC000000000001", "PC000000000000"]


def test_rules_lib_gen_and_bonus():
    df = _frame([
        {"tier": "lib", "lib_best_cos": 0.8, "lib_n_matched": 5},     # rule 1
        {"tier": "lib", "lib_best_cos": 0.8, "lib_n_matched": 3},     # too few peaks: rule 2
        {"tier": "lib"},                                              # no spectral data: rule 2
        {"tier": "np", "in_np": True, "pubmed_count": 10 ** 6},       # bonus capped
        {"tier": "gen"},                                              # rule 4
    ])
    s, z = fusion.fuse(df, CFG, return_z=True)
    assert s[0] == pytest.approx(3.8)
    assert s[1] == pytest.approx(1.0 + z[1])
    assert s[2] == pytest.approx(1.0 + z[2])
    assert s[3] == pytest.approx(1.0 + z[3] + CFG.fusion_prior_cap)
    assert s[4] == pytest.approx(CFG.fusion_gen_factor * z[4])
    assert z[4] == 1.0                                                # alone in its band
    assert fusion.fuse(df.iloc[:0], CFG).shape == (0,)


def test_fingerprint_branch_per_pool():
    df = _frame([
        {"molecule_id": "a", "tier": "np", "in_np": True, "fp_ll": -10.0, "pubmed_count": 1000},
        {"molecule_id": "a", "tier": "np", "in_np": True, "fp_ll": -1.0},
        {"molecule_id": "a", "tier": "np", "in_np": True, "fp_ll": np.nan},
        {"molecule_id": "b", "tier": "np", "in_np": True, "pubmed_count": 1000},
        {"molecule_id": "b", "tier": "np", "in_np": True},
    ])
    s, z = fusion.fuse(df, CFG, return_z=True)
    assert z[:3].tolist() == [0.0, 1.0, 0.0]     # fp ranks in pool a; no fp -> 0
    assert z[3:].tolist() == [1.0, 0.0]          # pool b has no fp: prior ranks
    assert s[1] > s[0]                           # fingerprint beats the popularity bonus


def test_sort_pool_tie_breaks():
    df = _frame([
        {"molecule_id": "m2", "tier": "np", "in_np": True, "ik14": "A", "ppm_err": 0.0},
        {"molecule_id": "m1", "tier": "np", "in_np": True, "ik14": "C", "ppm_err": -1.0},
        {"molecule_id": "m1", "tier": "np", "in_np": True, "ik14": "B", "ppm_err": 1.0},
        {"molecule_id": "m1", "tier": "np", "in_np": True, "ik14": "D", "ppm_err": 3.0},
        {"molecule_id": "m1", "tier": "pubchem", "in_np": False, "ik14": "A", "ppm_err": 0.0},
        {"molecule_id": "m1", "tier": "np", "in_np": True, "ik14": "E", "ppm_err": 0.0},
    ])
    df["score"] = [1.0, 1.0, 1.0, 1.0, 1.0, np.nan]
    out = fusion.sort_pool(df)
    assert list(zip(out["molecule_id"], out["ik14"])) == [
        ("m1", "B"), ("m1", "C"),      # equal |ppm|: ik14
        ("m1", "D"),                   # larger |ppm|
        ("m1", "A"),                   # PubChem after NP at equal score
        ("m1", "E"),                   # missing score last
        ("m2", "A"),
    ]


@pytest.mark.data("data/processed/db/np/np.parquet")
def test_quercetin_tops_its_real_np_pool():
    from casmi import db

    pool = db.lookup(302.0427, tiers=("np",))
    pool = pool.assign(molecule_id="q")
    pool["score"] = fusion.fuse(pool, CFG)
    out = fusion.sort_pool(pool)
    assert out["ik14"].iloc[0] == "REFJWTPEDVJJIY"

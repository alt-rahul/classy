"""WP12: class 3 analog generator (PLAN.md 4.7): edit rules, round trips, determinism."""

from __future__ import annotations

import time

import numpy as np
import pytest
from rdkit import Chem

from casmi import generate as G
from casmi.chem import formula as rd_formula
from casmi.chem import mono_mass
from casmi.config import Config
from casmi.formula import ELEMENTS, formula_counts, hill
from casmi.metric import ik14

QUE = "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12"          # quercetin
KAE = "O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12"             # kaempferol
API = "O=c1cc(-c2ccc(O)cc2)oc2cc(O)cc(O)c12"                # apigenin
ISORHAMNETIN = "COc1cc(-c2oc3cc(O)cc(O)c3c(=O)c2O)ccc1O"
ISOQUERCITRIN = "OCC1OC(Oc2c(-c3ccc(O)c(O)c3)oc3cc(O)cc(O)c3c2=O)C(O)C(O)C1O"
QUERCITRIN = "CC1OC(Oc2c(-c3ccc(O)c(O)c3)oc3cc(O)cc(O)c3c2=O)C(O)C(O)C1O"

# One positive example per edit rule: (reactant, one expected product).
EXAMPLES: dict[str, tuple[str, str]] = {
    "O_methylation": ("Oc1ccccc1", "COc1ccccc1"),
    "O_demethylation": ("COc1ccccc1", "Oc1ccccc1"),
    "N_methylation": ("NCc1ccccc1", "CNCc1ccccc1"),
    "N_demethylation": ("Cn1ccc2ccccc21", "c1ccc2[nH]ccc2c1"),
    "C_methylation": ("Oc1ccccc1", "Cc1ccccc1O"),
    "C_demethylation": ("Cc1ccccc1O", "Oc1ccccc1"),
    "hydroxylation_arom": (KAE, QUE),
    "hydroxylation_aliph": ("CCCC(=O)O", "CC(O)CC(=O)O"),
    "dehydroxylation": ("OCCc1ccccc1", "CCc1ccccc1"),
    "N_oxidation": ("CN1CCCC1c1cccnc1", "C[N+]1([O-])CCCC1c1cccnc1"),
    "N_oxidation_arom": ("c1ccncc1", "[O-][n+]1ccccc1"),
    "N_deoxygenation": ("C[N+]1([O-])CCCC1c1cccnc1", "CN1CCCC1c1cccnc1"),
    "epoxidation": ("CC=CC", "CC1OC1C"),
    "deepoxidation": ("CC1OC1C", "CC=CC"),
    "hydrogenation_CC": ("OC(=O)C=Cc1ccccc1", "OC(=O)CCc1ccccc1"),
    "hydrogenation_CO": ("CC(=O)c1ccccc1", "CC(O)c1ccccc1"),
    "dehydrogenation_CC": ("OC(=O)CCc1ccccc1", "OC(=O)C=Cc1ccccc1"),
    "dehydrogenation_CO": ("CC(O)c1ccccc1", "CC(=O)c1ccccc1"),
    "carbonyl_insertion": ("CCc1ccccc1", "CCC(=O)c1ccccc1"),
    "formylation_arom": ("Oc1ccccc1", "O=Cc1ccccc1O"),
    "N_formylation": ("NCc1ccccc1", "O=CNCc1ccccc1"),
    "carbonyl_removal": ("CCC(=O)c1ccccc1", "CCc1ccccc1"),
    "deformylation": ("O=Cc1ccccc1O", "Oc1ccccc1"),
    "O_acetylation": ("OCc1ccccc1", "CC(=O)OCc1ccccc1"),
    "N_acetylation": ("NCc1ccccc1", "CC(=O)NCc1ccccc1"),
    "deacetylation": ("CC(=O)Oc1ccccc1C(=O)O", "O=C(O)c1ccccc1O"),
    "C_prenylation": ("Oc1ccccc1", "CC(C)=CCc1ccccc1O"),
    "O_prenylation": ("Oc1ccccc1", "CC(C)=CCOc1ccccc1"),
    "deprenylation": ("CC(C)=CCc1ccccc1O", "Oc1ccccc1"),
    "O_hexosylation": (QUE, ISOQUERCITRIN),
    "C_hexosylation": (API, "OCC1OC(c2c(O)cc(O)c3c(=O)cc(-c4ccc(O)cc4)oc23)C(O)C(O)C1O"),
    "dehexosylation": ("OCc1ccccc1OC1OC(CO)C(O)C(O)C1O", "OCc1ccccc1O"),
    "O_deoxyhexosylation": ("Oc1ccccc1", "CC1OC(Oc2ccccc2)C(O)C(O)C1O"),
    "dedeoxyhexosylation": (QUERCITRIN, QUE),
    "O_pentosylation": ("Oc1ccccc1", "OC1COC(Oc2ccccc2)C(O)C1O"),
    "O_pentofuranosylation": ("Oc1ccccc1", "OCC1OC(Oc2ccccc2)C(O)C1O"),
    "depentosylation": ("OC1COC(Oc2ccccc2)C(O)C1O", "Oc1ccccc1"),
    "depentofuranosylation": ("Nc1ncnc2c1ncn2C1OC(CO)C(O)C1O", "Nc1ncnc2[nH]cnc12"),
    "O_glucuronidation": ("Oc1ccccc1", "O=C(O)C1OC(Oc2ccccc2)C(O)C(O)C1O"),
    "deglucuronidation": ("O=C(O)C1OC(Oc2ccccc2)C(O)C(O)C1O", "Oc1ccccc1"),
    "sulfation": ("Oc1ccccc1", "O=S(=O)(O)Oc1ccccc1"),
    "desulfation": ("O=S(=O)(O)Oc1ccccc1", "Oc1ccccc1"),
    "dehydration": ("CC(O)Cc1ccccc1", "CC=Cc1ccccc1"),
    "lactonization5": ("OCCCC(=O)O", "O=C1CCCO1"),
    "lactonization6": ("OCCCCC(=O)O", "O=C1CCCCO1"),
    "ether_closure5": ("OCCCCO", "C1CCOC1"),
    "ether_closure6": ("OCCCCCO", "C1CCOCC1"),
    "hydration": ("CC=CC", "CCC(C)O"),
    "lactone_opening": ("O=C1CCCO1", "O=C(O)CCCO"),
    "ether_opening": ("C1CCOC1", "OCCCCO"),
    "carboxylation": ("Oc1ccccc1", "O=C(O)c1ccccc1O"),
    "decarboxylation": ("O=C(O)c1ccccc1O", "Oc1ccccc1"),
    "OH_to_NH2": ("OCc1ccccc1", "NCc1ccccc1"),
    "NH2_to_OH": ("NCc1ccccc1", "OCc1ccccc1"),
    "ester_to_amide": ("CCOC(C)=O", "CCNC(C)=O"),
    "amide_to_ester": ("CCNC(C)=O", "CCOC(C)=O"),
    "chain_extension_C2H4": ("CCCC(=O)O", "CCCCCC(=O)O"),
    "chain_shortening_C2H4": ("CCCCCC(=O)O", "CCCC(=O)O"),
    "malonylation": ("OCc1ccccc1", "O=C(O)CC(=O)OCc1ccccc1"),
    "demalonylation": ("O=C(O)CC(=O)OCc1ccccc1", "OCc1ccccc1"),
    "benzoylation": ("OCc1ccccc1", "O=C(OCc1ccccc1)c1ccccc1"),
    "debenzoylation": ("O=C(OCc1ccccc1)c1ccccc1", "OCc1ccccc1"),
    "coumaroylation": ("OCc1ccccc1", "O=C(C=Cc1ccc(O)cc1)OCc1ccccc1"),
    "decoumaroylation": ("O=C(C=Cc1ccc(O)cc1)OCc1ccccc1", "OCc1ccccc1"),
    "caffeoylation": ("OCc1ccccc1", "O=C(C=Cc1ccc(O)c(O)c1)OCc1ccccc1"),
    "decaffeoylation": ("O=C(C=Cc1ccc(O)c(O)c1)OCc1ccccc1", "OCc1ccccc1"),
    "move_OH_arom": ("Cc1ccccc1O", "Cc1ccc(O)cc1"),
    "move_OH_aliph": ("CC(O)CC", "CCCCO"),
    "move_OMe_arom": ("COc1ccccc1C", "COc1ccc(C)cc1"),
    "swap_OMe_OH": ("COc1cccc(CO)c1", "COCc1cccc(O)c1"),
    "move_methyl_arom": ("Cc1ccccc1O", "Cc1ccc(O)cc1"),
    "move_prenyl_arom": ("CC(C)=CCc1ccccc1O", "CC(C)=CCc1ccc(O)cc1"),
    "move_hexosyl": ("OCc1ccccc1OC1OC(CO)C(O)C(O)C1O", "OCC1OC(OCc2ccccc2O)C(O)C(O)C1O"),
    "shift_double_bond": ("C=CCCC", "CC=CCC"),
}

# PLAN.md 4.7 delta units per family (malonyl corrected to C3H2O3, see generate.py).
FAMILY_DELTA = {
    "CH2": "CH2", "O": "O", "H2": "H2", "CO": "CO", "C2H2O": "C2H2O", "C5H8": "C5H8",
    "C6H10O5": "C6H10O5", "C6H10O4": "C6H10O4", "C5H8O4": "C5H8O4", "C6H8O6": "C6H8O6",
    "SO3": "SO3", "H2O": "H2O", "CO2": "CO2", "C2H4": "C2H4", "C3H2O3": "C3H2O3",
    "C7H4O": "C7H4O", "C9H6O2": "C9H6O2", "C9H6O3": "C9H6O3",
}

# 20 natural-product seeds for the timing test.
SEEDS20 = [
    QUE, KAE, API, ISORHAMNETIN,
    "O=C1CC(c2ccc(O)cc2)Oc2cc(O)cc(O)c21",                                   # naringenin
    "Oc1ccc(C2Oc3cc(O)cc(O)c3CC2O)cc1O",                                     # catechin
    "O=c1c(-c2ccc(O)cc2)coc2cc(O)cc(O)c12",                                  # genistein
    "OCC1OC(c2c(O)cc(O)c3c(=O)cc(-c4ccc(O)cc4)oc23)C(O)C(O)C1O",             # vitexin
    "CC1OC(OCC2OC(Oc3c(-c4ccc(O)c(O)c4)oc4cc(O)cc(O)c4c3=O)C(O)C(O)C2O)C(O)C(O)C1O",  # rutin
    "OCc1ccccc1OC1OC(CO)C(O)C(O)C1O",                                        # salicin
    "COC(=O)C1=COC(OC2OC(CO)C(O)C(O)C2O)C(=CC)C1CC(=O)OCCc1ccc(O)c(O)c1",    # oleuropein
    "CN1CCC23c4c5ccc(O)c4OC2C(O)C=CC3C1C5",                                  # morphine
    "COc1ccc2nccc(C(O)C3CC4CCN3CC4C=C)c2c1",                                 # quinine
    "COc1cc2c(c(OC)c1OC)-c1ccc(OC)c(=O)cc1C(NC(C)=O)CC2",                    # colchicine
    "COc1cc(CNC(=O)CCCCC=CC(C)C)ccc1O",                                      # capsaicin
    "CCC1(O)C(=O)OCc2c1cc1-c3nc4ccccc4cc3Cn1c2=O",                           # camptothecin
    "CC1CCC2C(C)C(=O)OC3OC4(C)CCC1C32OO4",                                   # artemisinin
    "C=C1CCC2C(C)(CO)C(O)CCC2(C)C1CC=C1C(O)COC1=O",                          # andrographolide
    "CC(=C)C1CCC2(C(=O)O)CCC3(C)C(CCC4C5(C)CCC(O)C(C)(C)C5CCC43C)C12",       # betulinic acid
    "C=CC1(C)CC(=O)C2(O)C(C)(O1)C(OC(C)=O)C(O)C1C(C)(C)CCC(O)C12C",          # forskolin
]


def _can(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False)


def _delta(product: Chem.Mol, reactant: Chem.Mol) -> tuple[int, ...]:
    return tuple(a - b for a, b in zip(G.mol_counts(product), G.mol_counts(reactant)))


# ------------------------------------------------------------------ the edit library

def test_every_rule_has_an_example_and_unique_id():
    assert set(EXAMPLES) == set(G.RULE_IDS)
    assert len(G.RULES) == len(G.RULE_IDS)
    table = G.rules_table()
    assert list(table["code"]) == list(range(len(G.RULES)))


def test_rule_deltas_follow_the_plan():
    for r in G.RULES:
        if r.family == "0":
            assert not any(r.delta), r.id
        elif r.family == "NH-O":
            nh_o = tuple(a - b for a, b in zip(formula_counts("NH"), formula_counts("O")))
            assert r.delta in (nh_o, tuple(-v for v in nh_o)), r.id
        else:
            unit = tuple(int(v) for v in formula_counts(FAMILY_DELTA[r.family]))
            assert r.delta in (unit, tuple(-v for v in unit)), r.id


@pytest.mark.parametrize("rule_id", list(EXAMPLES))
def test_edit_rule_positive_example(rule_id):
    rule = G.get_rule(rule_id)
    reactant_smiles, expected = EXAMPLES[rule_id]
    seed = G.prepare_seed(reactant_smiles)
    products = G.apply_rule(rule, seed)
    smiles = [s for s, _ in products]
    assert _can(expected) in smiles, (rule_id, smiles[:8])
    # the asserted formula delta: expected product minus reactant equals the rule delta,
    # and so does every product of the rule on this example
    assert _delta(Chem.MolFromSmiles(expected), seed) == rule.delta
    for _, m in products:
        assert _delta(m, seed) == rule.delta, rule_id
        assert Chem.GetFormalCharge(m) == 0
    assert _can(reactant_smiles) not in smiles


def test_delta_formula_text():
    assert G.get_rule("O_hexosylation").delta_formula == "+C6H10O5"
    assert G.get_rule("O_demethylation").delta_formula == "-CH2"
    assert G.get_rule("OH_to_NH2").delta_formula == "+HN-O"
    assert G.get_rule("move_OH_arom").delta_formula == "0"


def test_apply_rule_accepts_smiles_and_rejects_nothing_valid():
    assert G.apply_rule("O_methylation", "not a smiles") == []
    out = G.apply_rule("O_methylation", "Oc1ccccc1")
    assert [s for s, _ in out] == [_can("COc1ccccc1")]


# ------------------------------------------------------------------ generation

def test_roundtrip_quercetin_methylation_gives_isorhamnetin():
    target = rd_formula(ISORHAMNETIN)
    out = G.generate([QUE], [target])
    assert set(out["formula"]) == {target}
    hit = out[out["smiles"] == _can(ISORHAMNETIN)]
    assert len(hit) == 1
    assert hit["edits"].iloc[0] == "O_methylation" and hit["n_edits"].iloc[0] == 1
    assert hit["ik14"].iloc[0] == ik14(ISORHAMNETIN)


def test_roundtrip_kaempferol_hydroxylation_gives_quercetin():
    out = G.generate([KAE], [rd_formula(QUE)])
    hit = out[out["ik14"] == ik14(QUE)]
    assert len(hit) == 1 and hit["edits"].iloc[0] == "hydroxylation_arom"


def test_roundtrip_quercetin_hexose_gives_glucosides():
    out = G.generate([QUE], [rd_formula(ISOQUERCITRIN)])
    assert ik14(ISOQUERCITRIN) in set(out["ik14"])
    o_glyc = out[out["edits"] == "O_hexosylation"]
    assert len(o_glyc) == 5                       # one per quercetin OH
    assert (out["n_edits"] <= 2).all()


def test_products_are_valid_and_provenance_is_consistent():
    targets = ["C16H12O7", "C21H20O11", "C15H10O7"]
    seeds = [QUE, KAE, API, ISORHAMNETIN]
    out = G.generate(seeds, targets)
    assert list(out.columns) == list(G.GEN_COLUMNS)
    assert out["ik14"].is_unique and out["smiles"].is_unique
    for smi, f, t in zip(out["smiles"], out["formula"], out["target_idx"]):
        m = Chem.MolFromSmiles(smi)
        assert m is not None and "." not in smi and Chem.GetFormalCharge(m) == 0
        assert rd_formula(m) == f == targets[t]
    seed_ik = {ik14(s) for s in seeds}
    assert not seed_ik & set(out["ik14"])          # seeds are never products
    for row in out.itertuples():
        parts = row.edits.split("+")
        assert len(parts) == row.n_edits
        assert G.RULES[row.edit1].id == parts[0]
        assert (row.edit2 == -1) == (row.n_edits == 1)
        # summed deltas reach the target from the seed
        seed_c = G.mol_counts(G.prepare_seed(seeds[row.seed_idx]))
        total = [sum(G.get_rule(p).delta[k] for p in parts) for k in range(len(ELEMENTS))]
        assert tuple(a + b for a, b in zip(seed_c, total)) == tuple(formula_counts(row.formula))
    assert (out["n_seeds"] >= 1).all() and (out["n_paths"] >= out["n_seeds"]).all()
    # priority order: seed rank, then number of edits
    key = list(zip(out["seed_rank"], out["n_edits"]))
    assert key == sorted(key)


def test_exclusion_cap_and_scores():
    target = rd_formula(ISORHAMNETIN)
    full = G.generate([QUE], [target])
    excl = G.generate([QUE], [target], exclude_ik14={ik14(ISORHAMNETIN)})
    assert ik14(ISORHAMNETIN) not in set(excl["ik14"]) and len(excl) == len(full) - 1
    capped = G.generate([QUE], [target], max_products=5)
    assert list(capped["ik14"]) == list(full["ik14"][:5])
    # seed scores decide priority: the higher-scored seed comes first
    two = G.generate([KAE, QUE], ["C16H12O7"], seed_scores=[0.1, 0.9])
    assert two["seed_idx"].iloc[0] == 1 and two["seed_rank"].iloc[0] == 0
    small = G.generate([KAE, QUE], ["C16H12O7"], seed_scores=[0.1, 0.9], max_products=10)
    assert (small["seed_idx"] == 1).all()
    assert G.generate([], ["C15H10O7"]).empty
    assert G.generate([QUE], ["not a formula"]).empty
    assert G.generate(["[Na+].[Cl-]", "C[N+](C)(C)C"], ["C15H10O7"]).empty
    with pytest.raises(ValueError):
        G.generate([QUE, KAE], ["C15H10O7"], seed_scores=[1.0])


def test_same_formula_target_gives_positional_isomers():
    out = G.generate([ISORHAMNETIN], [rd_formula(ISORHAMNETIN)])
    assert len(out) > 0 and (out["n_edits"] == 1).all()   # inverse pairs are skipped
    assert set(out["edits"]) <= {r.id for r in G.RULES if not any(r.delta)}
    tamarixetin = "COc1ccc(-c2oc3cc(O)cc(O)c3c(=O)c2O)cc1O"
    assert ik14(tamarixetin) in set(out["ik14"])          # swap_OMe_OH


def test_generate_is_deterministic():
    targets = ["C16H12O7", "C21H20O11", "C15H10O7"]
    scores = np.linspace(1.0, 0.0, 8)
    a = G.generate(SEEDS20[:8], targets, seed_scores=scores)
    b = G.generate(SEEDS20[:8], targets, seed_scores=scores)
    assert a.equals(b)
    # the same seeds in another input order with the same scores give the same products
    perm = [3, 7, 0, 5, 1, 6, 2, 4]
    c = G.generate([SEEDS20[i] for i in perm], targets, seed_scores=scores[perm])
    assert list(c["ik14"]) == list(a["ik14"])
    assert [perm[i] for i in c["seed_idx"]] == list(a["seed_idx"])


def test_generation_timing_20_seeds_3_targets():
    """Acceptance target: under 3 s per molecule (20 seeds, 3 targets) on one core. The
    assertion allows 2x for a loaded shared machine; the measured time is printed."""
    G.generate([QUE], ["C16H12O7"])                       # warm-up: compile reactions
    targets = ["C16H12O7", "C21H20O11", "C15H10O7"]
    t = time.perf_counter()
    out = G.generate(SEEDS20, targets, seed_scores=np.linspace(1.0, 0.0, 20))
    dt = time.perf_counter() - t
    print(f"generate: 20 seeds x 3 targets -> {len(out)} products in {dt:.2f} s")
    assert len(out) > 100
    assert dt < 6.0


def test_seed_formulas_hook():
    sf = G.seed_formulas("C15H10O7")
    row = sf[sf["formula"] == "C15H10O6"]                # kaempferol formula: one +O edit
    assert len(row) == 1 and row["n_edits"].iloc[0] == 1
    assert row["mono_mass"].iloc[0] == pytest.approx(mono_mass(KAE), abs=1e-6)
    assert "C15H10O7" in set(sf["formula"])              # zero-delta moves
    two = sf[sf["n_edits"] == 2]
    assert len(two) > 0 and set(sf["n_edits"]) == {1, 2}
    assert sf["formula"].is_unique
    assert hill(formula_counts("C21H20O12")) in set(G.seed_formulas("C27H30O17")["formula"])


def test_config_defaults_used():
    cfg = Config()
    assert cfg.gen_max_edits == 2 and cfg.gen_max_products == 3000
    one = G.generate([QUE], ["C17H14O8"], cfg.replace(gen_max_edits=1))
    assert one.empty                                     # needs two edits (+CH2 +CH2O ...)
    two = G.generate([QUE], ["C17H14O7"])                 # two O-methylations
    assert (two["n_edits"] == 2).all() and len(two) > 0

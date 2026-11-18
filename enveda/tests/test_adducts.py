"""WP1 acceptance: adduct shifts, general parser, precursor m/z of known compounds."""

from __future__ import annotations

import pytest

from casmi import adducts
from casmi.adducts import (ELECTRON_MASS, TEST_ADDUCTS, Adduct, match_test_adduct,
                           neutral_mass, parse_adduct, precursor_mz, resolve_adduct)
from casmi.chem import mono_mass

# PLAN.md section 3.1, copied verbatim: precursor_mz = M + shift.
PLAN_TABLE = {
    "[M+H]+": (+1.007276, "positive"),
    "[M+NH4]+": (+18.033826, "positive"),
    "[M-H2O+H]+": (-17.003288, "positive"),
    "[M-2H2O+H]+": (-35.013853, "positive"),
    "[M+Na]+": (+22.989221, "positive"),
    "[M+K]+": (+38.963158, "positive"),
    "[M-H]-": (-1.007276, "negative"),
    "[M-H2O-H]-": (-19.017841, "negative"),
    "[M+CH2O2-H]-": (+44.998203, "negative"),
    "[M+Cl]-": (+34.969401, "negative"),
}

# Independent element masses (AME / IUPAC monoisotopic) for the hand checks below.
_M = dict(C=12.0, H=1.00782503207, N=14.0030740048, O=15.99491461956, Cl=34.96885268,
          Na=22.9897692809, K=38.96370668)
_E = 0.00054857990946


def _hand(**formula: int) -> float:
    return sum(_M[k] * v for k, v in formula.items())


def test_table_matches_plan():
    assert list(TEST_ADDUCTS) == list(PLAN_TABLE)
    for name, (shift, polarity) in PLAN_TABLE.items():
        a = TEST_ADDUCTS[name]
        assert a.multiplier == 1 and abs(a.charge) == 1
        assert a.polarity == polarity
        assert a.shift == pytest.approx(shift, abs=1e-5), name
        # M = precursor_mz - shift for every test adduct (all singly charged, monomers)
        assert neutral_mass(300.0 + a.shift, name) == pytest.approx(300.0, abs=1e-9)


def test_shifts_from_element_masses():
    """Each shift re-derived by hand from element masses and the electron mass."""
    hand = {
        "[M+H]+": _hand(H=1) - _E,
        "[M+NH4]+": _hand(N=1, H=4) - _E,
        "[M-H2O+H]+": -_hand(H=2, O=1) + _hand(H=1) - _E,
        "[M-2H2O+H]+": -2 * _hand(H=2, O=1) + _hand(H=1) - _E,
        "[M+Na]+": _hand(Na=1) - _E,
        "[M+K]+": _hand(K=1) - _E,
        "[M-H]-": -_hand(H=1) + _E,
        "[M-H2O-H]-": -_hand(H=2, O=1) - _hand(H=1) + _E,
        "[M+CH2O2-H]-": _hand(C=1, H=2, O=2) - _hand(H=1) + _E,
        "[M+Cl]-": _hand(Cl=1) + _E,
    }
    for name, value in hand.items():
        assert TEST_ADDUCTS[name].shift == pytest.approx(value, abs=1e-6), name
    assert ELECTRON_MASS == pytest.approx(0.000548580, abs=1e-9)


# Ten known compounds: (name, SMILES, adduct, hand-computed precursor m/z). Values from
# the molecular formula with the element masses above (computed outside casmi).
KNOWN = [
    ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "[M+H]+", 195.087652),
    ("amygdalin (glycoside)",
     "N#C[C@H](O[C@@H]1O[C@H](CO[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@H](O)"
     "[C@H]1O)c1ccccc1", "[M+NH4]+", 475.192236),
    ("cholesterol", "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC"
     "[C@]12C", "[M-H2O+H]+", 369.351578),
    ("glucose", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "[M-2H2O+H]+", 145.049535),
    ("diclofenac (Cl2)", "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl", "[M+Na]+", 318.005905),
    ("thyroxine (I4)", "N[C@@H](Cc1cc(I)c(Oc2cc(I)c(O)c(I)c2)c(I)c1)C(=O)O", "[M+K]+",
     815.649858),
    ("rutin (glycoside)",
     "C[C@@H]1O[C@@H](OC[C@H]2O[C@@H](Oc3c(-c4ccc(O)c(O)c4)oc4cc(O)cc(O)c4c3=O)[C@H](O)"
     "[C@@H](O)[C@@H]2O)[C@H](O)[C@H](O)[C@H]1O", "[M-H]-", 609.146108),
    ("quercetin", "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12", "[M-H2O-H]-", 283.024812),
    ("chloramphenicol (Cl2)", "OC[C@@H](NC(=O)C(Cl)Cl)[C@H](O)c1ccc([N+](=O)[O-])cc1",
     "[M+CH2O2-H]-", 367.010530),
    ("4-bromobenzoic acid (Br)", "OC(=O)c1ccc(Br)cc1", "[M+Cl]-", 234.916693),
]


@pytest.mark.parametrize("name,smiles,adduct,mz", KNOWN, ids=[k[0] for k in KNOWN])
def test_known_compounds(name, smiles, adduct, mz):
    m = mono_mass(smiles)
    assert precursor_mz(m, adduct) == pytest.approx(mz, abs=2e-6)
    assert neutral_mass(mz, adduct) == pytest.approx(m, abs=2e-6)


def test_general_parser_multimers_and_charges():
    h = TEST_ADDUCTS["[M+H]+"].shift
    rutin = 610.153385
    a = parse_adduct("[2M+H]+")
    assert a == Adduct(2, pytest.approx(h), 1)
    assert a.neutral_mass(2 * 194.080376 + h) == pytest.approx(194.080376, abs=1e-6)
    a = parse_adduct("[M+2H]2+")
    assert (a.multiplier, a.charge) == (1, 2)
    assert a.shift == pytest.approx(2 * h)
    assert a.precursor_mz(rutin) == pytest.approx(306.083969, abs=1e-6)
    assert a.neutral_mass(306.083969) == pytest.approx(rutin, abs=2e-6)
    a = parse_adduct("[M-2H]2-")
    assert (a.multiplier, a.charge, a.shift) == (1, -2, pytest.approx(-2 * h))
    a = parse_adduct("[3M+Na]+")
    assert (a.multiplier, a.charge) == (3, 1)
    assert a.shift == pytest.approx(TEST_ADDUCTS["[M+Na]+"].shift)
    # radical cation: only the electron
    assert parse_adduct("[M]+") == Adduct(1, pytest.approx(-ELECTRON_MASS), 1)
    assert parse_adduct("[M]+*") == parse_adduct("[M]+")
    # charge spellings
    for s in ("[M+H]1+", "[M+H]+1", "M+H+"):
        assert parse_adduct(s) == TEST_ADDUCTS["[M+H]+"], s
    assert parse_adduct("[M+2H]++") == parse_adduct("[M+2H]2+")


def test_general_parser_formula_terms_and_abbreviations():
    na = TEST_ADDUCTS["[M+Na]+"].shift
    h = adducts.element_mass("H")
    # sugar loss: [M+H-C6H10O5]+
    a = parse_adduct("[M+H-C6H10O5]+")
    assert a.shift == pytest.approx(TEST_ADDUCTS["[M+H]+"].shift - adducts.formula_mass("C6H10O5"))
    assert parse_adduct("[M+H-Hex]+") == a
    # [M+Na-2H]- : Na in, two H out, one electron in
    a = parse_adduct("[M+Na-2H]-")
    assert a.shift == pytest.approx(na + ELECTRON_MASS - 2 * h + ELECTRON_MASS)
    # spellings of formate and acetate adducts
    formate = TEST_ADDUCTS["[M+CH2O2-H]-"]
    for s in ("[M+FA-H]-", "[M+HCOO]-", "[M+HCOOH-H]-"):
        assert parse_adduct(s) == formate, s
        assert match_test_adduct(s) == "[M+CH2O2-H]-"
    acetate = parse_adduct("[M+CH3COO]-")
    assert parse_adduct("[M+Hac-H]-") == acetate
    assert acetate.shift == pytest.approx(adducts.formula_mass("C2H3O2") + ELECTRON_MASS)
    assert parse_adduct("[M+ACN+H]+").shift == pytest.approx(
        adducts.formula_mass("C2H3N") + TEST_ADDUCTS["[M+H]+"].shift)
    # reordered water loss is the same test adduct
    assert match_test_adduct("[M+H-H2O]+") == "[M-H2O+H]+"
    assert match_test_adduct("[M+H-2H2O]+") == "[M-2H2O+H]+"
    assert match_test_adduct("[2M+H]+") is None


def test_parser_rejects_garbage():
    for s in ("", "foo", "[M+Xyz]+", "[M+H]", "M+H", "[0M+H]+", "[M+0H]+", "[M+H]0+",
              "[M+e]-", "[M+H", None, 3.5):
        assert parse_adduct(s) is None, s


def test_mode_supplies_missing_charge_and_fallback():
    assert parse_adduct("[M+H]", "positive") == TEST_ADDUCTS["[M+H]+"]
    assert parse_adduct("M-H", "negative") == TEST_ADDUCTS["[M-H]-"]
    assert resolve_adduct("[M+Na]+", "positive") == (TEST_ADDUCTS["[M+Na]+"], True)
    assert resolve_adduct("[M+2H]2+", "positive")[1] is True
    assert resolve_adduct("[M+Foo]+", "positive") == (TEST_ADDUCTS["[M+H]+"], False)
    assert resolve_adduct("weird", "negative") == (TEST_ADDUCTS["[M-H]-"], False)
    assert resolve_adduct("[M+Foo]-", None) == (TEST_ADDUCTS["[M-H]-"], False)
    with pytest.raises(ValueError):
        resolve_adduct("weird", None)
    with pytest.raises(ValueError):
        neutral_mass(100.0, "[M+Foo]+")

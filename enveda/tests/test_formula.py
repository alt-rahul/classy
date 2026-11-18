"""WP12: formula enumeration, sub-formula explanation and the formula model (PLAN.md 3.4)."""

from __future__ import annotations

import itertools
import math
import time

import numpy as np
import pandas as pd
import pytest

from casmi import formula as F
from casmi.adducts import ELECTRON_MASS, TEST_ADDUCTS, precursor_mz
from casmi.chem import formula as rd_formula
from casmi.chem import mono_mass
from casmi.config import Config

# 54 natural products (name, SMILES, literature formula): flavonoids, glycosides,
# alkaloids, terpenoids, peptides up to 1,141 Da, and S / P / Cl / Br compounds.
NPS = [
    ("quercetin", "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12", "C15H10O7"),
    ("kaempferol", "O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12", "C15H10O6"),
    ("luteolin", "O=c1cc(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12", "C15H10O6"),
    ("apigenin", "O=c1cc(-c2ccc(O)cc2)oc2cc(O)cc(O)c12", "C15H10O5"),
    ("naringenin", "O=C1CC(c2ccc(O)cc2)Oc2cc(O)cc(O)c21", "C15H12O5"),
    ("catechin", "Oc1ccc(C2Oc3cc(O)cc(O)c3CC2O)cc1O", "C15H14O6"),
    ("genistein", "O=c1c(-c2ccc(O)cc2)coc2cc(O)cc(O)c12", "C15H10O5"),
    ("isorhamnetin", "COc1cc(-c2oc3cc(O)cc(O)c3c(=O)c2O)ccc1O", "C16H12O7"),
    ("myricetin", "O=c1c(O)c(-c2cc(O)c(O)c(O)c2)oc2cc(O)cc(O)c12", "C15H10O8"),
    ("rutin", "CC1OC(OCC2OC(Oc3c(-c4ccc(O)c(O)c4)oc4cc(O)cc(O)c4c3=O)C(O)C(O)C2O)"
     "C(O)C(O)C1O", "C27H30O16"),
    ("hesperidin", "COc1ccc(C2CC(=O)c3c(O)cc(OC4OC(COC5OC(C)C(O)C(O)C5O)C(O)C(O)C4O)"
     "cc3O2)cc1O", "C28H34O15"),
    ("naringin", "CC1OC(OC2C(Oc3cc(O)c4c(c3)OC(c3ccc(O)cc3)CC4=O)OC(CO)C(O)C2O)C(O)C(O)C1O",
     "C27H32O14"),
    ("vitexin", "OCC1OC(c2c(O)cc(O)c3c(=O)cc(-c4ccc(O)cc4)oc23)C(O)C(O)C1O", "C21H20O10"),
    ("amygdalin", "N#CC(OC1OC(COC2OC(CO)C(O)C(O)C2O)C(O)C(O)C1O)c1ccccc1", "C20H27NO11"),
    ("salicin", "OCc1ccccc1OC1OC(CO)C(O)C(O)C1O", "C13H18O7"),
    ("arbutin", "OCC1OC(Oc2ccc(O)cc2)C(O)C(O)C1O", "C12H16O7"),
    ("stevioside", "CC12CCCC(C)(C1CCC34CC(=C)C(OC5OC(CO)C(O)C(O)C5OC5OC(CO)C(O)C(O)C5O)"
     "(CCC23)C4)C(=O)OC1OC(CO)C(O)C(O)C1O", "C38H60O18"),
    ("digoxin", "CC1OC(OC2C(O)CC(OC3C(O)CC(OC4CCC5(C)C(CCC6C5CC(O)C5(C)C(C7=CC(=O)OC7)"
     "CCC65O)C4)OC3C)OC2C)CC(O)C1O", "C41H64O14"),
    ("glycyrrhizin", "CC1(C)C(OC2OC(C(=O)O)C(O)C(O)C2OC2OC(C(=O)O)C(O)C(O)C2O)CCC2(C)C1CCC1(C)"
     "C2C(=O)C=C2C3CC(C)(C(=O)O)CCC3(C)CCC21C", "C42H62O16"),
    ("ginsenoside_Rb1", "CC(C)=CCCC(C)(OC1OC(COC2OC(CO)C(O)C(O)C2O)C(O)C(O)C1O)C1CCC2(C)C1"
     "C(O)CC1C3(C)CCC(OC4OC(CO)C(O)C(O)C4OC4OC(CO)C(O)C(O)C4O)C(C)(C)C3CCC12C", "C54H92O23"),
    ("oleuropein", "COC(=O)C1=COC(OC2OC(CO)C(O)C(O)C2O)C(=CC)C1CC(=O)OCCc1ccc(O)c(O)c1",
     "C25H32O13"),
    ("caffeine", "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "C8H10N4O2"),
    ("nicotine", "CN1CCCC1c1cccnc1", "C10H14N2"),
    ("morphine", "CN1CCC23c4c5ccc(O)c4OC2C(O)C=CC3C1C5", "C17H19NO3"),
    ("strychnine", "O=C1CC2OCC=C3CN4CCC56C4CC3C2C6N1c1ccccc15", "C21H22N2O2"),
    ("quinine", "COc1ccc2nccc(C(O)C3CC4CCN3CC4C=C)c2c1", "C20H24N2O2"),
    ("reserpine", "COC(=O)C1C(OC)C(OC(=O)c2cc(OC)c(OC)c(OC)c2)CC2CN3CCc4c([nH]c5cc(OC)"
     "ccc45)C3CC21", "C33H40N2O9"),
    ("vinblastine", "CCC1(O)CC2CN(C1)CCc1c([nH]c3ccccc13)C(C(=O)OC)(c1cc3c(cc1OC)N(C)C1"
     "C34CCN3CC=CC(CC)(C34)C(OC(C)=O)C1(O)C(=O)OC)C2", "C46H58N4O9"),
    ("colchicine", "COc1cc2c(c(OC)c1OC)-c1ccc(OC)c(=O)cc1C(NC(C)=O)CC2", "C22H25NO6"),
    ("capsaicin", "COc1cc(CNC(=O)CCCCC=CC(C)C)ccc1O", "C18H27NO3"),
    ("piperine", "O=C(C=CC=Cc1ccc2c(c1)OCO2)N1CCCCC1", "C17H19NO3"),
    ("camptothecin", "CCC1(O)C(=O)OCc2c1cc1-c3nc4ccccc4cc3Cn1c2=O", "C20H16N2O4"),
    ("harmine", "COc1ccc2c(c1)[nH]c1c(C)nccc12", "C13H12N2O"),
    ("artemisinin", "CC1CCC2C(C)C(=O)OC3OC4(C)CCC1C32OO4", "C15H22O5"),
    ("paclitaxel", "CC(=O)OC1C(=O)C2(C)C(O)CC3OCC3(OC(C)=O)C2C(OC(=O)c2ccccc2)C2(O)CC("
     "OC(=O)C(O)C(NC(=O)c3ccccc3)c3ccccc3)C(C)=C1C2(C)C", "C47H51NO14"),
    ("betulinic_acid", "CC(=C)C1CCC2(C(=O)O)CCC3(C)C(CCC4C5(C)CCC(O)C(C)(C)C5CCC43C)C12",
     "C30H48O3"),
    ("oleanolic_acid", "CC1(C)CCC2(C(=O)O)CCC3(C)C(=CCC4C5(C)CCC(O)C(C)(C)C5CCC43C)C2C1",
     "C30H48O3"),
    ("abscisic_acid", "CC(C=CC1(O)C(C)=CC(=O)CC1(C)C)=CC(=O)O", "C15H20O4"),
    ("forskolin", "C=CC1(C)CC(=O)C2(O)C(C)(O1)C(OC(C)=O)C(O)C1C(C)(C)CCC(O)C12C", "C22H34O7"),
    ("andrographolide", "C=C1CCC2C(C)(CO)C(O)CCC2(C)C1CC=C1C(O)COC1=O", "C20H30O5"),
    ("limonin", "CC1(C)OC2CC(=O)OCC23C1CC(=O)C1(C)C3CCC2(C)C(c3ccoc3)OC(=O)C3OC321",
     "C26H30O8"),
    ("thymoquinone", "CC(C)C1=CC(=O)C(C)=CC1=O", "C10H12O2"),
    ("glutathione", "NC(CCC(=O)NC(CS)C(=O)NCC(=O)O)C(=O)O", "C10H17N3O6S"),
    ("leu_enkephalin", "CC(C)CC(NC(=O)C(Cc1ccccc1)NC(=O)CNC(=O)CNC(=O)C(N)Cc1ccc(O)cc1)"
     "C(=O)O", "C28H37N5O7"),
    ("oxytocin", "CCC(C)C1NC(=O)C(Cc2ccc(O)cc2)NC(=O)C(N)CSSCC(C(=O)N2CCCC2C(=O)NC(CC(C)C)"
     "C(=O)NCC(N)=O)NC(=O)C(CC(N)=O)NC(=O)C(CCC(N)=O)NC1=O", "C43H66N12O12S2"),
    ("gramicidin_S", "CC(C)CC1NC(=O)C(CCCN)NC(=O)C(C(C)C)NC(=O)C2CCCN2C(=O)C(Cc2ccccc2)NC(=O)"
     "C(CC(C)C)NC(=O)C(CCCN)NC(=O)C(C(C)C)NC(=O)C2CCCN2C(=O)C(Cc2ccccc2)NC1=O",
     "C60H92N12O10"),
    ("valinomycin", "CC(C)C1NC(=O)C(C)OC(=O)C(C(C)C)NC(=O)C(C(C)C)OC(=O)C(C(C)C)NC(=O)C(C)"
     "OC(=O)C(C(C)C)NC(=O)C(C(C)C)OC(=O)C(C(C)C)NC(=O)C(C)OC(=O)C(C(C)C)NC(=O)C(C(C)C)OC1=O",
     "C54H90N6O18"),
    ("sulforaphane", "CS(=O)CCCCN=C=S", "C6H11NOS2"),
    ("griseofulvin", "COC1=CC(=O)CC(C)C12Oc1c(Cl)c(OC)cc(OC)c1C2=O", "C17H17ClO6"),
    ("salinosporamide_A", "CC12OC(=O)C1(C(O)C1CCCC=C1)NC(=O)C2CCCl", "C15H20ClNO4"),
    ("glucoraphanin", "CS(=O)CCCCC(=NOS(=O)(=O)O)SC1OC(CO)C(O)C(O)C1O", "C12H23NO10S3"),
    ("NADH", "NC(=O)C1=CN(C2OC(COP(=O)(O)OP(=O)(O)OCC3OC(n4cnc5c(N)ncnc54)C(O)C3O)"
     "C(O)C2O)C=CC1", "C21H29N7O14P2"),
    ("penicillin_G", "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O", "C16H18N2O4S"),
    ("dibromoindigo", "O=C1C(=C2Nc3cc(Br)ccc3C2=O)Nc2cc(Br)ccc21", "C16H8Br2N2O2"),
]


# ------------------------------------------------------------------ strings and adducts

def test_hill_and_counts_roundtrip():
    for _, smi, lit in NPS:
        assert rd_formula(smi) == lit
        c = F.formula_counts(lit)
        assert c is not None and F.hill(c) == lit
        assert F.counts_mass(c) == pytest.approx(mono_mass(smi), abs=1e-6)
    assert F.hill(F.formula_counts("ClBrC2H4")) == rd_formula("CC(Cl)Br") == "C2H4BrCl"
    assert F.hill({"H": 2, "O": 1}) == "H2O"            # no carbon: alphabetical
    assert F.formula_counts("C6H12O6+") is None         # charge suffix rejected
    assert F.formula_counts("C2H6Si") is None           # outside the alphabet
    assert F.rdbe(F.formula_counts("C6H6")) == 4.0


def test_adduct_delta_matches_adduct_shifts():
    em = F.element_masses()
    for name, a in TEST_ADDUCTS.items():
        mult, delta, charge = F.adduct_delta(name)
        assert (mult, charge) == (a.multiplier, a.charge)
        shift = float(np.asarray(delta) @ em) - charge * ELECTRON_MASS
        assert shift == pytest.approx(a.shift, abs=1e-6), name
    assert F.adduct_delta("[M+Na]+")[1][F.ION_ELEMENTS.index("Na")] == 1
    assert F.adduct_delta("[2M+H]+")[0] == 2
    assert F.adduct_delta("[M+FA-H]-") == F.adduct_delta("[M+CH2O2-H]-")
    assert F.adduct_delta("[M+Ag]+") is None
    assert F.adduct_delta("garbage") is None


# ------------------------------------------------------------------ enumeration

@pytest.mark.parametrize("offset_ppm", [-7.0, 0.0, 7.0])
def test_enumeration_contains_known_natural_products(offset_ppm):
    """Acceptance: the true formula of 54 natural products is enumerated at 10 ppm."""
    missing = []
    for name, smi, lit in NPS:
        M = mono_mass(smi) * (1 + offset_ppm * 1e-6)
        fs = F.enumerate_formulas(M)
        if fs.index(lit) < 0:
            missing.append(name)
        tol = max(10e-6 * M, 0.002)                               # window with its floor
        assert np.all(np.abs(fs.mass - M) <= tol + 1e-9)
        assert np.all(np.diff(np.abs(fs.ppm)) >= -1e-9)          # sorted by |ppm|
    assert not missing


def _brute_force(mass: float, tol_ppm: float, cfg: Config) -> set[tuple[int, ...]]:
    """Independent enumeration: numpy grid over heavy atoms, H from the mass."""
    em = F.element_masses()[:10]
    lo, hi = mass * (1 - tol_ppm * 1e-6), mass * (1 + tol_ppm * 1e-6)
    hal = [h for h in itertools.product(range(5), repeat=4) if sum(h) <= cfg.formula_halogen_max]
    rows = []
    cmax = int(hi // 12)
    for c, n, o, p, s in itertools.product(range(1, cmax + 1), range(int(hi // 14) + 1),
                                           range(int(hi // 16) + 1), range(3), range(4)):
        base = c * em[0] + n * em[2] + o * em[3] + p * em[4] + s * em[5]
        if base > hi:
            continue
        for h4 in hal:
            m = base + float(np.dot(h4, em[6:10]))
            if m > hi:
                continue
            h0 = int(round((mass - m) / em[1]))
            for h in (h0 - 1, h0, h0 + 1):
                if h >= 0 and lo <= m + h * em[1] <= hi:
                    rows.append((c, h, n, o, p, s) + tuple(h4))
    if not rows:
        return set()
    arr = np.asarray(rows)
    return {tuple(int(v) for v in r) for r in arr[F.formula_ok(arr, cfg)]}


@pytest.mark.parametrize("mass", [160.0524, 231.1012, 305.0687])
def test_enumeration_equals_brute_force(mass):
    cfg = Config()
    fs = F.enumerate_formulas(mass, cfg, tol_ppm=50.0, min_tol_da=0.0)
    got = {tuple(int(v) for v in r) for r in fs.counts}
    assert got == _brute_force(mass, 50.0, cfg)
    assert F.formula_ok(fs.counts, cfg).all()


def test_enumeration_rules_and_window():
    cfg = Config()
    fs = F.enumerate_formulas(450.0, cfg)
    c = fs.counts.astype(int)
    assert len(fs) > 0
    assert np.all((c[:, 1] + c[:, 6:10].sum(1) + c[:, 2] + c[:, 4]) % 2 == 0)
    assert np.all(fs.rdbe >= 0) and np.all(fs.rdbe == np.round(fs.rdbe))
    assert np.all(c[:, 4] <= cfg.formula_p_max) and np.all(c[:, 5] <= cfg.formula_s_max)
    assert np.all(c[:, 6:10].sum(1) <= cfg.formula_halogen_max)
    assert np.all(c[:, 3] <= np.floor(cfg.formula_oc_max * c[:, 0] + 1e-9))
    tight = F.enumerate_formulas(450.0, cfg.replace(formula_halogen_max=0, formula_p_max=0,
                                                    formula_s_max=0))
    assert 0 < len(tight) < len(fs)
    assert np.all(tight.counts[:, 4:] == 0)
    assert len(F.enumerate_formulas(float("nan"))) == 0
    assert len(F.enumerate_formulas(-5.0)) == 0
    # the window floor (mass_tol_min_da) applies at low mass
    assert np.all(np.abs(F.enumerate_formulas(157.0).mass - 157.0) <= 0.002 + 1e-9)


def test_enumeration_speed():
    F.enumerate_formulas(300.0)                      # compile / load the numba cache
    def best(m: float) -> float:
        times = []
        for _ in range(5):
            t = time.perf_counter()
            F.enumerate_formulas(m)
            times.append(time.perf_counter() - t)
        return min(times)
    assert best(350.1) < 0.020
    assert best(1150.5) < 0.200


# ------------------------------------------------------------------ explanation

def _ion_sub_masses(ion: np.ndarray) -> np.ndarray:
    """All sub-formula masses of a small ion formula with fragment RDBE >= -0.5."""
    em = F.element_masses()
    ranges = [range(int(v) + 1) for v in ion]
    out = []
    for sub in itertools.product(*ranges):
        s = np.asarray(sub)
        x = s[6:10].sum() + s[10:].sum()
        if s[1] + x > 2 * s[0] + s[2] + s[4] + 3:
            continue
        out.append(float(s @ em))
    return np.asarray(out)


def test_explain_matches_brute_force():
    rng = np.random.default_rng(7)
    neutral = F.formula_counts("C8H10N4O2")          # caffeine
    for adduct in ("[M+H]+", "[M+Na]+", "[M-H]-"):
        mult, delta, charge = F.adduct_delta(adduct)
        ion = mult * np.r_[neutral, 0, 0] + np.asarray(delta)
        subs = _ion_sub_masses(ion)
        sign = 1.0 if charge > 0 else -1.0
        true_mz = rng.choice(subs[subs > 40], 30) - sign * ELECTRON_MASS
        mz = np.r_[true_mz + rng.normal(0, 0.003, 30), rng.uniform(40, 190, 30)]
        it = rng.uniform(0.05, 1.0, mz.size)
        prec = precursor_mz(F.counts_mass(neutral), adduct)
        spec = [(mz, it, prec, adduct)]
        peaks = F.prepare_peaks(spec, cleaned=True, tol_da=0.01)
        cands = np.vstack([neutral, F.formula_counts("C9H12N2O3"), F.formula_counts("C10H14O4")])
        ef, ep, ok = F.explain(cands, peaks)
        assert ok.all()
        for k, cnt in enumerate(cands):
            ion_k = mult * np.r_[cnt, 0, 0] + np.asarray(delta)
            sm = np.sort(_ion_sub_masses(ion_k))
            fm = mz + sign * ELECTRON_MASS
            pos = np.searchsorted(sm, fm)
            near = np.minimum(np.abs(fm - sm[np.clip(pos - 1, 0, sm.size - 1)]),
                              np.abs(fm - sm[np.clip(pos, 0, sm.size - 1)]))
            hit = near <= 0.01
            assert ef[k] == pytest.approx(it[hit].sum() / it.sum(), abs=1e-12), (adduct, k)
            assert ep[k] == pytest.approx(hit.mean(), abs=1e-12)


def test_explain_adducts_and_edge_cases():
    em = F.element_masses()
    neutral = F.formula_counts("C15H10O7")
    # [M+Na]+: a fragment that keeps Na (C6H5O2Na+) and one that does not (C7H5O+)
    na_frag = float(np.r_[F.formula_counts("C6H5O2"), 1, 0] @ em) - ELECTRON_MASS
    h_frag = float(np.r_[F.formula_counts("C7H5O"), 0, 0] @ em) - ELECTRON_MASS
    prec = precursor_mz(F.counts_mass(neutral), "[M+Na]+")
    spec = [(np.array([h_frag, na_frag]), np.array([1.0, 0.5]), prec, "[M+Na]+")]
    # 1 mDa: at 10 mDa a CHO sub-formula (C8H4O2, +2.4 mDa) would explain the Na fragment
    peaks = F.prepare_peaks(spec, cleaned=True, tol_da=0.001)
    ef, ep, ok = F.explain(neutral[None, :], peaks)
    assert ef[0] == pytest.approx(1.0) and ep[0] == pytest.approx(1.0)
    same_as_h = [(spec[0][0], spec[0][1], prec, "[M+H]+")]
    ef_h, _, _ = F.explain(neutral[None, :], F.prepare_peaks(same_as_h, cleaned=True,
                                                             tol_da=0.001))
    assert ef_h[0] == pytest.approx(1.0 / 1.5)        # the Na fragment is unexplained
    # [M-2H2O+H]+ needs at least two O: a formula with one O is not adduct-compatible
    one_o = F.formula_counts("C10H14O")
    spec2 = [(np.array([91.0542]), np.array([1.0]), 100.0, "[M-2H2O+H]+")]
    _, _, ok2 = F.explain(one_o[None, :], F.prepare_peaks(spec2, cleaned=True))
    assert not ok2[0]
    # no peaks -> NaN; unknown adducts are skipped
    empty = F.prepare_peaks([(np.zeros(0), np.zeros(0), 300.0, "[M+H]+")], cleaned=True)
    ef3, ep3, _ = F.explain(neutral[None, :], empty)
    assert math.isnan(ef3[0]) and math.isnan(ep3[0])
    skipped = F.prepare_peaks([(np.array([100.0]), np.array([1.0]), 300.0, "[M+Ag]+")])
    assert skipped.n_peaks == 0 and skipped.skipped == ("[M+Ag]+",)
    # string candidates are accepted
    ef4, _, _ = F.explain(["C15H10O7"], peaks)
    assert ef4[0] == pytest.approx(ef[0])


def test_explain_on_public_test(public_test_parquet):
    from casmi.io import molecule_table, read_test

    df = read_test(public_test_parquet)
    mt = molecule_table(df).sort_values("molecule_id").head(5)
    for mid, M in zip(mt["molecule_id"], mt["M"]):
        sub = df[df["molecule_id"] == mid]
        peaks = F.prepare_peaks(F.spectra_from_frame(sub))
        assert peaks.n_spectra == len(sub)
        fs = F.enumerate_formulas(float(M))
        ef, ep, ok = F.explain(fs, peaks)
        assert ef.shape == (len(fs),)
        assert np.nanmax(ef) <= 1.0 + 1e-12 and np.nanmin(ef) >= 0.0


# ------------------------------------------------------------------ features and model

def _spectrum_for(neutral_formula: str, adduct: str = "[M+H]+", n: int = 25, seed: int = 0):
    rng = np.random.default_rng(seed)
    neutral = F.formula_counts(neutral_formula)
    mult, delta, charge = F.adduct_delta(adduct)
    ion = mult * np.r_[neutral, 0, 0] + np.asarray(delta)
    em = F.element_masses()
    mz = []
    while len(mz) < n:
        sub = np.array([rng.integers(0, v + 1) for v in ion])
        m = float(sub @ em) - charge * ELECTRON_MASS
        if 50 < m < float(ion @ em) - 20:
            mz.append(m)
    prec = precursor_mz(F.counts_mass(neutral), adduct)
    return [(np.asarray(mz), rng.uniform(0.05, 1, n), prec, adduct)]


def test_formula_features_table():
    M = F.counts_mass(F.formula_counts("C15H10O7"))
    db_frame = pd.DataFrame({"formula": ["C15H10O7", "C15H10O7", "C16H14O6", "C99H2"],
                             "in_np": [True, False, True, True],
                             "in_pubchem": [True, True, False, False]})
    df = F.formula_features(M, _spectrum_for("C15H10O7"), db_frame=db_frame,
                            extra_formulas=["C14H6O8", "C15H10O7"],
                            lib_counts={"C15H10O7": 3})
    for col in F.FORMULA_FEATURES:
        assert col in df.columns
    row = df[df["formula"] == "C15H10O7"].iloc[0]
    assert (row["db_count_np"], row["db_count_pubchem"], row["lib_count"]) == (1, 2, 3)
    assert row["expl_frac"] == pytest.approx(1.0)
    assert row["expl_done"] and row["in_enumeration"]
    assert (df["formula"] == "C15H10O7").sum() == 1          # extras are deduplicated
    assert df["formula"].iloc[-1] == "C14H6O8" and not df["in_enumeration"].iloc[-1]
    assert (df["n_formulas"] == len(df)).all()
    # explanation cap: only expl_max rows get explained, DB-supported ones first
    capped = F.formula_features(M, _spectrum_for("C15H10O7"), db_frame=db_frame, expl_max=3)
    assert capped["expl_done"].sum() == 3
    assert capped.loc[capped["formula"] == "C15H10O7", "expl_done"].iloc[0]
    assert capped.loc[~capped["expl_done"], "expl_frac"].isna().all()
    # no spectra: NaN explanation, nothing filtered
    bare = F.formula_features(M)
    assert bare["expl_frac"].isna().all() and len(bare) == len(F.enumerate_formulas(M))


def test_formula_model_fit_predict_save_load(tmp_path):
    rng = np.random.default_rng(Config().seed)
    n = 4000
    X = pd.DataFrame({f: rng.normal(size=n) for f in F.FORMULA_FEATURES})
    true_w = np.zeros(len(F.FORMULA_FEATURES))
    true_w[[0, 1, 13]] = [-1.5, 2.0, 1.0]                 # abs_ppm, expl_frac, log_db_np
    p = 1 / (1 + np.exp(-(X.to_numpy() @ true_w - 1.0)))
    y = (rng.uniform(size=n) < p).astype(float)
    model = F.FormulaModel().fit(X, y, l2=1.0)
    assert model.fitted and model.meta["n_train"] == n
    w = dict(zip(model.features, model.coef * 1.0))
    assert w["abs_ppm"] < -1.0 and w["expl_frac"] > 1.4 and w["log_db_np"] > 0.6
    assert max(abs(w[f]) for f in F.FORMULA_FEATURES
               if f not in ("abs_ppm", "expl_frac", "log_db_np")) < 0.3
    prob = model.predict_proba(X)
    order = np.argsort(prob)
    assert y[order[-n // 10:]].mean() > 0.8 and y[order[: n // 10]].mean() < 0.1
    # NaN is imputed with the training mean (contributes nothing)
    Xn = X.head(3).copy()
    Xn["expl_frac"] = np.nan
    assert np.isfinite(model.predict_proba(Xn)).all()
    path = model.save(tmp_path / "formula_v1.json")
    again = F.FormulaModel.load(path)
    np.testing.assert_allclose(again.predict_proba(X), prob, rtol=0, atol=1e-12)
    assert F.load_formula_model(path).fitted
    with pytest.raises(KeyError):
        model.predict_proba(X.drop(columns=["abs_ppm"]))
    refit = F.FormulaModel().fit(X, y, l2=1.0)               # deterministic
    np.testing.assert_array_equal(refit.coef, model.coef)


def test_default_model_rank_and_targets(tmp_path):
    model = F.load_formula_model(tmp_path / "absent.json")   # falls back to the default
    assert not model.fitted
    M = F.counts_mass(F.formula_counts("C15H10O7"))
    db_frame = pd.DataFrame({"formula": ["C15H10O7"] * 40, "in_np": [True] * 40,
                             "in_pubchem": [True] * 40})
    ranked = F.rank_formulas(M, _spectrum_for("C15H10O7"), model=model, db_frame=db_frame)
    assert ranked["formula"].iloc[0] == "C15H10O7"
    assert np.all(np.diff(ranked["prob"].to_numpy()) <= 1e-12)
    assert list(ranked["rank"]) == list(range(len(ranked)))
    cfg = Config()
    targets = F.select_targets(ranked, cfg)
    assert 1 <= len(targets) <= cfg.gen_n_formulas and targets[0] == "C15H10O7"
    low = ranked.assign(prob=0.0)
    assert F.select_targets(low, cfg) == [ranked["formula"].iloc[0]]
    assert F.select_targets(low, cfg, min_one=False) == []


def test_fit_formula_model_hook(tmp_path):
    tables, truths = [], []
    for k, f in enumerate(["C15H10O7", "C15H10O6", "C16H12O7", "C21H20O12", "C10H14N2"]):
        M = F.counts_mass(F.formula_counts(f))
        db_frame = pd.DataFrame({"formula": [f] * 5, "in_np": [True] * 5,
                                 "in_pubchem": [True] * 5})
        tables.append(F.formula_features(M, _spectrum_for(f, seed=k), db_frame=db_frame))
        truths.append(f)
    model = F.fit_formula_model(tables, truths, save_path=tmp_path / "m.json")
    assert (tmp_path / "m.json").is_file() and model.meta["n_pos"] == 5
    for t, f in zip(tables, truths):
        p = model.predict_proba(t)
        assert t["formula"].iloc[int(np.argmax(p))] == f

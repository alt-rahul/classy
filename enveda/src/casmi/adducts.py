"""Adduct arithmetic: shifts, a general adduct parser, neutral mass <-> precursor m/z.

PLAN.md section 3.1. An adduct is described by three numbers:

- ``multiplier``: number of M in the ion (``[2M+H]+`` -> 2),
- ``shift``: mass added to ``multiplier * M`` to give the ion mass, in Da, with the
  electron mass included (``ion mass = multiplier * M + shift``),
- ``charge``: signed charge (``[M+2H]2+`` -> 2, ``[M-H]-`` -> -1).

so that ``precursor_mz = (multiplier * M + shift) / |charge|`` and
``M = (precursor_mz * |charge| - shift) / multiplier``.

``shift`` is the sum of the signed neutral masses of the terms minus ``charge``
electron masses: ``[M+H]+`` -> m(H) - m(e) = 1.007276; ``[M-H]-`` -> -m(H) + m(e).
Element masses are RDKit's most-common-isotope masses (the same numbers RDKit uses for
``CalcExactMolWt``), the electron mass is CODATA 2018.

Public API:
- ``TEST_ADDUCTS``: dict name -> Adduct for the ten test adducts (section 3.1 table).
- ``parse_adduct(s, ionization_mode=None) -> Adduct | None``: any ``[xM+A-B...]n+/-``.
- ``resolve_adduct(s, ionization_mode) -> (Adduct, known)``: never fails (falls back to
  ``[M+H]+`` / ``[M-H]-`` by mode, logged).
- ``neutral_mass(precursor_mz, adduct)``, ``precursor_mz(M, adduct)``.
- ``match_test_adduct(s)``: which of the ten test adducts ``s`` is (any spelling), or None.
- ``element_mass(symbol)``, ``formula_mass(formula)``.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import NamedTuple

__all__ = [
    "ELECTRON_MASS",
    "PROTON_MASS",
    "Adduct",
    "TEST_ADDUCT_NAMES",
    "TEST_ADDUCTS",
    "DEFAULT_ADDUCT",
    "ABBREVIATIONS",
    "element_mass",
    "formula_mass",
    "parse_formula",
    "parse_adduct",
    "resolve_adduct",
    "default_adduct",
    "match_test_adduct",
    "neutral_mass",
    "precursor_mz",
]

log = logging.getLogger(__name__)

ELECTRON_MASS: float = 0.000548579909   # Da, CODATA 2018 (the plan rounds to 0.000548580)


class Adduct(NamedTuple):
    """``ion mass = multiplier * M + shift``; ``m/z = ion mass / |charge|``."""

    multiplier: int
    shift: float
    charge: int

    @property
    def polarity(self) -> str:
        return "positive" if self.charge > 0 else "negative"

    def neutral_mass(self, mz: float) -> float:
        return (mz * abs(self.charge) - self.shift) / self.multiplier

    def precursor_mz(self, mass: float) -> float:
        return (self.multiplier * mass + self.shift) / abs(self.charge)


# The ten test adducts (COMPETITION.md), in the order of PLAN.md section 3.1.
TEST_ADDUCT_NAMES: tuple[str, ...] = (
    "[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+",
    "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-",
)

DEFAULT_ADDUCT: dict[str, str] = {"positive": "[M+H]+", "negative": "[M-H]-"}

# Abbreviations seen in library adduct strings, as neutral formulas. Plain formulas
# (H2O, NH4, HCOO, CH3COO, CH3CN, C6H10O5 ...) need no entry: they are parsed as formulas.
ABBREVIATIONS: dict[str, str] = {
    "FA": "CH2O2",        # formic acid
    "HFA": "CH2O2",
    "Hac": "C2H4O2",      # acetic acid
    "HAc": "C2H4O2",
    "AcOH": "C2H4O2",
    "OAc": "C2H3O2",      # acetate
    "ACN": "C2H3N",       # acetonitrile
    "MeCN": "C2H3N",
    "MeOH": "CH4O",
    "EtOH": "C2H6O",
    "IsoProp": "C3H8O",   # isopropanol
    "IPA": "C3H8O",
    "iPrOH": "C3H8O",
    "DMSO": "C2H6OS",
    "TFA": "C2HF3O2",     # trifluoroacetic acid
    "Hex": "C6H10O5",     # hexose residue (loss)
    "dHex": "C6H10O4",    # deoxyhexose
    "DeoxyHex": "C6H10O4",
    "Pen": "C5H8O4",      # pentose
    "HexA": "C6H8O6",     # hexuronic acid
}
_ABBREV_LOWER = {k.lower(): v for k, v in ABBREVIATIONS.items()}

_ELEMENT_RE = re.compile(r"([A-Z][a-z]?)(\d*)")
_FORMULA_RE = re.compile(r"^(?:[A-Z][a-z]?\d*)+$")
_TERM_RE = re.compile(r"([+-])(\d*)([A-Za-z][A-Za-z0-9]*)")
_HEAD_RE = re.compile(r"^(\d*)M")
_CHARGE_RE = re.compile(r"^(?:(\d*)([+-])|([+-])(\d+)|(\+{2,}|-{2,}))$")


@lru_cache(maxsize=1)
def _element_masses() -> dict[str, float]:
    from rdkit import Chem

    pt = Chem.GetPeriodicTable()
    out = {}
    for z in range(1, 119):
        mass = float(pt.GetMostCommonIsotopeMass(z))
        if mass > 0:
            out[pt.GetElementSymbol(z)] = mass
    return out


def element_mass(symbol: str) -> float:
    """Most-common-isotope mass of an element (RDKit periodic table). KeyError if unknown."""
    return _element_masses()[symbol]


def parse_formula(formula: str) -> dict[str, int] | None:
    """``"C6H10O5"`` -> ``{"C": 6, "H": 10, "O": 5}``; repeated symbols add up
    (``"CH3COO"`` -> C2H3O2). None if not a plain formula of known elements."""
    if not formula or not _FORMULA_RE.match(formula):
        return None
    counts: dict[str, int] = {}
    for sym, num in _ELEMENT_RE.findall(formula):
        try:
            element_mass(sym)
        except KeyError:
            return None
        counts[sym] = counts.get(sym, 0) + (int(num) if num else 1)
    return counts


def formula_mass(formula: str) -> float:
    """Monoisotopic mass of a neutral formula string (no charge suffix). ValueError if bad."""
    counts = parse_formula(formula)
    if counts is None:
        raise ValueError(f"not a formula: {formula!r}")
    return sum(element_mass(el) * n for el, n in counts.items())


PROTON_MASS: float = element_mass("H") - ELECTRON_MASS


def _term_mass(token: str) -> float | None:
    formula = ABBREVIATIONS.get(token) or _ABBREV_LOWER.get(token.lower())
    if formula is None:
        formula = token
    counts = parse_formula(formula)
    if counts is None:
        return None
    return sum(element_mass(el) * n for el, n in counts.items())


def _parse_charge(text: str) -> int | None:
    m = _CHARGE_RE.match(text)
    if m is None:
        return None
    if m.group(2):
        n = int(m.group(1)) if m.group(1) else 1
        sign = m.group(2)
    elif m.group(3):
        n, sign = int(m.group(4)), m.group(3)
    else:
        n, sign = len(m.group(5)), m.group(5)[0]
    if n == 0:
        return None
    return n if sign == "+" else -n


def _parse_body(body: str) -> tuple[int, float] | None:
    """``"2M+Na-2H"`` -> (2, summed neutral mass of the terms)."""
    head = _HEAD_RE.match(body)
    if head is None:
        return None
    mult = int(head.group(1)) if head.group(1) else 1
    if mult < 1:
        return None
    rest = body[head.end():]
    total = 0.0
    pos = 0
    for m in _TERM_RE.finditer(rest):
        if m.start() != pos:
            return None
        sign, count, token = m.groups()
        mass = _term_mass(token)
        if mass is None:
            return None
        k = int(count) if count else 1
        if k < 1:
            return None
        total += (1.0 if sign == "+" else -1.0) * k * mass
        pos = m.end()
    if pos != len(rest):
        return None
    return mult, total


@lru_cache(maxsize=4096)
def _parse(s: str, mode_sign: int) -> Adduct | None:
    text = re.sub(r"\s+", "", s)
    text = text.rstrip("*.\u2022")          # radical markers: [M]+*, [M]+.
    if not text:
        return None
    if text.startswith("["):
        close = text.find("]")
        if close < 0:
            return None
        body, charge_text = text[1:close], text[close + 1:]
    else:
        # Unbracketed ("M+H", "M+H+", "M-H-"): a trailing run of signs is the charge.
        m = re.match(r"^(.*?[A-Za-z0-9])((?:\d*[+-]|[+-]\d+|\+{2,}|-{2,}))?$", text)
        if m is None:
            return None
        # Shortest body first, so "M+2H2+" reads as [M+2H]2+ (the usual meaning).
        body, charge_text = m.group(1), m.group(2) or ""
    if charge_text:
        charge = _parse_charge(charge_text)
        if charge is None:
            return None
    elif mode_sign != 0:
        charge = mode_sign
    else:
        return None
    parsed = _parse_body(body)
    if parsed is None:
        return None
    mult, terms = parsed
    return Adduct(mult, terms - charge * ELECTRON_MASS, charge)


def _mode_sign(ionization_mode: str | None) -> int:
    if not ionization_mode:
        return 0
    m = str(ionization_mode).strip().lower()
    if m in ("positive", "pos", "+", "p"):
        return 1
    if m in ("negative", "neg", "-", "n"):
        return -1
    return 0


def parse_adduct(s: str | None, ionization_mode: str | None = None) -> Adduct | None:
    """Parse ``[xM+A-B...]n+/-`` into (multiplier, shift, charge); None if not understood.

    Terms are formulas (``H2O``, ``C6H10O5``, ``HCOO``) or abbreviations from
    ``ABBREVIATIONS`` (``FA``, ``ACN``, ``Hac`` ...), each with an optional count prefix
    (``2H``, ``2H2O``). Charge forms: ``+``, ``2+``, ``+2``, ``++`` (and negative). If the
    string has no charge, the sign comes from ``ionization_mode`` (charge +-1), else None.
    """
    if s is None or not isinstance(s, str):
        return None
    return _parse(s, _mode_sign(ionization_mode))


def default_adduct(ionization_mode: str) -> str:
    """``[M+H]+`` for positive mode, ``[M-H]-`` for negative. ValueError otherwise."""
    sign = _mode_sign(ionization_mode)
    if sign == 0:
        raise ValueError(f"unknown ionization mode: {ionization_mode!r}")
    return DEFAULT_ADDUCT["positive" if sign > 0 else "negative"]


def resolve_adduct(s: str | None, ionization_mode: str | None = None) -> tuple[Adduct, bool]:
    """Adduct for a spectrum, never failing: ``(adduct, known)``.

    Test table first, then the general parser; if both fail, the default adduct of the
    polarity (``ionization_mode``, else the sign at the end of ``s``) with
    ``known=False`` and a warning (PLAN.md 3.1). ValueError if no polarity is available.
    """
    if isinstance(s, str) and s in TEST_ADDUCTS:
        return TEST_ADDUCTS[s], True
    a = parse_adduct(s, ionization_mode)
    if a is not None:
        return a, True
    sign = _mode_sign(ionization_mode)
    if sign == 0 and isinstance(s, str) and s.strip()[-1:] in "+-":
        sign = 1 if s.strip()[-1] == "+" else -1
    if sign == 0:
        raise ValueError(f"cannot parse adduct {s!r} and no ionization mode given")
    name = DEFAULT_ADDUCT["positive" if sign > 0 else "negative"]
    log.warning("unknown adduct %r (mode %r): using %s", s, ionization_mode, name)
    return TEST_ADDUCTS[name], False


def _as_adduct(adduct: str | Adduct) -> Adduct:
    if isinstance(adduct, Adduct):
        return adduct
    a = TEST_ADDUCTS.get(adduct) if isinstance(adduct, str) else None
    if a is None:
        a = parse_adduct(adduct)
    if a is None:
        raise ValueError(f"cannot parse adduct {adduct!r}")
    return a


def neutral_mass(precursor_mz: float, adduct: str | Adduct) -> float:
    """M = (precursor_mz * |z| - shift) / multiplier. ValueError if the adduct is unparseable."""
    return _as_adduct(adduct).neutral_mass(precursor_mz)


def precursor_mz(mass: float, adduct: str | Adduct) -> float:
    """precursor m/z = (multiplier * M + shift) / |z|. ValueError if the adduct is unparseable."""
    return _as_adduct(adduct).precursor_mz(mass)


def _build_test_adducts() -> dict[str, Adduct]:
    out = {}
    for name in TEST_ADDUCT_NAMES:
        a = _parse(name, 0)
        assert a is not None, name
        out[name] = a
    return out


TEST_ADDUCTS: dict[str, Adduct] = _build_test_adducts()


def match_test_adduct(s: str | Adduct | None, ionization_mode: str | None = None) -> str | None:
    """Name of the test adduct that ``s`` is numerically identical to (any spelling, e.g.
    ``[M+H-H2O]+`` -> ``[M-H2O+H]+``, ``[M+FA-H]-`` and ``[M+HCOO]-`` -> ``[M+CH2O2-H]-``),
    else None."""
    if isinstance(s, str) and s in TEST_ADDUCTS:
        return s
    a = s if isinstance(s, Adduct) else parse_adduct(s, ionization_mode)
    if a is None:
        return None
    for name, t in TEST_ADDUCTS.items():
        if t.multiplier == a.multiplier and t.charge == a.charge and abs(t.shift - a.shift) < 1e-6:
            return name
    return None

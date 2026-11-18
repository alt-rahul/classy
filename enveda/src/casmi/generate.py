"""Class 3 generator: rule-based analogs of seed structures at target formulas (WP12).

PLAN.md section 4.7. Deterministic, RDKit only. Novel natural products are usually one
or two biosynthetic edits away from a known compound, so the generator applies an edit
library (reaction SMARTS with exact formula deltas) to seed structures and keeps the
products whose formula is exactly one of the target formulas (section 3.4).

Public API:

- ``EditRule`` and ``RULES``: the edit library (id, delta family, reaction SMARTS,
  formula delta in ``formula.ELEMENTS`` order). ``RULE_IDS`` maps id -> integer code
  (the ``edit1`` / ``edit2`` ranker features). ``rules_table()`` lists them.
- ``apply_rule(rule, mol) -> list[(smiles, Mol)]``: every distinct sanitised product of
  one rule at every matching site (seed excluded), sorted by SMILES.
- ``prepare_seed(smiles_or_mol) -> Mol | None``: ``chem.std_mol`` (cleanup, uncharge,
  no stereo) then a SMILES round trip so hydrogens are implicit.
- ``generate(seeds, target_formulas, cfg=None, exclude_ik14=set(), seed_scores=None)
  -> DataFrame``: up to ``cfg.gen_max_edits`` edits per seed, products sanitised, with
  exactly a target formula, net charge 0, one fragment, not a seed and not in
  ``exclude_ik14``, deduplicated by InChIKey first block, capped at
  ``cfg.gen_max_products`` with seeds taken in score order. Columns: ``GEN_COLUMNS``.
- ``seed_formulas(target, max_edits=2) -> DataFrame``: the formulas a seed must have to
  reach ``target`` within the edit budget, with the mass to query the DB at
  (seed selection hook, section 4.7 (b)).

Search: for seed formula ``s`` and target ``t`` the needed delta is ``D = t - s``. Edit
paths are only explored when their summed deltas can reach ``D`` (one edit with delta
``D``, or two edits ``r1, r2`` with ``delta(r1) + delta(r2) = D``, zero-delta moves
included), so seeds far from every target cost one formula comparison. Intermediates
(after the first edit) are deduplicated by canonical SMILES per seed.

Output order (deterministic): seed rank (score descending, then input position), then
number of edits, first edit code, second edit code, target index, SMILES. The first row
per ik14 is kept (its provenance is reported); ``n_seeds`` and ``n_paths`` count every
processed seed and edit path that produced the same ik14. Seeds after the cap is reached
are not processed.

NEEDS_TRAIN hooks (TODO WP12): choosing seeds (top analog hits from 4.3 and top DB/LIB
candidates by fingerprint score at ``seed_formulas`` masses) and scoring products with
the fingerprint model; both need train-derived artefacts. ``generate`` takes the seeds
and their scores as given.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import rdChemReactions, rdMolDescriptors

from .chem import mol_from_smiles, std_mol
from .config import Config
from .formula import ELEMENTS, counts_mass, formula_counts, hill
from .metric import ik14 as _ik14

__all__ = [
    "EditRule",
    "RULES",
    "RULE_IDS",
    "GEN_COLUMNS",
    "rules_table",
    "get_rule",
    "prepare_seed",
    "mol_counts",
    "apply_rule",
    "generate",
    "seed_formulas",
]

log = logging.getLogger(__name__)


def _d(formula: str, sign: int = 1) -> tuple[int, ...]:
    c = formula_counts(formula)
    assert c is not None, formula
    return tuple(int(sign * v) for v in c)


def _sum(*deltas: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(sum(v)) for v in zip(*deltas))


_ZERO: tuple[int, ...] = (0,) * len(ELEMENTS)


@dataclass(frozen=True)
class EditRule:
    """One edit: ``smarts`` is a one-reactant reaction SMARTS (intramolecular moves use
    component grouping), ``delta`` the exact formula change in ``ELEMENTS`` order."""

    id: str
    family: str
    smarts: str
    delta: tuple[int, ...]

    @property
    def delta_formula(self) -> str:
        """Signed delta as text, e.g. ``+C6H10O5``, ``-CH2``, ``+H+N-O``, ``0``."""
        if not any(self.delta):
            return "0"
        pos = {ELEMENTS[i]: v for i, v in enumerate(self.delta) if v > 0}
        neg = {ELEMENTS[i]: -v for i, v in enumerate(self.delta) if v < 0}
        out = ""
        if pos:
            out += "+" + hill(pos)
        if neg:
            out += "-" + hill(neg)
        return out


# Shared atom patterns.
_OH = "[OX2H1;$(O-[#6;!$([#6]=[O,S,N])])]"          # alcohol / phenol OH (not acid)
_OH_ANY = "[OX2H1;$(O-[#6])]"                        # any OH on carbon (incl. acid)
_NH = "[#7X3;!H0;+0]"                                # N-H: amine, amide, aromatic nH
_NH_AMINE = "[NX3;!H0;+0;!$(N-[#6]=[O,S,N])]"       # amine / aniline N-H (not amide)
_CH_SP3 = "[CX4;!H0;!$(C~[!#6;!#1])]"               # sp3 C-H not next to a heteroatom
_HEX = "OCC1OC({a})C(O)C(O)C1O"                       # hexopyranose (glucose, 2D)
_DHEX = "CC1OC({a})C(O)C(O)C1O"                       # 6-deoxyhexose (rhamnose)
_PENT_P = "OC1COC({a})C(O)C1O"                        # pentopyranose (xylose)
_PENT_F = "OCC1OC({a})C(O)C1O"                        # pentofuranose (arabinose, ribose)
_GLCA = "OC(=O)C1OC({a})C(O)C(O)C1O"                  # glucuronic acid
_HEX_RM = ("[#8,#6,#7:1]-[CH1]1-[OX2]-[CH1](-[CH2]-[OH1])-[CH1](-[OH1])-[CH1](-[OH1])"
           "-[CH1]1-[OH1]>>[*:1]")
_DHEX_RM = ("[#8,#6,#7:1]-[CH1]1-[OX2]-[CH1](-[CH3])-[CH1](-[OH1])-[CH1](-[OH1])"
            "-[CH1]1-[OH1]>>[*:1]")
_PENT_P_RM = "[#8,#6,#7:1]-[CH1]1-[OX2]-[CH2]-[CH1](-[OH1])-[CH1](-[OH1])-[CH1]1-[OH1]>>[*:1]"
_PENT_F_RM = ("[#8,#6,#7:1]-[CH1]1-[OX2]-[CH1](-[CH2]-[OH1])-[CH1](-[OH1])-[CH1]1-[OH1]"
              ">>[*:1]")
_GLCA_RM = ("[#8,#6,#7:1]-[CH1]1-[OX2]-[CH1](-[CX3](=O)-[OH1])-[CH1](-[OH1])-[CH1](-[OH1])"
            "-[CH1]1-[OH1]>>[*:1]")
_ESTER_RM = "[#6:2]-[OX2:1]-{acyl}>>[#6:2]-[O:1]"   # remove an acyl group from an ester O

_HEXOSE, _DEOXYHEX, _PENTOSE, _GLUCURONIC = (_d("C6H10O5"), _d("C6H10O4"), _d("C5H8O4"),
                                             _d("C6H8O6"))
_NH_MINUS_O = _sum(_d("NH"), _d("O", -1))

# The edit library (PLAN.md 4.7). Order matters only for output order and edit codes;
# append new rules at the end so codes stay stable.
RULES: tuple[EditRule, ...] = (
    # CH2: O-, N-, C-methylation and demethylation
    EditRule("O_methylation", "CH2", f"{_OH_ANY[:-1]}:1]>>C[O:1]", _d("CH2")),
    EditRule("O_demethylation", "CH2", "[CH3;X4]-[OX2;H0:1]>>[O:1]", _d("CH2", -1)),
    EditRule("N_methylation", "CH2", f"{_NH[:-1]}:1]>>C[#7:1]", _d("CH2")),
    EditRule("N_demethylation", "CH2", "[CH3;X4]-[#7;+0:1]>>[#7:1]", _d("CH2", -1)),
    EditRule("C_methylation", "CH2", "[cH:1]>>C[c:1]", _d("CH2")),
    EditRule("C_demethylation", "CH2", "[CH3;X4]-[#6:1]>>[#6:1]", _d("CH2", -1)),
    # O: hydroxylation, N-oxide, epoxide; and removal
    EditRule("hydroxylation_arom", "O", "[cH:1]>>O[c:1]", _d("O")),
    EditRule("hydroxylation_aliph", "O", f"{_CH_SP3[:-1]}:1]>>O[C:1]", _d("O")),
    EditRule("dehydroxylation", "O", "[#6;!$([#6]=[O,S,N]):1]-[OX2H1]>>[#6:1]", _d("O", -1)),
    EditRule("N_oxidation", "O",
             "[NX3;H0;+0;!$(N-[#6,#7,#16,#15]=[#6,#7,#8,#16]);!$(N-a):1]>>[O-][N+:1]",
             _d("O")),
    EditRule("N_oxidation_arom", "O", "[nX2;+0:1]>>[O-][n+:1]", _d("O")),
    EditRule("N_deoxygenation", "O", "[#7+:1]-[OX1-]>>[#7+0:1]", _d("O", -1)),
    EditRule("epoxidation", "O", "[C:1]=[C:2]>>[C:1]1-[C:2]-O1", _d("O")),
    EditRule("deepoxidation", "O", "[C:1]1-[C:2]-[OX2]-1>>[C:1]=[C:2]", _d("O", -1)),
    # H2: hydrogenation of C=C and C=O, dehydrogenation
    EditRule("hydrogenation_CC", "H2", "[C:1]=[C:2]>>[C:1]-[C:2]", _d("H2")),
    EditRule("hydrogenation_CO", "H2", "[CX3;!$(C-[!#6;!#1]):1]=[OX1:2]>>[C:1]-[O:2]",
             _d("H2")),
    EditRule("dehydrogenation_CC", "H2", "[CX4;!H0:1]-[CX4;!H0:2]>>[C:1]=[C:2]",
             _d("H2", -1)),
    EditRule("dehydrogenation_CO", "H2", "[CX4;!H0:1]-[OX2H1:2]>>[C:1]=[O:2]", _d("H2", -1)),
    # CO: carbonyl insertion, formylation; and removal
    EditRule("carbonyl_insertion", "CO", "[#6:1]-!@[CX4:2]>>[#6:1]-C(=O)-[C:2]", _d("CO")),
    EditRule("formylation_arom", "CO", "[cH:1]>>O=C[c:1]", _d("CO")),
    EditRule("N_formylation", "CO", f"{_NH_AMINE[:-1]}:1]>>O=C[#7:1]", _d("CO")),
    EditRule("carbonyl_removal", "CO", "[#6:1]-[CX3;R0](=O)-[#6:2]>>[#6:1]-[#6:2]",
             _d("CO", -1)),
    EditRule("deformylation", "CO", "[#6,#7:1]-[CH1;X3]=O>>[*:1]", _d("CO", -1)),
    # C2H2O: acetylation and deacetylation
    EditRule("O_acetylation", "C2H2O", f"{_OH[:-1]}:1]>>CC(=O)[O:1]", _d("C2H2O")),
    EditRule("N_acetylation", "C2H2O", f"{_NH_AMINE[:-1]}:1]>>CC(=O)[#7:1]", _d("C2H2O")),
    EditRule("deacetylation", "C2H2O", "[CH3]-[CX3](=O)-[#8,#7;X2,X3:1]>>[*:1]",
             _d("C2H2O", -1)),
    # C5H8: prenylation
    EditRule("C_prenylation", "C5H8", "[cH:1]>>CC(C)=CC[c:1]", _d("C5H8")),
    EditRule("O_prenylation", "C5H8", "[OX2H1;$(O-c):1]>>CC(C)=CC[O:1]", _d("C5H8")),
    EditRule("deprenylation", "C5H8", "[CH3]-[CX3](-[CH3])=[CH1]-[CH2]-[#6,#7,#8:1]>>[*:1]",
             _d("C5H8", -1)),
    # glycosylation (to OH, or C-glycoside on aromatic CH) and deglycosylation
    EditRule("O_hexosylation", "C6H10O5", f"{_OH[:-1]}:1]>>" + _HEX.format(a="[O:1]"),
             _HEXOSE),
    EditRule("C_hexosylation", "C6H10O5", "[cH:1]>>" + _HEX.format(a="[c:1]"), _HEXOSE),
    EditRule("dehexosylation", "C6H10O5", _HEX_RM, _d("C6H10O5", -1)),
    EditRule("O_deoxyhexosylation", "C6H10O4", f"{_OH[:-1]}:1]>>" + _DHEX.format(a="[O:1]"),
             _DEOXYHEX),
    EditRule("dedeoxyhexosylation", "C6H10O4", _DHEX_RM, _d("C6H10O4", -1)),
    EditRule("O_pentosylation", "C5H8O4", f"{_OH[:-1]}:1]>>" + _PENT_P.format(a="[O:1]"),
             _PENTOSE),
    EditRule("O_pentofuranosylation", "C5H8O4",
             f"{_OH[:-1]}:1]>>" + _PENT_F.format(a="[O:1]"), _PENTOSE),
    EditRule("depentosylation", "C5H8O4", _PENT_P_RM, _d("C5H8O4", -1)),
    EditRule("depentofuranosylation", "C5H8O4", _PENT_F_RM, _d("C5H8O4", -1)),
    EditRule("O_glucuronidation", "C6H8O6", f"{_OH[:-1]}:1]>>" + _GLCA.format(a="[O:1]"),
             _GLUCURONIC),
    EditRule("deglucuronidation", "C6H8O6", _GLCA_RM, _d("C6H8O6", -1)),
    # SO3: sulfation
    EditRule("sulfation", "SO3", f"{_OH[:-1]}:1]>>OS(=O)(=O)[O:1]", _d("SO3")),
    EditRule("desulfation", "SO3", "[#6:2]-[OX2:1]-S(=O)(=O)-[OX2H1]>>[#6:2]-[O:1]",
             _d("SO3", -1)),
    # H2O: dehydration, lactone / ether ring closure; hydration, ring opening
    EditRule("dehydration", "H2O", "[CX4;!H0:1]-[CX4:2]-[OX2H1]>>[C:1]=[C:2]",
             _d("H2O", -1)),
    EditRule("lactonization5", "H2O",
             "[OX2H1]-[CX3:1](=[O:2])-[#6:3]~[#6:4]~[#6:5]-[OX2H1:6]"
             ">>[C:1]1(=[O:2])-[#6:3]~[#6:4]~[#6:5]-[O:6]-1", _d("H2O", -1)),
    EditRule("lactonization6", "H2O",
             "[OX2H1]-[CX3:1](=[O:2])-[#6:3]~[#6:4]~[#6:5]~[#6:6]-[OX2H1:7]"
             ">>[C:1]1(=[O:2])-[#6:3]~[#6:4]~[#6:5]~[#6:6]-[O:7]-1", _d("H2O", -1)),
    EditRule("ether_closure5", "H2O",
             "[OX2H1:1]-[#6:2]~[#6:3]~[#6:4]-[CX4:5]-[OX2H1]"
             ">>[O:1]1-[#6:2]~[#6:3]~[#6:4]-[C:5]-1", _d("H2O", -1)),
    EditRule("ether_closure6", "H2O",
             "[OX2H1:1]-[#6:2]~[#6:3]~[#6:4]~[#6:5]-[CX4:6]-[OX2H1]"
             ">>[O:1]1-[#6:2]~[#6:3]~[#6:4]~[#6:5]-[C:6]-1", _d("H2O", -1)),
    EditRule("hydration", "H2O", "[C:1]=[C:2]>>[C:1]-[C:2]-O", _d("H2O")),
    EditRule("lactone_opening", "H2O",
             "[CX3:1](=[O:2])-@[OX2:3]-@[#6:4]>>([C:1](=[O:2])-O.[O:3]-[#6:4])", _d("H2O")),
    EditRule("ether_opening", "H2O",
             "[CX4;!$(C(~[#8])~[#8]):1]-@[OX2:2]-@[CX4;!$(C(~[#8])~[#8]):3]"
             ">>([C:1]-[O:2].[C:3]-O)", _d("H2O")),
    # CO2: carboxylation, decarboxylation
    EditRule("carboxylation", "CO2", "[cH:1]>>OC(=O)[c:1]", _d("CO2")),
    EditRule("decarboxylation", "CO2", "[#6:1]-[CX3](=O)-[OX2H1]>>[#6:1]", _d("CO2", -1)),
    # NH vs O swaps (amide / ester, amine / hydroxyl): delta +NH -O and back
    EditRule("OH_to_NH2", "NH-O", "[#6:1]-[OX2H1]>>[#6:1]-N", _NH_MINUS_O),
    EditRule("NH2_to_OH", "NH-O", "[#6:1]-[NX3H2;+0]>>[#6:1]-O",
             tuple(-v for v in _NH_MINUS_O)),
    EditRule("ester_to_amide", "NH-O", "[CX3:1](=[O:2])-[OX2]-[#6:4]>>[C:1](=[O:2])-N-[#6:4]",
             _NH_MINUS_O),
    EditRule("amide_to_ester", "NH-O",
             "[CX3:1](=[O:2])-[NX3H1;+0]-[#6:4]>>[C:1](=[O:2])-O-[#6:4]",
             tuple(-v for v in _NH_MINUS_O)),
    # C2H4: chain extension and shortening by two CH2
    EditRule("chain_extension_C2H4", "C2H4", "[CX4;!R:1]-[CH3;X4:2]>>[C:1]-C-C-[C:2]",
             _d("C2H4")),
    EditRule("chain_shortening_C2H4", "C2H4",
             "[#6:1]-[CH2;!R]-[CH2;!R]-[CH3:2]>>[#6:1]-[C:2]", _d("C2H4", -1)),
    # Acyl groups on OH: malonyl (C3H2O3), benzoyl (C7H4O), p-coumaroyl (C9H6O2),
    # caffeoyl (C9H6O3); and their removal
    EditRule("malonylation", "C3H2O3", f"{_OH[:-1]}:1]>>OC(=O)CC(=O)[O:1]", _d("C3H2O3")),
    EditRule("demalonylation", "C3H2O3",
             _ESTER_RM.format(acyl="[CX3](=O)-[CH2]-[CX3](=O)-[OX2H1]"), _d("C3H2O3", -1)),
    EditRule("benzoylation", "C7H4O", f"{_OH[:-1]}:1]>>O=C(c1ccccc1)[O:1]", _d("C7H4O")),
    EditRule("debenzoylation", "C7H4O",
             _ESTER_RM.format(acyl="[CX3](=O)-c1[cH][cH][cH][cH][cH]1"), _d("C7H4O", -1)),
    EditRule("coumaroylation", "C9H6O2", f"{_OH[:-1]}:1]>>O=C(C=Cc1ccc(O)cc1)[O:1]",
             _d("C9H6O2")),
    EditRule("decoumaroylation", "C9H6O2",
             _ESTER_RM.format(acyl="[CX3](=O)-[CH1]=[CH1]-c1[cH][cH]c(-[OH1])[cH][cH]1"),
             _d("C9H6O2", -1)),
    EditRule("caffeoylation", "C9H6O3", f"{_OH[:-1]}:1]>>O=C(C=Cc1ccc(O)c(O)c1)[O:1]",
             _d("C9H6O3")),
    EditRule("decaffeoylation", "C9H6O3",
             _ESTER_RM.format(acyl="[CX3](=O)-[CH1]=[CH1]-c1[cH][cH]c(-[OH1])c(-[OH1])[cH]1"),
             _d("C9H6O3", -1)),
    # Zero-delta moves (positional isomers): same group, another atom of the same class
    EditRule("move_OH_arom", "0", "([c:1]-[OX2H1].[cH:2])>>([c:1].O-[c:2])", _ZERO),
    EditRule("move_OH_aliph", "0",
             f"([CX4:1]-[OX2H1].{_CH_SP3[:-1]}:2])>>([C:1].O-[C:2])", _ZERO),
    EditRule("move_OMe_arom", "0", "([c:1]-[OX2]-[CH3].[cH:2])>>([c:1].C-O-[c:2])", _ZERO),
    EditRule("swap_OMe_OH", "0",
             f"([CH3]-[OX2:1]-[#6:3].{_OH[:-1]}:2]-[#6:4])>>([O:1]-[#6:3].C-[O:2]-[#6:4])",
             _ZERO),
    EditRule("move_methyl_arom", "0", "([c:1]-[CH3].[cH:2])>>([c:1].C-[c:2])", _ZERO),
    EditRule("move_prenyl_arom", "0",
             "([c:1]-[CH2]-[CH1]=[CX3](-[CH3])-[CH3].[cH:2])>>([c:1].CC(C)=CC-[c:2])", _ZERO),
    EditRule("move_hexosyl", "0",
             "([#6:20]-[OX2:1]-[CH1:3]1-[OX2:4]-[CH1:5](-[CH2:6]-[OH1:7])-[CH1:8](-[OH1:9])"
             f"-[CH1:10](-[OH1:11])-[CH1:12]1-[OH1:13].{_OH[:-1]}:2])"
             ">>([#6:20]-[O:1].[O:2]-[C:3]1-[O:4]-[C:5](-[C:6]-[O:7])-[C:8](-[O:9])"
             "-[C:10](-[O:11])-[C:12]1-[O:13])", _ZERO),
    EditRule("shift_double_bond", "0", "[C:1]=[C:2]-[CX4;!H0:3]>>[C:1]-[C:2]=[C:3]", _ZERO),
)

RULE_IDS: dict[str, int] = {r.id: i for i, r in enumerate(RULES)}
assert len(RULE_IDS) == len(RULES), "duplicate rule id"

GEN_COLUMNS: tuple[str, ...] = (
    "smiles", "ik14", "formula", "mono_mass", "target_idx", "seed_idx", "seed_rank",
    "seed_score", "n_edits", "edits", "edit1", "edit2", "n_seeds", "n_paths",
)


def get_rule(rule_id: str) -> EditRule:
    return RULES[RULE_IDS[rule_id]]


def rules_table() -> pd.DataFrame:
    """The edit library as a table: code, id, family, delta (text), SMARTS."""
    return pd.DataFrame({"code": list(range(len(RULES))), "id": [r.id for r in RULES],
                         "family": [r.family for r in RULES],
                         "delta": [r.delta_formula for r in RULES],
                         "smarts": [r.smarts for r in RULES]})


@lru_cache(maxsize=None)
def _reaction(smarts: str) -> rdChemReactions.ChemicalReaction:
    with rdBase.BlockLogs():
        rxn = rdChemReactions.ReactionFromSmarts(smarts)
        rxn.Initialize()
    if rxn.GetNumReactantTemplates() != 1 or rxn.GetNumProductTemplates() != 1:
        raise ValueError(f"edit SMARTS must have one reactant and one product: {smarts}")
    return rxn


# --------------------------------------------------------------------------------------
# Molecules
# --------------------------------------------------------------------------------------

def _smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def prepare_seed(m: str | Chem.Mol | None) -> Chem.Mol | None:
    """Standardised seed (``chem.std_mol``: cleanup, uncharge, no stereo), re-parsed from
    its SMILES so that hydrogens are implicit wherever possible. None if it fails."""
    mol = std_mol(m)
    if mol is None:
        return None
    with rdBase.BlockLogs():
        try:
            out = mol_from_smiles(_smiles(mol))
        except Exception:  # noqa: BLE001
            return None
    return out


def mol_counts(mol: Chem.Mol) -> tuple[int, ...] | None:
    """Element counts (``ELEMENTS`` order, implicit H included) or None if the molecule
    has an element outside the alphabet or a net charge."""
    if Chem.GetFormalCharge(mol) != 0:
        return None
    c = formula_counts(rdMolDescriptors.CalcMolFormula(mol))
    if c is None:
        return None
    return tuple(int(v) for v in c)


def _fix_hydrogens(prod: Chem.Mol, reactant: Chem.Mol) -> None:
    """Explicit-H bookkeeping RDKit leaves to the caller: an aromatic N that lost a
    neighbour gets an explicit H (``n-CH3 -> [nH]``); an atom with explicit Hs that gained
    neighbours loses as many."""
    for a in prod.GetAtoms():
        if not a.HasProp("react_atom_idx"):
            continue
        ra = reactant.GetAtomWithIdx(a.GetIntProp("react_atom_idx"))
        d_old, d_new = ra.GetDegree(), a.GetDegree()
        if d_new > d_old and a.GetNumExplicitHs() > 0:
            a.SetNumExplicitHs(max(0, a.GetNumExplicitHs() - (d_new - d_old)))
        elif (d_new < d_old and a.GetIsAromatic() and a.GetAtomicNum() == 7
              and a.GetFormalCharge() == 0):
            a.SetNumExplicitHs(a.GetNumExplicitHs() + (d_old - d_new))


def _sanitize(p: Chem.Mol, reactant: Chem.Mol) -> Chem.Mol | None:
    """Sanitised reaction product, or None. If plain sanitising fails, retry on a pristine
    copy with explicit Hs fixed (a failed attempt leaves the molecule half-modified)."""
    raw = Chem.Mol(p)
    if Chem.SanitizeMol(p, catchErrors=True) == Chem.SanitizeFlags.SANITIZE_NONE:
        return p
    _fix_hydrogens(raw, reactant)
    try:
        Chem.SanitizeMol(raw)
    except Exception:  # noqa: BLE001 - invalid product
        return None
    return raw


def _products(rule: EditRule, mol: Chem.Mol, self_smi: str | None = None,
              max_products: int = 1000) -> list[str]:
    """Sorted distinct canonical SMILES (no stereo) of sanitised single-fragment products
    of ``rule`` on ``mol``, excluding ``mol`` itself (``self_smi``: its SMILES)."""
    rxn = _reaction(rule.smarts)
    if self_smi is None:
        self_smi = _smiles(mol)
    found: set[str] = set()
    with rdBase.BlockLogs():
        try:
            outs = rxn.RunReactants((mol,), max_products)
        except Exception:  # noqa: BLE001
            return []
        for tup in outs:
            p = tup[0]
            try:
                q = _sanitize(p, mol)
                if q is None:
                    continue
                smi = _smiles(q)
            except Exception:  # noqa: BLE001 - invalid product: skip
                continue
            if "." in smi or smi == self_smi:
                continue
            found.add(smi)
    return sorted(found)


def apply_rule(rule: EditRule | str, mol: Chem.Mol | str, max_products: int = 1000
               ) -> list[tuple[str, Chem.Mol]]:
    """All distinct sanitised single-fragment products of ``rule`` on ``mol`` (every
    matching site), excluding ``mol`` itself, as ``(canonical SMILES without stereo,
    Mol)`` sorted by SMILES. Mols are fresh parses of the SMILES. A SMILES input is
    prepared with ``prepare_seed``."""
    r = get_rule(rule) if isinstance(rule, str) else rule
    m = prepare_seed(mol) if isinstance(mol, str) else mol
    if m is None:
        return []
    out = []
    for smi in _products(r, m, None, max_products):
        pm = mol_from_smiles(smi)
        if pm is not None:
            out.append((smi, pm))
    return out


# --------------------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _by_delta() -> dict[tuple[int, ...], tuple[int, ...]]:
    acc: dict[tuple[int, ...], list[int]] = {}
    for i, r in enumerate(RULES):
        acc.setdefault(r.delta, []).append(i)
    return {k: tuple(v) for k, v in acc.items()}


def _paths(need: Sequence[tuple[int, ...]], max_edits: int, inverse_pairs: bool = False,
           move_second: bool = False
           ) -> tuple[dict[int, set[int]], dict[int, dict[int, set[int]]]]:
    """For needed deltas (one per target): single-edit rules -> targets, and first-edit
    rules -> {second-edit rule -> targets}. Unless ``inverse_pairs``, two edits whose
    deltas cancel are skipped (a same-formula target then gets single moves only); unless
    ``move_second``, zero-delta moves are only used as the first of two edits (a move
    after an edit mostly repeats the edit at another site)."""
    by = _by_delta()
    one: dict[int, set[int]] = {}
    two: dict[int, dict[int, set[int]]] = {}
    for t, D in enumerate(need):
        if max_edits >= 1:
            for r in by.get(D, ()):
                one.setdefault(r, set()).add(t)
        if max_edits >= 2 and (inverse_pairs or any(D)):
            for r1, rule in enumerate(RULES):
                rest = tuple(a - b for a, b in zip(D, rule.delta))
                for r2 in by.get(rest, ()):
                    if not move_second and not any(RULES[r2].delta):
                        continue
                    two.setdefault(r1, {}).setdefault(r2, set()).add(t)
    return one, two


def _as_seed_list(seeds: Any) -> list[Any]:
    if isinstance(seeds, pd.DataFrame):
        return list(seeds["smiles"])
    if isinstance(seeds, (str, Chem.Mol)):
        return [seeds]
    return list(seeds)


def _empty() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype=object) for c in GEN_COLUMNS})
    for c in ("target_idx", "seed_idx", "seed_rank", "n_edits", "edit1", "edit2",
              "n_seeds", "n_paths"):
        df[c] = df[c].astype(np.int64)
    for c in ("mono_mass", "seed_score"):
        df[c] = df[c].astype(np.float64)
    return df


def generate(seeds: Sequence[str | Chem.Mol] | pd.DataFrame,
             target_formulas: Sequence[str] | str, cfg: Config | None = None,
             exclude_ik14: Iterable[str] = frozenset(), *,
             seed_scores: Sequence[float] | None = None, max_edits: int | None = None,
             max_products: int | None = None) -> pd.DataFrame:
    """Analogs of ``seeds`` with exactly one of ``target_formulas`` (module docstring).

    ``seeds``: SMILES or Mols (or a frame with ``smiles`` and optional ``score``);
    ``seed_scores``: higher is better, default the input order. ``exclude_ik14``: ik14s
    already in the candidate pool. Limits default to ``cfg.gen_max_edits`` and
    ``cfg.gen_max_products``. Returns ``GEN_COLUMNS`` (``seed_idx`` is the input
    position, ``target_idx`` the position in ``target_formulas``, ``edit1`` / ``edit2``
    rule codes or -1).
    """
    cfg = cfg or Config()
    max_edits = int(cfg.gen_max_edits if max_edits is None else max_edits)
    cap = int(cfg.gen_max_products if max_products is None else max_products)
    if isinstance(target_formulas, str):
        target_formulas = [target_formulas]
    targets: list[tuple[int, ...] | None] = []
    for f in target_formulas:
        c = formula_counts(f)
        targets.append(tuple(int(v) for v in c) if c is not None else None)
    seed_list = _as_seed_list(seeds)
    if seed_scores is None and isinstance(seeds, pd.DataFrame) and "score" in seeds.columns:
        seed_scores = list(seeds["score"])
    scores = (np.zeros(len(seed_list)) if seed_scores is None
              else np.asarray(seed_scores, dtype=np.float64).reshape(-1))
    if scores.shape[0] != len(seed_list):
        raise ValueError("generate: seeds and seed_scores differ in length")
    if not seed_list or not any(t is not None for t in targets) or cap <= 0:
        return _empty()
    order = sorted(range(len(seed_list)),
                   key=lambda i: (-scores[i] if np.isfinite(scores[i]) else np.inf, i))
    excluded = set(exclude_ik14)

    prepared: list[tuple[int, int, Chem.Mol, tuple[int, ...]]] = []
    for rank, i in enumerate(order):
        mol = prepare_seed(seed_list[i])
        if mol is None or Chem.GetFormalCharge(mol) != 0 or len(Chem.GetMolFrags(mol)) != 1:
            continue
        counts = mol_counts(mol)
        if counts is None:
            continue
        prepared.append((rank, i, mol, counts))
        k = _ik14(mol)
        if k:
            excluded.add(k)

    rows: dict[str, dict[str, Any]] = {}      # smiles -> first row (in priority order)
    smi_ik: dict[str, str | None] = {}
    seeds_per_ik: dict[str, set[int]] = {}
    paths_per_ik: dict[str, int] = {}
    kept: list[str] = []                      # ik14s in priority order
    kept_set: set[str] = set()
    memo: dict[tuple[int, str], list[str]] = {}
    budget = int(cfg.gen_max_raw_per_seed)
    for rank, i, mol, counts in prepared:
        need = [tuple(a - b for a, b in zip(t, counts)) if t is not None else None
                for t in targets]
        need_ok = [d if d is not None else (10 ** 6,) * len(ELEMENTS) for d in need]
        one, two = _paths(need_ok, max_edits, bool(cfg.gen_inverse_pairs),
                          bool(cfg.gen_move_second))
        if not one and not two:
            continue
        found: list[tuple[int, int, int, int, str]] = []   # n_edits, e1, e2, target, smiles
        seed_smi = _smiles(mol)
        first: dict[int, list[str]] = {}
        for r1 in sorted(set(one) | set(two)):
            first[r1] = _products(RULES[r1], mol, seed_smi)
            for t in sorted(one.get(r1, ())):
                found.extend((1, r1, -1, t, smi) for smi in first[r1])
        # Two-edit paths: both edits with a non-zero delta first, then paths that start
        # with a zero-delta move; stop once the per-seed raw budget is spent.
        raw = 0
        order1 = sorted(two, key=lambda r: (not any(RULES[r].delta), r))
        inter: dict[str, Chem.Mol | None] = {}
        for r1 in order1:
            if raw >= budget:
                break
            for smi in first[r1]:
                if raw >= budget:
                    break
                if smi not in inter:
                    inter[smi] = mol_from_smiles(smi)
                pm = inter[smi]
                if pm is None:
                    continue
                for r2 in sorted(two[r1]):
                    key2 = (r2, smi)
                    if key2 not in memo:
                        memo[key2] = _products(RULES[r2], pm, smi)
                    raw += len(memo[key2])
                    for smi2 in memo[key2]:
                        for t in sorted(two[r1][r2]):
                            found.append((2, r1, r2, t, smi2))
        if raw >= budget:
            log.debug("generate: seed %d hit the raw two-edit budget (%d)", i, budget)
        found.sort()
        new_here: list[str] = []
        for n_ed, e1, e2, t, smi in found:
            if smi not in smi_ik:
                pm = mol_from_smiles(smi)
                ok = pm is not None and mol_counts(pm) == targets[t]
                smi_ik[smi] = _ik14(pm) if ok else None
            key = smi_ik[smi]
            if key is None or key in excluded:
                continue
            seeds_per_ik.setdefault(key, set()).add(i)
            paths_per_ik[key] = paths_per_ik.get(key, 0) + 1
            if key in kept_set:
                continue
            kept_set.add(key)
            new_here.append(key)
            rows[key] = {"smiles": smi, "ik14": key, "formula": target_formulas[t],
                         "target_idx": t, "seed_idx": i, "seed_rank": rank,
                         "seed_score": float(scores[i]), "n_edits": n_ed,
                         "edits": RULES[e1].id + ("+" + RULES[e2].id if e2 >= 0 else ""),
                         "edit1": e1, "edit2": e2}
        kept.extend(new_here)
        if len(kept) >= cap:
            break
    if not kept:
        return _empty()
    kept = kept[:cap]
    out = pd.DataFrame([rows[k] for k in kept])
    out["formula"] = [hill(targets[t]) for t in out["target_idx"]]
    out["mono_mass"] = [float(counts_mass(np.asarray(targets[t]))) for t in out["target_idx"]]
    out["n_seeds"] = [len(seeds_per_ik[k]) for k in kept]
    out["n_paths"] = [paths_per_ik[k] for k in kept]
    out = out[list(GEN_COLUMNS)]
    for c in ("target_idx", "seed_idx", "seed_rank", "n_edits", "edit1", "edit2",
              "n_seeds", "n_paths"):
        out[c] = out[c].astype(np.int64)
    return out.reset_index(drop=True)


def seed_formulas(target: str, max_edits: int = 2) -> pd.DataFrame:
    """Seed formulas that reach ``target`` with at most ``max_edits`` rules: columns
    ``formula``, ``mono_mass`` (where to query the DB for seeds, PLAN.md 4.7 (b)),
    ``n_edits`` (fewest edits) and ``families`` (delta families of one shortest path).
    Sorted by n_edits then formula. The target itself (zero-delta moves) is included.
    TODO(WP12, NEEDS_TRAIN): rank DB/LIB candidates at these masses by fingerprint score
    and pass the top ``cfg.gen_n_db_seeds`` to ``generate``.
    """
    tc = formula_counts(target)
    if tc is None:
        raise ValueError(f"not a formula: {target!r}")
    t = tuple(int(v) for v in tc)
    best: dict[tuple[int, ...], tuple[int, str]] = {}
    frontier: list[tuple[tuple[int, ...], str]] = [(t, "")]
    for depth in range(1, max_edits + 1):
        nxt = []
        for f, fam in frontier:
            for r in RULES:
                s = tuple(a - b for a, b in zip(f, r.delta))
                if min(s) < 0 or s[0] < 1:
                    continue
                label = r.family if not fam else f"{r.family}+{fam}"
                if s not in best or best[s][0] > depth:
                    best[s] = (depth, label)
                    nxt.append((s, label))
        frontier = nxt
    rows = [{"formula": hill(s), "mono_mass": float(counts_mass(np.asarray(s))),
             "n_edits": d, "families": fam} for s, (d, fam) in best.items()]
    if not rows:
        return pd.DataFrame(columns=["formula", "mono_mass", "n_edits", "families"])
    return (pd.DataFrame(rows).sort_values(["n_edits", "formula"], kind="mergesort")
            .reset_index(drop=True))

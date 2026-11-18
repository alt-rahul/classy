# CASMI 2026 (Kaggle) - competition reference

Copied from the competition page for agents working in this repo.

## Task

Predict 2D chemical structures (SMILES) from LC-MS/MS spectra, for both known and novel molecules. CASMI 2026 focuses on molecules resembling those in natural samples from plants, mammals or microbes: confirmed natural products, hypothesised natural products, natural-product analogs, and synthetic molecules that might plausibly occur in nature. A molecule may have been measured several times at different collision energies or as different adducts. Predictions are per molecule, not per spectrum: aggregate all of a molecule's spectra into one ranked list.

## Background (MS/MS)

- Ionization: molecule gains/loses charged species (adduct), e.g. [M+H]+, [M+NH4]+, [M-H]-.
- Precursor m/z is measured with high accuracy and strongly constrains the molecular formula (z is usually +/-1).
- Fragmentation: precursor ions collide with neutral gas at a given collision energy; fragment m/z and intensities are recorded. Base peak = most intense fragment. Precursor peak may or may not be present.

## Evaluation: MRR@25

MRR@25 = (1/N) * sum_i 1/rank_i, where rank_i is the position of the first correct structure in the ranked list for molecule i (0 if not in top 25). Each molecule has exactly one correct structure.

Matching: both prediction and answer are passed through RDKit tautomer canonicalization (RDKit pinned at 2026.03.3) and reduced to the first block of the InChIKey (InChIKey14). Stereo and tautomer form are not penalised.

Metric notebook: https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank

## Submission file

```
molecule_id,smiles
m_0014ef,CC1=CC(=O)C=CC1=O;OC(=O)c1ccccc1O;CN1C=NC2=C1C(=O)N(C)C(=O)N2C
```

Up to 25 SMILES per molecule, best first, semicolon-joined. Every molecule_id exactly once. Rejected if: missing columns, empty, nulls, repeated molecule_id, or >25 guesses for any molecule. Fewer than 25 is allowed; wrong guesses cost nothing beyond rank occupancy.

This is a code competition: the public test.parquet is made of training examples and is replaced by the hidden test set on re-run (approximately the same size). The final pipeline must therefore run end-to-end on a new test.parquet, presumably offline on Kaggle, using precomputed artefacts attached as datasets.

## Files

- train.parquet - ~2.5M MS/MS spectra covering ~275k unique structures, with labels. One row per spectrum.
- test.parquet - spectra to identify. One row per spectrum, no structure.
- sample_submission.csv - valid format example.

## Hidden test set

~1,500 spectra of ~400 molecules, 1-16 spectra per molecule (median 3), all Bruker timsTOF. Monoisotopic masses 157 to 1,159 Da (median 348).

Novelty classes (distribution and assignment hidden):

| Class | Definition |
|---|---|
| 1 - in public spectral libraries | Structure has public reference MS/MS spectra; may be found by similarity to training spectra. |
| 2 - known structure, no public spectra | Structure is in PubChem or COCONUT; reachable by database retrieval. |
| 3 - novel structure | Not in PubChem at all; must be predicted de novo. |

Minimal cleaning applied to test spectra:
- Spectra whose precursor mass was inconsistent with structure+adduct were dropped.
- Spectra with base peak < 1,000 raw counts dropped.
- Peaks above precursor + 2 Da removed.

Test adducts (ten): [M+H]+, [M+NH4]+, [M-H2O+H]+, [M-2H2O+H]+, [M+Na]+, [M+K]+, [M-H]-, [M-H2O-H]-, [M+CH2O2-H]-, [M+Cl]-. Training set contains many more.

## Columns

Peak lists are two aligned arrays; intensities normalised so the largest peak = 1.0.

test.parquet (12 columns):

| Column | Description |
|---|---|
| molecule_id | Anonymous compound id; predictions per molecule_id |
| spectrum_id | Unique spectrum id |
| ms2_mzs | Fragment m/z array |
| ms2_normalized_intensities | Aligned intensities, base peak = 1.0 |
| base_peak_intensity | Raw intensity of largest peak before normalisation (higher usually = less noise) |
| adduct | Precursor adduct |
| ionization_mode | positive / negative |
| instrument_type | Always timsTOF in test |
| precursor_mz | Measured precursor m/z |
| collision_energy_ev | List of eV; single value = one acquisition, several = merged across energies |
| collision_energy_orig | As originally recorded |
| collision_energy_orig_units | Always eV in test |

train.parquet (18 columns) = the above plus:

| Column | Description |
|---|---|
| normalized_smiles | Label: RDKit-standardised SMILES |
| inchikey / inchikey14 | InChIKey and its first block |
| molecular_formula | Formula of neutral structure |
| ingest_lib | Source library |
| adduct_orig | Adduct string as recorded by source |
| precursor_error_ppm | ppm error between measured precursor m/z and the value implied by label+adduct; uncleaned, doubles as label-quality signal |
| num_peaks | Number of fragment peaks |

instrument_type in train is free text (null when missing); base_peak_intensity is null for libraries that shipped pre-normalised intensities. Only enveda-180 and enveda-np-examples are uniformly timsTOF, matching test.

## Training libraries

| ingest_lib | Spectra | Unique structures | Notes |
|---|---|---|---|
| enveda-180 | 1,153,785 | 182,941 | Enveda, same Bruker timsTOF as test. Synthetic drug-like screening compounds (different chemical space from test). |
| pluskal_ms2 | 527,581 | 46,821 | MSnLib (Orbitrap), commercial screening/bioactive libs, multiple CEs. NCE units. |
| riken | 347,171 | 15,892 | RIKEN, plant specialised metabolites. |
| gnps | 220,849 | 45,750 | GNPS community libraries, largest NP collection, heterogeneous. |
| massbank | 101,727 | 9,180 | MassBank, curated consortium. |
| mona | 92,416 | 11,681 | MassBank of North America. |
| spectraverse | 50,933 | 9,631 | Recent harmonised aggregation incl. obscure libs. |
| msdial | 40,765 | 9,127 | MS-DIAL collections, multi-instrument. |
| drug_plus | 2,545 | 2,539 | Pharma LC-MS/MS DB, ~1 spectrum each, no CE metadata. |
| enveda-np-examples | 1,151 | 250 | Closest to test: 250 common NPs on same instruments + pipeline as test. Also appear in other libs. |
| masaryk | 652 | 416 | RECETOX standards. |

Compounds overlap across libraries (exact duplicate spectra removed). Any freely and publicly available external data may be used.

## Collision energy

collision_energy_orig (lossless string), collision_energy_orig_units (eV, NCE, V, unknown), collision_energy_ev (best-effort eV; NCE converted as NCE * precursor_mz / 500 * charge factor; null if unresolved). Use collision_energy_ev to line up with test.

## Common curation methods (optional)

Relative-intensity floor (2%, 1%, 0.1%); drop peaks above precursor (+1-2 Da); min peak count ~5-6; keep top-N (e.g. 128) or top ~6 per 50 Da window; deisotoping (13C +1.0033 Da); sqrt/log intensity transforms.

## Other resources

- MIST-CF, SIRIUS: molecular formula annotation from MS/MS.
- matchms: spectra cleaning, curation, similarity.
- GNPS propagated "suspect" annotations (excluded from train; inferred labels).
- MassIVE: raw unannotated spectra for self-supervised pretraining.
- GeMS / DreaMS (https://zenodo.org/records/10997887): curated unannotated corpus + pretrained spectrum-embedding models.
- FragHub (https://github.com/eMetaboHUB/FragHub): merged open MS/MS libraries, may contain extra spectra.

License: CC BY-NC 4.0.

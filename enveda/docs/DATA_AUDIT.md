# Data audit (WP6)

Generated 2026-11-20 19:27 by `scripts/ingest_train.py --stage audit` from `/home/user/classy/enveda/data/raw/train.parquet` (2,539,608 rows, 21 row groups). Every number below is recomputed by that command; PLAN.md section numbers in brackets.

## 1. Rows and structures per library (WP6 acceptance)

| library | rows in | expected spectra | rows in vs expected % | dropped | kept | kept % of rows in | structures kept | expected structures | structures vs expected % |
|---|---|---|---|---|---|---|---|---|---|
| drug_plus | 2,545 | 2,545 | 0.00 | 0 | 2,545 | 100.00 | 2,539 | 2,539 | 0.00 |
| enveda-180 | 1,153,785 | 1,153,785 | 0.00 | 0 | 1,153,785 | 100.00 | 182,941 | 182,941 | 0.00 |
| enveda-np-examples | 1,184 | 1,151 | 2.87 | 0 | 1,184 | 100.00 | 250 | 250 | 0.00 |
| gnps | 220,849 | 220,849 | 0.00 | 595 | 220,254 | 99.73 | 45,729 | 45,750 | -0.05 |
| masaryk | 652 | 652 | 0.00 | 0 | 652 | 100.00 | 416 | 416 | 0.00 |
| massbank | 101,727 | 101,727 | 0.00 | 2,526 | 99,201 | 97.52 | 9,157 | 9,180 | -0.25 |
| mona | 92,416 | 92,416 | 0.00 | 257 | 92,159 | 99.72 | 11,679 | 11,681 | -0.02 |
| msdial | 40,765 | 40,765 | 0.00 | 709 | 40,056 | 98.26 | 9,097 | 9,127 | -0.33 |
| pluskal_ms2 | 527,581 | 527,581 | 0.00 | 442 | 527,139 | 99.92 | 46,816 | 46,821 | -0.01 |
| riken | 347,171 | 347,171 | 0.00 | 29,911 | 317,260 | 91.38 | 15,851 | 15,892 | -0.26 |
| spectraverse | 50,933 | 50,933 | 0.00 | 51 | 50,882 | 99.90 | 9,630 | 9,631 | -0.01 |

Rows read within 1 percent of COMPETITION.md for every library: **False** (outside: enveda-np-examples 1,184 vs 1,151, +2.87 percent). Total kept 2,505,117 of 2,539,608 (98.64 percent); every dropped row has a reason below. Structures are counted after the drops.

enveda-np-examples: COMPETITION.md says 1,151 spectra but the file has more; its structure count is checked in the table (expected 250). The extra rows are kept: they are the NP-TIMS query pool.

Drop reasons (first failing check, order as in the ingest docstring):

| library | null_label | bad_precursor | bad_peak_arrays | no_raw_peaks | no_polarity | duplicate_spectrum_id | empty_after_cleaning |
|---|---|---|---|---|---|---|---|
| drug_plus | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| enveda-180 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| enveda-np-examples | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| gnps | 0 | 0 | 0 | 0 | 0 | 0 | 595 |
| masaryk | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| massbank | 0 | 0 | 0 | 0 | 0 | 0 | 2,526 |
| mona | 0 | 0 | 0 | 0 | 0 | 0 | 257 |
| msdial | 0 | 0 | 0 | 0 | 0 | 0 | 709 |
| pluskal_ms2 | 0 | 0 | 0 | 0 | 0 | 0 | 442 |
| riken | 0 | 0 | 0 | 0 | 0 | 0 | 29,911 |
| spectraverse | 0 | 0 | 0 | 0 | 0 | 0 | 51 |

`empty_after_cleaning` by cause (`_empty_kind`: precursor_window_only = every raw peak within [precursor - 0.5, precursor + 2] Da, i.e. no fragment; nothing_below_precursor = no peak below that window but some above it; below_floor_or_other = fragments existed but none survived):

| library | precursor_window_only | nothing_below_precursor | below_floor_or_other |
|---|---|---|---|
| gnps | 470 | 125 | 0 |
| massbank | 2,357 | 169 | 0 |
| mona | 146 | 111 | 0 |
| msdial | 642 | 67 | 0 |
| pluskal_ms2 | 441 | 1 | 0 |
| riken | 10,720 | 19,191 | 0 |
| spectraverse | 44 | 7 | 0 |

Flags on kept rows (not dropped):

| library | abs_ppm_gt_20 | adduct_unparsed | bpi_null | ce_unresolved | peaks_above_precursor |
|---|---|---|---|---|---|
| drug_plus | 89 | 0 | 2,545 | 2,545 | 79 |
| enveda-180 | 0 | 0 | 0 | 0 | 282,613 |
| enveda-np-examples | 1 | 0 | 0 | 0 | 0 |
| gnps | 16,085 | 0 | 0 | 220,254 | 67,936 |
| masaryk | 0 | 0 | 0 | 652 | 0 |
| massbank | 6,931 | 8 | 99,201 | 1,904 | 4,222 |
| mona | 5,339 | 2 | 92,159 | 15,681 | 39,046 |
| msdial | 81 | 0 | 40,056 | 16,884 | 3,178 |
| pluskal_ms2 | 50,790 | 0 | 527,139 | 0 | 4,258 |
| riken | 35,302 | 5 | 317,260 | 31,376 | 88,493 |
| spectraverse | 5 | 0 | 50,882 | 47,026 | 6,384 |

## 2. Adducts per library [3.1]

| adduct | drug_plus | enveda-180 | enveda-np-examples | gnps | masaryk | massbank | mona | msdial | pluskal_ms2 | riken | spectraverse |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [M+H]+ | 2,012 | 588,507 | 569 | 134,706 | 652 | 63,117 | 46,105 | 15,172 | 288,485 | 155,808 | 14,099 |
| [M+NH4]+ | 71 | 4,702 | 98 | 6,898 | 0 | 317 | 6,576 | 2,051 | 8,644 | 7,281 | 16,810 |
| [M-H2O+H]+ | 0 | 0 | 24 | 6,096 | 0 | 329 | 1,005 | 1,105 | 11,401 | 1,681 | 0 |
| [M-2H2O+H]+ | 0 | 0 | 8 | 1,467 | 0 | 63 | 87 | 60 | 1,367 | 375 | 0 |
| [M+Na]+ | 225 | 15,957 | 80 | 19,638 | 0 | 759 | 8,901 | 3,247 | 5,291 | 105,972 | 1,575 |
| [M+K]+ | 164 | 1,084 | 16 | 6,326 | 0 | 19 | 4,976 | 186 | 0 | 5,422 | 728 |
| [M-H]- | 0 | 184,747 | 276 | 32,323 | 0 | 29,278 | 22,121 | 14,127 | 133,753 | 34,124 | 12,655 |
| [M-H2O-H]- | 0 | 0 | 1 | 39 | 0 | 8 | 41 | 110 | 0 | 86 | 0 |
| [M+CH2O2-H]- | 0 | 24,556 | 105 | 3,320 | 0 | 1,086 | 700 | 257 | 50,785 | 1,548 | 813 |
| [M+Cl]- | 0 | 2,407 | 7 | 1,416 | 0 | 42 | 102 | 167 | 22,987 | 219 | 126 |
| other parsed | 73 | 331,825 | 0 | 8,025 | 0 | 4,175 | 1,543 | 3,574 | 4,426 | 4,739 | 4,076 |
| unparsed | 0 | 0 | 0 | 0 | 0 | 8 | 2 | 0 | 0 | 5 | 0 |

Most frequent adduct strings outside the ten test adducts (spelling as in `adduct`):

| adduct | parsed | spectra |
|---|---|---|
| [2M+Na]+ | yes | 190,033 |
| [2M+H]+ | yes | 92,041 |
| [2M-H]- | yes | 56,172 |
| [M]+ | yes | 8,580 |
| [M+C2H4O2-H]- | yes | 7,436 |
| [M+2H]2+ | yes | 1,714 |
| [2M+Na-2H]- | yes | 1,431 |
| [2M+CH2O2-H]- | yes | 665 |
| [M-H2O]+ | yes | 624 |
| [M-CH3]- | yes | 617 |
| [M-2H]- | yes | 445 |
| [M]- | yes | 444 |
| [2M+NH4]+ | yes | 378 |
| [M-H5O3]+ | yes | 230 |
| [2M+C2H4O2-H]- | yes | 187 |
| [M+C2H4N]+ | yes | 158 |
| [M+3H]3+ | yes | 138 |
| [M+2Na-H]+ | yes | 110 |
| [2M+K]+ | yes | 109 |
| [M-H2]2- | yes | 103 |
| [M+Ca]2+ | yes | 85 |
| [M-C3H6NO2]- | yes | 72 |
| [M+CH5O]+ | yes | 67 |
| [M-H2N]+ | yes | 60 |
| [M+Br]- | yes | 45 |

## 3. Instrument family mapping [4.4, `instrument_family_of`]

Rules: Enveda libraries -> timstof; text with `tims` -> timstof; Orbitrap/Exactive/Lumos/Fusion/Eclipse/Astral/Exploris/ITFT/QFT/FT names -> orbitrap; Q-TOF/TOF/maXis/impact/Synapt/Xevo/TripleTOF/Zeno/Agilent 65xx/compact -> qtof; else pluskal_ms2 -> orbitrap, otherwise other.

| family | drug_plus | enveda-180 | enveda-np-examples | gnps | masaryk | massbank | mona | msdial | pluskal_ms2 | riken | spectraverse |
|---|---|---|---|---|---|---|---|---|---|---|---|
| orbitrap | 0 | 0 | 0 | 149,661 | 355 | 41,032 | 54,338 | 5,710 | 527,139 | 146,657 | 15,330 |
| other | 2,545 | 0 | 0 | 9,002 | 297 | 8,751 | 5,895 | 25,916 | 0 | 33,272 | 9,566 |
| qtof | 0 | 0 | 0 | 61,591 | 0 | 49,418 | 31,926 | 8,430 | 0 | 137,331 | 25,986 |
| timstof | 0 | 1,153,785 | 1,184 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Free-text values with their family (top 60 of 83 distinct texts, by spectra):

| instrument_type | family | spectra | libraries |
|---|---|---|---|
| timsTOF | timstof | 1,154,969 | enveda-180;enveda-np-examples |
| Orbitrap | orbitrap | 736,604 | gnps;mona;msdial;pluskal_ms2;riken |
| QTOF | qtof | 106,355 | riken |
| LC-ESI-QTOF | qtof | 101,447 | gnps;massbank;mona;msdial;riken |
| ESI-QFT | orbitrap | 77,236 | gnps;mona;msdial;riken |
| LC-ESI-QFT | orbitrap | 54,562 | gnps;massbank;mona;msdial;riken |
| LC-ESI-ITFT | orbitrap | 44,502 | gnps;massbank;mona;msdial;riken |
| qTof | qtof | 33,666 | gnps;mona;msdial;riken |
| <null> | other | 32,777 | drug_plus;mona;msdial;riken |
| ToF | qtof | 26,125 | gnps |
| qtof | qtof | 26,067 | mona;msdial;riken;spectraverse |
| LC-ESI-QQ | other | 18,917 | gnps;massbank;mona;msdial;riken |
| orbitrap | orbitrap | 15,354 | mona;riken;spectraverse |
| Linear Ion Trap | other | 10,232 | gnps;mona;msdial;riken |
| Hybrid FT | orbitrap | 9,128 | gnps;mona;riken |
| Flow-injection QqQ/MS | other | 8,647 | gnps;msdial;riken |
| iontrap | other | 8,595 | spectraverse |
| Maxis II HD Q-TOF Bruker | qtof | 7,589 | gnps;mona;riken |
| ESI-QTOF | qtof | 5,592 | gnps;massbank;mona;msdial;riken |
| LC-ESI-QQQ | other | 3,096 | mona;msdial;riken |
| qToF | qtof | 2,852 | gnps;mona;riken |
| Ion Trap | other | 1,951 | gnps;mona;riken |
| LC-ESI-IT | other | 1,708 | massbank;mona;msdial;riken |
| QQQ | other | 1,281 | gnps;mona |
| QIT | other | 1,231 | gnps;mona;msdial;riken |
| ESI-TOF | qtof | 1,215 | massbank;msdial;riken |
| LC-ESI-TOF | qtof | 1,192 | massbank;msdial;riken |
| LC-Q-TOF/MS | qtof | 996 | gnps;mona;msdial;riken |
| nan | other | 972 | drug_plus;spectraverse |
| LC-APPI-QQ | other | 954 | massbank;mona;msdial;riken |
| QqQ | other | 728 | gnps;msdial;riken |
| Quattro_QQQ:25eV | other | 720 | gnps |
| Quattro_QQQ:40eV | other | 714 | gnps |
| Quattro_QQQ:10eV | other | 700 | gnps |
| LC-ESI-QIT | other | 614 | massbank;mona;riken |
| LC-ESI-Orbitrap | orbitrap | 531 | mona |
| Q-Exactive Plus | orbitrap | 401 | gnps;mona;riken |
| LC-QTOF | qtof | 388 | msdial;riken |
| N/A | other | 376 | gnps |
| LC-Orbitrap MS | orbitrap | 347 | masaryk |
| FAB-EBEB | other | 340 | gnps;msdial;riken |
|  | other | 297 | masaryk |
| Q-Exactive Plus Orbitrap Res 70k | orbitrap | 277 | gnps;mona;riken |
| impact HD | qtof | 273 | gnps |
| APCI-ITFT | orbitrap | 271 | massbank;mona;msdial;riken |
| Maxis HD qTOF | qtof | 252 | gnps;mona;riken |
| LC-ESI-QEHF | other | 238 | mona |
| Q-TOF | qtof | 202 | gnps;mona;msdial;riken |
| Q-Exactive Plus Orbitrap Res 14k | orbitrap | 200 | gnps;mona;riken |
| qTOF | qtof | 161 | gnps;mona;riken |
| LC-APCI-ITFT | orbitrap | 151 | massbank;mona;msdial;riken |
| LC-ESI-ITTOF | qtof | 146 | gnps;massbank;mona;msdial;riken |
| LC-ESI-FT | orbitrap | 143 | massbank |
| ESI-ITFT | orbitrap | 136 | massbank;mona;msdial;riken |
| LC-APCI-QFT | orbitrap | 130 | massbank |
| LC-ESI-Q-Orbitrap | orbitrap | 111 | mona |
| ESI-FT | orbitrap | 110 | mona |
| GC-APCI-QTOF | qtof | 65 | msdial;riken |
| LC-ESI-Q | other | 56 | massbank;riken |
| MALDI-TOFTOF | qtof | 46 | msdial;riken |

## 4. precursor_error_ppm per library [3.3, risk 10]

| library | spectra | NaN | median (signed) | abs p50 | abs p90 | abs p99 | abs p99.5 | abs max | abs > 20 | abs M err p99.5 (recomputed) |
|---|---|---|---|---|---|---|---|---|---|---|
| drug_plus | 2,545 | 0 | 1.73 | 1.73 | 3.88 | 76,305.77 | 129,227.20 | 499,999.61 | 89 | 129,766.45 |
| enveda-180 | 1,153,785 | 0 | 0.98 | 0.98 | 2.76 | 4.40 | 4.93 | 7.09 | 0 | 4.61 |
| enveda-np-examples | 1,184 | 0 | 1.54 | 1.65 | 3.19 | 6.62 | 7.81 | 21.06 | 1 | 8.47 |
| gnps | 220,254 | 91 | 1.68 | 1.68 | 6.61 | 76,302.33 | 390,961.35 | 3,268,325.33 | 16,085 | 436,570.72 |
| masaryk | 652 | 0 | 1.48 | 1.48 | 2.82 | 3.97 | 4.34 | 15.02 | 0 | 4.08 |
| massbank | 99,201 | 3,107 | 1.95 | 1.95 | 4.97 | 11,866.43 | 126,375.04 | 2,380,470.45 | 6,931 | 178,301.96 |
| mona | 92,159 | 745 | 1.80 | 1.80 | 4.69 | 54,268.24 | 146,812.84 | 2,152,332.81 | 5,339 | 192,048.66 |
| msdial | 40,056 | 0 | 1.47 | 1.47 | 3.38 | 5.71 | 7.03 | 571,576.51 | 81 | 2.04 |
| pluskal_ms2 | 527,139 | 215 | 1.66 | 1.66 | 3.92 | 2,868.54 | 3,023.36 | 5,563.52 | 50,790 | 0.28 |
| riken | 317,260 | 61 | 3.59 | 3.59 | 22.03 | 112,426.63 | 212,607.77 | 2,590,672.22 | 35,302 | 254,673.64 |
| spectraverse | 50,882 | 0 | 0.65 | 0.65 | 1.64 | 4.88 | 6.25 | 7,344.61 | 5 | 6.23 |

Columns `abs ...` are quantiles of the absolute given error. `abs M err` is recomputed here: neutral mass from precursor m/z and the parsed adduct versus the formula mass of `molecular_formula`, in ppm of M (the quantity the retrieval window applies to).

Given and recomputed errors disagree: 52,460 kept spectra have |given precursor_error_ppm| > 20 but |recomputed M error| <= 5 ppm (the label and precursor agree; the given column is off). Largest groups:

| library | adduct | spectra | given_median | recomputed_median |
|---|---|---|---|---|
| pluskal_ms2 | [M+CH2O2-H]- | 50,785 | 2,414.52 | 0.01 |
| gnps | [M+CH2O2-H]- | 1,675 | 2,050.44 | -0.11 |

A label-quality filter on the given `precursor_error_ppm` (WP9 `fpm_max_precursor_error_ppm`) would drop these good spectra; filter on `m_err_ppm` from meta.parquet / the spectra files instead.

timsTOF libraries (the test instrument), test adducts only, recomputed M error per spectrum (signed, ppm). Molecule-level numbers and the window are in section 11.

| library | adduct | spectra | p0.5 | p5 | median | p95 | p99.5 | abs p99 | abs max | abs > 5 | abs > 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| enveda-180 | [M+CH2O2-H]- | 24,556 | -4.37 | -1.50 | 0.35 | 2.07 | 4.30 | 4.34 | 5.78 | 106 | 0 |
| enveda-180 | [M+Cl]- | 2,407 | -4.82 | -3.13 | -0.15 | 2.35 | 5.01 | 4.85 | 5.48 | 19 | 0 |
| enveda-180 | [M+H]+ | 588,507 | -0.89 | 0.68 | 2.22 | 3.64 | 4.27 | 4.13 | 5.14 | 32 | 0 |
| enveda-180 | [M+K]+ | 1,084 | -5.47 | -4.44 | 0.23 | 4.46 | 5.52 | 5.48 | 5.60 | 55 | 0 |
| enveda-180 | [M+NH4]+ | 4,702 | -5.23 | -4.41 | 1.50 | 4.03 | 5.11 | 5.18 | 5.33 | 114 | 0 |
| enveda-180 | [M+Na]+ | 15,957 | -4.74 | -2.23 | 1.03 | 3.41 | 4.91 | 4.84 | 5.50 | 98 | 0 |
| enveda-180 | [M-H]- | 184,747 | -1.74 | -0.66 | 0.80 | 2.17 | 2.95 | 2.75 | 5.06 | 2 | 0 |
| enveda-np-examples | [M+CH2O2-H]- | 105 | -3.04 | -2.00 | 0.22 | 2.96 | 6.07 | 3.87 | 8.44 | 1 | 0 |
| enveda-np-examples | [M+Cl]- | 7 | -2.55 | -1.97 | -0.07 | 1.36 | 1.60 | 2.55 | 2.61 | 0 | 0 |
| enveda-np-examples | [M+H]+ | 569 | -2.33 | -0.64 | 2.01 | 3.78 | 8.74 | 6.59 | 8.77 | 14 | 0 |
| enveda-np-examples | [M+K]+ | 16 | -4.28 | -3.44 | 1.26 | 9.19 | 12.30 | 11.95 | 12.65 | 2 | 1 |
| enveda-np-examples | [M+NH4]+ | 98 | -3.15 | -0.46 | 1.25 | 3.54 | 3.80 | 3.81 | 4.10 | 0 | 0 |
| enveda-np-examples | [M+Na]+ | 80 | -5.94 | -0.78 | 1.67 | 4.56 | 6.06 | 6.46 | 7.24 | 4 | 0 |
| enveda-np-examples | [M-2H2O+H]+ | 8 | 0.72 | 0.74 | 1.58 | 3.29 | 3.31 | 3.31 | 3.31 | 0 | 0 |
| enveda-np-examples | [M-H2O+H]+ | 24 | -6.39 | -1.87 | 1.96 | 3.89 | 4.17 | 6.33 | 6.97 | 1 | 0 |
| enveda-np-examples | [M-H2O-H]- | 1 | -4.45 | -4.45 | -4.45 | -4.45 | -4.45 | 4.45 | 4.45 | 0 | 0 |
| enveda-np-examples | [M-H]- | 276 | -6.31 | -1.60 | 0.56 | 2.11 | 4.60 | 6.08 | 21.01 | 7 | 1 |
| enveda-180 | all test adducts | 821,960 | -2.17 | -0.24 | 1.88 | 3.51 | 4.25 | 4.10 | 5.78 | 426 | 0 |
| enveda-np-examples | all test adducts | 1,184 | -5.57 | -1.41 | 1.55 | 3.66 | 8.07 | 6.60 | 21.01 | 29 | 2 |

## 5. Structure overlap between libraries (by inchikey14)

| library | drug_plus | enveda-180 | enveda-np-examples | gnps | masaryk | massbank | mona | msdial | pluskal_ms2 | riken | spectraverse |
|---|---|---|---|---|---|---|---|---|---|---|---|
| drug_plus | 2,539 | 3 | 145 | 2,185 | 226 | 1,643 | 1,758 | 1,277 | 1,612 | 1,841 | 0 |
| enveda-180 | 3 | 182,941 | 0 | 807 | 0 | 2 | 7 | 3 | 2,275 | 7 | 0 |
| enveda-np-examples | 145 | 0 | 250 | 233 | 53 | 208 | 231 | 227 | 224 | 246 | 0 |
| gnps | 2,185 | 807 | 233 | 45,729 | 397 | 5,636 | 7,551 | 5,542 | 15,136 | 9,225 | 0 |
| masaryk | 226 | 0 | 53 | 397 | 416 | 376 | 366 | 239 | 232 | 379 | 0 |
| massbank | 1,643 | 2 | 208 | 5,636 | 376 | 9,157 | 4,969 | 3,733 | 2,280 | 6,528 | 0 |
| mona | 1,758 | 7 | 231 | 7,551 | 366 | 4,969 | 11,679 | 6,802 | 4,715 | 11,180 | 94 |
| msdial | 1,277 | 3 | 227 | 5,542 | 239 | 3,733 | 6,802 | 9,097 | 3,274 | 8,332 | 0 |
| pluskal_ms2 | 1,612 | 2,275 | 224 | 15,136 | 232 | 2,280 | 4,715 | 3,274 | 46,816 | 5,430 | 0 |
| riken | 1,841 | 7 | 246 | 9,225 | 379 | 6,528 | 11,180 | 8,332 | 5,430 | 15,851 | 0 |
| spectraverse | 0 | 0 | 0 | 0 | 0 | 0 | 94 | 0 | 0 | 0 | 9,630 |

Same matrix by mkey:

| library | drug_plus | enveda-180 | enveda-np-examples | gnps | masaryk | massbank | mona | msdial | pluskal_ms2 | riken | spectraverse |
|---|---|---|---|---|---|---|---|---|---|---|---|
| drug_plus | 2,534 | 3 | 145 | 2,185 | 227 | 1,651 | 1,762 | 1,286 | 1,633 | 1,845 | 159 |
| enveda-180 | 3 | 182,941 | 0 | 807 | 0 | 2 | 7 | 3 | 2,276 | 7 | 0 |
| enveda-np-examples | 145 | 0 | 250 | 233 | 53 | 209 | 231 | 227 | 226 | 246 | 11 |
| gnps | 2,185 | 807 | 233 | 45,620 | 398 | 5,624 | 7,541 | 5,543 | 15,147 | 9,216 | 1,025 |
| masaryk | 227 | 0 | 53 | 398 | 415 | 377 | 366 | 240 | 233 | 379 | 12 |
| massbank | 1,651 | 2 | 209 | 5,624 | 377 | 9,127 | 4,968 | 3,734 | 2,288 | 6,524 | 264 |
| mona | 1,762 | 7 | 231 | 7,541 | 366 | 4,968 | 11,617 | 6,797 | 4,730 | 11,161 | 660 |
| msdial | 1,286 | 3 | 227 | 5,543 | 240 | 3,734 | 6,797 | 9,069 | 3,301 | 8,307 | 461 |
| pluskal_ms2 | 1,633 | 2,276 | 226 | 15,147 | 233 | 2,288 | 4,730 | 3,301 | 46,805 | 5,461 | 576 |
| riken | 1,845 | 7 | 246 | 9,216 | 379 | 6,524 | 11,161 | 8,307 | 5,461 | 15,801 | 730 |
| spectraverse | 159 | 0 | 11 | 1,025 | 12 | 264 | 660 | 461 | 576 | 730 | 9,630 |

## 6. C1 eligibility [5.3]: Enveda structures also in other libraries

| library | structures | in any other library | fraction | in a non-Enveda library | fraction | with a test-adduct spectrum there | fraction | mkeys | mkeys in a non-Enveda library | fraction |
|---|---|---|---|---|---|---|---|---|---|---|
| enveda-180 | 182,941 | 2,288 | 0.013 | 2,288 | 0.013 | 2,287 | 0.013 | 182,941 | 2,289 | 0.013 |
| enveda-np-examples | 250 | 250 | 1.000 | 250 | 1.000 | 250 | 1.000 | 250 | 250 | 1.000 |

enveda-np-examples structures (the NP-TIMS set) in non-Enveda libraries: median 6 other libraries and 158 reference spectra per structure (min 2, p10 28); per library: riken 246, gnps 233, mona 231, msdial 227, pluskal_ms2 224, massbank 208, drug_plus 145, masaryk 53.

## 7. Candidate recall ceiling: train structures in the NP and PubChem tiers [2, 5.5]

| library | structures | in NP tier | in PubChem tier | in either | passes 2.2 filters | in either / passes filters |
|---|---|---|---|---|---|---|
| drug_plus | 2,539 | 0.913 | 0.959 | 0.967 | 0.969 | 0.993 |
| enveda-180 | 182,941 | 0.002 | 0.973 | 0.973 | 1.000 | 0.973 |
| enveda-np-examples | 250 | 1.000 | 0.992 | 1.000 | 1.000 | 1.000 |
| gnps | 45,729 | 0.494 | 0.780 | 0.786 | 0.986 | 0.795 |
| masaryk | 416 | 0.971 | 0.976 | 0.976 | 0.978 | 0.998 |
| massbank | 9,157 | 0.816 | 0.954 | 0.958 | 0.955 | 0.998 |
| mona | 11,679 | 0.888 | 0.967 | 0.974 | 0.974 | 0.996 |
| msdial | 9,097 | 0.826 | 0.942 | 0.951 | 0.949 | 0.998 |
| pluskal_ms2 | 46,816 | 0.284 | 0.976 | 0.976 | 0.994 | 0.980 |
| riken | 15,851 | 0.854 | 0.960 | 0.971 | 0.970 | 0.996 |
| spectraverse | 9,630 | 0.262 | 0.803 | 0.806 | 0.994 | 0.808 |

Membership is by inchikey14 (train label vs the tier's ik14), a lower bound for the metric (a tier row with another ik14 but the same mkey, e.g. a tautomer, also counts; `validation.truth_keys` finds those per holdout molecule, and evaluate.py's candidate recall uses them). A structure that fails the 2.2 filters cannot be retrieved from either tier by construction.

enveda-np-examples (the NP-TIMS set, closest to the hidden test): 250 of 250 in the NP tier, 248 in the PubChem tier, 250 in either. Not in PubChem: FSASIHFSFGAIJM, YHKYFYCSTJGMID; not in the NP tier: none.

## 8. Element coverage and the other 2.2 filters

| library | structures | elements outside set | charged | multi-component | isotope label | < 3 C | mass outside range | RDKit fails | extra elements (structures) |
|---|---|---|---|---|---|---|---|---|---|
| drug_plus | 2,539 | 0.0000 | 0.0102 | 0.0000 | 0.0000 | 0.0079 | 0.0181 | 0.0000 |  |
| enveda-180 | 182,941 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| enveda-np-examples | 250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| gnps | 45,729 | 0.0010 | 0.0056 | 0.0000 | 0.0000 | 0.0010 | 0.0068 | 0.0000 | B 20, Si 10, As 8, Se 6 |
| masaryk | 416 | 0.0024 | 0.0000 | 0.0000 | 0.0000 | 0.0120 | 0.0144 | 0.0000 | Si 1 |
| massbank | 9,157 | 0.0015 | 0.0244 | 0.0000 | 0.0000 | 0.0064 | 0.0159 | 0.0000 | As 5, Se 4, Si 4, B 1 |
| mona | 11,679 | 0.0010 | 0.0086 | 0.0000 | 0.0003 | 0.0035 | 0.0142 | 0.0000 | B 4, Se 3, As 3, Si 2 |
| msdial | 9,097 | 0.0005 | 0.0077 | 0.0000 | 0.0004 | 0.0042 | 0.0406 | 0.0000 | Si 4, Se 1 |
| pluskal_ms2 | 46,816 | 0.0009 | 0.0038 | 0.0000 | 0.0000 | 0.0001 | 0.0011 | 0.0000 | B 26, Si 11, Se 4, Sn 1 |
| riken | 15,851 | 0.0009 | 0.0118 | 0.0000 | 0.0003 | 0.0028 | 0.0160 | 0.0000 | Si 5, B 4, Se 3, As 3 |
| spectraverse | 9,630 | 0.0002 | 0.0050 | 0.0000 | 0.0000 | 0.0000 | 0.0006 | 0.0000 | Si 2 |

NP-heavy libraries (enveda-np-examples, riken, gnps, msdial, massbank, mona; 55,093 structures): B: 0.038 percent; Si: 0.029 percent; Se: 0.011 percent. Rule (2.2): widen the element set only above 0.5 percent. Recommendation: **keep the element set**.

## 9. mkey versus inchikey14 [5.1]

mkey computed for 275,775 of 275,775 structures (status: {'ok': 275775}). mkey != ik14 for 4,362 (1.58 percent). mkeys shared by more than one ik14: 1,540 (covering 3,165 ik14).

| library | fraction mkey != ik14 | in another library only under another ik14 |
|---|---|---|
| drug_plus | 0.0748 | 9 |
| enveda-180 | 0.0050 | 1 |
| enveda-np-examples | 0.0640 | 0 |
| gnps | 0.0446 | 460 |
| masaryk | 0.0433 | 3 |
| massbank | 0.0474 | 18 |
| mona | 0.0649 | 12 |
| msdial | 0.0695 | 1 |
| pluskal_ms2 | 0.0278 | 122 |
| riken | 0.0768 | 46 |
| spectraverse | 0.0004 | 1,383 |

Last column: structures of the library whose mkey occurs in some other library but never under the same ik14 (tautomer or protonation variants labelled differently). Library matching, splits and the C1/C2 masks must group by mkey, not ik14 (`splits` and `validation` do).

## 10. Public test join [5.7]

Public test spectra matched to a train row: 1,213 of 1,213 (0 by shared spectrum_id, 1,213 by precursor m/z plus peak arrays). Molecules with a label: 400; label consistent across a molecule's spectra: 400.

| source library | spectra | molecules |
|---|---|---|
| enveda-180 | 1,213 | 400 |

M error of the public test spectra against their train label (ppm, after the organisers' cleaning): p50 2.07, p90 3.23, p99 4.13, max 4.65; per molecule (median over its spectra) p99.5 4.37, max 4.65.

Identical copies (every test peak matched, every train peak matched, same intensities): 1,213 of 1,213 matched spectra. Test spectra keeping peaks above precursor + 2 Da: 418; train peaks above the precursor missing from the test copy: 0. Public test base_peak_intensity minimum 1,018.

Peak cut evidence: largest kept peak above precursor +2.1846 Da; smallest train peak above precursor missing from the test spectrum none (no train peak is missing); intensity ratio test/train on matched peaks: median 1.0000, max spread 0 (1 and 0 mean the intensities were not renormalised).

Matched spectra per molecule: mean 3.03, max 9. Train spectra available for the same structure in the source library versus spectra the test kept (query emulation, 5.2):

kept / available: mean 0.584; available median 6, max 19.

## 11. Retrieval tolerance recommendation [3.3]

Error of the molecule mass M (median over a molecule's test-adduct spectra, as `io.mass_tables` computes it; molecule = structure for the two Enveda libraries, molecule_id for the public test) against the label's formula mass, in ppm. Coverage = fraction of molecules with |M error| within the window.

| source | molecules | signed median | abs p50 | abs p99 | abs p99.5 | abs max | cov 3 | cov 5 | cov 7.5 | cov 10 | cov 15 | cov 20 | n > 5 ppm | n > 10 ppm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| enveda-180 | 179,693 | 2.006 | 2.015 | 4.089 | 4.264 | 5.504 | 0.852 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 48 | 0 |
| enveda-np-examples | 250 | 1.451 | 1.487 | 6.026 | 8.159 | 8.769 | 0.876 | 0.976 | 0.992 | 1.000 | 1.000 | 1.000 | 6 | 0 |
| public test | 400 | 2.089 | 2.097 | 4.215 | 4.370 | 4.654 | 0.830 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 0 |

Pooled Enveda libraries (earlier definition): p99.5 4.27 ppm over 179,943 groups (max 8.77); public test single-spectrum max 4.65 ppm.

Recommended `mass_tol_ppm`: **10** (smallest of 3, 5, 7.5, 10, 15, 20 ppm, at least 5, that holds 99.5 percent of the molecules of every source; current default 10); decided by: enveda-np-examples (enveda-np-examples shares the hidden test's pipeline and chemistry). Signed medians above 0 mean M is heavier than the label mass (a calibration bias; per adduct in section 4): a window centred at M minus that bias would be narrower for the same coverage. `val_query_max_abs_ppm` (query emulation, organisers' precursor cut) should not be below the public single-spectrum maximum above.

## 12. structures.parquet

275,775 structures; RDKit parse ok 275,775; label SMILES variants per ik14 > 1: 1,608; RDKit mass vs formula mass differs by more than 0.001 Da: 24; NP-likeness median -1.31.

Ingestion: pass 1 78.9 s (32,180 rows/s), peak RSS 1,693 MB during pass 1; structures 181.69 s (RDKit 2772.4 structures/s on 3 jobs).

## 13. Holdout molecules (WP7, `splits/holdout_v1.csv`) [5.2]

| set | molecules (mkeys) | ik14 rows | eligible C1 | mass min / median / max | query spectra | spectra per molecule | query libraries |
|---|---|---|---|---|---|---|---|
| NP-TIMS | 250 | 265 | 250 | 133 / 354 / 1109 | 599 | 2.40 | enveda-np-examples 250 |
| SYN-TIMS | 3,000 | 3,001 | 44 | 235 / 331 / 499 | 8,289 | 2.76 | enveda-180 3000 |
| NP-OTHER | 3,000 | 3,188 | 1,871 | 157 / 353 / 1141 | 8,167 | 2.72 | gnps 1268, riken 670, massbank 574, mona 407, msdial 81 |

ik14 rows beyond the molecules are aliases (another ik14 with the same mkey), masked together with the molecule. Query spectra follow the public test's spectra-per-molecule and collision-energy distribution, capped by what the query library holds (`validation.build_queries`).

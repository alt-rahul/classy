# Experiments

One row per run, appended by `scripts/evaluate.py` (WP7) or by hand; newest at the bottom.
Rules (PLAN.md sections 5.5, 7 and WP15): one change at a time, one row each; every submission
in `submissions/` gets a row; numbers are local validation (out-of-fold where models are fitted),
plus the public leaderboard score in the notes once the user reports it.

Columns:
- date: YYYY-MM-DD.
- tag: as in the submission name (`b0_`, `b1_`, `b2_`, `r1_`, `final_` ...).
- config: `configs/<name>.yaml` plus any overrides.
- git commit: from `casmi.log.git_commit()` (`-dirty` if uncommitted changes).
- cells: MRR@25 per (set, condition) cell, NP-TIMS / SYN-TIMS / NP-OTHER x C1 / C2 / C3,
  the worst-case blend on NP-TIMS with its 90 percent bootstrap interval, and candidate recall.
- notes: what changed, public LB score, anything surprising.

| date | tag | config | git commit | cells | notes |
|---|---|---|---|---|---|
| 2026-11-18 | b0_npdb_prior | `configs/b0_npdb.yaml` | 42be57fd1474-dirty | none: no local validation exists yet (train not available, WP7 not built) | WP4 Baseline 0, NP tier only, prior-only fusion (PLAN 4.6) + assembly (4.8). `submissions/20261118_b0_npdb_prior.csv`, sha256 `5355da5c...3fcb8a`, 400 rows x 25 guesses. Public LB not meaningful (public test is drawn from train). Pool at 10 ppm: median 44.5, p05 10, p95 138, max 702; empty NP pool 1/400 (0.0025, m_2c2340); 94 molecules padded (89 from the 20 ppm window, 22 from nearest NP rows); 27 mkey duplicates skipped. 46 s cold mkey cache, 4.6 s warm, peak RSS 0.77 GB, 2 mkey workers. Cold and warm runs byte-identical. |
| 2026-11-18 | b0b_pubchem_prior | `configs/b0b_pubchem.yaml` | 42be57fd1474-dirty | none: no local validation exists yet (train not available, WP7 not built) | WP4 Baseline 0b, NP + PubChem tiers, same fusion: every NP-tier candidate ranks above every PubChem-only one; NP rows also use PubChem pmid_count in the literature prior. `submissions/20261118_b0b_pubchem_prior.csv`, sha256 `b29a3a54...ad515f`, 400 x 25. Public LB not meaningful. Pool at 10 ppm: median 10,470, p05 3,289, p95 21,180, max 33,523; no empty pool; no padding; first guess equals b0's on 84.8 percent of molecules. 78 s with mkey cache off (50 s warm), peak RSS 1.66 GB, 8 batches of 50. Cache-off and cached runs byte-identical. |

## Notes and spikes

Free-text results that do not fit a row (WP9 tautomer-consistency report, WP13 DreaMS spike,
feature importance, mix-sensitivity tables) go below, each under a dated heading.

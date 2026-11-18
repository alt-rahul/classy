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

## Notes and spikes

Free-text results that do not fit a row (WP9 tautomer-consistency report, WP13 DreaMS spike,
feature importance, mix-sensitivity tables) go below, each under a dated heading.

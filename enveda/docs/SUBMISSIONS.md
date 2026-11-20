# Submissions

Every leaderboard submission, appended by `scripts/kaggle_submit.py` (public score filled in by
`scripts/kaggle_submit.py --status`). Rules (docs/RUNBOOK.md, standing orders): at least 55
minutes between submissions, at most 5 per UTC day, always check with `--check` first.
Times are UTC. "kernel" is `owner/slug vN` of the private inference notebook that was run.

| # | submitted (UTC) | kernel | config | local validation | message | status | public score | ref |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-11-20 19:08:04 | ograhulrajkumar/casmi-inference v2 | b0_npdb | n/a (no local validation yet) | b0_npdb: NP-tier prior baseline (WP4), casmi-inference v2 | COMPLETE | 0.154 | 58630976 |
| 2 | 2026-11-20 20:04:10 | ograhulrajkumar/casmi26-taeyangg4-0495-fork v1 | public_fork_taeyangg4_0495 | n/a (public fork) | Private fork of public taeyangg4 0.495 notebook, unchanged (base for fork-and-extend) | PENDING |  | 58633848 |
| 3 | 2026-11-20 22:02:22 | ograhulrajkumar/casmi-inference v3 | b1_library | v1 NP-OTHER worst-case blend 0.343; v2 P(A) pending R1 | b1: library match + NP/PubChem prior (WP8) | PENDING |  | 58637333 |

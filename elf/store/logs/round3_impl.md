# Round 3 implementation notes (S1 chains, S2 global repair, seeding)

Start: store 162,003 (validated submission 162,003).
Workers: at most 3 processes (one core reserved for a research agent).

## S2 - global repair on independent puzzles <= 44 bits

Driver: `src/grepair.py` (new; resumable, per-puzzle log `store/runs_r3S2.json`, offers through
`offer` + `save_store`, tag `r3S2`). It is repair.improve with `self_thr` = 1e9 (L2 over all values of
the program), 3 passes, i.e. agent C's r1_grepair_scored.py without the list-position resume (C's
script resumes by list index over puzzles with len > 150, which shifts once stored lengths change).
Skips the 22 puzzles C already did (stored tag `r3C`).

    python3 src/grepair.py --tag r3S2 --classes 'indep>=20' --maxbits 44 --skip-tags r3C --workers 1

Resume: rerun the same command (done puzzles are in store/runs_r3S2.json).

## S1 - chain plans in regret_pick

core.c: `inv_forms`, `peel2` (copied from research/round3/ideas/core3.c), `chain_exact` (one tier, y in
exactly level ky, same scan order as peel2), `chain_at_cost`, `emit_chain`; `regret_pick` runs, at each
cost c >= 5 after the pair pass over all alive targets, a chain pass over the targets still alive (same
score-bound pruning). Tiers (v1 level, v2 level, forms): (0,0,all), (1,0,all), (0,1,all),
(1,1, t = y*v1 +- v2 and t = (y +- v1)*v2); cost = lev(y) + lev(v1) + lev(v2) + 2, y in L0..L3.
(tier, ky) combinations with ky + lev(v1) <= 2 are skipped: y op v1 is then in L<=3 and the pair search
at the same cost already covers them. Steps: expand(y), expand(v1), step, expand(v2), step.
`set_peel(ntiers)`, `last_pick_chain()`, `peel_stats()` exported; engine.py wrappers.
solver.py: `Cfg.peel` (default 0), `peel_tiers` (4), `peel_minpool` (10, as in the prototype).
Rebuild: `gcc -O3 -march=native -shared -fPIC -o src/core.so.new src/core.c && mv src/core.so.new src/core.so`
(build to a new file and rename: running processes have core.so mapped).

Reproduction check (peel off): practice_0004 and practice_0007 with the round-2 config give
byte-identical programs with the old library + old solver and the new ones (121 / 91 ops).

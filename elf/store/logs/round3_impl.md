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

Practice check, peel on (`{"pool_cap":46,"evict_size":1,"regret":0.1,"peel":1}`), C's 12 structured puzzles:
0015 200, 0001 130, 0004 117, 0014 130, 0020 98, 0021 124, 0027 184, 0017 155, 0032 126, 0022 129,
0026 142, 0036 124: total 1659 (base 1698, C's PEEL prototype 1660; 10 of 12 identical to the prototype,
0032 126 vs 128, 0036 124 vs 123). All programs valid.
Same-load timing (two processes side by side), 0001/0032/0017: peel off 137/129/158 in 108 s, peel on
130/126/155 in 107 s: no time overhead (prototype 1.5-1.9x).

## Seeding (task 3): built, not run
`src/seed.py` (CF relations, constant miner = A's seeded.mine top 20 <= 1e5, B's fam + 30% relation
test = `shared_like`, B's hub miner mine2 + seed_proto.pick STRICT, variants c / h / ch) and
`src/seed_run.py` (resumable runner, offers the best variant). Classifier on practice: shared 38 of 51
flagged, multiplicative 0 of 40, mixed 7 of 48. The practice test of the variants with peel on was not
run (stopped by the coordinator).

## Stop (18:35): coordinator merged public programs, store 133,977
S2 killed after 13 puzzles (store/runs_r3S2.json): 2,579 -> 2,490 ops, gain 89, 696 CPU-s, about 460
ops/CPU-h. All 13 are now held by shorter public programs (pub:gp_frontier), so the net effect on the
current store is 0. Checkpoint: `bash src/checkpoint.sh` -> invalid 0, score 133,977.
Resume S2: `python3 src/grepair.py --tag r3S2 --classes 'indep>=20' --maxbits 44 --skip-tags r3C --workers 1`
(46 puzzles left; note it now repairs the public programs, a different starting point).
S1 scaled runs and seeding runs: not started.
Other src changes: run.py `--order bits`, `--minbits` / `--maxbits`, saves every 15 s instead of 60 s.

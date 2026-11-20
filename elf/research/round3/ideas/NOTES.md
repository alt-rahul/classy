# Round 3 - solver-side ideas (agent C)

Working dir: research/round3/ideas. Engine copy: core3.c -> core3.so (loaded through eng3.py, ELF_CORE).
Nothing under src/ or data/ is modified.

## Log (progressive)

- 15:45 read README, PLAN2, round2 notes, solver/engine sources.
- 15:55 core3.c: added level_ptr, all_costs (pair cost of every target), hub_scan (for every u in level lu:
  how many targets become cheaper with u in the pool), resid_lev.
- 16:00 E1 (e1_hub_stats.py) on the three dumped mid-solve pools (40/50/60 bit independent):
  level sizes are L1 2.5k, L2 130-210k, L3 7-19M (much smaller than a naive 4^k n^(k+1) count).
  Base pair costs: 40-bit 6-7, 50/60-bit all > 7. Hubs u in L1/L2 with residual in L<=3 serving 2+ targets: 0
  found in all three pools. Verdict: cheap hubs (cost 1-2) do not exist for independent targets; a hub
  would need u in L3 with residual in L3 (expected about 2 per round at 40 bits, none at 50-60).
- 16:10 E2 (e2_hub_other.py, e2_a.log): same scan during normal solves of practice_0004 (shared, 42 bit) and
  practice_0001 (multiplicative, 49 bit). Here hubs exist: 10-370 L2 values per round that serve 2-5 targets,
  nominal gain 2-4 ops each. Base pair costs of the uncovered targets are mostly 4-6.
- 16:20 E3 (s3.py class S3, e3_a.log): commit the best hub before the regret pick. practice_0004: baseline 121,
  hub_g 1 -> 127 (22 hubs), hub_g 2 -> 126 (9 hubs). Negative: the nominal gain is not real, because building
  the first served target already puts equivalent intermediates into the pool; the hub op is extra.
  (practice_0001 run killed for time: 3x slower.)
- a1_marg.py (store analysis): other>=20 marginal cost per target by bits: 20b 4.0, 30b 4.9, 40b 5.8, 50b 7.4,
  55b 9.2, 60b 9.8. Targets >= 45 bits are 21% of the targets and 31% of the ops. Independent: 6.3 at 40 bits,
  7.9 at 50, 9.9 at 60; first three targets average 12.3 ops, steady state 7.7-8.0.
  So the "other" targets are being paid at about the price of random numbers of the same size.
- core3.c: enum_pairs (all pair plans of a given cost for a target). 40-bit pool: 4-6 plans at the minimal
  cost 6, 14-88 plans at cost 7.
- 16:35 E4 (s3.py class LA, e4_la.log): plan-choice lookahead. Target picked by regret score; all its minimal-cost
  pair plans are enumerated (up to la_k 6), each is scored by the sum of the other targets' pair costs with the
  plan's new values added to the pool. practice_0005 (independent 40 bit) 212 -> 209 at 3.5x time; practice_0004
  (shared 42 bit) 121 -> 119 at 2.8x time. About -1.5%: within run-to-run noise of restarts, not better per CPU.
- 16:40 E5 (e5_rootpeel.py, e5_a.log): root peel t = v op r, v in pool, r of pair cost <= 7 (a cost-8 shape the
  partial L4 cannot see) on the dumped 60-bit pool: 0 hits in 3 targets x 92 residuals; engine gives 10 via the
  window search. Killed (also kills "target = covered target +- cheap value" for independent targets).

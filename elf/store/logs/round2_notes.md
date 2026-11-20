# Round 2 notes (implementation agent)
(Times in this file are approximate wall-clock estimates; the logs have the exact sequence.)

Resume info: every run below is resumable by re-running the same command (runs_<tag>.json logs).

## Engine changes (src/core.c)
- L3 sort: two-level MSD radix on a log-linear key + recursive in-cache MSD (was 5-pass LSD).
- Global hash set dropped; per-level Bloom filters only. keep=1 builds (stale deep levels) no longer
  rebuild deep filters: build(...,2,1) went from 0.36 s to 0.01 s.
- Large buffers 2 MB aligned with MADV_HUGEPAGE.
- New regret_pick(ts, maxc, alpha): batch pair search over all targets, breadth-first by cost with
  score-bound pruning; identical choice to research/regret.py S3.
- Result: same programs as the old engine on the base config, ~30% faster overall.

## Solver options (src/solver.py Cfg)
- regret=alpha (>0 enables regret ordering), regret_maxc=7. Exact L2-only shortcut when the best
  cost<=3 score already beats any cost-4 score.
- perm_seed>=0: random target order mode (perm_free: take any target costing <= perm_free first).

## Runs launched
- g1: regret 0.1, pool_cap 46, evict_big 4, --repair, other>=20 and other<20, --order len, 3 workers. log store/logs/g1.log
- g1 restarted 05:31 with --order optasc (lowest ops/target first: mult/shared-like puzzles gain most and run fastest)
- sm1 (05:45): restarts.py portfolio on other<20 + indep<20 (regret, beam 16x6, 10 perm_mix 0.5 seeds, 8 perm seeds, beam variants), --repair, 120 s budget per puzzle, 1 worker. log store/logs/sm1.log
- sm1 restarted 05:50 with beam first (beam 16x6 g0.4 was best single method on 7 small practice puzzles: total 208 vs regret 216 vs base 227), budget 100 s
- engine: fused sort+dedupe installed (identical results); further sort micro-opts gave nothing measurable under load
- 06:05 practice measurements:
  - independent (0005, 0013, 0058): regret 0.1 + evict_size 212/205/252 vs evict_size alone 216/212/259 (-2.6%);
    regret + evict_big 4: 228/213/277 (evict_big is worse with regret on independent).
  - other (0021, 0083): regret 0.15: 124/138, 0.07: 126/141, jitter: 126/141, regret 0.1 + evict_size: 124/139,
    regret 0.1 + evict_big 4: 127/140. beam W4 B4: 125/135 at 4-5x time.
  - aggressive "skip L3 when an L2 plan is cheap enough" rules trigger in only 10-25% of rounds; dropped.
- sm1 stopped at 06:05 after 7 puzzles (gain 25, about 115 ops per CPU-hour: lowest rate); resumable.
- gi (06:05): regret 0.1 + evict_size, pool_cap 46, --repair, indep>=20 + indep<20, --order optasc, 1 worker.
- 06:12 rebalanced: g1 2 workers, gi 2 workers (gi gained 100 ops in 460 s on 1 core = ~780 ops/CPU-h vs g1 ~475)
- 06:30 practice: independent regret alpha 0.3 = 0.1 (212/205); win_k 2 same (249 vs 252 on 0058); regret with
  pure LRU eviction much worse (244/231). New option pair_eps (expected-hit pruning of level pairs in
  regret_pick): independent 60-bit practice_0018 same length 339, 123 s -> 50 s; mixed 60-bit practice_0053
  245 -> 254 (worse). So pair_eps 0.05 only for independent.
- gi restarted 06:31 with pair_eps 0.05 (same tag; done puzzles kept).
- g2 (06:36): regret 0.15 + evict_size, pool_cap 46, --repair, other>=20 + other<20, --order optasc, 1 worker (second config for the others)
- 06:40 FINDING: few-start puzzles waste ops on the first target (indep>=20 with 1 start: first target
  avg 65 ops; 2 starts: 36; 3: 19; 4: 14). New Cfg.warm=W: before the main loop grow the pool to W values,
  each step adding the one-op value that maximises |L0..L3|. 1-start scored_0015: 297 -> 258 (warm 5),
  scored_0036: 324 -> 275 (store had 294 / 329).
- gw (06:45): warm 5 + regret 0.1 + evict_size + pair_eps 0.05 on the 31 one-start independent puzzles, 1 worker; g2 paused (resumable)
- 06:55 gi restarted with warm 5 (applies only to <=3 distinct starts, Cfg.warm_ns) and --nstarts 2..8 (1-start ones are in gw). 2-start scored_0010 209->201, scored_0043 256->243, 3-start scored_0004 254->238 with warm 5.
- 06:55 warm on 2-start other puzzle scored_0003 ([4,22], 60-bit): no gain (254/254/258); others keep warm off
- gw done 06:58: 31 one-start independent puzzles, gain 1142 (first target 65 -> 23 ops). gw2 (07:00): warm 6, warm_lev 4 variant on the same 31, 1 worker. 4-start warm test (scored_0001): 265 base, warm5 268, warm6 271 -> warm only for <=3 starts.
- warm_one (force 1 = v/v first): worse on 1-start (267/276 vs 258/275); dropped
- 07:03 gi restarted on 2-3 start puzzles first (--nstarts 2,3, --order len); its rate on 4+ start 50-60-bit puzzles had dropped to ~110 ops/CPU-h
- gw2 done 07:20: gain 106 (best-of-2 over gw on the 31 one-start puzzles: 7832 -> 7726; warm6/lev4 avg 7786 vs warm5 7832). g2 resumed 07:20 with 1 worker.
- 07:30 engine: pair_search skips the repeated second direction of mul/div when both levels are the same (exact, identical results). A division-free two-pointer mul was slower (scans all of Y) and was reverted.
- regret_deep 30 (deeper levels + maxc 9 for regret rounds with pool < 30): no effect (212/205/123 vs 212/205/124); left off
- 07:45 gi 2-3 start phase done: 46 puzzles, gain 484 (~480 ops/CPU-h). Launched gi on 4-8 start (1 worker) and gw3 = warm 6 / warm_lev 4 second variant on 2-3 start independent (1 worker).
- oracle test (item 5): practice_0015 with the known hidden base h2 seeded as an extra target: 215 vs 211 plain. Hidden-base seeding does not help; dropped.
- warm on 2-start other puzzles (scored_0025/0062/0110): 158/170/205 vs 160/165/199 without; no warm for others (confirmed).
- 07:43 gw3 stopped (5 puzzles, gain 7: second warm variant on 2-3 start puzzles is not worth it); g1 restarted with 3 workers.
- 07:45 g2 paused again (rate ~166 ops/CPU-h); sm2: beam grid on other<20 (beam 16x6 g0.4, alpha 0.05 variant, 12x5 g0.6 a0.15), budget 60 s, 1 worker
- 08:00 mixed practice (0053/0029/0161) regret+evict_size 245/199/251 vs regret+evict_big4 252/200/252. g1 restarted for its remaining puzzles with evict_size instead of evict_big 4 (same tag).
- 08:05 gi on 4+ start independent stopped (21 puzzles, gain 35, ~95 ops/CPU-h: the store already had evict_size restarts for these); g1 now 4 workers.
- 08:10 g1 after the switch to evict_size: 40 puzzles, gain 500 in 697 s on 4 workers (~645 ops/CPU-h, vs ~350-445 with evict_big 4 on easier buckets). Next: g2 (alpha 0.15 + evict_size) first on the 344 others with original ops/target in [4,6) (done by g1 with evict_big 4); id list in store/logs/g2_ids.txt. Round-2 start store total was 170,274 (README's 170,375 predates the i1 merge).
- 08:12 sm2 stopped (38 puzzles, gain 62, ~145 ops/CPU-h); g2 started on g2_ids with 1 worker.
- 08:18 keep_small 8/64/1024 with regret+evict_size: no effect; evict_size=2 (4-bit size classes): same; evict_size=3 (keep last-round values): worse (142/136/207 vs 139/137/199).
- 08:50 warm 7 + warm_lev 4 on 1-start scored_0015/0036/0183: 254/270/340 vs store 258/275/343. gw4: that config on the 31 one-start puzzles, 1 worker.
- 09:05 found 37 two/three-start independent puzzles whose stored programs do not begin with the warm-up values (some were run by gi before warm was enabled). gw5: warm 5 config on those + scored_0010 (ids in store/logs/gw5_ids.txt), 1 worker. warm 7/lev 4 on 2-3 start: 254/246/199 vs 243/238/208 (mixed), not used there.
- 09:13 gw4 done: gain 60 (14 of 31 improved). g2 restarted with 2 workers on g2_ids.
- 09:21 gw5 done: gain 129 (19 of 38 improved).
- 09:52 g1 done (all 742 other puzzles). g2 chain (store/logs/g2chain.sh): g2_ids then g2b_ids (rest of the others, sorted by ops/target), 4 workers.
- 10:00 g2 vs g1 on 225 common g2_ids puzzles: g1 25,889, g2 25,726 (alpha 0.15 + evict_size 0.6% better on average), best-of-2 25,492 (-1.5%).
- 10:08 pair_eps 0.005 on mixed 60-bit (regret 0.15 + evict_size): 247/251 vs 245/251, 1.3-1.8x faster. g2 second stage (g2b_ids) will use pair_eps 0.005 (store/logs/g2chain2.sh waits for the g2_ids run).
- 10:22 queued store/logs/g3chain.sh: after g2, sm2 resume (1 worker) + g3 = regret 0.07 + evict_size + pair_eps 0.005 on other>=20, optasc (3 workers). Stop everything around 12:15 for the final checkpoint.
- 10:42 g3chain cancelled: g2's second stage runs at ~120 ops/CPU-h and the whole g2 pass is ~0.5% of the total; a third config would be below the 0.5% stop rule. Finish after g2.
- 11:13 g2 done (all 742 other puzzles; total g2 gain 1,085). Final checkpoint: validate.py invalid 0 score 162,152.

## Final state (round 2)
- Total 162,152 (start 170,274). Per-run gains from the logs: g1 4,331; gi 1,168 (484 of it from the
  warm-up on 2-3 start puzzles); gw 1,142; gw2 106; gw3 7; gw4 60; gw5 129; g2 1,085; sm1 42; sm2 66.
- Nothing is running. All runs are resumable with the same commands (runs_<tag>.json).
- If continuing: a third config on the others (e.g. regret 0.07 + evict_size + pair_eps 0.005, queued
  as store/logs/g3chain.sh but cancelled) is expected below 0.5% of the total; sm2 (beam on other<20)
  has 69 puzzles left at ~140 ops/CPU-h.

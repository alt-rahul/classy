# Elf calculator solver

Straight-line programs over 48 registers that make every target value appear, scored by the
total instruction count over 1,000 puzzles (lower is better).

## How to run

All commands are run from `/home/user/classy/elf`.

```
gcc -O3 -march=native -shared -fPIC -o src/core.so src/core.c   # build the C engine
python3 src/run.py --tag s3 --workers 4 --cfg '{"pool_cap":46}'   # solve all puzzles (resumable)
python3 src/repair_store.py --workers 4                            # post-pass on stored programs
python3 src/write_sub.py submissions/submission.csv                # re-validate all, write CSV
python3 src/validate.py data/puzzles.csv submissions/submission.csv
```

`run.py` keeps a per-tag log (`store/runs_<tag>.json`) so an interrupted run resumes where it
stopped; `--redo` reruns everything, `--skip <tag>` skips puzzles done under another tag,
`--ids` / `--classes` restrict the set. The store `store/best.json` holds one validated program
per puzzle and is only ever replaced by a strictly shorter valid program (saves are merge-safe
under a file lock, so several runners can work at once). It was seeded from the sample submission.

Practice-set benchmarking: `src/bench.sh '<cfg json>'` (12 practice puzzles, 3 per family) or
`python3 src/test_practice.py <ids|family|all> '<cfg json>'`.

## Files

- `src/validate.py` - independent exact validator (provided, unchanged).
- `src/common.py` - puzzle loading, class labels, store load/merge-save/offer (validates every
  program before storing), submission writer (re-validates all 1,000), per-class breakdown.
- `src/core.c` / `src/core.so` - C engine (ctypes). Builds level sets L0..Lk of values reachable
  from the current register pool with exact tree cost k (L0..L2 with witnesses, L3 always
  complete, deeper levels partial under a candidate cap, values re-derived on demand), with
  blocked Bloom filters for membership. Searches: pair / meet-in-the-middle `t = u op w`
  (lookups or two-pointer scans), window `t = X*Y +- Z` with a per-combination budget and
  expected-hit pruning, Horner candidates `t = X*Y +- Z` with `Z = t mod X`.
  Round 2: `regret_pick` (batch pair search over all uncovered targets, breadth-first by cost
  with score-bound pruning and optional expected-hit pruning `pair_eps`), two-level MSD radix
  sort fused with de-duplication for L3, per-level Bloom filters only (deep filters are kept on
  `keep` builds), huge-page buffers. Same programs as before, about 30% faster.
- `src/engine.py` - ctypes wrapper (`ELF_CORE` env var selects another build of the library).
- `src/solver.py` - greedy "cheapest target next" loop over a live pool of values, SSA by
  value; cheap tier (pairs on L<=2, optional lookahead), deep tier (fresh deep levels, pair search
  to cost 11, window search, Horner beam, guaranteed Horner fallback); LRU eviction that protects
  small constants; register allocation with dead-code elimination (`regalloc`).
  Round 2 options: `regret` (regret ordering, PLAN2 item 1), `warm` / `warm_lev` / `warm_ns`
  (grow a few-start pool before the main loop), `pair_eps`, `perm_seed` (random target orders),
  `regret_jit`, `regret_multi`, `regret_deep`, `evict_size` 2/3 (the last four measured without
  gain).
- `src/beam.py` - beam search over target order (item 4): expands each state with fresh L0..L3,
  one child per target among the best by regret score, ranked by ops + gamma * remaining costs.
- `src/restarts.py` - multi-config driver: runs a grid of configs (including beam) per puzzle
  and keeps the best; resumable like `run.py`.
- `src/tiny.py` - exact BFS over sets of produced values for tiny puzzles (limit <= 20).
- `src/repair.py` - ruin-and-recreate post-pass: drop a target's private subtree and rebuild the
  target from all values available before its first use, if strictly cheaper.
- `src/run.py` - parallel resumable runner (round 2: `--order optasc`, `--nstarts`).
  `src/repair_store.py` - parallel post-pass driver.
- `src/write_sub.py` - writes the submission from the store. `src/test_practice.py`,
  `src/bench.sh` - practice-set benchmarks.

## Score history

All totals are validated with `src/validate.py` (0 invalid).

| Stage | Total |
|---|---|
| Sample baseline | 1,490,379 |
| Stage 0: tiny puzzles solved exactly by BFS, rest baseline | 1,462,369 |
| Main engine on 466 of 1,000 puzzles (runs s2a, s3), rest baseline | 868,900 |
| Main engine on all 1,000 puzzles (runs s3-s7, `pool_cap` 46) | 179,208 |
| Ruin-and-recreate repair post-pass | 176,025 |
| Restarts on independent puzzles (r1, `evict_size`) | 174,337 |
| Restarts on other puzzles (r2, `evict_big` 4), partial: 70 of 473 | 173,016 |
| r2 completed on all 473 | **170,375** (final) |

Per-class breakdown at 170,375 (final):

| Class | Puzzles | Targets | Ops | Ops per target |
|---|---|---|---|---|
| indep<20 | 27 | 359 | 1,951 | 5.43 |
| indep>=20 | 215 | 6,450 | 55,922 | 8.67 |
| other<20 | 107 | 980 | 3,662 | 3.74 |
| other>=20 | 635 | 19,068 | 108,783 | 5.71 |
| tiny | 16 | 45 | 57 | 1.27 |

What helped most: the greedy cheapest-target-next loop with every intermediate kept in the pool
(about 8x over the baseline), a large pool (`pool_cap` 46), and eviction policies that drop big
values first (`evict_big`, `evict_size`, worth a few percent on independent puzzles). The repair
pass and multi-config restarts each give about 1-2%.

Tried without gain on the 6-puzzle experiment sets (`store/logs/exp_*.log`): `lookahead` 4,
`win_k` 3, random tie-break noise, `evict_unused`, a larger window budget.

## Round 2

Round 2 started from the validated store at 170,274 (the 170,375 above plus the i1 sweep merge).
Run notes with measurements, launch commands and resume info: `store/logs/round2_notes.md`.

| Stage | Total |
|---|---|
| Round 2 start (store) | 170,274 |
| g1 regret pass on the first other puzzles, gi regret + evict_size on 40-bit independent | 169,021 |
| gw warm-up on the 31 one-start independent puzzles | 167,084 |
| gw2, gi warm-up on 2-3 start independent, g1 continuing | 165,552 |
| g1 with evict_size on its last 204 puzzles (all 742 other puzzles done), gw4, gw5 | 162,750 |
| g2 second config (`regret` 0.15, `evict_size`) on all 742 other puzzles: final | **162,152** |

Per-class breakdown, round 2 start vs final:

| Class | Puzzles | Targets | Ops start | Ops final | Ops per target final |
|---|---|---|---|---|---|
| indep<20 | 27 | 359 | 1,951 | 1,880 | 5.24 |
| indep>=20 | 215 | 6,450 | 55,821 | 53,238 | 8.25 |
| other<20 | 107 | 980 | 3,662 | 3,497 | 3.57 |
| other>=20 | 635 | 19,068 | 108,783 | 103,480 | 5.43 |
| tiny | 16 | 45 | 57 | 57 | 1.27 |

Round 2 gained 8,122 ops (4.8%). The other team's 128,831 is still 33k away: other>=20 targets
of 50-60 bits cost about as much as independent targets of the same size (7.9 / 10.0 ops at 50 /
60 bits), so the generic search is not exploiting the shared/multiplicative structure there.

Gains by item (from the run logs; each pass only stores strictly shorter programs, so later
passes are measured against the improved store):

- Regret ordering (PLAN2 item 1), run g1 on all 742 other puzzles with `--repair`: about -4.4k.
  `regret` 0.1 commits the target minimising cost - 0.1 * bits over all pair plans of cost <= 7
  on fresh L0..L3. Implemented as one C call (`regret_pick`): breadth-first by cost over all
  targets with score-bound pruning, plus an exact L2-only shortcut; it picks exactly what
  `research/regret.py` picks. The last 204 g1 puzzles used `evict_size` instead of `evict_big` 4
  (better on 6 of 6 practice puzzles with regret).
- Warm-up for few-start independent puzzles (new; `warm` 5, applied only with at most 3 distinct
  start values): with one start value the first target cost 65 ops on average (36 with two
  starts, 19 with three, 14 with four). Growing the pool first, one op at a time with the value
  that maximises |L0..L3|, brings it to about 23. One-start puzzles: gw 1,142, then best-of
  variants gw2 (`warm` 6, `warm_lev` 4) 106 and gw4 (`warm` 7, `warm_lev` 4) 60; two/three-start
  puzzles: gi 484 and gw5 129. About -1.9k in total.
- Regret + `evict_size` on independent puzzles (gi): about 2.6% on practice against `evict_size`
  alone; on the scored 4+ start puzzles only about 0.5-1% over the store (which already held
  `evict_size` restarts). `pair_eps` 0.05 skips level pairs whose expected number of solutions is
  below 5%: 60-bit independent practice puzzle 2.5x faster at the same length.
- Second config on the other puzzles (item 2, run g2: `regret` 0.15, `evict_size`, `pair_eps`
  0.005 on the hardest 246): best-of-2 about 1.5% on the puzzles g1 ran with `evict_big` 4 and
  less where g1 already used `evict_size`; about -1.1k in total.
- Small puzzles (item 3/4): beam search over target order (`src/beam.py`, width 16, branch 6)
  was the best single method on 7 small practice puzzles (208 vs 216 regret vs 227 greedy), but
  on the scored small puzzles g1's regret pass had already taken most of it; sm1 + sm2 gained
  about 0.1k at about 140 ops per CPU-hour, so they were stopped.
- Engine: identical programs, about 30% faster (MSD radix sort fused with de-duplication for L3,
  per-level Bloom filters only, deep filters kept on `keep` builds which went from 0.36 s to
  0.01 s, huge-page buffers, no repeated second direction for same-level mul/div scans).

Measured without gain in round 2 (practice or small scored samples): random target orders
(`perm_seed`, best of 16-30 orders no better than regret), hidden-base seeding (item 5 oracle
test: practice_0015 with its true hidden base added as a target, 215 vs 211), warm-up for other
puzzles or for 4+ start independent puzzles, forcing 1 = v / v as the first warm-up value,
committing two targets per L3 build (`regret_multi`: 2x faster, 3.6% longer), deeper levels for
small pools in regret rounds (`regret_deep`), regret alpha jitter, `keep_small` 8 / 1024,
`evict_size` variants (4-bit size classes, keeping last-round values), regret with plain LRU
eviction on independent (much worse), `pair_eps` 0.05 on mixed puzzles (4% worse; 0.005 is
nearly free), beam width 4 on 30-target puzzles (1.6-3.6% at 4-5x the time, no better per
CPU-hour than restarts), approximate rules for skipping the L3 build, and several attempts to
speed up the L3 sort beyond the first MSD version (memory bound on this machine).
Not done: item 6 (exhaustive X*Y +- Z at 60 bits); a quick check showed a 10x window budget
finds one extra cost-9 plan in 8 targets at 1.8x the window time.

Best configs (round 2): other puzzles `{"pool_cap":46,"evict_size":1,"regret":0.1}` (or 0.15);
independent puzzles `{"pool_cap":46,"evict_size":1,"regret":0.1,"pair_eps":0.05,"warm":5}`
(`warm` only acts with at most 3 distinct start values).

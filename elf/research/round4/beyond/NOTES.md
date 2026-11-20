# Round 4 "beyond public" research notes (lead research agent)

## Log
- 18:40 Read public negative lists (goodpjw "what did not work", Asterios experiments E01-E17), our round-3 notes.
- Analysis of the stored public programs (store = 133,977), scripts: dagutil.py, show.py, cones.py
  - Score split by (goodpjw) family: independent 47.1k (6.91/target), shared 41.1k (4.12), mixed 34.0k (5.48),
    multiplicative 11.8k (3.03), tiny 57.
  - Independent ops/target is extremely tight across puzzles: 40-bit 5.70 (q10 5.57, q90 5.87), 50-bit 7.19
    (7.03..7.32), 60-bit 8.63 (8.47..8.75) = about 7 bits per instruction = the information rate of TREES over a
    48-leaf pool (leaf 5.5 bits + op 1.5 bits). The DAG counting bound (two free operands per op, 13 bits) is 3.8.
  - "shared" 56+ bits is bimodal: 3.2-4.4 (true shared) vs 6.5-7.4 (about a quarter; they behave like mixed/random).
  - Marginal ops per target along the program (50-bit indep): 28, 11.8, 9.6, ... about 8 until target 16, then 5.5
    falling to 3.5 for the last ones. Late targets are cheap because helpers are billed to their first user.
  - Register pressure: mean live 31-37 of 48 for independent; max 48 is hit in most 50/60-bit independents
    (29% of steps at >= 47 live for 60-bit), almost never in shared/multiplicative.
  - EXCLUSIVE CONE SIZES ARE TRUNCATED BY PUBLIC SEARCH DEPTH: independent 40-bit max 5, 50-bit max 6 (320 targets at 6),
    60-bit 7 (248) and 8 (52). Same in mixed and in the random-like tail of shared. If every cone > 6 became 6: -695;
    if every cone > 5 became 5: -2327. Public LNS is exact to depth 4 and partial at 5 over the whole DAG.
    => lever A: exact wide-pool (all DAG values) cone rebuild at depth 6-7 (wide.cpp).
- (clock labels in this file are rough; the session started 18:36 and lever B began about 19:15)
- Lever A (deeper single-cone rebuild over the whole DAG pool) built and measured: wide.cpp (flat shapes
  T = X*Y +- R, X,Y in S0..S2 with window scans, R of cost <= 3, exact-division shapes, chains), wide2.cpp (recursive:
  R solved recursively, chains on big pool values at depth 0/1, enumeration mode), drivers wlib.py / wrun.py / wtest.py.
  Sanity: both reproduce the existing cone cost instantly (cost 6 at 50 bits in 3 s; cost 8 at 60 bits in 1 s).
  RESULTS (negative):
  * 50-bit independent, cone 6 -> 5: 0 of 16 targets (4 puzzles, 50 s per target).
  * 60-bit independent, cone 7 -> 6: 0 of 10; cone 8 -> 7: found for 7 of 13 targets in < 30 s, but EVERY rebuild
    needs 50-54 registers (best of all block positions / DFS orders / target-order local search); none could be emitted.
    The original 60-bit programs sit at 48 live and a target-order local search cannot get the unchanged DAG below 48
    (0002: 52, 0020: 48, 0001: 47), so there is no slack to hold extra leaves alive.
  * Equal-cost alternatives per cone-6 target (50-bit, full enumeration 40 s): only 6-14, so pairs of targets
    sharing a new node: 0 found on scored_0001. (My a-priori estimate of 50-100 alternatives was 5-10x too high.)
  * Public elfopt on 60-bit independents with 48 vs 64 vs 200 registers, 40 s per puzzle: no gain in any, 0 strict
    moves rejected for pressure. Registers do not block the PUBLIC neighbourhoods; they block deeper ones.
  => single-cone / pair moves are exhausted even two levels deeper than public. Lever A is dead (worth < 50).
- 40-bit independent cone 5 -> 4 with wide2: 0 of about 100 targets (24 puzzles). Public is exact at depth 4.
- Public constructor (elfopt --construct, wide pool greedy) on scored_0001: 265-277 ops and 62-70 live values
  (PRESSURE-FAIL) vs 229 in the store. Restart-from-scratch with public tools is not a route to new basins.
- "Shared" tail (6.5-7.4 ops/target at 56+ bits) = mislabelled mixed puzzles: 14-20 of ~30 targets sit in one 18-digit
  decade (random half), few pair relations. True shared puzzles (17-30 of 30 targets in strong continued-fraction pair
  relations) cost 3.2-4.3 ops/target in the public programs, i.e. BELOW the hub recipe of the headroom study
  (practice_0015: 5.3 validated, 4.7 estimated). Hub recovery is already exploited by the public reverse beam.
- Information-rate reading: public independent programs deliver 7.0 bits per instruction at 40, 50 and 60 bits. A forest
  of trees over a 48-value pool gives log2(48) + 1.5..2 = 7.1..7.6 bits per instruction; with the 30! order symmetry the
  absolute ceiling is about 15% below public for the random targets (64k of the score), and nobody is near it
  (leader: -4.2% overall). A total under 120k (-10.4%) would need the whole theoretical slack on random targets plus
  -8% on the structured ones. Not reachable with any method measured here.
- Lever B (group rebuild, jcone.py / gcone.py / gcone2.py / jrun.py / grun.py): since a cost-6 rebuild of ANY single
  50-bit target exists over the full DAG (found in 3 s), a group A of targets whose joint exclusive cone JC(A) (values that
  only targets of A depend on) has more than 6|A| instructions can be shortened by rebuilding each target separately.
  Potential (store analysis, greedy disjoint groups): pairs 23 (i50) + 18 (m50); groups <= 10: about 155 (i50);
  60-bit at floor 7: about 320 (i60) but floor-7 rebuilds exist for only half the targets and are register-blocked.
  First validated gains: scored_0004 212 -> 211, scored_0033 208 -> 207 (tag wide-joint). Pair yield so far 2 ok,
  2 noemit (registers), 4 none of 8 attempts => about 25%. Expected total over all classes: 50-100 instructions.
- REGISTER SCHEDULING IS THE BLOCKER FOR DEEP MOVES. Half of the found pair rebuilds at 50 bits and all single-cone
  rebuilds at 60 bits were rejected because naive placement needs 50-54 registers. Tried: block insertion at 3 positions,
  DFS order, greedy list scheduling with noise (66+), single-op local search (no progress), target-permutation local
  search (cannot even re-find 48 for the unchanged DAG), CP-SAT cumulative model (valid, but no solution in 40-60 s
  from a 54-register hint; 12 s even with the exact hint). WORKS: simulated annealing over the topological order with
  SEGMENT moves (a run of 1-14 consecutive ops is moved as a block inside its dependency window), energy = 100 * excess
  over 48 + excess over 46; C++ sched.cpp: feasible in 3-5 s on the cases tested (51-53 -> 48). wlib.c_sched /
  try_emit5; splice_blocks inserts each rebuilt target at its best position first.
  With it: scored_0145 216 -> 215 (was noemit), 60-bit cone 8 -> 7 now lands (scored_0020, 0051, 0107, ...).
- KEY OBSERVATION (19:46): in every public file from 133,998 down to 133,977 the class totals of the big random-like
  puzzles are IDENTICAL (independent 40/50/60-bit: 14461 / 16041 / 15427; mixed 50-bit 13012). All public progress
  happens on shared / multiplicative / mixed-60. Public LNS is too shallow for puzzles of 180+ instructions (its
  authors say so), so 46k + 27k of the score is not being polished by anybody in public. wide2 + sched is the first
  tool here that moves them (independent 50-bit 16041 -> 16028, 60-bit 15427 -> 15416 within the first hour).
- Drivers now: wrun.py (single cones), jrun.py (pairs), grun.py (groups <= 8, best_groups from gcone2.py),
  wlns.py (plateau walk with equal-cost floor rebuilds + group pass after every accepted move).
  First group result: scored_0001 229 -> 227 (one group of 8 targets, joint cone 50).
- wide2 search order fixed (20:05): solutions that used to appear after 20-26 s came from the stage "T = u +- n with u a
  one-step value near T" which scanned 50k one-step values in value order. Now: candidates within a size window, closest
  first, before the pool-value chain and the heavy (2,2) window. Floor rebuilds at 50 bits: 12 of 12 found within the
  8 s limit (10 of 12 before), most in < 1 s after the 3 s set build.
  Re-test of single cones with the faster engine: 50-bit cone 6 -> 5: 0 of 27; 60-bit cone 7 -> 6: 0 of 8. Single
  cones really are at the floor of the whole-DAG pool (5 / 6 / 7 instructions at 40 / 50 / 60 bits).
- Statistics so far: 60-bit single cone 8 -> 7: found for 19 of 44 targets, 11 scheduled (old scheduler for the first
  ones); pairs at 50 bits: 47 attempts, 11 gains. Groups (grun): scored_0001 -2, then mostly 'nosol' (one target of the
  group has no floor rebuild) or 'noemit57-60' (a group rebuild pulls leaves from everywhere; 57-60 registers).
  wlns.py (plateau walk) 15 min x 3 puzzles: no gain; its time went into re-trying groups (fixed afterwards).
- Regression of independent program length on the number of targets n (store): 40-bit 32 + 4.64 n, 50-bit 33 + 6.09 n,
  60-bit 28 + 7.67 n; residual std after n and the number of starts: 0.8-1.2% (2 ops). So about 30 ops per puzzle are
  bootstrap (pool growth from the starts) and restarts of an equally good pipeline are worth at most about 1%.
- Existing DAGs vs registers (sched.cpp, 15 s): 60-bit programs cannot be scheduled in 46 registers, 50-bit ones can
  in 46 but not in 44. The public programs are register-tight; this is the global constraint for 50/60-bit puzzles.
- Group rebuild made register-aware (grun.attempt): per target up to 6 enumerated solutions, the one whose best insertion
  position gives the lowest register peak is taken; scored_0036 233 -> 231 in 48 s (was noemit59), scored_0015 -1.
  Most group attempts still end 'nosol' (one member has no floor rebuild). Pre-screen variant (attempt2, env GRUN2=1):
  over DAG - JC only 2-4 of 6-8 members have a floor rebuild, and the sub-group's joint cone is then too small; the
  sequential variant (new nodes of rebuilt members rejoin the pool) is the better one. Restore fallback: no gain.
  Realised: about 1 gain per 8 group attempts (45-70 s each).
- wide2 chain window: for budgets >= 5 the remainder is unrestricted again (the 60-bit shape T = big - 56-bit n needs it);
  13 puzzles searched with the narrow window are re-queued (tag t60e).
- 20:32 status: 43 validated instructions gained in 35 puzzles (tags wide, wide-joint, wide-group), store 133,923.
  Running in the background (resumable, results go to the store): wrun t60c (60-bit single cones, then t60e),
  jrun j60 (pairs on i60, m60, s60, s50), grun g50 (groups on i50 + m50, 6 per puzzle).
- Plateau walk test (walk_test.py, scored_0043): 8 accepted equal-cost rebuilds of random cones (23 tries, 190 s); the
  group potential stayed at 1 throughout. wlns.py 10 min on scored_0043: 9 group attempts, 0 gains. A deep LNS with
  plateau moves has no visible legs on these programs; the gains are the static ones (cones / pairs / groups above floor).
- CONCLUSION: no lever above about 0.4k was found. Public programs for big random-like puzzles sit at the floor of the
  whole-DAG pool for single targets, have few equal-cost alternatives, are register-tight, and do not respond to plateau
  walks. What remains for us: (1) lever B harvest, 150-350 instructions per pass, (2) fleet-scale public LNS + CP-SAT
  on the mid-size puzzles, (3) hourly merges of the public frontier.

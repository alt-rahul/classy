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
- 19:00-19:55 Lever A (deeper single-cone rebuild over the whole DAG pool) built and measured: wide.cpp (flat shapes
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

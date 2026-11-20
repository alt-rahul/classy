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

# Round 5 - structured families (shared / multiplicative / structured mixed): generator recovery and slack

Scripts: lib.py (helpers), stat.py, show.py (annotated value program), cone_stat.py, infra.py, sx.py (ILP exchange).

## Log
- 20:50 start. Store 133,919. Read round-3/4 notes (generator_shared, generator_meta, headroom, beyond).
- Family split of the store by goodpjw labels (fam.json) x "mixed-like" (>= 12 targets in one 12/15/18-digit decade):
  | class | puzzles | targets | ops | ops/target |
  | shared pure | 248 | 7454 | 27,856 | 3.74 |
  | shared mixed-like | 67 | 2036 | 11,691 | 5.74 |
  | multiplicative pure | 117 | 3516 | 10,849 | 3.09 |
  | mixed pure (no big random decade) | 53 | 1580 | 6,561 | 4.15 |
  | mixed mixed-like | 148 | 4423 | 26,656 | 6.03 |
  | < 20 targets (shared/mult/mixed) | 107 | 980 | 2,971 | 3.0 |
  So the clean structured part is 418 puzzles / 45.3k (+3k small); the other 38k is mixed-like (random half at the tree floor).
- Exclusive cone histogram (ops that die if one target is dropped), shared pure: 0:1519 1:1952 2:2137 3:1284 4:467 5:79 6:15 7:1.
  Sum of exclusive cones = 12.2k of 27.9k: 1.64 per target. The other 2.1 per target (63 per puzzle) is shared infrastructure.
- Anatomy (infra.py), mean per pure puzzle: shared 112.3 ops = 35.5 small-value ops (31.2 distinct smalls are operands of
  big ops, only 4.4 auxiliary) + 8.3 hubs (big value from two smalls) + 38.9 big non-target intermediates + 30 targets.
  multiplicative 92.7 = 32.2 smalls (28.0 needed) + 7.9 hubs + 28.5 + 25.8. mixed-pure 123.8 = 41.4 (35.7) + 9.9 + 43 + 29.8.
- Generator read-off on scored_0617 (29 targets, 90 ops, show.py): mids 3513, 5022, 5792; hubs = all 6 pairwise products
  (incl. squares); every target = chain of 1-4 ops from a hub with small operands: H-42, 18H-21, 627(H+174), 9(627H+22),
  39(2504 H-2), 14(H+195) ... The public program contains exactly these hubs (20 ops for the smalls + mids, 6 hubs) and
  finishes at 2.2 ops per target, sharing chain prefixes between targets (H1-24 serves two targets). Multipliers are
  reused pool values, additive offsets are mostly private (42, 174, 305, 43, 330, 306, 586) and cost 0-1 extra op each.
  => the public programs ALREADY sit on the hidden mids/hubs; a "build hubs, then offsets" recipe costs
  1 (target) + 1..2 (chain) + ~1 (private constant) per target, i.e. more than the 1 + 1.3 + 1.2 measured in the store.
- SX = exact multi-exchange test (sx.py): universe = all current values + every value one op away from a pair of current
  values; all recipes of every current value over that universe (operands current, or one new); CP-SAT picks the smallest
  acyclic (integer levels) value set covering the targets; asks for length <= current - 1. This contains every
  simultaneous "add k new one-op values, rewire anything, delete k+1" move, which the public LNS (cones of <= 3 targets,
  one helper subtree) and trajectory recombination do not enumerate.
  * small structured puzzles (< 20 targets, 2,971 ops), full universe, 60 s: 87 INFEASIBLE (proved: no shorter program in
    the universe), 3 UNKNOWN, 0 gains (17 largest not run).
  * pure hard puzzles, full universe (10-16k new values, 30-60k recipes): UNKNOWN at 150 s (too big to prove).
  * pure hard puzzles, new values restricted to smalls < 1e5 (all rewiring among current values allowed), 40 s:
    random sample of 40: 33 INFEASIBLE, 7 UNKNOWN, 0 gains.
  * sx2.py (no levels, lazy loop cuts, recipes with two new operands): no answer in 120 s even on 32-op puzzles. Dropped.
  => the small-value layer (35 ops per puzzle) cannot be shortened by swapping in new one-op constants, anywhere tested.
  * mixed-like puzzles (structured half + random decade), smalls < 1e5: sample of 16: 14 INFEASIBLE, 2 UNKNOWN, 0 gains.
- Plateau rigidity (divx.py: SX asked for an EQUAL-length program using as many new one-op smalls as possible):
  scored_0617: at most 1 new small value at equal length. The programs have almost no equal-cost neighbours in the
  small layer.
- Restart test (restart.py: public constructor elfopt --construct 3, then public LNS with kicks):
  scored_0617 stored 90: construct 124-127, after 60 s LNS 118 and 119; scored_0855 stored 63: construct 76, LNS 71;
  scored_0749 stored 89: construct 104, LNS 98. The landscape is rugged: LNS from a 30% worse start stays 10-30% worse.
  The frontier programs come from strong constructors (reverse beam on hubs), not from polishing.
- Planted-cost estimate for scored_0617 from the decoded chains: with every constant free the 29 targets need 48 big ops
  (public: 55, but it shares constants and prefixes); the simplest chains need about 35-40 distinct constants
  (42 = 2*21, 43 = 21+22, 63, 24, 9, 1, 198 = 9*22, 39, 29, ... plausible one-op pool values) => planted program about
  6 hubs + 48 + 35..40 = 89..94 vs 90 public. Public = planted cost, to within a few instructions.
- scored_0995 (4.5 ops/target, the most expensive pure shared): mid G = 874898 = 1493*586; the "hubs" are M*(G+a) with
  M in {2727, 2723, 6957, 1042, 546, 225, 75} and several targets are sums of two such products. The public program
  uses exactly this (874312 = G-586, 875484 = G+586, ...). Expensive because of many private constants, not because the
  structure was missed.
- Mixed-like puzzles (mixsplit.py): S = targets outside the random decade, R = targets in it.
  closure(S)/|S| = 6.2 / 6.8 / 6.7 (mixed 12/15/18 digits), 5.5 / 5.3 / 6.0 (shared label); marginal R given S =
  4.2 / 5.8 / 7.5 per random target; marginal S given R only 1.4-2.0. Control (subset.py): closure of 14 random targets
  of a PURE puzzle / 14 = 5.7-5.8 (shared), 4.7-5.2 (multiplicative), 6.3-6.6 (mixed-pure). So the structured half of
  a mixed puzzle costs what 14 targets of a pure puzzle cost; no excess to harvest there.

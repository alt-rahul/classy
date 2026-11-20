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

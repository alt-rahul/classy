# Round 3 - generator reverse engineering (shared / multiplicative) - agent A notes

Scripts here: lib.py (helpers, factor via coreutils), rel.py (pairwise CF relations), chain.py, decomp.py (annotated solver program), oracle.py (solve with extra free starts), t*.py (one-off tests).

## Log
- H0 lower_bound column carries planted length: REFUTED. lower_bound == number of targets not among the starts, on all 200 practice puzzles.
- Medium practice_0006 hand reconstruction (24 ops, 9 targets): 22=9+13, 286=22*13, 572=2*286, g=572^2 (hidden), targets 2g, 17g, 70g, g-4, 639^2, 572*639, 572*213+15, 572*213-70, 213^2+71 (213=3*71, 639=3*213). Targets look like leaves: (hidden product) then one or two tweaks by a small. Non-targets g, 572*213, 213^2 are each used by >=2 targets.
- practice_0015 (shared hard, starts [16,30]): hidden bases factor into shared mid-size hidden values: h2 = 2^6*5^3*31*421*7507, T4-2 = 2^4*5*17*31*421*8377 (T22 = 2036*(T4-2), T19 = 509*(T4+478)). 509 also divides T26; 1019 = 2*509+1 multiplies the T23/T30 base. So a persistent small/mid pool is reused across targets.
- Multipliers in strong pair relations over 46 shared practice puzzles: 1 (633 times), then 2,3,5,7,4,8,14,9,18,... i.e. start-sized numbers; no fixed cross-puzzle list. So "multipliers recur" is within a puzzle (pool reuse), not a generator constant table.
- Oracle: practice_0015 with 17 known smalls given as free starts (1,2,7,31,70,509,127,63,153,511,478,1330,10,5,99,14,25): 214 -> 172 ops. The smalls would cost about 20-25 ops to build, so net gain about 10%: constants alone are not the main lever.
- Pair relations T_j = m*T_i + r (m<=5000, >=14 bits surprise) over shared practice: pure multiples (r=0) essentially never occur (3 of 462); m=1 twins 169; m*(T_i+a) with c=0 144; with c!=0 145. So targets are never plain small multiples of other targets: both sides are siblings off a hidden base, T = M*(B+a)+c with M and c optional. The offsets a are mid-size (hundreds to 1e5+), c tiny.
- Bases are smooth products that share mid-size hidden factors (13051 = 31*421 in two bases of practice_0015), offsets/multipliers are reused pool values (127, 509, 31 used several times). Model: a hidden pool of small/mid values, bases = products of pool values, targets = M*(B+a)+c with M, a, c from the pool.
- Oracle 2: smalls free + two known bases as extra targets: 171 (vs 172 smalls only). Clusters with known base become cheap (base 6 ops, members 1-3 ops), but the solver already found them once the smalls existed.
- seeded.py prototype (honest cost): mine multipliers/offsets from strong pair relations, build them first with the solver (puzzle A), solve with them as extra starts (puzzle B), concatenate SSA steps, regalloc with DCE. practice_0015: 214 -> 184 (K=20 mined smalls), 183 (K=28), 194 with the hand list.

## Batch: seeded.py (K=20 mined relation constants, built first, then free starts) vs same-config baseline, shared hard practice
0015 214->184, 0027 189->184, 0071 172->158, 0084 151->148, 0091 211->204, 0111 164->161, 0141 143->146, 0149 192->184.
Sum 1436 -> 1369 (-4.7%). Positive on 7 of 8 but small outside 0015/0071. (batch1_k20.log)
Note agent B (generator_meta) independently found the hub hierarchy on practice_0015 (G = 522040 = 2^3*5*31*421 shared by three hubs) and has a hub-seeding prototype; see its NOTES.md. My seeding variant differs: constants go in as free starts after a separate build program (never evicted), hubs not seeded.

## Multiplicative family
- M-H1 targets share big hidden factors: REFUTED. Over 38 hard practice puzzles (1138 targets) no prime > 1e4 divides two targets of the same puzzle (m2.py). Pairwise gcds are small smooth numbers (products of starts-sized values, e.g. 58752 = 2^7*3^3*17, 118 = 2*59).
- 55% of multiplicative targets are 1e4-smooth, 5% are prime; the rest are (smooth multiplier) x (one private large prime-ish cofactor X).
- M-H2 cofactors X are related to each other (twins / affine): REFUTED. After stripping primes <= 1000, only 4% of cofactors have a >=10-bit pair relation (shared: 23% on cofactors, 53% on raw targets) (m1.py).
- M-H3 cofactor X = smooth +- tiny: REFUTED. P(X+-1 has largest prime < X^0.4) = 164/507 vs 152/507 for random odd numbers of the same size (m3.py). Shared cofactors show a mild signal (286 vs 211 of 817).
- Surviving model: multiplicative target = product of a few reused small pool values (smooth multiplier, shared between targets) times at most one private random-looking factor of any size; or fully smooth. No hub structure to mine. The private factor costs what a random number of its size costs.

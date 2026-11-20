# Round 3 - generator meta / mixed / independent (agent B)

## H1 lower_bound carries information - REFUTED
lower_bound = number of targets not already among the starts (tiny: 3 targets, 1 is a start -> lb 2). No info.

## H2 independent targets are planted (short formula from starts) - REFUTED
Test: exact tree cost from starts for ALL values <= limit on the 6-digit independent puzzles
(exact_small.py; practice_0038 + 8 scored). Target cost distribution = distribution over the whole
decade (e.g. scored_0005 mean 5.69 vs decade 5.56; 0121 5.53 vs 5.34; 0584 5.71 vs 5.92).
Verdict: targets are not cheaper than random values.

## H3 independent targets are uniform random integers in one decade - CONFIRMED
8116 targets (scored+practice): leading digit flat, position in decade flat, last digit flat, divisibility by
2,3,5,7,11,13 = random, omega and largest-prime-factor share = random control, pairwise gcd = control,
normalized sorted gaps exponential (frac<0.1 = 0.093 vs 0.095). indep_stats.py.
Model: hard: 1-6 starts uniform in ~[1,2000], 28-32 targets uniform in [10^(k-1),10^k), k in {12,15,18}.
medium: 2-8 starts in ~[1,500], 8-18 targets, k in {6,9,12}.

## H4 value_limit is a planted value - REFUTED (random slack)
max(target)/limit in [0.25,1]; limit = max_target * U(1,4) capped below 1e18 (for 18-digit decades the
limit is in (max, 1e18), ratio ~0.95-1.0). Factorisation of limits looks random. No info in the limit.

## H5 mixed = random-decade half + structured half - CONFIRMED
Every hard mixed practice puzzle has 14-18 targets (about n/2+1) in exactly one decade k in {12,15,18} digits
(medium: 6 digits), the rest are smaller shared/multiplicative-style values. mixed_dec.py, fam.py (classifier:
practice confusion 48/48 mixed found, 4 small false positives among medium shared/mult).
Scored (store 162,152), marginal ops per target (attrib.py):
 indep12 6.52, indep15 8.43, indep18 10.39
 mixed12 rand 5.95 / struct 5.27; mixed15 rand 7.77 / struct 6.39; mixed18 rand 9.77 / struct 7.06
 shared+mult (421 puzzles) 4.67
 => the structured half of mixed costs 5.3-7.1 per target, far above pure shared/mult. Investigating.

## H6 mixed decade targets are all uniform random - MOSTLY (about 95%)
mixed_smooth.py: P(largest prime < n^0.3) is 3-5% for mixed decade targets vs 2% for independent; omega slightly
higher. A few structured (multiplicative/shared-style) targets also fall in the top decade (e.g. scored_0003:
4 targets in the 18-digit decade share the factor 1350 and one shares 1619 with a 49-bit target).
So "random half" = n/2 uniform targets; the 1-4 extra decade members are structured ones that happen to be big.

## H7 structured targets are nodes of a dense random SLP (target = one op on two other targets/starts) - REFUTED
closure.py: fraction of targets that are 1 op from (targets U starts): shared 0.6%, mult 0.2%, mixed 0.0%.
2 ops: <2%. Targets are leaves of private finishing steps on hidden values, not neighbours of each other.

## H8 structured targets are chains of start-only steps (v -> v op start) from another target - REFUTED
peel.py depth 4, targets > 1e7: hit fraction shared 4.2%, mixed 1.2%, mult 0.8% (independent 0.0%).
Step operands are hidden small/medium values, not just the starts.

## H9 multipliers M in T = M*B + c come from a universal constant list - REFUTED
mstat.py over scored+practice: multiplier frequency decays smoothly with size (2,3,4,5,7,8,9,6,16,11,...);
puzzle-specific values (e.g. 31/63/127/509/1019 in practice_0015) are derived from that puzzle's starts.

## Cost shares (store 162,152), from attrib.py + fam.py
 pure random targets: indep>=20 53.2k + mixed random half 25.7k (+ small) = about 80k = 49% of the total.
 structured: shared+mult 59.1k (12,652 targets, 4.67), mixed structured half 18.3k (2,969 targets, 6.17).
 If random numbers are near the tree floor for everyone, the other team's 128.8k implies about 3.1 ops per
 structured target (we: 4.96).

## H10 hidden hubs are hierarchical and the constants come from a reused small pool - CONFIRMED on practice_0015
practice_0015 [16,30]: three hidden bases share G = 522040 = 2^3*5*31*421:
 B2 = T2+1080 = G*163451, Y = T4-2 = G*284818, h2 = G*1501400 (none is a target).
 Targets around B2: T2 = B2-1080, T7 = 63*(B2+2), T11 = 127*(B2-502), T12 = 160*B2-600, T27 = 63178*B2-31
 (63178 = 2*31*1019), T20 = 992*B2-8191 = 32*(31*B2-256)+1, T16 = 496*B2-6629.
 Around Y: T4 = Y+2, T22 = 2036*Y, T19 = 509*(Y+480), T29 = 126356*(Y-3)-2 (126356 = 4*31*1019).
 Around h2: T13 = 31*h2-2, T14 = 31*h2+151, T18 = 70*h2+1, T8 = 7*h2+9311.
 Constants form a chain of small hidden values: 31, 62, 63, 127, 509 (=4*127+1), 1019 (=2*509+1), 496 = 16*31,
 992, 2036 = 4*509, 480 = 16*30, 511, 256.
Model: (i) hidden small pool S built from the starts by a random short program, (ii) hidden hubs = products of
pool values (shared factor G), (iii) each target = hub modified by 1-3 steps (x m, +- c) with m, c from S.
Hub tree cost from the starts is NOT low (practice_0159 hub 591000 costs 7 like its neighbours), so hubs
cannot be found by enumerating cheap values; they must be mined from the targets.

## Oracle experiment (oracle.py, practice_0015, cfg pool_cap 46 + regret 0.1 + evict_size) - STRONG
Extra values are added as extra TARGETS (the solver has to build them; ops counted in full):
 plain 214 ops
 + 15 random small numbers (control) 216
 + 15 true constants (31,62,63,127,509,1019,496,992,2036,160,480,511,256,1080,600) 180 (-16%)
 + 4 true hubs (G, B2, Y, h2) 191 (-11%)
 + both 165 (-23%)   -> 5.3 ops/target instead of 6.9
So seeding mined hidden values as extra targets is exploitable (PLAN2's test with h2 alone, 215 vs 211, was too weak).

## Automatic hub miner + seeding prototype (mine2.py, seed_proto.py)
mine2.mine: for every target T >= 1e7 enumerate B = (T + c)/m for |c| <= 2500, m <= 2500 (numpy, 5 s per puzzle);
a value B reached from >= 2 targets is a hub candidate. seed_proto.pick ranks hubs by description gain
 gain(B) = sum_i max(0, bits(T_i) - bits(m_i) - bits(|c_i|) - 2) - bits(B)   (>= 12 bits required)
(the gain filter is essential: without it chance coincidences among 9-10 digit values flood mixed puzzles),
drops hubs within 6000 of an accepted hub, and also seeds gcds >= 1e4 between accepted hubs (the G level).
Seeds are passed to the unchanged solver as extra targets (ops for building them are counted).
First results (same cfg, single runs):
 practice_0015 shared: plain 214, auto hubs (no gain filter) 184 (-14%), auto hubs + auto constants 192.
 practice_0029 mixed: plain 199, unfiltered hubs 211 (false hubs hurt), unfiltered hubs + constants 232.
 Auto-mined constants (m and |c| of the relations) do not help; only the true pool constants did (oracle).

## Seeding results with the gain filter (seed_proto.py; STRICT=1 means gain >= 20 and >= 2 new targets per hub)
single runs, cfg pool_cap 46 + regret 0.1 + evict_size, ops include building the seeds:
 practice_0015 shared (2 starts, 56 bit): plain 214, hubs 184, STRICT hubs 184  (-14%)
 practice_0042 shared: plain 173, hubs 171 (-1%)
 practice_0084 shared: plain 151, hubs 163 (+8%), STRICT 158 (+5%)
 practice_0029 mixed: plain 199, hubs 210 (+5.5%), STRICT 200 (+0.5%)
Verdict: forced seeding is a per-puzzle gamble (run-to-run spread is 2-4%); it pays where the plain solver is
expensive (few starts, big hubs: 0015 at 6.9 ops/target) and costs a little elsewhere. Use it as one more
restart config (keep best), not as a replacement. Auto-mined constants never helped.
The real lever shown by the oracle (-23%) needs the TRUE small pool, which the miner does not recover yet.

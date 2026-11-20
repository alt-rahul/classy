# Round 4 - headroom study (search headroom vs unrecovered structure)

Files: rs.py (randomised PEEL restarts, exp 1a), rs1.jsonl (results), ...

## Log
- 18:20 read README, PLAN2, round-3 notes. Launched rs.py on core 3: practice_0006 (medium), 0015/0027 (shared),
  0029/0023 (mixed), 0001/0032 (mult); run 0 = base cfg and PEEL, then randomised PEEL (alpha, score noise, jitter).
- 18:35 coordinator: public programs are 15-19% shorter (store now 133,977), wrap up. Study stopped after ~25 min of
  wall clock; rs.py killed after 1 randomised round, no LNS/beam run. Everything below is what was measured.

## Exp 1 (search headroom) - only the start was run (rs1.jsonl)
single runs, cfg pool_cap 46 + evict_size + regret 0.1:
| puzzle | family | base | PEEL | 1 randomised PEEL run |
|---|---|---|---|---|
| practice_0006 | shared medium | 28 | 28 | 28, 28 |
| practice_0015 | shared | 214 | 200 | 207 |
| practice_0027 | shared | 189 | 184 | 188 |
| practice_0029 | mixed | 199 | 197 | 197 |
| practice_0023 | mixed | 176 | 171 | 169 |
| practice_0001 | mult | 137 | 130 | 136 |
| practice_0032 | mult | 129 | 128 | 128 |
No best-of-N curve exists (N = 2 per puzzle). The one randomised round moved results by -1%..+5%, i.e. the same
2-4% run-to-run spread agents saw in round 2/3. Nothing here contradicts "restarts of this solver buy a few %".

## Exp 2 (practice_0015) - validated 164-op program, hand1_G.pkl (base 214, PEEL 200)  => -23% / -18%
Hubs (CF pair relations between targets, then smallest |a|+|b| hub; hub.py):
 G = 522040; B2 = 85327960040 = G*163451; Y = 148686388720 = G*284818; h2 = 783790856000 = G*1501400 (agent B)
 new: P0 = T0-127 = 81121293120; H9 = T15/31 = 1127100976310 (T9 = 7*H9+25861); X3 = T3+115 = 122702665707
 (T17 = 350*X3+2912, T24 = 3906*X3+14, T21 = 1778*X3-16107, T26 = 509*(30*X3-89)); H23 = 23695877541009
 (T23 = 14*H23+1462, T30 = 1019*H23-261); T5 (T10 = 30*T5+3662, T25 = 3556*(T5+1157)-29).
 => 30 of 31 targets hang off 9 hubs with 2-4 finishing steps (only T6 has no relation).
Construction (hand.py, hand1.py): stage A = PEEL solver builds the 9 hubs + G as a puzzle: 54 ops. Stage B =
 every target finished from a hub by T = p*H + c, p*(H+a)+c or p1*(p2*H+a)+c, constants costed exactly on L0..L2
 of ALL values built so far (no 46 cap; regalloc still fits 48 registers): 110 ops for 31 targets = 3.55/target
 (costs 1,2,2,2,2,2,3,3,3,3,3,4x11,5x7,6). Total 164, common.check OK.
KEY STRUCTURE (read off stage A, which found it by itself once the hubs were the targets):
 mid pool values a = {522040 (G), 163451, 284818, 1501400, 750700 = 1501400/2, 509}; hubs = products of two mid
 values +- small:  T5 = G^2 - 140; B2 = G*163451; Y = G*284818; h2 = G*1501400; P0 = 284818^2 - 4;
 X3 = 163451*750700 + 7; H9 = 1501400*750700 - 3690; H23 = 163451*284818*509 - 253.
 Same template as the medium practice_0006 reconstruction (hubs 572^2, 572*639, 572*213, 213^2, 639^2).
 Generator model for shared: few mid values (4-6) -> hubs = pairwise products (+- small) -> targets = M*(hub+a)+c.
Remaining slack in the 164: stage A is 54 ops for 10 values; with the mid values built once (G 11 ops incl. smalls,
 163451/284818/1501400 about 4-5 each) and hubs at 1-3 ops each it should be about 35-40, so about 145-150 total
 (4.7-4.8 per target) is reachable for this puzzle by this recipe; not built.
Oracle variants with the unchanged solver (hubs/constants as extra TARGETS, orc.log): constants+4 hubs PEEL 160,
 constants + 10 hubs PEEL 175, base 179; 10 hubs only PEEL 181. So forcing hubs into the greedy loop is erratic;
 the gain comes from (a) the finishing search over a wide leaf set and (b) building hubs from the mid values.

## Reading against the public method (coordinator note)
The public reverse beam (t = child*factor +- atom, factor/atom from a 2-step dictionary over ~100 leaves) is the
same mechanism as stage B here: chain finishing from a big child with constants costed on L<=2 of a WIDE leaf set.
Measured here: 3.55 ops/target for finishing on a 55-bit 2-start shared puzzle, vs 6.5-6.9 for our loop.
Not run: two.py (seeds as free starts), LNS, beam, best-of-N curves.

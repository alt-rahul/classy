# Round 5 constructor research - running log

## Framing (before measuring)
- Tree-over-pool counting: a k-op tree has k+1 leaves; with N=48 live leaves that is about 5.6 bits per leaf + 1-2 bits per op,
  i.e. 6.5-7.5 bits/op. Realised public: 7.0 bits/op at 40/50/60 bits. So the store sits at the TREE bound for a 48-pool;
  only two things can beat it: (a) less bootstrap (intercept 28-33 ops/puzzle), (b) nodes shared by several targets
  chosen among alternative recipes (floor+1 / floor+2 recipes are 100x / 10^4x more numerous than floor recipes).
- H4 (targets uniform random): already tested on scored puzzles in round 3 (generator_meta/NOTES.md H2 refuted, H3 confirmed;
  peel hit fraction 0.0% for independent). Not repeated.

## Log (20 Nov, started 20:50)
- 21:00 E2/E3/E3b (xyr.cpp + rlib.py + e2/e3/e3b.py): recipes T = x*y +- r with x, y, r at most 2 chain ops from a pool.
  * Over the store's live 46-value pool at step 56 of scored_0001 only 10 of 32 targets have a recipe of cost <= 8:
    trees over a live 46-pool cost 8-9 at 50 bits (not 6-7). The store's 5.3 per target after its 56-op prefix is all sharing.
  * Whole-program one-shot rebuild over the store's shared core (values with reach >= 2, 119 values): sum of per-target
    tree minima 114 vs 108 in the store; union selection (min distinct new nodes, coordinate descent) 112. No gain.
  * Hub test (H2): 602,602 candidate hub values (all D1/D2 nodes in any recipe of cost <= 8): best gain 0, no hub
    lowers the summed cost by more than its own cost. H2 is dead on top of the stored programs.
- 21:10 Public reverse beam (juan/cpp/beam_search.cpp, mode 1) raw on the 28-puzzle benchmark: 7885 vs store 6041
  (ratio 1.305). It uses a FIXED palette (powers of two, small odds) and only 4 recent intermediates as leaves; all
  mid-value reuse in the store comes from LNS. elfopt polish of a raw 291 program: 279 after 90 s (slow).
  Our src/solver.py default cfg: 319 / 290 on scored_0001 / 0004.
- 21:15 nscale.py: the 50/60-bit regression intercepts come from n in 28..32 only (extrapolation). At 40 bits (n 10..32)
  L = 24.3 + 4.90 n; 30 bits 12.6 + 3.98 n; 20 bits 5.9 + 2.74 n.
- 21:20 cons.cpp (rolling-pool greedy, cheapest target next, recipes over the pool, from the 4 starts, no palette):
  * pool capped at 42 values (FIFO/LRU eviction): 270-274 on scored_0001; costs 14,10,9,9 then 8 per target.
    Bootstrap overhead from scratch is only about 10 ops; starting from the store's 40-op core gives the same total.
  * 1-ply lookahead over 24 alternatives: 268 (weak; H3 in this form is worth about 2%).
  * KEY: pool cap is not register use. cap 60 -> 258 using 32 registers; cap 80 -> 243 using 40; cap 120 -> 234 (54 regs).
    Values that are never used again cost no register, so the candidate pool can be far larger than 48.
  * EXACT liveness mode (pool = every value whose live range can be extended to now without exceeding the register
    cap at any earlier point; recipe applied only if all its extensions fit): scored_0001 236 valid, 46 registers,
    35 s, no LNS. Store 227; public raw constructors 265-291.

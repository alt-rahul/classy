# Round 2 plan (Fable analysis, after the 170,375 submission)

Goal: beat 128,831. Fable's projection with the levers below is about 158-162k (optimistic 154k). Reaching 128.8k would need other>=20 at about 3.7 ops/target (now 5.71), which implies generator structure not yet recovered.

Prototypes and experiment scripts are in `research/` (copied from Fable's scratch dir; they may reference /tmp/fable_scratch paths).

## Independent: nearly tapped out
- The engine returns the exact minimum tree cost for its pool at 40 and 50 bits (exhaustive check on dumped mid-solve pools, `research/exact.py`, `exact7.py`). At 60 bits a partial search found one cost-9 where the engine had 10.
- Floor with an ideal 46-value pool: 6.00 / 7.55 / 9.25 ops at 40 / 50 / 60 bits; current steady state 6.3-6.5 / ~8.0 / ~9.9. First target costs 19-31 ops (warm-up).
- Pool size is binding: at 40 bits, 20 values cost 7.8, 30 cost 6.9, 46 cost 6.0.
- Measured negative: deeper/exhaustive pair search at 40-50 bits, larger window budget, pool_cap 47, eviction-window policy, pre-seeded basis (break-even), further evict_big sweeps (the i1 sweep: 118 puzzles, gain 101 ops).
- Remaining independent levers: seed-only restart pass (about -0.5k), exhaustive X*Y +- Z at 60 bits only (about -0.6k, C work), warm-up seeding (break-even to -5 ops/puzzle).

## Other families: structure
- Shared: targets are `M*(B + a) + c` with small M, a, c around bases B that are often hidden non-targets (practice_0015: h2 = 783790856000 is not a target; T13 = 31*h2 - 2, T18 = 70*h2 + 1, T8 = 7*(h2 + 1330) + 1). Multipliers recur (31, 62, 63, 127, 509, 1019 and products); additive constants look private.
- Multiplicative: small smooth multiplier times a random-looking large factor.
- Mixed: near-limit targets look independent-style.
- Common-base clustering finds a base for 52% of shared, 19% of mixed, 18% of multiplicative targets (0.3% false-positive control on independent).
- Exploiting relations directly is weak (each relation needs its own small constants): modelled 5.87 vs 5.32 ops now on linked targets.
- But order matters a lot: on medium practice_0006, greedy = 35 ops, best of 50 random orders = 27, hand reconstruction about 24.

## Prioritized plan

| # | Build | Evidence | Gain | Compute |
|---|---|---|---|---|
| 1 | Regret ordering on all 742 other puzzles | measured | -3.5k to -5k | about 12 h on 4 cores as prototyped |
| 2 | Restart grid on top of item 1, plus repair | partly measured | -1.5k to -2.5k | 1-2 days on 4 cores |
| 3 | Random-order restarts on the 134 small puzzles | measured | about -0.4k | under 1 h |
| 4 | Beam or rollout over (target, plan) choices for other | untested | -2k to -4k | time-box 1 day |
| 5 | Hidden-base mining for shared | modelled, weak | 0 to -2k | time-box half a day |
| 6 | Exhaustive X*Y +- Z at 60 bits only | inconclusive | about -0.6k | hours of CPU, C work |

Item 1 (regret ordering): each round `engine.build(order, limit, 3, 0)`, then `find_pair(t, 7)` for every uncovered target; commit the target minimising `cost - alpha * t.bit_length()` with alpha = 0.10; if nothing costs <= 7, fall back to the existing `search_round`. Measured vs same config (pool_cap 46, evict_big 4), single runs: multiplicative 522 -> 492 (-5.7%), shared 659 -> 610 (-7.4%), mixed 562 -> 552 (-1.8%). Alpha sweep on 7 puzzles (base 996): 0 -> 964, 0.10 -> 930, 0.15 -> 937, 0.25 -> 968; best per puzzle 923. 2-5 min per puzzle as prototyped (3x slower); a C routine returning costs for all targets in one call would cut that. Prototype: `research/regret.py` (class `S3`).

Item 2: alpha in {0.07, 0.10, 0.13, 0.15} x eviction variant x seed; run-to-run spread 2-4% per puzzle. Seed-only pass for independent.

Item 3: 50 random orders gave 411 -> 378 (-8%) on 11 medium practice puzzles; apply to other<20 and indep<20 (5.6k ops).

Item 5: before building, run the oracle test (seed a known hidden base, count ops saved).

Practice single-run baseline (`research/prac_every4.log`): independent 10.12, mixed 7.25, shared 5.90, multiplicative 4.47 ops/target.

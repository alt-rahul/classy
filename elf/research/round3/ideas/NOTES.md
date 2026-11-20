# Round 3 - solver-side ideas (agent C)

Working dir: research/round3/ideas. Engine copy: core3.c -> core3.so (loaded through eng3.py, ELF_CORE).
Nothing under src/ or data/ is modified.

## Log (progressive; clock times are approximate labels, about one hour ahead of the machine clock)

- 15:45 read README, PLAN2, round2 notes, solver/engine sources.
- 15:55 core3.c: added level_ptr, all_costs (pair cost of every target), hub_scan (for every u in level lu:
  how many targets become cheaper with u in the pool), resid_lev.
- 16:00 E1 (e1_hub_stats.py) on the three dumped mid-solve pools (40/50/60 bit independent):
  level sizes are L1 2.5k, L2 130-210k, L3 7-19M (much smaller than a naive 4^k n^(k+1) count).
  Base pair costs: 40-bit 6-7, 50/60-bit all > 7. Hubs u in L1/L2 with residual in L<=3 serving 2+ targets: 0
  found in all three pools. Verdict: cheap hubs (cost 1-2) do not exist for independent targets; a hub
  would need u in L3 with residual in L3 (expected about 2 per round at 40 bits, none at 50-60).
- 16:10 E2 (e2_hub_other.py, e2_a.log): same scan during normal solves of practice_0004 (shared, 42 bit) and
  practice_0001 (multiplicative, 49 bit). Here hubs exist: 10-370 L2 values per round that serve 2-5 targets,
  nominal gain 2-4 ops each. Base pair costs of the uncovered targets are mostly 4-6.
- 16:20 E3 (s3.py class S3, e3_a.log): commit the best hub before the regret pick. practice_0004: baseline 121,
  hub_g 1 -> 127 (22 hubs), hub_g 2 -> 126 (9 hubs). Negative: the nominal gain is not real, because building
  the first served target already puts equivalent intermediates into the pool; the hub op is extra.
  (practice_0001 run killed for time: 3x slower.)
- a1_marg.py (store analysis): other>=20 marginal cost per target by bits: 20b 4.0, 30b 4.9, 40b 5.8, 50b 7.4,
  55b 9.2, 60b 9.8. Targets >= 45 bits are 21% of the targets and 31% of the ops. Independent: 6.3 at 40 bits,
  7.9 at 50, 9.9 at 60; first three targets average 12.3 ops, steady state 7.7-8.0.
  So the "other" targets are being paid at about the price of random numbers of the same size.
- core3.c: enum_pairs (all pair plans of a given cost for a target). 40-bit pool: 4-6 plans at the minimal
  cost 6, 14-88 plans at cost 7.
- 16:35 E4 (s3.py class LA, e4_la.log): plan-choice lookahead. Target picked by regret score; all its minimal-cost
  pair plans are enumerated (up to la_k 6), each is scored by the sum of the other targets' pair costs with the
  plan's new values added to the pool. practice_0005 (independent 40 bit) 212 -> 209 at 3.5x time; practice_0004
  (shared 42 bit) 121 -> 119 at 2.8x time. About -1.5%: within run-to-run noise of restarts, not better per CPU.
- 16:40 E5 (e5_rootpeel.py, e5_a.log): root peel t = v op r, v in pool, r of pair cost <= 7 (a cost-8 shape the
  partial L4 cannot see) on the dumped 60-bit pool: 0 hits in 3 targets x 92 residuals; engine gives 10 via the
  window search. Killed (also kills "target = covered target +- cheap value" for independent targets).
- 16:50 coordinator note (agent B): structured targets are where the gap is; forced seeding of mined hubs is
  hit-or-miss; asked for "soft seeding". E3 above is a soft-seeding test with a naive criterion (negative).
- 17:00 E7 (e7_grepair.py, e7b_diag.py): "global repair" = repair.improve with the 150-value threshold lifted, so
  each target is re-derived from ALL values of the program with L2 over the whole history.
  practice_0005 (212 ops): thr 150 -> 212, lifted -> 205 (-3.3%, 43 s).
  Diagnosis: L2 over the 210 history values has 6.3M entries, and EVERY target has a cost <= 5 pair plan on it
  (cost 6-7 on the 46-value pool). The saving is small only because private subtrees are small (avg 4.2 ops,
  the rest of each target's 6.6 marginal ops is reused by later targets).
  The base program really needs only 36 registers at its peak (mean 24.8): half of the 46-value pool is never
  used again. => KEY IDEA "wide pool": search over a much larger leaf set than 46 values and fix register
  feasibility afterwards (liveness), instead of capping the search pool at 46.
- 17:15 E8 (e8_big.py, class BIG): wide pool, no eviction. practice_0005: 212 -> 199-205 ops, needs 49-54
  registers. L2 over the history grows only like h^2.6 and a cost-5 plan needs L2 >= 2-6M (history >= 150-200
  values), which a 30-target puzzle reaches only at its very end. So the wide pool is worth 3-6% at best on
  40-bit independent, and needs a width-repair step. Parked; the cheap part of it is survivor S2 below.
- 17:25 r1_grepair_scored.py (survivor S2, global repair with the threshold lifted): first 8 scored independent
  40-bit puzzles: 8 of 8 improved, 1569 -> 1530 (-2.5%), 423 s, 332 ops/CPU-h. Stored with tag r3C.
- 17:30 E9 (e9_peel.py): chain plans t = (y op v1) op v2, y in L<=3, v1/v2 in L<=1 (core3.c peel2). These are the
  tree shapes with a deep subtree next to a cheap leaf ((4,0), (5,0), (4,1) ...), invisible to the pair search,
  and exactly the shape of structured targets M*(B+a)+c. Snapshots during normal solves: practice_0004 11 of 98
  target-states cheaper (each by 1), practice_0017 28 of 196 cheaper (35 ops), mostly the expensive targets
  (7 -> 6, 9 -> 7). Peel time 1-2 s per round.
- 17:40 E10 (class PEEL: regret ordering on min(pair, peel)): 0004 121 -> 117, 0017 158 -> 155, 0001 137 -> 130,
  0015 214 -> 200. Total 630 -> 602 (-4.4%), 4 of 4 better, 1.5-1.9x time.
- 17:55 E11: three-leaf chains (peel3_full, peel3_amc in core3.c; cfg peel3=8): 0015 199 vs 200 with peel2 only,
  0017 155 vs 155, 1.4x slower. No gain over two-leaf chains. Off by default.
- 18:05 E12a (e12_a.log) PEEL vs base on small-bit structured practice puzzles (33-42 bit): 0014 131/130,
  0020 99/98, 0021 124/124, 0022 131/129, 0026 142/142, 0036 123/123: 750 -> 746 (-0.5%). The chain search only
  matters when pair costs are >= 6, so the gain is concentrated in puzzles with targets >= 45 bits.
- 18:10 rel.py: relation-chain search with a hidden hub Z: t = (Z +- d)*M2 +- c2, Z = (v -+ c1)/M1, v in the
  pool, constants = small integers in L0..L2. Meet in the middle on reverse closures (numpy). Unit-tested on a
  constructed 0015 pool (plans valid). Integrated as cfg rel=1 in class PEEL. Being measured.
- 18:20 E12b: PEEL on 0027 (shared 59 bit) 189 -> 184, 0032 (mult 47 bit) 129 -> 128; peel_protect (keep chain
  bases from eviction) identical results: no effect.
- 18:35 E13/E14/E15 relation chains (rel.py): PEEL+rel on 0015/0017 = PEEL alone (200/155), rel fired once.
  Diagnostic with max cost 10: almost no relation exists between pool values and uncovered targets; covered
  targets are evicted fast (13 of 14 gone by round 15) and indexing them anyway yields only 3-4 targets at
  cost 8 late in the run. Keeping the 10 most recent covered targets in the pool: 207/159 (worse than 200/155).
  Killed: single-level M*(Z+d)+c relations between available values are too rare/expensive in this family;
  the hubs are deeper than one step from any target (agent B's oracle needed the true hubs AND constants).
- 18:50 r2_peel_scored.py (S1 on scored other>=20 puzzles, single PEEL run + global repair vs the store, which is
  the best of 2-3 configs + repair): sample A (>= 50 bit, 3 puzzles): 247/252, 197/195, 223/216;
  sample B (>= 47 bit, 6 puzzles): 189/173, 156/162, 145/150, 173/167, 158/153, 131/132.
  9 puzzles: 5 improved, accepted gain 36 of 1619 ops (2.2%), about 110-160 ops/CPU-h under 3-agent load.
  A single PEEL run is 1.0% shorter than the stored best-of-several on these 9. All gains stored (tag r3C).
- 19:00 E16 peel_deep (chain plans with the inner value in the partial deep levels, deep tier only): never
  fired on 0027 (184) or 0018 (independent 60 bit, 339 = baseline). Killed.
- 19:05 r1_b.log: S2 on 14 more scored independent 40-bit puzzles: 14 of 14 improved, 2739 -> 2669 (-2.56%),
  366 ops/CPU-h. Running total for S2: 22 puzzles, 109 ops stored.
- 19:10 E6 (e6_poolprofile.py, e6_a.log) pool profile at 40 bits, mean exact pair cost of 40 targets:
  organic mid-solve pool 6.70 (L3 7.4M); 46 random values log-uniform to 2^20 or 2^28: 6.62-6.65;
  log-uniform to 2^12: 7.6; to 2^40: 6.83; 46 random values uniform below 2^20 (all about half the target
  length): 5.90 (L3 19.8M; 33 targets at cost 6, 4 at 5, 1 at 4); uniform below 2^30 or 2^40: 8.0.
  So the pool profile is worth up to 0.8 op per target at 40 bits: the best pool is 46 unrelated values of
  about half the target bit length. Organic pools are made of correlated intermediates.
  Follow-up E17: diversity eviction (class DIV).
- 19:20 E17 diversity eviction (class DIV): 0005 212 -> 211, 0013 205 -> 201 with max-|L2|; with the
  target-range density metric 218 / 215 (worse). Weak.
- 19:30 S2 scope check: other>=20 puzzles >= 45 bits (4 puzzles) gain 1 op; independent 48-52 bit (4 puzzles)
  gain 3 ops. S2 pays only on independent puzzles <= 44 bits (81 puzzles, 15.7k ops).
- Store after this session: 162,003 (was 162,152; 30 puzzles carry tag r3C, 149 ops).

## Survivors

### S1 - chain ("peel") plans in the regret loop (class PEEL in s3.py, peel2 in core3.c)
What: besides pair plans t = u op w (both subtrees <= cost 3), search t = (y op1 v1) op2 v2 with y in L0..L3
and v1, v2 in L0/L1. Tiers: (v1 L0, v2 L0, all op forms), (L1, L0, all), (L0, L1, all), (L1, L1, only
t = y*v1 +- v2 and t = (y +- v1)*v2). Cost = lev(y) + lev(v1) + lev(v2) + 2. Only for targets whose pair
cost is >= 6 or unknown. Regret score uses min(pair, chain).
Measured: practice structured puzzles 1698 -> 1660 (-2.2%; 9 better, 3 equal, 0 worse of 12);
by size: >= 47 bit (0001, 0015, 0017, 0027, 0032): 827 -> 797 (-3.6%); <= 42 bit (7 puzzles): 871 -> 863 (-0.9%).
Scored, single run + global repair against the stored best-of-several: 9 puzzles, 36 ops (2.2%), 5 improved,
110-160 ops/CPU-h with three agents sharing the machine.
Projection: other>=20 with targets >= 47 bits = 384 puzzles, 71.1k ops: 2-3% => 1.4k-2.1k ops in about 12-15
CPU-h; other>=20 below 47 bits (251 puzzles, 32.4k ops): about 0.5% => 0.15k. A second alpha (0.15) should add
about a third more (as g2 did over g1).
Spec for src/: (1) copy peel2 + inv_forms from core3.c into core.c; (2) inside regret_pick, after the pair
loop for cost c, for alive targets try the chain tiers whose minimal cost is <= c (so the breadth-first
score-bound pruning also covers chains; this removes the Python all_costs loop and most of the 1.5x time);
(3) emit steps as expand(y), expand(v1), step, expand(v2), step; (4) Cfg.peel=1, on for all other>=20 runs.
Off: three-leaf chains, deep-tier chains, base protection, relation chains (all measured without gain).

### S2 - global repair (lift repair's 150-value threshold)
What: repair.improve with self_thr = 1e9, i.e. every target is re-derived with L2 over ALL values of the
finished program (6M+ L2 entries from about 210 values; cost <= 5 plans exist for every 40-bit target), the
private subtree is dropped when the new plan is shorter and regalloc still fits.
Measured on scored independent puzzles with targets <= 44 bits: 22 of 22 improved, 4308 -> 4199 (-2.5%),
30-60 s per puzzle, about 350 ops/CPU-h. Nothing on >= 48-bit puzzles (0.1-0.3%).
Projection: 81 puzzles, 15.7k ops => about 400 ops in total, 109 already stored, about 290 left in 1 CPU-h.
Spec: python3 research/round3/ideas/r1_grepair_scored.py 44 100 'indep>=20' 22 (resumes after the 22 done),
or set --thr 100000 in src/repair_store.py and restrict --ids to independent puzzles <= 44 bits; then apply
the same pass after any new independent run.

### Not survivors but worth knowing
- The finished programs need only 36 registers at the peak (mean 25) although the search pool is capped at
  46: half of the pool is never used again. A wide-pool search (E8) turns that slack into 3-6% at 40 bits but
  needs 49-54 registers as prototyped; it would need a liveness-aware plan choice. Estimated at most 0.5-0.9k
  ops (independent <= 44 bits only), with real engineering. Not recommended before the generator work.
- Pool profile: 46 unrelated half-length values give 5.9 instead of 6.7 mean pair cost at 40 bits, but they
  cost 1 op each to make; no policy tested (eviction by |L2|, lookahead on plan choice) recovers more than 1.5%.
## Idea table (25 ideas)

| # | Idea | Test | Result | Verdict |
|---|---|---|---|---|
| 1 | Hub seeding: commit a cheap value u (L1/L2) that lowers the pair cost of 2+ targets | E1 stats on 3 independent pools, E2/E3 on shared/mult | independent: 0 hubs exist; structured: hubs exist (gain 2-4 nominal) but committing them is worse (121 -> 126/127) | killed (the first served target already creates the same intermediates) |
| 2 | L3 x L3 hubs / DAG sharing between two random targets | density estimate from measured level sizes (L2 130-210k, L3 7-19M) | about 2 candidate hubs per round at 40 bits, none at 50-60; saving at most 1-2 ops per round | killed as a separate mechanism, folded into 3 |
| 3 | Plan-choice lookahead: among all minimal-cost plans of the chosen target, take the one that lowers the other targets' costs most | E4, class LA, plan enumeration in C (enum_pairs) | 212 -> 209 and 121 -> 119 (-1.5%) at 3x time; with 3 targets x 6 plans too slow to finish | weak: equal to a restart per CPU, not a survivor |
| 4 | Chain ("peel") plans t = (y op v1) op v2: tree shapes with a deep subtree next to cheap leaves, invisible to the pair search | E9 snapshots, E10/E12 class PEEL on 12 structured practice puzzles | 11-14% of target-states cheaper; solves 1698 -> 1660 (-2.2%), 9 better 3 equal 0 worse; -3% to -6.5% on puzzles with targets >= 47 bits, -0.5% below 43 bits; 1.3-1.9x time | SURVIVOR S1 |
| 5 | Three-leaf chains t = ((y op v1) op v2) op v3 and (y +- a)*m +- c | E11 | 199 vs 200, 155 vs 155; 1.4x slower | killed (two leaves are enough) |
| 6 | Relation chains through a hidden hub: t = (Z +- d)*M2 +- c2, Z = (v -+ c1)/M1, v in pool ("soft seeding": the hub is built only inside a plan that is cheaper than the alternatives) | rel.py, E13-E15 | PEEL+rel = PEEL alone (200/155); with max cost 10 almost no relation exists; keeping covered targets as partners is worse (207/159) | killed |
| 7 | Root peel at 60 bits: t = v op r, r of pair cost <= 7 (also "target = covered target +- cheap") | E5 on the dumped 60-bit pool | 0 hits in 276 residuals | killed |
| 8 | Global repair: re-derive every target from ALL values of the finished program (L2 over the whole history), i.e. lift repair's 150-value threshold | E7, r1 on 8 scored puzzles | 8 of 8 improved, -2.5% (39 ops), 332 ops/CPU-h, validated and stored | SURVIVOR S2 (small, cheap, ready) |
| 9 | Wide pool: search over the whole history (no eviction), fix register feasibility afterwards | E8 class BIG; liveness measurement | base program really needs only 36 registers (mean 25); wide pool 212 -> 199-205 but needs 49-54 registers; cost 5 needs a history >= 150-200 values | parked: 3-6% on 40-bit independent only, needs width repair; S2 captures the cheap part |
| 10 | DAG reuse inside one target (x op x, (x+c)*x ...) | counting: adds about 3M values to an L3 of 1e7-1e8 at 40+ bits | < 0.01 op per target | killed analytically |
| 11 | Eviction that protects chain bases (values y with t = (y op v1) op v2 for an uncovered t) | cfg peel_protect on 2 puzzles | 184 vs 184, 128 vs 128 | killed (no effect) |
| 12 | Eviction by usefulness / covered targets last (class EV) | rel_keep 10 (keep the 10 latest covered targets): 207/159 vs 200/155 | worse | killed |
| 13 | Pool profile: is a designed 46-value pool better than the organic one | E6: organic pool 6.70 mean pair cost vs 5.90 for 46 unrelated half-length values; E17 diversity eviction (class DIV, keep the max-|L2| subset) | 212/205 -> 211/201 (-1.2%); target-range density metric worse (218/215); building pool values on purpose costs 1 op each for about 0.4 op back | weak: eviction cannot make the pool unrelated; killed as a lever |
| 14 | Simulated annealing / late acceptance over target order | prior evidence: best of 16-30 random orders no better than regret; beam 4 equal per CPU | - | killed on prior evidence |
| 15 | LNS on the DAG: remove k targets and rebuild jointly | E7b diagnosis: private subtrees are only 4.2 ops of a 6.6 marginal cost, the rest is shared | a k-target rebuild can only win what S2 wins unless the shared part is rebuilt too | folded into S2 |
| 16 | Prefix-sharing restarts / limited-discrepancy on the regret order | estimate from the measured 2-4% run-to-run spread | best of 10 about -3% at 10x CPU: about 25 ops/CPU-h | not competitive per CPU; keep as the last filler |
| 17 | Exact search (SAT/ILP/BFS) for small puzzles | size of the prize: other<20 + indep<20 = 5.4k ops total, beam already applied | < 0.3k possible | deprioritised |
| 18 | Slack between targets and value_limit (build a multiple, divide down) | level sets already hold every value <= limit, division is in the pair search | nothing new to gain | killed |
| 19 | Subtraction/division with large pool values | same as 18: all four ops are in every level | - | killed |
| 20 | Covered targets as operands | they are pool values (L0) already; chains from them are idea 6 | - | folded into 6 |
| 21 | Soft seeding of small constants | analysis: a constant used by k chain plans is paid once by the first plan and is free afterwards, so seeding it separately cannot beat the greedy; what was missing was the visibility of chain plans (idea 4) | - | folded into 4 and 6 |
| 22 | Register choice as part of the search | liveness: 36 of 48 registers used at the peak, registers are not binding for the current search | - | killed as a lever (see 9) |
| 23 | Peephole superoptimiser on windows of the final program | subsumed by S2 (S2 re-derives each target from everything) | - | folded into S2 |
| 24 | Learning a potential (sum of remaining pair costs) to guide choices | this is the objective used in idea 3 | -1.5% | weak |
| 25 | Exhaustive X*Y +- Z at 60 bits | prior: one extra cost-9 plan in 8 targets at 10x budget | about -0.6k | unchanged, low priority |
| 26 | Chain plans in the deep tier (inner value in the partial deep levels) | E16 peel_deep | never fired on a 59-bit shared and a 60-bit independent puzzle | killed |

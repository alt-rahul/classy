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

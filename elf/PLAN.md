# Elf Calculator: solver plan (from the Fable planning agent)

## Bottom line
- One unified engine for all families: a greedy "cheapest target next" loop over a live pool of register values, using level sets of cheap values plus a divmod template `t = X*Y ± Z`. No family classifier is needed.
- Expected about 205k from the main algorithm (baseline 1,490,379), with refinements plausibly reaching 160-185k.
- Numbers come from a numpy prototype on 17 practice puzzles (hard tier) using a tree-cost model with the pool capped at 40 values plus starts. It did not emit or validate programs, so treat them as estimates until the validator confirms them.
- The trivial bound of 26,898 is not reachable: nothing is within 3 ops of the starts on hard puzzles.

## 1. Families
- `lower_bound` in practice = number of targets not in starts; carries no info.
- `max(target)/value_limit` in [0.25, 1.0] for every family: no headroom, never plan through values above the largest target.
- Tooling: Python 3.13, numpy 2.5.3, gcc and clang. No numba, no sympy.
- Scored split (`max(starting_values) > 40` separates independent; other families have starts <= 34):

| Class | Puzzles | Targets | Baseline ops | Baseline ops/target |
|---|---|---|---|---|
| Independent, >=20 targets | 215 | 6,450 | 462,738 | 72 |
| Independent, <20 targets | 27 | 359 | 15,350 | 43 |
| Other (shared / mult / mixed), >=20 targets | 635 | 19,068 | 986,048 | 52 |
| Other, <20 targets | 107 | 980 | 26,077 | 27 |
| Tiny (limit <= 16) | 16 | 45 | 166 | 3.7 |

- Independent: starts up to ~2000, targets within one decade, random-looking. Cost governed by information content (avg 46.7 bits).
- Shared: twin targets differ by < ~1000; many targets are Horner-related to an earlier target, `t = q*(u ± a) ± b` with small q, a, b (e.g. 5375661482646 = 63*(85327958960 + 1082)); squares of hidden values occur; a third to a half have no small relation to another target.
- Multiplicative: large shared gcds, squares of hidden factors, one target 2x another; shapes like 9*(884167 - 8). Cheapest family.
- Mixed: about half independent-like 57-60-bit targets near the limit, rest shared/multiplicative-style.
- Tiny: limit 7-16, 1-4 targets. Solve exactly.

## 2. Achievable (prototype, hard tier)

| Family | Puzzles | Ops/target | Range |
|---|---|---|---|
| Multiplicative | 5 | 4.68 | 3.45-5.67 |
| Shared | 5 | 7.17 | 4.55-9.17 |
| Mixed | 4 | 9.95 | 7.03-11.19 |
| Independent | 3 | 8.91 | 7.7-8.2 at 40 bits, 11.0 at 50 bits |

- 4-10 s per puzzle in pure numpy: ~2 CPU-hours per pass, ~30 min on 4 cores.
- Rule of thumb about 5 bits per op. Letting every intermediate join the pool is what brings 40-bit targets to 7-8 ops.
- Biggest improvable component: 60-bit random-like targets and unrelated shared targets hitting the fallback at 11-13 ops.

## 3. Core algorithm
State: `pool` = values currently in registers, capped at 40 (8 scratch). Record SSA steps `(op, a_val, b_val, result_val)`; assign registers afterwards.

Each round:
1. Level sets from the pool: F0 = pool; F1 = F0∘F0; F2 = F1∘F0, four ops, both operand orders. Keep positive, <= limit, exact division, strict subtraction. Sorted int64 array + min-cost + witness (op, a, b). Sizes at pool 40: F1 ~3-4k, F2 ~0.5-0.7M. Chunk outer products. Guard multiplication overflow with a float pre-check before exact int64 product (limit <= 1e18 < 2^63).
2. For each uncovered target, cheapest plan:
   - a. Direct lookup (cost <= 2).
   - b. `cost3`: for each pool value u, look up |t-u|, t+u, t/u, t*u in the table, +1.
   - c. Divmod template: for X in F0 ∪ F1 (F2 if time allows) with 2 <= X < t, Y = t // X, Z = t mod X. Variant 1: t = X*Y + Z. Variant 2: t = X*(Y+1) - (X-Z), needs X*(Y+1) <= limit. Cost c(X)+c(Y)+c(Z)+2; one fewer when Y = 1 or Z = 0. If exactly one of Y, Z is missing from the table, extend that side with `cost3` (catches `q*(u ± a) ± b`). Vectorise over X with searchsorted.
3. Commit the cheapest target (ties: smallest). Append ops, add intermediates and target to the pool, mark other covered targets.
4. If no finite plan: fallback on the smallest uncovered target. Horner recursion `t = X*Y + Z` with X in F0 ∪ F1 and Z of table cost <= 2, recursing on Y. Prototype picked X greedily by (ops spent)/log2(X). Build as a beam search (top 8-16 X per level, depth <= 6, score by total cost): highest-value upgrade.
5. Eviction when pool exceeds cap: never evict 1, starts, or small constants below ~64; else LRU non-targets first, then oldest targets.

Start-up: if 1 is absent, emit `["/",0,0,k]`. Pre-seeding extra constants is a restart parameter.

Value limit: enforced in level-set builder and in template variant 2; variant 1 has X*Y <= t automatically.

Register allocation: dead-code elimination first (drop ops whose result is neither a target nor used later), then linear scan on last-use; destination may reuse a source dying at that instruction; starting values can be freed. If max live > 48, re-run with smaller cap.

Tiny (limit <= ~20): exact BFS over bitmask of produced values in 1..limit. Optimal.

Improvement loop per puzzle, keep shortest valid: pool cap {32, 36, 40, 44}; target order; eviction policy; X from F2 or not; seed constants (none, {2,3}, a few powers); random tie-breaking.

## 4. Infrastructure
- Exact validator independent of the solver (`src/validate.py`, pure Python ints). Already confirmed to reproduce 1,490,379 on the sample.
- Best-solution store: one record per puzzle with program, length, config tag. Write only when validated and strictly shorter. Seed from the sample submission.
- CSV writer: reads store, re-validates all 1000, writes `id,program`, prints total and per-class breakdown.
- Runner: multiprocessing Pool(4), per-puzzle time budget, resumable.

## 5. Stages

| Stage | Work | Expected |
|---|---|---|
| 1 | Validator, store, writer, tiny BFS, regalloc + DCE, fallback alone (level sets + Horner per target, intermediates kept in pool) | 300-330k |
| 2 | Full greedy loop with template, cost3, beam fallback, 4 cores | ~205k (190-220k) |
| 3 | Refinements | 160-185k (guess) |

Stage 3 refinements, by value:
1. Multi-configuration restarts, best per puzzle (5-10%).
2. Beam fallback with X from F2, both template sides solved recursively.
3. Peephole re-derivation: for each value find a cheaper (<= 3 op) derivation from values live at that point; re-run DCE and validation.
4. Small C helper (gcc + ctypes) for exhaustive tree-cost search to cost 4-5 and F3 tables (~1e8 entries at pool 40); replace cost3, run template with Y, Z at cost <= 3.
5. Constant reuse in shared family: prefer plans whose small constants are already in the pool; protect frequent constants from eviction.

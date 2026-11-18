# Elf calculator solver

Straight-line programs over 48 registers that make every target value appear, scored by the
total instruction count over 1,000 puzzles (lower is better).

## How to run

All commands are run from `/home/user/classy/elf`.

```
gcc -O3 -march=native -shared -fPIC -o src/core.so src/core.c   # build the C engine
python3 src/run.py --tag s3 --workers 4 --cfg '{"pool_cap":46}'   # solve all puzzles (resumable)
python3 src/repair_store.py --workers 4                            # post-pass on stored programs
python3 src/write_sub.py submissions/submission.csv                # re-validate all, write CSV
python3 src/validate.py data/puzzles.csv submissions/submission.csv
```

`run.py` keeps a per-tag log (`store/runs_<tag>.json`) so an interrupted run resumes where it
stopped; `--redo` reruns everything, `--skip <tag>` skips puzzles done under another tag,
`--ids` / `--classes` restrict the set. The store `store/best.json` holds one validated program
per puzzle and is only ever replaced by a strictly shorter valid program (saves are merge-safe
under a file lock, so several runners can work at once). It was seeded from the sample submission.

Practice-set benchmarking: `src/bench.sh '<cfg json>'` (12 practice puzzles, 3 per family) or
`python3 src/test_practice.py <ids|family|all> '<cfg json>'`.

## Files

- `src/validate.py` - independent exact validator (provided, unchanged).
- `src/common.py` - puzzle loading, class labels, store load/merge-save/offer (validates every
  program before storing), submission writer (re-validates all 1,000), per-class breakdown.
- `src/core.c` / `src/core.so` - C engine (ctypes). Builds level sets L0..Lk of values reachable
  from the current register pool with exact tree cost k (L0..L2 with witnesses, L3 always
  complete, deeper levels partial under a candidate cap, values re-derived on demand), with
  blocked Bloom filters for membership. Searches: pair / meet-in-the-middle `t = u op w`
  (lookups or two-pointer scans), window `t = X*Y +- Z` with a per-combination budget and
  expected-hit pruning, Horner candidates `t = X*Y +- Z` with `Z = t mod X`.
- `src/engine.py` - ctypes wrapper.
- `src/solver.py` - greedy "cheapest target next" loop over a live pool of values, SSA by
  value; cheap tier (pairs on L<=2, optional lookahead), deep tier (fresh deep levels, pair search
  to cost 11, window search, Horner beam, guaranteed Horner fallback); LRU eviction that protects
  small constants; register allocation with dead-code elimination (`regalloc`).
- `src/tiny.py` - exact BFS over sets of produced values for tiny puzzles (limit <= 20).
- `src/repair.py` - ruin-and-recreate post-pass: drop a target's private subtree and rebuild the
  target from all values available before its first use, if strictly cheaper.
- `src/run.py` - parallel resumable runner. `src/repair_store.py` - parallel post-pass driver.
- `src/write_sub.py` - writes the submission from the store. `src/test_practice.py`,
  `src/bench.sh` - practice-set benchmarks.

## Score history

See the bottom of this file (filled in as stages complete).

# Fleet run spec - deep group rebuild on the big random-like puzzles (lever B)

Why: the public files do not change at all on independent 40/50/60-bit and mixed 50-bit puzzles (73k of the score);
public LNS is too shallow there. This engine is the only thing we have that shortens them.

What it does (grun.py): for a group A of 1-8 targets, remove the joint exclusive cone JC(A) (every value that only targets
of A depend on) and rebuild each target as a tree over ALL remaining values of the program (wide2: exact window /
chain search, floor = 6 new values at 50 bits, 7 at 60 bits). Among the enumerated solutions it takes the one whose
best insertion keeps the register peak lowest, then schedules to 48 registers with sched (annealing with segment
moves). Kept only if shorter and valid (common.check).

Build (repo root; python3 + gcc/g++ only):
  gcc -O3 -march=native -shared -fPIC -o src/core.so src/core.c
  g++ -O3 -march=native -o research/round4/beyond/wide2 research/round4/beyond/wide2.cpp
  g++ -O3 -march=native -o research/round4/beyond/sched research/round4/beyond/sched.cpp
Files needed: data/puzzles.csv, src/{common,validate,solver,engine}.py + core.so, store/best.json,
  research/round4/beyond/{wlib,dagutil,gcone,gcone2,jcone,grun,jrun,wrun}.py, fam.json, ids_*.txt
RAM 0.7 GB per worker. Resumable: runs_<TAG>.jsonl lists finished puzzles (delete it to redo on a new frontier).

Run (cwd research/round4/beyond):
  python3 -u grun.py <TAG> 8 <workers> <id lists, comma separated> 8 <groups per puzzle>
  e.g.  python3 -u grun.py g50 8 4 ids_i50.txt,ids_m50.txt,ids_s50.txt 8 10     (283 puzzles, 46-55 bits)
        python3 -u grun.py g60 8 4 ids_i60.txt,ids_m60.txt,ids_s60.txt 8 10     (254 puzzles, 56-62 bits)
  Shard by splitting the id lists (comma separated ids in a file); give every shard its own TAG.
Time: 40-70 s per group attempt, so about 8-12 minutes per independent/mixed puzzle at 10 groups; true shared puzzles
  finish in seconds (no group above the floor). Whole job about 60-90 core-hours.
Output: improved programs are offered to store/best.json and appended to improved_<TAG>.jsonl as [id, program]
  (one JSON per line; later lines for the same id are shorter). Merge remote results with common.offer.
Measured yield (local, 20 Nov): about 1 gain per 8 group attempts, 1-2 instructions each = 10-15 instructions per
  core-hour on puzzles where the public engines yield nothing. Expected 250-400 instructions for one pass over the
  537 puzzles; a second pass after other changes finds new groups.
Local runs already cover: ids_i50 + ids_m50 (tag g50) and ids_i60 + ids_m60 (tag g60), 6 groups per puzzle, in progress.

"""Diagnose global repair: for each target, |D|, level sizes over the whole history, best pair cost."""
import sys, time, pickle, os
from s3 import *
import repair
pid = sys.argv[1]
p = {q.id: q for q in load_puzzles('practice')}[pid]
fn = 'base_%s.pkl' % pid
if os.path.exists(fn): steps = pickle.load(open(fn, 'rb'))
else:
    prog, stats, steps = solve3(p, S3, pool_cap=46, evict_size=1, regret=0.1); pickle.dump(steps, open(fn, 'wb'))
steps = repair.dce(p.sv, steps, p.targets)
print('ops', len(steps))
# liveness of the base program
prog = regalloc(p.sv, steps, p.targets, p.nreg)
last = {}
for k, (op, a, b, r) in enumerate(steps): last[a] = k; last[b] = k
born = {v: -1 for v in p.sv}
for k, s in enumerate(steps): born.setdefault(s[3], k)
live = [sum(1 for v in born if born[v] <= k and last.get(v, -2) > k) for k in range(len(steps))]
print('live registers actually needed: max', max(live), 'mean %.1f' % (sum(live) / len(live)))
tset = set(p.targets) - set(p.sv)
vals, defs, nstart = repair._ssa(p.sv, steps)
first = {}
for i, d in enumerate(defs):
    v = vals[d[3]]
    if v in tset and v not in first: first[v] = i
tot = 0; n = 0
for t in sorted(tset)[::int(sys.argv[2]) if len(sys.argv) > 2 else 1]:
    k = first[t]
    D = repair.private_subtree(defs, k, set(first.values()) - {k})
    keep = list(dict.fromkeys([v for v in p.sv] + [vals[d[3]] for i, d in enumerate(defs) if i not in D and vals[d[3]] != t]))
    t0 = time.time()
    sz = engine.build(keep, p.limit, 2, 0)
    r = engine.find_pair(t, 5)
    tot += max(0, len(D) - (r[0] if r else len(D))); n += 1
    print('t %d bits %d |D| %d keep %d levels %s best %s  %.1fs' % (t, t.bit_length(), len(D), len(keep), sz, r[0] if r else None, time.time() - t0), flush=True)
print('sum of individual savings', tot, 'over', n, 'targets')

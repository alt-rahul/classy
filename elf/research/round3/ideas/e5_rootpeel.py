"""X3: root peel. t = v op r with v in the pool and r of pair cost <= 7 (total <= 8), against the
normal deep tier (search_round) on the dumped mid-solve pools."""
import sys, time, pickle
import eng3
from eng3 import engine
from common import *
from solver import Solver, Cfg
sys.path.insert(0, '/home/user/classy/elf/research')
def inv(t, s, limit):
    out = []
    if t > s: out.append(t - s)
    if t + s <= limit: out.append(t + s)
    if s > t: out.append(s - t)
    if s > 1:
        if t % s == 0: out.append(t // s)
        if t * s <= limit: out.append(t * s)
        if s % t == 0 and s // t > 1: out.append(s // t)
    return out
name = sys.argv[1]; nt = int(sys.argv[2])
d = pickle.load(open('/home/user/classy/elf/research/pool_%s.pkl' % name, 'rb'))
pool, ts, limit = list(d['pool']), sorted(d['ts']), d['limit']
class PP: pass
p = PP(); p.sv = d['sv']; p.targets = ts; p.limit = limit; p.nreg = 48
s = Solver(p, Cfg(pool_cap=46, evict_size=1, regret=0.1))
s.deep_age = None; s.evicted = False; s.n_l3 = 0
for t in ts[:nt]:
    t0 = time.time()
    r = s.search_round([t], pool, pool)
    t1 = time.time()
    engine.build(pool, limit, 3, 0)
    res = sorted(set(x for v in pool for x in inv(t, v, limit)))
    cs = eng3.all_costs(res, 7, None)
    known = [c for c in cs if c is not None]
    print(name, t.bit_length(), 'engine cost', r[0], r[3], '%.0fs' % (t1 - t0), '| root-peel best', (min(known) + 1) if known else None, 'n resid', len(res), 'n hits', len(known), '%.0fs' % (time.time() - t1), flush=True)
    s.deep_age = None

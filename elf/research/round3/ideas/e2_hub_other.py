"""Diagnostic: during a normal regret solve, every k-th round compute base costs of all uncovered
targets and scan for hubs (u in L1/L2, residual in L<=3)."""
import sys, time, collections
import eng3
from eng3 import engine
from common import *
from solver import Solver, Cfg, regalloc
P = {p.id: p for p in load_puzzles('practice')}
ids = sys.argv[1].split(','); every = int(sys.argv[2]) if len(sys.argv) > 2 else 4

class S(Solver):
    def regret_round(self, ts, pool_order, pool):
        self.rn = getattr(self, 'rn', 0) + 1
        if self.rn % every == 1:
            sz = engine.build(pool_order, self.p.limit, 3, 0)
            base = eng3.all_costs(ts, 7, 9)
            out = ['  rnd %d pool %d uncov %d L %s base %s' % (self.rn, len(pool), len(ts), sz[1:], ''.join(str(b) for b in base))]
            for lu, lw in ((1, 3), (2, 3)):
                hubs = eng3.hub_scan(ts, base, lu, lw, 1, 2)
                hubs.sort(key=lambda h: -h[1])
                out.append('   lu%d: n %d best %s' % (lu, len(hubs), [(h[1], h[2]) for h in hubs[:4]]))
            print('\n'.join(out), flush=True)
        return Solver.regret_round(self, ts, pool_order, pool)

for i in ids:
    p = P[i]
    t0 = time.time()
    s = S(p, Cfg(pool_cap=46, evict_size=1, regret=0.1))
    st = s.run()
    prog = regalloc(p.sv, st, p.targets, p.nreg)
    print(i, p.family, len(p.targets), 'bits', max(p.targets).bit_length(), 'len', len(prog), check(p, prog), '%.0fs' % (time.time() - t0), flush=True)

"""E9: do chain-shaped plans t = (y op v1) op v2 (y in L<=3, v1/v2 in L<=1) beat the pair search?
Sampled every k-th round of a normal regret solve."""
import sys, time, collections
from s3 import *
which = sys.argv[1]; ids = sys.argv[2].split(','); every = int(sys.argv[3])
P = {p.id: p for p in load_puzzles(which)}
AGG = collections.Counter()
class S(Solver):
    def regret_round(self, ts, pool_order, pool):
        self.rn = getattr(self, 'rn', 0) + 1
        if self.rn % every == 0 and len(pool) >= 20:
            engine.build(pool_order, self.p.limit, 3, 0)
            t0 = time.time()
            base = eng3.all_costs(ts, 7, 9)
            t1 = time.time()
            pc = []
            for t, b in zip(ts, base):
                r = eng3.peel(t, min(b - 1, 7))
                pc.append(r[0] if r else b)
            t2 = time.time()
            sav = [b - c for b, c in zip(base, pc)]
            AGG['targets'] += len(ts); AGG['improved'] += sum(1 for s in sav if s > 0); AGG['saved'] += sum(sav); AGG['base'] += sum(base)
            AGG['t_pair'] += t1 - t0; AGG['t_peel'] += t2 - t1
            print('  rnd %d pool %d  pair %s\n               peel %s  (pair %.1fs peel %.1fs)' % (self.rn, len(pool), ''.join(str(b) for b in base), ''.join(str(c) for c in pc), t1 - t0, t2 - t1), flush=True)
        return Solver.regret_round(self, ts, pool_order, pool)
for i in ids:
    p = P[i]
    s = S(p, Cfg(pool_cap=46, evict_size=1, regret=0.1)); st = s.run()
    print(i, p.family, len(p.targets), max(p.targets).bit_length(), 'ops', len(regalloc(p.sv, st, p.targets, p.nreg)), dict(AGG), flush=True)

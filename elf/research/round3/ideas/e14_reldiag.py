"""Snapshot diagnostic for the relation search: pair / peel / rel costs of the uncovered targets."""
import sys, time
from s3 import *
import rel
pid = sys.argv[1]; every = int(sys.argv[2]); cmax = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
p = {q.id: q for q in load_puzzles('practice')}[pid]
class S(PEEL):
    def regret_round(self, ts, pool_order, pool):
        self.rn = getattr(self, 'rn', 0) + 1
        if self.rn % every == 0:
            engine.build(pool_order, self.p.limit, 3, 0)
            base = eng3.all_costs(ts, 7, None)
            t0 = time.time(); R = rel.Rel(self.p.limit, cmax); cov = [x for x in self.p.targets if x not in set(ts)]; R.index_pool(list(dict.fromkeys(list(pool_order) + cov))); print("   covered targets not in pool:", len([x for x in cov if x not in set(pool)]), "of", len(cov)); t1 = time.time()
            out = []
            for t, b in zip(ts, base):
                pl = eng3.peel(t, 7); r = R.best(t, 10)
                out.append('%s/%s/%s' % (b if b else '-', pl[0] if pl else '-', r[0] if r else '-'))
            print(' rnd %d pool %d consts %d index %d (%.1fs + %.1fs)  pair/peel/rel: %s' % (self.rn, len(pool), len(R.cv), len(R.PZ), t1 - t0, time.time() - t1, ' '.join(out)), flush=True)
        return PEEL.regret_round(self, ts, pool_order, pool)
s = S(p, Cfg(pool_cap=46, evict_size=1, regret=0.1)); st = s.run()
print(pid, len(regalloc(p.sv, st, p.targets, p.nreg)), dict(s.stats))

"""Exp 1a: randomised restarts of the PEEL solver (agent C's chain plans) on selected practice puzzles.
usage: rs.py ids nruns out.jsonl   (round-robin over ids; run 0 = deterministic baselines)"""
import sys, time, json, random, pickle, os
sys.path.insert(0, '/home/user/classy/elf/research/round3/ideas')
sys.path.insert(0, '/home/user/classy/elf/src')
import eng3
from eng3 import engine
from common import *
from solver import Solver, Cfg, regalloc, simulate
from s3 import PEEL

class RP(PEEL):
    """PEEL with score noise (cfg.sig) and per-round alpha jitter (cfg.ajit)."""
    def regret_round(self, ts, pool_order, pool):
        cfg, limit = self.cfg, self.p.limit
        if len(pool_order) < getattr(cfg, 'peel_minpool', 10):
            return Solver.regret_round(self, ts, pool_order, pool)
        engine.build(pool_order, limit, 3, 0)
        self.n_l3 += 1
        self.deep_age = None
        base = eng3.all_costs(ts, cfg.regret_maxc, None)
        a = cfg.regret + cfg.ajit * (2 * self.rng.random() - 1)
        best = None
        for t, b in zip(ts, base):
            plan = None; c = b
            if c is None or c >= 6:
                r = eng3.peel(t, min(7, (c - 1) if c is not None else 7), eng3.TIERS)
                if r is not None:
                    c, plan = r
            if c is None:
                continue
            key = (c - a * t.bit_length() + cfg.sig * self.rng.random(), c, t)
            if best is None or key < best[0]:
                best = (key, c, t, plan)
        if best is None:
            return Solver.regret_round(self, ts, pool_order, pool)
        _, c, t, plan = best
        if plan is None:
            r = engine.find_pair(t, c)
            return (r[0], t, r[1], 'reg')
        return (c, t, plan, 'peel')

def run(p, cls, **kw):
    base = dict(pool_cap=46, evict_size=1, regret=0.1, sig=0.0, ajit=0.0)
    base.update(kw)
    s = cls(p, Cfg(**base))
    st = s.run()
    prog = regalloc(p.sv, st, p.targets, p.nreg)
    if prog is None or check(p, prog):
        return None, st
    return prog, st

if __name__ == '__main__':
    P = {p.id: p for p in load_puzzles('practice')}
    ids = sys.argv[1].split(','); N = int(sys.argv[2]); out = sys.argv[3]
    best = {}
    bd = out + '.best'
    os.makedirs(bd, exist_ok=True)
    rng = random.Random(12345)
    for k in range(N):
        for i in ids:
            p = P[i]
            if k == 0:
                variants = [('base', Solver, {}), ('peel', PEEL, {})]
            else:
                kw = dict(seed=1000 * k + 7, regret=rng.choice([0.05, 0.08, 0.1, 0.12, 0.15, 0.2]),
                          sig=rng.choice([0.0, 0.3, 0.6, 1.0]), ajit=rng.choice([0.0, 0.05]))
                variants = [('rp', RP, kw)]
            for name, cls, kw in variants:
                t0 = time.time()
                try:
                    prog, st = run(p, cls, **kw)
                    n = len(prog) if prog else None
                except Exception as e:
                    n = None; prog = None; print('ERR', i, e, flush=True)
                rec = dict(id=i, k=k, name=name, kw=kw, n=n, t=round(time.time() - t0, 1))
                open(out, 'a').write(json.dumps(rec) + '\n')
                if n and (i not in best or n < best[i]):
                    best[i] = n
                    pickle.dump((prog, st, rec), open('%s/%s.pkl' % (bd, i), 'wb'))

"""two-stage: stage A builds seeds (as a puzzle), stage B solves the targets with the seeds as extra starts.
two.py pid cls name seeds [cfg]  -> total validated ops on the original puzzle"""
import sys, json, time, pickle, copy
from rs import *
def two(p, seeds, cls=PEEL, **kw):
    base = dict(pool_cap=46, evict_size=1, regret=0.1, sig=0.0, ajit=0.0); base.update(kw)
    a = copy.copy(p); a.targets = sorted(set(seeds))
    sa = cls(a, Cfg(**base)).run()
    b = copy.copy(p); b.sv = list(p.sv) + [s for s in sorted(set(seeds)) if s not in p.sv]
    sb = cls(b, Cfg(**base)).run()
    st = sa + sb
    prog = regalloc(p.sv, st, p.targets, p.nreg)
    return prog, st, len(sa), len(sb)
if __name__ == '__main__':
    P = {p.id: p for p in load_puzzles('practice')}
    p = P[sys.argv[1]]; cls = {'PEEL': PEEL, 'RP': RP, 'S': Solver}[sys.argv[2]]; name = sys.argv[3]
    seeds = [int(x) for x in sys.argv[4].split(',') if x]
    kw = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
    t0 = time.time(); prog, st, na, nb = two(p, seeds, cls, **kw)
    ok = prog is not None and check(p, prog) is None
    print(name, sys.argv[2], 'seeds', len(seeds), 'A', na, 'B', nb, 'total', len(prog) if prog else None, ok, '%.0fs' % (time.time() - t0), flush=True)
    pickle.dump((prog, st), open('two_%s_%s.pkl' % (p.id, name), 'wb'))

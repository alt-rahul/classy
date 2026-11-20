"""oracle runs: orc.py pid cls name extras(comma) [json cfg]; extras added as targets, final regalloc on original targets (DCE)."""
import sys, json, time, pickle
from rs import *
def orc(p, extra, cls=PEEL, **kw):
    import copy
    q = copy.copy(p); q.targets = sorted(set(p.targets) | set(extra))
    base = dict(pool_cap=46, evict_size=1, regret=0.1, sig=0.0, ajit=0.0); base.update(kw)
    s = cls(q, Cfg(**base)); st = s.run()
    prog = regalloc(p.sv, st, p.targets, p.nreg)
    assert prog is not None and check(p, prog) is None
    return prog, st
if __name__ == '__main__':
    P = {p.id: p for p in load_puzzles('practice')}
    p = P[sys.argv[1]]; cls = {'PEEL': PEEL, 'RP': RP, 'S': Solver}[sys.argv[2]]; name = sys.argv[3]
    extra = [int(x) for x in sys.argv[4].split(',') if x]
    kw = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
    t0 = time.time(); prog, st = orc(p, extra, cls, **kw)
    print(name, sys.argv[2], 'extra', len(extra), 'ops', len(prog), '%.0fs' % (time.time() - t0), kw, flush=True)
    pickle.dump((prog, st), open('orc_%s_%s.pkl' % (p.id, name), 'wb'))

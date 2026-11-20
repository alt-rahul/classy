"""X-wide: solve with an unlimited pool (no eviction) and report ops and the register width needed."""
import sys, time, json, copy, pickle
from s3 import *
which = sys.argv[1]; ids = sys.argv[2].split(',')
kw0 = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
P = {p.id: p for p in load_puzzles(which)}
for i in ids:
    p = P[i]
    q = copy.copy(p); q.nreg = 100000
    t0 = time.time()
    kw = dict(pool_cap=100000, evict_size=1, regret=0.1); kw.update(kw0)
    s = BIG(q, Cfg(**kw)); st = s.run(); simulate(p.sv, st, p.limit)
    w, n = width(p.sv, st, p.targets)
    pickle.dump(st, open('big_%s.pkl' % i, 'wb'))
    print(i, p.family, len(p.targets), max(p.targets).bit_length(), 'wide ops', n, 'width', w, '%.0fs' % (time.time() - t0), dict(s.stats), flush=True)

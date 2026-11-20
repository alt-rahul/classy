import sys, time, pickle, glob
import eng3
from eng3 import engine
for fn in sorted(glob.glob('/home/user/classy/elf/research/pool_practice_*.pkl')):
    d = pickle.load(open(fn, 'rb'))
    pool, ts, limit = list(d['pool']), sorted(d['ts']), d['limit']
    t0 = time.time(); sz = engine.build(pool, limit, 3, 0); tb = time.time() - t0
    t0 = time.time(); base = eng3.all_costs(ts, 7, 9); tc = time.time() - t0
    print(fn.split('/')[-1], 'bits', ts[0].bit_length(), ts[-1].bit_length(), 'levels', sz, 'build %.1fs costs %.1fs' % (tb, tc))
    print(' base costs', base, 'sum', sum(base))
    for lu, lw in ((1, 3), (2, 3), (2, 2)) + (((3, 2),) if len(sys.argv) > 1 else ()):
        t0 = time.time()
        hubs = eng3.hub_scan(ts, base, lu, lw, 1, 2)
        hubs.sort(key=lambda h: -h[1])
        import collections
        print('  lu', lu, 'lw', lw, 'n hubs', len(hubs), 'gain hist', sorted(collections.Counter(h[1] for h in hubs).items()), 'nserve hist', sorted(collections.Counter(h[2] for h in hubs).items()), '%.1fs' % (time.time() - t0))
        for h in hubs[:3]:
            print('     u', h[0], 'gain', h[1], 'serves', h[2], [(i, eng3.resid_lev(ts[i], h[0], lw)) for i in range(len(ts)) if h[3] >> i & 1])

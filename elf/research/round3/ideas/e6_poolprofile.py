"""X16: which pool profile makes random targets cheap? Synthetic pools of 46 values (log-uniform up to
2^maxbits, optionally a few small constants) vs the dumped mid-solve pool, on the dump's own targets
plus random targets of the same size. Metric: mean exact pair cost (<= 7, else 8)."""
import sys, time, pickle, random
import eng3
from eng3 import engine
name = sys.argv[1]
d = pickle.load(open('/home/user/classy/elf/research/pool_%s.pkl' % name, 'rb'))
pool, ts, limit = list(d['pool']), sorted(d['ts']), d['limit']
rng = random.Random(1)
lo, hi = min(ts), max(ts)
T = ts + [rng.randrange(lo, hi) for _ in range(40 - len(ts))]
def ev(pl, label):
    t0 = time.time()
    sz = engine.build(pl, limit, 3, 0)
    cs = eng3.all_costs(T, 7, 8)
    import collections
    print('%-28s L1 %d L2 %d L3 %d  mean cost %.2f  hist %s  %.0fs' % (label, sz[1], sz[2], sz[3], sum(cs) / len(cs), sorted(collections.Counter(cs).items()), time.time() - t0), flush=True)
ev(pool, 'dumped pool')
print(' pool bits:', sorted(v.bit_length() for v in pool))
tb = hi.bit_length()
for mb in [int(x) for x in sys.argv[2].split(',')]:
    for seed in range(2):
        r = random.Random(100 * mb + seed)
        pl = set()
        while len(pl) < 46:
            b = r.uniform(1, mb)
            pl.add(max(2, int(2 ** b)))
        ev(sorted(pl), 'loguniform to 2^%d s%d' % (mb, seed))
# uniform-in-value pools
for mb in (20, 30, tb):
    r = random.Random(mb)
    pl = set(r.randrange(2, 2 ** mb) for _ in range(46))
    ev(sorted(pl), 'uniform to 2^%d' % mb)

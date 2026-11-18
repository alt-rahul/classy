import sys, time, json
sys.path.insert(0, '/home/user/classy/elf/src')
from common import *
from solver import solve, Cfg
import collections
P = load_puzzles('practice')
sel = sys.argv[1] if len(sys.argv) > 1 else 'all'
kw = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
if sel == 'all':
    ps = [p for p in P if p.family != 'tiny']
elif sel.startswith('every'):
    k = int(sel[5:])
    ps = [p for p in P if p.family != 'tiny'][::k]
else:
    ids = sel.split(',')
    ps = [p for p in P if p.id in ids or p.family in ids]
agg = collections.defaultdict(lambda: [0, 0, 0, 0.0])
T0 = time.time()
allst = collections.Counter()
for p in ps:
    t0 = time.time()
    prog, stats = solve(p, Cfg(**kw))
    allst.update(stats)
    err = 'regalloc fail' if prog is None else check(p, prog)
    dt = time.time() - t0
    a = agg[p.family]; a[0] += 1; a[1] += len(p.targets); a[2] += len(prog) if prog else 0; a[3] += dt
    print(p.id, p.family, len(p.targets), len(prog) if prog else None, '%.2f/t' % (len(prog) / len(p.targets)) if prog else '', err, '%.1fs' % dt, dict(stats), flush=True)
for f, a in sorted(agg.items()):
    print(f, a[0], a[1], a[2], '%.2f ops/t' % (a[2] / a[1]), '%.1fs' % a[3])
print('total', sum(a[2] for a in agg.values()), '%.1fs' % (time.time() - T0), dict(allst))

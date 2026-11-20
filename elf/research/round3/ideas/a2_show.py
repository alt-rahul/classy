"""Print a solve as per-target derivations (value-level steps), to look for waste by eye."""
import sys, json
from s3 import *
which, pid = sys.argv[1], sys.argv[2]
kw = dict(pool_cap=46, evict_size=1, regret=0.1)
if len(sys.argv) > 3: kw.update(json.loads(sys.argv[3]))
p = {q.id: q for q in load_puzzles(which)}[pid]
prog, stats, st = solve3(p, S3, **kw)
from repair import dce
st = dce(p.sv, st, p.targets)
print(pid, p.family, 'starts', p.sv, 'limit', p.limit, 'ops', len(st), stats)
tset = set(p.targets); n = 0; buf = []
idx = {t: i for i, t in enumerate(sorted(p.targets))}
for (op, a, b, r) in st:
    buf.append('%d%s%d=%d' % (a, op, b, r)); n += 1
    if r in tset:
        print('T%-2d %-20d cost %d : %s' % (idx[r], r, n, '  '.join(buf))); buf = []; n = 0

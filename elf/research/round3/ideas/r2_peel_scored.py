"""Survivor S1 on scored puzzles: PEEL solve + global repair, offered to the store (tag r3C)."""
import sys, time, json
from s3 import *
import repair
minbits = int(sys.argv[1]); skip = int(sys.argv[2]); lim = int(sys.argv[3]); step = int(sys.argv[4]) if len(sys.argv) > 4 else 1
kw0 = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
repair.self_thr[0] = 10**9
st = load_store()
ps = [p for p in load_puzzles('scored') if pclass(p) == 'other>=20' and max(p.targets).bit_length() >= minbits]
print('candidates', len(ps), 'ops', sum(st[p.id]['len'] for p in ps), flush=True)
gain = 0; base = 0; T0 = time.time(); new = 0
for p in ps[skip::step][:lim]:
    t0 = time.time()
    kw = dict(pool_cap=46, evict_size=1, regret=0.1); kw.update(kw0)
    prog, stats, steps = solve3(p, PEEL, **kw)
    n1 = len(prog)
    s2 = repair.improve(p, steps, passes=3)
    pr = regalloc(p.sv, s2, p.targets, p.nreg) or prog
    n0 = st[p.id]['len']
    ok = offer(st, p, pr, 'r3C')
    if ok:
        gain += n0 - len(pr); save_store(st)
    base += n0; new += len(pr)
    print(p.id, max(p.targets).bit_length(), 'store', n0, 'peel', n1, '+rep', len(pr), 'stored' if ok else '', '%.0fs' % (time.time() - t0), 'peel%d' % stats.get('peel', 0), flush=True)
dt = time.time() - T0
print('store %d  single PEEL run %d (%.2f%%)  accepted gain %d (%.2f%%)  %.0fs => %.0f ops/CPU-h' % (base, new, 100 * (new - base) / base, gain, 100 * gain / base, dt, gain / dt * 3600))

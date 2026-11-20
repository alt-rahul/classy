"""Survivor 1: global repair (threshold lifted) on stored scored programs; offers improvements (tag r3C)."""
import sys, time, json
from s3 import *
import repair
from repair_store import prog_to_steps
maxbits = int(sys.argv[1]); lim = int(sys.argv[2]); classes = sys.argv[3].split(',')
skip = int(sys.argv[4]) if len(sys.argv) > 4 else 0
minbits = int(sys.argv[5]) if len(sys.argv) > 5 else 0
repair.self_thr[0] = 10**9
st = load_store()
ps = [p for p in load_puzzles('scored') if pclass(p) in classes and minbits <= max(p.targets).bit_length() <= maxbits and st[p.id]['len'] > 150]
print('candidates', len(ps), 'ops', sum(st[p.id]['len'] for p in ps), flush=True)
gain = 0; T0 = time.time(); base = 0
for p in ps[skip:skip + lim]:
    t0 = time.time()
    n0 = st[p.id]['len']
    s2 = repair.improve(p, prog_to_steps(p, st[p.id]['prog']), passes=3)
    pr = regalloc(p.sv, s2, p.targets, p.nreg)
    ok = pr is not None and offer(st, p, pr, 'r3C')
    if ok:
        gain += n0 - len(pr); save_store(st)
    base += n0
    print(p.id, pclass(p), max(p.targets).bit_length(), n0, '->', len(pr) if pr else None, 'stored' if ok else '', '%.0fs' % (time.time() - t0), flush=True)
print('gain', gain, 'of', base, '(%.2f%%)' % (100 * gain / max(base, 1)), 'in %.0fs' % (time.time() - T0), '=> %.0f ops/CPU-h' % (gain / (time.time() - T0) * 3600))

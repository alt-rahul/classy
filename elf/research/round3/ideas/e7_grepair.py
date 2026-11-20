"""X24: global repair. Re-derive each target from ALL values of the program (not just a 46-value pool),
using L2 over the whole history (repair.improve with the size threshold lifted)."""
import sys, time, json
from s3 import *
import repair
from repair_store import prog_to_steps
which = sys.argv[1]; ids = sys.argv[2].split(','); thr = int(sys.argv[3]); passes = int(sys.argv[4]) if len(sys.argv) > 4 else 3
P = {p.id: p for p in load_puzzles(which)}
st = load_store() if which == 'scored' else None
for i in ids:
    p = P[i]
    if st is not None:
        prog = st[i]['prog']; steps = prog_to_steps(p, prog)
    else:
        prog, stats, steps = solve3(p, S3, pool_cap=46, evict_size=1, regret=0.1)
    n0 = len(prog)
    for th in (150, thr):
        repair.self_thr[0] = th
        t0 = time.time()
        s2 = repair.improve(p, steps, passes=passes, verbose=False)
        pr = regalloc(p.sv, s2, p.targets, p.nreg)
        print(i, p.family, max(p.targets).bit_length(), 'base', n0, 'thr', th, '->', len(pr), check(p, pr), '%.0fs' % (time.time() - t0), flush=True)
